"""Automatic FoundationPose tracking-mesh preparation.

The reconstruction mesh is the appearance source and remains the full-resolution
render asset.  FoundationPose receives a separate, automatically simplified
GLB whose texture and material are embedded in the file.  The simplifier never
recenters, rescales, rotates, or manually aligns geometry: it performs quadric
edge collapses in the generated object frame, assigns each output face a real
source UV chart, and splits vertices at texture seams instead of averaging UVs.

Every output is reloaded and checked before it can enter a scene manifest.
Missing texture/UV data, topology drift between the metric and Any6D-sized
geometry, a changed coordinate frame/scale, excessive surface error, or a
missing embedded texture fails closed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from oap.utils.io import sha256_file, write_json_atomic

logger = logging.getLogger("oap.reconstruct.tracking")

__all__ = [
    "DEFAULT_TRACKING_TARGET_FACES",
    "TRACKING_MESH_SCHEMA",
    "run_tracking_step",
    "validate_tracking_release",
]

TRACKING_MESH_SCHEMA = "oap_foundationpose_tracking_mesh_v1"

# NVIDIA's Isaac ROS FoundationPose documentation recommends simplifying mesh
# geometry to reduce the renderer's cost.  This fixed, object-agnostic ceiling
# is deliberately not exposed as a per-object reconstruction knob.
DEFAULT_TRACKING_TARGET_FACES = 20_000

_ALGORITHM = "quadric_edge_collapse_seam_safe_uv_v2"
_SIZED_GEOMETRY_THRESHOLD = 0.05
_MAX_CENTER_SHIFT_DIAGONAL_FRACTION = 0.005
_MAX_EXTENT_ERROR_DIAGONAL_FRACTION = 0.01
_MAX_AREA_RELATIVE_ERROR = 0.10
_MAX_SURFACE_P95_DIAGONAL_FRACTION = 0.03
_MAX_APPEARANCE_MEAN_ERROR = 0.01
_MAX_APPEARANCE_P95_ERROR = 0.03
_MAX_APPEARANCE_MAX_ERROR = 0.10
_SURFACE_VALIDATION_SAMPLES = 8_192
_APPEARANCE_BARYCENTRIC_SAMPLES = np.asarray(
    (
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (0.80, 0.10, 0.10),
        (0.10, 0.80, 0.10),
        (0.10, 0.10, 0.80),
        (0.50, 0.25, 0.25),
        (0.25, 0.50, 0.25),
        (0.25, 0.25, 0.50),
    ),
    dtype=float,
)
_MAX_EXPORT_FACE_FEEDBACK_ATTEMPTS = 3


class _TrackingFaceCeilingExceeded(RuntimeError):
    """An exported candidate exceeded the sealed renderer face ceiling."""

    def __init__(self, *, actual_faces: int, ceiling: int) -> None:
        self.actual_faces = int(actual_faces)
        self.ceiling = int(ceiling)
        super().__init__(
            "FoundationPose tracking mesh reload has "
            f"{self.actual_faces:,} faces; requires <= {self.ceiling:,}"
        )


def _tracking_face_budget(
    source_faces: int,
    target_faces: int,
) -> tuple[int, int]:
    """Return deterministic simplifier request and accepted near-ceiling floor.

    ``fast-simplification`` treats ``target_count`` as approximate for some
    large topologies.  Leave a fixed 0.5% export margin, while requiring the
    validated artifact to retain at least 99% of the public ceiling.  Sources
    already below the ceiling remain byte-for-byte unsimplified.
    """
    source = int(source_faces)
    target = int(target_faces)
    if source <= 0 or target < 100:
        raise ValueError("tracking face budget requires source>0 and target>=100")
    ceiling = min(source, target)
    if source <= target:
        return ceiling, ceiling
    margin = max(1, int(np.ceil(float(target) * 0.005)))
    floor = max(100, int(np.floor(float(target) * 0.99)))
    requested = max(floor, target - margin)
    return requested, floor


def _next_tracking_face_request(
    *,
    current_request: int,
    reloaded_faces: int,
    ceiling: int,
    near_ceiling_floor: int,
) -> int:
    """Return the next deterministic simplifier request after export overflow.

    Some high-face, multi-primitive sources make the simplifier or GLB
    round-trip land above its requested count. Feedback targets the midpoint of
    the *final artifact* acceptance band using only the observed round-trip
    ratio. The intermediate simplifier request may be below that final floor;
    only the reloaded artifact is required to contain 19,800--20,000 faces.
    """
    current = int(current_request)
    actual = int(reloaded_faces)
    limit = int(ceiling)
    floor = int(near_ceiling_floor)
    if actual <= limit:
        raise ValueError("face feedback requires an observed ceiling overflow")
    desired_reload = (floor + limit) // 2
    ratio_request = int(
        np.floor(float(current) * float(desired_reload) / float(actual))
    )
    # 100 is the same non-degenerate minimum accepted by run_tracking_step.
    # At the minimum, repeated overflow remains observable and is rejected by
    # the fixed attempt cap rather than by silently accepting a small artifact.
    return max(100, min(current - 1, ratio_request))


def _require_tracking_face_band(
    *,
    source_faces: int,
    target_faces: int,
    reloaded_faces: int,
) -> None:
    """Require the sealed face band on the final reloaded GLB only."""
    ceiling = min(int(source_faces), int(target_faces))
    _, near_ceiling_floor = _tracking_face_budget(source_faces, target_faces)
    actual = int(reloaded_faces)
    if actual > ceiling:
        raise _TrackingFaceCeilingExceeded(
            actual_faces=actual,
            ceiling=ceiling,
        )
    if int(source_faces) > int(target_faces) and actual < near_ceiling_floor:
        raise RuntimeError(
            "FoundationPose tracking mesh was simplified below its sealed "
            "near-ceiling floor"
        )


def _load_mesh(path: Path) -> Any:
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"tracking-mesh source is not one triangle mesh: {path}")
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(vertices) < 4
        or len(faces) < 4
        or not np.isfinite(vertices).all()
    ):
        raise RuntimeError(f"tracking-mesh source has invalid triangle geometry: {path}")
    return loaded


def _texture_uv(metric_mesh: Any, texture_path: Path) -> tuple[np.ndarray, Image.Image]:
    uv = getattr(getattr(metric_mesh, "visual", None), "uv", None)
    if uv is None:
        raise RuntimeError(
            "FoundationPose tracking mesh requires generated UV coordinates; "
            "re-run textured SAM3D reconstruction"
        )
    uv_array = np.asarray(uv, dtype=float)
    if (
        uv_array.shape != (len(metric_mesh.vertices), 2)
        or not np.isfinite(uv_array).all()
    ):
        raise RuntimeError(
            "FoundationPose tracking mesh has invalid per-vertex UV coordinates"
        )
    if not Path(texture_path).is_file():
        raise RuntimeError(
            f"FoundationPose tracking texture is missing: {texture_path}; "
            "re-run the textured scale step"
        )
    try:
        with Image.open(texture_path) as opened:
            image = opened.convert("RGBA").copy()
    except Exception as exc:
        raise RuntimeError(
            f"FoundationPose tracking texture is unreadable: {texture_path}"
        ) from exc
    if image.width < 2 or image.height < 2:
        raise RuntimeError(
            f"FoundationPose tracking texture is degenerate: {texture_path}"
        )
    return uv_array, image


def _selected_geometry(
    metric_mesh: Any,
    sized_mesh_path: Path | None,
) -> tuple[np.ndarray, str, list[float]]:
    """Select the same automatic final geometry policy the manifest used.

    The Any6D-sized mesh has the same face/vertex indexing as the metric mesh.
    That exact topology identity is what lets the baked texture be inherited
    without a manual UV transfer or an object-frame adjustment.
    """

    metric_vertices = np.asarray(metric_mesh.vertices, dtype=float)
    if sized_mesh_path is None or not Path(sized_mesh_path).is_file():
        return metric_vertices.copy(), "metric", [1.0, 1.0, 1.0]

    sized = _load_mesh(Path(sized_mesh_path))
    metric_faces = np.asarray(metric_mesh.faces, dtype=np.int64)
    sized_faces = np.asarray(sized.faces, dtype=np.int64)
    if (
        len(sized.vertices) != len(metric_mesh.vertices)
        or sized_faces.shape != metric_faces.shape
        or not np.array_equal(sized_faces, metric_faces)
    ):
        raise RuntimeError(
            "Any6D-sized geometry no longer has the metric mesh topology; "
            "refusing a heuristic texture/UV remap"
        )
    metric_extent = np.asarray(metric_mesh.extents, dtype=float)
    sized_extent = np.asarray(sized.extents, dtype=float)
    ratio = sized_extent / np.maximum(metric_extent, 1e-12)
    if not np.isfinite(ratio).all() or np.any(ratio <= 0.0):
        raise RuntimeError("Any6D-sized geometry has invalid metric extents")
    use_sized = float(np.max(np.abs(ratio - 1.0))) > _SIZED_GEOMETRY_THRESHOLD
    if use_sized:
        return np.asarray(sized.vertices, dtype=float).copy(), "any6d_sized", ratio.tolist()
    return metric_vertices.copy(), "metric", ratio.tolist()


def _weld_positions(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index-only weld of coincident positions.

    Returns ``(welded, inverse)`` with ``welded[inverse] == vertices`` exactly:
    no coordinate is snapped, rounded or moved, so anything computed on the
    welded array is computed on the same surface.  Only vertex IDENTITY changes,
    which is what an edge-collapse simplifier needs in order to see that two
    triangles on opposite sides of a UV seam share an edge.

    ``np.unique`` on the raw float rows is exact and order-independent, so the
    result does not depend on the source vertex order.
    """
    array = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    welded, inverse = np.unique(array, axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(len(array))
    return welded, inverse


def _replay_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    *,
    target_faces: int,
    requested_faces: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simplify geometry and transfer UVs without averaging across seams.

    Edge-collapse libraries operate only on geometry.  Averaging every source
    UV that maps to one collapsed vertex is not appearance preserving: values
    near ``u=0`` and ``u=1`` at a texture seam average to the middle of the
    image.  Instead, each output face inherits the UV chart of one surviving
    source face.  Output vertices are split per face, which is the standard way
    OBJ/GLB represents discontinuous face-corner attributes.

    The fourth return value is the source face-corner UV reference used by the
    post-export source-vs-mapped colour gate.
    """
    if len(faces) <= target_faces:
        source_faces = faces.astype(np.int64, copy=True)
        return (
            vertices.copy(),
            source_faces,
            uv.copy(),
            np.asarray(uv, dtype=float)[source_faces],
        )
    default_request, _ = _tracking_face_budget(len(faces), target_faces)
    requested_faces = (
        default_request if requested_faces is None else int(requested_faces)
    )
    if not 100 <= requested_faces <= default_request:
        raise ValueError(
            "tracking simplifier request must remain within the deterministic "
            "feedback budget"
        )
    try:
        import fast_simplification
    except ImportError as exc:
        raise RuntimeError(
            "automatic FoundationPose mesh simplification requires the "
            "fast-simplification package from OaP's declared dependencies"
        ) from exc

    # Collapse EDGES, so the simplifier has to see shared edges. ``vertices``
    # holds one entry per (position, uv) pair, so every UV chart boundary breaks
    # vertex sharing and the two triangles across a seam have no edge between
    # them. On the 2026-08-19 spacemouse_box that left 26,162 face-adjacency
    # components in a 1,126,856-face mesh and pinned the output at 26,356 faces
    # for every target tried (19,900 / 9,950 / 5,000) -- the component count is
    # the floor, so the caller's three-attempt ratio feedback could never
    # converge. Welding positions first drops it to 3 components and the same
    # target then lands exactly.
    #
    # Welding is pure index bookkeeping: welded[inverse] reproduces ``vertices``
    # exactly, so the collapse sequence is computed on the SAME surface.
    welded_vertices, welded_inverse = _weld_positions(vertices)
    welded_faces = welded_inverse[np.asarray(faces, dtype=np.int64)]
    degenerate = (
        (welded_faces[:, 0] == welded_faces[:, 1])
        | (welded_faces[:, 1] == welded_faces[:, 2])
        | (welded_faces[:, 0] == welded_faces[:, 2])
    )
    if bool(degenerate.any()):
        raise RuntimeError(
            "welding positions collapsed a source triangle to a degenerate "
            f"face ({int(degenerate.sum())} of {len(welded_faces)}); the source "
            "carries coincident-vertex triangles simplification cannot own"
        )
    simplified_vertices, simplified_faces, collapses = (
        fast_simplification.simplify(
            np.ascontiguousarray(welded_vertices, dtype=np.float64),
            np.ascontiguousarray(welded_faces, dtype=np.int32),
            target_count=requested_faces,
            agg=5.0,
            return_collapses=True,
        )
    )
    # Replay on the finite 3-D geometry to recover the exact welded-vertex
    # -> simplified-vertex mapping. Replaying QEM arithmetic in UV space is
    # invalid because UV charts are discontinuous and may contain zero-area
    # triangles even when the 3-D geometry is valid.
    _, uv_faces, welded_mapping = fast_simplification.replay_simplification(
        np.ascontiguousarray(welded_vertices, dtype=np.float32),
        np.ascontiguousarray(welded_faces, dtype=np.int32),
        np.ascontiguousarray(collapses),
    )
    # Compose source -> welded -> simplified, so the UV transfer below still
    # receives a mapping indexed by the ORIGINAL (UV-split) vertices.
    vertex_mapping = np.asarray(welded_mapping, dtype=np.int64)[welded_inverse]
    if not np.array_equal(
        np.asarray(simplified_faces, dtype=np.int64),
        np.asarray(uv_faces, dtype=np.int64),
    ):
        raise RuntimeError(
            "quadric simplification and UV replay produced different topology"
        )
    vertex_mapping = np.asarray(vertex_mapping, dtype=np.int64)
    if (
        vertex_mapping.shape != (len(uv),)
        or np.any(vertex_mapping >= len(simplified_vertices))
    ):
        raise RuntimeError("simplification returned an invalid UV vertex mapping")
    return _transfer_face_corner_uv(
        source_vertices=np.asarray(vertices, dtype=float),
        source_faces=np.asarray(faces, dtype=np.int64),
        source_uv=np.asarray(uv, dtype=float),
        simplified_vertices=np.asarray(simplified_vertices, dtype=float),
        simplified_faces=np.asarray(simplified_faces, dtype=np.int64),
        vertex_mapping=vertex_mapping,
    )


def _transfer_face_corner_uv(
    *,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    source_uv: np.ndarray,
    simplified_vertices: np.ndarray,
    simplified_faces: np.ndarray,
    vertex_mapping: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transfer a real source UV chart to every simplified face.

    Every triangle left by an edge-collapse sequence descends from at least one
    source triangle.  Candidate source triangles are keyed by their three
    mapped output vertices.  The largest source-area candidate (then lowest
    source index) is selected deterministically.  Vertices are expanded per
    output face so two adjacent faces never have to average incompatible UV
    charts.
    """

    if (
        vertex_mapping.shape != (len(source_vertices),)
        or np.any(vertex_mapping >= len(simplified_vertices))
        or np.any(vertex_mapping < -1)
    ):
        raise RuntimeError("simplification returned an invalid UV vertex mapping")

    candidates: dict[tuple[int, int, int], list[tuple[float, int]]] = {}
    for source_index, source_face in enumerate(source_faces):
        mapped_face = vertex_mapping[source_face]
        if np.any(mapped_face < 0) or len(set(mapped_face.tolist())) != 3:
            continue
        key = tuple(sorted(int(value) for value in mapped_face))
        triangle = source_vertices[source_face]
        area = 0.5 * float(
            np.linalg.norm(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            )
        )
        candidates.setdefault(key, []).append((-area, int(source_index)))

    reference_uv = np.empty((len(simplified_faces), 3, 2), dtype=float)
    for output_index, output_face in enumerate(simplified_faces):
        key = tuple(sorted(int(value) for value in output_face))
        source_candidates = candidates.get(key)
        if not source_candidates:
            raise RuntimeError(
                "simplification produced an output face with no source UV chart"
            )
        _, source_index = min(source_candidates)
        source_face = source_faces[source_index]
        mapped_source_face = vertex_mapping[source_face]
        for corner, output_vertex in enumerate(output_face):
            source_corners = np.flatnonzero(
                mapped_source_face == int(output_vertex)
            )
            if len(source_corners) != 1:
                raise RuntimeError(
                    "source UV chart does not map one-to-one onto output face"
                )
            reference_uv[output_index, corner] = source_uv[
                source_face[int(source_corners[0])]
            ]

    # Face-corner expansion intentionally duplicates geometry vertices at UV
    # seams.  The surface is unchanged; only the attribute representation is.
    expanded_vertices = simplified_vertices[simplified_faces.reshape(-1)]
    expanded_faces = np.arange(
        len(expanded_vertices), dtype=np.int64
    ).reshape((-1, 3))
    expanded_uv = reference_uv.reshape((-1, 2))
    if (
        not np.isfinite(expanded_vertices).all()
        or not np.isfinite(expanded_uv).all()
    ):
        raise RuntimeError("simplified FoundationPose geometry/UV output is invalid")
    return expanded_vertices, expanded_faces, expanded_uv, reference_uv


def _image_digest(image: Image.Image) -> str:
    rgba = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(np.asarray(rgba, dtype=np.uint8).tobytes())
    digest.update(str(rgba.size).encode("ascii"))
    return digest.hexdigest()


def _embedded_texture(mesh: Any) -> Image.Image | None:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return None
    image = (
        getattr(material, "baseColorTexture", None)
        or getattr(material, "image", None)
    )
    if image is None:
        return None
    return image.convert("RGBA")


def _sample_texture_rgba(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    """Bilinearly sample RGBA at normalized UVs, matching image v inversion."""

    pixels = np.asarray(image.convert("RGBA"), dtype=float) / 255.0
    coordinates = np.asarray(uv, dtype=float)
    # Generated UVs occasionally place a seam exactly at one; retain endpoints
    # rather than wrapping 1 -> 0 so the comparison reflects the authored map.
    u = np.clip(coordinates[..., 0], 0.0, 1.0) * (image.width - 1)
    v = (1.0 - np.clip(coordinates[..., 1], 0.0, 1.0)) * (
        image.height - 1
    )
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, image.width - 1)
    y1 = np.minimum(y0 + 1, image.height - 1)
    wx = (u - x0)[..., None]
    wy = (v - y0)[..., None]
    top = pixels[y0, x0] * (1.0 - wx) + pixels[y0, x1] * wx
    bottom = pixels[y1, x0] * (1.0 - wx) + pixels[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _appearance_metrics(
    image: Image.Image,
    source_face_uv: np.ndarray,
    mapped_face_uv: np.ndarray,
) -> dict[str, float]:
    """Measure normalized source-vs-mapped texture colour error.

    Sampling inside every output triangle catches both seam averaging and
    cross-chart interpolation; merely checking that the embedded PNG bytes are
    unchanged cannot detect either failure.
    """

    source = np.asarray(source_face_uv, dtype=float)
    mapped = np.asarray(mapped_face_uv, dtype=float)
    if source.shape != mapped.shape or source.ndim != 3 or source.shape[1:] != (
        3,
        2,
    ):
        raise RuntimeError("appearance gate received incompatible face UV arrays")
    source_samples = np.einsum(
        "sc,fcd->fsd", _APPEARANCE_BARYCENTRIC_SAMPLES, source
    )
    mapped_samples = np.einsum(
        "sc,fcd->fsd", _APPEARANCE_BARYCENTRIC_SAMPLES, mapped
    )
    source_rgba = _sample_texture_rgba(image, source_samples)
    mapped_rgba = _sample_texture_rgba(image, mapped_samples)
    # RGB mean absolute error is normalized to [0, 1]. Alpha is an authored
    # mask rather than appearance and is checked by embedded texture identity.
    errors = np.mean(
        np.abs(source_rgba[..., :3] - mapped_rgba[..., :3]), axis=-1
    )
    return {
        "source_vs_mapped_texture_color_mean_error": float(np.mean(errors)),
        "source_vs_mapped_texture_color_p95_error": float(
            np.percentile(errors, 95.0)
        ),
        "source_vs_mapped_texture_color_max_error": float(np.max(errors)),
    }


def _require_appearance_quality(metrics: dict[str, float]) -> None:
    failures: list[str] = []
    if (
        metrics["source_vs_mapped_texture_color_mean_error"]
        > _MAX_APPEARANCE_MEAN_ERROR
    ):
        failures.append("mean texture colour error")
    if (
        metrics["source_vs_mapped_texture_color_p95_error"]
        > _MAX_APPEARANCE_P95_ERROR
    ):
        failures.append("p95 texture colour error")
    if (
        metrics["source_vs_mapped_texture_color_max_error"]
        > _MAX_APPEARANCE_MAX_ERROR
    ):
        failures.append("maximum texture colour error")
    if failures:
        raise RuntimeError(
            "automatic FoundationPose tracking mesh failed appearance "
            "validation: " + "; ".join(failures)
        )


def _write_textured_obj(
    path: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
) -> None:
    """Write a deterministic OBJ with UVs and no implicit MTL/texture sidecar."""

    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# OaP generated metric visual mesh\n")
        for vertex in np.asarray(vertices, dtype=float):
            stream.write(
                f"v {vertex[0]:.10g} {vertex[1]:.10g} {vertex[2]:.10g}\n"
            )
        for texcoord in np.asarray(uv, dtype=float):
            stream.write(f"vt {texcoord[0]:.10g} {texcoord[1]:.10g}\n")
        for face in np.asarray(faces, dtype=np.int64):
            indices = [int(value) + 1 for value in face]
            stream.write(
                "f " + " ".join(f"{value}/{value}" for value in indices) + "\n"
            )


def _surface_metrics(source: Any, simplified: Any) -> dict[str, float]:
    import trimesh

    source_bounds = np.asarray(source.bounds, dtype=float)
    simplified_bounds = np.asarray(simplified.bounds, dtype=float)
    source_extent = source_bounds[1] - source_bounds[0]
    diagonal = float(np.linalg.norm(source_extent))
    if not np.isfinite(diagonal) or diagonal <= 1e-9:
        raise RuntimeError("tracking-mesh source has a degenerate metric diagonal")
    center_shift = float(
        np.linalg.norm(simplified_bounds.mean(axis=0) - source_bounds.mean(axis=0))
        / diagonal
    )
    extent_error = float(
        np.max(np.abs((simplified_bounds[1] - simplified_bounds[0]) - source_extent))
        / diagonal
    )
    source_area = float(source.area)
    simplified_area = float(simplified.area)
    if not np.isfinite(source_area) or source_area <= 0.0:
        raise RuntimeError("tracking-mesh source has invalid surface area")
    area_relative_error = abs(simplified_area - source_area) / source_area

    # Deterministic bidirectional surface samples make gross QEM deformation
    # observable without an rtree/Embree dependency.  The global threshold is
    # deliberately loose enough to include the finite sampling spacing, while
    # the tighter bounds/area checks separately protect frame and scale.
    source_points, _ = trimesh.sample.sample_surface(
        source, _SURFACE_VALIDATION_SAMPLES, seed=0
    )
    simplified_points, _ = trimesh.sample.sample_surface(
        simplified, _SURFACE_VALIDATION_SAMPLES, seed=1
    )
    source_to_simplified = cKDTree(simplified_points).query(
        source_points, k=1, workers=-1
    )[0]
    simplified_to_source = cKDTree(source_points).query(
        simplified_points, k=1, workers=-1
    )[0]
    surface_p95 = float(
        max(
            np.percentile(source_to_simplified, 95.0),
            np.percentile(simplified_to_source, 95.0),
        )
        / diagonal
    )
    return {
        "source_diagonal_m": diagonal,
        "center_shift_diagonal_fraction": center_shift,
        "extent_error_diagonal_fraction": extent_error,
        "area_relative_error": float(area_relative_error),
        "surface_p95_diagonal_fraction": surface_p95,
    }


def _validate_export(
    *,
    source_mesh: Any,
    expected_tracking_mesh: Any,
    source_face_uv: np.ndarray,
    tracking_path: Path,
    visual_path: Path,
    source_texture: Image.Image,
    target_faces: int,
) -> dict[str, Any]:
    tracking = _load_mesh(tracking_path)
    _require_tracking_face_band(
        source_faces=len(source_mesh.faces),
        target_faces=int(target_faces),
        reloaded_faces=len(tracking.faces),
    )
    expected_diagonal = max(float(np.linalg.norm(source_mesh.extents)), 1e-9)
    if (
        np.asarray(tracking.vertices).shape
        != np.asarray(expected_tracking_mesh.vertices).shape
        or np.asarray(tracking.faces).shape
        != np.asarray(expected_tracking_mesh.faces).shape
        or not np.array_equal(
            np.asarray(tracking.faces, dtype=np.int64),
            np.asarray(expected_tracking_mesh.faces, dtype=np.int64),
        )
        or not np.allclose(
            np.asarray(tracking.vertices, dtype=float),
            np.asarray(expected_tracking_mesh.vertices, dtype=float),
            rtol=0.0,
            atol=expected_diagonal * 1e-6,
        )
    ):
        raise RuntimeError(
            "reloaded tracking GLB changed validated geometry/face ordering"
        )
    visual = _load_mesh(visual_path)
    if (
        len(visual.faces) != len(source_mesh.faces)
        or not np.allclose(
            visual.bounds,
            source_mesh.bounds,
            rtol=0.0,
            atol=expected_diagonal * 1e-8,
        )
    ):
        raise RuntimeError(
            "full-resolution render mesh changed topology, coordinates, or scale"
        )
    visual_uv = getattr(getattr(visual, "visual", None), "uv", None)
    if (
        visual_uv is None
        or np.asarray(visual_uv).shape != (len(visual.vertices), 2)
        or not np.array_equal(
            np.asarray(visual.faces, dtype=np.int64),
            np.asarray(source_mesh.faces, dtype=np.int64),
        )
    ):
        raise RuntimeError(
            "full-resolution render OBJ did not preserve generated UV topology"
        )
    visual_appearance = _appearance_metrics(
        source_texture,
        np.asarray(source_mesh.visual.uv, dtype=float)[
            np.asarray(source_mesh.faces, dtype=np.int64)
        ],
        np.asarray(visual_uv, dtype=float)[
            np.asarray(visual.faces, dtype=np.int64)
        ],
    )
    _require_appearance_quality(visual_appearance)

    uv = getattr(getattr(tracking, "visual", None), "uv", None)
    embedded = _embedded_texture(tracking)
    if (
        uv is None
        or np.asarray(uv).shape != (len(tracking.vertices), 2)
        or not np.isfinite(np.asarray(uv, dtype=float)).all()
        or embedded is None
    ):
        raise RuntimeError(
            "reloaded tracking GLB did not preserve embedded UV texture/material"
        )
    if (
        embedded.size != source_texture.size
        or _image_digest(embedded) != _image_digest(source_texture)
    ):
        raise RuntimeError(
            "reloaded tracking GLB texture pixels differ from reconstruction"
        )

    metrics = _surface_metrics(source_mesh, tracking)
    mapped_face_uv = np.asarray(uv, dtype=float)[
        np.asarray(tracking.faces, dtype=np.int64)
    ]
    appearance = _appearance_metrics(
        source_texture,
        np.asarray(source_face_uv, dtype=float),
        mapped_face_uv,
    )
    _require_appearance_quality(appearance)
    failures: list[str] = []
    if (
        metrics["center_shift_diagonal_fraction"]
        > _MAX_CENTER_SHIFT_DIAGONAL_FRACTION
    ):
        failures.append("object-frame center shifted")
    if (
        metrics["extent_error_diagonal_fraction"]
        > _MAX_EXTENT_ERROR_DIAGONAL_FRACTION
    ):
        failures.append("metric extent changed")
    if metrics["area_relative_error"] > _MAX_AREA_RELATIVE_ERROR:
        failures.append("surface area changed")
    if (
        metrics["surface_p95_diagonal_fraction"]
        > _MAX_SURFACE_P95_DIAGONAL_FRACTION
    ):
        failures.append("sampled surface error is too large")
    if failures:
        raise RuntimeError(
            "automatic FoundationPose tracking mesh failed geometry validation: "
            + "; ".join(failures)
        )
    return {
        "passed": True,
        "limits": {
            "max_center_shift_diagonal_fraction": (
                _MAX_CENTER_SHIFT_DIAGONAL_FRACTION
            ),
            "max_extent_error_diagonal_fraction": (
                _MAX_EXTENT_ERROR_DIAGONAL_FRACTION
            ),
            "max_area_relative_error": _MAX_AREA_RELATIVE_ERROR,
            "max_surface_p95_diagonal_fraction": (
                _MAX_SURFACE_P95_DIAGONAL_FRACTION
            ),
            "max_appearance_mean_error": _MAX_APPEARANCE_MEAN_ERROR,
            "max_appearance_p95_error": _MAX_APPEARANCE_P95_ERROR,
            "max_appearance_max_error": _MAX_APPEARANCE_MAX_ERROR,
        },
        "appearance_transfer": "per_face_source_chart_no_uv_averaging",
        "visual_source_vs_mapped_texture_color_p95_error": (
            visual_appearance[
                "source_vs_mapped_texture_color_p95_error"
            ]
        ),
        **appearance,
        **metrics,
    }


def validate_tracking_release(
    metadata_path: Path,
    *,
    metric_obj: Path,
    metric_texture: Path,
    tracking_mesh: Path,
    visual_mesh: Path,
    sized_mesh: Path | None = None,
) -> dict[str, Any]:
    """Validate metadata and hashes against the *current* source/output files."""

    import json
    from collections.abc import Mapping

    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"tracking metadata is missing or unreadable: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("tracking metadata must be a JSON object")
    if metadata.get("schema") != TRACKING_MESH_SCHEMA:
        raise RuntimeError(
            f"tracking metadata schema is not {TRACKING_MESH_SCHEMA!r}"
        )
    if metadata.get("algorithm") != _ALGORITHM:
        raise RuntimeError("tracking metadata algorithm is stale or unsupported")
    if metadata.get("manual_alignment") is not False:
        raise RuntimeError("tracking metadata must declare manual_alignment=false")
    quality = metadata.get("quality_gate")
    if not isinstance(quality, Mapping) or quality.get("passed") is not True:
        raise RuntimeError("tracking metadata has no passed quality gate")
    if (
        quality.get("appearance_transfer")
        != "per_face_source_chart_no_uv_averaging"
    ):
        raise RuntimeError(
            "tracking metadata has no seam-safe appearance quality evidence"
        )
    required_hashes = {
        "source_metric_mesh_sha256": sha256_file(metric_obj),
        "source_texture_sha256": sha256_file(metric_texture),
        "tracking_mesh_sha256": sha256_file(tracking_mesh),
        "visual_mesh_sha256": sha256_file(visual_mesh),
    }
    for field, actual in required_hashes.items():
        if metadata.get(field) != actual:
            raise RuntimeError(
                f"tracking metadata {field} does not match the current artifact"
            )
    current_sized = (
        Path(sized_mesh)
        if sized_mesh is not None and Path(sized_mesh).is_file()
        else None
    )
    expected_sized_hash = (
        sha256_file(current_sized) if current_sized is not None else None
    )
    if metadata.get("sized_mesh_sha256") != expected_sized_hash:
        raise RuntimeError(
            "tracking metadata sized_mesh_sha256 does not match current lineage"
        )
    selected = metadata.get("selected_geometry")
    if selected not in {"metric", "any6d_sized"}:
        raise RuntimeError("tracking metadata selected_geometry is invalid")
    if selected == "any6d_sized" and current_sized is None:
        raise RuntimeError(
            "tracking metadata selected Any6D geometry without a current source"
        )
    return metadata


def run_tracking_step(
    step_dir: Path,
    metric_obj: Path,
    metric_texture: Path,
    *,
    name: str,
    sized_mesh: Path | None = None,
    target_faces: int = DEFAULT_TRACKING_TARGET_FACES,
) -> dict[str, Any]:
    """Generate, validate, then atomically promote one tracking asset set.

    Existing canonical outputs remain untouched until every temporary GLB/OBJ
    and its metadata pass validation. A validation failure leaves the previous
    generation intact. Each canonical file is then replaced atomically and
    metadata is promoted last; interruption between replacements may leave a
    mixed generation, but its recorded state/output hashes become stale so
    manifest assembly fails closed and requires regeneration.
    """

    import json
    import trimesh

    if int(target_faces) < 100:
        raise ValueError("tracking target_faces must be at least 100")
    step_dir = Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = step_dir / f"{name}_tracking.glb"
    visual_path = step_dir / f"{name}_visual.obj"
    metadata_path = step_dir / f"{name}_tracking.json"

    metric = _load_mesh(Path(metric_obj))
    uv, texture = _texture_uv(metric, Path(metric_texture))
    selected_vertices, selected_source, sized_ratio = _selected_geometry(
        metric, sized_mesh
    )
    faces = np.asarray(metric.faces, dtype=np.int64)
    source_material = trimesh.visual.material.PBRMaterial(
        name=f"{name}_reconstructed",
        baseColorTexture=texture,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    source_mesh = trimesh.Trimesh(
        vertices=selected_vertices,
        faces=faces,
        visual=trimesh.visual.texture.TextureVisuals(
            uv=uv, material=source_material
        ),
        process=False,
    )

    initial_request, near_ceiling_floor = _tracking_face_budget(
        len(source_mesh.faces), int(target_faces)
    )
    ceiling = min(len(source_mesh.faces), int(target_faces))
    current_request = initial_request
    request_history: list[int] = []
    reload_face_history: list[int] = []
    metadata: dict[str, Any] | None = None

    # Every feedback attempt owns a fresh temporary directory.  Only an
    # explicit reload face-ceiling overflow is retryable; appearance, geometry,
    # lineage, and near-ceiling-floor failures still stop immediately.
    for _attempt in range(_MAX_EXPORT_FACE_FEEDBACK_ATTEMPTS):
        request_history.append(current_request)
        (
            tracking_vertices,
            tracking_faces,
            tracking_uv,
            source_face_uv,
        ) = _replay_uv(
            selected_vertices,
            faces,
            uv,
            target_faces=int(target_faces),
            requested_faces=current_request,
        )
        tracking_mesh = trimesh.Trimesh(
            vertices=tracking_vertices,
            faces=tracking_faces,
            visual=trimesh.visual.texture.TextureVisuals(
                uv=tracking_uv,
                material=source_material,
            ),
            process=False,
        )

        # TemporaryDirectory cleanup covers export, reload, quality, JSON, and
        # replace failures. Suffixes remain .glb/.obj so exporters never guess
        # a format from a generic ".tmp" extension.
        with tempfile.TemporaryDirectory(
            dir=step_dir, prefix=f".{name}-tracking-"
        ) as temporary:
            temporary_dir = Path(temporary)
            temporary_tracking = temporary_dir / f"{name}_tracking.glb"
            temporary_visual = temporary_dir / f"{name}_visual.obj"
            temporary_metadata = temporary_dir / f"{name}_tracking.json"

            tracking_mesh.export(temporary_tracking)
            _write_textured_obj(
                temporary_visual,
                vertices=selected_vertices,
                faces=faces,
                uv=uv,
            )
            try:
                quality = _validate_export(
                    source_mesh=source_mesh,
                    expected_tracking_mesh=tracking_mesh,
                    source_face_uv=source_face_uv,
                    tracking_path=temporary_tracking,
                    visual_path=temporary_visual,
                    source_texture=texture,
                    target_faces=int(target_faces),
                )
            except _TrackingFaceCeilingExceeded as exc:
                reload_face_history.append(exc.actual_faces)
                current_request = _next_tracking_face_request(
                    current_request=current_request,
                    reloaded_faces=exc.actual_faces,
                    ceiling=ceiling,
                    near_ceiling_floor=near_ceiling_floor,
                )
                continue

            reloaded_tracking = _load_mesh(temporary_tracking)
            reload_face_history.append(len(reloaded_tracking.faces))
            metadata = {
                "schema": TRACKING_MESH_SCHEMA,
                "algorithm": _ALGORITHM,
                "source_metric_mesh": str(Path(metric_obj).resolve()),
                "source_metric_mesh_sha256": sha256_file(metric_obj),
                "source_texture": str(Path(metric_texture).resolve()),
                "source_texture_sha256": sha256_file(metric_texture),
                "sized_mesh": (
                    str(Path(sized_mesh).resolve())
                    if sized_mesh is not None and Path(sized_mesh).is_file()
                    else None
                ),
                "sized_mesh_sha256": (
                    sha256_file(sized_mesh)
                    if sized_mesh is not None and Path(sized_mesh).is_file()
                    else None
                ),
                "selected_geometry": selected_source,
                "sized_to_metric_extent_ratio": [float(v) for v in sized_ratio],
                "target_faces": int(target_faces),
                "simplification_requested_faces": current_request,
                "simplification_request_history": request_history.copy(),
                "export_reload_face_history": reload_face_history.copy(),
                "validated_near_ceiling_floor": near_ceiling_floor,
                "source_vertices": int(len(source_mesh.vertices)),
                "source_faces": int(len(source_mesh.faces)),
                "tracking_vertices": int(len(reloaded_tracking.vertices)),
                "tracking_faces": int(len(reloaded_tracking.faces)),
                # Paths are diagnostic only; hashes below are the portable contract.
                "tracking_mesh": str(tracking_path.resolve()),
                "tracking_mesh_sha256": sha256_file(temporary_tracking),
                "visual_mesh": str(visual_path.resolve()),
                "visual_mesh_sha256": sha256_file(temporary_visual),
                "texture_embedded_in_tracking_glb": True,
                "manual_alignment": False,
                "quality_gate": quality,
            }
            write_json_atomic(temporary_metadata, metadata)
            # Parse the generated JSON before promotion as well. Source/output
            # hashes are checked here and again against canonical artifacts.
            parsed = json.loads(temporary_metadata.read_text(encoding="utf-8"))
            if parsed != metadata:
                raise RuntimeError("tracking metadata changed during serialization")

            os.replace(temporary_tracking, tracking_path)
            os.replace(temporary_visual, visual_path)
            os.replace(temporary_metadata, metadata_path)
        break
    else:
        raise RuntimeError(
            "FoundationPose tracking mesh exceeded the face ceiling after "
            f"{_MAX_EXPORT_FACE_FEEDBACK_ATTEMPTS} deterministic attempts"
        )

    if metadata is None:
        raise RuntimeError("tracking generation ended without validated metadata")

    validate_tracking_release(
        metadata_path,
        metric_obj=Path(metric_obj),
        metric_texture=Path(metric_texture),
        tracking_mesh=tracking_path,
        visual_mesh=visual_path,
        sized_mesh=sized_mesh,
    )
    logger.info(
        "[tracking] %s: %d -> %d faces, geometry=%s, p95=%.5f diagonal, "
        "texture-p95=%.5f",
        name,
        len(source_mesh.faces),
        len(tracking_mesh.faces),
        selected_source,
        quality["surface_p95_diagonal_fraction"],
        quality["source_vs_mapped_texture_color_p95_error"],
    )
    return metadata
