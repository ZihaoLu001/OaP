"""Step SCALE: fixed-camera metric RGB-D + one isotropic mesh scale."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from oap.reconstruct.metric_scale import build_scale_report
from oap.utils.io import read_json

logger = logging.getLogger("oap.reconstruct.scale")

__all__ = [
    "build_scale_report",
    "convert_mesh_to_metric_obj",
    "run_scale_step",
    "read_scale_size",
]


def _load_as_mesh(path: Path) -> Any:
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type {type(mesh)!r}: {path}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(f"mesh has no usable triangle geometry: {path}")
    return mesh


def _save_texture(mesh: Any, out_obj: Path) -> tuple[Path | None, bool]:
    """Write the baked UV texture to the deterministic MuJoCo sibling path."""

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    has_uv = uv is not None and np.asarray(uv).size > 0
    material = getattr(visual, "material", None)
    image = None
    if material is not None:
        image = (
            getattr(material, "baseColorTexture", None)
            or getattr(material, "image", None)
        )
    if image is None or not has_uv:
        return None, bool(has_uv)
    png = out_obj.with_name(out_obj.stem + "_texture.png")
    image.save(png)
    return png, True


def convert_mesh_to_metric_obj(
    mesh_path: Path,
    out_obj: Path,
    target_size_m: tuple[float, float, float] | None,
    *,
    scale_factor: float | None = None,
    metadata_out: Path | None = None,
    min_extent: float = 1e-6,
) -> dict[str, Any]:
    """Scale a SAM3D mesh while preserving UVs and baked appearance.

    Production supplies ``scale_factor`` and therefore cannot alter the
    reconstruction's shape. ``target_size_m`` remains only for compatibility
    with older offline asset-conversion callers.
    """

    import trimesh

    mesh = _load_as_mesh(Path(mesh_path))
    bounds = np.asarray(mesh.bounds, dtype=float)
    center = bounds.mean(axis=0)
    extent = np.maximum(bounds[1] - bounds[0], float(min_extent))
    if (target_size_m is None) == (scale_factor is None):
        raise ValueError(
            "provide exactly one of target_size_m or scale_factor"
        )
    if scale_factor is not None:
        uniform_scale = float(scale_factor)
        if not np.isfinite(uniform_scale) or uniform_scale <= 0.0:
            raise ValueError(
                f"scale_factor must be finite and positive, got {scale_factor}"
            )
        target = extent * uniform_scale
        scale_xyz = np.full(3, uniform_scale, dtype=float)
        scaling_mode = "isotropic"
    else:
        target = np.asarray(target_size_m, dtype=float)
        if np.any(target <= 0.0):
            raise ValueError(
                f"target-size-m must be positive, got {target.tolist()}"
            )
        scale_xyz = target / extent
        scaling_mode = "legacy_axis_fit"

    vertices = (np.asarray(mesh.vertices, dtype=float) - center) * scale_xyz
    metric_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(mesh.faces),
        visual=getattr(mesh, "visual", None),
        process=False,
    )
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    metric_mesh.export(out_obj)
    texture_png, has_uv = _save_texture(metric_mesh, out_obj)
    metadata = {
        "schema": "sam3d_metric_visual_mesh_v2",
        "source_mesh": str(Path(mesh_path).resolve()),
        "out_obj": str(out_obj.resolve()),
        "texture_png": (
            str(texture_png.resolve()) if texture_png is not None else None
        ),
        "has_uv": bool(has_uv),
        "source_bounds": bounds.tolist(),
        "source_extent": extent.tolist(),
        "target_size_m": target.tolist(),
        "scaling_mode": scaling_mode,
        "scale_factor": (
            float(scale_factor) if scale_factor is not None else None
        ),
        "scale_xyz": scale_xyz.tolist(),
        "policy": (
            "metric fit preserving SAM3D baked appearance; production uses "
            "one isotropic scale and collision/physics are derived downstream"
        ),
    }
    metadata_path = metadata_out or out_obj.with_suffix(".metadata.json")
    Path(metadata_path).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def run_scale_step(
    step_dir: Path,
    packet_dir: Path,
    mesh_step_dir: Path,
    *,
    name: str,
) -> dict[str, Any]:
    """Build the RGB-D report, quality-gate it, then scale the mesh uniformly."""

    step_dir = Path(step_dir)
    packet_dir = Path(packet_dir)
    sam3d_output = Path(mesh_step_dir) / "sam3d_output"
    step_dir.mkdir(parents=True, exist_ok=True)

    metric_obj = step_dir / f"{name}_metric.obj"
    metric_obj.unlink(missing_ok=True)
    metric_obj.with_suffix(".metadata.json").unlink(missing_ok=True)
    metric_obj.with_name(metric_obj.stem + "_texture.png").unlink(
        missing_ok=True
    )

    source_mesh = sam3d_output / "mesh.glb"
    if not source_mesh.exists():
        source_mesh = sam3d_output / "mesh_raw.obj"
    if not source_mesh.exists():
        raise FileNotFoundError(
            f"mesh step output has neither mesh.glb nor mesh_raw.obj in "
            f"{sam3d_output}"
        )

    report = build_scale_report(
        packet_dir / "observation_manifest.json",
        mesh=source_mesh,
        out=step_dir / "scale_report.json",
    )
    if not bool(report.get("scale_reportable")):
        reasons = report.get("scale_rejection_reasons") or ["unknown"]
        raise RuntimeError(
            f"scale reconstruction for {name!r} failed the global quality "
            f"gate ({'; '.join(str(reason) for reason in reasons)}); no "
            f"executable metric asset was written"
        )
    size = report.get("size_xyz_m")
    if not size or len(size) != 3:
        raise RuntimeError(
            f"scale report for {name!r} has no size_xyz_m: "
            f"{step_dir / 'scale_report.json'}"
        )
    metric = convert_mesh_to_metric_obj(
        source_mesh,
        metric_obj,
        None,
        scale_factor=float(report["scale_factor"]),
    )
    logger.info(
        "[scale] %s: decision=%s uniform_scale=%.9f size=%s",
        name,
        report.get("scale_decision"),
        float(report["scale_factor"]),
        size,
    )
    return {"scale_report": report, "metric": metric}


def read_scale_size(step_dir: Path) -> tuple[float, float, float]:
    """Read ``size_xyz_m`` back from a completed scale step."""

    report = read_json(Path(step_dir) / "scale_report.json")
    size = report.get("size_xyz_m")
    if not size or len(size) != 3:
        raise RuntimeError(
            f"missing size_xyz_m in {Path(step_dir) / 'scale_report.json'}"
        )
    return (float(size[0]), float(size[1]), float(size[2]))
