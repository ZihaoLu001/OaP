"""Load and validate scene-bundle manifests (real2sim2real_scene_manifest_v1).

Role in the two-stage pipeline: ``oap-reconstruct bundle`` writes a
manifest listing every reconstructed object (metric mesh + refined size/axes +
physical priors + SAM3 prompt); ``oap-run`` loads it here into derived
:class:`~oap.twin.types.SceneObject` records that the twin builder
consumes. Production manifests are generated-only: every core artifact must
use its canonical bundle-relative path and match the SHA-256 recorded by the
reconstruction pipeline. Post-hoc pose, axis, scale, mesh, or collision edits
fail closed before the twin is built.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from oap.reconstruct.pose import load_fixed_camera_calibration
from oap.twin.obb import (
    mesh_axis_rotation,
    object_vertical_half_extent_from_q_om,
    project_unheld_pose_to_support,
)
from oap.twin.types import (
    LAB_TABLE_POS,
    LAB_TABLE_SIZE,
    TABLE_TOP_Z,
    SceneObject,
    load_physical_priors,
    mujoco_friction_from_priors,
)
from oap.utils.io import sha256_file
from oap.utils.se3 import mat_to_quat

logger = logging.getLogger("oap.twin.manifest")

MANIFEST_SCHEMA = "real2sim2real_scene_manifest_v1"

__all__ = ["MANIFEST_SCHEMA", "SceneManifestError", "load_scene_manifest"]


class SceneManifestError(RuntimeError):
    """A scene manifest is malformed or references missing files."""


_CORE_ARTIFACT_ROLES = (
    "mesh",
    "visual_mesh",
    "texture",
    "tracking_metadata",
    "collision",
    "refined_json",
    "priors_json",
)
_FIXED_CAMERA_BUNDLE_PATH = "calibration/T_base_zed2i.yaml"
_MANUAL_VALUE_KEYS = {
    "adjusted_by",
    "history",
    "method",
    "note",
    "notes",
    "provenance",
    "source",
}
_MANUAL_VALUE_MARKERS = (
    "caliper",
    "hand-adjust",
    "hand adjusted",
    "human-adjust",
    "human adjusted",
    "manual",
    "post-hoc",
    "post hoc",
    "posthoc",
    "user-adjust",
    "user adjusted",
)


def _generated_only_error(location: str, detail: str) -> SceneManifestError:
    return SceneManifestError(
        f"{location}: {detail}. Production scene bundles are generated-only; "
        "discard this bundle and re-run `oap-reconstruct all` instead of "
        "editing object pose, axes, scale, mesh, or collision artifacts."
    )


def _reject_manual_provenance(value: Any, *, location: str) -> None:
    """Reject explicit evidence of post-hoc human geometry adjustment."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
            if (
                key == "rescaled_by"
                or "manual" in key
                or key.endswith("_override")
                or key in {"post_hoc", "posthoc", "hand_adjusted_by", "human_adjusted_by"}
            ):
                raise _generated_only_error(
                    location, f"forbidden manual-provenance field {raw_key!r}"
                )
            if key in _MANUAL_VALUE_KEYS and isinstance(child, str):
                text = child.strip().lower().replace("_", " ")
                if any(marker in text for marker in _MANUAL_VALUE_MARKERS):
                    raise _generated_only_error(
                        location,
                        f"forbidden manual-provenance value in field {raw_key!r}",
                    )
            _reject_manual_provenance(child, location=location)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_manual_provenance(child, location=location)


def _require_file(path: Path, *, role: str, obj_name: str, manifest: Path) -> Path:
    """Return ``path`` if it exists, else raise an actionable error."""
    if not path.is_file():
        raise SceneManifestError(
            f"scene manifest {manifest} object {obj_name!r}: {role} file not found "
            f"at {path}. Bundle paths are resolved relative to the manifest "
            f"directory -- re-run `oap-reconstruct all` for this bundle or "
            f"sync the bundle payload (tools/sync_data.sh)."
        )
    return path


def _canonical_asset(
    raw_path: Any,
    *,
    role: str,
    obj_name: str,
    manifest: Path,
    allowed: set[str],
) -> tuple[str, Path]:
    """Resolve one artifact only when it has a canonical generated path."""
    if not isinstance(raw_path, str) or not raw_path:
        raise _generated_only_error(
            f"scene manifest {manifest} object {obj_name!r}",
            f"{role} must be a non-empty canonical bundle-relative path",
        )
    posix = PurePosixPath(raw_path)
    if (
        "\\" in raw_path
        or posix.is_absolute()
        or raw_path != posix.as_posix()
        or any(part in {"", ".", ".."} for part in posix.parts)
        or raw_path not in allowed
    ):
        expected = ", ".join(sorted(allowed))
        raise _generated_only_error(
            f"scene manifest {manifest} object {obj_name!r}",
            f"non-canonical {role} path {raw_path!r}; expected one of: {expected}",
        )

    base = manifest.parent.resolve()
    resolved = (base / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise _generated_only_error(
            f"scene manifest {manifest} object {obj_name!r}",
            f"{role} path escapes the bundle: {raw_path!r}",
        ) from exc
    return raw_path, _require_file(
        resolved, role=role, obj_name=obj_name, manifest=manifest
    )


def _allowed_paths(name: str) -> dict[str, set[str]]:
    return {
        "mesh": {
            f"{name}/tracking/{name}_tracking.glb",
        },
        "collision": {
            f"{name}/collision/{name}_collision.obj",
            f"{name}/pose/{name}_collision_sized.obj",
        },
        "refined_json": {f"{name}/pose/{name}_refined.json"},
        "priors_json": {f"{name}/priors/{name}_priors.json"},
        "visual_mesh": {
            f"{name}/tracking/{name}_visual.obj",
        },
        "texture": {f"{name}/scale/{name}_metric_texture.png"},
        "tracking_metadata": {
            f"{name}/tracking/{name}_tracking.json",
        },
    }


def _verify_artifact_hashes(
    entry: Mapping[str, Any],
    *,
    raw_paths: Mapping[str, str],
    paths: Mapping[str, Path],
    obj_name: str,
    manifest: Path,
) -> None:
    """Verify every executable/render artifact against portable digests."""
    location = f"scene manifest {manifest} object {obj_name!r}"
    hashes = entry.get("artifact_sha256")
    expected_keys = {raw_paths[role] for role in _CORE_ARTIFACT_ROLES}
    if not isinstance(hashes, Mapping) or set(hashes) != expected_keys:
        raise _generated_only_error(
            location,
            "artifact_sha256 must map exactly the canonical generated artifact "
            f"paths {sorted(expected_keys)!r}",
        )
    for role in _CORE_ARTIFACT_ROLES:
        rel = raw_paths[role]
        expected = hashes[rel]
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(ch not in "0123456789abcdef" for ch in expected)
        ):
            raise _generated_only_error(
                location, f"invalid SHA-256 for {role} at {rel!r}"
            )
        actual = sha256_file(paths[role])
        if actual != expected:
            raise _generated_only_error(
                location,
                f"SHA-256 mismatch for {role} at {rel!r}: expected "
                f"{expected}, observed {actual}",
            )


def _validate_camera_calibration(
    raw: Mapping[str, Any],
    *,
    manifest: Path,
) -> tuple[str, str]:
    """Bind a portable bundle to the one packaged fixed-camera calibration."""
    _, _, fixed_serial, fixed_sha256 = load_fixed_camera_calibration()
    recorded = raw.get("camera_calibration")
    if not isinstance(recorded, Mapping):
        raise _generated_only_error(
            f"scene manifest {manifest}",
            "missing top-level camera_calibration provenance",
        )
    observed = {
        "mode": recorded.get("mode"),
        "camera_serial": recorded.get("camera_serial"),
        "sha256": recorded.get("sha256"),
        "config": recorded.get("config"),
    }
    expected = {
        "mode": "fixed_eye_to_hand",
        "camera_serial": fixed_serial,
        "sha256": fixed_sha256,
        "config": _FIXED_CAMERA_BUNDLE_PATH,
    }
    if observed != expected:
        raise _generated_only_error(
            f"scene manifest {manifest}",
            f"camera_calibration mismatch: expected {expected!r}, observed "
            f"{observed!r}",
        )
    bundled = manifest.parent / _FIXED_CAMERA_BUNDLE_PATH
    if not bundled.is_file():
        raise _generated_only_error(
            f"scene manifest {manifest}",
            "bundled fixed-camera calibration is missing at "
            f"{_FIXED_CAMERA_BUNDLE_PATH!r}",
        )
    bundled_sha256 = sha256_file(bundled)
    if bundled_sha256 != fixed_sha256:
        raise _generated_only_error(
            f"scene manifest {manifest}",
            "bundled fixed-camera calibration SHA-256 mismatch: expected "
            f"{fixed_sha256}, observed {bundled_sha256}",
        )
    return fixed_serial, fixed_sha256


def load_scene_manifest(
    path: Path, *, z_real_to_plan: float | None = None,
) -> list[SceneObject]:
    """Parse a scene-bundle manifest into derived :class:`SceneObject` records.

    Per object this: validates canonical generated paths and artifact hashes;
    reads the refined metric size + mesh->object axis map (q_om); loads
    Phys2Real-style priors into mass/friction; validates the integrity-bound
    tracking metadata and visual texture; restores the twin pose exclusively
    from the reconstruction's ``refined_json``; maps it into the plan frame;
    and raises geometry below the known table support without lowering
    elevated objects. The reconstruction artifacts themselves remain unchanged.

    Args:
        path: The manifest JSON path (schema ``real2sim2real_scene_manifest_v1``).
        z_real_to_plan: Translation from the reconstruction's robot-base frame
            into the plan frame, shared with the robot and camera. If unknown,
            leave the raw poses unchanged until the caller obtains it.

    Returns:
        The scene's objects, at least one of which is movable. Poses remain in
        robot-base coordinates unless ``z_real_to_plan`` is supplied.

    Raises:
        SceneManifestError: On a wrong schema, non-canonical path, missing or
            modified artifact, manual provenance, missing reconstruction pose,
            or a scene with no movable object.
    """
    path = Path(path)
    if not path.exists():
        raise SceneManifestError(f"scene manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != MANIFEST_SCHEMA:
        raise SceneManifestError(
            f"unsupported scene manifest schema: {raw.get('schema')!r} "
            f"(expected {MANIFEST_SCHEMA!r}) in {path}")
    fixed_camera_serial, fixed_calibration_sha256 = _validate_camera_calibration(
        raw, manifest=path
    )
    _reject_manual_provenance(raw, location=f"scene manifest {path}")
    objects: list[SceneObject] = []

    for o in raw.get("objects", []):
        name = str(o["name"])
        if not name or name in {".", ".."} or PurePosixPath(name).name != name or "\\" in name:
            raise _generated_only_error(
                f"scene manifest {path}", f"invalid object name {name!r}"
            )
        if "pose_json" in o:
            raise _generated_only_error(
                f"scene manifest {path} object {name!r}",
                "pose_json overrides are forbidden",
            )
        allowed = _allowed_paths(name)
        raw_paths: dict[str, str] = {}
        core_paths: dict[str, Path] = {}
        for role in _CORE_ARTIFACT_ROLES:
            raw_paths[role], core_paths[role] = _canonical_asset(
                o.get(role),
                role=role,
                obj_name=name,
                manifest=path,
                allowed=allowed[role],
            )
        _verify_artifact_hashes(
            o,
            raw_paths=raw_paths,
            paths=core_paths,
            obj_name=name,
            manifest=path,
        )
        # Artifact roles are not interchangeable: `mesh` is the automatic,
        # self-contained, decimated FoundationPose tracking GLB; `visual_mesh`
        # is the full-resolution render-only OBJ; and `collision` is the
        # physics proxy.
        obj = SceneObject(
            name=name,
            label=str(o["label"]),
            sam3_prompt=str(o.get("sam3_prompt", o["label"])),
            mesh=core_paths["mesh"],
            visual_mesh=core_paths["visual_mesh"],
            tracking_metadata=core_paths["tracking_metadata"],
            collision=core_paths["collision"],
            refined_json=core_paths["refined_json"],
            priors_json=core_paths["priors_json"],
            movable=bool(o.get("movable", False)),
            rgba=tuple(o.get("rgba", (0.5, 0.5, 0.5, 1.0))),
            open_top_container=bool(o.get("open_top_container", False)),
            wall_thickness_m=float(o.get("wall_thickness_m", 0.008)),
        )
        # Tracking GLBs embed their generated UV texture/material. The
        # full-resolution OBJ stores automatically sampled vertex colours and
        # is deliberately self-contained as well.
        obj.texture = core_paths["texture"]
        metric_source = (
            path.parent / name / "scale" / f"{name}_metric.obj"
        ).resolve()
        sized_source = (
            path.parent
            / name
            / "pose"
            / "overlay"
            / f"{name}_metric_any6d_sized_mesh.obj"
        ).resolve()
        from oap.reconstruct.tracking import validate_tracking_release

        try:
            validate_tracking_release(
                core_paths["tracking_metadata"],
                metric_obj=_require_file(
                    metric_source,
                    role="tracking source metric mesh",
                    obj_name=name,
                    manifest=path,
                ),
                metric_texture=core_paths["texture"],
                tracking_mesh=core_paths["mesh"],
                visual_mesh=core_paths["visual_mesh"],
                sized_mesh=sized_source if sized_source.is_file() else None,
            )
        except RuntimeError as exc:
            raise _generated_only_error(
                f"scene manifest {path} object {name!r}",
                f"tracking release metadata is stale or invalid: {exc}",
            ) from exc
        ref = json.loads(obj.refined_json.read_text(encoding="utf-8"))
        _reject_manual_provenance(ref, location=f"refined artifact {obj.refined_json}")
        if (
            str(ref.get("camera_serial", "")) != fixed_camera_serial
            or ref.get("t_base_camera_sha256") != fixed_calibration_sha256
        ):
            raise _generated_only_error(
                f"refined artifact {obj.refined_json}",
                "camera provenance does not match the current packaged fixed "
                f"calibration (serial {fixed_camera_serial}, SHA-256 "
                f"{fixed_calibration_sha256})",
            )
        priors_raw = json.loads(obj.priors_json.read_text(encoding="utf-8"))
        _reject_manual_provenance(
            priors_raw, location=f"physical-priors artifact {obj.priors_json}"
        )
        lwh = ref["size_LWH_m"]
        obj.size_lwh = (float(lwh[0]), float(lwh[1]), float(lwh[2]))
        obj.q_om = mat_to_quat(mesh_axis_rotation(ref["assignment"]))
        axis = np.zeros(3)
        axis[ref["assignment"][0]] = 1.0
        obj.mesh_long_axis = axis
        obj.priors = load_physical_priors(
            obj.priors_json, size_m=obj.size_lwh, object_label=obj.label,
            wall_thickness_m=obj.wall_thickness_m)
        obj.mass_kg = float(obj.priors.mass_kg["nominal"])
        obj.friction = mujoco_friction_from_priors(obj.priors)
        # Real2Sim2Real principle: the twin pose is RESTORED FROM ITS OWN
        # RECONSTRUCTION -- the refined base-frame pose in refined_json
        # (pos_base + quat_wxyz). Runtime tracking updates the live state;
        # production manifests do not accept a separate pose override.
        if ref.get("pos_base") is not None:
            obj.pose_base = [float(v) for v in ref["pos_base"]]
            obj.pose_quat = [float(v) for v in ref.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])]
            if ref.get("yaw_base_rad") is not None:
                obj.yaw_ref_rad = float(ref["yaw_base_rad"])
        if not obj.movable and obj.pose_base is None:
            raise SceneManifestError(
                f"static object {obj.name!r} has no reconstructed pos_base in "
                f"{obj.refined_json.name}; discard and reconstruct the bundle")
        objects.append(obj)
    if not any(o.movable for o in objects):
        raise SceneManifestError(f"scene manifest {path} declares no movable object")

    if z_real_to_plan is not None:
        _project_objects_to_plan_support(objects, z_real_to_plan=z_real_to_plan)
    return objects


def _project_objects_to_plan_support(
    objects: list[SceneObject], *, z_real_to_plan: float,
) -> None:
    """Convert derived poses once, then apply the plan-frame support bound."""
    for obj in objects:
        if obj.pose_base is None:
            continue
        pos = np.asarray(obj.pose_base, dtype=float).copy()
        if not obj.z_snapped_plan_frame:
            pos[2] += float(z_real_to_plan)
        obj.pose_base = pos.tolist()
        obj.z_snapped_plan_frame = True

    # Apply the same physical-consistency seam used by initial observations
    # and post-prefix tracking.  This is intentionally a LOWER BOUND, not the
    # old exact table snap: elevated/off-table poses are preserved, while a
    # reconstruction that puts collision geometry through the known table is
    # lifted only enough to become support-consistent.  Contents of an
    # open-top container are excluded because their support is its inner floor.
    for obj in objects:
        if obj.pose_base is None:
            continue
        assert obj.pose_quat is not None and obj.q_om is not None
        container_footprints = [
            (
                container.name,
                tuple(container.pose_base[:2]),
                0.5 * float(max(container.size_lwh)),
            )
            for container in objects
            if (
                container is not obj
                and container.open_top_container
                and container.pose_base is not None
            )
        ]
        raw = np.asarray(obj.pose_base, dtype=float)
        projected, changed = project_unheld_pose_to_support(
            raw,
            obj.pose_quat,
            obj.size_lwh,
            obj.q_om,
            TABLE_TOP_Z,
            held=False,
            support_center_xy=LAB_TABLE_POS[:2],
            support_half_size_xy=LAB_TABLE_SIZE[:2],
            excluded_footprints=container_footprints,
        )
        if not changed:
            continue
        vhe = object_vertical_half_extent_from_q_om(
            obj.size_lwh,
            obj.q_om,
            obj.pose_quat,
        )
        logger.info(
            "[table-support] %s: z %.4f -> %.4f "
            "(orientation-aware lower bound; half-extent %.4f)",
            obj.name,
            float(raw[2]),
            float(projected[2]),
            vhe,
        )
        obj.pose_base = projected.tolist()
