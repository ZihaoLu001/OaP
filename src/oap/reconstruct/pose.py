"""Step POSE: Any6D joint 6-DoF pose + metric size, re-based through calibration.

Role in the two-stage pipeline: recovers the object's real-world pose (and a
refined metric size) that Stage 2 restores the twin from. The GPU half --
Any6D coarse-OBB init + FoundationPose render-and-compare joint pose+size on
the metric SAM3D mesh, with the OCCLUDED-AXIS GUARD (each object axis clamped
UP to the observed depth-OBB extent, single-view occlusion can only under-read)
and ``size_LWH_m`` taken from the guarded mesh's own AABB (a depth-OBB axis
projection over-reads a rotated footprint by up to ~1.4x) -- runs in the
external foundationpose env via ``payloads/any6d_pose_scale.py``.

The in-process half then applies the CAMERA->BASE transform: capture records
``T_base_camera = identity`` (camera-frame-only provenance), so this step
loads the packaged hand-eye calibration ``configs/calibration/T_base_zed2i.yaml``
and recomputes ``pos_base`` / ``quat_wxyz`` / ``yaw_base_rad``, writing the
final ``<obj>_refined.json`` (schema ``any6d_foundationpose_pose_scale_v1``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from oap.reconstruct.external import ExternalEnvs
from oap.reconstruct.quality import QUALITY_PROFILE
from oap.utils.io import (
    load_yaml,
    package_config_path,
    read_json,
    sha256_file,
    write_json_atomic,
)
from oap.utils.se3 import mat_to_quat, validate_se3_matrix

logger = logging.getLogger("oap.reconstruct.pose")

__all__ = [
    "load_fixed_camera_calibration",
    "load_t_base_camera",
    "normalize_camera_serial",
    "rebase_pose",
    "run_pose_step",
    "validate_refined_pose_quality",
]

_CALIB_RELATIVE = "calibration/T_base_zed2i.yaml"


def normalize_camera_serial(value: object) -> str:
    """Return the canonical numeric ZED serial used by capture + calibration."""
    if isinstance(value, bool):
        raise ValueError("camera serial must be a numeric device serial, not bool")
    serial = str(value).strip()
    if serial.upper().startswith("SN"):
        serial = serial[2:].strip()
    if not serial or not serial.isdecimal():
        raise ValueError(
            f"camera serial must contain decimal digits, got {value!r}"
        )
    return serial.lstrip("0") or "0"


def _load_calibration_payload(
    path: Path | None,
) -> tuple[np.ndarray, Path, Any]:
    """Read and validate one calibration document."""
    src = (
        Path(path)
        if path is not None
        else package_config_path(_CALIB_RELATIVE)
    )
    payload = load_yaml(src)
    node: Any = payload
    if isinstance(payload, dict):
        node = payload.get("T_base_zed2i", payload)
        if isinstance(node, dict):
            node = node.get("matrix", node.get("T_base_camera_opencv", node))
    matrix = validate_se3_matrix(
        node,
        name=f"base<-camera calibration {src}",
    )
    return matrix, src, payload


def load_t_base_camera(path: Path | None = None) -> tuple[np.ndarray, str]:
    """Load a base<-camera transform for inspection or calibration tests.

    Defaults to the packaged calibration ``configs/calibration/T_base_zed2i.yaml``
    (the current eye-to-hand ChArUco recalibration). An explicit YAML/JSON with
    either a ``T_base_zed2i.matrix`` node, a bare ``matrix`` key, a
    ``T_base_camera_opencv`` key, or a raw 4x4 array is accepted by this
    low-level loader for calibration tooling and tests. The production
    reconstruction path calls :func:`load_fixed_camera_calibration` instead
    and exposes no per-object override.

    Args:
        path: Optional override file.

    Returns:
        ``(T_base_camera, provenance)`` where the matrix maps camera-frame
        points into the robot base frame (``p_base = T @ [p_cam; 1]``).

    Raises:
        FileNotFoundError: If the packaged calibration is not shipped.
        ValueError: If the file does not contain a 4x4 matrix.
    """
    matrix, src, _ = _load_calibration_payload(path)
    return matrix, str(src)


def load_fixed_camera_calibration() -> tuple[np.ndarray, str, str, str]:
    """Load the one fixed-rig calibration and its immutable provenance.

    Returns:
        ``(T_base_camera, source, camera_serial, sha256)``. The serial binds
        every RGB-D packet to the physical camera calibrated by the YAML, and
        the digest makes calibration content part of pose-step idempotency.

    Raises:
        ValueError: If the packaged calibration has no valid ``serial``.
    """
    matrix, src, payload = _load_calibration_payload(None)
    if not isinstance(payload, dict) or "serial" not in payload:
        raise ValueError(
            f"fixed-camera calibration {src} has no top-level serial"
        )
    serial = normalize_camera_serial(payload["serial"])
    return matrix, str(src), serial, sha256_file(src)


def rebase_pose(raw: dict[str, Any], T_base_camera: np.ndarray) -> dict[str, Any]:
    """Recompute the base-frame pose fields from the camera-frame Any6D pose.

    The payload computes ``pos_base``/``quat_wxyz``/``yaw_base_rad`` with the
    manifest's ``T_base_camera`` (identity at capture time). Given the real
    calibration, rebuild ``T_base_obj = T_base_camera @ [R_cam_obj | center_cam]``
    and overwrite those fields (same rounding as the payload).

    Args:
        raw: The payload's output record (must carry ``R_cam_obj`` +
            ``center_cam_m``).
        T_base_camera: 4x4 base<-camera transform.

    Returns:
        A new record with the base-frame fields recomputed.
    """
    R_cam_obj = np.asarray(raw["R_cam_obj"], dtype=float).reshape(3, 3)
    center_cam = np.asarray(raw["center_cam_m"], dtype=float).reshape(3)
    T_co = np.eye(4)
    T_co[:3, :3] = R_cam_obj
    T_co[:3, 3] = center_cam
    T_bo = np.asarray(T_base_camera, dtype=float) @ T_co
    R_bo, p_bo = T_bo[:3, :3], T_bo[:3, 3]
    yaw = float(np.arctan2(R_bo[1, 0], R_bo[0, 0]))
    out = dict(raw)
    out["pos_base"] = [round(float(v), 5) for v in p_bo]
    out["quat_wxyz"] = [round(float(v), 6) for v in mat_to_quat(R_bo)]
    out["yaw_base_rad"] = round(yaw, 5)
    return out


def validate_refined_pose_quality(record: dict[str, Any]) -> dict[str, Any]:
    """Require measured silhouette and depth agreement for a refined pose.

    Any6D/FoundationPose already reports silhouette IoU and masked depth MAE
    after the occluded-axis guard. This function turns those diagnostics into
    one global fail-closed contract instead of leaving visual inspection or an
    object-specific repair step to decide whether a pose is executable.
    """
    try:
        iou = float(record["iou_after_guard"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "refined pose has no finite iou_after_guard quality evidence"
        ) from exc
    try:
        depth_mae_m = float(record["depth_mae_m_after_guard"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "refined pose has no finite depth_mae_m_after_guard quality evidence"
        ) from exc
    if not np.isfinite(iou) or not 0.0 <= iou <= 1.0:
        raise RuntimeError(
            f"refined pose has invalid iou_after_guard {iou!r}"
        )
    if not np.isfinite(depth_mae_m) or depth_mae_m < 0.0:
        raise RuntimeError(
            "refined pose has invalid depth_mae_m_after_guard "
            f"{depth_mae_m!r}"
        )
    failures: list[str] = []
    if iou < QUALITY_PROFILE.min_refined_pose_iou:
        failures.append(
            f"IoU {iou:.4f} < "
            f"{QUALITY_PROFILE.min_refined_pose_iou:.4f}"
        )
    if depth_mae_m > QUALITY_PROFILE.max_refined_pose_depth_mae_m:
        failures.append(
            f"depth MAE {depth_mae_m:.4f} m > "
            f"{QUALITY_PROFILE.max_refined_pose_depth_mae_m:.4f} m"
        )
    if failures:
        raise RuntimeError(
            "refined pose failed the global render-and-compare quality gate: "
            + "; ".join(failures)
        )
    return {
        "schema": "oap_refined_pose_quality_v1",
        "passed": True,
        "quality_profile": QUALITY_PROFILE.to_dict(),
        "iou_after_guard": iou,
        "depth_mae_m_after_guard": depth_mae_m,
    }


def run_pose_step(
    step_dir: Path,
    packet_dir: Path,
    metric_obj: Path,
    envs: ExternalEnvs,
    *,
    name: str,
    downscale: float = 0.5,
    iterations: int = 5,
) -> dict[str, Any]:
    """Run Any6D pose+scale for one object and write ``<name>_refined.json``.

    The payload runs with the Any6D checkout on ``PYTHONPATH`` (from the
    ``foundationpose`` tool's ``any6d_root``) and the 16GB memory recipe
    (``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``; keep
    ``downscale=0.5`` on a 16GB GPU).

    Args:
        step_dir: This step's output dir (``<bundle>/<obj>/pose``); receives
            the raw payload output, an ``overlay/`` render-and-compare dir
            (overlay PNG + the Any6D-sized mesh), and the final refined JSON.
        packet_dir: The imported observation packet.
        metric_obj: The metric visual mesh from the scale step.
        envs: Resolved external environments (needs ``foundationpose``).
        name: Object name (output file stem).
        downscale: Payload RGB-D downscale factor.
        iterations: Any6D refine iterations.

    Returns:
        The refined pose record (schema ``any6d_foundationpose_pose_scale_v1``).
    """
    step_dir = Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    raw_out = step_dir / f"{name}_any6d_raw.json"
    overlay_dir = step_dir / "overlay"
    refined_path = step_dir / f"{name}_refined.json"
    # A forced/restarted pose solve must not leave a previously accepted pose
    # available if the new measurement or GPU refinement fails.
    refined_path.unlink(missing_ok=True)

    any6d_root = envs.extra("foundationpose", "any6d_root")
    assert any6d_root is not None  # extra(required=True) raises when missing
    extra_pythonpath = [any6d_root, str(Path(any6d_root) / "foundationpose")]
    envs.run_payload(
        "foundationpose",
        "any6d_pose_scale",
        [
            "--packet", Path(packet_dir),
            "--mesh", Path(metric_obj),
            "--out", raw_out,
            "--downscale", str(downscale),
            "--iter", str(iterations),
            "--overlay-dir", overlay_dir,
        ],
        extra_pythonpath=extra_pythonpath,
        extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    raw = read_json(raw_out)
    quality_gate = validate_refined_pose_quality(raw)
    T_bc, calib_src, camera_serial, calib_sha256 = (
        load_fixed_camera_calibration()
    )
    refined = rebase_pose(raw, T_bc)
    refined["t_base_camera_source"] = calib_src
    refined["t_base_camera_sha256"] = calib_sha256
    refined["camera_serial"] = camera_serial
    refined["t_base_camera"] = np.asarray(T_bc, dtype=float).round(8).tolist()
    refined["quality_gate"] = quality_gate

    write_json_atomic(refined_path, refined)
    logger.info(
        "[pose] %s: pos_base=%s yaw=%.3f rad size_LWH=%s iou=%.3f (calib %s)",
        name,
        refined.get("pos_base"),
        float(refined.get("yaw_base_rad", 0.0)),
        refined.get("size_LWH_m"),
        float(refined.get("iou_after_guard", 0.0)),
        calib_src,
    )
    return refined
