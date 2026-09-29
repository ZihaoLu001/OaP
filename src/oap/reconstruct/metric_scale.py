"""Object-agnostic metric scale from one immutable fixed-camera RGB-D packet."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oap.reconstruct.pose import (
    load_fixed_camera_calibration,
    normalize_camera_serial,
)
from oap.reconstruct.quality import QUALITY_PROFILE
from oap.utils.io import read_json, sha256_file, write_json_atomic


def _resolve_manifest_path(manifest_path: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            f"{manifest_path} must contain non-empty packet path fields"
        )
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def _load_mesh(mesh_path: Path) -> Any:
    import trimesh

    mesh = trimesh.load(Path(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"unsupported mesh type {type(mesh)!r}: {mesh_path}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise RuntimeError(f"mesh has no triangle geometry: {mesh_path}")
    return mesh


def _sorted_oriented_extent(points_or_mesh: Any) -> np.ndarray:
    import trimesh

    _, extent = trimesh.bounds.oriented_bounds(
        points_or_mesh,
        angle_digits=None,
        ordered=True,
    )
    values = np.sort(np.asarray(extent, dtype=np.float64).reshape(3))[::-1]
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"oriented extent is degenerate: {values.tolist()}")
    return values


def _erode_mask_boundary(mask: np.ndarray, *, pixels: int) -> np.ndarray:
    """Discard segmentation-boundary pixels without splitting real surfaces."""

    interior = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(pixels)):
        eroded = np.zeros_like(interior)
        eroded[1:-1, 1:-1] = (
            interior[1:-1, 1:-1]
            & interior[:-2, 1:-1]
            & interior[2:, 1:-1]
            & interior[1:-1, :-2]
            & interior[1:-1, 2:]
        )
        interior = eroded
    return interior


def _backproject(
    support: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    T_base_camera: np.ndarray,
) -> np.ndarray:
    ys, xs = np.nonzero(support)
    z = depth[ys, xs].astype(np.float64)
    points_camera = np.column_stack(
        (
            (xs.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (ys.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    points_h = np.column_stack((points_camera, np.ones(len(points_camera))))
    return (T_base_camera @ points_h.T).T[:, :3]


def _gravity_aligned_extent(
    points: np.ndarray,
    *,
    trim_percent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Robust resting-frame extents: PCA in table XY and calibrated gravity Z."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"point cloud must be Nx3, got {points.shape}")
    if not 0.0 <= float(trim_percent) < 50.0:
        raise ValueError(f"trim_percent must be in [0, 50), got {trim_percent}")

    xy_center = np.median(points[:, :2], axis=0)
    centered_xy = points[:, :2] - xy_center
    orientation_low = np.percentile(
        centered_xy,
        float(trim_percent),
        axis=0,
    )
    orientation_high = np.percentile(
        centered_xy,
        100.0 - float(trim_percent),
        axis=0,
    )
    orientation_support = np.all(
        (centered_xy >= orientation_low)
        & (centered_xy <= orientation_high),
        axis=1,
    )
    if int(orientation_support.sum()) < 3:
        raise ValueError("too few points for robust gravity-frame orientation")
    _, _, xy_basis = np.linalg.svd(
        centered_xy[orientation_support],
        full_matrices=False,
    )
    frame_points = np.column_stack(
        (centered_xy @ xy_basis.T, points[:, 2])
    )
    low = np.percentile(frame_points, float(trim_percent), axis=0)
    high = np.percentile(
        frame_points,
        100.0 - float(trim_percent),
        axis=0,
    )
    extent = np.sort(np.asarray(high - low, dtype=np.float64))[::-1]
    if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
        raise ValueError(f"gravity-aligned extent is degenerate: {extent}")
    return extent, {
        "trim_percent_each_tail": float(trim_percent),
        "xy_center_m": xy_center.tolist(),
        "xy_basis": xy_basis.tolist(),
        "xy_orientation_support_points": int(orientation_support.sum()),
        "xy_orientation_support_fraction": float(
            orientation_support.mean()
        ),
        "xy_orientation_lower_quantile_m": orientation_low.tolist(),
        "xy_orientation_upper_quantile_m": orientation_high.tolist(),
        "lower_frame_quantile_m": low.tolist(),
        "upper_frame_quantile_m": high.tolist(),
        "gravity_axis_base": [0.0, 0.0, 1.0],
    }


def _uniform_scale(observed_extent: np.ndarray, mesh_extent: np.ndarray) -> tuple[float, np.ndarray]:
    ratios = observed_extent / mesh_extent
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise ValueError(f"invalid axis scale candidates: {ratios.tolist()}")
    return float(np.median(ratios)), ratios


def _fold(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.max() / values.min())


def _bootstrap_scale(
    points: np.ndarray,
    mesh_extent: np.ndarray,
    *,
    samples: int,
    seed: int,
    trim_percent: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selection = rng.integers(0, len(points), size=len(points))
        extent, _ = _gravity_aligned_extent(
            points[selection],
            trim_percent=trim_percent,
        )
        estimates[index], _ = _uniform_scale(extent, mesh_extent)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "samples": int(samples),
        "seed": int(seed),
        "scale_samples": [float(value) for value in estimates],
        "scale_mean": float(np.mean(estimates)),
        "scale_std": float(np.std(estimates)),
        "scale_cv": float(np.std(estimates) / np.mean(estimates)),
        "scale_ci95": [float(low), float(high)],
        "scale_ci95_fold": float(high / low),
    }


def build_scale_report(
    rgbd_observation_manifest: Path,
    *,
    mesh: Path,
    out: Path,
) -> dict[str, Any]:
    """Measure one isotropic mesh scale from fixed-camera metric RGB-D.

    All quality limits come from the immutable global reconstruction profile;
    this API intentionally exposes no tuning knob or per-object override.
    """

    manifest_path = Path(rgbd_observation_manifest)
    manifest = read_json(manifest_path)
    mask_path = _resolve_manifest_path(manifest_path, manifest.get("mask"))
    depth_path = _resolve_manifest_path(
        manifest_path,
        manifest.get("depth_m_npy"),
    )
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)

    T_base_camera, calibration_source, calibrated_serial, calibration_sha256 = (
        load_fixed_camera_calibration()
    )
    if "camera_serial" not in manifest:
        raise ValueError(f"{manifest_path} has no runtime camera_serial")
    packet_serial = normalize_camera_serial(manifest["camera_serial"])
    if packet_serial != calibrated_serial:
        raise ValueError(
            f"{manifest_path} camera serial {packet_serial} does not match "
            f"fixed calibration serial {calibrated_serial}"
        )

    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    depth = np.asarray(np.load(depth_path), dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError(
            f"mask/depth shape mismatch: {mask.shape} != {depth.shape}"
        )
    intrinsics = np.asarray(manifest.get("cam_K"), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"{manifest_path} cam_K must be 3x3")
    if (
        not np.all(np.isfinite(intrinsics))
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
    ):
        raise ValueError(f"{manifest_path} has invalid cam_K")

    valid = mask & np.isfinite(depth) & (depth > 0.0)
    valid_points = int(valid.sum())
    if valid_points < QUALITY_PROFILE.min_scale_points:
        raise ValueError(
            f"too few valid masked depth points: {valid_points} < "
            f"{QUALITY_PROFILE.min_scale_points}"
        )
    erosion_pixels = int(QUALITY_PROFILE.scale_boundary_probe_pixels)
    trim_percent = float(QUALITY_PROFILE.scale_extent_trim_percent)
    if erosion_pixels < 0:
        raise ValueError(
            f"scale_boundary_probe_pixels must be non-negative, got "
            f"{erosion_pixels}"
        )
    if not 0.0 < trim_percent < 25.0:
        raise ValueError(
            f"scale_extent_trim_percent must be in (0, 25), got "
            f"{trim_percent}"
        )
    outer_trim_percent = trim_percent / 2.0
    boundary_probe_mask = _erode_mask_boundary(
        mask,
        pixels=erosion_pixels,
    )
    boundary_probe = valid & boundary_probe_mask
    boundary_probe_points = int(boundary_probe.sum())
    if boundary_probe_points < QUALITY_PROFILE.min_scale_points:
        raise ValueError(
            "too few valid boundary-probe depth points: "
            f"{boundary_probe_points} < "
            f"{QUALITY_PROFILE.min_scale_points}"
        )
    boundary_probe_fraction = float(boundary_probe_points / valid_points)
    points = _backproject(valid, depth, intrinsics, T_base_camera)
    boundary_probe_cloud = _backproject(
        boundary_probe,
        depth,
        intrinsics,
        T_base_camera,
    )

    source_mesh = _load_mesh(Path(mesh))
    observed_extent, extent_frame = _gravity_aligned_extent(
        points,
        trim_percent=trim_percent,
    )
    outer_extent, outer_extent_frame = _gravity_aligned_extent(
        points,
        trim_percent=outer_trim_percent,
    )
    boundary_probe_extent, boundary_probe_extent_frame = (
        _gravity_aligned_extent(
            boundary_probe_cloud,
            trim_percent=trim_percent,
        )
    )
    mesh_extent = _sorted_oriented_extent(source_mesh)
    mesh_local_extent = np.asarray(
        source_mesh.bounds[1] - source_mesh.bounds[0],
        dtype=np.float64,
    )
    scale_factor, axis_candidates = _uniform_scale(
        observed_extent,
        mesh_extent,
    )
    outer_scale, _ = _uniform_scale(outer_extent, mesh_extent)
    boundary_probe_scale, _ = _uniform_scale(
        boundary_probe_extent,
        mesh_extent,
    )
    axis_fold = _fold(axis_candidates)
    extent_sensitivity_fold = _fold(
        np.asarray(
            [
                outer_scale,
                scale_factor,
                boundary_probe_scale,
            ],
            dtype=np.float64,
        )
    )

    depth_hash = sha256_file(depth_path)
    mask_hash = sha256_file(mask_path)
    mesh_hash = sha256_file(Path(mesh))
    seed_material = f"{depth_hash}:{mask_hash}:{mesh_hash}".encode()
    bootstrap_seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8],
        byteorder="big",
        signed=False,
    )
    bootstrap = _bootstrap_scale(
        points,
        mesh_extent,
        samples=QUALITY_PROFILE.scale_bootstrap_samples,
        seed=bootstrap_seed,
        trim_percent=trim_percent,
    )

    reasons: list[str] = []
    if (
        boundary_probe_fraction
        <= QUALITY_PROFILE.min_scale_boundary_probe_fraction
    ):
        reasons.append(
            "boundary_probe_not_strict_majority:"
            f"{boundary_probe_fraction:.6f}<="
            f"{QUALITY_PROFILE.min_scale_boundary_probe_fraction:.6f}"
        )
    if axis_fold > QUALITY_PROFILE.max_scale_fold:
        reasons.append(
            f"axis_scale_fold_too_high:{axis_fold:.6f}>"
            f"{QUALITY_PROFILE.max_scale_fold:.6f}"
        )
    if bootstrap["scale_ci95_fold"] > QUALITY_PROFILE.max_scale_fold:
        reasons.append(
            "bootstrap_scale_ci95_fold_too_high:"
            f"{bootstrap['scale_ci95_fold']:.6f}>"
            f"{QUALITY_PROFILE.max_scale_fold:.6f}"
        )
    if extent_sensitivity_fold > QUALITY_PROFILE.max_scale_fold:
        reasons.append(
            "extent_sensitivity_fold_too_high:"
            f"{extent_sensitivity_fold:.6f}>"
            f"{QUALITY_PROFILE.max_scale_fold:.6f}"
        )

    report = {
        "schema": "oap.fixed_camera_rgbd_gravity_scale.v1",
        "method": "rgbd_gravity_trimmed_extent_isotropic_scale",
        "scale_factor": float(scale_factor),
        "size_xyz_m": [
            float(value * scale_factor) for value in mesh_local_extent
        ],
        "scale_decision": (
            "fixed_camera_rgbd_isotropic"
            if not reasons
            else "fixed_camera_rgbd_rejected"
        ),
        "scale_reportable": not reasons,
        "scale_rejection_reasons": reasons,
        "observed_extent_sorted_m": [
            float(value) for value in observed_extent
        ],
        "outer_trim_observed_extent_sorted_m": [
            float(value) for value in outer_extent
        ],
        "boundary_probe_observed_extent_sorted_m": [
            float(value) for value in boundary_probe_extent
        ],
        "mesh_obb_extent_sorted_before_scale": [
            float(value) for value in mesh_extent
        ],
        "mesh_local_extent_xyz_before_scale": [
            float(value) for value in mesh_local_extent
        ],
        "axis_scale_candidates": [
            float(value) for value in axis_candidates
        ],
        "axis_scale_fold": float(axis_fold),
        "scaled_mesh_obb_extent_sorted_m": [
            float(value * scale_factor) for value in mesh_extent
        ],
        "outer_trim_scale_factor": float(outer_scale),
        "boundary_probe_scale_factor": float(boundary_probe_scale),
        "extent_sensitivity_fold": float(extent_sensitivity_fold),
        "bootstrap": bootstrap,
        "depth_cleanup": {
            "mask_points": int(mask.sum()),
            "valid_depth_points": valid_points,
            "valid_depth_fraction": float(valid_points / int(mask.sum())),
            "boundary_probe_erosion_pixels": erosion_pixels,
            "boundary_probe_mask_points": int(boundary_probe_mask.sum()),
            "boundary_probe_depth_points": boundary_probe_points,
            "boundary_probe_fraction_of_valid": boundary_probe_fraction,
            "raw_depth_range_m": [
                float(depth[valid].min()),
                float(depth[valid].max()),
            ],
            "boundary_probe_depth_range_m": [
                float(depth[boundary_probe].min()),
                float(depth[boundary_probe].max()),
            ],
            "extent_frame": extent_frame,
            "outer_extent_frame": outer_extent_frame,
            "boundary_probe_extent_frame": boundary_probe_extent_frame,
        },
        "quality_profile": QUALITY_PROFILE.to_dict(),
        "provenance": {
            "observation_manifest": str(manifest_path),
            "mask": str(mask_path),
            "mask_sha256": mask_hash,
            "depth_m_npy": str(depth_path),
            "depth_m_npy_sha256": depth_hash,
            "mesh": str(Path(mesh)),
            "mesh_sha256": mesh_hash,
            "camera_serial": packet_serial,
            "fixed_calibration_source": calibration_source,
            "fixed_calibration_sha256": calibration_sha256,
            "T_base_camera": np.asarray(
                T_base_camera,
                dtype=np.float64,
            ).tolist(),
            "note": (
                "The fixed calibration supplies gravity for the primary "
                "all-valid-depth extent. A one-pixel eroded estimate is only "
                "a sensitivity probe, never the selected object scale. The "
                "serial and calibration hash prevent a packet from another "
                "camera entering the production reconstruction path."
            ),
        },
    }
    write_json_atomic(Path(out), report)
    return report


__all__ = ["build_scale_report"]
