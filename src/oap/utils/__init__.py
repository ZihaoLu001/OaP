"""Shared low-level utilities (SE(3) math, IO, video) for both pipeline stages.

Nothing in this package imports from :mod:`oap.program`,
:mod:`oap.twin`, or :mod:`oap.loop` -- it sits at the bottom of the
import graph so every stage can use it.
"""
from __future__ import annotations

from . import io, se3
from .io import (
    load_package_config_json, load_yaml, package_config_path, read_json,
    sha256_bytes, sha256_file, timestamped_run_dir, write_json_atomic,
)
from .se3 import (
    T_to_pose, mat_to_quat, pose_to_T, quat_conj, quat_mul, quat_to_mat,
    quat_wxyz_from_rpy, wrap_yaw_half_pi, yaw_about_z, yaw_quat, yaw_residual_rad,
)

__all__ = [
    "io", "se3",
    "read_json", "write_json_atomic", "sha256_bytes", "sha256_file",
    "timestamped_run_dir", "package_config_path", "load_package_config_json",
    "load_yaml",
    "quat_mul", "quat_conj", "quat_to_mat", "mat_to_quat", "quat_wxyz_from_rpy",
    "yaw_quat", "yaw_about_z", "yaw_residual_rad", "wrap_yaw_half_pi",
    "pose_to_T", "T_to_pose",
]
