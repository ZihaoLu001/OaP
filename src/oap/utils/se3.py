"""Shared quaternion / SE(3) math for the twin and closed-loop stages.

Role in the two-stage pipeline: both ``oap-reconstruct`` (pose recovery,
OBB fitting) and ``oap-run`` (twin assembly, chunk conversion, residual
gates) need the same small wxyz-quaternion toolbox. This module deliberately
duplicates a little of :mod:`oap.program.geometry` so that ``program/``
stays fully self-contained (the crown-jewel package imports nothing from the
rest of the repo).

All quaternions are (w, x, y, z); all rotations are 3x3 row-major matrices;
all angles are radians.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "quat_mul",
    "quat_conj",
    "quat_to_mat",
    "mat_to_quat",
    "quat_wxyz_from_rpy",
    "yaw_quat",
    "yaw_about_z",
    "yaw_residual_rad",
    "wrap_yaw_half_pi",
    "pose_to_T",
    "T_to_pose",
    "validate_se3_matrix",
]


def validate_se3_matrix(
    matrix: np.ndarray | list[list[float]],
    *,
    name: str = "transform",
    atol: float = 1e-6,
) -> np.ndarray:
    """Return a validated homogeneous rigid transform.

    Calibration files authorize frame changes throughout reconstruction,
    annotation, and real-world twin updates.  A merely reshapeable 4x4 array
    is not enough: malformed projective/reflective matrices can produce finite
    but physically false object poses.  Validation is deliberately strict and
    never repairs or projects a supplied matrix onto SE(3).
    """
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4):
        raise ValueError(
            f"{name} must be a 4x4 SE(3) matrix, got shape {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN/Inf")
    tolerance = float(atol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name} validation tolerance must be finite and > 0")
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(value[3], expected_bottom, rtol=0.0, atol=tolerance):
        raise ValueError(
            f"{name} has invalid homogeneous bottom row "
            f"{value[3].tolist()}; expected [0, 0, 0, 1]"
        )
    rotation = value[:3, :3]
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > tolerance:
        raise ValueError(
            f"{name} rotation is not orthonormal "
            f"(Frobenius error {orthogonality_error:.3g} > {tolerance:.3g})"
        )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError(
            f"{name} rotation determinant is {determinant:.9g}, expected +1"
        )
    return value.copy()


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions) of a wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion (Shepperd's method)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4])


def quat_wxyz_from_rpy(rpy: tuple[float, float, float] | list[float]) -> tuple[float, float, float, float]:
    """Convert intrinsic XYZ roll/pitch/yaw (rad) to a wxyz quaternion."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r / 2.0), np.sin(r / 2.0)
    cp, sp = np.cos(p / 2.0), np.sin(p / 2.0)
    cy, sy = np.cos(y / 2.0), np.sin(y / 2.0)
    return (
        float(cr * cp * cy + sr * sp * sy),
        float(sr * cp * cy - cr * sp * sy),
        float(cr * sp * cy + sr * cp * sy),
        float(cr * cp * sy - sr * sp * cy),
    )


def yaw_quat(yaw: float) -> np.ndarray:
    """Build the wxyz quaternion for a rotation of ``yaw`` about world +z."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def yaw_about_z(quat_wxyz: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Return the in-plane yaw (rotation about world +z) of a wxyz quaternion."""
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def yaw_residual_rad(observed_yaw: float, predicted_yaw: float) -> float:
    """Return the smallest yaw mismatch, wrapped into [0, pi/2].

    The footprint long-axis is pi-symmetric (yaw and yaw+-pi describe the same
    OBB), so a half-turn is a MATCH, not a 180-deg error -- wrap modulo pi,
    then fold to <= pi/2.
    """
    d = (float(observed_yaw) - float(predicted_yaw) + np.pi / 2.0) % np.pi - np.pi / 2.0
    return abs(float(d))


def wrap_yaw_half_pi(yaw: float) -> float:
    """Wrap a pi-symmetric yaw into [-pi/2, pi/2).

    A parallel-jaw grasp at yaw and yaw+-pi is IDENTICAL, so commanded tool
    yaws are wrapped into the wrist joint's comfortable half-turn range
    (joint-limit avoidance for j7).
    """
    return float((float(yaw) + np.pi / 2.0) % np.pi - np.pi / 2.0)


def pose_to_T(pose: tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
    """Convert a 7-vector pose (x, y, z, qw, qx, qy, qz) to a 4x4 homogeneous T."""
    pose = np.asarray(pose, dtype=float).reshape(7)
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose[3:7])
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 homogeneous T into (position[3], quat_wxyz[4])."""
    T = np.asarray(T, dtype=float)
    return T[:3, 3].copy(), mat_to_quat(T[:3, :3])
