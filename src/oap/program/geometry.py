"""Provide tiny, dependency-light SE(3) / vector helpers for the program core.

Role in the two-stage pipeline: these are the only geometric operations the
TaskProgram core (planning stage, ``oap-run``) needs; they are also
reused by the reconstruction stage's pose utilities through the twin layer.
Quaternions are (w, x, y, z), matching flexiv-control and the MJCF convention.
Everything is numpy; no robot/sim/torch dependency, so the whole program core
is unit-testable on CPU without a GPU, MuJoCo, or a VLM.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def as_vec3(x: ArrayLike) -> np.ndarray:
    """Coerce input to a float64 3-vector copy."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.shape[0] < 3:
        raise ValueError(f"expected a 3-vector, got shape {v.shape}")
    return v[:3].copy()


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector, returning the zero vector below ``eps`` norm."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a (w,x,y,z) quaternion, defaulting to identity if degenerate."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w,x,y,z) quaternions."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a (w,x,y,z) quaternion from a rotation axis and angle (radians)."""
    a = unit(axis)
    h = 0.5 * float(angle)
    return quat_normalize(np.array([np.cos(h), *(a * np.sin(h))]))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle (radians) in [0, pi] between two vectors."""
    ua, ub = unit(a), unit(b)
    c = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    return float(np.arccos(c))


def point_line_distance(p: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
    """Perpendicular distance from a point to an infinite line."""
    p, lp, d = as_vec3(p), as_vec3(line_point), unit(line_dir)
    w = p - lp
    proj = w - np.dot(w, d) * d
    return float(np.linalg.norm(proj))


def signed_plane_distance(p: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> float:
    """Signed distance from a point to the plane (positive along the normal)."""
    return float(np.dot(as_vec3(p) - as_vec3(plane_point), unit(plane_normal)))


WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_DOWN = np.array([0.0, 0.0, -1.0])
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
