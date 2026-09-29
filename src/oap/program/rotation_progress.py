"""Signed principal SO(3) rotation progress in the initial raw body frame.

This experiment's angle is a rotation-vector component, not an unsigned
world-axis angle or a projected dihedral. NumPy and JAX use identical math.
"""
from __future__ import annotations

import numpy as np

# Numerical comparison allowance only (one millionth of a degree), not a
# task acceptance band.
ANGLE_NUMERICAL_EPS_DEG = 1e-6


def rotation_progress_radians(initial, current, axis, *, xp=np):
    """Return Log(R0.T @ Rt) dot axis, including identity and pi rotations.

    The largest quaternion component construction is Shepperd's method, also
    used by oap.utils.se3.mat_to_quat. All arrays may be batched. At the
    intrinsically ambiguous exact pi boundary the dominant vector component
    is chosen positive, independently of a quaternion's storage sign.
    """
    # Choose the exact-pi representative in world coordinates.
    relative = xp.einsum("...ij,...kj->...ik", current, initial)
    r = relative
    d0, d1, d2 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    squared = xp.stack((1+d0+d1+d2, 1+d0-d1-d2,
                        1-d0+d1-d2, 1-d0-d1+d2), axis=-1)
    rows = xp.stack((
        xp.stack((squared[..., 0], r[..., 2, 1]-r[..., 1, 2],
                  r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), axis=-1),
        xp.stack((r[..., 2, 1]-r[..., 1, 2], squared[..., 1],
                  r[..., 0, 1]+r[..., 1, 0], r[..., 0, 2]+r[..., 2, 0]), axis=-1),
        xp.stack((r[..., 0, 2]-r[..., 2, 0], r[..., 0, 1]+r[..., 1, 0],
                  squared[..., 2], r[..., 1, 2]+r[..., 2, 1]), axis=-1),
        xp.stack((r[..., 1, 0]-r[..., 0, 1], r[..., 0, 2]+r[..., 2, 0],
                  r[..., 1, 2]+r[..., 2, 1], squared[..., 3]), axis=-1),
    ), axis=-2)
    index = xp.argmax(squared, axis=-1)
    q = xp.take_along_axis(rows, index[..., None, None], axis=-2)[..., 0, :]
    q = q / xp.maximum(xp.linalg.norm(q, axis=-1, keepdims=True), 1e-15)
    dominant = xp.argmax(xp.abs(q[..., 1:]), axis=-1)
    dominant_value = xp.take_along_axis(q[..., 1:], dominant[..., None], axis=-1)[..., 0]
    near_pi = xp.abs(q[..., 0]) <= 1e-10
    flip = xp.where(near_pi, dominant_value < 0.0, q[..., 0] < 0.0)
    q = xp.where(flip[..., None], -q, q)
    vector = q[..., 1:]
    length = xp.linalg.norm(vector, axis=-1)
    angle = 2.0 * xp.arctan2(length, q[..., 0])
    direction = xp.asarray(axis)
    direction = direction / xp.linalg.norm(direction, axis=-1, keepdims=True)
    direction = xp.einsum("...ij,...j->...i", initial, direction)
    component = xp.sum(vector * direction, axis=-1)
    return xp.where(length > 1e-12,
                    angle * component / xp.maximum(length, 1e-12), 0.0)
