"""GPU (MJX/Warp) availability probe for exact joint-space rollouts.

The end-effector chunk screen (MjxScreen / screen_and_rerank) was deleted with the
EE pipeline (Stage D); the JOINT pipeline's GPU screen lives in
:mod:`oap.twin.batched_rollout` (BatchedRollout). What remains here is
:func:`is_available`, the Warp-bridge probe :mod:`oap.loop.sampling` uses
as a production prerequisite. When it returns false, physics validity is
unknown and planning fails closed with zero motion; there is no CPU fallback.

REQUIRED VERSION SET (hard pin): mujoco/mujoco-mjx/mujoco-warp==3.11.0,
warp-lang==1.15.0, and jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt==0.10.2.
Availability requires the fixed ``paper_standard_warp`` bootstrap: public
GraphMode.WARP with Warp's normal atomics and deterministic debug disabled.
Version drift, import-order drift, a missing GPU, or a non-public ModelWarp
path fails closed.
"""
from __future__ import annotations

import logging

from oap.twin.runtime import bootstrap_production_mjwarp

logger = logging.getLogger(__name__)

__all__ = ["is_available"]

def is_available() -> bool:
    """True only when the complete fixed production runtime proves ready."""
    try:
        identity = bootstrap_production_mjwarp()
        return bool(identity["gpu"]["jax_devices"])
    except Exception:  # pragma: no cover - environment-dependent
        return False
