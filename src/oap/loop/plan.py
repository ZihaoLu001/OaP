"""PLAN layer: initialize the sampler and prove the program is the objective.

Role in the two-stage pipeline: this wires the W_plan twin into the planner so
the paper's identity holds -- the SAME
:class:`~oap.program.TaskProgram` object that certifies success is
the objective the planner minimizes:

  * :func:`sample_candidates` returns one content-free nominal at each
    explicit stage entry: measured physical state at knot zero and the last
    acknowledged-control hold only in future knots. It does not read the object,
    goal, task name, predicate type, or a hand-written phase. Sampling is
    solely responsible for discovering motion from the program cost and
    contact physics.
  * The sampler evaluates the active stage's running-plus-terminal objective
    on every gated rollout and records it at the ``trajectory_cost`` evidence
    call site.
"""
from __future__ import annotations

import os

import logging
from typing import Any

import numpy as np

from oap.program import AnchorSet
from oap.program.geometry import quat_to_matrix, unit
from oap.twin import TABLE_TOP_Z, SimWorld

logger = logging.getLogger("oap.loop.plan")

__all__ = [
    "DEFAULT_N_KNOTS",
    "N_KNOTS_ENV",
    "num_knots_from_env",
    "predicted_end_anchors",
    "sample_candidates",
    "validate_num_knots",
]

N_KNOTS_ENV = "OAP_N_KNOTS"


def validate_num_knots(value: Any) -> int:
    """Return a valid spline-control-point count."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("num_knots must be an integer >= 2")
    try:
        num_knots = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("num_knots must be an integer >= 2") from exc
    if (
        num_knots < 2
        or not np.isfinite(numeric)
        or numeric != float(num_knots)
    ):
        raise ValueError(
            f"num_knots must be an integer >= 2, got {value!r}"
        )
    return num_knots


def num_knots_from_env() -> int:
    """Resolve the spline dimension once at episode configuration time."""
    raw = (os.environ.get(N_KNOTS_ENV) or "4").strip() or "4"
    try:
        return validate_num_knots(raw)
    except ValueError as exc:
        raise ValueError(f"{N_KNOTS_ENV}: {exc}") from exc


# Compatibility constant for library callers that do not yet carry a
# LoopConfig. Production resolves the value once into LoopConfig and passes it
# explicitly to every local or remote solver.
DEFAULT_N_KNOTS = num_knots_from_env()


def sample_candidates(
    *,
    world: SimWorld,
    num_knots: int = DEFAULT_N_KNOTS,
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    return _sample_candidates(
        world=world,
        num_knots=num_knots,
        continuous_jaw=False,
    )


def sample_mppi_candidates(
    *,
    world: SimWorld,
    num_knots: int = DEFAULT_N_KNOTS,
    control_profile: str | None = None,
    pick_v8_causal_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    """Build the generic nominal with a continuous signed-effort jaw latent."""
    return _sample_candidates(
        world=world,
        num_knots=num_knots,
        continuous_jaw=True,
        control_profile=control_profile,
        pick_v8_causal_profile=pick_v8_causal_profile,
        unified_mppi_effort_profile=unified_mppi_effort_profile,
    )


def _sample_candidates(
    *,
    world: SimWorld,
    num_knots: int,
    continuous_jaw: bool,
    control_profile: str | None = None,
    pick_v8_causal_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
) -> tuple[list[Any], dict[int, dict[str, Any]]]:
    """Build the generic measured-boundary plus applied-control nominal.

    No object pose, goal, stage command, state machine, or IK enters this
    initializer. Knot zero is measured arm q plus the nearest jaw latent.
    Future arm knots hold live ``ctrl``; the acknowledged physical jaw target
    is converted to the same latent before it enters the optimizer.
    Every later solve within a stage shifts the executed winner and holds its
    tail, then re-anchors only knot zero to the latest measurement.
    """
    from oap.twin.control_knots import (
        CONTROL_DIM,
        NJ,
        ControlBounds,
        ControlKnots,
        JawCommandState,
        action_knot_zero,
        jaw_command_state_from_native_control,
    )

    row = np.asarray(world.data.ctrl, dtype=float).copy()
    if row.shape != (CONTROL_DIM,):
        raise ValueError(
            "acknowledged-control hold requires exactly "
            f"{CONTROL_DIM} actuator targets, got {row.shape}"
        )
    if not np.all(np.isfinite(row)):
        raise ValueError("current actuator target contains NaN/Inf")
    # ``data.ctrl`` is the acknowledged actuator COMMAND, so its jaw column
    # inverts exactly to a binary latent. The measured aperture is never
    # consulted: an obstructed hold reads far from either command level.
    if (
        pick_v8_causal_profile is not None
        or unified_mppi_effort_profile is not None
    ):
        if control_profile is None:
            raise ValueError(
                "Pick-v8 causal MPPI nominal requires its control profile"
            )
        acknowledged_jaw = jaw_command_state_from_native_control(
            row[NJ],
            control_profile=control_profile,
            pick_v8_causal_profile=pick_v8_causal_profile,
            unified_mppi_effort_profile=unified_mppi_effort_profile,
            source="plan_world_acknowledged_ctrl",
        )
    else:
        acknowledged_jaw = JawCommandState.from_continuous_effort(
            row[NJ],
            source="plan_world_acknowledged_ctrl",
        )
    row[NJ] = float(acknowledged_jaw.latent)
    # Device execution returns float32 controls.  A target exactly on a model
    # ctrlrange endpoint (notably the fully-open gripper) can therefore arrive
    # on the host a few ulps outside the float64 endpoint.  Canonicalize the
    # acknowledged control to the physical actuator box before it becomes the
    # next stage's nominal.  This is model-derived numerical normalization,
    # not a task-, object-, predicate-, or stage-dependent initializer.
    # A production SimWorld always carries model-derived joint ranges.  Some
    # lightweight callers (including state-sync tests) intentionally expose
    # only the acknowledged actuator vector; in that case there is no bound
    # metadata to canonicalize against, so preserve the already-validated
    # control exactly as the pre-clipping implementation did.
    if "joint_ranges" in world.robot:
        row = ControlBounds.from_world(world).clip(row[None, :])[0]
    knots = np.tile(row, (validate_num_knots(num_knots), 1))
    knots[0] = action_knot_zero(world, acknowledged_jaw=acknowledged_jaw)
    return ([ControlKnots(candidate_id=0, knots=knots)],
            {0: {"seed": "measured_boundary_acknowledged_future_hold",
                 "source": "current_hold",
                 "acknowledged_jaw": acknowledged_jaw.to_dict()}})


# --------------------------------------------------------------------------
# Program-cost evaluation of gated candidates
# --------------------------------------------------------------------------
def predicted_end_anchors(anchors: AnchorSet, row: dict[str, Any], *,
                          table_top_z: float = TABLE_TOP_Z,
                          sub_h: float = 0.0,
                          start_quat_wxyz: Any = None) -> AnchorSet:
    """Build the AnchorSet a gated rollout PREDICTS at its end state.

    The subject (and everything attached to it) is moved to the rollout's
    ``final_object_xy`` at the height implied by ``final_object_lift_m``; every
    other anchor is unchanged. When the rollout also reports
    ``final_object_quat_wxyz`` AND the caller supplies the chunk-start
    orientation, the net rotation is applied too: the subject's axis turns
    with it, and attached anchors rotate ABOUT the subject (points and axes)
    instead of merely translating. Without the rotation an orientation
    predicate (a pour's tilt) reads the same residual for every candidate --
    the optimizer would be blind to the one thing the stage varies. This is
    the cheap, uniform end-state the program cost is evaluated on for
    candidate ranking evidence.
    """
    m = row.get("metrics") or {}
    out = anchors.copy()
    if "subject" not in out:
        return out
    sub = out["subject"]
    old_point = sub.point.copy()
    fxy = m.get("final_object_xy")
    if fxy is not None:
        new_xy = np.asarray(fxy, dtype=float)
    else:
        new_xy = old_point[:2]
    lift = float(m.get("final_object_lift_m", 0.0))
    new_z = float(table_top_z) + 0.5 * float(sub_h) + max(0.0, lift)
    new_point = np.array([new_xy[0], new_xy[1], new_z], dtype=float)
    delta = new_point - old_point
    fquat = m.get("final_object_quat_wxyz")
    rot = None
    if fquat is not None and start_quat_wxyz is not None:
        rot = (quat_to_matrix(np.asarray(fquat, dtype=float))
               @ quat_to_matrix(np.asarray(start_quat_wxyz, dtype=float)).T)
    sub.point = new_point
    if rot is not None and sub.axis is not None:
        sub.axis = unit(rot @ sub.axis)
    for name in out.names():
        a = out[name]
        if a.attached_to == "subject":
            if rot is not None:
                a.point = new_point + rot @ (a.point - old_point)
                if a.axis is not None:
                    a.axis = unit(rot @ a.axis)
            else:
                a.point = a.point + delta
    return out
