"""The JOINT-space action parameterization: arm and jaw effort.

The decision variable is

    action knots in R^(K x 8):  [7 normalized joint torques, gripper effort]

i.e. seven arm targets plus one dimensionless gripper-effort coordinate. The
jaw coordinate is interpolated continuously and maps linearly to signed GN01
force: ``force_n = 80 * clip(latent, -1, 1)``. Positive closes and negative
opens; no desired width appears in the action.
Source-verified as THE authoritative sampling-MPC decision variable for a
position-actuated arm (mjpc ``TimeSpline(dim=model->nu)`` policy.cc:39; hydrax
``zeros((num_knots, nu))`` alg_base.py:294): sampling MPC samples ctrl and
clamps to ``actuator_ctrlrange``; for a position servo that ctrl IS the joint
target. Absolute targets (never increments/velocities), per-actuator noise,
joint limits native, NO IK in the rollout or the executor.

Each explicit stage starts from measured physical state at knot zero and a
current-controller hold in future knots. Every subsequent MPC solve within
that stage shifts the previous winner, holds its tail, and re-anchors knot zero
to the latest measurement. No task geometry, phase label, Cartesian waypoint,
or IK goal
enters initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CONTROL_DIM",
    "GRIPPER_CLOSED_LATENT",
    "GRIPPER_OPEN_LATENT",
    "NJ",
    "ControlBounds",
    "ControlKnots",
    "JawCommandState",
    "JawCommandUnknown",
    "action_knot_zero",
    "jaw_command_state_from_native_control",
    "jaw_latent_from_commanded_width",
    "knots_to_ctrl",
    "measured_arm_boundary",
    "shift_mppi_effort_warm_start",
    "shift_warm_start",
]

# [j1..j7, dimensionless gripper latent]
NJ = 7
CONTROL_DIM = 8
GRIPPER_OPEN_LATENT = -1.0
GRIPPER_CLOSED_LATENT = 1.0

@dataclass(frozen=True)
class ControlBounds:
    """Box bounds on one action knot: joint limits + ``[-1, 1]`` jaw latent.

    The arm interval comes from the model. The jaw column is a normalized,
    task-independent signed effort used identically by simulation and real
    execution; physical aperture is state, not command.
    """

    joint_lo: np.ndarray  # (7,)
    joint_hi: np.ndarray  # (7,)
    gripper_latent_lo: float = GRIPPER_OPEN_LATENT
    gripper_latent_hi: float = GRIPPER_CLOSED_LATENT

    @classmethod
    def from_world(cls, world: Any) -> "ControlBounds":
        """Read arm limits and append the fixed jaw-latent interval."""
        jr = np.asarray(world.robot["joint_ranges"], dtype=float)[:NJ]
        return cls(
            joint_lo=jr[:, 0].copy(),
            joint_hi=jr[:, 1].copy(),
            gripper_latent_lo=GRIPPER_OPEN_LATENT,
            gripper_latent_hi=GRIPPER_CLOSED_LATENT,
        )

    def as_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lo, hi) arrays of shape (CONTROL_DIM,)."""
        lo = np.concatenate([
            self.joint_lo,
            [self.gripper_latent_lo],
        ]).astype(float)
        hi = np.concatenate([
            self.joint_hi,
            [self.gripper_latent_hi],
        ]).astype(float)
        return lo, hi

    def clip(self, knots: np.ndarray) -> np.ndarray:
        """Clip a (K, 8) joint-knot array into the box (== ctrlrange)."""
        lo, hi = self.as_array()
        return np.clip(np.asarray(knots, dtype=float), lo, hi)

    def noise_sigma(self, frac: float) -> np.ndarray:
        """Per-actuator Gaussian sigma = frac * 0.5 * (hi - lo).

        The mjpc convention (planner.cc:342-347): noise is scaled to each
        actuator's OWN range, never one global sigma across arm + gripper.
        """
        lo, hi = self.as_array()
        return float(frac) * 0.5 * (hi - lo)


@dataclass(frozen=True)
class ControlKnots:
    """An action sample: the (K, 8) decision carried through the loop.

    THIS is the decision variable -- a K x 8 joint-knot array (:data:`CONTROL_DIM`:
    j1..j7 targets + gripper latent) plus a candidate id that the sampler and
    audit log key on. Production
    (:func:`oap.loop.sampling.sample_once`) samples and selects ``knots``
    directly, then rolls them on
    :class:`~oap.twin.batched_rollout.BatchedRollout` (which consumes bare
    (K, 8) arrays); no intermediate execution format is ever built.
    """

    candidate_id: int
    knots: np.ndarray                                    # (K, 8)


class JawCommandUnknown(RuntimeError):
    """The last acknowledged jaw-effort command is not established.

    Planning and execution both fail closed here. Guessing open or closed
    would silently rewrite the action boundary of every candidate.
    """


def jaw_latent_from_commanded_width(
    width_m: float,
    *,
    open_width_m: float = 0.085,
    closed_width_m: float = 0.0,
    tol_m: float = 1e-3,
) -> float:
    """Legacy artifact reader: invert an exact width command to a binary latent.

    The command lane emits exactly two widths
    (:func:`~oap.twin.batched_rollout.width_command_ctrl`), so this
    inverse is exact -- and deliberately strict. Any other value is refused
    rather than snapped to the nearer level.

    This strictness is the fix for a measured 2026-08-07 defect: a jaw
    holding the eraser is obstructed at 44-67 mm, so a nearest-level rule
    (midpoint 42.5 mm) reported ``open`` for a jaw that had been commanded
    ``closed`` on all 30 pilot episodes. A physical aperture is a plant
    state, never a command. Feed this function commanded widths only
    (``data.ctrl``, an executor's acknowledged target); feed it a measured
    ``qpos`` width and it raises.
    """
    width = float(width_m)
    open_width = float(open_width_m)
    closed_width = float(closed_width_m)
    if (
        not np.isfinite(width)
        or not np.isfinite(open_width)
        or not np.isfinite(closed_width)
        or open_width <= closed_width
    ):
        raise ValueError(
            "commanded and level widths must be finite with open > closed"
        )
    if abs(width - closed_width) <= tol_m:
        return GRIPPER_CLOSED_LATENT
    if abs(width - open_width) <= tol_m:
        return GRIPPER_OPEN_LATENT
    raise JawCommandUnknown(
        f"width {width:.6f} m is not a commanded jaw level "
        f"({closed_width:.6f} / {open_width:.6f} m within {tol_m:.6f} m); "
        "an obstructed physical aperture is not a command state"
    )


@dataclass(frozen=True)
class JawCommandState:
    """The last acknowledged jaw command, carried explicitly.

    New MPC paths populate ``commanded_effort_n`` and leave width absent.
    Legacy width constructors remain only for reading older artifacts.
    """

    latent: float
    commanded_width_m: float | None
    source: str
    acknowledged_at_s: float | None = None
    continuous: bool = False
    commanded_effort_n: float | None = None

    def __post_init__(self) -> None:
        latent = float(self.latent)
        if self.continuous:
            if not np.isfinite(latent) or not -1.0 <= latent <= 1.0:
                raise ValueError(
                    "continuous jaw command latent must lie in [-1, 1]"
                )
        elif latent not in {GRIPPER_OPEN_LATENT, GRIPPER_CLOSED_LATENT}:
            raise ValueError(
                "jaw command latent must be exactly the open or closed level"
            )
        if not self.source:
            raise ValueError("jaw command state needs a provenance source")

    @property
    def is_closed(self) -> bool:
        if self.commanded_effort_n is not None:
            return float(self.commanded_effort_n) > 0.0
        assert self.commanded_width_m is not None
        return float(self.commanded_width_m) <= 1e-3

    @classmethod
    def from_commanded_width(
        cls,
        width_m: float,
        *,
        source: str,
        acknowledged_at_s: float | None = None,
        open_width_m: float = 0.085,
        closed_width_m: float = 0.0,
        tol_m: float = 1e-3,
    ) -> "JawCommandState":
        latent = jaw_latent_from_commanded_width(
            width_m,
            open_width_m=open_width_m,
            closed_width_m=closed_width_m,
            tol_m=tol_m,
        )
        return cls(
            latent=float(latent),
            commanded_width_m=float(width_m),
            source=str(source),
            acknowledged_at_s=(
                None if acknowledged_at_s is None else float(acknowledged_at_s)
            ),
        )

    @classmethod
    def from_continuous_commanded_width(
        cls,
        width_m: float,
        *,
        source: str,
        acknowledged_at_s: float | None = None,
        open_width_m: float = 0.085,
    ) -> "JawCommandState":
        """Legacy artifact reader for an older continuous-width controller."""
        width = float(width_m)
        open_width = float(open_width_m)
        if (
            not np.isfinite(width)
            or not np.isfinite(open_width)
            or open_width <= 0.0
            or width < -1e-9
            or width > open_width + 1e-9
        ):
            raise ValueError(
                "continuous commanded jaw width must lie inside the physical "
                f"range [0, {open_width:.6f}] m"
            )
        width = float(np.clip(width, 0.0, open_width))
        latent = 1.0 - 2.0 * width / open_width
        return cls(
            latent=float(latent),
            commanded_width_m=width,
            source=str(source),
            acknowledged_at_s=(
                None if acknowledged_at_s is None else float(acknowledged_at_s)
            ),
            continuous=True,
        )

    @classmethod
    def from_continuous_effort(
        cls,
        effort_n: float,
        *,
        source: str,
        acknowledged_at_s: float | None = None,
        force_limit_n: float = 80.0,
    ) -> "JawCommandState":
        effort = float(effort_n)
        limit = float(force_limit_n)
        if (
            not np.isfinite(effort)
            or not np.isfinite(limit)
            or limit <= 0.0
            or abs(effort) > limit + 1e-5
        ):
            raise ValueError(
                "continuous jaw effort must lie inside the physical force range"
            )
        return cls(
            latent=float(np.clip(effort / limit, -1.0, 1.0)),
            commanded_width_m=None,
            commanded_effort_n=float(np.clip(effort, -limit, limit)),
            source=str(source),
            acknowledged_at_s=(
                None if acknowledged_at_s is None else float(acknowledged_at_s)
            ),
            continuous=True,
        )

    @classmethod
    def from_asymmetric_continuous_effort(
        cls,
        effort_n: float,
        *,
        source: str,
        open_force_limit_n: float,
        close_force_limit_n: float = float(__import__("os").environ.get("OAP_JAW_CLOSE_FORCE_N", "80") or 80.0),  # V11F
        acknowledged_at_s: float | None = None,
    ) -> "JawCommandState":
        """Invert a signed latent with separate opening/closing force scales."""
        effort = float(effort_n)
        open_limit = float(open_force_limit_n)
        close_limit = float(close_force_limit_n)
        if (
            not np.isfinite(effort)
            or not np.isfinite(open_limit)
            or not np.isfinite(close_limit)
            or open_limit <= 0.0
            or close_limit <= 0.0
            or effort < -open_limit - 1e-5
            or effort > close_limit + 1e-5
        ):
            raise ValueError(
                "asymmetric continuous jaw effort lies outside its physical "
                "opening/closing force range"
            )
        latent = (
            effort / open_limit if effort < 0.0 else effort / close_limit
        )
        return cls(
            latent=float(np.clip(latent, -1.0, 1.0)),
            commanded_width_m=None,
            commanded_effort_n=float(
                np.clip(effort, -open_limit, close_limit)
            ),
            source=str(source),
            acknowledged_at_s=(
                None if acknowledged_at_s is None else float(acknowledged_at_s)
            ),
            continuous=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "oap_jaw_command_state_v1",
            "latent": float(self.latent),
            "state": (
                "continuous"
                if self.continuous
                else ("closed" if self.is_closed else "open")
            ),
            "commanded_width_m": (
                None if self.commanded_width_m is None else float(self.commanded_width_m)
            ),
            "commanded_effort_n": self.commanded_effort_n,
            "continuous": bool(self.continuous),
            "source": self.source,
            "acknowledged_at_s": self.acknowledged_at_s,
        }


def jaw_command_state_from_native_control(
    command: float,
    *,
    control_profile: str,
    pick_v8_causal_profile: str | None,
    unified_mppi_effort_profile: str | None = None,
    source: str,
    acknowledged_at_s: float | None = None,
) -> JawCommandState:
    """Invert the exact profile-native jaw command carried by execution."""
    from oap.twin.control_profile import (
        CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
        validate_control_profile,
    )

    profile = validate_control_profile(control_profile)
    if profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH:
        return JawCommandState.from_continuous_commanded_width(
            command,
            source=source,
            acknowledged_at_s=acknowledged_at_s,
        )
    if pick_v8_causal_profile is not None:
        from oap.twin.pick_v8_causal import (
            PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N,
            pick_v8_causal_uses_asymmetric_effort,
        )

        if pick_v8_causal_uses_asymmetric_effort(pick_v8_causal_profile):
            return JawCommandState.from_asymmetric_continuous_effort(
                command,
                source=source,
                open_force_limit_n=(
                    PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N
                ),
                acknowledged_at_s=acknowledged_at_s,
            )
    if unified_mppi_effort_profile is not None:
        from oap.twin.unified_mppi_effort import (
            unified_mppi_effort_uses_proven_pick_carrier,
        )

        if unified_mppi_effort_uses_proven_pick_carrier(
            unified_mppi_effort_profile
        ):
            return JawCommandState.from_asymmetric_continuous_effort(
                command,
                source=source,
                open_force_limit_n=1.0,
                acknowledged_at_s=acknowledged_at_s,
            )
    return JawCommandState.from_continuous_effort(
        command,
        source=source,
        acknowledged_at_s=acknowledged_at_s,
    )


def measured_arm_boundary(
    world: Any,
    *,
    qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Return the measured arm joint positions used as the action boundary.

    The first trajectory knot is the physical boundary from which the
    controller emits its first setpoint, so the arm columns come from
    measurement rather than from an acknowledged target the robot may lag.
    The jaw column is deliberately absent: commanded signed effort is not a
    position measurement, and aperture may be obstructed by an object.
    """
    robot = getattr(world, "robot", None)
    if not isinstance(robot, dict):
        raise ValueError("measured arm boundary requires world.robot metadata")
    arm_addr = np.asarray(robot.get("qpos_addr", ()), dtype=int)
    if arm_addr.shape != (NJ,):
        raise ValueError(
            f"measured arm boundary requires {NJ} arm qpos addresses, "
            f"got {arm_addr.shape}"
        )
    state = np.asarray(
        world.data.qpos if qpos is None else qpos,
        dtype=float,
    ).reshape(-1)
    if np.any(arm_addr < 0) or np.any(arm_addr >= len(state)):
        raise ValueError("measured arm-boundary qpos address is out of range")
    row = state[arm_addr].copy()
    if row.shape != (NJ,) or not np.all(np.isfinite(row)):
        raise ValueError("measured arm boundary must be finite")
    return row


def action_knot_zero(
    world: Any,
    *,
    acknowledged_jaw: JawCommandState | None,
    qpos: np.ndarray | None = None,
) -> np.ndarray:
    """Compose knot zero: measured arm boundary + acknowledged jaw command.

    The two columns come from genuinely different channels and must not be
    conflated. Arm targets track a measured joint state. The jaw column is
    the last acknowledged signed-effort command, because CEM never samples knot
    zero -- whatever lands here is executed verbatim at the start of every
    prefix. Inferring it from the measured aperture put a spurious open
    command at the head of all 30 pilot carries.
    """
    if acknowledged_jaw is None:
        raise JawCommandUnknown(
            "knot zero needs the last acknowledged jaw command; refusing to "
            "guess jaw effort"
        )
    row = np.empty(CONTROL_DIM, dtype=float)
    row[:NJ] = measured_arm_boundary(world, qpos=qpos)
    row[NJ] = float(acknowledged_jaw.latent)
    return row


def shift_warm_start(knots: np.ndarray, executed_frac: float, *,
                           bounds: ControlBounds | None = None) -> np.ndarray:
    """Receding-horizon warm start (M3): re-sample the previous best forward.

    THE authoritative shift-init, unanimous across sampling MPC (hydrax
    ``alg_base.py:139-148``; mjpc ``planner.cc`` resample-by-time; STORM
    ``roll(-shift)`` + hold): after executing a fraction of the plan, the next
    cycle's seed is the SAME winning knot spline re-sampled at knot times
    advanced by that fraction, NOT a cold re-seed. The executed prefix is
    discarded and the tail is HELD (the last knot repeats) rather than zeroed --
    hydrax clamps the advanced times to the spline's domain, which is exactly a
    hold-last. Episode entry itself is a content-free current-joint hold.

    Args:
        knots: the previous cycle's winning (K, 8) joint knots.
        executed_frac: fraction of the horizon executed since (in [0, 1]); the
            knot times advance by this and clamp to the spline's [0, 1] domain.
        bounds: optional box to clip the shifted knots into (== ctrlrange).

    Returns:
        The (K, 8) warm-start nominal for the next MPC cycle.
    """
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 2 or knots.shape[1] != CONTROL_DIM:
        raise ValueError(f"knots must be (K, {CONTROL_DIM}), got {knots.shape}")
    K = knots.shape[0]
    if K == 1:
        return bounds.clip(knots.copy()) if bounds is not None else knots.copy()
    tk = np.linspace(0.0, 1.0, K)
    # advance the knot times by the executed fraction, clamp to the old domain
    # (times past 1 map to the last knot == hold-last), then re-sample.
    new_t = np.clip(tk + float(executed_frac), 0.0, 1.0)
    shifted = np.stack([np.interp(new_t, tk, knots[:, c])
                        for c in range(CONTROL_DIM)], axis=1)
    return bounds.clip(shifted) if bounds is not None else shifted


def shift_mppi_effort_warm_start(
    action_endpoints: np.ndarray,
    executed_frac: float,
    *,
    bounds: ControlBounds | None = None,
) -> np.ndarray:
    """Shift MPPI's effort endpoints by elapsed action intervals.

    A paper MPPI row represents the physical state at the *end* of one
    torque action. It therefore has ``K`` equal-duration
    intervals, unlike a position spline whose ``K`` knots span ``K-1``
    segments.  With the paper timing, executing ``1/K`` of the horizon must
    drop exactly the first endpoint, shift rows 1..K-1 forward, and hold the
    last row.  Treating these endpoints as generic position knots introduces
    a fractional off-by-one resample at every MPC boundary.
    """
    values = np.asarray(action_endpoints, dtype=float)
    if values.ndim != 2 or values.shape[1] != CONTROL_DIM:
        raise ValueError(
            f"action_endpoints must be (K, {CONTROL_DIM}), got {values.shape}"
        )
    count = values.shape[0]
    if count == 1:
        shifted = values.copy()
    else:
        advance = float(executed_frac) * float(count)
        source = np.arange(count, dtype=float)
        query = np.clip(source + advance, 0.0, float(count - 1))
        shifted = np.stack(
            [np.interp(query, source, values[:, column])
             for column in range(CONTROL_DIM)],
            axis=1,
        )
    return bounds.clip(shifted) if bounds is not None else shifted


# Backward-compatible import for sealed episode readers. New production code
# uses the actuator-neutral name above.
shift_mppi_velocity_warm_start = shift_mppi_effort_warm_start


def knots_to_ctrl(knots: np.ndarray, n_steps: int, *,
                        order: str = "linear") -> np.ndarray:
    """Interpolate K action knots into an ``(n_steps, 8)`` latent sequence.

    Knots sit at FIXED equally-spaced times tk = linspace(0, K-1, K); the
    spline is sampled at every control step (n_steps = round(horizon/dt)), so
    the spline sample-rate is the control rate, far finer than the knot rate
    (mjpc/hydrax convention). ``order='linear'`` is the standard smooth choice
    for a stiff kp=15000 servo (smoother setpoints than zero-order-hold, no
    cubic overshoot past the joint limits the knots are clamped to). Each
    Arm columns are physical joint targets.  The final column remains the
    continuous gripper latent so the command boundary can threshold it after
    interpolation; it must never be written directly to ``data.ctrl``.
    """
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 2 or knots.shape[1] != CONTROL_DIM:
        raise ValueError(f"joint knots must be (K, {CONTROL_DIM}), got {knots.shape}")
    K = knots.shape[0]
    H = int(n_steps)
    if K == 1:
        return np.repeat(knots, H, axis=0)
    tk = np.linspace(0.0, K - 1.0, K)
    ts = np.linspace(0.0, K - 1.0, H)
    if order == "linear":
        return np.stack([np.interp(ts, tk, knots[:, c])
                         for c in range(CONTROL_DIM)], axis=1)
    if order == "cubic":
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as e:
            raise ValueError("order='cubic' needs scipy; use 'linear'") from e
        cs = CubicSpline(tk, knots, axis=0, bc_type="clamped")
        return np.asarray(cs(ts), dtype=float)
    raise ValueError(f"order must be 'linear' or 'cubic', got {order!r}")
