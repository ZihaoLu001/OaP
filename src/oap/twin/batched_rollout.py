"""GPU (MJX/warp) joint-space rollout backend for sampling-based MPC.

Where :class:`~oap.twin.mjx_screen.MjxScreen` rolls a reduced
floating-gripper model driven by EE pose, this rolls the full
twin -- the real 7-joint arm + gripper -- driven directly by seven arm actions
and one normalized signed GN01 effort action over JOINT knots
(:mod:`oap.twin.control_knots`). No IK or mocap palm is used: arm actions
drive their native actuators and ``force_n = 80 * jaw_latent`` drives the jaw.

The rollout uses the plan XML's contact-active convex arm meshes directly.
There is one geometry and one contact model for planning and execution: no
per-link capsule fit, safety buffer, phantom-contact band, or sampled
self-collision exclusion.  Arm/world, arm/object, and arm/arm penetration is
therefore measured from zero depth on the same hulls that define the twin.

Solver: warp has no NoSlip (raises on ``noslip_iterations>0``), so this uses
elliptic cone + impratio=10.

RETRACTED 2026-07-27 -- the 2026-07-17 "the gn01 needs NoSlip to HOLD a grasp"
finding does not reproduce and its mechanism was misattributed. NoSlip changes
a 150 mm CPU carry by 0.02 mm (4.11 mm with, 4.13 mm without). What actually
broke the warp grasp was this file: ``run()`` restored only
``gn01_finger_width_joint`` from a continued state and left the six
equality-coupled knuckles at the template default, so a held rollout started
with the hand OPEN and the object simply fell (16/16 candidates landed exactly
on the table datum, and the result was insensitive to condim, solref, solimp,
nativeccd, iterations, timestep, impratio and cone -- the tell that no contact
was being computed at all). Restoring all seven gn01 joints, plus raising the
complete linkage state prevents the false open start. This brings the real
linkage to 0.1 mm mean agreement with the CPU certifier on a
rank-breaking carry batch. Grip-critical outcomes no longer require CPU.

The per-candidate row schema is byte-identical to the floating-gripper screen's
so :func:`oap.loop.sampling._score_row` consumes either backend
unchanged. The contact extraction reuses the floating-gripper screen's
world-id + nacon masking verbatim (the warp contact arena is shared across the
vmapped batch; reading it unmasked cross-contaminates worlds).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from oap.program.cost_shaping import (
    cost_shaping_identity,
    cost_shaping_term_multipliers,
)
from oap.twin.assets import GN01_OPEN_WIDTH_M
from oap.twin.control_knots import NJ, knots_to_ctrl
from oap.twin.control_profile import (
    CONTROL_PROFILE_DIRECT_TORQUE_FORCE,
    CONTROL_PROFILE_JOINT_VELOCITY_FORCE,
    CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT,
    CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH,
    control_profile_uses_joint_velocity_arm,
    control_profile_uses_width_jaw,
    DIAL_ANNEALED_R2_EQUAL_BUDGET_V1,
    validate_control_profile,
    validate_joint_velocity_force_dial_arm,
)
from oap.twin.control_regularizer import (
    UNIFIED_CONTROL_REGULARIZER_V1,
    control_regularizer_base_valid_raw_violation,
    control_regularizer_cache_identity,
    control_regularizer_command_diagnostics,
    control_regularizer_formula_identity,
    control_regularizer_total_cost,
    control_regularizer_weighted_terms,
    require_control_regularizer_result_identity,
    validate_control_regularizer_calibration_profile,
)
from oap.twin.runtime import (
    bootstrap_production_mjwarp,
    verify_modelwarp,
)

logger = logging.getLogger("oap.twin.batched_rollout")

DEFAULT_EXECUTION_STEPS = 480
HORIZON_STEPS_ENV = "OAP_HORIZON_STEPS"
LOCAL_SIGMA_FRACTION_ENV = "OAP_LOCAL_SIGMA_FRACTION"
CEM_ROUNDS_ENV = "OAP_CEM_ROUNDS"
CEM_ELITE_FRACTION_ENV = "OAP_CEM_ELITE_FRACTION"
CEM_MIN_STD_FRACTION_ENV = "OAP_CEM_MIN_STD_FRACTION"
DEFAULT_CEM_ROUNDS = 4
DEFAULT_CEM_ELITE_FRACTION = 0.1
DEFAULT_CEM_MIN_STD_FRACTION = 0.01
# The reference halton-spline branch ignores ``lambda_`` and initializes its
# adaptive exponential-utility scale as ``self.beta = 1``.
DEFAULT_MPPI_TEMPERATURE = 1.0
DEFAULT_MPPI_ETA_MIN = 5.0
DEFAULT_MPPI_ETA_MAX = 10.0
DEFAULT_MPPI_TABLE_FORCE_WEIGHT = 0.1
MPPI_TABLE_FORCE_SENSOR = "oap_mppi_table_contact_force"
# Root of the robot kinematic chain in the canonical plan scene.  The movable
# objects (``support_eraser``, ``pick_object``) are siblings of this body, not
# descendants, so restricting the table-force sensor to this subtree excludes
# them by construction.
MPPI_TABLE_FORCE_ROBOT_ROOT_BODY = "base"
DEFAULT_GRIPPER_LATENT_STD = 1.0
# Bernoulli counterpart of min_std: keeps P(closed) off 0 and 1 so a Stage
# needing a jaw TRANSITION can still sample one after a unanimous round.
DEFAULT_JAW_P_FLOOR = 0.05
DEFAULT_ACTION_RATE_WEIGHT = 1e-3
# The public whole-body effort-pick controller penalizes the squared seven-arm
# joint velocity with weight 0.1 at each MPC control step. Our native horizon
# contains many 2 ms physics steps, so sample at the K held-action endpoints
# and sum exactly once per controller step rather than once per integrator step.
DEFAULT_ARM_VELOCITY_WEIGHT = 0.1
ARM_VELOCITY_WEIGHTS = (0.0, 0.01, 0.1)
PREDICTIVE_SAMPLER_SINGLE_SCALE = "single_scale"
PREDICTIVE_SAMPLER_TWO_SCALE = "two_scale"
PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL = "mtp_global_local"
PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON = "reference_effort_halton"
MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN = "softmax_weighted_mean"
MPPI_EXECUTION_BEST_VALID_SAMPLE = "best_valid_sample"
MPPI_EXECUTION_MODES = (
    MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    MPPI_EXECUTION_BEST_VALID_SAMPLE,
)
MTP_JAW_MODE_PRESERVE_NOMINAL = "preserve_nominal"
MTP_JAW_MODE_SAMPLE = "sample"
MTP_JAW_MODES = (
    MTP_JAW_MODE_PRESERVE_NOMINAL,
    MTP_JAW_MODE_SAMPLE,
)
PREDICTIVE_SAMPLER_MODES = (
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_SAMPLER_TWO_SCALE,
    PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
    PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON,
)
PREDICTIVE_SAMPLER_MPPI_HALTON_MODES = (
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON,
)

__all__ = [
    "BatchedRollout",
    "CEM_ELITE_FRACTION_ENV",
    "CEM_MIN_STD_FRACTION_ENV",
    "CEM_ROUNDS_ENV",
    "DEFAULT_CEM_ELITE_FRACTION",
    "DEFAULT_CEM_MIN_STD_FRACTION",
    "DEFAULT_CEM_ROUNDS",
    "DEFAULT_MPPI_TEMPERATURE",
    "DEFAULT_ACTION_RATE_WEIGHT",
    "DEFAULT_ARM_VELOCITY_WEIGHT",
    "ARM_VELOCITY_WEIGHTS",
    "DEFAULT_EXECUTION_STEPS",
    "PREDICTIVE_SIGMA_FRACTION",
    "PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION",
    "PREDICTIVE_SAMPLER_MODES",
    "PREDICTIVE_SAMPLER_MPPI_HALTON_MODES",
    "PREDICTIVE_SAMPLER_SINGLE_SCALE",
    "PREDICTIVE_SAMPLER_TWO_SCALE",
    "PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL",
    "PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON",
    "MTP_JAW_MODES",
    "MTP_JAW_MODE_PRESERVE_NOMINAL",
    "MTP_JAW_MODE_SAMPLE",
    "MPPI_EXECUTION_MODES",
    "MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN",
    "MPPI_EXECUTION_BEST_VALID_SAMPLE",
    "PredictiveSamplingResult",
    "HORIZON_STEPS_ENV",
    "LOCAL_SIGMA_FRACTION_ENV",
    "binary_gripper_width",
    "cem_elite_refit",
    "cem_elite_fraction_from_env",
    "cem_min_std_fraction_from_env",
    "cem_rounds_from_env",
    "cem_sampling_proposals",
    "mppi_halton_spline_proposals",
    "mtp_global_local_velocity_proposals",
    "mtp_elite_weighted_mean",
    "paper_mppi_filter_action_plan",
    "mppi_continuous_effort_ctrl",
    "mppi_continuous_width_ctrl",
    "mppi_gripper_command_channels",
    "mppi_prefix_carrier_ctrl",
    "mppi_realized_control_knots",
    "mppi_weighted_update",
    "mppi_weighted_mean",
    "dial_annealed_noise_scale",
    "dial_annealed_proposals",
    "dial_fixed_temperature_update",
    "binarized_width_ctrl",
    "width_command_ctrl",
    "build_paper_mppi_native_effort_xml",
    "build_paper_mppi_legacy_velocity_xml",
    "build_control_profile_rollout_xml",
    "build_paper_mppi_native_velocity_xml",
    "mppi_control_profile_parameters",
    "build_slide_gripper_xml",
    "device_action_feasibility",
    "device_action_rate_cost",
    "device_arm_velocity_cost",
    "exact_arm_collision_geoms",
    "full_fidelity_rollout",
    "horizon_steps_from_env",
    "local_sigma_fraction_from_env",
    "prepare_zero_sensor_mjx_model",
    "put_zero_sensor_mjx_model",
    "predictive_sampling_candidate_counts",
    "predictive_sampling_proposals",
    "stable_device_selection",
    "stable_terminal_trajectory_selection",
    "terminal_first_joint_hit_samples",
    "validate_local_sigma_fraction",
    "validate_mtp_jaw_mode",
    "validate_predictive_sampler_mode",
    "validate_horizon_steps",
    "validate_cem_elite_fraction",
    "validate_cem_rounds",
    "validate_mppi_temperature",
    "validate_mppi_execution_mode",
    "validate_arm_velocity_weight",
    "width_to_slide_ctrl",
]

POSE_SAMPLES = 600         # Compatibility constant; live rollouts retain all H states.


def physics_sample_indices(horizon: int) -> np.ndarray:
    """All post-step observations for Eq. (5), paired with H controls."""
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return np.arange(int(horizon), dtype=np.int32)
PREDICTIVE_SIGMA_FRACTION = 0.06
PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION = 0.02
_VALIDITY_COMPONENT_NAMES = (
    "contact_capacity",
    "constraint_capacity",
    "obj_table_pen",
    "obj_other_support_pen",
    "pad_table_pen",
    "arm_pen",
    "finite_program_cost",
    "subject_floor",
    "movable_floor",
    "controller_timing",
    "joint_position",
    "joint_velocity",
    "gripper_width",
    "gripper_velocity",
    "prefix_hold",
)


def control_profile_validity_component_names(
    control_profile: str | None,
) -> tuple[str, ...]:
    """Return truthful host labels without changing device validity order."""
    profile = validate_control_profile(control_profile, allow_none=True)
    if profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE:
        return _VALIDITY_COMPONENT_NAMES
    return tuple(
        "gripper_effort_bound"
        if name == "gripper_width"
        else "gripper_effort_slew"
        if name == "gripper_velocity"
        else name
        for name in _VALIDITY_COMPONENT_NAMES
    )

# Compiled screens, keyed by everything that changes the artifact. Bounded to a
# handful because a stage's plan XML is stable and an episode visits few of
# them; unbounded growth would pin GPU memory for models nobody rolls again.
_SCREEN_CACHE: dict = {}
_SCREEN_CACHE_MAX = 4


def prepare_zero_sensor_mjx_model(
    model: mujoco.MjModel,
    *,
    allow_mppi_table_force_sensor: bool = False,
) -> mujoco.MjModel:
    """Disable sensor evaluation for an asserted sensor-free MJX model.

    MuJoCo 3.11's public Warp bridge cannot evaluate the unused sensor path in
    this rollout model.  Disabling it is semantics-preserving only when the
    compiled model truly has no sensors, so non-empty sensor models fail
    closed instead of being silently altered.
    """
    sensor_ok = False
    if allow_mppi_table_force_sensor and int(model.nsensor) == 1:
        sensor_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_SENSOR, 0
        )
        sensor_ok = (
            sensor_name == MPPI_TABLE_FORCE_SENSOR
            and int(model.sensor_dim[0]) == 3
            and int(model.nsensordata) == 3
        )
    if (int(model.nsensor) != 0 or int(model.nsensordata) != 0) and not sensor_ok:
        raise RuntimeError(
            "zero-sensor MJX preparation requires nsensor=nsensordata=0, got "
            f"{int(model.nsensor)}/{int(model.nsensordata)}"
        )
    if not sensor_ok:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    return model


def put_zero_sensor_mjx_model(
    model: mujoco.MjModel,
    *,
    allow_mppi_table_force_sensor: bool = False,
) -> tuple[Any, str, dict[str, Any]]:
    """Convert through the public fixed-WARP paper-standard path.

    The signature intentionally exposes no graph-mode, debug, deterministic,
    or record-cap override.  The fixed profile uses Warp's normal atomics with
    deterministic debug disabled.  Runtime bootstrap is repeated here as a
    non-bypassable backstop for direct library callers.
    """
    runtime_identity = bootstrap_production_mjwarp()
    from mujoco import mjx

    prepare_zero_sensor_mjx_model(
        model,
        allow_mppi_table_force_sensor=allow_mppi_table_force_sensor,
    )
    conversion = mjx.put_model
    converted = conversion(model, impl="warp")
    actual = verify_modelwarp(converted)
    try:
        conversion_source = inspect.getsource(conversion).encode("utf-8")
    except (OSError, TypeError):
        conversion_source = b""
    module = importlib.import_module(conversion.__module__)
    module_path = Path(module.__file__).resolve() if module.__file__ else None
    module_sha256 = None
    if module_path is not None and module_path.is_file():
        module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    audit = {
        "function": f"{conversion.__module__}.{conversion.__name__}",
        "signature": str(inspect.signature(conversion)),
        "source_sha256": (
            hashlib.sha256(conversion_source).hexdigest()
            if conversion_source
            else None
        ),
        "module_path": str(module_path) if module_path is not None else None,
        "module_sha256": module_sha256,
        "requested": "warp",
        "actual": actual,
        "runtime_fingerprint_sha256": runtime_identity[
            "fingerprint_sha256"
        ],
        "runtime_identity": runtime_identity,
    }
    return converted, actual, audit


@dataclass(frozen=True)
class PredictiveSamplingResult:
    """Compact result of one predictive-sampling or iterative-CEM solve."""

    selected_index: int
    selected_knots: np.ndarray
    selected_row: dict[str, Any]
    pool_size: int
    n_gaussian: int
    n_valid: int
    mean_cost: float
    flat_objective: bool
    validity_counts: dict[str, int] = field(default_factory=dict)
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE
    n_local_gaussian: int = 0
    n_broad_gaussian: int = 0
    sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION
    local_sigma_fraction: float = PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
    refit_mean: np.ndarray | None = None
    refit_std: np.ndarray | None = None
    rounds: int = 1
    total_rollouts: int = 0
    round_summaries: tuple[dict[str, Any], ...] = ()
    diagnostics: dict[str, Any] | None = None
    # Which CEM round produced the returned trajectory (1-based). Equals
    # ``rounds`` for predictive sampling and for an all-invalid CEM solve.
    selected_round: int = 1
    mppi_effective_samples: float | None = None
    mppi_next_temperature: float | None = None
    selected_proposal: np.ndarray | None = None
    selected_family_override: str | None = None

    @property
    def selected_family(self) -> str:
        """Return the proposal family containing the selected device index."""
        if self.selected_family_override is not None:
            return self.selected_family_override
        if self.sampler_mode == "mppi":
            return "softmax_weighted_mean"
        if self.selected_index == 0:
            return "mean" if self.sampler_mode == "cem" else "nominal"
        if self.sampler_mode == "cem":
            return "diagonal_gaussian"
        if self.sampler_mode == PREDICTIVE_SAMPLER_TWO_SCALE:
            if self.selected_index <= self.n_local_gaussian:
                return "local_diagonal_gaussian"
            return "broad_diagonal_gaussian"
        return "diagonal_gaussian"


def validate_predictive_sampler_mode(value: Any) -> str:
    """Return one explicit task-agnostic predictive-sampler distribution."""
    if not isinstance(value, str):
        raise ValueError(
            "sampler_mode must be one of "
            f"{PREDICTIVE_SAMPLER_MODES}, got {value!r}"
        )
    mode = value.strip()
    if mode not in PREDICTIVE_SAMPLER_MODES:
        raise ValueError(
            "sampler_mode must be one of "
            f"{PREDICTIVE_SAMPLER_MODES}, got {value!r}"
        )
    return mode


def validate_mtp_jaw_mode(value: Any) -> str:
    """Return one explicit MTP jaw exploration policy."""
    if not isinstance(value, str):
        raise ValueError(
            f"mtp_jaw_mode must be one of {MTP_JAW_MODES}, got {value!r}"
        )
    mode = value.strip()
    if mode not in MTP_JAW_MODES:
        raise ValueError(
            f"mtp_jaw_mode must be one of {MTP_JAW_MODES}, got {value!r}"
        )
    return mode


def validate_cem_rounds(value: Any) -> int:
    """Return an integer CEM round count."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("cem_rounds must be an integer >= 1")
    try:
        rounds = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cem_rounds must be an integer >= 1") from exc
    if rounds < 1 or not np.isfinite(numeric) or numeric != float(rounds):
        raise ValueError(
            f"cem_rounds must be an integer >= 1, got {value!r}"
        )
    return rounds


def validate_cem_elite_fraction(value: Any) -> float:
    """Return a finite elite fraction in ``(0, 1)``."""
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cem_elite_fraction must be finite and in (0, 1)") from exc
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(
            f"cem_elite_fraction must be finite and in (0, 1), got {value!r}"
        )
    return fraction


def validate_mppi_temperature(value: Any) -> float:
    """Return a finite positive MPPI softmax temperature."""
    try:
        temperature = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mppi_temperature must be finite and positive") from exc
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"mppi_temperature must be finite and positive, got {value!r}"
        )
    return temperature


def validate_arm_velocity_weight(value: Any) -> float:
    """Return one registered task-independent arm-qvel ablation weight."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"arm_velocity_weight must be one of {ARM_VELOCITY_WEIGHTS}"
        )
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"arm_velocity_weight must be one of {ARM_VELOCITY_WEIGHTS}"
        ) from exc
    if not np.isfinite(weight) or weight not in ARM_VELOCITY_WEIGHTS:
        raise ValueError(
            "arm_velocity_weight must be one of "
            f"{ARM_VELOCITY_WEIGHTS}, got {value!r}"
        )
    return weight


def validate_mppi_execution_mode(value: Any) -> str:
    """Return one task-independent policy for applying an MPPI population.

    The softmax mean is the MPPI-ISAAC baseline.  Executing the best valid
    sampled rollout is the policy used by Predictive Sampling and by the
    real-world MTP controller to avoid averaging distinct contact modes.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"mppi_execution_mode must be one of {MPPI_EXECUTION_MODES}, "
            f"got {value!r}"
        )
    mode = value.strip()
    if mode not in MPPI_EXECUTION_MODES:
        raise ValueError(
            f"mppi_execution_mode must be one of {MPPI_EXECUTION_MODES}, "
            f"got {value!r}"
        )
    return mode


def predictive_sampling_candidate_counts(
    pool_size: int,
    *,
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
) -> tuple[int, int]:
    """Return deterministic ``(local, broad)`` two-scale family counts.

    Candidate zero is always the exact nominal.  In ``two_scale`` mode the
    remaining candidates are divided as evenly as integer allocation permits;
    the broad family deterministically receives the one-candidate remainder,
    matching the audited pre-standardization local-family ordering.
    ``single_scale`` uses neither named sub-family and therefore returns
    ``(0, 0)``.
    """
    count = int(pool_size)
    if count < 2:
        raise ValueError("pool_size must be >= 2")
    mode = validate_predictive_sampler_mode(sampler_mode)
    if mode == PREDICTIVE_SAMPLER_SINGLE_SCALE:
        return 0, 0
    stochastic = count - 1
    n_local = stochastic // 2
    n_broad = stochastic - n_local
    return n_local, n_broad


def predictive_sampling_proposals(
    nominal_knots: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    pool_size: int,
    seed: int,
    sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
    sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
    local_sigma_fraction: float = (
        PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
    ),
    uniform_jaw_effort_column: bool = False,
) -> tuple[Any, int]:
    """Build one single-round predictive-sampling population on device.

    Candidate zero is the exact nominal.  The other ``N - 1`` candidates are
    independent diagonal-Gaussian perturbations of every future control knot.
    ``single_scale`` preserves the standard one-Gaussian distribution exactly.
    The explicit ``two_scale`` ablation divides the stochastic population
    deterministically between local and broad Gaussian scales; it adds no
    uniform/global proposal and performs no refinement round.  Both scales use
    actuator half-range normalization and sample all seven arm commands plus
    the gripper latent within fixed ``[-1, 1]`` bounds. Knot zero is copied
    from the measured boundary. No program predicate or held state changes
    those proposal bounds.

    ``uniform_jaw_effort_column`` makes each FUTURE jaw-effort knot uniform over
    ``[-1, 1]`` instead of Gaussian-around-nominal. A
    thresholded scalar is searchable only if its proposals STRADDLE the
    threshold: from an open nominal at -1, the standard 0.06 Gaussian scale
    puts zero 16.7 sigma away, so the uniform option is required when the jaw
    must explore both opening and closing efforts in one round. Arm columns
    keep their exact Gaussian draws
    (the PRNG consumption is unchanged); the nominal and every knot zero
    stay untouched.
    """
    import jax
    import jax.numpy as jnp

    shape = tuple(getattr(nominal_knots, "shape", ()))
    if len(shape) != 2 or shape[0] < 2 or shape[1] != NJ + 1:
        raise ValueError(
            f"nominal_knots must be (K, {NJ + 1}) with K >= 2, got {shape}"
        )
    lo_shape = tuple(getattr(ctrl_low, "shape", np.shape(ctrl_low)))
    hi_shape = tuple(getattr(ctrl_high, "shape", np.shape(ctrl_high)))
    if lo_shape != (NJ + 1,) or hi_shape != (NJ + 1,):
        raise ValueError(
            f"control bounds must both have shape ({NJ + 1},), "
            f"got {lo_shape} and {hi_shape}"
        )
    count = int(pool_size)
    if count < 2:
        raise ValueError("pool_size must be >= 2")
    mode = validate_predictive_sampler_mode(sampler_mode)
    broad_sigma = validate_local_sigma_fraction(sigma_fraction)
    local_sigma = validate_local_sigma_fraction(local_sigma_fraction)
    if (
        mode == PREDICTIVE_SAMPLER_TWO_SCALE
        and local_sigma >= broad_sigma
    ):
        raise ValueError(
            "two_scale requires local_sigma_fraction < sigma_fraction "
            f"(broad), got {local_sigma} >= {broad_sigma}"
        )
    n_gaussian = count - 1
    nominal = jnp.asarray(nominal_knots)
    lo = jnp.asarray(ctrl_low, dtype=nominal.dtype)
    hi = jnp.asarray(ctrl_high, dtype=nominal.dtype)
    half_range = 0.5 * (hi - lo)

    future_shape = (shape[0] - 1, shape[1])
    key = jax.random.PRNGKey(int(seed))
    if mode == PREDICTIVE_SAMPLER_SINGLE_SCALE:
        # Preserve the production default bit-for-bit: the same PRNG key,
        # shape, order, and single scale as before this optional ablation.
        noise = jax.random.normal(
            key,
            shape=(n_gaussian,) + future_shape,
            dtype=nominal.dtype,
        )
        sampled_future = jnp.clip(
            nominal[None, 1:, :]
            + noise * (broad_sigma * half_range)[None, None, :],
            lo,
            hi,
        )
    else:
        n_local, n_broad = predictive_sampling_candidate_counts(
            count,
            sampler_mode=mode,
        )
        local_key, broad_key = jax.random.split(key, 2)
        local_noise = jax.random.normal(
            local_key,
            shape=(n_local,) + future_shape,
            dtype=nominal.dtype,
        )
        broad_noise = jax.random.normal(
            broad_key,
            shape=(n_broad,) + future_shape,
            dtype=nominal.dtype,
        )
        local_future = jnp.clip(
            nominal[None, 1:, :]
            + local_noise * (local_sigma * half_range)[None, None, :],
            lo,
            hi,
        )
        broad_future = jnp.clip(
            nominal[None, 1:, :]
            + broad_noise * (broad_sigma * half_range)[None, None, :],
            lo,
            hi,
        )
        sampled_future = jnp.concatenate(
            [local_future, broad_future],
            axis=0,
        )
    if uniform_jaw_effort_column:
        # An independent stream (fold_in) so the arm columns' Gaussian draws
        # stay bit-identical with the flag on or off.
        width_key = jax.random.fold_in(key, NJ)
        width_future = jax.random.uniform(
            width_key,
            shape=(n_gaussian, shape[0] - 1),
            dtype=nominal.dtype,
            minval=lo[NJ],
            maxval=hi[NJ],
        )
        sampled_future = sampled_future.at[..., NJ].set(width_future)
    first = jnp.broadcast_to(
        nominal[0],
        (count - 1, 1, shape[1]),
    )
    sampled = jnp.concatenate([first, sampled_future], axis=1)
    proposals = jnp.concatenate([nominal[None, ...], sampled], axis=0)
    return proposals, n_gaussian


def cem_sampling_proposals(
    mean_knots: Any,
    std_knots: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    pool_size: int,
    seed: int,
    jaw_bernoulli: bool = False,
) -> Any:
    """Sample one diagonal-Gaussian CEM population on latent action knots.

    Candidate zero is the current mean. All stochastic candidates sample every
    future arm and jaw-latent knot; knot zero remains the measured boundary.

    ``jaw_bernoulli`` is retained for the legacy CEM width-command path. Only
    the SIGN of a jaw latent survives :func:`width_command_ctrl`, so a
    Gaussian fits a spread that has no physical meaning and, when the elites
    disagree about the jaw, centres the next round on a mean of zero -- the
    decision boundary itself, the one value carrying no intent. Under
    Bernoulli the jaw channel of ``mean_knots`` carries ``2p - 1`` for
    ``p = P(closed)``, which keeps the array contract intact and makes
    candidate zero the MAP command. This follows from the action variable
    being binary, and so holds for every problem the sampler is pointed at.
    """
    import jax
    import jax.numpy as jnp

    mean = jnp.asarray(mean_knots)
    std = jnp.asarray(std_knots, dtype=mean.dtype)
    lo = jnp.asarray(ctrl_low, dtype=mean.dtype)
    hi = jnp.asarray(ctrl_high, dtype=mean.dtype)
    shape = tuple(mean.shape)
    if len(shape) != 2 or shape[0] < 2 or shape[1] != NJ + 1:
        raise ValueError(
            f"mean_knots must be (K, {NJ + 1}) with K >= 2, got {shape}"
        )
    if std.shape != mean.shape:
        raise ValueError("std_knots must have the same shape as mean_knots")
    if lo.shape != (NJ + 1,) or hi.shape != (NJ + 1,):
        raise ValueError(
            f"control bounds must both have shape ({NJ + 1},)"
        )
    count = int(pool_size)
    if count < 2:
        raise ValueError("pool_size must be >= 2")
    # An off-by-default option must not move the default path. Splitting the
    # key unconditionally changed 63 of 64 candidates with jaw_bernoulli off,
    # which would silently void any comparison spanning that change.
    root_key = jax.random.PRNGKey(int(seed))
    if jaw_bernoulli:
        arm_key, jaw_key = jax.random.split(root_key)
    else:
        arm_key, jaw_key = root_key, root_key
    noise = jax.random.normal(
        arm_key,
        shape=(count - 1, shape[0] - 1, shape[1]),
        dtype=mean.dtype,
    )
    future = mean[None, 1:, :] + noise * std[None, 1:, :]
    if jaw_bernoulli:
        # p is carried on the latent axis as 2p - 1, so the bounds already
        # constrain it to [0, 1] and no separate channel is needed.
        closed_p = 0.5 * (mean[1:, NJ] + 1.0)
        draw = jax.random.uniform(
            jaw_key, shape=(count - 1, shape[0] - 1), dtype=mean.dtype
        )
        future = future.at[:, :, NJ].set(
            jnp.where(draw < closed_p[None, :], hi[NJ], lo[NJ])
        )
    future = jnp.clip(future, lo, hi)
    first = jnp.broadcast_to(mean[0], (count - 1, 1, shape[1]))
    sampled = jnp.concatenate((first, future), axis=1)
    return jnp.concatenate((mean[None, ...], sampled), axis=0)


_HALTON_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
    191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251,
    257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
)


@lru_cache(maxsize=16)
def _halton_prime_bases(dimensions: int) -> tuple[int, ...]:
    """Return enough prime bases without changing the legacy first 64.

    The original local implementation stopped at the fixed table above.  A
    reference-style effort panel with 30 actions, knot scale two, and eight
    control dimensions needs 15 * 8 = 120 Halton dimensions.  Extend the
    exact historical table deterministically instead of changing generators
    (or adding a runtime dependency), so every existing <=64-dimensional
    panel remains byte-identical.
    """
    count = int(dimensions)
    if count < 1:
        raise ValueError("Halton dimensions must be positive")
    if count <= len(_HALTON_PRIMES):
        return _HALTON_PRIMES[:count]

    bases = list(_HALTON_PRIMES)
    candidate = bases[-1] + 2
    while len(bases) < count:
        prime = True
        for divisor in bases:
            if divisor * divisor > candidate:
                break
            if candidate % divisor == 0:
                prime = False
                break
        if prime:
            bases.append(candidate)
        candidate += 2
    return tuple(bases)


def _halton_uniform(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Return a deterministic, digitally shifted Halton panel in ``(0, 1)``.

    Pezzato et al. draw Gaussian Halton knot points once and fit degree-one
    B-splines.  This local implementation keeps the same low-discrepancy
    construction without adding the paper repository's ``ghalton`` runtime
    dependency.  A Cranley-Patterson shift makes episode seeds distinct while
    preserving the panel for every MPC solve in that episode.
    """
    n = int(count)
    d = int(dimensions)
    if n < 1 or d < 1:
        raise ValueError("Halton count and dimensions must both be positive")
    bases = _halton_prime_bases(d)
    indices = np.arange(1, n + 1, dtype=np.int64)
    points = np.empty((n, d), dtype=np.float64)
    for column, base in enumerate(bases):
        work = indices.copy()
        factor = 1.0 / float(base)
        radical = np.zeros(n, dtype=np.float64)
        while np.any(work > 0):
            radical += factor * (work % base)
            work //= base
            factor /= float(base)
        points[:, column] = radical
    rng = np.random.default_rng(int(seed))
    points = np.mod(points + rng.random(d), 1.0)
    return np.clip(points, np.finfo(np.float32).eps, 1.0 - np.finfo(np.float32).eps)


def mppi_halton_spline_proposals(
    mean_knots: Any,
    std_knots: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    pool_size: int,
    seed: int,
) -> Any:
    """Generate fixed-variance Gaussian Halton degree-one spline proposals.

    The future knot values are the spline coefficients; production's existing
    linear knot interpolator is exactly a degree-one B-spline evaluator.  Knot
    zero remains the measured control boundary.  The last proposal is the
    zero-noise warm start, matching the reference implementation's
    ``sample_previous_plan`` behavior.
    """
    import jax.numpy as jnp
    from jax.scipy.special import ndtri

    mean = jnp.asarray(mean_knots)
    std = jnp.asarray(std_knots, dtype=mean.dtype)
    lo = jnp.asarray(ctrl_low, dtype=mean.dtype)
    hi = jnp.asarray(ctrl_high, dtype=mean.dtype)
    shape = tuple(mean.shape)
    if len(shape) != 2 or shape[0] < 2 or shape[1] != NJ + 1:
        raise ValueError(
            f"mean_knots must be (K, {NJ + 1}) with K >= 2, got {shape}"
        )
    if std.shape != mean.shape:
        raise ValueError("std_knots must have the same shape as mean_knots")
    if lo.shape != (NJ + 1,) or hi.shape != (NJ + 1,):
        raise ValueError(f"control bounds must both have shape ({NJ + 1},)")
    count = int(pool_size)
    if count < 2:
        raise ValueError("pool_size must be >= 2")
    dimensions = (shape[0] - 1) * shape[1]
    uniform = jnp.asarray(
        _halton_uniform(count - 1, dimensions, int(seed)),
        dtype=mean.dtype,
    )
    noise = ndtri(uniform).reshape((count - 1, shape[0] - 1, shape[1]))
    future = jnp.clip(
        mean[None, 1:, :] + noise * std[None, 1:, :],
        lo,
        hi,
    )
    first = jnp.broadcast_to(mean[0], (count - 1, 1, shape[1]))
    sampled = jnp.concatenate((first, future), axis=1)
    # The reference implementation explicitly inserts a zero-noise previous
    # plan.  Put it last so the first K-1 rows remain a pure Halton panel.
    return jnp.concatenate((sampled, mean[None, ...]), axis=0)


def paper_mppi_effort_proposals(
    mean_knots: Any,
    std_knots: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    pool_size: int,
    seed: int,
    sample_null_action: bool = True,
    null_action: Any | None = None,
) -> Any:
    """Paper MPPI Gaussian-Halton effort sequences, including action zero.

    Unlike the position-control samplers, every coefficient is an action in
    the reference implementation.  There is therefore no measured-position
    boundary to freeze at action zero. The final row is the zero-noise shifted
    action plan, matching ``sample_previous_plan`` in ``mppi_torch``.
    """
    import jax.numpy as jnp

    mean = jnp.asarray(mean_knots)
    std = jnp.asarray(std_knots, dtype=mean.dtype)
    lo = jnp.asarray(ctrl_low, dtype=mean.dtype)
    hi = jnp.asarray(ctrl_high, dtype=mean.dtype)
    shape = tuple(mean.shape)
    if len(shape) != 2 or shape[0] < 2 or shape[1] != NJ + 1:
        raise ValueError(
            f"mean_knots must be (K, {NJ + 1}) with K >= 2, got {shape}"
        )
    count = int(pool_size)
    reserved = 2 if sample_null_action else 1
    if count <= reserved:
        raise ValueError(
            "pool_size must leave room for Halton, previous-plan, and "
            "optional null proposals"
        )
    # The reference implementation uses knot_scale=2 and degree-one splines:
    # an even T-action horizon is driven by T/2 Halton knots per dimension.
    if shape[0] % 2:
        raise ValueError("paper MPPI action horizon must be divisible by 2")
    halton_count = count - reserved
    # The reference implementation calls scipy.interpolate.splrep with
    # degree=1 and smoothing s=0.5 for every Halton knot row.  This is not the
    # same as merely connecting the six knots with straight segments: the fit
    # itself smooths the coefficients before evaluating twelve actions.
    # Cache the fixed episode panel exactly once, matching mppi_torch's
    # ``self.delta`` lifetime across all receding-horizon commands.
    noise = jnp.asarray(
        _paper_mppi_halton_degree_one_noise(
            halton_count,
            shape[0],
            shape[1],
            int(seed),
        ),
        dtype=mean.dtype,
    )
    sampled = jnp.clip(
        mean[None, ...] + noise * std[None, ...], lo, hi
    )
    rows = [sampled]
    if sample_null_action:
        if null_action is None:
            null = jnp.zeros_like(mean)
        else:
            null_vector = jnp.asarray(null_action, dtype=mean.dtype)
            if null_vector.shape != (shape[1],):
                raise ValueError(
                    f"null_action must have shape ({shape[1]},)"
                )
            null = jnp.broadcast_to(null_vector, mean.shape)
        rows.append(jnp.clip(null, lo, hi)[None, ...])
    # The final row is always the zero-noise shifted previous plan, as in the
    # paper implementation.  With ``sample_null_action`` enabled, the
    # penultimate row is the explicit braking/hold proposal.
    rows.append(mean[None, ...])
    return jnp.concatenate(tuple(rows), axis=0)


def mtp_global_local_velocity_proposals(
    mean_knots: Any,
    std_knots: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    pool_size: int,
    seed: int,
    beta: float = 0.5,
    graph_depth: int = 3,
    graph_width: int = 50,
    jaw_mode: str = MTP_JAW_MODE_PRESERVE_NOMINAL,
) -> tuple[Any, int, int]:
    """Build an MTP-style global/local velocity-action population.

    One row is the current nominal. ``beta`` of the full budget is allocated
    to smooth paths through a random M-partite control graph; the remainder is
    sampled locally around the nominal.  This is the exploration mechanism of
    Le et al.'s Model Tensor Planning, adapted to the GN01 embodiment. The
    seven arm-velocity dimensions are always explored. By default the jaw
    target remains nominal, preserving the Push-T adapter exactly. The
    explicit sampled jaw mode explores its bounded continuous latent and
    reserves deterministic fully-closed and fully-open global paths.

    The global paths use piecewise-linear interpolation, one of the three
    interpolation families specified by MTP.  Candidate ordering is stable:
    nominal, global tensor paths, then local Gaussian paths.
    """
    import jax
    import jax.numpy as jnp

    mean = jnp.asarray(mean_knots)
    std = jnp.asarray(std_knots, dtype=mean.dtype)
    lo = jnp.asarray(ctrl_low, dtype=mean.dtype)
    hi = jnp.asarray(ctrl_high, dtype=mean.dtype)
    shape = tuple(mean.shape)
    if len(shape) != 2 or shape[0] < 2 or shape[1] != NJ + 1:
        raise ValueError(
            f"mean_knots must be (K, {NJ + 1}) with K >= 2, got {shape}"
        )
    if std.shape != mean.shape:
        raise ValueError("std_knots must have the same shape as mean_knots")
    if lo.shape != (NJ + 1,) or hi.shape != (NJ + 1,):
        raise ValueError(f"control bounds must both have shape ({NJ + 1},)")
    count = int(pool_size)
    depth = int(graph_depth)
    width = int(graph_width)
    mix = float(beta)
    if count < 3:
        raise ValueError("MTP pool_size must be >= 3")
    if depth < 2 or width < 2:
        raise ValueError("MTP graph depth and width must both be >= 2")
    if not np.isfinite(mix) or not 0.0 < mix < 1.0:
        raise ValueError("MTP beta must be finite and in (0, 1)")
    jaw_policy = validate_mtp_jaw_mode(jaw_mode)

    global_count = min(count - 1, max(1, int(round(mix * count))))
    if jaw_policy == MTP_JAW_MODE_SAMPLE:
        # Both actuator extremes are part of the sampled-jaw contract, even
        # for the smallest otherwise-valid diagnostic population.
        global_count = max(2, global_count)
    local_count = count - 1 - global_count
    root = jax.random.PRNGKey(int(seed))
    point_key, path_key, local_key = jax.random.split(root, 3)

    # MTP samples graph nodes over the full bounded control space. The default
    # keeps the jaw nominal so existing push tasks retain their exact panel.
    control_points = jax.random.uniform(
        point_key,
        shape=(depth, width, NJ + 1),
        dtype=mean.dtype,
        minval=lo,
        maxval=hi,
    )
    if jaw_policy == MTP_JAW_MODE_PRESERVE_NOMINAL:
        control_points = control_points.at[..., NJ].set(mean[0, NJ])
    path_indices = jax.random.randint(
        path_key,
        shape=(global_count, depth),
        minval=0,
        maxval=width,
    )
    paths = control_points[jnp.arange(depth)[None, :], path_indices]
    query = jnp.linspace(0.0, float(depth - 1), shape[0])
    lower = jnp.minimum(jnp.floor(query).astype(jnp.int32), depth - 2)
    alpha = query - lower.astype(query.dtype)
    global_knots = (
        paths[:, lower, :] * (1.0 - alpha)[None, :, None]
        + paths[:, lower + 1, :] * alpha[None, :, None]
    )
    # Float32 interpolation can land one ulp outside an endpoint even though
    # every graph node is bounded.  The controller contract is closed bounds.
    global_knots = jnp.clip(global_knots, lo, hi)
    if jaw_policy == MTP_JAW_MODE_PRESERVE_NOMINAL:
        global_knots = global_knots.at[..., NJ].set(mean[None, :, NJ])
    else:
        # GN01 width = 0.5 * (1 - latent) * open_width: the upper actuator
        # bound is fully closed and the lower bound is fully open.
        global_knots = global_knots.at[0, :, NJ].set(hi[NJ])
        if global_count > 1:
            global_knots = global_knots.at[1, :, NJ].set(lo[NJ])

    if local_count:
        local_noise = jax.random.normal(
            local_key,
            shape=(local_count,) + shape,
            dtype=mean.dtype,
        )
        local_knots = jnp.clip(
            mean[None, ...] + local_noise * std[None, ...], lo, hi
        )
        if jaw_policy == MTP_JAW_MODE_PRESERVE_NOMINAL:
            local_knots = local_knots.at[..., NJ].set(mean[None, :, NJ])
    else:  # pragma: no cover - beta is strictly below one.
        local_knots = jnp.empty((0,) + shape, dtype=mean.dtype)
    proposals = jnp.concatenate(
        (mean[None, ...], global_knots, local_knots), axis=0
    )
    return proposals, global_count, local_count


@lru_cache(maxsize=8)
def _paper_mppi_halton_degree_one_noise(
    sample_count: int,
    horizon: int,
    action_dim: int,
    seed: int,
) -> np.ndarray:
    """Build the fixed smoothed Halton-spline noise panel from mppi_torch."""
    from scipy.interpolate import splev, splrep
    from scipy.special import ndtri

    if horizon < 2 or horizon % 2:
        raise ValueError("paper MPPI horizon must be positive and divisible by 2")
    knot_count = horizon // 2
    dimensions = knot_count * action_dim
    gaussian = ndtri(
        _halton_uniform(sample_count, dimensions, int(seed))
    ).reshape((sample_count, action_dim, knot_count))
    knot_times = np.linspace(0.0, float(knot_count), knot_count)
    action_times = np.linspace(0.0, float(knot_count), horizon)
    panel = np.empty((sample_count, horizon, action_dim), dtype=float)
    for sample_index in range(sample_count):
        for action_index in range(action_dim):
            representation = splrep(
                knot_times,
                gaussian[sample_index, action_index],
                k=1,
                s=0.5,
            )
            panel[sample_index, :, action_index] = splev(
                action_times,
                representation,
                ext=3,
            )
    panel.setflags(write=False)
    return panel


def _knot_filter_window(num_knots: int) -> int:
    """Largest odd Savitzky--Golay window this many knots can support, capped at 9.

    The filter runs over the knot sequence, so its window is denominated in
    knots and cannot exceed the knot count.  At the production K=12 this
    returns the historical 9 and nothing moves.  Below K=9 a literal 9 raises
    ``action horizon must be at least window_length`` rather than filtering,
    which is what made the registered ``k8`` ablation unrunnable.

    ``polynomial_order=2`` at the call site requires a window of at least 3.
    """
    usable = min(9, int(num_knots))
    if usable % 2 == 0:
        usable -= 1
    return max(3, usable)


def paper_mppi_filter_action_plan(
    actions: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    window_length: int = 9,
    polynomial_order: int = 2,
) -> np.ndarray:
    """Apply the optional mppi_torch Savitzky--Golay control filter.

    ``mppi_torch`` uses SciPy's ``savgol_filter(..., mode='interp')`` on the
    weighted action sequence when ``filter_u`` is enabled. The public
    MPPI-ISAAC whole-body effort-pick configuration sets ``filter_u: false``;
    other examples enable it. Keeping this helper in NumPy avoids making SciPy
    a production dependency while retaining the optional operation exactly.
    """
    values = np.asarray(actions, dtype=float)
    if values.ndim != 2:
        raise ValueError("actions must be a two-dimensional (T, nu) array")
    horizon, action_dim = values.shape
    window = int(window_length)
    order = int(polynomial_order)
    if window < 1 or window % 2 != 1:
        raise ValueError("window_length must be a positive odd integer")
    if order < 0 or order >= window:
        raise ValueError("polynomial_order must be in [0, window_length)")
    if horizon < window:
        raise ValueError("action horizon must be at least window_length")
    lo = np.asarray(ctrl_low, dtype=float)
    hi = np.asarray(ctrl_high, dtype=float)
    if lo.shape != (action_dim,) or hi.shape != (action_dim,):
        raise ValueError("control bounds must match the action dimension")

    half = window // 2
    filtered = np.empty_like(values)
    for index in range(horizon):
        if index < half:
            start = 0
        elif index >= horizon - half:
            start = horizon - window
        else:
            start = index - half
        sample_indices = np.arange(start, start + window, dtype=float)
        design = np.stack(
            [sample_indices ** power for power in range(order + 1)], axis=1
        )
        evaluation = np.asarray(
            [float(index) ** power for power in range(order + 1)]
        )
        weights = evaluation @ np.linalg.pinv(design)
        filtered[index] = weights @ values[start:start + window]
    return np.clip(filtered, lo, hi)


def cem_elite_refit(
    proposals: Any,
    cost: Any,
    valid: Any,
    *,
    previous_mean: Any,
    previous_std: Any,
    elite_count: int,
    min_std: Any,
    jaw_bernoulli: bool = False,
    jaw_p_floor: float = 0.05,
) -> tuple[Any, Any]:
    """Refit diagonal mean/std from lowest-cost valid elites on device.

    ``jaw_bernoulli`` refits the jaw column as ``P(closed)`` among the
    elites, carried back on the latent axis as ``2p - 1``. ``jaw_p_floor``
    is the Bernoulli counterpart of ``min_std``: without it the probability
    absorbs at 0 or 1 after one unanimous round and the other jaw state is
    never sampled again, which would make any Stage requiring a jaw
    TRANSITION unreachable no matter how many rounds remain.
    """
    import jax.numpy as jnp

    samples = jnp.asarray(proposals)
    costs = jnp.asarray(cost)
    validity = jnp.asarray(valid, dtype=bool)
    old_mean = jnp.asarray(previous_mean, dtype=samples.dtype)
    old_std = jnp.asarray(previous_std, dtype=samples.dtype)
    floor = jnp.asarray(min_std, dtype=samples.dtype)
    if samples.ndim != 3 or samples.shape[2] != NJ + 1:
        raise ValueError(
            f"proposals must be (N, K, {NJ + 1}), got {samples.shape}"
        )
    if costs.shape != (samples.shape[0],) or validity.shape != costs.shape:
        raise ValueError("cost and valid must match the proposal population")
    if old_mean.shape != samples.shape[1:] or old_std.shape != old_mean.shape:
        raise ValueError("previous mean/std must match one proposal")
    count = int(elite_count)
    if count < 1 or count > samples.shape[0]:
        raise ValueError("elite_count must be in [1, pool_size]")

    executable = validity & jnp.isfinite(costs)
    ranked = jnp.where(executable, costs, jnp.inf)
    elite_indices = jnp.argsort(ranked)[:count]
    elites = samples[elite_indices]
    elite_mask = executable[elite_indices].astype(samples.dtype)
    denominator = jnp.maximum(jnp.sum(elite_mask), 1.0)
    fitted_mean = jnp.sum(
        elites * elite_mask[:, None, None],
        axis=0,
    ) / denominator
    centered = elites - fitted_mean[None, ...]
    fitted_var = jnp.sum(
        centered * centered * elite_mask[:, None, None],
        axis=0,
    ) / denominator
    fitted_std = jnp.maximum(jnp.sqrt(fitted_var), floor)
    has_elite = jnp.any(executable)
    new_mean = jnp.where(has_elite, fitted_mean, old_mean)
    new_std = jnp.where(has_elite, fitted_std, old_std)
    if jaw_bernoulli:
        floor = float(jaw_p_floor)
        if not 0.0 < floor < 0.5:
            raise ValueError("jaw_p_floor must lie in (0, 0.5)")
        closed = (elites[..., NJ] > 0.0).astype(samples.dtype)
        closed_p = jnp.sum(
            closed * elite_mask[:, None], axis=0
        ) / denominator
        closed_p = jnp.clip(closed_p, floor, 1.0 - floor)
        fitted_jaw = 2.0 * closed_p - 1.0
        new_mean = new_mean.at[:, NJ].set(
            jnp.where(has_elite, fitted_jaw, old_mean[:, NJ])
        )
        # The spread of a Bernoulli is implied by p; carrying a Gaussian
        # width here as well would let the proposal reintroduce it.
        new_std = new_std.at[:, NJ].set(jnp.zeros_like(old_std[:, NJ]))
    # The measured boundary is not an optimization variable.
    new_mean = new_mean.at[0].set(old_mean[0])
    new_std = new_std.at[0].set(jnp.zeros_like(old_std[0]))
    return new_mean, new_std


def device_action_rate_cost(
    action_knots: Any,
    action_span: Any,
    *,
    weight: Any = DEFAULT_ACTION_RATE_WEIGHT,
    continuous_jaw: bool = False,
) -> Any:
    """Return one dimensionless commanded-action rate cost per candidate.

    Arm columns use the continuous joint-target knots. The jaw column is
    decoded to exact open/closed command levels before differencing, so latent
    motion within one sign has zero physical-action rate while any actual jaw
    switch pays the full normalized transition.
    """
    import jax.numpy as jnp

    knots = jnp.asarray(action_knots)
    span = jnp.asarray(action_span, dtype=knots.dtype)
    if knots.ndim != 3 or knots.shape[1] < 2 or knots.shape[2] != NJ + 1:
        raise ValueError(
            f"action_knots must be (N, K, {NJ + 1}) with K >= 2"
        )
    if span.shape != (NJ + 1,):
        raise ValueError(f"action_span must have shape ({NJ + 1},)")
    jaw_command = (
        knots[..., NJ]
        if continuous_jaw
        else jnp.where(knots[..., NJ] > 0.0, 1.0, -1.0)
    )
    commanded_knots = knots.at[..., NJ].set(jaw_command)
    normalized_delta = (
        jnp.diff(commanded_knots, axis=1)
        / span[None, None, :]
    )
    return jnp.asarray(weight, dtype=knots.dtype) * jnp.mean(
        jnp.sum(normalized_delta * normalized_delta, axis=2),
        axis=1,
    )


def device_arm_velocity_cost(
    arm_velocity_squared_sum: Any,
    *,
    weight: Any = DEFAULT_ARM_VELOCITY_WEIGHT,
) -> Any:
    """Weight the control-step sum of squared physical arm-joint velocity.

    The input is produced by the full-physics scan from actual ``qvel`` after
    each integration step.  It is not inferred from torque knots and is not a
    task-specific pose, waypoint, reference trajectory, or motion initializer.
    """
    import jax.numpy as jnp

    value = jnp.asarray(arm_velocity_squared_sum)
    if value.ndim != 1:
        raise ValueError(
            "arm_velocity_squared_sum must be one vector, got "
            f"{value.shape}"
        )
    return jnp.asarray(weight, dtype=value.dtype) * value


def device_action_rate_terms(
    action_knots: Any,
    action_span: Any,
    *,
    weight: Any = DEFAULT_ACTION_RATE_WEIGHT,
    continuous_jaw: bool = False,
) -> tuple[Any, Any]:
    """Return the arm and decoded-jaw components of the action-rate cost.

    Diagnostics only: selection keeps using :func:`device_action_rate_cost`,
    whose single mean-of-8-dim reduction this split reproduces up to float
    association (arm + jaw may differ from the total by rounding).
    """
    import jax.numpy as jnp

    knots = jnp.asarray(action_knots)
    span = jnp.asarray(action_span, dtype=knots.dtype)
    if knots.ndim != 3 or knots.shape[1] < 2 or knots.shape[2] != NJ + 1:
        raise ValueError(
            f"action_knots must be (N, K, {NJ + 1}) with K >= 2"
        )
    if span.shape != (NJ + 1,):
        raise ValueError(f"action_span must have shape ({NJ + 1},)")
    weight_value = jnp.asarray(weight, dtype=knots.dtype)
    arm_delta = jnp.diff(knots[..., :NJ], axis=1) / span[None, None, :NJ]
    arm = weight_value * jnp.mean(
        jnp.sum(arm_delta * arm_delta, axis=2),
        axis=1,
    )
    jaw_command = (
        knots[..., NJ]
        if continuous_jaw
        else jnp.where(knots[..., NJ] > 0.0, 1.0, -1.0)
    )
    jaw_delta = jnp.diff(jaw_command, axis=1) / span[NJ]
    jaw = weight_value * jnp.mean(jaw_delta * jaw_delta, axis=1)
    return arm, jaw


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    """Compact p10/p50/p90/max summary of one small host vector."""
    if values.size == 0:
        return {"p10": None, "p50": None, "p90": None, "max": None}
    p10, p50, p90 = (
        float(v) for v in np.percentile(values, [10.0, 50.0, 90.0])
    )
    return {"p10": p10, "p50": p50, "p90": p90, "max": float(np.max(values))}


def joint_feasibility_sets(
    *,
    term_descriptors: tuple[dict[str, Any], ...],
    per_term: np.ndarray,
    cost: np.ndarray,
    valid: np.ndarray,
    prefix_held: np.ndarray,
    endpoint_held: np.ndarray,
    endpoint_jaw_effort_latent: np.ndarray,
    selected_index: int,
) -> dict[str, Any]:
    """Which candidates in THIS production round were jointly feasible.

    A summary line cannot tell two failures apart: a feasible candidate that
    lost the ranking, and a population that never contained one. This reports
    the joint sets over the round's whole population, using the production
    validity flag and the production total cost (which already includes the
    action-rate term), so the answer is about the real optimizer rather than
    a post-hoc resample.

    Endpoint predicates are read from the per-term breakdown: a terminal term
    contributes exactly zero iff its residual is satisfied at the horizon
    endpoint. Jaw-open is the negative sign of the signed-effort action the
    candidate actually ends on.
    """
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    cost = np.asarray(cost, dtype=float).reshape(-1)
    prefix_held = np.asarray(prefix_held, dtype=bool).reshape(-1)
    endpoint_held = np.asarray(endpoint_held, dtype=bool).reshape(-1)
    command = np.asarray(endpoint_jaw_effort_latent, dtype=float).reshape(-1)
    term_multipliers = np.asarray(
        cost_shaping_term_multipliers(term_descriptors),
        dtype=float,
    )
    raw_per_term = np.asarray(per_term, dtype=float)
    shaped_per_term = raw_per_term * term_multipliers[:, None]

    def terminal_satisfied(kind: str) -> np.ndarray | None:
        for index, descriptor in enumerate(term_descriptors):
            if (
                descriptor.get("scope") == "terminal"
                and descriptor.get("type") == kind
                and index < per_term.shape[0]
            ):
                # Satisfaction is a property of the residual, not of how it is
                # weighted.  Reading the shaped value would make a registered
                # multiplier of 0 mark every candidate satisfied and silently
                # corrupt the joint-feasibility statistic this diagnoses.
                return raw_per_term[index] <= 1e-12
        return None

    members: dict[str, np.ndarray] = {
        "valid": valid,
        "prefix_held": prefix_held,
        "endpoint_held": endpoint_held,
        "endpoint_not_held": ~endpoint_held,
        "endpoint_jaw_open": command < 0.0,
    }
    for name, kind in (
        ("endpoint_supported", "supported_on"),
        ("endpoint_at_rest", "at_rest"),
    ):
        mask = terminal_satisfied(kind)
        if mask is not None:
            members[name] = mask

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        if not count:
            return {"count": 0, "min_cost": None, "argmin": None}
        indices = np.flatnonzero(mask)
        best = int(indices[np.argmin(cost[indices])])
        return {
            "count": count,
            "min_cost": float(cost[best]),
            "argmin": best,
        }

    sets: dict[str, Any] = {
        name: summarize(mask) for name, mask in members.items()
    }
    for name, mask in members.items():
        if name == "valid":
            continue
        sets[f"valid_and_{name}"] = summarize(valid & mask)

    # "Jointly feasible" means: production-valid AND satisfying every terminal
    # this Stage actually declares. Deriving that from the Stage's own terms
    # rather than a fixed list is what makes the statistic correct for a
    # release Stage (which requires NOT held) as well as a carry Stage (which
    # requires held) -- a hard-coded held requirement silently reported zero
    # for every release solve.
    joint = valid.copy()
    terminal_indices = [
        index for index, descriptor in enumerate(term_descriptors)
        if descriptor.get("scope") == "terminal" and index < per_term.shape[0]
    ]
    for index in terminal_indices:
        # Unweighted, for the same reason as ``terminal_satisfied`` above.
        joint = joint & (raw_per_term[index] <= 1e-12)
    sets["joint_feasible"] = summarize(joint)

    winner_sets = sorted(
        name for name, mask in members.items()
        if 0 <= selected_index < mask.size and bool(mask[selected_index])
    )
    # When a jointly feasible candidate exists but the winner is not it, the
    # decisive datum is WHICH term the winner is saving on. Record both
    # breakdowns so "ranking preferred something cheaper" becomes specific.
    def breakdown(index: int | None) -> dict[str, float] | None:
        if index is None or not (0 <= index < cost.size):
            return None
        return {
            f"{d.get('scope')}:{d.get('type')}": float(
                shaped_per_term[i, index]
            )
            for i, d in enumerate(term_descriptors)
            if i < per_term.shape[0]
        }

    joint_argmin = sets["joint_feasible"].get("argmin")
    return {
        "population": int(cost.size),
        # Every value in ``sets`` is a {count, min_cost, argmin} summary, so
        # the term list that DEFINES the joint set lives beside it.
        "joint_feasible_terms": [
            term_descriptors[index].get("type") for index in terminal_indices
        ],
        "sets": sets,
        "winner_index": int(selected_index),
        "winner_cost": (
            float(cost[selected_index])
            if 0 <= selected_index < cost.size
            else None
        ),
        "winner_member_of": winner_sets,
        "winner_is_joint_feasible": bool(
            0 <= selected_index < joint.size and joint[selected_index]
        ),
        "winner_per_term": breakdown(selected_index),
        "cheapest_joint_feasible_per_term": breakdown(joint_argmin),
    }


def solve_term_diagnostics(
    *,
    term_descriptors: tuple[dict[str, Any], ...],
    running_scales: np.ndarray,
    terminal_scales: np.ndarray,
    per_term: np.ndarray,
    arm_rate: np.ndarray,
    jaw_rate: np.ndarray,
    arm_velocity: np.ndarray,
    table_force: np.ndarray | None = None,
    first_fail_component: np.ndarray | None = None,
    validity_counts: dict[str, int] | None = None,
    control_profile: str | None = None,
    arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
    record_candidate_cost_telemetry: bool = False,
    control_regularizer_calibration_profile: str | None = None,
    control_regularizer_raw: np.ndarray | None = None,
    control_regularizer_base_cost: np.ndarray | None = None,
    control_regularizer_weighted: np.ndarray | None = None,
    control_regularizer_boundary_valid: bool | None = None,
    cost: np.ndarray,
    valid: np.ndarray,
    selected_index: int,
    elite_count: int,
    joint_sets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize one solve's per-term/rate/validity structure for logging.

    All inputs are small host arrays already copied off the device. Terminal
    raw endpoint residuals are recovered from their squared scaled
    contribution; running contributions are horizon integrals, so only their
    scaled value is reported (the fresh measured residual after each prefix is
    the raw-value channel).
    """
    valid = valid.astype(bool)
    term_multipliers = np.asarray(
        cost_shaping_term_multipliers(term_descriptors),
        dtype=float,
    )
    if term_multipliers.size != len(term_descriptors):
        raise ValueError("cost-shaping multipliers must align with descriptors")
    shaping_profiles = {
        str(descriptor["cost_shaping"]["profile"])
        for descriptor in term_descriptors
        if isinstance(descriptor.get("cost_shaping"), dict)
    }
    if len(shaping_profiles) > 1:
        raise ValueError("term descriptors mix cost_shaping profiles")
    shaping_profile = next(iter(shaping_profiles), None)
    per_term_post = (
        np.asarray(per_term, dtype=float)
        * term_multipliers[:, None]
    )
    valid_indices = np.flatnonzero(valid)
    elite_indices: np.ndarray = np.empty(0, dtype=int)
    if elite_count > 0 and valid_indices.size > 0:
        ranked_valid = valid_indices[np.argsort(cost[valid_indices])]
        elite_indices = ranked_valid[: int(elite_count)]
    scales = list(map(float, running_scales)) + list(
        map(float, terminal_scales)
    )
    terms: list[dict[str, Any]] = []
    for index, descriptor in enumerate(term_descriptors):
        row_pre = per_term[index] if index < per_term.shape[0] else np.empty(0)
        row = (
            per_term_post[index]
            if index < per_term_post.shape[0]
            else np.empty(0)
        )
        selected_value = (
            float(row[selected_index])
            if 0 <= selected_index < row.size
            else None
        )
        entry: dict[str, Any] = {
            **descriptor,
            "scale": scales[index] if index < len(scales) else None,
            "selected_scaled_contribution": selected_value,
            "valid_quantiles": _quantiles(row[valid_indices]),
            "elite_quantiles": (
                _quantiles(row[elite_indices])
                if elite_indices.size
                else None
            ),
        }
        if shaping_profile is not None:
            selected_pre = (
                float(row_pre[selected_index])
                if 0 <= selected_index < row_pre.size
                else None
            )
            entry.update({
                "selected_scaled_contribution_pre": selected_pre,
                "selected_scaled_contribution_post": selected_value,
                "cost_shaping_multiplier": float(term_multipliers[index]),
            })
        if (
            descriptor.get("scope") == "terminal"
            and selected_value is not None
            and entry["scale"] is not None
        ):
            entry["selected_raw_endpoint"] = float(
                entry["scale"] * math.sqrt(max(0.0, selected_value))
            )
        terms.append(entry)
    typed_task_pre_total = np.sum(per_term, axis=0)
    typed_task_total = np.sum(per_term_post, axis=0)
    selected = int(selected_index)
    selected_in_population = 0 <= selected < cost.size
    table_force_array = (
        np.asarray(table_force, dtype=float).reshape(-1)
        if table_force is not None
        else np.zeros_like(cost, dtype=float)
    )
    first_fail_array = (
        np.asarray(first_fail_component, dtype=np.int32).reshape(-1)
        if first_fail_component is not None
        else np.full(cost.size, -1, dtype=np.int32)
    )
    if table_force_array.size != cost.size:
        raise ValueError("table_force must align with the candidate population")
    if first_fail_array.size != cost.size:
        raise ValueError(
            "first_fail_component must align with the candidate population"
        )

    def optional_float(value: Any) -> float | None:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None

    def vector(values: np.ndarray) -> list[float | None]:
        return [optional_float(value) for value in np.asarray(values).reshape(-1)]

    validity_component_names = control_profile_validity_component_names(
        control_profile
    )
    first_fail_names: list[str | None] = [
        (
            validity_component_names[int(index)]
            if 0 <= int(index) < len(validity_component_names)
            else None
        )
        for index in first_fail_array
    ]
    winner_breakdown = {
        "candidate_index": selected if selected_in_population else None,
        "typed_task_total": (
            optional_float(typed_task_total[selected])
            if selected_in_population else None
        ),
        "typed_per_term": {
            f"{descriptor.get('scope')}:{descriptor.get('type')}": (
                optional_float(per_term_post[index, selected])
                if selected_in_population and index < per_term.shape[0]
                else None
            )
            for index, descriptor in enumerate(term_descriptors)
        },
        "action_rate_arm": (
            optional_float(arm_rate[selected]) if selected_in_population else None
        ),
        "action_rate_jaw": (
            optional_float(jaw_rate[selected]) if selected_in_population else None
        ),
        "arm_velocity_cost": (
            optional_float(arm_velocity[selected])
            if selected_in_population else None
        ),
        "table_force_cost": (
            optional_float(table_force_array[selected])
            if selected_in_population else None
        ),
        "total_cost": (
            optional_float(cost[selected]) if selected_in_population else None
        ),
        "valid": bool(valid[selected]) if selected_in_population else None,
        "first_fail_component": (
            first_fail_names[selected] if selected_in_population else None
        ),
    }
    diagnostics = {
        "schema": "oap_solve_diagnostics_v1",
        "joint_feasibility": joint_sets,
        "winner_cost_breakdown": winner_breakdown,
        "validity_fail_counts": dict(validity_counts or {}),
        "terms": terms,
        "action_rate": {
            "selected_arm": (
                float(arm_rate[selected_index])
                if 0 <= selected_index < arm_rate.size
                else None
            ),
            "selected_jaw_switch": (
                float(jaw_rate[selected_index])
                if 0 <= selected_index < jaw_rate.size
                else None
            ),
            "valid_arm": _quantiles(arm_rate[valid_indices]),
            "valid_jaw_switch": _quantiles(jaw_rate[valid_indices]),
        },
        "arm_velocity": {
            "weight": float(validate_arm_velocity_weight(arm_velocity_weight)),
            "selected_cost": (
                float(arm_velocity[selected_index])
                if 0 <= selected_index < arm_velocity.size
                else None
            ),
            "valid_cost": _quantiles(arm_velocity[valid_indices]),
        },
        "cost": {
            "selected": (
                float(cost[selected_index])
                if 0 <= selected_index < cost.size
                else None
            ),
            "valid": _quantiles(cost[valid_indices]),
            "elite": (
                _quantiles(cost[elite_indices])
                if elite_indices.size
                else None
            ),
        },
        "population": {
            "n": int(cost.size),
            "n_valid": int(valid_indices.size),
            "valid_fraction": (
                float(valid_indices.size / cost.size) if cost.size else 0.0
            ),
            "elite_count": int(elite_count),
        },
    }
    if record_candidate_cost_telemetry:
        diagnostics["candidate_cost_telemetry"] = {
            "schema": "oap_candidate_cost_telemetry_v1",
            "representation": "columnar_candidate_index_order",
            "n": int(cost.size),
            "typed_task_total": vector(typed_task_total),
            "typed_per_term": [vector(row) for row in per_term_post],
            "action_rate_arm": vector(arm_rate),
            "action_rate_jaw": vector(jaw_rate),
            "arm_velocity_cost": vector(arm_velocity),
            "table_force_cost": vector(table_force_array),
            "total_cost": vector(cost),
            "valid": np.asarray(valid, dtype=bool).tolist(),
            "first_fail_component": first_fail_names,
            "validity_fail_counts": dict(validity_counts or {}),
        }
        if shaping_profile is not None:
            diagnostics["candidate_cost_telemetry"].update({
                "typed_task_pre_total": vector(typed_task_pre_total),
                "typed_task_post_total": vector(typed_task_total),
                "typed_task_removed": vector(
                    typed_task_pre_total - typed_task_total
                ),
                "typed_per_term_pre": [vector(row) for row in per_term],
                "typed_per_term_post": [
                    vector(row) for row in per_term_post
                ],
            })
    if shaping_profile is not None:
        zeroed_running_terms = [
            descriptor["cost_shaping"]["identity"]
            for descriptor in term_descriptors
            if (
                descriptor.get("scope") == "running"
                and isinstance(descriptor.get("cost_shaping"), dict)
                and descriptor["cost_shaping"].get("zeroed_running_twin")
            )
        ]
        selected_typed_pre = (
            optional_float(typed_task_pre_total[selected])
            if selected_in_population else None
        )
        selected_typed_post = (
            optional_float(typed_task_total[selected])
            if selected_in_population else None
        )
        typed_closure = (
            optional_float(
                typed_task_total[selected]
                - np.sum(per_term_post[:, selected])
            )
            if selected_in_population else None
        )
        base_cost_array = (
            np.asarray(control_regularizer_base_cost, dtype=float).reshape(-1)
            if control_regularizer_base_cost is not None
            else np.asarray(cost, dtype=float).reshape(-1)
        )
        base_column = selected if base_cost_array.size == cost.size else 0
        base_closure = None
        if selected_in_population and base_cost_array.size in (1, cost.size):
            base_closure = optional_float(
                base_cost_array[base_column]
                - typed_task_total[selected]
                - arm_rate[selected]
                - jaw_rate[selected]
                - arm_velocity[selected]
                - table_force_array[selected]
            )
        diagnostics["cost_shaping"] = {
            "schema": "oap_cost_shaping_telemetry_v1",
            "identity": cost_shaping_identity(shaping_profile),
            "zeroed_running_terms": zeroed_running_terms,
            "selected": {
                "typed_task_pre": selected_typed_pre,
                "typed_task_post": selected_typed_post,
                "removed": (
                    None
                    if selected_typed_pre is None or selected_typed_post is None
                    else selected_typed_pre - selected_typed_post
                ),
                "typed_cost_closure_residual": typed_closure,
                "base_cost_closure_residual": base_closure,
            },
        }
    calibration_profile = validate_control_regularizer_calibration_profile(
        control_regularizer_calibration_profile,
        allow_none=True,
    )
    if calibration_profile is not None:
        raw = np.asarray(control_regularizer_raw, dtype=float)
        expected_columns = cost.size if record_candidate_cost_telemetry else 1
        if raw.shape != (3, expected_columns):
            raise ValueError(
                "control_regularizer_raw must have shape (3, population) "
                "when candidate telemetry is recorded and (3, 1) otherwise"
            )
        if not isinstance(control_regularizer_boundary_valid, bool):
            raise ValueError(
                "active control regularizer requires typed boundary validity"
            )
        raw_names = (
            "realized_arm_qvel_si_time_mean",
            "command_envelope_normalized_magnitude_time_mean",
            "boundary_complete_normalized_command_rate_time_mean",
        )
        calibration = {
            "schema": "oap_control_regularizer_calibration_v1",
            "identity": control_regularizer_formula_identity(
                calibration_profile
            ),
            "boundary_valid": control_regularizer_boundary_valid,
            "selected": {
                name: (
                    optional_float(
                        raw[
                            index,
                            selected if record_candidate_cost_telemetry else 0,
                        ]
                    )
                    if selected_in_population else None
                )
                for index, name in enumerate(raw_names)
            },
        }
        if record_candidate_cost_telemetry:
            calibration["population_raw"] = {
                name: vector(raw[index])
                for index, name in enumerate(raw_names)
            }
        if calibration_profile != UNIFIED_CONTROL_REGULARIZER_V1:
            diagnostics["control_regularizer_calibration"] = calibration
        else:
            weighted = np.asarray(control_regularizer_weighted, dtype=float)
            base_cost = np.asarray(
                control_regularizer_base_cost,
                dtype=float,
            ).reshape(-1)
            if weighted.shape != raw.shape:
                raise ValueError(
                    "control_regularizer_weighted must align with raw terms"
                )
            if base_cost.size != expected_columns:
                raise ValueError(
                    "control_regularizer_base_cost must align with raw columns"
                )
            raw_finite = np.all(np.isfinite(raw), axis=0)
            weighted_finite = np.all(np.isfinite(weighted), axis=0)
            selected_valid = (
                bool(valid[selected]) if selected_in_population else False
            )
            if record_candidate_cost_telemetry:
                if np.any(valid & ~(raw_finite & weighted_finite)):
                    raise ValueError(
                        "formal regularizer valid candidates must have finite raw "
                        "and weighted terms"
                    )
                nonfinite_matches_invalid = bool(
                    np.all((raw_finite & weighted_finite) | (~valid))
                )
                selected_column = selected
            else:
                selected_column = 0
                if selected_valid and not bool(
                    raw_finite[0] and weighted_finite[0]
                ):
                    raise ValueError(
                        "formal regularizer selected candidate must be finite"
                    )
                nonfinite_matches_invalid = bool(
                    (raw_finite[0] and weighted_finite[0])
                    or not selected_valid
                )
            selected_terms_finite = bool(
                raw_finite[selected_column] and weighted_finite[selected_column]
            )
            if selected_valid and not selected_terms_finite:
                raise ValueError(
                    "formal regularizer selected candidate must be finite"
                )
            selected_weighted = weighted[:, selected_column]
            selected_base = float(base_cost[selected_column])
            selected_total = float(cost[selected])
            weighted_sum = float(np.sum(selected_weighted))
            selected_closure_finite = bool(np.all(
                np.isfinite(
                    [selected_base, selected_total, weighted_sum]
                )
            ))
            if selected_valid and not selected_closure_finite:
                raise ValueError(
                    "formal regularizer selected cost closure must be finite"
                )
            selected_weighted_sum = optional_float(weighted_sum)
            selected_base_total = optional_float(selected_base)
            selected_total_cost = optional_float(selected_total)
            selected_closure_residual = (
                optional_float(selected_total - selected_base - weighted_sum)
                if selected_closure_finite else None
            )
            formal = {
                "schema": "oap_unified_control_regularizer_v1",
                "identity": control_regularizer_formula_identity(
                    calibration_profile
                ),
                "boundary_valid": control_regularizer_boundary_valid,
                "selected": {
                    "valid": selected_valid,
                    "raw": calibration["selected"],
                    "weighted": {
                        name: optional_float(selected_weighted[index])
                        for index, name in enumerate(raw_names)
                    },
                    "weighted_sum": selected_weighted_sum,
                    "base_total_cost": selected_base_total,
                    "total_cost": selected_total_cost,
                    "closure_residual": selected_closure_residual,
                    "raw_finite": bool(raw_finite[selected_column]),
                    "weighted_finite": bool(
                        weighted_finite[selected_column]
                    ),
                    "nonfinite_allowed_because_invalid": bool(
                        not selected_valid and not selected_terms_finite
                    ),
                    "invalid_reason": (
                        {
                            "device_program_valid": False,
                            "first_fail_component": (
                                first_fail_names[selected]
                                if selected_in_population else None
                            ),
                        }
                        if not selected_valid else None
                    ),
                },
                "nonfinite_raw_matches_invalid": nonfinite_matches_invalid,
            }
            if record_candidate_cost_telemetry:
                formal["population_raw"] = calibration["population_raw"]
                formal["population_weighted"] = {
                    name: vector(weighted[index])
                    for index, name in enumerate(raw_names)
                }
                population_weighted_sum = np.sum(weighted, axis=0)
                formal["population_weighted_sum"] = vector(
                    population_weighted_sum
                )
                formal["population_base_total_cost"] = vector(base_cost)
                formal["population_total_cost"] = vector(cost)
                # Invalid candidates may intentionally carry a non-finite raw
                # term.  Preserve that evidence without emitting a host-side
                # arithmetic warning; no value is filled or made selectable.
                with np.errstate(invalid="ignore"):
                    population_closure_residual = (
                        cost - base_cost - population_weighted_sum
                    )
                formal["population_closure_residual"] = vector(
                    population_closure_residual
                )
            diagnostics["control_regularizer"] = formal
    return diagnostics


def device_action_feasibility(
    action_knots: Any,
    *,
    continuous_jaw: bool = False,
    action_feasibility_enabled: Any,
    controller_timing_compatible: Any,
    controller_prefix_knot_times: Any,
    controller_segment_durations_s: Any,
    joint_position_min_rad: Any,
    joint_position_max_rad: Any,
    joint_velocity_max_rad_s: Any,
    gripper_width_min_m: Any,
    gripper_width_max_m: Any,
    gripper_velocity_min_m_s: Any,
    gripper_velocity_max_m_s: Any,
) -> tuple[Any, Any, Any, Any]:
    """Return device-resident actuator feasibility masks for every candidate.

    The mask covers every original linear-spline break that falls in the
    execution prefix plus the exact selected rollout endpoint. Slew is checked
    between the exact targets over their quantized controller durations before
    the device cost argmin is evaluated.
    """
    import jax.numpy as jnp

    knots = jnp.asarray(action_knots)
    if knots.ndim != 3 or knots.shape[1] < 2 or knots.shape[2] != NJ + 1:
        raise ValueError(
            f"action_knots must be (N, K, {NJ + 1}) with K >= 2, "
            f"got {knots.shape}"
        )
    enabled = jnp.asarray(action_feasibility_enabled, dtype=bool)
    knot_count = knots.shape[1]
    prefix_times = jnp.asarray(controller_prefix_knot_times)
    segment_durations = jnp.asarray(controller_segment_durations_s)
    scaled_times = prefix_times * (knot_count - 1)
    target_lower = jnp.minimum(
        jnp.floor(scaled_times).astype(jnp.int32),
        knot_count - 2,
    )
    target_alpha = scaled_times - target_lower.astype(scaled_times.dtype)
    controller_targets = (
        knots[:, target_lower, :] * (1.0 - target_alpha)[None, :, None]
        + knots[:, target_lower + 1, :] * target_alpha[None, :, None]
    )
    jaw_targets = mppi_continuous_effort_ctrl(
        controller_targets[..., NJ],
        clip=jnp.clip,
    )
    controller_targets = controller_targets.at[..., NJ].set(jaw_targets)

    joint_lo = jnp.asarray(joint_position_min_rad)
    joint_hi = jnp.asarray(joint_position_max_rad)
    joint_position_ok = (~enabled) | jnp.all(
        (controller_targets[..., :NJ] >= joint_lo)
        & (controller_targets[..., :NJ] <= joint_hi),
        axis=(1, 2),
    )

    controller_rates = jnp.abs(
        jnp.diff(controller_targets, axis=1)
    ) / segment_durations[None, :, None]
    joint_velocity_ok = (~enabled) | jnp.all(
        controller_rates[..., :NJ] <= jnp.asarray(joint_velocity_max_rad_s),
        axis=(1, 2),
    )

    width_lo = jnp.asarray(-GN01_DIRECT_FORCE_LIMIT_N)
    width_hi = jnp.asarray(GN01_DIRECT_FORCE_LIMIT_N)
    gripper_width_ok = (~enabled) | jnp.all(
        (controller_targets[..., NJ] >= width_lo)
        & (controller_targets[..., NJ] <= width_hi),
        axis=1,
    )
    # RDK Grasp is a direct effort command and specifies no effort slew bound.
    gripper_velocity_ok = jnp.ones(knots.shape[0], dtype=bool)
    timing_value = jnp.asarray(controller_timing_compatible, dtype=bool)
    controller_timing_ok = (~enabled) | jnp.broadcast_to(
        timing_value,
        (knots.shape[0],),
    )
    return (
        controller_timing_ok,
        joint_position_ok,
        joint_velocity_ok,
        gripper_width_ok,
        gripper_velocity_ok,
    )

def stable_device_selection(cost: Any, valid: Any) -> tuple[Any, Any, Any, Any]:
    """Lexicographically select on device: valid, then cost, earliest tie.

    Returns ``(index, n_valid, finite_mean_cost, flat_objective)`` as device
    scalars. If every candidate is invalid, the minimum finite cost is retained
    only for diagnostics; the caller still fails closed from ``n_valid == 0``.
    """
    import jax.numpy as jnp

    costs = jnp.asarray(cost)
    validity = jnp.asarray(valid, dtype=bool)
    if costs.ndim != 1 or validity.shape != costs.shape:
        raise ValueError(
            "cost and valid must be same-shape vectors, got "
            f"{costs.shape} and {validity.shape}"
        )
    finite = jnp.isfinite(costs)
    executable = validity & finite
    n_valid = jnp.sum(executable, dtype=jnp.int32)
    ranked_cost = jnp.where(
        n_valid > 0,
        jnp.where(executable, costs, jnp.inf),
        jnp.where(finite, costs, jnp.inf),
    )
    # JAX argmin returns the first occurrence, giving a stable earliest-tie
    # choice without transferring the vector to Python.
    selected = jnp.argmin(ranked_cost)

    finite_count = jnp.sum(finite, dtype=jnp.int32)
    mean_cost = jnp.where(
        finite_count > 0,
        jnp.sum(jnp.where(finite, costs, 0.0)) / finite_count,
        jnp.inf,
    )
    objective_mask = jnp.where(n_valid >= 2, executable, finite)
    objective_count = jnp.sum(objective_mask, dtype=jnp.int32)
    objective_min = jnp.min(jnp.where(objective_mask, costs, jnp.inf))
    objective_max = jnp.max(jnp.where(objective_mask, costs, -jnp.inf))
    objective_scale = jnp.maximum(
        1.0,
        jnp.max(jnp.where(objective_mask, jnp.abs(costs), 0.0)),
    )
    flat = (
        (objective_count > 1)
        & ((objective_max - objective_min) <= 1e-9 * objective_scale)
    )
    return selected, n_valid, mean_cost, flat


def terminal_first_joint_hit_samples(
    terminal_residual_trajectories: Any,
    *,
    atol: float = 1e-12,
) -> Any:
    """Return each candidate's first joint terminal hit, or the ``T`` sentinel.

    The input is raw device residuals with shape ``(n_terminal, N, T)``.
    ``T`` is the bounded pose-sample grid supplied by the rollout, not the
    native physics horizon.  This minimum-time ordering is a task-specific
    engineering inference for release, not a claim of theoretical equivalence
    to the ordinary soft-cost objective.
    """
    import jax.numpy as jnp

    residuals = jnp.asarray(terminal_residual_trajectories)
    if residuals.ndim != 3 or residuals.shape[0] < 1 or residuals.shape[2] < 1:
        raise ValueError(
            "terminal residual trajectories must have shape "
            f"(n_terminal,N,T), got {residuals.shape}"
        )
    pose_joint = jnp.all(
        jnp.isfinite(residuals) & (residuals <= float(atol)),
        axis=0,
    )
    sample_count = int(residuals.shape[2])
    sample_indices = jnp.arange(sample_count, dtype=jnp.int32)[None, :]
    return jnp.min(
        jnp.where(pose_joint, sample_indices, sample_count),
        axis=1,
    )


def stable_terminal_trajectory_selection(
    cost: Any,
    valid: Any,
    terminal_residual_trajectories: Any,
    *,
    atol: float = 1e-12,
) -> tuple[Any, Any, Any, Any, Any]:
    """Apply endpoint-joint, earliest-hit, cost, stable-index ordering.

    Returns ``(applied_index, endpoint_joint_count, selected_first_hit,
    first_hits)``. With no endpoint-joint candidate, ``applied_index`` is
    exactly :func:`stable_device_selection`'s result. Requiring the terminal
    still to hold at H prevents a transient pose-sample hit from masquerading
    as a persistent terminal trajectory.
    """
    import jax.numpy as jnp

    costs = jnp.asarray(cost)
    validity = jnp.asarray(valid, dtype=bool)
    if costs.ndim != 1 or validity.shape != costs.shape:
        raise ValueError(
            "cost and valid must be same-shape vectors, got "
            f"{costs.shape} and {validity.shape}"
        )
    residuals = jnp.asarray(terminal_residual_trajectories)
    first_hits = terminal_first_joint_hit_samples(residuals, atol=atol)
    if first_hits.shape != costs.shape:
        raise ValueError(
            "terminal trajectory candidate count must match cost, got "
            f"{first_hits.shape} and {costs.shape}"
        )
    sample_count = int(residuals.shape[2])
    endpoint_joint = jnp.all(
        jnp.isfinite(residuals[:, :, -1])
        & (residuals[:, :, -1] <= float(atol)),
        axis=0,
    )
    executable = validity & jnp.isfinite(costs) & endpoint_joint
    endpoint_joint_count = jnp.sum(executable, dtype=jnp.int32)
    best_hit = jnp.min(jnp.where(executable, first_hits, sample_count))
    earliest = executable & (first_hits == best_hit)
    # JAX argmin returns the first equal-cost row, making the last key the
    # original stable population index without a floating-point perturbation.
    requested_index = jnp.argmin(jnp.where(earliest, costs, jnp.inf))
    fallback_index, _, _, _ = stable_device_selection(costs, validity)
    applied_index = jnp.where(
        endpoint_joint_count > 0,
        requested_index,
        fallback_index,
    )
    no_hit = jnp.asarray(sample_count, dtype=jnp.int32)
    selected_first_hit = jnp.where(
        endpoint_joint_count > 0,
        first_hits[requested_index],
        no_hit,
    )
    return (
        applied_index,
        endpoint_joint_count,
        selected_first_hit,
        first_hits,
    )


def mppi_weighted_update(
    proposals: Any,
    cost: Any,
    valid: Any,
    *,
    previous_mean: Any,
    temperature: float,
    eta_min: float = DEFAULT_MPPI_ETA_MIN,
    eta_max: float = DEFAULT_MPPI_ETA_MAX,
) -> tuple[Any, Any, Any]:
    """Return the MPPI mean, normalization factor, and next temperature.

    The proposal variance remains fixed.  Invalid rollouts receive zero
    weight; if none are executable the nominal mean is returned unchanged.
    Every row is an action, including row zero; unlike position-control CEM,
    MPPI has no measured-position boundary to freeze.  The temperature rule
    is the reference halton-spline implementation: shrink beta by 0.9 above
    ``eta_max`` and grow it by 1.2 below ``eta_min`` so roughly 5--10 samples
    carry weight.
    """
    import jax.numpy as jnp

    knots = jnp.asarray(proposals)
    costs = jnp.asarray(cost)
    validity = jnp.asarray(valid, dtype=bool)
    prior = jnp.asarray(previous_mean)
    tau = validate_mppi_temperature(temperature)
    eta_lo = float(eta_min)
    eta_hi = float(eta_max)
    if not np.isfinite(eta_lo) or not np.isfinite(eta_hi):
        raise ValueError("MPPI eta bounds must be finite")
    if not 0.0 < eta_lo < eta_hi:
        raise ValueError("MPPI eta bounds must satisfy 0 < min < max")
    if knots.ndim != 3 or costs.shape != (knots.shape[0],):
        raise ValueError("MPPI proposals and costs have incompatible shapes")
    if validity.shape != costs.shape or prior.shape != knots.shape[1:]:
        raise ValueError("MPPI validity or previous mean has incompatible shape")
    executable = validity & jnp.isfinite(costs)
    count = jnp.sum(executable, dtype=jnp.int32)
    rho = jnp.min(jnp.where(executable, costs, jnp.inf))
    logits = jnp.where(executable, -(costs - rho) / tau, -jnp.inf)
    unnormalized = jnp.where(executable, jnp.exp(logits), 0.0)
    normalizer = jnp.sum(unnormalized)
    weights = jnp.where(
        normalizer > 0.0,
        unnormalized / normalizer,
        jnp.zeros_like(unnormalized),
    )
    updated = jnp.sum(weights[:, None, None] * knots, axis=0)
    updated = jnp.where(count > 0, updated, prior)
    next_temperature = jnp.where(
        count == 0,
        tau,
        jnp.where(
            normalizer > eta_hi,
            0.9 * tau,
            jnp.where(normalizer < eta_lo, 1.2 * tau, tau),
        ),
    )
    return updated, normalizer, next_temperature


def mppi_weighted_mean(
    proposals: Any,
    cost: Any,
    valid: Any,
    *,
    previous_mean: Any,
    temperature: float,
) -> Any:
    """Compatibility wrapper returning only the MPPI weighted mean."""
    mean, _, _ = mppi_weighted_update(
        proposals,
        cost,
        valid,
        previous_mean=previous_mean,
        temperature=temperature,
    )
    return mean


def dial_annealed_noise_scale(
    *,
    rounds: int,
    num_actions: int,
    trajectory_std_factor: float,
    horizon_std_factor: float,
    array_module: Any,
) -> Any:
    """Return official-code-inspired per-round/per-knot std multipliers.

    DIAL commit 871c84f applies ``traj_factor**round_index`` and
    ``horizon_factor**reversed_node_index`` directly to normal noise. They are
    therefore standard-deviation multipliers; covariance uses their square.
    """
    if isinstance(rounds, bool) or int(rounds) != rounds or int(rounds) < 1:
        raise ValueError("DIAL rounds must be a positive integer")
    if (
        isinstance(num_actions, bool)
        or int(num_actions) != num_actions
        or int(num_actions) < 2
    ):
        raise ValueError("DIAL num_actions must be an integer >= 2")
    trajectory = float(trajectory_std_factor)
    horizon = float(horizon_std_factor)
    if not np.isfinite(trajectory) or not 0.0 < trajectory <= 1.0:
        raise ValueError("DIAL trajectory std factor must be in (0, 1]")
    if not np.isfinite(horizon) or not 0.0 < horizon <= 1.0:
        raise ValueError("DIAL horizon std factor must be in (0, 1]")
    xp = array_module
    round_scale = trajectory ** xp.arange(int(rounds))
    horizon_scale = horizon ** xp.arange(int(num_actions))[::-1]
    return round_scale[:, None] * horizon_scale[None, :]


def dial_annealed_proposals(
    mean: Any,
    ctrl_low: Any,
    ctrl_high: Any,
    *,
    normalized_half_span: Any,
    normalized_noise_scale: Any,
    sample_count: int,
    key: Any,
    return_clipping_count: bool = False,
) -> Any:
    """Return Q-1 bounded normalized 8D samples plus the exact round mean."""
    import jax
    import jax.numpy as jnp

    if (
        isinstance(sample_count, bool)
        or int(sample_count) != sample_count
        or int(sample_count) < 2
    ):
        raise ValueError("DIAL sample_count must be an integer >= 2")
    nominal = jnp.asarray(mean)
    low = jnp.asarray(ctrl_low)
    high = jnp.asarray(ctrl_high)
    half_span = jnp.asarray(normalized_half_span)
    schedule = jnp.asarray(normalized_noise_scale)
    if nominal.ndim != 2 or nominal.shape[-1] != 8:
        raise ValueError("DIAL mean must have shape (K, 8)")
    if low.shape != (8,) or high.shape != (8,) or half_span.shape != (8,):
        raise ValueError("DIAL action bounds and half span must have shape (8,)")
    if schedule.shape != (nominal.shape[0],):
        raise ValueError("DIAL noise schedule must have shape (K,)")
    eps = jax.random.normal(
        key,
        (int(sample_count) - 1, *nominal.shape),
        dtype=nominal.dtype,
    )
    proposals = nominal[None, :, :] + (
        eps * schedule[None, :, None] * half_span[None, None, :]
    )
    # Match the cited code's already-committed first action and exact nominal
    # population member, then apply the registered optimizer-space bounds.
    proposals = proposals.at[:, 0, :].set(nominal[0, :])
    proposals = jnp.concatenate((proposals, nominal[None, :, :]), axis=0)
    below = proposals < low[None, None, :]
    above = proposals > high[None, None, :]
    clipped = jnp.clip(proposals, low[None, None, :], high[None, None, :])
    if return_clipping_count:
        return clipped, jnp.sum(below | above, dtype=jnp.int32)
    return clipped


def dial_fixed_temperature_update(
    proposals: Any,
    cost: Any,
    valid: Any,
    *,
    previous_mean: Any,
    temperature: float,
    enforce_host_contract: bool = True,
) -> tuple[Any, Any, Any]:
    """All-base-valid finite MPPI update with fixed per-round temperature."""
    import jax.numpy as jnp

    knots = jnp.asarray(proposals)
    costs = jnp.asarray(cost)
    validity = jnp.asarray(valid, dtype=bool)
    prior = jnp.asarray(previous_mean)
    tau = validate_mppi_temperature(temperature)
    if knots.ndim != 3 or costs.shape != (knots.shape[0],):
        raise ValueError("DIAL proposals and costs have incompatible shapes")
    if validity.shape != costs.shape or prior.shape != knots.shape[1:]:
        raise ValueError("DIAL validity or previous mean has incompatible shape")
    host_costs = host_validity = None
    if enforce_host_contract:
        try:
            host_costs = np.asarray(costs)
            host_validity = np.asarray(validity, dtype=bool)
        except Exception:
            # JAX tracer: production's base-valid/raw-finite integrity gate
            # owns this assertion before the pure device update.
            pass
    if host_costs is not None and host_validity is not None:
        if not np.any(host_validity):
            raise ValueError("DIAL round has no base-valid candidates")
        if not np.all(np.isfinite(host_costs[host_validity])):
            raise ValueError("DIAL base-valid costs must all be finite")
    executable = validity & jnp.isfinite(costs)
    rho = jnp.min(jnp.where(executable, costs, jnp.inf))
    logits = jnp.where(executable, -(costs - rho) / tau, -jnp.inf)
    unnormalized = jnp.where(executable, jnp.exp(logits), 0.0)
    normalizer = jnp.sum(unnormalized)
    weights = jnp.where(
        normalizer > 0.0,
        unnormalized / normalizer,
        jnp.zeros_like(unnormalized),
    )
    updated = jnp.sum(weights[:, None, None] * knots, axis=0)
    updated = jnp.where(normalizer > 0.0, updated, prior)
    ess = jnp.where(
        normalizer > 0.0,
        1.0 / jnp.sum(jnp.square(weights)),
        0.0,
    )
    return updated, ess, jnp.asarray(tau, dtype=costs.dtype)


def mtp_elite_weighted_mean(
    proposals: Any,
    cost: Any,
    valid: Any,
    *,
    previous_mean: Any,
    temperature: float,
    elite_count: int = 20,
) -> Any:
    """Fit the next local mean from the best valid MTP elites on device."""
    import jax
    import jax.numpy as jnp

    knots = jnp.asarray(proposals)
    costs = jnp.asarray(cost)
    validity = jnp.asarray(valid, dtype=bool)
    prior = jnp.asarray(previous_mean)
    tau = validate_mppi_temperature(temperature)
    elites = int(elite_count)
    if knots.ndim != 3 or costs.shape != (knots.shape[0],):
        raise ValueError("MTP proposals and costs have incompatible shapes")
    if validity.shape != costs.shape or prior.shape != knots.shape[1:]:
        raise ValueError("MTP validity or previous mean has incompatible shape")
    if not 1 <= elites <= knots.shape[0]:
        raise ValueError("MTP elite_count must be within the population")
    executable = validity & jnp.isfinite(costs)
    executable_count = jnp.sum(executable, dtype=jnp.int32)
    ranked = jnp.where(executable, costs, jnp.inf)
    _, elite_indices = jax.lax.top_k(-ranked, elites)
    elite_costs = costs[elite_indices]
    elite_valid = executable[elite_indices]
    baseline = jnp.min(jnp.where(executable, costs, jnp.inf))
    logits = jnp.where(
        elite_valid,
        -(elite_costs - baseline) / tau,
        -jnp.inf,
    )
    weights = jnp.nan_to_num(jax.nn.softmax(logits, axis=0))
    updated = jnp.sum(weights[:, None, None] * knots[elite_indices], axis=0)
    return jnp.where(executable_count > 0, updated, prior)


def binary_gripper_width(
    latent: Any,
    *,
    closed_width_m: float = 0.0,
    where: Any = np.where,
) -> Any:
    """Map an interpolated jaw latent to fixed widths per control step.

    Sampling-MPC prior art (Sumo, arXiv 2604.08508; robosuite's sign
    threshold) drives a jaw as one sampled scalar binarized per control step
    to fixed open/close positions: the close TIMING stays searchable (the
    spline's zero crossing decides WHEN), while no intermediate wedge-band
    width is ever commanded.  Positive means closed and non-positive means
    open. ``closed_width_m`` defaults to the standard close-on-object target;
    the servo saturates on the object, so physical width and held evidence
    remain measured simulation outcomes.
    ``where`` admits ``jnp.where`` so the device conversion applies the
    byte-identical rule to the same spline values.
    """
    closed = float(closed_width_m)
    if not 0.0 <= closed < GN01_OPEN_WIDTH_M:
        raise ValueError(
            "closed_width_m must lie inside [0, GN01_OPEN_WIDTH_M), got "
            f"{closed_width_m!r}"
        )
    return where(latent > 0.0, closed, GN01_OPEN_WIDTH_M)


# Compatibility spelling retained for external analysis scripts.  Its input is
# now a dimensionless latent, never a physical width.
binarized_width_ctrl = binary_gripper_width


def width_command_ctrl(
    latent: Any,
    *,
    binarize_closed_width_m: float = 0.0,
    where: Any = np.where,
) -> Any:
    """Legacy CEM helper mapping jaw latents to binary width commands.

    It remains for old CEM artifacts and the simulator-only slide gripper.
    Production MPPI and real GN01 execution use
    :func:`mppi_continuous_effort_ctrl` instead. Program predicates may assign
    a finite cost to a jaw command, but they never clamp or route the action.
    """
    return binary_gripper_width(
        latent,
        closed_width_m=binarize_closed_width_m,
        where=where,
    )


GN01_DIRECT_FORCE_LIMIT_N = 80.0
# Rizon 4s actuator envelope used by the calibrated planning model. Real
# dispatch additionally binds to the connected RobotInfo.tau_max and may only
# tighten this vector through the active safety profile.
RIZON4S_TORQUE_MAX_NM = np.asarray(
    [123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0], dtype=float
)
RIZON4S_MPC_TORQUE_SCALE = 0.30
RIZON4S_MPC_TORQUE_LIMIT_NM = (
    RIZON4S_TORQUE_MAX_NM * RIZON4S_MPC_TORQUE_SCALE
)


def mppi_continuous_width_ctrl(
    latent: Any,
    *,
    clip: Any = np.clip,
) -> Any:
    """Map the legacy MPPI jaw latent continuously to GN01 width.

    This is the exact ab10 convention: ``-1`` is the 85 mm open target and
    ``+1`` is the 0 mm closed target.  It is reachable only through the
    experiment-only legacy control profile.
    """
    bounded = clip(latent, -1.0, 1.0)
    return 0.5 * (1.0 - bounded) * GN01_OPEN_WIDTH_M


_V11F_JAW_CLOSE_CAP_N = float(__import__("os").environ.get("OAP_JAW_CLOSE_FORCE_N", "80") or 80.0)


def mppi_continuous_effort_ctrl(
    latent: Any,
    *,
    force_limit_n: float = GN01_DIRECT_FORCE_LIMIT_N,
    clip: Any = np.clip,
    where: Any = np.where,
    open_force_limit_n: float | None = None,
) -> Any:
    """Map the paper-MPPI jaw coordinate to signed GN01 force in Newtons.

    Positive closes and negative opens, matching Flexiv RDK
    ``Gripper.Grasp(force)``.  This action contains no desired aperture;
    physical width is produced by contact dynamics and remains measured state.
    """
    limit = float(force_limit_n)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("force_limit_n must be finite and positive")
    bounded = clip(latent, -1.0, 1.0)
    # V11F: env-gated CLOSING force cap at the single latent->Newton decode
    # (physics, events, and validators stay consistent; opening untouched).
    # Pure clip arithmetic: callers of the symmetric path may not supply a
    # tracer-safe `where`, so no branch may use it.
    close_scale = min(limit, _V11F_JAW_CLOSE_CAP_N)
    neg = clip(bounded, -1.0, 0.0)
    pos = clip(bounded, 0.0, 1.0)
    if open_force_limit_n is None:
        return neg * limit + pos * close_scale
    open_limit = float(open_force_limit_n)
    if not np.isfinite(open_limit) or open_limit <= 0.0 or open_limit > limit:
        raise ValueError(
            "open_force_limit_n must be finite and inside (0, force_limit_n]"
        )
    return neg * open_limit + pos * close_scale


def mppi_gripper_command_channels(
    latent: Any,
    *,
    control_profile: str,
    clip: Any = np.clip,
    where: Any = np.where,
    pick_v8_causal_profile: str | None = None,
    unified_mppi_effort_profile: str | None = None,
) -> tuple[Any, Any]:
    """Return physical actuator control and task-semantic command latent.

    ``GripperCommand`` has one common task meaning in both experiment arms:
    open is ``-1`` and closed is ``+1``.  Only the legacy actuator adapter
    decodes that latent to physical GN01 width.  Keeping both values explicit
    prevents a metre-valued width from entering the signed-latent objective.
    """
    profile = validate_control_profile(control_profile)
    command_latent = clip(latent, -1.0, 1.0)
    if control_profile_uses_width_jaw(profile):
        actuator_control = mppi_continuous_width_ctrl(command_latent, clip=clip)
    else:
        open_force_limit_n = None
        if pick_v8_causal_profile is not None:
            from oap.twin.pick_v8_causal import (
                PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N,
                pick_v8_causal_uses_asymmetric_effort,
            )

            if pick_v8_causal_uses_asymmetric_effort(pick_v8_causal_profile):
                open_force_limit_n = (
                    PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N
                )
        if unified_mppi_effort_profile is not None:
            from oap.twin.unified_mppi_effort import (
                unified_mppi_effort_uses_proven_pick_carrier,
            )

            if unified_mppi_effort_uses_proven_pick_carrier(
                unified_mppi_effort_profile
            ):
                open_force_limit_n = 1.0
        actuator_control = mppi_continuous_effort_ctrl(
            command_latent,
            clip=clip,
            where=where,
            open_force_limit_n=open_force_limit_n,
        )
    return actuator_control, command_latent


def mppi_arm_actuator_control(
    action: Any,
    *,
    control_profile: str,
    torque_scale_nm: Any,
) -> Any:
    """Decode the seven arm action columns to physical actuator control."""
    profile = validate_control_profile(control_profile)
    if control_profile_uses_joint_velocity_arm(profile):
        return action
    return action * torque_scale_nm


def mppi_prefix_carrier_ctrl(
    prefix_qpos: Any,
    arm_qpos_addresses: Any,
    gripper_actuator_command: Any,
    *,
    concatenate: Any = np.concatenate,
) -> Any:
    """Build the host orchestration carrier in profile-native command units.

    The arm columns are the exact realized endpoint positions.  The jaw
    column is deliberately *not* measured aperture: it is the native command
    acknowledged at the same endpoint (legacy width metres or direct effort
    Newtons).  ``world.data.ctrl`` is only an orchestration carrier here; it
    is never advanced as a second physics plant.
    """
    return concatenate(
        (
            prefix_qpos[arm_qpos_addresses],
            gripper_actuator_command[None],
        ),
        axis=0,
    )


def mppi_realized_control_knots(
    action_knots: Any,
    arm_action_endpoint_qpos: Any,
    *,
    concatenate: Any = np.concatenate,
) -> Any:
    """Return realized ControlKnots without reinterpreting MPPI actions.

    Both experiment profiles optimize in action space (legacy velocity or
    direct normalized effort).  The ordinary ``ControlKnots`` result carrier
    must instead contain the arm positions actually reached at every action
    endpoint plus the common semantic jaw latent.  Refit/selected proposals
    remain action-space arrays and are shifted separately by the MPC loop.
    """
    return concatenate(
        (
            arm_action_endpoint_qpos,
            action_knots[..., NJ: NJ + 1],
        ),
        axis=-1,
    )


def _linear_knots_to_ctrl_array(
    knots: Any,
    n_steps: int,
    *,
    array_module: Any,
) -> Any:
    """Linearly sample batched knots using the caller's array namespace."""
    count = int(knots.shape[1])
    times = array_module.linspace(0.0, float(count - 1), int(n_steps))
    lower = array_module.floor(times).astype(array_module.int32)
    upper = array_module.minimum(lower + 1, count - 1)
    alpha = (times - lower.astype(times.dtype))[None, :, None]
    return (
        knots[:, lower, :] * (1.0 - alpha)
        + knots[:, upper, :] * alpha
    )


def _linear_knots_to_ctrl_device(knots: Any, n_steps: int) -> Any:
    """Linearly sample ``(N,K,8)`` knots without leaving the accelerator."""
    import jax.numpy as jnp

    return _linear_knots_to_ctrl_array(
        knots,
        n_steps,
        array_module=jnp,
    )


def _mppi_zoh_action_indices(
    horizon_steps: int,
    num_knots: int,
    *,
    array_module: Any,
) -> Any:
    """Return the exact production ZOH knot selected at every physics step."""
    horizon = int(horizon_steps)
    knots = int(num_knots)
    if horizon < 2 or knots < 2:
        raise ValueError("MPPI ZOH expansion requires H>=2 and K>=2")
    return array_module.minimum(
        (array_module.arange(horizon) * knots) // horizon,
        knots - 1,
    ).astype(array_module.int32)


def _validate_paper_mppi_expansion_contract(
    *,
    spline_order: str,
    pick_v8_causal_profile: str | None,
    unified_mppi_effort_profile: str | None = None,
    control_profile: str,
) -> str | None:
    """Select the one allowed interpolation contract for paper MPPI."""
    from oap.twin.pick_v8_causal import (
        pick_v8_causal_uses_width_jaw,
        validate_pick_v8_causal_profile,
    )

    causal_profile = validate_pick_v8_causal_profile(
        pick_v8_causal_profile,
        allow_none=True,
    )
    from oap.twin.unified_mppi_effort import (
        unified_mppi_effort_uses_proven_pick_carrier,
        validate_unified_mppi_effort_profile_arg,
    )

    unified_profile = validate_unified_mppi_effort_profile_arg(
        unified_mppi_effort_profile,
        allow_none=True,
    )
    if causal_profile is not None and unified_profile is not None:
        raise ValueError("Pick-v8 causal and unified profiles conflict")
    validated_control = validate_control_profile(control_profile)
    proven_unified = (
        unified_profile is not None
        and unified_mppi_effort_uses_proven_pick_carrier(unified_profile)
    )
    if causal_profile is None and not proven_unified:
        if spline_order != "constant":
            raise ValueError(
                "paper MPPI requires constant actions outside a sealed "
                "Pick-v8 causal profile"
            )
        return unified_profile
    expected_control = (
        CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
        if causal_profile is not None
        and pick_v8_causal_uses_width_jaw(causal_profile)
        else CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT
    )
    if validated_control != expected_control:
        raise ValueError(
            "Pick-v8 causal MPPI control profile does not match its sealed "
            "single-variable contract"
        )
    if spline_order != "linear":
        raise ValueError(
            "Pick-v8 causal MPPI requires legacy degree-one linear actions"
        )
    return causal_profile or unified_profile


def _paper_mppi_knots_to_step_controls(
    knots: Any,
    horizon_steps: int,
    *,
    spline_order: str,
    pick_v8_causal_profile: str | None,
    unified_mppi_effort_profile: str | None = None,
    control_profile: str,
    array_module: Any,
) -> Any:
    """Expand MPPI knots according to the sealed production profile."""
    causal_profile = _validate_paper_mppi_expansion_contract(
        spline_order=spline_order,
        pick_v8_causal_profile=pick_v8_causal_profile,
        unified_mppi_effort_profile=unified_mppi_effort_profile,
        control_profile=control_profile,
    )
    if causal_profile is not None:
        return _linear_knots_to_ctrl_array(
            knots,
            horizon_steps,
            array_module=array_module,
        )
    action_index = _mppi_zoh_action_indices(
        horizon_steps,
        knots.shape[1],
        array_module=array_module,
    )
    return knots[:, action_index, :]


def validate_horizon_steps(value: Any) -> int:
    """Return one valid execution-grid horizon length.

    The horizon is a generic MPC discretization parameter.  It is expressed in
    execution-physics steps so changing the planning-grid timestep preserves
    the same physical duration.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"horizon_steps must be an integer >= 2, got {value!r}"
        )
    try:
        steps = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"horizon_steps must be an integer >= 2, got {value!r}"
        ) from exc
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(
            f"horizon_steps must be an integer >= 2, got {value!r}"
        )
    if steps < 2:
        raise ValueError(
            f"horizon_steps must be an integer >= 2, got {value!r}"
        )
    return steps


def validate_local_sigma_fraction(value: Any) -> float:
    """Return a finite local-noise scale in actuator-half-range units."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "local_sigma_fraction must be finite and in (0, 1], "
            f"got {value!r}"
        )
    try:
        sigma_fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "local_sigma_fraction must be finite and in (0, 1], "
            f"got {value!r}"
        ) from exc
    if not np.isfinite(sigma_fraction) or not 0.0 < sigma_fraction <= 1.0:
        raise ValueError(
            "local_sigma_fraction must be finite and in (0, 1], "
            f"got {value!r}"
        )
    return sigma_fraction


def local_sigma_fraction_from_env() -> float:
    """Resolve the single proposal scale from its production override."""
    default = str(PREDICTIVE_SIGMA_FRACTION)
    raw = (os.environ.get(LOCAL_SIGMA_FRACTION_ENV) or default).strip() or default
    try:
        return validate_local_sigma_fraction(raw)
    except ValueError as exc:
        raise ValueError(f"{LOCAL_SIGMA_FRACTION_ENV}: {exc}") from exc


def cem_rounds_from_env() -> int:
    """Resolve the production CEM round count."""
    default = str(DEFAULT_CEM_ROUNDS)
    raw = (os.environ.get(CEM_ROUNDS_ENV) or default).strip() or default
    try:
        return validate_cem_rounds(raw)
    except ValueError as exc:
        raise ValueError(f"{CEM_ROUNDS_ENV}: {exc}") from exc


def cem_elite_fraction_from_env() -> float:
    """Resolve the production CEM elite fraction."""
    default = str(DEFAULT_CEM_ELITE_FRACTION)
    raw = (
        os.environ.get(CEM_ELITE_FRACTION_ENV) or default
    ).strip() or default
    try:
        return validate_cem_elite_fraction(raw)
    except ValueError as exc:
        raise ValueError(f"{CEM_ELITE_FRACTION_ENV}: {exc}") from exc


def cem_min_std_fraction_from_env() -> float:
    """Resolve the task-independent CEM variance floor."""
    default = str(DEFAULT_CEM_MIN_STD_FRACTION)
    raw = (
        os.environ.get(CEM_MIN_STD_FRACTION_ENV) or default
    ).strip() or default
    try:
        return validate_local_sigma_fraction(raw)
    except ValueError as exc:
        raise ValueError(f"{CEM_MIN_STD_FRACTION_ENV}: {exc}") from exc


def horizon_steps_from_env() -> int:
    """Resolve the production horizon from ``OAP_HORIZON_STEPS``."""
    raw = os.environ.get(HORIZON_STEPS_ENV, str(DEFAULT_EXECUTION_STEPS))
    try:
        return validate_horizon_steps(raw)
    except ValueError as exc:
        raise ValueError(f"{HORIZON_STEPS_ENV}: {exc}") from exc


def _shared_gripper_qpos(world_model: mujoco.MjModel,
                         roll_model: mujoco.MjModel) -> tuple[tuple[int, int], ...]:
    """(world_qadr, roll_qadr) for every gn01 joint BOTH models define.

    A continued rollout has to install the gripper's real configuration, and for
    the gn01 that is seven scalars -- the width joint plus the six knuckle
    joints its <equality><joint> mimics drive. Matching by NAME rather than by
    address is what keeps the slide-gripper variant safe: its pad slides have no
    counterpart in the world model, so it yields an empty tuple and nothing
    changes for that backend.
    """
    pairs: list[tuple[int, int]] = []
    for j in range(roll_model.njnt):
        name = mujoco.mj_id2name(roll_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith("gn01_"):
            continue
        jw = mujoco.mj_name2id(world_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jw < 0:
            continue
        # scalar joints only; a gn01 hinge/slide is 1 qpos wide
        if int(roll_model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                               int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        pairs.append((int(world_model.jnt_qposadr[jw]),
                      int(roll_model.jnt_qposadr[j])))
    return tuple(pairs)


def _shared_dof_map(world_model: mujoco.MjModel,
                    roll_model: mujoco.MjModel) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(world_dof_indices, roll_dof_indices) for every joint BOTH models define.

    Velocity, unlike position, needs no per-category handling: one name-matched
    map covers the arm, the subject's freejoint, every other mover and the
    gripper at once, and a joint the rollout model does not share -- the slide
    variant's prismatic pads -- simply does not appear, so that backend is
    untouched. Widths come from the joint type (free 6, ball 3, hinge/slide 1),
    never assumed.
    """
    _ND = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3,
           int(mujoco.mjtJoint.mjJNT_HINGE): 1, int(mujoco.mjtJoint.mjJNT_SLIDE): 1}
    wsrc: list[int] = []
    rdst: list[int] = []
    for j in range(roll_model.njnt):
        name = mujoco.mj_id2name(roll_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name:
            continue
        jw = mujoco.mj_name2id(world_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jw < 0:
            continue
        t = int(roll_model.jnt_type[j])
        if t != int(world_model.jnt_type[jw]):
            continue                      # same name, different joint: skip
        n = _ND.get(t)
        if n is None:
            continue
        wa, ra = int(world_model.jnt_dofadr[jw]), int(roll_model.jnt_dofadr[j])
        wsrc.extend(range(wa, wa + n))
        rdst.extend(range(ra, ra + n))
    return tuple(wsrc), tuple(rdst)


def _shared_qpos_map(
    world_model: mujoco.MjModel,
    roll_model: mujoco.MjModel,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(world qpos indices, rollout qpos indices) for shared named joints.

    The selected GPU prefix becomes the simulated plant state for the next MPC
    cycle.  Mapping by joint name and type keeps that state transfer exact even
    when a rollout model has a different address layout; fixed/free/ball/scalar
    widths come from MuJoCo's joint type rather than from task-specific names.
    """
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    }
    world_indices: list[int] = []
    rollout_indices: list[int] = []
    for rollout_jid in range(roll_model.njnt):
        name = (
            mujoco.mj_id2name(
                roll_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                rollout_jid,
            )
            or ""
        )
        if not name:
            continue
        world_jid = mujoco.mj_name2id(
            world_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if world_jid < 0:
            continue
        joint_type = int(roll_model.jnt_type[rollout_jid])
        if joint_type != int(world_model.jnt_type[world_jid]):
            continue
        width = widths.get(joint_type)
        if width is None:
            continue
        world_address = int(world_model.jnt_qposadr[world_jid])
        rollout_address = int(roll_model.jnt_qposadr[rollout_jid])
        world_indices.extend(range(world_address, world_address + width))
        rollout_indices.extend(range(rollout_address, rollout_address + width))
    return tuple(world_indices), tuple(rollout_indices)


def _shared_ctrl_map(
    world_model: mujoco.MjModel,
    roll_model: mujoco.MjModel,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(world ctrl indices, rollout ctrl indices) for named actuators."""
    world_indices: list[int] = []
    rollout_indices: list[int] = []
    for rollout_aid in range(roll_model.nu):
        name = (
            mujoco.mj_id2name(
                roll_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                rollout_aid,
            )
            or ""
        )
        if not name:
            continue
        world_aid = mujoco.mj_name2id(
            world_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if world_aid < 0:
            continue
        world_indices.append(int(world_aid))
        rollout_indices.append(int(rollout_aid))
    return tuple(world_indices), tuple(rollout_indices)


def _plan_xml_text_and_sha256(plan_xml: Any) -> tuple[str, str]:
    """Read once and return exact XML text plus its byte-level identity."""
    value = str(plan_xml)
    payload = (
        value.encode("utf-8")
        if "<mujoco" in value
        else Path(value).read_bytes()
    )
    return payload.decode("utf-8"), hashlib.sha256(payload).hexdigest()


def _plan_xml_text(plan_xml: Any) -> str:
    """Backward-compatible text-only view for non-cache callers."""
    return _plan_xml_text_and_sha256(plan_xml)[0]


def _movable_and_robot_geoms(model: mujoco.MjModel) -> tuple[set[int], set[int]]:
    """(geoms of every free body except the subject, geoms of the robot).

    This structural split drives generic contact diagnostics in both lanes.
    It does not grant a world-obstacle exemption: only the current program's
    explicitly selected mediation target receives that semantic treatment.
    """
    mov = set().union(*_movable_geom_groups(model).values())

    def active(g: int) -> bool:
        return bool(
            int(model.geom_contype[g]) or int(model.geom_conaffinity[g])
        )

    rob = {
        g
        for g in range(model.ngeom)
        if (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        ).startswith("gn01_")
        and active(g)
    }
    return mov, rob


def robot_subject_contact_geom_roles(
    model: mujoco.MjModel,
    *,
    exact_arm_gids: tuple[int, ...] | None = None,
    require_complete: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return exact active geom roles for robot-to-controlled-subject touch.

    The controlled subject is the complete active-geom subtree owned by
    ``pick_object_freejoint``.  The robot is the union of active GN01 geoms and
    the same structurally derived convex-arm mesh set used by the GPU safety
    reducer.  The two roles are fixed by the model; a TaskProgram cannot
    substitute arbitrary actors.
    """
    subject_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "pick_object_freejoint",
    )
    if subject_joint < 0:
        if not require_complete:
            return (), ()
        raise ValueError(
            "robot-subject contact requires pick_object_freejoint"
        )
    subject_root = int(model.jnt_bodyid[subject_joint])

    def active(geom: int) -> bool:
        return bool(
            int(model.geom_contype[geom])
            or int(model.geom_conaffinity[geom])
        )

    subject: set[int] = set()
    for geom in range(model.ngeom):
        if not active(geom):
            continue
        body = int(model.geom_bodyid[geom])
        while body > 0:
            if body == subject_root:
                subject.add(int(geom))
                break
            body = int(model.body_parentid[body])
    if not subject and require_complete:
        raise ValueError(
            "controlled-subject subtree has no contact-active geoms"
        )

    gn01 = {
        int(geom)
        for geom in range(model.ngeom)
        if active(geom)
        and (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom,
            )
            or ""
        ).startswith("gn01_")
    }
    arm = set(
        exact_arm_collision_geoms(
            model,
            require_complete=require_complete,
        )
        if exact_arm_gids is None
        else (int(geom) for geom in exact_arm_gids)
    )
    robot = gn01 | arm
    overlap = subject & robot
    if overlap:
        raise ValueError(
            "robot and controlled-subject contact geom roles overlap: "
            f"{sorted(overlap)}"
        )
    if not robot and require_complete:
        raise ValueError("robot contact role has no active geoms")
    return tuple(sorted(subject)), tuple(sorted(robot))


def current_robot_subject_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    refresh: bool = True,
    require_complete: bool = True,
) -> bool:
    """Measure exact current robot-subject touch without advancing physics."""
    subject, robot = robot_subject_contact_geom_roles(
        model,
        require_complete=require_complete,
    )
    if refresh:
        mujoco.mj_forward(model, data)
    subject_set = set(subject)
    robot_set = set(robot)
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        if float(contact.dist) > 0.0:
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (
            (geom1 in robot_set and geom2 in subject_set)
            or (geom2 in robot_set and geom1 in subject_set)
        ):
            return True
    return False


def _movable_geom_groups(model: mujoco.MjModel) -> dict[str, set[int]]:
    """Map each non-subject freejoint body to all active geoms in its subtree."""
    roots: dict[int, str] = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if n.endswith("_freejoint") and n != "pick_object_freejoint":
            roots[int(model.jnt_bodyid[j])] = n[: -len("_freejoint")]
    groups = {name: set() for name in roots.values()}
    for geom in range(model.ngeom):
        if not (
            int(model.geom_contype[geom])
            or int(model.geom_conaffinity[geom])
        ):
            continue
        body = int(model.geom_bodyid[geom])
        while body > 0:
            owner = roots.get(body)
            if owner is not None:
                groups[owner].add(geom)
                break
            body = int(model.body_parentid[body])
    return {name: geoms for name, geoms in groups.items() if geoms}


def _contact_evidence_target_layout(
    *,
    subject_geom_id: int,
    movable_geom_groups: dict[str, Any],
    contact_target_bodies: tuple[str, ...] = (),
    mediation_target_bodies: tuple[str, ...] = (),
    controlled_subject_contact_evidence_name: str | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[int, ...]],
]:
    """Resolve cost targets separately from evidence-only subject contact.

    Explicit Contact/ToolMediation vocabulary alone controls the aggregate
    contact channels consumed by cost and validity. Independently, the
    controlled subject may be named as a per-body evidence target so an
    evaluator can qualify direct robot-to-subject contact even when the VLM
    objective intentionally contains no contact term (for example, a push
    objective expressed only by displacement and table support).

    The returned tuples are ``(contact, mediation, evidence, groups)``.  The
    controlled subject is added only to ``evidence`` and is structurally bound
    to the production subject geom, never looked up as a non-subject movable.
    """
    mediation_names = tuple(dict.fromkeys(
        str(name) for name in mediation_target_bodies if name
    ))
    contact_names = tuple(dict.fromkeys(
        (
            *(str(name) for name in contact_target_bodies if name),
            *mediation_names,
        )
    ))
    movable_groups = {
        str(name): tuple(int(geom) for geom in geoms)
        for name, geoms in movable_geom_groups.items()
    }
    unknown_targets = sorted(set(contact_names) - set(movable_groups))
    if unknown_targets:
        raise ValueError(
            "contact targets are not dynamic bodies in the rollout: "
            f"{unknown_targets}; available={sorted(movable_groups)}"
        )

    evidence_groups = {
        name: movable_groups[name] for name in mediation_names
    }
    controlled_name = (
        str(controlled_subject_contact_evidence_name)
        if controlled_subject_contact_evidence_name
        else None
    )
    if controlled_name is not None:
        if controlled_name in evidence_groups:
            raise ValueError(
                "controlled-subject evidence name collides with a distinct "
                f"mediation target: {controlled_name!r}"
            )
        evidence_groups[controlled_name] = (int(subject_geom_id),)
    evidence_names = tuple(evidence_groups)
    return contact_names, mediation_names, evidence_names, evidence_groups


def _terminal_contact_endpoint_targets(
    term_descriptors: tuple[dict[str, Any], ...],
    contact_target_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Bind native endpoint telemetry only for a declared terminal Contact.

    Running-only Contact retains its legacy public row schema. A terminal
    Contact can consume the exact subject-target trajectory only when the
    stage's structural contact binding names exactly one physical body.
    """
    terminal_contacts = tuple(
        descriptor
        for descriptor in term_descriptors
        if descriptor.get("scope") == "terminal"
        and descriptor.get("type") == "contact"
    )
    if not terminal_contacts:
        return ()
    if len(terminal_contacts) != 1:
        raise ValueError(
            "native terminal Contact endpoint requires exactly one terminal "
            "contact descriptor"
        )
    targets = tuple(dict.fromkeys(
        str(name) for name in contact_target_names if name
    ))
    if len(targets) != 1:
        raise ValueError(
            "native terminal Contact endpoint requires exactly one bound "
            f"contact target body, got {list(targets)}"
        )
    return targets


def _terminal_contact_evidence_layout(
    *,
    mediation_target_names: tuple[str, ...],
    evidence_target_names: tuple[str, ...],
    evidence_target_groups: dict[str, tuple[int, ...]],
    movable_geom_groups: dict[str, Any],
    terminal_contact_target_names: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, tuple[int, ...]],
]:
    """Inject only a terminal Contact body into native contact evidence.

    Existing running Contact does not enter ``terminal_contact_target_names``
    and therefore returns the legacy mediation/evidence layout unchanged.  A
    terminal target is added both to the aggregate subject-target mask (which
    feeds the exact sampled trajectory) and the per-body mask (which feeds the
    E-1/H-1 endpoint diagnostics).
    """
    scan_target_names = tuple(dict.fromkeys((
        *(str(name) for name in mediation_target_names if name),
        *(str(name) for name in terminal_contact_target_names if name),
    )))
    if not terminal_contact_target_names:
        return (
            scan_target_names,
            evidence_target_names,
            evidence_target_groups,
        )

    movable_groups = {
        str(name): tuple(int(geom) for geom in geoms)
        for name, geoms in movable_geom_groups.items()
    }
    missing = sorted(set(terminal_contact_target_names) - set(movable_groups))
    if missing:
        raise ValueError(
            "terminal Contact targets are not dynamic bodies in the rollout: "
            f"{missing}; available={sorted(movable_groups)}"
        )
    groups = dict(evidence_target_groups)
    for name in terminal_contact_target_names:
        geoms = movable_groups[name]
        existing = groups.get(name)
        if existing is not None and existing != geoms:
            raise ValueError(
                "terminal Contact evidence target collides with a different "
                f"physical owner: {name!r}"
            )
        groups[name] = geoms
    return scan_target_names, tuple(groups), groups


def _subject_world_target_contact_terms(
    dist: Any,
    geom1: Any,
    geom2: Any,
    slot_ok: Any,
    *,
    subject_geom_id: int,
    support_geom_mask: Any,
    contact_target_geom_mask: Any,
    target_geom_mask: Any,
) -> tuple[Any, Any, Any, Any]:
    """Reduce exact subject contact into obstacle/contact/mediation channels.

    A structurally declared contact target is not a world obstacle for the
    current program stage, even when its collision geom follows the generic
    ``support_*`` naming convention. The narrower mediation mask independently
    controls tool/robot-target evidence for finite compatibility residuals.

    This helper is JAX-only in production but deliberately lives outside
    :meth:`BatchedRollout.run` so the exact device reduction can be regression
    tested without compiling a full MJWarp scene.
    """
    import jax.numpy as jnp

    is_subject = (geom1 == subject_geom_id) | (geom2 == subject_geom_id)
    other = jnp.where(geom1 == subject_geom_id, geom2, geom1)
    penetrating = is_subject & slot_ok & (dist < 0)
    touching = is_subject & slot_ok & (dist <= 0)
    is_contact_target = contact_target_geom_mask[other]
    is_mediation_target = target_geom_mask[other]
    is_world_obstacle = support_geom_mask[other] & ~is_contact_target
    world_pen = jnp.where(
        penetrating & is_world_obstacle,
        -dist,
        0.0,
    ).max()
    contact_target_pen = jnp.where(
        penetrating & is_contact_target,
        -dist,
        0.0,
    ).max()
    target_pen = jnp.where(
        penetrating & is_mediation_target,
        -dist,
        0.0,
    ).max()
    target_touch = jnp.any(touching & is_mediation_target)
    return world_pen, contact_target_pen, target_pen, target_touch


def _initial_scene_collision_geoms(
    model: mujoco.MjModel,
    movable_gids: tuple[int, ...],
) -> tuple[int, ...]:
    """Contact-active task-scene geoms used by the episode-start diagnostic.

    This mirrors :func:`oap.twin.scene.assert_clean_episode_start` for the
    subject and named supports, then structurally adds every other free body's
    active geoms so a movable target whose geom has an arbitrary name cannot
    escape the check.  The table is deliberately excluded: robot/table contact
    can be a permanent property of the mounted model, while this signal asks
    whether every candidate inherited robot contact with a task object.
    """
    scene = set(int(gid) for gid in movable_gids)
    for gid in range(model.ngeom):
        if not (
            int(model.geom_contype[gid])
            or int(model.geom_conaffinity[gid])
        ):
            continue
        name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                gid,
            )
            or ""
        )
        if name.startswith(("pick_object", "support_")):
            scene.add(gid)
    return tuple(sorted(scene))


def exact_arm_collision_geoms(
    model: mujoco.MjModel,
    *,
    require_complete: bool = True,
) -> tuple[int, ...]:
    """Contact-active convex mesh hulls belonging to the articulated arm.

    The arm body set is derived from the seven robot joints and their ancestors,
    rather than from geom names or from "all meshes".  That keeps mesh-based
    task objects out of the robot safety set while including the fixed robot
    base. Every arm joint must exist and every derived arm body must expose at
    least one active mesh; otherwise the GPU model is incomplete and build
    fails closed. ``require_complete=False`` permits a robot-free unit-test
    model and returns an empty set; a partially defined robot still fails.
    """
    arm_bodies: set[int] = set()
    missing_joints: list[str] = []
    for i in range(NJ):
        name = f"joint{i + 1}"
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            missing_joints.append(name)
            continue
        bid = int(model.jnt_bodyid[jid])
        while bid > 0:
            arm_bodies.add(bid)
            bid = int(model.body_parentid[bid])
    if missing_joints:
        if not require_complete and len(missing_joints) == NJ:
            return ()
        raise ValueError(
            "exact GPU rollout is missing arm joints "
            f"{missing_joints}; refusing an incomplete collision model"
        )

    gids = tuple(
        gid
        for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) in arm_bodies
        and int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        and (
            int(model.geom_contype[gid])
            or int(model.geom_conaffinity[gid])
        )
    )
    covered_bodies = {int(model.geom_bodyid[gid]) for gid in gids}
    uncovered = sorted(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid)
        for bid in arm_bodies
        if bid not in covered_bodies
    )
    if uncovered:
        raise ValueError(
            "exact GPU rollout has no contact-active convex mesh for arm "
            f"bodies {uncovered}; refusing an unprotected robot"
        )
    return gids


def _xml_str(plan_xml: Any) -> str:
    """Accept a filesystem path OR an XML string; return the XML text. The plan's
    ``meshdir`` is absolute, so compiling the returned text via ``from_xml_string``
    still resolves the hull assets."""
    from pathlib import Path
    s = str(plan_xml)
    return s if "<mujoco" in s else Path(s).read_text()


# --- The slide-gripper GPU model (Option A) --------------------------------
# SUPERSEDED 2026-07-27; kept as an ablation arm, no longer the screen.
#
# This model existed because the real linkage was believed unable to hold a
# grasp on warp "because warp has no NoSlip". That was wrong twice over. NoSlip
# turns out to be irrelevant even on CPU (a 150 mm carry slips 4.11 mm with it
# and 4.13 mm without), and the linkage's failure on warp was OUR bug: run()
# restored only gn01_finger_width_joint from a continued state, leaving the six
# equality-coupled knuckles at the template default, so every rollout began
# with the hand OPEN. The slide model only escaped it because both of its pad
# joints happen to sit in grip_qpos_addrs.
#
# With that fixed and the width servo at kp=2700, the REAL linkage matches the
# CPU certifier to 0.1 mm mean / 0.2 mm max on a 16-candidate rank-breaking
# carry batch (spread 145 mm), with 16/16 agreement on which candidates drop.
# The slide surrogate on the same batch is 9.4 mm mean -- about 90x worse. Use
# gripper="linkage" for the screen; this stays only to ablate the surrogate.
_GRIP_KP = 900.0
_GRIP_KV = 16.0            # critically damp the kp=900 servo (Franka-MJX reduces kp/kv on GPU)
_GRIP_FORCE_N = 60.0       # matched to the real gn01 width-actuator clamp
_SLIDE_OPEN_M = 0.06       # pad-centre displacement at full open (ctrl semantics)


def width_to_slide_ctrl(width: float) -> float:
    """gn01 width command [0, 0.085] -> pad-slide ctrl [0, 0.06] (SAME for both
    pads; the opposite slide axes make them move symmetrically). Open(0.085)->
    0.06, closed(0)->0 so the object stops the pads and the +-60 N force clamps."""
    return float(min(_SLIDE_OPEN_M, max(0.0, width / 0.085 * _SLIDE_OPEN_M)))


def _measure_gn01_pad_geometry(model: mujoco.MjModel) -> dict[str, Any]:
    """Read the real gn01 finger-tip pad geometry (a fixed gripper feature).

    Closes the linkage gripper and reads each finger-tip pad's pose in the
    flange (``lab_gn01_tool``) frame, so the slide pads are placed EXACTLY where
    the real pads grip (pad centre ~2 cm below the grasp-centre site, tilted
    ~13 deg to seat flat on the object) -- derived from the model, never
    hardcoded per scene.
    """
    d = mujoco.MjData(model)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gn01_finger_width_pos")
    d.ctrl[aid] = 0.0
    for _ in range(300):
        mujoco.mj_step(model, d)
    flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lab_gn01_tool")
    T = np.asarray(d.xpos[flange]); R = np.asarray(d.xmat[flange]).reshape(3, 3)
    out: dict[str, Any] = {}
    for side in ("left", "right"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              f"gn01_{side}_finger_tip_collision")
        p_rel = R.T @ (np.asarray(d.geom_xpos[g]) - T)
        R_rel = R.T @ np.asarray(d.geom_xmat[g]).reshape(3, 3)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R_rel.flatten())
        out[f"{side}_z"] = float(p_rel[2])
        out[f"{side}_quat"] = q
        out["size"] = np.asarray(model.geom_size[g]).copy()
    return out


def plan_dt_from_env() -> float | None:
    """Parse ``OAP_PLAN_DT`` (seconds) or None. Raises a knob-named error
    on a malformed value so a typo'd study flag fails the run instead of
    silently running a different planning grid.
    Range validation lives in :meth:`BatchedRollout.build`, which knows the
    execution grid; callers that catch GPU screen-build errors
    try/except) must call this EAGERLY so a bad value still fails loud.
    """
    env = os.environ.get("OAP_PLAN_DT", "").strip()
    if not env:
        return None
    try:
        return float(env)
    except ValueError as e:
        raise ValueError(f"OAP_PLAN_DT={env!r} is not a float") from e


def _warp_contacts(d):
    """The warp lane's contact arena, read from ``mjx.Data._impl``.

    Private-underscore by upstream DESIGN, not by our shortcut -- the MJX docs
    state verbatim: "Since JAX and Warp diverge in their implementations of
    contact buffers, contacts are located in the private ``mjx.Data._impl``
    instead of ``mjx.Data.contact``." This accessor is the ONE place that
    touches those fields, so an upstream rename (the reviewed stack is
    mujoco/mujoco-warp==3.11.0 + warp-lang==1.15.0) breaks exactly one function.

    Returns ``(dist, geom1, geom2, nacon_scalar, worldid)``.
    """
    import jax.numpy as jnp
    impl = getattr(d, "_impl", d)
    geom = impl.contact__geom
    return (impl.contact__dist, geom[..., 0], geom[..., 1],
            jnp.asarray(impl.nacon).reshape(-1)[0], impl.contact__worldid)


def _reduce_warp_contact_arena(
    dist: Any,
    geom1: Any,
    geom2: Any,
    nacon: Any,
    worldid: Any,
    *,
    num_worlds: int,
    subject_geom_id: int,
    left_pad_geom_id: int,
    right_pad_geom_id: int,
    table_geom_id: int,
    support_geom_mask: Any,
    movable_geom_mask: Any,
    robot_geom_mask: Any,
    exact_arm_geom_mask: Any,
    subject_contact_geom_mask: Any,
    contact_target_geom_mask: Any,
    target_geom_mask: Any,
    target_geom_masks: Any,
    initial_scene_geom_mask: Any,
) -> tuple[Any, ...]:
    """Reduce MJWarp's shared contact arena once into per-world evidence.

    MJWarp 3.10 marks every ``contact__*`` field, ``nacon``, and ``nworld`` as
    ``DATA_NON_VMAP``.  A vmapped dynamics call therefore exposes one global
    arena whose slots are assigned to worlds by ``contact__worldid``.  Comparing
    that global array with a mapped scalar ``world_idx`` materializes an
    ``(num_worlds, arena_size)`` predicate at every physics step.  Since
    ``arena_size`` itself scales with the number of worlds, that is quadratic
    in the rollout batch.

    All existing evidence is a max/any reduction over non-negative slot
    features.  Pack those features and use one scatter-max to reduce the arena
    to ``(num_worlds, feature_count)``.  Under ``jax.vmap`` the inputs to this
    helper remain unbatched by MJWarp's custom rule, so the scatter is emitted
    once; each rollout lane only gathers its resulting row.

    The returned tuple is the historical ``contact_terms`` tuple without the
    global ``nacon`` scalar:

    ``(two_sided, pen_ot, pen_pt, pen_ow, pen_obj_pad, pen_arm,
    pen_subject_movable, pen_robot_movable, pen_subject_target,
    pen_robot_target, pen_subject_contact_target,
    touch_subject_movable, touch_robot_movable,
    touch_subject_target, touch_robot_target, pen_subject_targets,
    pen_robot_targets, touch_subject_targets, touch_robot_targets,
    touch_robot_scene, touch_robot_subject)``.
    """
    import jax.numpy as jnp

    arena_size = int(dist.shape[-1])
    slot = jnp.arange(arena_size, dtype=jnp.int32)
    worldid = jnp.asarray(worldid, dtype=jnp.int32)
    slot_ok = (
        (slot < jnp.asarray(nacon, dtype=jnp.int32))
        & (worldid >= 0)
        & (worldid < int(num_worlds))
    )
    safe_worldid = jnp.where(slot_ok, worldid, 0)

    # Contact geom ids are valid inside ``nacon``.  Slots beyond nacon are
    # unspecified; clip only for safe mask lookup, then zero them via slot_ok.
    ngeom = int(support_geom_mask.shape[0])
    safe_geom1 = jnp.clip(geom1, 0, ngeom - 1)
    safe_geom2 = jnp.clip(geom2, 0, ngeom - 1)
    penetrating = dist < 0
    touching = dist <= 0
    penetration = jnp.where(penetrating, -dist, jnp.zeros_like(dist))

    def pair(a: int, b: int) -> Any:
        return ((geom1 == a) & (geom2 == b)) | (
            (geom1 == b) & (geom2 == a)
        )

    subject1 = geom1 == subject_geom_id
    subject2 = geom2 == subject_geom_id
    is_subject = subject1 | subject2
    other_subject = jnp.where(subject1, safe_geom2, safe_geom1)
    subject_contact_target = contact_target_geom_mask[other_subject]
    subject_target = target_geom_mask[other_subject]
    subject_movable = movable_geom_mask[other_subject]
    subject_world = (
        support_geom_mask[other_subject] & ~subject_contact_target
        & (other_subject != table_geom_id)
    )

    arm1 = exact_arm_geom_mask[safe_geom1]
    arm2 = exact_arm_geom_mask[safe_geom2]
    is_arm = arm1 | arm2
    robot1 = robot_geom_mask[safe_geom1] | arm1
    robot2 = robot_geom_mask[safe_geom2] | arm2
    is_robot = robot1 | robot2
    other_robot = jnp.where(robot1, safe_geom2, safe_geom1)
    robot_target = target_geom_mask[other_robot]
    robot_movable = movable_geom_mask[other_robot]
    robot_scene = initial_scene_geom_mask[other_robot]
    robot_subject = (
        (robot1 & subject_contact_geom_mask[safe_geom2])
        | (robot2 & subject_contact_geom_mask[safe_geom1])
    )

    obj_left = pair(subject_geom_id, left_pad_geom_id)
    obj_right = pair(subject_geom_id, right_pad_geom_id)
    obj_table = pair(subject_geom_id, table_geom_id)
    pad_table = pair(left_pad_geom_id, table_geom_id) | pair(
        right_pad_geom_id,
        table_geom_id,
    )

    scalar_pen_masks = jnp.stack(
        (
            obj_table,
            pad_table,
            is_subject & subject_world,
            obj_left | obj_right,
            is_arm,
            is_subject & subject_movable,
            is_robot & robot_movable,
            is_subject & subject_target,
            is_robot & robot_target,
            is_subject & subject_contact_target,
        ),
        axis=1,
    )
    target_membership_subject = target_geom_masks[:, other_subject].T
    target_membership_robot = target_geom_masks[:, other_robot].T
    target_pen_subject = (
        target_membership_subject & is_subject[:, None]
    )
    target_pen_robot = target_membership_robot & is_robot[:, None]

    scalar_touch_masks = jnp.stack(
        (
            obj_left,
            obj_right,
            is_subject & subject_movable,
            is_robot & robot_movable,
            is_subject & subject_target,
            is_robot & robot_target,
            is_robot & robot_scene,
            robot_subject,
        ),
        axis=1,
    ) & touching[:, None]
    target_touch_subject = target_pen_subject & touching[:, None]
    target_touch_robot = target_pen_robot & touching[:, None]

    # Pack booleans as exact 0/1 in the contact-distance dtype.  This lets one
    # scatter-max produce every max and any result without changing any public
    # result dtype after the boolean slices are converted back below.
    target_count = int(target_geom_masks.shape[0])
    features = jnp.concatenate(
        (
            penetration[:, None] * scalar_pen_masks,
            penetration[:, None] * target_pen_subject,
            penetration[:, None] * target_pen_robot,
            scalar_touch_masks.astype(dist.dtype),
            target_touch_subject.astype(dist.dtype),
            target_touch_robot.astype(dist.dtype),
        ),
        axis=1,
    )
    features = jnp.where(slot_ok[:, None], features, 0)
    reduced = jnp.zeros(
        (int(num_worlds), int(features.shape[1])),
        dtype=dist.dtype,
    ).at[safe_worldid].max(features)

    scalar_pen = reduced[:, :10]
    cursor = 10
    pen_subject_targets = reduced[:, cursor:cursor + target_count]
    cursor += target_count
    pen_robot_targets = reduced[:, cursor:cursor + target_count]
    cursor += target_count
    scalar_touch = reduced[:, cursor:cursor + 8] > 0
    cursor += 8
    touch_subject_targets = (
        reduced[:, cursor:cursor + target_count] > 0
    )
    cursor += target_count
    touch_robot_targets = reduced[:, cursor:cursor + target_count] > 0

    two_sided = scalar_touch[:, 0] & scalar_touch[:, 1]
    return (
        two_sided,
        scalar_pen[:, 0],
        scalar_pen[:, 1],
        scalar_pen[:, 2],
        scalar_pen[:, 3],
        scalar_pen[:, 4],
        scalar_pen[:, 5],
        scalar_pen[:, 6],
        scalar_pen[:, 7],
        scalar_pen[:, 8],
        scalar_pen[:, 9],
        scalar_touch[:, 2],
        scalar_touch[:, 3],
        scalar_touch[:, 4],
        scalar_touch[:, 5],
        pen_subject_targets,
        pen_robot_targets,
        touch_subject_targets,
        touch_robot_targets,
        scalar_touch[:, 6],
        scalar_touch[:, 7],
    )


def _host_pair_touching(contact: Any, geom_a: int, geom_b: int) -> bool:
    """Exact contact predicate shared by both host-side parity lanes.

    MuJoCo may allocate a contact slot while geoms remain separated inside a
    positive contact margin. Only zero or negative distance is physical touch,
    matching the GPU ``ObjectHeld`` evidence.
    """
    return (
        {int(contact.geom1), int(contact.geom2)}
        == {int(geom_a), int(geom_b)}
        and float(contact.dist) <= 0.0
    )


def _required_named_ids(
    names: tuple[str, ...],
    resolver: Any,
    *,
    label: str,
) -> dict[str, int]:
    """Resolve required model ids and fail before negative-id indexing."""
    resolved = {name: int(resolver(name)) for name in names}
    missing = sorted(name for name, item_id in resolved.items() if item_id < 0)
    if missing:
        raise ValueError(
            f"rollout model is missing required {label}: "
            + ", ".join(missing)
        )
    return resolved


def build_slide_gripper_xml(plan_xml: Any) -> str:
    """Transform the plan XML: replace the gn01 4-bar linkage with two force-
    limited prismatic pad slides (Option A). Pad geometry is derived from the
    real gripper (:func:`_measure_gn01_pad_geometry`); warp stabilization uses
    armature, a critically damped servo, an interior control range, and a
    headroom range."""
    import xml.etree.ElementTree as ET

    xml = _xml_str(plan_xml)                            # path OR an already-transformed string
    src = mujoco.MjModel.from_xml_string(xml)
    pad = _measure_gn01_pad_geometry(src)
    z = 0.5 * (pad["left_z"] + pad["right_z"])          # pads grip the object centre
    size = " ".join(f"{v:.5f}" for v in pad["size"])

    r = ET.fromstring(xml)
    for kf in list(r.findall("keyframe")):               # qpos size changes
        r.remove(kf)
    eq = r.find("equality")
    if eq is not None:
        for c in list(eq):
            j1 = c.get("joint1", "") or ""
            if any(k in j1 for k in ("gn01", "finger", "knuckle", "width")):
                eq.remove(c)
    act = r.find("actuator")
    for a in list(act):
        if a.get("joint", "") == "gn01_finger_width_joint":
            act.remove(a)
    for side in ("left", "right"):
        ET.SubElement(act, "position", {
            "name": f"gn01_{side}_pad_pos", "joint": f"gn01_{side}_pad_j",
            "kp": f"{_GRIP_KP:.0f}", "kv": f"{_GRIP_KV:.0f}",
            "ctrlrange": f"0 {_SLIDE_OPEN_M}",           # interior: sampler cannot command impossible positions
            "forcerange": f"-{_GRIP_FORCE_N:.0f} {_GRIP_FORCE_N:.0f}"})
    root = next(b for b in r.iter("body") if b.get("name") == "gn01_articulated_root")
    for child in list(root.findall("body")):
        root.remove(child)
    for side, axis in (("left", "0 1 0"), ("right", "0 -1 0")):
        b = ET.SubElement(root, "body", {"name": f"gn01_{side}_pad_body", "pos": f"0 0 {z:.5f}"})
        ET.SubElement(b, "joint", {                       # headroom range so the operating point is not at a limit
            "name": f"gn01_{side}_pad_j", "type": "slide", "axis": axis,
            "range": f"-0.01 {_SLIDE_OPEN_M + 0.01:.3f}", "damping": "2.0", "armature": "0.02"})
        q = " ".join(f"{v:.6f}" for v in pad[f"{side}_quat"])
        ET.SubElement(b, "geom", {
            "name": f"gn01_{side}_finger_tip_collision", "type": "box", "size": size,
            "quat": q, "mass": "0.05", "friction": "1.0 0.5 0.02",
            "solref": "0.001 1", "solimp": "0.999 0.9999 0.0001",
            # Contact-neutral for the grasp: pad(2,1) vs
            # object/table(1,1) remains active.
            "condim": "3", "contype": "2", "conaffinity": "1"})
    return ET.tostring(r, encoding="unicode")


def add_mppi_table_force_sensor_xml(plan_xml: Any) -> str:
    """Add the paper's max table-contact force observation to a plan model."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_xml_str(plan_xml))
    if not any(
        geom.get("name") == "lab_table" for geom in root.iter("geom")
    ):
        raise ValueError("paper MPPI requires a named lab_table geom")
    sensors = root.find("sensor")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    if any(
        sensor.get("name") == MPPI_TABLE_FORCE_SENSOR
        for sensor in sensors
    ):
        raise ValueError("paper MPPI table-force sensor is already present")
    if not any(
        body.get("name") == MPPI_TABLE_FORCE_ROBOT_ROOT_BODY
        for body in root.iter("body")
    ):
        raise ValueError(
            "paper MPPI requires the named robot root body "
            f"{MPPI_TABLE_FORCE_ROBOT_ROOT_BODY!r} to scope the table-force "
            "sensor"
        )
    ET.SubElement(
        sensors,
        "contact",
        {
            "name": MPPI_TABLE_FORCE_SENSOR,
            "geom1": "lab_table",
            # Meter the robot pressing on the table, not the objects resting on
            # it.  Without this the sensor's maxforce reduction includes a
            # movable object's own weight -- measured 0.7590 N for the scene's
            # box -- which every candidate pays equally, so it cannot steer the
            # optimizer and only taxes the plans that make contact.
            "subtree2": MPPI_TABLE_FORCE_ROBOT_ROOT_BODY,
            "reduce": "maxforce",
            "num": "1",
            "data": "force",
        },
    )
    return ET.tostring(root, encoding="unicode")


def build_paper_mppi_native_effort_xml(plan_xml: Any) -> str:
    """Give paper-MPPI native arm-torque and signed jaw-force actuators.

    The paper's whole-body pick setup uses effort mode for every robot DoF.
    MuJoCo motors receive physical torque commands; normalized sampled actions
    are scaled by the same leased RobotInfo.tau_max-derived envelope used for
    real dispatch.
    GN01 exposes native signed direct force through RDK ``Grasp(force)``; its
    coupled semantic width coordinate therefore gets one motor whose positive
    command closes (negative generalized force because increasing width opens).
    This transformation is applied only to the private MPPI rollout model; the
    execution/CEM world remains unchanged.  Actuator names are retained so
    state/control address mapping stays exact.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_xml_str(plan_xml))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("paper MPPI requires an actuator section")
    expected = {f"joint{i}" for i in range(1, NJ + 1)}
    converted: set[str] = set()
    for item in list(actuator):
        name = item.get("name", "")
        if name not in expected:
            continue
        joint = item.get("joint")
        if not joint:
            raise ValueError(f"paper MPPI actuator {name!r} has no joint")
        item.tag = "motor"
        item.attrib.clear()
        item.set("name", name)
        item.set("joint", joint)
        joint_index = int(name[5:]) - 1
        limit = float(RIZON4S_MPC_TORQUE_LIMIT_NM[joint_index])
        item.set("gear", "1")
        item.set("ctrlrange", f"{-limit:g} {limit:g}")
        item.set("forcerange", f"{-limit:g} {limit:g}")
        converted.add(name)
    missing = sorted(expected - converted)
    if missing:
        raise ValueError(
            "paper MPPI effort conversion missed actuators: "
            + ", ".join(missing)
        )
    # The reference pick configuration explicitly sets ``gravity: false`` for
    # the controlled mobile manipulator. Flexiv RT_JOINT_TORQUE supplies its
    # nonlinear dynamics compensation separately. MuJoCo's per-body gravcomp
    # reproduces that actuator convention without disabling gravity for the
    # manipulated objects or the rest of the scene.
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("paper MPPI effort conversion needs a worldbody")

    def _contains_joint(body: ET.Element, joint_name: str) -> bool:
        return any(
            joint.get("name") == joint_name
            for joint in body.iter("joint")
        )

    arm_root = next(
        (
            body
            for body in worldbody.iter("body")
            if _contains_joint(body, "joint1")
        ),
        None,
    )
    if arm_root is None:
        raise ValueError("paper MPPI effort conversion missed joint1 body")
    for body in arm_root.iter("body"):
        body.set("gravcomp", "1")
    jaw = next(
        (
            item
            for item in actuator
            if item.get("joint") == "gn01_finger_width_joint"
            or item.get("name") == "gn01_finger_width_pos"
        ),
        None,
    )
    if jaw is None:
        raise ValueError("paper MPPI requires the GN01 coupled jaw actuator")
    jaw_name = jaw.get("name", "gn01_finger_width_pos")
    jaw_joint = jaw.get("joint", "gn01_finger_width_joint")
    jaw.tag = "motor"
    jaw.attrib.clear()
    jaw.set("name", jaw_name)
    jaw.set("joint", jaw_joint)
    jaw.set("gear", "-1")
    jaw.set("ctrlrange", "-80 80")
    jaw.set("forcerange", "-80 80")
    return ET.tostring(root, encoding="unicode")


def build_paper_mppi_legacy_velocity_xml(plan_xml: Any) -> str:
    """Reproduce the ab10 arm-velocity/GN01-width rollout actuator contract.

    Each arm actuator becomes a MuJoCo velocity servo with ``kv=600`` and
    physical command bounds ``[-0.2, 0.2]`` rad/s.  The clean-2ef scene enters
    here with a direct GN01 motor, so the jaw is also explicitly replaced by
    the exact historical ab10 position-servo row.  Merely leaving the current
    row untouched would send metre-valued width commands to a force motor.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_xml_str(plan_xml))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("paper MPPI requires an actuator section")
    expected = {f"joint{i}" for i in range(1, NJ + 1)}
    converted: set[str] = set()
    for item in list(actuator):
        name = item.get("name", "")
        if name not in expected:
            continue
        joint = item.get("joint")
        if not joint:
            raise ValueError(f"paper MPPI actuator {name!r} has no joint")
        item.tag = "velocity"
        item.attrib.clear()
        item.set("name", name)
        item.set("joint", joint)
        item.set("kv", "600")
        item.set("ctrlrange", "-0.2 0.2")
        converted.add(name)
    missing = sorted(expected - converted)
    if missing:
        raise ValueError(
            "paper MPPI velocity conversion missed actuators: "
            + ", ".join(missing)
        )
    jaw = next(
        (
            item
            for item in actuator
            if item.get("joint") == "gn01_finger_width_joint"
            or item.get("name") == "gn01_finger_width_pos"
        ),
        None,
    )
    if jaw is None:
        raise ValueError("legacy paper MPPI requires the GN01 jaw actuator")
    jaw.tag = "position"
    jaw.attrib.clear()
    jaw.set("name", "gn01_finger_width_pos")
    jaw.set("joint", "gn01_finger_width_joint")
    jaw.set("kp", "2700")
    jaw.set("ctrlrange", "0 0.100")
    jaw.set("forcerange", "-60 60")

    converted_xml = ET.tostring(root, encoding="unicode")
    _require_paper_mppi_legacy_xml(converted_xml)
    return converted_xml


def build_paper_mppi_joint_velocity_force_xml(plan_xml: Any) -> str:
    """Combine exact ab10 arm velocity rows with exact signed GN01 force.

    This profile changes only the legacy jaw row after the exact legacy arm
    conversion.  It intentionally does not apply the direct-torque model's
    arm gravity-compensation mutation.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(build_paper_mppi_legacy_velocity_xml(plan_xml))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("joint-velocity-force MPPI requires actuators")
    jaw = next(
        (
            item
            for item in actuator
            if item.get("name") == "gn01_finger_width_pos"
            or item.get("joint") == "gn01_finger_width_joint"
        ),
        None,
    )
    if jaw is None:
        raise ValueError("joint-velocity-force MPPI requires the GN01 jaw")
    jaw.tag = "motor"
    jaw.attrib.clear()
    jaw.set("name", "gn01_finger_width_pos")
    jaw.set("joint", "gn01_finger_width_joint")
    jaw.set("gear", "-1")
    jaw.set("ctrlrange", "-80 80")
    jaw.set("forcerange", "-80 80")
    converted = ET.tostring(root, encoding="unicode")
    _require_paper_mppi_joint_velocity_force_xml(converted)
    return converted


def build_paper_mppi_legacy_velocity_effort_xml(plan_xml: Any) -> str:
    """Change only the successful legacy MPPI jaw actuator to effort.

    The seven arm rows come byte-for-byte from the legacy velocity+width
    transform.  This diagnostic deliberately does not enter the later JVF/ZOH
    runtime path: its interpolation, filter, RNG panel, and arm carrier remain
    the successful Pick-v8 linear pipeline.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(build_paper_mppi_legacy_velocity_xml(plan_xml))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("legacy velocity-effort MPPI requires actuators")
    jaw = next(
        (
            item
            for item in actuator
            if item.get("name") == "gn01_finger_width_pos"
            or item.get("joint") == "gn01_finger_width_joint"
        ),
        None,
    )
    if jaw is None:
        raise ValueError("legacy velocity-effort MPPI requires the GN01 jaw")
    jaw.tag = "motor"
    jaw.attrib.clear()
    jaw.set("name", "gn01_finger_width_pos")
    jaw.set("joint", "gn01_finger_width_joint")
    jaw.set("gear", "-1")
    jaw.set("ctrlrange", "-80 80")
    jaw.set("forcerange", "-80 80")
    converted = ET.tostring(root, encoding="unicode")
    _require_paper_mppi_joint_velocity_force_xml(converted)
    return converted


def _require_paper_mppi_joint_velocity_force_xml(plan_xml: Any) -> None:
    """Fail closed unless the hybrid actuator rows are exact."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_xml_str(plan_xml))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("joint-velocity-force MPPI requires actuators")
    rows = {item.get("name", ""): item for item in actuator}
    for index in range(1, NJ + 1):
        name = f"joint{index}"
        row = rows.get(name)
        expected = {
            "name": name,
            "joint": name,
            "kv": "600",
            "ctrlrange": "-0.2 0.2",
        }
        if row is None or row.tag != "velocity" or row.attrib != expected:
            raise ValueError(f"joint-velocity-force arm row {name!r} is not exact")
    jaw = rows.get("gn01_finger_width_pos")
    expected_jaw = {
        "name": "gn01_finger_width_pos",
        "joint": "gn01_finger_width_joint",
        "gear": "-1",
        "ctrlrange": "-80 80",
        "forcerange": "-80 80",
    }
    if jaw is None or jaw.tag != "motor" or jaw.attrib != expected_jaw:
        raise ValueError("joint-velocity-force GN01 row is not exact")


def _require_paper_mppi_legacy_xml(plan_xml: Any) -> None:
    """Fail closed unless the private rollout XML has exact ab10 actuators."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(_xml_str(plan_xml))
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("legacy paper MPPI requires an actuator section")
    rows = {item.get("name", ""): item for item in actuator}
    for index in range(1, NJ + 1):
        name = f"joint{index}"
        row = rows.get(name)
        expected = {
            "name": name,
            "joint": name,
            "kv": "600",
            "ctrlrange": "-0.2 0.2",
        }
        if row is None or row.tag != "velocity" or row.attrib != expected:
            raise ValueError(
                f"legacy paper MPPI actuator {name!r} is not exact ab10"
            )
    jaw = rows.get("gn01_finger_width_pos")
    expected_jaw = {
        "name": "gn01_finger_width_pos",
        "joint": "gn01_finger_width_joint",
        "kp": "2700",
        "ctrlrange": "0 0.100",
        "forcerange": "-60 60",
    }
    if jaw is None or jaw.tag != "position" or jaw.attrib != expected_jaw:
        raise ValueError("legacy paper MPPI GN01 actuator is not exact ab10")


def build_control_profile_rollout_xml(
    plan_xml: Any,
    *,
    control_profile: str,
) -> tuple[str, dict[str, Any]]:
    """Build and identify the exact private XML compiled for one profile."""
    import xml.etree.ElementTree as ET

    profile = validate_control_profile(control_profile)
    if profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH:
        transformed = build_paper_mppi_legacy_velocity_xml(plan_xml)
    elif profile == CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT:
        transformed = build_paper_mppi_legacy_velocity_effort_xml(plan_xml)
    elif profile == CONTROL_PROFILE_JOINT_VELOCITY_FORCE:
        transformed = build_paper_mppi_joint_velocity_force_xml(plan_xml)
    else:
        transformed = build_paper_mppi_native_effort_xml(plan_xml)
    transformed = add_mppi_table_force_sensor_xml(transformed)
    root = ET.fromstring(transformed)
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("control-profile rollout XML omitted actuators")
    names = [f"joint{index}" for index in range(1, NJ + 1)] + [
        "gn01_finger_width_pos"
    ]
    by_name = {row.get("name", ""): row for row in actuator}
    if any(name not in by_name for name in names):
        raise ValueError("control-profile rollout XML actuator set is incomplete")
    _, transformed_sha256 = _plan_xml_text_and_sha256(transformed)
    return transformed, {
        "schema": "oap_control_profile_xml_identity_v1",
        "control_profile": profile,
        "transformed_exact_xml_sha256": transformed_sha256,
        "private_xml_mutators": [
            (
                "exact_ab10_arm_velocity_kv600_ctrl_pm0p2_and_"
                "gn01_position_kp2700_ctrl_0_0p100_force_pm60"
                if profile == CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
                else "exact_ab10_arm_velocity_linear_v8_and_"
                "gn01_direct_signed_force_pm80"
                if profile == CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT
                else "exact_ab10_arm_velocity_kv600_ctrl_pm0p2_and_"
                "gn01_direct_signed_force_motor_pm80"
                if profile == CONTROL_PROFILE_JOINT_VELOCITY_FORCE
                else "clean_2ef_native_direct_torque_force"
            ),
            "mppi_table_force_sensor_only",
        ],
        "actuator_rows": [
            {
                "tag": by_name[name].tag,
                "attributes": dict(by_name[name].attrib),
            }
            for name in names
        ],
        "arm_action_units": (
            "velocity_rad_s"
            if control_profile_uses_joint_velocity_arm(profile)
            else "normalized_torque_scaled_by_fixed_nm_envelope"
        ),
        "jaw_semantic_units": "signed_latent_open_minus1_closed_plus1",
        "jaw_actuator_units": (
            "width_m"
            if control_profile_uses_width_jaw(profile)
            else "effort_n"
        ),
    }


# Preserve the production 2ef0967 symbol/behavior.  The experiment-only legacy
# adapter has its own explicit name above so callers cannot silently change
# actuator semantics by importing an established production symbol.
build_paper_mppi_native_velocity_xml = build_paper_mppi_native_effort_xml


# Compatibility for code that reads sealed pre-effort proposals by symbol.
paper_mppi_velocity_proposals = paper_mppi_effort_proposals


def mppi_control_profile_parameters(
    *,
    control_profile: str,
    position_nominal: np.ndarray,
    measured_arm: np.ndarray,
    knot_dt_s: float,
    nominal_actions: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return the profile-specific action carrier without drawing any noise.

    Randomness stays downstream of this deterministic mapping.  Consequently
    one seed produces the same normalized MTP panel in every profile: direct
    uses unit arm bounds, while both velocity-arm profiles scale those exact
    normalized coordinates by 0.2 rad/s.  The jaw latent remains normalized in
    all profiles and only its actuator decode differs.
    """
    profile = validate_control_profile(control_profile)
    positions = np.asarray(position_nominal, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != NJ + 1:
        raise ValueError("MPPI nominal must be a (K, 8) knot array")
    measured = np.asarray(measured_arm, dtype=float)
    if measured.shape != (NJ,) or not np.all(np.isfinite(measured)):
        raise ValueError("measured_arm must be one finite seven-joint row")
    dt = float(knot_dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("knot_dt_s must be finite and positive")

    arm_half_range = (
        0.2
        if control_profile_uses_joint_velocity_arm(profile)
        else 1.0
    )
    lo = np.concatenate((np.full(NJ, -arm_half_range), np.array([-1.0])))
    hi = np.concatenate((np.full(NJ, arm_half_range), np.array([1.0])))
    if nominal_actions is None:
        if control_profile_uses_joint_velocity_arm(profile):
            previous = np.vstack((measured[None, :], positions[:-1, :NJ]))
            mean_arm = (positions[:, :NJ] - previous) / dt
        else:
            mean_arm = np.zeros((positions.shape[0], NJ), dtype=float)
        mean = np.concatenate(
            (mean_arm, np.clip(positions[:, NJ], -1.0, 1.0)[:, None]),
            axis=1,
        )
    else:
        mean = np.asarray(nominal_actions, dtype=float)
        if mean.shape != positions.shape:
            raise ValueError(
                "MPPI nominal_actions must match the (K, 8) action plan"
            )
        if not np.all(np.isfinite(mean)):
            raise ValueError("MPPI nominal_actions must be finite")
    return {
        "control_profile": profile,
        "mean": np.clip(mean, lo, hi),
        "lo": lo,
        "hi": hi,
        "half_range": 0.5 * (hi - lo),
        "arm_action": (
            "position_carrier_derived_velocity_kv600"
            if control_profile_uses_joint_velocity_arm(profile)
            else "normalized_gravity_compensated_joint_torque"
        ),
        "jaw_action": (
            "continuous_width_target"
            if control_profile_uses_width_jaw(profile)
            else "continuous_signed_effort"
        ),
    }


def _pick_v8_causal_mppi_summary_identity(
    *,
    pick_v8_causal_profile: str | None,
    control_profile: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Describe the actual causal MPPI carrier without changing defaults."""
    from oap.twin.pick_v8_causal import (
        PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N,
        pick_v8_causal_uses_asymmetric_effort,
        validate_pick_v8_causal_profile,
    )

    causal_profile = validate_pick_v8_causal_profile(
        pick_v8_causal_profile,
        allow_none=True,
    )
    if causal_profile is None:
        return {}
    # The device expansion contract owns the profile/control pairing and
    # interpolation.  Reuse it here so telemetry cannot describe a carrier
    # that production would refuse to execute.
    _validate_paper_mppi_expansion_contract(
        spline_order="linear",
        pick_v8_causal_profile=causal_profile,
        control_profile=control_profile,
    )
    width_jaw = control_profile_uses_width_jaw(control_profile)
    identity: dict[str, Any] = {
        "pick_v8_causal_profile": causal_profile,
        "control_profile": control_profile,
        "proposal": (
            "gaussian_halton_degree1_arm_velocity_jaw_width_actions"
            if width_jaw
            else "gaussian_halton_degree1_arm_velocity_jaw_effort_actions"
        ),
        "action_interpolation": "linear_degree_one",
        "arm_action": parameters["arm_action"],
        "arm_action_units": "velocity_rad_s",
        "jaw_action": parameters["jaw_action"],
        "jaw_command_semantics": "signed_latent_open_minus1_closed_plus1",
        "jaw_actuator_decode": (
            "continuous_width_target_m"
            if width_jaw
            else "asymmetric_continuous_effort_open_1n_close_80n"
            if pick_v8_causal_uses_asymmetric_effort(causal_profile)
            else "continuous_signed_effort_n"
        ),
    }
    if width_jaw:
        identity["jaw_width_range_m"] = [0.0, GN01_OPEN_WIDTH_M]
    else:
        identity["jaw_effort_range_n"] = [
            -(
                PICK_V8_GRIPPER_ONLY_V2_OPEN_FORCE_LIMIT_N
                if pick_v8_causal_uses_asymmetric_effort(causal_profile)
                else GN01_DIRECT_FORCE_LIMIT_N
            ),
            GN01_DIRECT_FORCE_LIMIT_N,
        ]
    return identity


@dataclass
class BatchedRollout:
    """A batched joint-space rollout of the full twin on the warp backend.

    ``gripper="linkage"`` (the default) rolls the real position-controlled gn01
    and HOLDS a grasp on warp: the 2026-07-17 "a grip needs CPU NoSlip" claim
    was a state-restore bug in :meth:`run`, not a solver gap, and NoSlip moves a
    150 mm CPU carry by 0.02 mm. ``gripper="slide"`` swaps in the force-limited
    pad slides (Option A); it is now an ABLATION arm only -- 9.4 mm mean error
    against the CPU certifier where the linkage is 0.1 mm. All qpos/actuator
    addresses are resolved from ``model``, which may be the slide-surgered model
    whose DoF layout differs from ``world``."""

    model: mujoco.MjModel
    mx: Any
    geom_ids: dict[str, int]
    support_gids: tuple[int, ...]
    object_qpos_addr: int
    arm_qpos_addr: np.ndarray
    arm_act_ids: np.ndarray
    grip_act_ids: np.ndarray
    grip_qpos_addrs: np.ndarray
    gripper: str
    nu: int
    control_profile: str = CONTROL_PROFILE_DIRECT_TORQUE_FORCE
    experiment_control_profile: bool = False
    control_regularizer_calibration_profile: str | None = None
    # Immutable identity of the public production conversion and the exact
    # runtime/model inputs that own every compiled executable below.
    mjx_graph_mode: str = "WARP"
    mjx_conversion_evidence: dict[str, Any] = field(default_factory=dict)
    runtime_fingerprint_sha256: str = ""
    plan_xml_sha256: str = ""
    site_id: int = -1
    table_contact_force_sensor_adr: int = -1
    table_contact_force_weight: float = 0.0
    # Exact contact-active convex arm meshes from the plan XML.  Kept separate
    # from gripper geoms so direct robot/movable contact includes both sets.
    exact_arm_gids: tuple[int, ...] = ()
    # Contact-active geoms of the movable (freejoint) bodies OTHER than the
    # subject, and of the robot itself. Kept apart because "the subject pressed
    # on a movable body" is manipulation while "a static support was
    # penetrated" is a violation, and one number could not say which.
    movable_gids: tuple[int, ...] = ()
    # Per-free-body geom ownership for mediation. Physical penetration remains
    # aggregated over ``movable_gids``; causal evidence uses only the body named
    # by the current program terminal.
    movable_geom_groups: tuple[tuple[str, tuple[int, ...]], ...] = ()
    robot_gids: tuple[int, ...] = ()
    # Fixed active-geom role used only by RobotSubjectContact. The robot role
    # is the exact union of ``robot_gids`` and ``exact_arm_gids``; this tuple
    # stores the complete controlled-subject subtree from the same resolver.
    subject_contact_gids: tuple[int, ...] = ()
    # Task-object/support collision set used only to expose whether the first
    # post-step state inherited robot/scene contact. It is evidence for the
    # episode gate, never a cost or candidate-validity term.
    initial_scene_gids: tuple[int, ...] = ()
    # (name, qpos address) for every OTHER free body, in a fixed order. The
    # scorer needs these because a goal about a body the robot does not hold --
    # a box a tool pushes -- is INERT unless the rollout reports where that body
    # ended up: every candidate would score identically and the optimizer would
    # have nothing to descend. Empty for a single-mover scene, where every path
    # below collapses to exactly its previous behaviour.
    movable_addrs: tuple[tuple[str, int], ...] = ()
    # World-side qpos addresses, so a caller's live state can be MAPPED in
    # rather than copied. The slide-gripper variant has a different DoF layout
    # (nu=9, prismatic pads instead of the 4-bar linkage), so a wholesale qpos
    # copy would scramble the gripper; only arm joints and free bodies are
    # positionally meaningful across the two models.
    world_arm_qpos_addr: tuple[int, ...] = ()
    world_object_qpos_addr: int = -1
    world_movable_addrs: tuple[tuple[str, int], ...] = ()
    # (world_qadr, model_qadr) for EVERY gripper joint this model shares with the
    # world by name. The gn01's finger pose is carried by six equality-coupled
    # knuckle joints, not by gn01_finger_width_joint alone; restoring only the
    # width joint left the knuckles at the template default (0.0 rad = wide
    # open) while the live grasp held them at 0.2797, so a continued rollout
    # began with the fingers OPEN and dropped the object every time. Empty for
    # the slide variant, whose pad joints do not exist in the world model --
    # that path keeps riding its control, exactly as before.
    grip_state_addrs: tuple[tuple[int, int], ...] = ()
    # Name-matched DOF indices for velocity. The CPU reference restores qvel
    # wholesale (full_fidelity_rollout does d.qvel[:] = start_state[1]); the GPU
    # path used to restore qpos only, so a continued rollout started FROM REST.
    # Harmless at a stage boundary, wrong mid-stage: the receding horizon
    # re-plans after executing a 34% prefix, with the arm still moving.
    dof_src: tuple[int, ...] = ()          # indices into the WORLD qvel
    dof_dst: tuple[int, ...] = ()          # indices into THIS model's qvel
    # Name-matched qpos indices for transferring the selected GPU prefix back
    # into the live simulation twin.  This is the reverse of state seeding:
    # the host copies arrays only; no CPU dynamics or scoring is run.
    qpos_src: tuple[int, ...] = ()         # indices into the WORLD qpos
    qpos_dst: tuple[int, ...] = ()         # indices into THIS model's qpos
    ctrl_src: tuple[int, ...] = ()         # indices into the WORLD ctrl
    ctrl_dst: tuple[int, ...] = ()         # indices into THIS model's ctrl
    # Planning-grid decoupling (mjpc/hydrax convention: the grid the planner
    # SCORES on is coarser than the grid the robot EXECUTES on). exec_dt is the
    # plan world's own timestep (the 2 ms execution grid); plan_dt is the grid
    # this screen ROLLS at. Equal by default (legacy behavior, byte-identical).
    exec_dt: float = 0.002
    plan_dt: float = 0.002
    # Mutable caches are allocated with the screen itself. ``dataclasses.replace``
    # (used on a build-cache hit to refresh world-side address maps) preserves
    # these dict identities, so every view of one compiled screen shares the same
    # Data buffers, JIT handles and executables.
    #
    # Data is keyed by naconmax (njmax is a constant). Reusing those buffers
    # avoids needless allocator churn and gives the warp FFI a chance to reuse
    # captured CUDA graphs; graph reuse is still backend- and pointer-key
    # dependent.  Each full rollout specialization retains its own JIT handle
    # and executable.  This is required for correctness: ``rollout_batch``
    # closes over N because the linear contact-arena reduction has a static
    # ``(N, feature_count)`` output. Reusing a handle first traced at another N
    # would retain that old world count even when lowering with new arguments.
    _data_templates: dict[int, Any] = field(
        default_factory=dict, repr=False, compare=False)
    _rollout_functions: dict[
        tuple[int, int, int, bool, int, bool, int], Any
    ] = field(
        default_factory=dict, repr=False, compare=False)
    _compiled_rollouts: dict[
        tuple[int, int, int, bool, int, bool, int], Any
    ] = field(
        default_factory=dict, repr=False, compare=False)
    _device_score_functions: dict[tuple[Any, tuple[str, ...]], Any] = field(
        default_factory=dict, repr=False, compare=False)
    _compiled_device_scores: dict[tuple[Any, ...], Any] = field(
        default_factory=dict, repr=False, compare=False)

    def compilation_cache_snapshot(
        self,
    ) -> tuple[frozenset[Any], frozenset[Any]]:
        """Return immutable executable-cache keys for startup verification.

        The persistent remote service calls this around its second direct
        warm-up execution.  Equality proves that the purported hot pass reused
        both the rollout specialization and the exact device-cost scorer rather
        than compiling a new executable behind a misleading ``ready`` flag.
        """
        return (
            frozenset(self._compiled_rollouts),
            frozenset(self._compiled_device_scores),
        )

    def compilation_cache_inventory(self) -> dict[str, Any]:
        """Return compact JSON-safe evidence about compiled GPU shapes."""
        rollout_signatures = [
            {
                "pool_size": int(pool_size),
                "horizon_steps": int(horizon_steps),
                "naconmax": int(naconmax),
                "continued_state": bool(continued),
                "mediation_target_count": int(target_count),
                "gpu_step_trace": bool(step_trace),
                "execution_prefix_steps": int(prefix_steps),
            }
            for (
                pool_size,
                horizon_steps,
                naconmax,
                continued,
                target_count,
                step_trace,
                prefix_steps,
            ) in sorted(self._compiled_rollouts)
        ]
        return {
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "plan_xml_sha256": self.plan_xml_sha256,
            "rollout_executable_count": len(self._compiled_rollouts),
            "device_score_executable_count": len(
                self._compiled_device_scores
            ),
            "rollout_signatures": rollout_signatures,
        }

    def production_runtime_evidence(self) -> dict[str, Any]:
        """Return the exact runtime/model identity serialized per solve.

        An explicit control-profile experiment compiles a private actuator
        model rather than executing the public planning XML verbatim.  Its
        transformed XML identity is therefore required evidence, not optional
        debug metadata.  Default production packets retain their historical
        schema when no explicit profile is active.
        """
        evidence = {
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "plan_xml_sha256": self.plan_xml_sha256,
            "identity": self.mjx_conversion_evidence.get("runtime_identity"),
            "mjx_graph_mode": self.mjx_graph_mode,
        }
        if self.experiment_control_profile:
            profile_xml = self.mjx_conversion_evidence.get(
                "control_profile_xml_identity"
            )
            if not isinstance(profile_xml, dict):
                raise RuntimeError(
                    "explicit control profile omitted transformed rollout XML "
                    "identity"
                )
            evidence["control_profile_xml_identity"] = dict(profile_xml)
        regularizer_profile = getattr(
            self,
            "control_regularizer_calibration_profile",
            None,
        )
        if regularizer_profile is not None:
            evidence[
                "control_regularizer"
                if regularizer_profile == UNIFIED_CONTROL_REGULARIZER_V1
                else "control_regularizer_calibration"
            ] = (
                control_regularizer_formula_identity(
                    regularizer_profile
                )
            )
        return evidence

    @classmethod
    def build(cls, world: Any, plan_xml: Any, *, gripper: str = "linkage",
              plan_timestep: float | None = None,
              paper_mppi: bool = False,
              control_profile: str | None = None,
              table_contact_force_weight: float | None = None,
              control_regularizer_calibration_profile: str | None = None,
              ) -> "BatchedRollout":
        """Build the fixed paper-standard public-WARP rollout.

        ``plan_timestep`` and ``OAP_PLAN_DT`` remain accepted only so an
        old configuration fails with an explicit explanation.  They may repeat
        the model timestep, but may not coarsen it.  A 4 ms grid once appeared
        acceptable on one grasp fixture; the tool-use fixture later produced
        only 9/16 lift-outcome agreement with the 2 ms grid.  Contact-rich
        manipulation therefore has no scene-general coarsening envelope.
        Production planning stays GPU-only and uses the exact execution grid;
        CPU rollout remains a test/offline parity oracle only.

        There is deliberately no graph-mode, deterministic-mode, debug, or
        record-cap override in this public production entry point.
        """
        # Reuse an identical screen instead of rebuilding it. This cache owns
        # all three expensive layers: the MjModel/mjx.Model, make_data templates,
        # and the stable JIT handles + shape-specialized executables populated by
        # run(). The last distinction matters: JAX keys its in-memory trace cache
        # by FUNCTION IDENTITY, so jitting a newly-defined rollout closure on
        # every call recompiles even when its source and shapes are identical.
        # run() now passes every cycle value (obj_q/z0 and the mapped live
        # qpos/qvel state) as a traced argument; the retained callable closes
        # over model structure only.
        #
        # Measured on an H100, one solve_mpc_step at 400 steps:
        # 16.3 s at pool 16 and 16.5 s at pool 1024 -- almost FLAT in pool
        # size, i.e. ~10 s of it was the repeatedly-compiled rollout, paid again
        # on every cycle even though the plan XML never changes within a stage.
        #
        # Safe only because nothing per-cycle is baked in any more: the lift
        # datum moved to run() (it is obj_pose7[2], the caller's own start
        # pose), and the world-side maps are recomputed below on a hit, since
        # the same plan XML can be paired with a different SimWorld instance.
        # The key covers everything that changes the compiled artifact.
        experiment_control_profile = control_profile is not None
        control_profile = validate_control_profile(
            control_profile or CONTROL_PROFILE_DIRECT_TORQUE_FORCE
        )
        regularizer_profile = validate_control_regularizer_calibration_profile(
            control_regularizer_calibration_profile,
            allow_none=True,
        )
        if regularizer_profile is not None and (
            not paper_mppi
            or not experiment_control_profile
            or control_profile != CONTROL_PROFILE_JOINT_VELOCITY_FORCE
        ):
            raise ValueError(
                "control regularizer calibration requires explicit "
                "joint_velocity_force paper MPPI"
            )
        if not paper_mppi and experiment_control_profile:
            raise ValueError("legacy control_profile requires paper_mppi=True")
        if table_contact_force_weight is None:
            resolved_table_force_weight = (
                DEFAULT_MPPI_TABLE_FORCE_WEIGHT if paper_mppi else 0.0
            )
        else:
            if (
                isinstance(table_contact_force_weight, bool)
                or not isinstance(table_contact_force_weight, (int, float))
            ):
                raise ValueError(
                    "table_contact_force_weight must be a finite "
                    "non-negative number or None"
                )
            resolved_table_force_weight = float(
                table_contact_force_weight
            )
            if (
                not math.isfinite(resolved_table_force_weight)
                or resolved_table_force_weight < 0.0
            ):
                raise ValueError(
                    "table_contact_force_weight must be a finite "
                    "non-negative number or None"
                )
            if resolved_table_force_weight > 0.0 and not paper_mppi:
                raise ValueError(
                    "a nonzero table_contact_force_weight requires "
                    "paper_mppi=True so the exact table-force sensor is "
                    "present"
                )

        runtime_identity = bootstrap_production_mjwarp()
        _xml_text, _xml_sha256 = _plan_xml_text_and_sha256(plan_xml)
        _key = (runtime_identity["fingerprint_sha256"], _xml_sha256,
                str(gripper),
                None if plan_timestep is None else float(plan_timestep),
                plan_dt_from_env(),
                bool(paper_mppi),
                control_profile,
                experiment_control_profile,
                resolved_table_force_weight) + control_regularizer_cache_identity(
                    regularizer_profile
                )
        _hit = _SCREEN_CACHE.get(_key)
        if _hit is not None:
            qpos_src, qpos_dst = _shared_qpos_map(world.model, _hit.model)
            dof_src, dof_dst = _shared_dof_map(world.model, _hit.model)
            ctrl_src, ctrl_dst = _shared_ctrl_map(world.model, _hit.model)
            return replace(
                _hit,
                world_arm_qpos_addr=tuple(int(a) for a in world.robot["qpos_addr"]),
                world_object_qpos_addr=int(world.object_qpos_addr),
                world_movable_addrs=tuple(
                    (n, int(a)) for n, a in
                    (getattr(world, "movable_qpos_addr", None) or {}).items()
                    if int(a) != int(world.object_qpos_addr)),
                grip_state_addrs=_shared_gripper_qpos(world.model, _hit.model),
                dof_src=dof_src,
                dof_dst=dof_dst,
                qpos_src=qpos_src,
                qpos_dst=qpos_dst,
                ctrl_src=ctrl_src,
                ctrl_dst=ctrl_dst)

        # An independent model fresh from the plan XML.  The plan's real
        # contact-active convex arm meshes remain enabled; there is no geometry
        # surgery on the production linkage path.  For the slide ablation only,
        # the XML is altered to replace the linkage with force pad slides, which
        # changes the gripper DoF count (so every address below is resolved from
        # this model, not from ``world``).
        exact_xml = _xml_text
        control_xml_identity: dict[str, Any] | None = None
        if gripper == "slide":
            exact_xml = build_slide_gripper_xml(exact_xml)
            grip_act_names = ("gn01_left_pad_pos", "gn01_right_pad_pos")
            grip_jnt_names = ("gn01_left_pad_j", "gn01_right_pad_j")
        elif gripper == "linkage":
            grip_act_names = ("gn01_finger_width_pos",)
            grip_jnt_names = ("gn01_finger_width_joint",)
        else:
            raise ValueError(f"gripper must be 'linkage' or 'slide', got {gripper!r}")
        if paper_mppi:
            if gripper != "linkage":
                raise ValueError(
                    "paper MPPI native effort drive requires the linkage gripper"
                )
            if experiment_control_profile:
                exact_xml, control_xml_identity = (
                    build_control_profile_rollout_xml(
                        exact_xml,
                        control_profile=control_profile,
                    )
                )
            else:
                exact_xml = build_paper_mppi_native_effort_xml(exact_xml)
                exact_xml = add_mppi_table_force_sensor_xml(exact_xml)
        m = mujoco.MjModel.from_xml_string(exact_xml)
        exact_arm_gids = exact_arm_collision_geoms(m)
        # warp-legal solver (no NoSlip on warp; elliptic + impratio replaces it).
        m.opt.noslip_iterations = 0
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = 10.0
        # Analytic box-box instead of GJK/EPA. mujoco_warp routes (BOX, BOX) to
        # CollisionType.CONVEX (collision_driver.py:76), whose CCD has a
        # DETECTION FLOOR -- measured 9.65-10.15 um for our pad pair, 40.34 um
        # for the 0.86 m table box, i.e. it scales with geometry. A gn01 grasp
        # penetrates 2.0-2.2 um, inside the blind band, so warp reported
        # nacon=0 and 0 N of pad force where CPU had 16.2053 N. With this flag
        # warp takes the same analytic collider CPU uses and the force matches
        # to +0.017% (16.2080 vs 16.2053 N).
        #
        # kp=2700 alone already clears the floor by squeezing deeper, and on
        # the rank-breaking carry batch the two are indistinguishable (0.1 mm
        # either way). This is belt-and-braces for the case kp cannot cover: a
        # LIGHTER grasp -- smaller object, lower clamp -- lands back under the
        # floor, and there the flag is what saves it. It is also not a cost:
        # measured 0.777 s vs 0.936 s for 16x480 steps.
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        exec_dt = float(m.opt.timestep)                  # the execution grid (2 ms)
        if exec_dt > 0.005:
            # The grip-parity envelope was validated at the 2 ms execution grid;
            # a plan XML that already carries a coarser timestep is outside it
            # on the DEFAULT path, where the knob guard below never runs.
            logger.warning("plan XML timestep %.4fs exceeds the validated 5 ms "
                           "grip envelope; screen verdicts are unvalidated", exec_dt)
        if plan_timestep is None:
            plan_timestep = plan_dt_from_env()
        if plan_timestep is not None:
            requested_dt = float(plan_timestep)
            if not np.isclose(requested_dt, exec_dt, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "plan_timestep/OAP_PLAN_DT must equal the model "
                    f"execution timestep {exec_dt}; contact outcome parity is "
                    f"not preserved by the requested {requested_dt}"
                )
        # Public MJX conversion only, after fixed production bootstrap.  The
        # conversion helper verifies the resulting ModelWarp and graph mode.
        if paper_mppi:
            mx, actual_graph_mode, conversion_evidence = (
                put_zero_sensor_mjx_model(
                    m,
                    allow_mppi_table_force_sensor=True,
                )
            )
        else:
            # Preserve the production sensor-free conversion route exactly.
            mx, actual_graph_mode, conversion_evidence = (
                put_zero_sensor_mjx_model(m)
            )
        conversion_evidence = dict(conversion_evidence)
        if experiment_control_profile:
            if control_xml_identity is None:
                raise RuntimeError("control-profile XML identity was not built")
            conversion_evidence["control_profile_xml_identity"] = {
                **control_xml_identity,
                "source_plan_xml_sha256": _xml_sha256,
            }
        logger.info(
            "[jointspace-screen] MJX backend=warp graph_mode=%s",
            actual_graph_mode,
        )

        def gid(name: str) -> int:
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

        def act(name: str) -> int:
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        def jqadr(name: str) -> int:
            return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])

        site_id = mujoco.mj_name2id(
            m,
            mujoco.mjtObj.mjOBJ_SITE,
            "gn01_grasp_center_site",
        )
        if site_id < 0:
            raise ValueError(
                "rollout model has no gn01_grasp_center_site; "
                "cannot ground reserved gripper_center"
            )

        geom_ids = _required_named_ids(
            (
                "pick_object_collision",
                "gn01_left_finger_tip_collision",
                "gn01_right_finger_tip_collision",
                "lab_table",
            ),
            gid,
            label="contact geoms",
        )
        _active = lambda g: bool(int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))
        # Geoms of every free body EXCEPT the subject: the things a plan may
        # legitimately push. Keep body ownership rather than only a union: tool
        # mediation must grade contact with the terminal's target, not a
        # distractor that merely happens to be movable.
        _movable_groups = _movable_geom_groups(m)
        movable_gids = tuple(sorted(set().union(*_movable_groups.values())))
        initial_scene_gids = _initial_scene_collision_geoms(
            m,
            movable_gids,
        )
        # The robot's own contact-active gripper geoms. Exact arm meshes are
        # tracked separately and unioned at use.
        robot_gids = tuple(
            g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("gn01_")
            and _active(g))
        subject_contact_gids, robot_subject_gids = (
            robot_subject_contact_geom_roles(
                m,
                exact_arm_gids=exact_arm_gids,
            )
        )
        if set(robot_subject_gids) != set(robot_gids) | set(exact_arm_gids):
            raise ValueError(
                "robot-subject contact role diverges from rollout robot masks"
            )
        # Every named support is a world obstacle by default, including a
        # support carried by a freejoint. ``run`` removes only targets selected
        # by typed subject-contact relations in the current Stage. Blanket
        # removal of every movable support would silently authorize contact
        # with unrelated movable bodies.
        supports = tuple(
            g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                "support_") and _active(g))
        obj_adr = jqadr("pick_object_freejoint")
        # Every OTHER free body, resolved from THIS model (the slide-surgered
        # variant has a different DoF layout, so world's map cannot be reused).
        mov_addrs = []
        for jid in range(m.njnt):
            if int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if not jname.endswith("_freejoint") or jname == "pick_object_freejoint":
                continue
            mov_addrs.append((jname[: -len("_freejoint")], int(m.jnt_qposadr[jid])))
        arm_qadr = np.asarray([jqadr(f"joint{i + 1}") for i in range(NJ)], dtype=int)
        arm_act = np.asarray([act(f"joint{i + 1}") for i in range(NJ)], dtype=int)
        dof_src, dof_dst = _shared_dof_map(world.model, m)
        qpos_src, qpos_dst = _shared_qpos_map(world.model, m)
        ctrl_src, ctrl_dst = _shared_ctrl_map(world.model, m)
        table_force_sensor_adr = -1
        if paper_mppi:
            sensor_id = mujoco.mj_name2id(
                m,
                mujoco.mjtObj.mjOBJ_SENSOR,
                MPPI_TABLE_FORCE_SENSOR,
            )
            if sensor_id < 0:
                raise RuntimeError("paper MPPI table-force sensor is missing")
            table_force_sensor_adr = int(m.sensor_adr[sensor_id])
        if len(_SCREEN_CACHE) >= _SCREEN_CACHE_MAX:
            _SCREEN_CACHE.pop(next(iter(_SCREEN_CACHE)))
        _SCREEN_CACHE[_key] = built = cls(
            model=m, mx=mx, geom_ids=geom_ids, support_gids=supports,
                   movable_gids=movable_gids,
                   movable_geom_groups=tuple(
                       (name, tuple(sorted(geoms)))
                       for name, geoms in _movable_groups.items()
                   ),
                   robot_gids=robot_gids,
                   subject_contact_gids=subject_contact_gids,
                   initial_scene_gids=initial_scene_gids,
                   object_qpos_addr=obj_adr, arm_qpos_addr=arm_qadr,
                   arm_act_ids=arm_act,
                   grip_act_ids=np.asarray([act(n) for n in grip_act_names], dtype=int),
                   grip_qpos_addrs=np.asarray([jqadr(n) for n in grip_jnt_names], dtype=int),
                   site_id=int(site_id), gripper=gripper, nu=int(m.nu),
                   control_profile=control_profile,
                   experiment_control_profile=experiment_control_profile,
                   control_regularizer_calibration_profile=regularizer_profile,
                   mjx_graph_mode=actual_graph_mode,
                   mjx_conversion_evidence=conversion_evidence,
                   runtime_fingerprint_sha256=runtime_identity[
                       "fingerprint_sha256"
                   ],
                   plan_xml_sha256=_xml_sha256,
                   table_contact_force_sensor_adr=table_force_sensor_adr,
                   table_contact_force_weight=resolved_table_force_weight,
                   exact_arm_gids=exact_arm_gids,
                   movable_addrs=tuple(mov_addrs),
                   world_arm_qpos_addr=tuple(int(a) for a in world.robot["qpos_addr"]),
                   world_object_qpos_addr=int(world.object_qpos_addr),
                   world_movable_addrs=tuple(
                       (n, int(a)) for n, a in
                       (getattr(world, "movable_qpos_addr", None) or {}).items()
                       if int(a) != int(world.object_qpos_addr)),
                   grip_state_addrs=_shared_gripper_qpos(world.model, m),
                   dof_src=dof_src, dof_dst=dof_dst,
                   qpos_src=qpos_src, qpos_dst=qpos_dst,
                   ctrl_src=ctrl_src, ctrl_dst=ctrl_dst,
                   exec_dt=exec_dt, plan_dt=float(m.opt.timestep))
        return built

    _NJMAX_PER_WORLD = 1024          # per-world STRICT constraint cap (see run())

    def _data_template(self, naconmax: int):
        """Cached mjx.make_data template (see the sizing comment in run())."""
        from mujoco import mjx
        key = int(naconmax)
        if key not in self._data_templates:
            self._data_templates[key] = mjx.make_data(
                self.model, impl="warp",
                njmax=self._NJMAX_PER_WORLD, naconmax=key)
        return self._data_templates[key]

    def _mapped_start_state(
        self,
        start_state: "tuple[np.ndarray, np.ndarray] | None",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Map a live world's state into fixed-shape rollout-model arrays.

        The compiled continued-state runner cannot close over a cycle's qpos or
        qvel, nor can optional address/value tuples change its pytree structure.
        Full-size value + mask arrays make all of those quantities ordinary
        traced arguments while preserving the old selective-copy semantics.
        """
        qpos_value = np.zeros(int(self.model.nq), dtype=float)
        qpos_mask = np.zeros(int(self.model.nq), dtype=bool)
        qvel_value = np.zeros(int(self.model.nv), dtype=float)
        qvel_mask = np.zeros(int(self.model.nv), dtype=bool)
        restore_grip = False
        if start_state is None:
            return qpos_value, qpos_mask, qvel_value, qvel_mask, restore_grip

        sq = np.asarray(start_state[0], dtype=float)
        arm_dst = np.asarray(self.arm_qpos_addr, dtype=int)
        arm_src = np.asarray(self.world_arm_qpos_addr, dtype=int)
        qpos_value[arm_dst] = sq[arm_src]
        qpos_mask[arm_dst] = True

        obj_dst = int(self.object_qpos_addr)
        obj_src = int(self.world_object_qpos_addr)
        qpos_value[obj_dst:obj_dst + 7] = sq[obj_src:obj_src + 7]
        qpos_mask[obj_dst:obj_dst + 7] = True

        world_movables = dict(self.world_movable_addrs)
        for name, dst in self.movable_addrs:
            if name not in world_movables:
                continue
            src = int(world_movables[name])
            qpos_value[int(dst):int(dst) + 7] = sq[src:src + 7]
            qpos_mask[int(dst):int(dst) + 7] = True

        # If at least one real gripper joint is shared, retain the historical
        # behavior: restore the mapped mechanism and do not overwrite any
        # unmatched joint with the knot's scalar width. The slide surrogate has
        # no shared joints and therefore keeps riding grip0 in the compiled body.
        if self.grip_state_addrs:
            restore_grip = True
            for src, dst in self.grip_state_addrs:
                qpos_value[int(dst)] = sq[int(src)]
                qpos_mask[int(dst)] = True

        if (self.dof_dst and len(start_state) > 1
                and start_state[1] is not None):
            sv = np.asarray(start_state[1], dtype=float)
            dof_dst = np.asarray(self.dof_dst, dtype=int)
            dof_src = np.asarray(self.dof_src, dtype=int)
            qvel_value[dof_dst] = sv[dof_src]
            qvel_mask[dof_dst] = True

        return qpos_value, qpos_mask, qvel_value, qvel_mask, restore_grip

    def selected_prefix_world_state(
        self,
        *,
        prefix_qpos: np.ndarray,
        prefix_qvel: np.ndarray,
        start_state: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a selected MJWarp prefix endpoint into the live world's layout.

        The returned arrays are the exact device-integrated state at the
        executed prefix for every joint shared by name and type.  Unshared
        host-only state is retained from ``start_state``.  This method performs
        no MuJoCo step, forward dynamics, contact query, or CPU re-ranking.
        """
        rollout_qpos = np.asarray(prefix_qpos, dtype=float)
        rollout_qvel = np.asarray(prefix_qvel, dtype=float)
        if rollout_qpos.shape != (int(self.model.nq),):
            raise ValueError(
                "selected GPU prefix qpos has shape "
                f"{rollout_qpos.shape}, expected {(int(self.model.nq),)}"
            )
        if rollout_qvel.shape != (int(self.model.nv),):
            raise ValueError(
                "selected GPU prefix qvel has shape "
                f"{rollout_qvel.shape}, expected {(int(self.model.nv),)}"
            )
        world_qpos = np.asarray(start_state[0], dtype=float).copy()
        world_qvel = np.asarray(start_state[1], dtype=float).copy()
        if self.qpos_src:
            world_qpos[np.asarray(self.qpos_src, dtype=int)] = rollout_qpos[
                np.asarray(self.qpos_dst, dtype=int)
            ]
        if self.dof_src:
            world_qvel[np.asarray(self.dof_src, dtype=int)] = rollout_qvel[
                np.asarray(self.dof_dst, dtype=int)
            ]
        return world_qpos, world_qvel

    def selected_prefix_world_control(
        self,
        *,
        prefix_ctrl: np.ndarray,
        start_control: np.ndarray,
    ) -> np.ndarray:
        """Map the selected endpoint control into the live controller layout."""
        rollout_ctrl = np.asarray(prefix_ctrl, dtype=float)
        if rollout_ctrl.shape != (int(self.model.nu),):
            raise ValueError(
                "selected GPU prefix ctrl has shape "
                f"{rollout_ctrl.shape}, expected {(int(self.model.nu),)}"
            )
        world_ctrl = np.asarray(start_control, dtype=float).copy()
        if self.ctrl_src:
            world_ctrl[np.asarray(self.ctrl_src, dtype=int)] = rollout_ctrl[
                np.asarray(self.ctrl_dst, dtype=int)
            ]
        return world_ctrl

    def _rollout_executable(
        self,
        key: tuple[int, int, int, bool, int, bool, int],
        make_jitted: Any,
        args: tuple[Any, ...],
    ) -> Any:
        """Return an executable per static rollout/trace specialization.

        ``make_jitted`` is deliberately lazy. ``run`` still constructs ordinary
        Python function objects while preparing a call, but JAX retains one
        function per full static specialization and identical keys reuse the
        exact compiled executable.  Pool size is part of that specialization:
        the contact reducer closes over it to create an ``(N, features)`` arena
        reduction, so a JIT handle traced for one N is not safe for another.
        """
        executable = self._compiled_rollouts.get(key)
        if executable is not None:
            return executable

        jitted = self._rollout_functions.get(key)
        if jitted is None:
            jitted = make_jitted()
            self._rollout_functions[key] = jitted

        compile_t0 = time.perf_counter()
        executable = jitted.lower(*args).compile()
        self._compiled_rollouts[key] = executable
        logger.info(
            "[jointspace-screen] compiled N=%d H=%d naconmax=%d continued=%s "
            "prefix_steps=%d in %.2fs",
            key[0], key[1], key[2], key[3], key[6],
            time.perf_counter() - compile_t0)
        return executable

    def _execute_rollout(
        self,
        key: tuple[int, int, int, bool, int, bool, int],
        make_jitted: Any,
        args: tuple[Any, ...],
    ) -> Any:
        """Execute a cached specialization with this call's dynamic arguments."""
        return self._rollout_executable(key, make_jitted, args)(*args)

    @staticmethod
    def _pytree_shape_layout(args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Hashable tree/shape/dtype signature with no dynamic values."""
        import jax

        leaves, tree = jax.tree_util.tree_flatten(args)
        return (
            str(tree),
            tuple(
                (
                    tuple(getattr(leaf, "shape", ())),
                    str(getattr(leaf, "dtype", type(leaf).__name__)),
                )
                for leaf in leaves
            ),
        )

    def _device_score_executable(
        self,
        *,
        spec: Any,
        movable_names: tuple[str, ...],
        make_jitted: Any,
        args: tuple[Any, ...],
    ) -> Any:
        """Cache one scorer executable per static program and array layout."""
        static_key = (spec, movable_names)
        key = static_key + (self._pytree_shape_layout(args),)
        executable = self._compiled_device_scores.get(key)
        if executable is not None:
            return executable

        jitted = self._device_score_functions.get(static_key)
        if jitted is None:
            jitted = make_jitted()
            self._device_score_functions[static_key] = jitted
        executable = jitted.lower(*args).compile()
        self._compiled_device_scores[key] = executable
        return executable

    def run_predictive_sampling(
        self,
        *,
        nominal_knots: np.ndarray,
        ctrl_low: np.ndarray,
        ctrl_high: np.ndarray,
        pool_size: int,
        seed: int,
        obj_pose7: np.ndarray,
        n_steps: int = DEFAULT_EXECUTION_STEPS,
        start_state: "tuple[np.ndarray, np.ndarray] | None" = None,
        execution_prefix_frac: float | None = None,
        device_cost_fn: Any,
        device_validity: dict[str, Any],
        sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
        sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
        local_sigma_fraction: float = (
            PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
        ),
        contact_target_bodies: tuple[str, ...] = (),
        mediation_target_bodies: tuple[str, ...] = (),
        controlled_subject_contact_evidence_name: str | None = None,
        record_step_trace: bool = False,
        arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
        record_candidate_cost_telemetry: bool = False,
    ) -> PredictiveSamplingResult:
        """Run one standard GPU predictive-sampling round.

        Candidate zero is the nominal and candidates ``1..N-1`` independently
        perturb all eight future control dimensions.  The default is one
        diagonal Gaussian; ``two_scale`` is one explicit same-round ablation.

        Every candidate's final knot column is a sampled latent. It is
        interpolated and thresholded per control step to the fixed physical
        open/closed targets, independent of stage semantics or held state.
        """
        import jax

        gpu_devices = [
            device for device in jax.devices() if device.platform == "gpu"
        ]
        if not gpu_devices:
            raise RuntimeError(
                "predictive sampling production requires a JAX GPU"
            )
        nominal = np.asarray(nominal_knots, dtype=float)
        lo = np.asarray(ctrl_low, dtype=float)
        hi = np.asarray(ctrl_high, dtype=float)
        if (
            nominal.ndim != 2
            or nominal.shape[0] < 2
            or nominal.shape[1] != NJ + 1
        ):
            raise ValueError(
                f"nominal_knots must be (K, {NJ + 1}) with K >= 2, "
                f"got {nominal.shape}"
            )
        if lo.shape != (NJ + 1,) or hi.shape != (NJ + 1,):
            raise ValueError(
                f"control bounds must both have shape ({NJ + 1},)"
            )
        if (
            not np.all(np.isfinite(nominal))
            or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi))
            or np.any(hi <= lo)
        ):
            raise ValueError(
                "nominal knots and ordered control bounds must be finite"
            )
        if np.any(nominal < lo) or np.any(nominal > hi):
            raise ValueError("nominal_knots must lie inside actuator bounds")
        mode = validate_predictive_sampler_mode(sampler_mode)
        sigma = validate_local_sigma_fraction(sigma_fraction)
        local_sigma = validate_local_sigma_fraction(local_sigma_fraction)
        if (
            mode == PREDICTIVE_SAMPLER_TWO_SCALE
            and local_sigma >= sigma
        ):
            raise ValueError(
                "two_scale requires local_sigma_fraction < sigma_fraction "
                f"(broad), got {local_sigma} >= {sigma}"
            )
        with jax.default_device(gpu_devices[0]):
            result = self.run(
                knot_arrays=None,
                obj_pose7=obj_pose7,
                n_steps=n_steps,
                spline_order="linear",
                start_state=start_state,
                execution_prefix_frac=execution_prefix_frac,
                device_cost_fn=device_cost_fn,
                device_validity=device_validity,
                contact_target_bodies=contact_target_bodies,
                mediation_target_bodies=mediation_target_bodies,
                controlled_subject_contact_evidence_name=(
                    controlled_subject_contact_evidence_name
                ),
                record_step_trace=record_step_trace,
                arm_velocity_weight=arm_velocity_weight,
                record_candidate_cost_telemetry=(
                    record_candidate_cost_telemetry
                ),
                _predictive_sampling_proposal=(
                    nominal,
                    lo,
                    hi,
                    int(pool_size),
                    int(seed),
                    sigma,
                    mode,
                    local_sigma,
                ),
            )
        if not isinstance(result, PredictiveSamplingResult):
            raise RuntimeError(
                "predictive sampler returned a host rollout batch"
            )
        return result

    def run_cem_sampling(
        self,
        *,
        nominal_knots: np.ndarray,
        ctrl_low: np.ndarray,
        ctrl_high: np.ndarray,
        total_rollouts: int,
        rounds: int,
        elite_fraction: float,
        min_std_fraction: float,
        seed: int,
        obj_pose7: np.ndarray,
        n_steps: int = DEFAULT_EXECUTION_STEPS,
        start_state: "tuple[np.ndarray, np.ndarray] | None" = None,
        execution_prefix_frac: float | None = None,
        device_cost_fn: Any,
        device_validity: dict[str, Any],
        sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
        gripper_latent_std: float = DEFAULT_GRIPPER_LATENT_STD,
        contact_target_bodies: tuple[str, ...] = (),
        mediation_target_bodies: tuple[str, ...] = (),
        controlled_subject_contact_evidence_name: str | None = None,
        jaw_bernoulli: bool = False,
        jaw_p_floor: float = DEFAULT_JAW_P_FLOOR,
        record_step_trace: bool = False,
        arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
        record_candidate_cost_telemetry: bool = False,
    ) -> PredictiveSamplingResult:
        """Run iterative diagonal-Gaussian CEM with a fixed rollout budget.

        ``jaw_bernoulli`` switches the binary jaw column to a Bernoulli
        proposal and refit; see :func:`cem_sampling_proposals`.
        """
        import jax

        gpu_devices = [
            device for device in jax.devices() if device.platform == "gpu"
        ]
        if not gpu_devices:
            raise RuntimeError("CEM production requires a JAX GPU")
        n_rounds = validate_cem_rounds(rounds)
        elite_fraction = validate_cem_elite_fraction(elite_fraction)
        min_std_fraction = validate_local_sigma_fraction(min_std_fraction)
        sigma_fraction = validate_local_sigma_fraction(sigma_fraction)
        budget = int(total_rollouts)
        if budget < 2 * n_rounds or budget % n_rounds != 0:
            raise ValueError(
                "total_rollouts must be divisible by cem_rounds with at least "
                "two candidates per round"
            )
        samples_per_round = budget // n_rounds
        mean = np.asarray(nominal_knots, dtype=float)
        lo = np.asarray(ctrl_low, dtype=float)
        hi = np.asarray(ctrl_high, dtype=float)
        if (
            mean.ndim != 2
            or mean.shape[0] < 2
            or mean.shape[1] != NJ + 1
            or lo.shape != (NJ + 1,)
            or hi.shape != (NJ + 1,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi))
            or np.any(hi <= lo)
        ):
            raise ValueError("CEM mean and ordered action bounds must be finite")
        if np.any(mean < lo) or np.any(mean > hi):
            raise ValueError("nominal_knots must lie inside action bounds")
        jaw_std = float(gripper_latent_std)
        if not np.isfinite(jaw_std) or jaw_std <= 0.0:
            raise ValueError("gripper_latent_std must be finite and positive")

        half_range = 0.5 * (hi - lo)
        std = np.broadcast_to(
            sigma_fraction * half_range,
            mean.shape,
        ).copy()
        std[:, NJ] = jaw_std
        if jaw_bernoulli:
            # The nominal's jaw is a hard command, so p starts at exactly 0 or
            # 1. Sampling from that draws one jaw value for every candidate in
            # round one -- no exploration at all -- so the same floor that
            # keeps the refit off the boundary is applied to the start.
            if not 0.0 < float(jaw_p_floor) < 0.5:
                raise ValueError("jaw_p_floor must lie in (0, 0.5)")
            start_p = np.clip(
                0.5 * (mean[1:, NJ] + 1.0),
                float(jaw_p_floor),
                1.0 - float(jaw_p_floor),
            )
            mean[1:, NJ] = 2.0 * start_p - 1.0
            std[:, NJ] = 0.0
        std[0] = 0.0
        min_std = np.broadcast_to(
            min_std_fraction * (hi - lo),
            mean.shape,
        ).copy()
        min_std[0] = 0.0
        elite_count = max(
            2,
            min(
                samples_per_round,
                int(math.ceil(elite_fraction * samples_per_round)),
            ),
        )
        summaries: list[dict[str, Any]] = []
        final: PredictiveSamplingResult | None = None
        # The best VALID candidate seen in ANY round, by production cost.
        # Returning only the last round's winner discards a better incumbent:
        # measured 2026-08-07, the final round was worse than the best earlier
        # round in 12 of 20 Stage-1 solves and 5 of 8 Stage-2 solves, once by
        # nearly 3x (3.30 -> 2.42 -> 3.23 -> 6.84). Keeping an incumbent is
        # standard in this family and entirely task-independent: the
        # distribution still refits from each round's elites exactly as
        # before; only the RETURNED trajectory becomes the best of the budget.
        incumbent: PredictiveSamplingResult | None = None
        incumbent_cost = float("inf")
        incumbent_round = 0
        for round_index in range(n_rounds):
            round_seed = int(
                (int(seed) + 104729 * round_index)
                % np.iinfo(np.int32).max
            )
            with jax.default_device(gpu_devices[0]):
                result = self.run(
                    knot_arrays=None,
                    obj_pose7=obj_pose7,
                    n_steps=n_steps,
                    spline_order="linear",
                    start_state=start_state,
                    execution_prefix_frac=execution_prefix_frac,
                    device_cost_fn=device_cost_fn,
                    device_validity=device_validity,
                    contact_target_bodies=contact_target_bodies,
                    mediation_target_bodies=mediation_target_bodies,
                    controlled_subject_contact_evidence_name=(
                        controlled_subject_contact_evidence_name
                    ),
                    record_step_trace=record_step_trace,
                    arm_velocity_weight=arm_velocity_weight,
                    record_candidate_cost_telemetry=(
                        record_candidate_cost_telemetry
                    ),
                    _cem_sampling_proposal=(
                        mean,
                        std,
                        lo,
                        hi,
                        samples_per_round,
                        round_seed,
                        elite_count,
                        min_std,
                        bool(jaw_bernoulli),
                        float(jaw_p_floor),
                    ),
                )
            if not isinstance(result, PredictiveSamplingResult):
                raise RuntimeError("CEM round returned a host rollout batch")
            if result.refit_mean is None or result.refit_std is None:
                raise RuntimeError("CEM round omitted its elite refit")
            mean = np.asarray(result.refit_mean, dtype=float)
            std = np.asarray(result.refit_std, dtype=float)
            round_summary: dict[str, Any] = {
                "round": float(round_index + 1),
                "samples": float(samples_per_round),
                "n_valid": float(result.n_valid),
                "mean_cost": float(result.mean_cost),
                "best_cost": float(
                    result.selected_row.get("device_program_cost", np.inf)
                ),
                # The refit distribution FEEDING the next round (or, for the
                # final round, the selected-round distribution): small K x 8
                # host copies for solve-level diagnostics.
                "refit_mean": mean.tolist(),
                "refit_std": std.tolist(),
                "first_invalid": {
                    name.removeprefix("first_fail_"): int(count)
                    for name, count in result.validity_counts.items()
                    if name.startswith("first_fail_") and int(count) > 0
                },
            }
            if result.diagnostics is not None:
                round_summary["term_valid_p50"] = [
                    term["valid_quantiles"].get("p50")
                    for term in result.diagnostics.get("terms", [])
                ]
                round_summary["latency_s"] = result.diagnostics.get(
                    "latency_s"
                )
                # Which candidates this round was actually able to produce,
                # jointly. This is the production population, not a resample.
                round_summary["joint_feasibility"] = (
                    result.diagnostics.get("joint_feasibility")
                )
            summaries.append(round_summary)
            final = result
            selected_valid = bool(
                result.selected_row.get("device_program_valid", False)
            )
            selected_cost = float(
                result.selected_row.get("device_program_cost", np.inf)
            )
            if (
                selected_valid
                and np.isfinite(selected_cost)
                and selected_cost < incumbent_cost
            ):
                incumbent = result
                incumbent_cost = selected_cost
                incumbent_round = round_index + 1
        if final is None:  # pragma: no cover - rounds validation excludes this.
            raise RuntimeError("CEM produced no rounds")
        # Return the best valid candidate of the whole budget. With no valid
        # candidate in any round there is nothing to keep, so the final round
        # is returned unchanged and the caller still fails closed on n_valid.
        returned = incumbent if incumbent is not None else final
        return replace(
            returned,
            rounds=n_rounds,
            total_rollouts=budget,
            round_summaries=tuple(summaries),
            sampler_mode="cem",
            # Last round's distribution state, carried as a diagnostic. It is
            # NOT the receding-horizon warm start: the caller shifts
            # `best.knots`, which is the incumbent's trajectory whenever an
            # earlier round won, so the next cycle continues from the RETURNED
            # trajectory. No consumer outside this module reads these fields.
            refit_mean=final.refit_mean,
            refit_std=final.refit_std,
            selected_round=(
                incumbent_round if incumbent is not None else n_rounds
            ),
        )

    def run_mppi_sampling(
        self,
        *,
        nominal_knots: np.ndarray,
        ctrl_low: np.ndarray,
        ctrl_high: np.ndarray,
        total_rollouts: int,
        temperature: float,
        execution_mode: str = MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
        seed: int,
        nominal_actions: np.ndarray | None = None,
        obj_pose7: np.ndarray,
        n_steps: int = DEFAULT_EXECUTION_STEPS,
        start_state: "tuple[np.ndarray, np.ndarray] | None" = None,
        execution_prefix_frac: float | None = None,
        device_cost_fn: Any,
        device_validity: dict[str, Any],
        sigma_fraction: float = PREDICTIVE_SIGMA_FRACTION,
        local_sigma_fraction: float = (
            PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
        ),
        sampler_mode: str = PREDICTIVE_SAMPLER_SINGLE_SCALE,
        mtp_jaw_mode: str = MTP_JAW_MODE_PRESERVE_NOMINAL,
        gripper_latent_std: float = DEFAULT_GRIPPER_LATENT_STD,
        prefer_earliest_joint_terminal_sample: bool = False,
        flip_v7_full_pose_terminal_earliest_authorized: bool = False,
        cup_v8_fine_terminal_earliest_authorized: bool = False,
        contact_target_bodies: tuple[str, ...] = (),
        mediation_target_bodies: tuple[str, ...] = (),
        controlled_subject_contact_evidence_name: str | None = None,
        record_step_trace: bool = False,
        arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
        record_candidate_cost_telemetry: bool = False,
        previous_executed_action: np.ndarray | None = None,
        control_regularizer_boundary_valid: bool | None = None,
        joint_velocity_force_dial_arm: str | None = None,
        unified_mppi_effort_profile: str | None = None,
        pick_v8_causal_profile: str | None = None,
        pick_v8_causal_temperature_state: Any | None = None,
    ) -> PredictiveSamplingResult:
        """Run one paper-style MPPI update and validate its returned mean.

        The budget is split into ``total_rollouts - 1`` optimization samples
        and one exact rollout of the softmax-weighted control.  This prevents
        execution from borrowing validity or trace evidence from a different
        sampled candidate.  Optimization uses fixed-variance Gaussian Halton
        degree-one splines and reports the effective-sample normalization used
        to adapt temperature on the next receding-horizon cycle.
        """
        import jax

        devices = [device for device in jax.devices() if device.platform == "gpu"]
        if not devices:
            raise RuntimeError("MPPI production requires a JAX GPU")
        budget = int(total_rollouts)
        if budget < 3:
            raise ValueError("MPPI total_rollouts must be >= 3")
        tau = validate_mppi_temperature(temperature)
        execution_policy = validate_mppi_execution_mode(execution_mode)
        proposal_mode = validate_predictive_sampler_mode(sampler_mode)
        from oap.twin.unified_mppi_effort import (
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8,
            UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7,
            require_unified_mppi_effort_device_contract,
            unified_mppi_effort_device_values,
            unified_mppi_effort_identity,
            unified_mppi_effort_uses_proven_pick_carrier,
            validate_unified_mppi_effort_profile_arg,
        )
        unified_profile = validate_unified_mppi_effort_profile_arg(
            unified_mppi_effort_profile,
            allow_none=True,
        )
        if (
            flip_v7_full_pose_terminal_earliest_authorized
            and unified_profile
            != UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_FLIP_FULL_POSE_EARLIEST_V7
        ):
            raise ValueError(
                "Flip-v7 full-pose terminal-earliest authorization requires "
                "its exact unified profile"
            )
        if (
            cup_v8_fine_terminal_earliest_authorized
            and unified_profile
            != UNIFIED_MPPI_EFFORT_UNIFORM20_N2001_CUP_FINE_EARLIEST_V8
        ):
            raise ValueError(
                "Cup-v8 fine terminal-earliest authorization requires its "
                "exact unified profile"
            )
        from oap.twin.pick_v8_causal import (
            pick_v8_causal_device_prefix_fractions,
            pick_v8_causal_uses_width_jaw,
            require_pick_v8_causal_mppi_temperature,
            validate_pick_v8_causal_profile,
        )
        causal_profile = validate_pick_v8_causal_profile(
            pick_v8_causal_profile,
            allow_none=True,
        )
        if unified_profile is not None and causal_profile is not None:
            raise ValueError("Pick-v8 causal and unified profiles conflict")
        proven_unified = (
            unified_profile is not None
            and unified_mppi_effort_uses_proven_pick_carrier(unified_profile)
        )
        if (
            causal_profile is None
            and not proven_unified
            and pick_v8_causal_temperature_state is not None
        ):
            raise ValueError(
                "Pick-v8 causal adaptive temperature state requires its profile"
            )
        dial_arm = validate_joint_velocity_force_dial_arm(
            joint_velocity_force_dial_arm,
            allow_none=True,
        )
        broad_sigma = validate_local_sigma_fraction(sigma_fraction)
        local_sigma = validate_local_sigma_fraction(local_sigma_fraction)
        if proposal_mode == PREDICTIVE_SAMPLER_TWO_SCALE:
            raise ValueError("two_scale is not an MPPI proposal mode")
        jaw_policy = validate_mtp_jaw_mode(mtp_jaw_mode)
        if (
            prefer_earliest_joint_terminal_sample
            and dial_arm is None
            and proposal_mode not in (
                *PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
                PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
            )
        ):
            raise ValueError(
                "earliest terminal pose-sample selection requires MPPI"
            )
        if (
            proposal_mode != PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
            and jaw_policy != MTP_JAW_MODE_PRESERVE_NOMINAL
        ):
            raise ValueError("sampled MTP jaw mode requires mtp_global_local")
        experiment_control_profile = bool(
            getattr(self, "experiment_control_profile", False)
        )
        control_profile = validate_control_profile(
            getattr(self, "control_profile", CONTROL_PROFILE_DIRECT_TORQUE_FORCE)
        )
        regularizer_profile = validate_control_regularizer_calibration_profile(
            getattr(self, "control_regularizer_calibration_profile", None),
            allow_none=True,
        )
        if unified_profile is not None:
            required = {
                "control_profile": control_profile,
                "total_rollouts": budget,
                "n_steps": int(n_steps),
                "num_knots": int(np.asarray(nominal_knots).shape[0]),
                "temperature": tau,
                "execution_mode": execution_policy,
                "sampler_mode": proposal_mode,
                "mtp_jaw_mode": jaw_policy,
                "seed": int(seed),
                "arm_velocity_weight": arm_velocity_weight,
                "prefer_earliest": prefer_earliest_joint_terminal_sample,
                "regularizer_profile": regularizer_profile,
                "table_contact_force_weight": self.table_contact_force_weight,
                "dial_arm": dial_arm,
                "execution_prefix_frac": execution_prefix_frac,
            }
            require_unified_mppi_effort_device_contract(
                required,
                profile=unified_profile,
                flip_v7_full_pose_terminal_earliest_authorized=(
                    flip_v7_full_pose_terminal_earliest_authorized
                ),
                cup_v8_fine_terminal_earliest_authorized=(
                    cup_v8_fine_terminal_earliest_authorized
                ),
            )
        if causal_profile is not None or proven_unified:
            require_pick_v8_causal_mppi_temperature(
                current_temperature=tau,
                state=pick_v8_causal_temperature_state,
            )
            expected_control = (
                CONTROL_PROFILE_LEGACY_VELOCITY_WIDTH
                if causal_profile is not None
                and pick_v8_causal_uses_width_jaw(causal_profile)
                else CONTROL_PROFILE_LEGACY_VELOCITY_EFFORT
            )
            proven_device_values = (
                unified_mppi_effort_device_values(unified_profile)
                if proven_unified
                else None
            )
            allowed_prefixes = (
                pick_v8_causal_device_prefix_fractions(causal_profile)
                if causal_profile is not None
                else (proven_device_values["execution_prefix_frac"],)
            )
            required = {
                "control_profile": control_profile,
                "total_rollouts": budget,
                "n_steps": int(n_steps),
                "num_knots": int(np.asarray(nominal_knots).shape[0]),
                "execution_mode": execution_policy,
                "sampler_mode": proposal_mode,
                "seed": int(seed),
                "arm_velocity_weight": arm_velocity_weight,
                "table_contact_force_weight": self.table_contact_force_weight,
            }
            expected = {
                "control_profile": expected_control,
                "total_rollouts": (
                    proven_device_values["total_rollouts"]
                    if proven_unified
                    else 1001
                ),
                "n_steps": (
                    proven_device_values["n_steps"]
                    if proven_unified
                    else 600
                ),
                # Resolve from the registered profile, like every value above
                # that varies across carriers.  A literal here made the
                # registered `k8` ablation unrunnable: it refused at stage 0
                # cycle 0 with zero rollouts, which is a refusal and not a
                # measurement of K=8.
                "num_knots": (
                    proven_device_values["num_knots"]
                    if proven_unified
                    else 12
                ),
                "execution_mode": (
                    proven_device_values["execution_mode"]
                    if proven_unified
                    else MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN
                ),
                "sampler_mode": PREDICTIVE_SAMPLER_SINGLE_SCALE,
                "seed": proven_device_values["seed"] if proven_unified else 0,
                "arm_velocity_weight": 0.1,
                # Resolve from the registered profile, like every other value
                # above that varies across carriers.  The module default only
                # applies when no unified profile is in force.
                "table_contact_force_weight": (
                    proven_device_values["table_contact_force_weight"]
                    if proven_unified
                    else DEFAULT_MPPI_TABLE_FORCE_WEIGHT
                ),
            }
            mismatches = {
                name: {"actual": required[name], "required": value}
                for name, value in expected.items()
                if required[name] != value
            }
            if execution_prefix_frac not in allowed_prefixes:
                mismatches["execution_prefix_frac"] = {
                    "actual": execution_prefix_frac,
                    "required": allowed_prefixes,
                }
            if mismatches:
                raise ValueError(
                    "pick_v8_causal device mismatch: " f"{mismatches!r}"
                )
        terminal_replay_joint_mismatch = False

        def require_formal_regularizer_identity(
            result: PredictiveSamplingResult,
            *,
            phase: str,
        ) -> None:
            require_control_regularizer_result_identity(
                result.diagnostics,
                profile=regularizer_profile,
                phase=phase,
            )

        def exact_replay_protocol_diagnostics(
            valid: bool,
            *,
            accepted_for_execution: bool | None = None,
        ) -> dict[str, Any]:
            # Preserve every pre-existing default/calibration schema except
            # the executable terminal-selector mismatch seam.  A physically
            # valid singleton whose endpoint joint feasibility changed after
            # exact replay is a retryable no-valid-plan refusal, never a hard
            # sampler error.  This semantic belongs to the selector itself;
            # it must not depend on an experiment-only control-profile label.
            if (
                regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
                and not terminal_replay_joint_mismatch
            ):
                return {}
            accepted = (
                bool(valid)
                if accepted_for_execution is None
                else bool(accepted_for_execution)
            )
            packet: dict[str, Any] = {
                "valid": bool(valid),
                "protocol_outcome": (
                    "validated_plan" if accepted else "no_valid_plan"
                ),
                "additional_rollouts": 0,
            }
            if accepted_for_execution is not None:
                packet["accepted_for_execution"] = accepted
            return {"exact_replay": packet}
        if (
            control_profile == CONTROL_PROFILE_JOINT_VELOCITY_FORCE
            and bool(device_validity.get("action_feasibility_enabled", False))
        ):
            raise ValueError(
                "joint_velocity_force diagnostic forbids the position-knot "
                "action-feasibility gate; velocity-action feasibility needs "
                "separate velocity/acceleration bounds"
            )
        if (
            experiment_control_profile and unified_profile is None
            and causal_profile is None
            and arm_velocity_weight != 0.0
        ):
            raise ValueError("control-profile diagnostic requires arm_velocity_weight=0")
        position_nominal = np.asarray(nominal_knots, dtype=float)
        if position_nominal.ndim != 2 or position_nominal.shape[1] != NJ + 1:
            raise ValueError("MPPI nominal must be a (K, 8) knot array")
        horizon_s = float(n_steps) * float(self.exec_dt)
        # The action hold time belongs to this run's horizon/parameterization.
        # Keep it explicit in diagnostics: the public MPPI-ISAAC whole-body
        # configuration uses six actions at 40 ms, whereas our 600-step,
        # 12-action screen holds each effort for 100 ms.
        knot_dt_s = horizon_s / float(position_nominal.shape[0])
        if control_profile_uses_joint_velocity_arm(control_profile):
            mapped_qpos, _, _, _, _ = self._mapped_start_state(start_state)
            measured_arm = np.asarray(mapped_qpos, dtype=float)[
                np.asarray(self.arm_qpos_addr, dtype=int)
            ]
        else:
            # The production direct-effort carrier never depends on measured
            # arm position.  Keep that clean-2ef behavior (and its test-double
            # interface) untouched.
            measured_arm = np.zeros(NJ, dtype=float)
        parameters = mppi_control_profile_parameters(
            control_profile=control_profile,
            position_nominal=position_nominal,
            measured_arm=measured_arm,
            knot_dt_s=knot_dt_s,
            nominal_actions=nominal_actions,
        )
        causal_summary_identity = _pick_v8_causal_mppi_summary_identity(
            pick_v8_causal_profile=causal_profile,
            control_profile=control_profile,
            parameters=parameters,
        )
        if proven_unified:
            causal_summary_identity = {
                "unified_mppi_effort_profile": unified_profile,
                "control_profile": control_profile,
                "proposal": (
                    "gaussian_halton_degree1_arm_velocity_"
                    "jaw_effort_actions"
                ),
                "action_interpolation": "linear_degree_one",
                "arm_action": parameters["arm_action"],
                "arm_action_units": "velocity_rad_s",
                "jaw_action": parameters["jaw_action"],
                "jaw_command_semantics": (
                    "signed_latent_open_minus1_closed_plus1"
                ),
                "jaw_actuator_decode": (
                    "asymmetric_continuous_effort_open_1n_close_80n"
                ),
                "jaw_effort_range_n": [-1.0, 80.0],
            }
        if experiment_control_profile:
            jaw_decode_suffix = (
                "width_m"
                if control_profile_uses_width_jaw(control_profile)
                else "effort_n"
            )
            sampled_jaw_action_label = (
                "sampled_continuous_signed_latent_decoded_to_"
                + jaw_decode_suffix
            )
            nominal_jaw_action_label = (
                "nominal_signed_latent_not_explored_decoded_to_"
                + jaw_decode_suffix
            )
        else:
            # Keep the clean-2ef default telemetry byte-compatible in meaning.
            sampled_jaw_action_label = "sampled_continuous_signed_effort"
            nominal_jaw_action_label = "nominal_effort_not_explored"
        mean = np.asarray(parameters["mean"], dtype=float)
        lo = np.asarray(parameters["lo"], dtype=float)
        hi = np.asarray(parameters["hi"], dtype=float)
        half_range = np.asarray(parameters["half_range"], dtype=float)
        dial_active = dial_arm == DIAL_ANNEALED_R2_EQUAL_BUDGET_V1
        if dial_active:
            prng_impl = str(
                getattr(getattr(jax, "config", None), "jax_default_prng_impl", "")
            )
            if prng_impl != "threefry2x32":
                raise ValueError(
                    "registered DIAL arm requires JAX default PRNG "
                    f"threefry2x32, got {prng_impl!r}"
                )
            required = {
                "control_profile": control_profile,
                "regularizer_profile": regularizer_profile,
                "total_rollouts": budget,
                "n_steps": int(n_steps),
                "num_knots": int(mean.shape[0]),
                "execution_prefix_frac": execution_prefix_frac,
                "proposal_mode": proposal_mode,
                "jaw_policy": jaw_policy,
                "execution_policy": execution_policy,
                "temperature": tau,
                "seed": int(seed),
            }
            expected = {
                "control_profile": CONTROL_PROFILE_JOINT_VELOCITY_FORCE,
                "regularizer_profile": UNIFIED_CONTROL_REGULARIZER_V1,
                "total_rollouts": 1001,
                "n_steps": 600,
                "num_knots": 30,
                "execution_prefix_frac": 50.0 / 600.0,
                "proposal_mode": PREDICTIVE_SAMPLER_SINGLE_SCALE,
                "jaw_policy": MTP_JAW_MODE_PRESERVE_NOMINAL,
                "execution_policy": MPPI_EXECUTION_BEST_VALID_SAMPLE,
                "temperature": 0.1,
                "seed": 1200,
            }
            mismatches = {
                name: {"actual": required[name], "required": value}
                for name, value in expected.items()
                if not (
                    math.isclose(
                        float(required[name]),
                        float(value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    if isinstance(value, float)
                    else required[name] == value
                )
            }
            if mismatches:
                raise ValueError(f"registered DIAL arm mismatch: {mismatches!r}")
        reference_effort_halton = (
            proposal_mode == PREDICTIVE_SAMPLER_REFERENCE_EFFORT_HALTON
        )
        if (
            reference_effort_halton
            and nominal_actions is None
            and control_profile == CONTROL_PROFILE_DIRECT_TORQUE_FORCE
        ):
            # Preserve the clean-2ef reference profile exactly: unlike the
            # current MTP arm, it starts every DoF (including jaw effort) at
            # the published zero-effort mean.
            mean[:, NJ] = 0.0
        if reference_effort_halton:
            # Public omni-Panda effort-pick covariance, normalized by its own
            # actuator bounds before mapping to the Rizon/GN01 safe envelope:
            # four arm axes sqrt(20)/87, three sqrt(8)/12, jaw sqrt(1)/6.
            # This preserves the reference distribution's relative scale; it
            # does not copy Panda torque/force limits onto another embodiment.
            reference_std = np.asarray(
                [math.sqrt(20.0) / 87.0] * 4
                + [math.sqrt(8.0) / 12.0] * 3
                + [1.0 / 6.0],
                dtype=float,
            )
            std = np.broadcast_to(reference_std, mean.shape).copy()
        else:
            normalized_std = (
                local_sigma
                if proposal_mode == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                else math.sqrt(0.1)
            )
            std = np.broadcast_to(
                half_range * normalized_std, mean.shape
            ).copy()
        if unified_profile is not None:
            required_std = half_range * float(
                unified_mppi_effort_identity(unified_profile)[
                    "proposal_std_normalized"
                ]
            )
            if not np.array_equal(std[0], required_std):
                raise ValueError(
                    "unified_mppi_effort_v1 requires exact normalized "
                    "proposal covariance 0.1"
                )
        # The paper reserves one rollout for an explicit null action.  For the
        # arm this is zero additional torque. The jaw holds its current signed
        # effort instead of commanding
        # the meaningless midpoint latent zero.
        null_action = np.zeros(NJ + 1, dtype=float)
        if not reference_effort_halton:
            null_action[NJ] = mean[0, NJ]
        legacy_spline_order = (
            "linear" if causal_profile is not None else "constant"
        )
        common = dict(
            obj_pose7=obj_pose7,
            n_steps=n_steps,
            spline_order=(
                "linear"
                if causal_profile is not None or proven_unified
                else legacy_spline_order
            ),
            pick_v8_causal_profile=causal_profile,
            unified_mppi_effort_profile=unified_profile,
            start_state=start_state,
            execution_prefix_frac=execution_prefix_frac,
            device_cost_fn=device_cost_fn,
            device_validity=device_validity,
            contact_target_bodies=contact_target_bodies,
            mediation_target_bodies=mediation_target_bodies,
            controlled_subject_contact_evidence_name=(
                controlled_subject_contact_evidence_name
            ),
            arm_velocity_weight=arm_velocity_weight,
            record_candidate_cost_telemetry=(
                record_candidate_cost_telemetry
            ),
            previous_executed_action=previous_executed_action,
            control_regularizer_boundary_valid=(
                control_regularizer_boundary_valid
            ),
        )
        dial_round_summaries: tuple[dict[str, Any], ...] | None = None
        dial_round_diagnostics: tuple[dict[str, Any], ...] | None = None
        if dial_active:
            schedule = np.asarray(
                dial_annealed_noise_scale(
                    rounds=2,
                    num_actions=mean.shape[0],
                    trajectory_std_factor=0.5,
                    horizon_std_factor=0.9,
                    array_module=np,
                ),
                dtype=float,
            )
            round_mean = mean.copy()
            dial_summaries: list[dict[str, Any]] = []
            dial_diagnostics: list[dict[str, Any]] = []
            optimization = None
            for round_index in range(2):
                round_seed = int(
                    (int(seed) + 104729 * round_index)
                    % np.iinfo(np.int32).max
                )
                with jax.default_device(devices[0]):
                    round_result = self.run(
                        knot_arrays=None,
                        record_step_trace=False,
                        _dial_annealed_proposal=(
                            round_mean,
                            lo,
                            hi,
                            half_range,
                            schedule[round_index],
                            500,
                            round_seed,
                            tau,
                            bool(
                                prefer_earliest_joint_terminal_sample
                                and round_index == 1
                            ),
                        ),
                        **common,
                    )
                if not isinstance(round_result, PredictiveSamplingResult):
                    raise RuntimeError("DIAL round returned a host rollout batch")
                require_formal_regularizer_identity(
                    round_result,
                    phase=f"DIAL optimization round {round_index + 1}",
                )
                if round_result.refit_mean is None:
                    raise RuntimeError("DIAL round omitted its weighted update")
                if int(round_result.n_valid) <= 0:
                    raise RuntimeError("DIAL round has no base-valid candidates")
                next_mean = np.asarray(round_result.refit_mean, dtype=float)
                selected_proposal = np.asarray(
                    round_result.selected_proposal,
                    dtype=float,
                )
                selected_at_bound = np.isclose(
                    selected_proposal,
                    lo[None, :],
                    rtol=0.0,
                    atol=0.0,
                ) | np.isclose(
                    selected_proposal,
                    hi[None, :],
                    rtol=0.0,
                    atol=0.0,
                )
                terminal_packet = (
                    (round_result.diagnostics or {}).get(
                        "terminal_trajectory_selection"
                    )
                    if round_index == 1 else None
                )
                clipping_packet = (round_result.diagnostics or {}).get(
                    "dial_proposal", {}
                )
                if not isinstance(clipping_packet, dict) or not all(
                    key in clipping_packet
                    for key in (
                        "preclip_out_of_bounds_coordinate_count",
                        "population_coordinate_count",
                    )
                ):
                    raise RuntimeError(
                        "DIAL round omitted clipping telemetry"
                    )
                dial_summaries.append({
                    "round": float(round_index + 1),
                    "samples": 500.0,
                    "gaussian_samples": 499.0,
                    "exact_nominal_samples": 1.0,
                    "n_valid": float(round_result.n_valid),
                    "mean_cost": float(round_result.mean_cost),
                    "best_cost": float(
                        round_result.selected_row.get(
                            "device_program_cost", np.inf
                        )
                    ),
                    "temperature": tau,
                    "next_temperature": tau,
                    "action_dt_s": knot_dt_s,
                    "effective_samples": (
                        round_result.mppi_effective_samples
                    ),
                    "trajectory_std_scale": float(0.5**round_index),
                    "horizon_std_first": float(0.9**29),
                    "horizon_std_last": 1.0,
                    "covariance_is_std_squared": True,
                    "update_l2_norm": float(
                        np.linalg.norm(next_mean - round_mean)
                    ),
                    "round_seed": round_seed,
                    "round_seed_policy": "seed_plus_104729_times_round_index",
                    "clipping_policy": "elementwise_optimizer_action_bounds",
                    "preclip_out_of_bounds_coordinate_count": int(
                        clipping_packet.get(
                            "preclip_out_of_bounds_coordinate_count", -1
                        )
                    ),
                    "population_coordinate_count": int(
                        clipping_packet.get("population_coordinate_count", -1)
                    ),
                    "selected_bound_coordinate_count": int(
                        np.count_nonzero(selected_at_bound)
                    ),
                    "selector_applied": round_index == 1,
                    "terminal_hit_telemetry": (
                        terminal_packet
                        if terminal_packet is not None
                        else "not_collected_before_final_round"
                    ),
                    **causal_summary_identity,
                })
                dial_diagnostics.append(dict(round_result.diagnostics or {}))
                round_mean = next_mean
                optimization = round_result
            assert optimization is not None
            dial_round_summaries = tuple(dial_summaries)
            dial_round_diagnostics = tuple(dial_diagnostics)
        else:
            mppi_proposal: tuple[Any, ...] = (
                mean,
                std,
                lo,
                hi,
                budget - 1,
                int(seed),
                tau,
                null_action,
                proposal_mode,
                20,
                0.5,
                3,
                50,
                jaw_policy,
            )
            if prefer_earliest_joint_terminal_sample:
                mppi_proposal = mppi_proposal + (True,)
            with jax.default_device(devices[0]):
                optimization = self.run(
                    knot_arrays=None,
                    record_step_trace=False,
                    _mppi_sampling_proposal=mppi_proposal,
                    **common,
                )
            if not isinstance(optimization, PredictiveSamplingResult):
                raise RuntimeError("MPPI optimization returned a host rollout batch")
            require_formal_regularizer_identity(
                optimization,
                phase="MPPI optimization",
            )
        if optimization.refit_mean is None:
            raise RuntimeError("MPPI optimization omitted its weighted mean")
        unfiltered_mean = np.asarray(optimization.refit_mean, dtype=float)
        final_refit_std = (
            np.asarray(optimization.refit_std, dtype=float)
            if dial_active and optimization.refit_std is not None
            else std
        )
        # The published omni-Panda effort-pick profile inherits
        # MPPIConfig.update_lambda=False.  Preserve its fixed lambda=0.1
        # contract for the reference profile; the other production samplers
        # retain OaP's adaptive effective-sample update.
        # Neither the cited reference profile nor MTP computes an
        # effective-sample temperature update. MTP refits from its best 20
        # valid proposals and the low-level diagnostic slots are intentionally
        # NaN in that branch; propagating that slot turns the registered fixed
        # temperature into NaN and aborts the next MPC cycle. Only ordinary
        # MPPI weighted update owns an adaptive next temperature.
        next_temperature = (
            tau
            if dial_active
            or reference_effort_halton
            or proposal_mode == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
            else optimization.mppi_next_temperature
        )
        trajectory_selection = (optimization.diagnostics or {}).get(
            "terminal_trajectory_selection"
        ) or {}
        endpoint_joint_count = int(
            trajectory_selection.get("endpoint_joint_candidate_count", 0)
        )
        if prefer_earliest_joint_terminal_sample and endpoint_joint_count > 0:
            if optimization.selected_proposal is None:
                raise RuntimeError(
                    "terminal-trajectory winner omitted its exact proposal"
                )
            # Keep the device-selected proposal's native dtype. Casting to
            # Python ``float`` promotes float32 to float64 and changes the
            # exact singleton replay/warm-start carrier bytes.
            selected_actions = np.asarray(optimization.selected_proposal)
            with jax.default_device(devices[0]):
                validation = self.run(
                    knot_arrays=None,
                    record_step_trace=record_step_trace,
                    _mppi_velocity_exact_proposal=(selected_actions, lo, hi),
                    **common,
                )
            if not isinstance(validation, PredictiveSamplingResult):
                raise RuntimeError(
                    "terminal-trajectory exact replay returned a host batch"
                )
            require_formal_regularizer_identity(
                validation,
                phase="terminal-trajectory exact replay",
            )
            replay_joint = (validation.diagnostics or {}).get(
                "joint_feasibility",
                {},
            )
            replay_valid = bool(
                validation.selected_row.get("device_program_valid", False)
            )
            replay_is_joint = bool(
                replay_joint.get("winner_is_joint_feasible", False)
            )
            replay_accepted_for_execution = bool(
                replay_valid and replay_is_joint
            )
            terminal_replay_joint_mismatch = bool(
                replay_valid
                and not replay_is_joint
                and regularizer_profile in (
                    None,
                    UNIFIED_CONTROL_REGULARIZER_V1,
                )
            )
            if (
                not replay_valid
                and regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
            ):
                raise RuntimeError(
                    "terminal-trajectory exact singleton replay is invalid"
                )
            replay_telemetry = {
                **trajectory_selection,
                "exact_replay_valid": replay_valid,
                "exact_replay_endpoint_joint": replay_is_joint,
                "exact_replay_joint_mismatch": not replay_is_joint,
            }
            is_mtp = (
                proposal_mode == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
            )
            return replace(
                validation,
                sampler_mode=(
                    PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    if is_mtp else "mppi"
                ),
                pool_size=budget - 1,
                n_gaussian=(
                    998 if dial_active else 0 if is_mtp else budget - 2
                ),
                n_local_gaussian=(
                    (budget - 1) - 1 - int(round(0.5 * (budget - 1)))
                    if is_mtp else 0
                ),
                n_broad_gaussian=(
                    int(round(0.5 * (budget - 1))) if is_mtp else 0
                ),
                rounds=2 if dial_active else 1,
                total_rollouts=budget,
                round_summaries=(dial_round_summaries if dial_active else ({
                    "round": 1.0,
                    "samples": float(budget - 1),
                    "n_valid": float(optimization.n_valid),
                    "mean_cost": float(optimization.mean_cost),
                    "best_cost": float(
                        optimization.selected_row.get(
                            "device_program_cost",
                            np.inf,
                        )
                    ),
                    "temperature": tau,
                    "proposal": (
                        (
                            "mtp_global_local_linear_m3_n50_arm_velocity_"
                            "jaw_latent"
                            if control_profile_uses_joint_velocity_arm(
                                control_profile
                            )
                            else "mtp_global_local_linear_m3_n50_arm_torque_"
                            "jaw_latent"
                        )
                        if is_mtp
                        else "gaussian_halton_degree1_arm_torque_"
                        "jaw_effort_actions"
                    ),
                    "arm_action": parameters["arm_action"],
                    "arm_action_units": (
                        "velocity_rad_s"
                        if control_profile_uses_joint_velocity_arm(
                            control_profile
                        )
                        else "normalized_torque"
                    ),
                    "jaw_command_semantics": (
                        "signed_latent_open_minus1_closed_plus1"
                    ),
                    "jaw_actuator_decode": (
                        "continuous_width_target_m"
                        if control_profile_uses_width_jaw(control_profile)
                        else "continuous_signed_effort_n"
                    ),
                    "jaw_action": (
                        sampled_jaw_action_label
                        if jaw_policy == MTP_JAW_MODE_SAMPLE
                        else nominal_jaw_action_label
                    ),
                    "action_dt_s": knot_dt_s,
                    "execution_selection": (
                        "endpoint_joint_then_earliest_pose_sample_then_"
                        "cost_then_stable_index_exact_singleton"
                    ),
                    "terminal_trajectory_selection": replay_telemetry,
                    **causal_summary_identity,
                },)),
                # Result selection decides what is executed. The MPPI/MTP
                # distribution update remains the weighted/elite mean, so a
                # diagnostic selector cannot silently replace the optimizer's
                # task-independent receding-horizon memory.
                refit_mean=unfiltered_mean,
                refit_std=final_refit_std,
                selected_round=2 if dial_active else 1,
                mppi_effective_samples=optimization.mppi_effective_samples,
                mppi_next_temperature=next_temperature,
                diagnostics={
                    **(validation.diagnostics or {}),
                    **exact_replay_protocol_diagnostics(
                        replay_valid,
                        accepted_for_execution=(
                            replay_accepted_for_execution
                            if (
                                regularizer_profile
                                == UNIFIED_CONTROL_REGULARIZER_V1
                                or terminal_replay_joint_mismatch
                            )
                            else None
                        ),
                    ),
                    "terminal_trajectory_selection": replay_telemetry,
                    "optimization_population_diagnostics": (
                        optimization.diagnostics
                    ),
                    **(
                        {
                            "dial_annealed": {
                                "schema": "dial_annealed_r2_equal_budget_v1",
                                "rounds": dial_round_diagnostics,
                                "final_selector_round": 2,
                                "total_optimization_rollouts": 1000,
                                "final_exact_replay_rollouts": 1,
                            }
                        }
                        if dial_active else {}
                    ),
                },
                selected_family_override=(
                    "terminal_joint_earliest_pose_sample_exact"
                ),
            )
        if proposal_mode == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL:
            if optimization.selected_proposal is None:
                raise RuntimeError("MTP omitted its best evaluated proposal")
            best_evaluated = np.asarray(optimization.selected_proposal)
            with jax.default_device(devices[0]):
                validation = self.run(
                    knot_arrays=None,
                    record_step_trace=record_step_trace,
                    _mppi_velocity_exact_proposal=(best_evaluated, lo, hi),
                    **common,
                )
            if not isinstance(validation, PredictiveSamplingResult):
                raise RuntimeError("MTP validation returned a host rollout batch")
            require_formal_regularizer_identity(
                validation,
                phase="MTP best-candidate exact replay",
            )
            replay_valid = bool(
                validation.selected_row.get("device_program_valid", False)
            )
            return replace(
                validation,
                sampler_mode=PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
                pool_size=budget - 1,
                n_gaussian=0,
                n_local_gaussian=(
                    (budget - 1) - 1 - int(round(0.5 * (budget - 1)))
                ),
                n_broad_gaussian=int(round(0.5 * (budget - 1))),
                rounds=1,
                total_rollouts=budget,
                round_summaries=({
                    "round": 1.0,
                    "samples": float(budget - 1),
                    "n_valid": float(optimization.n_valid),
                    "mean_cost": float(optimization.mean_cost),
                    "best_cost": float(
                        optimization.selected_row.get(
                            "device_program_cost", np.inf
                        )
                    ),
                    "temperature": tau,
                    "proposal": "mtp_global_local_linear_m3_n50",
                    "global_fraction": 0.5,
                    "global_graph_depth": 3,
                    "global_graph_width": 50,
                    "elite_count": 20,
                    "global_range_fraction": 1.0,
                    "configured_global_sigma_fraction": broad_sigma,
                    "local_std": local_sigma,
                    "jaw_action": (
                        sampled_jaw_action_label
                        if jaw_policy == MTP_JAW_MODE_SAMPLE
                        else nominal_jaw_action_label
                    ),
                    "arm_action": parameters["arm_action"],
                    "arm_action_units": (
                        "velocity_rad_s"
                        if control_profile_uses_joint_velocity_arm(
                            control_profile
                        )
                        else "normalized_torque"
                    ),
                    "jaw_command_semantics": (
                        "signed_latent_open_minus1_closed_plus1"
                    ),
                    "jaw_actuator_decode": (
                        "continuous_width_target_m"
                        if control_profile_uses_width_jaw(control_profile)
                        else "continuous_signed_effort_n"
                    ),
                    "jaw_mode": jaw_policy,
                    "execution_selection": "best_evaluated_valid_candidate",
                    "action_dt_s": knot_dt_s,
                    **causal_summary_identity,
                },),
                refit_mean=unfiltered_mean,
                refit_std=std,
                selected_round=1,
                mppi_effective_samples=None,
                mppi_next_temperature=tau,
                diagnostics={
                    **(validation.diagnostics or {}),
                    **exact_replay_protocol_diagnostics(replay_valid),
                    "optimization_population_diagnostics": (
                        optimization.diagnostics
                    ),
                },
                selected_family_override="mtp_best_evaluated_candidate",
            )
        if execution_policy == MPPI_EXECUTION_BEST_VALID_SAMPLE:
            if (
                int(optimization.n_valid) <= 0
                or optimization.selected_proposal is None
            ):
                raise RuntimeError("MPPI omitted its best valid sampled plan")
            best_valid_actions = np.asarray(optimization.selected_proposal)
            with jax.default_device(devices[0]):
                validation = self.run(
                    knot_arrays=None,
                    record_step_trace=record_step_trace,
                    _mppi_velocity_exact_proposal=(best_valid_actions, lo, hi),
                    **common,
                )
            if not isinstance(validation, PredictiveSamplingResult):
                raise RuntimeError(
                    "MPPI best-sample replay returned a host rollout batch"
                )
            require_formal_regularizer_identity(
                validation,
                phase="reference-effort exact replay",
            )
            replay_valid = bool(
                validation.selected_row.get("device_program_valid", False)
            )
            if (
                not replay_valid
                and regularizer_profile != UNIFIED_CONTROL_REGULARIZER_V1
            ):
                raise RuntimeError(
                    "MPPI best valid sampled plan failed exact singleton replay"
                )
            summary = {
                "round": 1.0,
                "samples": float(budget - 1),
                "n_valid": float(optimization.n_valid),
                "mean_cost": float(optimization.mean_cost),
                "best_cost": float(
                    optimization.selected_row.get(
                        "device_program_cost", np.inf
                    )
                ),
                "temperature": tau,
                "effective_samples": optimization.mppi_effective_samples,
                "next_temperature": next_temperature,
                "fixed_std": std.tolist(),
                "proposal": (
                    "gaussian_halton_degree1_arm_torque_jaw_effort_actions"
                ),
                "halton_panel_reused_across_mpc_cycles": True,
                "sample_null_action": True,
                "sample_previous_plan": True,
                "control_filter": "none_reference_whole_body_config",
                "variance_update": False,
                "arm_action": "normalized_gravity_compensated_joint_torque",
                "jaw_action": "continuous_signed_effort",
                **(
                    {
                        "jaw_effort_range_n": [
                            -GN01_DIRECT_FORCE_LIMIT_N,
                            GN01_DIRECT_FORCE_LIMIT_N,
                        ]
                    }
                    if causal_profile is None
                    else {}
                ),
                "action_std_normalized": std[0].tolist(),
                "action_dt_s": knot_dt_s,
                "execution_selection": "best_valid_sample_exact_singleton",
                "best_sample_replay_valid": bool(
                    replay_valid
                ),
                **causal_summary_identity,
            }
            return replace(
                validation,
                sampler_mode="mppi",
                pool_size=budget - 1,
                n_gaussian=998 if dial_active else budget - 2,
                rounds=2 if dial_active else 1,
                total_rollouts=budget,
                round_summaries=(
                    dial_round_summaries if dial_active else (summary,)
                ),
                refit_mean=unfiltered_mean,
                refit_std=final_refit_std,
                selected_round=2 if dial_active else 1,
                mppi_effective_samples=optimization.mppi_effective_samples,
                mppi_next_temperature=next_temperature,
                diagnostics={
                    **(validation.diagnostics or {}),
                    **exact_replay_protocol_diagnostics(replay_valid),
                    "optimization_population_diagnostics": (
                        optimization.diagnostics
                    ),
                    **(
                        {
                            "dial_annealed": {
                                "schema": "dial_annealed_r2_equal_budget_v1",
                                "rounds": dial_round_diagnostics,
                                "final_selector_round": 2,
                                "total_optimization_rollouts": 1000,
                                "final_exact_replay_rollouts": 1,
                            }
                        }
                        if dial_active else {}
                    ),
                },
                selected_proposal=best_valid_actions,
                selected_family_override="best_valid_sample_exact",
            )
        weighted_mean = (
            unfiltered_mean
            if reference_effort_halton
            else paper_mppi_filter_action_plan(
                unfiltered_mean,
                lo,
                hi,
                # The Savitzky--Golay window is measured in knots, so it cannot
                # exceed the knot count.  At the production K=12 this is 9,
                # unchanged; it only shrinks below K=9, where a fixed 9 raises
                # instead of filtering.  A consequence worth stating plainly:
                # the K ablation cannot move K alone -- the smoothing support
                # moves with it.
                window_length=_knot_filter_window(unfiltered_mean.shape[0]),
                polynomial_order=2,
            )
        )
        with jax.default_device(devices[0]):
            validation = self.run(
                knot_arrays=None,
                record_step_trace=record_step_trace,
                _mppi_velocity_exact_proposal=(weighted_mean, lo, hi),
                **common,
            )
        if not isinstance(validation, PredictiveSamplingResult):
            raise RuntimeError("MPPI validation returned a host rollout batch")
        require_formal_regularizer_identity(
            validation,
            phase="MPPI mean exact replay",
        )
        selected_action_plan = weighted_mean
        selection_family = "softmax_weighted_mean"
        weighted_mean_valid = bool(
            validation.selected_row.get("device_program_valid", False)
        )
        best_sample_endpoint_held = bool(
            optimization.selected_row.get("final_two_sided_now", False)
        )
        weighted_mean_endpoint_held = bool(
            validation.selected_row.get("final_two_sided_now", False)
        )
        best_sample_cost = float(
            optimization.selected_row.get("device_program_cost", np.inf)
        )
        weighted_mean_cost = float(
            validation.selected_row.get("device_program_cost", np.inf)
        )
        mean_loses_endpoint_hold = (
            best_sample_endpoint_held and not weighted_mean_endpoint_held
        )
        # Retain the pilot's cost-regression diagnostic. Execution and refit
        # use the exactly validated mean without an unused best-sample replay.
        # The additive guard avoids ratios around a near-zero cost.
        mean_cost_regressed = (
            weighted_mean_valid
            and optimization.n_valid > 0
            and math.isfinite(best_sample_cost)
            and math.isfinite(weighted_mean_cost)
            and weighted_mean_cost
            > max(2.0 * best_sample_cost, best_sample_cost + 1.0)
        )
        fallback_used = False
        summary = {
            "round": 1.0,
            "samples": float(budget - 1),
            "n_valid": float(optimization.n_valid),
            "mean_cost": float(optimization.mean_cost),
            "best_cost": float(
                optimization.selected_row.get("device_program_cost", np.inf)
            ),
            "temperature": tau,
            "effective_samples": optimization.mppi_effective_samples,
            "next_temperature": next_temperature,
            "fixed_std": std.tolist(),
            "proposal": "gaussian_halton_degree1_arm_torque_jaw_effort_actions",
            "halton_panel_reused_across_mpc_cycles": True,
            "sample_null_action": True,
            "sample_previous_plan": True,
            # Report the window that ran, not a literal.  The window is
            # derived from the knot count, so a hard-coded "window9" made the
            # k8 artifact record window 9 in all 33 cycles beside its own
            # num_knots: 8, while the filter had actually used window 7.
            "control_filter": (
                "none_reference_whole_body_config"
                if reference_effort_halton
                else (
                    "savitzky_golay_window"
                    f"{_knot_filter_window(int(unfiltered_mean.shape[0]))}"
                    "_order2_interp"
                )
            ),
            "variance_update": False,
            "arm_action": "normalized_gravity_compensated_joint_torque",
            "jaw_action": "continuous_signed_effort",
            **(
                {
                    "jaw_effort_range_n": [
                        -GN01_DIRECT_FORCE_LIMIT_N,
                        GN01_DIRECT_FORCE_LIMIT_N,
                    ]
                }
                if causal_profile is None
                else {}
            ),
            "action_std_normalized": std[0].tolist(),
            "action_dt_s": knot_dt_s,
            "weighted_mean_valid": weighted_mean_valid,
            "optimization_best_endpoint_held": best_sample_endpoint_held,
            "weighted_mean_endpoint_held": weighted_mean_endpoint_held,
            "weighted_mean_lost_endpoint_hold": mean_loses_endpoint_hold,
            "weighted_mean_cost": weighted_mean_cost,
            "optimization_best_valid_cost": best_sample_cost,
            "weighted_mean_cost_regressed": mean_cost_regressed,
            "best_valid_fallback_used": fallback_used,
            "unfiltered_action_total_variation": float(
                np.sum(np.abs(np.diff(unfiltered_mean, axis=0)))
            ),
            "filtered_action_total_variation": float(
                np.sum(np.abs(np.diff(weighted_mean, axis=0)))
            ),
            **causal_summary_identity,
        }
        if prefer_earliest_joint_terminal_sample:
            summary["terminal_trajectory_selection"] = trajectory_selection
        return replace(
            validation,
            sampler_mode="mppi",
            pool_size=budget - 1,
            n_gaussian=budget - 2,
            rounds=1,
            total_rollouts=budget,
            round_summaries=(summary,),
            refit_mean=selected_action_plan,
            refit_std=std,
            selected_round=1,
            mppi_effective_samples=optimization.mppi_effective_samples,
            mppi_next_temperature=next_temperature,
            diagnostics={
                **(validation.diagnostics or {}),
                **(
                    {"terminal_trajectory_selection": trajectory_selection}
                    if prefer_earliest_joint_terminal_sample
                    else {}
                ),
                "optimization_population_diagnostics": (
                    optimization.diagnostics
                ),
            },
            selected_family_override=selection_family,
        )

    def run(self, *, knot_arrays: list[np.ndarray] | None,
            obj_pose7: np.ndarray,
            n_steps: int = DEFAULT_EXECUTION_STEPS,
            spline_order: str = "linear",
            pick_v8_causal_profile: str | None = None,
            unified_mppi_effort_profile: str | None = None,
            start_state: "tuple[np.ndarray, np.ndarray] | None" = None,
            execution_prefix_frac: float | None = None,
            device_cost_fn: Any | None = None,
            device_validity: dict[str, Any] | None = None,
            contact_target_bodies: tuple[str, ...] = (),
            mediation_target_bodies: tuple[str, ...] = (),
            controlled_subject_contact_evidence_name: str | None = None,
            record_step_trace: bool = False,
            arm_velocity_weight: float = DEFAULT_ARM_VELOCITY_WEIGHT,
            record_candidate_cost_telemetry: bool = False,
            previous_executed_action: np.ndarray | None = None,
            control_regularizer_boundary_valid: bool | None = None,
            _predictive_sampling_proposal: (
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    int,
                    float,
                    str,
                    float,
                ] | None
            ) = None,
            _cem_sampling_proposal: (
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    int,
                    int,
                    np.ndarray,
                ] | None
            ) = None,
            _mppi_sampling_proposal: tuple[Any, ...] | None = None,
            _dial_annealed_proposal: tuple[Any, ...] | None = None,
            _exact_sampling_proposal: (
                tuple[np.ndarray, np.ndarray, np.ndarray] | None
            ) = None,
            _mppi_velocity_exact_proposal: (
                tuple[np.ndarray, np.ndarray, np.ndarray] | None
            ) = None,
            ) -> (
                dict[int, dict[str, Any]]
                | PredictiveSamplingResult
            ):
        """Roll every candidate joint-knot array; return per-candidate rows.

        ``knot_arrays[i]`` is a (K, 8) joint-knot array; the returned dict is
        keyed by the candidate INDEX i (the caller maps indices to ids). Rows
        carry the same keys the floating-gripper screen emits.

        Production supplies ``device_cost_fn`` plus the task-independent
        ``device_validity`` constants. Program cost and generic physical
        feasibility are then evaluated on the accelerator while the rollout
        trajectories are still device arrays; the selected compact row and,
        for CEM, the refitted ``K x 8`` mean/std cross to the host. The selected
        row includes prefix qpos/qvel and compact prefix contact evidence and is
        therefore also the
        dry-run plant transition; it is never replayed by CPU MuJoCo. A missing
        or failed device scorer is never replaced by the CPU interpreter.
        ``contact_target_bodies`` is resolved structurally from typed
        subject-contact relations in the current Stage. Its geoms are removed
        only from the subject/world-obstacle channel.
        ``mediation_target_bodies`` is the narrower set named by explicit
        ``ToolMediation`` running costs and alone supplies the aggregate
        subject-target and robot-target channels consumed by typed semantics.
        No other support is exempt.
        ``controlled_subject_contact_evidence_name`` adds a semantic key only
        to per-body contact evidence. It cannot change aggregate contact
        channels, cost, validity, candidate selection, or Stage progression.
        Per-body ``prefix_subject_target_now_by_body`` and
        ``final_subject_target_now_by_body`` are direct native-step Booleans at
        E-1 and H-1 respectively. They cross the public/compact boundary only
        when the device evaluator declares terminal Contact and exactly one
        physical contact target is bound; running-only Contact retains its legacy
        row schema. Neither Boolean is reconstructed from frame counts,
        penetration, or an any-step reduction.
        """
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        arm_velocity_weight = validate_arm_velocity_weight(
            arm_velocity_weight
        )
        regularizer_profile = validate_control_regularizer_calibration_profile(
            getattr(self, "control_regularizer_calibration_profile", None),
            allow_none=True,
        )
        regularizer_active = regularizer_profile is not None
        regularizer_score_active = (
            regularizer_profile == UNIFIED_CONTROL_REGULARIZER_V1
        )
        from oap.twin.pick_v8_causal import (
            validate_pick_v8_causal_profile,
        )
        causal_profile = validate_pick_v8_causal_profile(
            pick_v8_causal_profile,
            allow_none=True,
        )
        if regularizer_active:
            previous_action = np.asarray(
                previous_executed_action,
                dtype=float,
            )
            if previous_action.shape != (NJ + 1,) or not np.all(
                np.isfinite(previous_action)
            ):
                raise ValueError(
                    "active control regularizer requires finite physical 8D "
                    "previous_executed_action"
                )
            if not isinstance(control_regularizer_boundary_valid, bool):
                raise ValueError(
                    "active control regularizer requires typed boundary validity"
                )
        else:
            previous_action = np.zeros(NJ + 1, dtype=float)
            control_regularizer_boundary_valid = False

        device_select = (
            _predictive_sampling_proposal is not None
            or _cem_sampling_proposal is not None
            or _mppi_sampling_proposal is not None
            or _dial_annealed_proposal is not None
            or _exact_sampling_proposal is not None
            or _mppi_velocity_exact_proposal is not None
        )
        if record_step_trace and not device_select:
            raise ValueError(
                "GPU winner step tracing requires device-side candidate selection"
            )
        proposal_mode_count = sum(
            proposal is not None
            for proposal in (
                _predictive_sampling_proposal,
                _cem_sampling_proposal,
                _mppi_sampling_proposal,
                _dial_annealed_proposal,
                _exact_sampling_proposal,
                _mppi_velocity_exact_proposal,
            )
        )
        if proposal_mode_count > 1:
            raise ValueError(
                "device proposal modes are mutually exclusive"
            )
        if device_select and knot_arrays is not None:
            raise ValueError(
                "device proposal mode and host knot_arrays are mutually exclusive"
            )
        if not device_select and knot_arrays is None:
            raise ValueError("knot_arrays are required outside device proposal mode")
        proposal_knots = None
        n_gaussian = 0
        proposal_sampler_mode = PREDICTIVE_SAMPLER_SINGLE_SCALE
        proposal_local_sigma_fraction = (
            PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION
        )
        proposal_broad_sigma_fraction = PREDICTIVE_SIGMA_FRACTION
        n_local_gaussian = 0
        n_broad_gaussian = 0
        cem_previous_mean = None
        cem_previous_std = None
        cem_elite_count = 0
        cem_min_std = None
        cem_jaw_bernoulli = False
        cem_jaw_p_floor = DEFAULT_JAW_P_FLOOR
        mppi_temperature = DEFAULT_MPPI_TEMPERATURE
        mppi_null_action: np.ndarray | None = None
        mppi_proposal_mode = PREDICTIVE_SAMPLER_SINGLE_SCALE
        mtp_elite_count = 20
        mtp_beta = 0.5
        mtp_graph_depth = 3
        mtp_graph_width = 50
        mtp_jaw_mode = MTP_JAW_MODE_PRESERVE_NOMINAL
        mppi_prefer_earliest_joint_terminal_sample = False
        dial_half_span = None
        dial_noise_scale = None
        dial_clipping_count_device = None
        if device_select:
            if _cem_sampling_proposal is not None:
                (
                    nominal_knots,
                    cem_previous_std,
                    ctrl_low,
                    ctrl_high,
                    N,
                    proposal_seed,
                    cem_elite_count,
                    cem_min_std,
                    cem_jaw_bernoulli,
                    cem_jaw_p_floor,
                ) = _cem_sampling_proposal
                cem_previous_mean = nominal_knots
                proposal_sampler_mode = "cem"
            elif _mppi_sampling_proposal is not None:
                base_mppi = _mppi_sampling_proposal[:8]
                (
                    nominal_knots,
                    cem_previous_std,
                    ctrl_low,
                    ctrl_high,
                    N,
                    proposal_seed,
                    mppi_temperature,
                    mppi_null_action,
                ) = base_mppi
                if len(_mppi_sampling_proposal) not in (
                    8, 13, 14, 15
                ):
                    raise ValueError(
                        "MPPI proposal tuple must have 8, 13, 14, or 15 fields"
                    )
                if len(_mppi_sampling_proposal) >= 13:
                    (
                        mppi_proposal_mode,
                        mtp_elite_count,
                        mtp_beta,
                        mtp_graph_depth,
                        mtp_graph_width,
                    ) = _mppi_sampling_proposal[8:13]
                if len(_mppi_sampling_proposal) >= 14:
                    mtp_jaw_mode = validate_mtp_jaw_mode(
                        _mppi_sampling_proposal[13]
                    )
                if len(_mppi_sampling_proposal) >= 15:
                    mppi_prefer_earliest_joint_terminal_sample = bool(
                        _mppi_sampling_proposal[14]
                    )
                cem_previous_mean = nominal_knots
                proposal_sampler_mode = (
                    PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    if mppi_proposal_mode
                    == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    else "mppi"
                )
                if (
                    mppi_prefer_earliest_joint_terminal_sample
                    and mppi_proposal_mode
                    not in (
                        *PREDICTIVE_SAMPLER_MPPI_HALTON_MODES,
                        PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL,
                    )
                ):
                    raise ValueError(
                        "earliest terminal pose-sample selection requires MPPI"
                    )
            elif _dial_annealed_proposal is not None:
                if len(_dial_annealed_proposal) != 9:
                    raise ValueError("DIAL proposal tuple must have 9 fields")
                (
                    nominal_knots,
                    ctrl_low,
                    ctrl_high,
                    dial_half_span,
                    dial_noise_scale,
                    N,
                    proposal_seed,
                    mppi_temperature,
                    mppi_prefer_earliest_joint_terminal_sample,
                ) = _dial_annealed_proposal
                cem_previous_mean = nominal_knots
                cem_previous_std = (
                    np.asarray(dial_noise_scale, dtype=float)[:, None]
                    * np.asarray(dial_half_span, dtype=float)[None, :]
                )
                proposal_sampler_mode = "dial_annealed"
            elif _mppi_velocity_exact_proposal is not None:
                nominal_knots, ctrl_low, ctrl_high = (
                    _mppi_velocity_exact_proposal
                )
                N = 1
                proposal_seed = 0
                proposal_sampler_mode = "mppi_velocity_exact"
            elif _exact_sampling_proposal is not None:
                nominal_knots, ctrl_low, ctrl_high = _exact_sampling_proposal
                N = 1
                proposal_seed = 0
                proposal_sampler_mode = "exact"
            else:
                assert _predictive_sampling_proposal is not None
                (
                    nominal_knots,
                    ctrl_low,
                    ctrl_high,
                    N,
                    proposal_seed,
                    proposal_broad_sigma_fraction,
                    proposal_sampler_mode,
                    proposal_local_sigma_fraction,
                ) = _predictive_sampling_proposal
            if (
                int(N) < 2
                and _exact_sampling_proposal is None
                and _mppi_velocity_exact_proposal is None
            ):
                raise ValueError("pool_size must be >= 2")
            if (
                _cem_sampling_proposal is None
                and _mppi_sampling_proposal is None
                and _dial_annealed_proposal is None
                and _exact_sampling_proposal is None
                and _mppi_velocity_exact_proposal is None
            ):
                (
                    n_local_gaussian,
                    n_broad_gaussian,
                ) = predictive_sampling_candidate_counts(
                    N,
                    sampler_mode=proposal_sampler_mode,
                )
            knot_count = int(nominal_knots.shape[0])
            paper_effort_request = (
                _mppi_sampling_proposal is not None
                or _dial_annealed_proposal is not None
                or _mppi_velocity_exact_proposal is not None
            )
            if causal_profile is not None and (
                _dial_annealed_proposal is not None
                or not (
                    _mppi_sampling_proposal is not None
                    or _mppi_velocity_exact_proposal is not None
                )
            ):
                raise ValueError(
                    "Pick-v8 causal interpolation is restricted to MPPI "
                    "optimization and its exact replay"
                )
            if paper_effort_request:
                _validate_paper_mppi_expansion_contract(
                    spline_order=spline_order,
                    pick_v8_causal_profile=causal_profile,
                    unified_mppi_effort_profile=unified_mppi_effort_profile,
                    control_profile=self.control_profile,
                )
            elif causal_profile is not None:
                raise ValueError(
                    "Pick-v8 causal interpolation requires paper MPPI"
                )
            elif spline_order != "linear":
                raise ValueError(
                    "production samplers outside paper MPPI require a "
                    "linear spline"
                )
            if device_cost_fn is None or device_validity is None:
                raise ValueError(
                    "device cost and validity are required for production "
                    "predictive sampling"
                )
        else:
            if causal_profile is not None:
                raise ValueError(
                    "Pick-v8 causal interpolation requires device paper MPPI"
                )
            assert knot_arrays is not None
            N = len(knot_arrays)
            if N == 0:
                return {}
            knot_count = int(np.asarray(knot_arrays[0]).shape[0])
        if execution_prefix_frac is None:
            if knot_count < 2:
                raise ValueError("rollout needs at least two control knots")
            execution_prefix_frac = 1.0 / float(knot_count - 1)
        execution_steps = validate_horizon_steps(n_steps)
        # ``n_steps`` is in EXECUTION-grid units (480 x 2 ms = 0.96 s by
        # default) at every call site; rescale to this screen's planning grid so
        # the DURATION is preserved (96 x 10 ms = the same 0.96 s).
        # plan_dt == exec_dt makes this the identity.
        H = execution_steps if self.plan_dt == self.exec_dt else max(
            2, int(round(execution_steps * self.exec_dt / self.plan_dt)))
        # Paper Eq. (5): one running-cost observation per physics step. Keep
        # the existing fixed-shape carry, with its shape bound to this horizon.
        POSE_SAMPLES = H
        if not 0.0 < float(execution_prefix_frac) <= 1.0:
            raise ValueError(
                "execution_prefix_frac must be in (0, 1], got "
                f"{execution_prefix_frac!r}"
            )
        raw_prefix_steps = H * float(execution_prefix_frac)
        nearest_prefix_steps = round(raw_prefix_steps)
        prefix_steps = (
            int(nearest_prefix_steps)
            if math.isclose(
                raw_prefix_steps,
                nearest_prefix_steps,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            else int(np.ceil(raw_prefix_steps))
        )
        prefix_index = max(0, min(H - 1, prefix_steps - 1))
        t0 = time.time()
        model_ctrl = None
        grip_init_np = None
        if not device_select:
            assert knot_arrays is not None
            model_ctrl, grip_init_np = self._knots_to_model_ctrl(
                knot_arrays,
                H,
                spline_order,
            )

        # Arena sizing, current-semantics (mujoco_warp post-ca08923; audited
        # against the official docs 2026-07-23):
        #   * naconmax = max contacts ACROSS ALL WORLDS -> scales with N;
        #   * njmax    = max constraints PER WORLD, STRICT -> a CONSTANT.
        # The old code scaled njmax by N too (pre-change semantics; the
        # maintainers' guidance is "you should no longer multiply by nworld",
        # mujoco_warp#538) -- a 10-50x per-world over-allocation that measured
        # 15% of the N=256 batch wall-clock. 1024 is ~2x headroom over the
        # worst per-world constraint count these scenes produce (peer configs
        # use 40-250; playground go1 njmax=40); an overflow surfaces as a
        # runtime WARNING from warp, and the CPU-parity suite would catch any
        # silently dropped contact as a screen<->certifier divergence.
        # The template is CACHED per naconmax. Rebuilding it every call adds
        # allocator churn and makes CUDA-graph reuse less likely. The backend
        # may still recapture for changing input/output pointers, so H100
        # benchmarks must distinguish JAX executable reuse from graph reuse.
        # Fixed conservative capacity is part of the validated backend
        # contract. It is intentionally not user-tunable: lowering it can
        # silently drop contacts and invalidate safety/mechanism evidence.
        nacon_per_world = 24
        nacon_floor = 2048
        naconmax = max(nacon_floor, nacon_per_world * N)
        logger.info(
            "[jointspace-screen] naconmax=%d (%d/world, floor=%d)",
            naconmax, nacon_per_world, nacon_floor,
        )
        template = self._data_template(naconmax)
        state_dtype = template.qpos.dtype

        paper_mppi_active = (
            _mppi_sampling_proposal is not None
            or _dial_annealed_proposal is not None
            or _mppi_velocity_exact_proposal is not None
        )
        velocity_arm_profile = (
            paper_mppi_active
            and control_profile_uses_joint_velocity_arm(self.control_profile)
        )
        (start_qpos_np, start_qpos_mask_np, start_qvel_np,
         start_qvel_mask_np, restore_grip_np) = self._mapped_start_state(
            start_state
        )
        measured_arm_j = jnp.asarray(
            start_qpos_np[np.asarray(self.arm_qpos_addr, dtype=int)],
            dtype=state_dtype,
        )
        measured_width_j = jnp.asarray(
            start_qpos_np[int(self.grip_qpos_addrs[0])], dtype=state_dtype
        )
        requested_torque_max_nm = np.asarray(
            (device_validity or {}).get(
                "joint_torque_max_nm", RIZON4S_MPC_TORQUE_LIMIT_NM
            ),
            dtype=float,
        )
        if (
            paper_mppi_active
            and not velocity_arm_profile
            and (
            requested_torque_max_nm.shape != (NJ,)
            or not np.all(np.isfinite(requested_torque_max_nm))
            or np.any(requested_torque_max_nm <= 0.0)
            or not np.allclose(
                requested_torque_max_nm,
                RIZON4S_MPC_TORQUE_LIMIT_NM,
                rtol=0.0,
                atol=1e-12,
            )
            )
        ):
            raise ValueError(
                "joint_torque_max_nm must equal the fixed project-wide "
                "Rizon4s MPC torque envelope"
            )
        arm_torque_scale_j = jnp.asarray(
            requested_torque_max_nm, dtype=state_dtype
        )
        def device_knots_to_model_control(knots: Any) -> tuple[Any, Any, Any, Any]:
            """Expand one device knot population into Warp actuator controls.

            Direct MPPI sends normalized arm effort to native torque actuators
            and maps jaw coefficients to signed effort.  The legacy diagnostic
            sends derived arm velocity to kv=600 velocity actuators and maps
            the same jaw latent to a GN01 width target.  Only the two sealed
            Pick-v8 causal profiles retain the successful v8 degree-one linear
            expansion; all other paper MPPI profiles remain ZOH actions.
            """
            if paper_mppi_active:
                knot_ctrl = _paper_mppi_knots_to_step_controls(
                    knots,
                    H,
                    spline_order=spline_order,
                    pick_v8_causal_profile=causal_profile,
                    unified_mppi_effort_profile=unified_mppi_effort_profile,
                    control_profile=self.control_profile,
                    array_module=jnp,
                )
            else:
                knot_ctrl = _linear_knots_to_ctrl_device(knots, H)
            controls = jnp.zeros(
                (N, H, self.nu),
                dtype=state_dtype,
            )
            if paper_mppi_active:
                arm_actuator_control = mppi_arm_actuator_control(
                    knot_ctrl[..., :NJ],
                    control_profile=self.control_profile,
                    torque_scale_nm=arm_torque_scale_j,
                )
                controls = controls.at[
                    ..., jnp.asarray(self.arm_act_ids)
                ].set(arm_actuator_control)
            else:
                controls = controls.at[
                    ..., jnp.asarray(self.arm_act_ids)
                ].set(knot_ctrl[..., :NJ])
            jaw_actuator_control, jaw_target_latent = (
                mppi_gripper_command_channels(
                    knot_ctrl[..., NJ],
                    control_profile=self.control_profile,
                    clip=jnp.clip,
                    where=jnp.where,
                    pick_v8_causal_profile=causal_profile,
                    unified_mppi_effort_profile=(
                        unified_mppi_effort_profile
                    ),
                )
            )
            jaw_actuator_control = jaw_actuator_control.astype(state_dtype)
            jaw_target_latent = jaw_target_latent.astype(state_dtype)
            if self.gripper == "slide":
                width = width_command_ctrl(
                    knot_ctrl[..., NJ],
                    where=jnp.where,
                ).astype(state_dtype)
                slide = jnp.clip(
                    width / GN01_OPEN_WIDTH_M * _SLIDE_OPEN_M,
                    0.0,
                    _SLIDE_OPEN_M,
                )
                controls = controls.at[
                    ..., jnp.asarray(self.grip_act_ids)
                ].set(slide[..., None])
                initial_width = (
                    jnp.broadcast_to(measured_width_j, (N, 1))
                    if paper_mppi_active else width[:, :1]
                )
                initial_slide = jnp.clip(
                    initial_width / GN01_OPEN_WIDTH_M * _SLIDE_OPEN_M,
                    0.0,
                    _SLIDE_OPEN_M,
                )
                initial_grip = jnp.repeat(
                    initial_slide,
                    len(self.grip_qpos_addrs),
                    axis=1,
                )
            else:
                controls = controls.at[..., int(self.grip_act_ids[0])].set(
                    jaw_actuator_control
                )
                initial_grip = jnp.broadcast_to(measured_width_j, (N, 1))
            return (
                controls,
                initial_grip,
                knots,
                # The task-cost channel is always the common signed latent.
                jaw_target_latent,
            )

        if device_select:
            if (
                _exact_sampling_proposal is not None
                or _mppi_velocity_exact_proposal is not None
            ):
                proposal_knots = np.asarray(nominal_knots, dtype=float)[None, ...]
                n_gaussian = 0
            elif (
                _cem_sampling_proposal is not None
                or _mppi_sampling_proposal is not None
                or _dial_annealed_proposal is not None
            ):
                assert cem_previous_std is not None
                if _dial_annealed_proposal is not None:
                    assert dial_half_span is not None
                    assert dial_noise_scale is not None
                    (
                        proposal_knots,
                        dial_clipping_count_device,
                    ) = dial_annealed_proposals(
                        nominal_knots,
                        ctrl_low,
                        ctrl_high,
                        normalized_half_span=dial_half_span,
                        normalized_noise_scale=dial_noise_scale,
                        sample_count=N,
                        key=jax.random.PRNGKey(int(proposal_seed)),
                        return_clipping_count=True,
                    )
                elif _mppi_sampling_proposal is not None:
                    if (
                        mppi_proposal_mode
                        == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    ):
                        (
                            proposal_knots,
                            n_broad_gaussian,
                            n_local_gaussian,
                        ) = mtp_global_local_velocity_proposals(
                            nominal_knots,
                            cem_previous_std,
                            ctrl_low,
                            ctrl_high,
                            pool_size=N,
                            seed=proposal_seed,
                            beta=mtp_beta,
                            graph_depth=mtp_graph_depth,
                            graph_width=mtp_graph_width,
                            jaw_mode=mtp_jaw_mode,
                        )
                    else:
                        proposal_knots = paper_mppi_effort_proposals(
                            nominal_knots,
                            cem_previous_std,
                            ctrl_low,
                            ctrl_high,
                            pool_size=N,
                            seed=proposal_seed,
                            sample_null_action=True,
                            null_action=mppi_null_action,
                        )
                else:
                    proposal_knots = cem_sampling_proposals(
                        nominal_knots,
                        cem_previous_std,
                        ctrl_low,
                        ctrl_high,
                        pool_size=N,
                        seed=proposal_seed,
                        jaw_bernoulli=bool(cem_jaw_bernoulli),
                    )
                n_gaussian = (
                    0
                    if mppi_proposal_mode
                    == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    else N - 1
                )
            else:
                (
                    proposal_knots,
                    n_gaussian,
                ) = predictive_sampling_proposals(
                    nominal_knots,
                    ctrl_low,
                    ctrl_high,
                    pool_size=N,
                    seed=proposal_seed,
                    sigma_fraction=proposal_broad_sigma_fraction,
                    sampler_mode=proposal_sampler_mode,
                    local_sigma_fraction=proposal_local_sigma_fraction,
                    # The binary latent must explore both signs even when the
                    # measured/warm-start command is at one bound.
                    uniform_jaw_effort_column=True,
                )
            proposal_knots = jnp.asarray(proposal_knots, dtype=state_dtype)
            (
                model_ctrl_j,
                grip_init,
                _action_knots_carrier_j,
                gripper_command_j,
            ) = device_knots_to_model_control(proposal_knots)
        else:
            model_ctrl_j = jnp.asarray(model_ctrl)
            grip_init = jnp.asarray(grip_init_np, dtype=state_dtype)
            gripper_command_j = model_ctrl_j[:, :, int(self.grip_act_ids[0])]
        obj_pose_np = np.asarray(obj_pose7, dtype=float)
        obj_q = jnp.asarray(obj_pose_np, dtype=state_dtype)
        obj_adr = self.object_qpos_addr
        mov_adrs = tuple(int(a) for _, a in self.movable_addrs)
        start_qpos = jnp.asarray(start_qpos_np, dtype=state_dtype)
        start_qpos_mask = jnp.asarray(start_qpos_mask_np)
        start_qvel = jnp.asarray(start_qvel_np, dtype=template.qvel.dtype)
        start_qvel_mask = jnp.asarray(start_qvel_mask_np)
        restore_grip = jnp.asarray(restore_grip_np)
        continued = start_state is not None
        arm_adr = jnp.asarray(self.arm_qpos_addr)
        # Rizon joints are revolute (one qpos and one qvel each), but resolve
        # the DOF addresses by joint name rather than relying on qpos==qvel
        # indexing so the physical velocity regularizer remains model-safe.
        arm_dof_adr = jnp.asarray([
            int(self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"joint{index + 1}",
                )
            ])
            for index in range(NJ)
        ])
        arm_act_j = jnp.asarray(self.arm_act_ids)
        grip_act_j = int(self.grip_act_ids[0])
        grip_qadr = jnp.asarray(self.grip_qpos_addrs)
        # The lift datum is the object's height at the START OF THIS ROLLOUT,
        # which is exactly what the caller passes as obj_pose7. It used to be
        # self.z0, snapshotted in build() -- and the two agreed only because
        # build() happens immediately before run() with the same world state.
        # Both CPU lanes already use obj_pose7[2] (see full_fidelity_rollout and
        # cpu_rollout), so the GPU lane was the odd one out, and the accident
        # would become a bug the moment the screen is reused across cycles.
        z0 = float(np.asarray(obj_pose7, dtype=float)[2])
        z0_j = jnp.asarray(z0, dtype=state_dtype)
        obj_gid = self.geom_ids["pick_object_collision"]
        lp_gid = self.geom_ids["gn01_left_finger_tip_collision"]
        rp_gid = self.geom_ids["gn01_right_finger_tip_collision"]
        tbl_gid = self.geom_ids["lab_table"]
        table_force_sensor_adr = int(self.table_contact_force_sensor_adr)
        table_force_weight = float(self.table_contact_force_weight)
        # Pezzato et al. discount one cost sample per controller input.  The
        # reference pick configuration uses 12 inputs over 1.2 s (100 ms
        # each).  Resolve the same force integral on the native physics grid.
        paper_control_dt_s = (
            float(n_steps) * float(self.plan_dt)
            / float(knot_count)
        )
        table_force_gamma_step = float(
            0.95 ** (float(self.plan_dt) / paper_control_dt_s)
        )
        table_force_step_scale = float(self.plan_dt) / paper_control_dt_s
        support_geom_mask_np = np.zeros(self.model.ngeom, dtype=bool)
        support_geom_mask_np[list(self.support_gids)] = True
        support_geom_mask = jnp.asarray(support_geom_mask_np)
        movable_geom_mask_np = np.zeros(self.model.ngeom, dtype=bool)
        movable_geom_mask_np[list(self.movable_gids)] = True
        movable_geom_mask = jnp.asarray(movable_geom_mask_np)
        movable_groups = dict(self.movable_geom_groups)
        terminal_contact_endpoint_targets = (
            _terminal_contact_endpoint_targets(
                tuple(getattr(device_cost_fn, "term_descriptors", ())),
                contact_target_bodies,
            )
        )
        (
            contact_target_names,
            target_names,
            evidence_target_names,
            evidence_target_groups,
        ) = _contact_evidence_target_layout(
            subject_geom_id=obj_gid,
            movable_geom_groups=movable_groups,
            contact_target_bodies=contact_target_bodies,
            mediation_target_bodies=mediation_target_bodies,
            controlled_subject_contact_evidence_name=(
                controlled_subject_contact_evidence_name
            ),
        )
        (
            scan_target_names,
            evidence_target_names,
            evidence_target_groups,
        ) = _terminal_contact_evidence_layout(
            mediation_target_names=target_names,
            evidence_target_names=evidence_target_names,
            evidence_target_groups=evidence_target_groups,
            movable_geom_groups=movable_groups,
            terminal_contact_target_names=(
                terminal_contact_endpoint_targets
            ),
        )
        terminal_contact_endpoint_enabled = bool(
            terminal_contact_endpoint_targets
        )
        contact_target_geom_mask_np = np.zeros(
            self.model.ngeom,
            dtype=bool,
        )
        for contact_target_name in contact_target_names:
            contact_target_geom_mask_np[
                list(movable_groups[contact_target_name])
            ] = True
        contact_target_geom_mask = jnp.asarray(
            contact_target_geom_mask_np
        )
        target_geom_mask_np = np.zeros(self.model.ngeom, dtype=bool)
        target_geom_masks_np = np.zeros(
            (len(evidence_target_names), self.model.ngeom),
            dtype=bool,
        )
        for target_name in scan_target_names:
            target_geoms = list(movable_groups[target_name])
            target_geom_mask_np[target_geoms] = True
        for target_index, target_name in enumerate(evidence_target_names):
            target_geoms = list(evidence_target_groups[target_name])
            target_geom_masks_np[target_index, target_geoms] = True
        target_geom_mask = jnp.asarray(target_geom_mask_np)
        target_geom_masks = jnp.asarray(target_geom_masks_np)
        robot_geom_mask_np = np.zeros(self.model.ngeom, dtype=bool)
        robot_geom_mask_np[list(self.robot_gids)] = True
        robot_geom_mask = jnp.asarray(robot_geom_mask_np)
        exact_arm_geom_mask_np = np.zeros(self.model.ngeom, dtype=bool)
        exact_arm_geom_mask_np[list(self.exact_arm_gids)] = True
        exact_arm_geom_mask = jnp.asarray(exact_arm_geom_mask_np)
        subject_contact_geom_mask_np = np.zeros(
            self.model.ngeom,
            dtype=bool,
        )
        subject_contact_geom_mask_np[list(self.subject_contact_gids)] = True
        subject_contact_geom_mask = jnp.asarray(
            subject_contact_geom_mask_np
        )
        initial_scene_geom_mask_np = np.zeros(self.model.ngeom, dtype=bool)
        initial_scene_geom_mask_np[list(self.initial_scene_gids)] = True
        initial_scene_geom_mask = jnp.asarray(initial_scene_geom_mask_np)
        mx = self.mx

        def contact_terms(
            d,
            world_idx,
            contact_target_geom_mask_arg,
            target_geom_mask_arg,
            target_geom_masks_arg,
        ):
            dist, g1, g2, nacon, worldid = _warp_contacts(d)
            reduced = _reduce_warp_contact_arena(
                dist,
                g1,
                g2,
                nacon,
                worldid,
                num_worlds=N,
                subject_geom_id=obj_gid,
                left_pad_geom_id=lp_gid,
                right_pad_geom_id=rp_gid,
                table_geom_id=tbl_gid,
                support_geom_mask=support_geom_mask,
                movable_geom_mask=movable_geom_mask,
                robot_geom_mask=robot_geom_mask,
                exact_arm_geom_mask=exact_arm_geom_mask,
                subject_contact_geom_mask=subject_contact_geom_mask,
                contact_target_geom_mask=contact_target_geom_mask_arg,
                target_geom_mask=target_geom_mask_arg,
                target_geom_masks=target_geom_masks_arg,
                initial_scene_geom_mask=initial_scene_geom_mask,
            )
            # ``reduced`` is computed only from MJWarp DATA_NON_VMAP leaves, so
            # vmap emits one arena scatter.  The sole mapped operation is this
            # row gather for the current rollout lane.
            return tuple(value[world_idx] for value in reduced) + (nacon,)

        def rollout_batch(
            template_arg, ctrl_batch, grip_batch, command_latent_batch,
            obj_q_arg, z0_arg,
            start_qpos_arg, start_qpos_mask_arg,
            start_qvel_arg, start_qvel_mask_arg, restore_grip_arg,
            previous_executed_action_arg, regularizer_boundary_valid_arg,
            prefix_index_arg, contact_target_geom_mask_arg,
            target_geom_mask_arg, target_geom_masks_arg,
        ):
            """One retained JIT body; every cycle-dependent value is an argument."""

            def rollout_one(ctrl_seq, grip0, command_latent_t, world_idx):
                if self.gripper == "slide" and not paper_mppi_active:
                    command_latent_t = ctrl_seq[:, grip_act_j]
                    command_latent_t = (
                        command_latent_t
                        / _SLIDE_OPEN_M
                        * GN01_OPEN_WIDTH_M
                    )

                def physical_gripper_width(qpos_value):
                    width = qpos_value[grip_qadr[0]]
                    if self.gripper == "slide":
                        width = (
                            width / _SLIDE_OPEN_M * GN01_OPEN_WIDTH_M
                        )
                    return width

                d = template_arg
                qpos = d.qpos
                if not continued:
                    qpos = qpos.at[obj_adr:obj_adr + 7].set(obj_q_arg)
                    qpos = qpos.at[arm_adr].set(
                        measured_arm_j
                        if paper_mppi_active
                        else ctrl_seq[0][arm_act_j]
                    )
                    qpos = qpos.at[grip_qadr].set(
                        grip0)                  # gripper open at frame 0
                    d = d.replace(qpos=qpos, ctrl=ctrl_seq[0])
                else:
                    # CONTINUE from the caller's live state (a held carry, a
                    # later receding-horizon cycle). Fixed-shape masks preserve
                    # address-wise mapping while keeping qpos/qvel as traced
                    # arguments, so a retained compiled callable can never
                    # replay the first cycle's state.
                    qpos = jnp.where(start_qpos_mask_arg, start_qpos_arg, qpos)
                    # The linkage mapping contains all seven real gn01 joints.
                    # If it exists, keep that real configuration; the slide
                    # surrogate has no shared joints and rides the knot width.
                    grip_fallback = qpos.at[grip_qadr].set(grip0)
                    qpos = jnp.where(
                        restore_grip_arg, qpos, grip_fallback)
                    # CPU continuation carries velocity too. Unmapped DoFs keep
                    # the template value (normally zero), exactly as before.
                    qvel = jnp.where(
                        start_qvel_mask_arg, start_qvel_arg, d.qvel)
                    d = d.replace(
                        qpos=qpos, qvel=qvel, ctrl=ctrl_seq[0])

                # Retain each post-step state x_1,...,x_H and its applied
                # control u_0,...,u_{H-1}; the scorer averages all H terms.
                idx = jnp.asarray(physics_sample_indices(int(ctrl_seq.shape[0])))
                prefix_end = jnp.clip(
                    prefix_index_arg,
                    0,
                    ctrl_seq.shape[0] - 1,
                )
                prefix_begin = jnp.maximum(0, prefix_end - 4)
                prefix_window = jnp.minimum(
                    prefix_begin + jnp.arange(5, dtype=jnp.int32),
                    prefix_end,
                )
                final_window_begin = max(0, int(ctrl_seq.shape[0]) - 5)
                target_count = int(target_geom_masks_arg.shape[0])
                count_zero = jnp.zeros((), dtype=jnp.int32)
                scalar_zero = jnp.zeros((), dtype=d.qpos.dtype)
                stream = {
                    "data": d,
                    # Exact selected-prefix plant state.  This remains
                    # O(nq+nv) per candidate and is consumed by the GPU-offline
                    # plant transition; it is never reconstructed from samples.
                    "prefix_qpos": d.qpos,
                    "prefix_qvel": d.qvel,
                    "arm_action_endpoint_qpos": jnp.zeros(
                        (knot_count, NJ), dtype=d.qpos.dtype
                    ),
                    "pose_traj7": jnp.zeros(
                        (POSE_SAMPLES, 7), dtype=d.qpos.dtype),
                    "movable_traj7": jnp.zeros(
                        (POSE_SAMPLES, len(mov_adrs), 7),
                        dtype=d.qpos.dtype,
                    ),
                    "gripper_traj3": jnp.zeros(
                        (POSE_SAMPLES, 3), dtype=d.qpos.dtype),
                    "gripper_axis_traj3": jnp.zeros(
                        (POSE_SAMPLES, 3), dtype=d.qpos.dtype),
                    "gripper_closing_axis_traj3": jnp.zeros(
                        (POSE_SAMPLES, 3), dtype=d.qpos.dtype),
                    # Physical joint state and actuator target are deliberately
                    # separate. GripperState consumes this qpos-derived width;
                    # GripperCommand consumes command_latent_t below.
                    "gripper_width_traj": jnp.zeros(
                        (POSE_SAMPLES,), dtype=d.qpos.dtype),
                    "object_held_traj": jnp.zeros(
                        (POSE_SAMPLES,), dtype=jnp.bool_),
                    "tool_target_contact_traj": jnp.zeros(
                        (POSE_SAMPLES,), dtype=jnp.bool_),
                    "robot_target_contact_traj": jnp.zeros(
                        (POSE_SAMPLES,), dtype=jnp.bool_),
                    "robot_subject_contact_frames": count_zero,
                    "robot_subject_contact_suffix_frames": count_zero,
                    "robot_subject_contact_endpoint": jnp.asarray(False),
                    "prefix_robot_subject_contact_frames": count_zero,
                    "prefix_robot_subject_contact_now": jnp.asarray(False),
                    "prefix_pose7": jnp.zeros((7,), dtype=d.qpos.dtype),
                    "prefix_movable_pose7": jnp.zeros(
                        (len(mov_adrs), 7), dtype=d.qpos.dtype),
                    "prefix_gripper_center3": jnp.zeros(
                        (3,), dtype=d.qpos.dtype),
                    "prefix_gripper_axis3": jnp.zeros(
                        (3,), dtype=d.qpos.dtype),
                    "prefix_gripper_closing_axis3": jnp.zeros(
                        (3,), dtype=d.qpos.dtype),
                    "prefix_gripper_width": scalar_zero,
                    "two_sided_frames": count_zero,
                    "prefix_two_sided": count_zero,
                    "prefix_two_sided_now": jnp.asarray(False),
                    "prefix_two_sided_frames": count_zero,
                    "prefix_lift": scalar_zero,
                    "final_two_sided": count_zero,
                    "final_two_sided_now": jnp.asarray(False),
                    # Legacy max() observes post-step states only.  Starting at
                    # -inf preserves a horizon whose subject always moves down.
                    "max_lift": jnp.asarray(-jnp.inf, dtype=d.qpos.dtype),
                    "max_obj_world_pen": scalar_zero,
                    "max_obj_table_pen": scalar_zero,
                    "max_obj_other_support_pen": scalar_zero,
                    "max_subject_contact_target_pen": scalar_zero,
                    "max_pad_table_pen": scalar_zero,
                    "max_obj_pad_pen": scalar_zero,
                    "max_arm_pen": scalar_zero,
                    "arm_velocity_squared_sum": scalar_zero,
                    **(
                        {
                            "control_regularizer_arm_qvel_time_integral": (
                                scalar_zero
                            )
                        }
                        if regularizer_active
                        else {}
                    ),
                    # Paper MPPI collision term: discounted L1 table contact
                    # force.  The optional MuJoCo contact sensor is evaluated
                    # inside this exact MJWarp scan; CEM models have no sensor
                    # and therefore keep these reductions identically zero.
                    "table_contact_force_cost": scalar_zero,
                    "max_table_contact_force": scalar_zero,
                    # ``max_table_contact_force`` reduces over the whole
                    # horizon, but only the executable prefix is ever applied to
                    # the robot.  A full-horizon peak therefore reports a force
                    # the arm never exerted: the 678 N peak of the sealed
                    # geom-filtered Push run occurs at a horizon-tail step whose
                    # recorded states replay to zero table contacts.  Scope the
                    # statistic the same way the penetration reductions below
                    # already are, so a force number means "what the robot did".
                    "prefix_max_table_contact_force": scalar_zero,
                    "max_subject_movable_pen": scalar_zero,
                    "subject_movable_frames": count_zero,
                    "prefix_max_subject_movable_pen": scalar_zero,
                    "prefix_subject_movable_frames": count_zero,
                    "max_robot_movable_pen": scalar_zero,
                    "robot_movable_frames": count_zero,
                    "prefix_max_robot_movable_pen": scalar_zero,
                    "prefix_robot_movable_frames": count_zero,
                    "max_subject_target_pen": scalar_zero,
                    "subject_target_frames": count_zero,
                    "max_robot_target_pen": scalar_zero,
                    "robot_target_frames": count_zero,
                    "prefix_max_subject_target_pen_by_body": jnp.zeros(
                        (target_count,), dtype=d.qpos.dtype),
                    "prefix_max_robot_target_pen_by_body": jnp.zeros(
                        (target_count,), dtype=d.qpos.dtype),
                    "prefix_subject_target_frames_by_body": jnp.zeros(
                        (target_count,), dtype=jnp.int32),
                    # Exact Boolean contact at the independently selected
                    # executable-prefix endpoint E-1. This is captured from
                    # t_st_by at that native scan step, never inferred from
                    # accumulated frames or maximum penetration.
                    "prefix_subject_target_now_by_body": jnp.zeros(
                        (target_count,), dtype=jnp.bool_),
                    # Exact Boolean contact at the latest native physics step.
                    # After scan completion this is H-1 for each target body.
                    "final_subject_target_now_by_body": jnp.zeros(
                        (target_count,), dtype=jnp.bool_),
                    "prefix_robot_target_frames_by_body": jnp.zeros(
                        (target_count,), dtype=jnp.int32),
                    # Knot 0 is fixed across production candidates. Inspect
                    # only its first post-step contact state; later legal
                    # manipulation contacts must never contaminate this
                    # episode-start diagnostic.
                    "initial_robot_scene_contact": jnp.asarray(False),
                    "peak_global_nacon": count_zero,
                    # MJWarp's contact arena is shared across the batch, while
                    # nefc and solver_niter are per world.  Capacity saturation
                    # makes a rollout numerically incomplete, so retain exact
                    # device-side peaks and reject saturated candidates before
                    # selection instead of trusting backend warnings.
                    "peak_per_world_nefc": count_zero,
                    "peak_solver_niter": count_zero,
                }

                def body(carry, inputs):
                    u, step_index = inputs
                    dd = carry["data"].replace(ctrl=u)
                    dd = mjx.step(mx, dd)
                    if table_force_sensor_adr >= 0:
                        table_force_vector = dd.sensordata[
                            table_force_sensor_adr:
                            table_force_sensor_adr + 3
                        ]
                        table_force = jnp.sum(jnp.abs(table_force_vector))
                    else:
                        table_force = jnp.zeros((), dtype=dd.qpos.dtype)
                    discounted_table_force_cost = (
                        table_force_weight
                        * table_force_step_scale
                        * jnp.power(table_force_gamma_step, step_index)
                        * table_force
                    )
                    current_nefc = jnp.max(
                        jnp.asarray(dd._impl.nefc, dtype=jnp.int32)
                    )
                    current_solver_niter = jnp.max(
                        jnp.asarray(dd._impl.solver_niter, dtype=jnp.int32)
                    )
                    (two, p_ot, p_pt, p_ow, p_og, p_arm,
                     p_sm, p_rm, p_st, p_rt, p_sct, t_sm, t_rm,
                     t_st, t_rt, p_st_by, p_rt_by,
                     t_st_by, t_rt_by, t_robot_scene, t_robot_subject,
                     global_nacon) = contact_terms(
                         dd,
                         world_idx,
                         contact_target_geom_mask_arg,
                         target_geom_mask_arg,
                         target_geom_masks_arg,
                     )
                    objz = dd.qpos[obj_adr + 2]
                    # Zero-row stack when the scene has one mover: the arrays
                    # below stay well-formed and every consumer sees an empty
                    # mapping.
                    mov = (jnp.stack([dd.qpos[a:a + 7] for a in mov_adrs])
                           if mov_adrs else jnp.zeros((0, 7)))
                    site_rotation = dd.site_xmat[self.site_id].reshape((3, 3))
                    pose = dd.qpos[obj_adr:obj_adr + 7]
                    gripper = dd.site_xpos[self.site_id]
                    gripper_axis = site_rotation[:, 2]
                    gripper_closing_axis = site_rotation[:, 1]
                    gripper_width = physical_gripper_width(dd.qpos)
                    sample_mask = idx == step_index
                    action_endpoint_index = jnp.minimum(
                        (
                            jnp.ceil(
                                (jnp.arange(knot_count) + 1)
                                * float(ctrl_seq.shape[0])
                                / float(knot_count)
                            ).astype(jnp.int32)
                            - 1
                        ),
                        ctrl_seq.shape[0] - 1,
                    )
                    action_endpoint_mask = action_endpoint_index == step_index
                    capture = step_index == prefix_end
                    in_prefix = step_index <= prefix_end
                    # Preserve the clipped five-index window exactly.  When the
                    # prefix is below four, its endpoint occurs more than once.
                    prefix_weight = jnp.sum(
                        (prefix_window == step_index).astype(jnp.int32)
                    )
                    in_final_window = step_index >= final_window_begin

                    def as_count(value):
                        return value.astype(jnp.int32)

                    prefix_count = as_count(in_prefix)
                    next_carry = {
                        "data": dd,
                        "prefix_qpos": jnp.where(
                            capture, dd.qpos, carry["prefix_qpos"]),
                        "prefix_qvel": jnp.where(
                            capture, dd.qvel, carry["prefix_qvel"]),
                        "arm_action_endpoint_qpos": jnp.where(
                            action_endpoint_mask[:, None],
                            dd.qpos[arm_adr][None, :],
                            carry["arm_action_endpoint_qpos"],
                        ),
                        "pose_traj7": jnp.where(
                            sample_mask[:, None],
                            pose[None, :],
                            carry["pose_traj7"],
                        ),
                        "movable_traj7": jnp.where(
                            sample_mask[:, None, None],
                            mov[None, :, :],
                            carry["movable_traj7"],
                        ),
                        "gripper_traj3": jnp.where(
                            sample_mask[:, None],
                            gripper[None, :],
                            carry["gripper_traj3"],
                        ),
                        "gripper_axis_traj3": jnp.where(
                            sample_mask[:, None],
                            gripper_axis[None, :],
                            carry["gripper_axis_traj3"],
                        ),
                        "gripper_closing_axis_traj3": jnp.where(
                            sample_mask[:, None],
                            gripper_closing_axis[None, :],
                            carry["gripper_closing_axis_traj3"],
                        ),
                        "gripper_width_traj": jnp.where(
                            sample_mask,
                            gripper_width,
                            carry["gripper_width_traj"],
                        ),
                        "object_held_traj": jnp.where(
                            sample_mask,
                            two,
                            carry["object_held_traj"],
                        ),
                        "tool_target_contact_traj": jnp.where(
                            sample_mask,
                            t_st,
                            carry["tool_target_contact_traj"],
                        ),
                        "robot_target_contact_traj": jnp.where(
                            sample_mask,
                            t_rt,
                            carry["robot_target_contact_traj"],
                        ),
                        "robot_subject_contact_frames": (
                            carry["robot_subject_contact_frames"]
                            + as_count(t_robot_subject)
                        ),
                        "robot_subject_contact_suffix_frames": jnp.where(
                            t_robot_subject,
                            carry["robot_subject_contact_suffix_frames"] + 1,
                            count_zero,
                        ),
                        "robot_subject_contact_endpoint": t_robot_subject,
                        "prefix_robot_subject_contact_frames": (
                            carry["prefix_robot_subject_contact_frames"]
                            + as_count(t_robot_subject) * prefix_count
                        ),
                        "prefix_robot_subject_contact_now": jnp.where(
                            capture,
                            t_robot_subject,
                            carry["prefix_robot_subject_contact_now"],
                        ),
                        "prefix_pose7": jnp.where(
                            capture, pose, carry["prefix_pose7"]),
                        "prefix_movable_pose7": jnp.where(
                            capture, mov, carry["prefix_movable_pose7"]),
                        "prefix_gripper_center3": jnp.where(
                            capture,
                            gripper,
                            carry["prefix_gripper_center3"],
                        ),
                        "prefix_gripper_axis3": jnp.where(
                            capture,
                            gripper_axis,
                            carry["prefix_gripper_axis3"],
                        ),
                        "prefix_gripper_closing_axis3": jnp.where(
                            capture,
                            gripper_closing_axis,
                            carry["prefix_gripper_closing_axis3"],
                        ),
                        "prefix_gripper_width": jnp.where(
                            capture,
                            gripper_width,
                            carry["prefix_gripper_width"],
                        ),
                        "two_sided_frames": (
                            carry["two_sided_frames"] + as_count(two)
                        ),
                        "prefix_two_sided": (
                            carry["prefix_two_sided"]
                            + as_count(two) * prefix_weight
                        ),
                        "prefix_two_sided_now": jnp.where(
                            capture,
                            two,
                            carry["prefix_two_sided_now"],
                        ),
                        "prefix_two_sided_frames": (
                            carry["prefix_two_sided_frames"]
                            + as_count(two) * prefix_count
                        ),
                        "prefix_lift": jnp.where(
                            capture,
                            objz - z0_arg,
                            carry["prefix_lift"],
                        ),
                        "final_two_sided": (
                            carry["final_two_sided"]
                            + as_count(two) * as_count(in_final_window)
                        ),
                        "final_two_sided_now": two,
                        "max_lift": jnp.maximum(
                            carry["max_lift"],
                            objz - z0_arg,
                        ),
                        "max_obj_world_pen": jnp.maximum(
                            carry["max_obj_world_pen"],
                            jnp.maximum(p_ow, p_ot),
                        ),
                        "max_obj_table_pen": jnp.maximum(
                            carry["max_obj_table_pen"],
                            p_ot,
                        ),
                        "max_obj_other_support_pen": jnp.maximum(
                            carry["max_obj_other_support_pen"],
                            p_ow,
                        ),
                        "max_subject_contact_target_pen": jnp.maximum(
                            carry["max_subject_contact_target_pen"],
                            p_sct,
                        ),
                        "max_pad_table_pen": jnp.maximum(
                            carry["max_pad_table_pen"], p_pt),
                        "max_obj_pad_pen": jnp.maximum(
                            carry["max_obj_pad_pen"], p_og),
                        "max_arm_pen": jnp.maximum(
                            carry["max_arm_pen"], p_arm),
                        "arm_velocity_squared_sum": (
                            carry["arm_velocity_squared_sum"]
                            + jnp.where(
                                jnp.any(action_endpoint_mask),
                                jnp.sum(dd.qvel[arm_dof_adr] ** 2),
                                scalar_zero,
                            )
                        ),
                        **(
                            {
                                "control_regularizer_arm_qvel_time_integral": (
                                    carry[
                                        "control_regularizer_arm_qvel_time_integral"
                                    ]
                                    + self.exec_dt
                                    * jnp.mean(dd.qvel[arm_dof_adr] ** 2)
                                )
                            }
                            if regularizer_active
                            else {}
                        ),
                        "table_contact_force_cost": (
                            carry["table_contact_force_cost"]
                            + discounted_table_force_cost
                        ),
                        "max_table_contact_force": jnp.maximum(
                            carry["max_table_contact_force"], table_force
                        ),
                        "prefix_max_table_contact_force": jnp.maximum(
                            carry["prefix_max_table_contact_force"],
                            jnp.where(in_prefix, table_force, 0.0),
                        ),
                        "max_subject_movable_pen": jnp.maximum(
                            carry["max_subject_movable_pen"], p_sm),
                        "subject_movable_frames": (
                            carry["subject_movable_frames"] + as_count(t_sm)
                        ),
                        "prefix_max_subject_movable_pen": jnp.maximum(
                            carry["prefix_max_subject_movable_pen"],
                            jnp.where(in_prefix, p_sm, 0.0),
                        ),
                        "prefix_subject_movable_frames": (
                            carry["prefix_subject_movable_frames"]
                            + as_count(t_sm) * prefix_count
                        ),
                        "max_robot_movable_pen": jnp.maximum(
                            carry["max_robot_movable_pen"], p_rm),
                        "robot_movable_frames": (
                            carry["robot_movable_frames"] + as_count(t_rm)
                        ),
                        "prefix_max_robot_movable_pen": jnp.maximum(
                            carry["prefix_max_robot_movable_pen"],
                            jnp.where(in_prefix, p_rm, 0.0),
                        ),
                        "prefix_robot_movable_frames": (
                            carry["prefix_robot_movable_frames"]
                            + as_count(t_rm) * prefix_count
                        ),
                        "max_subject_target_pen": jnp.maximum(
                            carry["max_subject_target_pen"], p_st),
                        "subject_target_frames": (
                            carry["subject_target_frames"] + as_count(t_st)
                        ),
                        "max_robot_target_pen": jnp.maximum(
                            carry["max_robot_target_pen"], p_rt),
                        "robot_target_frames": (
                            carry["robot_target_frames"] + as_count(t_rt)
                        ),
                        "prefix_max_subject_target_pen_by_body": jnp.maximum(
                            carry[
                                "prefix_max_subject_target_pen_by_body"
                            ],
                            jnp.where(in_prefix, p_st_by, 0.0),
                        ),
                        "prefix_max_robot_target_pen_by_body": jnp.maximum(
                            carry["prefix_max_robot_target_pen_by_body"],
                            jnp.where(in_prefix, p_rt_by, 0.0),
                        ),
                        "prefix_subject_target_frames_by_body": (
                            carry["prefix_subject_target_frames_by_body"]
                            + as_count(t_st_by) * prefix_count
                        ),
                        "prefix_subject_target_now_by_body": jnp.where(
                            capture,
                            t_st_by,
                            carry["prefix_subject_target_now_by_body"],
                        ),
                        "final_subject_target_now_by_body": t_st_by,
                        "prefix_robot_target_frames_by_body": (
                            carry["prefix_robot_target_frames_by_body"]
                            + as_count(t_rt_by) * prefix_count
                        ),
                        "initial_robot_scene_contact": (
                            jnp.where(
                                step_index == 0,
                                t_robot_scene,
                                carry["initial_robot_scene_contact"],
                            )
                        ),
                        "peak_global_nacon": jnp.maximum(
                            carry["peak_global_nacon"],
                            global_nacon.astype(jnp.int32),
                        ),
                        "peak_per_world_nefc": jnp.maximum(
                            carry["peak_per_world_nefc"],
                            current_nefc,
                        ),
                        "peak_solver_niter": jnp.maximum(
                            carry["peak_solver_niter"],
                            current_solver_niter,
                        ),
                    }
                    # The experiment-only trace is a direct output of this same
                    # scan.  With the switch off, ``ys`` remains None exactly as
                    # in the production memory path.
                    return (
                        next_carry,
                        dd.qpos if record_step_trace else None,
                    )

                stream, post_step_qpos = jax.lax.scan(
                    body,
                    stream,
                    (
                        ctrl_seq,
                        jnp.arange(ctrl_seq.shape[0], dtype=jnp.int32),
                    ),
                )
                d_final = stream["data"]
                # ``mjx.step`` carries the contact arena computed for that
                # integration step, while ``d_final.qpos`` is already the
                # resulting state.  Those event contacts are authoritative for
                # the running missing-contact fraction above.  A terminal
                # RobotSubjectContact predicate instead asks whether x_H is in
                # contact, so refresh contact kinematics once at the final
                # state.  Keep the refreshed Data separate: dynamics, prefix
                # transfer, every other contact channel, and all returned
                # qpos/qvel/ctrl continue to use the unmodified scan carry.
                final_contact_terms = contact_terms(
                    mjx.forward(mx, d_final),
                    world_idx,
                    contact_target_geom_mask_arg,
                    target_geom_mask_arg,
                    target_geom_masks_arg,
                )
                final_robot_subject_contact = final_contact_terms[-2]
                final_movable = (
                    jnp.stack([d_final.qpos[a:a + 7] for a in mov_adrs])
                    if mov_adrs
                    else jnp.zeros((0, 7), dtype=d_final.qpos.dtype)
                )
                final_site_rotation = d_final.site_xmat[self.site_id].reshape(
                    (3, 3)
                )
                result = {
                    "final_pose7": d_final.qpos[obj_adr:obj_adr + 7],
                    "prefix_pose7": stream["prefix_pose7"],
                    # Exact selected-prefix plant state.  The scan captures the
                    # state in its carry, so this adds O(nq+nv) output per
                    # candidate rather than materializing an H-long state
                    # trajectory.
                    "prefix_qpos": stream["prefix_qpos"],
                    "prefix_qvel": stream["prefix_qvel"],
                    "arm_action_endpoint_qpos": (
                        stream["arm_action_endpoint_qpos"]
                    ),
                    # Native command actually applied to the rollout model.
                    "prefix_ctrl": ctrl_seq[prefix_end],
                    # Position-command carrier used only by the host planning
                    # twin after adopting the exact native-effort endpoint.
                    "prefix_carrier_ctrl": mppi_prefix_carrier_ctrl(
                        stream["prefix_qpos"],
                        arm_adr,
                        (
                            ctrl_seq[prefix_end, grip_act_j]
                            if self.experiment_control_profile
                            else command_latent_t[prefix_end]
                        ),
                        concatenate=jnp.concatenate,
                    ),
                    # Established task-semantic field: common signed latent.
                    "prefix_gripper_command": command_latent_t[prefix_end],
                    "prefix_gripper_command_latent": (
                        command_latent_t[prefix_end]
                    ),
                    # Exact native actuator command at the adopted endpoint:
                    # metres for legacy width, Newtons for direct effort.
                    "prefix_gripper_actuator_command": ctrl_seq[
                        prefix_end, grip_act_j
                    ],
                    "pose_traj7": stream["pose_traj7"],
                    # Reserved program anchor, grounded from this candidate's
                    # actual FK at the same sample times as every body pose.
                    "gripper_traj3": stream["gripper_traj3"],
                    "final_gripper_center": d_final.site_xpos[self.site_id],
                    # The reserved anchor's axis is the planning site's local
                    # +Z direction transformed by this candidate's measured
                    # FK, not a commanded or statically grounded orientation.
                    "gripper_axis_traj3": stream["gripper_axis_traj3"],
                    "final_gripper_axis": final_site_rotation[:, 2],
                    "gripper_closing_axis_traj3": (
                        stream["gripper_closing_axis_traj3"]
                    ),
                    "final_gripper_closing_axis": (
                        final_site_rotation[:, 1]
                    ),
                    # Physical qpos state for terminal GripperState.
                    "gripper_width_traj": stream["gripper_width_traj"],
                    "final_gripper_width": physical_gripper_width(
                        d_final.qpos
                    ),
                    # Candidate ctrl target for running GripperCommand. A held
                    # object may obstruct the physical jaw, so this must never
                    # masquerade as GripperState.
                    "gripper_command_traj": command_latent_t[idx],
                    "final_gripper_command": command_latent_t[-1],
                    # ObjectHeld evidence on the exact same sample grid as
                    # every trajectory consumed by the device running cost.
                    "object_held_traj": stream["object_held_traj"],
                    # ToolMediation evidence on that identical grid. MuJoCo can
                    # emit positive-distance candidate pairs, so contact is
                    # exactly ``dist <= 0`` rather than positive penetration.
                    "tool_target_contact_traj": (
                        stream["tool_target_contact_traj"]
                    ),
                    "robot_target_contact_traj": (
                        stream["robot_target_contact_traj"]
                    ),
                    # ObjectHeld uses exact full-physics-step counts. Whole-
                    # horizon loss is a finite running-cost residual. Prefix
                    # evidence below is retained for diagnostics and state
                    # transfer only; it is never a planning barrier.
                    "object_not_held_steps": (
                        ctrl_seq.shape[0] - stream["two_sided_frames"]
                    ),
                    "execution_prefix_steps": prefix_end + 1,
                    "prefix_object_not_held_steps": (
                        prefix_end
                        + 1
                        - stream["prefix_two_sided_frames"]
                    ),
                    "tool_target_contact_any_step": (
                        stream["subject_target_frames"] > 0
                    ),
                    "robot_target_contact_any_step": (
                        stream["robot_target_frames"] > 0
                    ),
                    # RobotSubjectContact is a distinct fixed-role primitive.
                    # Running consumes integration contact events on every
                    # native physics step. Terminal consumes the independently
                    # refreshed contact state at x_H. The measured post-prefix
                    # host terminal remains authoritative for Stage progress.
                    "robot_subject_contact_frames": (
                        stream["robot_subject_contact_frames"]
                    ),
                    "robot_subject_contact_suffix_frames": (
                        stream["robot_subject_contact_suffix_frames"]
                    ),
                    "robot_subject_contact_missing_fraction": (
                        1.0
                        - stream["robot_subject_contact_frames"]
                        / float(ctrl_seq.shape[0])
                    ),
                    "robot_subject_contact_endpoint": (
                        final_robot_subject_contact
                    ),
                    "prefix_robot_subject_contact_frames": (
                        stream["prefix_robot_subject_contact_frames"]
                    ),
                    "prefix_robot_subject_contact_now": (
                        stream["prefix_robot_subject_contact_now"]
                    ),
                    # (POSE_SAMPLES, n_movable, 7) and (n_movable, 7): the pushed
                    # body's trajectory, on the SAME sample grid as the
                    # subject's so the running cost sees both bodies at the same
                    # instants.
                    "movable_traj7": stream["movable_traj7"],
                    "final_movable_pose7": final_movable,
                    "prefix_movable_pose7": stream["prefix_movable_pose7"],
                    "prefix_gripper_center3": (
                        stream["prefix_gripper_center3"]
                    ),
                    "prefix_gripper_axis3": stream["prefix_gripper_axis3"],
                    "prefix_gripper_closing_axis3": (
                        stream["prefix_gripper_closing_axis3"]
                    ),
                    "prefix_gripper_width": (
                        stream["prefix_gripper_width"]
                    ),
                    "two_sided_frames": stream["two_sided_frames"],
                    "prefix_two_sided": stream["prefix_two_sided"],
                    "prefix_two_sided_now": stream["prefix_two_sided_now"],
                    "prefix_two_sided_frames": (
                        stream["prefix_two_sided_frames"]
                    ),
                    "prefix_lift": stream["prefix_lift"],
                    # Contact counts and endpoint evidence are diagnostics and
                    # explicit typed ObjectHeld cost inputs, never implicit
                    # task-independent validity.
                    "final_two_sided": stream["final_two_sided"],
                    # Exact endpoint evidence for explicit ObjectHeld terms;
                    # it carries no duration threshold.
                    "final_two_sided_now": stream["final_two_sided_now"],
                    "max_lift": stream["max_lift"],
                    "final_lift": (
                        d_final.qpos[obj_adr + 2] - z0_arg
                    ),
                    "max_obj_world_pen": stream["max_obj_world_pen"],
                    "max_obj_table_pen": stream["max_obj_table_pen"],
                    "max_obj_other_support_pen": (
                        stream["max_obj_other_support_pen"]
                    ),
                    "max_subject_contact_target_pen": (
                        stream["max_subject_contact_target_pen"]
                    ),
                    "max_pad_table_pen": stream["max_pad_table_pen"],
                    "max_obj_pad_pen": stream["max_obj_pad_pen"],
                    "max_arm_pen": stream["max_arm_pen"],
                    "arm_velocity_squared_sum": (
                        stream["arm_velocity_squared_sum"]
                    ),
                    "table_contact_force_cost": (
                        stream["table_contact_force_cost"]
                    ),
                    "max_table_contact_force": (
                        stream["max_table_contact_force"]
                    ),
                    "prefix_max_table_contact_force": (
                        stream["prefix_max_table_contact_force"]
                    ),
                    # Manipulation, not violation: how hard the SUBJECT pressed
                    # on a movable body, and for how many frames it was in
                    # contact at all. The frame count is an EXISTENCE signal
                    # (did the tool ever touch the goal), deliberately not a
                    # duration threshold -- the successful runs measured 8-11
                    # contact steps, so any "N frames" rule big enough to mean
                    # something rejects real successes.
                    "max_subject_movable_pen": (
                        stream["max_subject_movable_pen"]
                    ),
                    "subject_movable_frames": (
                        stream["subject_movable_frames"]
                    ),
                    "prefix_max_subject_movable_pen": (
                        stream["prefix_max_subject_movable_pen"]
                    ),
                    "prefix_subject_movable_frames": (
                        stream["prefix_subject_movable_frames"]
                    ),
                    # The tool being bypassed: the robot itself touching a
                    # movable body.
                    "max_robot_movable_pen": (
                        stream["max_robot_movable_pen"]
                    ),
                    "robot_movable_frames": stream["robot_movable_frames"],
                    "prefix_max_robot_movable_pen": (
                        stream["prefix_max_robot_movable_pen"]
                    ),
                    "prefix_robot_movable_frames": (
                        stream["prefix_robot_movable_frames"]
                    ),
                    # Causal evidence is scoped to the dynamic body named by
                    # the current terminal. Contacts with other movable bodies
                    # remain physical diagnostics but cannot satisfy or veto
                    # this stage's tool-use condition.
                    "max_subject_target_pen": (
                        stream["max_subject_target_pen"]
                    ),
                    "subject_target_frames": stream["subject_target_frames"],
                    "max_robot_target_pen": stream["max_robot_target_pen"],
                    "robot_target_frames": stream["robot_target_frames"],
                    "prefix_max_subject_target_pen_by_body": (
                        stream[
                            "prefix_max_subject_target_pen_by_body"
                        ]
                    ),
                    "prefix_max_robot_target_pen_by_body": (
                        stream["prefix_max_robot_target_pen_by_body"]
                    ),
                    "prefix_subject_target_frames_by_body": (
                        stream["prefix_subject_target_frames_by_body"]
                    ),
                    "prefix_robot_target_frames_by_body": (
                        stream["prefix_robot_target_frames_by_body"]
                    ),
                    "initial_robot_scene_contact": (
                        stream["initial_robot_scene_contact"]
                    ),
                    "peak_global_nacon": stream["peak_global_nacon"],
                    "peak_per_world_nefc": (
                        stream["peak_per_world_nefc"]
                    ),
                    "peak_solver_niter": stream["peak_solver_niter"],
                }
                if regularizer_active:
                    regularizer_raw = control_regularizer_command_diagnostics(
                        physical_ctrl=ctrl_seq,
                        previous_executed_action=(
                            previous_executed_action_arg
                        ),
                        native_dt_s=self.exec_dt,
                        boundary_valid=regularizer_boundary_valid_arg,
                        array_module=jnp,
                    )
                    result["control_regularizer_realized_arm_qvel"] = (
                        stream[
                            "control_regularizer_arm_qvel_time_integral"
                        ]
                        / (float(ctrl_seq.shape[0]) * self.exec_dt)
                    )
                    result["control_regularizer_command_magnitude"] = (
                        regularizer_raw[
                            "command_envelope_normalized_magnitude_time_mean"
                        ]
                    )
                    result["control_regularizer_command_rate"] = (
                        regularizer_raw[
                            "boundary_complete_normalized_command_rate_time_mean"
                        ]
                    )
                if terminal_contact_endpoint_enabled:
                    result["prefix_subject_target_now_by_body"] = (
                        stream["prefix_subject_target_now_by_body"]
                    )
                    result["final_subject_target_now_by_body"] = (
                        stream["final_subject_target_now_by_body"]
                    )
                if record_step_trace:
                    # ``prefix_steps`` is a Python-static specialization.  This
                    # slice contains every 2 ms post-step state actually adopted
                    # as the offline plant prefix; it is neither re-integrated
                    # nor reconstructed from the endpoint.
                    result["prefix_qpos_trace"] = post_step_qpos[:prefix_steps]
                return result

            return jax.vmap(rollout_one)(
                ctrl_batch,
                grip_batch,
                command_latent_batch,
                jnp.arange(ctrl_batch.shape[0]),
            )

        rollout_args = (
            template, model_ctrl_j, grip_init, gripper_command_j, obj_q, z0_j,
            start_qpos, start_qpos_mask, start_qvel, start_qvel_mask,
            restore_grip, jnp.asarray(previous_action),
            jnp.asarray(control_regularizer_boundary_valid),
            jnp.asarray(prefix_index, dtype=jnp.int32),
            contact_target_geom_mask, target_geom_mask, target_geom_masks,
        )
        # Per-body prefix evidence carries one target-mask row per evidence
        # target (explicit mediation targets, a terminal Contact target, plus
        # the optional controlled subject), so evidence-target count is part
        # of the JAX layout.
        # ``prefix_steps`` is also closure-static whenever the optional exact
        # post-step trace is materialized: its slice length changes the output
        # pytree shape.  Specialize it even with tracing disabled so one cache
        # contract covers both production modes and stage schedules safely.
        cache_key = (
            N,
            H,
            naconmax,
            continued,
            len(evidence_target_names),
            bool(record_step_trace),
            prefix_steps,
        )
        out = self._execute_rollout(
            cache_key, lambda: jax.jit(rollout_batch), rollout_args)
        # JAX dispatch is asynchronous.  Synchronize at the physics boundary
        # so this metric cannot accidentally charge rollout work to the scorer
        # (or report only enqueue latency).  This keeps arrays on device; the
        # compact host transfer remains part of end-to-end time below.
        jax.block_until_ready(out)
        physics_dt = time.time() - t0
        names = [n for n, _ in self.movable_addrs]
        selected_index_host: int | None = None
        selected_knots_host: np.ndarray | None = None
        selected_proposal_host: np.ndarray | None = None
        selected_n_valid_host: int | None = None
        selected_mean_cost_host: float | None = None
        selected_flat_host: bool | None = None
        selected_validity_counts_host: dict[str, int] | None = None
        population_capacity_peaks_host: dict[str, int] | None = None
        refit_mean_host: np.ndarray | None = None
        refit_std_host: np.ndarray | None = None
        mppi_effective_samples_host: float | None = None
        mppi_next_temperature_host: float | None = None
        device_score_dt: float | None = None
        solve_diagnostics_host: dict[str, Any] | None = None
        terminal_trajectory_selection_host: dict[str, Any] | None = None
        if device_cost_fn is not None:
            if device_validity is None:
                raise ValueError(
                    "device_validity is required with device_cost_fn"
                )
            from oap.program.device_cost import DeviceCostEvaluator

            if not isinstance(device_cost_fn, DeviceCostEvaluator):
                raise TypeError(
                    "device_cost_fn must be a DeviceCostEvaluator so current "
                    "grounding is passed as dynamic JAX arguments"
                )
            cost_function = device_cost_fn.breakdown_function
            if cost_function is None:
                raise TypeError(
                    "device_cost_fn must expose breakdown_function; rebuild "
                    "it with make_device_trajectory_cost"
                )
            trajectory_cost_function = getattr(
                device_cost_fn,
                "trajectory_breakdown_function",
                None,
            )
            if (
                mppi_prefer_earliest_joint_terminal_sample
                and trajectory_cost_function is None
            ):
                raise TypeError(
                    "earliest terminal pose-sample selection requires device "
                    "trajectory_breakdown_function"
                )
            cost_params = device_cost_fn.params
            cost_spec = device_cost_fn.spec
            cost_term_descriptors = tuple(device_cost_fn.term_descriptors)
            if (
                mppi_prefer_earliest_joint_terminal_sample
                and not any(
                    descriptor.get("scope") == "terminal"
                    for descriptor in cost_term_descriptors
                )
            ):
                raise ValueError(
                    "earliest terminal pose-sample selection requires at least "
                    "one terminal predicate"
                )
            movable_names = tuple(names)

            def score_on_device(
                rollout_out,
                grounding,
                validity,
                action_knots,
                feasibility_knots,
            ):
                movable = {
                    name: rollout_out["movable_traj7"][:, :, index, :]
                    for index, name in enumerate(movable_names)
                }
                cost_args = (
                    rollout_out["pose_traj7"],
                    movable,
                    rollout_out["gripper_traj3"],
                    rollout_out["gripper_width_traj"],
                    rollout_out["object_held_traj"],
                    rollout_out["tool_target_contact_traj"],
                    rollout_out["robot_target_contact_traj"],
                    rollout_out["gripper_axis_traj3"],
                    rollout_out["gripper_closing_axis_traj3"],
                    rollout_out["object_not_held_steps"] / float(H),
                    rollout_out["subject_target_frames"] / float(H),
                    rollout_out["robot_target_contact_any_step"],
                    rollout_out["gripper_command_traj"],
                    rollout_out[
                        "robot_subject_contact_missing_fraction"
                    ],
                    rollout_out["robot_subject_contact_endpoint"],
                    grounding,
                )
                cost, per_term = cost_function(*cost_args)
                if mppi_prefer_earliest_joint_terminal_sample:
                    assert trajectory_cost_function is not None
                    terminal_trajectories = trajectory_cost_function(
                        *cost_args
                    )
                action_rate_cost = device_action_rate_cost(
                    action_knots,
                    validity["action_span"],
                    weight=validity["action_rate_weight"],
                    continuous_jaw=True,
                )
                arm_velocity_cost = device_arm_velocity_cost(
                    rollout_out["arm_velocity_squared_sum"],
                    weight=validity["arm_velocity_weight"],
                ) if paper_mppi_active else jnp.zeros_like(cost)
                cost = (
                    cost
                    + action_rate_cost
                    + arm_velocity_cost
                    + rollout_out["table_contact_force_cost"]
                )
                # V11 disturbance penalty: undeclared movables that move
                # during the rollout add cost proportional to displacement
                # and rotation vs the cycle-start pose. Keyed by name, so
                # lane order cannot drift. Finite by construction.
                _disturb = getattr(grounding, "disturb_movables", None)
                if _disturb:
                    for _di, _dname in enumerate(movable_names):
                        if _dname not in _disturb:
                            continue
                        _dw, _dstart = _disturb[_dname]
                        _dfinal = rollout_out["final_movable_pose7"][:, _di, :]
                        _dpos = jnp.linalg.norm(
                            _dfinal[:, :3] - _dstart[:3], axis=-1
                        )
                        _dq = jnp.abs(jnp.sum(
                            _dfinal[:, 3:7] * _dstart[3:7], axis=-1
                        ))
                        _drot = 2.0 * jnp.arccos(
                            jnp.clip(_dq, 0.0, 1.0)
                        )
                        cost = cost + _dw * (
                            _dpos / 0.01 + _drot / 0.1
                        )
                base_cost = cost
                if regularizer_score_active:
                    regularizer_raw = jnp.stack(
                        (
                            rollout_out[
                                "control_regularizer_realized_arm_qvel"
                            ],
                            rollout_out[
                                "control_regularizer_command_magnitude"
                            ],
                            rollout_out[
                                "control_regularizer_command_rate"
                            ],
                        ),
                        axis=0,
                    )
                    cost = control_regularizer_total_cost(
                        cost,
                        regularizer_raw,
                        profile=regularizer_profile,
                        array_module=jnp,
                    )
                arm_rate_cost, jaw_rate_cost = device_action_rate_terms(
                    action_knots,
                    validity["action_span"],
                    weight=validity["action_rate_weight"],
                    continuous_jaw=True,
                )
                max_fake_pen_m = validity["max_fake_pen_m"]
                max_arm_pen_m = validity["max_arm_pen_m"]
                floor = validity["ejection_floor_z"]
                # Reaching an arena limit means MJWarp may have clipped
                # contacts or constraints.  These are backend integrity
                # predicates, not task semantics, and therefore apply to every
                # TaskProgram uniformly.
                contact_capacity_ok = (
                    rollout_out["peak_global_nacon"]
                    < validity["naconmax"]
                )
                constraint_capacity_ok = (
                    rollout_out["peak_per_world_nefc"]
                    < validity["njmax_per_world"]
                )
                obj_table_ok = (
                    rollout_out["max_obj_table_pen"] < max_fake_pen_m
                )
                obj_other_support_ok = (
                    rollout_out["max_obj_other_support_pen"]
                    < max_fake_pen_m
                )
                pad_table_ok = (
                    rollout_out["max_pad_table_pen"] < max_fake_pen_m
                )
                arm_ok = rollout_out["max_arm_pen"] < max_arm_pen_m
                # Non-finite program cost means the fixed typed objective could
                # not be evaluated. Task progress and explicit contact
                # relations themselves remain finite.
                # Validity belongs to the typed/base objective.  The formal
                # regularizer may rank finite base-valid candidates, but may
                # never manufacture a validity failure from its own raw terms.
                # A non-finite raw term on a base-valid candidate is rejected
                # as a whole-solve integrity failure below, before selection.
                finite_cost = jnp.isfinite(base_cost)
                subject_floor_ok = jnp.all(
                    rollout_out["pose_traj7"][..., 2] >= floor,
                    axis=1,
                )
                if names:
                    movable_floor_ok = jnp.all(
                        rollout_out["movable_traj7"][..., 2] >= floor,
                        axis=(1, 2),
                    )
                else:
                    movable_floor_ok = jnp.ones_like(
                        finite_cost,
                        dtype=bool,
                    )
                (
                    controller_timing_ok,
                    joint_position_ok,
                    joint_velocity_ok,
                    gripper_width_ok,
                    gripper_velocity_ok,
                ) = device_action_feasibility(
                    feasibility_knots,
                    continuous_jaw=paper_mppi_active,
                    action_feasibility_enabled=(
                        validity["action_feasibility_enabled"]
                    ),
                    controller_timing_compatible=(
                        validity["controller_timing_compatible"]
                    ),
                    controller_prefix_knot_times=(
                        validity["controller_prefix_knot_times"]
                    ),
                    controller_segment_durations_s=(
                        validity["controller_segment_durations_s"]
                    ),
                    joint_position_min_rad=(
                        validity["joint_position_min_rad"]
                    ),
                    joint_position_max_rad=(
                        validity["joint_position_max_rad"]
                    ),
                    joint_velocity_max_rad_s=(
                        validity["joint_velocity_max_rad_s"]
                    ),
                    gripper_width_min_m=validity["gripper_width_min_m"],
                    gripper_width_max_m=validity["gripper_width_max_m"],
                    gripper_velocity_min_m_s=(
                        validity["gripper_velocity_min_m_s"]
                    ),
                    gripper_velocity_max_m_s=(
                        validity["gripper_velocity_max_m_s"]
                    ),
                )
                components = (
                    contact_capacity_ok,
                    constraint_capacity_ok,
                    obj_table_ok,
                    obj_other_support_ok,
                    pad_table_ok,
                    arm_ok,
                    finite_cost,
                    subject_floor_ok,
                    movable_floor_ok,
                    controller_timing_ok,
                    joint_position_ok,
                    joint_velocity_ok,
                    gripper_width_ok,
                    gripper_velocity_ok,
                    (
                        (~validity["require_prefix_held"])
                        | rollout_out["prefix_two_sided_now"]
                    ),
                )
                valid = components[0]
                first_fail_counts = []
                surviving = jnp.ones_like(valid, dtype=bool)
                first_fail_component = jnp.full(
                    valid.shape, -1, dtype=jnp.int32
                )
                for component_index, component in enumerate(components):
                    first_fail_counts.append(
                        jnp.sum(
                            surviving & (~component),
                            dtype=jnp.int32,
                        )
                    )
                    first_fail_component = jnp.where(
                        surviving & (~component),
                        jnp.asarray(component_index, dtype=jnp.int32),
                        first_fail_component,
                    )
                    surviving = surviving & component
                pass_counts = [
                    jnp.sum(component, dtype=jnp.int32)
                    for component in components
                ]
                valid = surviving
                validity_counts = jnp.stack(
                    pass_counts + first_fail_counts,
                    axis=0,
                )
                scored = (
                    cost,
                    valid,
                    validity_counts,
                    per_term,
                    arm_rate_cost,
                    jaw_rate_cost,
                    arm_velocity_cost,
                    rollout_out["table_contact_force_cost"],
                    first_fail_component,
                )
                if mppi_prefer_earliest_joint_terminal_sample:
                    scored = scored + (terminal_trajectories,)
                if regularizer_score_active:
                    scored = scored + (base_cost,)
                return scored

            score_t0 = time.perf_counter()
            action_enabled = bool(
                device_validity.get("action_feasibility_enabled", False)
            )
            validity_params = {
                "naconmax": jnp.asarray(naconmax, dtype=jnp.int32),
                "njmax_per_world": jnp.asarray(
                    self._NJMAX_PER_WORLD,
                    dtype=jnp.int32,
                ),
                "max_fake_pen_m": jnp.asarray(device_validity["max_fake_pen_m"]),
                "max_arm_pen_m": jnp.asarray(device_validity["max_arm_pen_m"]),
                "ejection_floor_z": jnp.asarray(device_validity["ejection_floor_z"]),
                "action_span": jnp.asarray(
                    np.asarray(ctrl_high, dtype=float)
                    - np.asarray(ctrl_low, dtype=float)
                    if device_select
                    else np.ones(NJ + 1, dtype=float)
                ),
                "action_rate_weight": jnp.asarray(
                    DEFAULT_ACTION_RATE_WEIGHT
                ),
                "arm_velocity_weight": jnp.asarray(arm_velocity_weight),
                "action_feasibility_enabled": jnp.asarray(action_enabled),
                "require_prefix_held": jnp.asarray(
                    bool(device_validity.get("require_prefix_held", False))
                ),
                "controller_timing_compatible": jnp.asarray(
                    device_validity.get("controller_timing_compatible", True)
                ),
                "controller_prefix_knot_times": jnp.asarray(
                    device_validity.get(
                        "controller_prefix_knot_times",
                        [0.0, float(prefix_index) / float(H - 1)],
                    )
                ),
                "controller_segment_durations_s": jnp.asarray(
                    device_validity.get(
                        "controller_segment_durations_s",
                        [
                            execution_steps
                            * float(device_validity.get(
                                "execution_dt_s", self.exec_dt
                            ))
                        ],
                    )
                ),
                "joint_position_min_rad": jnp.asarray(
                    device_validity.get("joint_position_min_rad", [-jnp.inf] * NJ)
                ),
                "joint_position_max_rad": jnp.asarray(
                    device_validity.get("joint_position_max_rad", [jnp.inf] * NJ)
                ),
                "joint_velocity_max_rad_s": jnp.asarray(
                    device_validity.get("joint_velocity_max_rad_s", [jnp.inf] * NJ)
                ),
                "gripper_width_min_m": jnp.asarray(
                    device_validity.get("gripper_width_min_m", -jnp.inf)
                ),
                "gripper_width_max_m": jnp.asarray(
                    device_validity.get("gripper_width_max_m", jnp.inf)
                ),
                "gripper_velocity_min_m_s": jnp.asarray(
                    device_validity.get("gripper_velocity_min_m_s", 0.0)
                ),
                "gripper_velocity_max_m_s": jnp.asarray(
                    device_validity.get("gripper_velocity_max_m_s", jnp.inf)
                ),
            }
            action_knots = (
                proposal_knots
                if proposal_knots is not None
                else jnp.asarray(np.stack(knot_arrays, axis=0))
            )
            if paper_mppi_active:
                # Both explicit experiment profiles return realized arm qpos
                # as ControlKnots.  Their action-space winner/refit arrays are
                # carried separately and never interpreted as positions.
                feasibility_knots = mppi_realized_control_knots(
                    action_knots,
                    out["arm_action_endpoint_qpos"],
                    concatenate=jnp.concatenate,
                )
            else:
                feasibility_knots = action_knots
            score_args = (
                out,
                cost_params,
                validity_params,
                action_knots,
                feasibility_knots,
            )
            executable = self._device_score_executable(
                spec=(
                    (cost_spec, "result_terminal_earliest")
                    if mppi_prefer_earliest_joint_terminal_sample
                    else cost_spec
                ),
                movable_names=movable_names,
                make_jitted=lambda: jax.jit(score_on_device),
                args=score_args,
            )
            scored_device = executable(*score_args)
            (
                program_cost,
                program_valid,
                validity_counts_device,
                per_term_device,
                arm_rate_device,
                jaw_rate_device,
                arm_velocity_device,
                table_force_device,
                first_fail_component_device,
            ) = scored_device[:9]
            terminal_trajectories_device = (
                scored_device[9]
                if mppi_prefer_earliest_joint_terminal_sample
                else None
            )
            regularizer_base_cost_device = (
                scored_device[
                    10 if mppi_prefer_earliest_joint_terminal_sample else 9
                ]
                if regularizer_score_active
                else None
            )
            if regularizer_score_active:
                regularizer_raw_for_gate = jnp.stack(
                    (
                        out["control_regularizer_realized_arm_qvel"],
                        out["control_regularizer_command_magnitude"],
                        out["control_regularizer_command_rate"],
                    ),
                    axis=0,
                )
                base_valid_raw_nonfinite = (
                    control_regularizer_base_valid_raw_violation(
                        regularizer_raw_for_gate,
                        program_valid,
                        array_module=jnp,
                    )
                )
                if bool(np.asarray(base_valid_raw_nonfinite)):
                    raise RuntimeError(
                        "formal control regularizer observed non-finite raw "
                        "terms on a base-valid candidate"
                    )
            if _dial_annealed_proposal is not None:
                base_valid_total_nonfinite = jnp.any(
                    program_valid & (~jnp.isfinite(program_cost))
                )
                if bool(np.asarray(base_valid_total_nonfinite)):
                    raise RuntimeError(
                        "DIAL observed non-finite total cost on a base-valid "
                        "candidate"
                    )
            jax.block_until_ready(
                (program_cost, program_valid, validity_counts_device)
            )
            device_score_dt = time.perf_counter() - score_t0
            select_t0 = time.perf_counter()
            select_refit_sync_dt: float | None = None
            compact_keys = (
                "final_pose7",
                "prefix_pose7",
                "prefix_qpos",
                "prefix_qvel",
                "arm_action_endpoint_qpos",
                "prefix_ctrl",
                "prefix_carrier_ctrl",
                "prefix_gripper_command",
                "final_movable_pose7",
                "prefix_movable_pose7",
                "two_sided_frames",
                "prefix_two_sided",
                "prefix_two_sided_now",
                "prefix_two_sided_frames",
                "prefix_lift",
                "final_two_sided",
                "final_two_sided_now",
                "max_lift",
                "final_lift",
                "max_obj_world_pen",
                "max_obj_table_pen",
                "max_obj_other_support_pen",
                "max_subject_contact_target_pen",
                "max_pad_table_pen",
                "max_obj_pad_pen",
                "max_arm_pen",
                "table_contact_force_cost",
                "max_table_contact_force",
                "prefix_max_table_contact_force",
                "max_subject_movable_pen",
                "subject_movable_frames",
                "prefix_max_subject_movable_pen",
                "prefix_subject_movable_frames",
                "max_robot_movable_pen",
                "robot_movable_frames",
                "prefix_max_robot_movable_pen",
                "prefix_robot_movable_frames",
                "max_subject_target_pen",
                "subject_target_frames",
                "max_robot_target_pen",
                "robot_target_frames",
                "prefix_max_subject_target_pen_by_body",
                "prefix_max_robot_target_pen_by_body",
                "prefix_subject_target_frames_by_body",
                "prefix_robot_target_frames_by_body",
                "initial_robot_scene_contact",
                "peak_global_nacon",
                "peak_per_world_nefc",
                "peak_solver_niter",
                "object_not_held_steps",
                "execution_prefix_steps",
                "prefix_object_not_held_steps",
                "tool_target_contact_any_step",
                "robot_target_contact_any_step",
                "robot_subject_contact_frames",
                "robot_subject_contact_suffix_frames",
                "robot_subject_contact_missing_fraction",
                "robot_subject_contact_endpoint",
                "prefix_robot_subject_contact_frames",
                "prefix_robot_subject_contact_now",
            )
            if terminal_contact_endpoint_enabled:
                compact_keys = compact_keys + (
                    "prefix_subject_target_now_by_body",
                    "final_subject_target_now_by_body",
                )
            if record_step_trace:
                compact_keys = compact_keys + ("prefix_qpos_trace",)
            compact_out = {key: out[key] for key in compact_keys}
            # Added experiment telemetry stays optional at the retained-output
            # seam so existing native-executable test packets and production
            # default schemas remain compatible. The real MPPI body above
            # always emits both fields for control-profile runs.
            if self.experiment_control_profile:
                for key in (
                    "prefix_gripper_command_latent",
                    "prefix_gripper_actuator_command",
                ):
                    if key in out:
                        compact_out[key] = out[key]
            compact_out["object_not_held_fraction"] = (
                out["object_not_held_steps"] / float(H)
            )
            if device_select:
                if proposal_knots is None:
                    raise RuntimeError("device proposal population is missing")
                population_capacity_peaks = jnp.stack(
                    (
                        jnp.max(compact_out["peak_global_nacon"]),
                        jnp.max(compact_out["peak_per_world_nefc"]),
                        jnp.max(compact_out["peak_solver_niter"]),
                        jnp.sum(
                            compact_out["max_subject_contact_target_pen"]
                            >= validity_params["max_fake_pen_m"],
                            dtype=jnp.int32,
                        ),
                    )
                ).astype(jnp.int32)
                (
                    selected_index,
                    selected_n_valid,
                    selected_mean_cost,
                    selected_flat,
                ) = stable_device_selection(program_cost, program_valid)
                if mppi_prefer_earliest_joint_terminal_sample:
                    assert terminal_trajectories_device is not None
                    expected_terminal_count = sum(
                        descriptor.get("scope") == "terminal"
                        for descriptor in cost_term_descriptors
                    )
                    if terminal_trajectories_device.shape != (
                        expected_terminal_count,
                        program_cost.shape[0],
                        POSE_SAMPLES,
                    ):
                        raise RuntimeError(
                            "terminal trajectory diagnostics must have exact "
                            "shape (n_terminal,N,H), got "
                            f"{terminal_trajectories_device.shape}"
                        )
                    (
                        selected_index,
                        endpoint_joint_count_device,
                        selected_first_hit_device,
                        first_hits_device,
                    ) = stable_terminal_trajectory_selection(
                        program_cost,
                        program_valid,
                        terminal_trajectories_device,
                    )
                mppi_effective_samples_device = jnp.asarray(
                    jnp.nan, dtype=program_cost.dtype
                )
                mppi_next_temperature_device = jnp.asarray(
                    jnp.nan, dtype=program_cost.dtype
                )
                if _cem_sampling_proposal is not None:
                    if (
                        cem_previous_mean is None
                        or cem_previous_std is None
                        or cem_min_std is None
                    ):
                        raise RuntimeError("CEM distribution parameters are absent")
                    refit_mean_device, refit_std_device = cem_elite_refit(
                        proposal_knots,
                        program_cost,
                        program_valid,
                        previous_mean=cem_previous_mean,
                        previous_std=cem_previous_std,
                        elite_count=cem_elite_count,
                        min_std=cem_min_std,
                        jaw_bernoulli=bool(cem_jaw_bernoulli),
                        jaw_p_floor=float(cem_jaw_p_floor),
                    )
                elif _mppi_sampling_proposal is not None:
                    if cem_previous_mean is None or cem_previous_std is None:
                        raise RuntimeError("MPPI distribution parameters are absent")
                    if (
                        mppi_proposal_mode
                        == PREDICTIVE_SAMPLER_MTP_GLOBAL_LOCAL
                    ):
                        refit_mean_device = mtp_elite_weighted_mean(
                            proposal_knots,
                            program_cost,
                            program_valid,
                            previous_mean=cem_previous_mean,
                            temperature=mppi_temperature,
                            elite_count=mtp_elite_count,
                        )
                    else:
                        (
                            refit_mean_device,
                            mppi_effective_samples_device,
                            mppi_next_temperature_device,
                        ) = mppi_weighted_update(
                            proposal_knots,
                            program_cost,
                            program_valid,
                            previous_mean=cem_previous_mean,
                            temperature=mppi_temperature,
                        )
                    refit_std_device = jnp.asarray(cem_previous_std)
                elif _dial_annealed_proposal is not None:
                    if cem_previous_mean is None or cem_previous_std is None:
                        raise RuntimeError("DIAL distribution parameters are absent")
                    (
                        refit_mean_device,
                        mppi_effective_samples_device,
                        mppi_next_temperature_device,
                    ) = dial_fixed_temperature_update(
                        proposal_knots,
                        program_cost,
                        program_valid,
                        previous_mean=cem_previous_mean,
                        temperature=mppi_temperature,
                        # The formal regularizer integrity gate above proves
                        # base-valid raw/weighted costs finite without a full
                        # population host transfer. The orchestration seam
                        # separately refuses n_valid==0 after this device run.
                        enforce_host_contract=False,
                    )
                    refit_std_device = jnp.asarray(cem_previous_std)
                else:
                    refit_mean_device = proposal_knots[selected_index]
                    refit_std_device = jnp.zeros_like(refit_mean_device)
                # The gather is indexed by a device scalar. No proposal,
                # rollout, cost, or validity vector is materialized on the
                # host. A singleton leading dimension lets the existing row
                # serializer remain the sole schema conversion path.
                compact_out = {
                    key: value[selected_index][None, ...]
                    for key, value in compact_out.items()
                }
                compact_out["device_program_cost"] = (
                    program_cost[selected_index][None]
                )
                compact_out["device_program_valid"] = (
                    program_valid[selected_index][None]
                )
                selected_knots_device = feasibility_knots[selected_index]
                selected_proposal_device = proposal_knots[selected_index]
                # Population-level outcome flags, kept BEFORE the winner
                # gather so the joint feasibility sets describe every
                # candidate the round actually evaluated.
                population_prefix_held = out["prefix_two_sided_now"]
                population_endpoint_held = out["final_two_sided_now"]
                population_endpoint_command = out["final_gripper_command"]
                transfer = (
                    compact_out,
                    selected_knots_device,
                    selected_proposal_device,
                    selected_index,
                    selected_n_valid,
                    selected_mean_cost,
                    selected_flat,
                    validity_counts_device,
                    population_capacity_peaks,
                    refit_mean_device,
                    refit_std_device,
                    # Small fixed-size diagnostic vectors: (n_terms, N) plus
                    # a handful of (N,) arrays. Never rollout state; never a
                    # second objective.
                    per_term_device,
                    arm_rate_device,
                    jaw_rate_device,
                    arm_velocity_device,
                    table_force_device,
                    first_fail_component_device,
                    program_cost,
                    program_valid,
                    population_prefix_held,
                    population_endpoint_held,
                    population_endpoint_command,
                    mppi_effective_samples_device,
                    mppi_next_temperature_device,
                )
                control_regularizer_raw_device = None
                control_regularizer_base_cost_transfer_device = None
                if regularizer_active:
                    raw_population = jnp.stack(
                        (
                            out["control_regularizer_realized_arm_qvel"],
                            out["control_regularizer_command_magnitude"],
                            out["control_regularizer_command_rate"],
                        ),
                        axis=0,
                    )
                    control_regularizer_raw_device = (
                        raw_population
                        if record_candidate_cost_telemetry
                        else raw_population[:, selected_index][:, None]
                    )
                    transfer = transfer + (
                        control_regularizer_raw_device,
                    )
                    if regularizer_score_active:
                        assert regularizer_base_cost_device is not None
                        control_regularizer_base_cost_transfer_device = (
                            regularizer_base_cost_device
                            if record_candidate_cost_telemetry
                            else regularizer_base_cost_device[
                                selected_index
                            ][None]
                        )
                        transfer = transfer + (
                            control_regularizer_base_cost_transfer_device,
                        )
                if mppi_prefer_earliest_joint_terminal_sample:
                    transfer = transfer + (
                        endpoint_joint_count_device,
                        selected_first_hit_device,
                        first_hits_device,
                    )
                if _dial_annealed_proposal is not None:
                    if dial_clipping_count_device is None:
                        raise RuntimeError("DIAL clipping telemetry is absent")
                    transfer = transfer + (dial_clipping_count_device,)
                jax.block_until_ready(transfer)
                select_refit_sync_dt = time.perf_counter() - select_t0
                out = {
                    key: np.asarray(value)
                    for key, value in compact_out.items()
                }
                selected_knots_host = np.asarray(selected_knots_device)
                selected_proposal_host = np.asarray(selected_proposal_device)
                selected_index_host = int(np.asarray(selected_index))
                selected_n_valid_host = int(np.asarray(selected_n_valid))
                selected_mean_cost_host = float(
                    np.asarray(selected_mean_cost)
                )
                selected_flat_host = bool(np.asarray(selected_flat))
                if (
                    _cem_sampling_proposal is not None
                    or _mppi_sampling_proposal is not None
                    or _dial_annealed_proposal is not None
                ):
                    refit_mean_host = np.asarray(refit_mean_device)
                    refit_std_host = np.asarray(refit_std_device)
                if (
                    _mppi_sampling_proposal is not None
                    or _dial_annealed_proposal is not None
                ):
                    mppi_effective_samples_host = float(
                        np.asarray(mppi_effective_samples_device)
                    )
                    mppi_next_temperature_host = float(
                        np.asarray(mppi_next_temperature_device)
                    )
                if mppi_prefer_earliest_joint_terminal_sample:
                    endpoint_joint_count = int(
                        np.asarray(endpoint_joint_count_device)
                    )
                    selected_first_hit = int(
                        np.asarray(selected_first_hit_device)
                    )
                    first_hits = np.asarray(
                        first_hits_device,
                        dtype=np.int32,
                    )
                    sample_step_indices = physics_sample_indices(H)
                    terminal_trajectory_selection_host = {
                        "schema": "terminal_trajectory_selection_v1",
                        "mode": "result_terminal_earliest",
                        "pose_sample_count": POSE_SAMPLES,
                        "pose_sample_native_step_indices": (
                            sample_step_indices.tolist()
                        ),
                        "no_hit_sentinel": POSE_SAMPLES,
                        "first_joint_hit_sample_indices": first_hits.tolist(),
                        "endpoint_joint_candidate_count": (
                            endpoint_joint_count
                        ),
                        "any_pose_sample_joint_hit_count": int(
                            np.count_nonzero(first_hits < POSE_SAMPLES)
                        ),
                        "requested_candidate_index": (
                            selected_index_host
                            if endpoint_joint_count > 0
                            else None
                        ),
                        "applied_candidate_index": selected_index_host,
                        "selected_first_joint_hit_sample": selected_first_hit,
                        "selected_first_joint_hit_native_step": (
                            int(sample_step_indices[selected_first_hit])
                            if endpoint_joint_count > 0
                            else None
                        ),
                        "fallback_applied": endpoint_joint_count == 0,
                        "family": (
                            "terminal_joint_earliest_pose_sample"
                            if endpoint_joint_count > 0
                            else "stable_device_selection_fallback"
                        ),
                        "selection_order": [
                            "endpoint_joint_feasible",
                            "earliest_pose_sample_joint_hit",
                            "lowest_total_cost",
                            "stable_candidate_index",
                        ],
                        "semantics": (
                            "task_specific_engineering_inference_not_"
                            "theoretical_objective_equivalence"
                        ),
                    }
                validity_counts_array = np.asarray(
                    validity_counts_device,
                    dtype=np.int64,
                )
                validity_component_names = control_profile_validity_component_names(
                    self.control_profile
                )
                expected_count = 2 * len(validity_component_names)
                if validity_counts_array.shape != (expected_count,):
                    raise RuntimeError(
                        "device validity telemetry has shape "
                        f"{validity_counts_array.shape}, expected "
                        f"({expected_count},)"
                    )
                selected_validity_counts_host = {
                    **{
                        f"pass_{name}": int(validity_counts_array[index])
                        for index, name in enumerate(
                            validity_component_names
                        )
                    },
                    **{
                        f"first_fail_{name}": int(
                            validity_counts_array[
                                len(validity_component_names) + index
                            ]
                        )
                        for index, name in enumerate(
                            validity_component_names
                        )
                    },
                }
                population_capacity_peaks_array = np.asarray(
                    population_capacity_peaks,
                    dtype=np.int64,
                )
                if population_capacity_peaks_array.shape != (4,):
                    raise RuntimeError(
                        "device capacity peak telemetry has shape "
                        f"{population_capacity_peaks_array.shape}, expected "
                        "(4,)"
                    )
                population_capacity_peaks_host = {
                    "population_peak_global_nacon": int(
                        population_capacity_peaks_array[0]
                    ),
                    "population_peak_per_world_nefc": int(
                        population_capacity_peaks_array[1]
                    ),
                    "population_peak_solver_niter": int(
                        population_capacity_peaks_array[2]
                    ),
                    "population_subject_contact_target_over_world_pen_limit": (
                        int(population_capacity_peaks_array[3])
                    ),
                }
                regularizer_weighted_host: np.ndarray | None = None
                regularizer_base_cost_host: np.ndarray | None = None
                if regularizer_score_active:
                    assert control_regularizer_raw_device is not None
                    regularizer_raw_host = np.asarray(
                        control_regularizer_raw_device,
                        dtype=float,
                    )
                    regularizer_weighted_host = np.asarray(
                        control_regularizer_weighted_terms(
                            regularizer_raw_host,
                            profile=regularizer_profile,
                            array_module=np,
                        ),
                        dtype=float,
                    )
                    assert (
                        control_regularizer_base_cost_transfer_device
                        is not None
                    )
                    regularizer_base_cost_host = np.asarray(
                        control_regularizer_base_cost_transfer_device,
                        dtype=float,
                    )
                solve_diagnostics_host = solve_term_diagnostics(
                    joint_sets=joint_feasibility_sets(
                        term_descriptors=cost_term_descriptors,
                        per_term=np.asarray(per_term_device, dtype=float),
                        cost=np.asarray(program_cost, dtype=float),
                        valid=np.asarray(program_valid, dtype=bool),
                        prefix_held=np.asarray(
                            population_prefix_held, dtype=bool
                        ),
                        endpoint_held=np.asarray(
                            population_endpoint_held, dtype=bool
                        ),
                        endpoint_jaw_effort_latent=np.asarray(
                            population_endpoint_command, dtype=float
                        ),
                        selected_index=selected_index_host,
                    ),
                    term_descriptors=cost_term_descriptors,
                    running_scales=np.asarray(
                        cost_params.running_scales, dtype=float
                    ),
                    terminal_scales=np.asarray(
                        cost_params.terminal_scales, dtype=float
                    ),
                    per_term=np.asarray(per_term_device, dtype=float),
                    arm_rate=np.asarray(arm_rate_device, dtype=float),
                    jaw_rate=np.asarray(jaw_rate_device, dtype=float),
                    arm_velocity=np.asarray(
                        arm_velocity_device, dtype=float
                    ),
                    table_force=np.asarray(
                        table_force_device, dtype=float
                    ),
                    first_fail_component=np.asarray(
                        first_fail_component_device, dtype=np.int32
                    ),
                    validity_counts=selected_validity_counts_host,
                    control_profile=self.control_profile,
                    arm_velocity_weight=arm_velocity_weight,
                    record_candidate_cost_telemetry=(
                        bool(record_candidate_cost_telemetry)
                    ),
                    control_regularizer_calibration_profile=(
                        regularizer_profile
                    ),
                    control_regularizer_raw=(
                        np.asarray(control_regularizer_raw_device, dtype=float)
                        if control_regularizer_raw_device is not None
                        else None
                    ),
                    control_regularizer_base_cost=(
                        regularizer_base_cost_host
                    ),
                    control_regularizer_weighted=(
                        regularizer_weighted_host
                    ),
                    control_regularizer_boundary_valid=(
                        control_regularizer_boundary_valid
                    ),
                    cost=np.asarray(program_cost, dtype=float),
                    valid=np.asarray(program_valid, dtype=bool),
                    selected_index=selected_index_host,
                    elite_count=(
                        int(cem_elite_count)
                        if _cem_sampling_proposal is not None
                        else 0
                    ),
                )
                solve_diagnostics_host["latency_s"] = {
                    "physics_rollout": float(physics_dt),
                    "device_score": float(device_score_dt),
                    "select_refit_sync": (
                        float(select_refit_sync_dt)
                        if select_refit_sync_dt is not None
                        else None
                    ),
                }
                solve_diagnostics_host["production_runtime"] = (
                    self.production_runtime_evidence()
                )
                if terminal_trajectory_selection_host is not None:
                    solve_diagnostics_host["terminal_trajectory_selection"] = (
                        terminal_trajectory_selection_host
                    )
                if _dial_annealed_proposal is not None:
                    assert dial_clipping_count_device is not None
                    solve_diagnostics_host["dial_proposal"] = {
                        "schema": "dial_annealed_proposal_v1",
                        "preclip_out_of_bounds_coordinate_count": int(
                            np.asarray(dial_clipping_count_device)
                        ),
                        "population_coordinate_count": int(N * knot_count * 8),
                        "clipping_policy": "elementwise_optimizer_action_bounds",
                    }
            else:
                compact_out["device_program_cost"] = program_cost
                compact_out["device_program_valid"] = program_valid
                out = {
                    key: np.asarray(value)
                    for key, value in compact_out.items()
                }
            logger.info(
                "[jointspace-screen] device program cost + validity in %.3fs",
                device_score_dt,
            )
        else:
            out = {k: np.asarray(v) for k, v in out.items()}
        dt = time.time() - t0
        total_rollouts = N
        logger.info(
            "[jointspace-screen] %d candidates x %d steps: physics %.2fs, "
            "device-total %.2fs (%.1f rollouts/s)",
            total_rollouts,
            H,
            physics_dt,
            dt,
            total_rollouts / max(dt, 1e-9),
        )
        row_count = 1 if device_select else N
        rows = {i: {k: (out[k][i].tolist() if out[k][i].ndim else float(out[k][i]))
                    for k in out} for i in range(row_count)}
        for row in rows.values():
            row["contact_target_bodies"] = list(contact_target_names)
            row["mediation_target_bodies"] = list(target_names)
            row["contact_evidence_target_bodies"] = list(
                evidence_target_names
            )
            row["initial_robot_scene_contact"] = bool(
                row["initial_robot_scene_contact"]
            )
            row["tool_target_contact_any_step"] = bool(
                row["tool_target_contact_any_step"]
            )
            row["robot_target_contact_any_step"] = bool(
                row["robot_target_contact_any_step"]
            )
            row["robot_subject_contact_endpoint"] = bool(
                row["robot_subject_contact_endpoint"]
            )
            row["prefix_robot_subject_contact_now"] = bool(
                row["prefix_robot_subject_contact_now"]
            )
            row["robot_subject_contact_frames"] = int(
                row["robot_subject_contact_frames"]
            )
            row["robot_subject_contact_suffix_frames"] = int(
                row["robot_subject_contact_suffix_frames"]
            )
            row["prefix_robot_subject_contact_frames"] = int(
                row["prefix_robot_subject_contact_frames"]
            )
            row["robot_subject_contact_missing_fraction"] = float(
                row["robot_subject_contact_missing_fraction"]
            )
            row["peak_global_nacon"] = int(
                row["peak_global_nacon"]
            )
            row["peak_per_world_nefc"] = int(
                row["peak_per_world_nefc"]
            )
            row["peak_solver_niter"] = int(
                row["peak_solver_niter"]
            )
            row["contact_capacity"] = int(naconmax)
            row["constraint_capacity_per_world"] = int(
                self._NJMAX_PER_WORLD
            )
            row["batch_physics_elapsed_s"] = float(physics_dt)
            row["batch_device_score_elapsed_s"] = (
                float(device_score_dt)
                if device_score_dt is not None
                else None
            )
            row["batch_device_total_elapsed_s"] = float(dt)
            row["object_not_held_steps"] = int(
                row["object_not_held_steps"]
            )
            row["object_not_held_fraction"] = float(
                row.get(
                    "object_not_held_fraction",
                    row["object_not_held_steps"] / float(H),
                )
            )
            row["execution_prefix_steps"] = int(
                row["execution_prefix_steps"]
            )
            row["prefix_object_not_held_steps"] = int(
                row["prefix_object_not_held_steps"]
            )
        # Positional stacks -> {name: pose7}. The scorer must not have to know
        # the address order to read a body back; a mapping cannot silently
        # transpose two objects the way an index can.
        for i, row in rows.items():
            for key in ("final_movable_pose7", "prefix_movable_pose7"):
                if key in out:
                    mov_f = np.asarray(out[key][i], dtype=float)
                    row[key] = {
                        n: mov_f[j].tolist() for j, n in enumerate(names)
                    }
            if device_cost_fn is None:
                mov_t = np.asarray(out["movable_traj7"][i], dtype=float)
                row["movable_traj7"] = {
                    n: mov_t[:, j, :].tolist() for j, n in enumerate(names)
                }
            subject_target_frames = np.asarray(
                out["prefix_subject_target_frames_by_body"][i],
                dtype=int,
            )
            robot_target_frames = np.asarray(
                out["prefix_robot_target_frames_by_body"][i],
                dtype=int,
            )
            subject_target_pen = np.asarray(
                out["prefix_max_subject_target_pen_by_body"][i],
                dtype=float,
            )
            robot_target_pen = np.asarray(
                out["prefix_max_robot_target_pen_by_body"][i],
                dtype=float,
            )
            prefix_execution_contact = {
                "execution_backend": "mjwarp",
                "coverage": "every_mjwarp_step_in_executed_prefix",
                "subject_movable_frames": int(
                    row["prefix_subject_movable_frames"]
                ),
                "robot_movable_frames": int(
                    row["prefix_robot_movable_frames"]
                ),
                "max_subject_movable_pen_m": float(
                    row["prefix_max_subject_movable_pen"]
                ),
                "max_robot_movable_pen_m": float(
                    row["prefix_max_robot_movable_pen"]
                ),
                "contact_target_bodies": list(contact_target_names),
                "mediation_target_bodies": list(target_names),
                "contact_evidence_target_bodies": list(
                    evidence_target_names
                ),
                "subject_target_frames_by_body": {
                    name: int(subject_target_frames[j])
                    for j, name in enumerate(evidence_target_names)
                },
                "robot_target_frames_by_body": {
                    name: int(robot_target_frames[j])
                    for j, name in enumerate(evidence_target_names)
                },
                "max_subject_target_pen_m_by_body": {
                    name: float(subject_target_pen[j])
                    for j, name in enumerate(evidence_target_names)
                },
                "max_robot_target_pen_m_by_body": {
                    name: float(robot_target_pen[j])
                    for j, name in enumerate(evidence_target_names)
                },
                "subject_two_sided_frames": int(
                    row["prefix_two_sided_frames"]
                ),
                "subject_two_sided_recent": int(
                    row["prefix_two_sided"]
                ),
                "subject_two_sided_now": bool(
                    row["prefix_two_sided_now"]
                ),
                "object_not_held_steps": int(
                    row["prefix_object_not_held_steps"]
                ),
                "execution_prefix_steps": int(
                    row["execution_prefix_steps"]
                ),
            }
            if terminal_contact_endpoint_enabled:
                prefix_subject_target_now = np.asarray(
                    out["prefix_subject_target_now_by_body"][i],
                    dtype=bool,
                )
                final_subject_target_now = np.asarray(
                    out["final_subject_target_now_by_body"][i],
                    dtype=bool,
                )
                target_indices = {
                    name: evidence_target_names.index(name)
                    for name in terminal_contact_endpoint_targets
                }
                prefix_execution_contact.update({
                    "subject_target_now_by_body": {
                        name: bool(prefix_subject_target_now[index])
                        for name, index in target_indices.items()
                    },
                    "final_subject_target_now_by_body": {
                        name: bool(final_subject_target_now[index])
                        for name, index in target_indices.items()
                    },
                })
            row["prefix_execution_contact"] = prefix_execution_contact
        if device_select:
            if (
                selected_index_host is None
                or selected_knots_host is None
                or selected_n_valid_host is None
                or selected_mean_cost_host is None
                or selected_flat_host is None
                or selected_validity_counts_host is None
                or population_capacity_peaks_host is None
            ):
                raise RuntimeError("device selection did not produce compact results")
            rows[0].update(population_capacity_peaks_host)
            return PredictiveSamplingResult(
                selected_index=selected_index_host,
                selected_knots=selected_knots_host,
                selected_row=rows[0],
                pool_size=N,
                n_gaussian=n_gaussian,
                n_valid=selected_n_valid_host,
                mean_cost=selected_mean_cost_host,
                flat_objective=selected_flat_host,
                validity_counts=selected_validity_counts_host,
                sampler_mode=proposal_sampler_mode,
                n_local_gaussian=n_local_gaussian,
                n_broad_gaussian=n_broad_gaussian,
                sigma_fraction=proposal_broad_sigma_fraction,
                local_sigma_fraction=proposal_local_sigma_fraction,
                refit_mean=refit_mean_host,
                refit_std=refit_std_host,
                total_rollouts=N,
                diagnostics=solve_diagnostics_host,
                mppi_effective_samples=mppi_effective_samples_host,
                mppi_next_temperature=mppi_next_temperature_host,
                selected_proposal=selected_proposal_host,
                selected_family_override=(
                    "terminal_joint_earliest_pose_sample"
                    if (
                        terminal_trajectory_selection_host is not None
                        and not terminal_trajectory_selection_host[
                            "fallback_applied"
                        ]
                    )
                    else None
                ),
            )
        return rows

    def _knots_to_model_ctrl(
        self,
        knot_arrays: list[np.ndarray],
        H: int,
        spline_order: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(K,8) joint knots -> this model's (N,H,nu) actuator ctrl + (N,G)
        gripper qpos init. Arm targets go to the arm actuators; the jaw column
        maps to signed linkage effort, or on the legacy slide model to both
        pad slides. Shared by the warp ``run`` and the CPU
        ``cpu_rollout`` so both drive byte-identical actuator sequences.

        The final column remains normalized until after interpolation, then
        maps linearly to Newtons. ``grip_init`` is a physical mechanism seed,
        not an action.
        """
        knot_ctrl = np.stack([knots_to_ctrl(k, H, order=spline_order)
                              for k in knot_arrays], axis=0).astype(np.float32)  # (N,H,8)
        model_ctrl = np.zeros((len(knot_arrays), H, self.nu), dtype=np.float32)
        model_ctrl[..., self.arm_act_ids] = knot_ctrl[..., :NJ]
        raw_jaw = knot_ctrl[..., NJ]
        if self.gripper == "slide":
            width = width_command_ctrl(raw_jaw).astype(np.float32)
            slide = np.clip(width / GN01_OPEN_WIDTH_M * _SLIDE_OPEN_M, 0.0, _SLIDE_OPEN_M)
            model_ctrl[..., self.grip_act_ids] = slide[..., None]      # both pads together
            initial_slide = np.clip(
                width[:, :1] / GN01_OPEN_WIDTH_M * _SLIDE_OPEN_M,
                0.0,
                _SLIDE_OPEN_M,
            )
            grip_init = np.repeat(
                initial_slide, len(self.grip_qpos_addrs), axis=1
            )
        else:
            model_ctrl[..., self.grip_act_ids[0]] = mppi_continuous_effort_ctrl(
                raw_jaw
            ).astype(np.float32)
            grip_init = np.full(
                (len(knot_arrays), len(self.grip_qpos_addrs)),
                GN01_OPEN_WIDTH_M,
                dtype=np.float32,
            )
        return model_ctrl, grip_init

    def cpu_rollout(self, knots: np.ndarray, obj_pose7: np.ndarray, *,
                    n_steps: int = DEFAULT_EXECUTION_STEPS,
                    spline_order: str = "linear",
                    noslip: bool = False,
                    contact_target_bodies: tuple[str, ...] = (),
                    mediation_target_bodies: tuple[str, ...] = (),
                    ) -> dict[str, Any]:
        """Roll one candidate on this exact model as a test/parity oracle.

        Production planning never calls this method.  With ``noslip=False`` it
        differs from :meth:`run` only in compute backend; both use the same
        contact-active arm meshes, elliptic cone, actuator sequence, and initial
        state. ``noslip=True`` is retained for offline solver comparisons.
        """
        m = self.model
        d = mujoco.MjData(m)
        saved = int(m.opt.noslip_iterations)
        m.opt.noslip_iterations = 10 if noslip else 0
        try:
            # SAME exec->plan rescale as :meth:`run`: n_steps is in execution-
            # grid units, this model's timestep may be the coarser plan grid,
            # and the parity contract (same duration, same spline) must hold.
            execution_steps = validate_horizon_steps(n_steps)
            H = execution_steps if self.plan_dt == self.exec_dt else max(
                2, int(round(execution_steps * self.exec_dt / self.plan_dt)))
            model_ctrl, grip_init = self._knots_to_model_ctrl(
                [knots], H, spline_order)
            mc, gi = model_ctrl[0], grip_init[0]
            command_ctrl = knots_to_ctrl(
                knots, H, order=spline_order
            )[:, NJ]
            command_ctrl = np.clip(command_ctrl, -1.0, 1.0)
            mujoco.mj_resetData(m, d)
            oadr = self.object_qpos_addr
            d.qpos[oadr:oadr + 7] = np.asarray(obj_pose7, dtype=float)
            d.qpos[self.arm_qpos_addr] = mc[0, self.arm_act_ids]
            d.qpos[self.grip_qpos_addrs] = gi
            mujoco.mj_forward(m, d)
            z0 = float(obj_pose7[2])
            og = self.geom_ids["pick_object_collision"]
            lp = self.geom_ids["gn01_left_finger_tip_collision"]
            rp = self.geom_ids["gn01_right_finger_tip_collision"]
            tbl = self.geom_ids["lab_table"]
            supports = set(int(g) for g in self.support_gids)
            target_names = tuple(dict.fromkeys(
                str(name) for name in mediation_target_bodies if name
            ))
            contact_target_names = tuple(dict.fromkeys(
                (
                    *(str(name) for name in contact_target_bodies if name),
                    *target_names,
                )
            ))
            movable_groups = {
                name: set(int(g) for g in geoms)
                for name, geoms in self.movable_geom_groups
            }
            unknown_targets = sorted(
                set(contact_target_names) - set(movable_groups)
            )
            if unknown_targets:
                raise ValueError(
                    "CPU contact targets are not dynamic bodies in the "
                    f"rollout: {unknown_targets}; "
                    f"available={sorted(movable_groups)}"
                )
            target_groups = {
                name: movable_groups[name] for name in target_names
            }
            contact_target_geoms = set().union(
                *(movable_groups[name] for name in contact_target_names)
            )
            # Only a typed subject-contact relation exempts a support. The
            # table and every unrelated support retain obstacle semantics.
            other_support_world = supports - contact_target_geoms
            obj_world = other_support_world | {tbl}
            exact_arm = set(int(g) for g in self.exact_arm_gids)
            robot_subject_robot = set(self.robot_gids) | exact_arm
            robot_subject_subject = set(self.subject_contact_gids)
            two, objz, robot_subject_touch = [], [], []
            gripper_positions: list[np.ndarray] = []
            gripper_axes: list[np.ndarray] = []
            gripper_closing_axes: list[np.ndarray] = []
            gripper_widths: list[float] = []
            _mov_g, _rob_g = _movable_and_robot_geoms(m)
            obj_world_pen = obj_table_pen = other_support_pen = 0.0
            subject_contact_target_pen = 0.0
            pad_table_pen = arm_pen = 0.0
            subj_mov_pen = robot_mov_pen = 0.0
            subj_mov_frames = robot_mov_frames = 0
            subj_target_pen = robot_target_pen = 0.0
            subj_target_frames = robot_target_frames = 0
            subj_target_pen_by_body = {
                name: 0.0 for name in target_names
            }
            robot_target_pen_by_body = {
                name: 0.0 for name in target_names
            }
            subj_target_frames_by_body = {
                name: 0 for name in target_names
            }
            robot_target_frames_by_body = {
                name: 0 for name in target_names
            }
            for u in mc:
                d.ctrl[:] = u
                mujoco.mj_step(m, d)
                left = right = False
                _robot_subject = False
                _sm = _rm = False
                _st_targets: set[str] = set()
                _rt_targets: set[str] = set()
                for c in range(d.ncon):
                    contact = d.contact[c]
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    pair = {g1, g2}
                    contact_dist = float(contact.dist)
                    if _host_pair_touching(contact, og, lp):
                        left = True
                    elif _host_pair_touching(contact, og, rp):
                        right = True
                    pen = -contact_dist                     # >0 == penetration depth
                    if contact_dist <= 0.0:
                        if (
                            (
                                g1 in robot_subject_robot
                                and g2 in robot_subject_subject
                            )
                            or (
                                g2 in robot_subject_robot
                                and g1 in robot_subject_subject
                            )
                        ):
                            _robot_subject = True
                        if og in pair and (_mov_g & pair):
                            _sm = True
                        if (_mov_g & pair) and (
                            (_rob_g & pair) or (exact_arm & pair)
                        ):
                            _rm = True
                        for name, geoms in target_groups.items():
                            if og in pair and (geoms & pair):
                                _st_targets.add(name)
                            if (geoms & pair) and (
                                (_rob_g & pair) or (exact_arm & pair)
                            ):
                                _rt_targets.add(name)
                    if pen > 0.0:
                        if og in pair and (obj_world & pair):
                            obj_world_pen = max(obj_world_pen, pen)
                        if og in pair and tbl in pair:
                            obj_table_pen = max(obj_table_pen, pen)
                        if og in pair and (other_support_world & pair):
                            other_support_pen = max(other_support_pen, pen)
                        if og in pair and (contact_target_geoms & pair):
                            subject_contact_target_pen = max(
                                subject_contact_target_pen,
                                pen,
                            )
                        if tbl in pair and ({lp, rp} & pair):
                            pad_table_pen = max(pad_table_pen, pen)
                        if g1 in exact_arm or g2 in exact_arm:
                            arm_pen = max(arm_pen, pen)
                        if og in pair and (_mov_g & pair):
                            subj_mov_pen = max(subj_mov_pen, pen)
                        if (_mov_g & pair) and (
                            (_rob_g & pair) or (exact_arm & pair)
                        ):
                            robot_mov_pen = max(robot_mov_pen, pen)
                        for name, geoms in target_groups.items():
                            if og in pair and (geoms & pair):
                                subj_target_pen = max(subj_target_pen, pen)
                                subj_target_pen_by_body[name] = max(
                                    subj_target_pen_by_body[name],
                                    pen,
                                )
                            if (geoms & pair) and (
                                (_rob_g & pair) or (exact_arm & pair)
                            ):
                                robot_target_pen = max(
                                    robot_target_pen,
                                    pen,
                                )
                                robot_target_pen_by_body[name] = max(
                                    robot_target_pen_by_body[name],
                                    pen,
                                )
                subj_mov_frames += int(_sm)
                robot_mov_frames += int(_rm)
                subj_target_frames += int(bool(_st_targets))
                robot_target_frames += int(bool(_rt_targets))
                for name in _st_targets:
                    subj_target_frames_by_body[name] += 1
                for name in _rt_targets:
                    robot_target_frames_by_body[name] += 1
                two.append(left and right)
                robot_subject_touch.append(_robot_subject)
                objz.append(float(d.qpos[oadr + 2]))
                gripper_positions.append(
                    np.array(d.site_xpos[self.site_id], dtype=float)
                )
                gripper_axes.append(
                    np.array(
                        d.site_xmat[self.site_id],
                        dtype=float,
                    ).reshape(3, 3)[:, 2]
                )
                gripper_closing_axes.append(
                    np.array(
                        d.site_xmat[self.site_id],
                        dtype=float,
                    ).reshape(3, 3)[:, 1]
                )
                physical_width = float(d.qpos[self.grip_qpos_addrs[0]])
                if self.gripper == "slide":
                    physical_width = (
                        physical_width
                        / _SLIDE_OPEN_M
                        * GN01_OPEN_WIDTH_M
                    )
                gripper_widths.append(physical_width)
        finally:
            m.opt.noslip_iterations = saved
        two = np.asarray(two)
        robot_subject_touch_array = np.asarray(
            robot_subject_touch,
            dtype=bool,
        )
        gripper_path = np.asarray(gripper_positions, dtype=float)
        gripper_axis_path = np.asarray(gripper_axes, dtype=float)
        gripper_closing_axis_path = np.asarray(
            gripper_closing_axes,
            dtype=float,
        )
        gripper_width_path = np.asarray(gripper_widths, dtype=float)
        gripper_sample = physics_sample_indices(len(gripper_path))
        return {
            "final_pose7": np.asarray(d.qpos[oadr:oadr + 7], dtype=float).tolist(),
            "gripper_traj3": gripper_path[gripper_sample].tolist(),
            "final_gripper_center": gripper_path[-1].tolist(),
            "gripper_axis_traj3": gripper_axis_path[gripper_sample].tolist(),
            "final_gripper_axis": gripper_axis_path[-1].tolist(),
            "gripper_closing_axis_traj3": (
                gripper_closing_axis_path[gripper_sample].tolist()
            ),
            "final_gripper_closing_axis": (
                gripper_closing_axis_path[-1].tolist()
            ),
            "gripper_width_traj": (
                gripper_width_path[gripper_sample].tolist()
            ),
            "final_gripper_width": float(gripper_width_path[-1]),
            "gripper_command_traj": command_ctrl[gripper_sample].tolist(),
            "final_gripper_command": float(command_ctrl[-1]),
            "object_held_traj": two[gripper_sample].tolist(),
            "final_movable_pose7": {
                n: np.asarray(d.qpos[a:a + 7], dtype=float).tolist()
                for n, a in self.movable_addrs},
            "two_sided_frames": float(two.sum()),
            "final_two_sided": float(two[-5:].sum()),
            "final_two_sided_now": bool(two[-1]),
            "robot_subject_contact_frames": int(
                robot_subject_touch_array.sum()
            ),
            "robot_subject_contact_missing_fraction": float(
                np.mean(~robot_subject_touch_array)
            ),
            "robot_subject_contact_endpoint": bool(
                robot_subject_touch_array[-1]
            ),
            "final_lift": float(objz[-1] - z0),
            "max_lift": float(max(objz) - z0),
            "max_obj_world_pen": float(obj_world_pen),
            "max_obj_table_pen": float(obj_table_pen),
            "max_obj_other_support_pen": float(other_support_pen),
            "max_subject_contact_target_pen": float(
                subject_contact_target_pen
            ),
            "max_pad_table_pen": float(pad_table_pen),
            "max_arm_pen": float(arm_pen),
            "max_subject_movable_pen": float(subj_mov_pen),
            "subject_movable_frames": float(subj_mov_frames),
            "max_robot_movable_pen": float(robot_mov_pen),
            "robot_movable_frames": float(robot_mov_frames),
            "contact_target_bodies": list(contact_target_names),
            "mediation_target_bodies": list(target_names),
            "max_subject_target_pen": float(subj_target_pen),
            "subject_target_frames": float(subj_target_frames),
            "max_robot_target_pen": float(robot_target_pen),
            "robot_target_frames": float(robot_target_frames),
            "max_subject_target_pen_m_by_body": (
                subj_target_pen_by_body
            ),
            "subject_target_frames_by_body": (
                subj_target_frames_by_body
            ),
            "max_robot_target_pen_m_by_body": (
                robot_target_pen_by_body
            ),
            "robot_target_frames_by_body": (
                robot_target_frames_by_body
            ),
        }


def full_fidelity_rollout(world: Any, knots: np.ndarray, obj_pose7: np.ndarray, *,
                      n_steps: int = DEFAULT_EXECUTION_STEPS,
                      spline_order: str = "linear",
                      noslip: bool | None = None,
                      start_state: tuple[np.ndarray, np.ndarray] | None = None,
                      contact_target_bodies: tuple[str, ...] = (),
                      mediation_target_bodies: tuple[str, ...] = (),
                      ) -> dict[str, Any]:
    """Full-fidelity CPU parity oracle for tests and offline diagnostics.

    Production planning never calls this function. With ``noslip`` left at
    the model default it supplies a complete-solver comparison; with
    ``noslip=False`` it matches Warp's solver setting for backend parity tests.
    It drives the same eight actuators over the same host spline and returns
    the :meth:`BatchedRollout.run` row keys.

    ``start_state`` (a physical (qpos, qvel)) represents a continuing physical
    state -- e.g. a carry/lift while held -- instead of
    resetting the object onto the table with the gripper OPEN (which would drop
    a held object and fail every holding continuation). None keeps the fresh-grasp reset (object at
    ``obj_pose7`` on the table, arm at the plan's start).
    """
    from oap.twin.scene import set_arm_ctrl

    m, d = world.model, world.data
    _saved_noslip = int(m.opt.noslip_iterations)
    if noslip is not None:
        m.opt.noslip_iterations = _saved_noslip if noslip else 0
    obj_adr = int(world.object_qpos_addr)
    # Non-subject free bodies, from the LIVE world map (this path rolls
    # world.model itself, not a surgered copy).
    mov_addrs = tuple((n, int(a)) for n, a in
                      (getattr(world, 'movable_qpos_addr', None) or {}).items()
                      if int(a) != obj_adr)
    grip_aid = int(world.robot["gripper_actuator_id"])
    ctrl_seq = knots_to_ctrl(knots, n_steps, order=spline_order)
    command_ctrl = np.clip(ctrl_seq[:, NJ].copy(), -1.0, 1.0)
    ctrl_seq[:, NJ] = mppi_continuous_effort_ctrl(command_ctrl)
    if start_state is not None:
        # CONTINUE from the held physical state (like the executor): restore it,
        # do NOT re-seed the object/arm (they are in the state) -- the held grip
        # persists, so the certification measures the real held-continuation.
        d.qpos[:] = np.asarray(start_state[0], dtype=float)
        d.qvel[:] = np.asarray(start_state[1], dtype=float)
        mujoco.mj_forward(m, d)
        z0 = float(d.qpos[obj_adr + 2])
    else:
        # FRESH-GRASP reset: this reference is called repeatedly on the SAME
        # world.data, so residual qvel / a closed gripper from a previous roll
        # would poison the next. mj_resetData zeros qvel/act and restores qpos0
        # (gripper open); then override object + arm to this candidate's start.
        mujoco.mj_resetData(m, d)
        d.qpos[obj_adr:obj_adr + 7] = np.asarray(obj_pose7, dtype=float)
        d.qpos[world.robot["qpos_addr"]] = ctrl_seq[0, :NJ]
        mujoco.mj_forward(m, d)
        z0 = float(obj_pose7[2])

    def gid(name: str) -> int:
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    lp, rp = gid("gn01_left_finger_tip_collision"), gid("gn01_right_finger_tip_collision")
    og, tbl = gid("pick_object_collision"), gid("lab_table")
    supports = set(g for g in range(m.ngeom)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                       "support_") and (int(m.geom_contype[g]) or int(m.geom_conaffinity[g])))
    target_names = tuple(dict.fromkeys(
        str(name) for name in mediation_target_bodies if name
    ))
    contact_target_names = tuple(dict.fromkeys(
        (
            *(str(name) for name in contact_target_bodies if name),
            *target_names,
        )
    ))
    movable_groups = _movable_geom_groups(m)
    unknown_targets = sorted(
        set(contact_target_names) - set(movable_groups)
    )
    if unknown_targets:
        raise ValueError(
            "full-fidelity contact targets are not dynamic bodies in the "
            f"rollout: {unknown_targets}; available={sorted(movable_groups)}"
        )
    target_groups = {
        name: movable_groups[name] for name in target_names
    }
    contact_target_geoms = set().union(
        *(movable_groups[name] for name in contact_target_names)
    )
    other_support_world = supports - contact_target_geoms
    obj_world = other_support_world | {tbl}
    # Offline oracle: read the same exact arm meshes as the GPU production lane.
    arm_geoms = set(exact_arm_collision_geoms(m))
    _mov_g, _rob_g = _movable_and_robot_geoms(m)
    robot_subject_subject_ids, robot_subject_robot_ids = (
        robot_subject_contact_geom_roles(
            m,
            exact_arm_gids=tuple(sorted(arm_geoms)),
        )
    )
    robot_subject_subject = set(robot_subject_subject_ids)
    robot_subject_robot = set(robot_subject_robot_ids)
    subj_mov_pen = robot_mov_pen = 0.0
    subj_mov_frames = robot_mov_frames = 0
    subj_target_pen = robot_target_pen = 0.0
    subj_target_frames = robot_target_frames = 0
    subj_target_pen_by_body = {
        name: 0.0 for name in target_names
    }
    robot_target_pen_by_body = {
        name: 0.0 for name in target_names
    }
    subj_target_frames_by_body = {
        name: 0 for name in target_names
    }
    robot_target_frames_by_body = {
        name: 0 for name in target_names
    }
    two = []
    robot_subject_touch = []
    objz = []
    poses = []
    gripper_poses = []
    gripper_axes = []
    gripper_closing_axes = []
    gripper_widths = []
    mov_poses: list[dict] = []
    obj_world_pen = obj_table_pen = other_support_pen = 0.0
    subject_contact_target_pen = 0.0
    pad_table_pen = arm_pen = 0.0
    try:
        for u in ctrl_seq:
            set_arm_ctrl(d, world.robot, u[:NJ])
            d.ctrl[grip_aid] = float(u[NJ])
            mujoco.mj_step(m, d)
            left = right = False
            _robot_subject = False
            _sm = _rm = False
            _st_targets: set[str] = set()
            _rt_targets: set[str] = set()
            for c in range(d.ncon):
                contact = d.contact[c]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                pair = {g1, g2}
                contact_dist = float(contact.dist)
                if _host_pair_touching(contact, og, lp):
                    left = True
                elif _host_pair_touching(contact, og, rp):
                    right = True
                pen = -contact_dist
                if contact_dist <= 0.0:
                    if (
                        (
                            g1 in robot_subject_robot
                            and g2 in robot_subject_subject
                        )
                        or (
                            g2 in robot_subject_robot
                            and g1 in robot_subject_subject
                        )
                    ):
                        _robot_subject = True
                    if og in pair and (_mov_g & pair):
                        _sm = True
                    if (_mov_g & pair) and ((_rob_g | arm_geoms) & pair):
                        _rm = True
                    for name, geoms in target_groups.items():
                        if og in pair and (geoms & pair):
                            _st_targets.add(name)
                        if (geoms & pair) and (
                            (_rob_g | arm_geoms) & pair
                        ):
                            _rt_targets.add(name)
                if pen > 0.0:
                    if og in pair and (obj_world & pair):
                        obj_world_pen = max(obj_world_pen, pen)
                    if og in pair and tbl in pair:
                        obj_table_pen = max(obj_table_pen, pen)
                    if og in pair and (other_support_world & pair):
                        other_support_pen = max(other_support_pen, pen)
                    if og in pair and (contact_target_geoms & pair):
                        subject_contact_target_pen = max(
                            subject_contact_target_pen,
                            pen,
                        )
                    if tbl in pair and ({lp, rp} & pair):
                        pad_table_pen = max(pad_table_pen, pen)
                    if g1 in arm_geoms or g2 in arm_geoms:
                        arm_pen = max(arm_pen, pen)
                    if og in pair and (_mov_g & pair):
                        subj_mov_pen = max(subj_mov_pen, pen)
                    if (_mov_g & pair) and ((_rob_g | arm_geoms) & pair):
                        robot_mov_pen = max(robot_mov_pen, pen)
                    for name, geoms in target_groups.items():
                        if og in pair and (geoms & pair):
                            subj_target_pen = max(subj_target_pen, pen)
                            subj_target_pen_by_body[name] = max(
                                subj_target_pen_by_body[name],
                                pen,
                            )
                        if (geoms & pair) and (
                            (_rob_g | arm_geoms) & pair
                        ):
                            robot_target_pen = max(
                                robot_target_pen,
                                pen,
                            )
                            robot_target_pen_by_body[name] = max(
                                robot_target_pen_by_body[name],
                                pen,
                            )
            subj_mov_frames += int(_sm)
            robot_mov_frames += int(_rm)
            subj_target_frames += int(bool(_st_targets))
            robot_target_frames += int(bool(_rt_targets))
            for name in _st_targets:
                subj_target_frames_by_body[name] += 1
            for name in _rt_targets:
                robot_target_frames_by_body[name] += 1
            two.append(left and right)
            robot_subject_touch.append(_robot_subject)
            objz.append(float(d.qpos[obj_adr + 2]))
            # np.array (a COPY, not asarray): d.qpos is float64, so asarray would
            # return a VIEW aliasing the live buffer -- every row would collapse to
            # the FINAL pose and silently defeat the running cost.
            poses.append(np.array(d.qpos[obj_adr:obj_adr + 7], dtype=float))
            gripper_poses.append(
                np.array(d.site_xpos[int(world.site_id)], dtype=float)
            )
            gripper_axes.append(
                np.array(
                    d.site_xmat[int(world.site_id)],
                    dtype=float,
                ).reshape(3, 3)[:, 2]
            )
            gripper_closing_axes.append(
                np.array(
                    d.site_xmat[int(world.site_id)],
                    dtype=float,
                ).reshape(3, 3)[:, 1]
            )
            width_jid = (
                world.robot.get("gripper_joint_ids", {})
                .get("finger_width")
            )
            if width_jid is None:
                raise RuntimeError(
                    "world has no finger_width joint for GripperState"
                )
            width_qaddr = int(m.jnt_qposadr[int(width_jid)])
            gripper_widths.append(float(d.qpos[width_qaddr]))
            mov_poses.append({n: np.array(d.qpos[a:a + 7], dtype=float)
                              for n, a in mov_addrs})
    finally:
        # Restore the caller's NoSlip even if a step raises (consistency with
        # BatchedRollout.cpu_rollout's finally); world.data is self-healed by
        # the entry mj_resetData of the next call.
        m.opt.noslip_iterations = _saved_noslip
    two = np.asarray(two)
    robot_subject_touch_array = np.asarray(
        robot_subject_touch,
        dtype=bool,
    )
    # Match the GPU scorer's full physics-step grid (no temporal subsampling).
    # This reference is never an in-loop production backend.
    traj = np.asarray(poses, dtype=float)                     # (H, 7)
    gripper_traj = np.asarray(gripper_poses, dtype=float)     # (H, 3)
    gripper_axis_traj = np.asarray(gripper_axes, dtype=float) # (H, 3)
    gripper_closing_axis_traj = np.asarray(
        gripper_closing_axes,
        dtype=float,
    )
    gripper_width_traj = np.asarray(gripper_widths, dtype=float)
    sample = physics_sample_indices(len(traj))
    row = {
        "final_pose7": np.asarray(d.qpos[obj_adr:obj_adr + 7], dtype=float).tolist(),
        "pose_traj7": traj[sample].tolist(),
        "gripper_traj3": gripper_traj[sample].tolist(),
        "final_gripper_center": gripper_traj[-1].tolist(),
        "gripper_axis_traj3": gripper_axis_traj[sample].tolist(),
        "final_gripper_axis": gripper_axis_traj[-1].tolist(),
        "gripper_closing_axis_traj3": (
            gripper_closing_axis_traj[sample].tolist()
        ),
        "final_gripper_closing_axis": (
            gripper_closing_axis_traj[-1].tolist()
        ),
        "gripper_width_traj": gripper_width_traj[sample].tolist(),
        "final_gripper_width": float(gripper_width_traj[-1]),
        "gripper_command_traj": command_ctrl[sample].tolist(),
        "final_gripper_command": float(command_ctrl[-1]),
        "object_held_traj": two[sample].tolist(),
        # Same sample grid as the subject's, so a running-cost predicate that
        # mentions the pushed body is evaluated at the same instants.
        "final_movable_pose7": {n: np.asarray(d.qpos[a:a + 7], dtype=float).tolist()
                                for n, a in mov_addrs},
        "movable_traj7": {n: [mov_poses[k][n].tolist() for k in sample]
                          for n, _ in mov_addrs},
        "two_sided_frames": float(two.sum()),
        "final_two_sided": float(two[-5:].sum()),
        "final_two_sided_now": bool(two[-1]),
        "robot_subject_contact_frames": int(
            robot_subject_touch_array.sum()
        ),
        "robot_subject_contact_missing_fraction": float(
            np.mean(~robot_subject_touch_array)
        ),
        "robot_subject_contact_endpoint": bool(
            robot_subject_touch_array[-1]
        ),
        "final_lift": float(objz[-1] - z0),
        "max_lift": float(max(objz) - z0),
        "max_obj_world_pen": float(obj_world_pen),
        "max_obj_table_pen": float(obj_table_pen),
        "max_obj_other_support_pen": float(other_support_pen),
        "max_subject_contact_target_pen": float(
            subject_contact_target_pen
        ),
        "max_pad_table_pen": float(pad_table_pen),
        "max_arm_pen": float(arm_pen),
        "max_subject_movable_pen": float(subj_mov_pen),
        "subject_movable_frames": float(subj_mov_frames),
        "max_robot_movable_pen": float(robot_mov_pen),
        "robot_movable_frames": float(robot_mov_frames),
        "contact_target_bodies": list(contact_target_names),
        "mediation_target_bodies": list(target_names),
        "max_subject_target_pen": float(subj_target_pen),
        "subject_target_frames": float(subj_target_frames),
        "max_robot_target_pen": float(robot_target_pen),
        "robot_target_frames": float(robot_target_frames),
        "max_subject_target_pen_m_by_body": subj_target_pen_by_body,
        "subject_target_frames_by_body": subj_target_frames_by_body,
        "max_robot_target_pen_m_by_body": robot_target_pen_by_body,
        "robot_target_frames_by_body": robot_target_frames_by_body,
    }
    # This rolls the SHARED world.data; a slipped grasp would leave the object
    # tumbled for the next caller (who reads world.data for the object rest
    # pose). Reset so the helper is order-independent -- it must not pollute the
    # module-scoped world fixture other tests read.
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    return row
