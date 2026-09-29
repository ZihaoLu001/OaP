"""``oap-run``: the Stage-2 online closed loop (dry-run by default).

Role in the two-stage pipeline: consumes a scene bundle written by
``oap-reconstruct`` plus a natural-language task, synthesizes ONE
TaskProgram (or loads a pre-synthesized ``--program-json``), plans and
sim-verifies bounded chunks in the calibrated twin, and -- ONLY with
``--execute --i-confirm-real-motion`` and a home-gate pass -- drives the real
arm through flexiv-control serve. Every episode writes a gauntlet-ready
evidence packet (``episode.json``, schema ``oap_episode_v3``).

External-env interpreters (observer / FoundationPose / VLM judge) default
from an external ``external_envs.yaml`` config (select the file
with ``OAP_EXTERNAL_ENVS`` or ``--external-envs-file``, and any single
interpreter with its ``--*-python`` flag); heavy models are NEVER imported in
this environment.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from oap.twin.runtime import (
    ProductionRuntimeError,
    bootstrap_production_mjwarp,
)
from oap.execution_budget import (
    apply_execution_budget_protocol,
    validate_execution_budget_protocol,
)
from oap.loop import LoopConfig, measured_program_success, run_episode
from oap.loop.execution_prefix import (
    DEFAULT_EXECUTION_PREFIX_FRACTION,
    validate_execution_prefix_fraction,
    validate_execution_prefix_steps,
    validate_mppi_stage_execution_prefix_steps,
)
from oap.loop.plan import (
    num_knots_from_env,
    validate_num_knots,
)
from oap.loop.sampling import (
    CANDIDATE_SELECTION_MODES,
    CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
    pool_size_from_env,
    validate_candidate_selection_mode,
    validate_pool_size,
)
from oap.twin.batched_rollout import (
    DEFAULT_CEM_ELITE_FRACTION,
    DEFAULT_CEM_MIN_STD_FRACTION,
    DEFAULT_CEM_ROUNDS,
    DEFAULT_MPPI_TEMPERATURE,
    DEFAULT_ARM_VELOCITY_WEIGHT,
    MPPI_EXECUTION_MODES,
    MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
    MTP_JAW_MODES,
    MTP_JAW_MODE_PRESERVE_NOMINAL,
    cem_elite_fraction_from_env,
    cem_min_std_fraction_from_env,
    cem_rounds_from_env,
    validate_cem_elite_fraction,
    validate_cem_rounds,
    PREDICTIVE_SAMPLER_MODES,
    PREDICTIVE_SAMPLER_SINGLE_SCALE,
    PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
    horizon_steps_from_env,
    local_sigma_fraction_from_env,
    validate_horizon_steps,
    validate_local_sigma_fraction,
    validate_mtp_jaw_mode,
    validate_predictive_sampler_mode,
    validate_mppi_temperature,
    validate_mppi_execution_mode,
    validate_arm_velocity_weight,
)
from oap.loop.safety import SafetyError
from oap.program.cost_shaping import (
    COST_SHAPING_PROFILES,
    validate_cost_shaping_profile,
)
from oap.twin.control_profile import (
    CONTROL_PROFILES,
    DIRECT_TORQUE_S0_COVERAGE_K,
    DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS,
    JOINT_VELOCITY_FORCE_TASK_IDS,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGETS,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1,
    JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1,
    JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES,
    JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES,
    JOINT_VELOCITY_FORCE_ALGORITHM_ARMS,
    JOINT_VELOCITY_FORCE_DIAL_ARMS,
    JOINT_VELOCITY_FORCE_PHASE3_FINALIST_ARMS,
    JOINT_VELOCITY_FORCE_PHASE3_SEEDS,
    direct_torque_s0_horizon_prefix_values,
    joint_velocity_force_algorithm_arm_values,
    joint_velocity_force_dial_arm_values,
    joint_velocity_force_phase3_finalist_arm_values,
    validate_control_profile,
    validate_joint_velocity_force_task_id,
    validate_joint_velocity_force_trial_budget,
    validate_joint_velocity_force_algorithm_arm,
    validate_joint_velocity_force_dial_arm,
    validate_joint_velocity_force_phase3_finalist_arm,
    validate_joint_velocity_force_phase3_seed,
    validate_direct_torque_s0_coverage_k,
    validate_direct_torque_s0_horizon_prefix_arm,
)
from oap.twin.control_regularizer import (
    CONTROL_REGULARIZER_CALIBRATION_PROFILES,
    validate_control_regularizer_calibration_profile,
)
from oap.twin.unified_mppi_effort import (
    UNIFIED_MPPI_EFFORT_VALUES,
    unified_mppi_effort_cost_shaping_profile,
    unified_mppi_effort_uses_proven_pick_carrier,
    unified_mppi_effort_values,
    validate_unified_mppi_effort_profile_arg,
    unified_mppi_effort_task_id,
    unified_mppi_effort_task_registration,
)
from oap.twin.pick_v8_causal import (
    PICK_V8_CAUSAL_PROFILES,
    PICK_V8_CAUSAL_VALUES,
    PICK_V8_SUCCESS_STAGE_PREFIXES,
    pick_v8_causal_cost_shaping_profile,
    pick_v8_causal_values,
    pick_v8_causal_uses_original_prefix_schedule,
    pick_v8_causal_uses_width_jaw,
    validate_pick_v8_causal_profile,
)
from oap.utils.io import load_yaml, package_config_path

logger = logging.getLogger("oap.cli.run")

__all__ = ["build_parser", "config_from_args", "external_env_defaults", "main"]


class _StoreExplicitAction(argparse.Action):
    """Store a value and remember that its CLI option was explicitly used."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        explicit = set(
            getattr(namespace, "_oap_explicit_fields", ())
        )
        explicit.add(self.dest)
        setattr(namespace, "_oap_explicit_fields", tuple(sorted(explicit)))


# --------------------------------------------------------------------------
# External-env interpreter defaults (configs/external_envs.yaml)
# --------------------------------------------------------------------------
def external_env_defaults(profile: str | None = None,
                          envs_file: Path | None = None) -> dict[str, Any]:
    """Load external-env interpreter defaults for one host profile.

    Resolution order for the file: ``envs_file`` argument ->
    ``OAP_EXTERNAL_ENVS`` env var -> ``external_envs.yaml`` in the
    external config root (schema ``oap_external_envs_v1``:
    ``hosts.<profile>.<tool>.{python,root,model_path,model_id}``; the profile
    defaults to the file's ``default_profile``).

    Returns a FLAT dict of the keys the run stage consumes
    (``observer_python``, ``foundationpose_python``, ``foundationpose_root``,
    -- {} when no file exists, so a
    dry run without any external env still works.
    """
    path: Path | None = None
    if envs_file is not None:
        path = Path(envs_file)
    elif os.environ.get("OAP_EXTERNAL_ENVS"):
        path = Path(os.environ["OAP_EXTERNAL_ENVS"])
    else:
        try:
            path = package_config_path("external_envs.yaml")
        except FileNotFoundError:
            return {}
    if not path.exists():
        raise SystemExit(f"external-envs file not found: {path}")
    doc = load_yaml(path) or {}
    hosts = doc.get("hosts", {})
    if profile is None:
        profile = str(doc.get("default_profile", next(iter(hosts), "")))
    entry = hosts.get(profile)
    if entry is None:
        raise SystemExit(
            f"external-envs profile {profile!r} not found in {path} "
            f"(have: {sorted(hosts)}); pass --external-envs <profile> or add "
            f"hosts.{profile} to the yaml")

    def _tool(tool: str, key: str) -> Any:
        return (entry.get(tool) or {}).get(key)

    flat = {
        "observer_python": _tool("observer", "python"),
        "foundationpose_python": _tool("foundationpose", "python"),
        "foundationpose_root": _tool("foundationpose", "root"),
    }
    return {k: v for k, v in flat.items() if v is not None}


def _env_path(entry: dict[str, Any], key: str, cli_value: Path | None) -> Path | None:
    """Resolve one interpreter/tool path: CLI flag wins, else the yaml key."""
    if cli_value is not None:
        return Path(cli_value)
    v = entry.get(key)
    return Path(v) if v else None


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the ``oap-run`` argument parser (flags = ported lab truth)."""
    p = argparse.ArgumentParser(
        prog="oap-run", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # scene + task + output
    p.add_argument("--bundle", type=Path, required=True,
                   help="scene-bundle manifest JSON written by oap-reconstruct")
    p.add_argument("--task", required=True, help="natural-language task description")
    p.add_argument("--out", type=Path, required=True, help="episode output directory")
    # program synthesis
    p.add_argument("--synth-backend", default="anthropic",
                   help="program-synthesis backend: anthropic (default; model "
                        "$OAP_ANTHROPIC_MODEL, default claude-opus-4-8) | "
                        "openrouter. Live "
                        "backends receive the anchor-annotated planning-start "
                        "frame as an image block when one exists. API keys "
                        "come from a user-sourced env file; never entered here.")
    p.add_argument("--program-json", type=Path, default=None,
                   help="load a PRE-SYNTHESIZED program instead of synthesizing "
                        "in-process (synthesize where there is a network+key, run "
                        "the closed loop anywhere; still groundability-checked live)")
    # robot server
    p.add_argument("--server-host", default="ROBOT-HOST-PLACEHOLDER")
    p.add_argument("--server-port", type=int, default=8766)
    # observation
    p.add_argument("--foundationpose-mode", choices=("off", "actual"), default="off",
                   help="off = gravity-OBB pose from the depth capture (default); "
                        "actual = FoundationPose register-once -> track")
    p.add_argument("--subject-pose-policy", choices=("foundationpose", "gravity_obb"),
                   default="foundationpose",
                   help="how to localize the subject under --foundationpose-mode "
                        "actual (gravity_obb reads which side is vertical straight "
                        "from depth; FP render-and-compare can flip a textureless box)")
    p.add_argument("--foundationpose-timeout-s", type=float, default=300.0)
    p.add_argument(
        "--foundationpose-checkpoint-timeout-s",
        type=float,
        default=None,
        help=(
            "maximum wait for one fresh packet from the already-running "
            "continuous tracker; real execution requires an explicit value "
            "separate from the FoundationPose/camera RPC timeout"
        ),
    )
    p.add_argument("--fp-seed-from-obb", action="store_true",
                   help="seed FP's FIRST registration from the gravity-OBB pose "
                        "(resolves which face is up on textureless/symmetric objects)")
    p.add_argument("--relocalize-all-each-chunk", action="store_true",
                   help="re-observe & re-localize EVERY non-subject static object "
                        "each chunk (multi-object FP tracking) instead of the "
                        "one-time post-grasp reference relocalize")
    p.add_argument("--disable-held-verify", action="store_true",
                   help="do NOT re-check a believed-held subject each chunk; blind "
                        "FK track (legacy A/B). Default: verify held.")
    p.add_argument("--offline", action="store_true",
                   help="no camera/robot at all: observations come from the plan "
                        "world (a fully local closed-loop test)")
    p.add_argument("--vlm-generated-program", action="store_true",
                   help="the --program-json file is machine-synthesized (VLM), "
                        "not the sealed authority: relaxes ONLY the "
                        "program==authority equality under a unified profile, "
                        "requires machine provenance on the program, and runs "
                        "under the typed_residual_library_v1 cost shaping "
                        "(per-type weights fixed in the residual library)")
    p.add_argument("--initial-obs-json", type=Path, default=None,
                   help="offline mode: seed the initial subject pose from a "
                        "recorded observation JSON")
    p.add_argument("--offline-dataset-subject-source-name", default=None)
    p.add_argument("--offline-dataset-subject-name", default=None)
    p.add_argument("--offline-dataset-subject-label", default=None)
    p.add_argument("--offline-dataset-visual-mesh", type=Path, default=None)
    p.add_argument(
        "--offline-dataset-size-lwh",
        type=float,
        nargs=3,
        default=None,
        metavar=("L", "W", "H"),
    )
    p.add_argument("--offline-dataset-mass-kg", type=float, default=None)
    p.add_argument(
        "--offline-dataset-rgba",
        type=float,
        nargs=4,
        default=None,
        metavar=("R", "G", "B", "A"),
    )
    p.add_argument(
        "--offline-dataset-collision-primitive",
        choices=("box", "cylinder"),
        default=None,
    )
    # twin
    p.add_argument("--table-spec-json", type=Path, default=None,
                   help="patch the twin table geom to a measured spec (size + "
                        "xy-center + tilt + base plate)")
    p.add_argument("--real-table-z", type=float, default=None,
                   help="measured real table-surface z in the base frame (m), e.g. "
                        "-0.021 from the touch/ChArUco probe")
    p.add_argument("--sim-tcp-m", type=float, default=None,
                   help="place the sim grasp-center this far (m) from the flange to "
                        "MATCH the real calibrated tool (0.19812)")
    p.add_argument("--sim-tcp-anchor", choices=("gripper", "site"), default="site",
                   help="'site' (default) moves only the grasp-site + trims the tip "
                        "boxes so the sim hand matches the real 198mm tool; "
                        "'gripper' is the legacy shift (left sim fingertips 28mm "
                        "long and its rollouts exploded)")
    p.add_argument("--home-posture-json", type=Path, default=None,
                   help="canonical home posture json (default: external "
                        "calibration/lab_home_posture.json)")
    # grasp-point reconcile
    p.add_argument("--grasp-tcp-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   metavar=("X", "Y", "Z"),
                   help="RDK-TCP -> grasp-point offset in the TOOL frame (m); "
                        "commanded TCP = target - R(quat)@offset. Default 0 0 0 = "
                        "OFF; confirm with a power-on jog-test before relying on it.")
    # planning
    p.add_argument(
        "--optimizer",
        choices=("cem", "mppi"),
        default="cem",
        action=_StoreExplicitAction,
        help="trajectory optimizer (default: cem)",
    )
    p.add_argument(
        "--control-profile",
        choices=CONTROL_PROFILES,
        type=validate_control_profile,
        default=None,
        help=(
            "experiment-only offline Pick-v9 S0 actuator profile; selecting "
            "one enables the complete fail-closed diagnostic contract"
        ),
    )
    p.add_argument(
        "--unified-mppi-effort-profile",
        # No ``choices``: argparse applies ``type`` first and then tests the
        # RESULT against choices, so an ablated name survived the validator and
        # was rejected here instead.  The validator already refuses an unknown
        # base and an unknown ablation key, with a message naming both, so
        # choices was a redundant second gate that only knew about base names.
        type=validate_unified_mppi_effort_profile_arg,
        default=None,
        help=(
            "select the sole exact Pick-v8/Cup/Push/Flip MPPI effort profile; "
            "all controller values are derived and numeric overrides refuse"
        ),
    )
    p.add_argument(
        "--execution-budget-protocol",
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--pick-v8-causal-profile",
        choices=PICK_V8_CAUSAL_PROFILES,
        type=validate_pick_v8_causal_profile,
        default=None,
        help="select one closed Pick-v8 prefix-only or gripper-only diagnostic",
    )
    p.add_argument(
        "--control-profile-smoke-one-cycle",
        action="store_true",
        help=(
            "evidence-bound offline Pick-v9 S0 smoke: allow exactly one "
            "completed cycle; requires --control-profile and "
            "--max-stage-cycles 1"
        ),
    )
    p.add_argument(
        "--joint-velocity-force-task-id",
        choices=JOINT_VELOCITY_FORCE_TASK_IDS,
        type=validate_joint_velocity_force_task_id,
        default=None,
        help=(
            "select one sealed Pick/Cup/Flip/Push registration for the "
            "joint_velocity_force profile; no task or controller knobs are "
            "derived from free text"
        ),
    )
    p.add_argument(
        "--control-regularizer-profile",
        "--control-regularizer-calibration-profile",
        dest="control_regularizer_calibration_profile",
        choices=CONTROL_REGULARIZER_CALIBRATION_PROFILES,
        type=validate_control_regularizer_calibration_profile,
        default=None,
        help=(
            "select the registered zero-weight calibration or sole frozen "
            "score-active control regularizer; requires a four-task "
            "joint_velocity_force task ID"
        ),
    )
    p.add_argument(
        "--cost-shaping-profile",
        choices=COST_SHAPING_PROFILES,
        type=validate_cost_shaping_profile,
        default=None,
        help=(
            "select the registered task-agnostic trajectory-cost ablation; "
            "terminal success and hard validity remain unchanged"
        ),
    )
    p.add_argument(
        "--joint-velocity-force-trial-budget",
        choices=JOINT_VELOCITY_FORCE_TRIAL_BUDGETS,
        type=validate_joint_velocity_force_trial_budget,
        default=None,
        help=(
            "select an evidence-bound trial budget; the cost-focus stage90 "
            "budget requires the four-task joint_velocity_force registry, "
            "formal regularizer, and an explicit registered cost profile"
        ),
    )
    p.add_argument(
        "--joint-velocity-force-algorithm-arm",
        choices=tuple(JOINT_VELOCITY_FORCE_ALGORITHM_ARMS),
        type=validate_joint_velocity_force_algorithm_arm,
        default=None,
        help=(
            "select one closed Phase-A algorithm arm; requires the four-task "
            "joint_velocity_force formal-regularizer stage30 registry"
        ),
    )
    p.add_argument(
        "--joint-velocity-force-dial-arm",
        choices=JOINT_VELOCITY_FORCE_DIAL_ARMS,
        type=validate_joint_velocity_force_dial_arm,
        default=None,
        help=(
            "select the sole independent Phase-B DIAL-inspired arm; requires "
            "the four-task JVF formal-regularizer stage30 registry"
        ),
    )
    p.add_argument(
        "--joint-velocity-force-phase3-finalist-arm",
        choices=tuple(JOINT_VELOCITY_FORCE_PHASE3_FINALIST_ARMS),
        type=validate_joint_velocity_force_phase3_finalist_arm,
        default=None,
        help=(
            "select one of the two closed Phase-3 full360 finalists; requires "
            "the four-task JVF formal regularizer and a registered Phase-3 seed"
        ),
    )
    p.add_argument(
        "--joint-velocity-force-phase3-seed",
        choices=JOINT_VELOCITY_FORCE_PHASE3_SEEDS,
        type=int,
        default=None,
        help="select the preregistered Phase-3 episode seed",
    )
    p.add_argument(
        "--direct-torque-s0-coverage-k",
        choices=DIRECT_TORQUE_S0_COVERAGE_K,
        type=int,
        default=None,
        help=(
            "register one direct-torque Pick-v9 S0 sampling-coverage arm; "
            "K is the only variable and sets --num-knots when that option is "
            "omitted; requires --control-profile direct_torque_force"
        ),
    )
    p.add_argument(
        "--direct-torque-s0-horizon-prefix-arm",
        choices=tuple(DIRECT_TORQUE_S0_HORIZON_PREFIX_ARMS),
        type=validate_direct_torque_s0_horizon_prefix_arm,
        default=None,
        help=(
            "select one registered direct-torque Pick-v9 S0 horizon-by-prefix "
            "arm; the arm sets the paired horizon/K and exact execution "
            "prefix while preserving 2 ms physics and 40 ms action cadence"
        ),
    )
    p.add_argument(
        "--mppi-temperature",
        type=validate_mppi_temperature,
        default=DEFAULT_MPPI_TEMPERATURE,
        action=_StoreExplicitAction,
        help="positive raw-cost softmax temperature for MPPI (default: 0.1)",
    )
    p.add_argument(
        "--mppi-execution-mode",
        choices=MPPI_EXECUTION_MODES,
        type=validate_mppi_execution_mode,
        default=MPPI_EXECUTION_SOFTMAX_WEIGHTED_MEAN,
        action=_StoreExplicitAction,
        help=(
            "apply the softmax MPPI mean, or exactly replay the lowest-cost "
            "valid sampled trajectory while retaining the softmax warm start"
        ),
    )
    p.add_argument(
        "--arm-velocity-weight",
        type=validate_arm_velocity_weight,
        choices=(0.0, 0.01, 0.1),
        default=DEFAULT_ARM_VELOCITY_WEIGHT,
        action=_StoreExplicitAction,
        help=(
            "task-independent physical arm-qvel cost weight; registered "
            "ablation values are 0, 0.01, and 0.1 (default: 0.1)"
        ),
    )
    p.add_argument(
        "--record-candidate-cost-telemetry",
        action="store_true",
        help=(
            "record columnar per-candidate typed and regularizer cost "
            "breakdowns; evidence-only and selection-neutral"
        ),
    )
    p.add_argument("--pool-size", type=validate_pool_size, default=None,
                   action=_StoreExplicitAction,
                   help="candidates in the one nominal-plus-Gaussian predictive "
                        "batch (default: $OAP_POOL_SIZE or 2048)")
    p.add_argument(
        "--horizon-steps",
        type=validate_horizon_steps,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "prediction-horizon length in execution-physics steps "
            "(default: $OAP_HORIZON_STEPS or 480)"
        ),
    )
    p.add_argument(
        "--execution-prefix-fraction",
        type=validate_execution_prefix_fraction,
        default=DEFAULT_EXECUTION_PREFIX_FRACTION,
        action=_StoreExplicitAction,
        help=(
            "fraction of each planned horizon executed before re-observation "
            "(default: exact 1/(K-1)=1/3 for K=4). Real execution ignores "
            "this as a source of truth and requires exact "
            "--execution-prefix-steps in its authorized profile"
        ),
    )
    p.add_argument(
        "--num-knots",
        type=validate_num_knots,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "number of linear-spline control knots "
            "(default: $OAP_N_KNOTS or 4)"
        ),
    )
    p.add_argument(
        "--sigma-fraction",
        type=validate_local_sigma_fraction,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "Gaussian proposal standard deviation as a fraction of "
            "each actuator half-range "
            "(default: $OAP_LOCAL_SIGMA_FRACTION or 0.06)"
        ),
    )
    p.add_argument(
        "--sampler-mode",
        choices=PREDICTIVE_SAMPLER_MODES,
        type=validate_predictive_sampler_mode,
        default=PREDICTIVE_SAMPLER_SINGLE_SCALE,
        action=_StoreExplicitAction,
        help=(
            "one-round baseline proposal distribution: single_scale (default) "
            "or the two_scale simulation ablation; ignored by iterative CEM"
        ),
    )
    p.add_argument(
        "--candidate-selection-mode",
        choices=CANDIDATE_SELECTION_MODES,
        type=validate_candidate_selection_mode,
        default=CANDIDATE_SELECTION_VALID_THEN_TOTAL_COST,
        action=_StoreExplicitAction,
        help=(
            "candidate result policy (default: valid_then_total_cost); "
            "result_terminal_earliest is active for stages whose terminal "
            "vocabulary has sampled-trajectory evidence"
        ),
    )
    p.add_argument(
        "--mtp-jaw-mode",
        choices=MTP_JAW_MODES,
        type=validate_mtp_jaw_mode,
        default=MTP_JAW_MODE_PRESERVE_NOMINAL,
        action=_StoreExplicitAction,
        help="MTP jaw policy: preserve_nominal (legacy default) or sample",
    )
    p.add_argument(
        "--local-sigma-fraction",
        type=validate_local_sigma_fraction,
        default=PREDICTIVE_TWO_SCALE_LOCAL_SIGMA_FRACTION,
        action=_StoreExplicitAction,
        help=(
            "local Gaussian standard deviation in actuator half-range units "
            "for --sampler-mode two_scale (default: 0.02); "
            "--sigma-fraction is its broad scale"
        ),
    )
    p.add_argument(
        "--cem-rounds",
        type=validate_cem_rounds,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "CEM refit rounds under the fixed total rollout budget "
            f"(default: ${'OAP_CEM_ROUNDS'} or {DEFAULT_CEM_ROUNDS})"
        ),
    )
    p.add_argument(
        "--cem-elite-fraction",
        type=validate_cem_elite_fraction,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "lowest-cost valid fraction used for each CEM refit "
            f"(default: {DEFAULT_CEM_ELITE_FRACTION})"
        ),
    )
    p.add_argument(
        "--cem-min-std-fraction",
        type=validate_local_sigma_fraction,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "task-independent CEM standard-deviation floor as a fraction "
            f"of action range (default: {DEFAULT_CEM_MIN_STD_FRACTION})"
        ),
    )
    p.add_argument(
        "--execution-prefix-steps",
        type=validate_execution_prefix_steps,
        default=None,
        action=_StoreExplicitAction,
        help=(
            "exact execution-physics steps run before re-observation; when "
            "specified this takes precedence over --execution-prefix-fraction "
            "and must not exceed --horizon-steps"
        ),
    )
    p.add_argument(
        "--mppi-stage-execution-prefix-steps",
        type=validate_mppi_stage_execution_prefix_steps,
        default=None,
        help=(
            "offline local-MPPI-only comma-separated exact prefix steps, one "
            "per semantic stage (for example 50,10); default None preserves "
            "the single episode-wide execution prefix"
        ),
    )
    p.add_argument(
        "--max-stage-cycles",
        type=int,
        default=9,
        action=_StoreExplicitAction,
        help="maximum continuous receding-horizon cycles per stage (default: 9)",
    )
    p.add_argument(
        "--mppi-cycle-budget-scope",
        choices=("episode", "stage"),
        default="episode",
        action=_StoreExplicitAction,
        help=(
            "apply --max-stage-cycles once to the full paper-style MPPI "
            "episode, or independently to every explicit semantic stage"
        ),
    )
    p.add_argument(
        "--remote-planner-url",
        default=None,
        help=(
            "loopback HTTP URL for an SSH-forwarded H100 planner endpoint; "
            "requires every remote planner identity flag below"
        ),
    )
    p.add_argument("--remote-planner-model-sha256", default=None)
    p.add_argument("--remote-planner-assets-sha256", default=None)
    p.add_argument("--remote-planner-sha256", default=None)
    p.add_argument("--remote-planner-service-instance-id", default=None)
    p.add_argument(
        "--remote-planner-deadline-s",
        type=float,
        default=None,
        help=(
            "maximum end-to-end remote solve latency; must not exceed "
            "--max-plan-staleness-s"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=27,
        action=_StoreExplicitAction,
        help=(
            "evaluation random seed; unified MPPI profiles freeze controller "
            "values but permit an explicitly supplied evaluation seed"
        ),
    )
    # execution (dry-run default)
    p.add_argument("--execute", action="store_true",
                   help="request REAL joint motion; still refused while the "
                        "internal hardware-verified gate is false")
    p.add_argument("--prepare-real-execution", action="store_true",
                   help="connection-free readiness run: live-profile TCP/FK/safety "
                        "preflight with simulated execution; NEVER connects/moves")
    p.add_argument(
        "--joint-executor-verification-json",
        type=Path,
        default=None,
        help=(
            "attended joint-executor verification latch bound to the installed "
            "flexiv-control fingerprint; required for real motion"
        ),
    )
    p.add_argument(
        "--joint-executor-verification-sha256",
        default=None,
        help=(
            "full lowercase SHA-256 approving the exact joint-executor "
            "verification JSON; required with --execute"
        ),
    )
    p.add_argument("--i-confirm-real-motion", action="store_true",
                   help="explicit acknowledgement required WITH --execute")
    p.add_argument("--approved-program-sha256", default=None,
                   help="exact externally reviewed live-VLM program hash required "
                        "for real motion")
    p.add_argument(
        "--readiness-authorization-token",
        type=Path,
        default=None,
        help=(
            "short-lived Ed25519-signed, one-shot authorization bound to the "
            "exact source/program/bundle/server/calibrations/readiness evidence; "
            "required with --execute"
        ),
    )
    p.add_argument(
        "--operator-estop-packet-json",
        type=Path,
        default=None,
        help=(
            "attended operator/E-stop packet whose exact SHA-256 is bound by "
            "the signed execution authorization"
        ),
    )
    p.add_argument(
        "--hardware-evidence-calibration-json",
        type=Path,
        default=None,
        help=(
            "measured ObjectHeld evidence calibration. Required by real/preparation "
            "runs whose program contains ObjectHeld; supply measured values "
            "for your hardware to authorize motion"
        ),
    )
    p.add_argument(
        "--max-observation-age-s",
        type=float,
        default=None,
        help="measured maximum camera-observation age at use; required for "
             "real/readiness runs (no shipped safety default)",
    )
    p.add_argument(
        "--max-camera-robot-skew-s",
        type=float,
        default=None,
        help="measured maximum camera/robot timestamp skew; required for "
             "real/readiness runs",
    )
    p.add_argument(
        "--max-plan-staleness-s",
        type=float,
        default=None,
        help="maximum age of the measured planning state when a prefix begins; "
             "required for real/readiness runs",
    )
    p.add_argument(
        "--max-final-joint-tracking-error-rad",
        type=float,
        default=None,
        help="maximum measured final joint error after a prefix; required for "
             "real/readiness runs",
    )
    p.add_argument(
        "--min-terminal-anchor-reliability",
        type=float,
        default=None,
        help="minimum confidence*visibility for every measured terminal "
             "anchor; required for real/readiness runs",
    )
    p.add_argument("--home-first", action="store_true",
                   help="move the arm to canonical home before the run (REAL "
                        "MOTION -- needs --execute + e-stop in hand)")
    p.add_argument("--allow-home-drift", action="store_true",
                   help="diagnostic-only home-gate override; forbidden with "
                        "--execute")
    p.add_argument("--no-per-chunk-confirm", action="store_true",
                   help="diagnostic-only prompt suppression; forbidden with "
                        "--execute")
    p.add_argument(
        "--manual-post-experiment-restore",
        action="store_true",
        help=(
            "after a real run, write final_posture.json and stop/hold without "
            "automatic lift, gripper-open, or home; the operator must restore "
            "the verified initial_posture.json manually (automatic safe "
            "recovery remains the default)"
        ),
    )
    # external-env interpreters (defaults from configs/external_envs.yaml)
    p.add_argument("--external-envs", default=None, metavar="PROFILE",
                   help="host profile in configs/external_envs.yaml supplying the "
                        "external interpreter paths (e.g. a local or remote profile)")
    p.add_argument("--external-envs-file", type=Path, default=None,
                   help="explicit external_envs.yaml path (else "
                        "$OAP_EXTERNAL_ENVS, else the external config root)")
    p.add_argument("--observer-python", type=Path, default=None,
                   help="python with pyzed+SAM3 for capture/recording "
                        "(yaml key: observer_python)")
    p.add_argument("--foundationpose-python", type=Path, default=None,
                   help="python of the FoundationPose env (yaml key: "
                        "foundationpose_python)")
    p.add_argument("--foundationpose-root", type=Path, default=None,
                   help="FoundationPose checkout root (yaml key: foundationpose_root)")
    return p


def config_from_args(args: argparse.Namespace) -> LoopConfig:
    """Map parsed CLI arguments onto a :class:`LoopConfig`."""
    try:
        env = external_env_defaults(args.external_envs, args.external_envs_file)
    except SystemExit:
        if args.external_envs is not None or args.external_envs_file is not None:
            raise
        env = {}
    unified_profile = validate_unified_mppi_effort_profile_arg(
        args.unified_mppi_effort_profile,
        allow_none=True,
    )
    execution_budget_protocol = validate_execution_budget_protocol(
        args.execution_budget_protocol,
        allow_none=True,
    )
    if execution_budget_protocol is not None and unified_profile is None:
        raise ValueError(
            "execution-budget protocol requires a unified MPPI profile"
        )
    causal_profile = validate_pick_v8_causal_profile(
        args.pick_v8_causal_profile,
        allow_none=True,
    )
    if unified_profile is not None and causal_profile is not None:
        raise ValueError(
            "pick_v8_causal profile conflicts with unified_mppi_effort_v1"
        )
    unified_values: dict[str, Any] = {}
    unified_task_id: str | None = None
    if unified_profile is not None:
        forbidden_selectors = {
            "control_profile": args.control_profile,
            "joint_velocity_force_task_id": args.joint_velocity_force_task_id,
            "joint_velocity_force_trial_budget": (
                args.joint_velocity_force_trial_budget
            ),
            "joint_velocity_force_algorithm_arm": (
                args.joint_velocity_force_algorithm_arm
            ),
            "joint_velocity_force_dial_arm": args.joint_velocity_force_dial_arm,
            "joint_velocity_force_phase3_finalist_arm": (
                args.joint_velocity_force_phase3_finalist_arm
            ),
            "joint_velocity_force_phase3_seed": (
                args.joint_velocity_force_phase3_seed
            ),
            "direct_torque_s0_coverage_k": args.direct_torque_s0_coverage_k,
            "direct_torque_s0_horizon_prefix_arm": (
                args.direct_torque_s0_horizon_prefix_arm
            ),
            "control_regularizer_calibration_profile": (
                args.control_regularizer_calibration_profile
            ),
            "cost_shaping_profile": args.cost_shaping_profile,
            "mppi_stage_execution_prefix_steps": (
                args.mppi_stage_execution_prefix_steps
            ),
        }
        active = [name for name, value in forbidden_selectors.items() if value is not None]
        if active or args.control_profile_smoke_one_cycle:
            raise ValueError(
                "unified_mppi_effort_v1 conflicts with legacy selectors: "
                + ", ".join(active or ["control_profile_smoke_one_cycle"])
            )
        explicit = set(getattr(args, "_oap_explicit_fields", ()))
        # The evaluation seed identifies a stochastic replicate, not a
        # controller setting.  Keep every controller value frozen while
        # allowing paired evaluations to select their registered seeds.
        numeric_fields = set(UNIFIED_MPPI_EFFORT_VALUES) - {"seed"}
        overridden = sorted(explicit & numeric_fields)
        if overridden:
            raise ValueError(
                "unified_mppi_effort_v1 forbids numeric overrides: "
                + ", ".join(overridden)
            )
        unified_values = unified_mppi_effort_values(unified_profile)
        if "seed" in explicit:
            unified_values["seed"] = int(args.seed)
        unified_values = apply_execution_budget_protocol(
            unified_values,
            execution_budget_protocol,
        )
        unified_task_id = unified_mppi_effort_task_id(args)
        registration = unified_mppi_effort_task_registration(
            unified_task_id,
            unified_profile,
        )
        for name, value in registration["scene"].items():
            current = getattr(args, name)
            if current is not None and current != value:
                raise ValueError(
                    f"unified_mppi_effort_v1 scene field {name} conflicts "
                    "with canonical task registration"
                )
            setattr(args, name, value)
    if causal_profile is not None:
        forbidden_selectors = {
            "control_profile": args.control_profile,
            "joint_velocity_force_task_id": args.joint_velocity_force_task_id,
            "joint_velocity_force_trial_budget": args.joint_velocity_force_trial_budget,
            "joint_velocity_force_algorithm_arm": args.joint_velocity_force_algorithm_arm,
            "joint_velocity_force_dial_arm": args.joint_velocity_force_dial_arm,
            "joint_velocity_force_phase3_finalist_arm": (
                args.joint_velocity_force_phase3_finalist_arm
            ),
            "joint_velocity_force_phase3_seed": args.joint_velocity_force_phase3_seed,
            "direct_torque_s0_coverage_k": args.direct_torque_s0_coverage_k,
            "direct_torque_s0_horizon_prefix_arm": (
                args.direct_torque_s0_horizon_prefix_arm
            ),
            "control_regularizer_calibration_profile": (
                args.control_regularizer_calibration_profile
            ),
            "cost_shaping_profile": args.cost_shaping_profile,
            "mppi_stage_execution_prefix_steps": (
                args.mppi_stage_execution_prefix_steps
            ),
        }
        active = [name for name, value in forbidden_selectors.items() if value is not None]
        explicit = set(getattr(args, "_oap_explicit_fields", ()))
        overridden = sorted(explicit & set(PICK_V8_CAUSAL_VALUES))
        if active or overridden or args.control_profile_smoke_one_cycle:
            conflicts = active + overridden
            if args.control_profile_smoke_one_cycle:
                conflicts.append("control_profile_smoke_one_cycle")
            raise ValueError(
                "pick_v8_causal profile conflicts with overrides: "
                + ", ".join(conflicts)
            )
        unified_values = pick_v8_causal_values(causal_profile)
    runtime_unified_profile = unified_profile
    if causal_profile is not None:
        # Reuse only the frozen numeric-value selection below. The categorical
        # identities remain separate when LoopConfig is constructed.
        unified_profile = causal_profile

    matrix_arm = validate_direct_torque_s0_horizon_prefix_arm(
        args.direct_torque_s0_horizon_prefix_arm,
        allow_none=True,
    )
    matrix_values = (
        direct_torque_s0_horizon_prefix_values(matrix_arm)
        if matrix_arm is not None
        else {}
    )
    trial_budget = validate_joint_velocity_force_trial_budget(
        args.joint_velocity_force_trial_budget,
        allow_none=True,
    )
    cost_focus_stage90 = (
        trial_budget
        == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_STAGE90_V1
    )
    cost_focus_screen30 = (
        trial_budget
        == JOINT_VELOCITY_FORCE_TRIAL_BUDGET_COST_FOCUS_SCREEN30_V1
    )
    cost_focus = cost_focus_stage90 or cost_focus_screen30
    algorithm_arm = validate_joint_velocity_force_algorithm_arm(
        args.joint_velocity_force_algorithm_arm,
        allow_none=True,
    )
    dial_arm = validate_joint_velocity_force_dial_arm(
        args.joint_velocity_force_dial_arm,
        allow_none=True,
    )
    phase3_arm = validate_joint_velocity_force_phase3_finalist_arm(
        args.joint_velocity_force_phase3_finalist_arm,
        allow_none=True,
    )
    phase3_seed = validate_joint_velocity_force_phase3_seed(
        args.joint_velocity_force_phase3_seed,
        allow_none=True,
    )
    if sum(item is not None for item in (algorithm_arm, dial_arm, phase3_arm)) > 1:
        raise ValueError(
            "selection conflicts across Phase-A, Phase-B DIAL, and Phase-3 arms"
        )
    if phase3_arm is None and phase3_seed is not None:
        raise ValueError("Phase-3 seed requires a Phase-3 finalist arm")
    if phase3_arm is not None and phase3_seed is None:
        raise ValueError("Phase-3 finalist arm requires a registered Phase-3 seed")
    if cost_focus and any(
        item is not None
        for item in (algorithm_arm, dial_arm, phase3_arm, phase3_seed)
    ):
        raise ValueError(
            "cost-focus trial conflicts with Phase-A, Phase-B DIAL, and "
            "Phase-3 selectors"
        )
    registered_arm = (
        algorithm_arm
        or dial_arm
        or phase3_arm
        or ("cost_focus_stage90" if cost_focus_stage90 else None)
        or ("cost_focus_screen30" if cost_focus_screen30 else None)
    )
    registered_values = (
        joint_velocity_force_algorithm_arm_values(algorithm_arm)
        if algorithm_arm is not None
        else joint_velocity_force_dial_arm_values(dial_arm)
        if dial_arm is not None
        else joint_velocity_force_phase3_finalist_arm_values(phase3_arm)
        if phase3_arm is not None
        else {
            name: value
            for name, value in (
                (
                    JOINT_VELOCITY_FORCE_COST_FOCUS_STAGE90_VALUES
                    if cost_focus_stage90
                    else JOINT_VELOCITY_FORCE_COST_FOCUS_SCREEN30_VALUES
                ).items()
            )
            if name != "table_contact_force_weight"
        }
        if cost_focus
        else {}
    )
    algorithm_values = (
        {
            **registered_values,
            "optimizer": "mppi",
            "sigma_fraction": 1.0,
            "mppi_temperature": 0.1,
            "mppi_execution_mode": "best_valid_sample",
            "candidate_selection_mode": "result_terminal_earliest",
            "cem_rounds": 1,
            "cem_elite_fraction": 0.1,
            "cem_min_std_fraction": 0.01,
            "max_stage_cycles": (
                360
                if phase3_arm is not None
                else 90
                if cost_focus_stage90
                else 30
            ),
            "mppi_cycle_budget_scope": "stage",
            "seed": phase3_seed if phase3_arm is not None else 1200,
            "arm_velocity_weight": 0.0,
            "execution_prefix_fraction": DEFAULT_EXECUTION_PREFIX_FRACTION,
        }
        if registered_arm is not None
        else {}
    )
    if registered_arm is not None:
        if (
            args.direct_torque_s0_coverage_k is not None
            or matrix_arm is not None
        ):
            raise ValueError(
                "joint_velocity_force algorithm arm conflicts with Pick-only "
                "sampling matrices"
            )
        explicit = set(getattr(args, "_oap_explicit_fields", ()))
        for field, expected in algorithm_values.items():
            if field in explicit and getattr(args, field) != expected:
                option = "--" + field.replace("_", "-")
                raise ValueError(
                    f"{option}={getattr(args, field)!r} conflicts with "
                    f"registered arm {registered_arm!r} "
                    f"({expected!r})"
                )
    if matrix_arm is not None:
        if args.direct_torque_s0_coverage_k is not None:
            raise ValueError(
                "horizon-prefix matrix and coverage K are mutually exclusive"
            )
        for option, actual, key in (
            ("--horizon-steps", args.horizon_steps, "horizon_steps"),
            ("--num-knots", args.num_knots, "num_knots"),
            (
                "--execution-prefix-steps",
                args.execution_prefix_steps,
                "execution_prefix_steps",
            ),
        ):
            expected = matrix_values[key]
            if actual is not None and actual != expected:
                raise ValueError(
                    f"{option}={actual!r} conflicts with registered "
                    f"horizon-prefix arm {matrix_arm!r} ({expected!r})"
                )
    phase_a_algorithm_arm = algorithm_arm
    # The existing normalization block is shared by both closed categorical
    # registries. Preserve the distinct categorical fields in LoopConfig.
    algorithm_arm = registered_arm
    cfg = LoopConfig(
        bundle_manifest=args.bundle,
        task=args.task,
        out_dir=args.out,
        synth_backend=args.synth_backend,
        program_json=args.program_json,
        observer_python=_env_path(env, "observer_python", args.observer_python),
        foundationpose_python=_env_path(env, "foundationpose_python", args.foundationpose_python),
        foundationpose_root=_env_path(env, "foundationpose_root", args.foundationpose_root),
        # Generated-only object contract: FP always consumes the subject mesh
        # selected and integrity-bound by the scene bundle.
        foundationpose_mesh=None,
        foundationpose_timeout_s=float(args.foundationpose_timeout_s),
        foundationpose_checkpoint_timeout_s=(
            None
            if args.foundationpose_checkpoint_timeout_s is None
            else float(args.foundationpose_checkpoint_timeout_s)
        ),
        foundationpose_mode=args.foundationpose_mode,
        subject_pose_policy=args.subject_pose_policy,
        fp_seed_from_obb=bool(args.fp_seed_from_obb),
        relocalize_all_each_chunk=bool(args.relocalize_all_each_chunk),
        disable_held_verify=bool(args.disable_held_verify),
        offline=bool(args.offline),
        vlm_generated_program=bool(args.vlm_generated_program),
        initial_obs_json=args.initial_obs_json,
        offline_dataset_subject_source_name=(
            args.offline_dataset_subject_source_name
        ),
        offline_dataset_subject_name=args.offline_dataset_subject_name,
        offline_dataset_subject_label=args.offline_dataset_subject_label,
        offline_dataset_visual_mesh=args.offline_dataset_visual_mesh,
        offline_dataset_size_lwh=(
            None
            if args.offline_dataset_size_lwh is None
            else tuple(float(value) for value in args.offline_dataset_size_lwh)
        ),
        offline_dataset_mass_kg=args.offline_dataset_mass_kg,
        offline_dataset_rgba=(
            None
            if args.offline_dataset_rgba is None
            else tuple(float(value) for value in args.offline_dataset_rgba)
        ),
        offline_dataset_collision_primitive=(
            args.offline_dataset_collision_primitive
        ),
        table_spec_json=args.table_spec_json,
        table_spec_height=False,
        real_table_z=args.real_table_z,
        # No per-object CLI correction modes. Runtime tracking supplies pose;
        # collision geometry is selected by the reconstruction manifest.
        snap_subject_to_table=False,
        collision_from_mesh=False,
        sim_tcp_m=args.sim_tcp_m,
        sim_tcp_anchor=args.sim_tcp_anchor,
        home_posture_json=args.home_posture_json,
        pool_size=(
            unified_values["pool_size"]
            if unified_profile is not None
            else
            algorithm_values["pool_size"]
            if algorithm_arm is not None
            else pool_size_from_env()
            if args.pool_size is None
            else validate_pool_size(args.pool_size)
        ),
        horizon_steps=(
            unified_values["horizon_steps"]
            if unified_profile is not None
            else
            algorithm_values["horizon_steps"]
            if algorithm_arm is not None
            else matrix_values["horizon_steps"]
            if matrix_arm is not None
            else horizon_steps_from_env()
            if args.horizon_steps is None
            else validate_horizon_steps(args.horizon_steps)
        ),
        num_knots=(
            unified_values["num_knots"]
            if unified_profile is not None
            else
            algorithm_values["num_knots"]
            if algorithm_arm is not None
            else matrix_values["num_knots"]
            if matrix_arm is not None
            else validate_direct_torque_s0_coverage_k(
                args.direct_torque_s0_coverage_k
            )
            if args.num_knots is None
            and args.direct_torque_s0_coverage_k is not None
            else num_knots_from_env()
            if args.num_knots is None
            else validate_num_knots(args.num_knots)
        ),
        sigma_fraction=(
            unified_values["sigma_fraction"]
            if unified_profile is not None
            else
            algorithm_values["sigma_fraction"]
            if algorithm_arm is not None
            else local_sigma_fraction_from_env()
            if args.sigma_fraction is None
            else validate_local_sigma_fraction(
                args.sigma_fraction
            )
        ),
        sampler_mode=validate_predictive_sampler_mode(
            unified_values["sampler_mode"]
            if unified_profile is not None
            else algorithm_values["sampler_mode"]
            if algorithm_arm is not None
            else args.sampler_mode
        ),
        candidate_selection_mode=validate_candidate_selection_mode(
            unified_values["candidate_selection_mode"]
            if unified_profile is not None
            else algorithm_values["candidate_selection_mode"]
            if algorithm_arm is not None
            else args.candidate_selection_mode
        ),
        mtp_jaw_mode=validate_mtp_jaw_mode(
            unified_values["mtp_jaw_mode"]
            if unified_profile is not None
            else algorithm_values["mtp_jaw_mode"]
            if algorithm_arm is not None
            else args.mtp_jaw_mode
        ),
        local_sigma_fraction=validate_local_sigma_fraction(
            unified_values["local_sigma_fraction"]
            if unified_profile is not None
            else algorithm_values["local_sigma_fraction"]
            if algorithm_arm is not None
            else args.local_sigma_fraction
        ),
        cem_rounds=(
            unified_values["cem_rounds"]
            if unified_profile is not None
            else algorithm_values["cem_rounds"]
            if algorithm_arm is not None
            else cem_rounds_from_env()
            if args.cem_rounds is None
            else validate_cem_rounds(args.cem_rounds)
        ),
        cem_elite_fraction=(
            unified_values["cem_elite_fraction"]
            if unified_profile is not None
            else algorithm_values["cem_elite_fraction"]
            if algorithm_arm is not None
            else cem_elite_fraction_from_env()
            if args.cem_elite_fraction is None
            else validate_cem_elite_fraction(args.cem_elite_fraction)
        ),
        cem_min_std_fraction=(
            unified_values["cem_min_std_fraction"]
            if unified_profile is not None
            else algorithm_values["cem_min_std_fraction"]
            if algorithm_arm is not None
            else cem_min_std_fraction_from_env()
            if args.cem_min_std_fraction is None
            else validate_local_sigma_fraction(args.cem_min_std_fraction)
        ),
        optimizer=str(
            unified_values["optimizer"]
            if unified_profile is not None
            else algorithm_values["optimizer"]
            if algorithm_arm is not None
            else args.optimizer
        ),
        control_profile=(
            "legacy_velocity_width"
            if causal_profile is not None
            and pick_v8_causal_uses_width_jaw(causal_profile)
            else "legacy_velocity_effort"
            if causal_profile is not None
            else "legacy_velocity_effort"
            if runtime_unified_profile is not None
            and unified_mppi_effort_uses_proven_pick_carrier(
                runtime_unified_profile
            )
            else "joint_velocity_force"
            if runtime_unified_profile is not None
            else args.control_profile
        ),
        unified_mppi_effort_profile=runtime_unified_profile,
        execution_budget_protocol=execution_budget_protocol,
        pick_v8_causal_profile=causal_profile,
        unified_mppi_effort_task_id=unified_task_id,
        control_profile_smoke_one_cycle=bool(
            args.control_profile_smoke_one_cycle
        ),
        joint_velocity_force_task_id=validate_joint_velocity_force_task_id(
            args.joint_velocity_force_task_id,
            allow_none=True,
        ),
        joint_velocity_force_trial_budget=(
            trial_budget
        ),
        joint_velocity_force_algorithm_arm=phase_a_algorithm_arm,
        joint_velocity_force_dial_arm=dial_arm,
        joint_velocity_force_phase3_finalist_arm=phase3_arm,
        joint_velocity_force_phase3_seed=phase3_seed,
        control_regularizer_calibration_profile=(
            validate_control_regularizer_calibration_profile(
                args.control_regularizer_calibration_profile,
                allow_none=True,
            )
        ),
        cost_shaping_profile=validate_cost_shaping_profile(
            pick_v8_causal_cost_shaping_profile(causal_profile)
            if causal_profile is not None
            else (
                # Executor ablation (Addendum 77): OAP_EXEC_SHAPING_OVERRIDE
                # substitutes the typed library for VLM programs. Unset = the
                # sealed behaviour, byte for byte.
                (os.environ.get("OAP_EXEC_SHAPING_OVERRIDE")
                 or "typed_residual_library_v3")
                if args.vlm_generated_program
                else unified_mppi_effort_cost_shaping_profile(
                    runtime_unified_profile
                )
            )
            if runtime_unified_profile is not None
            else args.cost_shaping_profile,
            allow_none=True,
        ),
        direct_torque_s0_coverage_k=(
            validate_direct_torque_s0_coverage_k(
                args.direct_torque_s0_coverage_k,
                allow_none=True,
            )
        ),
        direct_torque_s0_horizon_prefix_arm=matrix_arm,
        mppi_temperature=validate_mppi_temperature(
            unified_values["mppi_temperature"]
            if unified_profile is not None
            else algorithm_values["mppi_temperature"]
            if algorithm_arm is not None
            else args.mppi_temperature
        ),
        mppi_execution_mode=validate_mppi_execution_mode(
            unified_values["mppi_execution_mode"]
            if unified_profile is not None
            else algorithm_values["mppi_execution_mode"]
            if algorithm_arm is not None
            else args.mppi_execution_mode
        ),
        arm_velocity_weight=validate_arm_velocity_weight(
            unified_values["arm_velocity_weight"]
            if unified_profile is not None
            else algorithm_values["arm_velocity_weight"]
            if algorithm_arm is not None
            else args.arm_velocity_weight
        ),
        record_candidate_cost_telemetry=(
            unified_values["record_candidate_cost_telemetry"]
            if unified_profile is not None
            else True
            if algorithm_arm is not None
            else bool(args.record_candidate_cost_telemetry)
        ),
        execution_prefix_fraction=(
            unified_values["execution_prefix_fraction"]
            if unified_profile is not None
            else algorithm_values["execution_prefix_fraction"]
            if algorithm_arm is not None
            else args.execution_prefix_fraction
        ),
        execution_prefix_steps=(
            unified_values["execution_prefix_steps"]
            if unified_profile is not None
            else algorithm_values["execution_prefix_steps"]
            if algorithm_arm is not None
            else matrix_values["execution_prefix_steps"]
            if matrix_arm is not None
            else args.execution_prefix_steps
        ),
        mppi_stage_execution_prefix_steps=(
            PICK_V8_SUCCESS_STAGE_PREFIXES
            if causal_profile is not None
            and pick_v8_causal_uses_original_prefix_schedule(causal_profile)
            else args.mppi_stage_execution_prefix_steps
        ),
        max_stage_cycles=int(
            unified_values["max_stage_cycles"]
            if unified_profile is not None
            else algorithm_values["max_stage_cycles"]
            if algorithm_arm is not None
            else args.max_stage_cycles
        ),
        mppi_cycle_budget_scope=str(
            unified_values["mppi_cycle_budget_scope"]
            if unified_profile is not None
            else algorithm_values["mppi_cycle_budget_scope"]
            if algorithm_arm is not None
            else args.mppi_cycle_budget_scope
        ),
        remote_planner_url=args.remote_planner_url,
        remote_planner_model_sha256=args.remote_planner_model_sha256,
        remote_planner_assets_sha256=args.remote_planner_assets_sha256,
        remote_planner_sha256=args.remote_planner_sha256,
        remote_planner_service_instance_id=(
            args.remote_planner_service_instance_id
        ),
        remote_planner_deadline_s=args.remote_planner_deadline_s,
        seed=int(
            unified_values["seed"]
            if unified_profile is not None
            else algorithm_values["seed"]
            if algorithm_arm is not None
            else args.seed
        ),
        execute=bool(args.execute),
        prepare_real_execution=bool(args.prepare_real_execution),
        joint_executor_verification_json=(
            args.joint_executor_verification_json
        ),
        joint_executor_verification_sha256=(
            args.joint_executor_verification_sha256
        ),
        i_confirm_real_motion=bool(args.i_confirm_real_motion),
        approved_program_sha256=args.approved_program_sha256,
        readiness_authorization_token=args.readiness_authorization_token,
        operator_estop_packet_json=args.operator_estop_packet_json,
        hardware_evidence_calibration_json=args.hardware_evidence_calibration_json,
        max_observation_age_s=args.max_observation_age_s,
        max_camera_robot_skew_s=args.max_camera_robot_skew_s,
        max_plan_staleness_s=args.max_plan_staleness_s,
        max_final_joint_tracking_error_rad=(
            args.max_final_joint_tracking_error_rad
        ),
        min_terminal_anchor_reliability=(
            args.min_terminal_anchor_reliability
        ),
        server_host=args.server_host,
        server_port=int(args.server_port),
        grasp_tcp_offset=(float(args.grasp_tcp_offset[0]),
                          float(args.grasp_tcp_offset[1]),
                          float(args.grasp_tcp_offset[2])),
        home_first=bool(args.home_first),
        allow_home_drift=bool(args.allow_home_drift),
        per_chunk_confirm=not bool(args.no_per_chunk_confirm),
        manual_post_experiment_restore=bool(
            args.manual_post_experiment_restore
        ),
    )
    return cfg


def main(argv: list[str] | None = None) -> int:
    """The ``oap-run`` console entry point."""
    logging.basicConfig(
        level=os.environ.get("OAP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        bootstrap_production_mjwarp()
    except ProductionRuntimeError as exc:
        print(f"GPU_ONLY_REFUSAL: {exc}", file=sys.stderr)
        return 2
    cfg = config_from_args(args)
    try:
        summary = run_episode(cfg)
    except SafetyError as exc:
        print(f"SAFETY REFUSAL: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("aborted by operator", file=sys.stderr)
        return 130
    verified = measured_program_success(
        terminal_success=summary.get("terminal_success"),
    )
    consistent = summary.get("success") is True and verified
    print(
        f"success={consistent} outcome={summary.get('outcome')} "
        f"chunks={summary.get('chunks')} out={cfg.out_dir}"
    )
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
