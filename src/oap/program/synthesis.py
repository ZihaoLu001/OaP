"""Synthesize a TaskProgram from an arbitrary instruction.

Role in the two-stage pipeline: this is the single generative interface of the
planning stage (``oap-run``) where open-task generality lives. A frontier
VLM is shown the instruction, the anchor list (id: kind @ base-frame xyz) and
-- when the caller provides one -- the anchor-ANNOTATED observation image
(:mod:`oap.program.annotate`: every anchor drawn under its exact id, so
the text ids and the image labels ground each other), and emits a JSON program
over the FIXED typed predicate set (:mod:`oap.program.predicates`). A new
unknown task = new VLM output, ZERO new code. Without an image the prompt is
text-only, byte-identical to the historical behavior (reconstructions without
a stored frame and deterministic JSON fixtures keep working unchanged).

Multi-VLM by design (test several providers and pick the best): backends
'anthropic' (default) and 'openrouter' (ONE key -> dozens of models through
the OpenAI-compatible gateway) call the respective API using a key the USER
provides in the environment (e.g. a chmod-600 env file that the process
sources) -- this module NEVER creates, copies, or stores keys, and raises
clear setup guidance if a key/SDK is absent.

Tests exercise this interface with explicit program JSON or a constant stub
backend. There is no keyword compiler or task-specific fallback in production:
without a live backend, callers must supply a reviewed ``--program-json``.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .anchors import AnchorSet
from .predicates import Predicate, SupportedOn, RotationProgress, relative_orientation_frame_failures
from .program import TaskProgram

# --- the schema the VLM is told to emit (compact) --------------------------------
PAPER_TERM_TYPES = frozenset({
    "point_point_distance", "relative_position", "axis_parallel", "axis_angle",
    "signed_axis_gap", "rotation_progress", "point_on_line", "in_region",
    "tool_mediation", "object_held", "not_held", "gripper_command",
})

PREDICATE_SCHEMA_DOC = """
OUTPUT FORMAT
Emit {"stages":[{"name":"stage_name","movable_objects":["body_name",...],
"running":[PREDICATE...],"terminal":[PREDICATE...]}]}.
Every stage requires movable_objects: exactly the scene bodies it intends to
move, or [] when none should move. Use finite numbers and literal anchor ids.

TABLE I: THE TWELVE RESIDUAL TERMS
Distances and widths are metres. JSON fields ending in _deg are degrees;
the evaluator converts angular residuals and scales to radians.
p(a) is the world-frame position and v(a) the unit world-frame axis of anchor a.
[z]+ = max(z,0). Dot products passed to acos are clipped to [-1,1].

 {"type":"point_point_distance","a","b","dist","tol"}
   r = abs(norm(p(a)-p(b)) - dist).
 {"type":"relative_position","a","b","offset":[dx,dy,dz],"tol"}
   r = norm(p(a)-p(b)-offset), with world-frame offset.
 {"type":"axis_parallel","a","b","antiparallel":false,"tol_deg"}
   r = acos(eta*dot(v(a),v(b))); eta=-1 if antiparallel, otherwise +1.
 {"type":"axis_angle","a","b","theta_deg","tol_deg"}
   r = abs(acos(dot(v(a),v(b))) - radians(theta_deg)).
 {"type":"signed_axis_gap","a","b","axis":[x,y,z],"margin","sign":1}
   r = [margin-sign*dot(unit(axis),p(b)-p(a))]+.
   For positive displacement, use a=initial, b=current and sign=1.
 {"type":"rotation_progress","body","axis":[x,y,z],"min_angle_deg","scale_deg"}
   alpha = dot(R0*unit(axis),rotvec(R*transpose(R0))).
   r = [radians(min_angle_deg)-alpha]+. axis is in the episode-start material
   frame; R0 is fixed at episode start. rotvec is the principal rotation vector.
   scale_deg is a positive cost scale (10, like the other angular terms), not an
   acceptance tolerance.
 {"type":"point_on_line","a","b","dir_anchor","tol"}
   r = norm(cross(p(a)-p(b),v(dir_anchor))). The line passes through p(b);
   movement along the line is free.
 {"type":"in_region","a","region"}
   r = max_k([abs(p(a)[k]-center[k])-halfsize[k]]+/halfsize[k]).
   region is a world-axis-aligned box with strictly positive half-sizes.
 {"type":"tool_mediation","tool","target","require_tool_target_contact":true}
   r = [M_robot_target, rho*(1-M_tool_target)], rho=1 when contact is required.
   Each M is a binary indicator of contact anywhere in the predicted trajectory.
   Running only: a horizon-wide contact event is not an endpoint measurement.
   Holding is a separate object_held term.
 {"type":"object_held","object_anchor"}
   r(h)=1-chi(h), where chi is simultaneous contact by both fingers at physics
   step h. This is a finite running penalty, not a hard rollout mask.
 {"type":"not_held","object_anchor"}
   r(h)=chi(h), the inverse of object_held.
 {"type":"gripper_command","state":"open","tol":0}
   state is open or closed; optional width_m in [0,0.085] sets a numeric target.
   g(u)=0.085/2*(1-clip(u,-1,1)); r=abs(g(u)-target_width).
   +1 effort closes and -1 opens. This measures the command, not physical width.
   For a grasp/release outcome use object_held/not_held, not the command alone.

COST MAPPING (EQUATIONS 3-5)
Each residual has a positive same-unit scale s and a terminal threshold tau.
Running: c_R = sqrt(1+(r/s)^2)-1. There is no tolerance dead zone.
Terminal: c_T = ([r-tau]+/s)^2, zero when r <= tau.
The stage objective averages weighted running costs over H predicted physics
steps, adds weighted terminal costs at x_H using u_(H-1), and adds C_physics.
Vector residual components are shaped separately and summed.
Running tool_mediation has weight 4; the other terms and all terminal terms
have weight 1. The library supplies positive scales; tol/tol_deg set
geometric terminal thresholds and also the default scale for those terms.
Boolean and lower-bound terms have zero terminal threshold. Do not change an
instruction's target to make a trial pass. C_physics penalizes control changes,
joint velocities, large contact forces, and unintended object motion.
Predicted satisfaction does not advance the stage. The stage verifier checks
all terminal conditions together using the measured state and, for a command
term, the last acknowledged control input.

ANCHORS AND STAGES
Reference supplied anchors, gripper_center and gripper_closing_axis.
A body's _center and material _frame_x/y/z anchors move with that body.
Its _initial_center and _initial_frame_x/y/z remain fixed for the episode.
Use geometry to guide approach; binary contact/held costs alone give no dense
approach signal. Account for finger clearance and identify opposing objectives.
Every stage needs at least one running and one terminal term.
"""

Backend = Callable[[str, AnchorSet], TaskProgram]
_BACKENDS: dict[str, Backend] = {}

# Production exposes one compact, question-driven synthesis contract.
PROMPT_REASONING_MODES = ("semantic_checklist",)
PRODUCTION_PROMPT_REASONING_MODE = "semantic_checklist"

_PROMPT_REASONING_SCAFFOLDS = {
    "semantic_checklist": """

Before returning JSON, silently answer this semantic checklist:
1. Which grounded entities are the controlled body, goal body, tool, support,
   and reference, and which instruction words establish those roles?
2. What smallest observable relation makes the instruction true? Distinguish
   equality, one-sided bound, containment, contact, held, and release.
3. Which intermediate results are logically necessary and independently
   re-observable? Omit a stage if no such result exists.
4. For each retained stage, which relation supplies dense progress before its
   terminal becomes true?
5. For every axis/sign expression, substitute the desired endpoint into the
   documented formula and verify the sign.
6. For every number, identify whether it comes from the instruction or an
   explicitly documented measured tolerance. Do not invent a waypoint,
   approach side, offset, orientation, or dwell.
Do not output the checklist answers or prose; return only the JSON program.
""",
}


def _prompt_reasoning_scaffold(reasoning_mode: str) -> str:
    if reasoning_mode not in PROMPT_REASONING_MODES:
        allowed = ", ".join(PROMPT_REASONING_MODES)
        raise ValueError(
            f"unknown prompt reasoning mode {reasoning_mode!r}; "
            f"expected one of {allowed}"
        )
    return _PROMPT_REASONING_SCAFFOLDS[reasoning_mode]


# --- synthesis trace --------------------------------------------------------------
# While set (via synthesize(trace_dir=...)), every exchange with the synthesis VLM
# is persisted VERBATIM: the exact prompts and each raw model reply (including ones
# that fail to parse -- those are the malformed-output evidence). The runner points
# this at <episode>/synthesis so the paper trail needs no re-run.
_TRACE_DIR: Path | None = None
# Set alongside _TRACE_DIR by synthesize(image_path=...): the anchor-annotated
# observation the VLM backends attach as an image block. A module-level
# context (like _TRACE_DIR) so registered custom backends keep their
# (instruction, anchors) signature; None = text-only, byte-identical requests.
_IMAGE_PATH: Path | None = None
# Set alongside _TRACE_DIR by synthesize(post_validate=...): an optional
# grounded validation the caller runs on every parsed program INSIDE the
# bounded-repair loop, so a program that parses but cannot ground (e.g. a
# supported_on support the scene cannot bind) is fed back to the VLM as a
# validator error instead of surfacing later as a 0-chunk CANNOT_VERIFY.
_POST_VALIDATE = None


def _trace_write(name: str, text: str) -> None:
    if _TRACE_DIR is None:
        return
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (_TRACE_DIR / name).write_text(text, encoding="utf-8")
    except OSError:  # tracing must never break synthesis
        pass


def _trace_write_bytes(name: str, data: bytes) -> None:
    if _TRACE_DIR is None:
        return
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (_TRACE_DIR / name).write_bytes(data)
    except OSError:  # tracing must never break synthesis
        pass


def _image_payload() -> str | None:
    """Base64 of the annotated observation; archives the EXACT bytes sent."""
    if _IMAGE_PATH is None:
        return None
    data = Path(_IMAGE_PATH).read_bytes()
    _trace_write_bytes("prompt_image.png", data)
    return base64.b64encode(data).decode("ascii")


def register_backend(name: str, fn: Backend) -> None:
    """Register a synthesis backend under a name."""
    _BACKENDS[name] = fn


def available_backends() -> list[str]:
    """Return the sorted names of all registered backends."""
    return sorted(_BACKENDS)


def build_prompt(
    instruction: str,
    anchors: AnchorSet,
    *,
    with_image: bool = False,
    reasoning_mode: str = PRODUCTION_PROMPT_REASONING_MODE,
    execution_environment: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Build the public Table I prompt from the same schema used by execution."""
    _prompt_reasoning_scaffold(reasoning_mode)  # Validate the supported mode.
    anchor_lines = []
    for name in anchors.names():
        anchor = anchors[name]
        data = {"position_m": np.asarray(anchor.point).tolist()}
        if anchor.attached_to is not None:
            data["owner"] = anchor.attached_to
        if anchor.axis is not None:
            data["axis"] = np.asarray(anchor.axis).tolist()
        if anchor.region_half is not None:
            data["half_extents_m"] = np.asarray(anchor.region_half).tolist()
        anchor_lines.append(f"  {name}: {json.dumps(data)}")
    system = (
        "Compile the instruction into a grounded multi-stage objective program. "
        "Use only the twelve Table I terms below. Choose goals and completion "
        "conditions; the shared physics-based MPC chooses controls. "
        "Check whether each stage can start from the preceding stage's end, "
        "whether geometric targets leave finger clearance, and whether any "
        "running objective conflicts with completion. Return JSON only.\n"
        + PREDICATE_SCHEMA_DOC
    )
    image_note = ("The image shows the scene with anchors annotated by id.\n"
                  if with_image else "")
    user = (image_note + f"Instruction: {instruction}\nScene anchors:\n"
            + "\n".join(anchor_lines)
            + "\nUse the supplied object dimensions and poses to derive targets. "
            "Keep the instruction's final goal unchanged. "
            "Return ONLY the JSON program.")
    _trace_write("prompt_system.txt", system)
    _trace_write("prompt_user.txt", user)
    return system, user


# --- deterministic structural validator -------------------------------------------
# Required anchor-reference fields per primitive type; absence/unbound -> typed error
# BEFORE the program is ever run (parse_program is a bare json.loads that would else
# fail late with a stack trace). Returns machine-readable error strings ([] == valid).
_REQUIRED_REFS: dict[str, tuple[str, ...]] = {
    "point_point_distance": ("a", "b"),
    "capture_distance": ("subject",),
    "relative_position": ("a", "b"),
    "min_distance": ("a", "b"), "above_by": ("a", "b"),
    "on_plane": ("a", "b"), "point_on_line": ("a", "b", "dir_anchor"),
    "axis_parallel": ("a", "b"),
    "axis_angle": ("a", "b"), "angle_about_axis": ("a", "b", "ref"),
    "relative_orientation": ("body",),
    "rotation_progress": ("body",),
    "in_region": ("a", "region"), "signed_axis_gap": ("a", "b"),
    "interaction_align": ("a", "b"), "along_axis_gap": ("a", "c", "b"),
    "contact": ("a", "b"), "no_contact": ("a", "b"),
    "robot_subject_contact": (),
    "tool_mediation": ("tool", "target"),
    "object_held": ("object_anchor",), "gripper_command": (),
    "gripper_state": (),
    "not_held": ("object_anchor",),
    "supported_on": ("subject", "support"),
    "at_rest": ("subject",),
    "temporal_hold": (), "non_geometric": (),
}
# Fields whose omission would silently select a dataclass default and can turn
# a malformed VLM relation into a different or inert constraint.  Robot
# Capability chooses the semantic gripper open/closed widths, but the command
# tolerance is an explicit structural zero: an actuator target is not a noisy
# measurement acceptance band.
_REQUIRED_VALUE_FIELDS: dict[str, tuple[str, ...]] = {
    "point_point_distance": ("dist", "tol"),
    "relative_position": ("offset", "tol"),
    "min_distance": ("clearance",),
    "above_by": ("dist", "tol", "axis"),
    "on_plane": ("axis", "tol"),
    "point_on_line": ("tol",),
    "axis_parallel": ("antiparallel", "tol_deg"),
    "axis_angle": ("theta_deg", "tol_deg"),
    "relative_orientation": ("target_axis", "target_angle_deg", "tol_deg"),
    "rotation_progress": ("axis", "min_angle_deg"),
    "angle_about_axis": ("theta_deg", "tol_deg"),
    "signed_axis_gap": ("axis", "margin", "sign"),
    "interaction_align": (
        "dist",
        "tol",
        "axis_a",
        "axis_b",
        "theta_deg",
        "tol_deg",
        "antiparallel",
    ),
    "along_axis_gap": ("dist", "tol"),
    "gripper_command": ("tol",),
    "temporal_hold": ("frames",),
}
# Numeric scalar fields any predicate may carry; each must be float-parseable
# when present (None == absent, legal for optional fields like width_m).
#
# A zero tolerance is useful only for an event-seeking running potential.
# Explicit VLM geometry remains part of the planning objective even after a
# separate held event becomes true; the planner must not silently rewrite it.
# Other running relations should measure distance to the terminal feasible set.
# Measured terminal equalities need a positive acceptance band because exact
# equality is not a robust stopping condition under sensor noise. Clearance is
# a lower bound rather than an equality tolerance and is always strictly
# positive.
_TOLERANCE_FIELDS = ("tol", "tol_deg")
_ALWAYS_POSITIVE_FIELDS = ("clearance",)
_NUMERIC_FIELDS = (
    "tol", "tol_deg", "clearance", "dist", "theta_deg", "margin",
    "target_angle_deg", "width_m", "w_d", "w_theta", "w_perp", "sign",
    "min_angle_deg", "scale_deg",
)
# primitives that DEREFERENCE an anchor's axis -> those anchors must carry one
# (else anchors.axis() raises at runtime; a live VLM 'open drawer' once pointed an
# axis primitive at an axis-less keypoint and crashed -- catch it as a typed error).
_AXIS_REQUIRED_FIELDS = {
    "axis_angle": ("a", "b"), "angle_about_axis": ("a", "b", "ref"),
    "axis_parallel": ("a", "b"),
    "point_on_line": ("dir_anchor",), "along_axis_gap": ("b",),
}
_RUNTIME_ANCHORS = {
    "gripper_center",
    "gripper_closing_axis",
    "robot",
}
_MATERIAL_FRAME_SUFFIXES = tuple(f"_frame_{axis}" for axis in "xyz")
_PROGRAM_FIELDS = {
    "instruction",
    "provenance",
    "notes",
    "stages",
    "table_contact_force_weight",
}
_STAGE_FIELDS = {"name", "running", "terminal", "movable_objects"}
_SYNTHESIS_WEIGHT_FIELDS = {"w_d", "w_theta", "w_perp"}


def _canonical_displacement_pair(
    predicate: dict[str, Any],
) -> tuple[str, str] | None:
    """Return ``(initial, current)`` for an episode-start displacement pair."""
    a = predicate.get("a")
    b = predicate.get("b")
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    suffix = "_initial_center"
    for initial, other in ((a, b), (b, a)):
        if not initial.endswith(suffix):
            continue
        stem = initial[: -len(suffix)]
        if other in {stem, f"{stem}_center"}:
            return initial, other
    return None


def _defers_to_grounded_pass(
    ref: str, *, require_grounded_frame_contract: bool
) -> bool:
    """Whether the manifest pass may leave this anchor id to the grounded pass.

    ``anchors_from_scene_manifest`` builds its vocabulary with
    ``include_material_frame=False``, so a material-frame triad member --
    ``<id>_frame_x/y/z``, or the frozen ``<id>_initial_frame_x/y/z`` derived
    from it -- cannot appear in it however the scene is configured.  The
    runtime scene mints the triad for every body carrying a verified signed
    pose.  This mirrors the deferral ``require_grounded_frame_contract``
    already performs for ``relative_orientation``, and is safe for the same
    reason: planning flows re-run the strict validation on the grounded
    vocabulary before any rollout, so an id that never binds is still refused.
    """
    if require_grounded_frame_contract:
        return False
    return any(
        ref.endswith(suffix) and len(ref) > len(suffix)
        for suffix in _MATERIAL_FRAME_SUFFIXES
    )


def _validate_predicate(
    p: dict,
    anchors: AnchorSet | None,
    where: str,
    *,
    terminal: bool,
    reject_synthesis_weights: bool,
    require_grounded_frame_contract: bool,
) -> list[str]:
    errs: list[str] = []
    t = p.get("type")
    if not isinstance(t, str) or t not in _REQUIRED_REFS:
        return [f"{where}: unknown primitive type {t!r} (closed basis only)"]
    allowed_fields = Predicate.allowed_fields(t)
    if allowed_fields is None:
        return [f"{where}: unknown primitive type {t!r} (closed basis only)"]
    unknown = sorted(set(p) - allowed_fields)
    if unknown:
        errs.append(
            f"{where}: {t} has unknown fields {unknown}; "
            f"allowed fields are {sorted(allowed_fields)}"
        )
    supplied_weights = sorted(set(p) & _SYNTHESIS_WEIGHT_FIELDS)
    if reject_synthesis_weights and supplied_weights:
        errs.append(
            f"{where}: synthesized predicates cannot set cost weights "
            f"{supplied_weights}; residual scaling belongs to the compiler"
        )
    for field in _REQUIRED_VALUE_FIELDS.get(t, ()):
        if field not in p or p[field] is None:
            errs.append(
                f"{where}: {t} missing required value field {field!r}; "
                "the compiler will not silently select a predicate default"
            )
    if t in {"gripper_command", "gripper_state"}:
        if "state" not in p:
            errs.append(
                f"{where}: {t}.state must be explicit ('open' or 'closed')"
            )
        state = p.get("state")
        if state not in {"open", "closed"}:
            errs.append(
                f"{where}: {t}.state must be 'open' or 'closed', got "
                f"{state!r}"
            )
        if p.get("width_m") is not None and t == "gripper_state":
            errs.append(
                f"{where}: synthesized {t} must omit numeric width_m; "
                "semantic open/closed is grounded by the robot capability"
            )
        if t == "gripper_command" and p.get("width_m") is not None:
            width = p["width_m"]
            if (isinstance(width, bool) or not isinstance(width, (int, float))
                    or not math.isfinite(width) or not 0 <= width <= 0.085):
                errs.append(f"{where}: gripper_command.width_m must be in [0,0.085] m")
        if t == "gripper_command":
            tol = p.get("tol")
            if (
                isinstance(tol, (int, float))
                and not isinstance(tol, bool)
                and math.isfinite(float(tol))
                and float(tol) != 0.0
            ):
                errs.append(
                    f"{where}: semantic gripper_command.tol must be exactly "
                    "0; it is an actuator target, not a measured acceptance "
                    "band"
                )
        if not terminal and t == "gripper_state":
            errs.append(
                f"{where}: gripper_state is terminal-only measured state; "
                "running uses gripper_command"
            )
    if t == "tool_mediation":
        required = p.get("require_tool_target_contact", True)
        if not isinstance(required, bool):
            errs.append(
                f"{where}: tool_mediation.require_tool_target_contact must "
                "be Boolean"
            )
    if (
        t == "robot_subject_contact"
        and anchors is not None
        and require_grounded_frame_contract
        and not _declares_controlled_subject(anchors)
    ):
        errs.append(
            f"{where}: robot_subject_contact requires a grounded controlled "
            "subject role"
        )
    if t in {"axis_parallel", "interaction_align"}:
        antiparallel = p.get("antiparallel")
        if not isinstance(antiparallel, bool):
            errs.append(f"{where}: {t}.antiparallel must be Boolean")
    if t == "temporal_hold":
        if "inner" not in p or not isinstance(p["inner"], dict):
            return [f"{where}: temporal_hold needs an 'inner' predicate"]
        try:
            frames = p.get("frames")
            if isinstance(frames, bool) or int(frames) != frames:
                raise ValueError
            if int(frames) < 1:
                errs.append(f"{where}: temporal_hold.frames must be >= 1")
        except (TypeError, ValueError):
            errs.append(f"{where}: temporal_hold.frames must be an integer")
        return errs + _validate_predicate(
            p["inner"],
            anchors,
            where + ".inner",
            terminal=terminal,
            reject_synthesis_weights=reject_synthesis_weights,
            require_grounded_frame_contract=require_grounded_frame_contract,
        )
    for f in _REQUIRED_REFS[t]:
        ref = p.get(f)
        if not ref:
            errs.append(f"{where}: {t} missing required anchor field {f!r}")
        elif not isinstance(ref, str):
            # anchor ids MUST be strings; a list/number here is what crashes
            # downstream dict/float ops -- catch it as a typed error, never a stack trace.
            errs.append(f"{where}: {t}.{f} anchor id must be a string, got {type(ref).__name__}")
        elif (
            anchors is not None
            and ref not in anchors
            and ref not in _RUNTIME_ANCHORS
            and not _defers_to_grounded_pass(
                ref,
                require_grounded_frame_contract=require_grounded_frame_contract,
            )
        ):
            errs.append(f"{where}: {t}.{f} references unbound anchor {ref!r}")
    if (
        t == "supported_on"
        and anchors is not None
        and _declares_controlled_subject(anchors)
        and isinstance(p.get("subject"), str)
        and isinstance(p.get("support"), str)
    ):
        try:
            SupportedOn(
                subject=p["subject"],
                support=p["support"],
            ).validate_grounded_contract(anchors)
        except ValueError as exc:
            errs.append(f"{where}: {exc}")
    if t == "rotation_progress" and require_grounded_frame_contract:
        try:
            progress = RotationProgress._from(p)
            if anchors is not None:
                errs.extend(f"{where}: {failure}" for failure in progress.frame_contract_failures(anchors))
        except (KeyError, ValueError, TypeError) as exc:
            errs.append(f"{where}: invalid rotation_progress: {exc}")
    if t == "relative_orientation" and require_grounded_frame_contract:
        body = p.get("body")
        if isinstance(body, str) and body and anchors is not None:
            frame_refs = [
                *(f"{body}_frame_{axis}" for axis in "xyz"),
                *(f"{body}_initial_frame_{axis}" for axis in "xyz"),
            ]
            for ref in frame_refs:
                anchor = anchors.get(ref)
                if anchor is None:
                    errs.append(
                        f"{where}: relative_orientation needs signed-pose "
                        f"anchor {ref!r}; a yaw/unsigned-axis fallback is not "
                        "accepted"
                    )
                elif anchor.axis is None:
                    errs.append(
                        f"{where}: relative_orientation needs anchor {ref!r} "
                        "to have a directed axis"
                    )
            for failure in relative_orientation_frame_failures(body, anchors):
                errs.append(
                    f"{where}: relative_orientation frame contract: {failure}"
                )
    # axis providers for interaction_align (when not the anchor's own 'self' axis)
    if t == "interaction_align":
        for f in ("axis_a", "axis_b"):
            ref = p.get(f, "self")
            if not isinstance(ref, str):
                errs.append(f"{where}: interaction_align.{f} must be a string")
            elif (
                ref != "self"
                and anchors is not None
                and ref not in anchors
                and ref not in _RUNTIME_ANCHORS
                and not _defers_to_grounded_pass(
                    ref,
                    require_grounded_frame_contract=(
                        require_grounded_frame_contract
                    ),
                )
            ):
                errs.append(f"{where}: interaction_align.{f} references unbound anchor {ref!r}")
    if t == "relative_position" and "offset" not in p:
        errs.append(f"{where}: relative_position missing required vector field 'offset'")
    displacement_pair = _canonical_displacement_pair(p)
    if displacement_pair is not None:
        initial, current = displacement_pair
        if (
            t == "signed_axis_gap"
            and (p.get("a"), p.get("b")) != (initial, current)
        ):
            errs.append(
                f"{where}: signed_axis_gap for an episode displacement must "
                f"use canonical a={initial!r}, b={current!r}; encode direction "
                "with axis/sign, never by swapping the anchor order"
            )
        if (
            t == "relative_position"
            and (p.get("a"), p.get("b")) != (current, initial)
        ):
            errs.append(
                f"{where}: relative_position for an episode displacement must "
                f"use canonical a={current!r}, b={initial!r}"
            )
    if t in {"signed_axis_gap", "point_on_line"}:
        a = p.get("a")
        b = p.get("b")
        if "gripper_center" in {a, b}:
            reference = b if a == "gripper_center" else a
            if isinstance(reference, str) and reference.endswith(
                "_initial_center"
            ):
                errs.append(
                    f"{where}: a gripper approach relation must reference the "
                    "body's current anchor, not episode-start anchor "
                    f"{reference!r}"
                )
            if (
                t == "signed_axis_gap"
                and isinstance(reference, str)
                and (a, b) != (reference, "gripper_center")
            ):
                errs.append(
                    f"{where}: a gripper side relation must use canonical "
                    f"a={reference!r}, b='gripper_center'; encode the side "
                    "with axis/sign, never by swapping the anchor order"
                )
    for f in _NUMERIC_FIELDS:
        if f in p and p[f] is not None:
            try:
                val = float(p[f])
            except (TypeError, ValueError):
                errs.append(f"{where}: {t}.{f} must be numeric, got {p[f]!r}")
                continue
            if not math.isfinite(val):
                errs.append(f"{where}: {t}.{f} must be finite")
                continue
            if f in _ALWAYS_POSITIVE_FIELDS and val <= 0.0:
                errs.append(f"{where}: {t}.{f} must be > 0")
            if f in _TOLERANCE_FIELDS and val < 0.0:
                errs.append(f"{where}: {t}.{f} must be >= 0")
            if f == "tol_deg" and val >= 180.0:
                errs.append(f"{where}: {t}.tol_deg must be < 180")
            if f == "sign" and val not in {-1.0, 1.0}:
                errs.append(f"{where}: signed_axis_gap.sign must be -1 or +1")
            if (
                f == "dist"
                and t in {"point_point_distance", "interaction_align"}
                and val < 0.0
            ):
                errs.append(f"{where}: {t}.dist must be >= 0")
            if (
                f == "theta_deg"
                and t in {"axis_angle", "interaction_align"}
                and not 0.0 <= val <= 180.0
            ):
                errs.append(f"{where}: {t}.theta_deg must be in [0, 180]")
            if (
                f == "target_angle_deg"
                and not -180.0 <= val <= 180.0
            ):
                errs.append(
                    f"{where}: {t}.target_angle_deg must be in [-180, 180]"
                )
            if (
                terminal
                and f in _TOLERANCE_FIELDS
                and val == 0.0
                # A commanded jaw state is discrete: the command lane emits
                # exactly two levels, so grading it exactly is correct and the
                # measurement-noise argument does not apply. Every other
                # terminal grades a noisy observation and still needs a band.
                and t != "gripper_command"
            ):
                errs.append(
                    f"{where}: {t}.{f} must be > 0 for measured terminal "
                    "evaluation (exact equality is not robust under noise)"
                )
    for vector_field in ("axis", "offset", "target_axis"):
        if vector_field not in p:
            continue
        vector = p[vector_field]
        if not isinstance(vector, (list, tuple)) or len(vector) != 3:
            errs.append(
                f"{where}: {t}.{vector_field} must be a numeric length-3 vector"
            )
        else:
            try:
                values = np.asarray(vector, dtype=float)
            except (TypeError, ValueError):
                errs.append(
                    f"{where}: {t}.{vector_field} must be a numeric length-3 vector"
                )
            else:
                if not np.all(np.isfinite(values)):
                    errs.append(
                        f"{where}: {t}.{vector_field} must contain finite numbers"
                    )
                elif (
                    vector_field in {"axis", "target_axis"}
                    and float(np.linalg.norm(values)) <= 1e-6
                ):
                    errs.append(
                        f"{where}: {t}.axis norm must be > 1e-06"
                    )
    # axis-presence: anchors fed to an axis-dereferencing primitive must HAVE an axis.
    if anchors is not None:
        axis_refs = [p.get(f) for f in _AXIS_REQUIRED_FIELDS.get(t, ())]
        if t == "interaction_align":
            axis_refs += [p.get("a") if p.get("axis_a", "self") == "self" else p.get("axis_a"),
                          p.get("b") if p.get("axis_b", "self") == "self" else p.get("axis_b")]
        for ref in axis_refs:
            if isinstance(ref, str) and ref in anchors and anchors[ref].axis is None:
                errs.append(f"{where}: {t} needs anchor {ref!r} to have an axis, but it has none")
        # region-presence: in_region dereferences region_half; a point anchor
        # passes the string-binding check then raises at residual() time
        # (InRegion.residual: "anchor is not a region"). Catch it as a typed error.
        if t == "in_region":
            reg = p.get("region")
            if isinstance(reg, str) and reg in anchors and anchors[reg].region_half is None:
                errs.append(f"{where}: in_region needs anchor {reg!r} to be a region "
                            f"(with region_half), but it has none")
    return errs


def _declares_controlled_subject(anchors: AnchorSet | None) -> bool:
    """True when this vocabulary binds task roles (a controlled ``subject``).

    Contact structural legality -- exactly one controlled-subject/robot actor,
    a distinct dynamic target, one shared target per Stage -- is a statement
    about the CONTROLLED subject. A role-free vocabulary (the scene manifest,
    or the pre-bind live-synthesis capture) cannot answer it: the same anchor
    id is subject-owned after binding and an ordinary body before it, so
    running the checks there both misses grounded-only violations and rejects
    the legal manifest-id spelling of a subject-mediated contact. The checks
    therefore run only against a role-bound vocabulary; every planning flow
    re-validates on its grounded anchors before any motion.
    """
    if anchors is None:
        return False
    if anchors.get("subject") is not None:
        return True
    return any(
        getattr(anchors.get(name), "attached_to", None) == "subject"
        for name in anchors.names()
    )


def validate_program(
    data: dict[str, Any] | TaskProgram,
    anchors: AnchorSet | None = None,
    *,
    require_grounded_frame_contract: bool = True,
) -> list[str]:
    """Validate TaskProgram schema and grounding without rewriting semantics.

    ``require_grounded_frame_contract`` may be disabled only for the initial
    manifest-vocabulary pass, before the runtime scene has minted the signed
    material-frame anchors used by ``relative_orientation``.  Planning flows
    must re-run the default strict validation on the grounded scene before
    any rollout or motion.
    """
    program_object = isinstance(data, TaskProgram)
    raw: Any = data.to_dict() if program_object else data
    if not isinstance(raw, dict):
        return ["program must be an object"]
    errors: list[str] = []
    unknown = sorted(set(raw) - _PROGRAM_FIELDS)
    if unknown:
        errors.append(
            f"program has unknown fields {unknown}; "
            f"allowed fields are {sorted(_PROGRAM_FIELDS)}"
        )
    table_force_weight = raw.get("table_contact_force_weight")
    if table_force_weight is not None and (
        isinstance(table_force_weight, bool)
        or not isinstance(table_force_weight, (int, float))
        or not math.isfinite(float(table_force_weight))
        or float(table_force_weight) < 0.0
    ):
        errors.append(
            "program.table_contact_force_weight must be a finite "
            "non-negative number or null"
        )
    stages = raw.get("stages")
    if not isinstance(stages, list) or not stages:
        return errors + ["program has no stages"]

    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"stage[{stage_index}] is not a stage object")
            continue

        unknown_stage = sorted(set(stage) - _STAGE_FIELDS)
        if unknown_stage:
            errors.append(
                f"stage[{stage_index}] has unknown fields {unknown_stage}; "
                f"allowed fields are {sorted(_STAGE_FIELDS)}"
            )
        if "name" in stage and not isinstance(stage["name"], str):
            errors.append(f"stage[{stage_index}].name must be a string")
        running = stage.get("running")
        terminal = stage.get("terminal")
        if not isinstance(running, list) or not running:
            errors.append(
                f"stage[{stage_index}] needs at least one running predicate"
            )
        if not isinstance(terminal, list) or not terminal:
            errors.append(
                f"stage[{stage_index}] needs at least one terminal predicate"
            )
        for scope, predicates in (("running", running), ("terminal", terminal)):
            if not isinstance(predicates, list):
                continue
            for predicate_index, predicate in enumerate(predicates):
                where = (
                    f"stage[{stage_index}].{scope}[{predicate_index}]"
                )
                if not isinstance(predicate, dict):
                    errors.append(f"{where} is not a predicate object")
                    continue
                errors.extend(
                    _validate_predicate(
                        predicate,
                        anchors,
                        where,
                        terminal=(scope == "terminal"),
                        reject_synthesis_weights=not program_object,
                        require_grounded_frame_contract=(
                            require_grounded_frame_contract
                        ),
                    )
                )
                scoped_predicate = predicate
                while (
                    isinstance(scoped_predicate, dict)
                    and scoped_predicate.get("type") == "temporal_hold"
                    and isinstance(scoped_predicate.get("inner"), dict)
                ):
                    scoped_predicate = scoped_predicate["inner"]
                predicate_type = scoped_predicate.get("type")
                if scope == "running" and predicate_type in {
                    "gripper_state",
                    "non_geometric",
                }:
                    errors.append(
                        f"{where} is not a computable running residual"
                    )
                if scope == "terminal" and predicate_type in {
                    "capture_distance",
                    "no_contact",
                    "tool_mediation",
                    "non_geometric",
                }:
                    errors.append(
                        f"{where} is not measurable from terminal tracking state"
                    )
        running_terms: list[dict[str, Any]] = []
        if isinstance(running, list):
            for predicate in running:
                if not isinstance(predicate, dict):
                    continue
                while (
                    predicate.get("type") == "temporal_hold"
                    and isinstance(predicate.get("inner"), dict)
                ):
                    predicate = predicate["inner"]
                running_terms.append(predicate)

        for scope_name, scope_predicates in (
            ("running", running), ("terminal", terminal),
        ):
            if not isinstance(scope_predicates, list):
                continue
            scope_types: set[str] = set()
            for predicate in scope_predicates:
                if not isinstance(predicate, dict):
                    continue
                while (
                    predicate.get("type") == "temporal_hold"
                    and isinstance(predicate.get("inner"), dict)
                ):
                    predicate = predicate["inner"]
                if predicate.get("type") in {"object_held", "not_held"}:
                    scope_types.add(str(predicate.get("type")))
            if len(scope_types) > 1:
                errors.append(
                    f"stage[{stage_index}] {scope_name} declares both "
                    "object_held and not_held; the held polarity must be "
                    "unambiguous"
                )

        gripper_commands = [
            predicate
            for predicate in running_terms
            if predicate.get("type") == "gripper_command"
        ]
        command_states = {
            predicate.get("state") for predicate in gripper_commands
        }
        if len(command_states) > 1:
            errors.append(
                f"stage[{stage_index}] has conflicting gripper_command states "
                f"{sorted(str(state) for state in command_states)}"
            )
        elif len(gripper_commands) > 1:
            errors.append(
                f"stage[{stage_index}] has duplicate gripper_command terms; "
                "at most one is allowed"
            )

        if _declares_controlled_subject(anchors):
            assert anchors is not None
            contact_targets: list[str] = []
            for predicate_index, predicate in enumerate(running_terms):
                if predicate.get("type") not in {"contact", "no_contact"}:
                    continue
                refs = (predicate.get("a"), predicate.get("b"))
                roles: list[tuple[str, str]] = []
                for ref in refs:
                    if not isinstance(ref, str):
                        continue
                    if ref == "robot":
                        roles.append(("robot", ref))
                        continue
                    anchor = anchors.get(ref)
                    owner = (
                        None
                        if anchor is None
                        else getattr(anchor, "attached_to", None)
                    )
                    role = (
                        "subject"
                        if ref == "subject" or owner == "subject"
                        else "target"
                    )
                    target_key = str(owner) if owner else ref
                    roles.append((role, target_key))
                actor_indices = [
                    index
                    for index, (role, _ref) in enumerate(roles)
                    if role in {"subject", "robot"}
                ]
                if len(roles) == 2 and len(actor_indices) != 1:
                    errors.append(
                        f"stage[{stage_index}].running[{predicate_index}] "
                        f"{predicate.get('type')} must relate the controlled "
                        "subject or robot to one distinct target"
                    )
                    continue
                if len(roles) != 2:
                    continue
                target_index = 1 - actor_indices[0]
                if roles[target_index][0] != "target":
                    errors.append(
                        f"stage[{stage_index}].running[{predicate_index}] "
                        f"{predicate.get('type')} has no distinct target"
                    )
                    continue
                target_ref = refs[target_index]
                target_anchor = (
                    anchors.get(target_ref)
                    if isinstance(target_ref, str)
                    else None
                )
                if (
                    target_anchor is not None
                    and not bool(getattr(target_anchor, "dynamic", False))
                ):
                    errors.append(
                        f"stage[{stage_index}].running[{predicate_index}] "
                        f"{predicate.get('type')} target {target_ref!r} must "
                        "be a dynamic physics body"
                    )
                    continue
                contact_targets.append(roles[target_index][1])
            unique_targets = sorted(set(contact_targets))
            if len(unique_targets) > 1:
                errors.append(
                    f"stage[{stage_index}] contact/no_contact terms must share "
                    f"one dynamic target, got {unique_targets}"
                )

    # Terminal Contact is measured through a dedicated fresh full-physics
    # endpoint channel, not its deliberately neutral AnchorSet residual. Reuse
    # the verdict adapter as the single semantic authority for which terminal
    # relation is supported. The first, role-free validation pass can enforce
    # topology but intentionally defers subject ownership; the grounded pass
    # below binds that role and the exact dynamic target before any motion.
    try:
        parsed_program = (
            data if program_object else TaskProgram.from_dict(raw)
        )
    except (KeyError, TypeError, ValueError):
        parsed_program = None
    if parsed_program is not None:
        from .anchors import AnchorSet
        from .verdict import terminal_contact_requirement

        contact_anchors = anchors if anchors is not None else AnchorSet()
        for stage_index, stage in enumerate(parsed_program.stages):
            try:
                requirement = terminal_contact_requirement(
                    stage,
                    contact_anchors,
                )
            except ValueError as exc:
                errors.append(f"stage[{stage_index}].terminal: {exc}")
                continue
            if requirement is None:
                continue
            if requirement.a == requirement.b:
                errors.append(
                    f"stage[{stage_index}].terminal contact must relate one "
                    "controlled subject anchor to one distinct target"
                )
                continue
            if anchors is None or not _declares_controlled_subject(anchors):
                continue
            target = anchors.get(requirement.b)
            if target is not None and not bool(
                getattr(target, "dynamic", False)
            ):
                errors.append(
                    f"stage[{stage_index}].terminal Contact target "
                    f"{requirement.b!r} must be a dynamic physics body"
                )
    return errors


def parse_program(
    json_text: str,
    instruction: str,
    provenance: str,
    anchors: AnchorSet | None = None,
) -> TaskProgram:
    """Parse + validate a VLM's JSON output into a TaskProgram."""
    if _TRACE_DIR is not None:
        tag = re.sub(r"[^A-Za-z0-9._-]", "_", provenance)
        n = len(list(_TRACE_DIR.glob("raw_response_*.txt"))) if _TRACE_DIR.exists() else 0
        _trace_write(f"raw_response_{n:02d}.{tag}.txt", json_text)
    text = json_text.strip()
    decoder = json.JSONDecoder()
    data: dict[str, Any] | None = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "stages" in candidate:
            data = candidate
            break
    if data is None:
        raise ValueError(
            f"invalid synthesized program ({provenance}): no JSON program object")
    # Reject malformed VLM output with a TYPED error BEFORE from_dict can crash
    # on it (e.g. a list where an anchor-id string belongs -> 'unhashable type',
    # or 'w' where a number belongs -> 'could not convert string to float').
    # The synthesis anchors are the pre-bind manifest vocabulary: the runtime
    # scene has not yet minted the signed material-frame anchors or bound the
    # contact roles, so the strict grounded-frame contract cannot be decided
    # here (validate_program's own docstring scopes the deferral to exactly
    # this pass). The strict pass re-runs on the grounded scene: at episode
    # start (runner's grounded re-validation) and, when the caller supplies
    # synthesize(post_validate=...), inside this same repair loop.
    errs = validate_program(data, anchors, require_grounded_frame_contract=False)
    if errs:
        raise ValueError(f"invalid synthesized program ({provenance}): " + "; ".join(errs[:4]))
    data["instruction"] = instruction
    data["provenance"] = provenance
    # Belt-and-suspenders: even if a malformed scalar slips past validate_program
    # (an unenumerated numeric field), surface it as a CLEAN typed error rather than
    # a raw 'could not convert string to float' from a primitive's from_dict.
    try:
        return TaskProgram.from_dict(data)
    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(f"invalid synthesized program ({provenance}): {type(e).__name__}: {e}")


# --- frontier backends (user supplies the key; we never touch keys) --------------
def _require_key(env_var: str, provider: str) -> str:
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"{provider} backend needs ${env_var}. Put your key in a chmod-600 env file "
            f"in the process environment before running; this module never "
            f"creates or stores keys. For an air-gapped run, synthesize and "
            f"review a program first, then pass it with --program-json.")
    return key


def _repair_user_prompt(
    original_user_prompt: str,
    raw_response: str,
    error: ValueError,
    *,
    repair_index: int = 1,
) -> str:
    """Request a complete regeneration after fail-closed validation."""
    repair = (
        original_user_prompt
        + "\n\nYour previous JSON was rejected by the deterministic validator. "
        "Regenerate the complete TaskProgram, preserving the instruction's "
        "meaning. Do not patch, insert, copy, reorder, or rewrite individual "
        "predicates from the rejected output. Return ONLY complete JSON.\n"
        + f"Validator error: {error}\n"
        + "Previous response:\n"
        + raw_response
    )
    trace_name = (
        "repair_prompt_user.txt"
        if repair_index == 1
        else f"repair_prompt_user_{repair_index}.txt"
    )
    _trace_write(trace_name, repair)
    return repair


def _parse_with_bounded_repairs(
    *,
    initial_text: str,
    initial_model: str,
    instruction: str,
    anchors: AnchorSet,
    original_user_prompt: str,
    provider: str,
    request: Callable[[str], tuple[str, str]],
    max_repairs: int = 2,
) -> TaskProgram:
    """Parse one VLM program with at most two validator-guided repairs."""
    text = initial_text
    model = initial_model
    for repair_index in range(max_repairs + 1):
        provenance = f"{provider}:{model}"
        if repair_index:
            provenance += f":repair{repair_index}"
        try:
            program = parse_program(
                text,
                instruction,
                provenance,
                anchors,
            )
            if _POST_VALIDATE is not None:
                grounded_errors = list(_POST_VALIDATE(program))
                if grounded_errors:
                    raise ValueError(
                        f"invalid synthesized program ({provenance}): "
                        "grounded: " + "; ".join(grounded_errors[:4])
                    )
            return program
        except ValueError as error:
            if repair_index >= max_repairs:
                raise
            repair_user = _repair_user_prompt(
                original_user_prompt,
                text,
                error,
                repair_index=repair_index + 1,
            )
            text, model = request(repair_user)
    raise RuntimeError("unreachable bounded VLM repair state")


def _anthropic_backend(instruction: str, anchors: AnchorSet) -> TaskProgram:  # pragma: no cover
    """Program synthesis through the ONE VLM call (:mod:`oap.vlm`).

    The request is unchanged -- image block first, then the text that refers
    back to its anchor ids -- it is just no longer a second, separately
    configured Anthropic client."""
    from oap.vlm import ask_vlm

    b64 = _image_payload()
    sys_p, usr_p = build_prompt(
        instruction,
        anchors,
        with_image=b64 is not None,
        reasoning_mode=PRODUCTION_PROMPT_REASONING_MODE,
    )
    # Reasoning-era models spend `max_completion_tokens` on hidden reasoning
    # BEFORE visible text; with the stage plan attached the 2048 budget was
    # consistently exhausted by reasoning alone and the completion came back
    # empty (23/24 in the first abl5 batch). Generation-side config only.
    text, model = ask_vlm(
        system=sys_p, user=usr_p, image_b64=b64, max_tokens=16384)
    def request(repair_user: str) -> tuple[str, str]:
        return ask_vlm(
            system=sys_p,
            user=repair_user,
            image_b64=b64,
            max_tokens=16384,
        )

    return _parse_with_bounded_repairs(
        initial_text=text,
        initial_model=model,
        instruction=instruction,
        anchors=anchors,
        original_user_prompt=usr_p,
        provider="anthropic",
        request=request,
    )


def _openrouter_backend(instruction: str, anchors: AnchorSet) -> TaskProgram:  # pragma: no cover
    """ONE key (OPENROUTER_API_KEY) -> dozens of models via the OpenAI-compatible
    gateway (set OAP_OPENROUTER_MODEL to a '<vendor>/<slug>'). No forced
    response_format (heterogeneous models); we rely on the prompt + parse_program's
    fence-stripping + validate_program. This is the cheapest way to A/B many VLMs."""
    _require_key("OPENROUTER_API_KEY", "openrouter")
    try:
        from openai import OpenAI  # OpenRouter is OpenAI-compatible
    except Exception as e:
        raise RuntimeError(f"`pip install openai` to use the openrouter backend ({e})")
    b64 = _image_payload()
    sys_p, usr_p = build_prompt(
        instruction,
        anchors,
        with_image=b64 is not None,
        reasoning_mode=PRODUCTION_PROMPT_REASONING_MODE,
    )
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"], max_retries=0,
                    timeout=float(os.environ.get("OAP_VLM_TIMEOUT", "90")))
    model = os.environ.get("OAP_OPENROUTER_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
    def request(user_text: str) -> str:
        # OpenAI-style multimodal content: image_url data URI first, text
        # after (mirrors the anthropic block order); plain string without one.
        user_content: Any = user_text if b64 is None else [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
            {"type": "text", "text": user_text},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                f"openrouter backend returned no text content (model {model})"
            )
        return content

    text = request(usr_p)
    return _parse_with_bounded_repairs(
        initial_text=text,
        initial_model=model,
        instruction=instruction,
        anchors=anchors,
        original_user_prompt=usr_p,
        provider="openrouter",
        request=lambda repair_user: (request(repair_user), model),
    )


register_backend("anthropic", _anthropic_backend)
register_backend("openrouter", _openrouter_backend)


def synthesize(instruction: str, anchors: AnchorSet, backend: str = "anthropic",
               *, trace_dir: Path | None = None,
               image_path: Path | None = None,
               post_validate=None) -> TaskProgram:
    """Synthesize a program via the named backend.

    Args:
        instruction: the natural-language task.
        anchors: the grounded scene the program may reference.
        backend: one of :func:`available_backends` ('anthropic' is the live default).
        trace_dir: when set, the verbatim prompts and every raw VLM reply are
            written here (episode evidence; parse failures included).
        image_path: the anchor-annotated observation PNG
            (:func:`oap.program.annotate.render_anchor_annotations`) the
            VLM backends attach as an image block; the exact bytes sent are
            archived in the trace. None = text-only, byte-identical to the
            historical request.

    Raises:
        ValueError: unknown backend name.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; have {available_backends()}")
    global _TRACE_DIR, _IMAGE_PATH, _POST_VALIDATE
    _TRACE_DIR = Path(trace_dir) if trace_dir is not None else None
    _IMAGE_PATH = Path(image_path) if image_path is not None else None
    _POST_VALIDATE = post_validate
    try:
        return _BACKENDS[backend](instruction, anchors)
    finally:
        _TRACE_DIR = None
        _IMAGE_PATH = None
        _POST_VALIDATE = None
