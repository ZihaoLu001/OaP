"""Measured feedback, joined to the stage that produced each control chunk.

Per-term satisfaction is the recorded evaluation of the executed program,
not a task-success label.
"""

from copy import deepcopy
import math


SCHEMA = "oap_execution_feedback_v2"
MEASURED_FIELDS = {"measured_movable_state_after", "subject_held", "gripper_width_m",
                   "measured_gripper_pose", "rotation_progress_reference"}
BODY_FIELDS = {"pose7", "linear_velocity_mps", "angular_velocity_rps",
               "velocity_observable", "source"}
CONTACT_FIELDS = {
    "subject_movable_frames", "robot_movable_frames", "max_subject_movable_pen_m",
    "max_robot_movable_pen_m", "subject_target_frames_by_body",
    "robot_target_frames_by_body", "max_subject_target_pen_m_by_body",
    "max_robot_target_pen_m_by_body", "required_subject_target_frames_by_body",
    "mediation_target_bodies", "subject_two_sided_frames",
    "subject_two_sided_recent", "subject_two_sided_now",
}
TERM_FIELDS = {"term_index", "type", "anchors", "residual", "satisfied",
               "measured", "unmeasured_channels"}
ROTATION_FIELDS = {"angle_deg", "min_angle_deg", "axis_initial_raw_body",
                   "reference_raw_frame"}
RUNNING_FIELDS = TERM_FIELDS | {"source", "unit", "residual_kind"}
GRIPPER_POSE_FIELDS = {"position_m", "tool_axis", "closing_axis", "frame", "source"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, allowed, label, required=()):
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(not set(value) - allowed and set(required) <= set(value),
             f"Unexpected or missing {label} fields")


def _integer(value):
    return type(value) is int and value >= 0


def _stage(value):
    _require(_integer(value.get("stage_index")), "Missing recorded stage index")
    _require(isinstance(value.get("stage_name"), str) and value["stage_name"].strip(),
             "Missing recorded stage name")


def _terms(rows, *, running=False):
    label = "running" if running else "terminal"
    fields = RUNNING_FIELDS if running else TERM_FIELDS
    _require(isinstance(rows, list), f"{label.capitalize()} measurements must be a list")
    for index, row in enumerate(rows):
        _fields(row, fields | ROTATION_FIELDS, f"{label} measurement", fields)
        _require(row["term_index"] == index and isinstance(row["type"], str),
                 f"Invalid {label} term identity")
        _require(isinstance(row["anchors"], list)
                 and all(isinstance(x, str) for x in row["anchors"]),
                 f"Invalid {label} anchors")
        _require(type(row["measured"]) is bool
                 and (row["satisfied"] is None or type(row["satisfied"]) is bool),
                 f"Invalid {label} observation flags")
        _require(isinstance(row["unmeasured_channels"], list)
                 and all(isinstance(x, str) for x in row["unmeasured_channels"]),
                 "Invalid unmeasured channels")
        residual = row["residual"]
        _require(residual is None or (type(residual) in (int, float)
                 and math.isfinite(residual)), f"Invalid measured {label} residual")
        if ROTATION_FIELDS & set(row):
            _require(row["type"] == "rotation_progress" and ROTATION_FIELDS <= set(row),
                     "Incomplete or misplaced recorded rotation metadata")
            for key in ("angle_deg", "min_angle_deg"):
                _require(type(row[key]) in (int, float) and math.isfinite(row[key]),
                         "Invalid recorded rotation angle")
            _vector3(row["axis_initial_raw_body"], "initial material axis")
            _frame(row["reference_raw_frame"])
        if running:
            _require(row["residual_kind"] in {"predicate_residual", "raw_table_i_residual"},
                     "Running observations must label raw or legacy predicate residuals, not predicted costs")
            _require(row["source"] in {"measured_anchor_state", "acknowledged_command", "unavailable"}
                     and row["unit"] in {"m", "rad", "dimensionless", "unknown"},
                     "Invalid running measurement provenance or unit")
            if row["measured"]:
                _require(residual is not None and type(row["satisfied"]) is bool
                         and not row["unmeasured_channels"] and row["source"] != "unavailable",
                         "Missing measured running value or source")
            else:
                _require(residual is None and row["satisfied"] is None
                         and row["unmeasured_channels"] and row["source"] == "unavailable",
                         "Unmeasured running term has a value")


def _vector3(value, label):
    _require(isinstance(value, list) and len(value) == 3
             and all(type(x) in (int, float) and math.isfinite(x) for x in value),
             f"Invalid {label}")


def _frame(value):
    _require(isinstance(value, list) and len(value) == 3, "Invalid recorded raw frame")
    for axis in value:
        _vector3(axis, "recorded raw frame axis")


def _rotation_reference(value):
    _fields(value, {"source", "raw_frames_by_body"}, "rotation reference",
            {"source", "raw_frames_by_body"})
    _require(value["source"] in {"first_recorded_measured_endpoint",
                                 "first_recorded_zero_action_endpoint",
                                 "episode_start_before_first_control"},
             "Rotation reference is not a recorded initial endpoint")
    frames = value["raw_frames_by_body"]
    _require(isinstance(frames, dict) and frames, "Missing recorded body reference")
    for name, frame in frames.items():
        _require(isinstance(name, str) and name, "Invalid recorded body name")
        _frame(frame)


def _gripper_pose(pose):
    _fields(pose, GRIPPER_POSE_FIELDS, "measured gripper pose", GRIPPER_POSE_FIELDS)
    _require(pose["source"] == "current_joint_state_fk" and pose["frame"] == "plan_world",
             "Gripper pose requires current joint-state FK in the plan-world frame")
    for key in ("position_m", "tool_axis", "closing_axis"):
        vector = pose[key]
        _require(isinstance(vector, list) and len(vector) == 3
                 and all(type(x) in (int, float) and math.isfinite(x) for x in vector),
                 f"Invalid measured gripper {key}")
        if key != "position_m":
            _require(any(x != 0 for x in vector), f"Missing measured gripper {key}")


def measurement_evidence(data):
    """Validate the v2 measurement envelope before any model request is made."""
    allowed = {"schema", "task", "executed_program_sha256", "attempt",
               "recorded_chunks", "task_roles", "measurements", "source_run_id",
               "stage_history"}
    _fields(data, allowed, "feedback envelope", allowed)
    _require(data["schema"] == SCHEMA,
             "Feedback requires stage-aligned v2 evidence")
    _require(_integer(data["attempt"]) and _integer(data["recorded_chunks"]),
             "Invalid feedback attempt or chunk count")
    _fields(data["task_roles"], {"source", "subject", "reference"}, "task roles")
    rows = data["measurements"]
    _require(isinstance(rows, list) and 1 <= len(rows) <= 6,
             "Feedback requires one to six measured snapshots")
    previous = -1
    for row in rows:
        required = {"index", "chunk_index", "stage_index", "stage_name",
                    "execution_evidence", "execution_contact_cumulative",
                    "terminal_measurements"}
        _fields(row, required | {"running_measurements"}, "snapshot", required)
        _stage(row)
        _require(_integer(row["index"]) and _integer(row["chunk_index"])
                 and row["chunk_index"] > previous, "Invalid snapshot chunk order")
        previous = row["chunk_index"]
        measured = row["execution_evidence"]
        _fields(measured, MEASURED_FIELDS, "execution evidence",
                {"measured_movable_state_after"})
        states = measured["measured_movable_state_after"]
        _require(isinstance(states, dict) and states, "Missing measured object states")
        for state in states.values():
            _fields(state, BODY_FIELDS, "measured body", {"pose7"})
        if "measured_gripper_pose" in measured:
            _gripper_pose(measured["measured_gripper_pose"])
        if "rotation_progress_reference" in measured:
            _rotation_reference(measured["rotation_progress_reference"])
        _fields(row["execution_contact_cumulative"], CONTACT_FIELDS, "contact evidence")
        _terms(row["terminal_measurements"])
        if "running_measurements" in row:
            _terms(row["running_measurements"], running=True)
    history = data["stage_history"]
    _require(isinstance(history, list) and history, "Missing measured stage history")
    previous = -1
    for stage in history:
        required = {"stage_index", "stage_name", "first_chunk_index", "last_chunk_index",
                    "measured_chunks", "last_terminal_measurements"}
        _fields(stage, required | {"last_running_measurements"}, "stage history", required)
        _stage(stage)
        first, last = stage["first_chunk_index"], stage["last_chunk_index"]
        _require(_integer(first) and _integer(last) and previous < first <= last
                 and type(stage["measured_chunks"]) is int and stage["measured_chunks"] > 0,
                 "Invalid measured stage history interval")
        previous = last
        _terms(stage["last_terminal_measurements"])
        if "last_running_measurements" in stage:
            _terms(stage["last_running_measurements"], running=True)
    for row in rows:
        aligned = [s for s in history
                   if s["first_chunk_index"] <= row["chunk_index"] <= s["last_chunk_index"]]
        _require(len(aligned) == 1 and all(row[k] == aligned[0][k]
                 for k in ("stage_index", "stage_name")), "Snapshot and stage history differ")
    # Check nested dynamic body-name maps too: no oracle, independent acceptance
    # result, prescribed correction, or aggregate runtime verdict enters a prompt.
    forbidden = {"success", "passed", "outcome", "grade", "grader", "oracle",
                 "primary", "tolerance", "threshold", "recommended_fix", "status"}
    def inspect(value):
        if isinstance(value, dict):
            _require(not set(value) & forbidden, "Feedback contains evaluation-only fields")
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)
        elif isinstance(value, float):
            _require(math.isfinite(value), "Feedback contains a non-finite observation")
    inspect(data)
    return data


def feedback_observations(data):
    """Build a prompt-only copy, labeling held state as a controller report.

    The raw v2 envelope stays unchanged. No physical grasp verdict is derived
    from this flag or from the measured width, contacts, and object states.
    """
    observations = deepcopy(measurement_evidence(data))
    for row in observations["measurements"]:
        measured = row["execution_evidence"]
        if "subject_held" in measured:
            measured["controller_reported_held"] = measured.pop("subject_held")
    return observations


def _terminal_measurements(chunk):
    gate = chunk.get("terminal_gate_evidence")
    _require(isinstance(gate, dict) and isinstance(gate.get("measured_residuals"), list),
             f"Missing recorded terminal measurements for chunk {chunk['chunk_index']}")
    rows = []
    for index, term in enumerate(gate["measured_residuals"]):
        row = {key: deepcopy(term[key])
               for key in (TERM_FIELDS | ROTATION_FIELDS) - {"term_index"} if key in term}
        row["term_index"] = index
        rows.append(row)
    _terms(rows)
    return rows


def feedback_evidence(job, episode):
    """Reconstruct feedback input without altering the recorded episode.

    A trajectory-cost record explicitly names the stage and chunk it planned.
    The terminal evaluation stored in that same chunk is the post-execution
    measurement of that stage, including the cycle on which a stage completes.
    No interpolation, next-stage assignment, or timestamp matching is used.
    Optional FK and running observations are copied only from the execution
    logger. Selected/predicted rollout states are never substituted for missing
    measurements. Running predicate residuals are not smooth planning costs.
    """
    stage_by_chunk = {}
    for call in episode.get("call_sites", []):
        if call.get("site") != "trajectory_cost":
            continue
        payload = call.get("payload") or {}
        _stage(payload)
        chunk_index = payload.get("chunk_index")
        _require(_integer(chunk_index), "Missing trajectory-cost chunk identity")
        stage = (payload["stage_index"], payload["stage_name"])
        _require(chunk_index not in stage_by_chunk or stage_by_chunk[chunk_index] == stage,
                 f"Conflicting stage identities for chunk {chunk_index}")
        stage_by_chunk[chunk_index] = stage
    chunks = episode.get("chunks", [])
    eligible = []
    previous = -1
    for index, chunk in enumerate(chunks):
        measured = chunk.get("execution_evidence") or {}
        if not measured.get("measured_movable_state_after"):
            continue
        chunk_index = chunk.get("chunk_index")
        _require(_integer(chunk_index) and chunk_index > previous,
                 "Measured chunk indices must be unique and ordered")
        previous = chunk_index
        _require(chunk_index in stage_by_chunk,
                 f"No recorded planning-stage identity for measured chunk {chunk_index}")
        stage_index, stage_name = stage_by_chunk[chunk_index]
        eligible.append({"index": index, "chunk_index": chunk_index,
            "stage_index": stage_index, "stage_name": stage_name,
            "execution_evidence": {key: deepcopy(measured[key])
                                   for key in MEASURED_FIELDS if key in measured},
            "execution_contact_cumulative": {key: deepcopy(value) for key, value in
                (chunk.get("execution_contact_cumulative") or {}).items() if key in CONTACT_FIELDS},
            "terminal_measurements": _terminal_measurements(chunk)})
        if "running_measurements" in measured:
            # Preserve the logger's order, term identities, and source labels.
            # The strict validator below rejects malformed or evaluation-only
            # fields instead of treating them as physical observations.
            eligible[-1]["running_measurements"] = deepcopy(measured["running_measurements"])
    _require(eligible, "Episode has no measured execution snapshots")
    history = []
    for row in eligible:
        if not history or (history[-1]["stage_index"], history[-1]["stage_name"]) != (
                row["stage_index"], row["stage_name"]):
            history.append({"stage_index": row["stage_index"], "stage_name": row["stage_name"],
                "first_chunk_index": row["chunk_index"], "last_chunk_index": row["chunk_index"],
                "measured_chunks": 0, "last_terminal_measurements": []})
        history[-1].update(last_chunk_index=row["chunk_index"],
            measured_chunks=history[-1]["measured_chunks"] + 1,
            last_terminal_measurements=deepcopy(row["terminal_measurements"]))
        if "running_measurements" in row:
            history[-1]["last_running_measurements"] = deepcopy(row["running_measurements"])
        else:
            history[-1].pop("last_running_measurements", None)
    indices = sorted({round(i * (len(eligible) - 1) / 5) for i in range(6)})
    roles = next((event["payload"] for event in episode.get("events", [])
                  if event.get("kind") == "task_roles"), {})
    data = {"schema": SCHEMA, "task": job["task"],
            "executed_program_sha256": job["program_sha256"], "attempt": job["attempt"],
            "recorded_chunks": len(chunks), "task_roles": deepcopy(roles),
            "measurements": [eligible[i] for i in indices], "stage_history": history,
            "source_run_id": job["id"]}
    return measurement_evidence(data)
