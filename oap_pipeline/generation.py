"""Generate, self-review and revise objective programs from measured execution."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import re
from datetime import datetime, timezone

if __package__:
    from .evidence import feedback_observations, measurement_evidence
    from .generation_adapter import external_workspace
else:
    from evidence import feedback_observations, measurement_evidence
    from generation_adapter import external_workspace

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts"
TASKS = ("pick", "cup", "push", "toolpush")
FEEDBACK_ARMS = ("draft_feedback", "review_feedback")
MAX_FEEDBACK_ROUNDS = 2


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_trial(trial):
    match = re.fullmatch(r"(pick|cup|push|toolpush)_t([0-9]+)", trial)
    if match is None or int(match.group(2)) < 1:
        raise ValueError("Trial id must be TASK_tNN with a positive trial number")
    return match.group(1)


def start_output(path, identity):
    """A completed invocation is idempotent; an interrupted one requires audit."""
    path = Path(path)
    receipt_path = path / "receipt.json"
    if receipt_path.exists():
        return read_json(receipt_path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "STARTED.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(identity, utc=datetime.now(timezone.utc).isoformat()), stream)
    return None


def load_loop(root, trial):
    if __package__:
        from .generation_adapter import load_loop as load_generation_loop
    else:
        from generation_adapter import load_loop as load_generation_loop
    return load_generation_loop(root, trial)


def parse_program(loop, text, instruction, provenance, anchors, body_names):
    envelope = loop.first_json_object(text, "program")
    if envelope is None:
        bare = loop.first_json_object(text, "stages")
        envelope = None if bare is None else {"program": bare, "reasoning": {}}
    if not isinstance(envelope, dict) or not isinstance(envelope.get("program"), dict):
        return None, {}, ["No JSON objective program"]
    source = copy.deepcopy(envelope["program"])
    # to_program strips movable_objects for the old parser. Never let that
    # strip affect the raw envelope or either paired branch's final program.
    try:
        program, errors = loop.to_program(copy.deepcopy(source), instruction,
            provenance, anchors, known_bodies=body_names)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return None, envelope, [f"Malformed program: {type(exc).__name__}"]
    if program is None:
        return None, envelope, errors
    final = loop.attach_movables(json.loads(program.to_json()), source)
    return final, envelope, []


def same_content(a, b):
    return {k: v for k, v in a.items() if k != "provenance"} == {
        k: v for k, v in b.items() if k != "provenance"}


def objective_content(program):
    """Admission ignores description-only changes."""
    value = {key: val for key, val in program.items()
             if key not in ("instruction", "provenance", "notes")}
    value["stages"] = [{key: val for key, val in stage.items()
                        if key not in ("name", "notes")} for stage in program["stages"]]
    return value


def objective_user(root, user_base, block):
    """Every stateless model request includes the same objective design rules."""
    shared = (root / "objective_design_rules.txt").read_text(encoding="utf-8")
    return user_base.replace("Return ONLY the JSON program.", "") + "\n" + shared + "\n" + block


def draft_user(root, user_base):
    return objective_user(root, user_base, (root / "draft_prompt.txt").read_text(encoding="utf-8"))


def review_user(root, user_base, envelope):
    block = (root / "review_prompt.txt").read_text(encoding="utf-8").replace(
        "__PROGRAM__", json.dumps(envelope["program"], indent=1)).replace(
        "__REASONING__", json.dumps(envelope.get("reasoning", {}), indent=1))
    return objective_user(root, user_base, block)


def feedback_user(root, user_base, current, data):
    block = (root / "feedback_prompt.txt").read_text(encoding="utf-8").replace(
        "__PROGRAM__", json.dumps(current, indent=1)).replace(
        "__MEASUREMENTS__", json.dumps(feedback_observations(data), indent=1))
    return objective_user(root, user_base, block)


def request(loop, root, output, system, user, image, tag):
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt_system.txt").write_text(system, encoding="utf-8")
    (output / "prompt_user.txt").write_text(user, encoding="utf-8")
    try:
        text, record = loop.ask(system, user, image, tag)
    except (Exception, SystemExit) as exc:
        # API/transport failures are not malformed model output, and are never
        # converted into a new draw or a task failure by this wrapper.
        save_json(output / "request_failure.json", {"tag": tag, "status": "INCOMPLETE",
                                                    "error_type": type(exc).__name__})
        raise
    (output / "raw.txt").write_text(text, encoding="utf-8")
    attempts = [p for p in (root / "api_attempts").glob("attempt_*.json")
                if ".result." not in p.name and read_json(p).get("tag") == tag]
    save_json(output / "request.json", dict(record, tag=tag,
        attempt_files=[str(p.relative_to(root)) for p in attempts]))
    return text


def bounded_program(loop, root, output, system, user, image, tag,
                    instruction, anchors, body_names, *, repair):
    provenance = f"openai:{loop.MODEL}:{tag}"
    text = request(loop, root, output / "request0", system, user, image, tag)
    final, envelope, errors = parse_program(loop, text, instruction, provenance, anchors, body_names)
    if final is None and repair:
        repair_user = (user + "\n\nYour previous JSON was rejected by the validator. Return the "
            "complete {\"program\": ..., \"reasoning\": ...} object again. "
            "Correct only the reported validation errors. Preserve all other "
            "stages, objectives, parameters and completion conditions from your previous "
            "response; this request is not another objective refinement. "
            "Every stage must include name, running, terminal and movable_objects.\n"
            f"Validator error: {'; '.join(errors[:4])}\nPrevious response:\n{text}")
        text = request(loop, root, output / "repair1", system, repair_user, image, tag + ":repair1")
        final, envelope, errors = parse_program(loop, text, instruction, provenance + ":repair1", anchors, body_names)
    save_json(output / "validation.json", {"valid": final is not None, "errors": errors})
    return final, envelope


def generate(root, trial, *, loaded=None):
    root = external_workspace(root)
    task = check_trial(trial)
    out = root / "generation" / trial
    prior = start_output(out, {"mode": "generate", "trial": trial})
    if prior is not None:
        return prior
    receipt = {"trial": trial, "task": task, "status": "INCOMPLETE",
               "draft_shared_by": ["draft_first", "draft_feedback"],
               "review_shared_by": ["review_first", "review_feedback"]}
    try:
        loop, _ = loaded or load_loop(root, trial)
        prompts = getattr(loop, "prompt_root", PROMPTS)
        anchors, image, instruction, system, user_base, names = loop.scene_context(task, out / "context")
        draft, envelope = bounded_program(loop, root, out / "draft_request", system,
            draft_user(prompts, user_base), image, trial + ":draft", instruction, anchors, names, repair=True)
        if draft is None:
            receipt.update(status="generation_failed", reason="draft_invalid_after_one_repair")
        else:
            save_json(out / "draft.json", draft)
            save_json(out / "draft_reasoning.json", envelope.get("reasoning", {}))
            review, _ = bounded_program(loop, root, out / "review_request", system,
                review_user(prompts, user_base, envelope), image, trial + ":review", instruction, anchors, names, repair=False)
            rejected = review is None
            changed = review is not None and not same_content(draft, review)
            if rejected or not changed:
                shutil.copyfile(out / "draft.json", out / "review.json")
            else:
                save_json(out / "review.json", review)
            receipt.update(status="complete", review_rejected=rejected, review_changed=changed,
                draft_sha256=digest(out / "draft.json"), review_sha256=digest(out / "review.json"))
    except (Exception, SystemExit) as exc:
        receipt.update(error_type=type(exc).__name__)
    save_json(out / "receipt.json", receipt)
    return receipt


def feedback(root, trial, arm, attempt, program, evidence, *, loaded=None):
    root = external_workspace(root)
    task = check_trial(trial)
    if arm not in FEEDBACK_ARMS or attempt not in range(1, MAX_FEEDBACK_ROUNDS + 1):
        raise ValueError("Unregistered feedback arm or attempt")
    expected = root / "generation" / trial / ("draft.json" if arm.startswith("draft") else "review.json")
    if attempt > 1:
        expected = root / "feedback" / trial / arm / f"r{attempt - 1}" / "program.json"
    if Path(program).resolve() != expected.resolve():
        raise ValueError("Input must be this trial/arm's immediately preceding program")
    data = measurement_evidence(read_json(evidence))
    if (data.get("task") != task or data.get("executed_program_sha256") != digest(program)
            or data.get("attempt") != attempt - 1):
        raise ValueError("Measured feedback differs from task, executed program, or preceding attempt")
    out = root / "feedback" / trial / arm / f"r{attempt}"
    prior = start_output(out, {"mode": "feedback", "trial": trial, "arm": arm, "attempt": attempt})
    if prior is not None:
        return prior
    receipt = {"trial": trial, "arm": arm, "attempt": attempt, "status": "INCOMPLETE",
               "input_program_sha256": digest(program), "evidence_sha256": digest(evidence)}
    try:
        loop, _ = loaded or load_loop(root, trial)
        prompts = getattr(loop, "prompt_root", PROMPTS)
        anchors, image, instruction, system, user_base, names = loop.scene_context(task, out / "context")
        current = read_json(program)
        user = feedback_user(prompts, user_base, current, data)
        final, _ = bounded_program(loop, root, out / "revision", system, user, image,
            f"{trial}:{arm}:r{attempt}", instruction, anchors, names, repair=True)
        rejected = final is None
        changed = final is not None and not same_content(current, final)
        if rejected or not changed:
            shutil.copyfile(program, out / "program.json")
        else:
            save_json(out / "program.json", final)
        receipt.update(status="complete", revision_rejected=rejected, changed=changed,
                       execution_eligible=not rejected and objective_content(current) != objective_content(final),
                       program_sha256=digest(out / "program.json"))
    except (Exception, SystemExit) as exc:
        receipt.update(error_type=type(exc).__name__)
    save_json(out / "receipt.json", receipt)
    return receipt


def initialize(root):
    """Create settings in an external workspace; scene data is supplied separately."""
    root = external_workspace(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "config.json").open("x", encoding="utf-8") as stream:
        stream.write((ROOT / "config.example.json").read_text(encoding="utf-8"))
    return {"status": "initialized", "root": str(root),
            "config": str(root / "config.json")}


def preview(root, trial):
    """Save the exact draft prompts without a model request or GPU execution."""
    root = external_workspace(root)
    task = check_trial(trial)
    loop, _ = load_loop(root, trial)
    out = root / "preview" / trial
    anchors, image, instruction, system, user, names = loop.scene_context(task, out)
    (out / "prompt_system.txt").write_text(system, encoding="utf-8")
    (out / "prompt_user.txt").write_text(draft_user(loop.prompt_root, user), encoding="utf-8")
    return {"status": "preview", "directory": str(out), "image": str(image),
            "image_exists": image.is_file(), "model": loop.MODEL,
            "residual_library": loop.residual_library}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="External workspace containing config.json and outputs")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create editable configuration for external scene inputs")
    preview_parser = commands.add_parser("preview", help="Write draft prompts without an API call")
    preview_parser.add_argument("--trial", required=True)
    draft = commands.add_parser("generate")
    draft.add_argument("--trial", required=True)
    revise = commands.add_parser("feedback")
    revise.add_argument("--trial", required=True)
    revise.add_argument("--arm", choices=FEEDBACK_ARMS, required=True)
    revise.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    revise.add_argument("--program", type=Path, required=True)
    revise.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = external_workspace(args.root)
    if args.command == "init":
        result = initialize(root)
    elif args.command == "preview":
        result = preview(root, args.trial)
    elif args.command == "generate":
        result = generate(root, args.trial)
    else:
        result = feedback(root, args.trial, args.arm, args.attempt, args.program, args.evidence)
    print(json.dumps(result, indent=2))
    raise SystemExit(2 if result["status"] == "INCOMPLETE" else 0)


if __name__ == "__main__":
    main()
