"""Load objective generation with external scene and model settings."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parent
GENERATOR_SOURCE = CODE_ROOT.parent
VENDOR = CODE_ROOT / "vendor" / "generation"
REPOSITORY = CODE_ROOT.parent
EXECUTION_FACTS = '''\nEXECUTION SETTINGS:
In simulation, ObjectHeld costs use bilateral finger contact, and stage
completion uses the measured bilateral-contact evidence. Running ObjectHeld
is a finite penalty, with no additional prefix-hold rollout mask.
The controller uses its smoothed weighted-mean sequence without best-sample replacement.
The closing-force cap is 25 N. A run allows at most 250 MPC updates.
Each update plans over 600 physics steps and executes its first 40 controls.
These are controller settings, not additional objective types.
'''
_LOADED_LIBRARY = None


def external_workspace(root):
    """Keep generated requests, programs and logs outside the source checkout."""
    root = Path(root).resolve()
    if root.is_relative_to(REPOSITORY):
        raise ValueError("Choose a workspace outside the source repository")
    return root


def validate_environment(environment):
    """Keep the configured objective and controller conventions consistent."""
    expected = {
        "OAP_JAW_CLOSE_FORCE_N": "25",
        "OAP_BLUE_RUNNING_NORM": "1",
        "OAP_RUNNING_SMOOTH_EPS": "0",
        "OAP_DECLARED_SCALE_EXACT": "1",
        "OAP_EXEC_SHAPING_OVERRIDE": "typed_residual_library_v4",
        "OAP_PREFIX_HOLD_BARRIER": "soft_only",
        "OAP_SIM_HELD_EVIDENCE": "frozen",
    }
    for key, value in expected.items():
        if environment.get(key) != value:
            raise ValueError(f"The controller profile requires {key}={value}")
    if float(environment.get("OAP_TERMINAL_GATE_TOL_FLOOR", "0")) != 0:
        raise ValueError("Terminal thresholds must not be enlarged by a tolerance floor.")


def input_path(root, value):
    """Resolve an external input path relative to its configured root."""
    return (Path(root) / Path(value).expanduser()).resolve()


def library_paths(settings, task):
    library = settings["tasks"][task]["residual_library"]
    expected = "paper_table1_12"
    if library != expected:
        raise ValueError(f"{task} requires the {expected} residual library")
    return library, REPOSITORY, VENDOR, CODE_ROOT / "prompts"


def read_settings(root):
    """Resolve configured input paths against the workspace configuration."""
    root = external_workspace(root)
    settings = json.loads((root / "config.json").read_text(encoding="utf-8"))
    data_root = input_path(root, settings["input_root"])
    settings["input_root"] = str(data_root)
    for spec in settings["tasks"].values():
        environment = {**settings["runtime_environment"], **spec.get("runtime_environment", {})}
        validate_environment(environment)
        spec["context_path"] = str(input_path(data_root, spec["context_path"]))
        spec["inputs"] = {key: str(input_path(data_root, value))
                          for key, value in spec["inputs"].items()}
    cap = settings["total_api_call_cap"]
    if type(cap) is not int or cap < 1:
        raise ValueError("total_api_call_cap must be a positive integer")
    return settings


def track_attempts(ask, root, limit, trial_id=None):
    """Reserve each attempted request atomically, including failed requests.

    Exclusive file creation is shared by concurrent processes on the same
    output filesystem. A crashed or failed request never releases its slot.
    """
    directory = Path(root) / "api_attempts"

    def tracked(system, user, image, tag):
        label = tag.split(":", 1)[0]
        if trial_id is not None and label != trial_id:
            raise ValueError("Provider request belongs to another trial")
        directory.mkdir(parents=True, exist_ok=True)
        record = {"trial_id": label, "task": label.split("_t", 1)[0], "tag": tag,
                  "utc": datetime.now(timezone.utc).isoformat(), "status": "started"}
        for index in range(1, limit + 1):
            path = directory / f"attempt_{index:02d}.json"
            try:
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(record, stream)
                    stream.write("\n")
                break
            except FileExistsError:
                continue
        else:
            raise SystemExit(f"Global API attempt limit ({limit}) reached; no request sent")
        result_path = directory / f"attempt_{index:02d}.result.json"
        try:
            result = ask(system, user, image, tag)
        except BaseException as exc:
            record.update(utc=datetime.now(timezone.utc).isoformat(), status="failed", error_type=type(exc).__name__)
            with result_path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream)
                stream.write("\n")
            raise
        record.update(utc=datetime.now(timezone.utc).isoformat(), status="returned")
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream)
            stream.write("\n")
        return result

    return tracked



def load_loop(root, trial_id=None):
    global _LOADED_LIBRARY
    root = external_workspace(root)
    settings = read_settings(root)
    if trial_id is None:
        raise ValueError("A trial identifies the task and scene")
    task = trial_id.rsplit("_t", 1)[0]
    library, generator_source, vendor, prompts = library_paths(settings, task)
    if _LOADED_LIBRARY not in (None, library):
        raise ValueError("Use a separate process for each residual-library family")
    imported = sys.modules.get("oap")
    if imported is not None and not Path(imported.__file__).resolve().is_relative_to(generator_source.resolve()):
        raise ValueError("Start generation in a fresh process, separate from the execution runtime")
    _LOADED_LIBRARY = library
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OAP_", "FR_"))}
    env.update(settings["runtime_environment"])
    env.update(PYTHONPATH=str(generator_source / "src"),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OAP_SOURCE=str(generator_source), OAP_SYNTH_DOCTRINE="1",
               OAP_INPUT_ROOT=settings["input_root"],
               OAP_VLM_MODEL=settings["model"],
               FR_REASONING_EFFORT=settings["reasoning_effort"],
               FR_MAX_TOKENS=str(settings["max_completion_tokens"]),
               JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", MUJOCO_GL="egl")
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(generator_source / "src"))
    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("oap_generation_loop", vendor / "loop_v11.py")
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    loop.HERE = root
    loop.LEDGER = root / "logs" / "api_calls.jsonl"
    loop.CALL_CAP = settings["total_api_call_cap"]
    loop.prompt_root = prompts
    loop.residual_library = library
    loop.ask = track_attempts(loop.ask, root, loop.CALL_CAP, trial_id)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    def scene_context(task, output):
        from oap.program.anchors import Anchor, AnchorSet
        from oap.program.synthesis import build_prompt, PRODUCTION_PROMPT_REASONING_MODE
        task_spec = settings["tasks"][task]
        data = json.loads(Path(task_spec["context_path"]).read_text(encoding="utf-8"))
        if data["instruction"] != task_spec["instruction"]:
            raise ValueError("Scene context and configured instruction differ")
        anchors = AnchorSet({name: Anchor(**fields) for name, fields in data["anchors"].items()})
        execution_environment = {**settings["runtime_environment"],
                                 **task_spec.get("runtime_environment", {}),
                                 "OAP_SYNTH_DOCTRINE": "1"}
        if execution_environment.get("OAP_JAW_CLOSE_FORCE_N") != "25":
            raise ValueError("The paper configuration uses 25 N closing force")
        system, user = build_prompt(task_spec["instruction"], anchors, with_image=False,
                                    reasoning_mode=PRODUCTION_PROMPT_REASONING_MODE,
                                    execution_environment=execution_environment)
        user = ("The image shows the scene without anchor labels. "
                "Anchor ids and their geometry are provided in the text below.\n" + user)
        system = loop.trim_schema.trim_system(system)
        user = loop.trim_schema.trim_user(user)
        output.mkdir(parents=True, exist_ok=True)
        return (anchors, Path(task_spec["inputs"]["generation_image_path"]),
                task_spec["instruction"], system, user + EXECUTION_FACTS, tuple(data["body_names"]))

    loop.scene_context = scene_context
    if len(loop.trim_schema.ALLOWED) != 12:
        raise ValueError("Expected the twelve Table I residual types")
    return loop, vendor
