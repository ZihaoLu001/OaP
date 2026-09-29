"""Execute one objective program with the shared MPC and export measured feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .evidence import feedback_evidence
from .generation_adapter import input_path, validate_environment, external_workspace

REPOSITORY = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_path(root, value):
    return input_path(root, value)


def execution_spec(root, trial, program, output, seed, attempt):
    root, output = external_workspace(root), external_workspace(output)
    settings = read(root / "config.json")
    if settings.get("runtime_profile") != "offline_only":
        raise ValueError("This entry supports offline simulation only, not robot execution")
    if settings.get("budget") != "paper_global_250_v1":
        raise ValueError("This entry uses the global 250-cycle execution budget")
    task = trial.rsplit("_t", 1)[0]
    spec = settings["tasks"][task]
    data_root = workspace_path(root, settings["input_root"])
    inputs = {name: workspace_path(data_root, value) for name, value in spec["inputs"].items()}
    calibration = settings["robot_calibration"]
    for name in ("real_table_z_m", "sim_tcp_m"):
        value = calibration.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Set robot_calibration.{name} from your calibrated scene before execution")
    program = Path(program).resolve()
    selected = (output / "program" / spec["canonical_program"]).resolve()
    if not selected.is_relative_to(output / "program"):
        raise ValueError("canonical_program must be a relative registration path")
    # Keep model credentials and generation-process imports out of the executor.
    environment = {name: os.environ[name] for name in
        ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH",
         "OAP_CONFIG_ROOT")
        if name in os.environ}
    environment.update(settings["runtime_environment"])
    environment.update(spec.get("runtime_environment", {}))
    validate_environment(environment)
    if "OAP_LAB_CUP_FIXTURE" in environment:
        environment["OAP_LAB_CUP_FIXTURE"] = str(workspace_path(data_root, environment["OAP_LAB_CUP_FIXTURE"]))
    if not 0 <= seed <= 2**31 - 1 or attempt not in (0, 1, 2):
        raise ValueError("Use a controller seed from 0 to 2147483647 and execution index 0, 1 or 2")
    environment.update(
        PYTHONPATH=str(REPOSITORY / "src"), PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", OAP_SOURCE=str(REPOSITORY),
        OAP_INPUT_ROOT=str(data_root),
        OAP_SYNTH_DOCTRINE="1", JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES="0",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", MUJOCO_GL="egl",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        OAP_EVAL_DEVICE_SEED=str(seed),
        OAP_JVF_SCENE_MANIFEST=str(inputs["bundle_path"]),
        OAP_ASSET_WORKSPACE=str(inputs["assets_path"]),
        OAP_CALIBRATION_SCENE_XML=str(inputs["calibration_xml_path"]))
    command = [sys.executable, "-B", "-m", "oap.cli.fixed_budget_run",
        "--bundle", str(inputs["bundle_path"]), "--task", spec["instruction"],
        "--program-json", str(selected), "--initial-obs-json", str(inputs["initial_obs_path"]),
        "--out", str(output / "result"), "--offline",
        "--real-table-z", str(calibration["real_table_z_m"]),
        "--sim-tcp-m", str(calibration["sim_tcp_m"]),
        "--sim-tcp-anchor", calibration["sim_tcp_anchor"],
        "--unified-mppi-effort-profile", spec["profile"], "--seed", str(seed),
        "--vlm-generated-program"]
    # Measured table (size, centre, tilt) and the robot's base plate, as the hardware runner uses.
    if calibration.get("table_spec_path"):
        command += ["--table-spec-json", str(workspace_path(data_root, calibration["table_spec_path"]))]
    job = {"id": f"{trial}_a{attempt}", "task": task, "trial": trial,
        "attempt": attempt, "seed": seed, "program_path": str(program),
        "program_sha256": digest(program), "output": str(output)}
    return job, command, environment, selected


def run(root, trial, program, output, seed, attempt):
    job, command, environment, selected = execution_spec(
        root, trial, program, output, seed, attempt)
    if attempt > 0:
        receipt_path = Path(program).resolve().parent / "receipt.json"
        receipt = read(receipt_path)
        if (receipt.get("status") != "complete" or receipt.get("execution_eligible") is not True
                or receipt.get("program_sha256") != digest(program)
                or receipt.get("trial") != trial or receipt.get("attempt") != attempt):
            raise ValueError("Feedback execution requires a valid objective-changing revision receipt")
    output = Path(job["output"])
    # An execution is never silently repeated in an existing output directory.
    output.mkdir(parents=True, exist_ok=False)
    selected.parent.mkdir(parents=True)
    shutil.copyfile(job["program_path"], selected)
    write(output / "identity.json", {**job, "command": command,
        "environment": {k: v for k, v in environment.items() if k.startswith("OAP_")}})
    try:
        with (output / "run.log").open("wb") as log:
            result = subprocess.run(command, cwd=REPOSITORY, env=environment,
                stdout=log, stderr=subprocess.STDOUT, timeout=4 * 3600)
        episode = read(output / "result" / "episode.json")
        identity = episode.get("config", {}).get("planner", {}).get("experiment_identity", {})
        if (result.returncode not in (0, 1) or not episode.get("chunks")
                or episode.get("finished_unix_s") is None
                or "EXECUTION_ABORTED" in str(episode.get("outcome", ""))
                or identity.get("device_seed") != seed
                or identity.get("proposal_seed_policy") != "explicit_evaluation_device_seed"):
            raise RuntimeError("Execution did not finish with the requested controller seed")
        if digest(selected) != job["program_sha256"]:
            raise RuntimeError("Selected program changed during execution")
        measured = any((row.get("execution_evidence") or {}).get("measured_movable_state_after")
            for row in episode["chunks"])
        if measured:
            write(output / "evidence.json", feedback_evidence(job, episode))
        else:
            write(output / "evidence.json", {"schema": "oap_no_measured_execution",
                "source_run_id": job["id"], "reason": "no_measured_execution_snapshots"})
        write(output / "completion.json", {"status": "COMPLETED", "returncode": result.returncode})
    except Exception as error:
        write(output / "completion.json", {"status": "INCOMPLETE", "error_type": type(error).__name__,
            "error": str(error)})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Workspace containing config.json")
    parser.add_argument("--trial", required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attempt", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--run", action="store_true", help="Execute one offline GPU simulation")
    args = parser.parse_args()
    if args.run:
        print(run(args.root, args.trial, args.program, args.out, args.seed, args.attempt))
    else:
        job, command, _, _ = execution_spec(args.root, args.trial, args.program,
            args.out, args.seed, args.attempt)
        print(json.dumps({"status": "COMMAND_ONLY", "job": job, "command": command}, indent=2))


if __name__ == "__main__":
    main()
