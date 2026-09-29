"""Client for the persistent external-env FoundationPose worker.

Role in the two-stage pipeline: the online loop (stage 2) calls these
functions to spawn each worker ONCE under its external-env interpreter, then
streams line-based JSONL requests over stdin/stdout with a ready-line
handshake and per-request deadlines. Keeping the workers resident is a
measured ~3x speedup over per-call subprocesses (model load dominates).

All worker stderr goes to per-worker log files; stdout stays a clean protocol
channel pumped by a daemon thread into a queue so reads can time out.
"""
from __future__ import annotations

import atexit
import importlib.resources
import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Any

import numpy as np

logger = logging.getLogger("oap.workers.client")

__all__ = [
    "FoundationPoseClient",
    "ZedStreamClient",
    "worker_script_path",
    "foundationpose_env",
    "fp_request",
    "fp_stop",
    "stop_all_workers",
]

_FP_WORKER: dict[str, Any] | None = None
_FP_LOCK = threading.RLock()
_ZED_STREAM_CLIENTS: list["ZedStreamClient"] = []


def worker_script_path(name: str) -> Path:
    """Resolve a worker script inside this package to a filesystem path.

    The workers are payload-style modules shipped as package files, so they can
    be handed verbatim to an EXTERNAL interpreter that has no ``oap``.

    Args:
        name: Worker file name, e.g. ``"foundationpose_worker.py"``.

    Returns:
        Absolute path of the worker script.
    """
    path = Path(str(importlib.resources.files("oap.workers") / name))
    if not path.exists():  # zip-installed wheels are not supported for workers
        raise FileNotFoundError(f"worker script not found on disk: {path}")
    return path


# --------------------------------------------------------------------------
# generic line-protocol plumbing
# --------------------------------------------------------------------------
def _pump(stdout: IO[str], out_q: "queue.Queue[str | None]") -> None:
    try:
        for line in stdout:
            out_q.put(line)
    finally:
        out_q.put(None)


def _spawn(argv: list[str], *, log_path: Path, env: dict[str, str] | None = None,
           cwd: Path | None = None) -> dict[str, Any]:
    """Start a worker subprocess with a pumped stdout queue and a stderr log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        argv, cwd=None if cwd is None else str(cwd), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_path.open("ab"),
        text=True, bufsize=1)
    q: "queue.Queue[str | None]" = queue.Queue()
    assert proc.stdout is not None
    threading.Thread(target=_pump, args=(proc.stdout, q), daemon=True).start()
    return {"proc": proc, "log": str(log_path), "queue": q}


def _readline(worker: dict[str, Any], tag: str, deadline: float, label: str) -> str:
    """Next protocol line starting with ``tag``, or raise on timeout/exit."""
    q = worker["queue"]
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"{label} worker response timeout")
        try:
            line = q.get(timeout=min(remaining, 5.0))
        except queue.Empty:
            if worker["proc"].poll() is not None:
                raise RuntimeError(f"{label} worker exited")
            continue
        if line is None:
            raise RuntimeError(f"{label} worker closed stdout")
        if line.startswith(tag):
            return str(line).strip()


def _stop(worker: dict[str, Any] | None, exit_line: str) -> None:
    if worker is None:
        return
    try:
        worker["proc"].stdin.write(exit_line + "\n")
        worker["proc"].stdin.flush()
        worker["proc"].wait(timeout=5)
    except Exception:
        try:
            worker["proc"].kill()
        except Exception:
            pass


def _request(worker: dict[str, Any], tag: str, req: dict[str, Any],
             deadline: float, label: str) -> dict[str, Any]:
    """Send one JSON request line and parse the tagged JSON response."""
    worker["proc"].stdin.write(json.dumps(req) + "\n")
    worker["proc"].stdin.flush()
    resp: dict[str, Any] = json.loads(
        _readline(worker, tag, deadline, label)[len(tag) + 1:])
    if not resp.get("ok"):
        raise RuntimeError(f"{label} worker error: {resp.get('error')}")
    return resp


# --------------------------------------------------------------------------
# FoundationPose worker (reconstruction-pose seed -> persistent track)
# --------------------------------------------------------------------------
def foundationpose_env(root: Path, py: Path) -> dict[str, str]:
    """Build the subprocess environment for the FoundationPose interpreter.

    Args:
        root: FoundationPose source root (holds ``estimater.py``).
        py: The foundationpose conda env's python executable.

    Returns:
        A copy of ``os.environ`` with CONDA/CUDA/LD_LIBRARY_PATH pointed at the
        env and PYTHONPATH arranged so ``import estimater`` works.
    """
    conda_prefix = py.parent.parent
    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(conda_prefix)
    env["CUDA_HOME"] = str(conda_prefix)
    env["PATH"] = f"{py.parent}{os.pathsep}" + env.get("PATH", "")
    ld = [str(conda_prefix / "lib"), str(conda_prefix / "targets" / "x86_64-linux" / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        ld.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld)
    # root is the FoundationPose dir (e.g. .../Any6D/foundationpose): put it on the
    # path for `import estimater`/`datareader`, AND its parent (.../Any6D) so the
    # absolute `from foundationpose.Utils import *` inside datareader resolves the
    # `foundationpose` package (matches the working Any6D PYTHONPATH layout).
    # ORDER MATTERS: root must come BEFORE root/mycpp/build. Utils does
    # `import mycpp.build.mycpp as mycpp`, which needs `mycpp` to resolve to the
    # PACKAGE at root/mycpp/ (with __init__.py). If root/mycpp/build is first,
    # `import mycpp` instead binds to the bare mycpp*.so module there, so
    # `mycpp.build` fails and Utils silently sets mycpp=None -> later
    # `mycpp.cluster_poses` AttributeErrors. Build dir last keeps it importable
    # without shadowing the package.
    pp = [str(root), str(root.parent), str(root / "mycpp" / "build")]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


def fp_stop() -> None:
    """Stop the FoundationPose worker (idempotent)."""
    global _FP_WORKER
    with _FP_LOCK:
        _stop(_FP_WORKER, "FPWORKER EXIT")
        _FP_WORKER = None


def fp_request(*, root: Path, py: Path, mesh: Path, recording_dir: Path,
               debug_dir: Path, mode: str, timeout_s: float,
               init_pose: np.ndarray | None = None,
               allow_fallback: bool = True,
               capture_id: str | None = None) -> dict[str, Any]:
    """One register/track/seed request to the persistent worker.

    The worker is MULTI-OBJECT: it holds one estimator per mesh (all sharing
    the heavy nets/glctx), so a request for a DIFFERENT object no longer tears
    the worker down. Each object keeps its own ``pose_last``. Normal first
    sight uses the reconstruction pose as a seed and refines it with
    ``track_one``; a global ``register`` is only the explicitly permitted
    fallback when that seed fails its mask-scored check. ``mesh`` is sent as
    ``mesh_file`` to select/build that object's estimator.

    ``init_pose`` (a 4x4 camera-frame T_cam_obj) is sent with mode="seed": the
    worker sets it as FoundationPose's previous pose and runs a track_one
    refine from it (the Any6D recipe -- OBB orientation prior +
    render-and-compare).

    Args:
        root: FoundationPose source root (worker cwd).
        py: The foundationpose env python.
        mesh: Object mesh file selecting the per-object estimator.
        recording_dir: Scene dir with the observation packet (YCBInEOAT layout).
        debug_dir: Where the worker writes the pose txt (+ sibling worker log).
        mode: ``"register"`` | ``"track"`` | ``"seed"``.
        timeout_s: Per-request deadline (capped at 300 s).
        init_pose: Optional 4x4 T_cam_obj prior for mode="seed".
        allow_fallback: False marks a PROBE -- return the low-confidence
            track verbatim instead of the ~2.6 GB register fallback (which
            also hallucinates under gripper occlusion).

    Returns:
        The worker response: ``pose_txt``, ``mode_used``, ``reregistered``,
        ``confidence``.
    """
    global _FP_WORKER
    with _FP_LOCK:
        worker_timeout = min(float(timeout_s), 300.0)
        if _FP_WORKER is not None and _FP_WORKER["proc"].poll() is not None:
            fp_stop()
        if _FP_WORKER is None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            log_path = debug_dir.parent / "foundationpose_worker.log"
            worker = _spawn(
                [str(py), str(worker_script_path("foundationpose_worker.py")),
                 "--mesh_file", str(mesh)],
                log_path=log_path, env=foundationpose_env(root, py), cwd=root)
            if _readline(worker, "FPWORKER", time.time() + worker_timeout,
                         "FoundationPose") != "FPWORKER READY":
                raise RuntimeError("unexpected FoundationPose worker handshake")
            _FP_WORKER = worker
        # mesh_file selects/builds the per-object estimator in the
        # multi-object worker.
        req: dict[str, Any] = {
            "test_scene_dir": str(recording_dir),
            "debug_dir": str(debug_dir),
            "mode": mode,
            "mesh_file": str(mesh),
            "allow_fallback": bool(allow_fallback),
        }
        if capture_id is not None:
            req["capture_id"] = str(capture_id)
        if init_pose is not None:
            req["init_pose"] = [
                float(v)
                for v in np.asarray(
                    init_pose, dtype=float
                ).reshape(-1)
            ]
        return _request(
            _FP_WORKER,
            "FPWORKER",
            req,
            time.time() + worker_timeout,
            "FoundationPose",
        )


# --------------------------------------------------------------------------
# Qwen3-VL live judge worker (VETO-ONLY consumer: loop.certify)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Concrete clients (what the loop constructs; satisfy the consumers' Protocols:
# loop.observe.FoundationPoseClient)
# --------------------------------------------------------------------------
class FoundationPoseClient:
    """Facade over :func:`fp_request` holding the env config (root + python)."""

    def __init__(self, *, py: Path | None, root: Path | None) -> None:
        if py is None or root is None:
            raise ValueError(
                "FoundationPoseClient needs the foundationpose interpreter and "
                "checkout root -- set hosts.<profile>.foundationpose.{python,root} "
                "in configs/external_envs.yaml (or the --foundationpose-* flags)")
        self.py, self.root = Path(py), Path(root)

    def request(self, *, mesh: Path, recording_dir: Path, debug_dir: Path,
                mode: str, timeout_s: float,
                init_pose: np.ndarray | None = None,
                allow_fallback: bool = True,
                capture_id: str | None = None) -> dict[str, Any]:
        """One register/seed/track call against the resident worker."""
        return fp_request(root=self.root, py=self.py, mesh=mesh,
                          recording_dir=recording_dir, debug_dir=debug_dir,
                          mode=mode, timeout_s=timeout_s, init_pose=init_pose,
                          allow_fallback=allow_fallback,
                          capture_id=capture_id)

    def stop(self) -> None:
        """Stop the resident worker (idempotent)."""
        fp_stop()


class ZedStreamClient:
    """Client for the episode-long, single-owner ZED RGB-D stream.

    The worker owns the physical camera. Snapshot requests and evidence-video
    windows are serialized over one JSONL channel, so continuous tracking and
    real-prefix recording never open competing ZED processes.
    """

    def __init__(
        self,
        *,
        py: Path | None,
        log_path: Path,
        timeout_s: float = 30.0,
        record_fps: float = 15.0,
    ) -> None:
        if py is None:
            raise ValueError(
                "ZedStreamClient needs the observer interpreter"
            )
        self.py = Path(py)
        self.timeout_s = float(timeout_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("ZED stream timeout_s must be finite and > 0")
        if not np.isfinite(float(record_fps)) or float(record_fps) <= 0.0:
            raise ValueError("ZED stream record_fps must be finite and > 0")
        script = Path(
            str(
                importlib.resources.files("oap.reconstruct")
                / "payloads"
                / "zed_stream_worker.py"
            )
        )
        if not script.exists():
            raise FileNotFoundError(
                f"ZED stream worker not found on disk: {script}"
            )
        self._lock = threading.Lock()
        self._closed = False
        self._worker = _spawn(
            [
                str(self.py),
                str(script),
                "--record-fps",
                str(float(record_fps)),
            ],
            log_path=Path(log_path),
        )
        ready = _readline(
            self._worker,
            "ZEDSTREAM",
            time.time() + self.timeout_s,
            "ZED stream",
        )
        if ready != "ZEDSTREAM READY":
            _stop(self._worker, '{"op":"exit"}')
            raise RuntimeError(
                f"unexpected ZED stream worker handshake: {ready!r}"
            )
        _ZED_STREAM_CLIENTS.append(self)

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("ZED stream client is closed")
            return _request(
                self._worker,
                "ZEDSTREAM",
                request,
                time.time() + self.timeout_s,
                "ZED stream",
            )

    def snapshot(
        self,
        *,
        out_dir: Path,
        objects: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Materialize one atomic latest RGB-D exposure for all objects."""
        if not objects:
            raise ValueError("ZED stream snapshot needs at least one object")
        return self._call(
            {
                "op": "snapshot",
                "out_dir": str(Path(out_dir).expanduser().resolve()),
                "objects": [
                    [str(name), str(prompt)]
                    for name, prompt in objects
                ],
            }
        )

    def start_recording(self, *, out_dir: Path) -> dict[str, Any]:
        """Begin an evidence window after saving its pre-motion frame."""
        return self._call(
            {
                "op": "start_recording",
                "out_dir": str(Path(out_dir).expanduser().resolve()),
            }
        )

    def stop_recording(
        self,
        *,
        checkpoint_out_dir: Path | None = None,
        objects: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Save one post-motion exposure as evidence and an RGB-D packet.

        When ``checkpoint_out_dir`` is supplied, the packet and final evidence
        frame come from the exact same camera grab.  No second capture is
        performed.
        """
        request: dict[str, Any] = {"op": "stop_recording"}
        if checkpoint_out_dir is not None:
            if not objects:
                raise ValueError(
                    "post-stop checkpoint packet needs at least one object"
                )
            request["checkpoint_out_dir"] = str(
                Path(checkpoint_out_dir).expanduser().resolve()
            )
            request["objects"] = [
                [str(name), str(prompt)]
                for name, prompt in objects
            ]
        return self._call(request)

    def close(self) -> None:
        """Release the sole physical-camera owner (idempotent)."""
        with self._lock:
            if self._closed:
                return
            try:
                _request(
                    self._worker,
                    "ZEDSTREAM",
                    {"op": "exit"},
                    time.time() + self.timeout_s,
                    "ZED stream",
                )
                self._worker["proc"].wait(timeout=5)
            except Exception:
                try:
                    self._worker["proc"].kill()
                    self._worker["proc"].wait(timeout=5)
                except Exception:
                    pass
            finally:
                self._closed = True
                try:
                    _ZED_STREAM_CLIENTS.remove(self)
                except ValueError:
                    pass

    stop = close


def stop_all_workers() -> None:
    """Stop every resident worker (registered with :mod:`atexit`)."""
    for client in list(_ZED_STREAM_CLIENTS):
        client.close()
    fp_stop()


atexit.register(stop_all_workers)
