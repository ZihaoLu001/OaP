"""Opt-in evidence for the exact GPU winner prefix used by simulation.

The production planner normally transfers only the selected prefix endpoint.
Paper experiments may opt in to a larger evidence path that retains every
post-step qpos from that *same* batched MJWarp rollout.  This module owns the
host-side mapping and immutable packet format; it never integrates physics or
interpolates between states.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

GPU_STEP_TRACE_ENV = "OAP_RECORD_GPU_STEP_TRACE"
GPU_STEP_TRACE_SCHEMA = "oap_gpu_winner_step_trace_v1"


def gpu_step_trace_enabled(environ: dict[str, str] | None = None) -> bool:
    """Return the explicit experiment switch, rejecting ambiguous values."""
    source = os.environ if environ is None else environ
    raw = str(source.get(GPU_STEP_TRACE_ENV, "")).strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(
        f"{GPU_STEP_TRACE_ENV} must be an explicit Boolean, got {raw!r}"
    )


def map_rollout_qpos_trace_to_world(
    rollout_qpos: np.ndarray,
    *,
    start_qpos: np.ndarray,
    qpos_src: tuple[int, ...],
    qpos_dst: tuple[int, ...],
    rollout_nq: int,
) -> np.ndarray:
    """Map a selected rollout's post-step qpos tape into the live layout.

    Unshared host-only coordinates retain the prefix start value on every row,
    matching :meth:`BatchedRollout.selected_prefix_world_state`.  This is an
    address-wise copy only; no MuJoCo call occurs here.
    """
    trace = np.asarray(rollout_qpos)
    initial = np.asarray(start_qpos)
    if trace.ndim != 2 or trace.shape[1] != int(rollout_nq):
        raise ValueError(
            "selected GPU qpos trace has shape "
            f"{trace.shape}, expected (steps, {int(rollout_nq)})"
        )
    if trace.shape[0] < 1:
        raise ValueError("selected GPU qpos trace must contain a post-step row")
    if len(qpos_src) != len(qpos_dst):
        raise ValueError("world/model qpos address maps have different lengths")
    world = np.broadcast_to(initial, (trace.shape[0], initial.size)).copy()
    if qpos_src:
        world[:, np.asarray(qpos_src, dtype=int)] = trace[
            :, np.asarray(qpos_dst, dtype=int)
        ]
    if not np.all(np.isfinite(world)):
        raise ValueError("selected GPU qpos trace contains non-finite state")
    return world


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_gpu_step_trace(
    *,
    episode_dir: Path,
    chunk_index: int,
    qpos: np.ndarray,
    start_qpos: np.ndarray,
    step_dt_s: float,
    selected_candidate_index: int | None,
    selected_round: int | None,
) -> dict[str, Any]:
    """Atomically write one immutable compressed per-chunk qpos tape."""
    root = Path(episode_dir).resolve()
    trace = np.asarray(qpos)
    initial = np.asarray(start_qpos)
    dt = float(step_dt_s)
    if trace.ndim != 2 or trace.shape[0] < 1:
        raise ValueError("qpos must have shape (positive_steps, nq)")
    if initial.shape != (trace.shape[1],):
        raise ValueError(
            f"start_qpos shape {initial.shape} does not match trace nq "
            f"{trace.shape[1]}"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("step_dt_s must be finite and positive")
    if not np.all(np.isfinite(trace)) or not np.all(np.isfinite(initial)):
        raise ValueError("GPU step trace contains non-finite qpos")

    trace_dir = root / "gpu_step_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    target = trace_dir / f"chunk_{int(chunk_index):04d}.npz"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite GPU step trace {target}")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary GPU step trace already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray(GPU_STEP_TRACE_SCHEMA),
                qpos=trace,
                start_qpos=initial,
                step_dt_s=np.asarray(dt, dtype=np.float64),
                chunk_index=np.asarray(int(chunk_index), dtype=np.int64),
                post_step_indices=np.arange(
                    1, trace.shape[0] + 1, dtype=np.int64
                ),
                selected_candidate_index=np.asarray(
                    -1
                    if selected_candidate_index is None
                    else int(selected_candidate_index),
                    dtype=np.int64,
                ),
                selected_round=np.asarray(
                    -1 if selected_round is None else int(selected_round),
                    dtype=np.int64,
                ),
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest = sha256_file(target)
    return {
        "schema": GPU_STEP_TRACE_SCHEMA,
        "path": str(target.relative_to(root)),
        "sha256": digest,
        "source": "same_mjwarp_selected_candidate_scan",
        "state_semantics": "post_step_qpos_no_reintegration_no_interpolation",
        "step_count": int(trace.shape[0]),
        "step_dt_s": dt,
        "qpos_shape": [int(v) for v in trace.shape],
        "qpos_dtype": str(trace.dtype),
        "includes_start_qpos": True,
        "selected_candidate_index": (
            None
            if selected_candidate_index is None
            else int(selected_candidate_index)
        ),
        "selected_round": (
            None if selected_round is None else int(selected_round)
        ),
    }


def read_gpu_step_trace(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and validate a trace packet without enabling pickle loading."""
    packet = Path(path)
    actual_sha256 = sha256_file(packet)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"GPU step trace SHA-256 mismatch for {packet}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    with np.load(packet, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        if schema != GPU_STEP_TRACE_SCHEMA:
            raise ValueError(f"unsupported GPU step trace schema {schema!r}")
        qpos = np.asarray(data["qpos"])
        start_qpos = np.asarray(data["start_qpos"])
        step_dt_s = float(np.asarray(data["step_dt_s"]).item())
        indices = np.asarray(data["post_step_indices"], dtype=np.int64)
        chunk_index = int(np.asarray(data["chunk_index"]).item())
        selected_candidate_index = int(
            np.asarray(data["selected_candidate_index"]).item()
        )
        selected_round = int(np.asarray(data["selected_round"]).item())
    if qpos.ndim != 2 or qpos.shape[0] < 1:
        raise ValueError("GPU step trace qpos must have shape (steps, nq)")
    if start_qpos.shape != (qpos.shape[1],):
        raise ValueError("GPU step trace start_qpos shape does not match qpos")
    expected_indices = np.arange(1, qpos.shape[0] + 1, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("GPU step trace post-step indices are not contiguous")
    if not np.isfinite(step_dt_s) or step_dt_s <= 0.0:
        raise ValueError("GPU step trace dt is not finite and positive")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(start_qpos)):
        raise ValueError("GPU step trace contains non-finite qpos")
    return {
        "schema": schema,
        "sha256": actual_sha256,
        "qpos": qpos,
        "start_qpos": start_qpos,
        "step_dt_s": step_dt_s,
        "post_step_indices": indices,
        "chunk_index": chunk_index,
        "selected_candidate_index": (
            None if selected_candidate_index < 0 else selected_candidate_index
        ),
        "selected_round": None if selected_round < 0 else selected_round,
    }
