"""Gauntlet-ready episode evidence log (schema ``oap_episode_v3``).

Role in the two-stage pipeline: every ``oap-run`` episode writes ONE
self-contained evidence packet sufficient to replay the judges offline:

  * the program JSON + its canonical sha256 (THE program identity);
  * a :class:`~oap.program.CallSiteRecord` for every program evaluation
    at the four call sites (``trajectory_cost``, ``physics_gate``,
    ``termination_gate``, ``success_verification``) -- the schema test asserts
    all four exist with ONE identical hash per episode;
  * per-chunk observations (poses + frame paths), tier-labelled verdicts,
    execution records, and per-chunk + stitched side-by-side videos;
  * free-form structured events (never parsed from log lines).

The log is checkpointed after every chunk (atomic write) so a crashed episode
still leaves a valid, replayable packet.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from oap.program import (
    CallSiteRecord,
    TaskProgram,
    EpisodeVerdict,
    call_site_violations,
    program_sha256,
)
from oap.utils.io import write_json_atomic

logger = logging.getLogger("oap.loop.episode")

__all__ = ["EPISODE_SCHEMA", "SUPPORTED_EPISODE_SCHEMAS", "EpisodeLog",
           "read_episode"]

EPISODE_SCHEMA = "oap_episode_v3"
SUPPORTED_EPISODE_SCHEMAS = frozenset({
    "oap_episode_v1",
    "oap_episode_v2",
    EPISODE_SCHEMA,
})


def _rel(path: Path | str | None, root: Path) -> str | None:
    """Store paths episode-dir-relative so the packet is portable."""
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p)


def read_episode(path: Path | str) -> dict[str, Any]:
    """Read v1, v2, or v3 evidence and normalize call-site records in memory.

    V2 changed the planner call-site spelling from the algorithm-specific
    ``cem_cost`` to ``trajectory_cost``.  This reader preserves the packet's
    original top-level schema while returning canonical call-site values, so
    historical v1 evidence remains auditable without letting new writers emit
    the retired name. V3 gives terminal-gate cycle evidence one unambiguous
    vocabulary; historical packets remain readable under their original
    schema, and new writers emit no legacy aliases.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = str(doc.get("schema", ""))
    if schema not in SUPPORTED_EPISODE_SCHEMAS:
        raise ValueError(f"unsupported episode schema {schema!r}")
    rows = doc.get("call_sites", [])
    if not isinstance(rows, list):
        raise ValueError("episode call_sites must be a list")
    doc["call_sites"] = [
        CallSiteRecord.from_dict(row).to_dict()
        for row in rows
    ]
    return doc


class EpisodeLog:
    """One episode's evidence packet: builder + atomic writer.

    Implements the :class:`oap.program.hashing.CallSiteSink` protocol
    (``record_call_site``), so it can be passed directly as the ``episode``
    argument of :func:`oap.program.log_call_site` -- the four-call-site
    hash rule persists through this object.
    """

    def __init__(self, out_dir: Path, *, instruction: str, bundle_manifest: Path | str,
                 config_snapshot: dict[str, Any] | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "episode.json"
        self.started_unix_s = time.time()
        self.episode_id = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{abs(hash(instruction)) % 10_000:04d}"
        self._doc: dict[str, Any] = {
            "schema": EPISODE_SCHEMA,
            "episode_id": self.episode_id,
            "instruction": instruction,
            "bundle_manifest": str(bundle_manifest),
            "config": dict(config_snapshot or {}),
            "program": None,
            "call_sites": [],
            "call_site_violations": None,
            "chunks": [],
            "verdicts": [],
            "events": [],
            "videos": {},
            "success": None,
            "outcome": None,
            "started_unix_s": self.started_unix_s,
            "finished_unix_s": None,
            "runtime_s": None,
        }
        self._call_site_records: list[CallSiteRecord] = []

    # ------------------------------------------------------------- program
    def set_program(self, program: TaskProgram) -> str:
        """Persist the program JSON + canonical sha256; returns the hash.

        Writes ``program.json`` (the canonical ``to_json`` form, byte-stable)
        next to the episode log; the sha256 recorded here is the ONE identity
        every call-site record must match.
        """
        sha = program_sha256(program)
        program_path = self.out_dir / "program.json"
        program_path.write_text(program.to_json(), encoding="utf-8")
        self._doc["program"] = {
            "path": _rel(program_path, self.out_dir),
            "sha256": sha,
            "provenance": program.provenance,
            "instruction": program.instruction,
            "n_stages": len(program.stages),
            "fully_geometric": program.is_fully_geometric(),
            "non_geometric_reasons": program.non_geometric_reasons(),
        }
        self.checkpoint()
        return sha

    # ---------------------------------------------------------- call sites
    def record_call_site(self, record: CallSiteRecord) -> None:
        """Receive one program-evaluation record (the CallSiteSink protocol)."""
        self._call_site_records.append(record)
        self._doc["call_sites"].append(record.to_dict())

    def record_planner_config(self, payload: dict[str, Any]) -> None:
        """Persist the derived MPC configuration actually used by this run.

        ``LoopConfig`` records user-facing inputs.  This method records the
        resulting planner semantics (horizon, knot interval, prefix and backend)
        after the MuJoCo model is loaded, so an evidence packet can be audited
        without reconstructing defaults from source code.
        """
        planner = dict(payload)
        config = self._doc.setdefault("config", {})
        previous = config.get("planner")
        if previous is not None and previous != planner:
            raise ValueError("planner config is immutable once recorded")
        config["planner"] = planner
        self.checkpoint()

    @property
    def call_site_records(self) -> list[CallSiteRecord]:
        """All program-evaluation records so far."""
        return list(self._call_site_records)

    # --------------------------------------------------------- observations
    def record_observation(self, chunk_idx: int, phase: str,
                           observation: dict[str, Any] | None,
                           frames: list[Path] | None = None) -> None:
        """Attach one observation (pose payload + frame paths) to a chunk.

        ``phase`` names when it was taken ('before' | 'after' | 'probe' |
        'relocalize:<name>' ...). A None observation (subject occluded/placed)
        is recorded explicitly -- absence of evidence is evidence too.
        """
        chunk = self._chunk(chunk_idx)
        chunk.setdefault("observations", []).append({
            "phase": phase,
            "observation": observation,
            "frames": [_rel(f, self.out_dir) for f in (frames or [])],
            "unix_s": time.time(),
        })

    # -------------------------------------------------------------- chunks
    def _chunk(self, chunk_idx: int) -> dict[str, Any]:
        """Get-or-create the record dict for a chunk index."""
        for c in self._doc["chunks"]:
            if c["chunk_index"] == int(chunk_idx):
                return c
        record: dict[str, Any] = {"chunk_index": int(chunk_idx)}
        self._doc["chunks"].append(record)
        return record

    def record_chunk(self, chunk_idx: int, payload: dict[str, Any]) -> None:
        """Merge a payload (selection, gates, execution record) into a chunk."""
        self._chunk(chunk_idx).update(payload)
        self.checkpoint()

    def record_chunk_video(self, chunk_idx: int, name: str, path: Path | None) -> None:
        """Attach one named video (real / sim / side-by-side) to a chunk."""
        self._chunk(chunk_idx).setdefault("videos", {})[name] = _rel(path, self.out_dir)

    # ------------------------------------------------------------ verdicts
    def record_verdict(self, verdict: EpisodeVerdict | dict[str, Any], *,
                       chunk_idx: int | None = None,
                       context: dict[str, Any] | None = None) -> None:
        """Record one tier-labelled verdict (episode-level and per-chunk)."""
        v = verdict.to_dict() if isinstance(verdict, EpisodeVerdict) else dict(verdict)
        entry = {"chunk_index": chunk_idx, "verdict": v,
                 "context": dict(context or {}), "unix_s": time.time()}
        self._doc["verdicts"].append(entry)
        if chunk_idx is not None:
            self._chunk(chunk_idx).setdefault("verdicts", []).append(entry)

    # -------------------------------------------------------------- events
    def record_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        """Append one structured event (never parse evidence from log lines)."""
        self._doc["events"].append({"kind": kind, "payload": dict(payload or {}),
                                    "unix_s": time.time()})

    # -------------------------------------------------------------- videos
    def add_video(self, name: str, path: Path | None) -> None:
        """Register a session-level video (full_real / full_sim / full_real_sim)."""
        if path is not None:
            self._doc["videos"][name] = _rel(path, self.out_dir)

    # ------------------------------------------------------------- persist
    def checkpoint(self) -> None:
        """Atomically write the current packet (crash-safe evidence)."""
        write_json_atomic(self.path, self._doc)

    def finalize(self, *, success: bool, outcome: str | None = None,
                 extra: dict[str, Any] | None = None) -> Path:
        """Close the packet: verify the four-call-site rule and write it out.

        The four-call-site violations list is COMPUTED AND STORED (an empty
        list is the paper invariant; the schema test asserts it) -- a violation
        is loudly logged but never silently dropped from the evidence.
        """
        violations = call_site_violations(self._call_site_records)
        if violations:
            logger.error("[episode] FOUR-CALL-SITE RULE VIOLATED: %s", violations)
        self._doc["call_site_violations"] = violations
        self._doc["success"] = bool(success)
        self._doc["outcome"] = outcome
        self._doc["finished_unix_s"] = time.time()
        self._doc["runtime_s"] = self._doc["finished_unix_s"] - self.started_unix_s
        if extra:
            self._doc.update(extra)
        self.checkpoint()
        logger.info("[episode] evidence packet written: %s (success=%s, %d call-site "
                    "records, %d chunks)", self.path, success,
                    len(self._call_site_records), len(self._doc["chunks"]))
        return self.path
