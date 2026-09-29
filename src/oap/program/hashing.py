"""Enforce the four-call-site hash rule: ONE program identity, provably shared.

Role in the two-stage pipeline: the planning stage's paper invariant is that
the SAME :class:`TaskProgram` object is evaluated at exactly four call
sites -- the trajectory cost, the physics gate, the termination gate, and the success
verification on re-observation. Every one of those evaluations MUST route
through :func:`log_call_site`, which hashes the program's CANONICAL JSON form
(:meth:`TaskProgram.to_json`) and appends a typed record to the episode
log. The episode-log schema test then asserts all four records exist with ONE
identical sha256 per episode -- no ad-hoc ``json.dumps`` hashing anywhere.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from .program import TaskProgram

logger = logging.getLogger("oap.program.hashing")


def program_sha256(program: TaskProgram) -> str:
    """Return the sha256 hex digest of the program's canonical JSON form.

    This is THE program identity. It is computed over
    ``program.to_json()`` (sorted keys, repr-stable floats), so two programs
    with equal content hash identically across processes and hosts.
    """
    return hashlib.sha256(program.to_json().encode("utf-8")).hexdigest()


class CallSite(str, Enum):
    """The four -- and only four -- places a program may be evaluated."""

    TRAJECTORY_COST = "trajectory_cost"
    PHYSICS_GATE = "physics_gate"
    TERMINATION_GATE = "termination_gate"
    SUCCESS_VERIFICATION = "success_verification"


#: Every episode log must contain at least one record per site, all with the
#: same program hash.
REQUIRED_CALL_SITES: tuple[CallSite, ...] = (
    CallSite.TRAJECTORY_COST,
    CallSite.PHYSICS_GATE,
    CallSite.TERMINATION_GATE,
    CallSite.SUCCESS_VERIFICATION,
)


@dataclass
class CallSiteRecord:
    """One logged program evaluation: where, which program, when, with what."""

    site: CallSite
    program_sha256: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record for the episode evidence log."""
        return {
            "site": self.site.value,
            "program_sha256": self.program_sha256,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "CallSiteRecord":
        """Deserialize current records and the v1 ``cem_cost`` spelling.

        Episode v2 renamed the planner call site to the algorithm-neutral
        ``trajectory_cost``.  Old evidence packets remain readable; only new
        packets use the canonical name.
        """
        raw_site = str(d["site"])
        if raw_site == "cem_cost":
            raw_site = CallSite.TRAJECTORY_COST.value
        return CallSiteRecord(
            site=CallSite(raw_site),
            program_sha256=str(d["program_sha256"]),
            timestamp=float(d.get("timestamp", 0.0)),
            payload=dict(d.get("payload", {})),
        )


@runtime_checkable
class CallSiteSink(Protocol):
    """Anything that can receive call-site records (the loop's episode log)."""

    def record_call_site(self, record: CallSiteRecord) -> None:
        ...


def log_call_site(episode: CallSiteSink | list[CallSiteRecord] | None,
                  site: CallSite | str,
                  program: TaskProgram,
                  payload: Mapping[str, Any] | None = None) -> CallSiteRecord:
    """Log ONE program evaluation at a call site (the single choke point).

    Every evaluation of the program at ``trajectory_cost`` / ``physics_gate`` /
    ``termination_gate`` / ``success_verification`` MUST route through this
    function so the episode log can prove the four-call-site identity.

    Args:
        episode: the episode log (anything with ``record_call_site``), a plain
            list to append to (unit tests), or None to only compute the record.
        site: which of the four call sites this evaluation happened at.
        program: the program being evaluated; hashed via its canonical JSON.
        payload: optional small JSON-compatible context (e.g. stage index,
            cost value, verdict) -- evidence, never a second source of truth.

    Returns:
        The record that was logged (also returned when ``episode`` is None).
    """
    record = CallSiteRecord(
        site=CallSite(site),
        program_sha256=program_sha256(program),
        timestamp=time.time(),
        payload=dict(payload or {}),
    )
    if episode is None:
        pass
    elif isinstance(episode, list):
        episode.append(record)
    elif isinstance(episode, CallSiteSink):
        episode.record_call_site(record)
    else:
        raise TypeError(
            f"episode must expose record_call_site(record), be a list, or None; "
            f"got {type(episode).__name__}")
    logger.debug("call_site=%s program=%s payload_keys=%s",
                 record.site.value, record.program_sha256[:12], sorted(record.payload))
    return record


def call_site_violations(records: Iterable[CallSiteRecord]) -> list[str]:
    """Check an episode's records against the four-call-site rule.

    Returns a typed error list ([] == compliant): one entry per missing call
    site and one entry if more than one distinct program hash appears. Used by
    the episode-log schema test and by post-hoc replay tooling.
    """
    recs = list(records)
    errs: list[str] = []
    seen_sites = {r.site for r in recs}
    for site in REQUIRED_CALL_SITES:
        if site not in seen_sites:
            errs.append(f"missing call site {site.value!r}")
    hashes = sorted({r.program_sha256 for r in recs})
    if len(hashes) > 1:
        errs.append("multiple distinct program hashes in one episode: "
                    + ", ".join(h[:12] for h in hashes))
    return errs
