"""Objectives as Policies: staged objective programs for physics-based MPC.

The public program types are re-exported here; residual definitions, cost
evaluation, synthesis, and stage verification live in :mod:`oap.program`.
"""
from __future__ import annotations

from .program import (
    Anchor,
    AnchorSet,
    CallSite,
    CallSiteRecord,
    ConstraintProgram,
    EpisodeVerdict,
    Stage,
    TaskProgram,
    VerificationOutcome,
    log_call_site,
    program_sha256,
    synthesize,
    verify,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Anchor",
    "AnchorSet",
    "CallSite",
    "CallSiteRecord",
    "ConstraintProgram",
    "EpisodeVerdict",
    "Stage",
    "TaskProgram",
    "VerificationOutcome",
    "log_call_site",
    "program_sha256",
    "synthesize",
    "verify",
]
