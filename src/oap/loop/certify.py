"""CERTIFY layer: re-evaluate the program on re-observed geometry.

Role in the two-stage pipeline: this is the success-verification call site of
the planning stage (site ``success_verification``) -- the SAME synthesized
program is re-evaluated on the RE-OBSERVED real anchors via
:func:`oap.program.verdict.verify`, and every evaluation routes through
:func:`log_call_site`.

The geometric gate is the only path to ``VERIFIED_SUCCESS``. There is no
task-specific lift gate or secondary VLM judge in this layer.
"""
from __future__ import annotations

import logging
from typing import Any

from oap.program import (
    AnchorSet,
    CallSite,
    TaskProgram,
    EpisodeVerdict,
    StageEvaluation,
    log_call_site,
    verify,
)

logger = logging.getLogger("oap.loop.certify")

__all__ = ["certify"]


# --------------------------------------------------------------------------
# The certification entry point (call site: success_verification)
# --------------------------------------------------------------------------
def certify(program: TaskProgram, real_anchors: AnchorSet, *,
            episode: Any = None,
            context: dict[str, Any] | None = None,
            execution_contact_valid: bool | None = None,
            execution_contact: dict[str, Any] | None = None,
            stage_evidence: StageEvaluation | None = None,
            table_top_z: float | None = None,
            displacement_tolerance_m: float | None = None,
            tau_reliable: float = 0.0) -> EpisodeVerdict:
    """Re-evaluate the SAME program on re-observed anchors (call site 4).

    Runs the single-tier router (:func:`oap.program.verify`) on the
    freshly grounded real scene and logs the evaluation through
    :func:`log_call_site` at ``success_verification`` with the evidence.

    Args:
        program: The episode's ONE program object.
        real_anchors: The re-observed grounded scene.
        episode: The episode log (``CallSiteSink``) or None.
        context: Small JSON-compatible evidence (cycle index, purpose, ...).
        execution_contact_valid: Optional tool-mediation evidence for the
            commands that actually ran.
        execution_contact: JSON-compatible causal evidence for the audit log.
            It does not replace or redefine the measured terminal predicates.
        stage_evidence: The composed per-stage evaluation produced by the MPC
            gate. Explicit endpoint evidence is reused here; legacy rest
            metadata remains diagnostic and cannot tighten the program.

    Returns:
        The final verdict from the same evaluator used by the stage gate.
    """
    hard = verify(
        program,
        real_anchors,
        tau_reliable=tau_reliable,
        table_top_z=table_top_z,
        tool_mediation_satisfied=execution_contact_valid,
        stage_evidence=stage_evidence,
        displacement_tolerance_m=displacement_tolerance_m,
    )
    final = hard
    payload = {
        **(context or {}),
        "hard": hard.to_dict(),
        "final": final.to_dict(),
        "execution_contact_valid": execution_contact_valid,
        "execution_contact": dict(execution_contact or {}),
    }
    log_call_site(episode, CallSite.SUCCESS_VERIFICATION, program, payload=payload)
    logger.info("[verdict] %s (%s: %s)", final.outcome.value, final.provenance,
                final.reason)
    return final


# --------------------------------------------------------------------------
