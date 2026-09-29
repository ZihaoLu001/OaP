"""Online closed loop (Stage 2): observe -> plan -> execute -> measure -> verify.

Role in the two-stage pipeline: this package is the lab-host runtime behind
``oap-run``. It consumes a scene bundle from the offline reconstruction
stage, synthesizes ONE TaskProgram from task text, and drives the
per-prefix closed loop with the four-call-site program identity, single hard
program verifier, explicit CANNOT_VERIFY router, and dry-run-by-default safety
policy. There is no second VLM success judge.
"""
from __future__ import annotations

from .runner import LoopConfig, measured_program_success, run_episode

__all__ = ["LoopConfig", "measured_program_success", "run_episode"]
