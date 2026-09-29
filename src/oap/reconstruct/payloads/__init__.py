"""Payload scripts executed by EXTERNAL environment interpreters.

Role in the two-stage pipeline: each module in this package is a
self-contained command-line program that Stage 1 launches through
:meth:`oap.reconstruct.external.ExternalEnvs.run_payload` using the
interpreter of a dedicated conda env (observer / sam3d / foundationpose /
qwen). THE PAYLOAD RULE: files here must not import ``oap`` (or anything
else from this repository) -- the external envs do not have it installed --
and all heavy imports (pyzed, SAM3, SAM3D, torch, transformers, Any6D) must
happen lazily inside functions so ``--help`` works everywhere.
"""
from __future__ import annotations
