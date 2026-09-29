"""Stage 1: the offline reconstruction pipeline (``oap-reconstruct``).

Role in the two-stage pipeline: this package turns a lab-host RGB-D capture
into a portable, metric, textured, physics-ready SCENE BUNDLE that Stage 2
(``oap-run``) loads through :mod:`oap.twin.manifest`. Every step is
idempotent and resumable (see :mod:`oap.reconstruct.bundle`); heavy
models never run in this process -- they execute in external conda envs via
:mod:`oap.reconstruct.external` and the self-contained scripts under
:mod:`oap.reconstruct.payloads`.

Step DAG (per object)::

    capture -> mesh -> scale -> collision -> pose -> tracking -> priors -> manifest
"""
from __future__ import annotations

from .bundle import STEP_ORDER, assemble_manifest, run_all, run_step
from .external import ExternalEnvError, ExternalEnvs, payload_path

__all__ = [
    "STEP_ORDER",
    "ExternalEnvError",
    "ExternalEnvs",
    "assemble_manifest",
    "payload_path",
    "run_all",
    "run_step",
]
