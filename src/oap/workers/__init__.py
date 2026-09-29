"""Persistent external-env worker processes and their in-package client.

Role in the two-stage pipeline: the online loop (stage 2) needs two heavy
models -- FoundationPose (6-DoF register+track) and Qwen3-VL (live YES/NO
judging) -- that live in EXTERNAL conda environments and must stay resident
between chunks (loading them per call costs 10-60 s).

Layout rule: every ``*_worker.py`` file in this package is PAYLOAD-STYLE -- it
imports NOTHING from :mod:`oap` (stdlib + the external env's own stack
only), because it is executed by a foreign interpreter that does not have this
package installed. Only :mod:`oap.workers.client` runs inside the
``oap`` env; it spawns the workers by file path (resolved with
:mod:`importlib.resources`) and speaks their line-based JSONL protocols.
"""
from __future__ import annotations

__all__ = ["client"]
