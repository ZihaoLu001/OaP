"""Define the task-agnostic anchor set a TaskProgram references.

Role in the two-stage pipeline: the reconstruction stage produces the scene a
grounding step turns into anchors; the planning stage (``oap-run``)
synthesizes and evaluates programs over those anchors.

An anchor is whatever the open-vocabulary perception stack (segmentation /
correspondence / pose tracking / mesh reconstruction) can localize and
re-localize: an object/part 6-DoF frame, a keypoint, a principal axis, or a
free-space region. Anchors are named by string id; a program references ids
only, so a new task is a new program over the SAME anchor substrate -- never
new code.

Crucially, anchors carry NO task semantics. The same anchor set serves any
instruction. The tracking-reliability fields (confidence, visibility,
age_since_seen, last_pose) feed the CANNOT_VERIFY router in
:mod:`oap.program.verdict`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import as_vec3, unit


@dataclass
class Anchor:
    """A grounded scene element.

    point     : 3D position (always present; the anchor's reference point).
    axis      : optional principal/normal direction (unit), e.g. a part's long
                axis, a drawer slide axis, a surface normal.
    region_half: optional axis-aligned half-extents (3,) -> the anchor is a box
                region centered at `point` (e.g. a tray interior, a color bin).
    kind      : provenance tag ('object' | 'part' | 'keypoint' | 'axis' |
                'region' | 'subject' | 'world'); informational only.
    attached_to: if set, this anchor is rigidly attached to that subject id and
                moves with it (e.g. 'spout' attached to 'held_can'); used by the
                outcome model to transform it under a candidate action.
    dynamic   : this anchor rides a NON-SUBJECT body the physics can move (a
                pushed box). Set only by the movable grounder.

    Tracking state defaults to fully observed in the current frame:
    confidence  : tracker score in [0,1] (pose-tracker score head / CoTracker3).
    visibility  : analytic visible-fraction in [0,1] (depth/mask), SEPARATE from
                  confidence.
    age_since_seen: frames since this anchor was last directly observed (0 == fresh).
    last_pose   : cached point from the last confident observation (occlusion fallback).
    """

    point: np.ndarray
    axis: np.ndarray | None = None
    region_half: np.ndarray | None = None
    kind: str = "keypoint"
    attached_to: str | None = None
    confidence: float = 1.0
    visibility: float = 1.0
    age_since_seen: int = 0
    last_pose: np.ndarray | None = None
    #: True when this anchor rides a body the PHYSICS can move that is not the
    #: subject -- a box a tool pushes. The inert-success gate asks "can the
    #: robot influence this?"; before free bodies existed the answer was
    #: exactly "is it the subject", and hardcoding that would now refuse every
    #: legitimate push. Only the movable grounder sets it, so a static support
    #: can never acquire it by accident.
    dynamic: bool = False

    def __post_init__(self) -> None:
        self.point = as_vec3(self.point)
        if self.axis is not None:
            self.axis = unit(self.axis)
        if self.region_half is not None:
            self.region_half = np.abs(as_vec3(self.region_half))
        if self.last_pose is not None:
            self.last_pose = as_vec3(self.last_pose)

    def is_finite(self) -> bool:
        """Return True when the anchor is finitely grounded (point and axis)."""
        ok = bool(np.all(np.isfinite(self.point)))
        if self.axis is not None:
            ok = ok and bool(np.all(np.isfinite(self.axis))) and float(np.linalg.norm(self.axis)) > 1e-9
        return ok

    def reliable(self, tau: float = 0.0) -> bool:
        """Grounded AND trustworthy: finite and visibility*confidence >= tau
        (CoTracker3's deployment recipe). tau=0 reproduces the legacy 'finite only'."""
        return self.is_finite() and (self.visibility * self.confidence) >= tau


class AnchorSet:
    """A dict of named anchors with helpers for grounding + subject updates."""

    def __init__(self, anchors: dict[str, Anchor] | None = None):
        self._a: dict[str, Anchor] = dict(anchors or {})

    def add(self, name: str, anchor: Anchor) -> "AnchorSet":
        """Insert or replace an anchor; returns self for chaining."""
        self._a[str(name)] = anchor
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._a

    def __getitem__(self, name: str) -> Anchor:
        return self._a[name]

    def get(self, name: str) -> Anchor | None:
        """Return the anchor or None when absent."""
        return self._a.get(name)

    def names(self) -> list[str]:
        """Return the anchor ids in insertion order."""
        return list(self._a.keys())

    def point(self, name: str) -> np.ndarray:
        """Return the named anchor's point."""
        return self._a[name].point

    def axis(self, name: str) -> np.ndarray:
        """Return the named anchor's axis, raising if it has none."""
        a = self._a[name].axis
        if a is None:
            raise ValueError(f"anchor {name!r} has no axis")
        return a

    def copy(self) -> "AnchorSet":
        """Deep-copy the set (arrays copied, ALL fields carried).

        Enumerated from ``dataclasses.fields`` rather than written out by hand.
        The hand-written version listed nine of the ten fields and dropped
        ``dynamic`` -- under a comment claiming it was written that way so a new
        field could not be silently dropped. It cost a night: ``dynamic`` marks
        the bodies the robot can move, ``predicted_end_anchors`` copies the set
        before scoring, and the tool-mediation rule keys on it, so that rule was
        inert in every production episode while its unit tests -- which build
        an AnchorSet by hand and never copy it -- stayed green.
        """
        from dataclasses import fields as _fields

        out = {}
        for k, v in self._a.items():
            kw = {}
            for f in _fields(Anchor):
                val = getattr(v, f.name)
                kw[f.name] = val.copy() if isinstance(val, np.ndarray) else val
            out[k] = Anchor(**kw)
        return AnchorSet(out)

    def missing(self, required: Iterable[str]) -> list[str]:
        """Anchor ids referenced but absent OR not finitely grounded."""
        bad = []
        for name in required:
            a = self._a.get(name)
            if a is None or not a.is_finite():
                bad.append(name)
        return bad

    def unreliable(self, required: Iterable[str], tau: float = 0.0) -> list[str]:
        """Anchor ids absent OR not RELIABLY tracked (finite & vis*conf >= tau).
        tau=0 == missing(); tau>0 catches occluded/low-confidence tracks that
        should route to CANNOT_VERIFY rather than feed the hard gate a stale point."""
        bad = []
        for name in required:
            a = self._a.get(name)
            if a is None or not a.reliable(tau):
                bad.append(name)
        return bad
