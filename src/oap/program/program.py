"""Define the TaskProgram the VLM synthesizes from an arbitrary instruction.

The VLM turns task text into one of these programs. The identical program is
evaluated by GPU sampling and measured termination/success verification, then
hashed via :mod:`oap.program.hashing` so the episode log proves it.

A program is an ordered list of STAGES. Each stage uses the standard
sampling-MPC cost split:
  * running: soft task costs accumulated over predicted rollout states;
  * terminal: task costs at the predicted endpoint and measured conditions for
    stage success/advance.
Generic physical safety is enforced by the task-independent feasibility gate,
not encoded as a second task-specific constraint language.
The program -- not a family router -- carries ALL task semantics. It is plain
data (JSON), so a new unknown task is a new program over the SAME primitive
library, with ZERO new code.

Canonical serialization: :meth:`TaskProgram.to_json` emits a byte-stable
JSON form (sorted keys, normalized floats, no non-finite values) so that
``sha256(to_json())`` is THE program identity logged at every call site.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .predicates import Predicate


@dataclass
class Stage:
    """One sampling-MPC stage: running and terminal program predicates."""

    name: str = ""
    running: list[Predicate] = field(default_factory=list)
    terminal: list[Predicate] = field(default_factory=list)
    # V11 (env-gated consumer): objects this stage INTENDS to move; every
    # other movable is disturbance-penalized when OAP_DISTURB_PENALTY_W
    # is set. Empty = field omitted = pre-v11 serialized form unchanged
    # (the table_contact_force_weight omit-when-empty idiom).
    movable_objects: tuple = ()

    def referenced_anchors(self) -> list[str]:
        """Return the de-duplicated anchor ids this stage references."""
        names: list[str] = []
        for p in list(self.running) + list(self.terminal):
            names.extend(p.referenced_anchors())
        # de-dup preserving order
        seen, out = set(), []
        for n in names:
            if n and n not in seen:
                seen.add(n); out.append(n)
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialize the stage to plain JSON-compatible data."""
        out = {
            "name": self.name,
            "running": [p.to_dict() for p in self.running],
            "terminal": [p.to_dict() for p in self.terminal],
        }
        if self.movable_objects:
            out["movable_objects"] = list(self.movable_objects)
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Stage":
        """Deserialize a stage from its JSON-object form."""
        allowed = {"name", "running", "terminal", "movable_objects"}
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"unknown Stage fields: {unknown}")
        running_raw = list(d.get("running", []))
        terminal_raw = list(d.get("terminal", []))
        # Running costs use unthresholded residuals (Eq. 3). Do not silently
        # rewrite the authored tolerances, which also define fixed scales.
        return Stage(
            name=d.get("name", ""),
            running=[Predicate.from_dict(p) for p in running_raw],
            terminal=[Predicate.from_dict(p) for p in terminal_raw],
            movable_objects=tuple(
                str(x) for x in d.get("movable_objects", ())
            ),
        )



def _relation_key(p: dict[str, Any]) -> tuple:
    """Identity of a relation for band matching: its type and its anchors."""
    return (p.get("type"),
            tuple(str(p.get(k)) for k in
                  ("a", "b", "subject", "body", "object_anchor", "dir_anchor",
                   "support", "axis_anchor") if p.get(k) is not None))


def _clamp_running_bands(running: list, terminal: list) -> list:
    """Never let a running band be wider than the acceptance band it shapes.

    A running residual is a hinge: it is exactly zero inside its tolerance. If
    that tolerance exceeds the acceptance tolerance of the SAME relation in the
    same stage, the residual is identically zero over the whole interval
    between the two bands, so nothing pulls the state through the last stretch
    before the gate and the stage parks just outside its own acceptance.
    Measured directly (abl17_push_gen01): running 0.040 against acceptance
    0.030, blocking distance flat at 31.1 mm for 240+ cycles, 0.8-1.1 mm short.
    The reference program uses equal bands and latches at cycle 22.

    The clamp is a MINIMUM, not a rescale: a program whose running band is
    already at or inside the acceptance band is left byte-identical, so only
    the defective pairing changes. Terminal predicates, weights, residual
    shapes and success bars are untouched.
    """
    if not terminal:
        return running
    accept: dict[tuple, float] = {}
    for t in terminal:
        if not isinstance(t, dict):
            continue
        for field in ("tol", "tol_deg"):
            if t.get(field) is None:
                continue
            try:
                accept[(field, _relation_key(t))] = float(t[field])
            except (TypeError, ValueError):
                pass
    if not accept:
        return running
    out = []
    for r in running:
        if not isinstance(r, dict):
            out.append(r)
            continue
        r2 = r
        for field in ("tol", "tol_deg"):
            if r.get(field) is None:
                continue
            cap = accept.get((field, _relation_key(r)))
            if cap is None:
                continue
            try:
                cur = float(r[field])
            except (TypeError, ValueError):
                continue
            if cur > cap:
                if r2 is r:
                    r2 = dict(r)
                r2[field] = cap
        out.append(r2)
    return out

@dataclass
class TaskProgram:
    """The full synthesized program: instruction, stages, provenance, notes."""

    instruction: str = ""
    stages: list[Stage] = field(default_factory=list)
    provenance: str = "unspecified"      # which VLM/backend synthesized it
    notes: str = ""
    # Optional full-physics running-cost coefficient.  ``None`` preserves the
    # optimizer profile's historical default.  An explicit value lets a
    # reviewed task match a published objective without weakening the generic
    # penetration/contact validity gates.  In particular, the reference arm
    # push objective uses zero table-force cost while its pick objective does
    # include that term.
    table_contact_force_weight: float | None = None

    def __post_init__(self) -> None:
        value = self.table_contact_force_weight
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "table_contact_force_weight must be a finite non-negative "
                "number or null"
            )
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "table_contact_force_weight must be a finite non-negative "
                "number or null"
            )
        self.table_contact_force_weight = value

    def referenced_anchors(self) -> list[str]:
        """Return the de-duplicated anchor ids the whole program references."""
        seen, out = set(), []
        for s in self.stages:
            for n in s.referenced_anchors():
                if n not in seen:
                    seen.add(n); out.append(n)
        return out

    def is_fully_geometric(self) -> bool:
        """True iff every TERMINAL primitive is geometric (=> Tier-1 verifiable)."""
        return all(p.is_geometric for s in self.stages for p in s.terminal)

    def non_geometric_reasons(self) -> list[str]:
        """Return the reasons of every NonGeometric terminal sentinel."""
        from .predicates import NonGeometric, TemporalHold

        reasons: list[str] = []
        for stage in self.stages:
            for predicate in stage.terminal:
                while isinstance(predicate, TemporalHold):
                    predicate = predicate.inner
                if isinstance(predicate, NonGeometric):
                    reasons.append(predicate.reason)
        return reasons

    def has_verifiable_success(self) -> bool:
        """Whether every Stage terminal can be measured.

        Every Stage terminal must be non-empty, and every predicate (after
        unwrapping :class:`TemporalHold`) must be geometric. An empty or mixed
        geometric/non-geometric terminal is not partially verifiable: silently
        dropping one conjunct would let another predicate redefine success.
        The runner therefore refuses such a program before motion.
        """
        if not self.stages:
            return False
        from .predicates import TemporalHold

        for stage in self.stages:
            if not stage.terminal:
                return False
            for predicate in stage.terminal:
                while isinstance(predicate, TemporalHold):
                    predicate = predicate.inner
                if not getattr(predicate, "is_geometric", False):
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the program to plain JSON-compatible data."""
        out = {"instruction": self.instruction, "provenance": self.provenance,
               "notes": self.notes, "stages": [s.to_dict() for s in self.stages]}
        if self.table_contact_force_weight is not None:
            out["table_contact_force_weight"] = float(
                self.table_contact_force_weight
            )
        return out

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TaskProgram":
        """Deserialize a program from its JSON-object form."""
        if not isinstance(d, dict):
            raise ValueError("program JSON must be an object")
        allowed = {
            "instruction",
            "provenance",
            "notes",
            "stages",
            "table_contact_force_weight",
        }
        unknown = sorted(set(d) - allowed)
        if unknown:
            raise ValueError(f"unknown TaskProgram fields: {unknown}")
        return TaskProgram(
            instruction=d.get("instruction", ""),
            provenance=d.get("provenance", "unspecified"),
            notes=d.get("notes", ""),
            stages=[Stage.from_dict(s) for s in d.get("stages", [])],
            table_contact_force_weight=d.get("table_contact_force_weight"),
        )

    # ---- canonical serialization (the sha256 hashing foundation) -------------
    def to_json(self) -> str:
        """Serialize to CANONICAL JSON: sorted keys, compact separators, floats
        normalized to repr-stable Python floats (-0.0 -> 0.0, non-finite rejected).

        Two programs with equal content produce byte-identical strings, so
        ``sha256(to_json().encode())`` is a stable identity across processes.
        Every one of the four call sites hashes THIS form -- never an ad-hoc
        ``json.dumps`` of ``to_dict()``.
        """
        return json.dumps(_canonicalize(self.to_dict()), sort_keys=True,
                          separators=(",", ":"), allow_nan=False, ensure_ascii=True)

    @staticmethod
    def from_json(text: str) -> "TaskProgram":
        """Deserialize a program from a JSON string (canonical or pretty)."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("program JSON must be an object")
        return TaskProgram.from_dict(data)


# Compatibility import for stored programs and downstream code. New APIs and
# documentation use TaskProgram; canonical JSON bytes are unchanged.
ConstraintProgram = TaskProgram


def _canonicalize(value: Any) -> Any:
    """Recursively normalize to pure-Python JSON types with stable floats.

    * dict keys become str (json sorts them at dump time);
    * tuples/lists/ndarrays become lists;
    * bools stay bool; ints (incl. numpy ints via ``int()``) stay int;
    * floats (incl. numpy floats) become plain Python floats with -0.0
      normalized to 0.0; NaN/inf raise (a hashable identity must be finite).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        f = float(value)
        if not math.isfinite(f):
            raise ValueError(f"non-finite float {value!r} in canonical program serialization")
        return 0.0 if f == 0.0 else f
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    # numpy scalars/arrays without importing numpy here: use duck typing.
    item = getattr(value, "item", None)
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and getattr(value, "ndim", None):
        return _canonicalize(tolist())
    if callable(item):
        return _canonicalize(item())
    raise TypeError(f"unserializable value of type {type(value).__name__} in program")
