"""Bind a synthesized program to scene objects without reading task text.

The binding seam is deliberately structural. ``ObjectHeld.object_anchor`` is
the explicit controlled-object declaration. Natural-language keywords,
manifest order, and single-movable fallbacks are never consulted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .predicates import ObjectHeld, TemporalHold


class ProgramBindingError(ValueError):
    """The program does not uniquely identify the controlled/goal bodies."""


@dataclass(frozen=True)
class ProgramBindings:
    """Scene objects bound to the two roles the current twin needs."""

    subject: Any
    reference: Any | None


def _refs(predicates: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for predicate in predicates:
        out.extend(str(ref) for ref in predicate.referenced_anchors() if ref)
    return out


def _object_for_anchor(ref: str, objects: Iterable[Any]) -> Any | None:
    """Return the manifest object whose anchor vocabulary contains ``ref``.

    Object names may themselves contain underscores, so match the longest
    exact ``name`` / ``name_*`` prefix rather than splitting on ``_``.
    """
    matches = [obj for obj in objects
               if ref == str(obj.name) or ref.startswith(f"{obj.name}_")]
    if not matches:
        return None
    matches.sort(key=lambda obj: len(str(obj.name)), reverse=True)
    if len(matches) > 1 and len(str(matches[0].name)) == len(str(matches[1].name)):
        raise ProgramBindingError(f"anchor {ref!r} matches multiple scene objects")
    return matches[0]


def _unique_objects(refs: Iterable[str], objects: list[Any]) -> list[Any]:
    by_name: dict[str, Any] = {}
    for ref in refs:
        obj = _object_for_anchor(str(ref), objects)
        if obj is not None:
            by_name[str(obj.name)] = obj
    return list(by_name.values())


def _object_held_refs(program: Any) -> list[str]:
    """Validate and return every explicit held-object anchor in the program."""
    refs: list[str] = []
    for index, stage in enumerate(getattr(program, "stages", ())):
        running: list[ObjectHeld] = []
        terminal: list[ObjectHeld] = []
        for where, predicates, out in (
            ("running", stage.running, running),
            ("terminal", stage.terminal, terminal),
        ):
            for predicate in predicates:
                if isinstance(predicate, TemporalHold):
                    inner = predicate
                    while isinstance(inner, TemporalHold):
                        inner = inner.inner
                    if isinstance(inner, ObjectHeld):
                        raise ProgramBindingError(
                            f"stage[{index}].{where} cannot wrap ObjectHeld "
                            "in TemporalHold"
                        )
                if isinstance(predicate, ObjectHeld):
                    out.append(predicate)
        if len(running) > 1 or len(terminal) > 1:
            raise ProgramBindingError(
                f"stage[{index}] may contain at most one ObjectHeld in each "
                "of running and terminal"
            )
        refs.extend(predicate.object_anchor for predicate in running + terminal)
    if refs and len(set(refs)) != 1:
        raise ProgramBindingError(
            "ObjectHeld predicates name multiple controlled anchors: "
            + ", ".join(sorted(set(refs)))
        )
    return refs


def bind_program_to_scene(program: Any, objects: list[Any]) -> ProgramBindings:
    """Resolve the controlled subject and singular goal/reference object.

    ``ObjectHeld.object_anchor`` binds the controlled body and final-stage
    terminal references bind the external goal body. Every held anchor must
    name a manifest object explicitly; no canonical alias or single-movable
    fallback may silently substitute a different body.
    """
    if not objects:
        raise ProgramBindingError("scene contains no objects")
    movable = [obj for obj in objects if bool(getattr(obj, "movable", False))]
    if not movable:
        raise ProgramBindingError("scene contains no movable object")

    held_refs = _object_held_refs(program)
    if held_refs:
        unresolved = [
            ref for ref in held_refs if _object_for_anchor(ref, objects) is None
        ]
        if unresolved:
            raise ProgramBindingError(
                "ObjectHeld anchors do not explicitly resolve to a manifest "
                "object: " + ", ".join(sorted(set(unresolved)))
            )
        named_subjects = _unique_objects(held_refs, objects)
        if len(named_subjects) > 1:
            raise ProgramBindingError(
                "ObjectHeld predicates name multiple controlled objects: "
                + ", ".join(str(obj.name) for obj in named_subjects))
        if len(named_subjects) != 1:
            raise ProgramBindingError(
                "ObjectHeld must identify exactly one controlled manifest object"
            )
    else:
        # A non-grasping program can still bind structurally, but only from an
        # object it names itself. This is not the retired "one movable in the
        # manifest, therefore pick it" fallback: an unreferenced object never
        # becomes the subject merely because it is alone.
        all_refs = [
            ref
            for stage in getattr(program, "stages", ())
            for ref in _refs(tuple(stage.running) + tuple(stage.terminal))
        ]
        named_subjects = [
            obj for obj in _unique_objects(all_refs, objects)
            if bool(getattr(obj, "movable", False))
        ]
        if len(named_subjects) != 1:
            raise ProgramBindingError(
                "program must explicitly reference exactly one controlled "
                "movable object when it has no ObjectHeld predicate"
            )
    subject = named_subjects[0]
    if not bool(getattr(subject, "movable", False)):
        raise ProgramBindingError(
            f"ObjectHeld binds non-movable object {subject.name!r}")

    stages = list(getattr(program, "stages", ()))
    final_refs = _refs(stages[-1].terminal) if stages else []
    external = [obj for obj in _unique_objects(final_refs, objects)
                if str(obj.name) != str(subject.name)]
    if len(external) > 1:
        raise ProgramBindingError(
            "final terminal references multiple external goal bodies: "
            + ", ".join(str(obj.name) for obj in external))
    if external:
        reference = external[0]
    else:
        canonical_target = any(
            ref == "target" or ref.startswith("target_")
            for ref in final_refs
        )
        if canonical_target:
            raise ProgramBindingError(
                "final terminal must name one manifest object's anchors; "
                "canonical target aliases do not infer scene roles"
            )
        reference = None
    return ProgramBindings(subject=subject, reference=reference)
