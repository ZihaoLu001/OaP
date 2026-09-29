"""Offline audit of a signed-pose *channel*, not self-certified observability.

Channel completeness proves only that every packet carries an explicitly
authorized, finite directed pose.  It cannot prove tracker accuracy or
symmetry disambiguation from those same packets.  A separately provisioned,
digest-bound reference authority manifest is required before an independently
recorded reference trace can validate accuracy or symmetry-branch continuity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from oap.loop.signed_pose_capability import (
    SignedPoseAuthorityRegistry,
    SignedPoseReferenceAuthorityRegistry,
    canonical_payload_sha256,
)
from oap.program.geometry import quat_to_matrix


_IDENTITY_FIELDS = ("source_id", "provenance", "recording_id")


@dataclass(frozen=True)
class SignedPoseChannelAudit:
    """Structured result of one offline channel/reference audit."""

    status: str
    body: str
    sample_count: int
    valid_sample_count: int
    reasons: tuple[str, ...]
    net_rotation_deg: float | None
    max_step_rotation_deg: float | None
    reference_sample_count: int | None = None
    max_reference_error_deg: float | None = None
    max_relative_step_error_deg: float | None = None
    symmetry_jump_count: int | None = None
    source_authority_id: str | None = None
    reference_authority_id: str | None = None
    reference_trust: str = "NOT_PROVIDED"

    @property
    def channel_complete(self) -> bool:
        return self.status in {
            "CHANNEL_COMPLETE",
            "REFERENCE_UNKNOWN",
            "REFERENCE_VALIDATED",
            "REFERENCE_FAILED",
        }

    @property
    def reference_validated(self) -> bool | None:
        if self.reference_sample_count is None:
            return None
        return self.status == "REFERENCE_VALIDATED"

    @property
    def observable(self) -> bool:
        """Compatibility property; only independent reference may make it true."""
        return self.status == "REFERENCE_VALIDATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "oap_signed_pose_channel_completeness_v3",
            "status": self.status,
            "channel_complete": self.channel_complete,
            "reference_validated": self.reference_validated,
            "observable": self.observable,
            "body": self.body,
            "sample_count": self.sample_count,
            "valid_sample_count": self.valid_sample_count,
            "reasons": list(self.reasons),
            "net_rotation_deg": self.net_rotation_deg,
            "max_step_rotation_deg": self.max_step_rotation_deg,
            "reference_sample_count": self.reference_sample_count,
            "max_reference_error_deg": self.max_reference_error_deg,
            "max_relative_step_error_deg": self.max_relative_step_error_deg,
            "symmetry_jump_count": self.symmetry_jump_count,
            "source_authority_id": self.source_authority_id,
            "reference_authority_id": self.reference_authority_id,
            "reference_trust": self.reference_trust,
            "required_runtime_behavior": (
                "missing capability/evidence keeps RelativeOrientation UNKNOWN"
            ),
            "does_not_establish_without_independent_reference": [
                "pose_accuracy",
                "object_symmetry_disambiguation",
                "tracking_latency",
                "closed_loop_task_success",
            ],
        }


# Kept as a type alias so existing import sites fail semantically closed while
# migrating to the more accurate name.
SignedPoseObservability = SignedPoseChannelAudit


def _packets(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        for field in ("observations", "frames", "packets"):
            values = payload.get(field)
            if isinstance(values, Sequence) and not isinstance(
                values,
                (str, bytes),
            ):
                return list(values)
        return [payload]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return list(payload)
    return []


def _body_row(packet: Mapping[str, Any], body: str) -> Mapping[str, Any] | None:
    bodies = packet.get("bodies")
    if isinstance(bodies, Mapping):
        value = bodies.get(body)
        return value if isinstance(value, Mapping) else None
    if packet.get("name") in {None, body}:
        return packet
    return None


def _finite_vector(value: Any, length: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _payload_identity(
    payload: Any,
    *,
    label: str,
) -> tuple[dict[str, str], list[str]]:
    """Read mandatory trace identity without accepting caller-supplied defaults."""
    identity: dict[str, str] = {}
    reasons: list[str] = []
    if not isinstance(payload, Mapping):
        return identity, [f"{label}_identity:wrapper_not_mapping"]
    for field in _IDENTITY_FIELDS:
        raw = payload.get(field)
        if not isinstance(raw, str) or not raw.strip():
            reasons.append(f"{label}_identity:{field}_missing")
            continue
        identity[field] = raw.strip()
    return identity, reasons


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def _trace_rotations(
    payload: Any,
    *,
    body: str,
    require_capability: bool,
    authority_registry: SignedPoseAuthorityRegistry | None = None,
    expected_source_id: str | None = None,
    expected_provenance: str | None = None,
    expected_recording_id: str | None = None,
) -> tuple[list[Any], list[np.ndarray], list[str], set[str]]:
    packets = _packets(payload)
    rotations: list[np.ndarray] = []
    reasons: list[str] = []
    authority_ids: set[str] = set()
    for index, packet in enumerate(packets):
        prefix = f"sample[{index}]"
        if not isinstance(packet, Mapping):
            reasons.append(f"{prefix}:packet_not_mapping")
            continue
        row = _body_row(packet, body)
        if row is None:
            reasons.append(f"{prefix}:body_missing")
            continue
        if require_capability:
            if authority_registry is None:
                reasons.append(f"{prefix}:authority_registry_missing")
                continue
            claims = authority_registry.verify_serialized_row(
                row,
                expected_body=body,
            )
            if claims is None:
                reasons.append(f"{prefix}:authority_verification_failed")
                continue
            if (
                expected_source_id is None
                or claims.authority_source_id != expected_source_id
            ):
                reasons.append(f"{prefix}:authority_source_id_mismatch")
                continue
            if (
                expected_provenance is None
                or claims.authority_provenance != expected_provenance
            ):
                reasons.append(f"{prefix}:authority_provenance_mismatch")
                continue
            if (
                expected_recording_id is None
                or claims.authority_recording_id != expected_recording_id
            ):
                reasons.append(f"{prefix}:authority_recording_id_mismatch")
                continue
            authority_ids.add(claims.authority_id)
            if row.get("track_lost") is True or row.get("fp_lost") is True:
                reasons.append(f"{prefix}:track_state_not_valid")
                continue
        elif row.get("track_lost") is True or row.get("fp_lost") is True:
            reasons.append(f"{prefix}:reference_track_lost")
            continue
        if require_capability:
            position = _finite_vector(
                row.get("pos_plan", row.get("pos_base")),
                3,
            )
            if position is None:
                reasons.append(f"{prefix}:position_missing_or_invalid")
                continue
        quaternion = _finite_vector(row.get("quat_wxyz"), 4)
        if quaternion is None:
            reasons.append(f"{prefix}:quaternion_missing_or_invalid")
            continue
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-9:
            reasons.append(f"{prefix}:quaternion_degenerate")
            continue
        rotations.append(quat_to_matrix(quaternion / norm))
    return packets, rotations, reasons, authority_ids


def evaluate_signed_pose_channel_completeness(
    payload: Any,
    *,
    body: str,
    min_samples: int = 2,
    authority_registry: SignedPoseAuthorityRegistry | None = None,
    reference_payload: Any | None = None,
    reference_authority_registry: (
        SignedPoseReferenceAuthorityRegistry | None
    ) = None,
    max_reference_error_deg: float = 10.0,
    symmetry_jump_deg: float = 45.0,
    source_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    source_provenance: str | None = None,
    reference_provenance: str | None = None,
) -> SignedPoseChannelAudit:
    """Audit channel completeness and optionally compare independent truth.

    ``reference_payload`` must come from an independent pose source or ground
    truth, and ``reference_authority_registry`` must bind its complete payload
    digest. Without that registry the reference is caller-attested only and
    can never validate observability. Per-step relative-rotation disagreement
    above ``symmetry_jump_deg`` is reported as a possible symmetry branch jump.
    """
    if not isinstance(body, str) or not body:
        raise ValueError("body must be a non-empty string")
    if isinstance(min_samples, bool) or int(min_samples) != min_samples:
        raise ValueError("min_samples must be an integer")
    minimum = int(min_samples)
    if minimum < 1:
        raise ValueError("min_samples must be >= 1")
    if not math.isfinite(max_reference_error_deg) or max_reference_error_deg < 0:
        raise ValueError("max_reference_error_deg must be finite and >= 0")
    if not math.isfinite(symmetry_jump_deg) or not 0 < symmetry_jump_deg <= 180:
        raise ValueError("symmetry_jump_deg must be finite and in (0, 180]")

    source_identity, source_identity_reasons = _payload_identity(
        payload,
        label="source",
    )
    packets, rotations, reasons, source_authority_ids = _trace_rotations(
        payload,
        body=body,
        require_capability=True,
        authority_registry=authority_registry,
        expected_source_id=source_identity.get("source_id"),
        expected_provenance=source_identity.get("provenance"),
        expected_recording_id=source_identity.get("recording_id"),
    )
    if source_provenance is not None and (
        not isinstance(source_provenance, str)
        or not source_provenance.strip()
        or source_provenance.strip() != source_identity.get("provenance")
    ):
        source_identity_reasons.append(
            "source_identity:provenance_argument_mismatch"
        )
    reasons.extend(source_identity_reasons)
    if len(source_authority_ids) != 1:
        reasons.append(
            "source_identity:authority_id_count_mismatch:"
            f"{len(source_authority_ids)}!=1"
        )
    if len(packets) < minimum:
        reasons.append(f"insufficient_samples:{len(packets)}<{minimum}")
    complete = bool(
        not source_identity_reasons
        and len(source_authority_ids) == 1
        and len(rotations) == len(packets)
        and len(rotations) >= minimum
    )
    source_authority_id = (
        next(iter(source_authority_ids))
        if len(source_authority_ids) == 1
        else None
    )

    net_rotation = None
    max_step = None
    if len(rotations) >= 2:
        net_rotation = math.degrees(
            _rotation_distance(rotations[0], rotations[-1])
        )
        max_step = math.degrees(
            max(
                _rotation_distance(first, second)
                for first, second in zip(rotations, rotations[1:])
            )
        )

    if reference_payload is None:
        return SignedPoseChannelAudit(
            status="CHANNEL_COMPLETE" if complete else "UNKNOWN",
            body=body,
            sample_count=len(packets),
            valid_sample_count=len(rotations),
            reasons=tuple(reasons),
            net_rotation_deg=net_rotation,
            max_step_rotation_deg=max_step,
            source_authority_id=source_authority_id,
        )

    reference_identity, reference_identity_reasons = _payload_identity(
        reference_payload,
        label="reference",
    )
    if reference_provenance is not None and (
        not isinstance(reference_provenance, str)
        or not reference_provenance.strip()
        or reference_provenance.strip() != reference_identity.get("provenance")
    ):
        reference_identity_reasons.append(
            "reference_identity:provenance_argument_mismatch"
        )
    independence_failures: list[str] = []
    reasons.extend(reference_identity_reasons)
    for field in _IDENTITY_FIELDS:
        source_value = source_identity.get(field)
        reference_value = reference_identity.get(field)
        if source_value is not None and source_value == reference_value:
            independence_failures.append(f"reference_same_{field}")
    if reference_payload is payload:
        independence_failures.append("reference_same_object")
    source_digest = canonical_payload_sha256(payload)
    reference_digest = canonical_payload_sha256(reference_payload)
    if source_digest is None or reference_digest is None:
        independence_failures.append("reference_digest_unavailable")
    elif source_digest == reference_digest:
        independence_failures.append("reference_same_digest")
    if source_path is not None and reference_path is not None:
        if Path(source_path).resolve() == Path(reference_path).resolve():
            independence_failures.append("reference_same_path")
    if reference_identity_reasons or independence_failures:
        reasons.extend(
            f"reference_independence:{failure}"
            for failure in independence_failures
        )
        return SignedPoseChannelAudit(
            status="UNKNOWN",
            body=body,
            sample_count=len(packets),
            valid_sample_count=len(rotations),
            reasons=tuple(reasons),
            net_rotation_deg=net_rotation,
            max_step_rotation_deg=max_step,
            reference_sample_count=len(_packets(reference_payload)),
            source_authority_id=source_authority_id,
            reference_trust="UNVERIFIED",
        )

    if reference_authority_registry is None:
        reasons.append(
            "reference_trust:trusted_reference_authority_registry_missing:"
            "caller_attested_only"
        )
        return SignedPoseChannelAudit(
            status="REFERENCE_UNKNOWN" if complete else "UNKNOWN",
            body=body,
            sample_count=len(packets),
            valid_sample_count=len(rotations),
            reasons=tuple(reasons),
            net_rotation_deg=net_rotation,
            max_step_rotation_deg=max_step,
            reference_sample_count=len(_packets(reference_payload)),
            source_authority_id=source_authority_id,
            reference_trust="CALLER_ATTESTED_ONLY",
        )

    reference_claims = reference_authority_registry.verify_payload(
        reference_payload,
        body=body,
    )
    if reference_claims is None:
        reasons.append("reference_trust:authority_manifest_verification_failed")
        return SignedPoseChannelAudit(
            status="UNKNOWN",
            body=body,
            sample_count=len(packets),
            valid_sample_count=len(rotations),
            reasons=tuple(reasons),
            net_rotation_deg=net_rotation,
            max_step_rotation_deg=max_step,
            reference_sample_count=len(_packets(reference_payload)),
            source_authority_id=source_authority_id,
            reference_trust="AUTHORITY_REJECTED",
        )
    if reference_claims.authority_id == source_authority_id:
        reasons.append("reference_independence:reference_same_authority_id")
        return SignedPoseChannelAudit(
            status="UNKNOWN",
            body=body,
            sample_count=len(packets),
            valid_sample_count=len(rotations),
            reasons=tuple(reasons),
            net_rotation_deg=net_rotation,
            max_step_rotation_deg=max_step,
            reference_sample_count=len(_packets(reference_payload)),
            source_authority_id=source_authority_id,
            reference_authority_id=reference_claims.authority_id,
            reference_trust="AUTHORITY_REJECTED",
        )

    ref_packets, references, ref_reasons, _ = _trace_rotations(
        reference_payload,
        body=body,
        require_capability=False,
    )
    reasons.extend(f"reference:{reason}" for reason in ref_reasons)
    reference_ready = (
        len(ref_packets) == len(packets)
        and len(references) == len(ref_packets)
        and len(references) >= minimum
    )
    if len(ref_packets) != len(packets):
        reasons.append(
            f"reference:sample_count_mismatch:{len(ref_packets)}!={len(packets)}"
        )

    max_error = None
    max_delta_error = None
    jump_count = None
    if complete and reference_ready:
        errors = [
            math.degrees(_rotation_distance(reference, observed))
            for reference, observed in zip(references, rotations, strict=True)
        ]
        max_error = max(errors, default=0.0)
        delta_errors = [
            math.degrees(
                _rotation_distance(
                    reference_first.T @ reference_second,
                    observed_first.T @ observed_second,
                )
            )
            for reference_first, reference_second, observed_first, observed_second
            in zip(
                references[:-1],
                references[1:],
                rotations[:-1],
                rotations[1:],
                strict=True,
            )
        ]
        max_delta_error = max(delta_errors, default=0.0)
        jump_count = sum(value > symmetry_jump_deg for value in delta_errors)
        if max_error > max_reference_error_deg:
            reasons.append(
                "reference:max_orientation_error_exceeded:"
                f"{max_error:.6g}>{max_reference_error_deg:.6g}"
            )
        if jump_count:
            reasons.append(f"reference:possible_symmetry_jumps:{jump_count}")

    validated = bool(
        complete
        and reference_ready
        and max_error is not None
        and max_error <= max_reference_error_deg
        and jump_count == 0
    )
    if not complete:
        status = "UNKNOWN"
    elif not reference_ready:
        status = "REFERENCE_UNKNOWN"
    else:
        status = "REFERENCE_VALIDATED" if validated else "REFERENCE_FAILED"
    return SignedPoseChannelAudit(
        status=status,
        body=body,
        sample_count=len(packets),
        valid_sample_count=len(rotations),
        reasons=tuple(reasons),
        net_rotation_deg=net_rotation,
        max_step_rotation_deg=max_step,
        reference_sample_count=len(ref_packets),
        max_reference_error_deg=max_error,
        max_relative_step_error_deg=max_delta_error,
        symmetry_jump_count=jump_count,
        source_authority_id=source_authority_id,
        reference_authority_id=reference_claims.authority_id,
        reference_trust="AUTHORITY_VERIFIED",
    )


def evaluate_signed_pose_observability(
    payload: Any,
    *,
    body: str,
    min_samples: int = 2,
    authority_registry: SignedPoseAuthorityRegistry | None = None,
    reference_payload: Any | None = None,
    reference_authority_registry: (
        SignedPoseReferenceAuthorityRegistry | None
    ) = None,
    max_reference_error_deg: float = 10.0,
    symmetry_jump_deg: float = 45.0,
    source_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    source_provenance: str | None = None,
    reference_provenance: str | None = None,
) -> SignedPoseChannelAudit:
    """Compatibility alias; a channel-only audit never returns observable."""
    return evaluate_signed_pose_channel_completeness(
        payload,
        body=body,
        min_samples=min_samples,
        authority_registry=authority_registry,
        reference_payload=reference_payload,
        reference_authority_registry=reference_authority_registry,
        max_reference_error_deg=max_reference_error_deg,
        symmetry_jump_deg=symmetry_jump_deg,
        source_path=source_path,
        reference_path=reference_path,
        source_provenance=source_provenance,
        reference_provenance=reference_provenance,
    )
