"""Authority-verified trust boundary for directed material-pose grounding.

Packets are untrusted data.  A packet cannot grant itself a signed-pose
capability by carrying a Boolean, a quaternion, or a capability-shaped JSON
object.  :func:`oap.loop.observe.observe_scene` verifies the packet
against a separately supplied authority registry, binds every relevant field,
and mints the process-local :class:`VerifiedSignedPose` token consumed by
grounding.  JSON round-tripping deliberately strips the token's Python type,
so replayed packet text cannot cross the runtime trust boundary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import math
import secrets
from typing import Any

import numpy as np


SIGNED_POSE_AUTHORITY_SCHEMA = "oap_signed_pose_authority_registry_v3"
SIGNED_POSE_VERIFICATION_SCHEMA = "oap_verified_signed_pose_v3"
SIGNED_POSE_REFERENCE_AUTHORITY_SCHEMA = (
    "oap_signed_pose_reference_authority_registry_v1"
)
_SHA256_HEX_LEN = 64
_TOKEN_SECRET = secrets.token_bytes(32)
_TOKEN_MINT_KEY = object()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LEN:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def quaternion_sha256(value: Any) -> str | None:
    """Hash one finite normalized wxyz quaternion in canonical float64 form."""
    try:
        quaternion = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return None
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    normalized = quaternion / norm
    # q and -q encode the same rotation. Canonicalize the sign so the digest
    # binds orientation rather than a transport-library sign convention.
    first_nonzero = next(
        (float(value) for value in normalized if abs(float(value)) > 1e-15),
        1.0,
    )
    if first_nonzero < 0.0:
        normalized = -normalized
    return hashlib.sha256(normalized.astype("<f8").tobytes()).hexdigest()


def canonical_payload_sha256(value: Any) -> str | None:
    """Hash one JSON payload using the audit's canonical representation."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SignedPoseAuthority:
    """One independently reviewed capability entry in a trusted registry."""

    authority_id: str
    source_id: str
    provenance: str
    recording_id: str
    body: str
    raw_schema: str
    calibration_sha256: str
    independent_validation_sha256: str

    def validate(self) -> None:
        for label, value in (
            ("authority_id", self.authority_id),
            ("source_id", self.source_id),
            ("provenance", self.provenance),
            ("recording_id", self.recording_id),
            ("body", self.body),
            ("raw_schema", self.raw_schema),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"signed-pose authority {label} is invalid")
        if not _is_sha256(self.calibration_sha256):
            raise ValueError("signed-pose authority calibration_sha256 is invalid")
        if not _is_sha256(self.independent_validation_sha256):
            raise ValueError(
                "signed-pose authority independent_validation_sha256 is invalid"
            )


@dataclass(frozen=True)
class VerifiedSignedPoseClaims:
    """Fields verified before the runtime token is minted."""

    authority_id: str
    authority_source_id: str
    authority_provenance: str
    authority_recording_id: str
    body: str
    observation_id: str
    capture_id: str
    frame_id: str
    raw_schema: str
    quaternion_sha256: str
    calibration_sha256: str
    independent_validation_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": SIGNED_POSE_VERIFICATION_SCHEMA,
            "authority_id": self.authority_id,
            "authority_source_id": self.authority_source_id,
            "authority_provenance": self.authority_provenance,
            "authority_recording_id": self.authority_recording_id,
            "body": self.body,
            "observation_id": self.observation_id,
            "capture_id": self.capture_id,
            "frame_id": self.frame_id,
            "raw_schema": self.raw_schema,
            "quaternion_sha256": self.quaternion_sha256,
            "calibration_sha256": self.calibration_sha256,
            "independent_validation_sha256": (
                self.independent_validation_sha256
            ),
        }


class VerifiedSignedPose(str):
    """Opaque process-local token; only observe_scene owns the mint key."""

    def __new__(
        cls,
        value: str,
        *,
        _mint_key: object | None = None,
    ) -> "VerifiedSignedPose":
        if _mint_key is not _TOKEN_MINT_KEY:
            raise TypeError(
                "VerifiedSignedPose tokens are minted only by observe_scene"
            )
        return str.__new__(cls, value)

    def __copy__(self) -> "VerifiedSignedPose":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "VerifiedSignedPose":
        memo[id(self)] = self
        return self


class SignedPoseAuthorityRegistry:
    """Immutable, separately provisioned signed-pose authority registry."""

    def __init__(self, authorities: Sequence[SignedPoseAuthority]) -> None:
        checked: dict[str, SignedPoseAuthority] = {}
        for authority in authorities:
            authority.validate()
            if authority.authority_id in checked:
                raise ValueError(
                    "duplicate signed-pose authority_id "
                    f"{authority.authority_id!r}"
                )
            checked[authority.authority_id] = authority
        self._authorities = checked

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignedPoseAuthorityRegistry":
        if value.get("schema") != SIGNED_POSE_AUTHORITY_SCHEMA:
            raise ValueError("signed-pose authority registry schema is invalid")
        raw_entries = value.get("authorities")
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries,
            (str, bytes),
        ):
            raise ValueError("signed-pose authority registry entries are invalid")
        entries: list[SignedPoseAuthority] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise ValueError("signed-pose authority entry is not an object")
            entries.append(
                SignedPoseAuthority(
                    authority_id=str(raw.get("authority_id", "")),
                    source_id=str(raw.get("source_id", "")),
                    provenance=str(raw.get("provenance", "")),
                    recording_id=str(raw.get("recording_id", "")),
                    body=str(raw.get("body", "")),
                    raw_schema=str(raw.get("raw_schema", "")),
                    calibration_sha256=str(raw.get("calibration_sha256", "")),
                    independent_validation_sha256=str(
                        raw.get("independent_validation_sha256", "")
                    ),
                )
            )
        return cls(entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SIGNED_POSE_AUTHORITY_SCHEMA,
            "authorities": [
                {
                    "authority_id": authority.authority_id,
                    "source_id": authority.source_id,
                    "provenance": authority.provenance,
                    "recording_id": authority.recording_id,
                    "body": authority.body,
                    "raw_schema": authority.raw_schema,
                    "calibration_sha256": authority.calibration_sha256,
                    "independent_validation_sha256": (
                        authority.independent_validation_sha256
                    ),
                }
                for authority in self._authorities.values()
            ],
        }

    def resolve_authority_id(
        self,
        *,
        body: str,
        raw_schema: str,
        calibration_sha256: str,
    ) -> str | None:
        matches = [
            authority.authority_id
            for authority in self._authorities.values()
            if authority.body == body
            and authority.raw_schema == raw_schema
            and authority.calibration_sha256 == calibration_sha256
        ]
        return matches[0] if len(matches) == 1 else None

    def verify_claims(
        self,
        *,
        authority_id: str,
        body: str,
        observation_id: str,
        capture_id: str,
        frame_id: str,
        raw_schema: str,
        quaternion_wxyz: Any,
        calibration_sha256: str,
    ) -> VerifiedSignedPoseClaims | None:
        authority = self._authorities.get(authority_id)
        if authority is None:
            return None
        for value in (body, observation_id, capture_id, frame_id, raw_schema):
            if not isinstance(value, str) or not value.strip():
                return None
        if authority.body != body or authority.raw_schema != raw_schema:
            return None
        if authority.calibration_sha256 != calibration_sha256:
            return None
        quaternion_digest = quaternion_sha256(quaternion_wxyz)
        if quaternion_digest is None:
            return None
        return VerifiedSignedPoseClaims(
            authority_id=authority.authority_id,
            authority_source_id=authority.source_id,
            authority_provenance=authority.provenance,
            authority_recording_id=authority.recording_id,
            body=body,
            observation_id=observation_id,
            capture_id=capture_id,
            frame_id=frame_id,
            raw_schema=raw_schema,
            quaternion_sha256=quaternion_digest,
            calibration_sha256=authority.calibration_sha256,
            independent_validation_sha256=(
                authority.independent_validation_sha256
            ),
        )

    def verify_serialized_row(
        self,
        row: Mapping[str, Any],
        *,
        expected_body: str | None = None,
    ) -> VerifiedSignedPoseClaims | None:
        verification = row.get("signed_pose_verification")
        if not isinstance(verification, Mapping):
            return None
        body = row.get("name")
        if not isinstance(body, str):
            return None
        if expected_body is not None and body != expected_body:
            return None
        claims = self.verify_claims(
            authority_id=str(verification.get("authority_id", "")),
            body=body,
            observation_id=str(row.get("observation_id", "")),
            capture_id=str(row.get("capture_id", "")),
            frame_id=str(row.get("frame_id", "")),
            raw_schema=str(row.get("source", row.get("schema", ""))),
            quaternion_wxyz=row.get("quat_wxyz"),
            calibration_sha256=str(row.get("calibration_sha256", "")),
        )
        if claims is None:
            return None
        expected = claims.to_dict()
        return claims if all(verification.get(k) == v for k, v in expected.items()) else None


@dataclass(frozen=True)
class SignedPoseReferenceAuthority:
    """Trusted manifest entry for one immutable independent reference trace."""

    authority_id: str
    source_id: str
    provenance: str
    recording_id: str
    body: str
    payload_sha256: str
    calibration_sha256: str
    independent_validation_sha256: str

    def validate(self) -> None:
        for label, value in (
            ("authority_id", self.authority_id),
            ("source_id", self.source_id),
            ("provenance", self.provenance),
            ("recording_id", self.recording_id),
            ("body", self.body),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"signed-pose reference authority {label} is invalid"
                )
        for label, value in (
            ("payload_sha256", self.payload_sha256),
            ("calibration_sha256", self.calibration_sha256),
            (
                "independent_validation_sha256",
                self.independent_validation_sha256,
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(
                    f"signed-pose reference authority {label} is invalid"
                )


@dataclass(frozen=True)
class VerifiedReferencePoseClaims:
    """Claims produced only by a digest-bound trusted reference registry."""

    authority_id: str
    source_id: str
    provenance: str
    recording_id: str
    body: str
    payload_sha256: str
    calibration_sha256: str
    independent_validation_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "authority_id": self.authority_id,
            "source_id": self.source_id,
            "provenance": self.provenance,
            "recording_id": self.recording_id,
            "body": self.body,
            "payload_sha256": self.payload_sha256,
            "calibration_sha256": self.calibration_sha256,
            "independent_validation_sha256": (
                self.independent_validation_sha256
            ),
        }


class SignedPoseReferenceAuthorityRegistry:
    """Separately provisioned manifest for immutable reference recordings."""

    def __init__(
        self,
        authorities: Sequence[SignedPoseReferenceAuthority],
    ) -> None:
        checked: dict[str, SignedPoseReferenceAuthority] = {}
        for authority in authorities:
            authority.validate()
            if authority.authority_id in checked:
                raise ValueError(
                    "duplicate signed-pose reference authority_id "
                    f"{authority.authority_id!r}"
                )
            checked[authority.authority_id] = authority
        self._authorities = checked

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "SignedPoseReferenceAuthorityRegistry":
        if value.get("schema") != SIGNED_POSE_REFERENCE_AUTHORITY_SCHEMA:
            raise ValueError(
                "signed-pose reference authority registry schema is invalid"
            )
        raw_entries = value.get("authorities")
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries,
            (str, bytes),
        ):
            raise ValueError(
                "signed-pose reference authority registry entries are invalid"
            )
        entries: list[SignedPoseReferenceAuthority] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise ValueError(
                    "signed-pose reference authority entry is not an object"
                )
            entries.append(
                SignedPoseReferenceAuthority(
                    authority_id=str(raw.get("authority_id", "")),
                    source_id=str(raw.get("source_id", "")),
                    provenance=str(raw.get("provenance", "")),
                    recording_id=str(raw.get("recording_id", "")),
                    body=str(raw.get("body", "")),
                    payload_sha256=str(raw.get("payload_sha256", "")),
                    calibration_sha256=str(
                        raw.get("calibration_sha256", "")
                    ),
                    independent_validation_sha256=str(
                        raw.get("independent_validation_sha256", "")
                    ),
                )
            )
        return cls(entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SIGNED_POSE_REFERENCE_AUTHORITY_SCHEMA,
            "authorities": [
                {
                    "authority_id": authority.authority_id,
                    "source_id": authority.source_id,
                    "provenance": authority.provenance,
                    "recording_id": authority.recording_id,
                    "body": authority.body,
                    "payload_sha256": authority.payload_sha256,
                    "calibration_sha256": authority.calibration_sha256,
                    "independent_validation_sha256": (
                        authority.independent_validation_sha256
                    ),
                }
                for authority in self._authorities.values()
            ],
        }

    def verify_payload(
        self,
        payload: Any,
        *,
        body: str,
    ) -> VerifiedReferencePoseClaims | None:
        """Verify wrapper identity and bytes against one trusted manifest row."""
        if not isinstance(payload, Mapping):
            return None
        payload_digest = canonical_payload_sha256(payload)
        if payload_digest is None:
            return None
        source_id = payload.get("source_id")
        provenance = payload.get("provenance")
        recording_id = payload.get("recording_id")
        calibration_digest = payload.get("calibration_sha256")
        validation_digest = payload.get("independent_validation_sha256")
        values = (
            source_id,
            provenance,
            recording_id,
            calibration_digest,
            validation_digest,
        )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return None
        matches = [
            authority
            for authority in self._authorities.values()
            if authority.source_id == source_id
            and authority.provenance == provenance
            and authority.recording_id == recording_id
            and authority.body == body
            and authority.payload_sha256 == payload_digest
            and authority.calibration_sha256 == calibration_digest
            and authority.independent_validation_sha256 == validation_digest
        ]
        if len(matches) != 1:
            return None
        authority = matches[0]
        return VerifiedReferencePoseClaims(
            authority_id=authority.authority_id,
            source_id=authority.source_id,
            provenance=authority.provenance,
            recording_id=authority.recording_id,
            body=authority.body,
            payload_sha256=authority.payload_sha256,
            calibration_sha256=authority.calibration_sha256,
            independent_validation_sha256=(
                authority.independent_validation_sha256
            ),
        )


def _token_payload(claims: VerifiedSignedPoseClaims) -> bytes:
    return json.dumps(
        claims.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mint_verified_signed_pose(
    claims: VerifiedSignedPoseClaims,
) -> VerifiedSignedPose:
    """Private mint called only after observe_scene registry verification."""
    payload = _token_payload(claims)
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    signature = hmac.new(_TOKEN_SECRET, payload, hashlib.sha256).hexdigest()
    return VerifiedSignedPose(
        f"vsp1.{encoded}.{signature}",
        _mint_key=_TOKEN_MINT_KEY,
    )


def _claims_from_token(value: Any) -> VerifiedSignedPoseClaims | None:
    if not isinstance(value, VerifiedSignedPose):
        return None
    try:
        prefix, encoded, signature = str(value).split(".", 2)
        if prefix != "vsp1":
            return None
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = base64.urlsafe_b64decode(padded.encode("ascii"))
        expected = hmac.new(_TOKEN_SECRET, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        raw = json.loads(payload.decode("utf-8"))
        if not isinstance(raw, Mapping):
            return None
        return VerifiedSignedPoseClaims(
            authority_id=str(raw["authority_id"]),
            authority_source_id=str(raw["authority_source_id"]),
            authority_provenance=str(raw["authority_provenance"]),
            authority_recording_id=str(raw["authority_recording_id"]),
            body=str(raw["body"]),
            observation_id=str(raw["observation_id"]),
            capture_id=str(raw["capture_id"]),
            frame_id=str(raw["frame_id"]),
            raw_schema=str(raw["raw_schema"]),
            quaternion_sha256=str(raw["quaternion_sha256"]),
            calibration_sha256=str(raw["calibration_sha256"]),
            independent_validation_sha256=str(
                raw["independent_validation_sha256"]
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def verified_signed_pose_failure(
    row: Mapping[str, Any] | Any,
    *,
    expected_body: str | None = None,
) -> str | None:
    """Validate the runtime token and every field to which it was bound."""
    if not isinstance(row, Mapping):
        return "verified_row_missing"
    claims = _claims_from_token(row.get("_verified_signed_pose"))
    if claims is None:
        return "verified_token_missing_or_invalid"
    body = row.get("name")
    if not isinstance(body, str) or claims.body != body:
        return "verified_body_mismatch"
    if expected_body is not None and body != expected_body:
        return "verified_expected_body_mismatch"
    comparisons = {
        "authority_source_id": row.get("signed_pose_source_id"),
        "authority_provenance": row.get("signed_pose_provenance"),
        "authority_recording_id": row.get("signed_pose_recording_id"),
        "observation_id": row.get("observation_id"),
        "capture_id": row.get("capture_id"),
        "frame_id": row.get("frame_id"),
        "raw_schema": row.get("source", row.get("schema")),
        "calibration_sha256": row.get("calibration_sha256"),
        "independent_validation_sha256": row.get(
            "independent_validation_sha256"
        ),
    }
    for field, observed in comparisons.items():
        if getattr(claims, field) != observed:
            return f"verified_{field}_mismatch"
    quaternion_digest = quaternion_sha256(row.get("quat_wxyz"))
    if claims.quaternion_sha256 != quaternion_digest:
        return "verified_quaternion_mismatch"
    verification = row.get("signed_pose_verification")
    if not isinstance(verification, Mapping) or any(
        verification.get(key) != value
        for key, value in claims.to_dict().items()
    ):
        return "verified_public_claims_mismatch"
    if row.get("track_lost") is True or row.get("fp_lost") is True:
        return "verified_track_lost"
    return None


def signed_pose_frames_authorized(
    row: Mapping[str, Any] | Any,
    *,
    expected_body: str | None = None,
) -> bool:
    """Whether observe_scene's typed, field-bound token authorizes a triad."""
    return verified_signed_pose_failure(
        row,
        expected_body=expected_body,
    ) is None


_SIM_CALIBRATION_SHA256 = hashlib.sha256(
    b"oap:sim:w_plan:mujoco-freejoint-qpos7:v1"
).hexdigest()
_SIM_VALIDATION_SHA256 = hashlib.sha256(
    b"oap:independent-contract-review:sim-exact-state:v1"
).hexdigest()


def _sim_exact_registry(body: str) -> SignedPoseAuthorityRegistry:
    """Internal authority used only by observe_scene's exact-state branch."""
    return SignedPoseAuthorityRegistry(
        [
            SignedPoseAuthority(
                authority_id=f"sim_exact_state:{body}",
                source_id="mujoco_exact_qpos7",
                provenance="trusted:internal:mujoco-exact-state:v1",
                recording_id="in_process_sim_exact_state",
                body=body,
                raw_schema="sim_scene_body_observation",
                calibration_sha256=_SIM_CALIBRATION_SHA256,
                independent_validation_sha256=_SIM_VALIDATION_SHA256,
            )
        ]
    )
