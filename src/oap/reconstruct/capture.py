"""Step CAPTURE: acquire and import the RGB-D observation packet.

Role in the two-stage pipeline: this is the ONLY reconstruction step that
touches hardware. :func:`capture_objects` runs on the lab host (the sole
machine with the ZED 2i) and drives the observer-env payload
``payloads/zed_capture.py``, which writes one observation packet per object
(schema ``real_zed2i_rgbd_observation_v1``). :func:`import_packet` then copies
a packet into a scene bundle and re-anchors its manifest path fields to the
copy, making the packet the IMMUTABLE input of every downstream step -- steps
never modify packet pixels/depth, and re-imports are refused unless forced.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from oap.reconstruct.external import ExternalEnvs
from oap.reconstruct.pose import (
    load_fixed_camera_calibration,
    normalize_camera_serial,
)
from oap.reconstruct.quality import QUALITY_PROFILE
from oap.utils.io import read_json, write_json_atomic

logger = logging.getLogger("oap.reconstruct.capture")

__all__ = [
    "PACKET_CONTENT_FILES",
    "capture_objects",
    "import_packet",
    "packet_dir_in_bundle",
    "reanchor_manifest",
    "validate_observation_packet",
    "validate_packet_camera_serial",
]

#: Files every observation packet must contain (completeness validation, run
#: after a capture and again on import).
PACKET_CONTENT_FILES = ("rgb.png", "depth_m.npy", "mask_00.png", "cam_K.txt")

#: Manifest keys holding a PACKET-LOCAL filesystem path, which must follow the
#: packet when it is copied between hosts (the path-valued fields
#: ``payloads/zed_capture.py`` writes). ``T_base_zed2i`` is deliberately absent:
#: it names the packaged hand-eye calibration YAML, which lives OUTSIDE the
#: packet, so re-anchoring it to ``<packet>/<basename>`` would break a working
#: path instead of fixing a stale one.
_MANIFEST_PATH_KEYS = (
    "rgb",
    "depth_m_npy",
    "depth_mm_png",
    "mask",
    "sam31_mask_packet",
    "foundationpose_recording",
)


def packet_dir_in_bundle(bundle: Path, name: str) -> Path:
    """Return the canonical imported-packet location ``<bundle>/<name>/capture``."""
    return Path(bundle) / name / "capture"


def validate_packet_camera_serial(
    manifest: Mapping[str, object],
    *,
    packet: Path | None = None,
) -> dict[str, str]:
    """Bind one packet to the camera named by the fixed-rig calibration.

    Per-object reconstruction must never repair a camera/object transform by
    hand. The only accepted base transform is the packaged, one-time
    eye-to-hand calibration, so the runtime device serial recorded at capture
    must match that calibration before any reconstruction step may run.
    """
    where = f"observation packet {packet}" if packet is not None else "observation packet"
    if "camera_serial" not in manifest:
        raise ValueError(f"{where} has no runtime camera_serial")
    runtime_serial = normalize_camera_serial(manifest["camera_serial"])
    _, source, calibrated_serial, digest = load_fixed_camera_calibration()
    if runtime_serial != calibrated_serial:
        raise ValueError(
            f"{where} was captured by camera serial {runtime_serial}, but "
            f"fixed-rig calibration {source} is for serial {calibrated_serial}"
        )
    return {
        "camera_serial": runtime_serial,
        "calibration_source": source,
        "calibration_sha256": digest,
    }


def validate_observation_packet(
    packet: Path,
    *,
    require_mask: bool = True,
    captured_after_s: float | None = None,
    captured_before_s: float | None = None,
) -> dict[str, object]:
    """Validate that a packet contains usable metric depth, not just filenames.

    Real terminal geometry may be certified only from an RGB-D observation.
    Existence checks alone accepted zero-filled depth arrays and manifests
    pointing at a stale packet on another host. Mask-bearing packets must also
    pass the one global SAM3/depth evidence floor: score, mask support, and the
    fraction of mask pixels carrying metric depth. This validator is shared by
    reconstruction and the online observer and fails before planning or robot
    connection when evidence is absent, malformed, or too sparse.
    """
    packet = Path(packet).expanduser().resolve()
    required = ["rgb.png", "depth_m.npy", "cam_K.txt",
                "observation_manifest.json"]
    if require_mask:
        required.append("mask_00.png")
    missing = [
        name
        for name in required
        if not (packet / name).is_file()
        or (packet / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"observation packet {packet} is missing {missing}; expected the "
            f"schema real_zed2i_rgbd_observation_v1 layout written by "
            f"`oap-reconstruct capture`."
        )
    manifest = read_json(packet / "observation_manifest.json")
    if manifest.get("schema") != "real_zed2i_rgbd_observation_v1":
        raise ValueError(
            f"observation packet {packet} has unsupported schema "
            f"{manifest.get('schema')!r}"
        )
    camera = validate_packet_camera_serial(manifest, packet=packet)
    fp_meta_path = packet / "foundationpose_recording" / "metadata.json"
    if fp_meta_path.exists():
        fp_meta = read_json(fp_meta_path)
        try:
            fp_serial = normalize_camera_serial(fp_meta["camera_serial"])
        except KeyError as exc:
            raise ValueError(
                f"observation packet {packet} FoundationPose metadata has no "
                f"runtime camera_serial"
            ) from exc
        if fp_serial != camera["camera_serial"]:
            raise ValueError(
                f"observation packet {packet} camera serial differs between "
                f"observation_manifest.json ({camera['camera_serial']}) and "
                f"FoundationPose metadata ({fp_serial})"
            )
        for identity_field in (
            "capture_id",
            "capture_unix_s",
            "capture_monotonic_s",
            "rgb_array_sha256",
            "depth_m_array_sha256",
        ):
            if (
                identity_field in manifest
                and fp_meta.get(identity_field)
                != manifest.get(identity_field)
            ):
                raise ValueError(
                    f"observation packet {packet} differs between manifest "
                    f"copies for {identity_field}"
                )
    try:
        capture_unix_s = float(manifest["capture_unix_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"observation packet {packet} has no valid capture_unix_s"
        ) from exc
    if not np.isfinite(capture_unix_s) or capture_unix_s <= 0.0:
        raise ValueError(
            f"observation packet {packet} has no valid capture_unix_s"
        )
    if (
        captured_after_s is not None
        and capture_unix_s < float(captured_after_s)
    ):
        raise ValueError(
            f"observation packet {packet} is stale: capture_unix_s "
            f"{capture_unix_s:.6f} predates this capture request "
            f"{float(captured_after_s):.6f}"
        )
    if (
        captured_before_s is not None
        and capture_unix_s > float(captured_before_s)
    ):
        raise ValueError(
            f"observation packet {packet} has a future capture_unix_s "
            f"{capture_unix_s:.6f} after validation bound "
            f"{float(captured_before_s):.6f}"
        )
    raw_depth = manifest.get("depth_m_npy")
    if not isinstance(raw_depth, str) or not raw_depth.strip():
        raise ValueError(
            f"observation packet {packet} does not declare depth_m_npy"
        )
    declared_depth = Path(raw_depth).expanduser()
    if not declared_depth.is_absolute():
        declared_depth = packet / declared_depth
    expected_depth = (packet / "depth_m.npy").resolve()
    if declared_depth.resolve() != expected_depth:
        raise ValueError(
            f"observation packet {packet} depth_m_npy points outside the "
            f"packet ({declared_depth}); expected {expected_depth}"
        )
    try:
        depth = np.load(expected_depth, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"observation packet {packet} depth_m.npy is unreadable: {exc}"
        ) from exc
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError(
            f"observation packet {packet} depth must be a non-empty 2-D array, "
            f"got shape {depth.shape}"
        )
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    valid_count = int(np.count_nonzero(valid_depth))
    if valid_count == 0:
        raise ValueError(
            f"observation packet {packet} has no finite positive metric depth"
        )
    depth_digest = hashlib.sha256(
        np.ascontiguousarray(depth, dtype=np.float32).tobytes()
    ).hexdigest()
    declared_depth_digest = manifest.get("depth_m_array_sha256")
    if (
        declared_depth_digest is not None
        and str(declared_depth_digest) != depth_digest
    ):
        raise ValueError(
            f"observation packet {packet} depth hash does not match manifest"
        )
    declared_rgb_digest = manifest.get("rgb_array_sha256")
    rgb_digest: str | None = None
    if declared_rgb_digest is not None:
        try:
            rgb = np.asarray(
                Image.open(packet / "rgb.png").convert("RGB")
            )
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"observation packet {packet} rgb.png is unreadable: {exc}"
            ) from exc
        rgb_digest = hashlib.sha256(
            np.ascontiguousarray(rgb).tobytes()
        ).hexdigest()
        if str(declared_rgb_digest) != rgb_digest:
            raise ValueError(
                f"observation packet {packet} RGB hash does not match "
                "manifest"
            )
    try:
        camera_matrix = np.loadtxt(packet / "cam_K.txt", dtype=float)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"observation packet {packet} cam_K.txt is unreadable: {exc}"
        ) from exc
    if (
        camera_matrix.shape != (3, 3)
        or not np.all(np.isfinite(camera_matrix))
        or float(camera_matrix[0, 0]) <= 0.0
        or float(camera_matrix[1, 1]) <= 0.0
    ):
        raise ValueError(
            f"observation packet {packet} has invalid camera intrinsics"
        )
    evidence: dict[str, object] = {
        "schema": "oap_rgbd_packet_validation_v1",
        "packet": str(packet),
        "capture_id": manifest.get("capture_id"),
        "capture_unix_s": capture_unix_s,
        "capture_monotonic_s": manifest.get("capture_monotonic_s"),
        "rgb_array_sha256": rgb_digest,
        "depth_m_array_sha256": depth_digest,
        "depth_available": True,
        "depth_shape": [int(v) for v in depth.shape],
        "valid_depth_samples": valid_count,
        "total_depth_samples": int(depth.size),
        "mask_required": bool(require_mask),
        **camera,
    }
    if not require_mask:
        return evidence

    mask_path = packet / "mask_00.png"
    try:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"observation packet {packet} mask_00.png is unreadable: {exc}"
        ) from exc
    if mask.shape != depth.shape:
        raise ValueError(
            f"observation packet {packet} mask/depth shape mismatch: "
            f"{mask.shape} vs {depth.shape}"
        )
    mask_pixels = int(np.count_nonzero(mask))
    try:
        declared_mask_pixels = int(manifest["mask_pixels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"observation packet {packet} has no valid mask_pixels"
        ) from exc
    if declared_mask_pixels != mask_pixels:
        raise ValueError(
            f"observation packet {packet} mask_pixels={declared_mask_pixels} "
            f"does not match mask_00.png ({mask_pixels})"
        )
    try:
        sam3_score = float(manifest["sam3_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"observation packet {packet} has no valid sam3_score"
        ) from exc
    if not np.isfinite(sam3_score) or not 0.0 <= sam3_score <= 1.0:
        raise ValueError(
            f"observation packet {packet} has invalid sam3_score "
            f"{sam3_score!r}"
        )
    if sam3_score < QUALITY_PROFILE.min_sam3_score:
        raise ValueError(
            f"observation packet {packet} SAM3 score {sam3_score:.3f} is "
            f"below the global minimum "
            f"{QUALITY_PROFILE.min_sam3_score:.3f}"
        )
    if mask_pixels < QUALITY_PROFILE.min_sam3_mask_pixels:
        raise ValueError(
            f"observation packet {packet} SAM3 mask has {mask_pixels} pixels, "
            f"below the global minimum "
            f"{QUALITY_PROFILE.min_sam3_mask_pixels}"
        )

    masked_depth_samples = int(np.count_nonzero(mask & valid_depth))
    masked_depth_coverage = masked_depth_samples / float(mask_pixels)
    if masked_depth_coverage < QUALITY_PROFILE.min_masked_depth_coverage:
        raise ValueError(
            f"observation packet {packet} masked-depth coverage "
            f"{masked_depth_coverage:.3f} is below the global minimum "
            f"{QUALITY_PROFILE.min_masked_depth_coverage:.3f}"
        )
    evidence.update(
        {
            "quality_profile": QUALITY_PROFILE.to_dict(),
            "sam3_score": sam3_score,
            "mask_pixels": mask_pixels,
            "masked_depth_samples": masked_depth_samples,
            "masked_depth_coverage": masked_depth_coverage,
        }
    )
    return evidence


def _validate_packet(packet: Path) -> None:
    """Compatibility wrapper for reconstruction's mask-bearing packets."""
    validate_observation_packet(packet, require_mask=True)


def capture_objects(
    capture_dir: Path,
    objects: Mapping[str, str],
    envs: ExternalEnvs,
    *,
    force: bool = False,
    settle_frames: int = 15,
) -> list[Path]:
    """Capture one ZED frame and write an observation packet per object.

    LAB HOST ONLY: requires the observer env (pyzed + SAM3). All requested
    objects are segmented from the SAME frame (one grab, N prompts), so the
    scene geometry is mutually consistent across packets.

    Idempotent step contract: an object whose packet already exists under
    ``capture_dir`` is skipped (the packet is immutable downstream input);
    pass ``force=True`` to recapture.

    Args:
        capture_dir: Output directory; packets land at ``<capture_dir>/<name>``.
        objects: Mapping of object name -> SAM3 text prompt.
        envs: Resolved external environments (needs the ``observer`` tool).
        force: Recapture even when packets already exist.
        settle_frames: ZED auto-exposure settle frames before the kept grab.

    Returns:
        The packet directories, one per requested object.

    Raises:
        ExternalEnvError: If the observer env is missing or the payload fails.
        FileNotFoundError: If a produced packet is incomplete.
    """
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)

    pending: dict[str, str] = {}
    for name, prompt in objects.items():
        packet = capture_dir / name
        if not force and (packet / "observation_manifest.json").exists():
            logger.info("[capture] %s: packet exists, skipping (use --force to recapture)", name)
            continue
        pending[name] = prompt

    if pending:
        args: list[str | Path] = ["--out", capture_dir, "--settle-frames", str(settle_frames)]
        for name, prompt in pending.items():
            args += ["--object", name, prompt]
        sam3_dir = envs.extra("observer", "sam3_service_dir", required=False)
        if sam3_dir:
            args += ["--sam3-service-dir", sam3_dir]
        envs.run_payload("observer", "zed_capture", args)

    packets = []
    for name in objects:
        packet = capture_dir / name
        _validate_packet(packet)
        packets.append(packet)
    return packets


def reanchor_manifest(packet: Path) -> dict:
    """Re-anchor the packet manifest's path fields to the packet's location.

    Packets are captured on the lab host and may be synced to another
    filesystem on another host. The packet manifest stores absolute paths and is
    the one artifact carried between hosts as downstream INPUT: bundle
    manifests are written bundle-relative (:func:`bundle.assemble_manifest`),
    and the step reports that also record absolutes (``scale_report.json``)
    are regenerated outputs, not inputs. So it is the one thing that must be
    re-anchored after a copy -- downstream payloads read those absolutes
    straight from ``observation_manifest.json`` -- and each path field is
    rewritten to ``<packet>/<basename>``. This is the only path rewriting in
    the repo and it only ever touches path STRINGS -- never pixels, depth,
    masks, or intrinsics, and never a ``null``.

    A null field is a FACT about the packet, not a stale path: a maskless
    ``--no-seg`` track packet has ``mask`` and ``sam31_mask_packet`` null
    because those artifacts do not exist. Re-anchoring a null would fabricate
    the literal ``"<packet>/None"``, which is truthy and absolute, so it
    survives ``scale._resolve_manifest_path`` and turns that step's clear
    "manifest must contain mask and depth_m_npy" into a FileNotFoundError for a
    file named ``None`` -- and it LATCHES, because the next pass finds the
    literal already equal to the recomputed value. So the guard is on the
    VALUE, mirroring ``scale._resolve_manifest_path``: never build a Path from
    a value not first proven a non-empty string.

    Args:
        packet: The packet directory containing ``observation_manifest.json``.

    Returns:
        The (possibly rewritten) manifest payload.
    """
    packet = Path(packet).resolve()

    def _reanchor(doc: dict) -> bool:
        """Point ``doc``'s path strings at ``packet``; True if any moved."""
        touched = False
        for key in _MANIFEST_PATH_KEYS:
            raw = doc.get(key)
            # Absent, null, "" or a non-string: not a path, so not ours to
            # rewrite. This is a best-effort fixup called from an unguarded
            # site (bundle.run_step), so a malformed field is left for its
            # consumer to report rather than aborting the whole import.
            if not isinstance(raw, str) or not raw:
                continue
            anchored = str(packet / Path(raw).name)
            if raw != anchored:
                doc[key] = anchored
                touched = True
        return touched

    manifest_path = packet / "observation_manifest.json"
    meta = read_json(manifest_path)
    if _reanchor(meta):
        write_json_atomic(manifest_path, meta)
        logger.info("[capture] re-anchored manifest paths: %s", manifest_path)
    # The FoundationPose recording carries a copy of the same metadata.
    fp_meta_path = packet / "foundationpose_recording" / "metadata.json"
    if fp_meta_path.exists():
        fp_meta = read_json(fp_meta_path)
        if _reanchor(fp_meta):
            write_json_atomic(fp_meta_path, fp_meta)
    return meta


def import_packet(
    bundle: Path,
    name: str,
    capture_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Copy an observation packet into the bundle and re-anchor its manifest.

    Idempotent step contract: if the packet was already imported the existing
    copy is kept (its manifest is cheaply re-anchored in case the bundle
    moved); ``force=True`` replaces the copy from ``capture_dir``.

    Args:
        bundle: Scene-bundle root directory.
        name: Object name (packet subdirectory under ``capture_dir``).
        capture_dir: Directory holding the source packets.
        force: Replace an existing imported packet.

    Returns:
        The imported packet directory ``<bundle>/<name>/capture``.

    Raises:
        FileNotFoundError: If the source packet is missing or incomplete.
    """
    source = Path(capture_dir) / name
    dest = packet_dir_in_bundle(Path(bundle), name)

    if dest.exists() and not force:
        reanchor_manifest(dest)
        _validate_packet(dest)
        logger.info("[capture] %s: already imported at %s", name, dest)
        return dest

    _validate_packet(source)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    reanchor_manifest(dest)
    _validate_packet(dest)
    logger.info("[capture] imported packet %s -> %s", source, dest)
    return dest
