#!/usr/bin/env python3
"""Capture a real ZED 2i RGB-D frame and SAM3 text-prompted object masks.

Stage-1 CAPTURE payload: runs under the OBSERVER env interpreter (pyzed +
SAM3; lab host only) and must not import ``oap``. Produces, per requested
object, an observation packet matching the schema every downstream
reconstruction step consumes::

  <out>/<name>/
    rgb.png  depth_m.npy  depth_mm.png  mask_00.png  overlay.png  cam_K.txt
    sam31_mask_packet/{input.png, mask_00.png, overlay.png}
    foundationpose_recording/{rgb,depth,masks}/frame_0000.png + cam_K.txt + metadata.json
    observation_manifest.json     (schema real_zed2i_rgbd_observation_v1)

T_base_camera is recorded as identity with provenance "camera_frame_only":
reconstruction and metric-scale recovery are invariant to a rigid base
transform; the calibrated hand-eye transform is applied later by the POSE
step from the packaged calibration YAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _link_or_copy(source: Path, destination: Path) -> None:
    """Reuse an already encoded frame in the FoundationPose packet.

    ``rgb.png`` and ``depth_mm.png`` are byte-identical to FoundationPose's
    frame-0000 inputs.  Encoding both copies separately dominated the measured
    maskless tracking loop.  A hard link preserves the exact packet layout and
    bytes without doing a second PNG encode or consuming a second copy on disk.
    Filesystems without hard-link support retain the old behaviour via copy.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def capture_zed_frame(settle_frames: int = 15):
    import pyzed.sl as sl

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.2
    # NEURAL requires TensorRT, which is NOT installed on the lab host (verified: no
    # `tensorrt` module / libnvinfer; a failed NEURAL open also corrupts the process
    # and segfaults any retry). ULTRA is version-stable and sufficient for object-scale
    # recovery at tabletop range. The ZED SDK prints a cosmetic "ULTRA is deprecated;
    # use NEURAL" WARNING (+ an INFO banner) at open() -- silence the SDK console
    # (sdk_verbose=0) so it stops cluttering the runner log. To actually switch to
    # NEURAL for better depth, install TensorRT first, then set depth_mode = NEURAL.
    init.sdk_verbose = 0
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    status = cam.open(init)
    print("[zed] depth mode: ULTRA")
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    try:
        runtime = sl.RuntimeParameters()
        img = sl.Mat()
        depth = sl.Mat()
        # Let auto-exposure settle, then keep the last grab.
        for _ in range(settle_frames):
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("ZED grab failed")
        capture_unix_s = time.time()
        cam.retrieve_image(img, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        rgb = img.get_data()[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
        depth_m = depth.get_data().astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0
        camera_info = cam.get_camera_information()
        camera_serial = str(camera_info.serial_number)
        calib = camera_info.camera_configuration.calibration_parameters.left_cam
        K = np.array([[calib.fx, 0.0, calib.cx], [0.0, calib.fy, calib.cy], [0.0, 0.0, 1.0]], dtype=float)
        return rgb, depth_m, K, capture_unix_s, camera_serial
    finally:
        cam.close()


def segment(rgb: np.ndarray, prompts: list[tuple[str, str]], sam3_service_dir: Path):
    sys.path.insert(0, str(sam3_service_dir))
    from serve_hf_sam3 import SAM3Server

    server = SAM3Server(
        model_name="facebook/sam3",
        device="auto",
        dtype="auto",
        threshold=0.4,
        mask_threshold=0.5,
        max_detections_per_prompt=3,
    )
    server.load()
    image_pil = Image.fromarray(rgb, mode="RGB")
    out: dict[str, dict] = {}
    for name, text in prompts:
        records = server._segment_prompt(image_pil, text)
        if not records:
            raise RuntimeError(f"SAM3 found nothing for prompt {text!r}")
        best = records[0]
        out[name] = {"mask": best["mask"].astype(bool), "score": best["score"], "bbox": best["bbox"], "text": text}
        print(f"[sam3] {name}: score={best['score']:.3f} bbox={best['bbox']} pixels={int(best['mask'].sum())}")
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    return overlay


def write_mask_group(out: Path, rgb: np.ndarray, det: dict) -> dict:
    """Write a packet's SAM3 mask artifacts and return the fields describing them.

    The mask FILES and the six mask-conditional manifest fields are ONE atomic
    group produced by ONE detection: all six null (a maskless ``--no-seg``
    track packet) or all six populated from the SAME ``det``. A populated
    ``mask`` beside a null ``sam3_score`` would describe a mask that no prompt
    produced, so one function writes both halves and they cannot drift --
    splitting them is exactly how the lazy pass shipped writing mask pixels
    next to a manifest still saying ``mask: null``.

    Called at capture time by :func:`write_packet` and lazily afterwards by
    :func:`update_packet_mask` (``--seg-only``): a maskless track packet whose
    cheap point-cloud drift gate tripped gets its SAM3 pass ON THE
    ALREADY-CAPTURED FRAME (same pixels the tracker saw -- no second camera
    shot, no pose inconsistency), so the follow-up global register has the
    ob_mask it requires.

    Args:
        out: The packet directory.
        rgb: The frame the mask was segmented from (overlays + sam31 input).
        det: One :func:`segment` record -- ``mask``/``score``/``bbox``/``text``.

    Returns:
        The six mask-conditional manifest fields, ready to ``dict.update``
        into the packet manifest.
    """
    mask = det["mask"]
    Image.fromarray((mask.astype(np.uint8) * 255)).save(out / "mask_00.png")
    Image.fromarray(overlay_mask(rgb, mask)).save(out / "overlay.png")
    sam31 = out / "sam31_mask_packet"
    sam31.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(sam31 / "input.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(sam31 / "mask_00.png")
    Image.fromarray(overlay_mask(rgb, mask)).save(sam31 / "overlay.png")
    fp_masks = out / "foundationpose_recording" / "masks"
    fp_masks.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(fp_masks / "frame_0000.png")
    return {
        "sam3_prompt": det["text"],
        "sam3_score": float(det["score"]),
        "sam3_bbox": [int(v) for v in det["bbox"]],
        "mask": str(out / "mask_00.png"),
        "sam31_mask_packet": str(sam31),
        "mask_pixels": int(mask.sum()),
    }


def clear_mask_group(out: Path) -> None:
    """Delete any mask artifacts left in a packet directory being recaptured.

    ``--out`` is a user-supplied episode directory and observe tags repeat
    across runs, so a ``--no-seg`` capture can land on a packet dir a PREVIOUS
    run's ``--seg-only`` pass left a mask in. The FoundationPose worker decides
    mask availability by file existence (``masks/frame_0000.png``), so a
    leftover mask silently gates this frame's pose against the previous run's
    silhouette instead of taking the maskless point-cloud branch.
    """
    for rel in ("mask_00.png", "overlay.png",
                "foundationpose_recording/masks/frame_0000.png"):
        (out / rel).unlink(missing_ok=True)
    shutil.rmtree(out / "sam31_mask_packet", ignore_errors=True)
    # ...including the now-empty masks/ dir write_mask_group mkdir'd, so a
    # recaptured maskless packet is shaped exactly like a fresh one.
    shutil.rmtree(out / "foundationpose_recording" / "masks", ignore_errors=True)


def write_manifests(out: Path, meta: dict) -> None:
    """Write ``observation_manifest.json`` AND its FoundationPose twin.

    The FP recording carries a copy of the same metadata under a leading
    ``format`` key, and the CAPTURE step re-anchors both together
    (``reconstruct/capture.py``), so every writer writes both: updating one
    alone only moves a stale copy one directory down, and the twin is what the
    correspondences payload reads for the extrinsic.

    Plain ``write_text``, not a tmp+rename: payloads must not import
    ``oap.utils.io.write_json_atomic`` (the payload rule) and every
    JSON-writing payload here uses this idiom. Crash consistency comes from
    write ORDER instead -- see :func:`update_packet_mask`.
    """
    (out / "observation_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    fp = out / "foundationpose_recording"
    fp.mkdir(parents=True, exist_ok=True)
    (fp / "metadata.json").write_text(
        json.dumps({"format": "foundationpose_rgbd_recording", **meta}, indent=2),
        encoding="utf-8")


def update_packet_mask(out: Path, rgb: np.ndarray, det: dict) -> dict:
    """Add a SAM3 mask to an ALREADY-CAPTURED packet (the ``--seg-only`` pass).

    The one in-place upgrade this schema allows, MASKLESS -> MASK-BEARING: it
    adds the mask artifacts and the six fields describing them and touches
    nothing else -- not the frame, not the intrinsics, not the extrinsic.

    Unconditional overwrite, never fill-if-null: :func:`write_mask_group`
    replaces the mask pixels every time it runs, so a preserved older
    ``sam3_score``/``sam3_bbox`` would describe a mask that no longer exists --
    a subtler lie than the honest null it replaces. Re-running with the same
    detection is byte-idempotent.

    Write ORDER is load-bearing: pixels first, manifests last, so an
    interrupted upgrade leaves the packet UNDER-claiming -- mask files on
    disk, fields still null. The consumer on this path tolerates that: the
    FoundationPose worker detects masks by file existence. Not every consumer
    does -- the scale step resolves the mask from the manifest FIELD and
    raises "observation manifest must contain mask and depth_m_npy" -- but
    under-claiming is still the better tear, because that is a named error
    about a null field, where the reverse order points the manifest at a mask
    that does not exist and the same step dies on a missing file instead.

    Returns:
        The updated manifest payload.
    """
    meta = json.loads((out / "observation_manifest.json").read_text(encoding="utf-8"))
    meta.update(write_mask_group(out, rgb, det))
    write_manifests(out, meta)
    return meta


def write_packet(
    out_dir: Path,
    name: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    det: dict | None,
    *,
    camera_serial: str | int,
    capture_unix_s: float | None = None,
    capture_monotonic_s: float | None = None,
    capture_id: str | None = None,
    reuse_frame_from: Path | None = None,
) -> None:
    """Write one packet from one synchronized RGB-D exposure.

    ``reuse_frame_from`` may name an already-written packet for another object
    in the *same* exposure.  RGB, depth, and intrinsics are exposure-level
    data, so the multi-object ZED stream hard-links those exact bytes instead
    of PNG-encoding and ``np.save``-ing them once per object.  The source
    manifest is checked against this call's capture id and array digests
    before reuse; object-specific masks and manifests remain independent.
    """
    serial = str(camera_serial).strip()
    if serial.upper().startswith("SN"):
        serial = serial[2:].strip()
    if not serial or not serial.isdecimal():
        raise ValueError(
            f"camera_serial must contain decimal digits, got {camera_serial!r}"
        )
    serial = serial.lstrip("0") or "0"
    captured_at = float(
        time.time() if capture_unix_s is None else capture_unix_s
    )
    if not np.isfinite(captured_at) or captured_at <= 0.0:
        raise ValueError("capture_unix_s must be a finite positive timestamp")
    captured_monotonic = (
        None
        if capture_monotonic_s is None
        else float(capture_monotonic_s)
    )
    if captured_monotonic is not None and (
        not np.isfinite(captured_monotonic) or captured_monotonic <= 0.0
    ):
        raise ValueError(
            "capture_monotonic_s must be a finite positive timestamp"
        )
    exposure_id = (
        f"zed:{serial}:standalone:{captured_at:.9f}"
        if capture_id is None
        else str(capture_id).strip()
    )
    if not exposure_id or any(ch.isspace() for ch in exposure_id):
        raise ValueError("capture_id must be a non-empty token")
    rgb_bytes = np.ascontiguousarray(rgb).tobytes()
    depth_bytes = np.ascontiguousarray(
        depth_m, dtype=np.float32
    ).tobytes()
    rgb_sha256 = hashlib.sha256(rgb_bytes).hexdigest()
    depth_sha256 = hashlib.sha256(depth_bytes).hexdigest()
    out = out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    rgb_path = out / "rgb.png"
    depth_npy = out / "depth_m.npy"
    depth_mm_path = out / "depth_mm.png"
    cam_k_path = out / "cam_K.txt"
    if reuse_frame_from is None:
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_npy, depth_m)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_mm_path)
        np.savetxt(cam_k_path, K, fmt="%.10f")
    else:
        source = Path(reuse_frame_from)
        source_manifest = json.loads(
            (source / "observation_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if source_manifest.get("capture_id") != exposure_id:
            raise ValueError(
                "reuse_frame_from belongs to a different capture_id"
            )
        if source_manifest.get("rgb_array_sha256") != rgb_sha256:
            raise ValueError("reuse_frame_from RGB bytes do not match")
        if source_manifest.get("depth_m_array_sha256") != depth_sha256:
            raise ValueError("reuse_frame_from depth bytes do not match")
        if not np.allclose(
            np.asarray(source_manifest.get("cam_K"), dtype=float),
            np.asarray(K, dtype=float),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("reuse_frame_from intrinsics do not match")
        for filename, destination in (
            ("rgb.png", rgb_path),
            ("depth_m.npy", depth_npy),
            ("depth_mm.png", depth_mm_path),
            ("cam_K.txt", cam_k_path),
        ):
            _link_or_copy(source / filename, destination)

    fp = out / "foundationpose_recording"
    (fp / "rgb").mkdir(parents=True, exist_ok=True)
    (fp / "depth").mkdir(parents=True, exist_ok=True)
    _link_or_copy(rgb_path, fp / "rgb" / "frame_0000.png")
    _link_or_copy(depth_mm_path, fp / "depth" / "frame_0000.png")
    np.savetxt(fp / "cam_K.txt", K, fmt="%.10f")

    # The MASKLESS (--no-seg) packet is the base document: the six
    # mask-conditional fields are declared null here, once, in schema order,
    # and write_mask_group fills exactly those six when a detection exists
    # (dict.update keeps each key in its place). The --seg-only upgrade fills
    # the same six through the same writer, so the two cannot disagree.
    meta = {
        "schema": "real_zed2i_rgbd_observation_v1",
        "capture_id": exposure_id,
        "capture_unix_s": captured_at,
        "capture_monotonic_s": captured_monotonic,
        "rgb_array_sha256": rgb_sha256,
        "depth_m_array_sha256": depth_sha256,
        "object_name": name,
        "sam3_prompt": None,
        "sam3_score": None,
        "sam3_bbox": None,
        "rgb": str(rgb_path),
        "depth_m_npy": str(depth_npy),
        "depth_mm_png": str(depth_mm_path),
        "mask": None,
        "sam31_mask_packet": None,
        "foundationpose_recording": str(fp),
        "cam_K": K.tolist(),
        "T_base_camera_opencv": np.eye(4).tolist(),
        "frame_provenance": "camera_frame_only: T_base_camera is identity; metric scale is frame-invariant; hand-eye calibration applied by the pose step",
        "mask_pixels": None,
        "camera_serial": serial,
        "camera": "ZED 2i HD720 left eye, ULTRA depth, real capture",
        "policy": "Real RGB-D observation; reconstruction must not use any privileged object model.",
    }
    if det is not None:
        meta.update(write_mask_group(out, rgb, det))
    else:
        clear_mask_group(out)
    write_manifests(out, meta)
    print(f"[packet] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--object", nargs=2, action="append", metavar=("NAME", "PROMPT"), required=True)
    parser.add_argument(
        "--sam3-service-dir",
        type=Path,
        default=Path.home() / "countersim-current" / "services" / "sam3_service",
        help="directory containing serve_hf_sam3.py (the SAM3 HF wrapper)",
    )
    parser.add_argument("--settle-frames", type=int, default=15)
    parser.add_argument("--no-seg", action="store_true",
                        help="capture RGB-D only; skip SAM3 (a maskless track "
                             "packet -- the worker's point-cloud gate covers drift)")
    parser.add_argument("--seg-only", action="store_true",
                        help="run SAM3 on an EXISTING packet's rgb.png (lazy "
                             "mask for a register after the cheap gate tripped); "
                             "no camera access")
    args = parser.parse_args()
    if args.seg_only:
        for name, prompt in args.object:
            out = args.out / name
            rgb = np.asarray(Image.open(out / "rgb.png").convert("RGB"))
            det = segment(rgb, [(name, prompt)], args.sam3_service_dir)[name]
            update_packet_mask(out, rgb, det)
            print(f"[seg-only] {out}")
        return
    rgb, depth_m, K, capture_unix_s, camera_serial = capture_zed_frame(
        settle_frames=int(args.settle_frames)
    )
    valid = depth_m[depth_m > 0]
    print(f"[zed] rgb {rgb.shape}, depth valid {valid.size}/{depth_m.size}, median {np.median(valid):.3f} m")
    if args.no_seg:
        for name, _prompt in args.object:
            write_packet(
                args.out,
                name,
                rgb,
                depth_m,
                K,
                None,
                camera_serial=camera_serial,
                capture_unix_s=capture_unix_s,
            )
        return
    detections = segment(rgb, [(n, p) for n, p in args.object], args.sam3_service_dir)
    for name, det in detections.items():
        write_packet(
            args.out,
            name,
            rgb,
            depth_m,
            K,
            det,
            camera_serial=camera_serial,
            capture_unix_s=capture_unix_s,
        )


if __name__ == "__main__":
    main()
