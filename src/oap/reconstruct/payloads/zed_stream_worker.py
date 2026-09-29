#!/usr/bin/env python3
"""Single-owner ZED RGB-D stream for tracking snapshots and motion evidence.

PAYLOAD-STYLE: this file runs under the external observer interpreter and must
not import :mod:`oap`.  It is the only process that opens the fixed ZED
during an executed episode.  Two operations share that one stream:

* ``snapshot`` atomically materializes the latest RGB-D exposure as the normal
  FoundationPose packet for every requested object;
* ``start_recording`` / ``stop_recording`` bracket a real-motion prefix and
  save timestamped RGB evidence from the same open camera.

Protocol (one JSON object per stdin line):

``{"op":"snapshot","out_dir":"...","objects":[["name","prompt"], ...]}``
``{"op":"start_recording","out_dir":"..."}``
``{"op":"stop_recording","checkpoint_out_dir":"...",
   "objects":[["name","prompt"], ...]}``
``{"op":"exit"}``

Every captured exposure has one ``stream_session_id + frame_sequence``
``capture_id``. The post-stop response, video sidecar, and every object packet
repeat that exact identity. Library chatter is redirected to stderr so stdout
remains a protocol channel.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import queue
import sys
import threading
import time
import uuid
from pathlib import Path


def _stdin_pump(out: "queue.Queue[str | None]") -> None:
    try:
        for line in sys.stdin:
            out.put(line)
    finally:
        out.put(None)


def _response(payload: dict) -> None:
    print("ZEDSTREAM " + json.dumps(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-fps", type=float, default=15.0)
    args = parser.parse_args()
    if args.record_fps <= 0.0:
        raise ValueError("--record-fps must be > 0")

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)

    import numpy as np
    import pyzed.sl as sl
    from PIL import Image
    from zed_capture import write_packet

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.2
    init.sdk_verbose = 0
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    commands: "queue.Queue[str | None]" = queue.Queue()
    threading.Thread(target=_stdin_pump, args=(commands,), daemon=True).start()
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    recording_dir: Path | None = None
    recording_timestamps = None
    recording_count = 0
    recording_first_monotonic_s: float | None = None
    recording_last_monotonic_s: float | None = None
    next_record_monotonic_s = 0.0
    record_period_s = 1.0 / float(args.record_fps)
    stream_session_id = uuid.uuid4().hex
    capture_sequence = 0

    def _grab() -> object:
        """Grab one exposure and advance its stream-global identity."""
        nonlocal capture_sequence
        result = cam.grab(runtime)
        if result == sl.ERROR_CODE.SUCCESS:
            capture_sequence += 1
        return result

    def _capture_id() -> str:
        return (
            f"zed:{serial}:session:{stream_session_id}:"
            f"frame:{capture_sequence}"
        )

    def _save_record_frame(rgb, monotonic_s: float) -> None:
        nonlocal recording_count
        nonlocal recording_first_monotonic_s
        nonlocal recording_last_monotonic_s
        if recording_dir is None or recording_timestamps is None:
            return
        Image.fromarray(rgb).save(
            recording_dir / f"frame_{recording_count:05d}.png"
        )
        recording_timestamps.write(f"{monotonic_s:.6f}\n")
        recording_timestamps.flush()
        if recording_first_monotonic_s is None:
            recording_first_monotonic_s = monotonic_s
        recording_last_monotonic_s = monotonic_s
        recording_count += 1

    def _finish_recording(rgb, monotonic_s: float) -> dict:
        nonlocal recording_dir
        nonlocal recording_timestamps
        nonlocal recording_count
        nonlocal recording_first_monotonic_s
        nonlocal recording_last_monotonic_s
        if recording_dir is None or recording_timestamps is None:
            raise RuntimeError("no ZED evidence recording is active")
        if recording_last_monotonic_s is None or (
            monotonic_s > recording_last_monotonic_s + 1e-6
        ):
            _save_record_frame(rgb, monotonic_s)
        recording_timestamps.close()
        payload = {
            "ok": True,
            "op": "stop_recording",
            "out_dir": str(recording_dir),
            "frame_count": int(recording_count),
            "first_monotonic_s": recording_first_monotonic_s,
            "last_monotonic_s": recording_last_monotonic_s,
        }
        recording_dir = None
        recording_timestamps = None
        recording_count = 0
        recording_first_monotonic_s = None
        recording_last_monotonic_s = None
        return payload

    try:
        # Warm up auto-exposure/depth before attesting ownership.
        for _ in range(15):
            if _grab() != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError("ZED warm-up grab failed")
        info = cam.get_camera_information()
        serial = str(info.serial_number)
        calib = info.camera_configuration.calibration_parameters.left_cam
        K = np.array(
            [
                [calib.fx, 0.0, calib.cx],
                [0.0, calib.fy, calib.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        print("ZEDSTREAM READY", flush=True)

        running = True
        while running:
            if _grab() != sl.ERROR_CODE.SUCCESS:
                continue
            capture_unix_s = time.time()
            capture_monotonic_s = time.monotonic()
            capture_id = _capture_id()
            cam.retrieve_image(image, sl.VIEW.LEFT)
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            rgb = image.get_data()[:, :, :3][:, :, ::-1].copy()
            depth_m = depth.get_data().astype(np.float32)
            depth_m[~np.isfinite(depth_m)] = 0.0

            if (
                recording_dir is not None
                and capture_monotonic_s >= next_record_monotonic_s
            ):
                _save_record_frame(rgb, capture_monotonic_s)
                next_record_monotonic_s = (
                    capture_monotonic_s + record_period_s
                )

            while True:
                try:
                    line = commands.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    running = False
                    break
                try:
                    request = json.loads(line)
                    op = str(request.get("op", ""))
                    if op == "snapshot":
                        out_dir = Path(request["out_dir"]).expanduser().resolve()
                        objects = request.get("objects")
                        if not isinstance(objects, list) or not objects:
                            raise ValueError(
                                "snapshot needs a non-empty objects list"
                            )
                        names: list[str] = []
                        shared_frame_packet: Path | None = None
                        # write_packet prints human diagnostics. Keep them off
                        # the JSONL protocol channel.
                        with contextlib.redirect_stdout(sys.stderr):
                            for item in objects:
                                if (
                                    not isinstance(item, list)
                                    or len(item) != 2
                                ):
                                    raise ValueError(
                                        "snapshot object must be [name, prompt]"
                                    )
                                name = str(item[0])
                                names.append(name)
                                write_packet(
                                    out_dir,
                                    name,
                                    rgb,
                                    depth_m,
                                    K,
                                    None,
                                    camera_serial=serial,
                                    capture_unix_s=capture_unix_s,
                                    capture_monotonic_s=(
                                        capture_monotonic_s
                                    ),
                                    capture_id=capture_id,
                                    reuse_frame_from=shared_frame_packet,
                                )
                                if shared_frame_packet is None:
                                    shared_frame_packet = out_dir / name
                        _response(
                            {
                                "ok": True,
                                "op": op,
                                "out_dir": str(out_dir),
                                "objects": names,
                                "capture_unix_s": capture_unix_s,
                                "capture_monotonic_s": capture_monotonic_s,
                                "capture_id": capture_id,
                                "camera_serial": serial,
                            }
                        )
                    elif op == "start_recording":
                        if recording_dir is not None:
                            raise RuntimeError(
                                "a ZED evidence recording is already active"
                            )
                        recording_dir = (
                            Path(request["out_dir"]).expanduser().resolve()
                        )
                        recording_dir.mkdir(parents=True, exist_ok=True)
                        recording_count = 0
                        recording_first_monotonic_s = None
                        recording_last_monotonic_s = None
                        recording_timestamps = (
                            recording_dir / "timestamps.txt"
                        ).open("w", encoding="utf-8")
                        _save_record_frame(rgb, capture_monotonic_s)
                        next_record_monotonic_s = (
                            capture_monotonic_s + record_period_s
                        )
                        (recording_dir / ".recording_live").write_text(
                            "live",
                            encoding="utf-8",
                        )
                        _response(
                            {
                                "ok": True,
                                "op": op,
                                "out_dir": str(recording_dir),
                                "first_monotonic_s":
                                    recording_first_monotonic_s,
                                "camera_serial": serial,
                            }
                        )
                    elif op == "stop_recording":
                        # The frame at the top of this loop may predate the
                        # stop request while that request waited in stdin.
                        # Grab once *after* dequeuing it so evidence provably
                        # brackets the caller's already-finished motion.
                        if _grab() != sl.ERROR_CODE.SUCCESS:
                            raise RuntimeError(
                                "ZED post-recording grab failed"
                            )
                        stop_unix_s = time.time()
                        stop_monotonic_s = time.monotonic()
                        stop_capture_id = _capture_id()
                        cam.retrieve_image(image, sl.VIEW.LEFT)
                        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
                        stop_rgb = (
                            image.get_data()[:, :, :3][:, :, ::-1].copy()
                        )
                        stop_depth_m = depth.get_data().astype(np.float32)
                        stop_depth_m[
                            ~np.isfinite(stop_depth_m)
                        ] = 0.0
                        packet_dir_raw = request.get(
                            "checkpoint_out_dir"
                        )
                        objects = request.get("objects")
                        packet_dir = (
                            None
                            if packet_dir_raw is None
                            else Path(packet_dir_raw).expanduser().resolve()
                        )
                        names: list[str] = []
                        shared_frame_packet = None
                        if packet_dir is not None:
                            if not isinstance(objects, list) or not objects:
                                raise ValueError(
                                    "checkpoint packet needs a non-empty "
                                    "objects list"
                                )
                            with contextlib.redirect_stdout(sys.stderr):
                                for item in objects:
                                    if (
                                        not isinstance(item, list)
                                        or len(item) != 2
                                    ):
                                        raise ValueError(
                                            "checkpoint object must be "
                                            "[name, prompt]"
                                        )
                                    name = str(item[0])
                                    names.append(name)
                                    write_packet(
                                        packet_dir,
                                        name,
                                        stop_rgb,
                                        stop_depth_m,
                                        K,
                                        None,
                                        camera_serial=serial,
                                        capture_unix_s=stop_unix_s,
                                        capture_monotonic_s=(
                                            stop_monotonic_s
                                        ),
                                        capture_id=stop_capture_id,
                                        reuse_frame_from=(
                                            shared_frame_packet
                                        ),
                                    )
                                    if shared_frame_packet is None:
                                        shared_frame_packet = (
                                            packet_dir / name
                                        )
                        stop_rgb_sha256 = hashlib.sha256(
                            np.ascontiguousarray(stop_rgb).tobytes()
                        ).hexdigest()
                        stop_depth_sha256 = hashlib.sha256(
                            np.ascontiguousarray(
                                stop_depth_m,
                                dtype=np.float32,
                            ).tobytes()
                        ).hexdigest()
                        result = _finish_recording(
                            stop_rgb,
                            stop_monotonic_s,
                        )
                        evidence_dir = Path(result["out_dir"])
                        final_frame = (
                            evidence_dir
                            / f"frame_{int(result['frame_count']) - 1:05d}.png"
                        )
                        sidecar = evidence_dir / "post_stop_capture.json"
                        sidecar_payload = {
                            "schema": (
                                "oap_zed_post_stop_capture_v1"
                            ),
                            "capture_id": stop_capture_id,
                            "capture_unix_s": stop_unix_s,
                            "capture_monotonic_s": stop_monotonic_s,
                            "camera_serial": serial,
                            "stream_session_id": stream_session_id,
                            "frame_sequence": capture_sequence,
                            "video_last_frame": str(final_frame),
                            "video_last_frame_sha256": hashlib.sha256(
                                final_frame.read_bytes()
                            ).hexdigest(),
                            "rgb_array_sha256": stop_rgb_sha256,
                            "depth_m_array_sha256": stop_depth_sha256,
                            "checkpoint_out_dir": (
                                None
                                if packet_dir is None
                                else str(packet_dir)
                            ),
                            "checkpoint_objects": names,
                        }
                        sidecar.write_text(
                            json.dumps(sidecar_payload, indent=2),
                            encoding="utf-8",
                        )
                        result.update(
                            {
                                "checkpoint_out_dir": (
                                    None
                                    if packet_dir is None
                                    else str(packet_dir)
                                ),
                                "checkpoint_objects": names,
                                "capture_id": stop_capture_id,
                                "post_stop_capture_unix_s": stop_unix_s,
                                "post_stop_capture_monotonic_s":
                                    stop_monotonic_s,
                                "post_stop_rgb_sha256": stop_rgb_sha256,
                                "post_stop_depth_m_sha256":
                                    stop_depth_sha256,
                                "post_stop_sidecar": str(sidecar),
                                "camera_serial": serial,
                            }
                        )
                        _response(result)
                    elif op == "exit":
                        if recording_dir is not None:
                            _finish_recording(rgb, capture_monotonic_s)
                        _response({"ok": True, "op": op})
                        running = False
                        break
                    else:
                        raise ValueError(f"unknown ZED stream op {op!r}")
                except Exception as exc:  # noqa: BLE001 - report and keep serving
                    logging.exception("ZED stream request failed")
                    _response(
                        {
                            "ok": False,
                            "error": repr(exc)[:500],
                        }
                    )
    finally:
        if recording_timestamps is not None:
            recording_timestamps.close()
        cam.close()


if __name__ == "__main__":
    main()
