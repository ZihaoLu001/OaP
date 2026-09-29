#!/usr/bin/env python3
"""Record the real ZED 2i left-eye stream to frames for a fixed duration.

The real half of the episode evidence video: the loop starts this payload
right before commanding a real action chunk (``loop.execute.start_zed_recording``)
and stitches the frames into the per-chunk and episode videos afterwards.
Runs under the EXTERNAL observer python (pyzed); imports nothing from oap.

Contract with the caller: flags are ``--out-dir --duration-s --fps``, and the
``.recording_live`` flag file is written in ``--out-dir`` only once frames are
actually streaming -- the caller holds the robot still until it appears, so
the clip starts at the first frame of real motion, not inside the ~1-2 s
camera-open gap.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import pyzed.sl as sl

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NONE  # RGB only: cheap + steady frame rate
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    try:
        runtime = sl.RuntimeParameters()
        img = sl.Mat()
        period = 1.0 / args.fps
        # Warm up the stream and SIGNAL LIVE before timing starts (see the
        # module docstring: the recording window must align with real motion).
        for _ in range(200):
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                break
        # Per-frame wall-clock timestamps: PNG-encode time makes the true
        # capture rate drift below the target fps, so the encoder must NOT
        # assume the nominal rate -- original-speed playback needs the
        # measured times. One float per line, flushed per frame (a killed
        # recorder still leaves timestamps for every frame it saved).
        with open(args.out_dir / "timestamps.txt", "w", encoding="utf-8") as ts:
            # Save and timestamp one actual frame before the LIVE attestation.
            # The controller starts motion only after seeing the flag, so this
            # gives the coverage checker a frame that provably precedes motion.
            cam.retrieve_image(img, sl.VIEW.LEFT)
            rgb = img.get_data()[:, :, :3][:, :, ::-1]
            Image.fromarray(rgb).save(args.out_dir / "frame_00000.png")
            ts.write(f"{time.monotonic():.6f}\n")
            ts.flush()
            (args.out_dir / ".recording_live").write_text("live")
            t_end = time.time() + args.duration_s
            i = 1
            next_t = time.time()
            while time.time() < t_end:
                if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                now = time.time()
                if now < next_t:
                    continue
                next_t = now + period
                cam.retrieve_image(img, sl.VIEW.LEFT)
                rgb = img.get_data()[:, :, :3][:, :, ::-1]
                Image.fromarray(rgb).save(args.out_dir / f"frame_{i:05d}.png")
                # Monotonic: only relative spacing matters downstream, and a
                # wall-clock (NTP) step must not corrupt the timeline.
                ts.write(f"{time.monotonic():.6f}\n")
                ts.flush()
                i += 1
        print(f"recorded {i} frames -> {args.out_dir}")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
