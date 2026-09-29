"""Video evidence helpers: frame annotation, mp4 encode, side-by-side stitching.

Role in the two-stage pipeline: the closed loop's standing directive is to
ALWAYS save the real and sim closed-loop footage. Per chunk the loop records a
real ZED clip and a twin replay; this module composites them into the
per-chunk ``chunk_XX_real_sim.mp4`` and stitches the FULL session into
``full_real.mp4`` / ``full_sim.mp4`` / ``full_real_sim.mp4``. The twin rollout
machinery also imports :func:`annotate_single` / :func:`write_mp4` from here
(the canonical implementations).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("oap.utils.video")

__all__ = [
    "VIDEO_FPS",
    "annotate_single",
    "latest_frame",
    "side_by_side_video",
    "sorted_frames",
    "stitch_full_session_videos",
    "write_mp4",
]

# Encode rate for ALL episode evidence videos. Playback is ORIGINAL SPEED by
# construction: real frames are resampled onto this grid by their MEASURED
# capture timestamps (the recorder's true rate drifts below its nominal fps
# with PNG-encode time -- encoding at the nominal rate played everything in
# slow motion), and sim frames are resampled to the chunk's measured executed
# duration. Demo/paper videos must be true-speed or carry an explicit speed
# label; this pipeline produces true speed, so no label is needed.
VIDEO_FPS = 15.0


def annotate_single(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    """Annotate a rendered frame with a title/subtitle banner (returns a copy)."""
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 48), fill=(244, 246, 242))
    draw.text((8, 8), title, fill=(10, 10, 10), font=font)
    draw.text((8, 28), subtitle, fill=(35, 35, 35), font=font)
    return np.asarray(canvas)


def write_mp4(frame_paths: list[Path], video_path: Path, *, fps: float) -> None:
    """Encode an ordered list of frame PNGs into an mp4 (imageio/ffmpeg).

    Raises when the output is missing/empty -- a silently absent evidence
    video must fail loudly, not surface at review time.
    """
    import imageio.v2 as iio

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(str(video_path), fps=float(fps)) as writer:
        for p in frame_paths:
            # imageio's v2 legacy API annotates the writer as the base class,
            # which lacks append_data; the runtime object is a Format.Writer.
            writer.append_data(iio.imread(str(p)))  # type: ignore[attr-defined]
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write video: {video_path}")


# --------------------------------------------------------------- frame grabs
def sorted_frames(directory: Path, pattern: str = "*.png") -> list[Path]:
    """Return a directory's frames in name order ([] when absent)."""
    d = Path(directory)
    return sorted(d.glob(pattern)) if d.exists() else []


def latest_frame(directory: Path, pattern: str = "frame_*.png") -> Path | None:
    """Return the LAST recorded frame of a clip (the judge's input), or None."""
    frames = sorted_frames(directory, pattern)
    return frames[-1] if frames else None


# --------------------------------------------------------- time resampling
def load_frame_timestamps(directory: Path) -> list[float] | None:
    """Per-frame capture timestamps the recorder saved (None when absent)."""
    p = Path(directory) / "timestamps.txt"
    if not p.exists():
        return None
    ts = [float(line) for line in p.read_text(encoding="utf-8").split()]
    return ts or None


def resample_real_frames(real_dir: Path, fps: float) -> list[Path]:
    """Real frames resampled onto a uniform ``fps`` grid by MEASURED capture
    times, so encoding at ``fps`` plays at original speed. Without timestamps
    (older recordings) the frames pass through unresampled -- the caller's fps
    is then only nominal."""
    real_paths = sorted(Path(real_dir).glob("frame_*.png")) if Path(real_dir).exists() else []
    ts = load_frame_timestamps(real_dir)
    if not real_paths:
        return real_paths
    if ts is None or len(ts) != len(real_paths):
        logger.warning("[video] %s: no usable per-frame timestamps (%s) -- "
                       "frames pass through at NOMINAL speed, playback is not "
                       "measured-original", real_dir,
                       "missing" if ts is None else f"{len(ts)} ts vs {len(real_paths)} frames")
        return real_paths
    # Sanitize: a wall-clock step (NTP) or a corrupt tail line must degrade,
    # never crash the evidence assembly.
    ts = list(np.maximum.accumulate(np.asarray(ts, float)))
    t0, t_last = ts[0], ts[-1]
    n_out = max(1, int(np.floor((t_last - t0) * fps)) + 1)
    out: list[Path] = []
    j = 0
    for i in range(n_out):
        t = t0 + i / fps
        while j + 1 < len(ts) and ts[j + 1] <= t:
            j += 1
        out.append(real_paths[j])
    # The last captured frame is the chunk's END-STATE evidence -- never let a
    # sub-tick remainder drop it.
    if out[-1] is not real_paths[-1]:
        out.append(real_paths[-1])
    return out


def resample_sim_frames(sim_paths: list[Path], duration_s: float | None,
                        fps: float, *, sim_fps_nominal: float = 12.0) -> list[Path]:
    """Sim replay frames resampled onto a uniform ``fps`` grid spanning the
    chunk's MEASURED executed duration (uniform time-warp: the replay depicts
    the same motion the robot ran, endpoint-aligned). Falls back to the
    replay's nominal timing when no execution happened (dry/sim-only runs)."""
    n = len(sim_paths)
    if n == 0:
        return []
    if duration_s is None or duration_s <= 0:
        duration_s = n / float(sim_fps_nominal)
    n_out = max(1, int(round(duration_s * fps)))
    if n_out == 1:
        # Degenerate (instant-stopped chunk): the end state is the evidence.
        return [sim_paths[-1]]
    return [sim_paths[min(n - 1, int(round(i / (n_out - 1) * (n - 1))))]
            for i in range(n_out)]


# ------------------------------------------------------------- side-by-side
def side_by_side_video(real_dir: Path, sim_paths: list[Path], out_mp4: Path,
                       fps: float = VIDEO_FPS, *,
                       executed_duration_s: float | None = None,
                       sim_fps_nominal: float = 12.0) -> None:
    """Composite real|sim frames into one 1280x360 side-by-side mp4.

    Both halves are resampled onto the same ``fps`` timeline first (real by
    measured capture timestamps, sim by the measured executed duration), so
    the output plays at ORIGINAL SPEED and the two halves stay time-aligned.
    Each side is held on its last frame when the other runs longer. A chunk
    with no frames at all writes nothing (dry chunks are legitimate).
    """
    from PIL import Image

    real_seq = resample_real_frames(real_dir, fps)
    sim_seq = resample_sim_frames(sim_paths, executed_duration_s, fps,
                                  sim_fps_nominal=sim_fps_nominal)
    n = max(len(real_seq), len(sim_seq))
    if n == 0:
        return
    combo_dir = Path(out_mp4).parent / (Path(out_mp4).stem + "_frames")
    combo_dir.mkdir(parents=True, exist_ok=True)
    combos = []
    cache: dict[Path, object] = {}
    for i in range(n):
        rp = real_seq[min(i, len(real_seq) - 1)] if real_seq else None
        sp = sim_seq[min(i, len(sim_seq) - 1)] if sim_seq else None
        real = cache.get(rp) if rp else None
        if rp and real is None:
            cache[rp] = real = Image.open(rp).resize((640, 360))
        sim = cache.get(sp) if sp else None
        if sp and sim is None:
            cache[sp] = sim = Image.open(sp).resize((640, 360))
        c = Image.new("RGB", (1280, 360))
        c.paste(real if real is not None else Image.new("RGB", (640, 360)), (0, 0))
        c.paste(sim if sim is not None else Image.new("RGB", (640, 360)), (640, 0))
        p = combo_dir / f"frame_{i:05d}.png"
        c.save(p)
        combos.append(p)
        if len(cache) > 64:
            cache.clear()
    write_mp4(combos, out_mp4, fps=fps)


def stitch_full_session_videos(output_dir: Path, fps: float = VIDEO_FPS, *,
                               chunk_durations: dict[int, float] | None = None,
                               sim_fps_nominal: float = 12.0) -> dict[str, Path]:
    """Stitch the FULL closed-loop footage across all chunks (always-save rule).

    Reads the per-chunk frame directories the loop wrote
    (``sim_video/chunk_*``, ``real_video/chunk_*``, and the per-chunk
    side-by-side combo frames) and writes ``full_sim.mp4`` / ``full_real.mp4``
    / ``full_real_sim.mp4`` next to them. Each chunk's frames are resampled to
    original speed first (real by capture timestamps, sim by the executed
    durations in ``chunk_durations``); inter-chunk planning gaps are hard cuts.
    Returns the videos actually written (a stream with zero frames is skipped,
    never an empty file).
    """
    output_dir = Path(output_dir)
    written: dict[str, Path] = {}
    durations = chunk_durations or {}

    def _chunk_idx(d: Path) -> int | None:
        try:
            return int(d.name.rsplit("_", 1)[-1])
        except ValueError:
            return None

    sim_all: list[Path] = []
    for d in sorted((output_dir / "sim_video").glob("chunk_*")):
        sim_all.extend(resample_sim_frames(
            sorted(d.glob("*.png")), durations.get(_chunk_idx(d)), fps,
            sim_fps_nominal=sim_fps_nominal))
    real_all: list[Path] = []
    for d in sorted((output_dir / "real_video").glob("chunk_*")):
        real_all.extend(resample_real_frames(d, fps))
    if sim_all:
        out = output_dir / "full_sim.mp4"
        write_mp4(sim_all, out, fps=fps)
        written["full_sim"] = out
    if real_all:
        out = output_dir / "full_real.mp4"
        write_mp4(real_all, out, fps=fps)
        written["full_real"] = out
    # Combo frames were already composed on the fps timeline by
    # side_by_side_video -- concatenate as-is.
    combo_all: list[Path] = []
    for f in sorted(output_dir.glob("chunk_*_real_sim_frames")):
        combo_all.extend(sorted(f.glob("*.png")))
    if combo_all:
        out = output_dir / "full_real_sim.mp4"
        write_mp4(combo_all, out, fps=fps)
        written["full_real_sim"] = out
    logger.info("[video] full-session footage (original speed, %.0f fps): %s",
                fps, {k: str(v) for k, v in written.items()} or "none (no frames)")
    return written
