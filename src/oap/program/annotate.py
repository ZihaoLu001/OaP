"""Render the anchor-annotated observation the synthesis VLM is shown.

Program synthesis references the scene through anchor ids. This module
projects every anchor of an :class:`~oap.program.anchors.AnchorSet` into
the planning-start RGB frame using the touch-validated hand-eye calibration
and draws it under ITS EXACT ID: the ids in the image match the ids in the
text list character-for-character, so the program's anchor references are
grounded in both modalities at once.

Conventions: base frame is the robot (flexiv_world) frame; the packaged
``T_base_zed2i.yaml`` stores ``p_base = T_base_zed2i @ p_camera`` with the
camera in the OpenCV convention (+z forward), so projection uses its inverse.
Intrinsics come from the SAME observation that produced the frame (each
``fp_obs/<tag>/<obj>/cam_K.txt``) -- they follow the ZED resolution and must
never be hardcoded.

Anchors that fall outside the image or behind the camera are NEVER silently
dropped: they are listed in an on-image margin note as ``id (off-screen)``,
because a program may legitimately reference an anchor the camera cannot see
and the VLM must know the id exists.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oap.utils.io import load_yaml, package_config_path
from oap.utils.se3 import validate_se3_matrix

from .anchors import AnchorSet

__all__ = [
    "AnnotatedScene",
    "load_base_camera_extrinsic",
    "project_base_points",
    "render_anchor_annotations",
]

# Longest image side sent to the VLM (token budget); the raw frame is archived
# at full resolution alongside.
MAX_SIDE_PX = 1024
# Drawn length of an anchor's axis arrow, metres in the base frame.
AXIS_ARROW_M = 0.06
# High-contrast label palette (cycled); every color is paired with a black
# outline so it reads on both the pale table and the dark gripper.
_PALETTE = (
    (255, 225, 25), (0, 230, 230), (245, 130, 48), (60, 255, 60),
    (240, 50, 230), (255, 105, 180), (0, 130, 255), (230, 25, 75),
)


@dataclass(frozen=True)
class AnnotatedScene:
    """The rendered evidence pair + which anchors made it into the frame."""

    annotated_path: Path
    raw_path: Path
    visible: tuple[str, ...]
    off_screen: tuple[str, ...]


def load_base_camera_extrinsic(path: Path | None = None) -> np.ndarray:
    """Load ``T_base_camera`` (4x4) from the packaged touch-validated yaml."""
    p = Path(path) if path is not None else package_config_path(
        "calibration/T_base_zed2i.yaml")
    doc = load_yaml(p)
    try:
        matrix = doc["T_base_zed2i"]["matrix"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{p} has no T_base_zed2i.matrix calibration"
        ) from exc
    return validate_se3_matrix(
        matrix,
        name=f"base<-camera calibration {p}",
    )


def project_base_points(points_base: np.ndarray, t_base_camera: np.ndarray,
                        cam_K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points into pixels.

    Returns ``(uv [N,2], z_cam [N])``; a point with ``z_cam <= 0`` is behind
    the camera and its uv is meaningless (callers must treat it off-screen).
    """
    pts = np.atleast_2d(np.asarray(points_base, dtype=float))
    T_cam_base = np.linalg.inv(np.asarray(t_base_camera, dtype=float))
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    p_cam = (T_cam_base @ homo.T).T[:, :3]
    K = np.asarray(cam_K, dtype=float).reshape(3, 3)
    z = p_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * p_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / safe_z + K[1, 2]
    return np.stack([u, v], axis=1), z


def _load_font(size: int):
    from PIL import ImageFont
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_anchor_annotations(
    rgb_path: Path,
    anchors: AnchorSet,
    cam_K: np.ndarray,
    t_base_camera: np.ndarray | None = None,
    *,
    out_dir: Path,
    max_side: int = MAX_SIDE_PX,
) -> AnnotatedScene:
    """Draw every anchor onto the observation frame and archive the pair.

    Writes ``anchors_annotated.png`` (resized to ``max_side``) and
    ``observation_raw.png`` (byte-copy of the original frame) into
    ``out_dir``. Each anchor gets a filled dot + its id label (staggered to
    avoid overlaps), an arrow when it carries an axis, and the projected
    horizontal rectangle when it carries a region; off-screen anchors are
    listed in a margin note.
    """
    from PIL import Image, ImageDraw

    rgb_path = Path(rgb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "observation_raw.png"
    if raw_path.resolve() != rgb_path.resolve():
        shutil.copyfile(rgb_path, raw_path)

    im = Image.open(rgb_path).convert("RGB")
    scale = min(1.0, float(max_side) / float(max(im.size)))
    if scale < 1.0:
        im = im.resize((round(im.size[0] * scale), round(im.size[1] * scale)),
                       Image.LANCZOS)
    # Scaling the image scales the intrinsics identically (fx, fy, cx, cy).
    K = np.asarray(cam_K, dtype=float).reshape(3, 3).copy()
    K[:2, :] *= scale
    T = (np.asarray(t_base_camera, dtype=float) if t_base_camera is not None
         else load_base_camera_extrinsic())
    W, H = im.size
    draw = ImageDraw.Draw(im)
    font = _load_font(max(14, H // 40))

    names = list(anchors.names())
    points = np.stack([anchors[n].point for n in names]) if names else np.zeros((0, 3))
    uv, z = project_base_points(points, T, K)

    visible: list[str] = []
    off_screen: list[str] = []
    placed_labels: list[tuple[float, float, float, float]] = []
    for i, name in enumerate(names):
        a = anchors[name]
        color = _PALETTE[i % len(_PALETTE)]
        on = bool(z[i] > 0 and 0 <= uv[i, 0] < W and 0 <= uv[i, 1] < H)
        if not on:
            off_screen.append(name)
            continue
        visible.append(name)
        u, v = float(uv[i, 0]), float(uv[i, 1])
        r = max(4, H // 120)
        # region rectangle FIRST (under the dot): the horizontal base-frame
        # AABB at the anchor's height, projected corner by corner.
        if a.region_half is not None:
            hx, hy = float(a.region_half[0]), float(a.region_half[1])
            corners = np.array([
                a.point + [sx * hx, sy * hy, 0.0]
                for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
            cuv, cz = project_base_points(corners, T, K)
            if np.all(cz > 0):
                draw.polygon([tuple(p) for p in cuv], outline=color, width=2)
        if a.axis is not None:
            tip_uv, tip_z = project_base_points(
                a.point[None, :] + AXIS_ARROW_M * np.asarray(a.axis, dtype=float)[None, :], T, K)
            if tip_z[0] > 0:
                tu, tv = float(tip_uv[0, 0]), float(tip_uv[0, 1])
                draw.line([(u, v), (tu, tv)], fill=color, width=3)
                # simple arrowhead: two short back-strokes
                d = np.array([tu - u, tv - v])
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-6:
                    d = d / nrm
                    perp = np.array([-d[1], d[0]])
                    for sgn in (1.0, -1.0):
                        back = np.array([tu, tv]) - 8.0 * d + sgn * 5.0 * perp
                        draw.line([(tu, tv), tuple(back)], fill=color, width=3)
        draw.ellipse([u - r, v - r, u + r, v + r], fill=color,
                     outline=(0, 0, 0), width=2)
        # Label with simple stagger: drop below the dot, push down while the
        # box overlaps an already-placed label.
        tb = draw.textbbox((0, 0), name, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx = min(max(0.0, u + r + 3), W - tw - 4)
        ly = min(max(0.0, v - th / 2), H - th - 6)
        def _hits(x0, y0):
            return any(not (x0 + tw + 4 < px0 or x0 > px1 or
                            y0 + th + 4 < py0 or y0 > py1)
                       for px0, py0, px1, py1 in placed_labels)
        guard = 0
        while _hits(lx, ly) and guard < 20:
            ly = (ly + th + 6) % max(1, H - th - 6)
            guard += 1
        placed_labels.append((lx, ly, lx + tw + 4, ly + th + 4))
        draw.rectangle([lx - 2, ly - 1, lx + tw + 3, ly + th + 4],
                       fill=(0, 0, 0))
        draw.text((lx, ly), name, fill=color, font=font)

    if off_screen:
        note = "off-screen: " + ", ".join(off_screen)
        tb = draw.textbbox((0, 0), note, font=font)
        draw.rectangle([4, H - (tb[3] - tb[1]) - 12, 10 + tb[2] - tb[0], H - 4],
                       fill=(0, 0, 0))
        draw.text((7, H - (tb[3] - tb[1]) - 9), note, fill=(255, 225, 25), font=font)

    annotated_path = out_dir / "anchors_annotated.png"
    im.save(annotated_path)
    return AnnotatedScene(
        annotated_path=annotated_path, raw_path=raw_path,
        visible=tuple(visible), off_screen=tuple(off_screen))
