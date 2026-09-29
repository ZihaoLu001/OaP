#!/usr/bin/env python3
"""Build SceneComplete-style DINO/RGB-D 3D correspondences for one object.

Stage-1 SCALE payload (GPU half): runs under the SAM3D/DINO env interpreter
(torch + torch.hub DINOv2) and must not import ``oap``. It renders the
unscaled SAM3D mesh into an object crop, extracts DINOv2 patch descriptors for
the rendered reconstruction and the observed RGB crop, matches foreground
patches, and backprojects the observation matches through RGB-D. The output is
the 3D correspondence JSON the in-process scale step turns into a depth-anchored
metric scale report.

The script does not use any privileged object mesh or physics metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _validate_se3_matrix(matrix: Any, *, name: str) -> np.ndarray:
    """Self-contained SE(3) gate for this external-environment payload."""
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 SE(3) matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN/Inf")
    if not np.allclose(
        value[3],
        [0.0, 0.0, 0.0, 1.0],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = value[:3, :3]
    orthogonality_error = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-6:
        raise ValueError(f"{name} rotation is not orthonormal")
    if abs(determinant - 1.0) > 1e-6:
        raise ValueError(f"{name} rotation determinant is not +1")
    return value.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-obj", type=Path, required=True)
    parser.add_argument("--crop-image", type=Path, required=True)
    parser.add_argument("--crop-mask", type=Path, required=True)
    parser.add_argument("--crop-metadata", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vis-out", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-matches", type=int, default=96)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument(
        "--samples-per-patch-side",
        type=int,
        default=1,
        help=(
            "Sample an NxN grid inside each matched DINO patch. DINO provides "
            "patch-level correspondences; this turns each accepted patch match "
            "into multiple rendered-surface/RGB-D 3D point pairs without using "
            "privileged object scale."
        ),
    )
    parser.add_argument("--model", default="dinov2_vits14_reg")
    parser.add_argument("--torch-hub-dir", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs = json.loads(args.observation_manifest.read_text(encoding="utf-8"))
    crop_meta = json.loads(args.crop_metadata.read_text(encoding="utf-8"))
    vertices, faces, colors = load_obj(args.mesh_obj)
    render_rgb, render_points, render_mask = render_mesh_ortho(
        vertices,
        faces,
        colors,
        size=int(args.image_size),
    )

    crop_rgb = np.asarray(Image.open(args.crop_image).convert("RGB").resize((args.image_size, args.image_size)))
    crop_mask = np.asarray(Image.open(args.crop_mask).convert("L").resize((args.image_size, args.image_size))) > 0

    src_feats, grid_hw = dino_patch_features(
        render_rgb,
        model_name=str(args.model),
        hub_dir=str(args.torch_hub_dir),
        device=str(args.device),
    )
    dst_feats, grid_hw_2 = dino_patch_features(
        crop_rgb,
        model_name=str(args.model),
        hub_dir=str(args.torch_hub_dir),
        device=str(args.device),
    )
    if grid_hw != grid_hw_2:
        raise RuntimeError(f"DINO grid mismatch: {grid_hw} vs {grid_hw_2}")

    source_points, target_points, matches = build_matches(
        src_feats=src_feats,
        dst_feats=dst_feats,
        grid_hw=grid_hw,
        render_points=render_points,
        render_mask=render_mask,
        crop_mask=crop_mask,
        crop_meta=crop_meta,
        obs=obs,
        max_matches=int(args.max_matches),
        samples_per_patch_side=max(1, int(args.samples_per_patch_side)),
    )
    if source_points.shape[0] < int(args.min_matches):
        raise RuntimeError(f"too few DINO/RGB-D correspondences: {source_points.shape[0]}")

    payload = {
        "schema": "countersim.scenecomplete_dino_rgbd_correspondences.v1",
        "method": "actual_dinov2_patch_matching_plus_rgbd_backprojection",
        "source_points_3d": source_points.tolist(),
        "target_points_3d": target_points.tolist(),
        "num_correspondences": int(source_points.shape[0]),
        "mesh": str(args.mesh_obj),
        "crop_image": str(args.crop_image),
        "crop_mask": str(args.crop_mask),
        "observation_manifest": str(args.observation_manifest),
        "dino_model": str(args.model),
        "notes": [
            "source_points_3d are unscaled SAM3D mesh coordinates from a rendered reconstruction depth map",
            "target_points_3d are RGB-D mask points backprojected into the base frame",
            "no privileged object mesh, collision, or physical metadata is used",
        ],
        "matches": matches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.vis_out:
        write_match_vis(args.vis_out, render_rgb, crop_rgb, matches, grid_hw)
    print(json.dumps({"out": str(args.out), "matches": int(source_points.shape[0])}, indent=2))


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    colors: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(parts) >= 7:
                colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            else:
                colors.append([0.7, 0.2, 0.2])
        elif line.startswith("f "):
            idx: list[int] = []
            for token in line.split()[1:]:
                idx.append(int(token.split("/")[0]) - 1)
            if len(idx) >= 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if len(vertices) < 3 or len(faces) < 1:
        raise ValueError(f"OBJ has too little geometry: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32), np.asarray(colors, dtype=np.float64)


def render_mesh_ortho(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = np.maximum(hi - lo, 1e-8)
    scale = 0.82 * float(size) / float(max(extent[0], extent[1]))
    xy = (vertices[:, :2] - center[:2]) * scale + 0.5 * size
    z = vertices[:, 2]
    rgb = np.full((size, size, 3), 245, dtype=np.uint8)
    point_img = np.full((size, size, 3), np.nan, dtype=np.float64)
    zbuf = np.full((size, size), -np.inf, dtype=np.float64)
    mask = np.zeros((size, size), dtype=bool)

    for tri in faces:
        pts = xy[tri]
        xmin = max(0, int(np.floor(pts[:, 0].min())))
        xmax = min(size - 1, int(np.ceil(pts[:, 0].max())))
        ymin = max(0, int(np.floor(pts[:, 1].min())))
        ymax = min(size - 1, int(np.ceil(pts[:, 1].max())))
        if xmax < xmin or ymax < ymin:
            continue
        v0, v1, v2 = pts
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(float(denom)) < 1e-12:
            continue
        for py in range(ymin, ymax + 1):
            for px in range(xmin, xmax + 1):
                p = np.array([px + 0.5, py + 0.5])
                w0 = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
                w1 = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
                w2 = 1.0 - w0 - w1
                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4:
                    continue
                depth = float(w0 * z[tri[0]] + w1 * z[tri[1]] + w2 * z[tri[2]])
                if depth <= zbuf[py, px]:
                    continue
                zbuf[py, px] = depth
                point = w0 * vertices[tri[0]] + w1 * vertices[tri[1]] + w2 * vertices[tri[2]]
                color = w0 * colors[tri[0]] + w1 * colors[tri[1]] + w2 * colors[tri[2]]
                point_img[py, px] = point
                rgb[py, px] = np.clip(color * 255.0, 0, 255).astype(np.uint8)
                mask[py, px] = True
    return rgb, point_img, mask


def dino_patch_features(
    image_rgb: np.ndarray,
    *,
    model_name: str,
    hub_dir: str,
    device: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    import torch

    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model = torch.hub.load(hub_dir, model_name, source="local", pretrained=True).to(dev).eval()
    arr = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.inference_mode():
        out = model.forward_features(x)
    tokens = out["x_norm_patchtokens"][0].detach().float().cpu().numpy()
    h = image_rgb.shape[0] // 14
    w = image_rgb.shape[1] // 14
    tokens = tokens[: h * w]
    tokens /= np.maximum(np.linalg.norm(tokens, axis=1, keepdims=True), 1e-8)
    return tokens, (h, w)


def build_matches(
    *,
    src_feats: np.ndarray,
    dst_feats: np.ndarray,
    grid_hw: tuple[int, int],
    render_points: np.ndarray,
    render_mask: np.ndarray,
    crop_mask: np.ndarray,
    crop_meta: dict[str, Any],
    obs: dict[str, Any],
    max_matches: int,
    samples_per_patch_side: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    gh, gw = grid_hw
    patch = 14
    src_valid: list[int] = []
    dst_valid: list[int] = []
    for iy in range(gh):
        for ix in range(gw):
            idx = iy * gw + ix
            x = int(ix * patch + patch // 2)
            y = int(iy * patch + patch // 2)
            if 0 <= y < render_mask.shape[0] and 0 <= x < render_mask.shape[1] and render_mask[y, x]:
                src_valid.append(idx)
            if 0 <= y < crop_mask.shape[0] and 0 <= x < crop_mask.shape[1] and crop_mask[y, x]:
                dst_valid.append(idx)
    if not src_valid or not dst_valid:
        raise RuntimeError("no valid foreground DINO patches")
    src_arr = np.asarray(src_valid, dtype=int)
    dst_arr = np.asarray(dst_valid, dtype=int)
    sim = dst_feats[dst_arr] @ src_feats[src_arr].T
    best_src_for_dst = sim.argmax(axis=1)
    best_dst_for_src = sim.argmax(axis=0)
    candidates: list[tuple[float, int, int]] = []
    for dst_i, src_j_local in enumerate(best_src_for_dst):
        if best_dst_for_src[src_j_local] == dst_i:
            candidates.append((float(sim[dst_i, src_j_local]), int(src_arr[src_j_local]), int(dst_arr[dst_i])))
    if len(candidates) < max_matches:
        # Mutual nearest-neighbor DINO patch matches are precise but too sparse
        # for low-texture or small objects. SceneComplete-style metric recovery
        # needs enough 3D pairs for robust similarity fitting, so we supplement
        # with the highest-similarity foreground patch pairs. These are still
        # DINO/RGB-D correspondences; they do not use hidden object geometry or
        # privileged scale.
        flat_count = min(max_matches * 8, sim.size)
        flat = np.argpartition(-sim.reshape(-1), flat_count - 1)[:flat_count]
        existing = {(src_idx, dst_idx) for _, src_idx, dst_idx in candidates}
        for flat_i in flat.tolist():
            dst_i, src_j_local = np.unravel_index(flat_i, sim.shape)
            src_idx = int(src_arr[src_j_local])
            dst_idx = int(dst_arr[dst_i])
            if (src_idx, dst_idx) in existing:
                continue
            candidates.append((float(sim[dst_i, src_j_local]), src_idx, dst_idx))
            existing.add((src_idx, dst_idx))
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    depth = np.load(obs["depth_m_npy"]).astype(float)
    K = np.asarray(obs["cam_K"], dtype=float)
    meta = json.loads(Path(obs["foundationpose_recording"]).joinpath("metadata.json").read_text(encoding="utf-8"))
    if "T_base_camera_opencv" in meta:
        T_base_camera = _validate_se3_matrix(
            meta["T_base_camera_opencv"],
            name="foundationpose metadata T_base_camera_opencv",
        )
    elif "T_base_zed2i" in meta:
        T_base_camera = load_T_base_zed2i(Path(meta["T_base_zed2i"]))
    elif "T_base_zed2i" in obs:
        T_base_camera = load_T_base_zed2i(Path(obs["T_base_zed2i"]))
    else:
        raise KeyError("metadata has neither T_base_camera_opencv nor T_base_zed2i")
    crop_box = crop_meta["crop_box_xyxy_in_padded_source"]
    x1, y1, x2, y2 = [float(v) for v in crop_box]
    # The crop is WRITTEN at crop_meta["output_size"] (512 by default,
    # reconstruct/mesh.py:41) but this payload RESIZES it to --image-size (224)
    # before DINO, and every coordinate below lives in that resized space: cx/cy
    # are grid indices times ``patch``, and ci/cj index crop_mask, which was
    # resized to --image-size. Normalising them by the PRE-resize 512 therefore
    # compressed every correspondence toward the crop's top-left by 224/512, so
    # the fitted similarity scale could never match the RGB-D mask extent -- the
    # DINO leg was rejected on 8/8 historical objects and again on both objects
    # of the 2026-07-26 bundle (dino_rgbd_median_size_ratio_too_high 7.15 >
    # 1.15), leaving the "consensus" a single-source fallback in every run that
    # has ever executed. Taking the divisor from the array these indices
    # actually address keeps the two in step by construction, instead of by a
    # constant that has to match across two files.
    out_size = float(crop_mask.shape[1])
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    match_payload: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for score, src_idx, dst_idx in candidates:
        if len(source_points) >= max_matches:
            break
        src_ix = src_idx % gw
        src_iy = src_idx // gw
        dst_ix = dst_idx % gw
        dst_iy = dst_idx // gw
        offsets = np.linspace(0.25, 0.75, samples_per_patch_side)
        for oy in offsets:
            for ox in offsets:
                if len(source_points) >= max_matches:
                    break
                sx = int(src_ix * patch + ox * patch)
                sy = int(src_iy * patch + oy * patch)
                cx = float(dst_ix * patch + ox * patch)
                cy = float(dst_iy * patch + oy * patch)
                if sy < 0 or sx < 0 or sy >= render_points.shape[0] or sx >= render_points.shape[1]:
                    continue
                if not np.all(np.isfinite(render_points[sy, sx])):
                    continue
                if not render_mask[sy, sx]:
                    continue
                ci = int(round(cx))
                cj = int(round(cy))
                if cj < 0 or ci < 0 or cj >= crop_mask.shape[0] or ci >= crop_mask.shape[1] or not crop_mask[cj, ci]:
                    continue
                u = x1 + (cx / out_size) * (x2 - x1)
                v = y1 + (cy / out_size) * (y2 - y1)
                ui = int(round(u))
                vi = int(round(v))
                if ui < 0 or vi < 0 or vi >= depth.shape[0] or ui >= depth.shape[1]:
                    continue
                if (ui, vi) in seen:
                    continue
                z = float(depth[vi, ui])
                if not np.isfinite(z) or z <= 0.05 or z >= 5.0:
                    continue
                x_cam = (u - float(K[0, 2])) / float(K[0, 0]) * z
                y_cam = (v - float(K[1, 2])) / float(K[1, 1]) * z
                p_base = (T_base_camera @ np.array([x_cam, y_cam, z, 1.0], dtype=float))[:3]
                source_points.append(render_points[sy, sx].copy())
                target_points.append(p_base)
                seen.add((ui, vi))
                match_payload.append(
                    {
                        "similarity": score,
                        "source_patch_xy": [int(sx), int(sy)],
                        "crop_patch_xy": [float(cx), float(cy)],
                        "observation_uv": [float(u), float(v)],
                    }
                )
            if len(source_points) >= max_matches:
                break
    return np.asarray(source_points), np.asarray(target_points), match_payload


def load_T_base_zed2i(path: Path) -> np.ndarray:
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        payload = json.loads(path.read_text(encoding="utf-8"))
    node = payload.get("T_base_zed2i", payload)
    if isinstance(node, dict) and "matrix" in node:
        node = node["matrix"]
    return _validate_se3_matrix(
        node,
        name=f"base<-camera calibration {path}",
    )


def write_match_vis(path: Path, left: np.ndarray, right: np.ndarray, matches: list[dict[str, Any]], grid_hw: tuple[int, int]) -> None:
    canvas = np.concatenate([left.copy(), right.copy()], axis=1)
    offset = left.shape[1]
    for m in matches[:40]:
        x1, y1 = m["source_patch_xy"]
        x2, y2 = m["crop_patch_xy"]
        color = (0, 255, 0)
        cv2.circle(canvas, (int(x1), int(y1)), 2, color, -1)
        cv2.circle(canvas, (int(x2) + offset, int(y2)), 2, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + offset, int(y2)), color, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


if __name__ == "__main__":
    main()
