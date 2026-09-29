#!/usr/bin/env python3
"""Real FoundationPose-Any6D joint POSE + metric SCALE recovery (mesh stays SAM3D).

Stage-1 POSE payload (GPU half): runs under the FOUNDATIONPOSE env interpreter
with the Any6D checkout on ``PYTHONPATH`` and must not import ``oap``.
This is the rigorous reconstruction front-end for the real2sim2real loop: the
REAL Any6D estimator -- coarse-OBB axis alignment + FoundationPose
render-and-compare refinement -- run on the SAM3D metric mesh. The coarse-OBB
pose init wins on flat/symmetric objects where plain FoundationPose
orientation is ambiguous.

OCCLUDED-AXIS GUARD (the one measured failure mode): single-view
render-and-compare cannot see the back of the object, so the size search can
collapse an unobserved axis. The masked depth point cloud is a LOWER BOUND on
the true extent, so we clamp every recovered object-axis UP to its observed
depth-OBB extent.

16GB fit: ``--downscale 0.5`` + ``make_rotation_grid(min_n_views=12,
inplane_step=120)`` + ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.

Output JSON (schema ``any6d_foundationpose_pose_scale_v1``) carries
``size_LWH_m`` + ``assignment`` plus the Any6D POSE: camera-frame
``R_cam_obj``/``center_cam_m`` and base-frame ``pos_base``/``quat_wxyz``/
``yaw_base_rad`` via the ``T_base_camera`` recorded in the observation
manifest (identity at capture time; the in-process pose step re-applies the
packaged hand-eye calibration).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


# ---------- pure-numpy helpers (testable without the Any6D/GPU stack) ----------
def fit_table_plane(pts_bg: np.ndarray, iters: int = 300, tol: float = 0.008, seed: int = 0):
    rng = np.random.default_rng(seed)
    best_inl, best = 0, None
    for _ in range(iters):
        p0, p1, p2 = pts_bg[rng.choice(len(pts_bg), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p0
        inl = int((np.abs(pts_bg @ n + d) < tol).sum())
        if inl > best_inl:
            best_inl, best = inl, (n, d)
    if best is None:
        raise RuntimeError("table-plane RANSAC found no non-degenerate 3-point sample")
    return (*best, best_inl)


def depth_obb_axes(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> dict:
    """Observed object frame from a single RGB-D view: plane normal = up, in-plane PCA
    = long/short axes; per-axis observed extents (1-99 pct) are LOWER BOUNDS on size."""
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth > 0.2) & (depth < 3.0)
    z = depth[valid]
    pts = np.stack([(us[valid] - K[0, 2]) / K[0, 0] * z,
                    (vs[valid] - K[1, 2]) / K[1, 1] * z, z], axis=1)
    m = mask[valid]
    n, d, _ = fit_table_plane(pts[~m])
    obj = pts[m]
    h = obj @ n + d
    if np.median(h) < 0:
        n, d, h = -n, -d, -h
    keep = h > 0.004
    obj, h = obj[keep], h[keep]
    height = float(np.percentile(h, 99))
    e1 = np.cross(n, np.array([0.0, 0.0, 1.0])); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    xy = np.stack([obj @ e1, obj @ e2], axis=1)
    c = xy.mean(axis=0)
    _, _, Vt = np.linalg.svd(xy - c, full_matrices=False)
    ab = (xy - c) @ Vt.T
    L = float(np.percentile(ab[:, 0], 99) - np.percentile(ab[:, 0], 1))
    Wd = float(np.percentile(ab[:, 1], 99) - np.percentile(ab[:, 1], 1))
    ax_long = Vt[0, 0] * e1 + Vt[0, 1] * e2; ax_long /= np.linalg.norm(ax_long)
    ax_short = np.cross(n, ax_long)
    # object axes in CAMERA frame: columns = [long, short, up]
    axes_cam = np.stack([ax_long, ax_short, n], axis=1)
    # object centre estimate (camera frame): in-plane centroid of the masked
    # points, lifted to HALF the observed height above the plane -- the visible
    # surface centroid sits on the object's front/top, so the geometric centre
    # is height/2 up the gravity normal from the footprint centroid.
    inplane_c = obj - (h[:, None]) * n[None, :]          # project each pt onto the plane
    centroid_cam = inplane_c.mean(axis=0) + 0.5 * height * n
    return {"axes_cam": axes_cam, "observed_LWH": np.array([L, Wd, height]),
            "plane_n": n, "plane_d": d, "centroid_cam": centroid_cam}


def splat_iou_mae(verts_cam: np.ndarray, depth: np.ndarray, mask: np.ndarray,
                  K: np.ndarray, stride: int = 4) -> tuple[float, float]:
    """Point-splat a posed mesh through K; silhouette IoU vs the SAM mask + depth MAE."""
    H, W = depth.shape
    w, h = W // stride, H // stride
    z = verts_cam[:, 2]; v = verts_cam[z > 0.05]
    u = (v[:, 0] / v[:, 2] * K[0, 0] + K[0, 2]) / stride
    vv = (v[:, 1] / v[:, 2] * K[1, 1] + K[1, 2]) / stride
    ui = np.clip(u.astype(int), 0, w - 1); vi = np.clip(vv.astype(int), 0, h - 1)
    zbuf = np.full((h, w), np.inf); np.minimum.at(zbuf, (vi, ui), v[:, 2])
    rendered = np.isfinite(zbuf)
    mask_lo, depth_lo = mask[::stride, ::stride], depth[::stride, ::stride]
    inter = rendered & mask_lo; union = rendered | mask_lo
    iou = float(inter.sum()) / max(1, int(union.sum()))
    both = inter & (depth_lo > 0.2)
    mae = float(np.abs(zbuf[both] - depth_lo[both]).mean()) if both.sum() > 20 else float("nan")
    return iou, mae


def mat_to_quat_wxyz(R: np.ndarray) -> list[float]:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    return q.tolist()


def occluded_axis_guard(verts_obj: np.ndarray, R_cam_obj: np.ndarray, center_cam: np.ndarray,
                        obb: dict, tol: float = 0.006) -> tuple[np.ndarray, dict]:
    """Clamp each object-axis extent UP to its observed depth-OBB extent. verts_obj are
    the Any6D-sized mesh vertices in the OBJECT frame (centered). Returns guarded verts
    + a report. The observed extent is a lower bound (we see at least that much), so the
    guard only ever enlarges a collapsed/unobserved axis -- never shrinks a real one."""
    verts_cam = verts_obj @ R_cam_obj.T + center_cam
    rel = verts_cam - center_cam
    recovered, scale = np.zeros(3), np.ones(3)
    clamped = []
    for i in range(3):
        proj = rel @ obb["axes_cam"][:, i]
        recovered[i] = float(proj.max() - proj.min())
        obs = float(obb["observed_LWH"][i])
        if recovered[i] < obs - tol and recovered[i] > 1e-4:
            scale[i] = obs / recovered[i]
            clamped.append({"axis": ["long", "short", "up"][i],
                            "recovered_m": round(recovered[i], 4), "observed_m": round(obs, 4),
                            "scale_up": round(float(scale[i]), 3)})
    if clamped:
        # apply per-OBJECT-axis scale: express verts in the OBB axis basis, scale, back.
        B = R_cam_obj.T @ obb["axes_cam"]            # object-frame -> obb-axis basis
        verts_obj = (verts_obj @ B) * scale @ B.T
    return verts_obj, {"recovered_LWH_m": recovered.round(4).tolist(),
                       "observed_LWH_m": obb["observed_LWH"].round(4).tolist(),
                       "clamped_axes": clamped}


def render_overlay(verts_cam: np.ndarray, faces: np.ndarray, K_full: np.ndarray,
                   rgb_full: np.ndarray, out_png: Path) -> None:
    """Render-and-compare: paint the posed+sized mesh (orange, normal-shaded) over the
    real RGB via a painter's-algorithm z-sort -- the direct fidelity view."""
    import cv2
    H, W = rgb_full.shape[:2]
    z = verts_cam[:, 2]
    u = verts_cam[:, 0] / np.clip(z, 1e-6, None) * K_full[0, 0] + K_full[0, 2]
    v = verts_cam[:, 1] / np.clip(z, 1e-6, None) * K_full[1, 1] + K_full[1, 2]
    uv = np.stack([u, v], axis=1)
    nrm = np.cross(verts_cam[faces[:, 1]] - verts_cam[faces[:, 0]],
                   verts_cam[faces[:, 2]] - verts_cam[faces[:, 0]])
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    shade = np.clip(np.abs(nrm[:, 2]), 0.25, 1.0)
    fz = z[faces].mean(axis=1)
    overlay = rgb_full.copy()
    for fi in np.argsort(-fz):                       # far faces first
        f = faces[fi]
        if np.any(z[f] <= 0.05):
            continue
        pts = uv[f].astype(np.int32)
        col = (np.array([255, 120, 0]) * shade[fi]).astype(np.uint8).tolist()
        cv2.fillConvexPoly(overlay, pts, col)
    blended = cv2.addWeighted(overlay, 0.5, rgb_full, 0.5, 0)
    cv2.imwrite(str(out_png), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def run(packet: Path, mesh_path: Path, downscale: float = 0.5,
        any6d_iter: int = 5, seed: int = 0, overlay_dir: Path | None = None) -> dict:
    # heavy imports. OBB-only mode needs ONLY numpy/cv2/trimesh (no torch, no
    # estimater/Any6D, no nvdiffrast) -- so it runs in any reconstruction env,
    # including the cluster's sam3d env; the FoundationPose path imports the
    # heavy deps lazily inside its branch below.
    import random

    import cv2
    import trimesh
    _obb_only = os.environ.get("OAP_POSE_OBB_ONLY") == "1"
    if not _obb_only:
        import torch
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    else:
        random.seed(seed); np.random.seed(seed)

    man = json.loads((packet / "observation_manifest.json").read_text())
    K = np.array(man["cam_K"], dtype=float)
    T_bc = np.array(man.get("T_base_camera_opencv", np.eye(4).tolist()), dtype=float)
    color_bgr = cv2.imread(str(packet / "rgb.png"))
    if color_bgr is None:
        raise FileNotFoundError(f"cannot read rgb image: {packet / 'rgb.png'}")
    color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(packet / "depth_m.npy").astype(np.float32); depth[~np.isfinite(depth)] = 0.0
    mask_img = cv2.imread(str(packet / "mask_00.png"), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"cannot read mask image: {packet / 'mask_00.png'}")
    mask = mask_img > 127
    if downscale != 1.0:
        H, W = color.shape[:2]; nW, nH = int(round(W * downscale)), int(round(H * downscale))
        color = cv2.resize(color, (nW, nH), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (nW, nH), interpolation=cv2.INTER_NEAREST) > 0
        K = K.copy(); K[:2, :] *= downscale

    obb = depth_obb_axes(depth, mask, K)
    mesh = trimesh.load(str(mesh_path), process=False)

    # OBB-ONLY mode (OAP_POSE_OBB_ONLY=1): a pure-NumPy coarse pose from
    # the depth-OBB, with NO FoundationPose render-and-compare -> no nvdiffrast.
    # For hosts where nvdiffrast's CUDA rasterizer is unavailable: the RTX 5080
    # (sm_120) fails `rasterize_fwd_cuda` (cudaMalloc err 2) even fully headless
    # with 16 GB free, and the cluster has no FoundationPose env. The pose is
    # the gravity-OBB orientation + centroid; the mesh's own axes are matched
    # to the OBB axes by extent (unambiguous for these upright objects). The
    # runtime gravity_obb policy re-localizes at execution, so this coarse
    # reconstruction pose only seeds the twin + carries the metric size.
    if _obb_only:
        verts = np.asarray(mesh.vertices, dtype=float)
        verts -= 0.5 * (verts.max(0) + verts.min(0))       # centre in mesh frame
        ext0 = verts.max(0) - verts.min(0)
        # Match each mesh axis to the OBB slot [long, short, up] by extent, but
        # assign the UP slot FIRST: its observed extent (height, measured along
        # the reliable gravity normal) pins WHICH mesh axis is vertical -- the
        # in-plane long/short observed extents are occlusion-shortened and
        # ambiguous (a tub reads L~=W~=6 cm), so extent-matching them first
        # steals the height's mesh axis and lays the object on its side.
        obb_lwh = np.asarray(obb["observed_LWH"], dtype=float)
        remaining = [0, 1, 2]
        slot_to_mesh = [0, 0, 0]
        for slot in (2, 0, 1):                             # up, then long, then short
            j = min(remaining, key=lambda a: abs(ext0[a] - obb_lwh[slot]))
            slot_to_mesh[slot] = j
            remaining.remove(j)
        # Permutation P: canonical object axis (slot) i <- mesh axis slot_to_mesh[i].
        P = np.zeros((3, 3))
        for slot, j in enumerate(slot_to_mesh):
            P[slot, j] = 1.0
        if np.linalg.det(P) < 0:                            # keep a proper rotation
            P[2] *= -1.0
        verts = verts @ P.T                                 # mesh -> canonical [L, short, up]
        R_cam_obj = np.asarray(obb["axes_cam"], dtype=float)  # canonical object -> camera
        center_cam = np.asarray(obb["centroid_cam"], dtype=float)
        pose = np.eye(4); pose[:3, :3] = R_cam_obj; pose[:3, 3] = center_cam

        class _M:                                           # minimal stand-in for est.mesh
            pass
        est = _M()
        est.mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces),
                                   process=False)
    else:
        from estimater import Any6D  # Any6D repo on PYTHONPATH (foundationpose env)
        dbg = packet / "_any6d_dbg"
        dbg.mkdir(parents=True, exist_ok=True)   # Any6D writes refine_init/debug meshes here
        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=str(dbg), debug=0)
        # 16GB fit: min_n_views=12, inplane_step=120.
        _mv = int(os.environ.get("OAP_POSE_MIN_VIEWS", "12"))
        est.make_rotation_grid(min_n_views=_mv, inplane_step=120)
        pose = np.asarray(est.register_any6d(K=K, rgb=color, depth=depth, ob_mask=mask,
                                             iteration=any6d_iter, name=mesh_path.stem,
                                             axis_align=True, coarse_est=True), dtype=float)
        R_cam_obj, center_cam = pose[:3, :3], pose[:3, 3]

    verts = np.asarray(est.mesh.vertices, dtype=float)
    verts -= 0.5 * (verts.max(0) + verts.min(0))            # center in object frame
    iou0, mae0 = splat_iou_mae(verts @ R_cam_obj.T + center_cam, depth, mask, K)
    verts_g, guard = occluded_axis_guard(verts, R_cam_obj, center_cam, obb)
    iou1, mae1 = splat_iou_mae(verts_g @ R_cam_obj.T + center_cam, depth, mask, K)

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        faces = np.asarray(est.mesh.faces)
        rgb_full_bgr = cv2.imread(str(packet / "rgb.png"))
        assert rgb_full_bgr is not None  # already read successfully above
        rgb_full = cv2.cvtColor(rgb_full_bgr, cv2.COLOR_BGR2RGB)
        K_full = np.array(man["cam_K"], dtype=float)
        render_overlay(verts_g @ R_cam_obj.T + center_cam, faces, K_full, rgb_full,
                       overlay_dir / f"{mesh_path.stem}_overlay.png")
        trimesh.Trimesh(vertices=verts_g, faces=faces, process=False).export(
            str(overlay_dir / f"{mesh_path.stem}_any6d_sized_mesh.obj"))

    # Metric size from the GRAVITY-ALIGNED depth-OBB extent, NOT the Any6D mesh's
    # own-axis AABB. Any6D's single-view render-and-compare SCALE under-fits the
    # self-occluded (away-from-camera) axis and its own frame is only partially
    # gravity-aligned, so the own-axis AABB both under-reads the far footprint
    # axis (amazon_box width read 0.089 vs true ~0.18) and misreports height. The
    # depth-OBB observed extent is a DIRECT, tight RGB-D measurement of the
    # object's visible bounding box in the table-normal frame -- correct in size
    # AND aspect (validated: amazon_box 0.21x0.18x0.07 open tray; round SUAVECITO
    # tub ~square). Map it onto the mesh's OWN axes (via B = mesh->OBB basis) so
    # size_LWH stays in the q_om-consistent order. A genuinely collapsed observed
    # axis (degenerate depth on a self-occluded axis) falls back to the mesh AABB
    # -- the depth extent is a lower bound there, never an over-read.
    ext_mesh = (verts_g.max(0) - verts_g.min(0))
    observed = np.asarray(obb["observed_LWH"], dtype=float)   # [long, short, up], gravity-aligned
    # B[:, j] = OBB axis j (long/short/up) expressed in the MESH frame; the mesh
    # axis most aligned with OBB slot j is that column's argmax. The canonical
    # object frame is L=long(x), W=short(y), H=up(z) -- the same "smallest/vertical
    # dim = H=z" order roll_q_om_for_standing_dims expects -- so size_LWH IS the
    # gravity-aligned observed extent and `assignment[i]` names the mesh axis that
    # is canonical axis i.
    B = R_cam_obj.T @ obb["axes_cam"]
    mesh_for_slot = [int(np.argmax(np.abs(B[:, j]))) for j in range(3)]
    if len(set(mesh_for_slot)) == 3 and float(observed.min()) > 1e-3:
        size_LWH = [float(observed[0]), float(observed[1]), float(observed[2])]
        assignment = mesh_for_slot
    else:                                                # degenerate depth -> fall back to mesh AABB
        size_LWH = [float(v) for v in ext_mesh]
        assignment = [int(a) for a in np.argsort(-ext_mesh)]

    T_base_obj = T_bc @ pose
    R_bo, p_bo = T_base_obj[:3, :3], T_base_obj[:3, 3]
    yaw = float(np.arctan2(R_bo[1, 0], R_bo[0, 0]))
    return {
        "schema": "any6d_foundationpose_pose_scale_v1",
        "method": ("depth-OBB coarse pose (gravity axes + centroid, mesh axes matched to "
                   "OBB extents) -- NO nvdiffrast; runtime re-localizes"
                   if _obb_only else
                   "real Any6D (coarse-OBB axis-align + FoundationPose render-and-compare) "
                   "on the SAM3D mesh + occluded-axis guard (clamp each object axis up to the "
                   "masked depth-OBB observed extent)"),
        "mesh": str(mesh_path), "downscale": downscale,
        "size_LWH_m": [round(v, 4) for v in size_LWH],
        "assignment": assignment,
        "R_cam_obj": R_cam_obj.round(6).tolist(), "center_cam_m": center_cam.round(5).tolist(),
        "pos_base": [round(float(v), 5) for v in p_bo],
        "quat_wxyz": [round(v, 6) for v in mat_to_quat_wxyz(R_bo)],
        "yaw_base_rad": round(yaw, 5),
        "iou_before_guard": round(iou0, 4), "iou_after_guard": round(iou1, 4),
        "depth_mae_m_after_guard": round(mae1, 4) if mae1 == mae1 else None,
        "occluded_axis_guard": guard,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--downscale", type=float, default=0.5)
    ap.add_argument("--iter", type=int, default=5)
    ap.add_argument("--overlay-dir", type=Path, default=None,
                    help="if set, write a render-and-compare overlay PNG + the Any6D-sized mesh here")
    args = ap.parse_args()
    res = run(args.packet, args.mesh, downscale=args.downscale, any6d_iter=args.iter,
              overlay_dir=args.overlay_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("size_LWH_m", "iou_before_guard", "iou_after_guard",
                                          "depth_mae_m_after_guard", "pos_base", "yaw_base_rad",
                                          "occluded_axis_guard")}, indent=2))


if __name__ == "__main__":
    main()
