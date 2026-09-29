#!/usr/bin/env python3
"""Persistent, STATEFUL FoundationPose worker: register ONCE, then track.

PAYLOAD-STYLE: no ``oap`` imports -- launched by
``oap.workers.client`` with the FoundationPose interpreter and
environment (cwd = FoundationPose root, root on PYTHONPATH). The score/refine
predictors, the CUDA rasterizer context, and the object mesh are initialized
ONCE. Production normally initializes the object track from its bound
reconstruction pose and refines it with `track_one`; subsequent observations
refine the previous accepted pose with the same single-hypothesis path. A
masked seed request may use global `register` only when its caller explicitly
allows that fallback. A rejected maskless production update is instead rolled
back: the continuous producer retries a later frame and the checkpoint fails
closed if no valid update arrives. It does not segment and re-register the
rejected frame.

MULTI-OBJECT: the two predictor nets + the nvdiffrast rasterizer context (the whole
~2.4 GB footprint) are SHARED across every object -- FoundationPose takes them as
constructor args -- while each object gets its OWN estimator (just per-mesh tensors +
pose_last: ~18 MB for a 75k-vert textured mesh, <0.1 MB for an untextured box). So a
single worker (one process, one GPU model load) tracks ALL scene objects at once, each
register-once-then-track on its own pose_last, and switching objects no longer tears
the worker down and re-registers. Estimators are built LAZILY from each request's
"mesh_file". PARK/LOAD move every estimator (shared nets moved once, redundantly-safe).

SEED mode (the Any6D recipe): a request may carry an "init_pose" (row-major 4x4
T_cam_obj, e.g. from the gravity-aligned depth OBB); the worker refines FROM that
prior instead of the blind global register, which stops render-and-compare from
FLIPPING a textureless/symmetric object.

Protocol (line-based):
  stdin:  one JSON per line: {"mesh_file", "test_scene_dir", "debug_dir",
          "mode": "register"|"track"|"seed", "init_pose"?} -- "mesh_file"
          selects/builds the per-object estimator (falls back to the startup
          --mesh_file when omitted); mode defaults to "register". Or the literal
          line "FPWORKER EXIT" / "FPWORKER PARK" / "FPWORKER LOAD".
  stdout: "FPWORKER READY" once, then one "FPWORKER {json}" line per request:
          {"ok": true, "pose_txt", "mode_used", "reregistered", "confidence"} or
          {"ok": false, "error"}.

All library logging is forced to stderr so stdout stays a clean protocol channel.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:   # annotations only -- never imported at runtime, so the
    import numpy as np   # payload rule (no third-party at module scope) holds


DEFAULT_PC_STRIDE = 1


def _load_foundationpose_mesh(trimesh_module, mesh_file: str):
    """Normalize OBJ/GLB loading to FoundationPose's expected mesh surface.

    ``trimesh.load`` returns a ``Scene`` for GLB by default, while
    FoundationPose requires ``vertices`` and ``faces`` directly. GLB textures
    also reload as ``PBRMaterial.baseColorTexture`` whereas upstream
    FoundationPose reads ``material.image``. Normalize both representation
    differences without changing geometry, UVs, object frame, or metric scale.
    """

    mesh = trimesh_module.load(mesh_file, force="mesh", process=False)
    if not isinstance(mesh, trimesh_module.Trimesh):
        raise RuntimeError(
            f"FoundationPose asset is not one triangle mesh: {mesh_file}"
        )
    visual = getattr(mesh, "visual", None)
    if isinstance(visual, trimesh_module.visual.texture.TextureVisuals):
        material = getattr(visual, "material", None)
        image = getattr(material, "image", None)
        if image is None:
            image = getattr(material, "baseColorTexture", None)
        if image is None:
            raise RuntimeError(
                f"FoundationPose textured asset has no material image: {mesh_file}"
            )
        if getattr(material, "image", None) is None:
            visual.material = trimesh_module.visual.material.SimpleMaterial(
                image=image,
                name=str(
                    getattr(material, "name", None)
                    or "oap_tracking"
                ),
            )
    return mesh


def _pointcloud_support(depth: "np.ndarray", K: "np.ndarray",
                        pose: "np.ndarray", diameter: float,
                        stride: int = DEFAULT_PC_STRIDE) -> int:
    """Depth-point support for a tracked pose (the Isaac ROS drift gate).

    Counts depth pixels that back-project within half the mesh diameter of the
    pose's translation. NVIDIA's productized FoundationPose Tracking node
    evaluates the full point cloud and rejects poses with fewer than
    ``min_pointcloud_support`` points (default 50). The production default is
    therefore ``stride=1`` so that threshold has the same units. A larger
    explicit stride remains available only for a calibrated ablation; retaining
    50 while using stride 4 would make the gate about 16 times stricter by area.
    The detector uses the depth frame already in memory, with no segmentation
    network. A mask, when present on seed/explicit-recovery requests, retains
    the stronger silhouette-IoU gate.
    """
    import numpy as np   # main()'s `import numpy as np` is local to main()

    stride = int(stride)
    if stride < 1:
        raise ValueError("point-cloud support stride must be >= 1")
    d = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    height, width = d.shape
    u = np.arange(width, dtype=np.float32) * stride
    v = np.arange(height, dtype=np.float32) * stride
    fx, fy, cx, cy = (
        np.float32(K[0, 0]),
        np.float32(K[1, 1]),
        np.float32(K[0, 2]),
        np.float32(K[1, 2]),
    )
    t = np.asarray(pose, dtype=np.float32).reshape(4, 4)[:3, 3]
    # Compute squared distance directly on the image grid. This preserves the
    # official full-cloud count without materializing an N-by-3 point array.
    distance_sq = (d * ((u - cx) / fx)[None, :] - t[0]) ** 2
    distance_sq += (d * ((v - cy) / fy)[:, None] - t[1]) ** 2
    distance_sq += (d - t[2]) ** 2
    radius_sq = np.float32(0.5 * float(diameter)) ** 2
    return int(np.count_nonzero(
        (d >= np.float32(0.001)) & (distance_sq < radius_sq)
    ))


def _mask_iou(mesh, pose, K, mask, stride: int = 4) -> float:
    """Render-and-compare confidence: silhouette IoU of the posed mesh (point-splat
    through K) vs the observed mask. A masked request may use it to trigger
    global registration only when its caller explicitly permits that fallback."""
    import numpy as np
    v = np.asarray(mesh.vertices, dtype=float)
    R, t = np.asarray(pose)[:3, :3], np.asarray(pose)[:3, 3]
    vc = v @ R.T + t
    z = vc[:, 2]
    vc = vc[z > 1e-3]
    if len(vc) == 0:
        return 0.0
    H, W = mask.shape
    hh, ww = H // stride, W // stride
    u = (vc[:, 0] / vc[:, 2] * K[0, 0] + K[0, 2]) / stride
    w = (vc[:, 1] / vc[:, 2] * K[1, 1] + K[1, 2]) / stride
    ui = np.clip(u.astype(int), 0, ww - 1)
    vi = np.clip(w.astype(int), 0, hh - 1)
    rend = np.zeros((hh, ww), dtype=bool)
    rend[vi, ui] = True
    m = mask[::stride, ::stride].astype(bool)
    inter = int((rend & m).sum())
    union = int((rend | m).sum())
    return float(inter) / max(1, union)


def _centered_pose_from_object_seed(
    init_pose: "np.ndarray", model_center: "np.ndarray"
) -> "np.ndarray":
    """Convert an external object-frame pose to FoundationPose track state.

    ``FoundationPose.reset_object`` translates every mesh by
    ``-model_center`` before refinement.  Its internal ``pose_last`` therefore
    maps the *centered* mesh frame into the camera, whereas reconstructed
    ``R_cam_obj``/``center_cam_m`` map the generated object's original mesh
    frame into the camera.  Upstream ``register`` performs this conversion
    implicitly; seed mode must do the same explicitly.

    Keeping this conversion here is important for two reasons: assigning a
    NumPy array directly makes upstream ``track_one`` crash when it calls
    ``pose_last.data.cpu()``, and omitting the center offset silently seeds a
    translated pose for non-origin-centred generated meshes.
    """
    import numpy as np

    pose = np.asarray(init_pose, dtype=np.float32).reshape(4, 4).copy()
    center = np.asarray(model_center, dtype=np.float32).reshape(3)
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def _request_used_global_registration(
    mode_used: str,
    reregistered: bool,
) -> bool:
    """Whether a request allocated FoundationPose's hypothesis-grid batch."""
    return str(mode_used) == "register" or bool(reregistered)


def _move_est(est, device: str) -> None:
    """Move FoundationPose's nets + tensors between GPU and CPU for warm-parking.

    Mirrors the estimator's own `to_device` but DELIBERATELY skips `glctx`: the
    nvdiffrast RasterizeCudaContext is CUDA-only, and `to_device` would try to
    re-create it as `RasterizeCudaContext('cpu')` (which fails). The two predictor
    nets (`scorer.model`, `refiner.model`) are the bulk of the ~2 GB; moving them
    to CPU frees it. `est.pose_last` (the register->track state) moves with the
    other tensors but its device is irrelevant -- track_one reads it via
    `.data.cpu().numpy()` -- so the track is preserved across a park/load cycle."""
    import torch
    from torch import nn
    for k in list(est.__dict__):
        v = est.__dict__[k]
        if torch.is_tensor(v) or isinstance(v, nn.Module):
            est.__dict__[k] = v.to(device)
    for k in list(getattr(est, "mesh_tensors", {}) or {}):
        est.mesh_tensors[k] = est.mesh_tensors[k].to(device)
    if getattr(est, "refiner", None) is not None:
        est.refiner.model.to(device)
    if getattr(est, "scorer", None) is not None:
        est.scorer.model.to(device)
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_file", default=None,
                        help="optional warm-up mesh; other objects are built lazily "
                             "from each request's mesh_file (multi-object worker).")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    # 4 refine iterations, ABOVE the paper's 1 and the demo's 2, deliberately:
    # our consecutive sights are separated by a whole executed prefix plus the
    # observation stop (~seconds of arm+object motion), not 30 fps video
    # frames. The maintainer's own guidance for large inter-frame motion is
    # exactly this knob (NVlabs/FoundationPose#83: "You can increase the
    # camera frame rate if possible. Otherwise, you can also try with higher
    # iter"). Cost is linear and still ~fast (track has no hypothesis batch).
    parser.add_argument("--track_refine_iter", type=int, default=4)
    parser.add_argument(
        "--min_pc_support",
        type=int,
        default=50,
        help=(
            "minimum full-resolution depth-point support; 50 matches the "
            "Isaac ROS FoundationPose tracking default"
        ),
    )
    parser.add_argument(
        "--pc_stride",
        type=int,
        default=DEFAULT_PC_STRIDE,
        help=(
            "depth support sampling stride (production 1, matching the official "
            "full point cloud); non-1 values are explicit ablations whose "
            "threshold must be recalibrated"
        ),
    )
    parser.add_argument("--track_min_iou", type=float, default=0.45,
                        help=(
                            "below this render-IoU, a masked request is lost; "
                            "global register occurs only when its caller "
                            "explicitly allows fallback"
                        ))
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)

    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    # estimater star-imports Utils, so trimesh / dr / set_seed and the
    # predictors are all reachable through its namespace (same surface the
    # upstream run_demo.py uses via `from estimater import *`).
    import estimater as fp  # noqa: E402
    from datareader import YcbineoatReader  # noqa: E402

    # Re-force stderr logging: library imports may have installed stdout
    # handlers, which would corrupt the protocol channel.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)

    fp.set_seed(0)
    # The score/refine nets + the nvdiffrast rasterizer context are the ENTIRE heavy
    # footprint (~2.4 GB) and are built ONCE, then SHARED by every per-object estimator
    # below (FoundationPose accepts them as ctor args). A second object therefore costs
    # only its per-mesh tensors + pose_last (~18 MB textured, <0.1 MB untextured) --
    # multi-object tracking is nearly free on VRAM and does NOT change the park/kill
    # mutex budget (FP's footprint is unchanged).
    scorer = fp.ScorePredictor()
    refiner = fp.PoseRefinePredictor()
    glctx = fp.dr.RasterizeCudaContext()
    _debug_dir = tempfile.mkdtemp(prefix="fpworker_debug_")

    estimators: dict = {}   # mesh_file -> FoundationPose (own pose_last => own track)
    meshes: dict = {}       # mesh_file -> trimesh (for the render-IoU confidence gate)

    def _get_est(mesh_file: str):
        """Lazily build (once) and cache a per-object estimator for `mesh_file`,
        reusing the shared scorer/refiner/glctx. Returns (est, mesh)."""
        key = str(mesh_file)
        if key not in estimators:
            m = _load_foundationpose_mesh(fp.trimesh, mesh_file)
            estimators[key] = fp.FoundationPose(
                model_pts=m.vertices, model_normals=m.vertex_normals, mesh=m,
                scorer=scorer, refiner=refiner, debug_dir=_debug_dir, debug=0, glctx=glctx)
            meshes[key] = m
            logging.info("[fpworker] built estimator for %s (now holding %d object(s))",
                         key, len(estimators))
        return estimators[key], meshes[key]

    def _gpu_mask_iou(est, K, mask, stride: int = 4) -> float:
        """Filled-triangle silhouette IoU on the existing CUDA rasterizer.

        The previous vertex point-splat score changed with mesh density and
        consequently under-scored generated meshes. Rasterizing faces gives
        the actual silhouette and keeps this quality gate on GPU.
        """
        observed_np = np.asarray(mask, dtype=bool)[::stride, ::stride]
        out_h, out_w = observed_np.shape
        _rgb, rendered_depth, _normal = fp.nvdiffrast_render(
            K=K,
            H=int(mask.shape[0]),
            W=int(mask.shape[1]),
            ob_in_cams=est.pose_last.reshape(1, 4, 4),
            glctx=est.glctx,
            mesh_tensors=est.mesh_tensors,
            output_size=np.asarray([out_h, out_w]),
            get_normal=False,
        )
        rendered = rendered_depth[0] > 0.0
        observed = torch.as_tensor(
            observed_np, dtype=torch.bool, device=rendered.device
        )
        intersection = torch.logical_and(rendered, observed).sum()
        union = torch.logical_or(rendered, observed).sum()
        return float(
            (intersection.float() / union.clamp_min(1).float()).item()
        )

    if args.mesh_file:  # warm up the primary object so the first observe doesn't stall
        _get_est(args.mesh_file)
    print("FPWORKER READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "FPWORKER EXIT":
            break
        if line == "FPWORKER PARK":
            # Warm-park: move the score/refine nets off the GPU. NOTE: this only
            # reclaims ~0.5 GB -- the nvdiffrast rasterizer context (~2 GB after a
            # register) CANNOT be freed in-process (proven: destroying glctx frees
            # nothing). So when the GPU is genuinely tight the client KILLS this
            # worker instead of parking it; this cheap park is just for the common
            # case where ~0.5 GB is enough and keeps the register->track state.
            try:
                for _e in estimators.values():   # shared nets move once (idempotent)
                    _move_est(_e, "cpu")
                print("FPWORKER PARKED", flush=True)
            except Exception as exc:  # noqa: BLE001 - report; client keeps FP resident
                print("FPWORKER " + json.dumps({"ok": False, "error": f"park failed: {exc!r}"[:300]}), flush=True)
            continue
        if line == "FPWORKER LOAD":
            # Restore to the GPU before the next register/track (predict assumes cuda).
            try:
                for _e in estimators.values():
                    _move_est(_e, "cuda")
                print("FPWORKER LOADED", flush=True)
            except Exception as exc:  # noqa: BLE001
                print("FPWORKER " + json.dumps({"ok": False, "error": f"load failed: {exc!r}"[:300]}), flush=True)
            continue
        active_est = None
        previous_pose_last = None
        try:
            req = json.loads(line)
            mesh_file = req.get("mesh_file") or args.mesh_file
            if not mesh_file:
                raise ValueError("request has no mesh_file and no startup --mesh_file")
            est, mesh = _get_est(mesh_file)
            active_est = est
            previous_pose_last = (
                None
                if getattr(est, "pose_last", None) is None
                else est.pose_last.detach().clone()
            )
            mode = str(req.get("mode", "register"))
            # allow_fallback=False marks a PROBE: the caller only needs a
            # cheap track refinement. Under gripper occlusion the register
            # fallback is wrong twice over -- its hypothesis grid transiently
            # allocates ~2.6 GB (the OOM source next to the resident VLM
            # judge on the 16 GB card) AND a global register on a half-
            # visible object hallucinates a pose. Probes return the low-
            # confidence track verbatim instead.
            allow_fallback = bool(req.get("allow_fallback", True))
            capture_id_raw = req.get("capture_id")
            capture_id = (
                None
                if capture_id_raw is None
                else str(capture_id_raw)
            )
            if capture_id is not None:
                metadata_path = os.path.join(
                    str(req["test_scene_dir"]),
                    "metadata.json",
                )
                with open(metadata_path, encoding="utf-8") as stream:
                    packet_metadata = json.load(stream)
                if packet_metadata.get("capture_id") != capture_id:
                    raise ValueError(
                        "FoundationPose request capture_id does not match "
                        "the RGB-D recording metadata"
                    )
            reader = YcbineoatReader(video_dir=str(req["test_scene_dir"]), shorter_side=None, zfar=np.inf)
            color = reader.get_color(0)
            depth = reader.get_depth(0)
            # Maskless packets are legal for TRACK (the payload's --no-seg
            # capture): track_one takes no mask, and drift is gated by the
            # point-cloud support check below. register/seed still require one.
            _mask_path = os.path.join(str(req["test_scene_dir"]), "masks", "frame_0000.png")
            mask = reader.get_mask(0).astype(bool) if os.path.exists(_mask_path) else None

            mode_used, reregistered = "register", False
            lost = False
            pc_support = None
            mask_iou = None
            init_pose = req.get("init_pose")
            if mask is None and not (mode == "track"
                                     and getattr(est, "pose_last", None) is not None):
                raise ValueError(
                    "maskless packet is only valid for TRACK with existing "
                    "state; register/seed require ob_mask -- capture with "
                    "segmentation (or run the payload's --seg-only pass)")
            if mode == "seed" and init_pose is not None:
                # SEED (the Any6D recipe): refine FoundationPose from an externally
                # supplied pose -- the gravity-OBB orientation prior -- instead of
                # the blind global register. The OBB resolves which face is up, so
                # render-and-compare no longer FLIPS a textureless/symmetric object.
                centered_seed = _centered_pose_from_object_seed(
                    init_pose, est.model_center
                )
                est.pose_last = torch.as_tensor(
                    centered_seed, dtype=torch.float, device="cuda"
                )
                pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                     iteration=int(args.est_refine_iter))
                mask_iou = _gpu_mask_iou(est, reader.K, mask)
                mode_used = "seed"
                if mask_iou < float(args.track_min_iou) and allow_fallback:
                    # the seed was too far off -> fall back to a global register.
                    torch.cuda.empty_cache()  # the hypothesis grid spikes ~2.6 GB
                    pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                                        iteration=int(args.est_refine_iter))
                    mask_iou = _gpu_mask_iou(est, reader.K, mask)
                    mode_used, reregistered = "register", True
            elif mode == "track" and getattr(est, "pose_last", None) is not None:
                # TRACK: single-hypothesis refinement from the previous pose (cheap,
                # temporally coherent). track_one keeps est.pose_last for the next call.
                pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                     iteration=int(args.track_refine_iter))
                mode_used = "track"
                pc_support = _pointcloud_support(
                    depth, reader.K, pose, float(getattr(est, "diameter", 0.1)),
                    stride=int(args.pc_stride))
                if mask is not None:
                    # Mask available (e.g. a seg-bearing packet): the silhouette
                    # IoU keeps the STRONG gate role, with in-worker fallback.
                    mask_iou = _gpu_mask_iou(est, reader.K, mask)
                    if mask_iou < float(args.track_min_iou) and allow_fallback:
                        # lost track (occlusion / large motion) -> fall back to estimation.
                        torch.cuda.empty_cache()  # the hypothesis grid spikes ~2.6 GB
                        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                                            iteration=int(args.est_refine_iter))
                        mask_iou = _gpu_mask_iou(est, reader.K, mask)
                        mode_used, reregistered = "register", True
                else:
                    # MASKLESS production track: apply the full-resolution
                    # Isaac ROS-style point-cloud gate. Below the support floor
                    # this request cannot re-register because it has no mask.
                    # The update is reported lost and transactionally rolled
                    # back below; the continuous producer may retry a later
                    # frame, while a checkpoint without a valid fresh update
                    # fails closed. There is no same-frame lazy segmentation.
                    lost = pc_support < int(args.min_pc_support)
            elif not allow_fallback:
                raise RuntimeError(
                    f"probe request (allow_fallback=false) for {mesh_file!r} has no "
                    f"track state to refine from -- register it first")
            else:
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask,
                                    iteration=int(args.est_refine_iter))
                mask_iou = _gpu_mask_iou(est, reader.K, mask)

            if mask_iou is not None and mask_iou < float(args.track_min_iou):
                lost = True
            # A mask provides a calibrated geometric quality score.  A
            # maskless frame deliberately reports no confidence: point support
            # is only a translation-neighbourhood lost gate and must not be
            # relabelled as pose confidence.
            conf = mask_iou
            # A rejected update is transactional: the next frame retries from
            # the last accepted pose instead of compounding a bad refinement.
            if lost:
                est.pose_last = previous_pose_last

            debug_dir = str(req["debug_dir"])
            os.makedirs(os.path.join(debug_dir, "ob_in_cam"), exist_ok=True)
            pose_txt = os.path.join(debug_dir, "ob_in_cam", f"{reader.id_strs[0]}.txt")
            np.savetxt(pose_txt, np.asarray(pose).reshape(4, 4))
            print("FPWORKER " + json.dumps({
                "ok": True, "pose_txt": pose_txt, "mode_used": mode_used,
                "reregistered": reregistered,
                "confidence": (round(float(conf), 4) if conf is not None else None),
                "mask_iou": (
                    round(float(mask_iou), 4)
                    if mask_iou is not None
                    else None
                ),
                "pc_support": pc_support, "lost": lost,
                "capture_id": capture_id,
            }), flush=True)
            # A global register's 252-hypothesis batch spikes ~2.6 GB, so
            # release that exceptional cache before returning to tracking.
            # Normal track_one calls deliberately keep the allocator warm:
            # empty_cache() after every frame synchronizes/releases reusable
            # blocks and was a measured throughput anti-pattern.
            if _request_used_global_registration(mode_used, reregistered):
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - report and keep serving
            if active_est is not None:
                active_est.pose_last = previous_pose_last
            logging.exception("FoundationPose worker request failed")
            torch.cuda.empty_cache()
            print("FPWORKER " + json.dumps({"ok": False, "error": repr(exc)[:300]}), flush=True)


if __name__ == "__main__":
    main()
