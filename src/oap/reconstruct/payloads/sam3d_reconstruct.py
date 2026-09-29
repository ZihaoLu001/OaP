#!/usr/bin/env python3
"""Run Meta SAM 3D Objects on a prepared ``image.png`` + ``0.png`` input dir.

Stage-1 MESH payload: runs under the SAM3D env interpreter (torch + SAM3D +
gsplat + nvdiffrast) and must not import ``oap``. Decodes the SAM3D
Gaussian representation and bakes it into a UV texture (``mesh.glb``) plus the
Gaussian splat (``splat.ply``); a low-memory mesh-only path exports the raw
vertex-color mesh (``mesh_raw.obj``). Writes ``metadata.json`` with a
``textured`` flag the mesh step FAILS CLOSED on: a texture-baking run that
produces no baked UV texture exits non-zero rather than silently emitting a
geometry-only twin.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Meta SAM 3D Objects on a prepared image.png + mask 0.png input directory."
    )
    parser.add_argument("--sam3d-root", required=True, help="Path to facebookresearch/sam-3d-objects clone.")
    parser.add_argument("--input-dir", required=True, help="Directory containing image.png and 0.png.")
    parser.add_argument("--out-dir", required=True, help="Output directory for splat/mesh/metadata artifacts.")
    parser.add_argument("--checkpoint-tag", default="hf", help="Checkpoint folder under SAM3D checkpoints/.")
    parser.add_argument("--mask-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--texture-baking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode SAM3D's Gaussian representation and bake it into a UV texture "
            "(with_texture_baking=True) so the metric mesh carries real appearance for "
            "the MuJoCo twin. DEFAULT ON -- use a sufficiently large GPU; the full "
            "Gaussian/texture decode can OOM a 16GB GPU, where you would use "
            "--no-texture-baking (falls back to vertex colors) + --mesh-only-lowmem."
        ),
    )
    parser.add_argument(
        "--texture-render-backend",
        choices=("auto", "inria", "gsplat"),
        default="auto",
        help=(
            "Gaussian renderer used for SAM3D texture-baking multiview renders. "
            "auto uses the official inria rasterizer when installed, otherwise "
            "falls back to gsplat, which is part of SAM3D's inference requirements."
        ),
    )
    parser.add_argument(
        "--texture-bake-engine",
        choices=("auto", "nvdiffrast", "pytorch3d"),
        default="auto",
        help=(
            "Texture optimization renderer. SAM3D's UV rasterization still requires "
            "nvdiffrast; auto selects nvdiffrast when available and otherwise fails "
            "with a clear dependency error."
        ),
    )
    parser.add_argument(
        "--no-vertex-color",
        action="store_true",
        help="Pass use_vertex_color=False to the official SAM3D pipeline.",
    )
    parser.add_argument(
        "--mesh-only-lowmem",
        action="store_true",
        help=(
            "Run the official SAM3D mesh path but skip Gaussian/texture decoding "
            "and export the raw vertex-color mesh as OBJ. This is intended for "
            "memory-constrained lab GPUs where the full GLB path can OOM."
        ),
    )
    parser.add_argument(
        "--max-sparse-coords",
        type=int,
        default=None,
        help=(
            "Optional low-memory cap for SAM3D sparse structure coordinates before "
            "the mesh decoder. Only used with --mesh-only-lowmem."
        ),
    )
    parser.add_argument(
        "--mesh-extract-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Device for the final FlexiCubes mesh extraction in --mesh-only-lowmem. "
            "CPU keeps SAM3D inference real while avoiding GPU OOM in the final "
            "dense mesh extraction step."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


def run(args) -> None:
    sam3d_root = Path(args.sam3d_root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = sam3d_root / "checkpoints" / args.checkpoint_tag / "pipeline.yaml"
    image_path = input_dir / "image.png"
    mask_path = input_dir / f"{args.mask_index}.png"
    if not config_path.exists():
        raise FileNotFoundError(f"SAM3D checkpoint config not found: {config_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"image.png not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    _prepare_sam3d_import_runtime()
    sys.path.insert(0, str(sam3d_root / "notebook"))
    from inference import Inference, load_image, load_single_mask  # type: ignore

    start = time.perf_counter()
    inference = Inference(str(config_path), compile=bool(args.compile))
    image = load_image(str(image_path))
    mask = load_single_mask(str(input_dir), index=int(args.mask_index))
    output = _run_inference(
        inference=inference,
        image=image,
        mask=mask,
        seed=int(args.seed),
        with_texture_baking=bool(args.texture_baking),
        use_vertex_color=not bool(args.no_vertex_color),
        texture_render_backend=args.texture_render_backend,
        texture_bake_engine=args.texture_bake_engine,
        mesh_only_lowmem=bool(args.mesh_only_lowmem),
        max_sparse_coords=args.max_sparse_coords,
        mesh_extract_device=str(args.mesh_extract_device),
    )
    elapsed_s = time.perf_counter() - start

    saved: dict[str, str] = {}
    gs = _find_gaussian(output)
    if gs is not None:
        ply_path = out_dir / "splat.ply"
        gs.save_ply(str(ply_path))
        saved["splat_ply"] = str(ply_path)

    glb = output.get("glb") if isinstance(output, dict) else None
    if glb is not None:
        glb_path = out_dir / "mesh.glb"
        if isinstance(glb, (str, Path)) and Path(glb).exists():
            shutil.copyfile(glb, glb_path)
        elif hasattr(glb, "export"):
            glb.export(str(glb_path))
        else:
            glb_path.write_bytes(bytes(glb))
        saved["mesh_glb"] = str(glb_path)

    mesh_obj = output.get("mesh_obj") if isinstance(output, dict) else None
    if mesh_obj is not None:
        obj_path = out_dir / "mesh_raw.obj"
        _export_sam3d_mesh_obj(mesh_obj, obj_path)
        saved["mesh_obj"] = str(obj_path)

    # Verify the baked Gaussian texture actually landed on the mesh -- the texture is
    # a FIXED part of this pipeline, so we fail closed if it is missing (rather than
    # silently emitting a geometry-only twin).
    textured, tex_size = False, None
    if saved.get("mesh_glb"):
        try:
            import trimesh as _tm
            _sc = _tm.load(saved["mesh_glb"], force=None)
            _m = list(_sc.geometry.values())[0] if isinstance(_sc, _tm.Scene) else _sc
            _v = getattr(_m, "visual", None)
            _uv = getattr(_v, "uv", None)
            _mat = getattr(_v, "material", None)
            _img = getattr(_mat, "baseColorTexture", None) or getattr(_mat, "image", None)
            textured = _uv is not None and len(_uv) > 0 and _img is not None
            tex_size = list(_img.size) if _img is not None else None
        except Exception:
            textured = False

    metadata = {
        "ok": True,
        "elapsed_s": elapsed_s,
        "sam3d_root": str(sam3d_root),
        "input_dir": str(input_dir),
        "config_path": str(config_path),
        "seed": int(args.seed),
        "with_texture_baking": bool(args.texture_baking),
        "use_vertex_color": not bool(args.no_vertex_color),
        "mesh_only_lowmem": bool(args.mesh_only_lowmem),
        "textured": textured,
        "texture_size": tex_size,
        "texture_runtime": output.pop("_oap_texture_runtime", None)
        if isinstance(output, dict)
        else None,
        "output_keys": sorted([str(k) for k in output.keys()]) if isinstance(output, dict) else [],
        "saved": saved,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

    if bool(args.texture_baking) and not bool(args.mesh_only_lowmem) and not textured:
        raise SystemExit(
            "FAIL-CLOSED: --texture-baking was on but the SAM3D mesh carries no baked UV "
            "texture. Most likely gsplat's CUDA kernels do not match the GPU arch "
            "('no kernel image is available'): rebuild gsplat for the target arch, or use "
            "--no-texture-baking for a geometry-only fallback.")


def _find_gaussian(output: dict[str, Any]):
    if "gs" in output:
        return output["gs"]
    if "gaussian" in output:
        gaussian = output["gaussian"]
        if isinstance(gaussian, (list, tuple)) and gaussian:
            return gaussian[0]
        return gaussian
    return None


def _run_inference(
    *,
    inference: Any,
    image: Any,
    mask: Any,
    seed: int,
    with_texture_baking: bool,
    use_vertex_color: bool,
    texture_render_backend: str,
    texture_bake_engine: str,
    mesh_only_lowmem: bool,
    max_sparse_coords: int | None,
    mesh_extract_device: str,
) -> dict[str, Any]:
    if mesh_only_lowmem:
        return _run_mesh_only_lowmem(
            inference=inference,
            image=image,
            mask=mask,
            seed=seed,
            max_sparse_coords=max_sparse_coords,
            mesh_extract_device=mesh_extract_device,
        )

    if not with_texture_baking and use_vertex_color:
        return inference(image, mask, seed=seed)

    if not hasattr(inference, "_pipeline") or not hasattr(inference, "merge_mask_to_rgba"):
        raise RuntimeError("SAM3D Inference object does not expose the official pipeline API.")

    texture_runtime = None
    if with_texture_baking:
        texture_runtime = _configure_texture_baking_runtime(
            inference=inference,
            texture_render_backend=texture_render_backend,
            texture_bake_engine=texture_bake_engine,
        )

    rgba = inference.merge_mask_to_rgba(image, mask)
    output = inference._pipeline.run(
        rgba,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=with_texture_baking,
        with_layout_postprocess=False,
        use_vertex_color=use_vertex_color,
        stage1_inference_steps=None,
        pointmap=None,
    )
    if isinstance(output, dict) and texture_runtime is not None:
        output["_oap_texture_runtime"] = texture_runtime
    return output


def _run_mesh_only_lowmem(
    *,
    inference: Any,
    image: Any,
    mask: Any,
    seed: int,
    max_sparse_coords: int | None,
    mesh_extract_device: str,
) -> dict[str, Any]:
    """Run the official SAM3D pipeline with only the mesh decoder active.

    The public SAM3D notebook path decodes mesh + Gaussian and then turns them
    into a textured GLB. That is useful for demos, but it needlessly keeps the
    Gaussian decoders resident when this pipeline only needs a visual mesh for
    the reconstructed twin. This mode keeps the core SAM3D geometry model
    unchanged, decodes the mesh, and exports the raw vertex-color mesh.
    """

    if not hasattr(inference, "_pipeline") or not hasattr(inference, "merge_mask_to_rgba"):
        raise RuntimeError("SAM3D Inference object does not expose the official pipeline API.")

    _release_optional_sam3d_decoders(inference._pipeline)
    if max_sparse_coords is not None:
        _patch_sparse_coord_cap(inference._pipeline, max_sparse_coords=max_sparse_coords)
    if mesh_extract_device == "cpu":
        _patch_mesh_extraction_to_cpu(inference._pipeline)

    def postprocess_mesh_only(outputs: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        meshes = outputs.get("mesh") or []
        outputs["mesh_obj"] = meshes[0] if meshes else None
        outputs["glb"] = None
        return outputs

    inference._pipeline.postprocess_slat_output = postprocess_mesh_only
    rgba = inference.merge_mask_to_rgba(image, mask)
    output = inference._pipeline.run(
        rgba,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=False,
        with_texture_baking=False,
        with_layout_postprocess=False,
        use_vertex_color=True,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        pointmap=None,
        decode_formats=["mesh"],
    )
    output["_oap_mesh_only_lowmem"] = True
    output["_oap_max_sparse_coords"] = max_sparse_coords
    output["_oap_mesh_extract_device"] = mesh_extract_device
    return output


def _patch_mesh_extraction_to_cpu(pipeline: Any) -> None:
    """Run only the final SAM3D mesh extraction step on CPU.

    The neural sparse-structure and sparse-latent stages still run normally on
    the configured SAM3D device. The final FlexiCubes conversion creates a dense
    grid and can exceed a 16GB lab GPU memory budget; moving that conversion
    to CPU keeps the reconstruction path genuine while avoiding a GPU-only
    allocation failure.
    """

    models = getattr(pipeline, "models", None)
    if models is None or "slat_decoder_mesh" not in models:
        raise RuntimeError("SAM3D pipeline has no slat_decoder_mesh model to patch.")
    decoder = models["slat_decoder_mesh"]
    try:
        from sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh import (  # type: ignore
            SparseFeatures2Mesh,
        )
    except Exception as exc:
        raise RuntimeError("Could not import SAM3D SparseFeatures2Mesh for CPU patch") from exc

    use_color = bool(getattr(decoder, "rep_config", {}).get("use_color", False))
    decoder.mesh_extractor = SparseFeatures2Mesh(
        res=int(decoder.resolution) * 4,
        use_color=use_color,
        device="cpu",
    )
    original_to_representation = decoder.__class__.to_representation

    def to_representation_cpu(self: Any, x: Any) -> list[Any]:
        return original_to_representation(self, x.cpu())

    decoder.to_representation = types.MethodType(to_representation_cpu, decoder)


def _patch_sparse_coord_cap(pipeline: Any, *, max_sparse_coords: int) -> None:
    if max_sparse_coords <= 0:
        raise ValueError("--max-sparse-coords must be positive.")

    import torch
    from sam3d_objects.pipeline.inference_utils import downsample_sparse_structure

    original = pipeline.sample_sparse_structure

    def capped_sample_sparse_structure(*args: Any, **kwargs: Any) -> dict[str, Any]:
        ret = original(*args, **kwargs)
        coords = ret.get("coords")
        if coords is None or int(coords.shape[0]) <= max_sparse_coords:
            ret["_coord_cap_before"] = int(coords.shape[0]) if coords is not None else None
            ret["_coord_cap_after"] = int(coords.shape[0]) if coords is not None else None
            return ret

        capped, extra_factor = downsample_sparse_structure(
            coords,
            max_coords=int(max_sparse_coords),
            downsample_factor=2,
        )
        if int(capped.shape[0]) > max_sparse_coords:
            generator = torch.Generator(device=capped.device)
            generator.manual_seed(0)
            idx = torch.randperm(capped.shape[0], generator=generator, device=capped.device)[
                : int(max_sparse_coords)
            ]
            capped = capped[idx]

        ret["coords"] = capped.int()
        ret["downsample_factor"] = ret.get("downsample_factor", 1) * int(extra_factor)
        ret["_coord_cap_before"] = int(coords.shape[0])
        ret["_coord_cap_after"] = int(capped.shape[0])
        return ret

    pipeline.sample_sparse_structure = capped_sample_sparse_structure


def _release_optional_sam3d_decoders(pipeline: Any) -> None:
    try:
        import torch
    except Exception:
        torch = None

    models = getattr(pipeline, "models", None)
    if models is not None:
        for name in ("slat_decoder_gs", "slat_decoder_gs_4"):
            if name in models:
                models[name].to("cpu")
                del models[name]
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _export_sam3d_mesh_obj(mesh: Any, obj_path: Path) -> None:
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().long().cpu().numpy()
    attrs = getattr(mesh, "vertex_attrs", None)
    colors = None
    if attrs is not None:
        colors = attrs.detach().float().cpu().numpy()
        if colors.ndim == 2 and colors.shape[1] >= 3:
            colors = np.clip(colors[:, :3], 0.0, 1.0)
        else:
            colors = None

    with obj_path.open("w", encoding="utf-8") as f:
        f.write("# SAM3D mesh-only-lowmem raw vertex-color OBJ\n")
        for i, v in enumerate(vertices):
            if colors is not None:
                c = colors[i]
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _prepare_sam3d_import_runtime() -> None:
    """Make the official SAM3D notebook wrapper importable in headless runs.

    The official `sam3d_objects` package imports `sam3d_objects.init` from its
    top-level `__init__`, but the public inference checkout may not contain that
    module. Upstream already provides the `LIDRA_SKIP_INIT` escape hatch for
    lightweight tools, so we set it before importing the notebook wrapper.

    The notebook also imports Kaolin's interactive visualizer for notebooks.
    That dependency is not part of the reconstruction path used here, so when
    Kaolin is absent we provide a tiny module stub for the visualizer import
    instead of forcing a heavyweight visualization package into the runtime.
    """

    os.environ.setdefault("LIDRA_SKIP_INIT", "1")
    # The official SAM3D notebook wrapper assumes a shell-activated conda env
    # and reads CONDA_PREFIX while importing. Slurm jobs often call the Python
    # executable directly, so provide the same value from sys.prefix.
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    _patch_utils3d_depth_edge_if_needed()

    if (
        _module_available("kaolin.visualize")
        and _module_available("kaolin.render.camera")
        and _module_available("kaolin.utils.testing")
    ):
        return

    kaolin_module = sys.modules.setdefault("kaolin", types.ModuleType("kaolin"))
    kaolin_module.__path__ = getattr(kaolin_module, "__path__", [])
    visualize_module = types.ModuleType("kaolin.visualize")
    render_module = types.ModuleType("kaolin.render")
    camera_module = types.ModuleType("kaolin.render.camera")
    utils_module = types.ModuleType("kaolin.utils")
    testing_module = types.ModuleType("kaolin.utils.testing")

    class IpyTurntableVisualizer:  # pragma: no cover - import compatibility shim
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class _KaolinCameraStub:  # pragma: no cover - notebook import shim
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    def check_tensor(tensor: Any, shape: tuple[Any, ...], throw: bool = True) -> bool:
        """Small subset of Kaolin's shape checker used by SAM3D FlexiCubes."""

        actual_shape = tuple(getattr(tensor, "shape", ()))
        ok = len(actual_shape) == len(shape) and all(
            expected is None or int(actual) == int(expected)
            for actual, expected in zip(actual_shape, shape)
        )
        if throw and not ok:
            raise AssertionError(f"Expected tensor shape {shape}, got {actual_shape}")
        return ok

    # Dynamically populate the shim modules (ModuleType has no static attrs).
    setattr(visualize_module, "IpyTurntableVisualizer", IpyTurntableVisualizer)
    setattr(camera_module, "Camera", _KaolinCameraStub)
    setattr(camera_module, "CameraExtrinsics", _KaolinCameraStub)
    setattr(camera_module, "PinholeIntrinsics", _KaolinCameraStub)
    setattr(testing_module, "check_tensor", check_tensor)
    setattr(kaolin_module, "visualize", visualize_module)
    setattr(kaolin_module, "render", render_module)
    setattr(kaolin_module, "utils", utils_module)
    setattr(render_module, "camera", camera_module)
    setattr(utils_module, "testing", testing_module)
    sys.modules.setdefault("kaolin.visualize", visualize_module)
    sys.modules.setdefault("kaolin.render", render_module)
    sys.modules.setdefault("kaolin.render.camera", camera_module)
    sys.modules.setdefault("kaolin.utils", utils_module)
    sys.modules.setdefault("kaolin.utils.testing", testing_module)


def _patch_utils3d_depth_edge_if_needed() -> None:
    """Provide the depth-edge helper expected by SAM3D if the installed
    `utils3d` package does not expose it.

    SAM3D uses this only for mask/depth edge cleanup during layout utilities.
    The implementation below follows the same contract: return a boolean mask
    where local finite depth jumps exceed a relative threshold.
    """

    try:
        import utils3d.numpy as utils3d_numpy  # type: ignore
    except Exception:
        return
    if hasattr(utils3d_numpy, "depth_edge"):
        return

    def depth_edge(depth: Any, rtol: float = 0.03, mask: Any | None = None) -> np.ndarray:
        depth_arr = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth_arr) & (depth_arr > 0)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)

        edge = np.zeros(depth_arr.shape, dtype=bool)
        for axis in (0, 1):
            diff = np.abs(np.diff(depth_arr, axis=axis))
            a = np.take(depth_arr, indices=range(depth_arr.shape[axis] - 1), axis=axis)
            b = np.take(depth_arr, indices=range(1, depth_arr.shape[axis]), axis=axis)
            ref = np.maximum(np.minimum(np.abs(a), np.abs(b)), 1e-6)
            valid_pair = (
                np.take(valid, indices=range(depth_arr.shape[axis] - 1), axis=axis)
                & np.take(valid, indices=range(1, depth_arr.shape[axis]), axis=axis)
            )
            pair_edge = valid_pair & (diff > (float(rtol) * ref))

            left_slice = [slice(None)] * depth_arr.ndim
            right_slice = [slice(None)] * depth_arr.ndim
            left_slice[axis] = slice(0, -1)
            right_slice[axis] = slice(1, None)
            edge[tuple(left_slice)] |= pair_edge
            edge[tuple(right_slice)] |= pair_edge
        return edge

    utils3d_numpy.depth_edge = depth_edge


def _configure_texture_baking_runtime(
    *,
    inference: Any,
    texture_render_backend: str,
    texture_bake_engine: str,
) -> dict[str, Any]:
    """Make SAM3D texture baking work in the cluster inference environment.

    SAM3D's official requirements include `gsplat`, but the texture-baking
    multiview path defaults to the optional Inria
    `diff_gaussian_rasterization` backend. If that optional module is absent,
    upstream only emits an ImportWarning and later fails with
    `NameError: GaussianRasterizationSettings is not defined`.

    We keep the upstream clone untouched and patch the runtime entry points from
    our wrapper: multiview Gaussian rendering falls back to `gsplat`, and texture
    optimization requires `nvdiffrast`, because SAM3D uses it to rasterize UV maps
    even when its later texture-sampling branch is set to `pytorch3d`.
    """

    has_inria = _module_available("diff_gaussian_rasterization")
    has_gsplat = _module_available("gsplat")
    has_nvdiffrast = _module_available("nvdiffrast.torch")

    render_backend, render_reason = _choose_texture_render_backend(
        requested=texture_render_backend,
        has_inria=has_inria,
        has_gsplat=has_gsplat,
    )
    bake_engine, bake_reason = _choose_texture_bake_engine(
        requested=texture_bake_engine,
        has_nvdiffrast=has_nvdiffrast,
    )

    pipeline = inference._pipeline
    previous_bake_engine = getattr(pipeline, "rendering_engine", None)
    setattr(pipeline, "rendering_engine", bake_engine)

    patched_multiview = False
    patched_gsplat_nonpacked = False
    if render_backend == "gsplat":
        _patch_sam3d_render_multiview(render_backend, force_gsplat_nonpacked=True)
        patched_multiview = True
        patched_gsplat_nonpacked = True

    return {
        "texture_render_backend": render_backend,
        "texture_render_backend_reason": render_reason,
        "texture_bake_engine": bake_engine,
        "texture_bake_engine_reason": bake_reason,
        "previous_pipeline_rendering_engine": previous_bake_engine,
        "patched_postprocessing_render_multiview": patched_multiview,
        "patched_gsplat_nonpacked_projection": patched_gsplat_nonpacked,
        "dependency_available": {
            "diff_gaussian_rasterization": has_inria,
            "gsplat": has_gsplat,
            "nvdiffrast.torch": has_nvdiffrast,
        },
    }


def _choose_texture_render_backend(
    *,
    requested: str,
    has_inria: bool,
    has_gsplat: bool,
) -> tuple[str, str]:
    if requested == "inria":
        if not has_inria:
            raise RuntimeError(
                "Requested --texture-render-backend inria, but "
                "diff_gaussian_rasterization is not installed."
            )
        return "inria", "requested_inria"
    if requested == "gsplat":
        if not has_gsplat:
            raise RuntimeError("Requested --texture-render-backend gsplat, but gsplat is not installed.")
        return "gsplat", "requested_gsplat"
    if has_inria:
        return "inria", "auto_selected_official_inria_backend"
    if has_gsplat:
        return "gsplat", "auto_fallback_diff_gaussian_rasterization_missing"
    raise RuntimeError(
        "SAM3D texture baking needs either diff_gaussian_rasterization or gsplat for multiview rendering."
    )


def _choose_texture_bake_engine(*, requested: str, has_nvdiffrast: bool) -> tuple[str, str]:
    if requested == "nvdiffrast":
        if not has_nvdiffrast:
            raise RuntimeError("Requested --texture-bake-engine nvdiffrast, but nvdiffrast.torch is not installed.")
        return "nvdiffrast", "requested_nvdiffrast"
    if requested == "pytorch3d":
        if not has_nvdiffrast:
            raise RuntimeError(
                "Requested --texture-bake-engine pytorch3d, but SAM3D still requires "
                "nvdiffrast.torch for UV rasterization before texture sampling."
            )
        return "pytorch3d", "requested_pytorch3d"
    if has_nvdiffrast:
        return "nvdiffrast", "auto_selected_nvdiffrast"
    raise RuntimeError(
        "SAM3D texture baking requires nvdiffrast.torch for UV rasterization. "
        "Install it in the SAM3D environment, for example with "
        "`CUDA_HOME=/cm/shared/apps/cuda12.2/toolkit/12.2.2 "
        "TORCH_CUDA_ARCH_LIST='8.0;9.0' python -m pip install --no-build-isolation "
        "git+https://github.com/NVlabs/nvdiffrast.git`."
    )


def _patch_sam3d_render_multiview(render_backend: str, *, force_gsplat_nonpacked: bool = False) -> None:
    from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils, render_utils  # type: ignore
    from sam3d_objects.model.backbone.tdfy_dit.utils.random_utils import sphere_hammersley_sequence  # type: ignore

    if render_backend == "gsplat" and force_gsplat_nonpacked:
        from sam3d_objects.model.backbone.tdfy_dit.renderers import gaussian_render  # type: ignore

        original_rasterization = gaussian_render.rasterization

        def rasterization_nonpacked(*args, **kwargs):
            kwargs.setdefault("packed", False)
            bg = kwargs.get("backgrounds")
            if bg is not None and getattr(bg, "ndim", None) == 1:
                kwargs["backgrounds"] = bg.unsqueeze(0)
            return original_rasterization(*args, **kwargs)

        gaussian_render.rasterization = rasterization_nonpacked

    def render_multiview_with_backend(sample, resolution=512, nviews=30):
        r = 2
        fov = 40
        cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitchs, r, fov
        )
        res = render_utils.render_frames(
            sample,
            extrinsics,
            intrinsics,
            {"resolution": resolution, "bg_color": (0, 0, 0), "backend": render_backend},
        )
        return res["color"], extrinsics, intrinsics

    postprocessing_utils.render_multiview = render_multiview_with_backend


if __name__ == "__main__":
    main()
