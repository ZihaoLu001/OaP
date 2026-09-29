"""Step MESH: SAM3D object reconstruction with baked-texture fail-closed gate.

Role in the two-stage pipeline: turns the imported observation packet into a
SAM3D shape. In-process this module only prepares the SAM3D crop input
(``image.png`` + ``0.png``) from the packet's RGB + SAM3 mask; the actual
reconstruction runs in the external SAM3D env via
``payloads/sam3d_reconstruct.py`` and yields ``mesh.glb`` (UV-baked texture),
``splat.ply``, optionally ``mesh_raw.obj`` (low-memory path), and
``metadata.json``.

TEXTURE FAIL-CLOSED: the baked Gaussian texture is a fixed part of this
pipeline (the twin renders real appearance, and FoundationPose needs it to
localize). A texture-baking run whose metadata reports ``textured: false``
raises instead of silently producing a geometry-only asset. The gsplat
CUDA extension must be built for the target GPU architecture.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from oap.reconstruct.external import ExternalEnvs
from oap.utils.io import read_json

logger = logging.getLogger("oap.reconstruct.mesh")

__all__ = ["prepare_sam3d_crop", "run_mesh_step"]


def prepare_sam3d_crop(
    image_path: Path,
    mask_path: Path,
    out_dir: Path,
    *,
    output_size: int = 512,
    margin_px: int = 48,
    name: str = "unknown_object",
) -> dict[str, Any]:
    """Create the SAM3D ``image.png`` / ``0.png`` input from an RGB + mask.

    The crop keeps the input tied to the observed RGB frame while giving SAM3D
    enough pixels for a small tabletop object: a square, mask-centered crop
    with margin, padded when it exceeds the frame, resized to ``output_size``.
    The written ``metadata.json`` records ``crop_box_xyxy_in_padded_source`` +
    ``output_size`` so the scale step can map crop pixels back to full-frame
    RGB-D pixels.

    Args:
        image_path: Full-frame RGB (the packet's ``rgb.png``).
        mask_path: Object mask (the packet's ``mask_00.png``).
        out_dir: Output directory for ``image.png``/``0.png``/``rgba.png``/
            ``overlay.png``/``metadata.json``.
        output_size: Side length of the square crop output.
        margin_px: Margin around the mask bounding box before squaring.
        name: Object name recorded in the metadata.

    Returns:
        The crop metadata payload.

    Raises:
        RuntimeError: If the mask is empty.
    """
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask) > 0
    if not np.any(mask_arr):
        raise RuntimeError(f"Mask is empty: {mask_path}")

    ys, xs = np.where(mask_arr)
    x0 = max(0, int(xs.min()) - int(margin_px))
    y0 = max(0, int(ys.min()) - int(margin_px))
    x1 = min(image.width, int(xs.max()) + 1 + int(margin_px))
    y1 = min(image.height, int(ys.max()) + 1 + int(margin_px))

    # Make a square crop so SAM3D sees an object-centered input.
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0)
    x0 = int(round(cx - 0.5 * side))
    y0 = int(round(cy - 0.5 * side))
    x1 = x0 + side
    y1 = y0 + side
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - image.width)
    pad_b = max(0, y1 - image.height)
    if any((pad_l, pad_t, pad_r, pad_b)):
        padded = Image.new("RGB", (image.width + pad_l + pad_r, image.height + pad_t + pad_b), (0, 0, 0))
        padded.paste(image, (pad_l, pad_t))
        image = padded
        padded_mask = Image.new("L", image.size, 0)
        padded_mask.paste(mask, (pad_l, pad_t))
        mask = padded_mask
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t

    crop_box = (x0, y0, x1, y1)
    crop_rgb = image.crop(crop_box).resize((output_size, output_size), Image.Resampling.BICUBIC)
    crop_mask = mask.crop(crop_box).resize((output_size, output_size), Image.Resampling.NEAREST)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_rgb.save(out_dir / "image.png")
    crop_mask.save(out_dir / "0.png")
    rgba = crop_rgb.convert("RGBA")
    rgba.putalpha(crop_mask)
    rgba.save(out_dir / "rgba.png")

    overlay = crop_rgb.copy()
    overlay_arr = np.asarray(overlay).copy()
    sel = np.asarray(crop_mask) > 0
    overlay_arr[sel] = (0.35 * overlay_arr[sel].astype(np.float32) + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr, mode="RGB")
    draw = ImageDraw.Draw(overlay)
    m = np.asarray(crop_mask) > 0
    ys2, xs2 = np.where(m)
    draw.rectangle((int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max())), outline=(255, 0, 255), width=2)
    overlay.save(out_dir / "overlay.png")

    metadata = {
        "schema": "sam3d_crop_from_existing_mask_v1",
        "name": name,
        "source_image": str(Path(image_path).resolve()),
        "source_mask": str(Path(mask_path).resolve()),
        "crop_box_xyxy_in_padded_source": [int(v) for v in crop_box],
        "output_size": int(output_size),
        "source_image_size": [int(image.width), int(image.height)],
        "source_mask_pixels": int(np.count_nonzero(mask_arr)),
        "resized_mask_pixels": int(np.count_nonzero(np.asarray(crop_mask) > 0)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_mesh_step(
    step_dir: Path,
    packet_dir: Path,
    envs: ExternalEnvs,
    *,
    name: str,
    texture_baking: bool = True,
    mesh_only_lowmem: bool = False,
    mesh_extract_device: str = "cuda",
    max_sparse_coords: int | None = None,
    checkpoint_tag: str = "hf",
    seed: int = 42,
    allow_untextured: bool = False,
) -> dict[str, Any]:
    """Reconstruct one object's SAM3D shape from its imported packet.

    Prepares the crop input in-process, then invokes the SAM3D payload in the
    external env. FAILS CLOSED when texture baking was requested but the
    output metadata reports ``textured: false`` (unless ``allow_untextured``
    was explicitly passed for a documented low-memory fallback).

    Args:
        step_dir: This step's output dir (``<bundle>/<obj>/mesh``); receives
            ``sam3d_input/`` and ``sam3d_output/``.
        packet_dir: The imported observation packet.
        envs: Resolved external environments (needs the ``sam3d`` tool with a
            ``root`` pointing at the SAM-3D-Objects checkout + checkpoints).
        name: Object name (crop metadata provenance).
        texture_baking: Decode + bake the Gaussian texture (default ON; run on
            a larger GPU host -- a 16GB GPU can OOM during texture decode).
        mesh_only_lowmem: 16GB fallback -- mesh decoder only, raw vertex-color
            OBJ output, no texture.
        mesh_extract_device: FlexiCubes extraction device for the lowmem path.
        max_sparse_coords: Optional lowmem sparse-coordinate cap.
        checkpoint_tag: Checkpoint folder under ``<sam3d_root>/checkpoints``.
        seed: SAM3D sampling seed.
        allow_untextured: Tolerate a texture-less result (explicit opt-out).

    Returns:
        The SAM3D output ``metadata.json`` payload.

    Raises:
        ExternalEnvError: If the SAM3D env is missing or the payload fails.
        RuntimeError: On the texture fail-closed gate.
    """
    step_dir = Path(step_dir)
    packet_dir = Path(packet_dir)
    input_dir = step_dir / "sam3d_input"
    output_dir = step_dir / "sam3d_output"

    prepare_sam3d_crop(
        packet_dir / "rgb.png",
        packet_dir / "mask_00.png",
        input_dir,
        name=name,
    )

    sam3d_root = envs.extra("sam3d", "root")
    args: list[str | Path] = [
        "--sam3d-root", str(sam3d_root),
        "--input-dir", input_dir,
        "--out-dir", output_dir,
        "--checkpoint-tag", checkpoint_tag,
        "--seed", str(seed),
    ]
    if not texture_baking:
        args.append("--no-texture-baking")
    if mesh_only_lowmem:
        args += ["--mesh-only-lowmem", "--mesh-extract-device", mesh_extract_device]
        if max_sparse_coords is not None:
            args += ["--max-sparse-coords", str(max_sparse_coords)]
    envs.run_payload("sam3d", "sam3d_reconstruct", args)

    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"SAM3D payload finished but wrote no metadata: {metadata_path}")
    metadata: dict[str, Any] = read_json(metadata_path)

    # FAIL-CLOSED texture gate (also enforced inside the payload for the
    # texture-baking path; re-checked here so a stale/partial output dir can
    # never certify a textured bundle).
    if texture_baking and not mesh_only_lowmem and not metadata.get("textured"):
        if not allow_untextured:
            raise RuntimeError(
                f"mesh step for {name!r}: texture baking was requested but "
                f"{metadata_path} reports textured=false. Rebuild gsplat for the "
                f"target GPU architecture or pass --allow-untextured / "
                f"--mesh-only-lowmem for an explicit geometry-only fallback."
            )
        logger.warning("[mesh] %s: proceeding UNTEXTURED (explicitly allowed)", name)

    saved = metadata.get("saved", {})
    if not (saved.get("mesh_glb") or saved.get("mesh_obj")):
        raise RuntimeError(
            f"mesh step for {name!r} produced neither mesh.glb nor mesh_raw.obj "
            f"(see {metadata_path})"
        )
    logger.info("[mesh] %s: %s", name, {k: Path(v).name for k, v in saved.items()})
    return metadata
