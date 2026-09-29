#!/usr/bin/env python3
"""Estimate Phys2Real-style physical priors with a LOCAL Qwen3-VL model.

Stage-1 PRIORS payload (GPU half): runs under the QWEN env interpreter
(transformers + torch) and must not import ``oap``. The VLM looks at the
object crop and returns broad, uncertain density/friction RANGES plus a
material/category guess as strict JSON. All clamping and the mass-from-volume
derivation happen in-process in the priors step -- this payload only produces
the raw visual estimate, and it never reads any hidden asset metadata.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="RGB observation or object crop seen by the VLM.")
    parser.add_argument("--out", required=True, help="Output JSON: {model_id, raw_text, estimate}.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Local transformers model id (e.g. Qwen/Qwen3-VL-4B-Instruct).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--object-label", default="unknown tabletop object")
    parser.add_argument("--task", default="Estimate broad physical priors for reconstructing this object in MuJoCo.")
    parser.add_argument("--size-xyz-m", type=float, nargs=3, required=True)
    parser.add_argument(
        "--scene-hint",
        default=(
            "The image is a real RGB-D tabletop observation. The target object is segmented "
            "and reconstructed separately with SAM3D; use visual cues only for material/category."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = load_resized_image(image_path, max_side=int(args.max_side))
    prompt = build_prompt(args)
    raw_text = run_qwen3vl(
        model_id=str(args.model_id),
        image=image,
        prompt=prompt,
        max_new_tokens=int(args.max_new_tokens),
    )
    estimate = parse_json_object(raw_text)
    payload = {
        "schema": "qwen3vl_physical_prior_estimate_response_v1",
        "backend": "qwen3vl_local",
        "model_id": str(args.model_id),
        "input_image": str(image_path),
        "object_label": str(args.object_label),
        "size_xyz_m": [float(v) for v in args.size_xyz_m],
        "raw_text": raw_text,
        "estimate": estimate,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "model_id": args.model_id}, indent=2))


def build_prompt(args: argparse.Namespace) -> str:
    size = [float(v) for v in args.size_xyz_m]
    volume = size[0] * size[1] * size[2]
    return f"""
You are estimating broad uncertain physical priors for a reconstructed tabletop object.
Look at the image and return JSON only, with no markdown.

Task:
{args.task}

Scene hint:
{args.scene_hint}

Target object label:
{args.object_label}

Metric reconstructed size from RGB-D/SAM3D, in meters:
{size}

Metric reconstructed bounding-box volume, in cubic meters:
{volume:.9g}

Return exactly this schema:
{{
  "schema_version": "countersim.qwen3vl_physical_prior_estimate.v1",
  "source": "qwen3vl_visual_physical_prior_estimate",
  "object_category": "best visible category, e.g. plastic cube, milk carton, metal can, wooden block, unknown",
  "material": "best visible material guess: foam/cardboard/paper/plastic/rubber/wood/metal/ceramic/unknown",
  "material_confidence": 0.0,
  "density_kg_m3": {{"range": [low, high], "nominal": mid, "reason": "brief reason"}},
  "friction_table": {{"range": [low, high], "nominal": mid, "reason": "brief reason for table contact"}},
  "friction_gripper": {{"range": [low, high], "nominal": mid, "reason": "brief reason for rubber finger pads"}},
  "wall_thickness_m": {{"range": [low, high], "nominal": mid, "reason": "wall/shell thickness"}},
  "mass_note": "Do not guess mass directly. The local code computes mass = density * min(bbox_volume, bbox_surface_area * wall_thickness_m), so wall_thickness_m is what decides whether this object is treated as a shell or as a solid block.",
  "uncertainty_note": "This is a visual prior, not measured truth."
}}

Rules:
- Use broad ranges, not precise measurements.
- Do not use hidden simulator metadata.
- Do not output robot actions, controller parameters, or task primitives.
- If uncertain, choose unknown material and wider ranges.
- Density must be in kg/m^3 and friction coefficients are dimensionless.
- wall_thickness_m is the thickness of the MATERIAL itself, in meters: about
  0.001-0.002 for a plastic bottle or drinks can, 0.003-0.005 for corrugated
  cardboard, 0.0005 for thin sheet metal. Most tabletop objects are SHELLS
  that enclose air, and the density you give is the material's, not the
  object's average -- the thickness is what reconciles the two.
- If the object is genuinely SOLID all the way through (a wooden block, a
  steel weight, a bar of soap), report wall_thickness_m as HALF ITS SMALLEST
  DIMENSION. That makes the formula above select the full bounding volume,
  which is the correct solid answer.
""".strip()


def run_qwen3vl(*, model_id: str, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=False,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_ids = generated[0, inputs["input_ids"].shape[-1] :]
    return processor.decode(output_ids, skip_special_tokens=True).strip()


def load_resized_image(path: Path, *, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
    return image


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


if __name__ == "__main__":
    main()
