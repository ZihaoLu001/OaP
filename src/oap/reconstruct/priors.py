"""Step PRIORS: bounded VLM physical priors (density/friction intervals + mass).

Role in the two-stage pipeline: the twin needs mass/friction to simulate the
object, but no sensor measures them -- so a VLM (local Qwen3-VL via
``payloads/qwen_priors.py``) estimates the visible material/category and BROAD
density/friction RANGES from the object crop. The in-process half clamps every
estimate to absolute physical bounds, derives mass from the RECONSTRUCTED
metric volume (mass = volume x density prior -- never a direct VLM mass guess),
and writes ``<obj>_priors.json`` (schema
``countersim.phys2real_vlm_physical_priors.v1``) plus the raw VLM response for
audit. Priors are bounded INTERVALS, not point estimates, and this step never
reads hidden asset metadata.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


from oap.reconstruct.external import ExternalEnvs
from oap.utils.io import read_json, write_json_atomic

import numpy as np

from oap.physical_priors import (  # the ONE implementation
    MATERIAL_DENSITY_PRIORS,
    MATERIAL_TABLE_FRICTION_PRIORS,
    PhysicalPriorReport,
    physical_prior_report_from_vlm_payload,
)
logger = logging.getLogger("oap.reconstruct.priors")

__all__ = [
    "MATERIAL_DENSITY_PRIORS",
    "MATERIAL_TABLE_FRICTION_PRIORS",
    "PhysicalPriorReport",
    "physical_prior_report_from_vlm_payload",
    "run_priors_step",
]
















def _aggregate_estimates(estimates: list[dict[str, Any]]) -> dict[str, Any]:
    """Median-of-nominals, union-of-ranges across repeated VLM replies.

    The median resists one wild reading (the run that decided a blackboard
    eraser was a smartphone); the union keeps the interval honest rather than
    letting agreement between two samples masquerade as confidence. Fields the
    VLM returns as plain scalars or strings take the FIRST reply's value, and
    the material takes the MODE -- a category is not averageable.
    """
    import collections

    if len(estimates) == 1:
        return estimates[0]
    out: dict[str, Any] = dict(estimates[0])
    mats = [str(e.get("material")) for e in estimates if e.get("material")]
    if mats:
        out["material"] = collections.Counter(mats).most_common(1)[0][0]
    for key in ("density_kg_m3", "friction_table", "friction_gripper",
                "wall_thickness_m"):
        dicts = [e[key] for e in estimates
                 if isinstance(e.get(key), dict)]
        if not dicts:
            continue
        noms, los, his = [], [], []
        for d in dicts:
            if d.get("nominal") is not None:
                noms.append(float(d["nominal"]))
            rng = d.get("range")
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                los.append(float(rng[0])); his.append(float(rng[1]))
        merged = dict(dicts[0])
        if noms:
            merged["nominal"] = float(np.median(noms))
        if los and his:
            merged["range"] = [float(min(los)), float(max(his))]
        merged["n_samples"] = len(dicts)
        out[key] = merged
    return out


def run_priors_step(
    step_dir: Path,
    image: Path,
    size_xyz_m: tuple[float, float, float],
    envs: ExternalEnvs,
    *,
    name: str,
    object_label: str,
    model_id: str | None = None,
    mask: Path | None = None,
    task: str = "Estimate broad physical priors for reconstructing this object in MuJoCo.",
    max_new_tokens: int = 512,
    max_side: int = 640,
) -> dict[str, Any]:
    """Estimate one object's physical priors and write ``<name>_priors.json``.

    Invokes the qwen payload (local Qwen3-VL) for the raw visual estimate,
    then clamps/derives the bounded prior report in-process. The raw VLM text
    is preserved next to the report (``.raw.txt``) for audit.

    Args:
        step_dir: This step's output dir (``<bundle>/<obj>/priors``).
        image: The object crop the VLM sees (the mesh step's ``image.png``).
        size_xyz_m: Reconstructed metric size (mass = volume x density prior).
        envs: Resolved external environments (needs the ``qwen`` tool).
        name: Object name (output file stem).
        object_label: Human label passed to the VLM prompt.
        model_id: Transformers model id (default: the profile's ``model_id``,
            else Qwen/Qwen3-VL-4B-Instruct).
        mask: Optional mask path recorded for provenance.
        task: Task text embedded in the prompt.
        max_new_tokens: VLM generation budget.
        max_side: Image resize bound before the VLM sees it.

    Returns:
        The priors payload (schema
        ``countersim.phys2real_vlm_physical_priors.v1``).
    """
    step_dir = Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    response_json = step_dir / f"{name}_vlm_response.json"

    # ONE VLM (oap.vlm). The local Qwen3-VL payload survives as the
    # ablation arm behind OAP_PRIORS_BACKEND=qwen: it needs an external
    # transformers environment and pays a ~10-12 s model load per object,
    # neither of which an API call has, so it is no longer the default.
    backend = (os.environ.get("OAP_PRIORS_BACKEND") or "claude").strip().lower()
    if backend not in ("claude", "qwen"):
        raise ValueError(
            f"OAP_PRIORS_BACKEND must be 'claude' or 'qwen', got {backend!r}")

    if backend == "claude":
        from oap.reconstruct.payloads.qwen_priors import (
            build_prompt as _build_priors_prompt,
            parse_json_object,
        )
        from oap.vlm import ask_vlm

        class _PromptArgs:                       # the payload's prompt, verbatim
            pass
        pa = _PromptArgs()
        pa.size_xyz_m = [float(v) for v in size_xyz_m]
        pa.object_label = str(object_label)
        pa.task = str(task)
        pa.scene_hint = (
            "The image is a real RGB-D tabletop observation. The target object is "
            "segmented and reconstructed separately with SAM3D; use visual cues "
            "only for material/category.")
        # POLL, do not sample once. Measured 2026-07-27 on the same object, the
        # same label and the same image: the box's nominal mass came back 162 g,
        # 262 g, 162 g on three consecutive calls -- a 60% swing with nothing
        # changed. A prior that moves that much between runs is not reportable,
        # and the JSON schema already had `n_queries` and
        # `vlm_raw_estimates_all` slots that nothing ever filled (n_queries was
        # the literal 1). The aggregate is the MEDIAN of the nominals with the
        # UNION of the ranges: the median is robust to one wild reading, and
        # widening the interval is the honest direction for a prior that is
        # explicitly not ground truth.
        n_queries = max(1, int(os.environ.get("OAP_PRIORS_QUERIES", "3")))
        estimates: list[dict[str, Any]] = []
        raw_texts: list[str] = []
        for _ in range(n_queries):
            raw_text, resolved_model = ask_vlm(
                system=("You estimate bounded, uncertain physical priors for a "
                        "reconstructed object. Return JSON only, no markdown."),
                user=_build_priors_prompt(pa), image=Path(image),
                max_tokens=int(max_new_tokens) * 2)
            raw_texts.append(raw_text)
            try:
                estimates.append(parse_json_object(raw_text))
            except Exception as exc:            # noqa: BLE001 - one bad reply
                logger.warning("[priors] %s: discarding an unparseable reply (%s)",
                               name, exc)
        if not estimates:
            raise RuntimeError(
                f"all {n_queries} VLM replies for {name!r} were unparseable")
        vlm_estimate = _aggregate_estimates(estimates)
        raw_text = "\n\n---- reply separator ----\n\n".join(raw_texts)
        response_json.write_text(json.dumps({
            "schema": "vlm_physical_prior_estimate_response_v1",
            "backend": "claude_api", "model_id": resolved_model,
            "input_image": str(image), "object_label": str(object_label),
            "size_xyz_m": [float(v) for v in size_xyz_m],
            "raw_text": raw_text, "estimate": vlm_estimate,
        }, indent=2), encoding="utf-8")
    else:
        resolved_model = (model_id or envs.extra("qwen", "model_id", required=False)
                          or "Qwen/Qwen3-VL-4B-Instruct")
        envs.run_payload(
            "qwen", "qwen_priors",
            [
                "--image", Path(image),
                "--out", response_json,
                "--model-id", resolved_model,
                "--object-label", object_label,
                "--task", task,
                "--max-new-tokens", str(max_new_tokens),
                "--max-side", str(max_side),
                "--size-xyz-m", str(float(size_xyz_m[0])), str(float(size_xyz_m[1])),
                str(float(size_xyz_m[2])),
            ],
        )
        response = read_json(response_json)
        vlm_estimate = response.get("estimate") or {}
        raw_text = str(response.get("raw_text") or "")

    all_estimates = estimates if backend == "claude" else None
    out_path = step_dir / f"{name}_priors.json"
    size_m = tuple(float(v) for v in size_xyz_m)
    priors = physical_prior_report_from_vlm_payload(
        vlm_estimate,
        size_m=(size_m[0], size_m[1], size_m[2]),
        object_label=str(object_label),
        source_path=out_path,
    )
    payload = {
        "schema_version": "countersim.phys2real_vlm_physical_priors.v1",
        "source": "actual_vlm_phys2real_physical_parameter_estimation",
        "backend": ("claude_api" if backend == "claude" else "qwen3vl_local"),
        "n_queries": int(len(all_estimates)) if all_estimates else 1,
        "vlm_raw_estimates_all": all_estimates or None,
        "model_id": str(resolved_model),
        "input_image": str(image),
        "input_mask": str(mask) if mask else None,
        "task_text": str(task),
        "object_label": str(object_label),
        "size_xyz_m": list(size_m),
        "volume_m3": float(size_m[0] * size_m[1] * size_m[2]),
        "vlm_raw_estimate": vlm_estimate,
        "physical_priors": priors.to_dict(),
        "control_role": "uncertain_physical_prior_for_simulation_not_hidden_truth",
    }
    write_json_atomic(out_path, payload)
    out_path.with_suffix(".raw.txt").write_text(raw_text, encoding="utf-8")
    logger.info(
        "[priors] %s: material=%s density=%s mass_nominal=%.4f kg",
        name, priors.material, priors.density_kg_m3["range"], priors.mass_kg["nominal"],
    )
    return payload
