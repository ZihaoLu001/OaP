"""Bounded physical priors for a reconstructed object -- ONE implementation.

This lived twice: once in :mod:`oap.twin.types` (read at manifest load) and
once in :mod:`oap.reconstruct.priors` (written by the reconstruct stage).
The copies drifted, and on 2026-07-26 they disagreed by 4x on the same object's
mass -- the twin said 0.205 kg where the priors JSON said 0.051 kg, and the twin
is the copy the physics uses. The mechanism was subtle in the way duplicated
code always is: both defined ``_shell_thickness_m(raw, size_m)``, same name,
same signature, but ``raw`` meant the DERIVED ``physical_priors`` block in one
and the RAW VLM estimate in the other, so a reader written against one
convention silently found nothing under the other.

Both call conventions still work and are exercised: the reconstruct stage hands
in the VLM estimate itself, the twin hands in the whole priors JSON. The
thickness lookup walks them in priority order -- the derived block, then the raw
estimate, then a caller-supplied default, then a quarter of the object's
smallest extent -- so the SAME code answers both without either caller having to
know about the other.

Dependency-light on purpose (numpy only, no mujoco, no scipy). That is what let
the duplication happen: importing ``twin.types`` from ``reconstruct`` drags in
the whole simulator, so the reconstruct side copied the formula instead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MATERIAL_DENSITY_PRIORS", "MATERIAL_TABLE_FRICTION_PRIORS",
    "PhysicalPriorReport", "physical_prior_report_from_vlm_payload",
    "load_physical_priors", "mujoco_friction_from_priors",
]


MATERIAL_DENSITY_PRIORS: dict[str, tuple[float, float, float]] = {
    # low, nominal, high kg/m^3. These are deliberately broad priors, not
    # measured truth (Phys2Real-style uncertainty).
    "foam": (20.0, 80.0, 250.0),
    "cardboard": (120.0, 350.0, 700.0),
    "paper": (150.0, 500.0, 900.0),
    "plastic": (600.0, 950.0, 1300.0),
    "rubber": (900.0, 1150.0, 1600.0),
    "wood": (350.0, 650.0, 950.0),
    "metal": (2400.0, 5000.0, 7900.0),
    "ceramic": (1200.0, 2200.0, 3500.0),
    "unknown": (200.0, 700.0, 1400.0),
}


MATERIAL_TABLE_FRICTION_PRIORS: dict[str, tuple[float, float]] = {
    "foam": (0.5, 1.3),
    "cardboard": (0.35, 0.9),
    "paper": (0.3, 0.8),
    "plastic": (0.25, 0.75),
    "rubber": (0.65, 1.4),
    "wood": (0.35, 0.9),
    "metal": (0.18, 0.55),
    "ceramic": (0.25, 0.7),
    "unknown": (0.25, 1.0),
}


@dataclass
class PhysicalPriorReport:
    """Bounded physical priors for one reconstructed object (not ground truth)."""

    source: str
    material: str
    density_kg_m3: dict[str, Any]
    mass_kg: dict[str, Any]
    friction_table: dict[str, Any]
    friction_gripper: dict[str, Any]
    inertia_diag_kgm2: tuple[float, float, float]
    update_policy: str
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain JSON-serializable dict."""
        return asdict(self)


def _infer_material(spec: dict[str, Any], object_label: str) -> str:
    """Infer a density-prior material key from loose VLM text fields."""
    candidates: list[Any] = [spec.get("material"), spec.get("object_material")]
    physical = spec.get("physical_priors")
    if isinstance(physical, dict):
        candidates.extend([physical.get("material"), physical.get("object_material")])
    target = spec.get("target_object")
    if isinstance(target, dict):
        candidates.extend([target.get("material"), target.get("category")])
    text = " ".join(str(v).lower() for v in candidates if v)
    text = f"{text} {object_label.lower()}"
    for key in MATERIAL_DENSITY_PRIORS:
        if key != "unknown" and key in text:
            return key
    if any(token in text for token in ("cube", "block", "tray", "bin")):
        return "plastic"
    if any(token in text for token in ("milk", "carton", "box")):
        return "cardboard"
    if any(token in text for token in ("can", "tin")):
        return "metal"
    return "unknown"


def _bounded_range_with_nominal(
    value: Any,
    *,
    fallback: tuple[float, float, float],
    absolute_bounds: tuple[float, float],
) -> tuple[float, float, float]:
    """Normalize a loose VLM range/nominal estimate into clamped (lo, mid, hi)."""
    lo_abs, hi_abs = absolute_bounds
    lo, mid, hi = fallback
    if isinstance(value, dict):
        raw_range = value.get("range")
        if raw_range is not None:
            try:
                arr = np.asarray(raw_range, dtype=np.float64).reshape(-1)
                if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                    lo, hi = float(arr[0]), float(arr[1])
            except (TypeError, ValueError):
                pass
        for key in ("nominal", "mean", "median", "best_estimate"):
            if key in value:
                try:
                    mid = float(value[key])
                    break
                except (TypeError, ValueError):
                    pass
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                lo, mid, hi = float(arr[0]), float(arr[1]), float(arr[2])
            elif arr.size >= 2:
                lo, hi = float(arr[0]), float(arr[1])
                mid = 0.5 * (lo + hi)
        except (TypeError, ValueError):
            pass
    elif value is not None:
        try:
            mid = float(value)
            lo = min(lo, mid)
            hi = max(hi, mid)
        except (TypeError, ValueError):
            pass
    lo = float(np.clip(min(lo, hi), lo_abs, hi_abs))
    hi = float(np.clip(max(lo, hi), lo_abs, hi_abs))
    mid = float(np.clip(mid, lo, hi))
    return lo, mid, hi


def _shell_thickness_m(raw: dict[str, Any], size_m: tuple[float, float, float],
                       fallback_raw: dict[str, Any] | None = None,
                       default_m: float | None = None) -> float:
    """Wall/shell thickness for the effective-volume correction.

    From the VLM when it estimates one, else a fraction of the object's own
    smallest dimension so the fallback scales with the object rather than
    being a hand-set constant. The fallback errs THICK (a quarter of the
    smallest extent) because that keeps V_eff near the solid answer, which is
    the conservative direction: an over-heavy object under-moves, an
    over-light one flies.
    """
    smallest = max(min(float(v) for v in size_m), 1e-6)
    # Precedence: this object's own measurement beats a global default. The
    # VLM's answer lives in the RAW estimate -- ``physical_priors`` is the
    # DERIVED block and its writer never copies the thickness across -- so
    # reading only the derived block discarded the one measurement this
    # correction exists to use. It then fell back to the manifest's blanket
    # 8 mm, which made the twin disagree with the priors JSON by exactly 4x on
    # the first two-object bundle (eraser 0.205 kg against 0.051 kg): the same
    # object, two masses, and the physics used the wrong one. Resolving it on
    # READ rather than at write time also repairs already-reconstructed bundles.
    for candidate in (raw.get("wall_thickness_m"),
                      (fallback_raw or {}).get("wall_thickness_m"),
                      default_m):
        val = candidate.get("nominal") if isinstance(candidate, dict) else candidate
        if val is not None:
            break
    else:
        val = None
    try:
        t = float(val)
    except (TypeError, ValueError):
        return 0.25 * smallest
    if not np.isfinite(t) or t <= 0.0:
        return 0.25 * smallest
    return float(np.clip(t, 0.0005, 0.5 * smallest))


def physical_prior_report_from_vlm_payload(
    payload: dict[str, Any],
    *,
    size_m: tuple[float, float, float],
    object_label: str,
    source_path: Path | None = None,
) -> PhysicalPriorReport:
    """Build bounded physical priors from an actual VLM estimate.

    The VLM is allowed to estimate category/material and broad physical
    ranges. The local code still clamps/normalizes those estimates and derives
    mass from the metric RGB-D/reconstruction volume so hidden simulator
    metadata cannot leak into W_plan.
    """
    raw = payload.get("physical_priors", payload)
    if not isinstance(raw, dict):
        raw = {}
    material = str(
        raw.get("material")
        or raw.get("vlm_material")
        or payload.get("material")
        or payload.get("object_material")
        or "unknown"
    ).lower()
    if material not in MATERIAL_DENSITY_PRIORS:
        material = _infer_material({"material": material}, object_label)
    if material not in MATERIAL_DENSITY_PRIORS:
        material = "unknown"

    density_low, density_nominal, density_high = _bounded_range_with_nominal(
        raw.get("density_kg_m3") or raw.get("density"),
        fallback=MATERIAL_DENSITY_PRIORS[material],
        absolute_bounds=(10.0, 9000.0),
    )
    friction_low, friction_nominal, friction_high = _bounded_range_with_nominal(
        raw.get("friction_table") or raw.get("table_friction") or raw.get("friction_coefficient"),
        fallback=(
            MATERIAL_TABLE_FRICTION_PRIORS.get(material, MATERIAL_TABLE_FRICTION_PRIORS["unknown"])[0],
            sum(MATERIAL_TABLE_FRICTION_PRIORS.get(material, MATERIAL_TABLE_FRICTION_PRIORS["unknown"])) * 0.5,
            MATERIAL_TABLE_FRICTION_PRIORS.get(material, MATERIAL_TABLE_FRICTION_PRIORS["unknown"])[1],
        ),
        absolute_bounds=(0.02, 2.0),
    )
    grip_low, grip_nominal, grip_high = _bounded_range_with_nominal(
        raw.get("friction_gripper") or raw.get("gripper_friction"),
        fallback=(0.55, 0.95, 1.35),
        absolute_bounds=(0.02, 2.5),
    )

    sx, sy, sz = [max(float(v), 1e-6) for v in size_m]
    volume = sx * sy * sz
    # EFFECTIVE volume, not bounding volume. Manipulanda are overwhelmingly
    # shells -- a box, a bottle, a cup enclose air -- so material density times
    # bounding volume is the SOLID answer, wrong by 3x-45x for containers
    # (YCB's measured table: real bulk density is 16-225 kg/m3 for boxes,
    # against the 700 a VLM correctly names for cardboard fibre).
    #
    #     V_eff = min(V_bbox, A_bbox * t)
    #
    # The ``min`` IS the branch: a genuinely solid object has t >= half its
    # smallest dimension, so A*t >= V, the min selects V, and the formula
    # collapses to the solid case bit-for-bit. No hollow flag, no per-object
    # special case. (Pham 2026 Eq. 4.1's eta factor; NeRF2Physics's thickness
    # prompt with its visual-hull cap -- the cap is this min.)
    #
    # It also retires a clamp that was hiding the error: mass was
    # np.clip(V*rho, 0.005, 0.250), so EVERY object above 250 g became exactly
    # 250 g. Both objects of the shipped bundle saturated it -- the amazon box
    # (1.73 kg solid) was accidentally RIGHT at 0.250, the hairgel bottle
    # (0.254 kg solid) was ~6x too heavy for an empty bottle. Dropping the
    # clamp WITHOUT this correction would have made the box strictly worse.
    area = 2.0 * (sx * sy + sx * sz + sy * sz)
    volume_eff = min(volume, area * _shell_thickness_m(
        raw, (sx, sy, sz), payload.get("vlm_raw_estimate"),
        payload.get("_default_wall_thickness_m")))
    mass_low = float(np.clip(volume_eff * density_low, 0.003, 5.0))
    mass_nominal = float(np.clip(volume_eff * density_nominal, 0.005, 5.0))
    mass_high = float(np.clip(max(volume_eff * density_high, mass_nominal), mass_nominal, 10.0))
    mass_low = min(mass_low, mass_nominal)

    ixx = mass_nominal * (sy * sy + sz * sz) / 12.0
    iyy = mass_nominal * (sx * sx + sz * sz) / 12.0
    izz = mass_nominal * (sx * sx + sy * sy) / 12.0
    provenance = dict(payload.get("provenance") or {})
    provenance.update(
        {
            "not_ground_truth": True,
            "object_label": object_label,
            "volume_m3": float(volume),
            "source_path": str(source_path) if source_path is not None else None,
            "vlm_payload_schema": payload.get("schema_version") or payload.get("schema"),
            "local_clamping_applied": True,
            "mass_formula": "rgbd_sam3d_volume_m3 * vlm_density_prior_kg_m3",
        }
    )
    return PhysicalPriorReport(
        source="phys2real_actual_vlm_uncertain_physical_prior_distribution",
        material=material,
        density_kg_m3={
            "distribution": "bounded_actual_vlm_material_prior",
            "range": [float(density_low), float(density_high)],
            "nominal": float(density_nominal),
        },
        mass_kg={
            "distribution": "volume_times_actual_vlm_density_prior",
            "range": [float(mass_low), float(mass_high)],
            "nominal": float(mass_nominal),
            "formula": "rgbd_sam3d_volume_m3 * vlm_density_prior_kg_m3",
        },
        friction_table={
            "distribution": "actual_vlm_material_pair_prior",
            "range": [float(friction_low), float(friction_high)],
            "nominal": float(friction_nominal),
        },
        friction_gripper={
            "distribution": "actual_vlm_rubber_pad_prior",
            "range": [float(grip_low), float(grip_high)],
            "nominal": float(grip_nominal),
        },
        inertia_diag_kgm2=(float(ixx), float(iyy), float(izz)),
        update_policy=(
            "After each executed chunk, compare observed object pose delta to W_plan; "
            "persistent slip/contact mismatch should update friction/mass priors."
        ),
        provenance=provenance,
    )


def load_physical_priors(
    path: Path,
    *,
    size_m: tuple[float, float, float],
    object_label: str,
    wall_thickness_m: float | None = None,
) -> PhysicalPriorReport:
    """Load a priors JSON (written by the reconstruct stage) into a report.

    ``wall_thickness_m`` (the manifest's, when the priors JSON carries none)
    feeds the effective-volume correction. The manifest has always held this
    number -- it sized the container collision decomposition -- so routing it
    here gives one quantity two consumers instead of one number and one guess.
    """
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if wall_thickness_m is not None:
        # NOT injected into ``physical_priors``: doing that made a blanket
        # manifest default outrank the VLM's per-object estimate, because the
        # reader looked in the derived block first and found the injection.
        payload["_default_wall_thickness_m"] = float(wall_thickness_m)
    return physical_prior_report_from_vlm_payload(
        payload, size_m=size_m, object_label=object_label, source_path=Path(path))


def mujoco_friction_from_priors(report: PhysicalPriorReport) -> tuple[float, float, float]:
    """Convert a prior report to a MuJoCo (sliding, torsional, rolling) triple."""
    mu = float(report.friction_table["nominal"])
    return (mu, 0.08, 0.01)
