"""Immutable, object-agnostic reconstruction evidence profile.

These limits validate sensor/reconstruction evidence before an asset can enter
the executable twin.  They are not MPC costs, task heuristics, object-class
priors, or per-run CLI knobs.  One named profile applies to every object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReconstructionQualityProfile:
    """Global validation limits for the fixed-camera reconstruction pipeline."""

    name: str
    min_sam3_score: float
    min_sam3_mask_pixels: int
    min_masked_depth_coverage: float
    min_refined_pose_iou: float
    max_refined_pose_depth_mae_m: float
    min_scale_points: int
    min_scale_boundary_probe_fraction: float
    scale_boundary_probe_pixels: int
    scale_extent_trim_percent: float
    max_scale_fold: float
    scale_bootstrap_samples: int
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready provenance for emitted validation evidence."""
        return asdict(self)


QUALITY_PROFILE = ReconstructionQualityProfile(
    name="fixed_zed2i_automatic_reconstruction_v1",
    min_sam3_score=0.40,
    min_sam3_mask_pixels=1024,
    min_masked_depth_coverage=0.70,
    min_refined_pose_iou=0.50,
    max_refined_pose_depth_mae_m=0.030,
    min_scale_points=256,
    min_scale_boundary_probe_fraction=0.50,
    scale_boundary_probe_pixels=1,
    scale_extent_trim_percent=2.0,
    max_scale_fold=1.25,
    scale_bootstrap_samples=256,
    provenance=(
        "Global fixed-camera sensor/reconstruction validation. The SAM3 floor "
        "matches the deployed detector threshold. The HD720 mask/depth floors "
        "sit conservatively below the measured accepted tool-scene range "
        "(minimum 7937 pixels and 0.834 masked-depth coverage). Pose limits "
        "bound the measured accepted range (minimum IoU 0.5833, maximum depth "
        "MAE 0.0222 m). Metric scale retains every valid visible surface and "
        "measures a globally fixed 2%-98% gravity-frame extent. Estimates at "
        "1%-99% and after one-pixel boundary erosion are sensitivity probes "
        "only; they never replace the primary scale. The same 1.25 "
        "multiplicative consistency limit validates mesh-axis agreement, "
        "bootstrap stability, and estimator sensitivity. No object, task, "
        "or planner may override this profile."
    ),
)
