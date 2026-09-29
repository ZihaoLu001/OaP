"""Typed contracts and lab-frame constants for the W_plan digital twin.

Role in the two-stage pipeline: ``oap-reconstruct`` writes a scene bundle
whose per-object contract is parsed here into :class:`SceneObject`;
``oap-run`` assembles those objects into the planning twin
(:mod:`oap.twin.scene`) and rolls candidate action chunks out in it
(:mod:`oap.twin.rollout`). The constants below are the calibrated lab
truth (table plane, home posture, GN01 stroke curve) shared
by every twin build.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Physical priors live in ONE module (oap.physical_priors) because two
# copies of this formula disagreed by 4x on the same object's mass. Re-exported
# here so every existing `from oap.twin.types import ...` keeps working.
from oap.physical_priors import (  # noqa: F401  (re-export)
    MATERIAL_DENSITY_PRIORS,
    MATERIAL_TABLE_FRICTION_PRIORS,
    PhysicalPriorReport,
    load_physical_priors,
    mujoco_friction_from_priors,
    physical_prior_report_from_vlm_payload,
)

__all__ = [
    "TABLE_TOP_Z", "TABLE_CENTER", "LAB_TABLE_POS", "LAB_TABLE_SIZE",
    "CAMERA_LOOKAT", "LAB_RESTORED_INITIAL_QPOS", "MJCF_HOME_QPOS",
    "MATERIAL_DENSITY_PRIORS", "MATERIAL_TABLE_FRICTION_PRIORS",
    "PhysicalPriorReport", "physical_prior_report_from_vlm_payload",
    "load_physical_priors", "mujoco_friction_from_priors",
    "ReconstructedObjectContract", "SceneObject",
    "SimWorld", "ObservationResult",
]

# ----------------------------------------------------------------- lab frame
TABLE_CENTER = np.asarray([0.7510895177674536, -0.0621738655708316, 0.0], dtype=float)
TABLE_TOP_Z = 0.020
LAB_TABLE_POS = np.asarray([0.56, -0.062173866, -0.005], dtype=float)
LAB_TABLE_SIZE = np.asarray([0.86, 0.62, 0.025], dtype=float)
CAMERA_LOOKAT = np.asarray([0.7510895177674536, -0.0621738655708316, 0.24], dtype=float)

# This is not the generic MJCF ``home`` key. It is the real lab restore target
# (the posture the home-gate enforces on the real arm). Twin plans start here.
LAB_RESTORED_INITIAL_QPOS = np.asarray(
    [
        -0.0010957308,
        -0.70017499,
        -0.0020403836,
        1.5730212,
        0.0051091854,
        0.68894786,
        -0.0028807865,
    ],
    dtype=float,
)
MJCF_HOME_QPOS = np.asarray([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0], dtype=float)

# ------------------------------------------------------------ physical priors

















# --------------------------------------------------------------- object types
@dataclass(frozen=True)
class ReconstructedObjectContract:
    """Raw per-object entry of a scene-bundle manifest (paths + flags only).

    This is the exact contract the reconstruction stage writes; it is turned
    into a derived :class:`SceneObject` by
    :func:`oap.twin.manifest.load_scene_manifest`.
    """

    name: str
    label: str
    sam3_prompt: str
    mesh: Path
    visual_mesh: Path
    texture: Path
    tracking_metadata: Path
    collision: Path
    refined_json: Path
    priors_json: Path
    movable: bool
    artifact_sha256: dict[str, str]
    open_top_container: bool = False
    wall_thickness_m: float = 0.008
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)


@dataclass
class SceneObject:
    """One reconstructed real object from the scene manifest (load-time derived)."""

    name: str
    label: str
    sam3_prompt: str
    mesh: Path
    collision: Path
    refined_json: Path
    priors_json: Path
    movable: bool
    rgba: tuple[float, float, float, float]

    # derived at load time
    size_lwh: tuple[float, float, float] = (0.0, 0.0, 0.0)
    q_om: np.ndarray | None = None
    mesh_long_axis: np.ndarray | None = None
    priors: PhysicalPriorReport | None = None
    mass_kg: float = 0.0
    friction: tuple[float, float, float] = (0.6, 0.08, 0.01)
    pose_base: list[float] | None = None
    pose_quat: list[float] | None = None  # mesh world quat (w,x,y,z) from recon
    yaw_ref_rad: float | None = None      # object-frame yaw at reconstruction
    visual_mesh: Path | None = None       # full-res generated mesh (render-only)
    tracking_metadata: Path | None = None # generated tracking QA/lineage record
    texture: Path | None = None           # baked UV texture PNG for the visual mesh
    open_top_container: bool = False      # open bin/tray: collision = floor + 4 walls (hollow),
    wall_thickness_m: float = 0.008       # so a subject can be placed INSIDE, not on top
    collision_primitive: str = "box"      # offline dataset demos may use a cylinder proxy
    # The snapped static z is a PLAN-frame height (TABLE_TOP_Z + half-extent);
    # marked so the scene builder does not add z_real_to_plan a second time.
    z_snapped_plan_frame: bool = False
    # Optional mesh-hugging collision-box center in the OBJECT frame, retained
    # for internal geometry tests (production uses the manifest artifact).
    collision_center_obj: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SimWorld:
    """A loaded W_plan MuJoCo world plus the resolved robot/object handles."""

    name: str
    model: Any            # mujoco.MjModel
    data: Any             # mujoco.MjData
    ik_data: Any          # mujoco.MjData (scratch data for the IK solver)
    robot: dict[str, Any]
    object_qpos_addr: int
    site_id: int
    q: np.ndarray
    # Every body the physics can move, name -> freejoint qpos address. Keys are
    # BODY names: a support appears under its manifest name, but the SUBJECT
    # appears as ``pick_object``, because that is what the scene builder calls
    # its body whatever the manifest named it. So the subject IS in here and
    # callers wanting "all the movers" need no special case -- but a caller
    # matching these keys against ``SceneObject.name`` silently skips the
    # subject. That is usually the correct behaviour (the subject is grounded
    # from its own observed pose, not read back from qpos), which is exactly
    # why the mismatch is easy to miss; ``_reground_movables`` relies on it.
    # Meanwhile ``object_qpos_addr`` stays exactly what it always was, so
    # the ~40 sites that index it are untouched. Whether a body can move is a
    # MANIFEST property (SceneObject.movable), not something the scene builder
    # decides: a goal that requires a second body to move is unrepresentable
    # until that body has a joint, and the inert-success gate refuses such a
    # program rather than pretending.
    movable_qpos_addr: dict[str, int] = field(default_factory=dict)


@dataclass
class ObservationResult:
    """One re-observation of an object in the ROBOT BASE frame."""

    tag: str
    pos_base: list[float]
    yaw_base_rad: float
    quat_wxyz: list[float] | None = None
    size_lwh_obb: tuple[float, float, float] | None = None
    table_z_base: float | None = None
    source: str = "unknown"
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the observation as a plain JSON-serializable dict."""
        return asdict(self)
