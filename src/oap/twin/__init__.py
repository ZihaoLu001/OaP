"""W_plan digital twin: scene bundle -> MuJoCo twin -> physics rollouts.

Role in the two-stage pipeline: this package is the bridge between the offline
reconstruction stage (scene bundles) and the online closed loop. It loads a
bundle manifest into typed scene objects, assembles the calibrated lab twin
(robot + curated GN01 + measured table + reconstructed objects), matches the
sim TCP to the real calibrated tool,
plans over a uniform knot array, rolls candidates out, and converts a
certified plan into a bounds-checked real-robot command.
Imports only :mod:`oap.program` and :mod:`oap.utils`.
"""
from __future__ import annotations

from .assets import (
    GN01_CLOSE_INTENT_WIDTH_M, GN01_OPEN_WIDTH_M, find_asset_workspace,
    gripper_joint_targets, official_gn01_flange_mount,
)
from .gripper_tcp import REAL_GN01_TCP_M, patch_sim_gripper_tcp
from .manifest import MANIFEST_SCHEMA, SceneManifestError, load_scene_manifest
from .obb import (
    gravity_aligned_obb, load_obj_vertices, mesh_axis_rotation,
    mesh_obb_from_vertices, obb_pose_matrix, object_body_quat,
    object_vertical_half_extent, object_vertical_half_extent_from_q_om,
    project_unheld_pose_to_support, roll_q_om_for_standing_dims,
    subject_obb_from_capture,
)
from .rollout import (
    clone_data, patch_static_body_pose, reset_object_pose, reset_robot_qpos,
    tcp_pose_T,
)
from .scene import build_plan_scene, load_world, map_robot, observe_object_pose
from .types import (
    TABLE_TOP_Z, ObservationResult, PhysicalPriorReport,
    ReconstructedObjectContract, SceneObject, SimWorld,
)

__all__ = [
    # types + constants
    "TABLE_TOP_Z", "SceneObject", "ReconstructedObjectContract",
    "SimWorld", "ObservationResult", "PhysicalPriorReport",
    # manifest
    "MANIFEST_SCHEMA", "SceneManifestError", "load_scene_manifest",
    # assets
    "find_asset_workspace", "official_gn01_flange_mount",
    "GN01_OPEN_WIDTH_M", "GN01_CLOSE_INTENT_WIDTH_M",
    "gripper_joint_targets",
    # obb
    "gravity_aligned_obb", "load_obj_vertices", "mesh_obb_from_vertices",
    "subject_obb_from_capture", "mesh_axis_rotation", "object_body_quat",
    "object_vertical_half_extent", "object_vertical_half_extent_from_q_om",
    "project_unheld_pose_to_support", "roll_q_om_for_standing_dims",
    "obb_pose_matrix",
    # scene
    "build_plan_scene", "load_world", "map_robot", "observe_object_pose",
    # sim TCP
    "REAL_GN01_TCP_M", "patch_sim_gripper_tcp",
    # rollout
    "clone_data", "reset_object_pose",
    "reset_robot_qpos", "patch_static_body_pose", "tcp_pose_T",
]
