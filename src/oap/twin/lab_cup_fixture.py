"""Explicit lab-reconstruction fixture, separate from the historical YCB task.

The visual mesh comes from the lab cup reconstruction. Contact uses the existing
solid-cylinder approximation for side grasping, not the mesh's hollow interior.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from oap.twin.control_profile import external_input_path

TASK_ID = "lab_cup_min20_v1"
PROGRAM_PATH = "experiments/task_programs/lab_cup_grasp_tilt_min20_v1.json"
INITIAL_OBSERVATION_PATH = "experiments/initial_observations/lab_cup_offline.json"
INSTRUCTION = (
    "Grasp the cup and tilt it by at least 20 degrees toward robot-base +X "
    "while keeping it held."
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture() -> tuple[dict[str, Any], Path]:
    """Load the explicitly supplied fixture and assets below OAP_INPUT_ROOT."""
    override = (os.environ.get("OAP_LAB_CUP_FIXTURE") or "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OAP_LAB_CUP_FIXTURE does not exist: {path}")
    else:
        path = external_input_path(
            "releases/lab_cup_hardmask_20260910/lab_cup_fixture.json",
            purpose="lab cup fixture",
        )
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if fixture.get("schema") != "oap_lab_cup_fixture_v1":
        raise ValueError("Expected the explicit lab cup fixture schema")
    if fixture.get("collision_primitive") != "cylinder":
        raise ValueError("Lab cup contact remains the frozen cylinder approximation")
    for field, length in (("size_lwh_m", 3), ("friction", 3), ("rgba", 4),
                          ("initial_position_xy_m", 2)):
        values = fixture[field]
        if len(values) != length or not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Invalid lab cup fixture field " + field)
    if min(fixture["size_lwh_m"]) <= 0 or not 0 < float(fixture["mass_kg"]) < math.inf:
        raise ValueError("Lab cup size and mass must be positive and finite")
    for field in ("visual_mesh", "texture"):
        asset = external_input_path(fixture[field], purpose="lab cup " + field)
        if _sha(asset) != fixture[field + "_sha256"]:
            raise ValueError("Lab cup asset hash mismatch: " + field)
    return fixture, path


def registration() -> dict[str, Any]:
    """Load scene metadata without requiring a historical generated program."""
    fixture, _ = load_fixture()
    return {
        "program_path": PROGRAM_PATH,
        "initial_observation_path": INITIAL_OBSERVATION_PATH,
        "instruction": INSTRUCTION,
        "provenance": "lab-reconstruction-fixture:2026-09-10",
        "subject": "cup", "reference": None,
        "scene": {
            "offline_dataset_subject_source_name": "eraser",
            "offline_dataset_subject_name": "cup",
            "offline_dataset_subject_label": fixture["label"],
            "offline_dataset_visual_mesh": str(external_input_path(
                fixture["visual_mesh"], purpose="lab cup visual_mesh"
            )),
            "offline_dataset_size_lwh": tuple(fixture["size_lwh_m"]),
            "offline_dataset_mass_kg": float(fixture["mass_kg"]),
            "offline_dataset_rgba": tuple(fixture["rgba"]),
            "offline_dataset_collision_primitive": "cylinder",
        },
        "initial_observation_identity": "lab_cup_fixture_initial_pose",
        "manifest_contains_registered_subject": False,
    }


def apply_materials(subject: Any) -> None:
    """Use the lab cup's texture and friction after the existing subject adapter."""
    fixture, _ = load_fixture()
    subject.texture = external_input_path(fixture["texture"], purpose="lab cup texture")
    subject.friction = tuple(float(v) for v in fixture["friction"])


def input_identity(*, control_profile: str, bundle_manifest: Any,
                   initial_observation: dict[str, Any], subject_name: str,
                   reference_name: str | None) -> dict[str, Any]:
    fixture, path = load_fixture()
    if control_profile != "legacy_velocity_effort" or (subject_name, reference_name) != ("cup", None):
        raise ValueError("Lab cup requires the unchanged D controller and cup subject")
    observation_path = external_input_path(
        INITIAL_OBSERVATION_PATH, purpose="lab cup initial observation"
    )
    authority = json.loads(observation_path.read_text(encoding="utf-8"))
    if initial_observation != authority:
        raise ValueError("Lab cup initial observation differs from its fixture authority")
    expected_pos = [*fixture["initial_position_xy_m"],
                    float(fixture["table_top_z_m"]) + .5 * float(fixture["size_lwh_m"][2])]
    position = authority["pos_base"]
    if len(position) != 3 or any(not math.isfinite(float(a)) or abs(float(a) - float(b)) > 1e-9
                                 for a, b in zip(position, expected_pos)):
        raise ValueError("Lab cup initial position must match its declared simulation placement")
    if authority["quat_wxyz"] != [1., 0., 0., 0.]:
        raise ValueError("Lab cup fixture must start in its upright material frame")
    return {
        "schema": "oap_lab_cup_input_v1", "registered_task_id": TASK_ID,
        "control_profile": control_profile, "subject_name": subject_name,
        "reference_name": reference_name, "manifest_contains_registered_subject": False,
        "subject_identity_source": "explicit_lab_reconstruction_fixture",
        "bundle_manifest": str(bundle_manifest), "bundle_manifest_sha256": _sha(Path(bundle_manifest)),
        "fixture_path": str(path), "fixture_sha256": _sha(path),
        "initial_observation_sha256": _sha(observation_path),
        "visual_mesh_sha256": fixture["visual_mesh_sha256"],
        "texture_sha256": fixture["texture_sha256"],
        "collision_representation": "solid_cylinder_approximation_not_hollow_mesh",
        "scene": registration()["scene"],
    }
