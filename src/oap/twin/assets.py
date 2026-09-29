"""Locate external robot assets and build the curated GN01 gripper subtree.

Role in the two-stage pipeline: the planning twin (``oap-run``) needs the
Flexiv Rizon4s MJCF meshes and the GN01/Grav gripper meshes, which are large
binaries supplied OUTSIDE this repo via ``OAP_ASSET_WORKSPACE`` or
``OAP_INPUT_ROOT``.
This module resolves that workspace and mounts the GN01 on the flange using
the link7-to-flange transform supplied in the external config file
``robot/gn01_mount_relative.yaml``.
"""
from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from oap.utils.io import load_yaml, package_config_path
from oap.utils.se3 import quat_wxyz_from_rpy

logger = logging.getLogger("oap.twin.assets")

__all__ = [
    "find_asset_workspace", "menagerie_assets_dir", "flexiv_description_root",
    "calibration_scene_xml", "official_gn01_flange_mount",
    "GN01_OPEN_WIDTH_M", "GN01_CLOSE_INTENT_WIDTH_M", "GN01_GEOM_QUAT",
    "GN01_PAD_CENTER_Z_M",
    "GN01_PAD_HALF_HEIGHT_M", "gn01_theta_for_width",
    "gripper_joint_targets",
    "ensure_gn01_meshes", "replace_static_gn01_with_articulated",
    "harden_gn01_fingertips_for_grasping", "append_gn01_articulated_subtree",
    "ensure_equality", "ensure_actuator", "mesh_geom",
    "rewrite_calibrated_robot_asset_paths", "strip_recorded_free_bodies",
    "ensure_material", "ensure_mesh", "ensure_child", "require_child",
    "find_body", "find_geom", "indent_xml",
]

def find_asset_workspace() -> Path:
    """Find the workspace that owns the calibrated robot / gripper assets.

    Runs are launched from a clean checkout while the large Flexiv / GN01 mesh
    assets live in an explicit external asset workspace. OAP_ASSET_WORKSPACE
    takes precedence over OAP_INPUT_ROOT; no source-checkout path is searched.

    Returns:
        The first candidate containing
        ``third_party/mujoco_menagerie/flexiv_rizon4/assets`` and
        ``third_party/flexiv_description`` (or the candidate that IS such a
        ``third_party`` layout).

    Raises:
        FileNotFoundError: With setup instructions when no candidate matches.
    """
    value = os.environ.get("OAP_ASSET_WORKSPACE") or os.environ.get("OAP_INPUT_ROOT")
    if not value:
        raise FileNotFoundError(
            "Set OAP_ASSET_WORKSPACE or OAP_INPUT_ROOT to your external "
            "robot asset workspace. Robot meshes are not included in the source."
        )
    root = Path(value).expanduser().resolve()
    assets = _third_party_root(root)
    if ((assets / "mujoco_menagerie" / "flexiv_rizon4" / "assets").is_dir()
            and (assets / "flexiv_description").is_dir()):
        return root
    raise FileNotFoundError(
        f"Missing external robot assets under {root}: expected "
        "third_party/mujoco_menagerie/flexiv_rizon4/assets and "
        "third_party/flexiv_description (or the same layout without third_party/)."
    )


def _third_party_root(workspace: Path) -> Path:
    """Return the ``third_party`` dir for a resolved workspace."""
    tp = workspace / "third_party"
    return tp if tp.exists() else workspace


def menagerie_assets_dir(workspace: Path | None = None) -> Path:
    """Return the Flexiv Rizon MuJoCo-Menagerie assets dir (raises if missing)."""
    workspace = workspace or find_asset_workspace()
    assets = _third_party_root(workspace) / "mujoco_menagerie" / "flexiv_rizon4" / "assets"
    if not assets.exists():
        raise FileNotFoundError(f"Missing Flexiv Rizon mesh assets: {assets}")
    return assets


def flexiv_description_root(workspace: Path | None = None) -> Path:
    """Return the flexiv_description root holding the GN01/Grav meshes."""
    workspace = workspace or find_asset_workspace()
    root = _third_party_root(workspace) / "flexiv_description"
    if not root.exists():
        raise FileNotFoundError(f"Missing flexiv_description assets: {root}")
    return root


def calibration_scene_xml(workspace: Path | None = None) -> Path:
    """Resolve the calibrated base lab-scene MJCF (robot base + table + framing).

    Checks OAP_CALIBRATION_SCENE_XML first, then calibration/scene.xml in
    the external configuration root. The workspace argument remains accepted
    for callers that also resolve meshes; calibration is configured separately.
    """
    env_value = os.environ.get("OAP_CALIBRATION_SCENE_XML")
    if env_value:
        p = Path(env_value).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"OAP_CALIBRATION_SCENE_XML points at a missing file: {p}")
    return package_config_path("calibration/scene.xml")


# --------------------------------------------------------- curated GN01 mount
_GN01_MOUNT_CACHE: tuple[str, str] | None = None


def official_gn01_flange_mount() -> tuple[str, str]:
    """Return (pos, quat_wxyz) MJCF strings for the link7->grav_base mount.

    Read once from the external ``configs/robot/gn01_mount_relative.yaml`` --
    the single source of truth encoding the official Rizon4s default
    kinematics link7_to_flange = xyz [0, 0, 0.124], rpy [0, 0, -pi].
    """
    global _GN01_MOUNT_CACHE
    if _GN01_MOUNT_CACHE is None:
        payload = load_yaml(package_config_path("robot/gn01_mount_relative.yaml"))
        xyz = [float(v) for v in payload["gn01_mount_xyz"]]
        rpy = [float(v) for v in payload["gn01_mount_rpy"]]
        pos = " ".join(f"{v:.6f}" for v in xyz)
        quat = " ".join(f"{v:.8f}" for v in quat_wxyz_from_rpy(rpy))
        _GN01_MOUNT_CACHE = (pos, quat)
    return _GN01_MOUNT_CACHE


# --------------------------------------------------------- GN01 stroke truth
GN01_OPEN_WIDTH_M = 0.085
# A width command below this is a CLOSE/grasp intent; at or above it is an
# OPEN intent. Must sit between the largest graspable just-touch close width
# (~0.078 for a 6.5 cm object on the measured gap curve) and the open 0.085.
GN01_CLOSE_INTENT_WIDTH_M = 0.082
GN01_GEOM_QUAT = "0.70710678 0 0 0.70710678"
# Finger-pad centre, metres from the flange along the tool axis. The sampler
# plans grasp heights at the PAD CENTRE and converts to a site command via
# (GN01_PAD_CENTER_Z_M - actual_site_flange_dist); the curated mount's site
# coincides with the pad centre, a calibrated fingertip site (--sim-tcp-m
# 0.19812) sits 24.1 mm beyond it. Never hard-code that conversion.
GN01_PAD_CENTER_Z_M = 0.174
# Half-height of the box-primitive fingertip collision (z extent of the pad
# below/above its center) -- keeps approach sweeps clear of tall objects.
GN01_PAD_HALF_HEIGHT_M = 0.0208

def gn01_theta_for_width(width_m: float) -> float:
    """Return the outer-knuckle hinge angle for a GN01 width command.

    ``-0.155 + 9.404 * width`` is copied from Flexiv's official GN01/Grav
    affine mimic definition.  The width coordinate is semantically metres in
    our control API, but the XML represents that virtual coordinate with a
    MuJoCo hinge, whose declared unit is radians.  The slope therefore maps two
    numeric conventions into the real knuckle angle; it is not a measured
    force gain or evidence of transmission loss.
    """
    return float(np.clip(9.404 * float(width_m) - 0.155, -0.155, 0.7854))


def gripper_joint_targets(width_m: float) -> list[float]:
    """Return the 7 GN01 joint qpos values realizing a width command."""
    theta = gn01_theta_for_width(width_m)
    return [float(width_m), theta, -theta, theta, theta, -theta, theta]


# ------------------------------------------------------------- XML utilities
def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    """Return the first ``tag`` child of ``parent``, creating it if absent."""
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def require_child(parent: ET.Element, tag: str) -> ET.Element:
    """Return the first ``tag`` child of ``parent`` or raise."""
    child = parent.find(tag)
    if child is None:
        raise RuntimeError(f"Missing <{tag}> in MJCF")
    return child


def find_body(root: ET.Element, name: str) -> ET.Element | None:
    """Find a ``<body name=...>`` element anywhere under ``root``."""
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


def find_geom(root: ET.Element, name: str) -> ET.Element | None:
    """Find a ``<geom name=...>`` element anywhere under ``root``."""
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == name:
            return geom
    return None


def ensure_material(asset: ET.Element, name: str, rgba: tuple[float, float, float, float]) -> None:
    """Create or update a flat-rgba material in ``<asset>``."""
    for material in asset.findall("material"):
        if material.attrib.get("name") == name:
            material.set("rgba", " ".join(f"{v:.4f}" for v in rgba))
            return
    ET.SubElement(asset, "material", {"name": name, "rgba": " ".join(f"{v:.4f}" for v in rgba)})


def ensure_mesh(asset: ET.Element, name: str, path: Path, scale: tuple[float, float, float]) -> None:
    """Create or update a named mesh asset pointing at ``path``."""
    file_str = str(path.resolve()).replace("\\", "/")
    for mesh in asset.findall("mesh"):
        if mesh.attrib.get("name") == name:
            mesh.set("file", file_str)
            mesh.set("scale", " ".join(f"{v:.6g}" for v in scale))
            return
    ET.SubElement(asset, "mesh", {"name": name, "file": file_str, "scale": " ".join(f"{v:.6g}" for v in scale)})


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Pretty-indent an ElementTree in place (2-space)."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        child = elem[0]
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


# --------------------------------------------------- calibrated base scene fixups
def rewrite_calibrated_robot_asset_paths(root: ET.Element, workspace: Path | None = None) -> None:
    """Make the legacy calibrated robot XML portable across machines.

    The saved calibration XML preserves the original replay robot/table/camera
    geometry but carries absolute mesh paths from the capture workstation;
    point the compiler meshdir at the local menagerie assets instead.
    """
    compiler = root.find("compiler")
    assets = menagerie_assets_dir(workspace)
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(assets.resolve()).replace("\\", "/"))


def strip_recorded_free_bodies(root: ET.Element) -> None:
    """Reduce a calibrated replay scene to robot and fixed infrastructure.

    The calibration XML establishes the robot, camera, and workspace. Any free
    body in that replay is an object from the old recorded episode; live task
    objects must instead come from the current scene bundle. Remove those
    bodies by joint type, then remove mesh assets no remaining geometry uses.
    This keeps scene assembly task- and object-name agnostic.
    """
    world = require_child(root, "worldbody")

    def _strip(parent: ET.Element) -> None:
        for body in list(parent.findall("body")):
            has_free_joint = body.find("freejoint") is not None or any(
                joint.attrib.get("type") == "free"
                for joint in body.findall("joint")
            )
            if has_free_joint:
                parent.remove(body)
            else:
                _strip(body)

    _strip(world)

    referenced_meshes = {
        mesh_name
        for geom in root.findall(".//geom")
        if (mesh_name := geom.attrib.get("mesh"))
    }
    asset = require_child(root, "asset")
    for mesh in list(asset.findall("mesh")):
        mesh_name = mesh.attrib.get("name")
        if mesh_name is None:
            mesh_name = Path(mesh.attrib.get("file", "")).stem
        if mesh_name and mesh_name not in referenced_meshes:
            asset.remove(mesh)


# ----------------------------------------------------------- GN01 MJCF builder
def ensure_gn01_meshes(asset: ET.Element, workspace: Path | None = None) -> None:
    """Register the official GN01/Grav visual + collision meshes in ``<asset>``."""
    desc = flexiv_description_root(workspace)
    specs = {
        "gn01_visual_base_mesh": desc / "meshes" / "Grav" / "visual" / "base.stl",
        "gn01_visual_outer_bar_mesh": desc / "meshes" / "Grav" / "visual" / "outer_bar.stl",
        "gn01_visual_inner_bar_mesh": desc / "meshes" / "Grav" / "visual" / "inner_bar.stl",
        "gn01_visual_finger_mount_mesh": desc / "meshes" / "Grav" / "visual" / "finger_mount.stl",
        "gn01_visual_finger_tip_mesh": desc / "meshes" / "Grav" / "visual" / "finger_tip.obj",
        "gn01_collision_base_mesh": desc / "meshes" / "Grav" / "collision" / "base.stl",
        "gn01_collision_static_assembly_mesh": desc / "meshes" / "Grav" / "collision" / "static_assembly.stl",
        "gn01_collision_outer_bar_mesh": desc / "meshes" / "Grav" / "collision" / "outer_bar.stl",
        "gn01_collision_inner_bar_mesh": desc / "meshes" / "Grav" / "collision" / "inner_bar.stl",
        "gn01_collision_finger_tip_mesh": desc / "meshes" / "Grav" / "collision" / "finger_tip.stl",
    }
    for name, file_path in specs.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing official GN01 mesh: {file_path}")
        ensure_mesh(asset, name, file_path, (0.001, 0.001, 0.001))


def mesh_geom(
    parent: ET.Element,
    name: str,
    mesh: str,
    *,
    material: str,
    collision: bool,
    mass: str | None = None,
    friction: str = "2.0 0.08 0.008",
    margin: str = "0.0005",
) -> None:
    """Append a GN01 mesh geom (contact-active only for the distal fingertips)."""
    attrs = {"name": name, "type": "mesh", "mesh": mesh, "quat": GN01_GEOM_QUAT, "material": material}
    if material == "lab_collision":
        # Collision-envelope OVERLAYS (translucent green) are debug visuals:
        # render group 3 is hidden by the default MjvOption, so evidence
        # videos show the black gripper like the real one; toggle geomgroup[3]
        # to inspect contact shells. (The static_assembly shell also carries a
        # vendor-STL origin offset -- it floats beside the wrist -- one more
        # reason not to draw it by default.) Physics is untouched: contact
        # flags are independent of render groups.
        attrs["group"] = "3"
    if collision:
        attrs.update({"contype": "2", "conaffinity": "1", "friction": friction,
                      "solref": "0.002 1", "solimp": "0.995 0.999 0.0005", "margin": margin})
        if mass is not None:
            attrs["mass"] = mass
    else:
        attrs.setdefault("group", "1")
        attrs.update({"contype": "0", "conaffinity": "0"})
    ET.SubElement(parent, "geom", attrs)


def append_gn01_articulated_subtree(tool: ET.Element) -> None:
    """Append the articulated GN01 4-bar linkage subtree under the tool body."""
    root = ET.SubElement(tool, "body", {"name": "gn01_articulated_root", "pos": "0 0 0", "quat": "1 0 0 0"})
    width_link = ET.SubElement(root, "body", {"name": "gn01_unused_finger_width_link", "pos": "0 0 0"})
    # Semantic total aperture in metres. A slide coordinate makes generalized
    # motor effort a force in Newtons, so the private MPPI motor and real GN01
    # Grasp(force) share physical units. The transparent carrier has no contact;
    # equality constraints alone drive the visible four-bar linkage.
    ET.SubElement(width_link, "joint", {"name": "gn01_finger_width_joint", "type": "slide", "axis": "1 0 0", "range": "0 0.100", "damping": "0.03", "armature": "0.001"})
    ET.SubElement(width_link, "geom", {"name": "gn01_unused_finger_width_mass", "type": "sphere", "size": "0.001", "mass": "0.0001", "rgba": "0 0 0 0", "contype": "0", "conaffinity": "0"})

    mesh_geom(root, "gn01_base_visual", "gn01_visual_base_mesh", material="gn01_visual_dark", collision=False)
    # Only the distal fingertip collisions are contact-active; the palm /
    # linkage stays visual so it cannot scrape the table.
    mesh_geom(root, "gn01_base_collision", "gn01_collision_base_mesh", material="lab_collision", collision=False)
    mesh_geom(root, "gn01_static_assembly_collision", "gn01_collision_static_assembly_mesh", material="lab_collision", collision=False)

    left_outer = ET.SubElement(root, "body", {"name": "gn01_left_outer_bar_body", "pos": "0 -0.0325 0.0825"})
    ET.SubElement(left_outer, "joint", {"name": "gn01_left_outer_knuckle_joint", "type": "hinge", "axis": "1 0 0", "range": "-0.155 0.7854", "damping": "0.1", "armature": "0.002"})
    mesh_geom(left_outer, "gn01_left_outer_bar_visual", "gn01_visual_outer_bar_mesh", material="gn01_visual_dark", collision=False)
    mesh_geom(left_outer, "gn01_left_outer_bar_collision", "gn01_collision_outer_bar_mesh", material="lab_collision", collision=False)
    left_mount = ET.SubElement(left_outer, "body", {"name": "gn01_left_finger_mount_body", "pos": "0 0 0.058"})
    ET.SubElement(left_mount, "joint", {"name": "gn01_left_inner_finger_joint", "type": "hinge", "axis": "1 0 0", "range": "-0.7854 0.155", "damping": "0.1", "armature": "0.002"})
    mesh_geom(left_mount, "gn01_left_finger_mount_visual", "gn01_visual_finger_mount_mesh", material="gn01_visual_dark", collision=False)
    left_tip = ET.SubElement(left_mount, "body", {"name": "gn01_left_finger_tip_body", "pos": "0 0.010 0.0235"})
    mesh_geom(left_tip, "gn01_left_finger_tip_visual", "gn01_visual_finger_tip_mesh", material="gn01_tip", collision=False)
    mesh_geom(left_tip, "gn01_left_finger_tip_collision", "gn01_collision_finger_tip_mesh", material="lab_collision", collision=True, mass="0.02", friction="1.0 0.20 0.02", margin="0.001")

    left_inner = ET.SubElement(root, "body", {"name": "gn01_left_inner_bar_body", "pos": "0 -0.0165 0.0995"})
    ET.SubElement(left_inner, "joint", {"name": "gn01_left_inner_knuckle_joint", "type": "hinge", "axis": "1 0 0", "range": "-0.155 0.7854", "damping": "0.1", "armature": "0.002"})
    mesh_geom(left_inner, "gn01_left_inner_bar_visual", "gn01_visual_inner_bar_mesh", material="gn01_visual_dark", collision=False)
    mesh_geom(left_inner, "gn01_left_inner_bar_collision", "gn01_collision_inner_bar_mesh", material="lab_collision", collision=False)

    right_outer = ET.SubElement(root, "body", {"name": "gn01_right_outer_bar_body", "pos": "0 0.0325 0.0825", "quat": "0 0 0 1"})
    ET.SubElement(right_outer, "joint", {"name": "gn01_right_outer_knuckle_joint", "type": "hinge", "axis": "1 0 0", "range": "-0.155 0.7854", "damping": "0.1", "armature": "0.002"})
    mesh_geom(right_outer, "gn01_right_outer_bar_visual", "gn01_visual_outer_bar_mesh", material="gn01_visual_dark", collision=False)
    mesh_geom(right_outer, "gn01_right_outer_bar_collision", "gn01_collision_outer_bar_mesh", material="lab_collision", collision=False)
    right_mount = ET.SubElement(right_outer, "body", {"name": "gn01_right_finger_mount_body", "pos": "0 0 0.058"})
    ET.SubElement(right_mount, "joint", {"name": "gn01_right_inner_finger_joint", "type": "hinge", "axis": "1 0 0", "range": "-0.7854 0.155", "damping": "0.1", "armature": "0.002"})
    mesh_geom(right_mount, "gn01_right_finger_mount_visual", "gn01_visual_finger_mount_mesh", material="gn01_visual_dark", collision=False)
    right_tip = ET.SubElement(right_mount, "body", {"name": "gn01_right_finger_tip_body", "pos": "0 0.010 0.0235"})
    mesh_geom(right_tip, "gn01_right_finger_tip_visual", "gn01_visual_finger_tip_mesh", material="gn01_tip", collision=False)
    mesh_geom(right_tip, "gn01_right_finger_tip_collision", "gn01_collision_finger_tip_mesh", material="lab_collision", collision=True, mass="0.02", friction="1.0 0.20 0.02", margin="0.001")

    right_inner = ET.SubElement(root, "body", {"name": "gn01_right_inner_bar_body", "pos": "0 0.0165 0.0995", "quat": "0 0 0 1"})
    ET.SubElement(right_inner, "joint", {"name": "gn01_right_inner_knuckle_joint", "type": "hinge", "axis": "1 0 0", "range": "-0.155 0.7854", "damping": "0.1", "armature": "0.002"})
    mesh_geom(right_inner, "gn01_right_inner_bar_visual", "gn01_visual_inner_bar_mesh", material="gn01_visual_dark", collision=False)
    mesh_geom(right_inner, "gn01_right_inner_bar_collision", "gn01_collision_inner_bar_mesh", material="lab_collision", collision=False)


def harden_gn01_fingertips_for_grasping(scope: ET.Element) -> None:
    """Replace fingertip MESH collision with box primitives (object-agnostic).

    The convex fingertip meshes are geometrically degenerate as graspers: once
    a pad sinks a few mm into a flat face, MPR flips the contact normal to
    VERTICAL and no horizontal grip exists. A box primitive with the same AABB
    gives clean face normals and a 4-point manifold per pad; torsional
    friction 0.5 keeps long objects from yaw-spinning out of the grip.
    """
    for geom in scope.iter("geom"):
        if geom.get("name") in ("gn01_left_finger_tip_collision", "gn01_right_finger_tip_collision"):
            geom.set("type", "box")
            geom.attrib.pop("mesh", None)
            geom.attrib.pop("margin", None)
            # Pad position calibrated so the inner GAP equals the width command
            # (measured gap(0.085)=0.0850): the collision-mesh AABB extends
            # ~7 mm past the flat rubber face per side and ate open clearance.
            geom.set("pos", "0.0001 -0.00055 0.0162")
            geom.set("quat", "0.015600 -0.002800 0.113800 0.993400")
            geom.set("size", "0.00740 0.01070 0.02080")
            # PHYSICAL rubber-pad friction (sliding 1.0, torsional 0.5, rolling
            # 0.02). NOT the former mu=8: mu=8 was a "stealth weld" -- with a 60 N
            # clamp it gives ~480 N tangential capacity against a ~2 N object
            # (~240x), so EVERY closure trivially passed the lift test and the
            # screen ranked "did contact happen", not "is the grip sound". The
            # grasp-and-transport model uses Coulomb friction rather than a
            # welded contact; the rubber-pad sliding coefficient here is 1.0.
            #   mu=8 was also masking a SOLVER gap: MuJoCo soft contacts leak a
            # residual tangential creep that a NoSlip post-pass removes. The CPU
            # twin runs NoSlip (noslip_iterations=10) and HOLDS a 110 deg pour at
            # this physical mu (measured 15/16). The warp GPU screen does NOT
            # implement NoSlip, so at physical mu it cannot hold a sustained
            # torqued grip (pour) -- which is correct and expected: grip-fidelity
            # ranking for large-reorientation belongs on the complete CPU solver
            # (mjpc's own division of labour), and the GPU screen stays the cheap
            # breadth pre-rank for lift/place/push. Do NOT re-inflate mu to make
            # the incomplete GPU solver hold pour; that is the hack this removes.
            # Canonical refinement still TODO (needs its own CPU verification):
            # priority=1 on the pad so its params win the contact combination,
            # which then also requires condim=6 on the pad (else priority drops
            # the contact to condim=3 and loses torsional friction).
            geom.set("friction", "1.0 0.50 0.02")
            # Firm rubber pad: a few newtons of clamp must not produce
            # millimetres of surface deformation, otherwise the penetration
            # gate (1.5 mm) caps the achievable grip force below 1 N.
            geom.set("solref", "0.001 1")
            geom.set("solimp", "0.999 0.9999 0.0001")


def replace_static_gn01_with_articulated(world: ET.Element) -> None:
    """Replace the static GN01 visual subtree with the articulated linkage.

    ALWAYS re-normalizes the link7->flange mount from the externally configured
    yaml (the pre-baked calibration XML carries pos="0 0 0.16" quat="1 0 0 0":
    +36 mm too long and missing the -pi wrist flip).
    """
    tool = find_body(world, "lab_gn01_tool")
    if tool is None:
        raise RuntimeError("Calibrated lab scene does not contain lab_gn01_tool.")
    mount_pos, mount_quat = official_gn01_flange_mount()
    tool.set("pos", mount_pos)
    tool.set("quat", mount_quat)
    for child in list(tool):
        tool.remove(child)
    ET.SubElement(tool, "site", {"name": "lab_gn01_tcp_site", "group": "4", "type": "sphere", "size": "0.006", "pos": "0 0 0.15", "rgba": "0.0 0.8 0.2 1"})
    ET.SubElement(tool, "site", {"name": "lab_gn01_closed_tcp_site", "group": "4", "type": "sphere", "size": "0.006", "pos": "0 0 0.20", "rgba": "0.9 0.6 0.0 1"})
    ET.SubElement(tool, "site", {"name": "gn01_grasp_center_site", "group": "4", "type": "sphere", "size": "0.008", "pos": f"0 0 {GN01_PAD_CENTER_Z_M:.6f}", "rgba": "0.0 0.9 0.2 1"})
    append_gn01_articulated_subtree(tool)
    harden_gn01_fingertips_for_grasping(tool)


def ensure_equality(root: ET.Element) -> None:
    """(Re)write the GN01 mimic-joint equality constraints on the scene root.

    Five constraints below are unit-magnitude mirrors from the main knuckle.
    The sixth is different in kind: it converts the semantic width coordinate
    into that knuckle angle using Flexiv's inherited affine mimic.  This is not
    structurally equivalent to ALOHA's directly actuated slide finger plus one
    opposite-finger mirror, despite both models containing 1:1 mirror terms.
    """
    eq = ensure_child(root, "equality")
    for item in list(eq):
        if item.attrib.get("name", "").startswith("gn01_"):
            eq.remove(item)
    defs = [
        ("gn01_left_outer_from_width", "gn01_left_outer_knuckle_joint", "gn01_finger_width_joint", "-0.155 9.404 0 0 0"),
        ("gn01_left_inner_knuckle_mimic", "gn01_left_inner_knuckle_joint", "gn01_left_outer_knuckle_joint", "0 1 0 0 0"),
        ("gn01_left_inner_finger_mimic", "gn01_left_inner_finger_joint", "gn01_left_outer_knuckle_joint", "0 -1 0 0 0"),
        ("gn01_right_outer_knuckle_mimic", "gn01_right_outer_knuckle_joint", "gn01_left_outer_knuckle_joint", "0 1 0 0 0"),
        ("gn01_right_inner_knuckle_mimic", "gn01_right_inner_knuckle_joint", "gn01_left_outer_knuckle_joint", "0 1 0 0 0"),
        ("gn01_right_inner_finger_mimic", "gn01_right_inner_finger_joint", "gn01_left_outer_knuckle_joint", "0 -1 0 0 0"),
    ]
    for name, joint1, joint2, polycoef in defs:
        ET.SubElement(eq, "joint", {"name": name, "joint1": joint1, "joint2": joint2, "polycoef": polycoef, "solref": "0.001 1", "solimp": "0.99 0.999 0.0002"})


def ensure_actuator(root: ET.Element) -> None:
    """(Re)write the GN01 signed direct-force actuator on the scene root.

    The one sampled jaw coordinate is generalized force, not desired width.
    Its sign follows Flexiv RDK ``Gripper.Grasp``: positive closes and negative
    opens. Because the semantic width coordinate increases while opening, the
    MuJoCo motor uses gear=-1. Width remains qpos/telemetry only.
    """
    actuator = ensure_child(root, "actuator")
    for item in list(actuator):
        if item.attrib.get("name") == "gn01_finger_width_pos":
            actuator.remove(item)
    ET.SubElement(
        actuator,
        "motor",
        {
            # Retain the actuator name so model/address manifests remain
            # structurally compatible; its type and physical command changed.
            "name": "gn01_finger_width_pos",
            "joint": "gn01_finger_width_joint",
            # Increasing the semantic width opens; RDK positive force closes.
            "gear": "-1",
            "ctrlrange": "-80 80",
            "forcerange": "-80 80",
        },
    )
