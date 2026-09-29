"""Assemble the W_plan MJCF digital twin from a scene bundle.

Role in the two-stage pipeline: this is the seam between the offline
reconstruction stage (scene bundle: meshes + refined poses + priors) and the
online closed loop. :func:`build_plan_scene` writes a self-contained MuJoCo
scene containing the calibrated lab robot + curated GN01, the measured table
(patched by an optional table-spec json), every reconstructed object (textured
visual mesh + analytic object-axis collision cuboids sized by the refined
LWH), prior-derived mass/friction, and a home-posture keyframe seeded from the
same json the real home-gate enforces. Object poses are RESTORED FROM
RECONSTRUCTION only; live tracking updates runtime state, while the movable
subject becomes the free ``pick_object`` body.

:func:`load_world` compiles a written twin into a :class:`SimWorld` handle for
the rollout machinery.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from oap.twin.assets import (
    GN01_OPEN_WIDTH_M,
    ensure_child,
    ensure_gn01_meshes,
    ensure_material,
    ensure_actuator,
    ensure_equality,
    find_body,
    find_geom,
    gripper_joint_targets,
    indent_xml,
    strip_recorded_free_bodies,
    replace_static_gn01_with_articulated,
    require_child,
    rewrite_calibrated_robot_asset_paths,
    calibration_scene_xml,
    find_asset_workspace,
)
from oap.twin.obb import load_base_camera_matrix, object_body_quat
from oap.twin.types import (
    LAB_RESTORED_INITIAL_QPOS,
    LAB_TABLE_POS,
    LAB_TABLE_SIZE,
    SceneObject,
    SimWorld,
)
from oap.utils.io import load_yaml, package_config_path, sha256_file
from oap.utils.se3 import quat_conj, quat_mul, quat_to_mat

logger = logging.getLogger("oap.twin.scene")

__all__ = [
    "build_plan_scene", "load_world", "map_robot", "set_arm_ctrl",
    "observe_object_pose", "subject_is_held", "assert_clean_episode_start",
    "require_id",
    "add_calibrated_zed_camera",
    "make_table_physical_and_visible", "set_grasp_solver_options",
    "strengthen_arm_position_controller", "add_visible_mount_adapter",
]

TWIN_XML_NAME = "W_plan_real2sim2real.xml"


# ------------------------------------------------------------- scene styling
def _upsert_texture(asset: ET.Element, *, name: str, attrs: dict[str, str]) -> None:
    for texture in asset.findall("texture"):
        if texture.attrib.get("name") == name:
            texture.attrib.clear()
            texture.set("name", name)
            texture.attrib.update(attrs)
            return
    elem = ET.SubElement(asset, "texture", {"name": name})
    elem.attrib.update(attrs)


def _upsert_material(asset: ET.Element, *, name: str, attrs: dict[str, str]) -> None:
    for material in asset.findall("material"):
        if material.attrib.get("name") == name:
            material.attrib.update(attrs)
            return
    elem = ET.SubElement(asset, "material", {"name": name})
    elem.attrib.update(attrs)


def _remove_named_world_geoms(world: ET.Element, names: set[str]) -> None:
    for parent in world.iter():
        for child in list(parent):
            if child.tag == "geom" and child.attrib.get("name") in names:
                parent.remove(child)


# Gamma applied to baked object textures for display. MuJoCo's classic
# renderer shades texels linearly and never gamma-encodes the framebuffer,
# so sRGB-authored textures (SAM3D bakes) render with crushed mid-tones.
TEXTURE_DISPLAY_GAMMA = 1.6

_FILL_LIGHTS: tuple[dict[str, str], ...] = (
    {"name": "qa_fill_key", "pos": "1.2 0.6 1.5", "dir": "-0.5 -0.4 -1",
     "diffuse": "0.45 0.45 0.45", "specular": "0.05 0.05 0.05"},
    {"name": "qa_fill_side", "pos": "0.2 -0.8 1.4", "dir": "0.3 0.6 -1",
     "diffuse": "0.35 0.35 0.35", "specular": "0.05 0.05 0.05"},
)


def apply_lab_lighting(root: ET.Element) -> None:
    """Bright, even lighting so twin renders match the well-lit real lab.

    The base scene ships a single overhead light with near-zero ambient;
    combined with the linear (no-sRGB) classic renderer that makes textured
    objects look 2-3x darker than the real ZED frames. Headlight ambient plus
    two directional fills close most of that gap; lights never affect physics.
    """
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.35 0.35 0.35")
    headlight.set("diffuse", "0.45 0.45 0.45")
    headlight.set("specular", "0.10 0.10 0.10")
    world = require_child(root, "worldbody")
    fill_names = {spec["name"] for spec in _FILL_LIGHTS}
    for light in list(world.findall("light")):
        if light.attrib.get("name") in fill_names:
            world.remove(light)
    for spec in _FILL_LIGHTS:
        ET.SubElement(world, "light", dict(spec))


def _display_texture(path: Path, output_dir: Path) -> Path:
    """Return a gamma-lifted, content-addressed display texture.

    The reconstruction bundle is immutable input.  Derived renderer assets
    therefore live under the twin build directory, keyed by the source
    texture's SHA-256 (and namespaced by the transform version).  A temporary
    file in that same directory is atomically renamed into place so concurrent
    builders never expose a partial PNG.  Display conversion is optional:
    every failure falls back to the read-only source texture.
    """
    temporary: Path | None = None
    try:
        source_sha256 = sha256_file(path)
        gamma_namespace = f"gamma-{TEXTURE_DISPLAY_GAMMA:g}".replace(".", "p")
        cache_dir = (
            Path(output_dir)
            / "derived"
            / "display_textures"
            / gamma_namespace
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = cache_dir / f"{source_sha256}.png"
        if out.is_file():
            return out

        from PIL import Image

        with Image.open(path) as source_image:
            im = (
                np.asarray(source_image.convert("RGB"), dtype=np.float32)
                / 255.0
            )
        lifted = (np.power(im, 1.0 / TEXTURE_DISPLAY_GAMMA) * 255.0).astype(np.uint8)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=cache_dir,
            prefix=f".{out.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            Image.fromarray(lifted).save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, out)
        except OSError:
            # Windows can reject one of two simultaneous replacements even
            # after both writers have closed their unique temporary handles.
            # The other writer has nevertheless published the same
            # content-addressed artifact atomically, so reuse it.
            if not out.is_file():
                raise
            temporary.unlink(missing_ok=True)
        temporary = None
        return out
    except Exception as exc:  # noqa: BLE001 - display conversion is optional
        logger.warning(
            "[texture] could not derive display texture for %s (%s); "
            "using read-only source",
            path,
            exc,
        )
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def apply_scene_style(root: ET.Element, workspace: Path) -> None:
    """Force the canonical lab-scene visual style (wood table, grid floor).

    Texture files are optional (they live in the asset workspace under
    ``assets/lab_scene/textures``); a flat material / builtin checker is the
    fallback so a texture-less workspace still builds a valid twin.
    """
    asset = require_child(root, "asset")
    world = require_child(root, "worldbody")

    table_texture = workspace / "assets" / "lab_scene" / "textures" / "lab_table_wood_maniskill_tabletop.png"
    if table_texture.exists():
        _upsert_texture(asset, name="qa_tabletop_texture",
                        attrs={"type": "2d", "file": str(table_texture.resolve()).replace("\\", "/")})
        _upsert_material(asset, name="lab_table", attrs={
            "texture": "qa_tabletop_texture",
            "texrepeat": "2 1.25",
            "texuniform": "true",
            "rgba": "1 1 1 1",
            "specular": "0.18",
            "shininess": "0.25",
        })
    else:
        _upsert_material(asset, name="lab_table", attrs={"rgba": "0.82 0.62 0.38 1", "reflectance": "0.12"})

    floor_texture = workspace / "assets" / "lab_scene" / "textures" / "maniskill_grid_texture.png"
    if floor_texture.exists():
        _upsert_texture(asset, name="qa_dark_grid_texture",
                        attrs={"type": "2d", "file": str(floor_texture.resolve()).replace("\\", "/")})
    else:
        _upsert_texture(asset, name="qa_dark_grid_texture", attrs={
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.36 0.37 0.39",
            "rgb2": "0.46 0.47 0.49",
            "width": "512",
            "height": "512",
            "mark": "edge",
            "markrgb": "0.64 0.66 0.68",
        })
    _upsert_material(asset, name="qa_dark_grid_floor", attrs={
        "texture": "qa_dark_grid_texture",
        "texrepeat": "24 24",
        "rgba": "0.68 0.70 0.72 1",
        "specular": "0.08",
        "shininess": "0.08",
    })
    _remove_named_world_geoms(world, {"lab_floor", "qa_dark_grid_floor", "qa_wall_back", "qa_wall_left", "qa_wall_right"})
    ET.SubElement(world, "geom", {
        "name": "lab_floor",
        "type": "plane",
        "pos": f"0 0 {-0.899643:.6f}",
        "size": "50.000000 50.000000 0.010000",
        "material": "qa_dark_grid_floor",
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    })


def make_table_physical_and_visible(world: ET.Element) -> None:
    """Ensure the lab table exists as a collidable box at the calibrated pose."""
    table = find_geom(world, "lab_table")
    if table is None:
        ET.SubElement(world, "geom", {
            "name": "lab_table",
            "type": "box",
            "pos": f"{LAB_TABLE_POS[0]:.8f} {LAB_TABLE_POS[1]:.8f} {LAB_TABLE_POS[2]:.8f}",
            "size": f"{LAB_TABLE_SIZE[0]:.8f} {LAB_TABLE_SIZE[1]:.8f} {LAB_TABLE_SIZE[2]:.8f}",
            "material": "lab_table",
            "friction": "1.0 0.005 0.0001",
        })
        return
    table.set("type", "box")
    table.set("pos", f"{LAB_TABLE_POS[0]:.8f} {LAB_TABLE_POS[1]:.8f} {LAB_TABLE_POS[2]:.8f}")
    table.set("size", f"{LAB_TABLE_SIZE[0]:.8f} {LAB_TABLE_SIZE[1]:.8f} {LAB_TABLE_SIZE[2]:.8f}")
    table.set("material", table.attrib.get("material", "lab_table"))
    table.set("contype", "1")
    table.set("conaffinity", "1")
    table.set("friction", "1.0 0.005 0.0001")
    # Match the manipulated objects' contact model.  MuJoCo's softer default
    # lets the controller sink a supported object into the tabletop and can
    # make the unchanged initial state fail the penetration validity gate.
    table.set("solref", "0.001 1")
    table.set("solimp", "0.999 0.9999 0.0001")


def add_visible_mount_adapter(world: ET.Element) -> None:
    """Add visible flange/adapter geometry without moving the calibrated tool."""
    link7 = find_body(world, "link7")
    if link7 is None:
        return
    for child in list(link7):
        if child.tag == "geom" and child.attrib.get("name") in {"lab_link7_to_gn01_adapter_visual", "lab_gn01_flange_visual"}:
            link7.remove(child)
    ET.SubElement(link7, "geom", {"name": "lab_gn01_flange_visual", "type": "cylinder", "pos": "0 0 0.083", "size": "0.050 0.010", "material": "lab_flange_silver", "contype": "0", "conaffinity": "0", "group": "1"})
    ET.SubElement(link7, "geom", {"name": "lab_link7_to_gn01_adapter_visual", "type": "cylinder", "pos": "0 0 0.120", "size": "0.040 0.045", "material": "gn01_visual_dark", "contype": "0", "conaffinity": "0", "group": "1"})


def set_grasp_solver_options(root: ET.Element, *, noslip_iterations: int = 10) -> None:
    """Use one contact model in GPU planning and full-fidelity execution.

    MJWarp supports elliptic friction with ``impratio=10`` but not MuJoCo's
    NoSlip post-pass. The live/full-fidelity twin uses the same cone and
    impedance ratio so a rollout cannot be selected under one friction model
    and executed under another; it additionally retains NoSlip to suppress
    sustained-grasp creep.
    """
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")
    opt.set("noslip_iterations", str(int(noslip_iterations)))


def strengthen_arm_position_controller(root: ET.Element) -> None:
    """Use a strong position servo for simulator-only planning rollouts.

    The calibrated lab XML inherits conservative torque limits from the raw
    Rizon arm model; a lower-gain controller left the grasp-center site
    2-3 cm below its IK target under gravity, producing false gripper/table
    interpenetration. Strict contact/penetration gates still decide whether
    the resulting motion is certifiable.
    """
    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")
        if name.startswith("joint") and name[5:].isdigit():
            joint.set("actuatorfrcrange", "-12000 12000")
    actuator = ensure_child(root, "actuator")
    for item in actuator.findall("position"):
        name = item.attrib.get("name", "")
        if name.startswith("joint") and name[5:].isdigit():
            item.set("kp", "15000")
            item.set("kv", "800")
            item.set("forcerange", "-12000 12000")


# -------------------------------------------------------------- table spec
def _apply_table_spec(world: ET.Element, spec: dict[str, Any], *, table_spec_height: bool, spec_name: str) -> None:
    """Patch the twin table geom from a measured table-spec json.

    Keys: ``table_half_xy_m``, ``table_surface_center_base_m`` (z applied only
    when ``table_spec_height``), ``table_normal_base`` (tilt), ``base_plate``
    (the robot's real mounting plate, a collidable obstacle the twin was
    missing).
    """
    tbl = next((g for g in world.iter("geom") if g.get("name") == "lab_table"), None)
    if tbl is None:
        return
    sz = [float(v) for v in tbl.attrib["size"].split()]
    ps = [float(v) for v in tbl.attrib["pos"].split()]
    if spec.get("table_half_xy_m"):
        sz[0], sz[1] = float(spec["table_half_xy_m"][0]), float(spec["table_half_xy_m"][1])
    c = spec.get("table_surface_center_base_m")
    if c is not None:
        ps[0], ps[1] = float(c[0]), float(c[1])
        if table_spec_height:
            ps[2] = float(c[2]) - sz[2]
    n = spec.get("table_normal_base")
    if n is not None:
        n = np.asarray(n, dtype=float)
        n = n / np.linalg.norm(n)
        ax = np.cross([0.0, 0.0, 1.0], n)
        s = float(np.linalg.norm(ax))
        q = [1.0, 0.0, 0.0, 0.0]
        if s > 1e-9:
            ax = ax / s
            ang = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
            q = [np.cos(ang / 2.0), *(np.sin(ang / 2.0) * ax)]
        tbl.set("quat", " ".join(f"{float(v):.8f}" for v in q))
    tbl.set("size", " ".join(f"{v:.6f}" for v in sz))
    tbl.set("pos", " ".join(f"{v:.6f}" for v in ps))
    logger.info(
        "[table-spec] twin table <- %s: size %s center_xy %s height_z=%.3f%s",
        spec_name, [round(v, 3) for v in sz], [round(ps[0], 3), round(ps[1], 3)], ps[2],
        "" if table_spec_height else " (sim height kept; objects rest)")
    plate = spec.get("base_plate")
    if plate is not None:
        # The robot's real mounting plate sits ON the table around the pedestal --
        # a collidable obstacle the twin was missing. Placed relative to the SIM
        # table top (same convention as objects), so gripper<->plate clearance
        # matches reality regardless of the plan-frame z offset.
        phx, phy = (float(v) for v in plate.get("half_xy_m", [0.11, 0.11]))
        pt = float(plate.get("thickness_m", 0.019))
        pcx, pcy = (float(v) for v in plate.get("center_xy_base_m", [0.0, 0.0]))
        table_quat = tbl.get("quat", "1 0 0 0")
        normal = quat_to_mat(np.fromstring(table_quat, sep=" "))[:, 2]
        # Seat the plate on the tilted top plane at its own XY position.
        table_top = ps[2] + (
            sz[2] - normal[0] * (pcx - ps[0]) - normal[1] * (pcy - ps[1])
        ) / normal[2]
        plate_z = table_top + 0.5 * pt / normal[2]
        _remove_named_world_geoms(world, {"lab_base_plate"})
        ET.SubElement(world, "geom", {
            "name": "lab_base_plate", "type": "box",
            "pos": f"{pcx:.6f} {pcy:.6f} {plate_z:.9f}",
            "size": f"{phx:.6f} {phy:.6f} {pt / 2.0:.6f}",
            "quat": table_quat,
            "material": "lab_flange_silver",
            "contype": "1", "conaffinity": "1",
            "friction": "1.0 0.005 0.0001",
        })
        logger.info(
            "[table-spec] base plate added: %.2fx%.2fx%.3f m at xy=(%.2f,%.2f), "
            "resting on the twin table top (z=%.3f)",
            2 * phx, 2 * phy, pt, pcx, pcy, table_top)


# ------------------------------------------------------------------ camera
def _scaled_intrinsics_for_render(
    intrinsics: dict[str, Any], *, render_width: int, render_height: int
) -> tuple[np.ndarray, int, int]:
    """Scale recovered camera intrinsics to the requested render resolution."""
    k = np.asarray(intrinsics["K"], dtype=float).copy()
    native_width = int(intrinsics["width"])
    native_height = int(intrinsics["height"])
    if render_width <= 0 or render_height <= 0:
        raise ValueError("render_width/render_height must be positive")
    sx = float(render_width) / float(native_width)
    sy = float(render_height) / float(native_height)
    k[0, :] *= sx
    k[1, :] *= sy
    return k, int(render_width), int(render_height)


def _load_intrinsics(path: Path) -> dict[str, Any]:
    """Load a camera-intrinsics yaml/json ({K, width, height})."""
    doc = load_yaml(path)
    k = np.asarray(doc.get("K"), dtype=float)
    if k.shape != (3, 3):
        raise ValueError(f"{path} does not contain a 3x3 K matrix")
    width = int(doc.get("width", doc.get("image_width")))
    height = int(doc.get("height", doc.get("image_height")))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} must contain positive width/height or image_width/image_height")
    return {"doc": doc, "K": k, "width": width, "height": height}


def _camera_xyaxes_from_opencv_pose(t_base_camera_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV camera axes to MuJoCo camera xyaxes (R_mj = R_cv diag(1,-1,-1))."""
    r_base_cv = t_base_camera_cv[:3, :3]
    r_base_mj = r_base_cv @ np.diag([1.0, -1.0, -1.0])
    return r_base_mj[:, 0], r_base_mj[:, 1]


def _camera_xyaxes_from_lookat(camera_pos: np.ndarray, lookat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build MuJoCo fixed-camera axes from a position and target point."""
    camera_pos = np.asarray(camera_pos, dtype=float)
    lookat = np.asarray(lookat, dtype=float)
    forward = lookat - camera_pos
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        raise ValueError("camera position and look-at point are identical")
    forward = forward / norm
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-8:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
    right = right / right_norm
    up = np.cross(right, forward)
    up = up / float(np.linalg.norm(up))
    return right, up


def _mujoco_principalpixel_from_opencv(k: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """Convert OpenCV's image principal point to MuJoCo's frustum offset."""
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    return float(width) * 0.5 - cx, float(height) * 0.5 - cy


def _plan_frame_z_offset(world: ET.Element) -> float:
    """Read the real-base to plan-frame translation from the seated base."""
    base = world.find("body[@name='base']")
    if base is None:
        return 0.0
    return float(base.get("pos", "0 0 0").split()[2])


def _vec(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def add_calibrated_zed_camera(
    *,
    xml_path: Path,
    camera_name: str = "zed2i_rgbd_sim",
    candidate_path: Path | None = None,
    intrinsics_path: Path | None = None,
    camera_mode: str = "exact",
    lookat: tuple[float, float, float] = (0.78, -0.045, 0.02),
    render_width: int = 1280,
    render_height: int = 720,
) -> None:
    """Insert the calibrated simulated ZED RGB-D camera into a written twin.

    Shares the real ZED extrinsics (packaged hand-eye calibration by default)
    and intrinsics so the sim render is an observation-consistent comparison
    with the real frames, not an unrelated debug camera. When no intrinsics
    file is available the camera is skipped with a log line -- planning does
    not depend on it, only side-by-side evidence videos do.
    """
    if intrinsics_path is None or not Path(intrinsics_path).exists():
        logger.info("[camera] no ZED intrinsics file (%s) -> sim camera skipped", intrinsics_path)
        return
    intrinsics = _load_intrinsics(Path(intrinsics_path))
    k, width, height = _scaled_intrinsics_for_render(
        intrinsics, render_width=int(render_width), render_height=int(render_height))
    t_base_camera = load_base_camera_matrix(candidate_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    world = require_child(root, "worldbody")
    for camera in list(world.findall("camera")):
        if camera.attrib.get("name") == camera_name:
            world.remove(camera)
    camera_pos = np.asarray(t_base_camera[:3, 3], dtype=float).copy()
    camera_pos[2] += _plan_frame_z_offset(world)
    if camera_mode == "exact":
        x_axis, y_axis = _camera_xyaxes_from_opencv_pose(t_base_camera)
    elif camera_mode == "lookat":
        x_axis, y_axis = _camera_xyaxes_from_lookat(camera_pos, np.asarray(lookat, dtype=float))
    else:
        raise ValueError(f"Unsupported camera mode: {camera_mode}")
    principal_offset = _mujoco_principalpixel_from_opencv(k, width, height)
    ET.SubElement(world, "camera", {
        "name": camera_name,
        "pos": _vec(camera_pos),
        "xyaxes": f"{_vec(x_axis)} {_vec(y_axis)}",
        "resolution": f"{int(width)} {int(height)}",
        "sensorsize": "1 1",
        "focalpixel": f"{float(k[0, 0]):.9g} {float(k[1, 1]):.9g}",
        "principalpixel": f"{principal_offset[0]:.9g} {principal_offset[1]:.9g}",
    })
    indent_xml(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


# --------------------------------------------------------------- twin build
def build_plan_scene(
    objects: list[SceneObject],
    subject: SceneObject,
    subject_pose0: tuple[float, float, float, float],
    z_real_to_plan: float,
    output_dir: Path,
    *,
    workspace: Path | None = None,
    table_spec_json: Path | None = None,
    table_spec_height: bool = False,
    home_posture_json: Path | None = None,
    camera_candidate: Path | None = None,
    camera_intrinsics: Path | None = None,
    camera_name: str = "zed2i_rgbd_sim",
    render_width: int = 1280,
    render_height: int = 720,
) -> Path:
    """Write the W_plan twin: the calibrated lab scene + every bundle object.

    The subject becomes the free ``pick_object`` body (the contact bookkeeping
    matches geoms BY NAME containing "pick_object"); every other object is a
    static body at its reconstructed base-frame pose. All collisions are
    analytic cuboids sized by the refined LWH and aligned to the OBJECT axes
    (reconstructed hulls have rounded bottoms that rock and slide), with
    prior-derived mass/friction and condim=6.

    Args:
        objects: All scene objects from :func:`load_scene_manifest`.
        subject: The movable object (must be one of ``objects``).
        subject_pose0: The subject's observed PLAN-frame (x, y, z, yaw).
        z_real_to_plan: Real-frame -> plan-frame z offset (measured table).
        output_dir: Where the twin XML is written.
        workspace: Asset workspace (default: :func:`find_asset_workspace`).
        table_spec_json: Optional measured table spec (size/center/tilt/plate).
        table_spec_height: Also apply the spec's z as the sim table height.
        home_posture_json: Posture json seeding the keyframe (default: the
            packaged ``calibration/lab_home_posture.json`` the home-gate uses).
        camera_candidate: Hand-eye T_base_camera yaml (default packaged).
        camera_intrinsics: ZED intrinsics yaml; camera skipped when None.
        camera_name: Name of the inserted sim camera.
        render_width: Sim camera render width.
        render_height: Sim camera render height.

    Returns:
        The written twin XML path (``W_plan_real2sim2real.xml``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = workspace or find_asset_workspace()
    tree = ET.parse(calibration_scene_xml(workspace))
    root = tree.getroot()
    rewrite_calibrated_robot_asset_paths(root, workspace)
    strip_recorded_free_bodies(root)
    asset = require_child(root, "asset")
    world = require_child(root, "worldbody")
    base = world.find("body[@name='base']")
    if base is None:
        raise RuntimeError("the calibrated scene has no robot base body")
    base_pos = np.fromstring(base.get("pos", "0 0 0"), sep=" ")
    base_pos[2] = float(z_real_to_plan)
    base.set("pos", _vec(base_pos))
    visual = root.find("visual")
    g = visual.find("global") if visual is not None else None
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if g is None:
        g = ET.SubElement(visual, "global")
    g.set("offwidth", "1280")
    g.set("offheight", "720")
    ensure_material(asset, "lab_flange_silver", (0.62, 0.64, 0.62, 1.0))
    ensure_material(asset, "gn01_visual_dark", (0.015, 0.015, 0.014, 1.0))
    ensure_material(asset, "gn01_tip", (0.035, 0.033, 0.030, 1.0))
    ensure_material(asset, "lab_collision", (0.05, 0.45, 0.25, 0.55))
    apply_scene_style(root, workspace)
    apply_lab_lighting(root)
    ensure_gn01_meshes(asset, workspace)
    make_table_physical_and_visible(world)
    if table_spec_json is not None:
        spec = json.loads(Path(table_spec_json).read_text(encoding="utf-8"))
        _apply_table_spec(world, spec, table_spec_height=table_spec_height,
                          spec_name=Path(table_spec_json).name)
    add_visible_mount_adapter(world)
    # Includes the fingertip box-primitive hardening (degenerate mesh-pad
    # contact normals) AND the curated flange mount (never bypassed).
    replace_static_gn01_with_articulated(world)
    # Suppress solver friction creep for sustained grasps (noslip post-pass).
    set_grasp_solver_options(root)

    for obj in objects:
        m = ET.SubElement(asset, "mesh")
        m.set("name", f"{obj.name}_visual_mesh")
        # Render-only channel: the full-res textured mesh when the bundle
        # ships one; the FP-safe (decimated) base mesh otherwise.
        m.set("file", str((obj.visual_mesh or obj.mesh).resolve()))
        # Baked UV texture -> a MuJoCo 2D texture + material so the twin renders
        # real appearance (the OBJ carries texcoords). Falls back to flat rgba.
        if obj.texture is not None and Path(obj.texture).exists():
            tex = ET.SubElement(asset, "texture")
            tex.set("name", f"{obj.name}_tex")
            tex.set("type", "2d")
            tex.set(
                "file",
                str(_display_texture(Path(obj.texture), output_dir).resolve()),
            )
            mat = ET.SubElement(asset, "material")
            mat.set("name", f"{obj.name}_mat")
            mat.set("texture", f"{obj.name}_tex")
            mat.set("specular", "0.1")
            mat.set("shininess", "0.3")

    # load_scene_manifest derives q_om for every object at load time.
    assert subject.q_om is not None
    subject_quat = object_body_quat(subject_pose0[3], subject.q_om)
    # The subject arrives ALREADY SEATED on its support: runner.py seats it
    # before calling this, so the emitted body pos, the keyframe and the
    # runtime reset_object_pose all carry the same height.  Seating here
    # instead was measured to be useless -- the XML was right and the run
    # still started 6.850 mm high, because runner.py:1998 resets the pose
    # from its own copy of subject_pose0 and this function cannot reach it.
    # Free-body poses in EMISSION order, which is the order MuJoCo assigns
    # freejoint qpos addresses. The keyframe below is one flat qpos vector, so
    # it needs a 7-tuple per free body: a keyframe shorter than nq leaves the
    # trailing bodies at zero -- spawned at the world origin, inside the table,
    # and the settle loop then ejects them half a metre. Accumulating here
    # (rather than appending the subject last) also keeps the vector right when
    # a movable support precedes the subject in the manifest.
    free_qpos: list[float] = []
    for obj in objects:
        assert obj.q_om is not None
        L, W, H = obj.size_lwh
        if obj is subject:
            body = ET.SubElement(world, "body")
            body.set("name", "pick_object")
            body.set("pos", f"{subject_pose0[0]:.6f} {subject_pose0[1]:.6f} {subject_pose0[2]:.6f}")
            body.set("quat", " ".join(f"{float(v):.6f}" for v in subject_quat))
            ET.SubElement(body, "freejoint", {"name": "pick_object_freejoint"})
            free_qpos.extend([subject_pose0[0], subject_pose0[1], subject_pose0[2],
                              *(float(v) for v in subject_quat)])
            cname = "pick_object_collision"
        else:
            # Objects always carry their generated reconstruction pose:
            # load_scene_manifest raises otherwise, and sets both together.
            assert obj.pose_base is not None and obj.pose_quat is not None
            pose = list(obj.pose_base)
            if not obj.z_snapped_plan_frame:
                pose[2] += z_real_to_plan   # real-frame pose -> plan frame
            body = ET.SubElement(world, "body")
            body.set("name", f"support_{obj.name}")
            if obj.movable:
                # A NON-SUBJECT movable body: the robot does not hold it, but
                # the world can move it -- a box a tool pushes, a lid that
                # slides. It needs a joint or the physics cannot represent the
                # goal at all. The body keeps its ``support_`` name so every
                # existing geom/name lookup is unchanged; only the joint is new.
                # Its z stays TABLE-SNAPPED at t=0 (manifest.py pins the least
                # reliable reconstruction axis to the measured table) -- physics
                # takes over from a seated pose rather than from a guess that
                # may start it interpenetrating or afloat.
                ET.SubElement(body, "freejoint", {"name": f"{obj.name}_freejoint"})
                free_qpos.extend([float(pose[0]), float(pose[1]), float(pose[2]),
                                  *(float(v) for v in obj.pose_quat)])
            body.set("pos", " ".join(f"{v:.6f}" for v in pose))
            # The body frame IS the mesh frame: its world quat comes from the
            # reconstruction pose, NOT from the mesh->object axis map.
            body.set("quat", " ".join(f"{float(v):.6f}" for v in obj.pose_quat))
            cname = f"support_{obj.name}_collision"
        gv = ET.SubElement(body, "geom")
        gv.set("type", "mesh")
        gv.set("mesh", f"{obj.name}_visual_mesh")
        # The visual mesh is RENDER-ONLY: without mass="0" MuJoCo adds its own
        # solid-mesh mass on top of the collision geom's, so the body weighs
        # roughly twice what the priors say (measured on the shipped bear twin:
        # 0.5588 -> 0.2224 kg). Menagerie's robotiq_2f85 uses the same
        # convention for its pad visuals.
        gv.set("mass", "0")
        if obj is subject and obj.pose_quat is not None and obj.yaw_ref_rad is not None:
            # Cancel the body's axis map before applying the reconstruction
            # visual rotation, so visuals are not rotated twice.
            half = 0.5 * float(obj.yaw_ref_rad)
            q_yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            q_vis = quat_mul(
                quat_conj(np.asarray(obj.q_om, float)),
                quat_mul(quat_conj(q_yaw), np.asarray(obj.pose_quat, float)),
            )
            gv.set("quat", " ".join(f"{float(v):.6f}" for v in q_vis))
        if obj.texture is not None and Path(obj.texture).exists():
            gv.set("material", f"{obj.name}_mat")   # baked texture
        else:
            gv.set("rgba", " ".join(f"{float(v):.3f}" for v in obj.rgba))
        gv.set("contype", "0")
        gv.set("conaffinity", "0")
        # The cuboid(s) are axis-aligned in the OBJECT frame; the body frame is
        # the MESH frame, so the local rotation is mesh<-object = conj(q_om).
        # Using q_om here composes the axis map TWICE (a reference's collision
        # once stood UP as an invisible wall across the carry corridor).
        cquat = quat_conj(obj.q_om)
        cR = quat_to_mat(cquat)  # object-frame offset -> body(mesh)-frame pos

        def _emit_box(suffix: str, half_xyz: tuple[float, float, float],
                      center_obj: tuple[float, float, float],
                      mass_share: float = 1.0) -> None:
            geo = ET.SubElement(body, "geom")
            geo.set("name", f"{cname}{suffix}")
            dataset_cylinder = (
                suffix == ""
                and getattr(obj, "collision_primitive", "box") == "cylinder"
            )
            if dataset_cylinder:
                geo.set("type", "cylinder")
                geo.set(
                    "size",
                    f"{min(float(half_xyz[0]), float(half_xyz[1])):.5f} "
                    f"{float(half_xyz[2]):.5f}",
                )
            else:
                geo.set("type", "box")
                geo.set("size", " ".join(f"{float(v):.5f}" for v in half_xyz))
            geo.set("pos", " ".join(f"{float(v):.6f}" for v in (cR @ np.asarray(center_obj, float))))
            geo.set("quat", " ".join(f"{float(v):.6f}" for v in cquat))
            geo.set("rgba", "0 0 0 0")
            # Physical params from the priors stage, not hand-set. EVERY body
            # gets its mass, not only the subject: a static support never needed
            # one, but a body that can move does, and which bodies move is a
            # manifest property, not something the scene builder should assume.
            # ``mass_share`` splits an object's mass across a multi-geom
            # decomposition (a container's floor + 4 walls) by geom volume.
            geo.set("mass", f"{obj.mass_kg * float(mass_share):.5f}")
            # Shell inertia: these collision geoms are primitive BOXES, which is
            # the one case MuJoCo still honours shellinertia (it is ignored for
            # meshes since 3.3.0). Real manipulanda are shells, not solid
            # blocks; measured on the amazon-box extents the tensor grows
            # 1.40-1.54x, and on the thin wall slabs it is a 1.05-1.15x
            # near-no-op -- so it applies uniformly with no branch. This is
            # rotational dynamics (tipping, pouring), not just mass.
            if not dataset_cylinder:
                geo.set("shellinertia", "true")
            geo.set("friction", " ".join(f"{float(v):.4f}" for v in obj.friction))
            geo.set("condim", "6")          # sliding+torsional+rolling (else free spin on the normal)
            geo.set("solref", "0.001 1")    # rigid: match the stiffened pad contact
            geo.set("solimp", "0.999 0.9999 0.0001")

        if obj.open_top_container:
            # Open bin/tray: hollow box = thin FLOOR + 4 WALLS (open top), so a
            # subject can be lowered INSIDE and rest on the floor -- a solid
            # cuboid would make it rest ON TOP and fail the "inside" criterion.
            # Object frame: x=L, y=W, z=H(up); floor at -H/2, rim at +H/2.
            t = float(obj.wall_thickness_m)
            hl, hw, hh = L / 2.0, W / 2.0, H / 2.0
            parts = [("_floor", (hl, hw, t / 2), (0.0, 0.0, -hh + t / 2)),
                     ("_wxp", (t / 2, hw, hh), (+hl - t / 2, 0.0, 0.0)),
                     ("_wxn", (t / 2, hw, hh), (-hl + t / 2, 0.0, 0.0)),
                     ("_wyp", (hl - t, t / 2, hh), (0.0, +hw - t / 2, 0.0)),
                     ("_wyn", (hl - t, t / 2, hh), (0.0, -hw + t / 2, 0.0))]
            # Split the object's mass across the shell by geom VOLUME, so the
            # five pieces sum to obj.mass_kg. Without the split each piece
            # would carry the full mass and the container would weigh 5x its
            # prior -- invisible while containers were static, load-bearing the
            # moment one of them can move.
            vols = [8.0 * hx * hy * hz for _, (hx, hy, hz), _ in parts]
            total_v = sum(vols) or 1.0
            for (suffix, half_xyz, center), v in zip(parts, vols):
                _emit_box(suffix, half_xyz, center, mass_share=v / total_v)
        else:
            _emit_box("", (L / 2, W / 2, H / 2), obj.collision_center_obj)

    ensure_equality(root)
    ensure_actuator(root)
    strengthen_arm_position_controller(root)
    # Seed the twin's start posture from the CANONICAL HOME json (the posture
    # the home-gate enforces on the real arm), not the benchmark constant --
    # identical today, but if the lab home is ever re-recorded the twin must
    # plan chunk 0 from the same posture.
    home_qpos = list(LAB_RESTORED_INITIAL_QPOS.tolist())
    if home_posture_json is None:
        try:
            home_posture_json = package_config_path("calibration/lab_home_posture.json")
        except FileNotFoundError:
            home_posture_json = None
    if home_posture_json is not None:
        try:
            hq = json.loads(Path(home_posture_json).read_text(encoding="utf-8"))["q_rad"]
            if len(hq) == len(home_qpos):
                home_qpos = [float(v) for v in hq]
        except Exception as e:  # noqa: BLE001 - fall back to the known-good constant
            logger.warning("[twin-home] could not read %s (%s); using LAB_RESTORED_INITIAL_QPOS",
                           home_posture_json, e)
    # One 7-tuple per emitted freejoint, or the keyframe is short and MuJoCo
    # pads the tail silently -- it even normalizes the zero quaternion, so a
    # body would spawn at the world origin with no compiler error. Check the
    # counts here, where the discrepancy is still visible.
    n_free = len(root.findall(".//freejoint"))
    if len(free_qpos) != 7 * n_free:
        raise RuntimeError(
            f"scene has {n_free} free bodies but the keyframe carries "
            f"{len(free_qpos) / 7:.3g} poses; every freejoint needs one.")
    qpos = [
        *home_qpos,
        *gripper_joint_targets(GN01_OPEN_WIDTH_M),
        *free_qpos,
    ]
    # Negative signed force actively opens GN01; width remains qpos state.
    ctrl = [*home_qpos, -80.0]
    for kf in root.findall("keyframe"):
        for key in kf.findall("key"):
            # Seed EVERY keyframe (the calibrated base scene ships a stale
            # 'home' key with a generic posture; leaving it un-seeded is a
            # trap -- any renderer resetting to 'home' shows the arm ~20 cm
            # off the true raised home, while the loop itself uses
            # 'lab_snapshot' and never noticed).
            key.set("qpos", " ".join(f"{float(v):.9g}" for v in qpos))
            key.set("ctrl", " ".join(f"{float(v):.9g}" for v in ctrl))

    out_xml = output_dir / TWIN_XML_NAME
    tree.write(out_xml)
    if camera_candidate is None:
        try:
            camera_candidate = package_config_path("calibration/T_base_zed2i.yaml")
        except FileNotFoundError:
            camera_candidate = None
    if camera_intrinsics is None:
        try:
            camera_intrinsics = package_config_path("calibration/zed2i_intrinsics.yaml")
        except FileNotFoundError:
            camera_intrinsics = None
    add_calibrated_zed_camera(
        xml_path=out_xml, camera_name=camera_name, candidate_path=camera_candidate,
        intrinsics_path=camera_intrinsics, camera_mode="exact",
        render_width=render_width, render_height=render_height)
    return out_xml


# ----------------------------------------------------------- world handles
def require_id(model: Any, obj_type: Any, name: str) -> int:
    """Resolve a named MuJoCo object id or raise."""
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Missing MuJoCo object {obj_type}: {name}")
    return int(idx)


def map_robot(model: Any) -> dict[str, Any]:
    """Resolve the arm/gripper joint + actuator handles of a compiled twin."""
    robot_joint_ids = [require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    return {
        "joint_ids": robot_joint_ids,
        "qpos_addr": np.asarray([model.jnt_qposadr[jid] for jid in robot_joint_ids], dtype=int),
        "dof_addr": np.asarray([model.jnt_dofadr[jid] for jid in robot_joint_ids], dtype=int),
        "joint_ranges": np.asarray([model.jnt_range[jid] for jid in robot_joint_ids], dtype=float),
        "gripper_joint_ids": {
            "finger_width": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_finger_width_joint"),
            "left_outer": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_left_outer_knuckle_joint"),
            "left_inner": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_left_inner_knuckle_joint"),
            "left_finger": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_left_inner_finger_joint"),
            "right_outer": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_right_outer_knuckle_joint"),
            "right_inner": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_right_inner_knuckle_joint"),
            "right_finger": require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "gn01_right_inner_finger_joint"),
        },
        "gripper_actuator_id": require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gn01_finger_width_pos"),
        "arm_actuator_ids": np.asarray([require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{i}") for i in range(1, 8)], dtype=int),
    }


def set_arm_ctrl(data: Any, robot: dict[str, Any], q: np.ndarray) -> None:
    """Write the 7 arm position-actuator targets."""
    for actuator_id, value in zip(robot["arm_actuator_ids"], np.asarray(q, dtype=float)):
        data.ctrl[int(actuator_id)] = float(value)


def load_world(xml_path: Path, name: str) -> SimWorld:
    """Compile a written twin XML into a keyframed :class:`SimWorld` handle.

    Physics belongs to the batched MJWarp rollout. The host-side ``MjData`` is
    only an orchestration/observation mirror, so loading it must not advance CPU
    MuJoCo dynamics or contacts. The complete ``lab_snapshot`` keyframe supplies
    the initial generalized state and control; ``mj_kinematics`` refreshes only
    pose-dependent caches before that state is copied to the GPU.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    key = require_id(model, mujoco.mjtObj.mjOBJ_KEY, "lab_snapshot")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_resetDataKeyframe(model, ik_data, key)
    robot = map_robot(model)
    set_arm_ctrl(data, robot, LAB_RESTORED_INITIAL_QPOS)
    data.ctrl[robot["gripper_actuator_id"]] = -80.0
    mujoco.mj_kinematics(model, data)
    object_joint = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "pick_object_freejoint")
    object_qpos_addr = int(model.jnt_qposadr[object_joint])
    # name -> qpos address for EVERY free body. The subject is included under
    # its manifest name; ``object_qpos_addr`` above stays the subject's, so
    # nothing that indexes it changes.
    movable_qpos_addr: dict[str, int] = {}
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if jname.endswith("_freejoint"):
            movable_qpos_addr[jname[: -len("_freejoint")]] = int(model.jnt_qposadr[jid])
    site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "gn01_grasp_center_site")
    return SimWorld(
        name=name,
        model=model,
        data=data,
        ik_data=ik_data,
        robot=robot,
        object_qpos_addr=object_qpos_addr,
        movable_qpos_addr=movable_qpos_addr,
        site_id=site_id,
        q=LAB_RESTORED_INITIAL_QPOS.copy(),
    )


def subject_is_held(world: SimWorld) -> bool:
    """CPU parity utility: measure whether both pads touch the subject.

    Production uses sample-aligned MJWarp contact evidence and never calls this
    helper.

    Both finger pads must be in contact with the subject's collider. The
    held-state the loop carries between stages was previously pure MEMORY: a
    width was written when a closing stage finished and no code path ever wrote
    it back, so after a release -- or after the object simply fell out -- the
    system still believed it was holding. That belief is load-bearing twice
    over: it activates the measured holding-continuity gate (which keeps the
    jaw shut while a grasp is physically present), and ``safety.safe_to_home``
    refuses to home while it is set. Stale memory therefore disabled homing
    and made loss of grasp invisible for the rest of the episode.

    Measured on a SCRATCH MjData (``world.ik_data``) seeded from the live state,
    so asking the question cannot perturb the answer for anything downstream.
    """
    m = world.model
    names = ("pick_object_collision",
             "gn01_left_finger_tip_collision", "gn01_right_finger_tip_collision")
    gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names]
    if min(gids) < 0:
        return False
    obj, lp, rp = gids
    d = world.ik_data
    d.qpos[:] = world.data.qpos
    d.qvel[:] = world.data.qvel
    mujoco.mj_forward(m, d)
    left = right = False
    for c in range(d.ncon):
        pair = {int(d.contact[c].geom1), int(d.contact[c].geom2)}
        if pair == {obj, lp}:
            left = True
        elif pair == {obj, rp}:
            right = True
    return left and right


def assert_clean_episode_start(world: SimWorld) -> None:
    """CPU scene diagnostic; not the production episode-start gate.

    Production checks the same invariant from candidate-consistent MJWarp
    ``initial_robot_scene_contact`` evidence before the global first prefix.
    This helper remains only for parity tests and offline scene diagnosis.

    A physics-in-the-loop planner scores candidates by rolling them from the
    CURRENT state. If the arm is already resting on a body it has not grasped,
    every rollout inherits that contact: the body is pressed into the table, its
    friction cone is loaded by a position-controlled servo that behaves like a
    wall, and a push that the real scene permits is impossible in the twin. The
    2026-07-26 tool-use bundle hit exactly this -- the home posture's right
    finger tip rested on the reconstructed box (dist -0.03 mm), which sank it
    3 mm into the table and pinned it so hard that 10 N of direct force moved it
    0.1 mm. Nothing failed; the numbers were just quietly wrong.

    All four bundles that predate that scene start with ZERO such contacts, so
    this is a real defect signal, not a tolerance to tune. The fix belongs on
    the TABLE (move the object out of the arm's rest volume and re-capture), not
    in a threshold here.
    """
    from oap.twin.batched_rollout import exact_arm_collision_geoms

    m, d = world.model, world.data
    robot_gids = set(
        exact_arm_collision_geoms(m, require_complete=False)
    )
    robot_gids.update(
        gid
        for gid in range(m.ngeom)
        if (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        ).startswith("gn01_")
        and (
            int(m.geom_contype[gid])
            or int(m.geom_conaffinity[gid])
        )
    )
    offenders = []
    for c in range(d.ncon):
        gids = (int(d.contact[c].geom1), int(d.contact[c].geom2))
        robot = [gid for gid in gids if gid in robot_gids]
        scene = [
            gid
            for gid in gids
            if (
                mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_GEOM, gid
                ) or ""
            ).startswith(("support_", "pick_object"))
        ]
        if robot and scene:
            robot_name = (
                mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_GEOM, robot[0]
                )
                or mujoco.mj_id2name(
                    m,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(m.geom_bodyid[robot[0]]),
                )
                or f"geom_{robot[0]}"
            )
            scene_name = (
                mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_GEOM, scene[0]
                )
                or f"geom_{scene[0]}"
            )
            offenders.append(
                (
                    robot_name,
                    scene_name,
                    1000.0 * float(d.contact[c].dist),
                )
            )
    if offenders:
        detail = "; ".join(f"{r} touches {s} ({p:+.2f} mm)" for r, s, p in offenders[:4])
        raise RuntimeError(
            f"episode would start with the robot already in contact with "
            f"{len(offenders)} scene geom(s): {detail}. Every rollout inherits "
            f"this contact, so the physics the planner scores is not the physics "
            f"of the real scene. Move the object out of the arm's rest volume on "
            f"the table and re-capture.")


def observe_object_pose(world: SimWorld) -> tuple[float, float, float, float, float, float, float]:
    """Read the subject's freejoint qpos (x, y, z, qw, qx, qy, qz)."""
    qpos = np.asarray(world.data.qpos[world.object_qpos_addr : world.object_qpos_addr + 7], dtype=float)
    x, y, z, qw, qx, qy, qz = (float(v) for v in qpos.tolist())
    return (x, y, z, qw, qx, qy, qz)
