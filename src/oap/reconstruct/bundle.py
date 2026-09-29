"""Scene-bundle assembly, per-step idempotency engine, and the `all` DAG.

Role in the two-stage pipeline: a SCENE BUNDLE is the portable product of
Stage 1 -- one directory holding, per object, the imported observation packet
and every derived artifact, plus a ``manifest.json`` (schema
``real2sim2real_scene_manifest_v1``) with BUNDLE-RELATIVE paths that Stage 2
loads via :mod:`oap.twin.manifest`.

Idempotency contract (every step): the step directory
``<bundle>/<object>/<step>/`` carries a ``state.json`` recording the sha256 of
each input, the parameters, timestamps, and ``ok``. A step re-runs only when
an input hash or parameter changed, or on ``--force``; the ``all`` orchestrator
therefore resumes mid-DAG after any crash. Steps read only immutable prior
outputs and never modify another step's directory.

Step DAG (per object)::

    capture -> mesh -> scale -> collision -> pose -> tracking -> priors -> manifest
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from oap.reconstruct import capture as capture_step
from oap.reconstruct import collision as collision_step
from oap.reconstruct import mesh as mesh_step
from oap.reconstruct import pose as pose_step
from oap.reconstruct import priors as priors_step
from oap.reconstruct import scale as scale_step
from oap.reconstruct import tracking as tracking_step
from oap.reconstruct.external import ExternalEnvs
from oap.reconstruct.quality import QUALITY_PROFILE
from oap.utils.io import (
    read_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

logger = logging.getLogger("oap.reconstruct.bundle")

__all__ = [
    "BUNDLE_SCHEMA",
    "MANIFEST_SCHEMA",
    "STEP_ORDER",
    "assemble_manifest",
    "load_registry",
    "run_all",
    "run_step",
    "update_registry",
]

BUNDLE_SCHEMA = "oap_scene_bundle_v1"
MANIFEST_SCHEMA = "real2sim2real_scene_manifest_v1"
_FIXED_CAMERA_BUNDLE_PATH = Path("calibration/T_base_zed2i.yaml")

#: Canonical per-object step order (``manifest`` is the bundle-level final
#: step handled by :func:`assemble_manifest`).
STEP_ORDER = (
    "capture",
    "mesh",
    "scale",
    "collision",
    "pose",
    "tracking",
    "priors",
)


# --------------------------------------------------------------------------- #
# bundle registry (objects + prompts + roles, so per-step CLIs need no re-spec)
# --------------------------------------------------------------------------- #
def _registry_path(bundle: Path) -> Path:
    return Path(bundle) / "bundle.json"


def load_registry(bundle: Path) -> dict[str, Any]:
    """Load ``<bundle>/bundle.json`` (empty skeleton if absent)."""
    path = _registry_path(bundle)
    if path.exists():
        reg = read_json(path)
        if reg.get("schema") != BUNDLE_SCHEMA:
            raise RuntimeError(
                f"{path} has schema {reg.get('schema')!r}, expected {BUNDLE_SCHEMA!r}"
            )
        return reg
    return {
        "schema": BUNDLE_SCHEMA,
        "capture_dir": None,
        "objects": {},
        # name -> the label handed to the VLM for PHYSICAL PRIORS, when it must
        # differ from the SAM3 prompt. One string used to serve both, and they
        # pull opposite ways: segmentation wants a broad phrase that matches,
        # priors want a specific noun that identifies the material. Measured
        # 2026-07-27 -- widening "black blackboard eraser" to "black eraser"
        # lifted the SAM3 score 0.408 -> 0.883 and simultaneously made the VLM
        # call the object a "smartphone", moving its mass 43 g -> 263 g.
        "labels": {},
        "static": [],
        "open_top": [],
    }


def _object_label(reg: Mapping[str, Any], name: str, prompt: str) -> str:
    """The label the VLM sees for physical priors: explicit if given, else the
    SAM3 prompt (which is what every bundle written before 2026-07-27 has)."""
    return str((reg.get("labels") or {}).get(name) or prompt)


def update_registry(
    bundle: Path,
    *,
    objects: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    capture_dir: Path | None = None,
    static: Iterable[str] | None = None,
    open_top: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Merge object names/prompts + roles into the bundle registry and save it.

    Args:
        bundle: Bundle root directory.
        objects: name -> SAM3 prompt entries to add/update.
        capture_dir: Source capture directory (recorded for provenance and as
            the default for later per-step ``capture`` imports).
        static: Object names whose twin is static (``movable: false``).
        open_top: Object names modeled as open-top containers.

    Returns:
        The updated registry payload.
    """
    reg = load_registry(bundle)
    if objects:
        reg["objects"] = {**reg.get("objects", {}), **{str(k): str(v) for k, v in objects.items()}}
    if labels:
        reg["labels"] = {**reg.get("labels", {}), **{str(k): str(v) for k, v in labels.items()}}
    if capture_dir is not None:
        reg["capture_dir"] = str(Path(capture_dir).resolve())
    if static is not None:
        reg["static"] = sorted(set(reg.get("static", [])) | {str(s) for s in static})
    if open_top is not None:
        reg["open_top"] = sorted(set(reg.get("open_top", [])) | {str(s) for s in open_top})
    Path(bundle).mkdir(parents=True, exist_ok=True)
    write_json_atomic(_registry_path(bundle), reg)
    return reg


# --------------------------------------------------------------------------- #
# per-step state (idempotency)
# --------------------------------------------------------------------------- #
def _hash_inputs(inputs: Sequence[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for p in inputs:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(
                f"step input missing: {p} -- run the upstream step first "
                f"(see the capture->mesh->scale->collision->pose->priors DAG)."
            )
        hashes[str(p)] = sha256_file(p)
    return hashes


def _iter_output_files(value: Any) -> Iterable[Path]:
    """Yield existing output files named anywhere in a step result record."""
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_output_files(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _iter_output_files(child)
    elif isinstance(value, (str, Path)):
        candidate = Path(value)
        if candidate.is_file():
            yield candidate


def _hash_outputs(outputs: Mapping[str, Any]) -> dict[str, str]:
    """Hash every materialized file referenced by a step's output record."""
    hashes = {
        str(path): sha256_file(path)
        for path in _iter_output_files(outputs)
    }
    if not hashes:
        raise RuntimeError(
            "step completed without a hashable output file; generated-only "
            "bundles require every derived artifact to be integrity-bound"
        )
    return hashes


def _recorded_outputs_are_current(state: Mapping[str, Any]) -> bool:
    recorded = state.get("output_files")
    if not isinstance(recorded, Mapping) or not recorded:
        # Old state records did not bind their outputs. Force one automatic
        # regeneration instead of trusting artifacts that may have been edited.
        return False
    for raw_path, expected in recorded.items():
        path = Path(str(raw_path))
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def _step_is_current(step_dir: Path, inputs: Sequence[Path], params: Mapping[str, Any]) -> bool:
    state_path = Path(step_dir) / "state.json"
    if not state_path.exists():
        return False
    try:
        state = read_json(state_path)
    except Exception:
        return False
    if not state.get("ok"):
        return False
    if state.get("params") != dict(params):
        return False
    try:
        current = _hash_inputs(inputs)
    except FileNotFoundError:
        # A missing input makes lineage unverifiable. Never promote a cached
        # derived artifact merely because it survived a partial bundle sync.
        return False
    return state.get("inputs") == current and _recorded_outputs_are_current(state)


def _mark_step_ok(
    step_dir: Path,
    inputs: Sequence[Path],
    params: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    started_unix_s: float,
) -> None:
    write_json_atomic(
        Path(step_dir) / "state.json",
        {
            "ok": True,
            "inputs": _hash_inputs(inputs),
            "params": dict(params),
            "outputs": dict(outputs),
            "output_files": _hash_outputs(outputs),
            "started_unix_s": started_unix_s,
            "finished_unix_s": time.time(),
        },
    )


def _packet(bundle: Path, name: str) -> Path:
    return capture_step.packet_dir_in_bundle(Path(bundle), name)


def _refined_json(bundle: Path, name: str) -> Path:
    """The manifest-facing refined pose (the pose step's output)."""
    return Path(bundle) / name / "pose" / f"{name}_refined.json"


def _tracking_lineage(
    bundle: Path,
    name: str,
) -> tuple[Path, Path, Path | None, list[Path], dict[str, Any]]:
    """Return the one canonical tracking-step source lineage and parameters."""

    metric_obj = Path(bundle) / name / "scale" / f"{name}_metric.obj"
    metric_texture = metric_obj.with_name(metric_obj.stem + "_texture.png")
    sized = (
        Path(bundle)
        / name
        / "pose"
        / "overlay"
        / f"{name}_metric_any6d_sized_mesh.obj"
    )
    sized_mesh = sized if sized.is_file() else None
    inputs = [
        metric_obj,
        metric_texture,
        _refined_json(bundle, name),
    ]
    if sized_mesh is not None:
        inputs.append(sized_mesh)
    params = {
        "step": "tracking",
        "algorithm": "quadric_edge_collapse_seam_safe_uv_v2",
        "target_faces": tracking_step.DEFAULT_TRACKING_TARGET_FACES,
        "output_format": "self_contained_textured_glb",
        "appearance_transfer": "per_face_source_chart_no_uv_averaging",
        "manual_alignment": False,
    }
    return metric_obj, metric_texture, sized_mesh, inputs, params


def _require_current_tracking_step(bundle: Path, name: str) -> dict[str, Any]:
    """Fail closed unless tracking state, sources, outputs and QA are current."""

    step_dir = Path(bundle) / name / "tracking"
    (
        metric_obj,
        metric_texture,
        sized_mesh,
        inputs,
        params,
    ) = _tracking_lineage(bundle, name)
    if not _step_is_current(step_dir, inputs, params):
        raise RuntimeError(
            f"manifest assembly: object {name!r} tracking step is missing, "
            "stale, or has modified outputs; re-run the tracking step"
        )
    expected_outputs = {
        "tracking_mesh": str(step_dir / f"{name}_tracking.glb"),
        "visual_mesh": str(step_dir / f"{name}_visual.obj"),
        "metadata": str(step_dir / f"{name}_tracking.json"),
    }
    state = read_json(step_dir / "state.json")
    if state.get("outputs") != expected_outputs:
        raise RuntimeError(
            f"manifest assembly: object {name!r} tracking state does not name "
            "the canonical release outputs"
        )
    return tracking_step.validate_tracking_release(
        step_dir / f"{name}_tracking.json",
        metric_obj=metric_obj,
        metric_texture=metric_texture,
        tracking_mesh=step_dir / f"{name}_tracking.glb",
        visual_mesh=step_dir / f"{name}_visual.obj",
        sized_mesh=sized_mesh,
    )


# --------------------------------------------------------------------------- #
# step dispatch
# --------------------------------------------------------------------------- #
def run_step(
    bundle: Path,
    name: str,
    step: str,
    envs: ExternalEnvs,
    *,
    capture_dir: Path | None = None,
    force: bool = False,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one DAG step for one object, honoring the idempotency contract.

    Args:
        bundle: Bundle root directory.
        name: Object name.
        step: One of :data:`STEP_ORDER`.
        envs: Resolved external environments.
        capture_dir: Source capture dir (``capture`` step only; falls back to
            the registry's recorded ``capture_dir``).
        force: Re-run even when inputs/params are unchanged.
        options: Step-specific options (see the CLI flags).

    Returns:
        ``{"skipped": True}`` when the step was current, else the step's
        result payload.

    Raises:
        ValueError: On an unknown step name.
        FileNotFoundError: When an upstream output is missing.
    """
    bundle = Path(bundle)
    opts = dict(options or {})
    step_dir = bundle / name / step
    packet = _packet(bundle, name)
    reg = load_registry(bundle)
    prompt = str(reg.get("objects", {}).get(name, name))
    started = time.time()

    if step == "capture":
        src_dir = Path(capture_dir) if capture_dir else (
            Path(reg["capture_dir"]) if reg.get("capture_dir") else None
        )
        if (packet / "observation_manifest.json").exists() and not force:
            capture_step.reanchor_manifest(packet)
            capture_step.validate_observation_packet(packet, require_mask=True)
            logger.info("[all:%s] capture: current, skipping", name)
            return {"skipped": True}
        if src_dir is None:
            raise FileNotFoundError(
                f"object {name!r} has no imported packet and no --capture-dir was "
                f"given (registry has none either); run `oap-reconstruct "
                f"capture` on the lab host first."
            )
        capture_step.import_packet(bundle, name, src_dir, force=force)
        return {"packet": str(packet)}

    if step == "pose" and opts.get("t_base_camera") not in (None, ""):
        raise ValueError(
            "per-object t_base_camera overrides are forbidden: the fixed "
            "camera uses the one packaged rig calibration"
        )

    packet_manifest = packet / "observation_manifest.json"
    if step in {"mesh", "scale", "pose", "tracking", "priors"}:
        # Every packet-consuming step revalidates the immutable fixed-camera
        # evidence. This prevents a modified score, mask, depth, intrinsics, or
        # camera serial from being hidden behind an otherwise-current cache.
        capture_step.validate_observation_packet(packet, require_mask=True)

    if step == "mesh":
        inputs = [
            packet / "rgb.png",
            packet / "mask_00.png",
            packet_manifest,
        ]
        # Heterogeneous JSON-ish step-param record (hashed by _step_is_current).
        params: dict[str, Any] = {
            "step": "mesh",
            "texture_baking": bool(opts.get("texture_baking", True)),
            "mesh_only_lowmem": bool(opts.get("mesh_only_lowmem", False)),
            "mesh_extract_device": str(opts.get("mesh_extract_device", "cuda")),
            "max_sparse_coords": opts.get("max_sparse_coords"),
            "checkpoint_tag": str(opts.get("checkpoint_tag", "hf")),
            "seed": int(opts.get("seed", 42)),
        }
        if not force and _step_is_current(step_dir, inputs, params):
            logger.info("[all:%s] mesh: current, skipping", name)
            return {"skipped": True}
        result = mesh_step.run_mesh_step(
            step_dir,
            packet,
            envs,
            name=name,
            texture_baking=bool(params["texture_baking"]),
            mesh_only_lowmem=bool(params["mesh_only_lowmem"]),
            mesh_extract_device=str(params["mesh_extract_device"]),
            max_sparse_coords=params["max_sparse_coords"],
            checkpoint_tag=str(params["checkpoint_tag"]),
            seed=int(params["seed"]),
            allow_untextured=bool(opts.get("allow_untextured", False)),
        )
        _mark_step_ok(step_dir, inputs, params, {"metadata": result.get("saved", {})},
                      started_unix_s=started)
        return result

    if step == "scale":
        sam3d_output = bundle / name / "mesh" / "sam3d_output"
        mesh_file = sam3d_output / "mesh.glb"
        if not mesh_file.exists():
            mesh_file = sam3d_output / "mesh_raw.obj"
        _, scale_calibration, _, _ = (
            pose_step.load_fixed_camera_calibration()
        )
        inputs = [
            mesh_file,
            packet / "depth_m.npy",
            packet / "mask_00.png",
            packet / "cam_K.txt",
            packet_manifest,
            Path(scale_calibration),
        ]
        params = {
            "step": "scale",
            "algorithm": (
                "rgbd_gravity_trimmed_extent_isotropic_scale.v1"
            ),
            "quality_profile": QUALITY_PROFILE.to_dict(),
        }
        if not force and _step_is_current(step_dir, inputs, params):
            logger.info("[all:%s] scale: current, skipping", name)
            return {"skipped": True}
        result = scale_step.run_scale_step(
            step_dir,
            packet,
            bundle / name / "mesh",
            name=name,
        )
        _mark_step_ok(step_dir, inputs, params,
                      {"metric_obj": str(step_dir / f"{name}_metric.obj"),
                       "scale_decision": result["scale_report"].get("scale_decision")},
                      started_unix_s=started)
        return result

    metric_obj = bundle / name / "scale" / f"{name}_metric.obj"

    if step == "collision":
        inputs = [metric_obj]
        params = {"step": "collision"}
        if not force and _step_is_current(step_dir, inputs, params):
            logger.info("[all:%s] collision: current, skipping", name)
            return {"skipped": True}
        result = collision_step.run_collision_step(step_dir, metric_obj, name=name)
        _mark_step_ok(step_dir, inputs, params,
                      {"collision_obj": str(step_dir / f"{name}_collision.obj")},
                      started_unix_s=started)
        return result

    if step == "pose":
        # Bind this packet to the fixed camera before spending GPU minutes.
        packet_camera = capture_step.validate_packet_camera_serial(
            read_json(packet / "observation_manifest.json"),
            packet=packet,
        )
        _, calib_source, camera_serial, calib_sha256 = (
            pose_step.load_fixed_camera_calibration()
        )
        if packet_camera["camera_serial"] != camera_serial:
            raise ValueError(
                f"object {name!r} packet camera serial "
                f"{packet_camera['camera_serial']} does not match fixed camera "
                f"{camera_serial}"
            )
        inputs = [
            metric_obj,
            packet / "rgb.png",
            packet / "depth_m.npy",
            packet / "mask_00.png",
            packet / "cam_K.txt",
            packet_manifest,
            Path(calib_source),
        ]
        params = {
            "step": "pose",
            "downscale": float(opts.get("downscale", 0.5)),
            "iterations": int(opts.get("iterations", 5)),
            "camera_serial": camera_serial,
            "t_base_camera_sha256": calib_sha256,
        }
        if not force and _step_is_current(step_dir, inputs, params):
            logger.info("[all:%s] pose: current, skipping", name)
            return {"skipped": True}
        result = pose_step.run_pose_step(
            step_dir,
            packet,
            metric_obj,
            envs,
            name=name,
            downscale=float(params["downscale"]),
            iterations=int(params["iterations"]),
        )
        _mark_step_ok(step_dir, inputs, params,
                      {"refined_json": str(step_dir / f"{name}_refined.json")},
                      started_unix_s=started)
        return result

    if step == "tracking":
        (
            tracking_metric,
            metric_texture,
            sized_mesh,
            inputs,
            params,
        ) = _tracking_lineage(bundle, name)
        if tracking_metric != metric_obj:
            raise RuntimeError(
                "internal tracking lineage disagrees with metric source"
            )
        if not force and _step_is_current(step_dir, inputs, params):
            logger.info("[all:%s] tracking: current, skipping", name)
            return {"skipped": True}
        result = tracking_step.run_tracking_step(
            step_dir,
            metric_obj,
            metric_texture,
            name=name,
            sized_mesh=sized_mesh,
            target_faces=tracking_step.DEFAULT_TRACKING_TARGET_FACES,
        )
        _mark_step_ok(
            step_dir,
            inputs,
            params,
            {
                "tracking_mesh": str(step_dir / f"{name}_tracking.glb"),
                "visual_mesh": str(step_dir / f"{name}_visual.obj"),
                "metadata": str(step_dir / f"{name}_tracking.json"),
            },
            started_unix_s=started,
        )
        return result

    if step == "priors":
        crop_image = bundle / name / "mesh" / "sam3d_input" / "image.png"
        refined = _refined_json(bundle, name)
        inputs = [crop_image, refined, packet / "mask_00.png", packet_manifest]
        params = {
            "step": "priors",
            "model_id": str(opts.get("model_id") or "profile_default"),
            "object_label": _object_label(reg, name, prompt),
        }
        if not force and _step_is_current(step_dir, inputs, params):
            logger.info("[all:%s] priors: current, skipping", name)
            return {"skipped": True}
        size = read_json(refined).get("size_LWH_m")
        if not size or len(size) != 3:
            raise RuntimeError(f"refined pose {refined} has no size_LWH_m for the priors step")
        result = priors_step.run_priors_step(
            step_dir,
            crop_image,
            (float(size[0]), float(size[1]), float(size[2])),
            envs,
            name=name,
            object_label=_object_label(reg, name, prompt),
            model_id=opts.get("model_id"),
            mask=packet / "mask_00.png",
        )
        _mark_step_ok(step_dir, inputs, params,
                      {"priors_json": str(step_dir / f"{name}_priors.json")},
                      started_unix_s=started)
        return result

    raise ValueError(f"unknown step {step!r}; expected one of {STEP_ORDER}")


# --------------------------------------------------------------------------- #
# manifest assembly + the `all` orchestrator
# --------------------------------------------------------------------------- #
def _materialize_fixed_camera_calibration(
    bundle: Path,
    *,
    source: Path,
    expected_sha256: str,
) -> str:
    """Atomically embed the exact packaged fixed-camera calibration."""
    source_path = Path(source).expanduser().resolve()
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            "manifest assembly: packaged fixed-camera calibration is "
            f"unavailable at {source_path}"
        ) from exc
    source_sha256 = sha256_bytes(payload)
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            "manifest assembly: fixed-camera calibration changed while the "
            "bundle was assembled "
            f"(expected {expected_sha256}, observed {source_sha256})"
        )

    target = Path(bundle) / _FIXED_CAMERA_BUNDLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    materialized_sha256 = sha256_file(target)
    if materialized_sha256 != expected_sha256:
        raise RuntimeError(
            "manifest assembly: embedded fixed-camera calibration failed its "
            f"SHA-256 check at {target}"
        )
    return _FIXED_CAMERA_BUNDLE_PATH.as_posix()


def assemble_manifest(
    bundle: Path,
    *,
    wall_thickness_m: float = 0.006,
    description: str | None = None,
) -> Path:
    """Assemble ``<bundle>/manifest.json`` with bundle-relative paths.

    Reads the bundle registry (objects, static/open-top roles) and points each
    entry at the reconstructed artifacts: the self-contained simplified
    tracking GLB, full-resolution visual mesh, convex collision proxy, refined
    pose, and physical priors. Paths are RELATIVE to the manifest so the
    bundle can be synced between hosts and loaded anywhere.

    Args:
        bundle: Bundle root directory.
        wall_thickness_m: Wall thickness applied to open-top containers.
        description: Optional human description embedded in the manifest.

    Returns:
        The manifest path.

    Raises:
        FileNotFoundError: If a referenced artifact is missing (its step has
            not completed for some object).
    """
    bundle = Path(bundle)
    reg = load_registry(bundle)
    _, calibration_source, fixed_camera_serial, calibration_sha256 = (
        pose_step.load_fixed_camera_calibration()
    )
    objects = reg.get("objects", {})
    if not objects:
        raise RuntimeError(
            f"bundle {bundle} has no registered objects; run "
            f"`oap-reconstruct all --objects \"name=prompt\" ...` first."
        )
    static = set(reg.get("static", []))
    open_top = set(reg.get("open_top", []))

    entries: list[dict[str, Any]] = []
    for name, prompt in objects.items():
        refined_abs = _refined_json(bundle, name)
        _require_current_tracking_step(bundle, name)
        capture_step.validate_observation_packet(
            _packet(bundle, name), require_mask=True
        )
        packet_camera = capture_step.validate_packet_camera_serial(
            read_json(_packet(bundle, name) / "observation_manifest.json"),
            packet=_packet(bundle, name),
        )
        refined = read_json(refined_abs)
        if (
            packet_camera["camera_serial"] != fixed_camera_serial
            or str(refined.get("camera_serial", "")) != fixed_camera_serial
        ):
            raise RuntimeError(
                f"manifest assembly: object {name!r} is not bound to fixed "
                f"camera serial {fixed_camera_serial}; re-capture with the "
                f"calibrated camera and re-run the pose step."
            )
        if refined.get("t_base_camera_sha256") != calibration_sha256:
            raise RuntimeError(
                f"manifest assembly: object {name!r} pose was not produced "
                f"with the current fixed-camera calibration "
                f"{calibration_sha256}; re-run the pose step."
            )
        rel = {
            "mesh": f"{name}/tracking/{name}_tracking.glb",
            "visual_mesh": f"{name}/tracking/{name}_visual.obj",
            "texture": f"{name}/scale/{name}_metric_texture.png",
            "tracking_metadata": (
                f"{name}/tracking/{name}_tracking.json"
            ),
            "collision": f"{name}/collision/{name}_collision.obj",
            "refined_json": refined_abs.relative_to(bundle).as_posix(),
            "priors_json": f"{name}/priors/{name}_priors.json",
        }
        # Prefer the pose step's jointly-refined geometry when it exists. The
        # scale step measures a WORLD-axis-aligned extent -- an upper bound
        # that over-reads a yawed box by up to sqrt(2) -- while Any6D refines
        # the mesh's own object-frame size against depth (render-and-compare,
        # overlay-verifiable). Keep visual + collision consistent by rescaling
        # the convex collision proxy with the same per-axis ratio.
        metric_source = bundle / name / "scale" / f"{name}_metric.obj"
        sized = bundle / name / "pose" / "overlay" / f"{name}_metric_any6d_sized_mesh.obj"
        if sized.exists():
            import trimesh

            metric = trimesh.load(str(metric_source), force="mesh", process=False)
            sized_m = trimesh.load(str(sized), force="mesh", process=False)
            assert isinstance(metric, trimesh.Trimesh) and isinstance(sized_m, trimesh.Trimesh)
            ratio = np.asarray(sized_m.extents) / np.maximum(np.asarray(metric.extents), 1e-9)
            if float(np.max(np.abs(ratio - 1.0))) > 0.05:
                # convex hull of the sized mesh: frame-consistent by construction
                # (open-top containers get analytic walls in the twin instead).
                sized_coll = bundle / name / "pose" / f"{name}_collision_sized.obj"
                sized_m.convex_hull.export(str(sized_coll))
                logger.warning(
                    "[manifest] %s: using the Any6D-sized geometry (per-axis ratio %s "
                    "vs the world-AABB metric mesh); collision = its convex hull.",
                    name, np.round(ratio, 3).tolist())
                rel["collision"] = sized_coll.relative_to(bundle).as_posix()
        for role, rel_path in rel.items():
            if not (bundle / rel_path).exists():
                raise FileNotFoundError(
                    f"manifest assembly: object {name!r} is missing its {role} at "
                    f"{bundle / rel_path}; run the corresponding step first."
                )
        entry: dict[str, Any] = {
            "name": name,
            "label": _object_label(reg, name, prompt),
            "sam3_prompt": prompt,
            "movable": name not in static,
            **rel,
            "artifact_sha256": {
                rel_path: sha256_file(bundle / rel_path)
                for rel_path in rel.values()
            },
        }
        if name in open_top:
            entry["open_top_container"] = True
            entry["wall_thickness_m"] = float(wall_thickness_m)
        entries.append(entry)

    if not any(e["movable"] for e in entries):
        logger.warning(
            "[manifest] no movable object in %s -- Stage 2 will refuse this "
            "manifest; check the --static flags.", bundle)

    calibration_config = _materialize_fixed_camera_calibration(
        bundle,
        source=Path(calibration_source),
        expected_sha256=calibration_sha256,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "description": description
        or f"Reconstructed scene bundle ({', '.join(objects)}); paths are bundle-relative.",
        "camera_calibration": {
            "mode": "fixed_eye_to_hand",
            "camera_serial": fixed_camera_serial,
            "sha256": calibration_sha256,
            "config": calibration_config,
        },
        "objects": entries,
    }
    path = bundle / "manifest.json"
    write_json_atomic(path, manifest)
    logger.info("[manifest] wrote %s (%d objects)", path, len(entries))
    return path


def run_all(
    bundle: Path,
    envs: ExternalEnvs,
    *,
    capture_dir: Path | None = None,
    objects: Mapping[str, str] | None = None,
    static: Iterable[str] = (),
    open_top: Iterable[str] = (),
    force: bool = False,
    step_options: Mapping[str, Mapping[str, Any]] | None = None,
    wall_thickness_m: float = 0.006,
    description: str | None = None,
    labels: Mapping[str, str] | None = None,
) -> Path:
    """Walk the full reconstruction DAG for every object, then write the manifest.

    Resume-safe: completed steps (unchanged inputs + params, ``ok`` state) are
    skipped, so re-running after a crash or an OOM continues where it stopped.

    Args:
        bundle: Bundle root directory.
        envs: Resolved external environments.
        capture_dir: Source capture directory (required on first run).
        objects: name -> SAM3 prompt map (merged into the registry).
        static: Objects to mark ``movable: false`` in the manifest.
        open_top: Objects modeled as open-top containers.
        force: Re-run every step regardless of state.
        step_options: Per-step option maps, e.g. ``{"pose": {"downscale": 0.5}}``.
        wall_thickness_m: Open-top container wall thickness.
        description: Manifest description.

    Returns:
        The assembled manifest path.
    """
    bundle = Path(bundle)
    reg = update_registry(
        bundle,
        objects=objects,
        labels=labels,
        capture_dir=capture_dir,
        static=static,
        open_top=open_top,
    )
    names = list(reg.get("objects", {}))
    if not names:
        raise RuntimeError("no objects registered; pass --objects \"name=prompt\" ...")

    opts = {k: dict(v) for k, v in (step_options or {}).items()}
    for name in names:
        for step in STEP_ORDER:
            logger.info("[all:%s] step %s", name, step)
            run_step(
                bundle,
                name,
                step,
                envs,
                capture_dir=capture_dir,
                force=force,
                options=opts.get(step),
            )
    return assemble_manifest(
        bundle, wall_thickness_m=wall_thickness_m, description=description
    )
