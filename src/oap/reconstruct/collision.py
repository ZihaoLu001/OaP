"""Step COLLISION: reconstructed-only convex collision proxy.

Role in the two-stage pipeline: MuJoCo needs a convex, watertight collision
geometry; the SAM3D visual mesh is neither. This step builds a trimesh convex
hull FROM THE RECONSTRUCTED METRIC VISUAL MESH -- never from any official or
privileged asset (anti-leakage: ``official_asset_geometry_used: false`` is
recorded in the metadata, schema
``countersim.reconstructed_convex_collision_mesh.v1``). Manifest assembly
selects the generated collision artifact automatically and records its
SHA-256; production exposes no per-object collision override.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("oap.reconstruct.collision")

__all__ = ["build_convex_collision", "run_collision_step"]


def build_convex_collision(
    mesh_path: Path,
    out_obj: Path,
    *,
    metadata_out: Path | None = None,
    method: str = "trimesh_convex_hull_from_reconstructed_metric_mesh",
) -> dict[str, Any]:
    """Build a reconstructed convex collision mesh from a metric visual mesh.

    Args:
        mesh_path: The metric visual mesh (from the scale step).
        out_obj: Output convex-hull OBJ path.
        metadata_out: Metadata path (default ``<stem>_metadata.json`` sibling).
        method: Provenance string recorded in the metadata.

    Returns:
        The metadata payload
        (schema ``countersim.reconstructed_convex_collision_mesh.v1``).
    """
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=True)
    hull = mesh.convex_hull
    out_obj = Path(out_obj)
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    hull.export(out_obj)
    metadata = {
        "schema": "countersim.reconstructed_convex_collision_mesh.v1",
        "method": method,
        "source_metric_mesh": str(mesh_path),
        "collision_mesh": str(out_obj),
        "official_asset_geometry_used": False,
        "extent_xyz_m": [float(v) for v in hull.extents],
        "watertight": bool(hull.is_watertight),
        "vertices": int(len(hull.vertices)),
        "faces": int(len(hull.faces)),
    }
    metadata_path = metadata_out or out_obj.with_name(out_obj.stem + "_metadata.json")
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_collision_step(step_dir: Path, metric_obj: Path, *, name: str) -> dict[str, Any]:
    """Build ``<name>_collision.obj`` in the step directory.

    Args:
        step_dir: This step's output dir (``<bundle>/<obj>/collision``).
        metric_obj: The metric visual mesh from the scale step.
        name: Object name (output file stem).

    Returns:
        The collision metadata payload.
    """
    step_dir = Path(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_convex_collision(Path(metric_obj), step_dir / f"{name}_collision.obj")
    logger.info(
        "[collision] %s: extent=%s watertight=%s",
        name, metadata["extent_xyz_m"], metadata["watertight"],
    )
    return metadata
