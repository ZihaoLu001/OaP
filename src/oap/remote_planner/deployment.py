"""Content-addressed release loading for the persistent H100 service."""
from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from oap.loop.planner_profile import PlannerProfile
from oap.twin import load_world
from oap.utils.io import sha256_file, write_json_atomic

from .protocol import IdentityMismatch, PlanningProtocolError
from .service import PlanningContext

RELEASE_SCHEMA = "oap_remote_planner_release_v2"
_ASSET_TAGS = ("mesh", "texture", "hfield", "skin")
_SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml"}


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise PlanningProtocolError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return text


def _xml_path(plan_xml: Path | str) -> Path:
    path = Path(plan_xml).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_dir(
    literal: str | None,
    *,
    xml_dir: Path,
    fallback: Path,
) -> Path:
    if not literal:
        return fallback
    path = Path(literal).expanduser()
    return path.resolve() if path.is_absolute() else (xml_dir / path).resolve()


def referenced_asset_records(plan_xml: Path | str) -> list[dict[str, Any]]:
    """Hash every external MuJoCo asset using path-independent logical keys."""
    xml_path = _xml_path(plan_xml)
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    xml_dir = xml_path.parent
    asset_dir = _resolve_dir(
        None if compiler is None else compiler.get("assetdir"),
        xml_dir=xml_dir,
        fallback=xml_dir,
    )
    bases = {
        "mesh": _resolve_dir(
            None if compiler is None else compiler.get("meshdir"),
            xml_dir=xml_dir,
            fallback=asset_dir,
        ),
        "texture": _resolve_dir(
            None if compiler is None else compiler.get("texturedir"),
            xml_dir=xml_dir,
            fallback=asset_dir,
        ),
        "hfield": asset_dir,
        "skin": asset_dir,
    }
    records: list[dict[str, Any]] = []
    for kind in _ASSET_TAGS:
        for index, element in enumerate(root.findall(f"./asset/{kind}")):
            literal = element.get("file")
            if not literal:
                continue
            raw = Path(literal).expanduser()
            resolved = (
                raw.resolve()
                if raw.is_absolute()
                else (bases[kind] / raw).resolve()
            )
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            records.append(
                {
                    "key": f"{kind}:{element.get('name') or index}:{index}",
                    "sha256": sha256_file(resolved),
                    "size_bytes": int(resolved.stat().st_size),
                    "resolved_path": str(resolved),
                }
            )
    records.sort(key=lambda row: str(row["key"]))
    return records


def assets_sha256(plan_xml: Path | str) -> str:
    """Return the path-independent identity of all referenced asset bytes."""
    digest = hashlib.sha256()
    for record in referenced_asset_records(plan_xml):
        digest.update(str(record["key"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonicalize_element(element: ET.Element) -> None:
    element.text = None
    element.tail = None
    element.attrib = dict(sorted(element.attrib.items()))
    for child in element:
        _canonicalize_element(child)


def model_sha256(plan_xml: Path | str) -> str:
    """Hash model structure independently of host paths and live free state.

    External asset bytes have their own identity.  Asset path literals are
    replaced by stable slots.  Initial poses of free-joint bodies are state,
    not model structure: every request supplies the complete measured qpos.
    """
    root = ET.parse(_xml_path(plan_xml)).getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for name in ("assetdir", "meshdir", "texturedir"):
            compiler.attrib.pop(name, None)
    for kind in _ASSET_TAGS:
        for index, element in enumerate(root.findall(f"./asset/{kind}")):
            if element.get("file"):
                element.set("file", f"asset://{kind}/{index}")
    for body in root.iter("body"):
        has_freejoint = body.find("freejoint") is not None
        has_free_joint = any(
            joint.get("type") == "free" for joint in body.findall("joint")
        )
        if has_freejoint or has_free_joint:
            body.set("pos", "0 0 0")
            body.set("quat", "1 0 0 0")
    _canonicalize_element(root)
    encoded = ET.tostring(
        root,
        encoding="utf-8",
        short_empty_elements=True,
    )
    return hashlib.sha256(encoded).hexdigest()


def planner_source_sha256(package_root: Path | str | None = None) -> str:
    """Hash installed OaP source independent of its installation path."""
    root = (
        Path(package_root).expanduser().resolve()
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (
            path.suffix in _SOURCE_SUFFIXES
            or path.name == "py.typed"
        )
    ]
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class RemotePlannerRelease:
    """Verified release descriptor for one preloaded planning context."""

    path: Path
    plan_xml: Path
    model_sha256: str
    assets_sha256: str
    planner_sha256: str
    planner_profile: PlannerProfile
    planner_profile_sha256: str

    def to_dict(self) -> dict[str, Any]:
        try:
            xml_literal = os.path.relpath(self.plan_xml, self.path.parent)
        except ValueError:
            xml_literal = str(self.plan_xml)
        return {
            "schema": RELEASE_SCHEMA,
            "plan_xml": xml_literal,
            "model_sha256": self.model_sha256,
            "assets_sha256": self.assets_sha256,
            "planner_sha256": self.planner_sha256,
            "planner_profile": self.planner_profile.to_dict(),
            "planner_profile_sha256": self.planner_profile_sha256,
        }


def create_release_bundle(
    *,
    plan_xml: Path | str,
    output: Path | str,
    planner_profile: PlannerProfile,
    package_root: Path | str | None = None,
) -> RemotePlannerRelease:
    """Create a descriptor after assets have been staged/relocated."""
    xml_path = _xml_path(plan_xml)
    output_path = Path(output).expanduser().resolve()
    if not isinstance(planner_profile, PlannerProfile):
        raise TypeError("planner_profile must be a PlannerProfile")
    release = RemotePlannerRelease(
        path=output_path,
        plan_xml=xml_path,
        model_sha256=model_sha256(xml_path),
        assets_sha256=assets_sha256(xml_path),
        planner_sha256=planner_source_sha256(package_root),
        planner_profile=planner_profile,
        planner_profile_sha256=planner_profile.sha256,
    )
    write_json_atomic(output_path, release.to_dict())
    return release


def read_release_bundle(path: Path | str) -> RemotePlannerRelease:
    """Parse and verify all declared identities without loading MuJoCo."""
    release_path = Path(path).expanduser().resolve()
    data = json.loads(release_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "plan_xml",
        "model_sha256",
        "assets_sha256",
        "planner_sha256",
        "planner_profile",
        "planner_profile_sha256",
    }
    if not isinstance(data, dict) or set(data) != required:
        raise PlanningProtocolError(
            "remote planner release has missing or unknown fields"
        )
    if data["schema"] != RELEASE_SCHEMA:
        raise PlanningProtocolError(
            f"unsupported remote planner release schema {data['schema']!r}"
        )
    raw_xml = Path(str(data["plan_xml"])).expanduser()
    xml_path = (
        raw_xml.resolve()
        if raw_xml.is_absolute()
        else (release_path.parent / raw_xml).resolve()
    )
    if not isinstance(data["planner_profile"], Mapping):
        raise PlanningProtocolError(
            "release planner_profile must be a JSON object"
        )
    try:
        planner_profile = PlannerProfile.from_dict(
            data["planner_profile"]
        )
    except (TypeError, ValueError) as exc:
        raise PlanningProtocolError(str(exc)) from exc
    release = RemotePlannerRelease(
        path=release_path,
        plan_xml=_xml_path(xml_path),
        model_sha256=_require_sha256(
            data["model_sha256"], "model_sha256"
        ),
        assets_sha256=_require_sha256(
            data["assets_sha256"], "assets_sha256"
        ),
        planner_sha256=_require_sha256(
            data["planner_sha256"], "planner_sha256"
        ),
        planner_profile=planner_profile,
        planner_profile_sha256=_require_sha256(
            data["planner_profile_sha256"],
            "planner_profile_sha256",
        ),
    )
    if release.planner_profile.sha256 != release.planner_profile_sha256:
        raise IdentityMismatch(
            "release planner_profile_sha256 does not match planner_profile"
        )
    actual_model = model_sha256(release.plan_xml)
    if actual_model != release.model_sha256:
        raise IdentityMismatch(
            "release model_sha256 does not match plan XML structure"
        )
    actual_assets = assets_sha256(release.plan_xml)
    if actual_assets != release.assets_sha256:
        raise IdentityMismatch(
            "release assets_sha256 does not match referenced asset bytes"
        )
    actual_planner = planner_source_sha256()
    if actual_planner != release.planner_sha256:
        raise IdentityMismatch(
            "release planner_sha256 does not match installed OaP source"
        )
    return release


def load_release_context(path: Path | str) -> tuple[
    RemotePlannerRelease,
    PlanningContext,
]:
    """Verify a release, compile its world, and return a service context."""
    release = read_release_bundle(path)
    world = load_world(release.plan_xml, "W_plan_remote_h100")
    context = PlanningContext(
        model_sha256=release.model_sha256,
        assets_sha256=release.assets_sha256,
        world=world,
        plan_xml=release.plan_xml,
        planner_profile=release.planner_profile,
    )
    return release, context
