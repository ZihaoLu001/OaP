"""Filesystem helpers and explicit external configuration resolution.

Calibration and machine configuration are user inputs, not packaged data.
Set OAP_CONFIG_ROOT to their directory, or OAP_INPUT_ROOT to a directory
containing a configs/ subdirectory. Execution outputs should also be directed
outside the source checkout.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = [
    "read_json",
    "write_json_atomic",
    "sha256_bytes",
    "sha256_file",
    "timestamped_run_dir",
    "package_config_path",
    "load_package_config_json",
    "load_yaml",
]


def read_json(path: Path | str) -> Any:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as JSON atomically (tmp file + rename).

    The temporary file lives in the same directory as ``path`` so the final
    ``os.replace`` is a same-filesystem atomic rename. Parent directories are
    created as needed.

    Args:
        path: Destination JSON file.
        payload: Any ``json.dumps``-serializable object.
        indent: Pretty-print indent (2 matches the episode-log convention).

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=indent, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def sha256_bytes(data: bytes) -> str:
    """Return the hex sha256 of a bytes payload."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, *, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, streamed in ``chunk_bytes`` blocks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def timestamped_run_dir(base: Path | str, tag: str) -> Path:
    """Create and return ``<base>/<tag>_<YYYY-MM-DD_HHMMSS>`` (collision-safe).

    If the directory for the current second already exists, a numeric suffix
    is appended so two runs launched in the same second never share a dir.
    """
    base = Path(base)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    run_dir = base / f"{tag}_{stamp}"
    n = 1
    while run_dir.exists():
        run_dir = base / f"{tag}_{stamp}_{n}"
        n += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def package_config_path(relative: str) -> Path:
    """Resolve a relative configuration name from explicit external inputs.

    The retained function name is used by configuration consumers; no config
    files are shipped in the package. OAP_CONFIG_ROOT takes precedence over
    OAP_INPUT_ROOT/configs.
    """
    config_root = os.environ.get("OAP_CONFIG_ROOT")
    input_root = os.environ.get("OAP_INPUT_ROOT")
    if config_root:
        root = Path(config_root).expanduser().resolve()
    elif input_root:
        root = (Path(input_root).expanduser() / "configs").resolve()
    else:
        raise FileNotFoundError(
            f"External configuration {relative!r} is required. Set "
            "OAP_CONFIG_ROOT, or OAP_INPUT_ROOT with a configs/ directory."
        )
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Configuration names must stay inside the configured root")
    if not path.is_file():
        raise FileNotFoundError(
            f"External configuration {relative!r} was not found at {path}"
        )
    return path


def load_package_config_json(relative: str) -> Any:
    """Load an external JSON config (see :func:`package_config_path`)."""
    return read_json(package_config_path(relative))


def load_yaml(path: Path | str) -> Any:
    """Load a YAML (or JSON) mapping file.

    YAML import is local so that modules which never touch YAML configs do
    not pay for it at import time.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)
