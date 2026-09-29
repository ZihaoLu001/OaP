"""Fail-closed bootstrap for the paper-standard MJWarp planning runtime.

Warp execution controls are compile-time module options.  They must be fixed
before JAX or the MJWarp bridge is imported; setting them after a kernel has
been created produces plausible but invalid evidence.  This module is the one
production authority for that ordering and for the exact runtime version set.

The paper profile intentionally uses Warp's normal-atomic default rather than
bit-exact deterministic lowering.  Independent seeds quantify run variation;
the expensive deterministic debug path is reserved for debugging, not the
paper task runs.

Importing this module is deliberately inert.  Production entry points call
:func:`bootstrap_production_mjwarp` before any JAX/MJWarp import, and the model
conversion path calls it again as a non-bypassable backstop.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
from importlib import metadata
import json
import os
import platform
import sys
import threading
from enum import Enum
from typing import Any, Callable, Mapping

RUNTIME_PROFILE_NAME = "paper_standard_warp"
RUNTIME_SCHEMA = "oap_paper_standard_warp_runtime_v1"
RUNTIME_FINGERPRINT_SCHEMA = (
    "oap_paper_standard_warp_fingerprint_v1"
)
EXPECTED_DISTRIBUTIONS = {
    "mujoco": "3.11.0",
    "mujoco-mjx": "3.11.0",
    "mujoco-warp": "3.11.0",
    "warp-lang": "1.15.0",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "jax-cuda12-plugin": "0.10.2",
    "jax-cuda12-pjrt": "0.10.2",
}
DETERMINISTIC_MODE = "NOT_GUARANTEED"
DETERMINISTIC_DEBUG = False
DETERMINISTIC_MAX_RECORDS = 0
XLA_PREALLOCATE = "false"

_FORBIDDEN_BEFORE_BOOTSTRAP = (
    "jax",
    "jaxlib",
    "mujoco.mjx",
    "mujoco_warp",
    "warp",
)
_LOCK = threading.Lock()
_RUNTIME_IDENTITY: dict[str, Any] | None = None

__all__ = [
    "DETERMINISTIC_DEBUG",
    "DETERMINISTIC_MAX_RECORDS",
    "DETERMINISTIC_MODE",
    "EXPECTED_DISTRIBUTIONS",
    "ProductionRuntimeError",
    "RUNTIME_SCHEMA",
    "RUNTIME_PROFILE_NAME",
    "XLA_PREALLOCATE",
    "bootstrap_production_mjwarp",
    "production_runtime_identity",
    "verify_modelwarp",
]


class ProductionRuntimeError(RuntimeError):
    """The process cannot prove the production MJWarp runtime contract."""


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_imported_prefix(modules: Mapping[str, Any], prefix: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for name in modules)


def _require_clean_import_order(modules: Mapping[str, Any]) -> None:
    imported = [
        prefix
        for prefix in _FORBIDDEN_BEFORE_BOOTSTRAP
        if _has_imported_prefix(modules, prefix)
    ]
    if imported:
        raise ProductionRuntimeError(
            "production MJWarp bootstrap was called after GPU runtime import: "
            + ", ".join(imported)
        )


def _installed_versions(
    version_reader: Callable[[str], str],
) -> dict[str, str]:
    versions: dict[str, str] = {}
    failures: list[str] = []
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = str(version_reader(distribution))
        except Exception as exc:
            failures.append(
                f"{distribution}: unavailable ({type(exc).__name__}: {exc})"
            )
            continue
        versions[distribution] = actual
        if actual != expected:
            failures.append(
                f"{distribution}: expected {expected}, found {actual}"
            )
    if failures:
        raise ProductionRuntimeError(
            "production MJWarp version contract failed: " + "; ".join(failures)
        )
    return versions


def _integer_components(value: Any, label: str) -> list[int]:
    raw = list(value) if isinstance(value, (tuple, list)) else [value]
    if not raw:
        raise ProductionRuntimeError(f"{label} returned an empty sequence")
    result: list[int] = []
    for index, component in enumerate(raw):
        if isinstance(component, bool):
            raise ProductionRuntimeError(
                f"{label}[{index}] is boolean, not an integer"
            )
        try:
            normalized = component.__index__()
        except (AttributeError, TypeError) as exc:
            raise ProductionRuntimeError(
                f"{label}[{index}] is not an integer: {component!r}"
            ) from exc
        result.append(int(normalized))
    return result


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _jax_gpu_identity(jax: Any, *, require_gpu: bool) -> list[dict[str, Any]]:
    try:
        devices = list(jax.devices("gpu"))
    except Exception as exc:
        if require_gpu:
            raise ProductionRuntimeError(
                f"GPU_ONLY_REFUSAL: JAX GPU discovery failed: {type(exc).__name__}: {exc}"
            ) from exc
        devices = []
    if require_gpu and len(devices) != 1:
        raise ProductionRuntimeError(
            "GPU_ONLY_REFUSAL: expected exactly one visible JAX GPU, "
            f"found {len(devices)}"
        )
    if require_gpu and str(jax.default_backend()).lower() != "gpu":
        raise ProductionRuntimeError(
            f"GPU_ONLY_REFUSAL: JAX backend is {jax.default_backend()!r}"
        )
    return [
        {
            "string": str(device),
            "platform": _json_scalar(getattr(device, "platform", None)),
            "device_kind": _json_scalar(getattr(device, "device_kind", None)),
            "local_hardware_id": _json_scalar(
                getattr(device, "local_hardware_id", None)
            ),
        }
        for device in devices
    ]


def _warp_gpu_identity(wp: Any, *, require_gpu: bool) -> list[dict[str, Any]]:
    try:
        devices = list(wp.get_cuda_devices())
    except Exception as exc:
        if require_gpu:
            raise ProductionRuntimeError(
                f"GPU_ONLY_REFUSAL: Warp GPU discovery failed: {type(exc).__name__}: {exc}"
            ) from exc
        devices = []
    if require_gpu and len(devices) != 1:
        raise ProductionRuntimeError(
            "GPU_ONLY_REFUSAL: expected exactly one visible Warp GPU, "
            f"found {len(devices)}"
        )
    return [
        {
            "string": str(device),
            "alias": _json_scalar(getattr(device, "alias", None)),
            "name": _json_scalar(getattr(device, "name", None)),
            "uuid": _json_scalar(getattr(device, "uuid", None)),
            "arch": _json_scalar(getattr(device, "arch", None)),
        }
        for device in devices
    ]


def _expected_warp_mode(wp: Any) -> Any:
    config = getattr(wp, "config", None)
    mode_type = getattr(config, "DeterministicMode", None)
    if mode_type is None or not hasattr(mode_type, DETERMINISTIC_MODE):
        raise ProductionRuntimeError(
            "required Warp deterministic mode is unavailable: "
            f"{DETERMINISTIC_MODE}"
        )
    return getattr(mode_type, DETERMINISTIC_MODE)


def _configure_warp_profile(wp: Any) -> None:
    config = wp.config
    expected_mode = _expected_warp_mode(wp)
    config.deterministic = expected_mode
    config.deterministic_debug = DETERMINISTIC_DEBUG
    config.deterministic_max_records = DETERMINISTIC_MAX_RECORDS


def _require_warp_profile(wp: Any) -> None:
    config = wp.config
    expected_mode = _expected_warp_mode(wp)
    if config.deterministic is not expected_mode:
        raise ProductionRuntimeError("Warp deterministic mode drifted after bootstrap")
    if config.deterministic_debug is not DETERMINISTIC_DEBUG:
        raise ProductionRuntimeError("Warp deterministic_debug profile drifted")
    record_cap = config.deterministic_max_records
    if (
        isinstance(record_cap, bool)
        or not isinstance(record_cap, int)
        or record_cap != DETERMINISTIC_MAX_RECORDS
    ):
        raise ProductionRuntimeError("Warp deterministic record cap profile drifted")


def _graph_mode_identity(mjxw: Any, mjxw_types: Any) -> dict[str, Any]:
    if getattr(mjxw, "WARP_INSTALLED", False) is not True:
        raise ProductionRuntimeError("MJWarp reports WARP_INSTALLED != True")
    graph_mode_type = getattr(mjxw_types, "GraphMode", None)
    if (
        not isinstance(graph_mode_type, type)
        or not issubclass(graph_mode_type, Enum)
        or not hasattr(graph_mode_type, "WARP")
    ):
        raise ProductionRuntimeError(
            "public mujoco.mjx.warp.types.GraphMode.WARP is unavailable"
        )
    model_type = getattr(mjxw_types, "ModelWarp", None)
    if not isinstance(model_type, type):
        raise ProductionRuntimeError(
            "public mujoco.mjx.warp.types.ModelWarp is unavailable"
        )
    warp_member = graph_mode_type.WARP
    return {
        "graph_mode_type": (
            f"{graph_mode_type.__module__}.{graph_mode_type.__qualname__}"
        ),
        "warp_name": warp_member.name,
        "warp_value": int(warp_member.value),
        "modelwarp_type": f"{model_type.__module__}.{model_type.__qualname__}",
    }


def _revalidate_cached(
    identity: Mapping[str, Any],
    *,
    require_gpu: bool = True,
    version_reader: Callable[[str], str],
) -> None:
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != XLA_PREALLOCATE:
        raise ProductionRuntimeError("XLA preallocation setting drifted after bootstrap")
    live_versions = _installed_versions(version_reader)
    if live_versions != identity.get("versions"):
        raise ProductionRuntimeError(
            "installed runtime versions drifted after bootstrap"
        )
    wp = importlib.import_module("warp")
    _require_warp_profile(wp)
    jax = importlib.import_module("jax")
    mjxw = importlib.import_module("mujoco.mjx.warp")
    mjxw_types = importlib.import_module("mujoco.mjx.warp.types")
    live_bridge = _graph_mode_identity(mjxw, mjxw_types)
    if live_bridge != identity.get("bridge"):
        raise ProductionRuntimeError("MJWarp bridge identity drifted after bootstrap")
    live_gpu = {
        "jax_devices": _jax_gpu_identity(jax, require_gpu=require_gpu),
        "warp_devices": _warp_gpu_identity(wp, require_gpu=require_gpu),
        "warp_cuda_driver_version": _integer_components(
            wp.get_cuda_driver_version(), "Warp CUDA driver version"
        ),
        "warp_cuda_toolkit_version": _integer_components(
            wp.get_cuda_toolkit_version(), "Warp CUDA toolkit version"
        ),
    }
    if live_gpu != identity.get("gpu"):
        raise ProductionRuntimeError(
            "GPU runtime identity drifted after bootstrap"
        )
    live_profile = {
        "name": RUNTIME_PROFILE_NAME,
        "deterministic_mode": DETERMINISTIC_MODE,
        "deterministic_debug": DETERMINISTIC_DEBUG,
        "deterministic_max_records": DETERMINISTIC_MAX_RECORDS,
        "xla_python_client_preallocate": XLA_PREALLOCATE,
    }
    live_fingerprint = {
        "schema": RUNTIME_FINGERPRINT_SCHEMA,
        "versions": live_versions,
        "profile": live_profile,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "bridge": live_bridge,
        "gpu": live_gpu,
    }
    fingerprint = identity.get("fingerprint")
    if (
        not isinstance(fingerprint, Mapping)
        or identity.get("schema") != RUNTIME_SCHEMA
        or identity.get("versions") != live_versions
        or identity.get("profile") != live_profile
        or identity.get("bridge") != live_bridge
        or identity.get("gpu") != live_gpu
        or fingerprint != live_fingerprint
        or identity.get("fingerprint_sha256")
        != _canonical_sha256(live_fingerprint)
    ):
        raise ProductionRuntimeError("cached runtime fingerprint is corrupt")


def _bootstrap_production_mjwarp(
    *,
    require_gpu: bool,
    importer: Callable[[str], Any],
    version_reader: Callable[[str], str],
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    """Internal dependency-injected implementation for CPU contract tests."""
    global _RUNTIME_IDENTITY
    with _LOCK:
        if _RUNTIME_IDENTITY is not None:
            _revalidate_cached(
                _RUNTIME_IDENTITY,
                require_gpu=require_gpu,
                version_reader=version_reader,
            )
            if require_gpu and not _RUNTIME_IDENTITY["gpu"]["jax_devices"]:
                raise ProductionRuntimeError(
                    "GPU_ONLY_REFUSAL: cached bootstrap has no GPU identity"
                )
            return copy.deepcopy(_RUNTIME_IDENTITY)

        _require_clean_import_order(modules)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = XLA_PREALLOCATE
        if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != XLA_PREALLOCATE:
            raise ProductionRuntimeError(
                "could not set XLA_PYTHON_CLIENT_PREALLOCATE=false"
            )
        versions = _installed_versions(version_reader)

        # Warp must be first: these module-compilation options affect modules
        # created later.  Only after the profile is frozen may JAX/MJWarp load.
        wp = importer("warp")
        _configure_warp_profile(wp)
        _require_warp_profile(wp)
        init = getattr(wp, "init", None)
        if callable(init):
            init()
        jax = importer("jax")
        mjxw = importer("mujoco.mjx.warp")
        mjxw_types = importer("mujoco.mjx.warp.types")
        bridge = _graph_mode_identity(mjxw, mjxw_types)
        jax_devices = _jax_gpu_identity(jax, require_gpu=require_gpu)
        warp_devices = _warp_gpu_identity(wp, require_gpu=require_gpu)

        profile = {
            "name": RUNTIME_PROFILE_NAME,
            "deterministic_mode": DETERMINISTIC_MODE,
            "deterministic_debug": DETERMINISTIC_DEBUG,
            "deterministic_max_records": DETERMINISTIC_MAX_RECORDS,
            "xla_python_client_preallocate": XLA_PREALLOCATE,
        }
        gpu = {
            "jax_devices": jax_devices,
            "warp_devices": warp_devices,
            "warp_cuda_driver_version": _integer_components(
                wp.get_cuda_driver_version(), "Warp CUDA driver version"
            ),
            "warp_cuda_toolkit_version": _integer_components(
                wp.get_cuda_toolkit_version(), "Warp CUDA toolkit version"
            ),
        }
        fingerprint = {
            "schema": RUNTIME_FINGERPRINT_SCHEMA,
            "versions": versions,
            "profile": profile,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "bridge": bridge,
            "gpu": gpu,
        }
        identity = {
            "schema": RUNTIME_SCHEMA,
            "versions": versions,
            "profile": profile,
            "bridge": bridge,
            "gpu": gpu,
            "fingerprint": fingerprint,
            "fingerprint_sha256": _canonical_sha256(fingerprint),
        }
        _RUNTIME_IDENTITY = identity
        return copy.deepcopy(identity)


def bootstrap_production_mjwarp() -> dict[str, Any]:
    """Configure and verify the single paper-standard WARP profile.

    There is intentionally no parameter, environment, or CLI override for the
    public graph mode, normal-atomic mode, disabled debug flag, default record
    cap, dependency versions, import registry, or GPU requirement.  A process
    that imported Warp, JAX, or MJWarp first is irrecoverably ambiguous and is
    refused.
    """
    return _bootstrap_production_mjwarp(
        require_gpu=True,
        importer=importlib.import_module,
        version_reader=metadata.version,
        modules=sys.modules,
    )


def production_runtime_identity() -> dict[str, Any]:
    """Return bootstrapped runtime evidence or fail closed."""
    with _LOCK:
        if _RUNTIME_IDENTITY is None:
            raise ProductionRuntimeError(
                "production MJWarp runtime has not been bootstrapped"
            )
        _revalidate_cached(
            _RUNTIME_IDENTITY,
            require_gpu=True,
            version_reader=metadata.version,
        )
        return copy.deepcopy(_RUNTIME_IDENTITY)


def verify_modelwarp(converted: Any) -> str:
    """Prove that public conversion produced the fixed WARP graph mode."""
    production_runtime_identity()
    mjxw_types = importlib.import_module("mujoco.mjx.warp.types")
    model_type = getattr(mjxw_types, "ModelWarp", None)
    model_impl = getattr(converted, "_impl", None)
    if not isinstance(model_type, type) or not isinstance(model_impl, model_type):
        raise ProductionRuntimeError(
            "mjx.put_model result does not wrap public "
            "mujoco.mjx.warp.types.ModelWarp"
        )
    actual_member = getattr(
        getattr(getattr(converted, "opt", None), "_impl", None),
        "graph_mode",
        None,
    )
    graph_mode_type = getattr(mjxw_types, "GraphMode", None)
    if not isinstance(graph_mode_type, type) or not issubclass(
        graph_mode_type, Enum
    ):
        raise ProductionRuntimeError("public MJWarp GraphMode type is invalid")
    if not isinstance(actual_member, graph_mode_type):
        raise ProductionRuntimeError(
            "converted ModelWarp does not expose a public GraphMode"
        )
    actual = actual_member.name
    if actual != "WARP":
        raise ProductionRuntimeError(
            f"MJX graph mode degraded from 'WARP' to {actual!r}"
        )
    return actual
