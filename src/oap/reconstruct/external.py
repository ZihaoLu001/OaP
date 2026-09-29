"""Resolve and drive the external heavy-model environments for Stage 1.

Role in the two-stage pipeline: the reconstruction steps never import pyzed,
SAM3, SAM3D, FoundationPose/Any6D, torch, or transformers in-process. Each
heavy tool lives in its own conda env. Users provide interpreter paths in an
external YAML named by ``OAP_EXTERNAL_ENVS`` or ``--external-envs``, or in
``external_envs.yaml`` under their configured external config root.
This module applies those settings and per-flag CLI
overrides, then runs the payload scripts from
:mod:`oap.reconstruct.payloads` as subprocesses under the resolved
interpreters.

A missing or misconfigured env always raises :class:`ExternalEnvError` naming
the exact YAML key (``hosts.<profile>.<tool>``) to fix -- never a bare
``ImportError`` deep inside a model stack.
"""
from __future__ import annotations

import importlib.resources
import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oap.utils.io import load_yaml, package_config_path

logger = logging.getLogger("oap.reconstruct.external")

__all__ = ["ExternalEnvError", "ToolEnv", "ExternalEnvs", "payload_path", "KNOWN_TOOLS"]

#: Tools an external-envs host profile may define. Each maps to one conda env.
KNOWN_TOOLS = ("observer", "sam3d", "foundationpose", "qwen")

_CONFIG_RELATIVE = "external_envs.yaml"
_RESERVED_KEYS = {"python", "pythonpath_extra", "ld_library_path_extra"}


class ExternalEnvError(RuntimeError):
    """An external tool env is missing or misconfigured (actionable message)."""


@dataclass(frozen=True)
class ToolEnv:
    """One external tool environment resolved from ``external_envs.yaml``.

    Attributes:
        name: Tool key (one of :data:`KNOWN_TOOLS`).
        python: Interpreter path, or ``None`` when the active profile does not
            provide this tool on the current host.
        pythonpath_extra: Entries prepended to ``PYTHONPATH`` for payload runs
            (e.g. a staged dependency or architecture-specific rebuild).
        ld_library_path_extra: Entries prepended to ``LD_LIBRARY_PATH``.
        extras: Tool-specific string settings (``root``, ``any6d_root``,
            ``model_id``, ``model_path``, ``torch_hub_dir``, ...).
    """

    name: str
    python: Path | None
    pythonpath_extra: tuple[str, ...] = ()
    ld_library_path_extra: tuple[str, ...] = ()
    extras: dict[str, str] = field(default_factory=dict)


def payload_path(module_name: str) -> Path:
    """Resolve a payload script packaged under ``oap/reconstruct/payloads``.

    Args:
        module_name: Bare module name, e.g. ``"sam3d_reconstruct"``.

    Returns:
        A concrete filesystem path to ``<module_name>.py`` suitable for
        passing to an external interpreter.

    Raises:
        ExternalEnvError: If the payload is not shipped with the package.
    """
    root = importlib.resources.files("oap.reconstruct.payloads")
    candidate = root / f"{module_name}.py"
    try:
        with importlib.resources.as_file(candidate) as p:
            path = Path(p)
    except FileNotFoundError as exc:
        raise ExternalEnvError(
            f"payload script {module_name!r} not found in "
            f"oap.reconstruct.payloads -- was the wheel built with "
            f"[tool.setuptools.package-data] reconstruct/payloads/*.py?"
        ) from exc
    if not path.exists():
        raise ExternalEnvError(
            f"payload script {module_name!r} resolved to {path} which does not exist"
        )
    return path


def _deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base`` (mappings merge, else replace)."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _as_str_list(raw: Any, *, where: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, Sequence):
        raise ExternalEnvError(f"{where} must be a list of paths, got {type(raw).__name__}")
    return tuple(str(v) for v in raw)


def _load_config_mapping(path: Path) -> dict[str, Any]:
    raw = load_yaml(path)
    if not isinstance(raw, Mapping) or "hosts" not in raw:
        raise ExternalEnvError(
            f"{path} is not an external-envs config: expected a mapping with a "
            f"'hosts' key (schema oap_external_envs_v1)"
        )
    return dict(raw)


class ExternalEnvs:
    """Per-host external interpreter registry + payload subprocess runner."""

    def __init__(self, *, profile: str, tools: dict[str, ToolEnv], source: Path) -> None:
        self.profile = profile
        self.source = source
        self._tools = tools

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        spec: str | None = None,
        *,
        overrides: Mapping[str, str | Path | None] | None = None,
    ) -> "ExternalEnvs":
        """Load a host profile with the documented override chain.

        A YAML path supplied through ``spec`` takes precedence over a YAML
        at ``OAP_EXTERNAL_ENVS``; per-flag ``overrides`` are applied last.
        If neither file is supplied, load ``external_envs.yaml`` from the
        external config root. A profile name supplied through ``spec``
        selects a host from that config instead of its ``default_profile``.

        Args:
            spec: A profile name defined under ``hosts`` (for example,
                ``local``), a filesystem path to a YAML with the same
                schema, or ``None``.
            overrides: Dotted-key overrides applied last, e.g.
                ``{"sam3d.python": "/env/bin/python",
                "foundationpose.any6d_root": "/repos/Any6D"}``. ``None``
                values are ignored.

        Returns:
            The resolved registry.

        Raises:
            ExternalEnvError: On an unknown profile or malformed config.
        """
        user_yaml = os.environ.get("OAP_EXTERNAL_ENVS")
        spec_is_path = spec is not None and (
            "/" in spec or "\\" in spec or spec.endswith((".yaml", ".yml"))
        )
        config: dict[str, Any] = {}
        if user_yaml:
            user_path = Path(user_yaml).expanduser()
            if not user_path.exists():
                raise ExternalEnvError(
                    f"OAP_EXTERNAL_ENVS points at a missing YAML: {user_path}"
                )
            config = _load_config_mapping(user_path)
            source = user_path
        elif spec_is_path:
            # The explicit file is loaded below without requiring a default
            # config or any source-checkout data files.
            source = Path(spec).expanduser()
        else:
            source = package_config_path(_CONFIG_RELATIVE)
            config = _load_config_mapping(source)

        profile_name: str | None = None
        if spec_is_path:
            spec_path = Path(spec).expanduser()
            if not spec_path.exists():
                raise ExternalEnvError(f"--external-envs YAML not found: {spec_path}")
            spec_config = _load_config_mapping(spec_path)
            config = _deep_merge(config, spec_config)
            source = spec_path
            profile_name = str(spec_config.get("default_profile") or "") or None
            if profile_name is None and len(spec_config.get("hosts") or {}) == 1:
                profile_name = next(iter(spec_config["hosts"]))
        elif spec is not None:
            profile_name = spec

        hosts = config.get("hosts")
        if not isinstance(hosts, Mapping) or not hosts:
            raise ExternalEnvError(f"{source} defines no hosts")
        if profile_name is None:
            profile_name = str(config.get("default_profile") or "")
        if profile_name not in hosts:
            raise ExternalEnvError(
                f"external-envs profile {profile_name!r} not found under hosts in "
                f"{source}; available: {sorted(hosts)}. Pass --external-envs "
                f"{{{','.join(sorted(hosts))}}} or a YAML path."
            )

        tools: dict[str, ToolEnv] = {}
        profile_raw = hosts[profile_name] or {}
        if not isinstance(profile_raw, Mapping):
            raise ExternalEnvError(
                f"hosts.{profile_name} in {source} must be a mapping of tools"
            )
        for tool_name, tool_raw in profile_raw.items():
            where = f"hosts.{profile_name}.{tool_name}"
            if not isinstance(tool_raw, Mapping):
                raise ExternalEnvError(f"{where} in {source} must be a mapping")
            extras = {
                str(k): str(v)
                for k, v in tool_raw.items()
                if k not in _RESERVED_KEYS and v is not None
            }
            python_raw = tool_raw.get("python")
            tools[str(tool_name)] = ToolEnv(
                name=str(tool_name),
                python=Path(str(python_raw)).expanduser() if python_raw else None,
                pythonpath_extra=_as_str_list(
                    tool_raw.get("pythonpath_extra"), where=f"{where}.pythonpath_extra"
                ),
                ld_library_path_extra=_as_str_list(
                    tool_raw.get("ld_library_path_extra"),
                    where=f"{where}.ld_library_path_extra",
                ),
                extras=extras,
            )

        envs = cls(profile=str(profile_name), tools=tools, source=source)
        for key, value in (overrides or {}).items():
            if value:
                envs._apply_override(key, str(value))
        return envs

    def _apply_override(self, dotted_key: str, value: str) -> None:
        """Apply one ``tool.field`` override (from a CLI flag)."""
        if "." not in dotted_key:
            raise ExternalEnvError(f"override key must be 'tool.field', got {dotted_key!r}")
        tool_name, field_name = dotted_key.split(".", 1)
        current = self._tools.get(tool_name, ToolEnv(name=tool_name, python=None))
        if field_name == "python":
            updated = ToolEnv(
                name=current.name,
                python=Path(value).expanduser(),
                pythonpath_extra=current.pythonpath_extra,
                ld_library_path_extra=current.ld_library_path_extra,
                extras=current.extras,
            )
        else:
            extras = dict(current.extras)
            extras[field_name] = value
            updated = ToolEnv(
                name=current.name,
                python=current.python,
                pythonpath_extra=current.pythonpath_extra,
                ld_library_path_extra=current.ld_library_path_extra,
                extras=extras,
            )
        self._tools[tool_name] = updated

    # --------------------------------------------------------------- queries
    def has(self, name: str) -> bool:
        """Return whether the active profile defines an interpreter for ``name``."""
        return self._tools.get(name, ToolEnv(name=name, python=None)).python is not None

    def tool(self, name: str) -> ToolEnv:
        """Return the :class:`ToolEnv` for ``name`` (empty stub if undefined)."""
        return self._tools.get(name, ToolEnv(name=name, python=None))

    def python(self, name: str) -> Path:
        """Return the interpreter for ``name``, or raise an actionable error."""
        tool = self.tool(name)
        if tool.python is None:
            raise ExternalEnvError(
                f"external tool '{name}' has no interpreter for profile "
                f"'{self.profile}'. Set hosts.{self.profile}.{name}.python in "
                f"{self.source}, point "
                f"OAP_EXTERNAL_ENVS at a YAML that defines it, or pass "
                f"--{name}-python."
            )
        if not tool.python.exists():
            raise ExternalEnvError(
                f"external tool '{name}' interpreter does not exist: {tool.python} "
                f"(hosts.{self.profile}.{name}.python in {self.source}). Is this "
                f"the right --external-envs profile for this host?"
            )
        return tool.python

    def extra(self, name: str, key: str, *, required: bool = True) -> str | None:
        """Return a tool-specific setting (e.g. ``sam3d`` -> ``root``)."""
        value = self.tool(name).extras.get(key)
        if value is None and required:
            raise ExternalEnvError(
                f"external tool '{name}' is missing setting '{key}' for profile "
                f"'{self.profile}'. Set hosts.{self.profile}.{name}.{key} in "
                f"{self.source}."
            )
        return value

    # ------------------------------------------------------------------ run
    def run_payload(
        self,
        name: str,
        payload_module: str,
        args: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        extra_env: Mapping[str, str] | None = None,
        extra_pythonpath: Sequence[str | Path] = (),
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """Run a packaged payload script under the tool's external interpreter.

        The subprocess inherits stdout/stderr (long GPU steps stream live);
        the tool's ``pythonpath_extra``/``ld_library_path_extra`` entries plus
        ``extra_pythonpath`` are prepended to the child's search paths.

        Args:
            name: Tool key (see :data:`KNOWN_TOOLS`).
            payload_module: Payload module name under
                :mod:`oap.reconstruct.payloads`.
            args: Command-line arguments for the payload.
            cwd: Optional working directory.
            extra_env: Extra environment variables for this invocation.
            extra_pythonpath: Extra ``PYTHONPATH`` entries (e.g. the Any6D
                checkout for the pose step), prepended before the tool's own.
            timeout_s: Optional wall-clock timeout.

        Returns:
            The completed process (returncode 0).

        Raises:
            ExternalEnvError: If the env is missing or the payload fails.
        """
        python = self.python(name)
        tool = self.tool(name)
        script = payload_path(payload_module)
        cmd = [str(python), str(script)] + [str(a) for a in args]

        env = dict(os.environ)
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        pythonpath = [str(p) for p in extra_pythonpath] + list(tool.pythonpath_extra)
        if pythonpath:
            existing = env.get("PYTHONPATH", "")
            joined = os.pathsep.join(pythonpath)
            env["PYTHONPATH"] = f"{joined}{os.pathsep}{existing}" if existing else joined
        if tool.ld_library_path_extra:
            existing = env.get("LD_LIBRARY_PATH", "")
            joined = os.pathsep.join(tool.ld_library_path_extra)
            env["LD_LIBRARY_PATH"] = (
                f"{joined}{os.pathsep}{existing}" if existing else joined
            )

        logger.info("[external:%s] %s", name, " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalEnvError(
                f"payload {payload_module!r} in env '{name}' timed out after "
                f"{timeout_s}s: {' '.join(cmd)}"
            ) from exc
        if proc.returncode != 0:
            raise ExternalEnvError(
                f"payload {payload_module!r} in env '{name}' (profile "
                f"'{self.profile}') exited with code {proc.returncode}. Command: "
                f"{' '.join(cmd)}. If the interpreter is wrong for this host, fix "
                f"hosts.{self.profile}.{name} in {self.source}."
            )
        return proc
