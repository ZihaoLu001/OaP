"""Read-only measurement primitives for real-release readiness evidence.

This module measures facts; it never chooses or authorizes readiness limits.
It intentionally has no robot-control API.  A caller supplies a read-only
``get_state`` function and the production tracking/shadow-planning callbacks.

The three measured quantities are:

* observation age: current host wall time minus the RGB-D exposure timestamp;
* camera/robot skew: exposure timestamp minus the nearest *actually sampled*
  ``RobotState.stamp`` (with a real before/after telemetry bracket);
* shadow-plan staleness: response receive time minus the exact exposure bound
  into the planning request.

No commanded state, post-inference state, or historical planning state may be
substituted for those measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from oap.loop.executor_readiness import validate_robot_state_telemetry
from oap.utils.io import write_json_atomic

READINESS_MEASUREMENT_SCHEMA = "oap_readiness_measurement_v1"


class ReadinessMeasurementError(RuntimeError):
    """A measurement could not establish the requested physical fact."""


@dataclass(frozen=True)
class TelemetrySample:
    """One real read-only controller observation and its local receive times."""

    sequence: int
    robot_state_stamp_s: float
    received_unix_s: float
    received_monotonic_s: float
    q_rad: tuple[float, ...]
    tcp_pose: tuple[float, ...]
    gripper_width_m: float
    gripper_force_n: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExposureTelemetryPair:
    """A camera exposure bracketed by independently sampled RobotState rows."""

    capture_unix_s: float
    capture_monotonic_s: float
    before: TelemetrySample
    after: TelemetrySample
    nearest: TelemetrySample

    @property
    def skew_s(self) -> float:
        return abs(
            float(self.capture_unix_s)
            - float(self.nearest.robot_state_stamp_s)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_unix_s": self.capture_unix_s,
            "capture_monotonic_s": self.capture_monotonic_s,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "nearest": self.nearest.to_dict(),
            "camera_robot_timestamp_skew_s": self.skew_s,
            "paired_from": (
                "independent_continuous_readonly_robot_state_samples"
            ),
            "post_inference_state_substitution": False,
        }


def _finite_positive(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReadinessMeasurementError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ReadinessMeasurementError(
            f"{label} must be finite and positive"
        )
    return number


class ReadOnlyRobotStateBuffer:
    """Continuously sample ``get_state`` without acquiring a control lease.

    The caller must pass a connection that was opened with the controller's
    observation-only ``connect`` operation, never its context-manager/lease
    operation.  Only the supplied ``read_state`` callback is retained here;
    there is no command-capable object on this class.
    """

    def __init__(
        self,
        read_state: Callable[[], Any],
        *,
        expected_joint_count: int = 7,
        period_s: float = 0.01,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        max_samples: int = 20_000,
    ) -> None:
        if period_s <= 0.0 or not math.isfinite(period_s):
            raise ValueError("period_s must be finite and positive")
        if max_samples < 2:
            raise ValueError("max_samples must be at least two")
        self._read_state = read_state
        self._expected_joint_count = int(expected_joint_count)
        self._period_s = float(period_s)
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._max_samples = int(max_samples)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[TelemetrySample] = []
        self._failures: list[dict[str, Any]] = []
        self._sequence = 0

    @property
    def failures(self) -> list[dict[str, Any]]:
        with self._condition:
            return [dict(row) for row in self._failures]

    @property
    def samples(self) -> list[TelemetrySample]:
        with self._condition:
            return list(self._samples)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RobotState sampler may only be started once")
        self._thread = threading.Thread(
            target=self._run,
            name="oap-readonly-robot-state-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = self._monotonic_clock()
            try:
                raw = self._read_state()
                received_unix = float(self._wall_clock())
                received_monotonic = float(self._monotonic_clock())
                checked = validate_robot_state_telemetry(
                    raw,
                    expected_joint_count=self._expected_joint_count,
                )
                q = tuple(float(v) for v in getattr(raw, "q"))
                tcp = tuple(float(v) for v in getattr(raw, "tcp_pose"))
                self._sequence += 1
                sample = TelemetrySample(
                    sequence=self._sequence,
                    robot_state_stamp_s=float(checked["stamp_s"]),
                    received_unix_s=received_unix,
                    received_monotonic_s=received_monotonic,
                    q_rad=q,
                    tcp_pose=tcp,
                    gripper_width_m=float(checked["gripper_width_m"]),
                    gripper_force_n=(
                        None
                        if checked["gripper_force_n"] is None
                        else float(checked["gripper_force_n"])
                    ),
                )
                with self._condition:
                    if (
                        self._samples
                        and sample.robot_state_stamp_s
                        <= self._samples[-1].robot_state_stamp_s
                    ):
                        raise ReadinessMeasurementError(
                            "RobotState stamps are not strictly increasing"
                        )
                    self._samples.append(sample)
                    if len(self._samples) > self._max_samples:
                        del self._samples[
                            : len(self._samples) - self._max_samples
                        ]
                    self._condition.notify_all()
            except BaseException as exc:  # noqa: BLE001 - retain raw failure
                with self._condition:
                    self._failures.append(
                        {
                            "received_unix_s": float(self._wall_clock()),
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    self._condition.notify_all()
                self._stop.wait(min(self._period_s, 0.05))
            elapsed = float(self._monotonic_clock()) - float(started)
            self._stop.wait(max(0.0, self._period_s - elapsed))

    def wait_for_exposure_bracket(
        self,
        *,
        capture_unix_s: float,
        capture_monotonic_s: float,
        timeout_s: float,
    ) -> ExposureTelemetryPair:
        """Return real telemetry immediately before and after an exposure."""
        capture = _finite_positive(
            capture_unix_s, label="camera exposure timestamp"
        )
        capture_monotonic = _finite_positive(
            capture_monotonic_s,
            label="camera exposure monotonic timestamp",
        )
        timeout = _finite_positive(timeout_s, label="pairing timeout")
        deadline = float(self._monotonic_clock()) + timeout
        with self._condition:
            while True:
                before = [
                    row
                    for row in self._samples
                    if row.robot_state_stamp_s <= capture
                ]
                after = [
                    row
                    for row in self._samples
                    if row.robot_state_stamp_s >= capture
                ]
                if before and after:
                    left = before[-1]
                    right = after[0]
                    nearest = min(
                        (left, right),
                        key=lambda row: abs(
                            row.robot_state_stamp_s - capture
                        ),
                    )
                    return ExposureTelemetryPair(
                        capture_unix_s=capture,
                        capture_monotonic_s=capture_monotonic,
                        before=left,
                        after=right,
                        nearest=nearest,
                    )
                remaining = deadline - float(self._monotonic_clock())
                if remaining <= 0.0:
                    raise ReadinessMeasurementError(
                        "camera exposure is not bracketed by real RobotState "
                        "timestamps in the same clock domain"
                    )
                self._condition.wait(timeout=remaining)

    def stop(self, *, timeout_s: float = 5.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                raise ReadinessMeasurementError(
                    "read-only RobotState sampler did not stop"
                )


def quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    """Deterministic compact distribution summary."""
    numeric = np.asarray([float(value) for value in values], dtype=float)
    if numeric.size == 0:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    if not np.all(np.isfinite(numeric)):
        raise ValueError("summary values contain NaN/Inf")
    return {
        "count": int(numeric.size),
        "min": float(np.min(numeric)),
        "p50": float(np.quantile(numeric, 0.50)),
        "p90": float(np.quantile(numeric, 0.90)),
        "p95": float(np.quantile(numeric, 0.95)),
        "p99": float(np.quantile(numeric, 0.99)),
        "max": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
    }


def summarize_measurements(
    samples: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize samples without converting evidence into policy limits."""

    def successful(name: str) -> list[float]:
        return [
            float(row[name])
            for row in samples
            if row.get("status") == "ok" and row.get(name) is not None
        ]

    ok_rows = [row for row in samples if row.get("status") == "ok"]
    stamps = [
        float(row["robot_state_stamp_s"])
        for row in samples
        if row.get("status") == "ok"
        and row.get("robot_state_stamp_s") is not None
    ]
    offsets = [
        float(row["robot_state_received_unix_s"])
        - float(row["robot_state_stamp_s"])
        for row in samples
        if row.get("status") == "ok"
        and row.get("robot_state_received_unix_s") is not None
        and row.get("robot_state_stamp_s") is not None
    ]
    return {
        "schema": READINESS_MEASUREMENT_SCHEMA,
        "sample_count_recorded": len(samples),
        "sample_count_ok": len(ok_rows),
        "sample_count_failed": sum(
            row.get("status") != "ok" for row in samples
        ),
        "failure_record_count": len(failures),
        "metrics": {
            "production_maskless_observation_age_s": quantiles(
                successful("observation_age_s")
            ),
            "camera_robot_timestamp_skew_s": quantiles(
                successful("camera_robot_timestamp_skew_s")
            ),
            "remote_shadow_plan_staleness_s": quantiles(
                successful("shadow_plan_staleness_s")
            ),
            "remote_shadow_round_trip_s": quantiles(
                successful("shadow_round_trip_s")
            ),
            "robot_state_receive_minus_stamp_s": quantiles(offsets),
        },
        "clock_evidence": {
            "robot_state_stamp_strictly_increasing": all(
                right > left for left, right in zip(stamps, stamps[1:])
            ),
            "camera_exposures_bracketed_by_robot_state": all(
                bool(row.get("robot_state_bracketed"))
                for row in ok_rows
            ),
            "clock_domain_claim": (
                "measurable_from_real_before_after_brackets"
                if ok_rows
                and all(
                    bool(row.get("robot_state_bracketed"))
                    for row in ok_rows
                )
                else "NO-GO"
            ),
        },
        "readiness_limits": {
            "max_observation_age_s": {
                "status": "MEASURED_NOT_AUTHORIZED",
                "value": None,
            },
            "max_camera_robot_skew_s": {
                "status": "MEASURED_NOT_AUTHORIZED",
                "value": None,
            },
            "max_plan_staleness_s": {
                "status": "MEASURED_NOT_AUTHORIZED",
                "value": None,
            },
            "max_final_joint_tracking_error_rad": {
                "status": "NO-GO",
                "value": None,
                "reason": "requires authorized measured-motion trials",
            },
            "min_terminal_anchor_reliability": {
                "status": "NO-GO",
                "value": None,
                "reason": (
                    "requires held-out labelled 6D pose/occlusion/OOD data"
                ),
            },
        },
        "authorization_created": False,
    }


def source_identity(repo_root: Path) -> dict[str, Any]:
    """Describe the exact local source tree without mutating it."""
    root = Path(repo_root).resolve()
    import subprocess

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            check=True,
            capture_output=True,
        )
        return result.stdout.strip()

    return {
        "repo_root": str(root),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
    }


def write_evidence_package(
    out_dir: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
    telemetry_samples: Sequence[TelemetrySample],
    identities: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write raw evidence and a SHA-256 manifest atomically."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_samples.jsonl"
    telemetry_path = out / "robot_state_samples.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True) + "\n"
            for row in samples
        ),
        encoding="utf-8",
    )
    telemetry_path.write_text(
        "".join(
            json.dumps(row.to_dict(), sort_keys=True) + "\n"
            for row in telemetry_samples
        ),
        encoding="utf-8",
    )
    write_json_atomic(out / "summary.json", summarize_measurements(
        samples, failures
    ))
    write_json_atomic(out / "identities.json", dict(identities))
    write_json_atomic(out / "failures.json", {
        "schema": READINESS_MEASUREMENT_SCHEMA,
        "failures": [dict(row) for row in failures],
    })
    files = sorted(
        path for path in out.iterdir()
        if path.is_file() and path.name != "sha256_manifest.json"
    )
    manifest = {
        "schema": "oap_sha256_manifest_v1",
        "files": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    write_json_atomic(out / "sha256_manifest.json", manifest)
    return manifest


def drain_queue(
    rows: "queue.Queue[Mapping[str, Any]]",
) -> list[dict[str, Any]]:
    """Test/helper utility for non-blocking failure collection."""
    result: list[dict[str, Any]] = []
    while True:
        try:
            result.append(dict(rows.get_nowait()))
        except queue.Empty:
            return result
