"""Thread-safe latest-state handoff for a continuously running pose tracker.

This module deliberately owns no camera and imports no perception backend.  A
single producer callback is the *only* code allowed to acquire frames and run
tracking.  The real-robot checkpoint consumes an already-published scene
snapshot from memory; it never calls the producer and therefore cannot create
an extra "verdict" capture.

The producer is expected to be backed by one long-lived camera owner.  Wiring
it to the current one-shot ``zed_capture.py`` process is invalid because the
real executor also opens ``zed_record.py`` during motion.  The buffer is kept
separate from that transport so its atomicity and fail-closed semantics can be
tested without a camera or robot.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "ContinuousSceneTracker",
    "ContinuousTrackingError",
]


class ContinuousTrackingError(RuntimeError):
    """A current, trustworthy tracking snapshot was unavailable."""


SceneProducer = Callable[[int, Mapping[str, Any] | None], Mapping[str, Any]]


def _finite_timestamp(value: object, *, label: str) -> float:
    if value is None:
        raise ContinuousTrackingError(f"{label} is missing")
    try:
        stamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ContinuousTrackingError(f"{label} is not numeric") from exc
    if not math.isfinite(stamp):
        raise ContinuousTrackingError(f"{label} is not finite")
    return stamp


def _validate_scene(
    scene: Mapping[str, Any],
    *,
    now_s: float,
    accept_unscored_tracking: bool = False,
) -> float:
    """Validate the backend-neutral fields required by a real checkpoint."""
    capture_stamp = _finite_timestamp(
        scene.get("capture_timestamp_s"),
        label="continuous tracking capture timestamp",
    )
    if capture_stamp > now_s:
        raise ContinuousTrackingError(
            "continuous tracking capture timestamp is in the future"
        )
    if scene.get("synchronized_capture") is not True:
        raise ContinuousTrackingError(
            "continuous tracking packet is not a synchronized camera capture"
        )
    bodies = scene.get("bodies")
    if not isinstance(bodies, Mapping) or not bodies:
        raise ContinuousTrackingError(
            "continuous tracking packet has no body poses"
        )
    for name, raw_body in bodies.items():
        if not isinstance(raw_body, Mapping):
            raise ContinuousTrackingError(
                f"continuous tracking body {name!r} is not a mapping"
            )
        track_lost = raw_body.get("track_lost")
        if not isinstance(track_lost, bool):
            raise ContinuousTrackingError(
                f"continuous tracking body {name!r} has no explicit boolean "
                "track_lost state"
            )
        if track_lost:
            raise ContinuousTrackingError(
                f"continuous tracking body {name!r} is lost"
            )
        confidence_raw = raw_body.get("confidence")
        mask_scored = raw_body.get("mask_scored") is True
        if (
            confidence_raw is None
            and accept_unscored_tracking
            and not mask_scored
        ):
            # A maskless frame may advance and expose the persistent pose
            # state. It deliberately carries no fabricated numeric
            # confidence; callers that require a mask score must say so.
            confidence = None
        else:
            confidence = _finite_timestamp(
                confidence_raw,
                label=f"continuous tracking body {name!r} confidence",
            )
            if not 0.0 <= confidence <= 1.0:
                raise ContinuousTrackingError(
                    f"continuous tracking body {name!r} confidence is outside "
                    "[0, 1]"
                )
        for field, length in (("pos_plan", 3), ("quat_wxyz", 4)):
            values = raw_body.get(field)
            if not isinstance(values, (list, tuple)) or len(values) != length:
                raise ContinuousTrackingError(
                    f"continuous tracking body {name!r} has no valid {field}"
                )
            try:
                numeric = [float(value) for value in values]
            except (TypeError, ValueError) as exc:
                raise ContinuousTrackingError(
                    f"continuous tracking body {name!r} has non-numeric "
                    f"{field}"
                ) from exc
            if not all(math.isfinite(value) for value in numeric):
                raise ContinuousTrackingError(
                    f"continuous tracking body {name!r} has non-finite "
                    f"{field}"
                )
    return capture_stamp


class ContinuousSceneTracker:
    """Run one scene producer continuously and atomically publish its latest pose.

    ``consume_latest`` is intentionally a pure buffer read.  It waits for a
    *new* packet that is no older than ``max_age_s`` and (optionally) was
    exposed no earlier than ``min_capture_timestamp_s``.  Missing, stale,
    future-dated, unsynchronized, malformed, or lost-track packets are never
    returned.  If a good packet does not arrive before the caller's existing
    perception timeout, the checkpoint fails closed.

    Only one thread invokes ``producer``.  ``start`` may be called once and a
    second start is rejected, making the camera-owner contract explicit.
    """

    def __init__(
        self,
        producer: SceneProducer,
        *,
        previous: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        accept_unscored_tracking: bool = False,
    ) -> None:
        self._producer = producer
        self._previous = (
            None if previous is None else copy.deepcopy(dict(previous))
        )
        self._clock = clock
        self._monotonic = monotonic
        self._accept_unscored_tracking = bool(
            accept_unscored_tracking
        )
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._sequence = 0
        self._consumed_sequence = 0
        self._latest: dict[str, Any] | None = None
        self._last_error: BaseException | None = None

    @property
    def is_alive(self) -> bool:
        """Whether the sole producer thread is currently running."""
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def start(self) -> None:
        """Start the sole producer thread; a second start is a contract error."""
        with self._condition:
            if self._started:
                raise ContinuousTrackingError(
                    "continuous scene tracker may only be started once"
                )
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="oap-continuous-scene-tracker",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        previous = self._previous
        while not self._stop_event.is_set():
            with self._condition:
                next_sequence = self._sequence + 1
            try:
                produced = self._producer(next_sequence, previous)
                if not isinstance(produced, Mapping):
                    raise ContinuousTrackingError(
                        "continuous scene producer returned a non-mapping"
                    )
                scene = copy.deepcopy(dict(produced))
                _validate_scene(
                    scene,
                    now_s=float(self._clock()),
                    accept_unscored_tracking=(
                        self._accept_unscored_tracking
                    ),
                )
            except BaseException as exc:  # noqa: BLE001 - publish fault, keep tracking
                with self._condition:
                    self._last_error = exc
                    self._condition.notify_all()
                # Avoid a hot failure loop while still permitting prompt
                # recovery. This affects only retry scheduling, never planning.
                self._stop_event.wait(0.01)
                continue
            previous = scene
            with self._condition:
                self._sequence = next_sequence
                self._latest = scene
                self._last_error = None
                self._condition.notify_all()

    def consume_latest(
        self,
        *,
        max_age_s: float,
        timeout_s: float,
        min_capture_timestamp_s: float | None = None,
        min_capture_monotonic_s: float | None = None,
        required_body_names: set[str] | None = None,
    ) -> dict[str, Any]:
        """Atomically consume one new, fresh pose snapshot.

        This function never invokes ``producer``. ``min_capture_timestamp_s``
        lets a post-prefix checkpoint wait for a tracker frame exposed at or
        after the measured endpoint robot-state timestamp. The monotonic
        counterpart proves ordering on one host without depending on wall
        clock synchronization. ``required_body_names`` makes a missing task
        body fail closed instead of silently reusing an older pose.
        """
        max_age = float(max_age_s)
        timeout = float(timeout_s)
        if not math.isfinite(max_age) or max_age <= 0.0:
            raise ValueError("max_age_s must be finite and > 0")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and > 0")
        minimum = (
            None
            if min_capture_timestamp_s is None
            else _finite_timestamp(
                min_capture_timestamp_s,
                label="minimum checkpoint capture timestamp",
            )
        )
        minimum_monotonic = (
            None
            if min_capture_monotonic_s is None
            else _finite_timestamp(
                min_capture_monotonic_s,
                label="minimum checkpoint capture monotonic timestamp",
            )
        )
        required = (
            set()
            if required_body_names is None
            else {str(name) for name in required_body_names}
        )
        deadline = float(self._monotonic()) + timeout
        last_problem = "no tracking packet has been published"
        with self._condition:
            if not self._started:
                raise ContinuousTrackingError(
                    "continuous scene tracker was not started"
                )
            while True:
                now = float(self._clock())
                latest = self._latest
                sequence = self._sequence
                if latest is not None and sequence > self._consumed_sequence:
                    try:
                        capture_stamp = _validate_scene(
                            latest,
                            now_s=now,
                            accept_unscored_tracking=(
                                self._accept_unscored_tracking
                            ),
                        )
                        if minimum is not None and capture_stamp < minimum:
                            raise ContinuousTrackingError(
                                "latest continuous tracking packet predates "
                                "the checkpoint endpoint"
                            )
                        bodies = latest.get("bodies")
                        missing = sorted(required - set(bodies or {}))
                        if missing:
                            raise ContinuousTrackingError(
                                "latest continuous tracking packet is missing "
                                f"required bodies: {missing}"
                            )
                        capture_monotonic = latest.get(
                            "capture_monotonic_s"
                        )
                        if minimum_monotonic is not None:
                            capture_monotonic = _finite_timestamp(
                                capture_monotonic,
                                label=(
                                    "continuous tracking capture monotonic "
                                    "timestamp"
                                ),
                            )
                            if capture_monotonic < minimum_monotonic:
                                raise ContinuousTrackingError(
                                    "latest continuous tracking packet predates "
                                    "the checkpoint endpoint in the host "
                                    "monotonic clock"
                                )
                        age = now - capture_stamp
                        if age > max_age:
                            raise ContinuousTrackingError(
                                f"latest continuous tracking packet is stale "
                                f"({age:.6f}s > {max_age:.6f}s)"
                            )
                    except ContinuousTrackingError as exc:
                        last_problem = str(exc)
                    else:
                        scene = copy.deepcopy(latest)
                        self._consumed_sequence = sequence
                        scene["continuous_tracking"] = {
                            "sequence": int(sequence),
                            "capture_timestamp_s": float(capture_stamp),
                            "consumed_timestamp_s": now,
                            "age_s": float(age),
                            "checkpoint_min_capture_timestamp_s": minimum,
                            "capture_monotonic_s": capture_monotonic,
                            "checkpoint_min_capture_monotonic_s": (
                                minimum_monotonic
                            ),
                            "required_body_names": sorted(required),
                            "source": "persistent_tracker_latest_state",
                            "extra_verdict_capture": False,
                            "state_valid": True,
                            "numeric_confidence_current_frame": all(
                                body.get("confidence") is not None
                                for body in scene["bodies"].values()
                            ),
                        }
                        return scene
                if self._last_error is not None:
                    last_problem = (
                        "continuous scene producer failed: "
                        f"{type(self._last_error).__name__}: "
                        f"{self._last_error}"
                    )
                remaining = deadline - float(self._monotonic())
                if remaining <= 0.0:
                    raise ContinuousTrackingError(
                        "no fresh continuous tracking packet before checkpoint "
                        f"timeout: {last_problem}"
                    )
                self._condition.wait(timeout=remaining)

    def stop(self, *, timeout_s: float | None = None) -> None:
        """Stop the producer and wait for camera ownership to be released."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise ContinuousTrackingError(
                "continuous scene producer did not stop; camera ownership "
                "remains unresolved"
            )
