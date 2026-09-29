"""CLI adapter for attended, empty-gripper ObjectHeld calibration."""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from oap.loop.object_held_recording import (
    ObjectHeldEvidenceRecorder,
    hash_evidence_tree,
)


class FlexivCalibrationRobot:
    """Strict gripper-only adapter over flexiv-control runtime v5."""

    def __init__(self, *, host: str, port: int) -> None:
        from flexiv_control import RemoteRobot

        self._control_hz: float | None = None
        self._robot = RemoteRobot(
            str(host),
            port=int(port),
            owner="oap-objectheld-empty-calibration",
        )
        self._observer = RemoteRobot(
            str(host),
            port=int(port),
            owner="oap-objectheld-readback-observer",
        )
        self._robot.__enter__()
        try:
            # Observation RPCs require no lease. Keeping this connection
            # separate lets it sample while the owner connection blocks in the
            # controller's audited gripper-wait implementation.
            self._observer.connect()
        except Exception:
            self._robot.__exit__(None, None, None)
            raise

    def get_server_info(self) -> Mapping[str, Any]:
        from flexiv_control.server import protocol as protocol_contract

        info = dict(self._robot.get_server_info())
        expected = {
            "protocol_fingerprint_sha256": (
                protocol_contract.PROTOCOL_FINGERPRINT_SHA256
            ),
            "source_fingerprint_sha256": (
                protocol_contract.SOURCE_FINGERPRINT_SHA256
            ),
        }
        for key, local_value in expected.items():
            if info.get(key) != local_value:
                raise RuntimeError(
                    f"controller {key} does not match the installed client"
                )
        self._control_hz = float(info["control_hz"])
        return info

    def get_safety_profile(self) -> Any:
        return self._robot.get_safety_profile()

    def get_state(self) -> Any:
        return self._robot.get_state()

    def command_gripper(
        self,
        *,
        width_m: float,
        force_n: float,
        velocity_m_s: float,
    ) -> Mapping[str, Any]:
        from flexiv_control import GripperCommand

        if self._control_hz is None or self._control_hz <= 0.0:
            raise RuntimeError(
                "runtime control_hz must be validated before a gripper command"
            )
        transition: list[dict[str, Any]] = []
        observer_errors: list[Exception] = []
        stop_observer = threading.Event()
        sample_period_s = 1.0 / self._control_hz

        def observe_transition() -> None:
            while not stop_observer.is_set():
                try:
                    state = self._observer.get_state()
                    transition.append(
                        {
                            "stamp_s": float(state.stamp),
                            "gripper_width_m": float(state.gripper_width),
                            "gripper_is_moving": bool(
                                state.gripper_is_moving
                            ),
                        }
                    )
                except Exception as exc:
                    observer_errors.append(exc)
                    return
                stop_observer.wait(sample_period_s)

        observer = threading.Thread(
            target=observe_transition,
            name="objectheld-gripper-readback",
            daemon=True,
        )
        observer.start()
        try:
            settled = self._robot.command_gripper(
                GripperCommand(
                    width=float(width_m),
                    force=float(force_n),
                    velocity=float(velocity_m_s),
                    grasp=False,
                ),
                wait=True,
                timeout=10.0,
            )
        finally:
            stop_observer.set()
            observer.join()
        if observer_errors:
            error = observer_errors[0]
            raise RuntimeError(
                "read-only gripper transition observation failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        return {
            "server_returned_width_m": (
                None if settled is None else float(settled)
            ),
            "transition_telemetry": transition,
            "transition_sample_period_s": sample_period_s,
        }

    def close(self) -> None:
        robot = getattr(self, "_robot", None)
        observer = getattr(self, "_observer", None)
        if observer is not None:
            observer.close()
            self._observer = None
        if robot is not None:
            robot.__exit__(None, None, None)
            self._robot = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record attended, object-independent ObjectHeld calibration: "
            "empty GN01 closes and width readback only."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("empty",),
        default="empty",
        help="Only the object-independent empty phase exists.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--server-host", default="ROBOT-HOST-PLACEHOLDER")
    parser.add_argument("--server-port", type=int, default=8766)
    parser.add_argument("--holdout-trials", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--i-confirm-real-motion", action="store_true")
    parser.add_argument("--i-confirm-estop-in-hand", action="store_true")
    parser.add_argument(
        "--i-confirm-empty-gripper-and-workspace-clear",
        action="store_true",
    )
    return parser


def _require_attended_confirmation(args: argparse.Namespace) -> None:
    if not args.execute or not args.i_confirm_real_motion:
        raise RuntimeError(
            "calibration is dry-disabled; pass --execute and "
            "--i-confirm-real-motion only while attending the robot"
        )
    if not args.i_confirm_estop_in_hand:
        raise RuntimeError("calibration requires the operator to hold the e-stop")
    if not args.i_confirm_empty_gripper_and_workspace_clear:
        raise RuntimeError(
            "calibration requires explicit empty-gripper/workspace-clear "
            "confirmation"
        )
    if not str(args.operator_id).strip():
        raise RuntimeError("--operator-id must be non-empty")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_attended_confirmation(args)
    out_dir = Path(args.out).expanduser().resolve()
    robot: FlexivCalibrationRobot | None = None
    try:
        robot = FlexivCalibrationRobot(
            host=args.server_host,
            port=args.server_port,
        )
        recorder = ObjectHeldEvidenceRecorder(robot, out_dir=out_dir)
        recorder.record_empty(
            confirmed_empty_and_clear=True,
            holdout_trials=int(args.holdout_trials),
        )
    finally:
        if robot is not None:
            robot.close()
        if out_dir.exists():
            confirmation = {
                "schema": "oap_attended_calibration_confirmation_v2",
                "phase": "empty",
                "operator_id": str(args.operator_id).strip(),
                "recorded_at_unix_s": time.time(),
                "real_motion_confirmed": bool(args.i_confirm_real_motion),
                "estop_in_hand_confirmed": bool(
                    args.i_confirm_estop_in_hand
                ),
                "empty_gripper_confirmed": bool(
                    args.i_confirm_empty_gripper_and_workspace_clear
                ),
                "arm_motion_command_count": 0,
                "phase_completed": (
                    out_dir / "empty" / "phase.json"
                ).is_file(),
            }
            path = out_dir / "attended_confirmation_empty.json"
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(
                json.dumps(confirmation, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
            hash_evidence_tree(out_dir)
    print(out_dir / "empty" / "phase.json")
    return 0
