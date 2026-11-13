"""Calibration client for the existing resident controller; never opens a robot port.

Start `raccoon_control serve` explicitly, then use one jog/measurement at a time.
There is no command sequence, automatic retry or workspace-verification toggle.
"""
import argparse
import json
from pathlib import Path
import uuid

from .errors import ControlError
from .transport import Client
from .validation import json_loads


def execute(client, *, action=None, measurement=None, command_id=None):
    """Own one short lease; retain the command ID even when the reply is uncertain."""
    command_id = command_id or "calibration-" + uuid.uuid4().hex
    report = {"command_id": command_id if action is not None else None, "physical_success": None}
    lease = client.call("acquire", owner="calibration-client")["lease_id"]
    try:
        if action is not None:
            observation = client.call("observe", lease_id=lease)
            report["result"] = client.call("apply_action", lease_id=lease, command_id=command_id,
                                           observation_id=observation["observation_id"], action=action)
            report["ok"] = report["result"]["status"] in {
                "completed", "already_at_target", "gripper_reported", "gripper_commanded_unverified"}
        else:
            report["sample"] = client.call("record_calibration", lease_id=lease, **measurement)
            report["ok"] = True
    except (ControlError, KeyboardInterrupt) as exc:
        error = exc.as_dict() if isinstance(exc, ControlError) else {"code": "CANCELLED", "message": "Calibration interrupted"}
        report.update(ok=False, error=error)
        try:
            report["stop"] = client.call("stop", reason="CALIBRATION_REQUEST_FAILED")
        except ControlError as stop_error:
            report["stop_error"] = stop_error.as_dict()
    finally:
        try:
            report["ended"] = client.call("end_episode", lease_id=lease)
            if not report["ended"]["stop"]["confirmed"]:
                report["ok"] = False
        except ControlError as end_error:
            report.update(ok=False, end_error=end_error.as_dict())
    return report


def samples(runtime):
    path = Path(runtime) / "events.jsonl"
    rows = [json_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"schema": "raccoon-calibration-samples.v1", "calibration_complete": False,
            "samples": [row["sample"] for row in rows if row.get("event") == "calibration_sample_recorded"]}


def parser():
    result = argparse.ArgumentParser(description="실물 교정: 소량 이동·집게 시험·좌표 기록. 실행 중인 제어기에 접속합니다.")
    result.add_argument("--runtime", default=".runtime/calibration")
    sub = result.add_subparsers(dest="operation", required=True)
    sub.add_parser("show", help="현재 관절각·명목 손끝 좌표·집게 원값")
    joint = sub.add_parser("joint", help="관절 하나를 현재 위치에서 최대 ±2° 이동")
    joint.add_argument("joint", type=int, choices=range(1, 5))
    joint.add_argument("delta", type=float)
    joint.add_argument("--speed", type=float, default=12)
    tcp = sub.add_parser("tcp", help="현재 손끝에서 XYZ 증분 이동, 한 번에 최대 5mm")
    tcp.add_argument("delta", type=float, nargs=3, metavar=("DX", "DY", "DZ"))
    tcp.add_argument("--speed", type=float, default=12)
    grip = sub.add_parser("gripper", help="집게 시험. 원통 파지 성공을 자동으로 판정하지 않습니다.")
    grip.add_argument("state", choices=("open", "closed"))
    record = sub.add_parser("record", help="현재 수신과 현장에서 확인한 기준·설명을 함께 기록")
    record.add_argument("label")
    record.add_argument("--reference-mm", type=float, nargs=3, metavar=("X", "Y", "Z"))
    record.add_argument("--note", default="")
    record.add_argument("--gripper-seen", choices=("open", "closed"))
    export = sub.add_parser("samples", help="저장된 교정 표본을 읽습니다. 로봇 연결 불필요")
    export.add_argument("--output")
    sub.add_parser("stop", help="현재 팔 동작 정지")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.operation == "samples":
            value = samples(args.runtime)
            if args.output:
                # Do not overwrite prior measurement evidence accidentally.
                with Path(args.output).open("x", encoding="utf-8") as output:
                    json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            report = {"ok": True, **value}
        else:
            client = Client(args.runtime)
            if args.operation == "show":
                report = {"ok": True, "status": client.call("status"), "observation": client.call("observe")}
            elif args.operation == "stop":
                stopped = client.call("stop", reason="CALIBRATION_OPERATOR_STOP")
                report = {"ok": stopped["confirmed"], "stop": stopped}
            elif args.operation == "record":
                report = execute(client, measurement={"label": args.label, "reference_mm": args.reference_mm,
                                 "note": args.note, "gripper_seen": args.gripper_seen})
            else:
                if args.operation == "joint":
                    action = {"kind": "jog_joint", "joint": args.joint, "delta_deg": args.delta, "speed_pct": args.speed}
                elif args.operation == "tcp":
                    action = {"kind": "jog_tcp", "delta_mm": args.delta, "speed_pct": args.speed}
                else:
                    action = {"kind": "set_gripper", "state": args.state}
                report = execute(client, action=action)
    except ControlError as exc:
        report = {"ok": False, "error": exc.as_dict()}
    except (OSError, ValueError) as exc:
        report = {"ok": False, "error": {"code": "INPUT_ERROR", "message": str(exc)}}
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
