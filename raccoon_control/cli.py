import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

from . import __version__
from .config import Config
from .controller import Controller
from .errors import ControlError
from .transport import Client, ControllerServer
from .validation import json_loads


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def serve(args):
    config = Config.load(args.config) if args.config else Config(workspace_verified=True)
    expected = getattr(args, 'expected_config_sha256', None)
    if expected is not None and config.fingerprint() != expected:
        raise ControlError('CONTROLLER_CONFIG_MISMATCH', 'Controller configuration changed before startup')
    if config.backend == "sdk" and not args.allow_hardware:
        raise ControlError("HARDWARE_DISABLED", "Real hardware also requires the explicit --allow-hardware switch")
    controller = Controller(config, args.runtime)
    server = None
    try:
        controller.start()
        server = ControllerServer(controller, args.port)
        endpoint = server.write_endpoint(args.runtime)
        emit({"ok": True, "event": "ready", "endpoint_file": str(endpoint), "status": controller.status()})
        try:
            server.serve_forever(poll_interval=0.05)
        except KeyboardInterrupt:
            pass
    finally:
        if server is not None:
            server.server_close()
        closed = controller.close()
        emit({"ok": True, "event": "closed", "result": closed})
    return 0


def doctor():
    from .sdk_adapter import inspect_sdk
    try:
        sdk = {"available": True, **inspect_sdk()}
    except ControlError as exc:
        sdk = {"available": False, "error": exc.as_dict()}
    return {"version": __version__, "python": sys.version.split()[0], "platform": platform.system(), "default_backend": "mock", "hardware_connected": False, "mock_requires_installation": False, "sdk": sdk}


def demo(runtime):
    """Own one temporary mock daemon, exercise it, then shut down both processes."""
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    demo_root = runtime / ("demo-" + uuid.uuid4().hex)
    demo_root.mkdir()
    output = (demo_root / "server-output.txt").open("w", encoding="utf-8")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, "-m", "raccoon_control", "serve", "--runtime", str(demo_root)], stdout=output, stderr=subprocess.STDOUT, creationflags=creationflags)
    client = None
    try:
        deadline = time.monotonic() + 10
        while not (demo_root / "controller.json").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise ControlError("DEMO_START_FAILED", "The mock daemon did not start", log=str(demo_root / "server-output.txt"))
            time.sleep(0.03)
        client = Client(demo_root)
        lease = client.call("acquire", owner="mock-demo")
        observation = client.call("observe", lease_id=lease["lease_id"])
        request = {"lease_id": lease["lease_id"], "command_id": "demo-correction", "observation_id": observation["observation_id"], "action": {"kind": "correct_xy", "reference_id": "tcp", "delta_xy_mm": [2, 0], "speed_pct": 12}}
        result = client.call("apply_action", **request)
        repeat = client.call("apply_action", **request)
        ended = client.call("end_episode", lease_id=lease["lease_id"])
        return {"simulated": True, "runtime": str(demo_root), "motion_status": result["status"], "replayed": repeat["replayed"], "device_move_count": repeat["observation"]["move_count"], "stop": ended["stop"], "verified_task_success": None}
    finally:
        if client is not None:
            try:
                client.call("shutdown")
            except ControlError:
                pass
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=3)
        output.close()


def build_parser():
    parser = argparse.ArgumentParser(description="라쿤봇 상주 제어기 — 기본값은 모의 장치이며 실제 연결은 명시적으로 활성화해야 합니다.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("doctor", help="설치·SDK 지문 확인. 로봇에는 연결하지 않습니다.")
    server = sub.add_parser("serve", help="상주 제어기 시작")
    server.add_argument("--runtime", default=".runtime")
    server.add_argument("--config")
    server.add_argument("--expected-config-sha256", help=argparse.SUPPRESS)
    server.add_argument("--port", type=int, default=0)
    server.add_argument("--allow-hardware", action="store_true")
    for operation in ("status", "diagnostics", "acquire", "observe", "apply", "stop", "end", "result", "shutdown", "demo"):
        command = sub.add_parser(operation)
        command.add_argument("--runtime", default=".runtime")
        if operation == "acquire":
            command.add_argument("--owner", required=True)
        elif operation == "observe":
            command.add_argument("--lease")
        elif operation == "apply":
            command.add_argument("--request-file", required=True, help="lease_id, command_id, observation_id, action이 있는 JSON")
        elif operation == "end":
            command.add_argument("--lease", required=True)
        elif operation == "result":
            command.add_argument("--command-id", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.operation == "serve":
            return serve(args)
        if args.operation == "doctor":
            result = doctor()
        elif args.operation == "demo":
            result = demo(args.runtime)
        else:
            client = Client(args.runtime)
            if args.operation == "acquire":
                result = client.call("acquire", owner=args.owner)
            elif args.operation == "observe":
                result = client.call("observe", **({"lease_id": args.lease} if args.lease else {}))
            elif args.operation == "apply":
                request = json_loads(Path(args.request_file).read_text(encoding="utf-8"))
                result = client.call("apply_action", **request)
            elif args.operation == "end":
                result = client.call("end_episode", lease_id=args.lease)
            elif args.operation == "result":
                result = client.call("result", command_id=args.command_id)
            else:
                result = client.call(args.operation)
        emit({"ok": True, "value": result})
        return 0
    except ControlError as exc:
        emit({"ok": False, "error": exc.as_dict()})
        return 2
    except (OSError, TypeError, ValueError) as exc:
        emit({"ok": False, "error": {"code": "INPUT_ERROR", "message": str(exc), "details": {}}})
        return 2
