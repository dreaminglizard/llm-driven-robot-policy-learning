from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import urllib.error
import urllib.request

from .errors import ControlError
from .validation import json_loads, keys, number

MAX_REQUEST_BYTES = 65536


def _owner_process_identity():
    """Persist an exact Windows process lifetime, not merely a reusable PID."""
    if os.name != 'nt':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        birth, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), ctypes.byref(birth), ctypes.byref(exited),
                                      ctypes.byref(kernel_time), ctypes.byref(user_time)):
            return None
        created = (birth.dwHighDateTime << 32) | birth.dwLowDateTime
        return {'pid': os.getpid(), 'created_filetime': str(created)}
    except (OSError, AttributeError, TypeError, ValueError):
        return None


def dispatch(controller, method, params):
    allowed = {
        "status": ((), ()),
        "diagnostics": ((), ()),
        "acquire": (("owner", "expected_revision", "expected_instance_id"), ("owner",)),
        "recover": (("observation_id","layout_confirmed","layout_basis","expected_instance_id"), ("observation_id","layout_confirmed")),
        "confirm_idle_stop": (("observation_id",), ("observation_id",)),
        "observe": (("lease_id",), ()),
        "record_calibration": (("lease_id", "label", "reference_mm", "note", "gripper_seen"), ("lease_id", "label")),
        "apply_action": (("lease_id", "command_id", "observation_id", "action"), ("lease_id", "command_id", "observation_id", "action")),
        "motion_interruption_capability": (("controller_instance_id",), ("controller_instance_id",)),
        "interrupt_motion": (("lease_id", "command_id", "controller_instance_id"), ("lease_id", "command_id", "controller_instance_id")),
        "resume_interrupted_motion": (("lease_id", "interruption_id", "observation_id", "controller_instance_id"), ("lease_id", "interruption_id", "observation_id", "controller_instance_id")),
        "interrupt_mock_motion": (("lease_id", "command_id", "controller_instance_id"), ("lease_id", "command_id", "controller_instance_id")),
        "resume_mock_interrupted_motion": (("lease_id", "interruption_id", "observation_id", "controller_instance_id"), ("lease_id", "interruption_id", "observation_id", "controller_instance_id")),
        "stop": (("reason",), ()),
        "end_episode": (("lease_id",), ("lease_id",)),
        "result": (("command_id",), ("command_id",)),
        "shutdown": ((), ()),
    }
    if not isinstance(method, str) or method not in allowed:
        raise ControlError("METHOD_NOT_FOUND", "Unknown controller operation")
    permitted, required = allowed[method]
    keys(params, allowed=permitted, required=required)
    if method == 'recover' and params.get('layout_basis') in ('expected', 'unregistered') and not params.get('expected_instance_id'):
        raise ControlError('CONTROLLER_CHANGED', '자동 복구할 상주 제어기 식별자가 필요합니다.')
    if method == "shutdown":
        return {"accepted": True}
    if method == "apply_action":
        return controller.apply_action(params)
    if method in ("motion_interruption_capability", "interrupt_motion", "resume_interrupted_motion", "interrupt_mock_motion", "resume_mock_interrupted_motion"):
        if params["controller_instance_id"] != controller.instance_id:
            raise ControlError("CONTROLLER_CHANGED", "Retarget request belongs to another controller instance")
        handler = getattr(controller, method, None)
        if not callable(handler):
            raise ControlError("UNSUPPORTED_ACTION", "This controller does not provide motion interruption")
        return handler(**{key: value for key, value in params.items() if key != "controller_instance_id"})
    return getattr(controller, method)(**params)


class ControllerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, controller, port=0):
        self.controller = controller
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), RequestHandler)

    def write_endpoint(self, runtime):
        path = Path(runtime).resolve() / "controller.json"
        document = {"schema_version": 1, "host": "127.0.0.1", "port": self.server_address[1], "token": self.token, "instance_id": self.controller.instance_id,
                    "owner_process": _owner_process_identity()}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(document), encoding="utf-8")
        temporary.replace(path)
        return path


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_POST(self):
        self.connection.settimeout(3)
        if self.path != "/rpc":
            self._reply(404, {"ok": False, "error": {"code": "NOT_FOUND", "message": "Use /rpc", "details": {}}})
            return
        authorization = self.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization, "Bearer " + self.server.token):
            self._reply(401, {"ok": False, "error": {"code": "UNAUTHORIZED", "message": "A local controller token is required", "details": {}}})
            return
        shutdown = False
        try:
            if self.headers.get_content_type() != "application/json":
                raise ControlError("INVALID_CONTENT_TYPE", "Use application/json")
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as exc:
                raise ControlError("INVALID_LENGTH", "Invalid Content-Length") from exc
            if not 0 <= size <= MAX_REQUEST_BYTES:
                raise ControlError("REQUEST_TOO_LARGE", "The request must be at most 64 KiB")
            body = self.rfile.read(size)
            if len(body) != size:
                raise ControlError("INVALID_LENGTH", "Incomplete request")
            request = json_loads(body)
            keys(request, allowed={"method", "params"}, required={"method"})
            value = dispatch(self.server.controller, request["method"], request.get("params", {}))
            shutdown = request["method"] == "shutdown"
            self._reply(200, {"ok": True, "value": value})
        except ControlError as exc:
            self._reply(400, {"ok": False, "error": exc.as_dict()})
        except (OSError, UnicodeError):
            return
        except Exception:
            self._reply(500, {"ok": False, "error": {"code": "SERVER_ERROR", "message": "Unexpected controller error", "details": {}}})
        if shutdown:
            threading.Thread(target=self.server.shutdown, daemon=True).start()


class Client:
    def __init__(self, runtime, timeout=20):
        try:
            endpoint = json_loads((Path(runtime) / "controller.json").read_text(encoding="utf-8"))
        except OSError as exc:
            raise ControlError("CONTROLLER_NOT_FOUND", "Start the resident controller first") from exc
        if endpoint.get("host") != "127.0.0.1":
            raise ControlError("INVALID_ENDPOINT", "Only the local controller endpoint is supported")
        port = endpoint.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ControlError("INVALID_ENDPOINT", "Invalid controller port")
        self.endpoint = endpoint
        self.timeout = timeout

    def call(self, method, **params):
        return self.call_with_timeout(method, timeout=self.timeout, **params)

    def call_with_timeout(self, method, *, timeout, **params):
        timeout = number(timeout, 'client_timeout', minimum=.001)
        body = json.dumps({"method": method, "params": params}, allow_nan=False).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.endpoint['port']}/rpc", data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.endpoint["token"]},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json_loads(response.read())
        except urllib.error.HTTPError as exc:
            with exc:
                result = json_loads(exc.read())
        except (urllib.error.URLError, OSError) as exc:
            raise ControlError("CONTROLLER_UNREACHABLE", "The resident controller did not respond; do not resend movement with a new command ID") from exc
        if not result.get("ok"):
            error = result.get("error", {})
            raise ControlError(error.get("code", "RPC_ERROR"), error.get("message", "Controller request failed"), **error.get("details", {}))
        return result["value"]
