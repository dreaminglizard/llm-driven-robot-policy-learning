"""The only process allowed to construct a vendor device."""
import multiprocessing
import os
import threading
import time

from .config import Config
from .errors import ControlError


def worker_main(connection, config_data):
    holder = {"device": None}
    parent = multiprocessing.parent_process()

    def watch_parent():
        while parent is not None and parent.is_alive():
            time.sleep(0.1)
        if parent is None:
            return
        def request_stop():
            device = holder["device"]
            if device is not None:
                try:
                    device.stop()
                except Exception:
                    pass
        stopping = threading.Thread(target=request_stop, daemon=True)
        stopping.start()
        stopping.join(0.5)
        os._exit(72)  # This worker only; no other user's process is touched.

    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        config = Config.from_dict(config_data)
        if config.backend == "mock":
            from .mock_device import MockDevice
            device = MockDevice(config)
        else:
            from .sdk_adapter import SDKDevice
            device = SDKDevice(config)
        holder["device"] = device
        connection.send({"ok": True, "metadata": {"backend": config.backend, "simulated": config.backend == "mock", "pid": os.getpid(), "sdk": getattr(device, "metadata", None)}})
        while True:
            request = connection.recv()
            seq = request["seq"]
            try:
                method, arguments = request["method"], request.get("arguments", {})
                if method == "snapshot":
                    value = device.snapshot().as_dict()
                elif method == "diagnostics":
                    if config.backend != "sdk":
                        raise ControlError("NOT_AVAILABLE", "Raw diagnostics require the SDK backend")
                    value = device.diagnostics()
                elif method == "move_joints":
                    value = device.move_joints(**arguments)
                elif method in ("move_joint_trajectory", "trajectory_trace"):
                    if config.backend != "mock" or config.hardware_enabled is not False:
                        raise ControlError("UNSUPPORTED_ACTION", "Trajectory execution is mock-only")
                    value = getattr(device, method)(**arguments)
                elif method in ("move_blended_joint_path", "blended_path_trace", "blended_start_reference"):
                    if config.backend == 'sdk' and config.hardware_enabled is not True:
                        raise ControlError('HARDWARE_DISABLED', 'The SDK worker is not enabled')
                    value = getattr(device, method)(**arguments)
                elif method in ('position_target_capability', 'position_target_frame'):
                    if config.backend != 'sdk' or config.hardware_enabled is not True:
                        raise ControlError('UNSUPPORTED_ACTION', 'Position-target evidence requires the SDK worker')
                    value = getattr(device, method)(**arguments)
                elif method == "set_gripper":
                    value = device.set_gripper(**arguments)
                elif method == "stop":
                    value = device.stop()
                elif method == "close":
                    device.close()
                    holder["device"] = None
                    connection.send({"seq": seq, "ok": True, "value": {"closed": True}})
                    break
                else:
                    raise ControlError("INVALID_WORKER_METHOD", "Unknown worker method")
                connection.send({"seq": seq, "ok": True, "value": value})
            except ControlError as exc:
                connection.send({"seq": seq, "ok": False, "error": exc.as_dict()})
            except Exception as exc:
                connection.send({"seq": seq, "ok": False, "error": {"code": "DEVICE_ERROR", "message": type(exc).__name__, "details": {}}})
    except (EOFError, BrokenPipeError, OSError):
        pass
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, ControlError) else {"code": "DEVICE_INIT_ERROR", "message": type(exc).__name__, "details": {}}
        try:
            connection.send({"ok": False, "error": error})
        except (OSError, EOFError):
            pass
    finally:
        device = holder["device"]
        if device is not None:
            def cleanup():
                try:
                    device.stop()
                finally:
                    device.close()
            thread = threading.Thread(target=cleanup, daemon=True)
            thread.start()
            thread.join(0.5)
            if thread.is_alive():
                os._exit(73)
        connection.close()
