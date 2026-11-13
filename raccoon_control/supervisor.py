import multiprocessing
from multiprocessing.reduction import ForkingPickler
import threading

from .errors import ControlError
from .worker import worker_main


class DeviceProcess:
    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self._seq = 0
        self.connection = None
        self.process = None
        self.metadata = None
        self.terminated = False

    @property
    def alive(self):
        return self.process is not None and self.process.is_alive()

    @property
    def supports_mock_blended_joint_path(self):
        # A real SDK process does not advertise the virtual timing contract.
        return self.config.backend == 'mock' and self.config.hardware_enabled is False

    @property
    def supports_sdk_blended_joint_path(self):
        # Capability is only exposed after this exact SDK worker started.
        # It does not claim that timing or physical motion is verified.
        return (self.config.backend == 'sdk' and self.config.hardware_enabled is True
                and isinstance(self.metadata, dict) and self.metadata.get('backend') == 'sdk')

    def start(self):
        context = multiprocessing.get_context("spawn")
        parent_end, child_end = context.Pipe()
        self.connection = parent_end
        self.process = context.Process(target=worker_main, args=(child_end, self.config.as_dict()), name="raccoon-device-worker", daemon=True)
        self.process.start()
        child_end.close()
        try:
            if not parent_end.poll(self.config.limits.connect_timeout):
                raise ControlError("CONNECT_TIMEOUT", "Device initialization exceeded its deadline")
            result = parent_end.recv()
            if not result.get("ok"):
                self._raise_error(result)
            self.metadata = result["metadata"]
            return self.metadata
        except (EOFError, OSError) as exc:
            self._terminate()
            raise ControlError("WORKER_EXITED", "Device worker exited during initialization") from exc
        except Exception:
            self._terminate()
            raise

    @staticmethod
    def _raise_error(result):
        error = result.get("error", {})
        raise ControlError(error.get("code", "WORKER_ERROR"), error.get("message", "Device worker failed"), **error.get("details", {}))

    def call(self, method, arguments=None, *, timeout=None, send_guard=None):
        """Send one request and match its reply.

        When provided, ``send_guard`` is a context-manager factory around the
        pipe write only. It is entered after acquiring the RPC lock and after
        serializing the request, so a caller can order a cancellation against
        the actual send without locking cancellation during reply waits.
        """
        timeout = self.config.limits.rpc_timeout if timeout is None else timeout
        with self._lock:
            if not self.alive or self.connection is None:
                raise ControlError("WORKER_EXITED", "Device worker is not running")
            self._seq += 1
            seq = self._seq
            try:
                request = {"seq": seq, "method": method, "arguments": arguments or {}}
                if send_guard is None:
                    self.connection.send(request)
                else:
                    # Connection.recv() accepts the same pickle via send_bytes.
                    # Prepare it outside the cancellation boundary, especially
                    # for larger mock-only trajectory requests.
                    payload = bytes(ForkingPickler.dumps(request))
                    with send_guard():
                        self.connection.send_bytes(payload)
                if not self.connection.poll(timeout):
                    self._terminate()
                    raise ControlError("WORKER_TIMEOUT", "Device operation exceeded its deadline", operation=method, physical_stop_confirmed=False)
                result = self.connection.recv()
            except (EOFError, OSError) as exc:
                self._terminate()
                raise ControlError("WORKER_EXITED", "Lost the device worker", physical_stop_confirmed=False) from exc
            if result.get("seq") != seq:
                self._terminate()
                raise ControlError("WORKER_PROTOCOL", "Unexpected device response sequence")
            if not result.get("ok"):
                self._raise_error(result)
            return result["value"]

    def _terminate(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.75)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(0.75)
                self.terminated = True
            else:
                self.process.join(0.1)
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def close(self):
        error = None
        if self.alive:
            try:
                self.call("close", timeout=self.config.limits.close_timeout)
            except ControlError as exc:
                error = exc.as_dict()
        with self._lock:
            if self.process is not None:
                self.process.join(0.5)
            self._terminate()
        return {"worker_exited": not self.alive, "forced_termination": self.terminated, "error": error}
