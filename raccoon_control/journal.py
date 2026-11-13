from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading

from .errors import ControlError


class Journal:
    """Admission is flushed before dispatch, so uncertain commands never replay."""
    def __init__(self, root, backend):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self._lock = threading.Lock()
        self.records = {}
        if self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.endswith("\n"):
                            raise ValueError("Incomplete journal tail")
                        item = json.loads(line)
                        if item.get("schema_version") != 1:
                            raise ValueError("Unknown journal schema")
                        if item["event"] == "command_admitted":
                            self.records[item["command_id"]] = {"fingerprint": item["fingerprint"], "result": None}
                        elif item["event"] == "command_result":
                            self.records[item["command_id"]]["result"] = item["result"]
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
                raise ControlError("JOURNAL_CORRUPT", "Inspect the existing journal before starting another controller") from exc
        self.backend = backend
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        self._trace = (self.root / "telemetry.jsonl").open("a", encoding="utf-8", newline="\n")

    @property
    def pending(self):
        return [key for key, record in self.records.items() if record["result"] is None]

    def append(self, event, **data):
        item = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "backend": self.backend, "simulated": self.backend == "mock", "event": event, **data}
        try:
            with self._lock:
                self._stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                self._stream.flush()
                os.fsync(self._stream.fileno())
        except (OSError, ValueError) as exc:
            raise ControlError("JOURNAL_FAILED", "The command journal could not be saved") from exc

    def sample(self, instance_id, snapshot):
        try:
            with self._lock:
                self._trace.write(json.dumps({"instance_id": instance_id, "simulated": self.backend == "mock", **snapshot}, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            raise ControlError("JOURNAL_FAILED", "Telemetry could not be recorded") from exc

    def admit(self, command_id, fingerprint, request, input_observation=None):
        self.append("command_admitted", command_id=command_id, fingerprint=fingerprint, request=request, input_observation=input_observation)
        self.records[command_id] = {"fingerprint": fingerprint, "result": None}

    def finish(self, command_id, result):
        self.append("command_result", command_id=command_id, result=result)
        self.records[command_id]["result"] = result

    def close(self):
        with self._lock:
            self._trace.close()
            self._stream.close()
