from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re

from .errors import ControlError
from .validation import json_loads, keys, number, vector


@dataclass(frozen=True)
class Limits:
    max_speed_pct: float = 35.0
    joint_tolerance_deg: float = 0.25
    tcp_tolerance_mm: float = 1.0
    coarse_residual_mm: float = 5.0
    stable_seconds: float = 0.2
    motion_timeout: float = 8.0
    sensor_max_age: float = 0.3
    observation_max_age: float = 30.0
    observation_joint_drift_deg: float = 0.5
    sample_period: float = 0.05
    rpc_timeout: float = 0.75
    connect_timeout: float = 15.0
    close_timeout: float = 3.0
    lease_seconds: float = 30.0
    stop_timeout: float = 1.0
    max_corrections: int = 3
    max_xy_correction_mm: float = 5.0
    path_step_deg: float = 0.5
    # Real devices have no assumed workspace. Only Config(backend="mock")
    # supplies the historical fixture when no explicit bounds were provided.
    workspace_min_mm: tuple | None = None
    workspace_max_mm: tuple | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.startswith("workspace_"):
                if value is not None:
                    object.__setattr__(self,name,vector(value,3,name))
            else:
                number(value, name, minimum=0.0001)
        if not isinstance(self.max_corrections, int) or isinstance(self.max_corrections, bool):
            raise ControlError("INVALID_CONFIG", "max_corrections must be an integer")
        if (self.workspace_min_mm is None)!=(self.workspace_max_mm is None):
            raise ControlError("INVALID_CONFIG", "Both workspace bounds must be specified together")
        if self.workspace_configured:
            for lo, hi in zip(self.workspace_min_mm, self.workspace_max_mm):
                if lo >= hi:
                    raise ControlError("INVALID_CONFIG", "Workspace lower bounds must be below upper bounds")
        if self.max_speed_pct > 35 or self.max_xy_correction_mm > 5 or self.coarse_residual_mm > 5:
            raise ControlError("INVALID_CONFIG", "Phase-one speed/correction limits cannot be enlarged")
        if self.motion_timeout > 30 or self.connect_timeout > 15 or self.close_timeout > 3:
            raise ControlError("INVALID_CONFIG", "Motion, connect and close deadlines are capped at 30, 15 and 3 seconds")
        if self.path_step_deg > 0.5 or self.sample_period > self.sensor_max_age:
            raise ControlError("INVALID_CONFIG", "Unsafe path or sampling interval")

    @property
    def workspace_configured(self):
        return self.workspace_min_mm is not None and self.workspace_max_mm is not None

    def with_mock_workspace(self):
        return replace(self,workspace_min_mm=(-180.0,30.0,70.0),workspace_max_mm=(180.0,280.0,290.0))


@dataclass(frozen=True)
class Config:
    backend: str = "mock"
    mode: str = "calibration"
    port: str | None = None
    hardware_enabled: bool = False
    workspace_verified: bool = False
    workspace_source: str = "unspecified"
    precision_validated: bool = False
    tool_forward_mm: float = 80.0
    tool_vertical_mm: float = -6.0
    limits: Limits = field(default_factory=Limits)
    mock: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.backend, str) or not isinstance(self.mode, str) or self.backend not in {"mock", "sdk"} or self.mode not in {"calibration", "collection"}:
            raise ControlError("INVALID_CONFIG", "Unsupported backend or operating mode")
        for name in ("hardware_enabled", "workspace_verified", "precision_validated"):
            if not isinstance(getattr(self, name), bool):
                raise ControlError("INVALID_CONFIG", f"{name} must be boolean")
        number(self.tool_forward_mm, "tool_forward_mm", minimum=0, maximum=200)
        number(self.tool_vertical_mm, "tool_vertical_mm", minimum=-100, maximum=100)
        if not isinstance(self.mock, dict):
            raise ControlError("INVALID_CONFIG", "mock must be an object")
        if not isinstance(self.limits,Limits):
            raise ControlError("INVALID_CONFIG", "limits must be a Limits object")
        if self.workspace_source not in ("unspecified", "fixed_layout_photo"):
            raise ControlError("INVALID_CONFIG", "Unknown workspace reference source")
        if self.workspace_source == "fixed_layout_photo" and (self.workspace_verified or self.mode != "calibration"):
            raise ControlError("INVALID_CONFIG", "Photo-based trial coordinates are not measured calibration and require calibration mode")
        if self.backend=="mock" and not self.limits.workspace_configured:
            object.__setattr__(self,"limits",self.limits.with_mock_workspace())
        if self.backend == "sdk":
            if not isinstance(self.port, str) or not re.fullmatch(r"COM[1-9][0-9]{0,3}", self.port):
                raise ControlError("INVALID_CONFIG", "SDK mode requires an explicit COM port")
            if not self.hardware_enabled:
                raise ControlError("HARDWARE_DISABLED", "Real hardware is disabled in this configuration")
            if self.mode != "calibration":
                raise ControlError("PERCEPTION_UNAVAILABLE", "Phase one supports real hardware calibration only")
            if self.workspace_verified and not self.limits.workspace_configured:
                raise ControlError("WORKSPACE_UNSPECIFIED", "A real workspace cannot be verified without explicit bounds")
            if self.workspace_source == "fixed_layout_photo" and not self.limits.workspace_configured:
                raise ControlError("WORKSPACE_UNSPECIFIED", "A photo-based trial needs explicit reviewed path bounds")

    @property
    def workspace_ready(self):
        return self.workspace_verified or (self.workspace_source == "fixed_layout_photo"
                and self.mode == "calibration" and self.limits.workspace_configured)

    def workspace_status(self):
        return {"configured":self.limits.workspace_configured,"verified":self.workspace_verified,
                "reference_source":self.workspace_source,"trial_ready":self.workspace_ready,
                "scope":"simulation" if self.backend=="mock" else "hardware",
                "min_mm":self.limits.workspace_min_mm,"max_mm":self.limits.workspace_max_mm}

    @classmethod
    def from_dict(cls, value):
        allowed = set(cls.__dataclass_fields__)
        keys(value, allowed=allowed)
        values = dict(value)
        if "limits" in values:
            keys(values["limits"], allowed=Limits.__dataclass_fields__)
            values["limits"] = Limits(**values["limits"])
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json_loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self):
        return asdict(self)

    def fingerprint(self):
        document=json.dumps(self.as_dict(),sort_keys=True,separators=(",",":"),allow_nan=False)
        return hashlib.sha256(document.encode("utf-8")).hexdigest()
