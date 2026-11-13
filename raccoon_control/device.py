from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Snapshot:
    connected: bool
    joints_deg: tuple | None
    sensor_sequence: int
    sensor_time: float | None
    moving: bool | None
    completion_sequence: int
    gripper_reported_state: str | None = None
    gripper_time: float | None = None
    gripper_raw: int | None = None
    stop_count: int = 0
    move_count: int = 0
    gripper_module_type: int | None = None
    firmware_warning_raw: int | None = None
    collision_event_counts: tuple | None = None
    battery_state_raw: int | None = None
    signal_strength_raw: int | None = None

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        fields = dict(data)
        if fields.get("joints_deg") is not None:
            fields["joints_deg"] = tuple(fields["joints_deg"])
        if fields.get("collision_event_counts") is not None:
            fields["collision_event_counts"] = tuple(fields["collision_event_counts"])
        return cls(**fields)
