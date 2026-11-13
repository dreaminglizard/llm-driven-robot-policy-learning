"""Simulation-only interpolation of arm targets and optional virtual finger opening.

The policy and the one-second command contract do not change.  This profile
ramps each arm target from the last applied control to the new target, then
holds it for the remainder of the command.  It does not impose a validated
joint velocity, acceleration, collision, or real-device safety limit.
The default ramp is linear; minimum jerk is an explicit alternative.
"""
from __future__ import annotations

import math

import mujoco

from raccoon_control.kinematics import check_joints
from .motion_sim import MAX_OPENING_M, _finite_number
from .scene_motion_sim import SceneMotionSim


FINGER_OPEN_INSTANT = "instant"
FINGER_OPEN_HALF_SECOND = "minimum_jerk_0p5s"
FINGER_OPEN_RAMP_SECONDS = 0.5
FULL_OPENING_M = 0.04
CLOSED_MOVE_MINIMUM_JERK = "minimum_jerk"
CLOSED_MOVE_SMOOTHSTEP = "smoothstep"


def _minimum_jerk(progress):
    return 10 * progress**3 - 15 * progress**4 + 6 * progress**5


def _smoothstep(progress):
    return progress * progress * (3. - 2. * progress)


class RampedSceneMotionSim(SceneMotionSim):
    """A virtual scene using the same policy with an arm target ramp."""

    def __init__(self, layout, seed=0, randomize=False, *, ramp_seconds=0.05, threshold_degrees=0.0,
                 ramp_shape="linear", finger_open_profile=FINGER_OPEN_INSTANT, closed_move_profile=None):
        self.ramp_seconds = _finite_number(ramp_seconds, "ramp_seconds", minimum=0)
        self.threshold_degrees = _finite_number(threshold_degrees, "threshold_degrees", minimum=0)
        if self.ramp_seconds <= 0:
            raise ValueError("ramp_seconds must be positive")
        if not isinstance(ramp_shape, str) or ramp_shape not in ("linear", "minimum_jerk"):
            raise ValueError("ramp_shape must be 'linear' or 'minimum_jerk'")
        self.ramp_shape = ramp_shape
        if not isinstance(finger_open_profile, str) or finger_open_profile not in (FINGER_OPEN_INSTANT, FINGER_OPEN_HALF_SECOND):
            raise ValueError("unsupported virtual finger opening profile")
        self._finger_open_profile = finger_open_profile
        # Plain legacy simulators inherit their own arm shape, including linear.
        selected = ramp_shape if closed_move_profile is None else closed_move_profile
        if not isinstance(selected, str) or selected not in (ramp_shape, CLOSED_MOVE_MINIMUM_JERK, CLOSED_MOVE_SMOOTHSTEP):
            raise ValueError("unsupported virtual closed MOVE profile")
        self._closed_move_profile = selected
        super().__init__(layout, seed=seed, randomize=randomize)

    @property
    def finger_open_profile(self):
        return self._finger_open_profile

    @property
    def closed_move_profile(self):
        return self._closed_move_profile

    def reset(self, layout=None, seed=None, randomize=None):
        layout = self._reset_info["initial_layout"] if layout is None else layout
        seed = self._reset_info["seed"] if seed is None else seed
        randomize = self._reset_info["randomize"] if randomize is None else randomize
        self.__init__(layout, seed=seed, randomize=randomize, ramp_seconds=self.ramp_seconds,
                      threshold_degrees=self.threshold_degrees, ramp_shape=self.ramp_shape,
                      finger_open_profile=self.finger_open_profile, closed_move_profile=self.closed_move_profile)
        return self.observation()

    def command(self, joints_deg, gripper_opening_m, seconds=0.5, *, cancel=None):
        if self.layout_status == "uncertain":
            raise RuntimeError("layout is uncertain; explicit reset is required")
        if self.layout_status == "registered":
            self.layout_status = "in_progress"
            self._pending_transfer = None
        try:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("virtual command cancelled before motion")
            joints = check_joints(joints_deg)
            opening = _finite_number(gripper_opening_m, "gripper_opening_m", minimum=0, maximum=MAX_OPENING_M)
            duration = _finite_number(seconds, "seconds", minimum=0)
            if duration <= 0:
                raise ValueError("seconds must be positive")
            previous = [float(self.data.ctrl[i]) for i in self._arm_actuators]
            check_joints([math.degrees(value) for value in previous])
            target = [math.radians(value) for value in joints]
            difference = max(abs(math.degrees(end - start)) for start, end in zip(previous, target))
            dt = float(self.model.opt.timestep)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("invalid physics timestep")
            steps = max(1, math.ceil(duration / dt - 1e-12))
            ramp = min(self.ramp_seconds, duration)
            # The virtual student issues a 2 s full-ramp MOVE between confirmed
            # closed setpoints. Select the arm curve before updating the fingers.
            closed_profile_move = ((duration == 2. and ramp == 2.) or
                                   (getattr(self, '_demo_one_second_move', False) and duration == 1. and ramp == .5))
            closed_move = (opening == 0. and closed_profile_move and
                           all(float(self.data.ctrl[i]) == 0. for i in self._finger_actuators))
            # A preflighted path is attached to this simulator instance only,
            # never to MuJoCo's process-global step function. Other simulators
            # and every ordinary command retain their existing interpolation.
            descent = getattr(self, '_checked_straight_descent', None)
            path_cancel = getattr(self, '_checked_descent_cancel', None)
            if descent is not None:
                from .straight_descent import CheckedDescent, TICKS, TIMESTEP
                if (type(descent) is not CheckedDescent or not closed_move or dt != TIMESTEP
                        or steps != TICKS or len(descent.targets_rad) != steps
                        or any(abs(a-b) > 1e-7 for a, b in zip(
                            map(math.degrees, previous), descent.start, strict=True))
                        or any(abs(a-b) > 1e-7 for a, b in zip(joints, descent.target, strict=True))):
                    raise ValueError('checked virtual descent does not match this command')

            def cancelled():
                return ((cancel is not None and cancel.is_set()) or
                        (path_cancel is not None and path_cancel.is_set()))

            arm_shape = self.closed_move_profile if closed_move else self.ramp_shape
            finger_start = None
            if self.finger_open_profile == FINGER_OPEN_HALF_SECOND and opening == FULL_OPENING_M:
                left, right = (float(self.data.ctrl[i]) for i in self._finger_actuators)
                if not (0. <= left <= MAX_OPENING_M / 2 and 0. <= right <= MAX_OPENING_M / 2 and abs(left - right) <= 1e-9):
                    raise ValueError("invalid previous virtual finger targets")
                if left + right < opening:
                    if duration != 1.:
                        raise ValueError("gradual full finger opening requires a one-second OPEN command")
                    finger_start = left + right
            if finger_start is None:
                if cancelled():
                    raise InterruptedError("virtual command cancelled before gripper target")
                for actuator in self._finger_actuators:
                    self.data.ctrl[actuator] = opening / 2
            for step in range(steps):
                if cancelled():
                    raise InterruptedError("virtual command cancelled during motion")
                alpha = (min((step + 1) * dt / ramp, 1.0) if difference > self.threshold_degrees else 1.0)
                if alpha < 1.0:
                    if arm_shape == CLOSED_MOVE_MINIMUM_JERK:
                        alpha = _minimum_jerk(alpha)
                    elif arm_shape == CLOSED_MOVE_SMOOTHSTEP:
                        alpha = _smoothstep(alpha)
                for index, (actuator, start, end) in enumerate(zip(self._arm_actuators, previous, target)):
                    self.data.ctrl[actuator] = (descent.targets_rad[step][index] if descent is not None
                                                else end if alpha == 1.0 else start + (end - start) * alpha)
                if finger_start is not None:
                    progress = (step + 1) * dt / FINGER_OPEN_RAMP_SECONDS
                    gap = opening if progress >= 1. else finger_start + (opening - finger_start) * _minimum_jerk(progress)
                    for actuator in self._finger_actuators:
                        self.data.ctrl[actuator] = gap / 2
                if cancelled():  # demo-step-cancel-boundary
                    raise InterruptedError("virtual command cancelled before physics step")
                mujoco.mj_step(self.model, self.data)
            if cancelled():
                raise InterruptedError("virtual command cancelled after motion")
            mujoco.mj_forward(self.model, self.data)
            if cancelled():
                raise InterruptedError("virtual command cancelled before result")
            return self.observation()
        except Exception as exc:
            self.mark_layout_uncertain(f"simulation_command_error: {type(exc).__name__}: {exc}")
            raise
