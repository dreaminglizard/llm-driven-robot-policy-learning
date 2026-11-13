"""Optional observation and one-command target preview for the demo only.

The original simulator command remains the only code that changes physics.
The trace runs on its own calling thread at the existing physics-step boundary;
it reads public robot state only, never object/contact state. It is not a
real-device trajectory, a collision check, or a full-task prediction.
"""
from __future__ import annotations

import inspect
import math
import sys

from raccoon_control.kinematics import Geometry, check_joints
from .scene_ramped_motion_sim import RampedSceneMotionSim


def remaining_target_path(start, target, elapsed, ramp, *, samples=25, cartesian=False):
    """Nominal FK of the remaining minimum-jerk *servo setpoint* path.

    It starts at the last applied servo target, which need not equal the
    measured joints. The actual simulated joints may lag these setpoints.
    """
    start = check_joints(start)
    target = check_joints(target)
    if (not math.isfinite(elapsed) or not math.isfinite(ramp)
            or elapsed < 0 or ramp <= 0 or samples < 2):
        raise ValueError('invalid command preview interval')
    if max(abs(a-b) for a, b in zip(start, target)) <= 1e-7 or elapsed >= ramp:
        return []
    fraction = min(elapsed / ramp, 1.)
    alpha = 10*fraction**3 - 15*fraction**4 + 6*fraction**5
    geometry = Geometry()
    if cartesian:
        p0, p1 = geometry.forward(start), geometry.forward(target)
        return [[a+(b-a)*index/(samples-1) for a, b in zip(p0, p1, strict=True)]
                for index in range(samples)]
    result = []
    for index in range(samples):
        t = alpha + (1-alpha) * index / (samples-1)
        q = [a+(b-a)*t for a, b in zip(start, target)]
        result.append(list(geometry.forward(q)))
    return result


def _step_line():
    lines, first = inspect.getsourcelines(RampedSceneMotionSim.command)
    matches = [first + i for i, line in enumerate(lines) if '# demo-step-cancel-boundary' in line]
    return matches[0] if len(matches) == 1 else None


def attach_motion_monitor(sim, emit, cancelled, *, pause_per_sample=0., command_hook=None, session_token=None,
                          request_owner=None):
    """Instrument one app-owned simulator instance without modifying its class.

    Existing debuggers/coverage tracers take precedence: the UI then receives
    boundary observations only. Pacing changes wall-clock display speed, not
    the MuJoCo time step, control targets, or command duration.
    """
    original = sim.command
    code = RampedSceneMotionSim.command.__code__
    try:
        step_line = _step_line()
    except (OSError, IOError, TypeError):
        step_line = None
    sequence = 0

    def command(target, opening, seconds=.5):
        nonlocal sequence
        sequence += 1
        number = sequence
        request_id = request_owner() if request_owner is not None else None
        started = float(sim.data.time)
        ramp = min(float(sim.ramp_seconds), float(seconds))
        start = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
        start_opening = sum(float(sim.data.ctrl[i]) for i in sim._finger_actuators)
        target = list(target)
        moving = max(abs(a-b) for a, b in zip(start, target)) > 1e-7
        finger_delta = float(opening) - start_opening
        finger_change = abs(finger_delta) > 1e-7
        command_kind = ('arm_and_gripper' if moving and finger_change else 'arm_move' if moving
                        else 'open_gripper' if finger_delta > 1e-7
                        else 'close_gripper' if finger_delta < -1e-7 else 'hold')
        old_trace = sys.gettrace()
        can_sample = old_trace is None and step_line is not None
        period = max(1, round(.1 / float(sim.model.opt.timestep)))

        def publish(phase, elapsed, observation=None):
            # A UI exception must never alter the student's physics or result.
            try:
                current = dict(sim.observation() if observation is None else observation)
                # During mj_step, site transforms may lag integrated qpos by
                # one physics tick. Use the same nominal FK as the preview so
                # the displayed TCP is consistent with the sampled joints.
                current['tcp_mm'] = list(Geometry().forward(current['joints_deg']))
                current['tcp_source'] = 'nominal_fk_from_observed_virtual_joints'
                # The servo target is already advanced before the physics-step
                # trace fires.  Read that target directly so the remaining
                # geometry starts at the applied value for either arm curve,
                # instead of reconstructing it with a minimum-jerk clock.
                applied = [math.degrees(float(sim.data.ctrl[i])) for i in sim._arm_actuators]
                preview = (remaining_target_path(applied, target, 0., ramp,
                           cartesian=getattr(sim, '_checked_straight_descent', None) is not None)
                           if moving and phase in ('start', 'progress') and not cancelled.is_set() else [])
                emit('motion', {'phase': phase, 'number': number, 'elapsed': max(0., elapsed),
                     'forecast_session': session_token,
                     'request_id': request_id,
                     'duration': float(seconds), 'ramp': ramp, 'target_joints_deg': target,
                     'target_opening_m': float(opening), 'observation': current,
                     'remaining_target_path_mm': preview, 'arm_move': moving, 'command_kind': command_kind,
                     'sampling': 'during_command' if can_sample else 'boundaries_only',
                     'cancel_requested': cancelled.is_set()})
            except Exception:
                pass

        def local_trace(frame, event, arg):
            if event == 'line' and frame.f_lineno == step_line:
                step = frame.f_locals.get('step')
                if isinstance(step, int) and step % period == 0:
                    if command_hook is not None:
                        try:
                            command_hook('progress', number, target, opening)
                        except Exception:
                            pass
                    publish('progress', float(sim.data.time) - started)
                    if pause_per_sample > 0:
                        cancelled.wait(pause_per_sample)
            return local_trace

        def trace(frame, event, arg):
            if event == 'call' and frame.f_code is code:
                return local_trace
            return None

        publish('start', 0.)
        if command_hook is not None:
            try:
                command_hook('start', number, target, opening)
            except Exception:
                pass
        try:
            if can_sample:
                sys.settrace(trace)
            result = original(target, opening, seconds=seconds, cancel=cancelled)
        except InterruptedError:
            # Preserve a final robot-only sample, but never label a partially
            # applied command as complete or physical state as confirmed.
            publish('interrupted', float(sim.data.time) - started)
            raise
        except BaseException:
            # No end observation or expected path is promoted to success.
            try:
                emit('motion_error', {'number': number})
            except Exception:
                pass
            raise
        finally:
            if can_sample:
                sys.settrace(old_trace)
            if command_hook is not None:
                try:
                    command_hook('end', number, target, opening)
                except Exception:
                    pass
        publish('end', float(sim.data.time)-started, result)
        return result

    sim.command = command
