"""Conditional, independent look-ahead for the virtual demo.

The copy starts from expected symbolic state, public robot observations and a
separate snapshot of the learned memory. It never reads live object state,
contacts, checkpoints or future observations. Its output is illustrative;
it cannot issue live commands or establish success.
"""
from __future__ import annotations

import copy
import math
import threading


# At most one forecast worker may retain a shadow scene across registrations.
# A stuck native/model call cannot be killed safely from a Python thread.
_FORECAST_GATE = threading.Lock()


def forecast_worker_active():
    """Report whether a worker, including one closing, still owns the slot."""
    return _FORECAST_GATE.locked()


def _clone_guard(model):
    """Copy bounded policy state and weights, excluding accumulated guard logs."""
    from raccoon_control.near_joint_limit_guard import GuardedPolicy
    from .native_approach_session import AppliedHistoryPolicy

    if type(model) is not GuardedPolicy or type(model.policy) is not AppliedHistoryPolicy:
        raise TypeError('unsupported virtual forecast policy')
    return GuardedPolicy(copy.deepcopy(model.policy))


def _snapshot_guard(model):
    """Copy only the bounded live policy state, never the live network."""
    from raccoon_control.near_joint_limit_guard import GuardedPolicy
    from .native_approach_session import AppliedHistoryPolicy

    if type(model) is not GuardedPolicy or type(model.policy) is not AppliedHistoryPolicy:
        raise TypeError('unsupported virtual forecast policy')
    policy = model.policy
    return copy.deepcopy({key: getattr(policy, key) for key in
                          ('current_target', 'previous_action', 'pending', 'last_context', 'committed', 'count')})


def _restore_guard(student, state):
    """Build a guard around the forecaster's private, read-only student."""
    from raccoon_control.near_joint_limit_guard import GuardedPolicy
    from .native_approach_session import AppliedHistoryPolicy

    policy = AppliedHistoryPolicy(student)
    for key, value in state.items():
        setattr(policy, key, copy.deepcopy(value))
    return GuardedPolicy(policy)


_MAX_FUTURE_STEPS = 16
_MAX_FUTURE_ROUNDS = 2
_MAX_FUTURE_COMMANDS = 120


def _future_schedule(snapshot):
    """Bound the admitted coordinator schedule after its current step."""
    captured = snapshot['program']
    plan, program = captured['plan'], captured['program']
    position = snapshot['position']
    index, current_round = position['step_index'], position['round']
    if type(index) is not int or type(current_round) is not int or current_round < 1:
        raise ValueError('invalid active program position')
    repeat_index = plan['repeat_index']
    initial = program['steps'] if repeat_index is None else program['steps'][:repeat_index]
    body = plan['body'] if repeat_index is not None else []
    initial_phase = repeat_index is None or (plan['prior_body'] and current_round == 1)
    active_steps = initial if initial_phase else body
    if index < 0 or index >= len(active_steps):
        raise ValueError('active step is outside the admitted program')
    source = captured['validation_source'] if initial_phase else plan['body_validation_source']
    schedule = [(copy.deepcopy(step), copy.deepcopy(source), current_round)
                for step in active_steps[index + 1:]]
    repeat_mode = 'none'
    more_rounds = 0
    if repeat_index is not None:
        repeat_mode = plan['count']['mode']
        if repeat_mode == 'continuous':
            more_rounds = _MAX_FUTURE_ROUNDS
        elif repeat_mode == 'total':
            count = plan['count']['value']
            if type(count) is not int or count < 0:
                raise ValueError('invalid admitted repeat count')
            completed_or_active = 0 if initial_phase else current_round - (1 if plan['prior_body'] else 0)
            more_rounds = max(0, count - completed_or_active)
        else:
            raise ValueError('unsupported admitted repeat count')
    limited = repeat_mode == 'continuous' or more_rounds > _MAX_FUTURE_ROUNDS
    for offset in range(min(more_rounds, _MAX_FUTURE_ROUNDS)):
        round_number = current_round + offset + 1
        schedule.extend((copy.deepcopy(step), copy.deepcopy(plan['body_validation_source']), round_number)
                        for step in body)
    # Preserve round ends before truncation so a partially calculated round
    # cannot be mistaken for a completed one.
    round_ends = {round_number: offset + 1
                  for offset, (_, _, round_number) in enumerate(schedule)}
    if len(schedule) > _MAX_FUTURE_STEPS:
        schedule = schedule[:_MAX_FUTURE_STEPS]
        limited = True
    through_round = schedule[-1][2] if schedule else current_round
    return {'steps': schedule, 'limited': limited, 'repeat_mode': repeat_mode,
            'through_round': through_round, 'round_ends': round_ends,
            'current_round_complete': index == len(active_steps) - 1}


class _Superseded:
    def __init__(self, owner, generation):
        self.owner, self.generation = owner, generation

    def is_set(self):
        with self.owner._condition:
            superseded = self.owner._closed or self.owner._generation != self.generation
        return superseded or self.owner._control_stopping()


class _PredictionCancellation:
    """A shadow-owned stop flag with read-only awareness of supersession."""
    def __init__(self, source):
        self._local = threading.Event()
        self._source = source

    def is_set(self):
        return self._local.is_set() or self._source.is_set()

    def set(self):
        self._local.set()


class DemoForecast:
    """Run one replaceable forecast at a time without owning the live session."""

    def __init__(self, sim, session, emit, cancelled, *, coordinator=None, session_token=None):
        self.sim, self.session, self.emit, self.cancelled = sim, session, emit, cancelled
        self.coordinator = coordinator
        self.session_token = session_token
        self._condition = threading.Condition()
        self._generation = 0
        self._job = None
        self._current_number = None
        self._current_ready = False
        self._cached = None
        self._event_sequence = 0
        self._closed = False
        self._plan = None
        self._plan_index = 0
        self._plan_memory = None
        self._transfer = None
        self._thread_program = threading.local()
        self._cleared_number = None
        self._thread = None
        self._student_template = None
        self._memory_template = None
        if coordinator is not None:
            try:
                # These immutable model copies are made at registration, not
                # while a live command or its stop request is running.
                self._student_template = copy.deepcopy(session.student.policy.student)
                self._memory_template = copy.deepcopy(session.memory.model)
            except Exception:
                self._student_template = self._memory_template = None

        # These are app-instance adapters around existing execution, not new
        # planning or control rules. The original bound methods remain in use.
        original_path = session._execute_path
        original_runner = session._unscored_transfer_runner

        def planned_path(path, *args, **kwargs):
            # The coordinator enters this callback only after checking its
            # expected memory stamp and while the current thread owns both
            # memory locks for the entire step. Record that baseline before
            # the first transfer; each committed transfer advances the GRU
            # once, even though the coordinator's stamp changes only at the
            # end of the step.
            plan_memory = None
            if coordinator is not None:
                try:
                    memory = session.memory
                    binding_owned = getattr(session._memory_binding_lock, '_is_owned', lambda: False)()
                    memory_owned = getattr(memory.synchronized(), '_is_owned', lambda: False)()
                    if binding_owned and memory_owned:
                        with coordinator._lock:
                            active = coordinator._active
                            stamp = coordinator._expected_memory_stamp
                            if active is not None and stamp == (id(memory), memory.state_version):
                                plan_memory = {'active': active, 'memory': memory, 'stamp': stamp,
                                               'thread': threading.get_ident(), 'event_start': len(session.events)}
                except Exception:
                    # Missing inspection support only disables the extended
                    # preview; it must never prevent the live task.
                    plan_memory = None
            with self._condition:
                self._plan = copy.deepcopy(path)
                self._plan_index = 0
                self._plan_memory = plan_memory
            try:
                return original_path(path, *args, **kwargs)
            finally:
                with self._condition:
                    self._plan = None
                    self._plan_memory = None
                    self._transfer = None
                self.invalidate()

        def transfer(model, live, before, task, **kwargs):
            with self._condition:
                planned = None
                if self._plan is not None and self._plan_index < len(self._plan):
                    item = self._plan[self._plan_index]
                    if (item.get('source_slot') == task.get('source_slot')
                            and item.get('destination_slot') == task.get('destination_slot')):
                        planned = copy.deepcopy(self._plan[self._plan_index:])
                    self._plan_index += 1
                self._transfer = {'model': model, 'before': copy.deepcopy(before),
                                  'task': copy.deepcopy(task), 'planned': planned,
                                  'max_decisions': kwargs['max_decisions']}
            try:
                return original_runner(model, live, before, task, **kwargs)
            finally:
                with self._condition:
                    self._transfer = None
                self.invalidate()

        session._execute_path = planned_path
        session._unscored_transfer_runner = transfer

        if (coordinator is not None and callable(getattr(coordinator, 'execute', None))
                and hasattr(coordinator, '_pending')):
            original_execute = coordinator.execute

            def execute(ticket, *, cancel_event=None):
                # Admission already validated this private ticket. A detached
                # copy is for prediction only and is never executed live.
                with coordinator._lock:
                    entry = coordinator._pending.get(ticket)
                captured = None
                if entry is not None and entry.get('plan', {}).get('kind') == 'work':
                    try:
                        captured = copy.deepcopy({key: entry[key] for key in
                            ('program', 'plan', 'validation_source', 'registration_scene')})
                    except Exception:
                        captured = None
                # Only the thread that actually enters this ticket can use
                # its copy. Simultaneously admitted tickets cannot overwrite
                # or withhold one another's prediction input.
                previous = getattr(self._thread_program, 'program', None)
                self._thread_program.program = captured
                try:
                    return original_execute(ticket, cancel_event=cancel_event)
                finally:
                    self._thread_program.program = previous

            coordinator.execute = execute

    def _capture_program(self):
        """Detach the admitted program, active position and learned memory."""
        from .scene_recall import SceneRecall

        if self.coordinator is None or self._memory_template is None:
            return None
        program = getattr(self._thread_program, 'program', None)
        with self._condition:
            plan_memory = self._plan_memory
            plan_index = self._plan_index
        if program is None or plan_memory is None or plan_memory['thread'] != threading.get_ident():
            return None
        coordinator = self.coordinator
        with coordinator._lock:
            active = coordinator._active
            if (active is None or active['request_id'] != program['program']['request_id']
                    or active['stop_requested'] or active['stop_repeat']
                    or active['generation'] != coordinator._generation
                    or type(active['step_index']) is not int):
                return None
            marker = (active['program_id'], active['generation'], active['round'], active['step_index'],
                      active['task_id'], active['transfer_id'])
            position = {'round': active['round'], 'step_index': active['step_index'],
                        'task_start_scene': copy.deepcopy(active['task_start_scene']),
                        'registered_initial': copy.deepcopy(coordinator._registered_initial),
                        'last_registered': copy.deepcopy(self.session._last_confirmed_scene),
                        'train_groups': frozenset(self.session.train_groups)}
            memory = self.session.memory
            stamp = coordinator._expected_memory_stamp
        if (type(memory) is not SceneRecall or memory is not plan_memory['memory']
                or active is not plan_memory['active'] or stamp != plan_memory['stamp']):
            return None
        # Never wait for another thread to release the memory locks, and do
        # not accept a version change merely because a step is in progress.
        # Only the current executor can own both reentrant locks here.
        if (not getattr(self.session._memory_binding_lock, '_is_owned', lambda: False)()
                or not getattr(memory.synchronized(), '_is_owned', lambda: False)()):
            return None
        # The live executor owns this reentrant memory lock throughout its
        # current step. No coordinator or preview lock is held during copies.
        with memory.synchronized():
            committed = sum(1 for event in self.session.events[plan_memory['event_start']:]
                            if event.get('event') == 'transfer' and event.get('committed'))
            version = stamp[1] + committed
            if committed != plan_index - 1 or (id(memory), memory.state_version) != (stamp[0], version):
                return None
            hidden = memory.hidden
            active_ids = memory.active_ids
        with coordinator._lock:
            current = coordinator._active
            if (current is not active or current['stop_requested'] or current['stop_repeat']
                    or marker != (current['program_id'], current['generation'], current['round'],
                                  current['step_index'], current['task_id'], current['transfer_id'])
                    or coordinator._generation != marker[1]
                    or coordinator._expected_memory_stamp != stamp
                    or self.session.memory is not memory or memory.state_version != version):
                return None
        return {'program': copy.deepcopy(program), 'position': position,
                'memory_model': self._memory_template, 'memory_hidden': hidden,
                'memory_active_ids': active_ids}

    def command_started(self, number, target, opening):
        """Capture only public inputs while the live command thread owns them."""
        from .scene_oracle_free_session import _servo_targets

        if self._control_stopping():
            return
        try:
            with self._condition:
                context = self._transfer
                if self._closed or context is None:
                    return
                self._cleared_number = None
                self._current_number = number
                self._current_ready = False
                generation = self._generation
                live_model = context['model']
                before = copy.deepcopy(context['before'])
                task = copy.deepcopy(context['task'])
                planned = copy.deepcopy(context['planned'])
                maximum = context['max_decisions']
            # Never keep the cancellation lock while copying weights. The
            # live command thread owns this policy until this callback returns.
            # Copy only the bounded policy history, not growing guard logs.
            if self._student_template is not None:
                model = None
                model_state = _snapshot_guard(live_model)
                pending = model_state['pending']
            else:
                model = _clone_guard(live_model)
                model_state = None
                pending = model.policy.pending
            if pending is None or self._control_stopping():
                return
            try:
                program = self._capture_program()
            except Exception:
                # An incomplete snapshot never invents later steps.
                program = None
            if self._memory_template is not None and program is None:
                # A failed consistency check also retires any older compound
                # prediction for this transfer; never reuse it after a
                # memory/reference mismatch or stop.
                with self._condition:
                    if generation != self._generation or self._closed:
                        return
                    self._generation += 1
                    generation = self._generation
                    self._job = None
                    self._cached = None
            observation = self.sim.observation()
            targets = _servo_targets(self.sim)
            job = {'number': number, 'observation': observation, 'targets': targets,
                   'model': model, 'model_snapshot': model_state, 'student_template': self._student_template,
                   'before': before, 'task': task, 'planned': planned,
                   'max_decisions': maximum, 'target': list(target), 'opening': float(opening),
                   'slots': [list(point) for point in self.sim.slot_xy_mm],
                   'finger_profile': self.sim.finger_open_profile,
                   'move_profile': self.sim.closed_move_profile,
                   'control_timing_profile': getattr(self.sim, '_control_timing_profile', 'app_2s'),
                   'straight_descent': getattr(self.sim, '_straight_descent_enabled', False),
                   'support_height_guard_enabled': getattr(self.sim, '_support_height_guard_enabled', False),
                   'support_height_guard': copy.deepcopy(getattr(self.sim, '_active_support_height_guard', None)),
                   'program_snapshot': program}
            skipped = False
            reuse = None
            with self._condition:
                if self._closed or self.cancelled.is_set() or generation != self._generation:
                    return
                self._job = (self._generation, job)
                self._current_ready = True
                if self._thread is None:
                    if not _FORECAST_GATE.acquire(blocking=False):
                        self._job = None
                        skipped = True
                    else:
                        try:
                            self._thread = threading.Thread(target=self._work, daemon=True, name='demo-forecast')
                            self._thread.start()
                        except BaseException:
                            self._thread = None
                            self._job = None
                            _FORECAST_GATE.release()
                            raise
                if not skipped and self._cached is not None:
                    cached, basis = self._cached
                    reuse = self._event_locked(cached, basis, number)
            if skipped:
                self.emit('forecast_skipped', {'command_number': number, 'forecast_session': self.session_token})
            elif reuse is not None and not self._control_stopping():
                self.emit('forecast', reuse)
        except Exception:
            # Preview failure cannot change the live command or its outcome.
            self.invalidate()

    def _control_stopping(self):
        if self.cancelled.is_set() or (getattr(self.session, 'cancel', None) is not None
                                       and self.session.cancel.is_set()):
            return True
        if self.coordinator is not None:
            with self.coordinator._lock:
                active = self.coordinator._active
                return bool(active and (active['stop_requested'] or active['stop_repeat']))
        return False

    def command_progress(self, number):
        # Repeat-stop and voice stop use the coordinator's own control state.
        # Stop showing the old plan even when the app stop Event was not set.
        if not self._control_stopping():
            return
        with self._condition:
            if self._cleared_number == number:
                return
            self._cleared_number = number
        self.invalidate()
        try:
            self.emit('forecast_clear', {'command_number': number, 'forecast_session': self.session_token})
        except Exception:
            pass

    def invalidate(self):
        with self._condition:
            self._generation += 1
            self._job = None
            self._current_number = None
            self._current_ready = False
            self._cached = None

    def command_finished(self, number):
        """Clear the active command, but let the same transfer finish predicting."""
        with self._condition:
            if self._current_number == number:
                self._current_number = None
                self._current_ready = False

    def _event_locked(self, value, basis, current):
        self._event_sequence += 1
        result = copy.deepcopy(value)
        result.update(command_number=current, forecast_session=self.session_token,
                      basis_command_number=basis, forecast_sequence=self._event_sequence,
                      stale=basis != current)
        return result

    def close(self, *, wait=False):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._job = None
            self._current_number = None
            self._current_ready = False
            self._cached = None
            thread = self._thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.)
        return thread is None or not thread.is_alive()

    def _work(self):
        released = False
        try:
            while True:
                with self._condition:
                    if self._closed or self._job is None:
                        self._thread = None
                        _FORECAST_GATE.release()
                        released = True
                        return
                    generation, job = self._job
                    self._job = None
                cancelled = _Superseded(self, generation)

                def publish(value):
                    if cancelled.is_set():
                        return
                    result = copy.deepcopy(value)
                    result.setdefault('scope', 'planned_task' if job['planned'] is not None else 'current_transfer')
                    result.setdefault('transfers', len(job['planned']) if job['planned'] is not None else 1)
                    with self._condition:
                        if self._closed or self._generation != generation:
                            return
                        # Keep a completed whole-program prediction visible
                        # while a newer job is still publishing only its
                        # current-task prefix. A later full or partial result
                        # replaces it, including an explicit failure.
                        previous = self._cached[0] if self._cached is not None else None
                        prefix = result['scope'] in ('planned_task', 'current_transfer')
                        if (previous and previous.get('scope') == 'program' and prefix
                                and result.get('reason') in ('computing', 'model_done')):
                            return
                        current = self._current_number if self._current_ready else None
                        if result.get('reason') == 'computing':
                            # A short prefix may already be in the past once
                            # its basis command ends. Do not cache or reuse it.
                            if current != job['number']:
                                return
                        else:
                            self._cached = (result, job['number'])
                        event = self._event_locked(result, job['number'], current) if current is not None else None
                    if event is None or cancelled.is_set():
                        return
                    try:
                        self.emit('forecast', event)
                    except Exception:
                        pass

                job['_emit_prefix'] = publish
                try:
                    result = _forecast(job, cancelled)
                except Exception:
                    result = {'segments': [], 'complete': False, 'reason': 'unavailable'}
                publish(result)
        finally:
            if not released:
                with self._condition:
                    self._thread = None
                    self._job = None
                _FORECAST_GATE.release()


def _forecast(job, cancelled):
    """Roll detached policy and memory in an independently constructed scene."""
    import mujoco
    import torch
    from . import (motion_policy, native_approach_session as native, scene_features, scene_recall,
                   scene_program_coordinator as coordinator, task_catalog)
    from . import scene_oracle_free_session as runtime
    from .demo_motion import remaining_target_path

    shadow_cancel = _PredictionCancellation(cancelled)
    if shadow_cancel.is_set():
        raise InterruptedError('forecast superseded')
    snapshot = job.get('program_snapshot') if job.get('planned') is not None else None
    schedule = None
    if snapshot is not None:
        try:
            schedule = _future_schedule(snapshot)
        except Exception:
            snapshot = None
    memory = None
    if snapshot is not None:
        memory = scene_recall.SceneRecall(snapshot['memory_model'])
        memory.hidden = snapshot['memory_hidden']
        memory.active_ids = snapshot['memory_active_ids']
    observation, applied = job['observation'], copy.deepcopy(job['targets'])
    shadow = native.PublicCustomSlotsSceneMotionSim(
        job['before'], seed=0, randomize=False, slot_xy_mm=job['slots'],
        ramp_seconds=.5, threshold_degrees=0., ramp_shape='minimum_jerk',
        finger_open_profile=job['finger_profile'], closed_move_profile=job['move_profile'])
    shadow._straight_descent_enabled = bool(job.get('straight_descent', False))
    shadow._control_timing_profile = job.get('control_timing_profile', 'app_2s')
    shadow._support_height_guard_enabled = bool(job.get('support_height_guard_enabled', False))
    height_guard = copy.deepcopy(job.get('support_height_guard'))
    # Unobserved velocities are assumed zero and finger positions symmetric.
    # No live MuJoCo object or contact state is copied into this scene.
    for joint, actuator, angle, target in zip(shadow._joint_ids, shadow._arm_actuators,
                                              observation['joints_deg'], applied['joints_deg']):
        shadow.data.qpos[shadow.model.jnt_qposadr[joint]] = math.radians(float(angle))
        shadow.data.ctrl[actuator] = math.radians(float(target))
    for joint, actuator in zip(shadow._finger_ids, shadow._finger_actuators):
        shadow.data.qpos[shadow.model.jnt_qposadr[joint]] = float(observation['gripper_opening_m']) / 2
        shadow.data.ctrl[actuator] = applied['gripper_opening_m'] / 2
    mujoco.mj_forward(shadow.model, shadow.data)
    before, task = scene_features.transfer_task(job['before'], job['task'])
    shadow.begin_transfer(before, task['source_slot'], task['destination_slot'])
    guard = (_restore_guard(job['student_template'], job['model_snapshot'])
             if job.get('model_snapshot') is not None else job['model'])
    pending = guard.policy.pending
    history = [list(row) for row, valid in zip(pending['window'][0], pending['mask'][0]) if valid]
    if not history:
        raise ValueError('missing public policy history')
    row = history[-1]
    gripper = 'open' if row[7] else 'closed' if row[8] else None
    action = motion_policy.ACTIONS[pending['predicted_action']]
    target, opening = job['target'], job['opening']
    if shadow_cancel.is_set():
        raise InterruptedError('forecast superseded')
    _, last = native._apply(shadow, target, opening, action, cancel=shadow_cancel)
    guard.on_control_applied(action, target, last, opening)
    applied.update(joints_deg=list(last), gripper_opening_m=opening)
    if action in ('open', 'close'):
        gripper = 'open' if action == 'open' else 'closed'
    segments = []

    def add_segment(start, end, *, cartesian=False):
        path = remaining_target_path(start, end, 0., 1., samples=12, cartesian=cartesian)
        if path:
            segments.append(path)

    # Publish a growing, explicitly incomplete prefix during the current
    # command. The first moving segment is followed by at most one update per
    # three additional segments, limiting copies on the prediction worker.
    last_partial_size = 0
    termination = 'decision_limit'
    for _ in range(job['max_decisions'] - guard.policy.count):
        if shadow_cancel.is_set():
            raise InterruptedError('forecast superseded')
        robot_observation = runtime._robot_observation(shadow, gripper)
        features = scene_features.encode_scene_features(robot_observation, before, task, job['slots'])
        history.append(features)
        window, mask = scene_features.history_window(history, 8)
        with torch.inference_mode():
            logits, joints = guard.forward_with_control_state(torch.tensor([window], dtype=torch.float32),
                torch.tensor([mask], dtype=torch.bool), torch.tensor([last], dtype=torch.float32))
        action = motion_policy.ACTIONS[int(logits.argmax(-1).item())]
        if action == 'done':
            termination = 'model_done'
            break
        target = [float(v) * 180. for v in joints[0].tolist()] if action == 'move' else list(last)
        if not runtime._in_range(target):
            termination = 'out_of_range'
            break
        if height_guard is not None:
            target, _ = height_guard.apply(action=action, proposed_target_deg=target,
                                           previous_target_deg=last, gripper_opening_m=opening,
                                           observed_tcp_mm=robot_observation['tcp_mm'])
        opening = (.04 if action == 'open' else 0.) if action in ('open', 'close') else opening
        timing, actual = native._apply(shadow, target, opening, action, cancel=shadow_cancel)
        add_segment(last, target, cartesian=timing.get('cartesian', {}).get('changed', False))
        if (len(segments) > last_partial_size
                and (last_partial_size == 0 or len(segments) - last_partial_size >= 3)
                and callable(job.get('_emit_prefix'))):
            if shadow_cancel.is_set():
                raise InterruptedError('forecast superseded')
            job['_emit_prefix']({'segments': segments, 'complete': False, 'reason': 'computing',
                                 'scope': 'current_transfer', 'transfers': 1})
            last_partial_size = len(segments)
        guard.on_control_applied(action, target, actual, opening)
        last = list(actual)
        applied.update(joints_deg=list(actual), gripper_opening_m=opening)
        if action in ('open', 'close'):
            gripper = 'open' if action == 'open' else 'closed'

    remaining = (job['planned'] or [job['task']])[1:]
    for next_task in remaining:
        if termination != 'model_done':
            break
        if shadow_cancel.is_set():
            raise InterruptedError('forecast superseded')
        runtime._hold_one_second(shadow, shadow_cancel)
        # Commit only an expected symbolic scene in the separate simulation.
        before = task_catalog.transfer(before, task['source_slot'], task['destination_slot'])
        shadow._committed_layout = before
        shadow._pending_transfer = None
        shadow.layout_status = 'registered'
        if memory is not None:
            memory.observe(before, write=False)
        task = next_task
        original = shadow.command

        def collect(target, next_opening, seconds=.5):
            if shadow_cancel.is_set():
                raise InterruptedError('forecast superseded')
            start = runtime._servo_targets(shadow)['joints_deg']
            cartesian = getattr(shadow, '_checked_straight_descent', None) is not None
            result = original(target, next_opening, seconds=seconds)
            add_segment(start, target, cartesian=cartesian)
            return result

        shadow.command = collect
        try:
            result = native.run_native_transfer(guard, shadow, before, task, max_decisions=job['max_decisions'],
                device='cpu', cancel=shadow_cancel, applied_targets=applied, slot_xy_mm=job['slots'])
        finally:
            shadow.command = original
        termination = result['termination']
    current = {'segments': segments, 'complete': termination == 'model_done', 'reason': termination}
    if snapshot is None or termination != 'model_done':
        return current

    # Show the already-calculated current task while later steps are still
    # being simulated; the worker publishes detached copies of the points.
    if schedule['steps'] and callable(job.get('_emit_prefix')):
        job['_emit_prefix'](current)
    if shadow_cancel.is_set():
        raise InterruptedError('forecast superseded')
    runtime._hold_one_second(shadow, shadow_cancel)
    before = task_catalog.transfer(before, task['source_slot'], task['destination_slot'])
    shadow._committed_layout = before
    shadow._pending_transfer = None
    shadow.layout_status = 'registered'
    shadow.layout_uncertainty_reason = None
    memory.observe(before, write=False)
    position = snapshot['position']

    if not schedule['steps']:
        return {'segments': segments, 'complete': not schedule['limited'],
                'reason': 'horizon' if schedule['limited'] else 'model_done',
                'scope': 'program_horizon' if schedule['limited'] else 'program',
                'through_round': schedule['through_round'], 'repeat_mode': schedule['repeat_mode'],
                'completed_round': (position['round'] if schedule['current_round_complete']
                                    else position['round'] - 1), 'completed_steps': 0}

    shadow_session = runtime.OracleFreeSceneModelSession(
        shadow, memory, guard, device='cpu', max_decisions=job['max_decisions'],
        cancel=shadow_cancel, slot_xy_mm=job['slots'], unscored_transfer_runner=native.run_native_transfer)
    shadow_session.train_groups = position['train_groups']
    shadow_session.current = before
    shadow_session.status = 'registered'
    shadow_session._applied_targets = applied
    shadow_session._last_confirmed_scene = position['last_registered']
    shadow_session._current_confirmed_scene = None
    previous_task_start = position['task_start_scene']
    current_round = position['round']
    round_transfers = 1  # The current transfer is part of this round.
    commands = 0
    completed_steps = 0
    through_round = current_round
    completed_round = current_round if schedule['current_round_complete'] else current_round - 1
    original = shadow.command

    class HorizonReached(Exception):
        pass

    def collect(target, next_opening, seconds=.5):
        nonlocal commands, through_round
        if shadow_cancel.is_set():
            raise InterruptedError('forecast superseded')
        if commands >= _MAX_FUTURE_COMMANDS:
            raise HorizonReached('forecast command limit')
        commands += 1
        start = runtime._servo_targets(shadow)['joints_deg']
        cartesian = getattr(shadow, '_checked_straight_descent', None) is not None
        result = original(target, next_opening, seconds=seconds)
        add_segment(start, target, cartesian=cartesian)
        through_round = current_round
        return result

    def before_transfer():
        nonlocal round_transfers
        if shadow_cancel.is_set():
            raise InterruptedError('forecast superseded')
        round_transfers += 1
        return True

    reason = 'model_done'
    limited = schedule['limited']
    shadow.command = collect
    try:
        for offset, (step, validation_source, round_number) in enumerate(schedule['steps'], 1):
            if shadow_cancel.is_set():
                raise InterruptedError('forecast superseded')
            if round_number != current_round:
                if round_transfers == 0:
                    reason, limited = 'no_progress', False
                    break
                current_round, round_transfers = round_number, 0
            if (coordinator._requires_registered_top_scene(step, validation_source)
                    and not coordinator._registered_top_scene(shadow_session, shadow_session.current)):
                reason, limited = 'future_unavailable', True
                break
            if not coordinator._unique_unspecified_target(step, shadow_session.current,
                                                           validation_source=validation_source):
                reason, limited = 'future_unavailable', True
                break
            step_start = copy.deepcopy(shadow_session.current)
            shadow_session._execute_validated_step(
                step, registered_initial=position['registered_initial'], task_start=previous_task_start,
                before_transfer=before_transfer, planned_tasks=True, validation_source=validation_source)
            previous_task_start = step_start
            completed_steps += 1
            through_round = round_number
            if schedule['round_ends'][round_number] == offset:
                completed_round = round_number
        else:
            if schedule['repeat_mode'] != 'none' and round_transfers == 0:
                reason, limited = 'no_progress', False
    except HorizonReached:
        reason, limited = 'horizon', True
    except Exception:
        if cancelled.is_set():
            raise InterruptedError('forecast superseded')
        reason, limited = 'future_unavailable', True
    finally:
        shadow.command = original
    if shadow_cancel.is_set():
        raise InterruptedError('forecast superseded')
    if limited and reason == 'model_done':
        reason = 'horizon'
    return {'segments': segments, 'complete': not limited, 'reason': reason,
            'scope': 'program_horizon' if limited else 'program', 'through_round': through_round,
            'completed_round': completed_round, 'repeat_mode': schedule['repeat_mode'],
            'completed_steps': completed_steps}
