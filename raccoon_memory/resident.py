"""Use an existing controller without owning its device connection lifetime."""
import time

from raccoon_control.errors import ControlError
from raccoon_control.blended_execution import virtual_blend_time_budget
from raccoon_control.transport import Client


def require_ready_after_start(status, *, allow_restored_connection_fault=False):
    """A previously ready child reporting a fault or lost feedback is not waiting to start."""
    if not isinstance(status, dict) or 'fault' not in status:
        raise ControlError('APP_CONTROLLER_STATUS_INVALID', '시작한 제어기의 상태를 확인할 수 없습니다.')
    fault = status['fault']
    restored = (allow_restored_connection_fault and status.get('restored_connection_fault') is True
                and isinstance(fault, dict) and fault.get('code') in ('DISCONNECTED', 'SENSOR_STALE'))
    if fault is not None and not restored:
        detail = fault.get('message') if isinstance(fault, dict) else None
        message = '시작한 제어기에 오류가 발생했습니다.'
        if isinstance(detail, str) and detail:
            message += ' ' + detail
        raise ControlError('APP_CONTROLLER_START_FAULT', message, fault=fault)
    worker, feedback = status.get('worker_alive'), status.get('fresh_feedback')
    if type(worker) is not bool or type(feedback) is not bool:
        raise ControlError('APP_CONTROLLER_STATUS_INVALID', '시작한 제어기의 작업자 또는 수신 상태가 불명확합니다.')
    if not worker:
        raise ControlError('APP_CONTROLLER_WORKER_LOST', '시작한 제어기의 장치 작업자가 종료됐습니다.')
    if not feedback:
        raise ControlError('APP_CONTROLLER_FEEDBACK_LOST', '시작한 제어기의 장치 수신이 끊겼습니다.')


class ResidentController:
    def __init__(self,config,runtime):
        self.config=config
        # A legal 30-second move must not lose its reply at the client's old
        # 20-second default. The controller still enforces its own deadlines.
        self.client=Client(runtime,timeout=config.limits.motion_timeout+config.limits.stop_timeout+5)
        self.instance_id=None
        self._untouched_start_revision=None
        self._untouched_start_lease=None

    def start(self, *, pending_start_instance=None, allow_restored_connection_fault=False):
        try:
            status=self.client.call("status")
        except ControlError as exc:
            if pending_start_instance is not None and exc.code == 'NOT_READY':
                raise ControlError('APP_CONTROLLER_STATUS_INVALID', '시작한 제어기의 상태 응답이 불명확합니다.') from exc
            raise
        if pending_start_instance is not None and not isinstance(status, dict):
            raise ControlError('APP_CONTROLLER_STATUS_INVALID', '시작한 제어기의 상태 응답이 불명확합니다.')
        if status.get("configuration_sha256")!=self.config.fingerprint():
            raise ControlError("CONTROLLER_CONFIG_MISMATCH","실행 중인 제어기와 경로 계산 설정이 다릅니다.")
        if status.get('resident_api_revision')!=2:
            raise ControlError('CONTROLLER_UPGRADE_REQUIRED',
                '상주 제어기가 이전 판본입니다. 연결을 유지했습니다. 현장에서 정지·종료 후 새 제어기로 시작해 주세요.')
        if pending_start_instance is not None:
            if status.get('instance_id') != pending_start_instance:
                raise ControlError('CONTROLLER_CHANGED', '시작한 제어기와 응답한 제어기가 다릅니다.')
            require_ready_after_start(status, allow_restored_connection_fault=allow_restored_connection_fault)
        if not status["worker_alive"] or not status["fresh_feedback"]:
            raise ControlError("NOT_READY","상주 제어기의 새 장치 수신을 확인하지 못했습니다.")
        self.instance_id=status["instance_id"]
        return status

    def acquire(self,owner):
        if self.instance_id is None:
            raise ControlError("NOT_READY","상주 제어기 설정을 먼저 확인해야 합니다.")
        revision, self._untouched_start_revision = self._untouched_start_revision, None
        if revision is not None:
            result = self.client.call("acquire",owner=owner,expected_revision=revision,
                                      expected_instance_id=self.instance_id)
            if revision == 0 and isinstance(result, dict):
                self._untouched_start_lease = result.get('lease_id')
            return result
        return self.client.call("acquire",owner=owner)

    def status(self):
        status=self.client.call("status")
        if self.instance_id is not None and status["instance_id"]!=self.instance_id:
            raise ControlError("CONTROLLER_CHANGED","실행 중 상주 제어기가 바뀌었습니다.")
        return status

    def display_status(self):
        """Bound UI polling without shortening motion-command RPC timeouts."""
        status=self.client.call_with_timeout("status", timeout=2)
        if self.instance_id is not None and status["instance_id"] != self.instance_id:
            raise ControlError("CONTROLLER_CHANGED", "실행 중 상주 제어기가 바뀌었습니다.")
        return status

    def observe(self,lease_id=None):
        observation=self.client.call("observe",**({"lease_id":lease_id} if lease_id else {}))
        if observation["instance_id"]!=self.instance_id:
            raise ControlError("CONTROLLER_CHANGED","실행 중 상주 제어기가 바뀌었습니다.")
        return observation

    def apply_action(self,request):
        action = request.get('action') if isinstance(request, dict) else None
        try:
            if isinstance(action, dict) and action.get('kind') == 'blended_joint_path':
                waypoints = action.get('waypoints_deg')
                if isinstance(waypoints, list) and 1 <= len(waypoints) <= 64:
                    if getattr(self.config, 'backend', None) == 'sdk':
                        # A lost reply is resolved by the admitted-command
                        # status below, not a virtual waypoint time estimate.
                        timeout = self.client.timeout
                    else:
                        # Preserve the virtual path's aggregate HTTP budget.
                        budget = virtual_blend_time_budget(self.config.limits.motion_timeout, len(waypoints))
                        timeout = max(self.client.timeout, budget + self.config.limits.stop_timeout + 5.)
                    return self.client.call_with_timeout('apply_action', timeout=timeout, **request)
            return self.client.call("apply_action",**request)
        except ControlError as exc:
            if (exc.code != 'CONTROLLER_UNREACHABLE' or getattr(self.config, 'backend', None) != 'sdk'
                    or not isinstance(action, dict) or action.get('kind') not in ('move_joints', 'blended_joint_path')
                    or not isinstance(request.get('command_id'), str) or not self.instance_id):
                raise
            return self._await_admitted_result(request['command_id'], self.instance_id, exc)

    def _await_admitted_result(self, command_id, instance_id, original_error):
        """Recover only an admitted command; never submit the action again."""
        while True:
            try:
                status = self.status()
                if not isinstance(status, dict) or status.get('instance_id') != instance_id:
                    raise ControlError('CONTROLLER_CHANGED', '실행 중 상주 제어기가 바뀌었습니다.')
                saved = self.client.call('result', command_id=command_id)
            except ControlError as exc:
                if exc.code == 'CONTROLLER_CHANGED':
                    raise
                raise original_error from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise original_error from exc

            if (self.instance_id != instance_id or not isinstance(saved, dict)
                    or saved.get('command_id') != command_id):
                raise original_error
            active = status.get('active_command_id')
            if active not in (None, command_id):
                raise original_error
            outcome = saved.get('status')
            if outcome == 'outcome_unknown_or_in_progress':
                fault = status.get('fault')
                if isinstance(fault, dict):
                    raise ControlError(fault.get('code', 'CONTROLLER_ERROR'),
                                       fault.get('message', '제어기 오류가 발생했습니다.'),
                                       **(fault.get('details') if isinstance(fault.get('details'), dict) else {}))
                if (active != command_id or status.get('worker_alive') is not True
                        or status.get('fresh_feedback') is not True):
                    raise original_error
                time.sleep(.2)
                continue

            observation = saved.get('observation')
            if (not isinstance(observation, dict) or observation.get('instance_id') != instance_id
                    or saved.get('replayed') is not False
                    or outcome not in ('completed', 'already_at_target', 'settled_needs_nominal_correction',
                                       'settled_needs_nominal_retry', 'settled_needs_target_follow_up',
                                       'settled_needs_observation',
                                       'cancelled', 'failed')):
                raise original_error
            if outcome not in ('cancelled', 'failed'):
                fault = status.get('fault')
                if isinstance(fault, dict):
                    raise ControlError(fault.get('code', 'CONTROLLER_ERROR'),
                                       fault.get('message', '제어기 오류가 발생했습니다.'),
                                       **(fault.get('details') if isinstance(fault.get('details'), dict) else {}))
                if status.get('worker_alive') is not True or status.get('fresh_feedback') is not True:
                    raise original_error
            return saved

    def _interrupt_call(self, method, **params):
        if self.instance_id is None:
            raise ControlError("NOT_READY", "상주 제어기 설정을 먼저 확인해야 합니다.")
        result = self.client.call(method, controller_instance_id=self.instance_id, **params)
        if not isinstance(result, dict) or result.get('instance_id') != self.instance_id:
            raise ControlError('CONTROLLER_CHANGED', '목표 변경 중 상주 제어기가 바뀌었습니다.')
        return result

    def interrupt_motion(self, lease_id, command_id):
        return self._interrupt_call('interrupt_motion', lease_id=lease_id, command_id=command_id)

    def motion_interruption_capability(self):
        try:
            return self._interrupt_call('motion_interruption_capability')
        except ControlError as exc:
            # Existing revision-2 residents predate this optional RPC. Keep
            # their connection and ordinary commands available; the user can
            # attach a newer resident when interruption is needed.
            if exc.code == 'METHOD_NOT_FOUND':
                return {'instance_id': self.instance_id, 'supported': False,
                        'reason': 'resident_upgrade_required', 'physical_execution_verified': False,
                        'stop_delay_bounded': False}
            raise

    def resume_interrupted_motion(self, lease_id, interruption_id, observation_id):
        return self._interrupt_call('resume_interrupted_motion', lease_id=lease_id,
                                    interruption_id=interruption_id, observation_id=observation_id)

    def interrupt_mock_motion(self, lease_id, command_id):
        return self._interrupt_call('interrupt_mock_motion', lease_id=lease_id, command_id=command_id)

    def resume_mock_interrupted_motion(self, lease_id, interruption_id, observation_id):
        return self._interrupt_call('resume_mock_interrupted_motion', lease_id=lease_id,
                                    interruption_id=interruption_id, observation_id=observation_id)

    def recover(self,observation_id,layout_confirmed,*,layout_basis=None):
        params={'observation_id':observation_id,'layout_confirmed':layout_confirmed}
        if layout_basis is not None:
            params['layout_basis']=layout_basis
        if layout_basis in ('expected', 'unregistered'):
            if self.instance_id is None:
                raise ControlError('NOT_READY','상주 제어기 설정을 먼저 확인해야 합니다.')
            params['expected_instance_id']=self.instance_id
        result=self.client.call('recover',**params)
        if result.get('instance_id')!=self.instance_id:
            raise ControlError('CONTROLLER_CHANGED','복구 중 상주 제어기가 바뀌었습니다.')
        return result

    def end_episode(self,lease_id):
        guarded = self._untouched_start_lease is not None and self._untouched_start_lease == lease_id
        self._untouched_start_lease = None
        result = self.client.call("end_episode",lease_id=lease_id)
        if guarded:
            # Start revision 0 -> acquire +1 -> end's stop +1 -> release +1.
            # Any intervening command, stop or fault makes the next acquire fail.
            self._untouched_start_revision = 3
        return result

    def confirm_idle_stop(self,observation_id):
        result=self.client.call('confirm_idle_stop',observation_id=observation_id)
        if result.get('instance_id')!=self.instance_id:
            raise ControlError('CONTROLLER_CHANGED','정지 확인 중 상주 제어기가 바뀌었습니다.')
        return result

    def stop(self,reason="TRANSFER_CANCELLED"):
        return self.client.call("stop",reason=reason)

    def close(self):
        # Ending a client does not send shutdown, reset or a gripper command.
        return {"controller_connection_retained":True,"lifecycle":"externally_owned"}
