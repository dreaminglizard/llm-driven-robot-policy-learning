"""Application executor; it never owns or disconnects the resident device."""
from datetime import datetime,timezone
import json
from pathlib import Path
import threading
import uuid

from raccoon_control.errors import ControlError
from raccoon_control.timing import measured,span
from .robot_bridge import TransferRunner


class RobotExecutor:
    requires_registration=True

    def __init__(self,controller,calibration,log_dir,*,timeout=None):
        self.controller=controller
        self.calibration=calibration
        self.log_dir=Path(log_dir)
        self.timeout=timeout
        self.mode="hardware_expected" if controller.config.backend=="sdk" else "controller_simulation"
        self._lock=threading.Lock()
        self._runner=None
        self._last_result=None
        self._idle=threading.Event()
        self._idle.set()

    @measured('robot_executor')
    def run(self,initial,actions,*,cancel,on_progress,on_transfer):
        with self._lock:
            if self._runner is not None:
                raise ControlError("BUSY","다른 실물 요청이 실행 중입니다.")
            runner=TransferRunner(self.controller,self.calibration,self.timeout)
            runner.cancelled=cancel
            self._runner=runner
            self._idle.clear()
        try:
            result=runner.run(initial,actions,on_progress=on_progress,on_transfer=on_transfer)
            self._last_result=result
            self.log_dir.mkdir(parents=True,exist_ok=True)
            name=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex+".json"
            path=self.log_dir/name
            with span('result_encode'):
                encoded=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
            with span('result_write'):
                path.write_text(encoded,encoding="utf-8")
            return {**result,"execution_reference":str(path.resolve())}
        finally:
            with self._lock:
                self._runner=None
                self._idle.set()

    def request_stop(self):
        with self._lock:
            runner=self._runner
        if runner is not None:
            return runner.cancel()
        status=self.controller.status()
        if status.get('fault') and status.get('last_stop',{}).get('confirmed') is not True:
            observation=self.controller.observe()
            return self.controller.confirm_idle_stop(observation['observation_id'])
        return {"requested":False,"reason":"no_active_app_request"}

    def prepare_registration(self):
        """Called only for a new, explicit arrangement registration."""
        with self._lock:
            if self._runner is not None:
                raise ControlError('BUSY','진행 중인 작업을 먼저 정지해 주세요.')
            status=self.controller.status()
            if status.get('active_command_id') is not None or status.get('lease_owner') is not None:
                raise ControlError('RECOVERY_BUSY','기존 작업이 아직 종료되지 않았습니다.')
            if not status.get('fault'):
                return {'recovered':False}
            observation=self.controller.observe()
            return self.controller.recover(observation['observation_id'],layout_confirmed=True)

    def close(self):
        self.request_stop()
        deadline=self.controller.config.limits.motion_timeout+self.controller.config.limits.stop_timeout+5
        if not self._idle.wait(deadline):
            raise ControlError("APP_CLOSE_PENDING","정지 결과를 기다리고 있습니다. 창을 유지합니다.")
        last=self._last_result or {}
        owned=bool(last.get("stages") or "end" in last or "end_error" in last)
        if owned and last.get("end",{}).get("stop",{}).get("confirmed") is not True:
            # A later confirmed stop can resolve a lost end reply. A request
            # rejected before acquiring the device has no stop to confirm.
            if self.controller.status().get("last_stop",{}).get("confirmed") is not True:
                raise ControlError("STOP_UNCONFIRMED","마지막 작업의 정지가 확인되지 않았습니다. 창을 유지합니다.")
        # A read-only check; do not stop work owned by a different client.
        observation=self.controller.observe()
        if not observation["sensor_valid"] or observation["moving"] is not False:
            raise ControlError("STOP_UNCONFIRMED","새 정지 관측을 확인하지 못했습니다. 창을 유지합니다.")
        return {"controller_connection_retained":True,"stationary_feedback":True}
