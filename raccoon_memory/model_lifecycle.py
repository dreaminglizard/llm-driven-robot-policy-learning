"""Keep one Windows model process tree owned until its exit is observable."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

if os.name == 'nt':
    import psutil
    import pywintypes
    import win32api
    import win32con
    import win32event
    import win32job
    import win32process
    import winerror


def _open_process(pid):
    try:
        return win32api.OpenProcess(
            win32con.SYNCHRONIZE | win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    except pywintypes.error as exc:
        if exc.winerror == winerror.ERROR_INVALID_PARAMETER:
            return None  # This PID no longer exists.
        raise


def _birth(handle):
    return win32process.GetProcessTimes(handle)['CreationTime'].isoformat()


def _pending(handle):
    state = win32event.WaitForSingleObject(handle, 0)
    if state == win32event.WAIT_OBJECT_0:
        return False
    if state == win32event.WAIT_TIMEOUT:
        return True
    raise OSError(f'프로세스 종료 신호를 확인하지 못했습니다 ({state}).')


class WindowsModelJob:
    """Create the wrapper suspended, track its descendants and fence restarts."""

    def __init__(self, model_path, *, state_dir=None):
        key = hashlib.sha256(os.path.normcase(str(Path(model_path).resolve())).encode()).hexdigest()
        if state_dir is None:
            state_dir = Path(__file__).resolve().parents[1] / '.runtime/model-processes'
        self.state = Path(state_dir) / (key + '.json')
        self._handles = {}
        self._births = {}
        self._stopping = False
        win32api.SetLastError(0)
        self.job = win32job.CreateJobObject(None, 'Local\\RaccoonModel_' + key)
        try:
            if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
                raise RuntimeError('같은 모델의 기존 프로세스가 종료될 때까지 기다려 주세요.')
            limits = win32job.QueryInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation)
            limits['BasicLimitInformation']['LimitFlags'] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(self.job, win32job.JobObjectExtendedLimitInformation, limits)
            if self.state.exists():
                for pid, birth in json.loads(self.state.read_text(encoding='utf-8')).items():
                    handle = _open_process(int(pid))
                    if handle is not None:
                        try:
                            if _pending(handle) and _birth(handle) == birth:
                                raise RuntimeError(f'이전 모델 프로세스 {pid}의 종료를 기다리고 있습니다.')
                        finally:
                            handle.Close()
                self.state.unlink()
        except BaseException:
            self.job.Close()
            self.job = None
            raise

    def capture(self):
        """Retain real process handles; later a PID or an exit code alone is insufficient."""
        members = win32job.QueryInformationJobObject(self.job, win32job.JobObjectBasicProcessIdList)
        changed = False
        for pid in members:
            if pid in self._handles:
                continue
            handle = _open_process(pid)
            if handle is None:
                continue
            try:
                if win32job.IsProcessInJob(handle, self.job):
                    self._births[pid] = _birth(handle)
                    self._handles[pid] = handle
                    changed = True
            finally:
                if pid not in self._handles:
                    handle.Close()
        if changed:
            self.state.parent.mkdir(parents=True, exist_ok=True)
            temp = self.state.with_suffix('.tmp')
            temp.write_text(json.dumps(self._births), encoding='utf-8')
            os.replace(temp, self.state)
        return members

    def popen(self, args, **kwargs):
        # A Windows venv redirector can spawn the real Python before its first line.
        kwargs['creationflags'] = kwargs.get('creationflags', 0) | 0x00000004  # CREATE_SUSPENDED
        process = subprocess.Popen(args, **kwargs)
        try:
            win32job.AssignProcessToJobObject(self.job, int(process._handle))
            self.capture()
            psutil.Process(process.pid).resume()
        except BaseException:
            process.kill()
            process.wait(timeout=5)
            raise
        return process

    def close(self, timeout=10):
        if self.job is None:
            return
        members = self.capture()
        if members and not self._stopping:
            win32job.TerminateJobObject(self.job, 1)
            self._stopping = True
        deadline = time.monotonic() + timeout
        while True:
            members = self.capture()
            pending = {pid for pid, handle in self._handles.items() if _pending(handle)}
            if not members and not pending:
                break
            if time.monotonic() >= deadline:
                pids = ', '.join(map(str, sorted(pending.union(members))))
                raise TimeoutError(f'모델 프로세스의 실제 종료를 기다리고 있습니다 (PID {pids}).')
            time.sleep(.05)
        self.state.unlink(missing_ok=True)
        for handle in self._handles.values():
            handle.Close()
        self._handles.clear()
        self.job.Close()
        self.job = None
