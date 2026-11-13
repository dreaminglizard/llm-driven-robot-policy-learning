"""Explicit app profiles and receive-only attachment to a resident controller."""
from dataclasses import dataclass,replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from raccoon_control.config import Config
from raccoon_control.errors import ControlError
from raccoon_control.locking import FileLock
from raccoon_control.transport import Client
from raccoon_control.validation import keys
from .executor import RobotExecutor
from .resident import ResidentController, require_ready_after_start
from .robot_bridge import SlotCalibration
from .scope import enabled_slots as validate_slots
from .qualification import check_expansion
from .session import Session


ROOT=Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PendingControllerStart:
    """Evidence retained only for the process started by this app request."""
    request_id: str
    runtime: Path
    output_path: Path
    config: Config
    process: object


def _windows_listener_owner(port, *, recorded_pid=None):
    """Look up an OS process using its saved PID or its loopback listener."""
    script = r'''
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if ($env:RACCOON_RECOVERY_PID) {
    $targetPid = [int]$env:RACCOON_RECOVERY_PID
} else {
    $listeners = @(Get-NetTCPConnection -LocalPort ([int]$env:RACCOON_RECOVERY_PORT) -State Listen |
        Where-Object { $_.LocalAddress -eq '127.0.0.1' })
    if ($listeners.Count -ne 1) { exit 3 }
    $targetPid = $listeners[0].OwningProcess
}
$p = Get-CimInstance Win32_Process -Filter ("ProcessId = " + $targetPid)
if ($null -eq $p -or $null -eq $p.CreationDate) { exit 4 }
[pscustomobject]@{ pid = [int]$p.ProcessId; path = $p.ExecutablePath; command = $p.CommandLine;
                   created = [string]$p.CreationDate.ToFileTimeUtc() } | ConvertTo-Json -Compress
'''
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            encoding='utf-8', timeout=8, creationflags=subprocess.CREATE_NO_WINDOW,
                            env={**os.environ, 'RACCOON_RECOVERY_PORT': str(port),
                                 'RACCOON_RECOVERY_PID': str(recorded_pid) if recorded_pid is not None else ''})
    if result.returncode != 0:
        raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기 프로세스를 식별하지 못했습니다.')
    try:
        return json.loads(result.stdout)
    except (TypeError, ValueError) as exc:
        raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기 프로세스 정보를 읽지 못했습니다.') from exc


def _windows_argv(command):
    import ctypes
    from ctypes import wintypes
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel.LocalFree.restype = wintypes.HLOCAL
    count = ctypes.c_int()
    arguments = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not arguments:
        raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기의 실행 정보를 읽지 못했습니다.')
    try:
        return [arguments[i] for i in range(count.value)]
    finally:
        kernel.LocalFree(arguments)


def _terminate_unresponsive_owned_controller(runtime, config_path, config, expected_instance):
    """Stop only the exact Raccoon process belonging to this instance.

    No application motion or gripper request is issued here. A PID alone is
    insufficient: endpoint, recorded owner or port, executable, arguments and
    process creation time must all name the same still-running controller.
    """
    if os.name != 'nt' or not isinstance(expected_instance, str) or not expected_instance:
        raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기의 소유 프로세스를 식별할 수 없습니다.')
    endpoint = Client(runtime, timeout=2).endpoint
    if endpoint.get('instance_id') != expected_instance:
        raise ControlError('CONTROLLER_CHANGED', '기존 제어기가 이미 바뀌었습니다.')
    try:
        owner = endpoint.get('owner_process')
        if owner is not None:
            if (not isinstance(owner, dict) or type(owner.get('pid')) is not int or owner['pid'] <= 0
                    or not isinstance(owner.get('created_filetime'), str)
                    or not owner['created_filetime'].isdigit()):
                raise ValueError('invalid saved owner identity')
            info = _windows_listener_owner(endpoint['port'], recorded_pid=owner['pid'])
        else:
            # Older residents have no owner record. A live listener is the
            # only OS evidence that links their endpoint to a process.
            info = _windows_listener_owner(endpoint['port'])
        pid = info['pid']
        command = info['command']
        created = int(info['created'])
        if type(pid) is not int or pid <= 0 or not isinstance(command, str):
            raise ValueError('invalid process identity')
        if owner is not None and pid != owner['pid']:
            raise ValueError('changed process identity')
        executables = {Path(sys.executable).resolve(),
                       Path(getattr(sys, '_base_executable', sys.executable)).resolve()}
        if not isinstance(info.get('path'), str) or Path(info['path']).resolve() not in executables:
            raise ValueError('unexpected executable')
        args = _windows_argv(command)
        modules = [i for i, value in enumerate(args) if value == '-m']
        if len(modules) != 1 or args[modules[0] + 1:modules[0] + 3] != ['raccoon_control', 'serve']:
            raise ValueError('unexpected process command')
        def option_path(name):
            positions = [i for i, value in enumerate(args) if value == name]
            if len(positions) != 1 or positions[0] + 1 >= len(args):
                raise ValueError('missing controller path')
            value = Path(args[positions[0] + 1])
            return (value if value.is_absolute() else ROOT / value).resolve()
        if option_path('--runtime') != Path(runtime).resolve() or option_path('--config') != Path(config_path).resolve():
            raise ValueError('unexpected controller runtime or config')
    except (KeyError, TypeError, ValueError, IndexError, OSError, subprocess.SubprocessError) as exc:
        raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기의 실행 정보가 일치하지 않습니다.') from exc

    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateProcess.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x0001 | 0x1000 | 0x00100000, False, pid)
    if not handle:
        raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기 프로세스에 접근하지 못했습니다.')
    try:
        birth, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(birth), ctypes.byref(exited),
                                      ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise ControlError('CONTROLLER_OWNER_UNKNOWN', '기존 제어기의 시작 시각을 확인하지 못했습니다.')
        actual_created = (birth.dwHighDateTime << 32) | birth.dwLowDateTime
        # CIM exposes microsecond precision; process IDs may have been reused.
        if abs(actual_created - created) > 10:
            raise ControlError('CONTROLLER_CHANGED', '기존 제어기 프로세스가 바뀌었습니다.')
        if owner is not None and actual_created != int(owner['created_filetime']):
            raise ControlError('CONTROLLER_CHANGED', '저장된 제어기 프로세스가 바뀌었습니다.')
        current = Client(runtime, timeout=2).endpoint
        if current.get('instance_id') != expected_instance or current.get('port') != endpoint['port']:
            raise ControlError('CONTROLLER_CHANGED', '기존 제어기 연결 정보가 바뀌었습니다.')
        if not kernel.TerminateProcess(handle, 74):
            raise ControlError('CONTROLLER_SHUTDOWN_PENDING', '기존 제어기를 종료하지 못했습니다.')
        if kernel.WaitForSingleObject(handle, 5000) != 0:
            raise ControlError('CONTROLLER_SHUTDOWN_PENDING', '기존 제어기 종료를 기다리고 있습니다.')
    finally:
        kernel.CloseHandle(handle)

    # The worker's parent watchdog stops and exits independently. Do not
    # unlink lock files: wait until the OS has released both actual locks.
    deadline = time.monotonic() + config.limits.close_timeout + 5.
    locks = [Path(runtime) / 'controller.lock']
    if config.backend == 'sdk':
        locks.append(Path(tempfile.gettempdir()) / 'raccoon-control-port-locks' / (config.port + '.lock'))
    while True:
        try:
            held = []
            try:
                for path in locks:
                    held.append(FileLock(path).acquire())
            finally:
                for lock in reversed(held):
                    lock.close()
            return
        except ControlError as exc:
            if exc.code != 'ALREADY_OWNED' or time.monotonic() >= deadline:
                raise ControlError('CONTROLLER_SHUTDOWN_PENDING', '기존 장치 연결 해제를 기다리고 있습니다.') from exc
            time.sleep(.1)


def _ready_connection_unknown(start, cause):
    if start.process.poll() is not None:
        raise ControlError('APP_CONTROLLER_START_FAILED', '시작한 제어기 프로세스가 종료됐습니다.',
                           log=str(start.output_path)) from cause
    raise ControlError('APP_CONTROLLER_READY_CONNECTION_UNKNOWN',
                       '시작한 제어기의 연결 상태를 확인할 수 없습니다. 같은 연결을 다시 확인해 주세요.',
                       cause=cause.code, log=str(start.output_path)) from cause


def _recover_restored_connection_fault(remote, status):
    """Acknowledge a prior process's connection fault without asserting a scene."""
    if (not isinstance(status, dict) or status.get('restored_connection_fault') is not True
            or not isinstance(status.get('fault'), dict)
            or status['fault'].get('code') not in ('DISCONNECTED', 'SENSOR_STALE')
            or status.get('instance_id') != remote.instance_id
            or status.get('worker_alive') is not True or status.get('fresh_feedback') is not True
            or status.get('active_command_id') is not None or status.get('lease_owner') is not None):
        raise ControlError('NOT_READY', '새 제어기의 수신과 종료된 작업을 확인하지 못했습니다.')
    observation = remote.observe()
    if (observation.get('instance_id') != remote.instance_id
            or observation.get('sensor_valid') is not True or observation.get('moving') is not False):
        raise ControlError('NOT_READY', '새 제어기의 정지 관측을 확인하지 못했습니다.')
    if (status.get('last_stop') or {}).get('confirmed') is not True:
        stopped = remote.confirm_idle_stop(observation['observation_id'])
        if stopped.get('confirmed') is not True:
            raise ControlError('STOP_UNCONFIRMED', '새 제어기의 정지를 확인하지 못했습니다.')
        observation = remote.observe()
    remote.recover(observation['observation_id'], False, layout_basis='unregistered')
    current = remote.status()
    require_ready_after_start(current)
    remote._recovered_restored_connection_fault = True
    return current


def _started_controller(start, *, untouched=False, allow_restored_connection_fault=False):
    """Read the unique child's ready record and the exact endpoint it named."""
    if start.process.poll() is not None:
        raise ControlError('APP_CONTROLLER_START_FAILED', '시작한 제어기 프로세스가 종료됐습니다.', log=str(start.output_path))
    try:
        output = start.output_path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise ControlError('NOT_READY', '시작한 제어기의 준비 결과를 기다리고 있습니다.') from exc
    # An incomplete line is not a ready event. Other output is not evidence.
    lines = output.splitlines(keepends=True)
    ready = []
    for line in lines:
        if not line.endswith(('\n', '\r')):
            continue
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and value.get('event') == 'ready':
            ready.append(value)
    if not ready:
        raise ControlError('NOT_READY', '시작한 제어기의 준비 결과를 기다리고 있습니다.')
    if len(ready) != 1:
        raise ControlError('CONTROLLER_CHANGED', '시작한 제어기의 준비 결과를 구분할 수 없습니다.')
    value = ready[0]
    initial = value.get('status')
    endpoint_file = value.get('endpoint_file')
    if (value.get('ok') is not True or not isinstance(initial, dict)
            or not isinstance(endpoint_file, str)
            or Path(endpoint_file).resolve() != (start.runtime / 'controller.json').resolve()
            or not isinstance(initial.get('instance_id'), str) or not initial['instance_id']
            or initial.get('configuration_sha256') != start.config.fingerprint()
            or initial.get('resident_api_revision') != 2):
        raise ControlError('CONTROLLER_CHANGED', '시작한 제어기의 설정 또는 식별 정보를 확인하지 못했습니다.')
    require_ready_after_start(initial, allow_restored_connection_fault=allow_restored_connection_fault)
    try:
        remote = ResidentController(start.config, start.runtime)
    except ControlError as exc:
        if exc.code in {'CONTROLLER_NOT_FOUND', 'CONTROLLER_UNREACHABLE'}:
            _ready_connection_unknown(start, exc)
        raise
    endpoint = remote.client.endpoint
    if endpoint.get('instance_id') != initial['instance_id']:
        raise ControlError('CONTROLLER_CHANGED', '시작한 제어기와 현재 연결 대상이 다릅니다.')
    timeout = remote.client.timeout
    remote.client.timeout = 2
    try:
        status = remote.start(pending_start_instance=initial['instance_id'],
                              allow_restored_connection_fault=allow_restored_connection_fault)
        if status.get('instance_id') != initial['instance_id'] or remote.instance_id != initial['instance_id']:
            raise ControlError('CONTROLLER_CHANGED', '시작한 제어기와 응답한 제어기가 다릅니다.')
        if allow_restored_connection_fault and status.get('restored_connection_fault') is True:
            status = _recover_restored_connection_fault(remote, status)
        if untouched:
            if (initial.get('fault') is not None or status.get('fault') is not None
                    or status.get('state') != 'idle' or status.get('active_command_id') is not None
                    or status.get('lease_owner') is not None or status.get('worker_alive') is not True
                    or status.get('fresh_feedback') is not True):
                raise ControlError('CONTROLLER_CHANGED', '시작 이후 제어기의 오류 또는 작업 상태가 바뀌었습니다.')
            observation = remote.observe()
            if (observation.get('instance_id') != initial['instance_id']
                    or type(observation.get('revision')) is not int or observation['revision'] != 0
                    or observation.get('state') != 'idle' or observation.get('sensor_valid') is not True
                    or observation.get('moving') is not False):
                raise ControlError('CONTROLLER_CHANGED', '시작 이후 제어기 상태가 바뀌었습니다.')
            # The first acquisition must atomically check this revision again.
            remote._untouched_start_revision = 0
        if start.process.poll() is not None:
            raise ControlError('APP_CONTROLLER_START_FAILED', '시작한 제어기 프로세스가 종료됐습니다.')
    except ControlError as exc:
        if exc.code in {'CONTROLLER_NOT_FOUND', 'CONTROLLER_UNREACHABLE'}:
            _ready_connection_unknown(start, exc)
        raise
    finally:
        remote.client.timeout = timeout
    return remote, status


def reattach_pending_controller(start):
    """Never search other runtimes or start a replacement for a pending child."""
    remote, status = _started_controller(start, untouched=True)
    remote.spawned_process = start.process
    return remote, {'created': True, 'runtime': str(start.runtime), 'status': status,
                    'resumed_start': start.request_id}


@dataclass(frozen=True)
class AppProfile:
    controller_config: Path
    calibration: Path
    runtime: Path
    state_reference: Path | None
    enabled_slots: tuple
    max_height: int
    route_evidence: Path | None = None
    compact_waypoints: bool | None = None

    def __post_init__(self):
        if self.compact_waypoints is not None and type(self.compact_waypoints) is not bool:
            raise ValueError("경유점 병합 설정은 참 또는 거짓이어야 합니다.")

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        data=json.loads(path.read_text(encoding="utf-8"))
        keys(data,allowed=cls.__dataclass_fields__,required=set(cls.__dataclass_fields__)-{"state_reference","route_evidence","compact_waypoints"})
        if type(data["max_height"]) is not int or not 1<=data["max_height"]<=4:
            raise ValueError("앱의 최대 높이는 1~4층이어야 합니다.")
        for name in ("controller_config","calibration","runtime","state_reference","route_evidence"):
            value=data.get(name)
            data[name]=(path.parent/value).resolve() if value else None
        data["enabled_slots"]=validate_slots(data["enabled_slots"])
        return cls(**data)

    def settings(self):
        config=Config.load(self.controller_config)
        calibration=SlotCalibration.load(self.calibration)
        if self.max_height>calibration.max_height or not set(self.enabled_slots)<=set(calibration.enabled_slots):
            raise ValueError("앱 범위가 선택한 실물 설정의 범위를 넘습니다.")
        calibration=replace(calibration,max_height=self.max_height,enabled_slots=self.enabled_slots)
        if self.compact_waypoints is not None:
            calibration=replace(calibration,compact_waypoints=self.compact_waypoints)
        calibration.require_ready(config)
        check_expansion(self,config)
        return config,calibration


def attach_controller(profile,*,allow_hardware=False,allow_start=False,_validated_config=None,
                      force_reconnect=False, previous_owner=None):
    # The demo's physical preflight supplies the same config that was checked
    # with its calibration. Other callers still validate the profile here.
    config=_validated_config if _validated_config is not None else profile.settings()[0]
    if config.backend=="sdk":
        if not allow_hardware:
            raise ControlError("HARDWARE_DISABLED","실물 앱에는 --allow-hardware가 필요합니다.")
        from raccoon_control.hardware_policy import check_hardware_time
        check_hardware_time()
    candidates=[profile.runtime]
    if profile.state_reference and profile.state_reference.exists():
        state=json.loads(profile.state_reference.read_text(encoding="utf-8"))
        if state.get("controller_runtime"):
            previous=Path(state["controller_runtime"])
            candidates.insert(0,(ROOT/previous).resolve() if not previous.is_absolute() else previous)
    profile.runtime.mkdir(parents=True,exist_ok=True)
    with FileLock(profile.runtime/"app-startup.lock"):
        reconnecting = False
        for runtime in dict.fromkeys(candidates):
            try:
                status=Client(runtime,timeout=2).call("status")
            except ControlError as exc:
                if exc.code not in {"CONTROLLER_NOT_FOUND","CONTROLLER_UNREACHABLE"}:
                    raise
                lock=runtime/"controller.lock"
                if lock.exists():
                    try:
                        with FileLock(lock):
                            pass
                    except ControlError as locked:
                        if locked.code != 'ALREADY_OWNED' or not allow_start:
                            raise
                        if isinstance(previous_owner, dict):
                            if Path(previous_owner.get('runtime', '')).resolve() != Path(runtime).resolve():
                                raise
                            expected_instance = previous_owner.get('instance_id')
                        else:
                            # A newly opened app may not yet have a wrapper for
                            # its old resident. The endpoint identity is still
                            # tied to the verified owner/process lifetime below.
                            expected_instance = Client(runtime, timeout=2).endpoint.get('instance_id')
                        _terminate_unresponsive_owned_controller(runtime, profile.controller_config, config,
                                                                 expected_instance)
                        reconnecting = True
                if force_reconnect:
                    reconnecting = True
                continue
            if status.get("configuration_sha256")!=config.fingerprint():
                raise ControlError("CONTROLLER_CONFIG_MISMATCH","기존 제어기의 설정이 앱 설정과 다릅니다. 기존 연결을 유지합니다.")
            fault=status.get('fault')
            if (force_reconnect and isinstance(previous_owner, dict)
                    and Path(previous_owner.get('runtime', '')).resolve() == Path(runtime).resolve()
                    and status.get('instance_id') != previous_owner.get('instance_id')):
                # A previous restart attempt may have completed after its RPC
                # timed out. Attach that new instance; never kill it and start
                # another one merely because the old wrapper is still held.
                if (status.get('active_command_id') is not None or status.get('lease_owner') is not None):
                    raise ControlError('RECOVERY_BUSY', '새 제어기의 작업이 끝나기를 기다립니다.')
                remote=ResidentController(config,runtime)
                remote.start()
                if status.get('restored_connection_fault') is True:
                    status=_recover_restored_connection_fault(remote,status)
                elif fault is not None:
                    raise ControlError('APP_CONTROLLER_START_FAULT', '새 제어기에 오류가 남아 있습니다.', fault=fault)
                return remote, {'created':False, 'runtime':str(runtime), 'status':status, 'reconnected':True}
            if (force_reconnect or (isinstance(fault, dict)
                    and fault.get('code') in ('DISCONNECTED', 'SENSOR_STALE')
                    and status.get('restored_connection_fault') is not True)):
                if ('active_command_id' not in status or 'lease_owner' not in status
                        or status.get('active_command_id') is not None or status.get('lease_owner') is not None
                        or (status.get('last_stop') or {}).get('reason') == 'stop_in_progress'):
                    raise ControlError('RECOVERY_BUSY', '기존 제어기의 작업이 종료되기를 기다립니다.')
                client=Client(runtime, timeout=2)
                if client.endpoint.get('instance_id') != status.get('instance_id'):
                    raise ControlError('CONTROLLER_CHANGED', '기존 제어기가 바뀌어 재연결을 중단했습니다.')
                accepted=client.call('shutdown')
                if not isinstance(accepted, dict) or accepted.get('accepted') is not True:
                    raise ControlError('CONTROLLER_SHUTDOWN_PENDING', '기존 제어기 종료를 확인하지 못했습니다.')
                deadline=time.monotonic()+config.limits.connect_timeout+5
                while True:
                    try:
                        with FileLock(runtime/'controller.lock'):
                            break
                    except ControlError as exc:
                        if exc.code != 'ALREADY_OWNED' or time.monotonic() >= deadline:
                            raise ControlError('CONTROLLER_SHUTDOWN_PENDING', '기존 제어기 종료를 기다리고 있습니다.') from exc
                        time.sleep(.05)
                reconnecting = True
                continue
            remote=ResidentController(config,runtime)
            remote.start()
            if status.get('restored_connection_fault') is True:
                status=_recover_restored_connection_fault(remote,status)
                reconnecting=True
            return remote,{"created":False,"runtime":str(runtime),"status":status,
                           "reconnected":reconnecting}
        if not allow_start:
            raise ControlError("CONTROLLER_NOT_FOUND","유지 중인 연결이 없습니다. 로봇 연결 버튼으로 연결해 주세요.")
        request_id=uuid.uuid4().hex
        runtime=profile.runtime.resolve()
        output_path=runtime/('app-controller-'+request_id+'.txt')
        flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        command=[sys.executable,"-m","raccoon_control","serve","--config",str(profile.controller_config),
                 "--runtime",str(runtime),"--expected-config-sha256",config.fingerprint()]
        if config.backend=="sdk":
            command.append("--allow-hardware")
        with output_path.open("x",encoding="utf-8") as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                     creationflags=flags,stdin=subprocess.DEVNULL,
                                     env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        pending=PendingControllerStart(request_id,runtime,output_path,config,process)
        # Keep and reap the child handle without tying device lifetime to the UI.
        threading.Thread(target=process.wait,daemon=True).start()
        deadline=time.monotonic()+config.limits.connect_timeout+5
        while time.monotonic()<deadline:
            if process.poll() is not None:
                error=ControlError("APP_CONTROLLER_START_FAILED","제어기 연결에 실패했습니다.",log=str(output_path))
                error.pending_start=pending
                raise error
            try:
                remote,status=_started_controller(pending, allow_restored_connection_fault=True)
                remote.spawned_process=process
                return remote,{"created":True,"runtime":str(runtime),"status":status,
                               "reconnected":reconnecting or bool(getattr(remote, '_recovered_restored_connection_fault', False))}
            except ControlError as exc:
                if exc.code not in {"CONTROLLER_NOT_FOUND", "CONTROLLER_UNREACHABLE", "NOT_READY"}:
                    exc.pending_start = pending
                    raise
            time.sleep(.05)
        # Do not kill/restart a process whose connection result is unknown.
        error=ControlError("APP_CONTROLLER_START_PENDING","연결 준비가 계속되고 있습니다. 잠시 뒤 다시 연결해 주세요.",log=str(output_path))
        error.pending_start=pending
        raise error


def create_session(model_dir,log_path,*,profile_path=None,allow_hardware=False,allow_start=False):
    if profile_path is None:
        return Session(model_dir,log_path)
    profile=AppProfile.load(profile_path)
    _,calibration=profile.settings()
    remote,connection=attach_controller(profile,allow_hardware=allow_hardware,allow_start=allow_start)
    executor=RobotExecutor(remote,calibration,Path(log_path).parent/"runs")
    session=Session(model_dir,log_path,executor=executor)
    session.record("controller_attached",**connection,message="연결 완료. 현재 배치를 등록해 주세요.")
    return session
