"""Windows product-instance ownership and a token-free reopen request.

An owner acquires the pipe before constructing the controller or preparing a
model, keeps it through shutdown, and installs a reopener only when its own
HTTP server is serving.  The reopener serializes with server shutdown.

Only an authenticated request to open a screen crosses this pipe: neither a
web address nor its token is sent to the other process.  The protected record,
Windows pipe peer PID, process creation time, and challenge are all required;
an old or unrecognized owner is not taken over.  This is not a defense against
malicious programs already running as the same Windows user.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from urllib.parse import quote


_SCHEMA = 'raccoonbot.product-instance.v1'
_OUTCOMES = frozenset(('open_requested', 'starting', 'closing', 'unavailable'))
_HANDSHAKE_TIMEOUT = 2.0
# A competing launcher may have a process before it reaches claim_or_reopen.
# Give it time to register or exit; an unresolved process still blocks startup.
_LAUNCH_GRACE = 3.0
_LAUNCH_POLL = .05
_claimant_handles = {}
_claimant_lock = threading.Lock()


@dataclass(frozen=True)
class InstanceAddress:
    """The stable pipe name and protected record for one Windows user."""
    pipe: str
    record: Path


@dataclass(frozen=True)
class ClaimResult:
    owner: 'ProductInstance | None'
    # owner, open_requested, starting, closing, unavailable, or unconfirmed.
    status: str


def _windows_modules():
    if os.name != 'nt':
        raise RuntimeError('제품 실행 확인은 현재 Windows에서만 사용할 수 있습니다.')
    import pywintypes
    import win32api
    import win32con
    import win32event
    import win32file
    import win32pipe
    import win32process
    import win32security
    return (pywintypes, win32api, win32con, win32event, win32file,
            win32pipe, win32process, win32security)


def _current_user_sid():
    _, api, con, _, _, _, _, security = _windows_modules()
    token = security.OpenProcessToken(api.GetCurrentProcess(), con.TOKEN_QUERY)
    try:
        return security.GetTokenInformation(token, security.TokenUser)[0]
    finally:
        token.Close()


def default_address():
    """Use the same product slot across installations for the current user."""
    *_, security = _windows_modules()
    sid = security.ConvertSidToStringSid(_current_user_sid())
    user_key = hashlib.sha256(sid.encode('ascii')).hexdigest()[:24]
    local_app_data = os.environ.get('LOCALAPPDATA')
    if not local_app_data:
        raise RuntimeError('현재 사용자의 앱 저장 위치를 확인하지 못했습니다.')
    return InstanceAddress(r'\\.\pipe\RaccoonBot.Product.v1.' + user_key,
                           Path(local_app_data) / 'RaccoonBot' / 'product-instance.json')


def _birth(pid):
    _, api, con, event, _, _, process, _ = _windows_modules()
    handle = api.OpenProcess(con.PROCESS_QUERY_LIMITED_INFORMATION | con.SYNCHRONIZE, False, pid)
    try:
        if event.WaitForSingleObject(handle, 0) != event.WAIT_TIMEOUT:
            raise RuntimeError('인스턴스 프로세스가 이미 종료됐습니다.')
        return process.GetProcessTimes(handle)['CreationTime'].isoformat()
    finally:
        handle.Close()


def _claimant_name(address, pid, birth):
    """Bind a launcher marker to its product slot and process identity."""
    identity = f'{address.pipe.casefold()}\0{pid}\0{birth}'.encode('utf-8')
    return r'Local\RaccoonBot.Product.Claimant.' + hashlib.sha256(identity).hexdigest()


def _register_claimant(address):
    """Advertise this process before attempting to reserve the product slot.

    Keep the handle for the process lifetime, including after a failed claim:
    another starter may still be checking processes while this one exits.
    """
    pywintypes, _, con, event, _, _, _, security = _windows_modules()
    pid = os.getpid()
    name = _claimant_name(address, pid, _birth(pid))
    with _claimant_lock:
        if name not in _claimant_handles:
            sid = _current_user_sid()
            acl = security.ACL()
            acl.AddAccessAllowedAce(security.ACL_REVISION, con.GENERIC_ALL, sid)
            descriptor = security.SECURITY_DESCRIPTOR()
            descriptor.SetSecurityDescriptorOwner(sid, 0)
            descriptor.SetSecurityDescriptorDacl(1, acl, 0)
            descriptor.SetSecurityDescriptorControl(security.SE_DACL_PROTECTED, security.SE_DACL_PROTECTED)
            attributes = pywintypes.SECURITY_ATTRIBUTES()
            attributes.SECURITY_DESCRIPTOR = descriptor
            _claimant_handles[name] = event.CreateEvent(attributes, True, True, name)


def _is_claimant(address, pid):
    """Check a live process's marker without accepting a reused PID."""
    try:
        _, _, con, event, _, _, _, security = _windows_modules()
        name = _claimant_name(address, pid, _birth(pid))
        handle = event.OpenEvent(con.SYNCHRONIZE | con.READ_CONTROL, False, name)
        try:
            descriptor = security.GetSecurityInfo(handle, security.SE_KERNEL_OBJECT,
                                                  security.OWNER_SECURITY_INFORMATION)
            return (descriptor.GetSecurityDescriptorOwner() == _current_user_sid()
                    and event.WaitForSingleObject(handle, 0) == event.WAIT_OBJECT_0)
        finally:
            handle.Close()
    except Exception:
        # pywin32 uses its own error type for a missing or inaccessible event.
        return False


def _write_private_record(path, record):
    """Publish the IPC key atomically with an explicit current-user-only DACL."""
    pywintypes, _, con, _, file_api, _, _, security = _windows_modules()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    acl = security.ACL()
    acl.AddAccessAllowedAce(security.ACL_REVISION, con.GENERIC_ALL, _current_user_sid())
    descriptor = security.SECURITY_DESCRIPTOR()
    descriptor.SetSecurityDescriptorDacl(1, acl, 0)
    descriptor.SetSecurityDescriptorControl(security.SE_DACL_PROTECTED, security.SE_DACL_PROTECTED)
    attributes = pywintypes.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR = descriptor
    temporary = path.with_name(path.name + '.' + secrets.token_hex(12) + '.tmp')
    try:
        handle = file_api.CreateFile(str(temporary), con.GENERIC_WRITE, 0, attributes,
                                     con.CREATE_NEW, con.FILE_ATTRIBUTE_NORMAL, None)
        try:
            file_api.WriteFile(handle, json.dumps(record, separators=(',', ':')).encode('ascii'))
            file_api.FlushFileBuffers(handle)
        finally:
            handle.Close()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_record(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError('인스턴스 기록이 너무 큽니다.')
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get('schema') != _SCHEMA
            or type(value.get('pid')) is not int or value['pid'] <= 0
            or not isinstance(value.get('birth'), str) or not value['birth']
            or not isinstance(value.get('key'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', value['key'])):
        raise ValueError('인스턴스 기록을 확인할 수 없습니다.')
    return value


def _connect(pipe, timeout):
    """Connect without the stdlib PipeClient's potentially long retry loop."""
    _, _, con, _, file_api, pipe_api, _, _ = _windows_modules()
    from multiprocessing.connection import PipeConnection
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            handle = file_api.CreateFile(pipe, con.GENERIC_READ | con.GENERIC_WRITE, 0, None,
                                         con.OPEN_EXISTING, con.FILE_FLAG_OVERLAPPED, None)
            try:
                pipe_api.SetNamedPipeHandleState(handle, pipe_api.PIPE_READMODE_MESSAGE, None, None)
                return PipeConnection(handle.Detach())
            except BaseException:
                handle.Close()
                raise
        except OSError as exc:
            # Busy is transient; a missing or inaccessible pipe is unknown,
            # not permission to start a second product instance.
            if getattr(exc, 'winerror', None) != 231 or time.monotonic() >= deadline:
                raise
            time.sleep(min(.05, max(0.0, deadline - time.monotonic())))


def _pipe_server_pid(connection):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GetNamedPipeServerProcessId
    query.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG))
    query.restype = wintypes.BOOL
    pid = wintypes.ULONG()
    if not query(wintypes.HANDLE(connection.fileno()), ctypes.byref(pid)):
        raise ctypes.WinError(ctypes.get_last_error())
    return pid.value


def _mac(key, label, nonce, result=b''):
    return hmac.new(key, label + b'\0' + nonce + b'\0' + result, hashlib.sha256).digest()


def _receive(connection, timeout, limit):
    if not connection.poll(timeout):
        raise TimeoutError('기존 실행의 응답을 확인하지 못했습니다.')
    return connection.recv_bytes(limit)


def _request_reopen(address, timeout):
    """Return an honest outcome; no web token or address enters the pipe."""
    command_sent = False
    try:
        record = _read_record(address.record)
        key = bytes.fromhex(record['key'])
        with _connect(address.pipe, timeout) as connection:
            pid = _pipe_server_pid(connection)
            if pid != record['pid'] or _birth(pid) != record['birth']:
                return 'unavailable'
            nonce = secrets.token_bytes(32)
            connection.send_bytes(nonce)
            if not hmac.compare_digest(_receive(connection, timeout, 32), _mac(key, b'server', nonce)):
                return 'unavailable'
            # A write error can happen after the other side received a full
            # message, so treat even an interrupted write as unconfirmed.
            command_sent = True
            connection.send_bytes(_mac(key, b'open', nonce))
            response = _receive(connection, timeout, 128)
            status, separator, signature = response.partition(b'\0')
            if (separator != b'\0' or not hmac.compare_digest(signature, _mac(key, b'result', nonce, status))):
                return 'unconfirmed'
            result = status.decode('ascii')
            return result if result in _OUTCOMES else 'unconfirmed'
    except (OSError, EOFError, TimeoutError, ValueError, TypeError, UnicodeError, RuntimeError):
        # Once the authenticated command was sent, a lost response does not
        # prove whether the existing instance opened the browser.
        return 'unconfirmed' if command_sent else 'unavailable'


class ProductInstance:
    """Keep the named-pipe reservation until the web app has fully stopped."""

    def __init__(self, address, listener, record):
        self.address, self._listener, self._record = address, listener, record
        self._key = bytes.fromhex(record['key'])
        self._closing = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reopener = None
        self._thread = threading.Thread(target=self._serve, name='product-reopen', daemon=True)
        self._thread.start()

    def set_reopener(self, reopener):
        """Install a zero-argument callback that checks the server's own lifecycle."""
        if not callable(reopener):
            raise TypeError('화면 열기 함수를 확인해 주세요.')
        with self._lock:
            if self._closing.is_set():
                raise RuntimeError('종료 중인 앱은 화면 열기를 시작할 수 없습니다.')
            self._reopener = reopener

    def begin_close(self):
        """Keep ownership, but report closing to later launch attempts.

        The web app must call this while holding the same lifecycle lock that
        its reopener uses and include ``owner.closing`` in the closing check.
        """
        self._closing.set()

    @property
    def closing(self):
        return self._closing.is_set()

    def _serve(self):
        while not self._stop.is_set():
            try:
                connection = self._listener.accept()
            except (OSError, EOFError):
                # A broken listener is not evidence of a successful reopen.
                # Retain the reservation until the owning process cleans up.
                break
            with connection:
                if self._stop.is_set():
                    break
                try:
                    nonce = _receive(connection, _HANDSHAKE_TIMEOUT, 32)
                    if len(nonce) != 32:
                        continue
                    connection.send_bytes(_mac(self._key, b'server', nonce))
                    proof = _receive(connection, _HANDSHAKE_TIMEOUT, 32)
                    if not hmac.compare_digest(proof, _mac(self._key, b'open', nonce)):
                        continue
                    if self._closing.is_set():
                        result = 'closing'
                    else:
                        with self._lock:
                            reopener = self._reopener
                        if reopener is None:
                            result = 'starting'
                        else:
                            try:
                                # The callback must recheck shutdown under the
                                # same lock that the web server uses to close.
                                result = reopener()
                            except Exception:
                                result = 'unavailable'
                            if not isinstance(result, str) or result not in _OUTCOMES:
                                result = 'unavailable'
                    status = result.encode('ascii')
                    connection.send_bytes(status + b'\0' + _mac(self._key, b'result', nonce, status))
                except (OSError, EOFError, TimeoutError, ValueError):
                    continue

    def close(self):
        """Release after controller and HTTP socket cleanup, not on tab close."""
        self.begin_close()
        self._stop.set()
        try:
            with _connect(self.address.pipe, .5):
                pass  # Wake the accept loop; no request is sent.
        except OSError:
            pass
        self._thread.join(_HANDSHAKE_TIMEOUT + 2)
        if self._thread.is_alive():
            # Do not drop ownership while an unresolved callback may be live.
            raise TimeoutError('기존 실행의 화면 열기 처리가 아직 끝나지 않았습니다.')
        try:
            if _read_record(self.address.record) == self._record:
                self.address.record.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError, UnicodeError):
            # A later owner or an unknown record must not be removed.
            pass
        # Remove our record while the pipe is still reserved: after releasing
        # the pipe, a new owner may publish its own record immediately.
        self._listener.close()


def claim_or_reopen(address=None, *, timeout=2.0, request_reopen=True):
    """Acquire before model startup, or ask the verified owner to reopen."""
    _windows_modules()
    address = address or default_address()
    _register_claimant(address)
    from multiprocessing.connection import Listener
    try:
        # CPython's Windows PipeListener uses FILE_FLAG_FIRST_PIPE_INSTANCE.
        listener = Listener(address.pipe, family='AF_PIPE')
    except OSError:
        return ClaimResult(None, _request_reopen(address, timeout) if request_reopen else 'unavailable')
    try:
        record = {'schema': _SCHEMA, 'pid': os.getpid(), 'birth': _birth(os.getpid()),
                  'key': secrets.token_hex(32)}
        _write_private_record(address.record, record)
        return ClaimResult(ProductInstance(address, listener, record), 'owner')
    except BaseException:
        listener.close()
        raise


def legacy_app_status(address=None):
    """Detect recognizable older web apps before starting a new product model.

    Older app versions do not hold our pipe and cannot answer a reopen request.
    A Python process of this user whose command cannot be read is inconclusive.
    Ancestors are excluded because the Windows venv launcher can create a
    second Python process with the same command line.  Another new launcher
    that reached our guard has a process-bound marker and cannot own this
    product slot.  An unmarked process is retried briefly, never ignored.
    """
    try:
        import psutil
        current = psutil.Process(os.getpid())
        username = current.username().casefold()
        ignored = {os.getpid(), *(parent.pid for parent in current.parents())}
        deadline = time.monotonic() + _LAUNCH_GRACE
        while True:
            processes = psutil.process_iter(['pid', 'name', 'username', 'cmdline'], ad_value=None)
            status = 'clear'
            for process in processes:
                info = process.info
                if info.get('pid') in ignored:
                    continue
                name = info.get('name')
                if not isinstance(name, str) or not re.fullmatch(r'(?:python(?:w|[0-9.]*)?|pyw?)\.exe', name, re.I):
                    continue
                user = info.get('username')
                if isinstance(user, str) and user.casefold() != username:
                    continue
                command = info.get('cmdline')
                if not isinstance(user, str) or not isinstance(command, (list, tuple)):
                    status = 'unknown'
                    continue
                args = [arg.casefold() for arg in command if isinstance(arg, str)]
                if any((arg == '-m' and index + 1 < len(args) and args[index + 1] == 'raccoon_memory.demo_web')
                       or arg == '-mraccoon_memory.demo_web' for index, arg in enumerate(args)):
                    address = address or default_address()
                    if not _is_claimant(address, info.get('pid')):
                        status = 'found'
                        break
            remaining = deadline - time.monotonic()
            if status == 'clear' or remaining <= 0:
                return status
            time.sleep(min(_LAUNCH_POLL, remaining))
    except Exception:
        return 'unknown'


def open_owned_screen(server, *, lifecycle_lock, is_serving, is_closing, browser_open):
    """Open this process's live, exclusive loopback server while holding its lifecycle lock.

    The caller must update serving/closing state and close the socket under the
    same lock.  The browser callback is invoked locally; no URL is returned to
    or transmitted by the duplicate launcher.
    """
    with lifecycle_lock:
        if is_closing():
            return 'closing'
        if not is_serving():
            return 'starting'
        try:
            sock = server.socket
            host, port = sock.getsockname()[:2]
            exclusive = getattr(socket, 'SO_EXCLUSIVEADDRUSE', None)
            if (sock.family != socket.AF_INET or host != '127.0.0.1'
                    or port != server.server_port or not sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
                    or exclusive is None or not sock.getsockopt(socket.SOL_SOCKET, exclusive)):
                return 'unavailable'
            if not isinstance(server.token, str) or not server.token:
                return 'unavailable'
            url = f'http://127.0.0.1:{port}/?token={quote(server.token, safe="")}'
            return 'open_requested' if browser_open(url) else 'unavailable'
        except (OSError, ValueError, AttributeError, TypeError):
            return 'unavailable'
