"""Numbered-slot application state and auditable offline execution."""
from dataclasses import asdict
from datetime import datetime, timezone
from copy import copy
import json
from pathlib import Path
import threading
from raccoon_control.timing import measured,span,snapshot as timing_snapshot

from .commands import goal_for, parse, registered_state, summary
from .inference import LearnedRecall, LocalPlanner
from .world import INITIAL, MOVES, state, step


class Session:
    def __init__(self, model_dir="artifacts/memory_v2", log_path=None, *, executor=None,
                 max_height=4, enabled_slots=(1,2,3,4)):
        self.executor=executor
        if executor is not None:
            max_height=executor.calibration.max_height
            enabled_slots=executor.calibration.enabled_slots
        self.planner = LocalPlanner(model_dir,max_height=max_height,enabled_slots=enabled_slots)
        self.memory = LearnedRecall(model_dir)
        self.current = INITIAL if executor is None else None
        self.initial = self.current
        self.layout_status="simulation" if executor is None else "unregistered"
        if self.current is not None:
            self.memory.observe(self.current)
        self.cancel = threading.Event()
        self.log_path = Path(log_path) if log_path else None
        self.sequence = 0
        self.registered_by_user = False
        self._run_lock=threading.Lock()
        self._state_lock=threading.Lock()
        self._record_lock=threading.Lock()
        self._closing=False
        self._stop_generation=0
        self._request_sequence=0
        self._stop_requests={}
        self._request=None

    @property
    def mode(self):
        return self.executor.mode if self.executor is not None else "symbolic_simulation"

    def record(self, event, **fields):
        with self._record_lock:
            value = {"at":datetime.now(timezone.utc).isoformat(), "event":event,
                     "sequence":self.sequence, "mode":self.mode,"layout_status":self.layout_status,
                     "memory_saved":self.memory.saved,"physical_success":None, **fields}
            self.sequence += 1
            if self.log_path:
                with span('app_event_write'):
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
        return value

    def signal_stop(self, *, closing=False):
        """Accept a stop immediately; device communication is a separate step."""
        with self._state_lock:
            if closing:
                self._closing=True
            self._stop_generation+=1
            self.cancel.set()
            ticket=object()
            self._stop_requests[ticket]=False
            return ticket

    def finish_stop(self, ticket):
        with self._state_lock:
            if ticket not in self._stop_requests or self._stop_requests[ticket]:
                raise ValueError("이미 처리했거나 알 수 없는 정지 요청입니다.")
            self._stop_requests[ticket]=True
        try:
            return self.executor.request_stop() if self.executor is not None else {"requested":True}
        finally:
            with self._state_lock:
                del self._stop_requests[ticket]

    def request_stop(self):
        return self.finish_stop(self.signal_stop())

    def begin_request(self, *, _generation=None):
        """Reserve one explicit new request before it is handed to a worker."""
        if not self._run_lock.acquire(blocking=False):
            raise ValueError("다른 요청이 실행 중입니다.")
        try:
            with self._state_lock:
                if self._closing:
                    raise ValueError("앱을 종료하는 중입니다.")
                if self._stop_requests:
                    raise ValueError("정지 요청을 처리하는 중입니다.")
                ticket=object()
                if _generation is None:
                    generation=self._stop_generation
                else:
                    generation,sequence=_generation
                    if sequence != self._request_sequence:
                        raise InterruptedError("더 최근 요청이 시작되어 이전 요청을 실행하지 않습니다.")
                self._request_sequence+=1
                self._request={"ticket":ticket,"generation":generation,"claimed":False}
                if generation == self._stop_generation:
                    self.cancel.clear()
                return ticket
        except BaseException:
            self._run_lock.release()
            raise

    def abandon_request(self, ticket):
        """Release a reservation only if its worker could not be started."""
        with self._state_lock:
            if self._request is None or self._request["ticket"] is not ticket or self._request["claimed"]:
                raise ValueError("실행 중이거나 알 수 없는 요청입니다.")
            self._request=None
        self._run_lock.release()

    def _observe_memory(self,value,*,write=False):
        try:
            with span('memory_update'):
                self.memory.observe(value,write=write)
        except Exception:
            self.memory.clear()
            self.layout_status='uncertain'
            raise

    def close(self, *, stop_ticket=None):
        self.finish_stop(stop_ticket if stop_ticket is not None else self.signal_stop(closing=True))
        timeout=40 if self.executor is not None else 5
        if not self._run_lock.acquire(timeout=timeout):
            raise RuntimeError("진행 중인 요청의 종료를 기다리고 있습니다.")
        try:
            result=self.executor.close() if self.executor is not None else {"closed":True}
            self.memory.clear()
            return result
        finally:
            self._run_lock.release()

    @measured('session')
    def execute(self, text, on_step=None, on_progress=None, *, request=None):
        command=None
        if request is None:
            with self._state_lock:
                generation=(self._stop_generation,self._request_sequence)
            with span('command_parse'):
                command=parse(text)
            if command.kind == "stop":
                self.request_stop()
                return self.record("stopped", message="정지를 요청했습니다.")
            request=self.begin_request(_generation=generation)
        with self._state_lock:
            if self._request is None or self._request["ticket"] is not request or self._request["claimed"]:
                raise ValueError("이미 실행했거나 알 수 없는 요청입니다.")
            self._request["claimed"]=True
            generation=self._request["generation"]
        try:
            if command is None:
                with span('command_parse'):
                    command=parse(text)
            if command.kind == "stop":
                self.request_stop()
                return self.record("stopped", message="정지를 요청했습니다.")
            return self._execute(text,command,on_step,on_progress,generation)
        finally:
            with self._state_lock:
                self._request=None
            self._run_lock.release()

    def _registration_stopped(self,generation):
        with self._state_lock:
            if generation == self._stop_generation and not self.cancel.is_set():
                return False
            self.memory.clear()
            self.layout_status="uncertain"
            return True

    def _stopped_registration(self,*,started=True):
        return self.record("stopped",current=self.current,
            message=("배치 등록을 중단했습니다. 현재 배치를 다시 등록해 주세요." if started
                     else "배치 등록을 시작하기 전에 중단했습니다. 기존 배치와 기억을 유지했습니다."))

    def _execute(self,text,command,on_step,on_progress,generation):
        with self._state_lock:
            stopped=generation != self._stop_generation or self.cancel.is_set()
        if stopped and command.kind not in {"register","status"}:
            raise InterruptedError("정지된 요청은 재개하지 않습니다. 새 요청으로 시작해 주세요.")
        if command.kind == "register":
            value=registered_state(command.counts)
            if max(map(len,value))>self.planner.max_height:
                raise ValueError(f"현재 지원하는 최대 높이는 {self.planner.max_height}층입니다.")
            with self._state_lock:
                stopped=generation != self._stop_generation or self.cancel.is_set()
                if not stopped:
                    self.layout_status='uncertain'
                    self.memory.clear()
            if stopped:
                return self._stopped_registration(started=False)
            if self._registration_stopped(generation):
                return self._stopped_registration()
            recovery=None
            prepare=getattr(self.executor,'prepare_registration',None)
            if prepare is not None:
                recovery=prepare()
            if self._registration_stopped(generation):
                return self._stopped_registration()
            self._observe_memory(value)
            with self._state_lock:
                if generation != self._stop_generation or self.cancel.is_set():
                    self.memory.clear()
                    self.layout_status="uncertain"
                    stopped=True
                else:
                    self.current=value
                    self.initial=value
                    self.registered_by_user=True
                    self.layout_status="registered" if self.executor is not None else "simulation"
                    stopped=False
            if stopped:
                return self._stopped_registration()
            return self.record("registered", current=self.current, source="human_position_counts",
                               controller_recovery=recovery,
                               message=("고장 해제를 확인했습니다. " if recovery and recovery.get('recovered') else '')+
                               "배치를 등록하고 이전 기억을 비웠습니다. "+summary(self.current))
        if command.kind == "status":
            message="시작 배치를 등록해 주세요." if self.current is None else summary(self.current)
            if self.layout_status=="uncertain":
                message="배치가 불확실합니다. 마지막 완료 배치: "+message
            return self.record("status", current=self.current, message=message)
        if self.current is None or self.layout_status=="uncertain":
            raise ValueError("현재 배치를 먼저 등록해 주세요. 등록하면 이전 기억을 비웁니다.")
        if command.slots and any(slot+1 not in self.planner.enabled_slots for slot in command.slots):
            raise ValueError("현재 사용 가능한 자리는 "+"·".join(map(str,self.planner.enabled_slots))+"번입니다.")
        if command.kind == "remember":
            # A write starts a new hidden state. Stage it on a shallow copy so
            # cancellation cannot alter the previous saved state; the model is
            # shared and only the completed state is published under the lock.
            candidate=copy(self.memory)
            with self._state_lock:
                stopped=generation != self._stop_generation or self.cancel.is_set()
            if not stopped:
                with span('memory_update'):
                    candidate.observe(self.current,write=True)
                with self._state_lock:
                    stopped=generation != self._stop_generation or self.cancel.is_set()
                    if not stopped:
                        self.memory=candidate
            if stopped:
                return self.record("stopped",message="기억 요청을 중단했습니다. 기존 기억 상태를 유지했습니다.")
            return self.record("remembered", message="현재 배치를 기억했습니다.", hidden_state_used=True)
        if command.kind == "restore":
            goal = self.memory.recall()
            goal_source = "decoded_gru_hidden_state"
        elif command.kind == "initial":
            goal, goal_source = self.initial, "registered_initial_state"
        else:
            goal, goal_source = goal_for(command,self.current,enabled_slots=self.planner.enabled_slots), "compiled_numbered_command"
        with span('model_planning'):
            plan = self.planner.plan(self.current, goal, cancel=self.cancel)
        planned=self.record("plan", text=text, command=asdict(command), current=self.current,
                    goal=goal, goal_source=goal_source, timings=timing_snapshot(),**asdict(plan))
        if on_progress:
            on_progress(planned)
        if self.executor is not None:
            def transferred(item):
                before=self.current
                self.current=state(item["after"])
                self._observe_memory(self.current)
                self.layout_status="expected"
                src,dst=MOVES[plan.actions[item["transfer"]]]
                event=self.record("expected_transfer",before=before,after=self.current,
                                  message=f"{src+1}번 → {dst+1}번 이동 명령 완료 · 예상 배치 갱신")
                if on_step:
                    on_step(event)
            try:
                result=self.executor.run(self.current,plan.actions,cancel=self.cancel,
                    on_transfer=transferred,on_progress=on_progress)
            except Exception:
                self.layout_status="uncertain"
                self.request_stop()
                raise
            if result["layout_uncertain"]:
                self.layout_status="uncertain"
            if result["status"]!="command_sequence_completed":
                message="작업을 중단했습니다. "+("현재 배치를 다시 등록해 주세요." if self.layout_status=='uncertain' else "완료한 배치는 유지했습니다.")
                return self.record("stopped",current=self.current,message=message,error=result.get("error"),
                    execution_reference=result.get("execution_reference"),completed_transfers=result["completed_transfers"],
                    timings=result.get('timings',{}))
            return self.record("completed",current=self.current,method=plan.method,
                search_expanded_states=plan.expanded_states,execution_reference=result.get("execution_reference"),
                timings=result.get('timings',{}),
                message="로봇 명령 완료 · 예상 배치: "+summary(self.current))
        for action in plan.actions:
            if self.cancel.is_set():
                return self.record("stopped", current=self.current, message="모의 작업을 정지했습니다.")
            previous = self.current
            self.current = step(previous,action)
            self._observe_memory(self.current)
            src,dst = MOVES[action]
            item = self.record("simulated_transfer", action=action, before=previous, after=self.current,
                               message=f"{src+1}번 → {dst+1}번: 맨 위 원통 1개")
            if on_step:
                on_step(item)
        if self.cancel.is_set():
            return self.record("stopped", current=self.current, message="모의 작업을 정지했습니다.")
        return self.record("completed", current=self.current, symbolic_success=self.current==goal,
                           method=plan.method, search_expanded_states=plan.expanded_states,
                           message="모의 작업 완료. "+summary(self.current))
