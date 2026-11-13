"""Local desktop app with explicit symbolic or resident-controller execution."""
import queue
import threading
import uuid
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from raccoon_control.config import Config

from .app_runtime import AppProfile,create_session
from .scope import presentation
from .voice import LocalVoice


class Application:
    def __init__(self, root, model_dir, voice_dir, *, profile_path=None, allow_hardware=False, session_factory=None):
        self.root = root
        self.events = queue.Queue()
        self.busy = False
        self.session = None
        self.model_dir,self.profile_path,self.allow_hardware=model_dir,profile_path,allow_hardware
        self.session_factory=session_factory
        self.stopping=self.closing=self.close_pending=False
        self.current = None if profile_path else ((1,), (2,), (3,), (4,))
        self.cancel_record = threading.Event()
        self.pending_voice_id = None
        self.voice = LocalVoice(voice_dir)
        profile=AppProfile.load(profile_path) if profile_path else None
        self.connection_label=('실물' if Config.load(profile.controller_config).backend=='sdk' else '제어기 모의') if profile else '모의'
        scope=presentation(profile.enabled_slots,profile.max_height) if profile else presentation((1,2,3,4),4)
        root.title("라쿤 기억 실험실 · 번호로 말하고, 기억으로 복원")
        width=min(1080,root.winfo_screenwidth()-40)
        height=min(900,root.winfo_screenheight()-90)
        root.geometry(f"{width}x{height}")
        root.minsize(min(900,width),min(650,height))
        root.configure(bg="#f3f2ed")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", font=("맑은 고딕",11), padding=(12,9))
        style.configure("Primary.TButton", background="#24443e", foreground="white")
        style.configure("Stop.TButton", background="#802f36", foreground="white")
        top = tk.Frame(root,bg="#f3f2ed")
        top.pack(fill="x",padx=28,pady=(22,10))
        tk.Label(top,text="라쿤 기억 실험실",font=("맑은 고딕",24,"bold"),fg="#1d2c2a",bg="#f3f2ed").pack(side="left")
        self.mode_text=tk.StringVar(value=self.connection_label+" 연결 확인 중" if profile_path else "모의 실행 · 실제 로봇 연결 없음")
        tk.Label(top,textvariable=self.mode_text,font=("맑은 고딕",11,"bold"),fg="#24443e",bg="#e0e9e3",padx=14,pady=8).pack(side="right")
        self.scope_text=tk.StringVar(value=scope['description'])
        tk.Label(root,textvariable=self.scope_text,font=("맑은 고딕",12),fg="#54635f",bg="#f3f2ed").pack(anchor="w",padx=30)
        self.canvas = tk.Canvas(root,height=max(150,min(265,height-550)),bg="#fbfaf7",highlightthickness=0)
        self.canvas.pack(fill="x",padx=28,pady=(18,12))
        self.canvas.bind("<Configure>",lambda event:self.draw())
        registration=tk.Frame(root,bg="#f3f2ed")
        registration.pack(fill="x",padx=28,pady=(0,8))
        self.counts=[]
        self.count_inputs=[]
        for i in range(4):
            ttk.Label(registration,text=f"{i+1}번").pack(side="left",padx=(8,4))
            value=tk.StringVar(value="1")
            self.counts.append(value)
            spin=ttk.Spinbox(registration,from_=0,to=scope['max_height'],width=3,textvariable=value,state="readonly")
            spin.pack(side="left");self.count_inputs.append(spin)
        self.register_button=ttk.Button(registration,text="시작 배치 등록",command=self.register)
        self.register_button.pack(side="left",padx=12)
        self.connect_button=ttk.Button(registration,text="로봇 연결",command=lambda:self.connect(True))
        if profile_path:
            self.connect_button.pack(side="right")
        self.detail=tk.StringVar(value="기억 없음")
        tk.Label(root,textvariable=self.detail,font=("맑은 고딕",10),fg="#64716d",bg="#f3f2ed").pack(anchor="w",padx=30)
        self.status = tk.StringVar(value="로컬 모델을 불러오는 중입니다.")
        tk.Label(root,textvariable=self.status,font=("맑은 고딕",11),fg="#24443e",bg="#f3f2ed",anchor="w",wraplength=1000).pack(fill="x",padx=30,pady=(0,10))
        row = tk.Frame(root,bg="#f3f2ed")
        row.pack(fill="x",padx=28)
        self.entry = ttk.Entry(row,font=("맑은 고딕",13))
        self.entry.pack(side="left",fill="x",expand=True,ipady=8)
        self.entry.bind("<Return>",lambda event:self.submit())
        self.run_button = ttk.Button(row,text="요청 실행",style="Primary.TButton",command=self.submit)
        self.run_button.pack(side="left",padx=(8,0))
        self.mic_button = ttk.Button(row,text="7초 녹음",command=self.record)
        self.mic_button.pack(side="left",padx=8)
        ttk.Button(row,text="정지 · Esc",style="Stop.TButton",command=self.stop).pack(side="left")
        root.bind("<Escape>",lambda _:self.stop())
        examples = tk.Frame(root,bg="#f3f2ed")
        examples.pack(fill="x",padx=28,pady=10)
        self.examples=examples
        self.show_scope(scope)
        for column in range(3):
            examples.columnconfigure(column,weight=1)
        self.note=tk.StringVar(value="버튼을 누를 때만 녹음합니다. 인식한 지원 명령은 바로 실행합니다. 즉시 정지는 화면 버튼 또는 Esc를 사용하세요.")
        tk.Label(root,textvariable=self.note,font=("맑은 고딕",10),fg="#64716d",bg="#f3f2ed",wraplength=1000).pack(anchor="w",padx=30,pady=(0,8))
        self.log = tk.Text(root,height=4,font=("맑은 고딕",10),bg="#fffefa",fg="#25342f",wrap="word",relief="flat",padx=12,pady=10,state="disabled")
        self.log.pack(fill="both",expand=True,padx=28,pady=(0,22))
        root.protocol("WM_DELETE_WINDOW",self.close)
        self.connect(False)
        self.root.after(50,self.poll)

    def show_scope(self,scope):
        self.scope_text.set(scope['description'])
        for spin in self.count_inputs:spin.configure(to=scope['max_height'])
        for widget in self.examples.winfo_children():widget.destroy()
        for i,text in enumerate(scope['examples']):
            ttk.Button(self.examples,text=text,command=lambda text=text:self.fill(text)).grid(
                row=i//3,column=i%3,sticky='ew',padx=(0,6),pady=(0,5))

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(),840)
        baseline=max(105,canvas.winfo_height()-40)
        layer_height=min(40,(baseline-30)/self.session.planner.max_height) if self.session else 32
        for i,items in enumerate(self.current if self.current is not None else ((),(),(),())):
            x = (i+0.5)*width/4
            canvas.create_line(x-78,baseline,x+78,baseline,fill="#c4c9c3",width=2)
            canvas.create_text(x,baseline+24,text=f"{i+1}번"+(f" · {len(items)}개" if self.current is not None else ""),font=("맑은 고딕",13,"bold"),fill="#273b35")
            if not items:
                canvas.create_text(x,baseline-38,text="빈자리" if self.current is not None else "미등록",font=("맑은 고딕",12),fill="#98a19c")
            for level,_ in enumerate(items):
                y = baseline-5-level*layer_height
                canvas.create_rectangle(x-45,y-layer_height*.75,x+45,y,fill="#d7ddd8",outline="#75887f",width=2)
                canvas.create_oval(x-45,y-layer_height,x+45,y-layer_height*.55,fill="#e9ede9",outline="#75887f",width=2)
                canvas.create_text(x,y-layer_height*.3,text=f"{level+1}층",font=("맑은 고딕",9),fill="#354c40")
        labels={"unregistered":"시작 배치 미등록","registered":"사용자가 등록한 배치","expected":"명령 완료에 따른 예상 배치",
                "uncertain":"배치 불확실 · 마지막 완료 배치","simulation":"모의 배치"}
        label=self.session.layout_status if self.session else ("unregistered" if self.profile_path else "simulation")
        canvas.create_text(16,16,anchor="nw",text=labels[label],font=("맑은 고딕",10),fill="#6a7770")
        if self.session:
            self.detail.set(("기억 저장됨" if self.session.memory.saved else "기억 없음")+" · 사용 자리 "+
                "·".join(map(str,self.session.planner.enabled_slots))+f"번 · 최대 {self.session.planner.max_height}층")

    def connect(self,allow_start=False):
        if self.busy or self.session is not None or self.closing:
            return
        self.set_busy(True)
        directory=("controller-mock-app" if self.connection_label=='제어기 모의' else "hardware-app") if self.profile_path else "memory-app"
        factory=self.session_factory or (lambda:create_session(self.model_dir,Path(".runtime")/directory/"events.jsonl",
            profile_path=self.profile_path,allow_hardware=self.allow_hardware,allow_start=allow_start))
        self.run_async(factory,"ready","connection_error")

    def register(self):
        self.fill(", ".join(f"{i+1}번에 {v.get()}개" for i,v in enumerate(self.counts))+" 있어")
        self.submit()

    def fill(self,text):
        if not self.busy and not self.stopping and not self.closing:
            self.entry.delete(0,"end")
            self.entry.insert(0,text)
            self.entry.focus_set()

    def set_busy(self,value):
        self.busy = value
        disabled=value or self.stopping or self.closing or self.session is None
        for widget in (self.run_button,self.mic_button,self.register_button):
            widget.configure(state="disabled" if disabled else "normal")
        self.connect_button.configure(state="disabled" if value or self.session is not None or self.closing else "normal")

    def append(self,text):
        self.log.configure(state="normal")
        self.log.insert("end",text+"\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def run_async(self,operation,event,error_event="error"):
        def run():
            try:
                self.events.put((event,operation()))
            except Exception as exc:
                self.events.put((error_event,str(exc)))
        threading.Thread(target=run,daemon=True).start()

    def submit(self):
        if self.busy or self.stopping or self.closing or self.session is None:
            return
        text = self.entry.get().strip()
        self.append("요청: "+text)
        try:
            request=self.session.begin_request()
        except Exception as exc:
            self.status.set(str(exc))
            self.append("확인 필요: "+str(exc))
            return
        self.set_busy(True)
        self.status.set("작업을 계획하고 있습니다.")
        try:
            self.run_async(lambda:self.session.execute(text,self.on_step,self.on_progress,request=request),"result")
        except Exception:
            self.session.abandon_request(request)
            self.set_busy(False)
            raise

    def on_step(self,item):
        self.events.put(("step",item))
        if self.session.executor is None:
            self.session.cancel.wait(0.3)

    def on_progress(self,item):
        self.events.put(("progress",item))

    def record(self):
        if self.busy or self.stopping or self.closing or self.session is None:
            return
        self.set_busy(True)
        self.cancel_record = threading.Event()
        cancel = self.cancel_record
        request_id = uuid.uuid4().hex
        self.pending_voice_id = request_id
        self.status.set("7초 동안 듣고 있습니다.")
        def recognize():
            try:
                return {"request_id":request_id,"result":self.voice.record(cancel=cancel)}
            except Exception as exc:
                return {"request_id":request_id,"error":str(exc)}
        self.run_async(recognize,"voice")

    def stop(self):
        self.cancel_record.set()
        if self.stopping:
            return
        session=self.session
        ticket=session.signal_stop() if session else None
        self.stopping=True
        self.set_busy(self.busy)
        self.status.set("정지를 요청하고 있습니다.")
        self.run_async(lambda:session.finish_stop(ticket) if session else None,"stop_result","stop_error")

    def poll(self):
        try:
            while True:
                event,item = self.events.get_nowait()
                if event == "ready":
                    self.session = item
                    if hasattr(self,'examples'):
                        self.show_scope(presentation(item.planner.enabled_slots,item.planner.max_height))
                    self.mode_text.set({"hardware_expected":"실물 · 예상 배치 추적","controller_simulation":"제어기 모의 실행",
                                        "symbolic_simulation":"모의 실행"}[item.mode])
                    self.current=item.current
                    self.draw()
                    self.set_busy(False)
                    self.status.set("현재 배치를 등록해 주세요." if item.current is None else "준비됐습니다. 텍스트로 요청하거나 녹음 버튼을 누르세요.")
                    if self.closing:
                        self.finish_close()
                elif event == "step":
                    self.current = item["after"]
                    self.append(item["message"])
                    self.draw()
                elif event == "voice":
                    if self.pending_voice_id is None or item.get("request_id")!=self.pending_voice_id:
                        continue
                    # Consume before submitting: a late duplicate must neither
                    # unlock an active task nor execute the same recording twice.
                    self.pending_voice_id=None
                    self.set_busy(False)
                    if self.cancel_record.is_set() or self.stopping or self.closing:
                        self.status.set("음성 요청을 취소했습니다.")
                    elif "error" in item:
                        self.status.set(item["error"])
                        self.append("확인 필요: "+item["error"])
                    else:
                        if item['result'].get('timings'):
                            try:
                                self.session.record('voice_recognized',text=item['result']['text'],
                                    voice_request_id=item['request_id'],timings=item['result']['timings'])
                            except Exception as exc:
                                self.status.set('음성 기록 저장 실패: '+str(exc))
                                self.append('음성 기록을 저장하지 못해 요청을 실행하지 않았습니다.')
                                continue
                        self.fill(item["result"]["text"])
                        self.append("음성 인식: "+item["result"]["text"])
                        self.submit()
                elif event == "result":
                    self.set_busy(False)
                    self.current = self.session.current
                    self.draw()
                    self.append(item["message"])
                    suffix = " · 탐색 보조 사용" if item.get("method")=="learned_policy_with_search" else ""
                    self.status.set(item["message"]+suffix)
                    if item.get('timings'):
                        try:
                            self.session.record('request_timing',execution_reference=item.get('execution_reference'),
                                                timings=item['timings'])
                        except Exception as exc:
                            self.status.set(item['message']+' · 시간 기록 저장 실패: '+str(exc))
                            self.append('시간 기록을 저장하지 못했습니다. 완료한 요청은 재실행하지 않습니다.')
                elif event == "progress":
                    if self.stopping or self.closing:
                        continue
                    if item.get("event")=="plan":
                        self.append("모델이 작업 순서를 정했습니다."+(" 탐색 보조 사용." if item["method"]=="learned_policy_with_search" else ""))
                    else:
                        stages={"raise_before_transfer":"이동 높이로 상승","open_before_pick":"집게 열기","above_source":"출발 자리로 이동",
                            "descend_to_pick":"집기 위치로 하강","close_for_pick":"집게 닫기","lift_from_source":"들어 올리기",
                            "above_destination":"도착 자리로 이동","descend_to_place":"놓기 위치로 하강","release_at_destination":"원통 놓기",
                            "retreat_from_destination":"위로 퇴피"}
                        self.status.set(stages.get(item["stage"],item["stage"])+f" · {item['completed_stages']}/{item['total_stages']}단계")
                elif event in {"error","connection_error"}:
                    self.set_busy(False)
                    if self.session:
                        self.current=self.session.current
                        self.draw()
                    self.status.set(item)
                    self.append("확인 필요: "+item)
                    if event=="connection_error":
                        label=getattr(self,'connection_label','실물')
                        self.mode_text.set(label+" · 앱 연결 안 됨")
                        self.note.set("새 연결은 로봇 연결 버튼을 누르면 시작합니다."+
                                      (" 연결 시 초기 자세로 돌아갈 수 있습니다." if label=='실물' else " 모의 장치만 사용합니다."))
                        if self.closing:
                            self.root.destroy()
                            return
                elif event in {"stop_result","stop_error"}:
                    self.stopping=False
                    self.set_busy(self.busy)
                    if event=="stop_error":
                        self.status.set("정지 확인 실패: "+item)
                    elif not self.closing:
                        self.status.set("정지 요청을 처리했습니다.")
                elif event=="closed":
                    self.root.destroy()
                    return
                elif event=="close_error":
                    self.closing=self.close_pending=False
                    self.set_busy(True)
                    self.status.set("종료 전 정지 확인이 필요합니다: "+item)
        except queue.Empty:
            pass
        self.root.after(50,self.poll)

    def finish_close(self):
        if not self.close_pending:
            self.close_pending=True
            session=self.session
            ticket=session.signal_stop(closing=True)
            self.run_async(lambda:session.close(stop_ticket=ticket),"closed","close_error")

    def close(self):
        if self.closing:
            return
        was_busy=self.busy
        self.closing=True
        self.cancel_record.set()
        self.set_busy(True)
        self.status.set("정지를 확인하고 있습니다. 로봇 연결은 유지합니다.")
        if self.session:
            self.finish_close()
        elif not was_busy:
            self.root.destroy()


def launch(model_dir,voice_dir,*,profile_path=None,allow_hardware=False):
    root = tk.Tk()
    Application(root,model_dir,voice_dir,profile_path=profile_path,allow_hardware=allow_hardware)
    root.mainloop()
