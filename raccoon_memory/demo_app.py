"""Local virtual demonstration window for the current student session."""
from __future__ import annotations

import math
from functools import partial
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .demo_service import (BEST_CLOSE_SHA256, BEST_CLOSE_STUDENT, CLOSE_SHA256, CLOSE_STUDENT,
                           CLOSED_MOVE_PROFILES, CONTROL_TIMING_PROFILES, DEFAULT_STUDENT, FINGER_OPEN_PROFILES,
                           RL_SHA256, RL_STUDENT, SLOTS, DemoService, _AuxiliaryOperation,
                           _is_direct_stop_text, _is_text_stop_candidate)

BG = '#f3f5f2'
INK = '#1d302c'
TEAL = '#176a5d'
COLORS = ('#e79b56', '#6ca2c7', '#cc7f86', '#a28dbb')
CHAT_LANGUAGES = {'자동': 'auto', '한국어': 'ko', '영어': 'en'}
STUDENT_DEFAULT_PROFILE = '학생별 기본'


def parse_layout(values):
    stacks = []
    for value in values:
        value = value.strip()
        if not value:
            stacks.append([])
            continue
        parts = [part.strip() for part in value.split(',')]
        if any(part not in ('1', '2', '3', '4') for part in parts):
            raise ValueError('원통 번호는 1~4이며, 쉼표로 아래층부터 입력해 주세요.')
        stacks.append([int(part) for part in parts])
    ids = [item for stack in stacks for item in stack]
    if any(len(stack) > 2 for stack in stacks) or not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids):
        raise ValueError('원통은 1~4개, 번호는 중복 없이, 각 자리는 최대 2층이어야 합니다.')
    return stacks


def robot_points(joints):
    """Plan-view schematic using the nominal link geometry and observed joints."""
    q1, q2, q3, q4 = map(math.radians, joints)
    pitch = q2 + q3 + q4 + math.pi / 2
    radii = [0, -100 * math.sin(q2),
             -100 * math.sin(q2) - 100 * math.sin(q2 + q3)]
    radii.append(radii[-1] + 80 * math.cos(pitch) + 6 * math.sin(pitch))
    return [(-r * math.sin(q1), r * math.cos(q1)) for r in radii]


def describe_program(program):
    """Show the model's proposal without treating it as an executed result."""
    names = {'move': '이동', 'stack': '쌓기', 'unstack': '풀기', 'swap': '교환',
             'arrange': '배치', 'remember': '현재 배치 기억', 'register': '배치 등록'}
    lines = []
    for step in program.get('steps', []):
        op = step.get('op')
        if op == 'task':
            label = names.get(step.get('kind'), step.get('kind', '작업'))
            if step.get('source_slot') is not None and step.get('destination_slot') is not None:
                label += f" · {step['source_slot']}번 → {step['destination_slot']}번"
            if step.get('object_id') is not None:
                label += f" · 원통 {step['object_id']}"
        elif op == 'return':
            targets = {'registered_initial': '등록된 처음 배치로 복원', 'learned_memory': '기억한 배치로 복원',
                       'task_start': '직전 작업 시작 배치로 복원', 'arm_home': '팔 대기 자세로 복귀',
                       'unresolved': '복귀 대상 확인 필요'}
            label = targets.get(step.get('target', {}).get('kind'), '복귀')
        elif op == 'clarify':
            # The verifier checks question structure, not its relevance.
            label = '추가 확인 필요'
        else:
            label = {'repeat': '반복', 'stop_motion': '중단', 'stop_repeat': '반복 종료',
                     'continue': '계속', 'status': '상태 확인'}.get(op, str(op))
        lines.append(label)
    return ' → '.join(lines) if lines else '해석된 작업이 없습니다.'


class DemoWindow:
    def __init__(self, root, *, prepare_model=True):
        self.root = root
        self.root.title('라쿤 · 가상 시연')
        self.root.geometry('1200x790')
        self.root.minsize(1040, 780)
        self.root.configure(bg=BG)
        self.events = queue.Queue()
        events = self.events
        service_holder = {}
        def emit(kind, value):
            if kind == 'motion' and value.get('phase') == 'start':
                service = service_holder.get('service')
                coordinator = service.coordinator if service is not None else None
                if coordinator is not None:
                    try:
                        value = {**value, 'scene_snapshot': coordinator.snapshot()}
                    except Exception:
                        pass
            events.put((kind, value))
        self.service = DemoService(emit, visual_pace_seconds=.02)
        service_holder['service'] = self.service
        self.worker = None
        self.text_control_worker = None
        self._text_target = None
        self._record_target = None
        self._record_cancelled = False
        self._aux_operation = None
        self.current_action = None
        self.voice_stage = None
        self.chat = None
        self.chat_error = None
        self.closing = False
        self._closing_own_requests = set()
        self._interpreter_close_requested = False
        self.cleanup_thread = None
        self.layout = [[], [], [1, 2], [3, 4]]
        self.observation = None
        self.samples = []
        self.predicted_path = []
        self.forecast_segments = []
        self.forecast_sequence = None
        self.motion_number = None
        self.motion_session = None
        self.motion_interrupted = False
        self._stopped_motion_requests = set()
        self._display_stop_request = None
        self._latest_motion_numbers = {}
        self._motion_request_id = None
        self.model_file = BEST_CLOSE_STUDENT
        self.model_hash = BEST_CLOSE_SHA256
        self.model_pending = True
        self._recovery_required = False
        self._recovery_guarded = False
        self._busy = False
        self.model = tk.StringVar(value='집게·거리 결합 학생 · 기본')
        self.control_timing_profile = tk.StringVar(value=STUDENT_DEFAULT_PROFILE)
        self.control_timing_label = tk.StringVar(value='선택한 이동 시간은 배치 등록 시 적용됩니다.')
        self.finger_profile = tk.StringVar(value=STUDENT_DEFAULT_PROFILE)
        self.finger_profile_label = tk.StringVar(value='선택한 집게 열기는 배치 등록 시 적용됩니다.')
        self.closed_move_profile = tk.StringVar(value=STUDENT_DEFAULT_PROFILE)
        self.closed_move_profile_label = tk.StringVar(value='선택한 이동 곡선은 배치 등록 시 적용됩니다.')
        self.chat_language = tk.StringVar(value='자동')
        self.status = tk.StringVar(value='가상 시작 배치와 모델을 선택하고 등록해 주세요.')
        self.scene_label = tk.StringVar(value='미등록 · 화면은 설정값입니다')
        self.model_label = tk.StringVar(value='모델은 배치 등록 시 적용됩니다.')
        self.motion_label = tk.StringVar(value='가상 명령 대기 중')
        self.forecast_label = tk.StringVar(value='현재 표시할 이후 예상 경로가 없습니다.')
        self.llm_status = tk.StringVar(value='Gemma 4 E4B 준비 중' if prepare_model else '언어 모델 자동 준비 생략')
        self._build()
        self.busy(False)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Escape>', lambda _event: self.stop())
        self._poll_id = self.root.after(35, self.poll)
        if prepare_model:
            self.service.interpreter.prepare_async(lambda value: events.put(('llm_prepare', value)))

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TCombobox', padding=5)
        style.configure('VoiceLanguage.TCombobox', padding=1)
        header = tk.Frame(self.root, bg=INK, padx=24, pady=16)
        header.pack(fill='x')
        tk.Label(header, text='라쿤 가상 시연', font=('맑은 고딕', 21, 'bold'), fg='white', bg=INK).pack(side='left')
        tk.Label(header, text='SIMULATION ONLY  ·  실물 연결 없음', font=('맑은 고딕', 10, 'bold'),
                 fg='#a9e0cf', bg=INK).pack(side='right')
        body = tk.Frame(self.root, bg=BG, padx=16, pady=16)
        body.pack(fill='both', expand=True)
        left = tk.Frame(body, bg='white', padx=17, pady=16, width=315)
        left.pack(side='left', fill='y', padx=(0, 12))
        left.pack_propagate(False)
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        self.label(left, '1  시작 배치', 14, bold=True).pack(anchor='w')
        self.label(left, '아래층부터 쉼표로 입력 · 빈 자리는 비움', 9, color='#5c706b', justify='left').pack(anchor='w', pady=(2, 5))
        self.entries = []
        for index, value in enumerate(('', '', '1,2', '3,4')):
            row = tk.Frame(left, bg='white')
            row.pack(fill='x', pady=1)
            self.label(row, f'{index + 1}번 자리', 10).pack(side='left')
            entry = tk.Entry(row, font=('맑은 고딕', 11), relief='solid', bd=1, width=15)
            entry.insert(0, value)
            entry.pack(side='right', ipady=1)
            self.entries.append(entry)
        self.label(left, '2  학생 모델', 14, bold=True).pack(anchor='w', pady=(9, 3))
        self.model_select = ttk.Combobox(left, textvariable=self.model, state='readonly',
            values=('집게·거리 결합 학생 · 기본', '집게·거리 결합 학생 · 기존 시드', '기존 접근거리 학생',
                    '강화학습 후보 · 실험용', '직접 파일 선택'))
        self.model_select.pack(fill='x')
        self.model_select.bind('<<ComboboxSelected>>', self.choose_model)
        self.label(left, '직접 선택은 지원되는 학생 모델만 사용합니다.', 9, color='#607570').pack(anchor='w', pady=(2, 0))
        self.label(left, textvariable=self.model_label, size=9, color='#607570', justify='left', wraplength=275).pack(anchor='w', pady=(3, 4))
        self.register_button = self.button(left, '가상 배치 등록', self.register, filled=True)
        self.register_button.pack(fill='x', ipady=3)
        self.candidate_note = self.label(left, '개발 12과제 × 2조건: 23/24 · 실물·봉인 미실시', 9,
                                         color='#785b46', justify='left', wraplength=275)
        self.candidate_note.pack(anchor='w', before=self.register_button, pady=(0, 3))
        self.label(left, '배치를 다시 등록하면 이전에 기억한 배치가 지워집니다.', 9,
                   color='#607570', justify='left', wraplength=275).pack(anchor='w', pady=(3, 0))
        self.label(left, '3  요청', 14, bold=True).pack(anchor='w', pady=(9, 3))
        self.text = tk.Text(left, height=2, wrap='word', font=('맑은 고딕', 10), relief='solid', bd=1)
        self.text.pack(fill='x')
        self.text.insert('1.0', '3번 자리에서 1번 자리로 원통 하나를 옮겨 줘.')
        request_row = tk.Frame(left, bg='white')
        request_row.pack(fill='x', pady=(5, 3))
        self.run_button = self.button(request_row, '가상 실행', self.submit, filled=True)
        self.run_button.pack(side='left', fill='x', expand=True, ipady=3, padx=(0, 7))
        self.label(request_row, '대화 언어', 9, color='#607570').pack(side='left', padx=(0, 3))
        self.chat_language_select = ttk.Combobox(request_row, textvariable=self.chat_language,
            state='readonly', width=7, values=tuple(CHAT_LANGUAGES), style='VoiceLanguage.TCombobox')
        self.chat_language_select.pack(side='right')
        self.chat_button = self.button(left, '음성 대화 시작', self.toggle_chat, filled=True)
        self.chat_button.pack(fill='x', ipady=2, pady=(0, 3))
        voice_row = tk.Frame(left, bg='white')
        voice_row.pack(fill='x')
        self.record_button = self.button(voice_row, '한 번 녹음', self.record)
        self.record_button.pack(side='left', fill='x', expand=True, ipady=2, padx=(0, 3))
        self.audio_button = self.button(voice_row, '음성 파일 열기', self.audio)
        self.audio_button.pack(side='left', fill='x', expand=True, ipady=2)
        self.label(left, '로컬 로봇 명령 · 제한된 일반 대화\n응답 중 듣기 중지 · 중단: “멈춰”',
                   9, color='#607570', justify='left', wraplength=275).pack(anchor='w', pady=2)
        self.stop_button = self.button(left, '가상 작업 중단  ·  Esc', self.stop, color='#a53a36')
        self.stop_button.pack(fill='x', ipady=2, pady=(3, 0))

        info = tk.Frame(right, bg='white', padx=17, pady=13)
        info.pack(fill='x')
        self.label(info, '가상 작업대', 15, bold=True).pack(side='left')
        self.label(info, textvariable=self.scene_label, size=10, color='#a36532').pack(side='right')
        profile_panel = tk.Frame(right, bg='white', padx=17, pady=6)
        profile_panel.pack(fill='x')
        timing_row = tk.Frame(profile_panel, bg='white')
        timing_row.pack(fill='x')
        self.label(timing_row, '가상 이동 시간', 10, bold=True).pack(side='left', padx=(0, 10))
        self.control_timing_select = ttk.Combobox(timing_row, textvariable=self.control_timing_profile,
            state='readonly', width=24, values=(STUDENT_DEFAULT_PROFILE, *CONTROL_TIMING_PROFILES.values()))
        self.control_timing_select.pack(side='left')
        self.control_timing_select.bind('<<ComboboxSelected>>', self.choose_control_timing_profile)
        self.label(timing_row, textvariable=self.control_timing_label, size=9, color='#607570',
                   wraplength=300, justify='left').pack(side='left', padx=(10, 0))
        profile_row = tk.Frame(profile_panel, bg='white')
        profile_row.pack(fill='x')
        self.label(profile_row, '집게 열기', 10, bold=True).pack(side='left', padx=(0, 10))
        self.finger_profile_select = ttk.Combobox(profile_row, textvariable=self.finger_profile,
            state='readonly', width=24, values=(STUDENT_DEFAULT_PROFILE, *FINGER_OPEN_PROFILES.values()))
        self.finger_profile_select.pack(side='left')
        self.finger_profile_select.bind('<<ComboboxSelected>>', self.choose_finger_profile)
        self.label(profile_row, textvariable=self.finger_profile_label, size=9, color='#607570',
                   wraplength=300, justify='left').pack(side='left', padx=(10, 0))
        move_row = tk.Frame(profile_panel, bg='white')
        move_row.pack(fill='x', pady=(4, 0))
        self.label(move_row, '잡고 이동', 10, bold=True).pack(side='left', padx=(0, 10))
        self.closed_move_profile_select = ttk.Combobox(move_row, textvariable=self.closed_move_profile,
            state='readonly', width=24, values=(STUDENT_DEFAULT_PROFILE, *CLOSED_MOVE_PROFILES.values()))
        self.closed_move_profile_select.pack(side='left')
        self.closed_move_profile_select.bind('<<ComboboxSelected>>', self.choose_closed_move_profile)
        self.label(move_row, textvariable=self.closed_move_profile_label, size=9, color='#607570',
                   wraplength=300, justify='left').pack(side='left', padx=(10, 0))
        self.label(profile_panel, '기본 학생은 시연 시간과 즉시 열기를 사용합니다. 이전 설정을 복구하려면 기존 앱 시간을 고르고 집게 열기와 이동 곡선은 학생별 기본으로 선택하세요.\n가상 전용 · 실물에서는 확인하지 않았습니다.',
                   9, color='#785b46', justify='left', wraplength=700).pack(anchor='w', pady=(4, 0))
        self.canvas = tk.Canvas(right, bg='#e6efeb', highlightthickness=0, height=370)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda _event: self.draw())
        legend = tk.Frame(right, bg='white', padx=15, pady=8)
        legend.pack(fill='x')
        self.label(legend, '초록: 관측 가상 관절 · 주황: 관절로 계산한 손끝 · 파란 점선: 현재 명령 목표 · 보라 점선: 이후 가상 예상', 9, color='#45665e').pack(anchor='w')
        self.label(legend, '이후 경로는 예상 배치와 명목 원통 위치, 정지 관절을 가정합니다. 현재 요청의 후속 단계도 조건부로 계산하며 반복은 현재 회차 이후 최대 두 회차까지 예상합니다. 계산이 늦으면 경로가 표시되지 않을 수 있고, 앞선 명령을 기준으로 계산한 경로에는 이미 지난 부분이 포함될 수 있습니다. 계산 제한·오류가 있으면 일부만 표시하며 물체 성공은 확인하지 않습니다. 이전 세션 계산이 끝나지 않으면 이 명령의 예상을 생략하고 다음 명령에서 다시 시도합니다.', 9, color='#785b46', wraplength=700, justify='left').pack(anchor='w')
        self.label(legend, textvariable=self.forecast_label, size=9, color='#785596', wraplength=700, justify='left').pack(anchor='w')
        self.label(legend, textvariable=self.motion_label, size=9, color='#315d91', wraplength=700, justify='left').pack(anchor='w')
        log_frame = tk.Frame(right, bg='white', padx=15, pady=12)
        log_frame.pack(fill='x', pady=(12, 0))
        self.label(log_frame, '진행', 12, bold=True).pack(anchor='w')
        self.label(log_frame, textvariable=self.llm_status, size=9, color='#536d65',
                   wraplength=700, justify='left').pack(anchor='w')
        self.label(log_frame, textvariable=self.status, size=10, color=TEAL, wraplength=700, justify='left').pack(anchor='w', pady=(3, 5))
        self.log = tk.Text(log_frame, height=5, font=('맑은 고딕', 9), bg='#f8faf8', relief='flat', wrap='word')
        self.log.pack(fill='x')
        self.log.configure(state='disabled')

    @staticmethod
    def label(parent, text=None, size=10, bold=False, color=INK, **kwargs):
        return tk.Label(parent, text=text, font=('맑은 고딕', size, 'bold' if bold else 'normal'),
                        fg=color, bg=parent.cget('bg'), **kwargs)

    @staticmethod
    def button(parent, text, command, filled=False, color=TEAL):
        return tk.Button(parent, text=text, command=command, font=('맑은 고딕', 10, 'bold'),
                         bg=color if filled else 'white', fg='white' if filled else color,
                         activebackground='#d9ebe5', relief='solid', bd=1, cursor='hand2')

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def show_voice_device(self, result):
        if result.get('device') == 'GPU':
            self.append('음성 인식: GPU · 로컬 라이브러리 ' + str(result.get('cuda_runtime')))
        elif result.get('device') == 'CPU':
            self.append('음성 인식: CPU로 대체 · ' + str(result.get('fallback_reason') or 'GPU 사용 안 함'))

    def choose_model(self, _event=None):
        selected = self.model.get()
        if selected == '기존 접근거리 학생':
            self.model_file, self.model_hash = DEFAULT_STUDENT, None
        elif selected == '집게·거리 결합 학생 · 기본':
            self.model_file, self.model_hash = BEST_CLOSE_STUDENT, BEST_CLOSE_SHA256
        elif selected == '집게·거리 결합 학생 · 기존 시드':
            self.model_file, self.model_hash = CLOSE_STUDENT, CLOSE_SHA256
        elif selected == '강화학습 후보 · 실험용':
            self.model_file, self.model_hash = RL_STUDENT, RL_SHA256
        else:
            name = filedialog.askopenfilename(title='호환되는 학생 모델 선택', filetypes=[('PyTorch 모델', '*.pt'), ('모든 파일', '*.*')])
            if not name:
                self.model.set('집게·거리 결합 학생 · 기본')
                self.model_file, self.model_hash = BEST_CLOSE_STUDENT, BEST_CLOSE_SHA256
            else:
                self.model_file, self.model_hash = Path(name), None
        if self.model.get() in ('집게·거리 결합 학생 · 기본', '집게·거리 결합 학생 · 기존 시드'):
            self.candidate_note.pack(anchor='w', before=self.register_button, pady=(0, 3))
        else:
            self.candidate_note.pack_forget()
        self.model_pending = True
        self.busy(False)
        self.model_label.set(self.model_file.name + ' · 다음 배치 등록에 적용')
        self.status.set('선택한 모델을 적용하려면 가상 배치를 등록해 주세요.')

    def choose_finger_profile(self, _event=None):
        self.model_pending = True
        self.finger_profile_label.set('선택을 적용하려면 가상 배치를 다시 등록해 주세요.')
        self.status.set('선택한 집게 열기를 적용하려면 가상 배치를 등록해 주세요.')
        self.busy(False)

    def choose_closed_move_profile(self, _event=None):
        self.model_pending = True
        self.closed_move_profile_label.set('선택을 적용하려면 가상 배치를 다시 등록해 주세요.')
        self.status.set('선택한 이동 곡선을 적용하려면 가상 배치를 등록해 주세요.')
        self.busy(False)

    def choose_control_timing_profile(self, _event=None):
        self.model_pending = True
        self.control_timing_label.set('선택을 적용하려면 가상 배치를 다시 등록해 주세요.')
        self.status.set('선택한 이동 시간을 적용하려면 가상 배치를 등록해 주세요.')
        self.busy(False)

    def busy(self, value):
        self._busy = value
        chat_active = self.chat is not None
        state = 'disabled' if value or chat_active else 'normal'
        text_interrupt = chat_active or (value and self.current_action == 'submit')
        recovery_blocks_input = self._recovery_required and not self._recovery_guarded
        for widget in (self.register_button, self.audio_button, *self.entries):
            widget.configure(state=state)
        self.run_button.configure(text=('정지 요청' if text_interrupt else
                                        '요청 확인' if self._recovery_required else '가상 실행'),
            state='disabled' if (not text_interrupt and (value or chat_active or self.model_pending
                                                         or recovery_blocks_input
                                                         or self.service.coordinator is None)) else 'normal')
        self.record_button.configure(state='disabled' if value or chat_active or self.model_pending or recovery_blocks_input or self.service.coordinator is None else 'normal')
        self.chat_button.configure(state=('disabled' if (self.chat.stop_event.is_set() if chat_active else
                                   value or self.model_pending or recovery_blocks_input or self.service.coordinator is None) else 'normal'))
        self.text.configure(state='normal' if text_interrupt else state)
        self.model_select.configure(state='disabled' if value or chat_active else 'readonly')
        self.control_timing_select.configure(state='disabled' if value or chat_active else 'readonly')
        self.finger_profile_select.configure(state='disabled' if value or chat_active else 'readonly')
        self.closed_move_profile_select.configure(state='disabled' if value or chat_active else 'readonly')
        self.chat_language_select.configure(state='disabled' if value or chat_active else 'readonly')

    def replace_text(self, value):
        state = self.text.cget('state')
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', value)
        self.text.configure(state=state)

    def run_async(self, action, operation, *, auxiliary=None):
        if ((self.worker is not None and self.worker.is_alive())
                or (self.text_control_worker is not None and self.text_control_worker.is_alive())):
            return
        if action in ('submit', 'record', 'register'):
            self.motion_interrupted = False
            self._display_stop_request = None
        self._aux_operation = auxiliary
        self.current_action = action
        self.busy(True)
        events = self.events
        shutdown = self.service.cancelled
        def run():
            try:
                result = operation()
                if action == 'audio' and auxiliary is not None:
                    auxiliary.commit(lambda: None, shutdown)
                events.put(('finished', (action, result, None, auxiliary)))
            except Exception as exc:
                events.put(('finished', (action, None, exc, auxiliary)))
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def register(self):
        try:
            layout = parse_layout([entry.get() for entry in self.entries])
            finger_label = self.finger_profile.get()
            profile = next((key for key, label in FINGER_OPEN_PROFILES.items() if label == finger_label), None)
            if profile is None and finger_label != STUDENT_DEFAULT_PROFILE:
                raise ValueError('가상 집게 열기 방식을 다시 선택해 주세요.')
            if finger_label == STUDENT_DEFAULT_PROFILE:
                profile = 'auto'
            move_label = self.closed_move_profile.get()
            move_profile = next((key for key, label in CLOSED_MOVE_PROFILES.items() if label == move_label), None)
            if move_profile is None and move_label != STUDENT_DEFAULT_PROFILE:
                raise ValueError('가상 이동 곡선을 다시 선택해 주세요.')
            if move_label == STUDENT_DEFAULT_PROFILE:
                move_profile = 'auto'
            timing_label = self.control_timing_profile.get()
            timing_profile = next((key for key, label in CONTROL_TIMING_PROFILES.items() if label == timing_label), None)
            if timing_profile is None and timing_label != STUDENT_DEFAULT_PROFILE:
                raise ValueError('가상 이동 시간을 다시 선택해 주세요.')
            if timing_label == STUDENT_DEFAULT_PROFILE:
                timing_profile = 'auto'
        except ValueError as exc:
            messagebox.showerror('배치 확인', str(exc), parent=self.root)
            return
        self.layout, self.observation, self.samples = layout, None, []
        self.model_pending = True
        self.predicted_path = []
        self.forecast_segments = []
        self.motion_number = None
        self.motion_session = None
        self.finger_profile_label.set('선택한 집게 열기를 등록하고 있습니다.')
        self.control_timing_label.set('선택한 이동 시간을 등록하고 있습니다.')
        self.closed_move_profile_label.set('선택한 이동 곡선을 등록하고 있습니다.')
        self.motion_label.set('가상 명령 대기 중')
        self.scene_label.set('등록 중 · 사용자 설정값')
        self.status.set('가상 세션과 학생 모델을 준비하고 있습니다.')
        self.draw()
        auxiliary = _AuxiliaryOperation()
        self.run_async('register', partial(self.service.register, layout, self.model_file, self.model_hash,
                                           finger_open_profile=profile, closed_move_profile=move_profile,
                                            control_timing_profile=timing_profile,
                                           operation=auxiliary), auxiliary=auxiliary)

    def submit(self):
        if self.model_pending or self.service.coordinator is None:
            self.status.set('선택한 설정으로 가상 배치를 먼저 등록해 주세요.')
            return
        text = self.text.get('1.0', 'end').strip()
        if self._recovery_required and not self._recovery_guarded and not _is_direct_stop_text(text):
            self.status.set('가상 배치를 다시 등록해 주세요. 재등록하면 이전 기억이 지워집니다.')
            return
        if ((self.worker is not None and self.worker.is_alive()) or self.chat is not None):
            if (self.current_action not in ('submit', None) and self.chat is None) or not _is_text_stop_candidate(text):
                self.status.set('진행 중에는 명확한 즉시 정지 요청만 처리할 수 있습니다.')
                return
            if self.text_control_worker is not None and self.text_control_worker.is_alive():
                self.status.set('앞선 정지 요청을 확인하고 있습니다.')
                return
            chat = self.chat
            if chat is not None:
                target = chat.current_target()
            else:
                target = self._text_target
            if target is None:
                self.status.set('중단할 작업이 없습니다.')
                return
            if _is_direct_stop_text(text):
                self._stopped_motion_requests.add(target)
                self._display_stop_request = target
            events = self.events
            service = self.service
            self.append('정지 요청: ' + text)
            def control():
                try:
                    language = 'ko' if any('가' <= c <= '힣' for c in text) else 'en'
                    handled = (chat._stop_target(target, language=language) if chat is not None and _is_direct_stop_text(text)
                               else chat.stop_pending_prefix(target, text, language=language) if chat is not None
                               else None)
                    if handled is None:
                        events.put(('text_stop_finished', (service.submit(text, stop_target=target), None)))
                except Exception as exc:
                    events.put(('text_stop_finished', (None, exc)))
            self.text_control_worker = threading.Thread(target=control, daemon=True, name='demo-text-stop')
            self.text_control_worker.start()
            return
        if self.text_control_worker is not None and self.text_control_worker.is_alive():
            self.status.set('앞선 정지 요청을 확인하고 있습니다.')
            return
        self.samples = []
        self.predicted_path = []
        self.forecast_segments = []
        self.motion_number = None
        self.motion_session = None
        self.motion_label.set('가상 명령 대기 중')
        self.append('요청: ' + text)
        try:
            request = None if _is_direct_stop_text(text) else self.service.reserve_text_request()
        except RuntimeError as exc:
            self.status.set(str(exc))
            return
        self._text_target = request
        self.run_async('submit', partial(self.service.submit, text, request_id=request))

    def audio(self):
        path = filedialog.askopenfilename(title='저장된 음성 파일 선택',
            filetypes=[('음성 파일', '*.wav *.flac *.ogg'), ('모든 파일', '*.*')])
        if path:
            self.replace_text('')
            self.status.set('저장된 음성을 이 PC에서 글자로 변환하고 있습니다.')
            auxiliary = _AuxiliaryOperation()
            self.run_async('audio', partial(self.service.transcribe, path, operation=auxiliary), auxiliary=auxiliary)

    def record(self):
        if self.model_pending or self.service.coordinator is None:
            self.status.set('선택한 설정으로 가상 배치를 먼저 등록해 주세요.')
            return
        if self._recovery_required and not self._recovery_guarded:
            self.status.set('가상 배치를 다시 등록해 주세요. 재등록하면 이전 기억이 지워집니다.')
            return
        if self.worker is not None and self.worker.is_alive():
            return
        # An earlier transcript must never survive a failed/cancelled capture.
        self.replace_text('')
        self.samples = []
        self.predicted_path = []
        self.forecast_segments = []
        self.motion_number = None
        self.motion_session = None
        self.motion_label.set('가상 명령 대기 중')
        self.voice_stage = 'recording'
        self.status.set('7초 동안 녹음합니다. 중단 또는 Esc로 취소할 수 있습니다.')
        try:
            self._record_target = self.service.reserve_record_request()
        except RuntimeError as exc:
            self.status.set(str(exc))
            return
        self._record_cancelled = False
        self.run_async('record', partial(self.service.record_and_submit, request_id=self._record_target))

    def toggle_chat(self):
        if self.chat is not None:
            self.chat.stop()
            self.chat_button.configure(text='음성 대화 종료 중', state='disabled')
            self.status.set('음성 대화를 종료하고 있습니다.')
            return
        if self.model_pending or self.service.coordinator is None:
            self.status.set('선택한 설정으로 가상 배치를 먼저 등록해 주세요.')
            return
        if self._recovery_required and not self._recovery_guarded:
            self.status.set('가상 배치를 다시 등록해 주세요. 재등록하면 이전 기억이 지워집니다.')
            return
        if self.worker is not None and self.worker.is_alive():
            return
        language_label = self.chat_language.get()
        if language_label not in CHAT_LANGUAGES:
            self.status.set('음성 인식 언어를 다시 선택해 주세요.')
            return
        from .demo_voice_chat import LocalConversation
        self.replace_text('')
        self.samples = []
        self.predicted_path = []
        self.forecast_segments = []
        self.motion_number = None
        self.motion_session = None
        self.motion_interrupted = False
        self.chat_error = None
        events = self.events
        self.chat = LocalConversation(self.service, lambda kind, value: events.put((kind, value)),
                                      language=CHAT_LANGUAGES[language_label])
        self.chat_button.configure(text='음성 대화 종료')
        self.busy(False)
        self.status.set('음성 대화를 시작하고 있습니다.')
        self.append('음성 대화 인식 언어: ' + language_label)
        self.append('음성 대화: 로봇 명령과 제한된 일반 대화를 처리합니다. 앱이 말하는 동안에는 듣지 않습니다.')
        self.append('음성 중단은 인식에 시간이 걸립니다. 버튼이나 Esc로도 중단을 요청할 수 있습니다.')
        self.append('잡음이나 되울림이 말로 인식될 수 있습니다. 반복 입력과 실행 결과를 확인해 주세요.')
        try:
            self.chat.start()
        except Exception as exc:
            self.chat = None
            self.chat_button.configure(text='음성 대화 시작')
            self.busy(False)
            self.status.set('음성 대화를 시작하지 못했습니다: ' + str(exc))

    def stop(self):
        target = (self.chat.current_target() if self.chat is not None else
                  self._text_target if self.current_action == 'submit' else
                  self._record_target if self.current_action == 'record' else None)
        if target is not None:
            self._stopped_motion_requests.add(target)
            self._display_stop_request = target
        self.predicted_path = []
        self.forecast_segments = []
        self.motion_number = None
        self.motion_session = None
        self.forecast_label.set('중단 요청으로 이후 예상 경로를 지웠습니다.')
        self.draw()
        if self.chat is not None:
            chat = self.chat
            events = self.events
            self.status.set('가상 작업 중단 요청을 처리하고 있습니다.')
            def stop_chat_motion():
                try:
                    chat.stop_motion()
                except Exception as exc:
                    events.put(('chat_error', '중단 요청 오류: ' + str(exc)))
            threading.Thread(target=stop_chat_motion, daemon=True).start()
            return
        if self.worker is None or not self.worker.is_alive():
            return
        try:
            if self.current_action in ('submit', 'record'):
                result = self.service.stop(target_request_id=(self._text_target if self.current_action == 'submit'
                                                               else self._record_target))
                if self.current_action == 'record' and result.get('status') != 'already_completed':
                    self._record_cancelled = True
                    self.replace_text('')
                message = ('해당 가상 작업은 이미 완료됐습니다. 중단을 요청하지 않았습니다.'
                           if result.get('status') == 'already_completed' else
                           '해당 시점의 작업이 끝났거나 바뀌어 중단하지 않았습니다.'
                           if result.get('status') == 'no_matching_active_request' else
                           '중단할 요청을 확인하지 못했습니다.'
                           if result.get('status') in ('target_required', 'ambiguous_target', 'no_active_task') else
                           '녹음을 취소하고 있습니다.' if self.current_action == 'record' and self.voice_stage == 'recording'
                           else '중단을 요청했습니다. 가상 명령의 중단 결과를 확인 중입니다.')
            else:
                operation = self._aux_operation
                cancelled = operation is not None and operation.cancel()
                result = {'status': 'cancel_requested' if cancelled else 'already_completed'}
                message = (('음성 변환이 끝나면 결과를 버립니다.' if self.current_action == 'audio'
                            else '가상 배치 등록을 취소하고 있습니다.') if cancelled
                           else '해당 작업은 이미 완료됐습니다.')
            self.status.set(message)
            self.append('중단 요청: ' + ((result.get('reply') or {}).get('text') or result.get('status', '요청됨')))
        except Exception as exc:
            self.append('중단 요청 오류: ' + str(exc))

    def poll(self):
        self._poll_id = None
        self.service.interpreter.check_liveness()
        needs_draw = False
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'observation':
                    self.observation = value
                    needs_draw = True
                elif kind == 'motion_stop_requested':
                    target = value.get('target_request_id')
                    current_target = (self.chat.current_target() if self.chat is not None else
                                      self._record_target if self.current_action == 'record' else self._text_target)
                    current_target = current_target or self._motion_request_id
                    if not isinstance(target, str) or (current_target is not None and current_target != target):
                        continue
                    self._stopped_motion_requests.add(target)
                    self._display_stop_request = target
                    self.predicted_path = []
                    self.forecast_segments = []
                    self.motion_number = None
                    self.motion_session = None
                    self.forecast_label.set('중단 요청으로 이후 예상 경로를 지웠습니다.')
                    if not self.motion_interrupted:
                        self.motion_label.set('중단 요청 · 가상 명령 결과 확인 중')
                        self.status.set('가상 작업 중단을 요청했습니다. 결과는 아직 확인되지 않았습니다.')
                    needs_draw = True
                elif kind == 'motion':
                    request_id = value.get('request_id')
                    live_session = getattr(getattr(self.service, 'forecast', None), 'session_token', None)
                    if (live_session and value.get('forecast_session')
                            and value['forecast_session'] != live_session):
                        continue
                    current_target = (self.chat.current_target() if self.chat is not None else
                                      self._record_target if self.current_action == 'record' else self._text_target)
                    if request_id is not None and current_target is not None and request_id != current_target:
                        continue
                    session, number = value.get('forecast_session'), value.get('number')
                    if session and type(number) is int:
                        if number < self._latest_motion_numbers.get(session, 0):
                            continue
                        self._latest_motion_numbers[session] = number
                    if request_id is not None and self.service.motion_stop_requested(request_id):
                        self._stopped_motion_requests.add(request_id)
                    stopped = request_id is not None and request_id in self._stopped_motion_requests
                    if stopped and value['phase'] not in ('interrupted', 'end'):
                        self._display_stop_request = request_id
                        self.predicted_path = []
                        self.forecast_segments = []
                        self.motion_number = None
                        self.motion_session = None
                        self.motion_label.set('중단 요청 · 가상 명령 결과 확인 중')
                        needs_draw = True
                        continue
                    self._motion_request_id = request_id
                    self.observation = value['observation']
                    self.samples.append(self.observation['tcp_mm'])
                    self.samples = self.samples[-250:]
                    self.predicted_path = value['remaining_target_path_mm']
                    if value['phase'] == 'start':
                        self.motion_interrupted = False
                        snapshot = value.get('scene_snapshot')
                        self.show_snapshot(snapshot if isinstance(snapshot, dict) else {}, in_motion=True, draw=False)
                        self.motion_number = value['number']
                        self.motion_session = value.get('forecast_session')
                        self.forecast_segments = []
                        self.forecast_sequence = None
                        self.forecast_label.set('현재 관측을 바탕으로 이후 예상 경로를 준비하고 있습니다.')
                    if value['phase'] in ('end', 'interrupted') or value['cancel_requested'] or self.service.cancelled.is_set():
                        self.motion_number = None
                        self.motion_session = None
                        self.forecast_segments = []
                        self.forecast_label.set('현재 표시할 이후 예상 경로가 없습니다.')
                    if value['phase'] == 'interrupted':
                        self.motion_interrupted = True
                        self.motion_label.set('가상 명령 중단 · 정지 여부와 물체 배치 미확인')
                    elif value['phase'] == 'end' and (value['cancel_requested'] or stopped):
                        self.motion_label.set('가상 명령 종료 · 중단 결과 확인 중')
                    elif value['phase'] == 'end':
                        self.motion_label.set(f"가상 명령 {value['number']} 완료 · 다음 판단 대기")
                    elif value['cancel_requested']:
                        self.motion_label.set('중단 요청 · 현재 가상 명령이 끝나기를 기다리는 중')
                    else:
                        detail = ('팔 목표 경로' if value['arm_move'] else '집게 조작 · 팔 목표 경로 없음')
                        angles = ', '.join(f'{angle:.0f}°' for angle in value['target_joints_deg'])
                        sampling = '진행 표본' if value['sampling'] == 'during_command' else '명령 경계 표본'
                        self.motion_label.set(f"가상 명령 {value['number']} · {value['elapsed']:.1f}/{value['duration']:.1f}초 · {detail} · 목표각 {angles} · {sampling}")
                    needs_draw = True
                elif kind == 'forecast':
                    if (self.motion_session and value.get('forecast_session') == self.motion_session
                            and value.get('command_number') == self.motion_number and not self.service.cancelled.is_set()):
                        sequence = value.get('forecast_sequence')
                        if (type(sequence) is int and type(self.forecast_sequence) is int
                                and sequence < self.forecast_sequence):
                            continue
                        self.forecast_sequence = sequence
                        self.forecast_segments = value.get('segments', [])
                        if value.get('reason') == 'unavailable':
                            self.forecast_label.set('이후 예상 경로를 계산하지 못했습니다.')
                        elif value.get('scope') == 'program':
                            suffix = '추가 진행 없음으로 계산됨' if value.get('reason') == 'no_progress' else '실제 동작과 다를 수 있음'
                            self.forecast_label.set(f'검증된 요청의 남은 단계 경로 예상 · {suffix}')
                        elif value.get('scope') == 'program_horizon':
                            if value.get('reason') == 'future_unavailable':
                                self.forecast_label.set('이번 요청의 일부 경로만 예상 · 이후 단계 계산을 멈췄습니다.')
                            else:
                                repeat = ' · 이후 반복은 미계산' if value.get('repeat_mode') == 'continuous' else ''
                                completed = value.get('completed_round')
                                completed_text = (f'{completed}회차 예상 계산 완료 · '
                                    if type(completed) is int and completed > 0 and completed < value.get('through_round', 0)
                                    else '')
                                self.forecast_label.set(f"이번 요청의 일부 경로 예상 · {completed_text}{value.get('through_round')}회차 일부까지{repeat}")
                        else:
                            scope = '현재 작업에서 계획된 남은 이동' if value.get('scope') == 'planned_task' else '현재 원통 이동'
                            coverage = '예상' if value.get('complete') else '일부 예상'
                            progress = ' · 이후 경로 계산 중' if value.get('reason') == 'computing' else ''
                            self.forecast_label.set(f"{scope} {coverage} · {value.get('transfers', 1)}개 이동 기준{progress} · 실제 동작과 다를 수 있음")
                        if value.get('stale') and type(value.get('basis_command_number')) is int:
                            self.forecast_label.set(f"가상 명령 {value['basis_command_number']} 기준 · 이미 지난 경로 포함 가능 · 현재 명령으로 갱신 중 · "
                                                    + self.forecast_label.get())
                        needs_draw = True
                elif kind == 'forecast_skipped':
                    if (self.motion_session and value.get('forecast_session') == self.motion_session
                            and value.get('command_number') == self.motion_number and not self.service.cancelled.is_set()):
                        self.forecast_segments = []
                        self.forecast_label.set('이 명령의 예상 경로는 생략했습니다. 다음 명령에서 다시 시도합니다.')
                        needs_draw = True
                elif kind == 'forecast_clear':
                    if (self.motion_session and value.get('forecast_session') == self.motion_session
                            and value.get('command_number') == self.motion_number):
                        self.forecast_segments = []
                        self.motion_number = None
                        self.motion_session = None
                        self.forecast_label.set('중단 요청으로 이후 예상 경로를 지웠습니다.')
                        needs_draw = True
                elif kind == 'motion_error':
                    self.predicted_path = []
                    self.forecast_segments = []
                    self.motion_number = None
                    self.motion_session = None
                    self.forecast_label.set('가상 명령 오류 · 이후 경로 표시 중지')
                    self.motion_label.set('가상 명령 오류 · 경로 표시 중지')
                    needs_draw = True
                elif kind == 'status':
                    status_target = (self.chat.current_target() if self.chat is not None else self._text_target)
                    if ((self._display_stop_request is not None
                         and (status_target is None or status_target == self._display_stop_request))
                            or self.service.motion_stop_requested(status_target)):
                        continue
                    self.status.set(value)
                    if '실행 중' in value:
                        self.scene_label.set('이동 중 · 물체 배치 미확인')
                elif kind == 'llm_prepare':
                    self.llm_status.set(value['message'])
                    if value['state'] in ('ready', 'failed'):
                        self.append(value['message'])
                elif kind == 'interpretation':
                    source = '로컬 규칙' if value.get('source') == 'local_rule' else '로컬 E4B'
                    self.append('요청 해석: ' + source + ' · ' + value['status'])
                    if value['program']:
                        self.append('이해한 작업: ' + describe_program(value['program']))
                    elif value['detail']:
                        self.append(str(value['detail']))
                elif kind == 'transcript':
                    if not self.service.cancelled.is_set() and not self._record_cancelled:
                        self.voice_stage = 'interpreting'
                        self.replace_text(value['text'])
                        self.append('음성 인식: ' + value['text'])
                        self.show_voice_device(value)
                elif kind == 'chat_transcript':
                    if self.chat is not None and not self.chat.stop_event.is_set():
                        self.chat_error = None
                        if self.text.focus_get() is not self.text:
                            self.replace_text(value['text'])
                        self.append('음성 인식: ' + value['text'])
                        self.show_voice_device(value)
                elif kind == 'chat_utterance_start':
                    if self.chat is not None and not self.chat.stop_event.is_set():
                        if self.text.focus_get() is not self.text:
                            self.replace_text('')
                elif kind == 'chat_state':
                    chat_target = self.chat.current_target() if self.chat is not None else None
                    if (self._display_stop_request is not None
                            and chat_target in (None, self._display_stop_request)
                            and ('듣' in value or '음성 응답 중' in value)):
                        continue
                    self.status.set(value + ' 배치를 다시 등록해 주세요. 재등록하면 이전 기억이 지워집니다.'
                                    if self._recovery_required and '중단' in value else value)
                elif kind == 'chat_response':
                    self.append('앱 음성: ' + value)
                elif kind == 'voice_route':
                    labels = {'validated_program_proposal': '완성된 단문을 로컬 규칙으로 확인합니다.',
                              'full_command_interpretation': '로컬 E4B로 전체 요청을 해석합니다.',
                              'read_only_conversation': '일반 대화 · 가상 작업 실행 없음',
                              'acknowledge_prohibition': '금지 요청을 확인하고 실행하지 않습니다.',
                              'clarify_action_verb': '동작 표현을 다시 확인합니다.',
                              'clarify_memory_scope': '기억할 범위를 다시 확인합니다.'}
                    self.append(labels.get(value['kind'], '음성 요청 내용을 확인합니다.'))
                elif kind == 'chat_error':
                    coordinator = self.service.coordinator
                    if coordinator is not None:
                        try:
                            self.show_snapshot(coordinator.snapshot())
                        except Exception:
                            self.show_snapshot({})
                    self.chat_error = ('가상 실행 결과가 미확인입니다. 배치를 다시 등록해 주세요. '
                                       '재등록하면 이전 기억이 지워집니다.' if self._recovery_required else value)
                    self.status.set(self.chat_error)
                    self.append(value)
                    if self.chat_error != value:
                        self.append(self.chat_error)
                elif kind == 'chat_result':
                    self.show_snapshot(value['snapshot'])
                    execution = value.get('result')
                    if value.get('reply') is not None:
                        if not value['reply'].get('speak', True):
                            self.append('안내: ' + value['reply']['text'])
                    elif value.get('voice_feedback'):
                        self.append('요청 처리 안내가 있습니다.')
                    else:
                        self.append('가상 실행: ' + str(execution.get('status') if execution else value.get('admission')))
                elif kind == 'chat_closed':
                    self.chat = None
                    self.chat_button.configure(text='음성 대화 시작')
                    self.busy(False)
                    self.replace_text('')
                    self.status.set(self.chat_error or '음성 대화가 종료됐습니다.')
                elif kind == 'finished':
                    self.finished(*value)
                elif kind == 'text_stop_finished':
                    result, error = value
                    if error is not None:
                        self.status.set('정지 요청을 처리하지 못했습니다: ' + str(error))
                    else:
                        message = ((result.get('reply') or {}).get('text') or result.get('voice_feedback')
                                   or '정지 요청 결과를 확인해 주세요.')
                        self.status.set(message)
                        self.append(message)
                        self.show_snapshot(result.get('snapshot'))
        except queue.Empty:
            pass
        if needs_draw:
            self.draw()
        if self.closing:
            foreign = self.service.has_other_requests(self._closing_own_requests)
            if not foreign and not self._interpreter_close_requested:
                self.service.interpreter.request_close()
                self._interpreter_close_requested = True
            if (not foreign and self.chat is None and (self.worker is None or not self.worker.is_alive())
                    and (self.text_control_worker is None or not self.text_control_worker.is_alive())):
                if self.cleanup_thread is None:
                    self.cleanup_thread = threading.Thread(target=self.service.close, daemon=True,
                                                           name='demo-cleanup')
                    self.cleanup_thread.start()
                elif not self.cleanup_thread.is_alive():
                    self.root.destroy()
                    return
        self._poll_id = self.root.after(35, self.poll)

    def finished(self, action, result, error, auxiliary=None):
        if auxiliary is not None and auxiliary is self._aux_operation:
            self._aux_operation = None
        if action == 'submit':
            self._text_target = None
        if action == 'record':
            self._record_target = None
        self.busy(False)
        if action in ('submit', 'record'):
            self.predicted_path = []
            self.forecast_segments = []
            self.motion_number = None
            self.motion_session = None
            self.forecast_label.set('현재 표시할 이후 예상 경로가 없습니다.')
            if not self.motion_interrupted:
                self.motion_label.set('현재 표시할 예상 목표 경로가 없습니다.')
        if error is not None:
            if action in ('record', 'audio'):
                self.replace_text('')
            self.voice_stage = None
            self.status.set(('음성 변환 결과를 취소했습니다.' if action == 'audio' else '요청을 취소했습니다.')
                            if isinstance(error, InterruptedError)
                            else '완료하지 못했습니다. 자세한 내용을 확인해 주세요.')
            self.append(f'{type(error).__name__}: {error}')
            if self.service.coordinator is not None:
                self.show_snapshot(self.service.coordinator.snapshot())
            elif action == 'register':
                self.scene_label.set('미등록 · 화면은 설정값입니다')
            if action == 'register':
                self.finger_profile_label.set('집게 열기가 등록되지 않았습니다. 다시 등록해 주세요.')
                self.control_timing_label.set('이동 시간이 등록되지 않았습니다. 다시 등록해 주세요.')
                self.closed_move_profile_label.set('이동 곡선이 등록되지 않았습니다. 다시 등록해 주세요.')
            return
        if action == 'register':
            self.model_pending = False
            self._recovery_required = False
            self._recovery_guarded = False
            self.busy(False)
            self.show_snapshot(result['snapshot'])
            self.model_label.set(Path(result['model_path']).name + ' · ' + result['model_sha256'][:12])
            self.finger_profile_label.set('적용됨 · ' + FINGER_OPEN_PROFILES[result['finger_open_profile']])
            self.control_timing_label.set('적용됨 · ' + CONTROL_TIMING_PROFILES[result['control_timing_profile']])
            self.closed_move_profile_label.set('적용됨 · ' + CLOSED_MOVE_PROFILES[result['closed_move_profile']]
                + (' · 받침 위 직선 하강' if result.get('straight_descent') else ''))
            warnings = result.get('post_commit_warnings') or []
            self.status.set('가상 시작 배치는 등록됐지만 부가 처리에 오류가 있었습니다. 로그를 확인해 주세요.'
                            if warnings else '가상 시작 배치를 등록했습니다. 물체의 실제 상태는 확인하지 않았습니다.')
            for warning in warnings:
                self.append('등록 후 오류 (' + str(warning.get('stage')) + '): ' + str(warning.get('detail')))
            self.append('학생 모델 적용 · SHA-256 ' + result['model_sha256'])
            self.append('가상 집게 열기 적용 · ' + FINGER_OPEN_PROFILES[result['finger_open_profile']])
            self.append('가상 잡고 이동 곡선 적용 · ' + CLOSED_MOVE_PROFILES[result['closed_move_profile']])
        elif action == 'audio':
            if self.service.cancelled.is_set():
                self.status.set('음성 변환 결과를 취소했습니다.')
                return
            self.replace_text(result['text'])
            self.status.set('인식한 문장을 확인한 뒤 가상 실행을 누르세요.')
            self.append('음성 인식: ' + result['text'])
            self.show_voice_device(result)
        else:
            if action == 'record':
                self.voice_stage = None
            self.show_snapshot(result['snapshot'])
            execution = result.get('result')
            if result.get('reply') is not None:
                self.status.set(result['reply']['text'])
                self.append('안내: ' + result['reply']['text'])
            elif execution is None:
                feedback = result.get('voice_feedback')
                self.status.set(feedback or '요청을 실행하지 않았습니다. 해석 결과를 확인해 주세요.')
                self.append('안내: ' + feedback if feedback else '접수 결과: ' + str(result.get('admission')))
            else:
                status = execution.get('status', 'unknown')
                if status == 'completed':
                    self.status.set('동작을 마쳤습니다.')
                elif status == 'retargeted_expected':
                    self.status.set('변경한 동작을 마쳤습니다.')
                else:
                    self.status.set('가상 실행 상태: ' + status)
                self.append('가상 실행: ' + status + ((' · ' + str(execution.get('detail'))) if execution.get('detail') else ''))
            if self._recovery_required:
                self.status.set(self.status.get().rstrip() +
                                ' 배치를 다시 등록해 주세요. 재등록하면 이전 기억이 지워집니다.')

    def show_snapshot(self, snapshot, *, in_motion=False, draw=True):
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        state = snapshot.get('scene_state') or {}
        expected = state.get('expected_scene')
        if expected is not None:
            self.layout = expected
        confirmation = state.get('confirmation')
        context = snapshot.get('reference_context') or {}
        guarded = (snapshot.get('session_status') == 'uncertain'
                   or context.get('execution_state') == 'uncertain'
                   or snapshot.get('stop_confirmation') == 'unconfirmed')
        # The coordinator rejects work while these flags are set. Keep
        # read-only conversation available only when that protection is known.
        if state or not in_motion:
            self._recovery_guarded = guarded
            self._recovery_required = (self._recovery_required or guarded or expected is None
                                       or confirmation not in ('explicit_registration', 'unconfirmed_after_motion'))
        if expected is None:
            self.layout = [[], [], [], []]
            label = '배치 표시 불가 · 다시 등록 필요'
        elif self._recovery_required:
            label = '마지막 예상 배치 · 이동 결과 미확인 · 다시 등록 필요'
        elif in_motion:
            label = '가상 조작 중 · 마지막 예상 배치 · 현재 물체 위치 미확인'
        elif confirmation == 'explicit_registration':
            label = '등록된 가상 시작 배치 · 물체 실측 없음'
        elif confirmation == 'unconfirmed_after_motion':
            label = '기대 배치 · 물체 위치 미확인'
        else:
            label = '배치 미확인 · 재등록 필요'
        self.scene_label.set(label)
        if self._recovery_required:
            self.status.set('가상 실행 결과가 미확인입니다. 배치를 다시 등록해 주세요. 재등록하면 이전 기억이 지워집니다.')
        self.busy(self._busy)
        if draw:
            self.draw()

    def draw(self):
        c = self.canvas
        c.delete('all')
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        cx, cy = w * .50, h * .72
        scale = min(w / 500, h / 420)
        def xy(point):
            return cx + point[0] * scale, cy - point[1] * scale
        c.create_rectangle(20, 14, w - 20, h - 14, fill='#d8e8e1', outline='#b9d1c7', width=2)
        for i, (point, stack) in enumerate(zip(SLOTS, self.layout)):
            x, y = xy(point)
            c.create_oval(x-29, y-29, x+29, y+29, fill='#edf5f1', outline='#a4c4b7', width=2)
            c.create_text(x, y+44, text=f'{i+1}번', fill='#537469', font=('맑은 고딕', 10, 'bold'))
            for level, obj in enumerate(stack):
                offset = level * 13
                c.create_oval(x-19, y-16-offset, x+19, y+17-offset, fill=COLORS[obj-1], outline='white', width=2)
                c.create_text(x, y-offset, text=str(obj), fill='white', font=('맑은 고딕', 11, 'bold'))
        for segment in self.forecast_segments:
            if len(segment) >= 2:
                coordinates = [value for point in segment for value in xy(point)]
                c.create_line(*coordinates, fill='#9869a8', width=3, dash=(2, 6))
        if len(self.predicted_path) >= 2:
            coordinates = [value for point in self.predicted_path for value in xy(point)]
            c.create_line(*coordinates, fill='#307ac5', width=3, dash=(7, 5))
            x, y = xy(self.predicted_path[-1])
            c.create_oval(x-7, y-7, x+7, y+7, outline='#307ac5', width=2)
        for sample in self.samples:
            x, y = xy(sample)
            c.create_oval(x-2, y-2, x+2, y+2, fill='#d4894b', outline='')
        if self.observation:
            points = [xy(p) for p in robot_points(self.observation['joints_deg'])]
            for a, b in zip(points, points[1:]):
                c.create_line(*a, *b, fill=TEAL, width=9, capstyle='round')
            for x, y in points:
                c.create_oval(x-6, y-6, x+6, y+6, fill='#f8fcfa', outline=TEAL, width=3)
            x, y = xy(self.observation['tcp_mm'])
            c.create_oval(x-6, y-6, x+6, y+6, fill='#e5a05b', outline='white', width=2)
        c.create_text(30, 27, anchor='w', text='위에서 본 가상 장면 · 도식', fill='#537469', font=('맑은 고딕', 9))

    def close(self):
        if self.closing:
            return
        self.closing = True
        self._closing_own_requests = {self._text_target, self._record_target}
        if self.current_action in ('register', 'audio') and self._aux_operation is not None:
            self._closing_own_requests.add(self._aux_operation.request_id)
        if self.chat is not None:
            self._closing_own_requests.add(self.chat.current_target())
            self._closing_own_requests.add(getattr(self.chat, '_session_id', None))
        self.service.begin_shutdown()
        if self.chat is not None:
            self.chat.stop()
        else:
            self.stop()
        # Hide immediately. If another caller owns work, let it finish before
        # terminating the shared model; no new service request can start now.
        self.root.withdraw()
        if not self.service.has_other_requests(self._closing_own_requests):
            self.service.interpreter.request_close()
            self._interpreter_close_requested = True
        self.status.set('가상 작업과 로컬 모델을 정리하고 있습니다.')


def main():
    root = tk.Tk()
    DemoWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
