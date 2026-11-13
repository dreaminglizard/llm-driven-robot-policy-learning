(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const slots = [...document.querySelectorAll('.slot-input')];
  const coordinateInputs = [1, 2, 3, 4].map((index) => [$(`slot-${index}-x`), $(`slot-${index}-y`)]);
  const coordinateFields = coordinateInputs.flat();
  const editedCoordinates = new Set();
  // Do not offer the HTML's example coordinates as a connected profile.
  coordinateFields.forEach((input) => { input.value = ''; });
  const token = new URLSearchParams(location.search).get('token') || '';
  const modelSelect = $('model-select');
  const modeSelect = $('mode-select');
  const deviceConnect = $('device-connect');
  const devicePrepare = $('device-prepare');
  const physicalJoints = [1, 2, 3, 4].map((index) => $(`physical-joint-${index}`));
  const physicalReady = $('physical-ready');
  const useConfirmedTarget = $('use-confirmed-target');
  const recoveryConfirm = $('recovery-confirm');
  const controlTimingSelect = $('control-timing-select');
  const controlTimingNames = {demo_1s: '시연 시간 · 1초', app_2s: '기존 앱 시간 · 2초'};
  const fingerProfileSelect = $('finger-profile-select');
  const fingerProfileNames = {instant: '즉시 열기', minimum_jerk_0p5s: '부드럽게 열기 · 0.5초'};
  const closedMoveProfileSelect = $('closed-move-profile-select');
  const closedMoveProfileNames = {minimum_jerk: '기존 곡선', smoothstep: '부드러운 3차 곡선'};
  const command = $('command-text');
  const registerButton = $('register-button');
  const submitButton = $('submit-button');
  const chatButton = $('chat-button');
  const stopButton = $('stop-button');
  const mobileStopButton = $('mobile-stop-button');
  const language = $('language-select');
  // [top face, side] colours of cylinders 1-4 on the dark stage.
  const cylinderColors = [['#fbbf24', '#b45309'], ['#38bdf8', '#0369a1'], ['#f472b6', '#be185d'], ['#a78bfa', '#6d28d9']];
  const modeButtons = {virtual: $('mode-virtual'), mock: $('mode-mock'), physical: $('mode-physical')};
  // Layout presets only fill the inputs; the user still registers explicitly.
  const presets = [[$('preset-default'), ['1', '2', '3', '4']], [$('preset-stacked'), ['1, 2', '', '3, 4', '']]];
  // Example chips only fill the text box; nothing is submitted until the user runs it.
  const examples = [[$('example-1'), '3번에서 1번으로 옮겨 줘'], [$('example-2'), '지금 배치 기억해'],
    [$('example-3'), '초기 배치로 돌아가'], [$('example-4'), '아까대로 복원해']];
  let state = null;
  let initialStateSeen = false;
  let dirty = true;
  let editRevision = 0;
  let layoutEdited = false;
  // Only a registration adopted by this page can be restored by reverting
  // layout/coordinate edits. Other changes and uncertain registration results
  // must keep their existing explicit recovery behavior.
  let formRestoreEligible = false;
  let registrationRequest = null;
  let awaitingRegisteredCoordinates = false;
  let registrationRevision = null;
  let registrationChangedElsewhere = false;
  let pending = false;
  let stopPending = false;
  const stoppedMotionRequests = new Set();
  const stoppedControlCommands = new Set();
  let connected = false;
  let ending = false;
  let refreshIssued = 0;
  let refreshApplied = 0;
  let refreshBarrier = 0;
  let lastViewRevision = null;
  let toastTimer = null;
  let autoTarget = null;
  let targetEdited = false;
  const point = ([x, y]) => [350 + x * 1.32, 370 - y * 1.32];
  const virtualPositions = [[152.149, 150.349], [149.651, 2.149], [-152.149, 149.651], [-149.651, -2.149]];
  const modeNames = {virtual: '가상', mock: '모의', physical: '실물'};
  const virtualForecastNote = $('forecast-note').textContent;
  const currentMode = () => state?.control?.mode || state?.mode || 'physical';
  const modeName = () => modeNames[currentMode()] || '실물';
  // The server ties this to its own unclaimed pending start. Keep the
  // underlying recovery and registration guards intact.
  const pendingConnectionRetry = () => {
    const control = state?.control || {};
    const controller = control.controller;
    const activeCommand = controller?.active_command_id;
    const noOwner = !controller && !control.ownership;
    const healthyOwner = control.ownership === 'resident_retained' && controller
      && controller.fault === null && controller.worker_alive === true && controller.fresh_feedback === true
      && ['idle', 'armed', 'executing', 'stopped'].includes(controller.state)
      && [null, 'demo-connect-stop-check', 'demo-connect-initial-pose'].includes(controller.lease_owner)
      && (activeCommand == null || activeCommand === control.current_command?.command_id);
    return connected && currentMode() === 'physical' && state?.pending_connect_retry === true
      && !state?.registered && !state?.stop_requested && !state?.closing && state?.recovery_required
      && control.physical_recovery_required && !control.connected && (noOwner || healthyOwner)
      && ['disconnected', 'connecting'].includes(control.connection);
  };
  const controllerConnected = () => currentMode() === 'virtual' || state?.control?.connected === true;
  // A temporary, registered-session feedback interruption is handled by the
  // service: it retains a new request until this same controller returns.
  // Do not treat a real fault or an unregistered/uncertain scene as available.
  const waitingForController = () => {
    const control = state?.control || {};
    return Boolean(connected && currentMode() === 'physical' && state?.registered
      && !state?.recovery_required && control.status_read_recovery_pending === true
      && control.connection !== 'fault' && !control.controller?.fault
      && control.controller?.worker_alive !== false);
  };
  const requestConnectionAvailable = () => {
    const control = state?.control || {};
    if (currentMode() === 'physical' && (control.connection === 'fault'
        || control.controller?.fault || control.controller?.worker_alive === false
        || (control.physical_recovery_required && !waitingForController()))) return false;
    return controllerConnected() || waitingForController();
  };
  const freshObservation = () => {
    const observation = state?.observation;
    return connected && controllerConnected() && observation && observation.sensor_valid !== false
      ? observation : null;
  };
  const controlWasStopped = () => Boolean(state?.text_target &&
    (state.text_target === state.motion_stop_target || stoppedMotionRequests.has(state.text_target)));
  const retargetUnconfirmed = () => state?.control?.current_command?.status === 'retarget_interrupt_unconfirmed';
  const currentControlCommand = () => currentMode() !== 'virtual' && connected && controllerConnected()
    && !stopPending && !state?.stop_requested && !state?.recovery_required
    && state?.control?.connection !== 'fault' && !controlWasStopped() && !retargetUnconfirmed()
    && !stoppedControlCommands.has(state?.control?.current_command?.command_id)
    ? state?.control?.current_command : null;
  const retargetProgress = () => {
    if (currentMode() === 'virtual' || !connected || stopPending || state?.stop_requested
        || state?.recovery_required || controlWasStopped() || retargetUnconfirmed()) return null;
    const sceneState = state?.control?.scene_state;
    const retarget = sceneState?.retarget;
    const redirectedSlot = sceneState?.unresolved_transfer?.redirect_destination_slot;
    if (Number.isInteger(redirectedSlot)) return `새 목표 경로 시작 중 · 목적지 ${redirectedSlot}번 자리`;
    if (retarget?.status === 'waiting') return '목표 변경 요청 전달 중';
    if (retarget?.status === 'requested') return '목표 변경 요청 접수 · 적용 대기 중';
    const targetSlot = retarget?.destination_slot ?? retarget?.source_slot;
    if (retarget?.status === 'applied') return Number.isInteger(targetSlot)
      ? `목표 변경 적용 · 목적지 ${targetSlot}번 자리` : '목표 변경 적용';
    return null;
  };
  const currentControlForecast = () => {
    const current = currentControlCommand();
    const forecast = state?.control?.forecast;
    return current && forecast?.basis_command_id === current.command_id ? forecast : null;
  };
  const preparationReady = () => {
    const observation = freshObservation();
    const controller = state?.control?.controller;
    return Boolean(observation && observation.sensor_valid === true && observation.moving === false
      && observation.gripper_command === 'open' && observation.gripper_reported_state === 'open'
      && observation.gripper_reported_valid === true && observation.gripper_feedback_fresh !== false
      && (!controller || (!controller.fault && ['idle', 'stopped'].includes(controller.state))));
  };
  // An operating arm or closed gripper is expected during a registered task;
  // do not confuse that with the separate preparation needed to register.
  const activePhysicalOperation = () => {
    const control = state?.control;
    return Boolean(currentMode() === 'physical' && connected && state?.registered
      && control?.connected === true && control.connection === 'connected'
      && !control.physical_recovery_required && !control.controller?.fault
      && control.controller?.state !== 'fault' && !state?.recovery_required
      && !state?.stop_requested && !retargetUnconfirmed()
      && freshObservation()?.sensor_valid === true
      && (state?.busy === 'submit' || state?.text_target || state?.arm_home_active));
  };
  const physicalTargetsValid = () => physicalJoints.every((input) => input.value.trim() !== '' && Number.isFinite(Number(input.value)));
  const confirmedTarget = () => {
    const control = state?.control;
    const target = control?.confirmed_applied_target;
    return currentMode() === 'physical' && connected && control?.connected === true && control?.connection === 'connected'
      && !control?.physical_recovery_required && typeof target?.instance_id === 'string'
      && target.instance_id === control?.controller?.instance_id && typeof target?.command_id === 'string'
      && Array.isArray(target?.joints_deg) && target.joints_deg.length === 4 && target.joints_deg.every(Number.isFinite)
      ? target : null;
  };

  function numbers(values, suffix) {
    return Array.isArray(values) && values.length && values.every(Number.isFinite)
      ? values.map((value) => `${Number(value).toFixed(1)}${suffix}`).join(' · ') : null;
  }

  function renderReadings() {
    const observation = freshObservation();
    $('observation-label').textContent = `${modeName()} 로봇 상태${currentMode() === 'virtual' ? '' : ' · 최신 수신'}`;
    $('joint-reading').textContent = observation
      ? (numbers(observation.joints_deg, '°') || '관절 상태 확인 불가') : '관절 상태 확인 불가';
    let gripper = '집게 상태 확인 불가';
    if (observation && currentMode() === 'virtual' && Number.isFinite(observation.gripper_opening_m)) {
      gripper = `가상 집게 간격 ${(observation.gripper_opening_m * 1000).toFixed(1)} mm`;
    } else if (observation && observation.gripper_reported_valid === true && observation.gripper_feedback_fresh !== false) {
      const reported = {open: '열림', closed: '닫힘'}[observation.gripper_reported_state] || '상태 확인 불가';
      gripper = `집게 보고 · ${reported}`;
    }
    $('gripper-reading').textContent = gripper;
    const motion = currentMode() === 'virtual' && connected && !stopPending && !state?.stop_requested
      && !motionWasStopped(state?.motion) && !['end', 'interrupted'].includes(state?.motion?.phase)
      ? state?.motion : null;
    const target = motion || currentControlCommand();
    const joints = numbers(target?.target_joints_deg, '°');
    const opening = Number.isFinite(target?.target_opening_m) ? `집게 목표 ${(target.target_opening_m * 1000).toFixed(1)} mm` : null;
    const gripperTarget = target?.gripper_state ? `집게 목표 ${target.gripper_state === 'open' ? '열기' : target.gripper_state === 'closed' ? '닫기' : target.gripper_state}` : null;
    $('target-reading').textContent = target ? ([joints && `관절 ${joints}`, opening || gripperTarget].filter(Boolean).join(' · ') || '목표 확인 중') : '표시할 목표 없음';
  }

  function toast(message) {
    const node = $('toast');
    node.textContent = displayMessage(message);
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 7000);
  }

  function displayMessage(message) {
    return String(message).replace(/the requested quantity does not fit the source and destination/gi,
      '출발 자리의 원통 수와 도착 자리의 빈 공간을 확인해 주세요.');
  }

  async function request(path, body, signal) {
    const options = {headers: {'X-Demo-Token': token}, cache: 'no-store'};
    if (signal) options.signal = signal;
    if (body !== undefined) {
      options.method = 'POST';
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(data.error || '요청을 처리하지 못했습니다.');
      error.status = response.status;
      error.code = data.code;
      throw error;
    }
    return data;
  }

  async function requestState() {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    try {
      return await request('/api/state', undefined, controller.signal);
    } finally {
      clearTimeout(timeout);
    }
  }

  async function act(path, body, accepted, rejected) {
    if (pending || ending) return;
    pending = true;
    render();
    try {
      const result = await request(path, body);
      if (accepted) accepted(result);
      await refresh();
    } catch (error) {
      if (rejected) rejected(error);
      toast(error.message || '로컬 시연에 연결할 수 없습니다.');
    } finally {
      pending = false;
      render();
    }
  }

  function inputLayout() {
    return slots.map((input) => input.value.trim());
  }

  function previewLayout() {
    return inputLayout().map((value) => value ? value.split(',').map((part) => Number(part.trim())).filter((n) => [1, 2, 3, 4].includes(n)).slice(0, 2) : []);
  }

  function coordinateValid(input) {
    return input.value.trim() !== '' && Number.isFinite(Number(input.value));
  }

  function inputCoordinates() {
    return coordinateInputs.map((pair) => pair.map((input) => Number(input.value)));
  }

  function validCoordinatePairs(xy) {
    return Array.isArray(xy) && xy.length === 4
      && xy.every((pair) => Array.isArray(pair) && pair.length === 2 && pair.every(Number.isFinite))
      && xy.every((pair, index) => xy.slice(0, index).every((other) => pair[0] !== other[0] || pair[1] !== other[1]));
  }

  function calibrationCoordinates() {
    const xy = state?.control?.calibration?.slot_xy_mm;
    return validCoordinatePairs(xy) ? {xy} : null;
  }

  function fillCoordinates(coordinates, preserveEdits = false) {
    coordinateInputs.forEach((pair, index) => pair.forEach((input, axis) => {
      if (!preserveEdits || !editedCoordinates.has(input)) input.value = String(coordinates.xy[index][axis]);
    }));
  }

  function coordinateProblem() {
    const invalid = coordinateInputs.map((pair) => pair.map((input) => !coordinateValid(input)));
    if (invalid.some((pair) => pair.some(Boolean))) {
      return {message: '네 자리의 X·Y에 숫자를 입력해 주세요.', invalid};
    }
    const xy = inputCoordinates();
    const duplicate = xy.map((pair, index) => xy.some((other, otherIndex) => otherIndex !== index
      && pair[0] === other[0] && pair[1] === other[1]));
    return {message: duplicate.some(Boolean) ? '자리 좌표는 서로 달라야 합니다.' : '',
      invalid: duplicate.map((bad) => [bad, bad])};
  }

  // Mirrors demo_web.parse_layout so the problem is visible before registering; the server still validates.
  function layoutProblem() {
    const bad = [false, false, false, false];
    const stacks = inputLayout().map((value) => value ? value.split(',').map((part) => part.trim()) : []);
    let message = '';
    stacks.forEach((parts, index) => {
      if (parts.some((part) => !['1', '2', '3', '4'].includes(part))) {
        bad[index] = true;
        message ||= `${index + 1}번 자리: 원통 번호는 1~4이며, 쉼표로 아래층부터 입력해 주세요.`;
      } else if (parts.length > 2) {
        bad[index] = true;
        message ||= `${index + 1}번 자리: 한 자리에는 최대 2개까지 쌓을 수 있습니다.`;
      }
    });
    if (!message) {
      const seen = new Map();
      stacks.forEach((parts, index) => parts.forEach((part) => {
        if (seen.has(part)) {
          bad[index] = true;
          bad[seen.get(part)] = true;
          message ||= `원통 ${part}번이 두 자리에 입력됐습니다. 번호는 중복 없이 입력해 주세요.`;
        } else {
          seen.set(part, index);
        }
      }));
    }
    if (!message && !stacks.some((parts) => parts.length)) message = '원통을 한 개 이상 입력해 주세요.';
    return {message, bad};
  }

  function setStep(id, {done = false, active = false, running = false, warn = false}, sub) {
    const node = $(id);
    node.classList.toggle('done', done);
    node.classList.toggle('active', active);
    node.classList.toggle('running', running);
    node.classList.toggle('warn', warn);
    $(`${id}-sub`).textContent = sub;
  }

  function renderSteps(ready, problem) {
    const control = state?.control || {};
    const mode = currentMode();
    const linked = connected && controllerConnected();
    const registered = connected && Boolean(state?.registered) && !dirty && !awaitingRegisteredCoordinates;
    const deviceReady = mode === 'virtual' ? connected : linked && control.connection !== 'fault'
      && (preparationReady() || registered || activePhysicalOperation());
    const pendingConnection = pendingConnectionRetry();
    const recovery = connected && Boolean(state?.recovery_required) && !pendingConnection;
    const working = connected && Boolean(state?.busy || state?.chat);
    const modeSub = !connected ? '로컬 연결 확인 불가'
      : mode === 'virtual' ? '가상 시뮬레이션'
      : control.connection === 'fault' ? '제어기 오류 · 확인 필요'
      : pendingConnection ? (state?.pending_connect_unknown && !state?.busy ? '제어기 연결 확인 필요' : '연결 준비 중')
      : waitingForController() ? '연결 복귀 대기'
      : !control.connected ? `${modeName()} 연결 필요`
      : !deviceReady ? '집게 열기 준비 필요' : `${modeName()} 연결됨`;
    setStep('step-mode', {done: deviceReady, active: !deviceReady, warn: connected && control.connection === 'fault'}, modeSub);
    const layoutSub = recovery ? '다시 등록 필요'
      : awaitingRegisteredCoordinates && !dirty ? '자리 좌표 확인 중'
      : registrationChangedElsewhere ? '다른 화면에서 변경됨'
      : registered ? '등록됨'
      : problem.message ? '입력 확인 필요'
      : dirty && state?.registered ? '변경됨 · 다시 등록' : '등록 전';
    setStep('step-layout', {done: registered && !recovery, active: deviceReady && (!registered || recovery),
      warn: recovery || registrationChangedElsewhere || Boolean(problem.message)}, layoutSub);
    const commandSub = !connected ? '상태 확인 불가'
      : state?.busy === 'register' ? '등록 중'
      : state?.busy ? '실행 중'
      : state?.chat ? '음성 대화 중'
      : waitingForController() && ready ? '연결 복귀 후 실행'
      : ready ? '명령 가능' : '등록 후 가능';
    setStep('step-command', {active: ready || working, running: working}, commandSub);
  }

  function renderAdvancedSummary() {
    const node = $('advanced-summary');
    const allDefault = modelSelect.value === 'close' && controlTimingSelect.value === 'auto'
      && fingerProfileSelect.value === 'auto' && closedMoveProfileSelect.value === 'auto';
    let text = allDefault ? '학생별 기본' : '선택 변경됨';
    if (currentMode() !== 'virtual') text = '제어기 기본값';
    else if (connected && state?.registered && !dirty) text = '적용됨';
    else if (connected && state?.registered && dirty) text = '다시 등록 시 적용';
    node.textContent = text;
    node.classList.toggle('applied', text === '적용됨');
    node.classList.toggle('changed', text === '다시 등록 시 적용' || text === '선택 변경됨');
  }

  // A freshly opened page adopts the server's current registration instead of forcing a re-registration
  // (which would erase the robot's memory). Later changes from other tabs still require re-registration.
  function adoptRegistration() {
    registrationRevision = state.registration_revision;
    registrationChangedElsewhere = false;
    dirty = false;
    layoutEdited = false;
    formRestoreEligible = true;
    if (Array.isArray(state.layout) && state.layout.length === 4) {
      slots.forEach((input, index) => { input.value = Array.isArray(state.layout[index]) ? state.layout[index].join(', ') : ''; });
    }
    const coordinates = calibrationCoordinates();
    awaitingRegisteredCoordinates = currentMode() === 'physical' && !coordinates;
    if (coordinates) fillCoordinates(coordinates);
    // Any earlier coordinate edits are part of the registration just adopted.
    // Later edits are tracked separately while coordinates are unavailable.
    editedCoordinates.clear();
    const model = (state.models || []).find((candidate) => candidate.name === state.model?.name);
    if (model) modelSelect.value = model.id;
    if (controlTimingNames[state.control_timing_profile]) controlTimingSelect.value = state.control_timing_profile;
    if (fingerProfileNames[state.finger_open_profile]) fingerProfileSelect.value = state.finger_open_profile;
    if (closedMoveProfileNames[state.closed_move_profile]) closedMoveProfileSelect.value = state.closed_move_profile;
  }

  function svgElement(name, attributes = {}, text = '') {
    const node = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    if (text) node.textContent = text;
    return node;
  }

  function add(parent, name, attrs, text) {
    const node = svgElement(name, attrs, text);
    parent.appendChild(node);
    return node;
  }

  function drawScene() {
    const groups = ['scene-objects', 'scene-forecast', 'scene-path', 'scene-samples', 'scene-robot'].map($);
    groups.forEach((group) => group.replaceChildren());
    const [objects, forecast, path, samples, robot] = groups;
    // Keep diagram labels readable as the SVG scales down on narrow screens.
    const labelScale = Math.max(1, Math.min(2.25, 700 / Math.max($('scene').clientWidth || 700, 1)));
    const layout = dirty || !state ? previewLayout() : state.layout;
    const positions = currentMode() === 'physical'
      ? (awaitingRegisteredCoordinates && !dirty ? [] : coordinateInputs.map((pair) =>
        pair.every(coordinateValid) ? pair.map((input) => Number(input.value)) : null))
      : virtualPositions;
    positions.forEach((position, index) => {
      if (!position) return;
      const [x, y] = point(position);
      add(objects, 'circle', {cx: x, cy: y, r: 44, fill: '#16233b', stroke: '#3b5079', 'stroke-width': 1.5});
      add(objects, 'circle', {cx: x, cy: y, r: 34, fill: 'none', stroke: '#2f4266', 'stroke-width': 1, 'stroke-dasharray': '4 5'});
      add(objects, 'text', {x, y: y + 66, 'text-anchor': 'middle', fill: '#b4c2dc', 'font-size': 14 * labelScale, 'font-weight': 700}, `${index + 1}번 자리`);
      const stack = Array.isArray(layout?.[index]) ? layout[index] : [];
      if (!stack.length) {
        add(objects, 'text', {x, y: y + 4, 'text-anchor': 'middle', fill: '#5a6f96', 'font-size': 11 * labelScale, 'font-weight': 600}, '비어 있음');
      }
      stack.forEach((object, level) => {
        const cy = y + 6 - level * 14;
        const [top, side] = cylinderColors[object - 1] || ['#94a3b8', '#475569'];
        const size = 15 * labelScale;
        add(objects, 'ellipse', {cx: x, cy: cy + 8, rx: 25, ry: 21, fill: side});
        add(objects, 'ellipse', {cx: x, cy, rx: 25, ry: 21, fill: top, stroke: '#f8fafc', 'stroke-width': 2});
        add(objects, 'text', {x, y: cy + size * 0.36, 'text-anchor': 'middle', fill: '#0f172a', 'font-size': size, 'font-weight': 800}, String(object));
      });
      if (stack.length > 1) {
        add(objects, 'text', {x: x + 30, y: y - 30, 'text-anchor': 'start', fill: '#5eead4', 'font-size': 11 * labelScale, 'font-weight': 800}, `${stack.length}층`);
      }
    });
    const showObserved = freshObservation();
    const showPrediction = showObserved && !stopPending && !state?.stop_requested && !motionWasStopped(state?.motion)
      && !controlWasStopped() && !['end', 'interrupted'].includes(state?.motion?.phase) && !state?.motion?.cancel_requested;
    const preview = showPrediction ? (currentMode() === 'virtual' ? state.path : (currentControlCommand() ? state?.control?.remaining_target_path_mm : [])) : [];
    const forecastSegments = showPrediction ? (currentMode() === 'virtual' ? state?.forecast?.segments : currentControlForecast()?.segments) : [];
    if (Array.isArray(forecastSegments)) {
      forecastSegments.forEach((segment) => {
        if (!Array.isArray(segment) || segment.length < 2) return;
        const points = segment.map((p) => point(p).join(',')).join(' ');
        add(forecast, 'polyline', {points, fill: 'none', stroke: '#c084fc', 'stroke-width': 3,
          'stroke-dasharray': '2 8', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
      });
    }
    if (preview?.length >= 2) {
      const points = preview.map((p) => point(p).join(',')).join(' ');
      add(path, 'polyline', {points, fill: 'none', stroke: '#38bdf8', 'stroke-width': 3, 'stroke-dasharray': '9 7', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
      const [x, y] = point(preview[preview.length - 1]);
      add(path, 'circle', {cx: x, cy: y, r: 11, fill: 'none', stroke: '#38bdf8', 'stroke-width': 3});
    }
    if (showObserved) {
      (state.samples || []).forEach((sample) => {
        const [x, y] = point(sample);
        add(samples, 'circle', {cx: x, cy: y, r: 3, fill: '#fb923c', opacity: .9});
      });
      const joints = showObserved.joints_deg;
      if (Array.isArray(joints) && joints.length >= 4) {
        const [q1, q2, q3, q4] = joints.map((value) => value * Math.PI / 180);
        const pitch = q2 + q3 + q4 + Math.PI / 2;
        const radii = [0, -100 * Math.sin(q2), -100 * Math.sin(q2) - 100 * Math.sin(q2 + q3)];
        radii.push(radii[2] + 80 * Math.cos(pitch) + 6 * Math.sin(pitch));
        const points = radii.map((r) => point([-r * Math.sin(q1), r * Math.cos(q1)]));
        const line = points.map((p) => p.join(',')).join(' ');
        add(robot, 'circle', {cx: 350, cy: 370, r: 26, fill: '#1e293b', stroke: '#5eead4', 'stroke-width': 2});
        add(robot, 'polyline', {points: line, fill: 'none', stroke: '#14b8a6', 'stroke-opacity': .45, 'stroke-width': 20, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
        add(robot, 'polyline', {points: line, fill: 'none', stroke: '#e2e8f0', 'stroke-width': 10, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
        points.forEach(([x, y]) => add(robot, 'circle', {cx: x, cy: y, r: 7, fill: '#0f172a', stroke: '#5eead4', 'stroke-width': 3}));
      }
      if (Array.isArray(showObserved.tcp_mm)) {
        const [x, y] = point(showObserved.tcp_mm);
        add(robot, 'circle', {cx: x, cy: y, r: 8, fill: '#fb923c', stroke: '#fff7ed', 'stroke-width': 2.5});
      }
    }
    $('canvas-empty').hidden = Boolean(showObserved);
    $('canvas-empty').lastChild.textContent = !connected
      ? ' 연결을 확인할 수 없어 현재 로봇 자세와 경로를 표시하지 않습니다'
      : (dirty && state?.registered ? ' 설정을 바꿨습니다. 다시 등록해 주세요' : ` ${modeName()} 로봇 상태를 확인하면 표시됩니다`);
  }

  function motionWasStopped(motion) {
    return Boolean(motion?.request_id &&
      (motion.request_id === state?.motion_stop_target || stoppedMotionRequests.has(motion.request_id)));
  }

  function motionText(motion) {
    if (!motion) return '현재 표시할 예상 경로가 없습니다.';
    if (motion.phase === 'interrupted') return `${modeName()} 명령 중단 · 정지 여부와 물체 배치 미확인`;
    if ((motion.cancel_requested || motionWasStopped(motion)) && motion.phase === 'end') return `${modeName()} 명령 종료 · 중단 결과 확인 중`;
    if (motion.cancel_requested || motionWasStopped(motion)) return `중단 요청 중 · 현재 ${modeName()} 명령이 끝나기를 기다립니다.`;
    if (motion.phase === 'end') return `${modeName()} 명령 ${motion.number} 종료 · 다음 판단 대기`;
    const sampling = motion.sampling === 'during_command' ? '진행 표본' : '명령 경계 표본';
    const details = {arm_move: '팔 목표 이동', arm_and_gripper: '팔과 집게 목표 변경',
      open_gripper: '집게 열기', close_gripper: '집게 닫기', hold: '목표 유지'};
    const detail = details[motion.command_kind] || (motion.arm_move ? '팔 목표 이동' : '집게 조작');
    return `${modeName()} 명령 ${motion.number} · ${Number(motion.elapsed).toFixed(1)} / ${Number(motion.duration).toFixed(1)}초 · ${detail} · ${sampling}`;
  }

  function forecastText() {
    if (!connected || stopPending || state?.stop_requested) return '현재 표시할 이후 예상 경로가 없습니다.';
    if (currentMode() !== 'virtual') {
      if (!currentControlCommand()) return '현재 표시할 이후 예상 경로가 없습니다.';
      const forecast = currentControlForecast();
      if (!forecast || forecast.reason === 'computing') return '현재 명령을 바탕으로 이후 예상 경로를 계산하고 있습니다.';
      if (!Array.isArray(forecast.segments) || !forecast.segments.length) return '이후 예상 경로를 표시할 수 없습니다.';
      return `현재 원통 이동의 이후 ${forecast.complete ? '예상' : '일부 예상'} 경로 · 실제 물체 위치와 충돌 여부 미확인`;
    }
    const motion = state?.motion;
    if (!motion || motion.phase === 'end' || motion.phase === 'interrupted' || motion.cancel_requested || motionWasStopped(motion)) return '현재 표시할 이후 예상 경로가 없습니다.';
    const forecast = state?.forecast;
    if (!forecast) return '현재 관측을 바탕으로 이후 예상 경로를 준비하고 있습니다.';
    if (forecast.reason === 'cancelled') return '중단 요청으로 이후 예상 경로를 지웠습니다.';
    if (forecast.reason === 'skipped') return '이 명령의 예상 경로는 생략했습니다. 다음 명령에서 다시 시도합니다.';
    if (forecast.reason === 'unavailable') return '이후 예상 경로를 계산하지 못했습니다.';
    const basis = forecast.stale && Number.isInteger(forecast.basis_command_number)
      ? `가상 명령 ${forecast.basis_command_number} 기준 · 이미 지난 경로 포함 가능 · 현재 명령으로 갱신 중 · ` : '';
    if (forecast.scope === 'program') {
      return forecast.reason === 'no_progress'
        ? basis + '검증된 요청의 남은 단계 경로 예상 · 추가 진행 없음으로 계산됨 · 실제 동작은 달라질 수 있음'
        : basis + '검증된 요청의 남은 단계 경로 예상 · 실제 동작과 다를 수 있음';
    }
    if (forecast.scope === 'program_horizon') {
      if (forecast.reason === 'future_unavailable') return basis + '이번 요청의 일부 경로만 예상 · 이후 단계 계산을 멈췄습니다.';
      const repeat = forecast.repeat_mode === 'continuous' ? ' · 이후 반복은 미계산' : '';
      const completed = Number.isInteger(forecast.completed_round) && forecast.completed_round > 0 &&
        forecast.completed_round < forecast.through_round ? `${forecast.completed_round}회차 예상 계산 완료 · ` : '';
      return basis + `이번 요청의 일부 경로 예상 · ${completed}${forecast.through_round}회차 일부까지${repeat}`;
    }
    const scope = forecast.scope === 'planned_task' ? '현재 작업에서 계획된 남은 이동' : '현재 원통 이동';
    const coverage = forecast.complete ? '예상' : '일부 예상';
    const progress = forecast.reason === 'computing' ? ' · 이후 경로 계산 중' : '';
    return basis + `${scope} ${coverage} · ${forecast.transfers}개 이동 기준${progress} · 실제 동작과 다를 수 있음`;
  }

  function render() {
    const busy = Boolean(state?.busy);
    const chatting = Boolean(state?.chat);
    const setupChat = chatting && state?.chat_setup_only === true;
    const development = state?.development_mode === true;
    const locked = !connected || pending || ending || busy || (chatting && !setupChat) || Boolean(state?.stop_requested || state?.chat_stopping);
    const canPauseListeningForRegistration = !development && currentMode() === 'physical'
      && Boolean(state?.registered) && chatting && !setupChat;
    const registrationLocked = !connected || pending || ending || busy
      || (chatting && !setupChat && !canPauseListeningForRegistration)
      || Boolean(state?.stop_requested || state?.chat_stopping)
      || (canPauseListeningForRegistration && Boolean(state?.text_target));
    const ready = Boolean(state?.registered) && requestConnectionAvailable() && !state?.recovery_required && !dirty && !awaitingRegisteredCoordinates;
    const dialogueReady = Boolean(state?.registered)
      && (requestConnectionAvailable() || (controllerConnected() && state?.recovery_guarded))
      && !dirty && !awaitingRegisteredCoordinates && (!state?.recovery_required || state?.recovery_guarded);
    const homeStopAvailable = Boolean(state?.arm_home_active && state?.text_target);
    const textStopAvailable = ((state?.busy === 'submit' || chatting) && Boolean(state?.text_target)) || homeStopAvailable;
    const activeTextControlReady = textStopAvailable && Boolean(state?.registered)
      && (requestConnectionAvailable() || (controllerConnected() && homeStopAvailable))
      && (!state?.recovery_required || homeStopAvailable);
    const textLocked = !connected || pending || ending || !(ready || activeTextControlReady) || Boolean(state?.stop_requested || state?.chat_stopping)
      || (busy && !textStopAvailable);
    slots.forEach((input) => { input.disabled = ending; });
    coordinateFields.forEach((input) => { input.disabled = ending; });
    $('coordinate-settings').hidden = currentMode() !== 'physical';
    $('development-mode-options').hidden = !development;
    $('development-mode-note').hidden = !development;
    $('advanced-settings').hidden = !development;
    $('mode-group-label').textContent = development ? '제어 모드' : '로봇 연결';
    $('step-mode-title').textContent = development ? '모드 · 연결' : '로봇 연결';
    $('physical-confirmation').hidden = !development;
    $('physical-target-inputs').hidden = !development;
    $('register-confirmation').hidden = development;
    $('physical-target-instructions').textContent = development
      ? '확인된 마지막 적용 목표각을 사용합니다. 확인된 값이 없으면 직접 확인해 입력하세요.'
      : '연결된 기기의 초기 상태를 확인합니다.';
    $('voice-note').textContent = (state?.chat_audio_mode === 'aec' && chatting
      ? '듣고 있습니다. 답변 중에도 말씀할 수 있습니다.'
      : state?.chat_audio_mode === 'half_duplex' && chatting
        ? '답변이 끝나면 다시 듣습니다.'
        : development ? '음성 대화를 시작하면 마이크를 사용합니다.'
          : '로봇 연결 후 마이크 듣기를 시작합니다. 운영체제의 마이크 권한이 필요합니다.')
      + (setupChat ? ' 인식 언어 변경은 배치 등록 후 적용됩니다.' : '');
    if (!pending) modeSelect.value = currentMode();
    modeSelect.disabled = locked;
    for (const [mode, button] of Object.entries(modeButtons)) {
      const selected = modeSelect.value === mode;
      button.disabled = locked;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    }
    const root = $('app-root');
    root.classList.toggle('mode-mock', currentMode() === 'mock');
    root.classList.toggle('mode-physical', currentMode() === 'physical');
    const problem = layoutProblem();
    const coordinates = coordinateProblem();
    const invalidCoordinates = currentMode() === 'physical' && Boolean(coordinates.message);
    const waitingForCoordinates = !controllerConnected() && !editedCoordinates.size
      && coordinateFields.every((input) => input.value.trim() === '');
    const registrationProtectionMissing = currentMode() === 'physical'
      && state?.coordinate_registration_revision_required !== true;
    coordinateInputs.forEach((pair, index) => pair.forEach((input, axis) => {
      input.setAttribute('aria-invalid', String(currentMode() === 'physical' && !waitingForCoordinates && coordinates.invalid[index][axis]));
    }));
    const coordinateError = invalidCoordinates && !waitingForCoordinates ? coordinates.message
      : registrationProtectionMissing && connected && controllerConnected() ? '설정을 적용하려면 앱을 종료한 뒤 다시 열어 주세요.' : '';
    $('coordinate-error').hidden = !coordinateError;
    $('coordinate-error').textContent = coordinateError;
    slots.forEach((input, index) => {
      input.classList.toggle('invalid', problem.bad[index]);
      input.setAttribute('aria-invalid', String(problem.bad[index]));
    });
    $('layout-error').hidden = !problem.message;
    $('layout-error').textContent = problem.message;
    presets.forEach(([button]) => { button.disabled = ending; });
    const control = state?.control || {};
    const activeSlots = control.calibration?.active_slots;
    $('physical-scope-note').hidden = currentMode() !== 'physical';
    $('physical-scope-note').textContent = Array.isArray(activeSlots) && activeSlots.length
      ? `배치는 네 자리 모두 등록할 수 있습니다. 현재 이동 가능 설정: ${activeSlots.join('·')}번 자리.`
      : '배치는 네 자리 모두 등록할 수 있습니다. 이동 가능한 자리는 연결 후 표시됩니다.';
    const ownsController = Boolean(control.controller || control.connected || control.ownership);
    const needsPhysicalRecovery = currentMode() === 'physical' && !waitingForController()
      && Boolean(state?.recovery_required || control.connection === 'fault' || control.physical_recovery_required);
    const pendingConnection = pendingConnectionRetry();
    const physicalPrepared = currentMode() === 'physical' && control.connection === 'connected' && preparationReady() && !needsPhysicalRecovery;
    const physicalOperating = activePhysicalOperation();
    deviceConnect.hidden = currentMode() === 'virtual';
    deviceConnect.disabled = locked || currentMode() === 'virtual' || (control.connection === 'unknown' && !ownsController);
    deviceConnect.textContent = ownsController ? '제어기 연결 해제' : `${modeName()} ${currentMode() === 'physical' ? '로봇' : '제어기'} 연결`;
    $('mode-indicator').textContent = currentMode() === 'virtual' ? '가상 모드 · 실물 연결 없음'
      : currentMode() === 'mock' ? '모의 모드 · 실물 연결 없음' : '실물 모드';
    $('mode-caption').hidden = !development && currentMode() === 'physical' && connected
      && (control.connected || waitingForController());
    $('mode-caption').textContent = currentMode() === 'virtual' ? '가상 로봇으로 실행합니다. 실물 장치에 연결하지 않습니다.'
      : currentMode() === 'mock' ? '모의 제어기를 별도로 연결합니다. 실제 로봇은 움직이지 않습니다.'
      : '실물 로봇은 아래 버튼을 눌러 연결합니다.';
    $('device-state').textContent = !connected ? '로컬 연결 확인 불가'
      : currentMode() === 'virtual' ? '가상 시뮬레이션 준비'
      : pendingConnection ? (state?.pending_connect_unknown && !busy ? '제어기 연결 결과 미확인' : '실물 연결 준비 중')
      : waitingForController() ? '실물 제어기 연결 복귀 대기'
      : control.connection === 'unknown' ? `${modeName()} 제어기 연결과 장치 상태 확인 불가`
      : control.connection === 'fault' ? `${modeName()} 제어기 상태 오류 · 연결 해제 후 확인 필요`
      : control.connected ? `${modeName()} 연결됨${needsPhysicalRecovery ? ' · 복구 필요' : (preparationReady() ? ' · 집게 열림' : '')}`
      : `${modeName()} 제어기 ${ownsController ? '연결 상태 확인 불가 · 연결 해제 가능' : '연결 전'}`;
    devicePrepare.hidden = currentMode() === 'virtual' || (!development && currentMode() === 'physical');
    devicePrepare.disabled = locked || currentMode() === 'virtual' || !control.connected || control.connection === 'fault' || Boolean(state?.registered || state?.recovery_required || control.preparation);
    devicePrepare.textContent = `${modeName()} 집게 열기 준비`;
    $('prepare-note').hidden = currentMode() === 'virtual' || (!development && (physicalPrepared || physicalOperating || waitingForController()));
    $('prepare-note').textContent = pendingConnection
      ? (state?.busy === 'connect' ? '연결 상태를 확인하고 있습니다.'
        : state?.pending_connect_unknown ? '연결을 다시 눌러 확인해 주세요.' : '잠시 뒤 연결을 다시 눌러 주세요.')
      : (state?.recovery_required || needsPhysicalRecovery) && currentMode() !== 'virtual'
      ? (!ownsController && currentMode() === 'physical'
        ? '결과가 불확실합니다. 제어기 상태와 팔 정지를 확인해 주세요.'
        : '결과가 불확실합니다. 연결을 해제하고 다시 연결한 뒤 집게를 준비하고 배치를 재등록해 주세요.')
      : control.preparation ? '집게 열기 준비 진행 중 · 중단 시 팔 정지와 집게 상태를 각각 확인해 주세요.'
      : preparationReady() ? '최신 집게 명령과 피드백이 열림으로 확인됐습니다.'
      : !development && currentMode() === 'physical' ? '배치 등록을 누르면 필요한 경우 집게를 연 뒤 준비 상태를 확인합니다.'
      : '연결 후 버튼을 누르면 집게 열기 명령을 보냅니다. 준비 결과는 집게 피드백으로 확인합니다.';
    $('physical-setup').hidden = currentMode() !== 'physical';
    $('physical-alert').hidden = pendingConnection || (!development && (physicalPrepared || physicalOperating || waitingForController()));
    $('physical-warning').hidden = development || needsPhysicalRecovery || waitingForController();
    $('physical-warning').textContent = !connected ? '로컬 연결 상태를 확인해 주세요.'
      : !control.connected ? '로봇을 연결해 준비 상태를 확인해 주세요.'
      : control.preparation ? '집게 준비가 진행 중입니다.'
      : state?.busy ? '작업이 진행 중입니다. 로봇 상태를 확인해 주세요.'
      : !freshObservation() ? '로봇 상태를 아직 확인하지 못했습니다.'
      : freshObservation().moving !== false ? '로봇의 정지 상태를 아직 확인하지 못했습니다.'
      : (freshObservation().gripper_command === 'open' && freshObservation().gripper_reported_state === 'open'
          && freshObservation().gripper_reported_valid === true && freshObservation().gripper_feedback_fresh !== false)
        ? '제어기의 정지와 준비 상태를 확인해 주세요.'
      : '집게가 열렸는지 확인해 주세요.';
    $('physical-recovery').hidden = pendingConnection || !needsPhysicalRecovery;
    recoveryConfirm.disabled = locked || !needsPhysicalRecovery;
    $('recover-controller').disabled = locked || !needsPhysicalRecovery || !ownsController || !recoveryConfirm.checked;
    const calibration = control.calibration || {};
    const calibrationSource = calibration.coordinate_source === 'fixed_layout_photo' ? '사진 기반 좌표'
      : calibration.coordinate_source === 'operator_input' ? '사용자 입력 좌표'
      : calibration.coordinate_source ? `좌표 출처: ${calibration.coordinate_source}` : '좌표 출처 미확인';
    $('calibration-info').textContent = `${calibrationSource} · ${calibration.verified === true ? '교정 확인됨' : '교정 미검증'}`;
    physicalJoints.forEach((input) => { input.disabled = locked || currentMode() !== 'physical'; });
    physicalReady.disabled = locked || currentMode() !== 'physical';
    const target = confirmedTarget();
    if (autoTarget && (!target || autoTarget.instance_id !== target.instance_id || autoTarget.command_id !== target.command_id
        || autoTarget.joints_deg.some((value, index) => value !== target.joints_deg[index]))) {
      physicalJoints.forEach((input) => { input.value = ''; });
      autoTarget = null;
    }
    if (development && !state?.registered && !targetEdited && !autoTarget && target && physicalJoints.every((input) => input.value.trim() === '')) {
      physicalJoints.forEach((input, index) => { input.value = String(target.joints_deg[index]); });
      autoTarget = {instance_id: target.instance_id, command_id: target.command_id, joints_deg: [...target.joints_deg]};
    }
    useConfirmedTarget.hidden = !development || !target;
    useConfirmedTarget.disabled = locked || !target || needsPhysicalRecovery;
    $('target-provenance').hidden = currentMode() !== 'physical';
    $('target-provenance').textContent = development
      ? (autoTarget ? '같은 제어기에서 확인된 적용 목표각을 채웠습니다.'
        : '확인된 적용 목표각이 없거나 직접 수정했습니다. 마지막 적용 목표각을 직접 확인해 입력하세요. 관절 관측값을 대신 사용하지 않습니다.')
      : (freshObservation()?.sensor_valid === true && freshObservation()?.moving === false
        ? '정지 관측을 수신했습니다. 배치를 등록할 때 다시 확인합니다.'
        : '기기의 정지 상태를 아직 확인하지 못했습니다.');
    $('stage-title').textContent = `${modeName()} 작업대`;
    $('arm-legend').textContent = `${modeName()} 관절`;
    $('forecast-note').textContent = currentMode() === 'virtual' ? virtualForecastNote
      : '현재 명령 목표 경로는 제어기 목표의 명목 경로입니다. 이후 예상 경로는 현재 관절과 마지막 예상 배치를 바탕으로 계산합니다. 미래 물체 위치와 충돌 여부는 확인하지 않으며 실제 동작과 다를 수 있습니다.';
    registerButton.querySelector('span').textContent = !development && currentMode() === 'physical' && !preparationReady()
      ? '집게 열고 배치 등록' : `${modeName()} 배치 등록`;
    submitButton.querySelector('span').textContent = `${modeName()} 실행`;
    $('stop-label').textContent = `${modeName()} 작업 중단`;
    if (currentMode() !== 'virtual') modelSelect.value = 'close';
    modelSelect.disabled = locked || currentMode() !== 'virtual';
    controlTimingSelect.disabled = locked || currentMode() !== 'virtual';
    fingerProfileSelect.disabled = locked || currentMode() !== 'virtual';
    closedMoveProfileSelect.disabled = locked || currentMode() !== 'virtual';
    $('finger-profile-note').hidden = currentMode() !== 'virtual';
    $('closed-move-profile-note').hidden = currentMode() !== 'virtual';
    $('control-timing-note').hidden = currentMode() !== 'virtual';
    command.disabled = ending;
    language.disabled = locked;
    registerButton.disabled = registrationLocked || !controllerConnected() || Boolean(problem.message) || invalidCoordinates || registrationProtectionMissing
      || (currentMode() !== 'virtual' && (state?.recovery_required
          || (!preparationReady() && (development || currentMode() !== 'physical'))))
      || (currentMode() === 'physical' && (development ? (!physicalTargetsValid() || !physicalReady.checked)
          : (freshObservation()?.sensor_valid !== true || freshObservation()?.moving !== false)));
    const registrationMemoryNote = state?.registration_preserves_memory
      ? '배치를 복구해도 이전 기억은 유지됩니다.' : '다시 등록하면 이전 기억이 지워집니다.';
    $('registration-help').textContent = canPauseListeningForRegistration && dirty && !busy
      ? `등록할 때 음성 듣기를 잠시 중지한 뒤 다시 시작합니다. ${registrationMemoryNote}`
      : registrationMemoryNote;
    // The server already distinguishes a direct stop from another follow-up
    // while a text-control request is pending. Do not duplicate that language
    // classifier here and accidentally prevent a supported stop expression.
    submitButton.disabled = textLocked || !command.value.trim();
    examples.forEach(([button]) => { button.disabled = ending; });
    $('command-hint').textContent = state?.text_control_pending && textStopAvailable
      ? '이전 요청을 확인하는 중입니다. 그동안 직접 정지 요청을 보낼 수 있습니다.'
      : waitingForController() ? '연결이 돌아오면 요청을 실행합니다. 기다리는 동안 중단할 수 있습니다.'
      : activeTextControlReady ? '실행 중에도 목표 변경이나 정지를 요청할 수 있습니다.'
      : !ready ? '등록 후 명령을 실행할 수 있습니다.' : '예시를 누르면 입력란에 채워집니다.';
    const canRetrySetup = !development && currentMode() === 'physical' && controllerConnected() && !state?.registered;
    chatButton.disabled = !connected || pending || ending || busy || Boolean(state?.chat_stopping)
      || (!chatting && !dialogueReady && !canRetrySetup);
    chatButton.querySelector('span').textContent = state?.chat_stopping ? '음성 대화 종료 중'
      : (chatting ? (state?.chat_listening ? '듣기 중지' : '연결 취소') : (development ? '음성 대화 시작' : '다시 듣기'));
    chatButton.classList.toggle('live', connected && chatting && state?.chat_listening === true);
    stopButton.disabled = stopPending || ending || (connected && (Boolean(state?.stop_requested)
      || (!busy && !homeStopAvailable && (Boolean(state?.chat_stopping) || !chatting || setupChat))));
    stopButton.classList.toggle('armed', !stopButton.disabled);
    const showMobileStop = !ending && (!connected || busy || homeStopAvailable || (chatting && !setupChat) || stopPending || Boolean(state?.stop_requested || state?.chat_stopping));
    $('mobile-stop').hidden = !showMobileStop;
    mobileStopButton.disabled = stopButton.disabled;
    $('mobile-stop-status').textContent = !connected ? '연결 상태 확인 불가'
      : (stopPending || state?.stop_requested) ? '중단 요청 중 · 확인 필요'
      : state?.chat_stopping && !busy ? '음성 대화 종료 중'
      : chatting ? '음성 대화 중' : '작업 진행 중';
    $('mobile-stop-label').textContent = stopPending || state?.stop_requested ? '중단 요청 중' : '중단 요청';
    root.classList.toggle('has-mobile-stop', showMobileStop);
    $('exit-button').disabled = pending || ending;
    $('connection-indicator').classList.toggle('offline', !connected);
    $('connection-label').textContent = connected ? '이 PC에서 실행 중' : '로컬 연결 확인 불가';
    $('connection-short').textContent = connected ? '로컬 연결됨' : '로컬 연결 불가';
    $('connection-indicator').setAttribute('aria-label', connected ? '이 PC에서 실행 중' : '로컬 연결 확인 불가');
    $('scene-label').textContent = !connected
      ? (state && !dirty ? `연결 확인 불가 · 마지막 수신: ${state.scene_label || '배치 미확인'}` : '연결 확인 불가 · 입력한 설정')
      : pendingConnection ? (state?.pending_connect_unknown && !busy
        ? '제어기 연결 결과 미확인 · 배치 등록 전' : '연결 준비 중 · 배치 등록 전')
      : (registrationRequest
        ? '입력한 배치 · 등록 중'
        : awaitingRegisteredCoordinates && !dirty
        ? '등록 배치 · 자리 좌표 확인 중'
        : registrationChangedElsewhere
        ? '다른 화면에서 배치가 바뀜 · 다시 등록 필요'
        : (dirty && state?.registered ? '입력한 설정 · 다시 등록 필요' : (state?.scene_label || '입력한 설정 · 아직 등록되지 않음')));
    $('motion-label').textContent = !connected
      ? '연결을 확인할 수 없어 현재 작업 상태와 예상 경로를 표시하지 않습니다.'
      : (!busy && dirty ? `입력한 배치는 등록 후 적용됩니다.`
        : ((currentMode() !== 'virtual' && retargetUnconfirmed() && !stopPending && !state?.stop_requested
              && !controlWasStopped() ? '목표 변경 응답 미확인 · 로봇 상태 확인 중' : null)
          || retargetProgress() || (currentMode() !== 'virtual' && currentControlCommand()
          ? `${modeName()} 명령 진행 중 · ${currentControlCommand().kind || '목표 확인 중'} · 명목 경로`
          : currentMode() !== 'virtual' && (stopPending || state?.stop_requested || controlWasStopped())
          ? '중단 요청 중 · 정지 여부와 물체 배치 확인 필요' : motionText(state?.motion))));
    $('forecast-label').textContent = forecastText();
    $('status').textContent = connected ? displayMessage(state?.status || '시작 배치와 학생 모델을 선택하고 등록해 주세요.')
      : '로컬 시연에 연결할 수 없습니다. 실행 상태를 확인해 주세요.';
    const activity = $('activity-state');
    activity.classList.toggle('running', connected && (busy || chatting));
    activity.classList.toggle('offline', !connected);
    activity.querySelector('span').textContent = !connected ? '상태 확인 불가'
      : (busy ? (state.busy === 'register' ? '등록 중' : '처리 중') : (chatting ? '음성 대화 중' : '대기 중'));
    $('llm-status').textContent = connected ? (state?.llm?.message || '로컬 언어 모델 준비 상태를 확인하고 있습니다.')
      : '연결을 확인할 수 없어 로컬 언어 모델 상태를 확인할 수 없습니다.';
    $('llm-dot').className = connected ? (state?.llm?.state || '') : '';
    renderReadings();
    renderSteps(ready, problem);
    renderAdvancedSummary();
    if (!state) {
      drawScene();
      return;
    }
    $('model-caption').textContent = !connected ? '연결 후 적용 상태를 확인할 수 있습니다.'
      : currentMode() !== 'virtual' ? '제어기 실행에는 기본 결합 학생이 적용됩니다.'
      : (state.model && !dirty ? `${state.model.name} 적용 · ${state.model.sha256.slice(0, 12)}` : '배치를 등록할 때 선택한 모델이 적용됩니다.');
    $('control-timing-caption').textContent = !connected ? '연결 후 적용 상태를 확인할 수 있습니다.'
      : currentMode() !== 'virtual' ? '제어기에서 이동 시간을 관리합니다.'
      : state.registered && !dirty && controlTimingNames[state.control_timing_profile]
      ? `적용됨 · ${controlTimingNames[state.control_timing_profile]}`
      : (dirty && state.registered ? '선택을 적용하려면 배치를 다시 등록해 주세요.' : '배치를 등록할 때 선택한 시간이 적용됩니다.');
    $('finger-profile-caption').textContent = !connected ? '연결 후 적용 상태를 확인할 수 있습니다.'
      : currentMode() !== 'virtual' ? '제어기에서 집게를 관리합니다. 가상 집게 설정은 적용되지 않습니다.'
      : state.registered && !dirty && fingerProfileNames[state.finger_open_profile]
      ? `적용됨 · ${fingerProfileNames[state.finger_open_profile]}`
      : (dirty && state.registered ? '선택을 적용하려면 배치를 다시 등록해 주세요.' : '배치를 등록할 때 선택한 집게 열기가 적용됩니다.');
    $('closed-move-profile-caption').textContent = !connected ? '연결 후 적용 상태를 확인할 수 있습니다.'
      : currentMode() !== 'virtual' ? '제어기의 이동 방식을 사용합니다. 가상 곡선 선택은 적용되지 않습니다.'
      : state.registered && !dirty && closedMoveProfileNames[state.closed_move_profile]
      ? `적용됨 · ${closedMoveProfileNames[state.closed_move_profile]}${state.straight_descent ? ' · 받침 위 직선 하강' : ''}`
      : (dirty && state.registered ? '선택을 적용하려면 배치를 다시 등록해 주세요.' : '배치를 등록할 때 선택한 이동 곡선이 적용됩니다.');
    const list = $('log-list');
    list.replaceChildren();
    (state.logs?.length ? state.logs.slice().reverse() : ['아직 처리한 요청이 없습니다.']).forEach((message) => {
      const item = document.createElement('li');
      item.textContent = message;
      list.appendChild(item);
    });
    (state.models || []).forEach((model) => {
      const option = [...modelSelect.options].find((candidate) => candidate.value === model.id);
      if (option) {
        option.disabled = !model.available;
        option.textContent = model.name + (model.available ? '' : ' · 파일 없음');
      }
    });
    drawScene();
  }

  async function refresh() {
    const sequence = ++refreshIssued;
    try {
      const nextState = await requestState();
      if (sequence < refreshBarrier) return;
      const viewRevision = Number.isSafeInteger(nextState.view_revision) ? nextState.view_revision : null;
      if (viewRevision !== null && lastViewRevision !== null) {
        if (viewRevision <= lastViewRevision) return;
      } else if (sequence < refreshApplied) return;
      if (state && nextState.registration_revision < state.registration_revision) return;
      if (registrationRevision !== null && nextState.registration_revision < registrationRevision) return;
      refreshApplied = Math.max(refreshApplied, sequence);
      if (viewRevision !== null) lastViewRevision = viewRevision;
      state = nextState;
      if (!initialStateSeen) {
        initialStateSeen = true;
        if (state.registered && !state.recovery_required) {
          if (editRevision === 0) adoptRegistration();
          else registrationRevision = state.registration_revision;
        } else if (state.busy === 'register' && editRevision === 0 && Array.isArray(state.layout) && state.layout.length === 4) {
          slots.forEach((input, index) => { input.value = Array.isArray(state.layout[index]) ? state.layout[index].join(', ') : ''; });
          if (validCoordinatePairs(state.registration_coordinates?.slot_xy_mm)) fillCoordinates({xy: state.registration_coordinates.slot_xy_mm});
          registrationRevision = state.registration_revision;
          registrationRequest = {revision: registrationRevision, editRevision};
        }
      }
      if (registrationRequest?.revision === state.registration_revision && state.registered && !state.recovery_required) {
        // Exclude requests that began before this completed registration was observed.
        refreshApplied = ++refreshIssued;
        refreshBarrier = refreshApplied;
        if (editRevision === registrationRequest.editRevision) adoptRegistration();
        registrationRequest = null;
      } else if (registrationRequest?.revision === state.registration_revision && !state.registered && !state.busy) {
        registrationRequest = null;
      }
      if (registrationRevision !== null && state.registration_revision !== registrationRevision && registrationRequest?.revision !== null) {
        dirty = true;
        formRestoreEligible = false;
        registrationChangedElsewhere = true;
        registrationRequest = null;
      }
      // Keep an unedited layout in step with the current expected scene. A
      // layout draft survives polls, while independent coordinate edits survive
      // via editedCoordinates. Never adopt over an in-flight registration.
      if (state.registered && !state.recovery_required && !layoutEdited && !registrationChangedElsewhere && !registrationRequest
          && state.registration_revision === registrationRevision && Array.isArray(state.layout) && state.layout.length === 4) {
        slots.forEach((input, index) => { input.value = Array.isArray(state.layout[index]) ? state.layout[index].join(', ') : ''; });
      }
      if (awaitingRegisteredCoordinates && state.registered && state.registration_revision === registrationRevision) {
        const coordinates = calibrationCoordinates();
        if (coordinates) { fillCoordinates(coordinates, true); awaitingRegisteredCoordinates = false; }
      }
      if (currentMode() === 'physical') {
        const coordinates = state.busy === 'register'
          && validCoordinatePairs(state.registration_coordinates?.slot_xy_mm)
          ? {xy: state.registration_coordinates.slot_xy_mm}
          : calibrationCoordinates();
        if (coordinates) fillCoordinates(coordinates, true);
        else {
          coordinateFields.forEach((input) => { if (!editedCoordinates.has(input)) input.value = ''; });
          if (state.registered && !dirty) awaitingRegisteredCoordinates = true;
        }
      }
      connected = true;
      // A draft may have been restored while state reads were unavailable.
      // Recheck it against this accepted snapshot, preserving any other edits.
      if (dirty) reconcileRestoredForm();
    } catch (_) {
      if (sequence < refreshApplied || sequence < refreshBarrier) return;
      refreshApplied = sequence;
      connected = false;
    }
    render();
  }

  function canRestoreForm() {
    return Boolean(formRestoreEligible && connected && state?.registered && !state?.recovery_required
      && !awaitingRegisteredCoordinates && !registrationRequest && !registrationChangedElsewhere
      && registrationRevision !== null && state.registration_revision === registrationRevision
      && Array.isArray(state.layout) && state.layout.length === slots.length
      && state.layout.every((stack) => Array.isArray(stack)));
  }

  function matchesCurrentLayout() {
    return canRestoreForm() && inputLayout().every((value, index) => {
      const entered = value ? value.split(',').map((part) => part.trim()) : [];
      return entered.length === state.layout[index].length
        && entered.every((part, position) => part === String(state.layout[index][position]));
    });
  }

  function matchesCurrentCoordinates() {
    if (currentMode() !== 'physical') return true;
    const coordinates = calibrationCoordinates();
    return Boolean(coordinates && coordinateFields.every(coordinateValid)
      && inputCoordinates().every((pair, index) => pair.every((value, axis) => value === coordinates.xy[index][axis])));
  }

  function reconcileRestoredForm() {
    if (matchesCurrentLayout()) {
      // The layout is no longer a draft, even when a coordinate edit remains.
      layoutEdited = false;
      if (matchesCurrentCoordinates()) {
        dirty = false;
        editedCoordinates.clear();
      }
    }
  }

  function markEdited(source = 'other') {
    editRevision += 1;
    if (source !== 'form') formRestoreEligible = false;
    dirty = true;
    if (source === 'form') reconcileRestoredForm();
    render();
  }

  slots.forEach((input) => input.addEventListener('input', () => { layoutEdited = true; markEdited('form'); }));
  coordinateFields.forEach((input) => input.addEventListener('input', () => {
    editedCoordinates.add(input);
    markEdited('form');
  }));
  presets.forEach(([button, values]) => button.addEventListener('click', () => {
    if (button.disabled) return;
    slots.forEach((input, index) => { input.value = values[index]; });
    layoutEdited = true;
    markEdited('form');
  }));
  examples.forEach(([button, text]) => button.addEventListener('click', () => {
    if (button.disabled) return;
    command.value = text;
    render();
    if (typeof command.focus === 'function') command.focus();
  }));
  const changeMode = () => {
    if (state?.development_mode !== true) return;
    return act('/api/mode', {mode: modeSelect.value}, (result) => {
      registrationRevision = result.registration_revision;
      registrationChangedElsewhere = false;
      dirty = true;
      formRestoreEligible = false;
      awaitingRegisteredCoordinates = false;
      autoTarget = null;
      targetEdited = false;
      physicalReady.checked = false;
      recoveryConfirm.checked = false;
    });
  };
  modeSelect.addEventListener('change', changeMode);
  for (const [mode, button] of Object.entries(modeButtons)) {
    button.addEventListener('click', () => {
      if (button.disabled || modeSelect.disabled || mode === currentMode()) return;
      modeSelect.value = mode;
      changeMode();
    });
  }
  deviceConnect.addEventListener('click', () => {
    const disconnecting = Boolean(state?.control?.controller || state?.control?.connected || state?.control?.ownership);
    const body = {mode: currentMode()};
    if (!disconnecting) body.language = language.value;
    act(disconnecting ? '/api/disconnect' : '/api/connect', body, (result) => {
      if (disconnecting) {
        registrationRevision = result.registration_revision;
        registrationChangedElsewhere = false;
        dirty = true;
        formRestoreEligible = false;
        autoTarget = null;
        targetEdited = false;
        physicalJoints.forEach((input) => { input.value = ''; });
        physicalReady.checked = false;
        recoveryConfirm.checked = false;
      }
    });
  });
  devicePrepare.addEventListener('click', () => act('/api/prepare-gripper', {mode: currentMode()}));
  recoveryConfirm.addEventListener('change', render);
  useConfirmedTarget.addEventListener('click', () => {
    const target = confirmedTarget();
    if (!target) return;
    physicalJoints.forEach((input, index) => { input.value = String(target.joints_deg[index]); });
    autoTarget = {instance_id: target.instance_id, command_id: target.command_id, joints_deg: [...target.joints_deg]};
    targetEdited = false;
    physicalReady.checked = false;
    markEdited();
  });
  $('recover-controller').addEventListener('click', () => act('/api/recover-controller', {
    mode: currentMode(), layout_confirmed: recoveryConfirm.checked === true, confirm_stop: recoveryConfirm.checked === true,
  }, (result) => {
    recoveryConfirm.checked = false;
    physicalReady.checked = false;
    dirty = true;
    formRestoreEligible = false;
    registrationRevision = result.registration_revision;
    registrationChangedElsewhere = false;
  }));
  physicalJoints.forEach((input) => input.addEventListener('input', () => {
    autoTarget = null;
    targetEdited = true;
    physicalReady.checked = false;
    markEdited();
  }));
  physicalReady.addEventListener('change', () => { if (state?.registered) markEdited(); else render(); });
  modelSelect.addEventListener('change', markEdited);
  controlTimingSelect.addEventListener('change', markEdited);
  fingerProfileSelect.addEventListener('change', markEdited);
  closedMoveProfileSelect.addEventListener('change', markEdited);
  command.addEventListener('input', render);
  registerButton.addEventListener('click', async () => {
    if (registerButton.disabled || pending || ending) return;
    const body = {layout: inputLayout(), model: modelSelect.value, mode: currentMode(), language: language.value,
      finger_open_profile: fingerProfileSelect.value, closed_move_profile: closedMoveProfileSelect.value,
      control_timing_profile: controlTimingSelect.value};
    if (currentMode() === 'physical') {
      body.slot_xy_mm = inputCoordinates();
      body.table_z_mm = 0;
      body.registration_revision = state?.registration_revision;
      body.controller_ready_confirmed = state?.development_mode !== true || physicalReady.checked === true;
      if (state?.development_mode !== true) body.prepare_gripper_if_needed = true;
      if (state?.development_mode === true) {
        body.initial_applied_target_deg = physicalJoints.map((input) => Number(input.value));
        if (autoTarget && body.initial_applied_target_deg.every((value, index) => value === autoTarget.joints_deg[index])) {
          body.initial_applied_target_evidence = {instance_id: autoTarget.instance_id, command_id: autoTarget.command_id};
        }
      }
    }
    const request = registrationRequest = {revision: null, editRevision, previousDirty: dirty};
    let explicitlyRejected = false;
    let stateChanged = false;
    dirty = true;
    await act('/api/register', body, (result) => {
      registrationRevision = result.registration_revision;
      registrationChangedElsewhere = false;
      formRestoreEligible = false;
      request.revision = result.registration_revision;
      // Ignore state requests issued before the server accepted this request.
      refreshApplied = ++refreshIssued;
      refreshBarrier = refreshApplied;
    }, (error) => {
      explicitlyRejected = error.status >= 400 && error.status < 500;
      stateChanged = error.code === 'registration_state_changed';
    });
    if (registrationRequest === request && request.revision === null) {
      registrationRequest = null;
      // A lost response could still have changed the server registration.
      // Preserve the form until the user explicitly resolves that state.
      if (!explicitlyRejected) { layoutEdited = true; formRestoreEligible = false; }
      if (explicitlyRejected && editRevision === request.editRevision && !registrationChangedElsewhere) {
        dirty = request.previousDirty;
      }
      if (stateChanged) await refresh();
      else render();
    }
  });
  submitButton.addEventListener('click', () => act('/api/submit', {text: command.value,
    registration_revision: registrationRevision, stop_target: state?.text_target ?? null}));
  command.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && event.ctrlKey && !submitButton.disabled) {
      event.preventDefault();
      submitButton.click();
    }
  });
  chatButton.addEventListener('click', () => {
    if (state?.chat) act('/api/chat/end', {});
    else act('/api/chat/start', {language: language.value, registration_revision: registrationRevision});
  });
  stopButton.addEventListener('click', async () => {
    if (stopPending || ending) return;
    const stopTarget = state?.text_target ?? state?.motion?.request_id;
    const controlCommand = state?.control?.current_command?.command_id;
    if (stopTarget) stoppedMotionRequests.add(stopTarget);
    if (controlCommand) stoppedControlCommands.add(controlCommand);
    stopPending = true;
    render();
    try {
      await request('/api/stop', {stop_target: stopTarget ?? null});
      if (!connected) toast('중단 요청을 전달했습니다. 실행 종료는 아직 확인되지 않았습니다.');
      await refresh();
    } catch (error) {
      if (stopTarget) stoppedMotionRequests.delete(stopTarget);
      if (controlCommand) stoppedControlCommands.delete(controlCommand);
      toast(error.message ? `중단 요청을 전달하지 못했습니다: ${error.message}` : '중단 요청을 전달하지 못했습니다. 연결과 작업 상태를 확인해 주세요.');
    } finally {
      stopPending = false;
      render();
    }
  });
  mobileStopButton.addEventListener('click', () => {
    if (!stopButton.disabled) stopButton.click();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !stopButton.disabled) stopButton.click();
  });
  window.addEventListener('resize', drawScene);
  $('exit-button').addEventListener('click', () => act('/api/close', {}, () => {
    ending = true;
    $('closed-overlay').hidden = false;
  }));
  refresh();
  setInterval(() => { if (!ending) refresh(); }, 350);
})();
