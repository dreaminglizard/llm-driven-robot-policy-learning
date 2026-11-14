"""Preflight learned transfers and execute them through the resident controller.

This module does not import the robot SDK. Execution requires measured coordinates
or an explicit photo-based fixed-layout trial. Trial coordinates remain unverified.
Finished commands yield EXPECTED object state, never observed success.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
import uuid

from raccoon_control.config import Config
from raccoon_control.errors import ControlError
from raccoon_control.kinematics import Geometry, check_joints, quantize_joints
from raccoon_control.paths import radial_yaw_segments, vertical_segments
from raccoon_control.motion_envelope import bounds as motion_bounds, check as check_envelope
from raccoon_control.validation import keys, number, vector
from raccoon_control.timing import measured,span
from .commands import goal_for, parse, registered_state
from .world import MAX_HEIGHT, MOVES, SLOTS, state, step
from .scope import enabled_slots as validate_slots

DEFAULT_BLENDED_MAX_VELOCITY_DEG_S = 12.
SLOT_COUNT = len(SLOTS)
MAX_STACK_HEIGHT = MAX_HEIGHT
MAX_SAFE_SPEED_PCT = 35
MAX_WAYPOINT_TOLERANCE_MM = 3


def validate_slot_coordinates(value):
    if len(value) != SLOT_COUNT:
        raise ValueError("자리 네 개의 좌표가 필요합니다.")
    xy = tuple(vector(pair, 2, "slot_xy_mm") for pair in value)
    if len(set(xy)) != SLOT_COUNT:
        raise ValueError("자리 좌표는 서로 달라야 합니다.")
    return xy


@dataclass(frozen=True)
class SlotCalibration:
    calibration_id: str
    verified: bool
    simulation_only: bool
    slot_xy_mm: tuple
    table_z_mm: float
    cylinder_height_mm: float
    grip_offset_mm: float
    travel_z_mm: float
    clearance_mm: float
    vertical_corridor_mm: float
    max_height: int
    speed_pct: float
    tool_forward_mm: float
    tool_vertical_mm: float
    coordinate_source: str = "measured"
    transport_path: str = "cartesian"
    waypoint_tolerance_mm: float | None = None
    travel_speed_pct: float | None = None
    compact_waypoints: bool = False
    enabled_slots: tuple = tuple(range(1, SLOT_COUNT + 1))
    path_ceiling_mm: float | None = None
    blended_max_velocity_deg_s: float = DEFAULT_BLENDED_MAX_VELOCITY_DEG_S

    def __post_init__(self):
        object.__setattr__(self,"enabled_slots",validate_slots(self.enabled_slots))
        if not isinstance(self.calibration_id,str) or not self.calibration_id:
            raise ValueError("교정 기록의 식별자가 필요합니다.")
        if type(self.verified) is not bool or type(self.simulation_only) is not bool:
            raise ValueError("교정·모의 여부는 명시한 참/거짓 값이어야 합니다.")
        if self.coordinate_source not in ("measured", "fixed_layout_photo", "operator_input"):
            raise ValueError("알 수 없는 좌표 출처입니다.")
        if self.coordinate_source in ("fixed_layout_photo", "operator_input") and self.verified:
            raise ValueError("사진 또는 사용자 입력 좌표를 실측 교정 완료로 표시할 수 없습니다.")
        if self.transport_path not in ("cartesian", "radial_yaw"):
            raise ValueError("알 수 없는 이송 경로 방식입니다.")
        if self.waypoint_tolerance_mm is not None:
            number(self.waypoint_tolerance_mm,"waypoint_tolerance_mm",minimum=.01,maximum=MAX_WAYPOINT_TOLERANCE_MM)
        if self.travel_speed_pct is not None:
            number(self.travel_speed_pct,"travel_speed_pct",minimum=1,maximum=MAX_SAFE_SPEED_PCT)
        number(self.blended_max_velocity_deg_s,"blended_max_velocity_deg_s",minimum=.0001)
        if type(self.compact_waypoints) is not bool:
            raise ValueError("경유점 병합 여부는 참/거짓 값이어야 합니다.")
        object.__setattr__(self, "slot_xy_mm", validate_slot_coordinates(self.slot_xy_mm))
        if type(self.max_height) is not int or not 1<=self.max_height<=MAX_STACK_HEIGHT:
            raise ValueError(f"교정된 층수는 1~{MAX_STACK_HEIGHT} 사이여야 합니다.")
        for name in ("table_z_mm","tool_forward_mm","tool_vertical_mm"):
            number(getattr(self,name),name)
        for name in ("cylinder_height_mm","grip_offset_mm","clearance_mm","vertical_corridor_mm"):
            number(getattr(self,name),name,minimum=0.01)
        number(self.travel_z_mm,"travel_z_mm")
        if self.path_ceiling_mm is not None:
            number(self.path_ceiling_mm,"path_ceiling_mm",minimum=self.travel_z_mm)
        number(self.speed_pct,"speed_pct",minimum=1,maximum=MAX_SAFE_SPEED_PCT)
        if self.grip_offset_mm >= self.cylinder_height_mm:
            raise ValueError("집기 기준점은 원통 높이 안에 있어야 합니다.")
        if self.travel_z_mm < self.transport_floor_mm:
            raise ValueError("이송 높이가 교정한 탑과 원통 아래쪽의 여유 높이보다 낮습니다.")

    @property
    def transport_floor_mm(self):
        return self.table_z_mm+self.max_height*self.cylinder_height_mm+self.grip_offset_mm+self.clearance_mm

    @classmethod
    def load(cls,path):
        raw=json.loads(Path(path).read_text(encoding="utf-8"))
        keys(raw,allowed=cls.__dataclass_fields__,required=set(cls.__dataclass_fields__)-{
            "coordinate_source","transport_path","waypoint_tolerance_mm","travel_speed_pct","compact_waypoints","enabled_slots","path_ceiling_mm","blended_max_velocity_deg_s"})
        return cls(**raw)

    def require_ready(self,config):
        nominal_trial = (self.coordinate_source in ("fixed_layout_photo", "operator_input")
                         and config.mode == "calibration"
                         and (config.workspace_source == "fixed_layout_photo"
                              or (self.coordinate_source == "operator_input" and config.workspace_verified)))
        if nominal_trial and self.max_height > 2:
            raise ControlError("TRIAL_LIMIT","시험 좌표는 최대 2층 범위입니다.")
        if max(self.speed_pct,self.travel_speed_pct or self.speed_pct)>config.limits.max_speed_pct:
            raise ControlError("TRANSFER_SPEED_LIMIT","작업 속도가 제어기의 속도 상한을 넘습니다.")
        if not config.workspace_ready or (not self.verified and not nominal_trial):
            raise ControlError("CALIBRATION_UNVERIFIED","실측 교정값 또는 출처를 명시한 고정 배치 시험 좌표가 필요합니다.")
        if not config.limits.workspace_configured:
            raise ControlError("WORKSPACE_UNSPECIFIED","작업별 공간 범위를 명시해야 합니다.")
        if config.backend=="sdk" and self.simulation_only:
            raise ControlError("SIMULATION_CALIBRATION","모의 좌표를 실제 장치에 사용할 수 없습니다.")
        if (config.tool_forward_mm,config.tool_vertical_mm)!=(self.tool_forward_mm,self.tool_vertical_mm):
            raise ControlError("CALIBRATION_MISMATCH","교정 당시 손끝 설정과 제어기 설정이 다릅니다.")


class TransferCompiler:
    def __init__(self,config,calibration):
        self.config,self.calibration=config,calibration
        self.geometry=Geometry(tool_forward_mm=config.tool_forward_mm,tool_vertical_mm=config.tool_vertical_mm)
        self.path_limits=config.limits
        if calibration.path_ceiling_mm is not None and config.limits.workspace_configured:
            upper=config.limits.workspace_max_mm
            self.path_limits=replace(config.limits,workspace_max_mm=(*upper[:2],min(upper[2],calibration.path_ceiling_mm)))

    def check_segment(self,start,target,*,transport=False,vertical_xy=None,observed_start=False):
        evidence=self.geometry.check_path(start,target,self.path_limits)
        if transport and evidence["minimum_tcp_z_mm"]<self.calibration.transport_floor_mm:
            raise ControlError("TRANSFER_CLEARANCE","이송 경로가 원통 아래쪽의 여유 높이를 벗어납니다.")
        count=max(1,math.ceil(max(abs(a-b) for a,b in zip(start,target))/self.config.limits.path_step_deg))
        if vertical_xy is not None:
            deviation=0.0
            for i in range(count+1):
                point=self.geometry.forward(tuple(a+(b-a)*i/count for a,b in zip(start,target)))
                deviation=max(deviation,math.hypot(point[0]-vertical_xy[0],point[1]-vertical_xy[1]))
            if deviation>self.calibration.vertical_corridor_mm:
                raise ControlError("VERTICAL_CORRIDOR","집기·놓기 경로가 교정된 수직 통로를 벗어납니다.",
                                   maximum_lateral_deviation_mm=deviation,corridor_mm=self.calibration.vertical_corridor_mm)
            evidence["maximum_lateral_deviation_mm"]=deviation
        if self.config.backend=="sdk" or self.calibration.coordinate_source in ("fixed_layout_photo", "operator_input"):
            envelope=motion_bounds(self.geometry,start,target,self.config.limits.joint_tolerance_deg,
                                   start_tolerance_deg=0 if observed_start else None)
            evidence["independent_joint_envelope"]=check_envelope(envelope,self.path_limits,
                vertical_xy=vertical_xy,corridor_mm=self.calibration.vertical_corridor_mm,
                minimum_z_mm=self.calibration.transport_floor_mm if transport else None)
        return evidence

    def compile(self,start_joints,initial,actions):
        """Check the ENTIRE request before returning the first executable stage."""
        self.calibration.require_ready(self.config)
        current=state(initial)
        actions=tuple(actions)
        if len(actions)>20 or max(map(len,current))>self.calibration.max_height:
            raise ControlError("TASK_LIMIT","이동 수 또는 시작 배치가 교정 범위를 넘습니다.")
        q=check_joints(start_joints)
        operations=[]
        c=self.calibration

        def move(xyz,transfer,label,*,transport=False,vertical_xy=None):
            nonlocal q
            validate=lambda start,target:self.check_segment(start,target,transport=transport,vertical_xy=vertical_xy)
            independent=self.config.backend=="sdk" or c.coordinate_source in ("fixed_layout_photo", "operator_input")
            if transport and c.transport_path=="radial_yaw":
                segments=radial_yaw_segments(self.geometry,q,xyz,validate,compact=c.compact_waypoints)
            elif vertical_xy is not None or independent:
                segments=vertical_segments(self.geometry,q,xyz,validate,allow_async_subdivision=independent,
                                           compact=c.compact_waypoints)
            else:
                target=self.geometry.inverse(xyz)
                _,effective=quantize_joints(target)
                segments=[{"requested":target,"effective":effective,"path":validate(q,effective)}]
            for index,segment in enumerate(segments,1):
                action={"kind":"move_joints","joints_deg":segment["requested"],"speed_pct":c.speed_pct}
                # Only the final pick/place points require contact accuracy.
                # Intermediate waypoints retain the joint, envelope and live
                # corridor checks; the next segment starts from actual feedback.
                contact_point=label in {"descend_to_pick","descend_to_place"} and index==len(segments)
                envelope=segment["path"].get("independent_joint_envelope")
                minimum_z=envelope["minimum_mm"][2] if envelope else segment["path"]["minimum_tcp_z_mm"]
                faster=(c.travel_speed_pct is not None and c.travel_speed_pct>c.speed_pct
                        and not contact_point and minimum_z>=c.transport_floor_mm)
                if faster:
                    action["speed_pct"]=c.travel_speed_pct
                if c.waypoint_tolerance_mm is not None and not contact_point:
                    action["arrival_tolerance_mm"]=c.waypoint_tolerance_mm
                if independent:
                    action["path_constraints"]={"vertical_xy_mm":vertical_xy,
                        "corridor_mm":c.vertical_corridor_mm if vertical_xy is not None else None,
                        "minimum_z_mm":c.transport_floor_mm if transport or faster else None,
                        "maximum_z_mm":c.path_ceiling_mm}
                operations.append({"transfer":transfer,"stage":label,"segment":index,"segments":len(segments),
                                   "transport_phase":segment.get("transport_phase"),
                                   "expected_start_joints_deg":q,
                                   "action":action,
                                   "path":segment["path"],"constraints":{"transport":transport or faster,"vertical_xy":vertical_xy}})
                q=segment["effective"]

        def gripper(value,transfer,label):
            operations.append({"transfer":transfer,"stage":label,"expected_start_joints_deg":q,
                               "action":{"kind":"set_gripper","state":value}})

        for index,action in enumerate(actions):
            if type(action) is not int or not 0<=action<len(MOVES):
                raise ControlError("INVALID_TRANSFER","원통 이동만 실행할 수 있습니다.")
            src,dst=MOVES[action]
            if src+1 not in c.enabled_slots or dst+1 not in c.enabled_slots:
                raise ControlError("SLOT_DISABLED","현재 사용 범위 밖의 자리로 이동할 수 없습니다.")
            after=step(current,action)
            if max(map(len,after))>c.max_height:
                raise ControlError("TASK_LIMIT","계획의 중간 배치가 교정된 최대 층수를 넘습니다.")
            source,destination=c.slot_xy_mm[src],c.slot_xy_mm[dst]
            point=self.geometry.forward(q)
            move((*point[:2],c.travel_z_mm),index,"raise_before_transfer",vertical_xy=point[:2])
            gripper("open",index,"open_before_pick")
            move((*source,c.travel_z_mm),index,"above_source",transport=True)
            pickup=c.table_z_mm+(len(current[src])-1)*c.cylinder_height_mm+c.grip_offset_mm
            drop=c.table_z_mm+len(current[dst])*c.cylinder_height_mm+c.grip_offset_mm
            move((*source,pickup),index,"descend_to_pick",vertical_xy=source)
            gripper("closed",index,"close_for_pick")
            move((*source,c.travel_z_mm),index,"lift_from_source",vertical_xy=source)
            move((*destination,c.travel_z_mm),index,"above_destination",transport=True)
            move((*destination,drop),index,"descend_to_place",vertical_xy=destination)
            gripper("open",index,"release_at_destination")
            move((*destination,c.travel_z_mm),index,"retreat_from_destination",vertical_xy=destination)
            operations[-1]["expected_after_transfer"]=after
            current=after
        return {"calibration_id":c.calibration_id,"coordinate_source":c.coordinate_source,
                "transport_path":c.transport_path,
                "waypoint_tolerance_mm":c.waypoint_tolerance_mm,
                "speed_profile":{"contact_speed_pct":c.speed_pct,"travel_speed_pct":c.travel_speed_pct,
                                 "fast_motion_minimum_z_mm":c.transport_floor_mm,
                                 "compact_waypoints":c.compact_waypoints},
                "physical_calibration_verified":c.verified and self.config.workspace_verified,
                "enabled_slots":c.enabled_slots,
                "initial":state(initial),"expected_final":current,
                "operations":operations,"full_collision_check":False,"physical_success":None}


class TransferRunner:
    """No retries after an uncertain response; caller owns controller lifetime."""
    def __init__(self,controller,calibration,timeout=None):
        self.controller=controller
        self.calibration=calibration
        self.cancelled=threading.Event()
        if timeout is None:
            timeout=300 if calibration.coordinate_source in ("fixed_layout_photo", "operator_input") else 120
        self.timeout=number(timeout,"task_timeout",minimum=1,maximum=300)
        self._run_lock=threading.Lock()
        self._lease_lock=threading.Lock()
        self._active_lease=None

    def cancel(self):
        self.cancelled.set()
        return self._stop_owned("TRANSFER_CANCELLED")

    def _stop_owned(self,reason):
        with self._lease_lock:
            if self._active_lease is None:
                return {"requested":False,"reason":"no_active_transfer_lease"}
            return self.controller.stop(reason)

    @staticmethod
    def _stationary_observation(observation,instance_id=None):
        if not isinstance(observation,dict) or observation.get('sensor_valid') is not True or observation.get('moving') is not False:
            raise ControlError('NOT_READY','새 정지 관측을 확인하지 못했습니다.')
        if instance_id is not None and observation.get('instance_id')!=instance_id:
            raise ControlError('CONTROLLER_CHANGED','실행 중 관측의 제어기가 바뀌었습니다.')
        if observation.get('device_diagnostics',{}).get('firmware_warning_raw')==1:
            raise ControlError('DEVICE_WARNING','장치 경고가 있어 후속 동작을 중단했습니다.')
        vector(observation.get('joints_deg'),4,'observed_joints')

    @measured('transfer_runner')
    def run(self,initial,actions,*,on_progress=None,on_transfer=None):
        if not self._run_lock.acquire(blocking=False):
            raise ControlError("BUSY","다른 원통 작업이 진행 중입니다.")
        lease=None
        watchdog=None
        watchdog_finished=threading.Event()
        deadline_lock=threading.Lock()
        deadline=[None]
        in_flight=False
        current_transfer=None
        result={"started_at":datetime.now(timezone.utc).isoformat(),"mode":"mock_controller" if self.controller.config.backend=="mock" else "hardware_expected",
                "coordinate_source":self.calibration.coordinate_source,
                "physical_calibration_verified":self.calibration.verified and self.controller.config.workspace_verified,
                "physical_success":None,"status":"not_started","stages":[],"expected_final":None,
                "expected_current":None,"completed_transfers":0,"layout_uncertain":False}
        try:
            result["expected_current"]=state(initial)
            self.calibration.require_ready(self.controller.config)
            if self.cancelled.is_set():
                raise ControlError("CANCELLED","취소된 요청은 다시 실행하지 않습니다.")
            with span('acquire_rpc'):
                lease=self.controller.acquire("learned-transfer")["lease_id"]
            with self._lease_lock:
                self._active_lease=lease
            with span('observe_rpc'):
                observation=self.controller.observe(lease)
            self._stationary_observation(observation)
            instance_id=observation.get('instance_id')
            compiler=TransferCompiler(self.controller.config,self.calibration)
            with span('path_compile'):
                compiled=compiler.compile(observation["joints_deg"],initial,actions)
            expected_observed_joints=observation["joints_deg"]
            result["compiled"]=compiled
            def watch_deadline():
                while not watchdog_finished.wait(.02):
                    with deadline_lock:
                        expired=deadline[0] is not None and time.monotonic()>=deadline[0]
                        if expired:
                            self.cancelled.set()
                    if expired:
                        try:
                            self._stop_owned("TRANSFER_TIMEOUT")
                        except ControlError:
                            pass
                        return
            watchdog=threading.Thread(target=watch_deadline,daemon=True)
            watchdog.start()
            run_id=uuid.uuid4().hex
            for index,operation in enumerate(compiled["operations"]):
                if operation["transfer"]!=current_transfer:
                    current_transfer=operation["transfer"]
                    with deadline_lock:
                        deadline[0]=time.monotonic()+self.timeout
                if self.cancelled.is_set():
                    raise ControlError("CANCELLED","요청을 취소했거나 작업 시간 상한에 도달했습니다.")
                with span('observe_rpc'):
                    observation=self.controller.observe(lease)
                self._stationary_observation(observation,instance_id)
                if max(abs(a-b) for a,b in zip(observation["joints_deg"],expected_observed_joints))>self.controller.config.limits.observation_joint_drift_deg:
                    raise ControlError("STATE_CHANGED","검사한 출발 자세에서 로봇이 움직여 계획을 중단했습니다.")
                if operation["action"]["kind"]=="move_joints":
                    _,effective=quantize_joints(operation["action"]["joints_deg"])
                    with span('live_path_check'):
                        compiler.check_segment(observation["joints_deg"],effective,observed_start=True,**operation["constraints"])
                if self.cancelled.is_set():
                    raise ControlError("CANCELLED","전송 전에 정지가 요청됐습니다.")
                # A missing reply may still have changed the real world.
                in_flight=True
                command_id=f"{run_id}-{index}"
                with span('apply_rpc'):
                    outcome=self.controller.apply_action({"lease_id":lease,"command_id":command_id,
                        "observation_id":observation["observation_id"],"action":operation["action"]})
                result["stages"].append({"transfer":operation["transfer"],"stage":operation["stage"],
                                         "segment":operation.get("segment"),"segments":operation.get("segments"),
                                         "transport_phase":operation.get("transport_phase"),"outcome":outcome})
                if not isinstance(outcome,dict) or outcome.get('command_id')!=command_id:
                    raise ControlError('COMMAND_REPLY_MISMATCH','요청한 동작과 응답의 명령 번호가 다릅니다.')
                allowed={"gripper_reported"} if operation["action"]["kind"]=="set_gripper" else {"completed","already_at_target"}
                if outcome.get("status") not in allowed:
                    raise ControlError("TRANSFER_UNCONFIRMED","동작 도달 또는 집게 상태를 확인하지 못해 후속 동작을 중단했습니다.",stage=operation["stage"],outcome_status=outcome.get("status"))
                self._stationary_observation(outcome.get('observation'),instance_id)
                expected_observed_joints=outcome["observation"]["joints_deg"]
                if on_progress:
                    with span('progress_callback'):
                        on_progress({"transfer":current_transfer,"stage":operation["stage"],
                            "segment":operation.get("segment"),"segments":operation.get("segments"),
                            "completed_stages":index+1,"total_stages":len(compiled["operations"])})
                if "expected_after_transfer" in operation:
                    before=result["expected_current"]
                    after=state(operation["expected_after_transfer"])
                    result.update(expected_current=after,completed_transfers=current_transfer+1)
                    in_flight=False
                    with deadline_lock:
                        deadline[0]=None
                    if on_transfer:
                        with span('transfer_commit_callback'):
                            on_transfer({"transfer":current_transfer,"before":before,"after":after,
                                         "physical_success":None})
            if self.cancelled.is_set():
                raise ControlError("CANCELLED","작업 완료 전에 정지가 요청됐습니다.")
            result.update(status="command_sequence_completed",expected_final=compiled["expected_final"])
        except (ControlError,ValueError) as exc:
            result.update(status="blocked" if not result["stages"] else "stopped_unverified",
                          error=exc.as_dict() if isinstance(exc,ControlError) else {"code":"INVALID_PLAN","message":str(exc)})
        except Exception as exc:
            result.update(status="stopped_unverified",expected_final=None,
                          error={"code":"TRANSFER_INTERNAL_ERROR","message":type(exc).__name__})
        finally:
            watchdog_finished.set()
            if watchdog:
                watchdog.join(self.controller.config.limits.stop_timeout+self.controller.config.limits.rpc_timeout+0.2)
            result["layout_uncertain"]=in_flight
            if self.cancelled.is_set():
                result.update(status="stopped_unverified",expected_final=None)
            if lease:
                with self._lease_lock:
                    try:
                        with span('end_rpc'):
                            result["end"]=self.controller.end_episode(lease)
                        if not result["end"]["stop"]["confirmed"] or result["end"]["fault"]:
                            result.update(status="stopped_unverified",expected_final=None,layout_uncertain=True)
                    except Exception as exc:
                        error=exc.as_dict() if isinstance(exc,ControlError) else {'code':'END_UNCONFIRMED','message':type(exc).__name__}
                        result.update(status="stopped_unverified",expected_final=None,layout_uncertain=True,end_error=error)
                    finally:
                        self._active_lease=None
            result["finished_at"]=datetime.now(timezone.utc).isoformat()
            self._run_lock.release()
        return result


def main():
    parser=argparse.ArgumentParser(description="번호 명령 → 학습 계획 → 상주 제어기. 기본은 모의 장치입니다.")
    parser.add_argument("--request",default="1번에서 3번으로 옮겨")
    parser.add_argument("--config",default="configs/mock.json")
    parser.add_argument("--calibration",default="configs/slots.mock.json")
    parser.add_argument("--model-dir",default="artifacts/memory_v2")
    parser.add_argument("--runtime",default=".runtime/bridge-mock")
    parser.add_argument("--controller-runtime",help="실행 중인 상주 제어기 폴더. 작업 종료 뒤에도 장치 연결을 유지합니다.")
    parser.add_argument("--output",default=".runtime/bridge-result.json")
    parser.add_argument("--initial-counts",nargs=4,type=int)
    parser.add_argument("--allow-hardware",action="store_true")
    args=parser.parse_args()
    config=Config.load(args.config)
    calibration=SlotCalibration.load(args.calibration)
    # All readiness checks happen before constructing/starting a controller.
    calibration.require_ready(config)
    if config.backend=="sdk":
        if not args.allow_hardware:
            parser.error("실물 실행에는 --allow-hardware가 필요합니다.")
        if not args.controller_runtime:
            parser.error("실물은 --controller-runtime으로 상주 제어기에 연결해야 합니다. 작업 종료 시 장치 연결을 끊지 않습니다.")
        if args.initial_counts is None:
            parser.error("실물 시작 배치는 --initial-counts로 명시해야 합니다. 기본 배치로 초기화하지 않습니다.")
        from raccoon_control.hardware_policy import check_hardware_time
        check_hardware_time()
    initial=registered_state(args.initial_counts if args.initial_counts is not None else [1,1,1,1])
    command=parse(args.request)
    goal=goal_for(command,initial,enabled_slots=calibration.enabled_slots)
    from .inference import LocalPlanner
    planner=LocalPlanner(args.model_dir,max_height=calibration.max_height,enabled_slots=calibration.enabled_slots)
    plan=planner.plan(initial,goal)
    if args.controller_runtime:
        from .resident import ResidentController
        controller=ResidentController(config,args.controller_runtime)
    else:
        from raccoon_control.controller import Controller
        controller=Controller(config,args.runtime)
    report={"request":args.request,"plan":asdict(plan),"goal":goal}
    try:
        controller.start()
        report["execution"]=TransferRunner(controller,calibration).run(initial,plan.actions)
    except ControlError as exc:
        report["error"]=exc.as_dict()
    finally:
        report["close"]=controller.close()
        path=Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    execution=report.get("execution",{})
    print(json.dumps({"status":execution.get("status"),"error":report.get("error",execution.get("error")),
                      "method":plan.method,"stages":len(execution.get("stages",[])),
                      "expected_final":execution.get("expected_final"),"physical_success":None,
                      "controller_connection_retained":report["close"].get("controller_connection_retained",False),
                      "worker_exited":report["close"].get("worker",{}).get("worker_exited")},ensure_ascii=False,indent=2))
    return 0 if execution.get("status")=="command_sequence_completed" else 1


if __name__=="__main__":
    raise SystemExit(main())
