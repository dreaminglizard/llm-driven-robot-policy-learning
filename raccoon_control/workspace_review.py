"""Offline candidate review. Produces no controller requests or verification flag.

This entry point cannot connect to a device. It consumes an earlier observation,
which is explicitly NOT considered a fresh observation for motion admission.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

from .config import Config,Limits
from .errors import ControlError
from .kinematics import Geometry,JOINT_LIMITS,quantize_joints
from .paths import vertical_segments
from .validation import json_loads,keys
from raccoon_memory.robot_bridge import SlotCalibration,TransferCompiler


def review(candidate,calibration,start_joints):
    fields={"schema","profile_id","review_only","verified","workspace_min_mm","workspace_max_mm","source"}
    keys(candidate,allowed=fields,required=fields)
    if candidate["schema"]!="workspace-review.v1" or candidate["review_only"] is not True or candidate["verified"] is not False:
        raise ControlError("INVALID_REVIEW_PROFILE","This format is strictly unverified and review-only")
    limits=Limits(workspace_min_mm=candidate["workspace_min_mm"],workspace_max_mm=candidate["workspace_max_mm"])
    config=Config(backend="mock",workspace_verified=False,tool_forward_mm=calibration.tool_forward_mm,
                  tool_vertical_mm=calibration.tool_vertical_mm,limits=limits)
    compiler=TransferCompiler(config,calibration)
    g=compiler.geometry
    source=g.forward(start_joints)
    poses={"recorded_start":tuple(start_joints)}
    positions={"raised_start":(*source[:2],calibration.travel_z_mm)}
    for index,xy in enumerate(calibration.slot_xy_mm,1):
        positions[f"slot_{index}_above"]=(*xy,calibration.travel_z_mm)
        for level in range(1,calibration.max_height+1):
            positions[f"slot_{index}_level_{level}"]=(*xy,calibration.table_z_mm+(level-1)*calibration.cylinder_height_mm+calibration.grip_offset_mm)
    point_results=[]
    for name,point in positions.items():
        try:
            requested=g.inverse(point)
            _,effective=quantize_joints(requested)
            poses[name]=effective
            point_results.append({"name":name,"candidate_mm":point,"nominal_joint_degrees":effective,"joint_limits_valid":True})
        except ControlError as exc:
            point_results.append({"name":name,"candidate_mm":point,"joint_limits_valid":False,"error":exc.as_dict()})
    edges=[("recorded_start","raised_start",False,source[:2])]
    for index,xy in enumerate(calibration.slot_xy_mm,1):
        above=f"slot_{index}_above"
        edges.append(("raised_start",above,True,None))
        for level in range(1,calibration.max_height+1):
            below=f"slot_{index}_level_{level}"
            edges.extend(((above,below,False,xy),(below,above,False,xy)))
        for destination in range(1,5):
            if destination!=index:
                edges.append((above,f"slot_{destination}_above",True,None))
    results=[]
    for a,b,transport,xy in edges:
        entry={"from":a,"to":b,"transport":transport,"vertical_corridor":xy is not None}
        try:
            if a not in poses or b not in poses:
                raise ControlError("INVALID_ENDPOINT","One endpoint failed inverse kinematics")
            validate=lambda start,target:compiler.check_segment(start,target,transport=transport,vertical_xy=xy)
            direct_error=None
            try:
                validate(poses[a],poses[b])
            except ControlError as exc:
                direct_error=exc.as_dict()
            if xy is not None:
                segments=vertical_segments(g,poses[a],positions[b],validate)
            else:
                segments=[{"effective":poses[b],"path":validate(poses[a],poses[b])}]
            waypoints=[]
            for segment in segments:
                q=segment["effective"]
                waypoints.append({"nominal_joint_degrees":q,"nominal_tcp_mm":g.forward(q),"path":segment["path"]})
            entry.update(nominal_path_valid=True,segments=len(segments),direct_path_error=direct_error,waypoints=waypoints)
        except ControlError as exc:
            entry.update(nominal_path_valid=False,error=exc.as_dict())
        results.append(entry)
    return {"schema":"workspace-review-result.v1","at":datetime.now(timezone.utc).isoformat(),
            "candidate_id":candidate["profile_id"],"executable":False,"calibration_verified":False,
            "controller_requests_generated":False,"live_connection_attempted":False,"physical_success":None,
            "workspace":config.workspace_status(),"geometry_mm":asdict(g),"joint_limits_deg":JOINT_LIMITS,
            "points":point_results,"paths":results,
            "summary":{"candidate_paths":len(results),"valid_nominal_paths":sum(x["nominal_path_valid"] for x in results),
                       "subdivided_paths":sum(x.get("segments",0)>1 for x in results),
                       "maximum_direct_vertical_deviation_mm":max((x.get("direct_path_error") or {}).get("details",{}).get("maximum_lateral_deviation_mm",0) for x in results),
                       "maximum_planned_vertical_deviation_mm":max((w["path"].get("maximum_lateral_deviation_mm",0) for x in results for w in x.get("waypoints",[])),default=0)},
            "full_collision_check":False,"input_is_fresh_motion_observation":False}


def main():
    parser=argparse.ArgumentParser(description="저장된 관측으로 작업 영역 후보만 검사합니다. 장치 연결·동작은 없습니다.")
    parser.add_argument("--candidate",default="configs/workspace.candidate.json")
    parser.add_argument("--slots",default="configs/slots.hardware.template.json")
    parser.add_argument("--observation",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    candidate=json_loads(Path(args.candidate).read_text(encoding="utf-8"))
    calibration=SlotCalibration.load(args.slots)
    observation=json_loads(Path(args.observation).read_text(encoding="utf-8"))
    if observation.get("connection_verified") is not True or observation.get("last_sample",{}).get("joints_deg") is None:
        raise ControlError("OBSERVATION_UNAVAILABLE","The stored record has no valid received joint pose")
    result=review(candidate,calibration,observation["last_sample"]["joints_deg"])
    result["inputs"]={name:{"path":str(Path(path).resolve()),"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                      for name,path in (("candidate",args.candidate),("slots",args.slots),("observation",args.observation))}
    result["recorded_observation_finished_at"]=observation.get("finished_at")
    result["sdk_imported"]="roboid" in sys.modules
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"output":str(path),**result["summary"],"executable":False,"sdk_imported":result["sdk_imported"]},ensure_ascii=False,indent=2))
    return 0 if all(x["nominal_path_valid"] for x in result["paths"]) else 1


if __name__=="__main__":
    raise SystemExit(main())
