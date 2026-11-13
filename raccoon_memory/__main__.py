import argparse
from dataclasses import asdict
import json


def main():
    parser = argparse.ArgumentParser(description="라쿤 번호·기억 과제 — 로봇에 연결하지 않는 로컬 모의 실행")
    parser.add_argument("--model-dir",default="artifacts/memory_v2")
    sub = parser.add_subparsers(dest="operation",required=True)
    gui = sub.add_parser("gui")
    gui.add_argument("--voice-dir",default="artifacts/whisper-medium")
    gui.add_argument("--app-profile",help="실물 앱 프로필. 생략하면 모의 실행입니다.")
    gui.add_argument("--allow-hardware",action="store_true")
    command = sub.add_parser("command")
    command.add_argument("text",nargs="+")
    voice = sub.add_parser("transcribe")
    voice.add_argument("audio_file")
    voice.add_argument("--voice-dir",default="artifacts/whisper-medium")
    voice.add_argument("--language",choices=("auto","ko","en"),default="auto")
    sub.add_parser("demo")
    args = parser.parse_args()
    if args.operation == "gui":
        from .gui import launch
        launch(args.model_dir,args.voice_dir,profile_path=args.app_profile,allow_hardware=args.allow_hardware)
        return 0
    if args.operation == "transcribe":
        from .voice import LocalVoice
        result = LocalVoice(args.voice_dir).transcribe(args.audio_file,language=args.language)
    else:
        from .session import Session
        session = Session(args.model_dir,".runtime/memory-cli/events.jsonl")
        requests = args.text if args.operation=="command" else ["1번과 3번에 두 개씩 있어","지금 배치 기억해","2번에 모두 쌓아","아까대로 복원해"]
        result = [{"request":text,**session.execute(text)} for text in requests]
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
