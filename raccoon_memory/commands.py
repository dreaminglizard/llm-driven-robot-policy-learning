"""Bounded Korean task grammar. It compiles goals, never joint/move sequences."""
from dataclasses import dataclass
import re
import unicodedata

from .world import state


@dataclass(frozen=True)
class Command:
    kind: str
    slots: tuple = ()
    height: int | None = None
    counts: tuple | None = None
    all_objects: bool = False


def normalize(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 300:
        raise ValueError("짧은 작업 지시를 입력해 주세요.")
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"[\s.!?。]", "", value)
    for word, number in (("일",1),("이",2),("삼",3),("사",4),("오",5),("육",6)):
        value = value.replace(word+"번",str(number)+"번")
        value = value.replace(word+"층",str(number)+"층")
    for word, number in (("네",4),("넷",4),("세",3),("셋",3),("두",2),("둘",2),("한",1),("하나",1),("영",0),("제로",0)):
        value = value.replace(word+"개",str(number)+"개").replace(word+"층",str(number)+"층")
        value = value.replace(word+"씩",str(number)+"개씩")
    return value


def parse(text):
    t = normalize(text)
    if re.search(r"지마|지말|않|말고|말아|말라|마라|안돼|금지|안(?:옮|쌓|바꾸|기억|복원|이동|교환|돌아|가져)", t):
        raise ValueError("부정·정정이 섞인 지시는 실행하지 않습니다. 원하는 작업을 한 문장으로 말해 주세요.")
    if re.search(r"그리고|한뒤|한후|다음에|고나서|하면서|기억하고|복원하고|쌓고|바꾸고|옮기고",t):
        raise ValueError("한 번에 한 가지 작업을 요청해 주세요.")
    if "기억해" in t and any(word in t for word in ("복원","돌아가","아까대로","원래대로")):
        raise ValueError("기억과 복원은 각각 한 번씩 요청해 주세요.")
    if t in {"정지","정지해","멈춰","멈춰줘","멈춰라","그만","그만해","스톱","stop",
             "취소","취소해","취소해줘","작업취소","중지","중지해","중지해줘","작업중지"}:
        return Command("stop")
    if any(word in t for word in ("빨간","빨강","붉은","초록","녹색","노란","노랑","파란","파랑","푸른",
                                  "색","이것","저것","그것","이거","저거","그거","여기","저기",
                                  "왼쪽","오른쪽","가까운","먼쪽","앞쪽","뒤쪽")):
        raise ValueError("색상이나 상대 위치로 물체를 구분하지 않습니다. 출발·도착 자리 번호를 말해 주세요.")
    raw_slots = tuple(int(n) for n in re.findall(r"(\d+)번",t))
    if any(n not in range(1,5) for n in raw_slots):
        raise ValueError("자리 번호는 1번부터 4번까지만 있습니다.")
    slots = tuple(n-1 for n in raw_slots)
    if any(word in t for word in ("복원","아까대로","기억한대로","원래대로")):
        reference=r"(?:아까대로|기억한대로|원래대로|(?:(?:아까)?기억한|저장한|이전|아까|원래)(?:배치|상태)로)"
        restore=rf"(?:{reference}|(?:{reference})?(?:복원해|돌아가)(?:줘|주세요|줘요)?)"
        if slots or not re.fullmatch(restore,t):
            raise ValueError("복원은 ‘기억한 배치로 복원해’라고 말해 주세요.")
        return Command("restore")
    if "기억" in t:
        if slots or not re.fullmatch(r"(?:지금|현재)?(?:(?:전체)?(?:배치|상태)(?:를)?)?기억해(?:줘|주세요|줘요)?",t):
            raise ValueError("전체 배치를 기억합니다. ‘지금 배치 기억해’라고 말해 주세요.")
        return Command("remember")
    if re.fullmatch(r"(?:(?:초기|처음)배치로돌아가(?:줘|주세요|줘요)?|초기화)",t):
        return Command("initial")
    if any(word in t for word in ("어디","현재배치","배치알려","상태알려")):
        return Command("status")
    if any(word in t for word in ("있어","있다","등록","놓여","위치설정","배치설정")):
        count_body=re.sub(r"[1-4]번|\d+개","",t)
        if not re.fullmatch(r"(?:지금|현재|자리|원통|에는|에|은|는|과|와|,|씩|놓여|있어|있다|등록|위치설정|배치설정|해|줘|주세요|를)*",count_body):
            raise ValueError("현재 배치의 자리별 개수만 말해 주세요. 다른 조건이 섞인 등록은 실행하지 않습니다.")
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("각 자리의 원통 개수를 한 번씩 알려 주세요.")
        counts = [0]*4
        groups = re.findall(r"([1-4])번(?:자리)?(?:에는|에|은|는)?(\d+)개",t)
        if "씩" in t:
            values = re.findall(r"(\d+)개씩",t)
            if len(values) != 1 or len(re.findall(r"\d+개",t)) != 1:
                raise ValueError("자리마다 다른 개수는 ‘1번에 2개, 3번에 2개’처럼 말해 주세요.")
            for slot in slots:
                counts[slot] = int(values[0])
        elif len(groups) == len(slots):
            for slot, count in groups:
                counts[int(slot)-1] = int(count)
        else:
            raise ValueError("각 자리의 원통 개수가 필요합니다. 예: ‘1번과 3번에 두 개씩 있어’")
        if sum(counts) != 4 or any(n<0 or n>4 for n in counts):
            raise ValueError("등록할 원통은 합계 4개여야 합니다. 말하지 않은 자리는 빈자리로 처리합니다.")
        return Command("register", counts=tuple(counts))
    kinds = {"swap": any(w in t for w in ("바꿔","교환","맞바꾸")),
             "stack": any(w in t for w in ("쌓","탑만들","탑세워")),
             "move": any(w in t for w in ("옮겨","이동","가져다","가져가"))}
    chosen = [k for k,v in kinds.items() if v]
    if len(chosen) != 1:
        raise ValueError("한 번에 이동·쌓기·교환 중 한 가지를 요청해 주세요.")
    kind = chosen[0]
    endings = {
        "swap": r"(?:바꿔(?:줘|주세요|줘요|라)?|교환해(?:줘|주세요|줘요)?|맞바꾸어줘)$",
        "stack": r"(?:쌓아(?:줘|주세요|줘요|라)?|탑만들어(?:줘|주세요)?|탑세워(?:줘|주세요)?)$",
        "move": r"(?:옮겨(?:줘|주세요|줘요|라)?|이동해(?:줘|주세요|줘요)?|이동시켜(?:줘|주세요)?|가져다줘|가져가(?:줘|주세요)?)$",
    }
    ending = re.search(endings[kind],t)
    if not ending:
        raise ValueError("작업 지시를 확인하지 못했습니다. 화면의 예시처럼 요청해 주세요.")
    # Consume the whole request vocabulary: unsupported object/order/speed
    # qualifiers must not disappear merely because two slot numbers are present.
    body=t[:ending.start()]
    body=re.sub(r"[1-4]번|\d+(?:개|층)","",body)
    words=("에있는","맨위에있는","맨위","원통","물체","자리","위치","으로","에서",
           "에는","모두","전부","전체","지금","있는","과","와","의","에","로","을","를","만","좀")
    vocabulary="(?:"+"|".join(map(re.escape,sorted(words,key=len,reverse=True)))+")*"
    if not re.fullmatch(vocabulary,body):
        raise ValueError("대상·순서·조건을 명확히 해석하지 못했습니다. 자리 번호와 한 가지 작업만 말해 주세요.")
    has_all=any(w in t for w in ("모두","전부","전체"))
    if has_all and re.search(r"\d+(?:개|층)",t):
        raise ValueError("전체 이동과 개수·층수를 동시에 지정하지 말아 주세요.")
    if kind == "swap":
        if re.search(r"\d+(?:개|층)",t) or "맨위" in t:
            raise ValueError("위치 교환은 두 자리의 원통 전체를 바꿉니다. 일부 원통 교환은 지원하지 않습니다.")
        if len(slots) != 2 or slots[0] == slots[1]:
            raise ValueError("교환할 서로 다른 자리 두 개를 말해 주세요.")
        return Command(kind, slots)
    if kind == "stack":
        levels = re.findall(r"(\d+)(?:층|개)",t)
        if len(slots) != 1 or len(levels)>1 or "더" in t:
            raise ValueError("‘3번에 3층 쌓아’처럼 최종 자리와 층수를 말해 주세요.")
        if not levels and not any(word in t for word in ("모두","전부","전체")):
            raise ValueError("‘3번에 3층 쌓아’ 또는 ‘3번에 모두 쌓아’처럼 개수를 말해 주세요.")
        height = int(levels[0]) if levels else 4
        if not 1 <= height <= 4:
            raise ValueError("탑 높이는 1층부터 4층까지입니다.")
        return Command(kind, slots, height)
    if len(slots) != 2 or slots[0] == slots[1]:
        raise ValueError("‘1번에서 3번으로 옮겨’처럼 출발과 도착 번호를 말해 주세요.")
    if "층" in t:
        raise ValueError("이동은 맨 위 한 개 또는 전체로 요청해 주세요. 특정 층 선택은 지원하지 않습니다.")
    # The first number is the source only when its phrase is explicit.
    source = re.search(r"([1-4])번(?:자리)?(?:에서|에있는|의원통|원통|을|를)",t)
    destination = re.search(r"([1-4])번(?:자리)?(?:으로|로|에(?!있는|서))",t)
    if not source or not destination or source.group(1) == destination.group(1):
        raise ValueError("출발은 ‘1번에서’, 도착은 ‘3번으로’처럼 구분해 주세요.")
    amounts = re.findall(r"(\d+)개",t)
    if amounts and (len(amounts)!=1 or int(amounts[0])!=1):
        raise ValueError("이동은 맨 위 한 개 또는 ‘모두’로 요청해 주세요.")
    return Command(kind, (int(source.group(1))-1,int(destination.group(1))-1),
                   all_objects=any(w in t for w in ("모두","전부","전체")))


def registered_state(counts):
    stacks, obj = [], 1
    for count in counts:
        stacks.append(tuple(range(obj,obj+count)))
        obj += count
    return state(stacks)


def goal_for(command, current, *, enabled_slots=(1,2,3,4)):
    current = state(current)
    stacks = [list(s) for s in current]
    if command.kind == "move":
        src, dst = command.slots
        if not stacks[src]:
            raise ValueError("출발 자리가 비어 있습니다.")
        moved = stacks[src][:] if command.all_objects else [stacks[src][-1]]
        stacks[src] = [] if command.all_objects else stacks[src][:-1]
        stacks[dst].extend(moved)
    elif command.kind == "swap":
        a,b = command.slots
        stacks[a], stacks[b] = stacks[b], stacks[a]
    elif command.kind == "stack":
        dst = command.slots[0]
        if len(stacks[dst]) > command.height:
            raise ValueError("현재 탑보다 낮은 층수입니다. 먼저 다른 자리로 옮겨 주세요.")
        needed = command.height - len(stacks[dst])
        selected = [(src,obj) for src, items in enumerate(stacks)
                    if src != dst and src+1 in enabled_slots for obj in reversed(items)][:needed]
        if len(selected)<needed:
            raise ValueError("현재 사용 가능한 자리의 원통만으로 요청한 층수를 만들 수 없습니다.")
        for src,obj in selected:
            stacks[src].remove(obj)
            stacks[dst].append(obj)
    else:
        raise ValueError("이 요청은 이동 목표를 만들지 않습니다.")
    return state(stacks)


def summary(current):
    return ", ".join(f"{i+1}번 {len(items)}개" for i,items in enumerate(state(current)))
