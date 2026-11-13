"""Numbered task scope shared by planning and physical execution."""


def enabled_slots(value):
    result=tuple(value)
    if (len(result)<2 or len(set(result))!=len(result)
            or any(type(slot) is not int or slot not in range(1,5) for slot in result)):
        raise ValueError("사용 자리는 서로 다른 1~4번 중 두 개 이상이어야 합니다.")
    return result


def presentation(slots,max_height):
    slots=enabled_slots(slots)
    first,second=slots[:2]
    examples=[f'{first}번에서 {second}번으로 옮겨']
    if max_height>=2:
        examples.append(f'{second}번에 2층 쌓아')
    else:
        examples.append(f'{second}번에서 {first}번으로 옮겨')
    if len(slots)>2:
        examples.append(f'{slots[-1]}번과 {first}번 위치 바꿔')
    examples.extend(['지금 배치 기억해','초기 배치로 돌아가','아까대로 복원해'])
    return {'description':'시작 배치를 등록하고, '+ '·'.join(map(str,slots))+
            f'번의 이동·기억·복원을 요청하세요. 최대 {max_height}층입니다.',
            'examples':tuple(examples),'max_height':max_height}
