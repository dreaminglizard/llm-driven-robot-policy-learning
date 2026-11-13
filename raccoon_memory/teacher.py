"""GPT-authored symbolic examples from this working conversation.

These explicit decisions were written by the assistant, not returned by a new
external API call. Relabellings are programmatic augmentation, and exact-search
labels (when used) have a different source. None are physical grasp successes.
"""
from itertools import permutations

from .world import INITIAL, MOVES, action_id, relabel, state, step


CASES = (
    {"name": "gather_four_at_position_one", "start": INITIAL,
     "goal": ((1,2,3,4), (), (), ()), "moves": (("B","A"), ("C","A"), ("D","A"))},
    {"name": "separate_four_to_registered_positions", "start": ((1,2,3,4), (), (), ()),
     "goal": INITIAL, "moves": (("A","D"), ("A","C"), ("A","B"))},
    {"name": "reverse_tower", "start": ((1,2,3,4), (), (), ()),
     "goal": ((), (4,3,2,1), (), ()), "moves": (("A","B"), ("A","B"), ("A","B"), ("A","B"))},
    {"name": "move_tower_preserving_order", "start": ((1,2,3,4), (), (), ()),
     "goal": ((), (1,2,3,4), (), ()),
     "moves": (("A","C"), ("A","C"), ("A","C"), ("A","B"), ("C","B"), ("C","B"), ("C","B"))},
    {"name": "retrieve_bottom_and_restore_remaining", "start": ((1,2,3,4), (), (), ()),
     "goal": ((2,3,4), (), (), (1,)),
     "moves": (("A","B"), ("A","B"), ("A","B"), ("A","D"), ("B","A"), ("B","A"), ("B","A"))},
    {"name": "split_tower_into_two", "start": ((1,2,3,4), (), (), ()),
     "goal": ((1,2), (), (3,4), ()), "moves": (("A","B"), ("A","C"), ("B","C"))},
    {"name": "combine_two_towers", "start": ((1,2), (), (3,4), ()),
     "goal": ((1,2,3,4), (), (), ()), "moves": (("C","B"), ("C","A"), ("B","A"))},
    {"name": "remember_two_towers", "start": INITIAL,
     "goal": ((1,2), (), (3,4), ()), "moves": (("B","A"), ("D","C"))},
    {"name": "remember_other_past", "start": INITIAL,
     "goal": ((1,4), (), (3,2), ()), "moves": (("D","A"), ("B","C"))},
    {"name": "unstack_to_registered_slots", "start": ((1,2), (), (3,4), ()),
     "goal": INITIAL, "moves": (("A","B"), ("C","D"))},
    {"name": "swap_remembered_tops", "start": ((1,2), (), (3,4), ()),
     "goal": ((1,4), (), (3,2), ()), "moves": (("A","B"), ("C","D"), ("D","A"), ("B","C"))},
    {"name": "exchange_two_flat_objects", "start": INITIAL,
     "goal": ((2,), (1,), (3,), (4,)), "moves": (("A","C"), ("B","A"), ("C","B"))},
    {"name": "retrieve_lower_object", "start": ((1,2), (), (3,4), ()),
     "goal": ((), (2,), (3,4), (1,)), "moves": (("A","B"), ("A","D"))},
    {"name": "exchange_tower_bases", "start": ((1,2), (), (3,4), ()),
     "goal": ((3,2), (), (1,4), ()),
     "moves": (("A","B"), ("C","D"), ("A","B"), ("C","A"), ("B","C"), ("B","A"), ("D","C"))},
    {"name": "four_object_rotation", "start": INITIAL,
     "goal": ((4,), (1,), (2,), (3,)),
     "moves": (("A","B"), ("D","A"), ("C","D"), ("B","C"), ("B","D"), ("C","B"), ("D","C"))},
)


def validated_cases():
    result = []
    for case in CASES:
        current, goal = state(case["start"]), state(case["goal"])
        steps = []
        for src, dst in case["moves"]:
            action = action_id(src, dst)
            nxt = step(current, action)
            steps.append({"current": current, "action": action, "next": nxt})
            current = nxt
        if current != goal:
            raise ValueError(f"GPT example does not achieve its goal: {case['name']}")
        result.append({**case, "steps": steps, "symbolic_success": True,
                       "physical_success": None, "source": "gpt_authored_in_current_conversation",
                       "external_model_call": False, "observed_runtime_model": None})
    return result


def augmented_examples(allowed_goals):
    allowed_goals = set(allowed_goals)
    for case in validated_cases():
        for objects in permutations(range(1,5)):
            for slots in permutations(range(4)):
                goal = relabel(case["goal"], objects, slots)
                if goal not in allowed_goals:
                    continue
                for item in case["steps"]:
                    current = relabel(item["current"], objects, slots)
                    src, dst = MOVES[item["action"]]
                    action = MOVES.index((slots[src], slots[dst]))
                    yield current, goal, action, case["name"]
