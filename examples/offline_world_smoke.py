"""Offline symbolic-world smoke check; no model, SDK, network, or robot calls.

Run from the repository root: python -m examples.offline_world_smoke
This checks the task representation, NOT learned policy quality or robot success.
"""
from raccoon_memory.world import INITIAL, action_id, features, key, state, step


def main():
    before = state(INITIAL)
    after = step(before, action_id("A", "B"))
    assert after == ((), (2, 1), (3,), (4,))
    assert len(features(after)) == 80
    restored = step(after, action_id("B", "A"))
    assert restored == before
    try:
        state(((1,), (1,), (3,), (4,)))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate cylinder IDs must be rejected")
    print("PASS: symbolic move, restoration, 80-value encoding, invalid-state rejection")
    print(f"{key(before)} -> {key(after)} -> {key(restored)}")
    print("Not tested: learned weights, LLM inference, physics, or hardware.")


if __name__ == "__main__":
    main()
