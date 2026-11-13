"""Isolated development intervention: remove only held-object placement attraction.

Install this only in the single-candidate worker, never in a shared runtime.
The original policy calls this function for held-object MOVE proposals; returning
the proposal and zero weight follows its existing no-correction branch.  The
grasp approach, frame conversion, independent IK rejection and guard are not
replaced.  There is no scene, task, trajectory or action-dependent rule here.
"""
from contextlib import contextmanager

from . import scene_placement_belief_policy as placement


@contextmanager
def without_placement_attraction():
    """Temporarily change the sole placement field and restore it on exit."""
    original = placement.attracted_position
    if getattr(original, "_placement_attraction_disabled", False):
        raise RuntimeError("placement attraction is already disabled")
    audit = {"calls": 0, "restored": False}

    def identity(proposed, source_xy, goal, peak_height):
        audit["calls"] += 1
        return list(proposed), 0.0

    identity._placement_attraction_disabled = True
    placement.attracted_position = identity
    try:
        yield audit
    finally:
        intact = placement.attracted_position is identity
        placement.attracted_position = original
        audit["restored"] = placement.attracted_position is original
        if not intact:
            raise RuntimeError("the intervention was modified while installed")
