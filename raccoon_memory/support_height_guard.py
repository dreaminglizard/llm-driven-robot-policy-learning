"""Nominal support-height target adjustment for simulation and explicit mock use.

Inputs are registered destination coordinates, expected source/destination
levels, the arm's own observations, and its previous applied control target.
This is not a measured object pose, contact sensor, or physical safety guard.
"""
from __future__ import annotations

import math
import struct

from raccoon_control.kinematics import Geometry, check_joints


GEOMETRY = Geometry()
NOMINAL_CYLINDER_HEIGHT_MM = 20.0
CLEARANCE_MM = 1.0
DESTINATION_FOOTPRINT_MM = 10.0


def _finite_vector(value, length, name):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f'{name} must contain {length} finite values')
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f'{name} must contain {length} finite values')
    return tuple(float(item) for item in value)


class SupportHeightGuard:
    """Limit a downward target relative to the measured robot grasp height."""

    def __init__(self, task, destination_xy_mm):
        if (not isinstance(task, dict) or type(task.get('source_level')) is not int
                or type(task.get('destination_level')) is not int
                or task['source_level'] not in (1, 2) or task['destination_level'] not in (1, 2)):
            raise ValueError('expected source and destination levels are required')
        self.source_level = task['source_level']
        self.destination_level = task['destination_level']
        self.destination_xy_mm = _finite_vector(destination_xy_mm, 2, 'destination_xy_mm')
        self.close_observed_z_mm = None

    def apply(self, *, action, proposed_target_deg, previous_target_deg,
              gripper_opening_m, observed_tcp_mm):
        """Return the command target and a record only when a target is adjusted.

        ``observed_tcp_mm`` is a robot observation at the current decision,
        never a physical object measurement. A failed inverse calculation raises
        before command transmission; callers must halt the active transfer.
        """
        if action not in ('move', 'open', 'close', 'done'):
            raise ValueError('unsupported action')
        proposed = tuple(check_joints(proposed_target_deg))
        previous = tuple(check_joints(previous_target_deg))
        tcp = _finite_vector(observed_tcp_mm, 3, 'observed_tcp_mm')
        if type(gripper_opening_m) not in (int, float) or not math.isfinite(gripper_opening_m):
            raise ValueError('gripper target must be finite')
        if not 0. <= gripper_opening_m <= .05:
            raise ValueError('gripper target is outside its virtual range')
        if action == 'close':
            self.close_observed_z_mm = tcp[2]
            return list(proposed), None
        if (action != 'move' or gripper_opening_m != 0. or self.close_observed_z_mm is None
                or self.destination_level <= 1):
            return list(proposed), None

        start, target = GEOMETRY.forward(previous), GEOMETRY.forward(proposed)
        floor = (self.close_observed_z_mm + NOMINAL_CYLINDER_HEIGHT_MM
                 * (self.destination_level - self.source_level) + CLEARANCE_MM)
        if (target[2] >= start[2] or target[2] >= floor
                or max(math.dist(point[:2], self.destination_xy_mm) for point in (start, target))
                > DESTINATION_FOOTPRINT_MM):
            return list(proposed), None

        pitch = 90. + sum(proposed[1:])
        q = GEOMETRY.inverse((target[0], target[1], floor), pitch)
        # The studied controller returned normalized float32 angles to the
        # ordinary executor. Preserve that rounding in both execution paths.
        adjusted = [struct.unpack('f', struct.pack('f', value / 180.))[0] * 180. for value in q]
        check_joints(adjusted)
        return adjusted, {'proposed_joints_deg': list(proposed), 'adjusted_joints_deg': list(adjusted),
                          'source_close_observed_tcp_z_mm': self.close_observed_z_mm,
                          'source_level': self.source_level, 'destination_level': self.destination_level,
                          'nominal_object_height_mm': NOMINAL_CYLINDER_HEIGHT_MM,
                          'clearance_mm': CLEARANCE_MM, 'floor_tcp_z_mm': floor,
                          'proposed_tcp_z_mm': target[2]}
