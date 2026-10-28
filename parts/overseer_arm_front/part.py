"""Overseer front arm (arm 1 at +y, arm 4 is its y-mirror), v6 setup: one piece, spokes end on the arm, every bolt a case.
Geometry, cases and numbers: parts/overseer_arm_lib.py. Motor pad at motors_xy[0] (205.97, 139.5), arm top z 45, thrust plane z 82;
spokes 1.1 / 1.2 / 1.3 end in drawn 3.2 x 10 sockets at (178.94, 146.7, 8). See the setup docstring of the lib for supports and cases."""
from parts.overseer_arm_lib import Arm, build

ARM = Arm("front", 45.0, 0, ["spoke 1.1", "spoke 1.2", "spoke 1.3"], [178.94, 146.7, 8.0])
SPEC = dict(h=1.2, gen_cm3=48.0, max_worst=0.5)


def setup(h):
    return build(ARM, h)
