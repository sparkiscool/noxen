"""Overseer rear arm (arm 2 at +y, arm 3 is its y-mirror), v6 setup: one piece, spokes end on the arm, every bolt a case.
Geometry, cases and numbers: parts/overseer_arm_lib.py. Motor pad at motors_xy[1] (-73.03, 139.5), arm top z 45, thrust plane z 82;
spokes 2.1 / 2.2 / 2.3 end in drawn 3.2 x 10 sockets at (-53.26, 119.7, 25). See the setup docstring of the lib for supports and cases."""
from parts.overseer_arm_lib import Arm, build

ARM = Arm("rear", 135.0, 1, ["spoke 2.1", "spoke 2.2", "spoke 2.3"], [-53.26, 119.7, 25.0])
SPEC = dict(h=2.0, gen_cm3=30.0, max_worst=0.5)   # h 2.0 (29 Oct): at 1.2 the grid was 1.0 M / 0.4 M cells, multigrid never converged, no attempt reached a fine iteration


def setup(h):
    return build(ARM, h)
