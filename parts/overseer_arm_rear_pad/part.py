"""Overseer rear arm, PAD: the printed motor pad at the end of the two 6/4 tubes (29 Oct, Spark: tube arms, as light as possible, no
clamps): motor slots, shaft bore, the three 3 mm spoke sockets and two tube sockets facing the root. Geometry, cases and numbers:
parts/overseer_tube_arm_lib.py on parts/overseer_arm_lib.py. Motor at motors_xy[1] (-73.03, 139.5), pad top z 45, thrust plane z 82;
spokes 2.1 / 2.2 / 2.3 end at (-53.26, 119.7, 25). Arm 3 is the y-mirror."""
from parts.overseer_tube_arm_lib import TubeArm, build_piece
ARM = TubeArm("rear", 135.0, 1, ["spoke 2.1", "spoke 2.2", "spoke 2.3"], [-53.26, 119.7, 25.0], "pad")
SPEC = dict(h=1.5, gen_cm3=7.0, max_worst=0.5)      # as the front pad (E)


def setup(h):
    return build_piece(ARM, h)
