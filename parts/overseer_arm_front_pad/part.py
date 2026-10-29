"""Overseer front arm, PAD: the printed motor pad at the end of the two 6/4 tubes (29 Oct, Spark: tube arms, as light as possible,
no clamps): motor slots, shaft bore, the three 3 mm spoke sockets and two tube sockets facing the root. Geometry, cases and numbers:
parts/overseer_tube_arm_lib.py on parts/overseer_arm_lib.py. Motor at motors_xy[0] (205.97, 139.5), pad top z 45, thrust plane z 82;
spokes 1.1 / 1.2 / 1.3 end at (178.94, 146.7, 8). Arm 4 is the y-mirror."""
from parts.overseer_tube_arm_lib import TubeArm, build_piece
ARM = TubeArm("front", 45.0, 0, ["spoke 1.1", "spoke 1.2", "spoke 1.3"], [178.94, 146.7, 8.0], "pad")
SPEC = dict(h=1.5, gen_cm3=7.0, max_worst=0.5)      # the tip piece passed at 12 with a joint; the pad has two tube collars instead (E)


def setup(h):
    return build_piece(ARM, h)
