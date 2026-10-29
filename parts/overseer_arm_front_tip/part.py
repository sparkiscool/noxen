"""Overseer front arm, TIP piece (joint to the motor pad with the 3 spoke sockets, ~74 mm): the front arm in two bolted prints (Spark 29 Oct: one piece is 226 x 65 mm, over
the A1 mini's 180 plate at every rotation). Geometry, joint, cases and numbers: parts/overseer_arm_split_lib.py on parts/overseer_arm_lib.py.
Motor pad at motors_xy[0] (205.97, 139.5), arm top z 45, thrust plane z 82; spokes 1.1 / 1.2 / 1.3 end in drawn 3.2 x 10 sockets at
(178.94, 146.7, 8); joint plane 130 mm out along the arm axis: 2 x 6.0 carbon pins + 2 x M3 along the axis, contact pads."""
from parts.overseer_arm_split_lib import SplitArm, build_piece
ARM = SplitArm("front", 45.0, 0, ["spoke 1.1", "spoke 1.2", "spoke 1.3"], [178.94, 146.7, 8.0], "tip")
SPEC = dict(h=2.0, gen_cm3=9.0, max_worst=0.5)      # 30 gave index 0.10 on the frame-model loads (29 Oct, Spark: cut it); 20 gave 1.38 on the old rigid-core 1465 N spoke pull; 9 (29 Oct, Spark: super light)


def setup(h):
    return build_piece(ARM, h)
