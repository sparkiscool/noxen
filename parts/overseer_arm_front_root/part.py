"""Overseer front arm, ROOT piece (plate bosses to the joint, ~144 mm): the front arm in two bolted prints (Spark 29 Oct: one piece is 226 x 65 mm, over
the A1 mini's 180 plate at every rotation). Geometry, joint, cases and numbers: parts/overseer_arm_split_lib.py on parts/overseer_arm_lib.py.
Motor pad at motors_xy[0] (205.97, 139.5), arm top z 45, thrust plane z 82; spokes 1.1 / 1.2 / 1.3 end in drawn 3.2 x 10 sockets at
(178.94, 146.7, 8); joint plane 130 mm out along the arm axis: 2 x 6.0 carbon pins + 2 x M3 along the axis, contact pads."""
from parts.overseer_arm_split_lib import SplitArm, build_piece
ARM = SplitArm("front", 45.0, 0, ["spoke 1.1", "spoke 1.2", "spoke 1.3"], [178.94, 146.7, 8.0], "root")
SPEC = dict(h=2.0, gen_cm3=22.0, max_worst=0.5)      # 28 (10 % of the design space, ten attachments at the plates) ended with a plate boss off and a bolt case at 2.5e4 of solid (29 Oct); the tip gave back 18; 50 (29 Oct): 40 with the joint block passed the solid gate at 0.72 in the stiffness case at the top plate block edge; 22 (29 Oct, Spark: super light): the arm-alone case is handling at 2 g, no thrust


def setup(h):
    return build_piece(ARM, h)
