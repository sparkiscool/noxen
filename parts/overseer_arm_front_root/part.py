"""Overseer front arm, ROOT: the printed block between the plates that the two 6/4 tubes of the arm glue into (29 Oct, Spark: tube
arms, as light as possible, no clamps). Geometry, cases and numbers: parts/overseer_tube_arm_lib.py on parts/overseer_arm_lib.py.
Motor at motors_xy[0] (205.97, 139.5); spokes 1.1 / 1.2 / 1.3 end on the PAD at (178.94, 146.7, 8); tubes at z 14 and 36 along the
arm axis from this block (socket floors 12 out from the root centre) to the pad. Arm 4 is the y-mirror."""
from parts.overseer_tube_arm_lib import TubeArm, build_piece
ARM = TubeArm("front", 45.0, 0, ["spoke 1.1", "spoke 1.2", "spoke 1.3"], [178.94, 146.7, 8.0], "root")
SPEC = dict(h=1.5, gen_cm3=6.0, max_worst=0.5)      # a block ~45 x 45 x 45 joining two plate blocks, two pegs and two tube collars (E)


def setup(h):
    return build_piece(ARM, h)
