"""Overseer rear arm, ROOT: the printed block between the plates that the two 6/4 tubes of the arm glue into (29 Oct, Spark: tube
arms, as light as possible, no clamps). Geometry, cases and numbers: parts/overseer_tube_arm_lib.py on parts/overseer_arm_lib.py.
Motor at motors_xy[1] (-73.03, 139.5); spokes 2.1 / 2.2 / 2.3 end on the PAD at (-53.26, 119.7, 25); tubes at z 14 and 36 along the
arm axis. Arm 3 is the y-mirror."""
from parts.overseer_tube_arm_lib import TubeArm, build_piece
ARM = TubeArm("rear", 135.0, 1, ["spoke 2.1", "spoke 2.2", "spoke 2.3"], [-53.26, 119.7, 25.0], "root")
SPEC = dict(h=1.5, gen_cm3=6.0, max_worst=0.5)      # as the front root (E)


def setup(h):
    return build_piece(ARM, h)
