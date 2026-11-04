"""Scout arm clip 4 (4 Nov): the printed clip under arm 4's tip on the motor's four M2 screws, with the socket of the cage tie
from the arm tip to the base rail corner (tools/scout_cage.TIES). Geometry, loads and the other three: parts/scout_units_lib.py."""
from parts.scout_units_lib import arm_clip_setup
SPEC = dict(h=0.8, gen_cm3=1.5, max_worst=0.5)


def setup(h):
    return arm_clip_setup(4, h)
