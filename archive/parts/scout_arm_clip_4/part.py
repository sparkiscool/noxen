"""Scout arm clip 4 (4 Nov): the printed clip under arm 4's tip on the motor's four M2 screws, with the socket of the cage tie
from the arm tip down to the skid end (tools/scout_cage.TIES; v20). Geometry, loads and the other three: parts/scout_units_lib.py."""
from parts.scout_units_lib import arm_clip_setup
SPEC = dict(h=0.8, gen_cm3=2.1, max_worst=0.3)   # 8 Nov (the Overseer brief, PCTG): limit 0.3, budget x1.4 (cage v20 run 1 at 1.5: 3.7 g, 0.19..0.27)


def setup(h):
    return arm_clip_setup(4, h)
