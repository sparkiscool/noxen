"""Box stand v2, rail left (-X) (14 Nov 2026): the floor rail forward of the lap to the front foot at W Y 232. Setup in parts/box_stand2_lib.py."""
from parts.box_stand2_lib import rail_setup


def setup(h):
    return rail_setup(h, -1)


SPEC = dict(h=1.6, gen_cm3=80.0, max_worst=0.5)
