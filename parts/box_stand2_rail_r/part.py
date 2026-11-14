"""Box stand v2, front rail right (+X) (setup 2, 14 Nov 2026): from the splice at W Y 110 to the front foot at W Y 232. Setup in parts/box_stand2_frame_lib.py.
Setup 1 (lapped onto the base half's cheek, split in X) failed 0.71 / 0.64 at the tongue's root in box handling, uneven floor."""
from parts.box_stand2_frame_lib import rail_setup


def setup(h):
    return rail_setup(h, 1)


SPEC = dict(h=1.6, gen_cm3=100.0, max_worst=0.5)
