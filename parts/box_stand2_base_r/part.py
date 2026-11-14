"""Box stand v2, base half right (+X, under the PSU) (14 Nov 2026): the mast's foot out to its rail, the rear foot and the lap's cheek. Setup in parts/box_stand2_lib.py."""
from parts.box_stand2_lib import base_setup


def setup(h):
    return base_setup(h, 1)


SPEC = dict(h=2.3, gen_cm3=220.0, max_worst=0.5)
