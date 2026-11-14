"""Box stand v2, yoke left (-X) (setup 2, 14 Nov 2026): the mast's foot out to the rear rail's inner face. Setup in parts/box_stand2_frame_lib.py."""
from parts.box_stand2_frame_lib import yoke_setup


def setup(h):
    return yoke_setup(h, -1)


SPEC = dict(h=1.8, gen_cm3=160.0, max_worst=0.5)
