"""Box stand v2, rear rail left (-X) (setup 2, 14 Nov 2026): the rail from the yoke's joint and the rear foot to the splice at W Y 110. Setup in parts/box_stand2_frame_lib.py."""
from parts.box_stand2_frame_lib import rear_setup


def setup(h):
    return rear_setup(h, -1)


SPEC = dict(h=1.8, gen_cm3=110.0, max_worst=0.5)
