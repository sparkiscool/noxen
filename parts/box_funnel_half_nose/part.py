"""Landing funnel, nose half of the frame (v5, 13 Nov): setup, loads and room in parts/box_funnel_lib.py and parts/box_funnel/make.py."""
from parts.box_funnel_lib import half_setup, SPEC_HALF
SPEC = dict(SPEC_HALF)


def setup(h):
    return half_setup(1, h)
