"""Landing funnel, cross arm (v5, 13 Nov; four: nose +y and tail -y as drawn, nose -y and tail +y mirrored): setup, loads and room in
parts/box_funnel_lib.py and parts/box_funnel/make.py."""
from parts.box_funnel_lib import arm_setup, SPEC_ARM
SPEC = dict(SPEC_ARM)


def setup(h):
    return arm_setup(h)
