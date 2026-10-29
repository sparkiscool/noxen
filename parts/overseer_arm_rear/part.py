"""Overseer rear arm (arm 2 at +y, arm 3 is its y-mirror), v9 setup (29 Oct, Spark: spoke 2.3 ran through the rear motor pad and the
rear end sits close to the body anyway: no rods there, one printed PETG piece from the plates to the motor, no tubes, no spokes).
Geometry and cases: parts/overseer_arm_lib.py; with no spokes every flight case holds at the plates alone: 4 g + 21 N thrust with
+-0.5 N m motor torque, landing 6 g, the frame model's crash set (the motor's inertia only; the rear arm carries no cage load now),
HANDLING. Motor pad at motors_xy[1] (-73.03, 139.5), arm top z 45, thrust plane z 82. Replaces the tube-arm root + pad pieces (v8)."""
from parts.overseer_arm_lib import Arm, build

ARM = Arm("rear", 135.0, 1, [], [-73.03, 139.5, 25.0])
SPEC = dict(h=2.0, gen_cm3=32.0, max_worst=0.5)   # h 2.0 (29 Oct: at 1.2 the one-piece grid was 1.0 M cells); 32 (E): the v6 one-piece arm took 40 with the flight-at-the-plates case and 22 without it; the rear spokes are gone, so that case is back


def setup(h):
    return build(ARM, h)
