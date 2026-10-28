"""Generic drone motor mount (a demo, not a Nox part: every dimension here is an assumption, E).

A 2207-class motor (16 x 19 mm M3 pattern, ~1.3 kg max thrust) stands 38 mm above a 3 mm carbon plate on a printed
PETG mount bolted with 4 x M3 on a 24 mm square. Drawn: 4 bolt pads (OD 10, 4 tall, M3 web 3.3) and the motor seat
(disc r 15, 4 thick) with its 4 M3 holes and the shaft bore. Cuts: bolt holes, a key / nut access column over each
plate bolt, the motor screw heads' access under the seat, the shaft / bearing clearance. Supports: the pads' undersides.
Loads: thrust, yaw torque reaction, crash pushes on the motor, landing on the motor, flight vibration (body force).
Build +z (plate face on the bed)."""
import numpy as np
import manifold3d as mf
from ..part import Part

PLATE_BOLTS = [(sx * 12.0, sy * 12.0) for sx in (1, -1) for sy in (1, -1)]
MOTOR_BOLTS = [(8.0, 0.0), (-8.0, 0.0), (0.0, 9.5), (0.0, -9.5)]          # 16 x 19 pattern
Z_SEAT0, Z_TOP = 32.0, 38.0
# v3 (23 Oct): pads 6 mm tall so the solid gate (2.0 mm erosion) keeps a 2 mm core (v2: Ø13 x 5 failed by 4 voxels of 2370).
# v2 (23 Oct): pads Ø13 x 5 (README pad rule >= Ø12; 4.8 web round the M3 so a core survives the solid gate's 2.0 mm
# erosion) and a 6 mm seat (v1: Ø10 x 4 pads and a 4 mm seat had no core at 2.0 mm erosion and failed the gate)
PAD_R, PAD_T = 6.5, 6.0


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl(x, y, z0, z1, r, n=48):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def bodies():
    pads = [cyl(x, y, 0.0, PAD_T, PAD_R) for x, y in PLATE_BOLTS]
    seat = cyl(0, 0, Z_SEAT0, Z_TOP, 15.0, 96)
    drawn = seat
    for p in pads:
        drawn += p
    cuts = cyl(0, 0, 18.0, Z_TOP + 1, 4.0)                                   # shaft / bearing / circlip clearance
    for x, y in PLATE_BOLTS:
        cuts += cyl(x, y, -1.0, PAD_T + 1, 1.7) + cyl(x, y, PAD_T, Z_TOP + 1, 3.3)   # M3 + key / nut access from above
    for x, y in MOTOR_BOLTS:
        cuts += cyl(x, y, Z_SEAT0 - 1, Z_TOP + 1, 1.7) + cyl(x, y, 24.0, Z_SEAT0, 3.0)   # screw + head from below
    env = cyl(0, 0, 0.0, Z_TOP, 21.0, 96) + box(-18, 18, -18, 18, 0.0, Z_TOP)
    return dict(pads=pads, seat=seat, drawn=drawn, cuts=cuts, env=env)


def setup(h):
    B = bodies()
    p = Part("motor_mount", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG")
    seats = None
    for x, y in PLATE_BOLTS:
        s = cyl(x, y, -1.0, 1.5 * h, PAD_R)
        seats = s if seats is None else seats + s
    p.support("default", seats)
    seat = B["seat"]
    half = lambda x0, x1: seat ^ box(x0, x1, -20, 20, 0, 40)
    p.load("thrust 13 N up", seat, (0, 0, 13.0), cls="flight")
    p.load("max thrust 20 N up", seat, (0, 0, 20.0), cls="misuse")
    # yaw torque reaction 0.35 N m as two opposite tangential pushes at r ~ 10 mm: 17.5 N each side
    p.load_multi("yaw torque 0.35 Nm", [(half(2, 20), (0, 17.5, 0)), (half(-20, -2), (0, -17.5, 0))], cls="flight")
    for v, nm in (((40, 0, 0), "+X"), ((-40, 0, 0), "-X"), ((0, 40, 0), "+Y"), ((0, -40, 0), "-Y")):
        p.load(f"crash push 40 N {nm} on the motor", seat, v, cls="crash")
    p.load("landing on the motor 60 N down", seat, (0, 0, -60.0), cls="crash")
    p.load("vibration 30 g + thrust", seat, (0, 0, 13.0), cls="vibration", accel=(0, 0, 30.0))
    return p
