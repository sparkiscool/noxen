"""Box stand v2, the frame under the mast (setup 2, 14 Nov 2026): per side a yoke, a rear rail and a front rail, in W. Geometry, the
arm's cases, the floor and the helpers from parts/box_stand2_lib.py (the mast's lib, unchanged).

Why setup 2 (14 Nov, the first runs): the base halves (the yoke and the rail's rear part in one, 177 x 172 x 76) were refused for memory
before their build (16.8 GB: the gate grid and the mesh of a part that size), and the rails, lapped onto the halves' cheeks side by
side in X, failed 0.71 / 0.64 at the tongue's root in box handling on an uneven floor (the 2 g side friction at the front foot bends the
rail about Z, and a lap split in X leaves the root half its width). Now every rail section is whole:
  yoke_r / _l     under the mast's foot (X 0..120, Y -64..62; its 4 pads take 4 M5 x 16 up into the foot's inserts) out to its rear rail's
                  inner face at |X| 120, where 6 M5 x 60 come in from the rail's outer face (heads outboard) into heat-set inserts in the yoke.
  rear_r / _l     the rail from Y -64 to its splice plate at Y 110 (|X| 120..177.5), with the rear foot at (+-149, -44).
  rail_r / _l     the rail from its splice plate at Y 110 to the front foot at (+-149, 232).
The splice: a butt joint, 4 M5 x 16 hex bolts along Y at the section's corners, from heads in pockets behind the front rail's plate
(turned with an 8 mm spanner through windows to the rail's side faces, no tool sweep along the rail) into heat-set inserts in the
rear rail's plate.
Supports are the joint toward the mast (the yoke: the mast's foot; the rear rail: its face on the yoke; the front rail: its splice
plate); loads the floor at the feet, carried through the joints rigidly; own weight as body forces.
"""
import numpy as np
from topo.part import Part
from parts.box_stand2_lib import (box, cyl, U, mirror_x, sweep, PSU, FOOT_BOLTS, FEET, FOOT_X, FOOT_R, STUD_R, Z_FOOT, BED, FLOOR,
                                  RAIL_X, RAIL_TOP, floor_cases, rigid_split)

X_IN = RAIL_X[0]                    # the yoke's end / the rail's inner face
Y_SPLICE = 110.0
YOKE_Y = (-64.0, 62.0)                # 40: the right yoke's room filled solid read 0.489 (handling, under the PSU)
JOINT = [(-54.0, -200.0), (-14.0, -200.0), (26.0, -200.0), (52.0, -198.0), (-14.0, -152.0), (26.0, -152.0)]      # yoke to rear rail, along X (Y, Z)
SPLICE = [(128.0, -203.0), (170.0, -203.0), (128.0, -176.0), (170.0, -142.0)]                    # rear to front rail, along Y (X, Z)
PLATE = 8.0                         # each splice boss, along Y
POCKET = 6.0                        # the nut / head pocket behind it


def _side(m, s):
    return m if s > 0 else mirror_x(m)


def _foot(y):
    pad = cyl("z", FOOT_X, y, FOOT_R, BED, BED + 6.0, 64)               # 6 thick, 3 over a 3 deep stud hole
    return pad, cyl("z", FOOT_X, y, STUD_R, BED - 1.0, BED + 3.0)


def _foot_region(y, s):
    return _side(cyl("z", FOOT_X, y, FOOT_R - 1.0, BED, BED + 2.0, 48) - cyl("z", FOOT_X, y, STUD_R + 0.6, BED - 2, BED + 6), s)


def _joint_cuts():
    """the yoke-rail bolts: through the rear rail, the head's counterbore from its outer face, the insert in the yoke"""
    return U(*[cyl("x", y, z, 2.75, X_IN - 1.0, RAIL_X[1] + 1.0) for y, z in JOINT],
             *[cyl("x", y, z, 5.5, RAIL_X[1] - 6.0, RAIL_X[1] + 30.0) for y, z in JOINT],
             *[cyl("x", y, z, 3.2, X_IN - 9.0, X_IN + 0.5) for y, z in JOINT])


def _splice_cuts(rear=False):
    """the splice: the rear rail's side takes M5 heat-set inserts in its bosses (a nut pocket and window behind its outboard boss left
    it backed on two sides: room_full 1.48); the front rail's side has the bolt, the hex head in a pocket behind its plate, turned with
    an 8 mm spanner through a window to the nearer side face"""
    out = []
    y1 = Y_SPLICE + PLATE + POCKET
    for x, z in SPLICE:
        if rear:
            out.append(cyl("y", x, z, 3.2, Y_SPLICE - 9.0, Y_SPLICE + 1.0))
            continue
        out.append(cyl("y", x, z, 2.75, Y_SPLICE - 1.0, y1 + 1.0))
        xa, xb = (X_IN - 5.0, x + 6.5) if x < 150 else (x - 6.5, RAIL_X[1] + 5.0)
        za = z - 6.5 if z - 6.5 > BED + 6.0 else BED - 1.0         # a lower pocket opens through the floor face (no fin over the bed)
        out.append(box(xa, xb, Y_SPLICE + PLATE, y1, za, z + 6.5))
    return U(*out)


def _splice_boss(x, z, y0, y1):
    """a splice boss: round, the outboard ones squared out to the rail's outer face (round left a 1.8 fin there: room_full 1.48 in it)"""
    b = cyl("y", x, z, 8.0, y0, y1)
    return b + box(x, RAIL_X[1], y0, y1, z - 8.0, z + 8.0) if x > 150 else b


def _rail_room(y0, y1):
    return box(X_IN, RAIL_X[1], y0, y1, BED, RAIL_TOP)


# ------------------------------------------------------------------ yoke
def yoke_setup(h, s):
    pads = [cyl("z", x, y, 8.0, Z_FOOT - 10.0, Z_FOOT) for x, y in FOOT_BOLTS if x > 0]
    seat = box(2.0, 40.0, -62.0, -4.0, Z_FOOT - 4.0, Z_FOOT)                                    # the face the mast's foot bears on (pads alone: 0.55 filled solid at a pad)
    ends = [cyl("x", y, z, 7.0, X_IN - 12.0, X_IN) for y, z in JOINT]                           # the insert bosses, their end the contact face
    rim = box(X_IN - 8.0, X_IN, YOKE_Y[0] + 4.0, YOKE_Y[1] - 4.0, BED, BED + 5.0)               # the end face's foot: the bed (build +z)
    drawn = U(*pads, seat, *ends, rim)
    cuts = U(sweep(), _joint_cuts(), *[cyl("z", x, y, 2.75, Z_FOOT - 11.0, Z_FOOT + 1.0) for x, y in FOOT_BOLTS],
             *[cyl("z", x, y, 5.5, FLOOR - 2.0, Z_FOOT - 10.0) for x, y in FOOT_BOLTS])          # the M5 x 16 heads and the key from below
    if s > 0:
        cuts = cuts + box(*PSU)
    env = box(0.25, X_IN - 0.25, YOKE_Y[0], YOKE_Y[1], BED, Z_FOOT)
    p = Part("box_stand2_yoke_" + ("r" if s > 0 else "l"), h, _side(drawn, s), _side(cuts, s), _side(env, s), build="+z", material="PCTG")
    p.bed_min_mm2 = 500.0                                                # the drawn rim under the end face (topo/audit.py bed)
    p.support("default", _side(box(2.0, 40.0, -62.0, -4.0, Z_FOOT - 2.0, Z_FOOT) - U(*[cyl("z", x, y, 3.0, -300, 0) for x, y in FOOT_BOLTS if x > 0]), s))
    regs = [(np.array([s * (X_IN - 5.0), y, z]), _side(cyl("x", y, z, 6.5, X_IN - 10.0, X_IN) - cyl("x", y, z, 3.4, 0, 300), s)) for y, z in JOINT]
    for cn, cls, feet, acc in floor_cases():
        F, P = [], []
        for k, y in FEET.items():                                       # the rear rail passes both its feet's forces (the front through the splice)
            f = feet.get((s, k))
            if f is not None and np.linalg.norm(f) > 0.5:
                F.append(f); P.append(np.array([s * FOOT_X, y, BED]))
        if not F:
            continue
        Ft = sum(F); Mt = sum(np.cross(q, f) for q, f in zip(P, F))
        fs = _split_wrench([q for q, _ in regs], Ft, Mt)
        p.load_multi(cn, [(r, f) for (_, r), f in zip(regs, fs)], cls=cls, accel=acc)
    return p


def _split_wrench(pts, F, M):
    """least-norm forces at pts with resultant F and moment M about the origin"""
    pts = np.asarray(pts, float); n = len(pts); A = np.zeros((6, 3 * n))
    for i, q in enumerate(pts):
        A[:3, 3 * i:3 * i + 3] = np.eye(3); A[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
    return (np.linalg.pinv(A) @ np.r_[F, M]).reshape(n, 3)


# ------------------------------------------------------------------ rear rail
def rear_setup(h, s):
    pad, stud = _foot(FEET["rear"])
    inner = [cyl("x", y, z, 7.0, X_IN, X_IN + 6.0) for y, z in JOINT]                           # its contact face on the yoke
    outer = [cyl("x", y, z, 8.5, RAIL_X[1] - 10.0, RAIL_X[1]) for y, z in JOINT]                # the heads' seats (r 7 left 1.5 round the Ø11 counterbore: drawn_wall)
    plate = [_splice_boss(x, z, Y_SPLICE - PLATE, Y_SPLICE - 0.25) for x, z in SPLICE]
    drawn = U(pad, *inner, *outer, *plate)
    cuts = U(sweep(), _joint_cuts(), _splice_cuts(rear=True), stud)
    if s > 0:
        cuts = cuts + box(*PSU)
    env = _rail_room(-64.0, Y_SPLICE - 0.25)
    p = Part("box_stand2_rear_" + ("r" if s > 0 else "l"), h, _side(drawn, s), _side(cuts, s), _side(env, s), build="+z", material="PCTG")
    p.bed_min_mm2 = 800.0                                                # the drawn rear foot pad
    p.support("default", _side(U(*[cyl("x", y, z, 7.0, X_IN, X_IN + 2.5) - cyl("x", y, z, 3.0, 0, 300) for y, z in JOINT]), s))
    regs = [(np.array([s * x, Y_SPLICE - 1.0, z]), _side(cyl("y", x, z, 7.5, Y_SPLICE - 2.5, Y_SPLICE - 0.25) - cyl("y", x, z, 3.0, 0, 300), s)) for x, z in SPLICE]
    Pf = np.array([s * FOOT_X, FEET["front"], BED])
    for cn, cls, feet, acc in floor_cases():
        parts = []
        fr = feet.get((s, "rear"))
        if fr is not None and np.linalg.norm(fr) > 0.5:
            parts.append((_foot_region(FEET["rear"], s), fr))
        ff = feet.get((s, "front"))
        if ff is not None and np.linalg.norm(ff) > 0.5:                 # the front rail's foot force, through the splice plate
            fs = rigid_split([q for q, _ in regs], ff, Pf)
            parts += [(r, f) for (_, r), f in zip(regs, fs)]
        if parts:
            p.load_multi(cn, parts, cls=cls, accel=acc)
    return p


# ------------------------------------------------------------------ front rail
def rail_setup(h, s):
    pad, stud = _foot(FEET["front"])
    plate = [_splice_boss(x, z, Y_SPLICE + 0.25, Y_SPLICE + PLATE) for x, z in SPLICE]
    drawn = U(pad, *plate)
    cuts = U(sweep(), _splice_cuts(), stud)
    env = _rail_room(Y_SPLICE + 0.25, 250.0)
    p = Part("box_stand2_rail_" + ("r" if s > 0 else "l"), h, _side(drawn, s), _side(cuts, s), _side(env, s), build="+z", material="PCTG")
    p.bed_min_mm2 = 800.0                                                # the drawn front foot pad
    p.support("default", _side(U(*[cyl("y", x, z, 7.5, Y_SPLICE + 0.25, Y_SPLICE + 2.5) - cyl("y", x, z, 3.0, 0, 300) for x, z in SPLICE]), s))
    for cn, cls, feet, acc in floor_cases():
        f = feet.get((s, "front"))
        if f is None or np.linalg.norm(f) < 0.5:
            continue
        p.load(cn, _foot_region(FEET["front"], s), f, cls=cls, accel=acc)
    return p
