"""Box stand v2, mast (14 Nov 2026): the arm v8 stub's face (W Y 0) down behind the hub to the foot on the base halves. Geometry, loads and
helpers from parts/box_stand2_lib.py; the mast's own setup lives here since its second setup (run 1, 14 Nov: early fail 2.39 at the face
ring's inner edge in every case, the flange's bearing on Ø9 points of a 4 mm ring with a Ø26 bore behind it): the bearing on 30 deg
sectors of the ring, the ring 6 thick, no bore. Setup 3 (run 2: 259 g, gate PASS, 1.43 across the layers at the top bolt in every
case, face down): printed on its +X side, so the bolts' pull and the column's bending both run in the layers."""
import numpy as np
import manifold3d as mf
from scipy.optimize import lsq_linear
from topo.part import Part
from parts.box_stand2_lib import box, cyl, U, MOTOR, PSU, STUB_BOLTS, FOOT_BOLTS, Z_FOOT, arm_cases, wrench, rigid_split

FACE_T, BOSS_T = 6.0, 10.0          # the boss ends at the washer seat (r 12 x 14 made one drawn body with a 7 mm core: audit drawn_blocks)
X_BED = 40.0                        # the mast's +X side, on the bed (setup 2, face down, read 1.43 ACROSS the layers behind the one top bolt: its pull ran along Y, the build axis)
Z_TOP = 44.0                        # the mast's top (the flange's at 45): 179 tall, in the 180 bed
SEAT_Y = -10.0                      # the stub bolts' seats, under an M5 x 15 washer: the face 10 thick at a bolt (6 read 0.72 filled solid at the one top bolt, 10 with the head alone 0.55)
RING = [(np.cos(a) * 39.0, np.sin(a) * 39.0) for a in np.radians(np.arange(0, 360, 30))]


def _sector(deg, half=15.0):
    """the face ring's 30 deg sector round deg (in the XZ plane), its first 2 mm: a bearing region"""
    a = np.radians(np.linspace(deg - half, deg + half, 7))
    pts = [[0.0, y, 0.0] for y in (-2.0, 0.0)] + [[50 * np.cos(t), y, 50 * np.sin(t)] for t in a for y in (-2.0, 0.0)]
    wedge = mf.Manifold.hull_points(pts)
    return wedge ^ (cyl("y", 0, 0, 44.5, -2.0, 0.0, 96) - cyl("y", 0, 0, 33.5, -3.0, 1.0, 96))


def _motor_cut():
    return box(*MOTOR)


def mast_parts():
    face = (cyl("y", 0, 0, 45.0, -FACE_T, 0.0, 96) - cyl("y", 0, 0, 33.0, -FACE_T - 1, 1.0, 96)) ^ box(-50, X_BED, -20, 1, -50, Z_TOP)
    bosses = [cyl("y", x, z, 9.5, -BOSS_T, 0.0) for x, z in STUB_BOLTS]
    feet = [cyl("z", x, y, 6.0, Z_FOOT, Z_FOOT + 12.0) for x, y in FOOT_BOLTS]                # r 6: the outer ones end on the bed plane X 40
    rims = [box(X_BED - 3.0, X_BED, y0, y1, Z_FOOT, Z_TOP) for y0, y1 in ((-64.0, -60.0), (-4.0, 0.0))]   # the column's edges on the bed (build -x)
    drawn = U(face, *bosses, *feet, *rims)
    cuts = U(_motor_cut(), box(*PSU),                                                                # (no bore behind the stub's: nothing passes it; v1's Ø26 took the face's back-up, 14 Nov run 1)
             *[cyl("y", x, z, 2.75, SEAT_Y - 1.0, 1.0) for x, z in STUB_BOLTS],
             *[cyl("y", x, z, 7.75, -70.0, SEAT_Y) for x, z in STUB_BOLTS],                        # the Ø15 washers, heads and the hex key, back through the mast
             *[cyl("z", x, y, 3.2, Z_FOOT - 1.0, Z_FOOT + 8.0) for x, y in FOOT_BOLTS])            # M5 x D7 x L6 heat-set inserts
    env = U(box(-42.0, X_BED, -64.0, 0.0, Z_FOOT, Z_TOP), cyl("y", 0, 0, 45.0, -14.0, 0.0, 96) ^ box(-50, X_BED, -20, 1, -50, Z_TOP))
    return drawn, cuts, env


def mast_setup(h):
    drawn, cuts, env = mast_parts()
    p = Part("box_stand2_mast", h, drawn, cuts, env, build="-x", material="PCTG")       # on its +X side (X 40): the stub bolts' pull (Y) and the column's bending (Z) in the layers
    p.bed_min_mm2 = 1000.0                                                               # the two drawn rims and the face ring's edge at X 40 (topo/audit.py bed)
    p.support("default", U(*[cyl("z", x, y, 6.0, Z_FOOT, Z_FOOT + 3.0) for x, y in FOOT_BOLTS]))
    ring = [(np.array([x, -1.0, z]), _sector(np.degrees(np.arctan2(z, x)))) for x, z in RING
            if not (x < -24.0 and z > 14.0)]                                # the flange bears on the face ring: 30 deg sectors of it (run 1 put up to 280 N on Ø9 points: 2.39 early)
    heads = [(np.array([x, SEAT_Y, z]), cyl("y", x, z, 7.4, SEAT_Y, SEAT_Y + 1.5) - cyl("y", x, z, 2.9, -12, 2)) for x, z in STUB_BOLTS]
    lin = [(np.array([x, -3.0, z]), cyl("y", x, z, 4.5, SEAT_Y, 0.0) - cyl("y", x, z, 2.9, -12, 2)) for x, z in STUB_BOLTS]
    for nm, cls, loads, My, acc in arm_cases():
        F, M = wrench(loads, My)
        # the normal part (Fy, Mx, Mz): bearing on the ring (force -Y on the face) and bolt pull at the head seats (+Y), least squares >= 0
        pts = [q for q, _ in ring] + [q for q, _ in heads]; sg = [-1.0] * len(ring) + [1.0] * len(heads)
        A = np.array([[s for s in sg], [-s * q[2] for s, q in zip(sg, pts)], [s * q[0] for s, q in zip(sg, pts)]])
        b = np.array([F[1], M[0], M[2]])
        n = len(pts); Aa = np.vstack([A, 1e-3 * np.eye(n)]); ba = np.r_[b, np.zeros(n)]
        x = lsq_linear(Aa, ba, bounds=(0, np.inf), method="bvls").x
        assert np.allclose(A @ x, b, atol=0.02 * max(1.0, np.abs(b).max())), (nm, A @ x, b)
        parts = [(reg, np.array([0, s * xi, 0])) for (q, reg), s, xi in zip(ring + heads, sg, x) if xi > 1e-6]
        # the in-plane part (Fx, Fz, My) on the three bolt linings, least norm: sum fx = Fx, sum fz = Fz, sum (z fx - x fz) = My
        B = np.zeros((3, 6))
        for i, (q, _) in enumerate(lin):
            B[0, 2 * i] = 1.0; B[1, 2 * i + 1] = 1.0; B[2, 2 * i] = q[2]; B[2, 2 * i + 1] = -q[0]
        y = np.linalg.lstsq(B, np.array([F[0], F[2], M[1]]), rcond=None)[0]
        fi = [np.array([y[2 * i], 0.0, y[2 * i + 1]]) for i in range(len(lin))]
        parts += [(reg, f) for (q, reg), f in zip(lin, fi)]
        p.load_multi(nm, parts, cls=cls, accel=acc)
    return p



setup = mast_setup
SPEC = dict(h=2.0, gen_cm3=180.0, max_worst=0.5)   # 110 (14 % of the room) thinned to 3.0 x max_worst against 0.42 filled solid: audit budget
