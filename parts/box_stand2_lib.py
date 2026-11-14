"""Box stand v2 (14 Nov 2026): the stand the swap arm v8 bolts to, as five engine parts (parts/box_stand2_<piece>), all in W
(X across, Y along the axle toward the front, Z up; the arm's hub at the origin, the floor at W Z -216, Spark 13 Nov: hub 15 higher).

Pieces (each in the 180 bed):
  mast        the face the v8 stub's Ø90 flange bolts to (W Y 0), down behind the hub to its foot at W Z -135. Room X -42..40,
              Y -64..0 (the rear sheet at -66, the PSU at X >= 42 behind Y -36, the axle stepper X -71..-29, Y -48..0, Z 19..61 cut out).
              Printed face down (build -y): the face is the first layer; the column's bending runs in the layers.
  base_r/_l   one half each side of X 0, under the mast's foot (8 M5 up into heat-set inserts in the foot, 4 a half) out to its rail
              at |X| 120..177.5 (the ring's sweep leaves 42 at |X| 120, 61+ outboard), the rail's rear part with the rear foot at
              (+-149, -44), and the lap's outer cheek (|X| 149..177.5, Y 72..110). Printed on its floor face (build +z).
  rail_r/_l   the rail forward of the lap (Y 72..250) with the front foot at (+-149, 232); its tongue (|X| 120..149, Y 72..110)
              laps the base's cheek with 4 M5 x 60 across (X), nuts outboard. Printed on its floor face.
  feet        four TPU 95A pads Ø36 x 5 (a Ø8 x 2.8 stud pressed into each foot pad): the stand stands on four points, so every floor
              reaction is known (flat floor: four; an uneven floor: three, each foot lifted in turn).
Joints (all M5, steel): stub flange to the mast face, 3 x M5 x 16 from the mast side (the heads in the face's counterbores, the tool from
behind through the mast), threading into M5 heat-set inserts in the stub's flange; mast foot to the halves, 8 x M5 x 16 up from under
the halves' top pads into M5 inserts in the foot; lap, 4 x M5 x 60 + nyloc a side.

Loads are the system's, per case (parts/box_stand_mast/check.py: the arm v8 with 11 bays and packs 9717 g at W (-2.1, 129, -21.4), the
ring at its worst unbalance, the insert at the drone, carrying, the float limit, a dropped Overseer, the axle stall, 3 g + 2 g handling):
the mast takes the arm's wrench at the stub face (the flange bearing on the face ring, the bolts pulling at their head seats, the
in-plane part on the bolt linings); the halves and the rails take the floor at their feet (the whole stand's free body: arm + stand),
their supports being the joint toward the mast. Own weight as body forces.
"""
import numpy as np
import manifold3d as mf
from scipy.optimize import lsq_linear
from topo.part import Part

FLOOR = -216.0
BED = -211.0                   # the PCTG pieces' floor face: the TPU feet are 5 under it
G = 9.81e-3                    # N per g

# ------------------------------------------------------------------ geometry helpers
def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl(ax, a, b, r, t0, t1, n=40):
    """a cylinder of radius r along axis ax ('x', 'y', 'z') from t0 to t1, centred at (a, b) in the other two (in x, y, z order)"""
    c = mf.Manifold.cylinder(t1 - t0, r, r, n)
    if ax == "z":
        return c.translate([a, b, t0])
    if ax == "y":
        return c.rotate([-90, 0, 0]).translate([a, t0, b])          # rotate z -> +y: (x, y, z) -> (x, -z, y)... see the check below
    return c.rotate([0, 90, 0]).translate([t0, a, b])               # z -> +x


def U(*ms):
    ms = [m for m in ms if m is not None]
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add) if len(ms) > 1 else ms[0]


def _check_cyl():
    for ax, k in (("x", 0), ("y", 1), ("z", 2)):
        bb = np.array(cyl(ax, 1.0, 2.0, 0.5, 10.0, 20.0).bounding_box())
        assert abs(bb[k] - 10.0) < 1e-6 and abs(bb[3 + k] - 20.0) < 1e-6, (ax, bb)
_check_cyl()


def mirror_x(m):
    return m.mirror([1.0, 0.0, 0.0])


# ------------------------------------------------------------------ what is round the stand (W, fixed: copied here so a change elsewhere reruns nothing)
# the arm's sweep: the largest radius off the axle of every turning piece (hub, heads, bays, packs) per Y band, from systems/box.assembly.json
# on 14 Nov (sampled surfaces): Y 8..20 50.2, 20..30 78.1, 30..50 154.3, 50..70 158.1, 70..266 204.2. Held 3.3 off, bands overlapped 2.
SWEEP = [(5.0, 21.0, 53.5), (19.0, 31.0, 81.5), (29.0, 51.0, 157.6), (49.0, 71.0, 161.5), (69.0, 270.0, 207.5)]
MOTOR = (-73.0, -27.4, -64.5, 1.0, 17.4, 63.0)      # the axle stepper (X -71.2..-28.9, Y -48..0, Z 18.9..61.2) grown 1.5, cut through Y (no 14 mm sliver behind it)
PSU = (40.0, 159.0, -68.0, -34.0, -172.0, 47.0)     # the PSU box (X 42..157, Y -66..-36, Z -170..45) grown 2
BOLT_R = 38.0
STUB_BOLTS = [(BOLT_R * np.cos(a), BOLT_R * np.sin(a)) for a in np.radians([45.0, 225.0, 315.0])]   # (X, Z); the 135 deg hole (-26.9, 26.9) is 0.5 into the stepper behind the flange
FOOT_BOLTS = [(sx * x, y) for sx in (1, -1) for x in (10.0, 34.0) for y in (-54.0, -12.0)]          # mast foot to the halves (X, Y), Z -135
LAP_BOLTS = [(y, z) for y in (80.0, 102.0) for z in (-200.0, -176.0)]                              # rail tongue to the base cheek, along X (Y, Z)
RAIL_X = (120.0, 177.5); LAP_X = 149.0; FOOT_X = 149.0      # 175 / 148 and a top at -140: the rail's room filled solid read 0.476 (handling, uneven floor)
RAIL_TOP = -128.0
FEET = {"rear": -44.0, "front": 232.0}             # foot Y; at X +-FOOT_X
FOOT_R, STUD_R = 18.0, 4.0
Z_FOOT = -135.0                                     # the mast foot / base top plane


def sweep():
    return U(*[cyl("y", 0, 0, r, y0, y1, 160) for y0, y1, r in SWEEP])


# ------------------------------------------------------------------ loads: the arm's cases (parts/box_stand_mast/check.py, the same numbers)
ARM_G, ARM_CG = 9717.0, np.array([-2.1, 129.0, -21.4])
TURN_G, TURN_R = 8688.0, 25.9
STAND_G, STAND_CG = 700.0, np.array([0.0, 70.0, -185.0])          # E: the five pieces at their budgets x 1.27 g/cm3 + hardware
DROP = np.array([0.0, 192.5, 176.0]); CARRY = np.array([0.0, 167.1, 176.0])


def arm_cases():
    """(name, cls, arm loads [(F, at)], extra moment on the arm about Y (N mm), accel of the stand in g)"""
    D = (np.array([0, 0, -ARM_G * G]), ARM_CG + np.array([TURN_R * TURN_G / ARM_G, 0, 0]))
    g1 = np.array([0, 0, -1.0])
    return [("standing: arm v8, 11 bays with packs, the ring at its worst unbalance", "sustained", [D], 0.0, g1),
            ("insert at the drone, 106.6 N every swap", "flight", [D, (np.array([0, 0, -106.6]), DROP)], 0.0, g1),
            ("carrying an Overseer at full reach", "flight", [D, (np.array([0, 0, -10.0]), CARRY)], 0.0, g1),
            ("insert at the float limit, 118 N", "misuse", [D, (np.array([0, 0, -118.0]), DROP)], 0.0, g1),
            ("dropped Overseer onto the head at the drone, 150 N", "misuse", [D, (np.array([0, 0, -150.0]), CARRY)], 0.0, g1),
            ("axle stall 5.53 N m about Y, head held at the drone", "misuse", [D], 5530.0, g1),
            ("box handling 3 g down + 2 g side (+X)", "misuse", [(3 * D[0], D[1]), (np.array([2 * ARM_G * G, 0, 0]), D[1])], 0.0, np.array([2.0, 0, -3.0])),
            ("box handling 3 g down + 2 g side (-X)", "misuse", [(3 * D[0], D[1]), (np.array([-2 * ARM_G * G, 0, 0]), D[1])], 0.0, np.array([-2.0, 0, -3.0]))]


UNEVEN = ("standing", "insert at the drone", "box handling")       # the cases also run on an uneven floor (three feet)


def wrench(loads, My, at=np.zeros(3)):
    F = sum(f for f, _ in loads); M = sum(np.cross(p - at, f) for f, p in loads) + np.array([0, My, 0])
    return F, M


def feet_xy():
    return {(s, k): np.array([s * FOOT_X, y]) for s in (1, -1) for k, y in FEET.items()}


def floor_cases():
    """(name, cls, {(side, 'rear'|'front'): force on the stand at that foot (3,)}, accel) for every arm case and floor"""
    out = []
    xy = feet_xy(); keys = list(xy)
    for nm, cls, loads, My, acc in arm_cases():
        st = (acc * STAND_G * G, STAND_CG)
        F, M = wrench(loads + [st], My, at=np.array([0, 0, FLOOR]))
        V = -F[2]; h = -np.array([F[0], F[1]]) / V                      # friction share per newton of floor reaction
        floors = [("flat floor", keys)] + ([(f"uneven floor, {'right' if k[0] > 0 else 'left'} {k[1]} foot off", [q for q in keys if q != k]) for k in keys]
                                           if nm.startswith(UNEVEN) else [])
        for fn, on in floors:
            A = [[1.0] * len(on), [xy[q][1] for q in on], [xy[q][0] for q in on]]; b = [V, -M[0], M[1]]
            if len(on) == 4:                                            # four equal springs: the planar warp mode off
                A.append([np.sign(xy[q][0]) * np.sign(xy[q][1] - 94.0) for q in on]); b.append(0.0)
            R = np.linalg.solve(np.array(A), np.array(b))
            if (R < -1e-6).any():
                continue                                                # the stand rocks onto the fourth foot: not a state
            out.append((f"{nm}, {fn}", cls, {q: np.array([h[0] * r, h[1] * r, r]) for q, r in zip(on, R)}, acc))
    return out


def rigid_split(pts, F, P):
    pts = np.asarray(pts, float); c = pts.mean(0); n = len(pts); A = np.zeros((6, 3 * n))
    for i, q in enumerate(pts - c):
        A[:3, 3 * i:3 * i + 3] = np.eye(3); A[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
    return (np.linalg.pinv(A) @ np.r_[F, np.cross(np.asarray(P, float) - c, F)]).reshape(n, 3)


# ------------------------------------------------------------------ mast
FACE_T, BOSS_T = 4.0, 10.0          # the boss ends at the washer seat (r 12 x 14 made one drawn body with a 7 mm core: audit drawn_blocks)
Z_TOP = 44.0                        # the mast's top (the flange's at 45): 179 tall, in the 180 bed
SEAT_Y = -10.0                      # the stub bolts' seats, under an M5 x 15 washer: the face 10 thick at a bolt (6 read 0.72 filled solid at the one top bolt, 10 with the head alone 0.55)
RING = [(np.cos(a) * 39.0, np.sin(a) * 39.0) for a in np.radians(np.arange(0, 360, 30))]


def _motor_cut():
    return box(*MOTOR)


def mast_parts():
    face = (cyl("y", 0, 0, 45.0, -FACE_T, 0.0, 96) - cyl("y", 0, 0, 33.0, -FACE_T - 1, 1.0, 96)) ^ box(-50, 50, -20, 1, -50, Z_TOP)
    bosses = [cyl("y", x, z, 9.5, -BOSS_T, 0.0) for x, z in STUB_BOLTS]
    feet = [cyl("z", x, y, 7.0, Z_FOOT, Z_FOOT + 12.0) for x, y in FOOT_BOLTS]
    drawn = U(face, *bosses, *feet)
    cuts = U(_motor_cut(), box(*PSU), cyl("y", 0, 0, 13.0, -70.0, 1.0, 64),                        # the stub's bore: the post's screw from behind
             *[cyl("y", x, z, 2.75, SEAT_Y - 1.0, 1.0) for x, z in STUB_BOLTS],
             *[cyl("y", x, z, 7.75, -70.0, SEAT_Y) for x, z in STUB_BOLTS],                        # the Ø15 washers, heads and the hex key, back through the mast
             *[cyl("z", x, y, 3.2, Z_FOOT - 1.0, Z_FOOT + 8.0) for x, y in FOOT_BOLTS])            # M5 x D7 x L6 heat-set inserts
    env = U(box(-42.0, 40.0, -64.0, 0.0, Z_FOOT, Z_TOP), cyl("y", 0, 0, 45.0, -14.0, 0.0, 96) ^ box(-50, 50, -20, 1, -50, Z_TOP))
    return drawn, cuts, env


def mast_setup(h):
    drawn, cuts, env = mast_parts()
    p = Part("box_stand2_mast", h, drawn, cuts, env, build="-y", material="PCTG")       # the face on the bed (Y 0), the column up
    p.bed_min_mm2 = 1500.0                                                               # the drawn face ring and bosses at Y 0 (topo/audit.py bed)
    p.support("default", U(*[cyl("z", x, y, 7.0, Z_FOOT, Z_FOOT + 3.0) for x, y in FOOT_BOLTS]))
    ring = [(np.array([x, -1.0, z]), cyl("y", x, z, 4.5, -2.0, 0.0)) for x, z in RING
            if not (x < -24.0 and z > 14.0)]                                # the flange bears on the face ring
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


# ------------------------------------------------------------------ base halves and rails (side s = +1 right, -1 left; drawn at +X and mirrored)
def _env_base():
    a = box(0.25, RAIL_X[1], -64.0, 66.0, BED, Z_FOOT)
    b = box(118.0, RAIL_X[1], 60.0, 72.0, BED, RAIL_TOP)
    c = box(LAP_X + 0.25, RAIL_X[1], 72.0, 109.75, BED, RAIL_TOP)
    return U(a, b, c)


def _env_rail():
    return U(box(RAIL_X[0], LAP_X - 0.25, 72.25, 110.5, BED, RAIL_TOP), box(RAIL_X[0], RAIL_X[1], 110.25, 250.0, BED, RAIL_TOP))


def _lap_cuts():
    return U(*[cyl("x", y, z, 2.75, RAIL_X[0] - 20.0, RAIL_X[1] + 20.0) for y, z in LAP_BOLTS],
             *[cyl("x", y, z, 5.5, RAIL_X[0] - 30.0, RAIL_X[0]) for y, z in LAP_BOLTS],          # heads inboard
             *[cyl("x", y, z, 6.5, RAIL_X[1], RAIL_X[1] + 30.0) for y, z in LAP_BOLTS])          # nuts outboard


def _foot(y):
    pad = cyl("z", FOOT_X, y, FOOT_R, BED, BED + 6.0, 64)              # 6 thick, 3 over a 3 deep stud hole (4.5 deep left 1.5: drawn_wall; 9 thick was a block: drawn_blocks)
    return pad, cyl("z", FOOT_X, y, STUD_R, BED - 1.0, BED + 3.0)


def base_parts():
    pads = [cyl("z", x, y, 8.0, Z_FOOT - 10.0, Z_FOOT) for x, y in FOOT_BOLTS if x > 0]
    pad_f, stud = _foot(FEET["rear"])
    cheek = [cyl("x", y, z, 7.0, LAP_X, LAP_X + 6.0) for y, z in LAP_BOLTS] + [cyl("x", y, z, 7.0, RAIL_X[1] - 6.0, RAIL_X[1]) for y, z in LAP_BOLTS]
    drawn = U(*pads, pad_f, *cheek)
    cuts = U(sweep(), _lap_cuts(), stud,
             *[cyl("z", x, y, 2.75, Z_FOOT - 11.0, Z_FOOT + 1.0) for x, y in FOOT_BOLTS],
             *[cyl("z", x, y, 5.5, FLOOR - 2.0, Z_FOOT - 10.0) for x, y in FOOT_BOLTS])          # the M5 x 16 heads and the key from below
    return drawn, cuts, _env_base(), pads, pad_f, cheek


def rail_parts():
    pad_f, stud = _foot(FEET["front"])
    tongue = [cyl("x", y, z, 7.0, RAIL_X[0], RAIL_X[0] + 6.0) for y, z in LAP_BOLTS] + [cyl("x", y, z, 7.0, LAP_X - 6.0, LAP_X) for y, z in LAP_BOLTS]
    drawn = U(pad_f, *tongue)
    cuts = U(sweep(), _lap_cuts(), stud)
    return drawn, cuts, _env_rail(), pad_f, tongue


def _side(m, s):
    return m if s > 0 else mirror_x(m)


def _vec(v, s):
    return np.array([s * v[0], v[1], v[2]])


def _foot_region(y, s):
    return _side(cyl("z", FOOT_X, y, FOOT_R - 1.0, BED, BED + 2.0, 48) - cyl("z", FOOT_X, y, STUD_R + 0.6, BED - 2, BED + 6), s)


def _lap_regions(x0, x1, s):
    return [(np.array([s * (x0 + x1) / 2, y, z]), _side(cyl("x", y, z, 5.0, x0, x1) - cyl("x", y, z, 3.0, x0 - 1, x1 + 1), s)) for y, z in LAP_BOLTS]


def rail_setup(h, s):
    drawn, cuts, env, pad_f, tongue = rail_parts()
    nm = "box_stand2_rail_" + ("r" if s > 0 else "l")
    p = Part(nm, h, _side(drawn, s), _side(cuts, s), _side(env, s), build="+z", material="PCTG")
    p.bed_min_mm2 = 800.0                                                # the drawn front foot pad on the bed (topo/audit.py bed)
    lap = _lap_regions(RAIL_X[0], LAP_X, s)                              # the tongue's bolt linings, its whole width (the bolts bear along it)
    p.support("default", U(*[r for _, r in lap]))
    P = np.array([s * FOOT_X, FEET["front"], BED])
    for cn, cls, feet, acc in floor_cases():
        f = feet.get((s, "front"))
        if f is None or np.linalg.norm(f) < 0.5:
            continue
        p.load(cn, _foot_region(FEET["front"], s), f, cls=cls, accel=acc)
    return p


def base_setup(h, s):
    drawn, cuts, env, pads, pad_f, cheek = base_parts()
    nm = "box_stand2_base_" + ("r" if s > 0 else "l")
    if s > 0:
        cuts = cuts + box(*PSU)                  # the PSU stands over the right half only
    p = Part(nm, h, _side(drawn, s), _side(cuts, s), _side(env, s), build="+z", material="PCTG")
    p.bed_min_mm2 = 800.0                                                # the drawn rear foot pad on the bed
    p.support("default", _side(U(*[cyl("z", x, y, 8.0, Z_FOOT - 3.0, Z_FOOT) - cyl("z", x, y, 3.0, -300, 0) for x, y in FOOT_BOLTS if x > 0]), s))
    lap = _lap_regions(LAP_X, RAIL_X[1], s)
    Pf = np.array([s * FOOT_X, FEET["front"], BED])
    for cn, cls, feet, acc in floor_cases():
        parts = []
        fr = feet.get((s, "rear"))
        if fr is not None and np.linalg.norm(fr) > 0.5:
            parts.append((_foot_region(FEET["rear"], s), fr))
        ff = feet.get((s, "front"))
        if ff is not None and np.linalg.norm(ff) > 0.5:                 # the rail passes its front foot's force to the cheek's four bolts
            fs = rigid_split([q for q, _ in lap], ff, Pf)
            parts += [(reg, f) for (_, reg), f in zip(lap, fs)]
        if parts:
            p.load_multi(cn, parts, cls=cls, accel=acc)
    return p


# ------------------------------------------------------------------ the TPU foot (no run: a pad)
def foot_mesh():
    return U(cyl("z", 0, 0, FOOT_R, 0.0, 5.0, 64), cyl("z", 0, 0, STUD_R - 0.1, 5.0, 7.8, 32))
