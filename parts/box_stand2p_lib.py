"""Box stand v2, PROTOTYPE (15 Nov 2026, CLAUDE.md stages: "we don't need generated parts for bench testing or even the first test"):
the stand of the engine stream (parts/box_stand2_lib.py, parts/box_stand2_frame_lib.py: those setups stay in the repo for OPTIMISE,
after Spark's test) as drawn parts, simple generous sections sized by hand below. W frame (X across, Y along the axle toward the front,
Z up; the arm's hub at the origin, the floor at W Z -216: Spark 13 Nov, hub 15 higher).

Pieces (each in the A1 mini's 180 bed, flat on it, PCTG):
  box_stand2p_mast      a 82 x 64 tube (5 walls, a mid web) from the foot plate (W Z -135..-125: the front row M5 inserts, the rear row
                        through to nuts inside, set through windows in the side walls) up to the
                        stub's face (W Y 0): a 12 thick face plate (the flange's Ø90 seat, three M5 x 16 on Ø15 washers from behind into
                        inserts in the stub's flange, through Ø18 access holes in the back wall and the web), a 5 thick top cap. Clear of
                        the axle stepper (X -71..-29, Y -48..0, Z 19..61) and the PSU (X 42..157, Y -66..-36). Printed on its +X side.
  box_stand2p_base_r/_l the yoke (a closed box X 0..120, Y -64..28 under the mast's deck, Y -30..28 out to the rail, Z -211..-135, 5 walls,
                        a mid web, the deck plate 6 thick with 4 M5 x 16 up into the mast's foot, Ø14 key holes in the floor plate) in one
                        piece with the rear rail (an open-bottom channel |X| 120..177.5, Y -64..110: inner, mid and outer walls 5 thick,
                        its top stepped down under the arm's sweep and the PSU, bulkheads where the yoke meets it), the rear foot pad at
                        (+-149, -44), the splice plate at Y 102..110 with 5 M5 heat-set inserts. Printed on its floor face.
  box_stand2p_rail_r/_l the front rail: the same channel from its splice plate (Y 110.25..118.25, five M5 x 25 through both plates, Ø15
                        washers, nylocs, all reached from under the open channels; a heat-set insert there read 0.88) to the front foot pad
                        at (+-149, 232) and an end cap. Printed on its floor face.
  box_stand2p_foot      four TPU 95A pads Ø36 x 5, a Ø8 x 2.8 stud pressed into each foot pad's Ø8 x 3 hole.
Holes (no systems/fit_table.json yet: the repo's values until Spark's coupons): M5 clearance Ø5.5, M5 heat-set insert (D7 x L6) Ø6.4 x 8,
M5 head counterbore Ø10 / Ø16 under a Ø15 washer, TPU stud hole Ø8 (stud Ø7.8), a 0.25 gap between mating faces.
Assembly path: (1) feet into the four foot pads; (2) each front rail to its base half: five M5 x 25 with Ø15 washers and nylocs through both
splice plates, from under the open channels; (3) the two base halves side by side at X 0 on the floor; (4) the mast onto both decks: up from
under each deck through the yoke floor's key holes, the front row M5 x 16 into the mast foot's inserts, the rear row M5 x 25 on Ø15 washers
through the foot to nylocs set through the windows low in the mast's side walls (a heat-set insert there read 0.59); (5) the arm's stub onto the mast's face, three M5 x
16 on Ø15 washers from behind through the mast's back-wall holes into inserts in the stub's flange (the arm stream: its flange holes are Ø5.5
now, its 135 deg hole sits 0.5 into the axle stepper and is not used); (6) the hub on the stub (the arm's own order).
Loads: every case of parts/box_stand_mast/check.json (the arm v8 with 11 bays and packs, 9.7 kg at W Y 129; the insert at the drone 106.6 /
118 N; a dropped Overseer 150 N; the axle stall 5.53 N m; 3 g + 2 g handling) with the stand's own weight, the floor on the four feet (flat,
and each foot lifted in turn), parts/box_stand2_lib.py arm_cases / floor_cases. Hand check: hand_check() below, on the sections sliced from
these meshes; its table is in each make.py's docstring.
"""
import os, sys, json, subprocess
import numpy as np, trimesh, manifold3d as mf
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from parts.box_stand2_lib import (box, cyl, U, mirror_x, sweep, PSU, MOTOR, STUB_BOLTS, FOOT_BOLTS, FEET, FOOT_X, FOOT_R, STUD_R,
                                  FLOOR, BED, Z_FOOT, arm_cases, floor_cases, wrench)

PCTG = 1.27                                     # g / cm3
XT, ZT, S_, KD = 46.0, 23.0, 23.0, 0.85        # topo/material.py PCTG (E): in-layer, across, shear; knockdown
SF = {"flight": 4.0, "misuse": 2.5, "sustained": 10.0}
T = 5.0                                         # walls
X_IN, X_MID, X_OUT = 120.0, 149.0, 177.5        # the rail's inner face, its mid wall's centre, its outer face
Y_SPL = 110.0
SPLICE = [(130.0, -204.0), (168.0, -204.0), (130.0, -184.0), (168.0, -160.0), (149.0, -172.0)]   # (X, Z)
TOP_IN, TOP_OUT = -172.0, -148.0               # the channel's top: inboard of the mid wall, outboard (the sweep: -169 at X 120, -143 at X 150, Y >= 69)
YOKE_Y0, YOKE_Y1, DECK_Y1 = -30.0, 28.0, 0.0
Z_TOPY = Z_FOOT                                 # the yoke's top (the deck under the mast's foot)
INS_R, CLR_R = 3.2, 2.75


# ------------------------------------------------------------------ helpers
def hollow_box(x0, x1, y0, y1, z0, z1, t=T, open_=()):
    """a box with walls t; open_ names the faces left open ('z0' the bottom, ...)"""
    outer = box(x0, x1, y0, y1, z0, z1)
    inner = box(x0 + t if "x0" not in open_ else x0 - 1, x1 - t if "x1" not in open_ else x1 + 1,
                y0 + t if "y0" not in open_ else y0 - 1, y1 - t if "y1" not in open_ else y1 + 1,
                z0 + t if "z0" not in open_ else z0 - 1, z1 - t if "z1" not in open_ else z1 + 1)
    return outer - inner


def to_tri(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def keepouts(side):
    k = U(sweep(), box(*MOTOR))
    return k + box(*PSU) if side > 0 else k + mirror_x(box(*PSU))      # both halves clear of the PSU: one drawing, mirrored


# ------------------------------------------------------------------ mast
def mast():
    xa, xb, ya, yb = -42.0, 40.0, -64.0, 0.0
    tube = hollow_box(xa, xb, ya, yb, Z_FOOT + 10.0, 44.0, open_=("z0",))
    web = box(xa, xb, -34.0, -30.0, Z_FOOT + 10.0, 44.0)
    face = cyl("y", 0, 0, 45.0, -12.0, 0.0, 96) ^ box(-50, xb, -13, 1, -50, 44.0)
    foot = box(xa, xb, ya, yb, Z_FOOT, Z_FOOT + 10.0)
    bosses = [cyl("y", x, z, 10.0, -16.0, 0.0) for x, z in STUB_BOLTS]           # the seats' backs, into the walls
    body = U(tube, web, face, foot, *bosses)
    cuts = U(box(*MOTOR), box(*PSU),
             *[cyl("y", x, z, CLR_R, -20.0, 1.0) for x, z in STUB_BOLTS],
             *[cyl("y", x, z, 8.0, -70.0, -10.0) for x, z in STUB_BOLTS],        # Ø16 washer seat at Y -10 and the key's way back out
             *[cyl("y", x, z, 9.0, -70.0, -16.0) for x, z in STUB_BOLTS],        # Ø18 holes in the web and the back wall
             *[cyl("z", x, y, INS_R, Z_FOOT - 1.0, Z_FOOT + 8.0) for x, y in FOOT_BOLTS if y > -30],      # the front row: inserts
             *[cyl("z", x, y, CLR_R, Z_FOOT - 1.0, Z_FOOT + 11.0) for x, y in FOOT_BOLTS if y < -30],     # the rear row: through, a nut inside
             *[box(x0, x1, -56.0, -38.0, Z_FOOT + 13.0, Z_FOOT + 33.0) for x0, x1 in ((-43.0, -36.0), (34.0, 41.0))])   # the nuts' windows in the side walls
    return body - cuts


# ------------------------------------------------------------------ base half (drawn at +X; the left is its mirror)
def channel(y0, y1, end_caps=()):
    """the rail's open-bottom channel from y0 to y1: inner / mid / outer walls and the stepped top"""
    w_in = box(X_IN, X_IN + T, y0, y1, BED, TOP_IN)
    w_mid = box(X_MID - T / 2, X_MID + T / 2, y0, y1, BED, TOP_OUT)
    w_out = box(X_OUT - T, X_OUT, y0, y1, BED, TOP_OUT)
    top_i = box(X_IN, X_MID, y0, y1, TOP_IN - T, TOP_IN)
    top_o = box(X_MID - T / 2, X_OUT, y0, y1, TOP_OUT - T, TOP_OUT)
    caps = [box(X_IN, X_OUT, ya, yb, BED, TOP_OUT) for ya, yb in end_caps]
    return U(w_in, w_mid, w_out, top_i, top_o, *caps)


def foot_pad(y):
    return cyl("z", FOOT_X, y, FOOT_R, BED, BED + 10.0, 64), cyl("z", FOOT_X, y, STUD_R, BED - 1.0, BED + 3.0)


def base(side=1):
    yoke = U(hollow_box(0.25, X_IN + T, YOKE_Y0, YOKE_Y1, BED, Z_TOPY),
             hollow_box(0.25, 40.0, -64.0, YOKE_Y1, BED, Z_TOPY),
             box(0.25, X_IN, -1.5, 1.5, BED, Z_TOPY),                                # mid web
             box(0.25, 40.0, -64.0, DECK_Y1, Z_TOPY - 6.0, Z_TOPY))                  # the deck under the mast's foot, 6 thick
    pads = [cyl("z", x, y, 8.0, Z_TOPY - 10.0, Z_TOPY) for x, y in FOOT_BOLTS if x > 0]
    rail = channel(-64.0, Y_SPL, end_caps=((-64.0, -64.0 + T),))
    bulk = [box(X_IN, X_OUT, ya, ya + T, BED, TOP_OUT) for ya in (YOKE_Y0, YOKE_Y1 - T)]   # where the yoke's walls meet the channel
    plate = box(X_IN, X_OUT, Y_SPL - 8.0, Y_SPL, BED, TOP_OUT)
    pad, stud = foot_pad(FEET["rear"])
    body = U(yoke, *pads, rail, *bulk, plate, pad)
    cuts = U(keepouts(1), stud,
             *[cyl("z", x, y, CLR_R, Z_TOPY - 12.0, Z_TOPY + 1.0) for x, y in FOOT_BOLTS],
             *[cyl("z", x, y, 7.0, FLOOR - 2.0, Z_TOPY - 10.0) for x, y in FOOT_BOLTS],    # the key's way up from under the yoke
             *[cyl("z", x, y, 8.0, FLOOR - 2.0, BED + T + 1.0) for x in (60.0, 100.0) for y in (-16.0, 15.0)],   # the yoke's cells open below (no sealed void)
             *[cyl("z", 20.0, y, 8.0, FLOOR - 2.0, BED + T + 1.0) for y in (-33.0, 15.0)],
             *[cyl("y", x, z, CLR_R, Y_SPL - 9.0, Y_SPL + 1.0) for x, z in SPLICE])
    m = body - cuts
    return m if side > 0 else mirror_x(m)


def rail(side=1):
    ch = channel(Y_SPL + 0.25, 250.0, end_caps=((250.0 - T, 250.0),))
    plate = box(X_IN, X_OUT, Y_SPL + 0.25, Y_SPL + 8.25, BED, TOP_OUT)
    pad, stud = foot_pad(FEET["front"])
    body = U(ch, plate, pad)
    cuts = U(keepouts(1), stud, *[cyl("y", x, z, CLR_R, Y_SPL - 1.0, Y_SPL + 10.0) for x, z in SPLICE])
    m = body - cuts
    return m if side > 0 else mirror_x(m)


def foot():
    return U(cyl("z", 0, 0, FOOT_R, 0.0, 5.0, 64), cyl("z", 0, 0, STUD_R - 0.1, 5.0, 7.8, 32))


# ------------------------------------------------------------------ hand check: sections sliced from the mesh, the cases' moments on them
def allow(cls):
    return XT * KD / SF[cls], ZT * KD / SF[cls], S_ * KD / SF[cls]


def section(tm, origin, normal, pitch=0.25):
    """area, centroid (3D), in-plane axes (ex, ey) and section moduli about them of the section of tm on a plane: the section's loops
    rasterised at `pitch` (even-odd, so holes are holes)"""
    from matplotlib.path import Path as MPath
    s = tm.section(plane_origin=origin, plane_normal=normal)
    p2, Tm = s.to_planar()
    loops = [np.asarray(d) for d in p2.discrete]
    lo = np.min([l.min(0) for l in loops], 0); hi = np.max([l.max(0) for l in loops], 0)
    gx = np.arange(lo[0] + pitch / 2, hi[0], pitch); gy = np.arange(lo[1] + pitch / 2, hi[1], pitch)
    X, Y = np.meshgrid(gx, gy); P = np.c_[X.ravel(), Y.ravel()]
    inside = np.zeros(len(P), bool)
    for l in loops:
        inside ^= MPath(l).contains_points(P)
    q = P[inside]; a = pitch * pitch; A = len(q) * a
    cx, cy = q.mean(0)
    Iu = ((q[:, 1] - cy) ** 2).sum() * a; Iv = ((q[:, 0] - cx) ** 2).sum() * a
    return dict(A=A, c=(Tm @ np.array([cx, cy, 0, 1.0]))[:3], ex=Tm[:3, 0], ey=Tm[:3, 1],
                Wx=Iu / np.abs(q[:, 1] - cy).max(), Wy=Iv / np.abs(q[:, 0] - cx).max())


def stress_index(sec, F, M, cls, Am=None, t=T):
    """in-layer index at a section: |N|/A + |M about its x|/Wx + |M about its y|/Wy, torsion through a closed cell (Am) if given, else
    the walls' share of shear only; von Mises against the in-layer allowable"""
    n = np.cross(sec["ex"], sec["ey"])
    N = abs(np.dot(F, n)); Mx = abs(np.dot(M, sec["ex"])); My = abs(np.dot(M, sec["ey"])); Tz = abs(np.dot(M, n))
    sig = N / sec["A"] + Mx / sec["Wx"] + My / sec["Wy"]
    V = np.linalg.norm(F - np.dot(F, n) * n)
    tau = V / (0.5 * sec["A"]) + (Tz / (2 * Am * t) if Am else 0.0)
    a = allow(cls)
    return np.sqrt(sig ** 2 + 3 * tau ** 2) / a[0]


def hand_check(name, tm, sections, loads):
    """sections: [(label, origin, normal, Am or None, the side of the plane the loads act on: +1 / -1)];
    loads(case) -> [(F, at)] acting on the part beyond the section. Returns rows and prints a table."""
    out = []
    for lab, o, nrm, Am, sd in sections:
        sec = section(tm, o, nrm)
        worst = (0.0, None)
        for cn, cls, fl in loads():
            F = np.zeros(3); M = np.zeros(3)
            for f, at in fl:
                if np.dot(np.asarray(at) - o, nrm) * sd > 0:
                    F += f; M += np.cross(np.asarray(at) - sec["c"], f)
            ix = stress_index(sec, F, M, cls, Am)
            if ix > worst[0]:
                worst = (ix, cn)
        out.append(dict(section=lab, A_mm2=round(sec["A"]), Wx=round(sec["Wx"]), Wy=round(sec["Wy"]), index=round(worst[0], 3), case=worst[1]))
    for r in out:
        print(f"  {name} {r['section']:44s} A {r['A_mm2']:6d} Wx {r['Wx']:7d} Wy {r['Wy']:7d}  index {r['index']:.3f}  {r['case']}")
    return out


def arm_loads():
    """the arm's cases at the stub's face, the mast's own weight left out (its section sees the arm only): [(name, cls, [(F, at)])]"""
    out = []
    for nm, cls, loads, My, acc in arm_cases():
        F, M = wrench(loads, My)                         # about the origin (the face's centre)
        # as two forces: F at the origin, and the couple M as +-M x e / |e|^2 forces 100 apart along the axis it is not about
        fl = [(F, np.zeros(3))]
        for e in np.eye(3):
            m = np.dot(M, e)
            if abs(m) < 1e-9:
                continue
            d = np.cross(e, [1.0, 0, 0]) if abs(e[0]) < 0.9 else np.cross(e, [0, 1.0, 0])
            d /= np.linalg.norm(d); f = np.cross(e, d) * m / 100.0
            fl += [(f, d * 50.0), (-f, -d * 50.0)]
        out.append((nm, cls, fl))
    return out


def floor_loads(side, which):
    """the floor at the side's feet per case (which: ('rear', 'front') or ('front',)): [(name, cls, [(F, at)])]"""
    out = []
    for cn, cls, feet, acc in floor_cases():
        fl = [(f, np.array([side * FOOT_X, FEET[k], BED])) for (s, k), f in feet.items() if s == side and k in which]
        if fl:
            out.append((cn, cls, fl))
    return out


# ------------------------------------------------------------------ write
def save(name, m_w, R, note):
    """parts/<name>/part.stl in W (the page, the box check), print/<name>.stl in its print pose (R: 4x4 from W), laid on z 0;
    the solid check on the print mesh; one line"""
    tm = to_tri(m_w) if not isinstance(m_w, trimesh.Trimesh) else m_w
    assert tm.is_watertight and tm.is_volume, name
    d = os.path.join(ROOT, "parts", name); os.makedirs(d, exist_ok=True)
    tm.export(os.path.join(d, "part.stl"))
    pr = tm.copy(); pr.apply_transform(R); pr.apply_translation(-pr.bounds[0])
    os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
    p = os.path.join(ROOT, "print", f"{name}.stl"); pr.export(p)
    sc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solid_check.py"), p], capture_output=True, text=True)
    ok = "PASS" if sc.returncode == 0 else "FAIL"
    v = tm.volume / 1000
    bed_mm2 = section(pr, [0, 0, pr.vertices[:, 2].min() + 0.1], [0, 0, 1.0], pitch=0.5)["A"]      # what lies on the plate
    line = dict(name=name, cm3=round(v, 2), g=round(v * PCTG, 1), as_printed_mm=np.round(pr.extents, 1).tolist(), bed_mm2=round(bed_mm2),
                solid_check=ok, note=note)
    print(f"{name}: {v:.1f} cm3, {v * PCTG:.0f} g PCTG, {np.round(pr.extents, 1).tolist()} mm as printed, {bed_mm2:.0f} mm2 on the bed, solid check {ok}; {note}")
    if ok == "FAIL":
        print(sc.stdout[-1500:], sc.stderr[-800:])
    json.dump(line, open(os.path.join(d, "make.json"), "w"), indent=1)
    return tm, line


def rot(axis, deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), axis)


WASHER_D = 15.0
INSERT_PULL = 1500.0        # N, an M5 x D7 x L6 heat-set insert's pull-out in PCTG (E: the vendor's PETG figure, no coupon yet)


def splice_check(side):
    """the splice per case: the front foot's force taken at the splice plane by the plate's bearing (its four corners, compression only)
    and five bolts (tension only, into the inserts), least squares; the bolts' share of the shear bearing on 8 of plate each. Index:
    the worst bolt's pull against the insert's pull-out / SF, and the bolt's bearing against the in-layer allowable."""
    from scipy.optimize import lsq_linear
    y = Y_SPL
    corners = [(X_IN + 2, BED + 2), (X_OUT - 2, BED + 2), (X_IN + 2, TOP_IN - 2), (X_OUT - 2, TOP_OUT - 2)]
    pts = [np.array([side * x, y, z]) for x, z in corners] + [np.array([side * x, y, z]) for x, z in SPLICE]
    sg = [-1.0] * len(corners) + [1.0] * len(SPLICE)     # on the rail: the base pushes it -Y at the corners... the bolts pull it +Y? signs below
    c = np.mean(pts, 0)
    worst = [(0.0, None), (0.0, None)]
    for cn, cls, fl in floor_loads(side, ("front",)):
        F = sum(f for f, _ in fl); M = sum(np.cross(at - c, f) for f, at in fl)
        # the rail is held at the splice: reactions R at the points with sum R = -F, sum (p - c) x R = -M. Normal (Y) parts:
        # corners can only push the rail toward +Y (bearing, the base is at -Y), bolts only pull it toward -Y.
        A = np.array([[(-s) for s in sg], [(-s) * (p - c)[2] * -1 for s, p in zip(sg, pts)], [(-s) * (p - c)[0] for s, p in zip(sg, pts)]])
        # a Y force r_y at (dx, 0, dz): moment = (dz * 0 - ... ) -> M_x = -dz r_y ... written out: Mx += -dz * ry, Mz += dx * ry
        A = np.array([[1.0] * len(pts), [-(p - c)[2] for p in pts], [(p - c)[0] for p in pts]])
        b = np.array([-F[1], -M[0], -M[2]])
        lo = [0.0 if s < 0 else -np.inf for s in sg]; hi = [np.inf if s < 0 else 0.0 for s in sg]
        n = len(pts); r = lsq_linear(np.vstack([A, 1e-3 * np.eye(n)]), np.r_[b, np.zeros(n)], bounds=(lo, hi), method="bvls").x
        pull = max(-r[i] for i in range(len(corners), n))
        ix = pull / (np.pi * WASHER_D * 8.0 * allow(cls)[2])          # a Ø15 washer pulling through the 8 plate (shear round it)
        if ix > worst[0][0]:
            worst[0] = (ix, cn, pull)
        V = np.hypot(F[0], F[2]) / len(SPLICE)
        ib = V / (5.0 * 8.0) / allow(cls)[0]
        if ib > worst[1][0]:
            worst[1] = (ib, cn)
    out = [dict(section=f"splice bolt pull through its washer ({worst[0][2]:.0f} N)", index=round(worst[0][0], 3), case=worst[0][1]),
           dict(section="splice bolt bearing", index=round(worst[1][0], 3), case=worst[1][1])]
    for r_ in out:
        print(f"  splice {r_['section']:44s} index {r_['index']:.3f}  {r_['case']}")
    return out


def extra(line, rows):
    d = os.path.join(ROOT, "parts", line["name"])
    line["hand_check"] = rows
    line["worst"] = max([r["index"] for r in rows], default=None)
    json.dump(line, open(os.path.join(d, "make.json"), "w"), indent=1)
    print(f"  worst hand-check index {line['worst']}")


def joint_check(label, bear, bolts, cases, cap, caps=None):
    """a bolted face: bearing points (compression only) and bolts (tension only) take the normal part of each case's wrench (least
    squares); cap(cls, pull) -> index for the worst bolt. bear, bolts: [(point, normal into the part)]; cases: [(name, cls, F, M about
    the origin)] on the part. Returns a row."""
    from scipy.optimize import lsq_linear
    n = bear[0][1] / np.linalg.norm(bear[0][1])
    pts = [p for p, _ in bear] + [p for p, _ in bolts]; nb = len(bear)
    c = np.mean(pts, 0); worst = (0.0, None, 0.0)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    for cn, cls, F, M in cases:
        Mc = M - np.cross(c, F)
        # reactions r_i n at p_i (r > 0 pushes the part along n: bearing; r < 0 pulls: a bolt): sum r = -F.n, sum (p-c) x r n = -Mc (in-plane parts)
        A = np.array([[1.0] * len(pts)] + [[np.dot(np.cross(p - c, n), e) for p in pts] for e in (e1, e2)])
        b = np.array([-np.dot(F, n), -np.dot(Mc, e1), -np.dot(Mc, e2)])
        lo = [0.0] * nb + [-np.inf] * len(bolts); hi = [np.inf] * nb + [0.0] * len(bolts)
        r = lsq_linear(np.vstack([A, 1e-3 * np.eye(len(pts))]), np.r_[b, np.zeros(len(pts))], bounds=(lo, hi), method="bvls").x
        pulls = [-x for x in r[nb:]]; pull = max([0.0] + pulls)
        ix = max(cap(cls, p_) if caps is None else caps[i](cls, p_) for i, p_ in enumerate(pulls))
        if ix > worst[0]:
            worst = (ix, cn, pull)
    row = dict(section=f"{label} ({worst[2]:.0f} N)", index=round(worst[0], 3), case=worst[1])
    print(f"  {label:52s} pull {worst[2]:6.0f} N  index {worst[0]:.3f}  {worst[1]}")
    return row


def arm_wrenches():
    return [(nm, cls, *wrench(loads, My)) for nm, cls, loads, My, acc in arm_cases()]


def mast_joints():
    """the mast's foot on the decks (8 M5 into the foot's inserts) and the stub on its face (the washers' seats in the face)"""
    zf = Z_FOOT
    bear = [(np.array([x, y, zf]), np.array([0, 0, 1.0])) for x in (-40.0, 38.0) for y in (-62.0, -2.0)]
    bolts = [(np.array([x, y, zf]), np.array([0, 0, 1.0])) for x, y in FOOT_BOLTS]
    # the arm's wrench acts on the mast; the decks hold it: the foot's reactions balance the arm's loads (the mast's own 0.46 kg added at its CG)
    cases = []
    for nm, cls, F, M in arm_wrenches():
        g = 3.0 if "handling" in nm else 1.0
        w = np.array([0, 0, -0.462 * 9.81 * g]); cw = np.array([0, -32.0, -45.0])
        cases.append((nm, cls, F + w, M + np.cross(cw, w)))
    ins = lambda cls, p: p / (INSERT_PULL / SF[cls]); thr = lambda cls, p: p / (np.pi * WASHER_D * 6.0 * allow(cls)[2])   # the weaker side of a through bolt: its washer under the 6 deck
    r1 = joint_check("mast foot: front row inserts (1500 N, E), rear row through", bear, bolts, cases, None,
                     caps=[ins if y > -30 else thr for x, y in FOOT_BOLTS])
    bear2 = [(np.array([39.0 * np.cos(a), 0.0, 39.0 * np.sin(a)]), np.array([0, -1.0, 0])) for a in np.radians(np.arange(0, 360, 30))
             if not (39.0 * np.cos(a) < -24 and 39.0 * np.sin(a) > 14)]
    bolts2 = [(np.array([x, 0.0, z]), np.array([0, -1.0, 0])) for x, z in STUB_BOLTS]
    # the stub pushes the face (-Y into the mast) and its bolts pull the face toward +Y: on the mast, the stub's reaction = the arm's loads
    r2 = joint_check("stub on the face: Ø15 washer through 10 of face", bear2, bolts2, arm_wrenches(),
                     lambda cls, p: p / (np.pi * WASHER_D * 10.0 * allow(cls)[2]))
    r3 = joint_check("  the same pull on the stub flange's inserts (arm stream)", bear2, bolts2, arm_wrenches(),
                     lambda cls, p: p / (INSERT_PULL / SF[cls]))
    r3["section"] = "for the arm stream, not this part: " + r3["section"]
    return [r1, r2], r3
