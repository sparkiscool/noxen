"""Swap arm v8 PROTOTYPES (15 Nov 2026, Spark: "we don't need generated parts for bench testing or even the first test"): the drawn hub, carriage
and lift (geometry: parts/box_arm8/make.py hub_proto, parts/box_arm8_lib.py carriage_proto / lift_proto: the engine setups' drawn attachments,
cuts and room, as simple solid sections) and the stub's joint to the stand's mast. Here: the hand checks (sections sliced from the mesh, joints by
formula, every case of box_arm8_lib.cases, each its own) and save(). Index = stress over the PCTG allowable (topo/material.py E: in-layer 46,
across layers 23, shear 23 MPa, x 0.85, over SF 4 flight / 2.5 misuse / 10 sustained), or a bolt's pull over its capacity / SF; 1.0 = the SF is just met.
"""
import importlib.util, json, math, os, subprocess, sys
import numpy as np, trimesh
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p_ in (HERE, ROOT, os.path.join(ROOT, "tools")):
    if p_ not in sys.path: sys.path.insert(0, p_)
import box_arm8_lib as L
_M = None
def M():
    global _M
    if _M is None:
        sp = importlib.util.spec_from_file_location("box_arm8_make", os.path.join(HERE, "box_arm8", "make.py")); _M = importlib.util.module_from_spec(sp); sp.loader.exec_module(_M)
    return _M

RHO = 1.23
XT, ZT, S_, KD = 46.0, 23.0, 23.0, 0.85
SF = {"flight": 4.0, "misuse": 2.5, "sustained": 10.0}
INSERT = {"M5": 1500.0, "M3": 900.0}     # N pull-out in PCTG, E: M5 x D7 x L6 the vendor's PETG figure (parts/box_stand2p_lib.py); M3 x 5.7 scaled by its knurl's area
def allow(cls, across=False): return (ZT if across else XT) * KD / SF[cls]
def shear(cls): return S_ * KD / SF[cls]
def rot(axis, deg): return trimesh.transformations.rotation_matrix(np.radians(deg), axis)


def cases():
    """[(name, cls, F, point with the head's travel)]: both packs' cases, the Scout's carrying and handling left out (the Overseer's envelope them)"""
    out = []
    for pk in ("overseer", "scout"):
        for nm, cls, F, P, s in L.cases(pk):
            if pk == "scout" and ("handling" in nm or "carrying" in nm): continue
            out.append((nm, cls, np.asarray(F, float), np.asarray(P, float), float(s)))
    return out


def section(tm, origin, normal, clip=None, pitch=0.25):
    """area, centroid, axes and moduli of tm's section on a plane (loops rasterised, even-odd), clip: (x0, x1, y0, y1, z0, z1) keeps that box of it"""
    from matplotlib.path import Path as MPath
    s = tm.section(plane_origin=origin, plane_normal=normal); p2, Tm = s.to_planar()
    loops = [np.asarray(d) for d in p2.discrete]
    lo = np.min([l.min(0) for l in loops], 0); hi = np.max([l.max(0) for l in loops], 0)
    X, Y = np.meshgrid(np.arange(lo[0] + pitch / 2, hi[0], pitch), np.arange(lo[1] + pitch / 2, hi[1], pitch)); P = np.c_[X.ravel(), Y.ravel()]
    ins = np.zeros(len(P), bool)
    for l in loops: ins ^= MPath(l).contains_points(P)
    q = P[ins]
    if clip is not None:
        w = (Tm @ np.c_[q, np.zeros(len(q)), np.ones(len(q))].T).T[:, :3]; c = np.asarray(clip, float)
        q = q[np.all((w >= c[0::2] - 1e-6) & (w <= c[1::2] + 1e-6), axis=1)]
    a = pitch * pitch; A = len(q) * a; cx, cy = q.mean(0)
    Iu = ((q[:, 1] - cy) ** 2).sum() * a; Iv = ((q[:, 0] - cx) ** 2).sum() * a
    return dict(A=A, c=(Tm @ np.array([cx, cy, 0, 1.0]))[:3], ex=Tm[:3, 0], ey=Tm[:3, 1], Wx=Iu / np.abs(q[:, 1] - cy).max(), Wy=Iv / np.abs(q[:, 0] - cx).max())


def index(sec, loads, cls, across=False):
    """von Mises at a section over the allowable: |N|/A + |Mx|/Wx + |My|/Wy; shear V / (A/2) and the torsion over 1.2 x the smaller modulus (a solid section)"""
    F = sum((f for f, _ in loads), np.zeros(3)); Mo = sum((np.cross(np.asarray(at, float) - sec["c"], f) for f, at in loads), np.zeros(3))
    n = np.cross(sec["ex"], sec["ey"])
    sig = abs(F @ n) / sec["A"] + abs(Mo @ sec["ex"]) / sec["Wx"] + abs(Mo @ sec["ey"]) / sec["Wy"]
    tau = np.linalg.norm(F - (F @ n) * n) / (0.5 * sec["A"]) + abs(Mo @ n) / (1.2 * min(sec["Wx"], sec["Wy"]))
    return float(np.sqrt(sig ** 2 + 3 * tau ** 2) / allow(cls, across))


def worst(rows, label, items):
    """items: [(index, case)] -> one row"""
    ix, cn = max(items, key=lambda t: t[0]) if items else (0.0, None)
    rows.append(dict(check=label, index=round(float(ix), 3), case=cn)); print(f"  {label:78s} {ix:.3f}  {cn}")


def joint(bear, bolts, n, F, Mo):
    """a bolted face: bearing points (compression only) and bolts (tension only) take the wrench (F, M about the origin) along n (least squares); -> the bolts' pulls"""
    from scipy.optimize import lsq_linear
    pts = [np.asarray(p, float) for p in bear + bolts]; nb = len(bear); n = np.asarray(n, float); c = np.mean(pts, 0)
    e1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1); Mc = Mo - np.cross(c, F)
    A = np.array([[1.0] * len(pts)] + [[np.cross(p - c, n) @ e for p in pts] for e in (e1, e2)]); b = np.array([-(F @ n), -(Mc @ e1), -(Mc @ e2)])
    lo = [0.0] * nb + [-np.inf] * len(bolts); hi = [np.inf] * nb + [0.0] * len(bolts)
    r = lsq_linear(np.vstack([A, 1e-3 * np.eye(len(pts))]), np.r_[b, np.zeros(len(pts))], bounds=(lo, hi), method="bvls").x
    return [max(0.0, -x) for x in r[nb:]], r[:nb]


# ------------------------------------------------------------------ hub
def hub_check(tm):
    m = M(); rows = []; y0, y1, R = L.PLATE; x, y = L.SCREW; w = L.THRUST_WALL
    both = lambda nm: ("handling" in nm or "carrying" in nm)
    sec = section(tm, [0, m.FLARE_Y0 - 1.0, 0], [0, 1.0, 0])
    it = []
    for nm, cls, F, P, s in cases():
        Pa = P + [0, 0, s]; ld = [(F, Pa)] + ([(F, Pa * [-1, 1, -1])] if both(nm) else [])
        it.append((index(sec, ld, cls, across=True), nm))
    worst(rows, f"sleeve under the cone (Y {m.FLARE_Y0 - 1:.1f}, across the layers; A {sec['A']:.0f} mm2, W {min(sec['Wx'], sec['Wy']):.0f} mm3)", it)
    bear = [[sx * 9.0, y1, u] for u in (L.RAIL[0], 36.0, 62.0, R - 0.5) for sx in (-1, 1)]; bolts = [[0.0, y1, u] for u in L.RAIL_BOLTS]
    keel = section(tm, [0, 47.0, 70.5], [0, 0, 1.0], clip=(-21, 21, 39, 55, 60, 80))
    ins, kl, th = [], [], []
    rm, rc, t = (8.0 + 4.3) / 2, 8.5, w["u"][1] - w["u"][0]
    for nm, cls, F, P, s in cases():
        Pa = P + [0, 0, s]; Fu = np.array([0, 0, F[2]]); rest = F - Fu
        Mo = np.cross(Pa, F) - np.cross([x, y, w["u"][0]], Fu)                       # the screw takes the bar direction at its axis; the rail the rest
        pull, press = joint(bear, bolts, [0, 1.0, 0], -rest, -Mo)                    # on the rail: the seat pushes it +Y, the bolts pull it -Y
        ins.append((max(pull) / (INSERT["M5"] / SF[cls]), nm))
        ld = [(np.array([0, pull[1], 0]), bolts[1])] + [(np.array([0, -press[i], 0]), bear[i]) for i in (6, 7)]
        kl.append((index(keel, ld, cls), nm))
        sig = 6 * (abs(F[2]) / (2 * math.pi * rm)) * (rc - rm) * 1.6 / t ** 2          # the wall as a 4 mm diaphragm in its cradle: the ring load 2.35 inside the cradle's edge, x 1.6 for the open upper half
        th.append((sig / allow(cls), nm))
    worst(rows, "rail bolts: M5 heat-set inserts, 1500 N (E)", ins)
    worst(rows, f"keel beyond the cone under a rail's outer bolt (u 70.5; W {keel['Wx']:.0f} mm3)", kl)
    worst(rows, "thrust wall, 4.0 between the two F8-16M, in its cradle", th)
    tq = 1000 * L.AXLE_STALL_NM / m.TOOTH_R; rows.append(dict(check="ring teeth at the axle's stall: %.0f N over 37 teeth in mesh (a quarter wrap), 0.6 x 6 of root each" % tq,
                                                              index=round(tq / (37 * 0.6 * 6.0) / shear("misuse"), 3), case="axle stall"))
    print("  " + rows[-1]["check"], rows[-1]["index"])
    return rows


# ------------------------------------------------------------------ carriage
def carriage_check(tm):
    rows = []; a = L.AXLE; lev = a["lever"]
    cp = [np.array([x - math.copysign(lev, x), 62.75, L.Z0 + c]) for x, c in L.WHEELS]         # each wheel's contact on the ridge, park frame
    W15 = math.pi * (15.0 ** 4 - 5.4 ** 4) / (32 * 15.0); A15 = math.pi / 4 * (15.0 ** 2 - 5.4 ** 2); web = L.PLATE_Y[1] - L.PLATE_Y[0] - a["back"] - a["front"]
    secs = [("spine and plate under the top wheel, above the rib (u 64)", section(tm, [30, 68, 64.0], [0, 0, 1.0], clip=(11, 51, 54, 79, 0, 200)), [1]),
            ("spine and plate over the low wheel, below the rib (u 28)", section(tm, [30, 68, 28.0], [0, 0, 1.0], clip=(11, 51, 54, 79, 0, 200)), [0]),
            ("strap to the third wheel, beside the rail (x 0)", section(tm, [0.0, 73, L.Z0 + 14], [1.0, 0, 0]), [2])]
    its = [[] for _ in secs]; ax, sh, mg = [], [], []
    for nm, cls, F, P, s in cases():
        wl, _, _ = L.wheel_loads(F, P, s); f = [np.array([fx, fy, 0.0]) for fx, fy in wl]
        R_, Mm, C, _ = L.mgn_wrench(F, P)
        for k, (lab, sec, wh) in enumerate(secs):
            ld = [(f[i], cp[i]) for i in wh]                                          # beyond the section: that wheel's force at its contact (the block's share on that side only relieves it)
            its[k].append((index(sec, ld, cls), nm))
        for i in range(3):
            fy = abs(f[i][1]) + (10.0 if i == 2 else 0.0)
            ax.append(((fy * lev / W15 + fy / A15) / allow(cls), nm)); sh.append((fy / (math.pi * 15.0 * web) / shear(cls), nm))
        T = abs(Mm[0]) * 37.5 / (2 * (37.5 ** 2 + 12.5 ** 2)) + abs(Mm[2]) / 6.0 / 4 + max(0.0, -R_[1]) / 4
        mg.append((T / (INSERT["M3"] / SF[cls]), nm))
    for k, (lab, sec, wh) in enumerate(secs): worst(rows, f"{lab}; A {sec['A']:.0f} mm2", its[k])
    worst(rows, "wheel axle: the V's contact %.1f off the axle, pried on the plate through the o 15 washers" % lev, ax)
    worst(rows, "wheel axle: the washers through the plate's %.2f web" % web, sh)
    worst(rows, "MGN rail on the spine: four M3 heat-set inserts, 900 N (E)", mg)
    return rows


def couple(Mo):
    """a moment as pairs of forces 40 apart about a point: [(force, offset)]"""
    out = []
    for e in np.eye(3):
        mm = Mo @ e
        if abs(mm) < 1e-9: continue
        d = np.cross(e, [1.0, 0, 0]) if abs(e[0]) < 0.9 else np.cross(e, [0, 1.0, 0]); d /= np.linalg.norm(d); f = np.cross(e, d) * mm / 40.0
        out += [(f, d * 20.0), (-f, -d * 20.0)]
    return out


# ------------------------------------------------------------------ lift
def lift_check(tm):
    rows = []; x, y = L.SCREW
    sec = section(tm, [26.0, 97, 20], [1.0, 0, 0]); lug = section(tm, [59, 72.6, 15], [0, 1.0, 0], clip=(50, 68, 70, 75, 2, 28))
    yk, bl, so, lg = [], [], [], []
    Aw = math.pi / 4 * (L.MOUNT_WASHER ** 2 - 3.4 ** 2); h = L.MGN["pattern"]; depth = L.SOCKET[1] - L.SOCKET[0]
    for nm, cls, F, P, s in cases():
        R_, Mm, C, _ = L.mgn_wrench(F, P)
        yk.append((index(sec, [(F, P)], cls), nm))
        T = abs(Mm[0]) / (2 * h) + abs(Mm[2]) / (2 * h) + max(0.0, R_[1]) / 4
        bl.append((T / Aw / allow(cls), nm))
        Mr = np.linalg.norm(np.cross(P - [0, L.SOCKET[1], L.BOOMS[0][1]], F)) / 3.0     # each rod's moment at its socket's mouth
        so.append((6 * Mr / (6.2 * depth ** 2) / allow(cls), nm))
        lg.append((index(lug, [(np.array([0, 0, -F[2]]), [x, y, sum(L.NUT_U) / 2])], cls), nm))
    worst(rows, f"yoke into the plate (x 26: its twist, in the layers; A {sec['A']:.0f} mm2)", yk)
    worst(rows, "plate on the MGN block: four M3, each head on an o %g washer" % L.MOUNT_WASHER, bl)
    worst(rows, "boom sockets: a o 6 rod's moment on %g of bore" % depth, so)
    worst(rows, f"nut lug at the arm (Y 72.6; A {lug['A']:.0f} mm2)", lg)
    return rows


# ------------------------------------------------------------------ stub (made by parts/box_arm8/make.py)
def stub_check(stub):
    """the stub on the stand v2 mast: the mast's own cases (parts/box_stand2p_lib.py arm_wrenches, on the mast) taken by the flange on the face and the three bolts"""
    import parts.box_stand2p_lib as SP
    m = M(); tm = m.tm(stub); rows = []
    bolts = [[m.BOLT_R * math.cos(math.radians(a)), 0.0, m.BOLT_R * math.sin(math.radians(a))] for a in m.BOLT_A]
    bear = [[43.0 * math.cos(a), 0.0, 43.0 * math.sin(a)] for a in np.radians(np.arange(0, 360, 30)) if not (43.0 * math.cos(a) < -24.0 and 43.0 * math.sin(a) > 14.0)]
    sec = section(tm, [0, m.HUB_Y0 + 1.0, 0], [0, 1.0, 0], clip=(-30, 30, 0, 50, -30, 30))
    pt, fl, sp = [], [], []
    for nm, cls, F, Mo in SP.arm_wrenches():
        pull, _ = joint(bear, bolts, [0, 1.0, 0], np.asarray(F, float), np.asarray(Mo, float))        # the arm's wrench on the stub; the face pushes it +Y, the bolts pull it -Y
        T = max(pull)
        pt.append((T / (math.pi * 15.0 * m.FL_SKIN * shear(cls)), f"{nm} ({T:.0f} N)"))
        fl.append((6 * T * 9.0 / ((15.0 + 2 * 9.0) * m.FL_T ** 2) / allow(cls), f"{nm} ({T:.0f} N)"))
        sp.append((index(sec, [(np.asarray(F, float), np.zeros(3))] + [(g, d) for g, d in couple(np.asarray(Mo, float))], cls, across=True), nm))
    worst(rows, "flange bolt: its o 15 washer through the %.1f skin (was an M5 insert in a 6 flange: 0.887)" % m.FL_SKIN, pt)
    worst(rows, "flange %g thick at the spigot's shoulder, the bolt 9 out (its o 15 washer's load spread at 45 deg: 33 of root; the 6 mm flange read 2.4)" % m.FL_T, fl)
    worst(rows, f"spigot at the shoulder (across the layers; A {sec['A']:.0f} mm2)", sp)
    return rows


# ------------------------------------------------------------------ save
def save(name, man, R, note, check):
    """parts/<name>/part.stl in the arm frame (head A, park), print/<name>.stl in its print pose (R from the arm frame) on z 0, the solid check on it,
    preview.png of the print file, make.json with the hand check's rows"""
    me = man.to_mesh(); tm = trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=False)    # the manifold's own mesh: closed as it is
    assert tm.is_watertight and len(man.decompose()) == 1, name
    d = os.path.join(ROOT, "parts", name); os.makedirs(d, exist_ok=True); tm.export(os.path.join(d, "part.stl"))
    pr = tm.copy(); pr.apply_transform(R); pr.apply_translation(-pr.bounds[0])
    os.makedirs(os.path.join(ROOT, "print"), exist_ok=True); f = os.path.join(ROOT, "print", name + ".stl"); pr.export(f)
    sc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solid_check.py"), f], capture_output=True, text=True); ok = "PASS" if sc.returncode == 0 else "FAIL"
    if ok == "FAIL": print(sc.stdout[-1500:], sc.stderr[-600:])
    g = tm.volume / 1000 * RHO; e = np.round(pr.extents, 1).tolist()
    print(f"{name}: {g:.0f} g PCTG, {e} mm as printed, solid check {ok}; {note}")
    rows = check(tm); w = max(rows, key=lambda r: r["index"])
    print(f"  worst hand-check index {w['index']}: {w['check']} ({w['case']})")
    json.dump(dict(name=name, g=round(g, 1), as_printed_mm=e, fits_180=bool(max(e) <= 180.0), solid_check=ok, note=note, worst=w["index"], worst_check=w["check"], worst_case=w["case"], hand_check=rows),
              open(os.path.join(d, "make.json"), "w"), indent=1)
    from topo.build import render
    render(pr, os.path.join(d, "preview.png"), title=f"{name} (drawn prototype) on its bed", info=f"PCTG {g:.0f} g | {e[0]} x {e[1]} x {e[2]} mm | solid check {ok} | hand calc worst {w['index']}: {w['check'][:60]}")
    if ok == "FAIL": sys.exit(1)
    return tm
