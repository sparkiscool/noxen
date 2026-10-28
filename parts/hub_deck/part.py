"""Large test part: Overseer hub deck (v9 setup, 368k cells at h 1.25, 5.3 GB peak, 0.9 h on 4 cores).
Copied from nox/overseer/core_topo_v3.py with the command-line handling removed; its inputs live in inputs/."""
import json, os, sys
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
from topo import Part
from topo.part import man

SPEC = dict(h=1.25, gen_cm3=24.0, max_worst=0.5)
LEAN = False

R = json.load(open(os.path.join(INP, "core_result.json")))
C = R["coupon"]
A = json.load(open(os.path.join(INP, "core.assembly.json")))
M = {p["name"]: trimesh.load(os.path.join(INP, p["mesh"])) for p in A["parts"]}
PI_NM = "Pi 5 + Active Cooler"
G = 9.81e-3                                     # N per g per g
DIRS14 = [np.array(v, float) / np.linalg.norm(v) for v in
          [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
FLIGHT = {"flight 4 g": np.array([0, 0, 4.0]), "flight 2 g x": np.array([2.0, 0, 1]), "flight 2 g y": np.array([0, 2.0, 1]), "landing 6 g": np.array([0, 0, 6.0])}


def cyl(x, y, z0, z1, r, n=40):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def U(ms):
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)


def grown(name, d):
    """an assembly part grown by d (its convex pieces' hulls, offset): a keep-out"""
    t = M[name]
    hulls = [b.convex_hull for b in t.split(only_watertight=False) if len(b.faces) > 3] or [t.convex_hull]
    out = []
    for h_ in hulls:
        v = h_.vertices; c = v.mean(0); v2 = c + (v - c) * 1.0
        hh = trimesh.convex.convex_hull(np.vstack([v2 + d * np.array(s_) for s_ in [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]]))
        out.append(man(hh))
    return U(out)


def rigid_split(pts, F, P):
    """least-norm forces at pts that carry a force F acting at point P (force and moment): the payload on a rigid link"""
    pts = np.asarray(pts, float); c = pts.mean(0); n = len(pts); Am = np.zeros((6, 3 * n))
    for i, q in enumerate(pts - c):
        Am[:3, 3 * i:3 * i + 3] = np.eye(3); Am[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
    Mc = np.cross(np.asarray(P, float) - c, F)
    return (np.linalg.pinv(Am) @ np.r_[F, Mc]).reshape(n, 3)


# ================================================================ hub deck
KEELS = {int(k): (np.array(v[0], float), np.array(v[1], float)) for k, v in R["keels"].items()}
SK = R["socket"]; INS = R["insert"]; T = R["foot_h"]
KEEL_N = R["keel_n"]                                                       # max pull / push per keel (F)
RF = json.load(open(os.path.join(INP, "core_root_forces_low.json")))
ROOTS = {k: (np.array(v["core"], float), np.array(v["lug"], float)) for k, v in RF["points"].items()}   # core = the tube end
BS = R["brace_socket"]
Z_UNDER = 45.0                                                             # the top plate underside (S500 STEP)
FOOT_R = 7.0                                                               # feet Ø14: 5 of wall round the Ø4.0 insert hole (the gate erodes 2.0; v3's Ø12 left 4.0)


def along(a, d, l0, l1, r, n=40):
    """a cylinder of radius r on the axis a + d t, t in l0..l1"""
    q0, q1 = a + d * l0, a + d * l1
    t = trimesh.creation.cylinder(r, l1 - l0, sections=n, segment=np.vstack([q0, q1]))
    return man(t)


def arm_env(deg):
    """the arm session's root envelope along the arm at deg (27 Oct: roots u >= 58 for |v| <= 21); v3 used a box |y| > 20 ahead of
    x 56, which cut the mid feet (65, ±22.5) to a 3.5 sliver and capped every member at 4.44 (the solid gate FAILED)"""
    return box(58.0, 400.0, -21.0, 21.0, -1.0, 20.0).rotate([0, 0, deg])


def root_axis(name):
    a, b = ROOTS[name]; return a, (b - a) / np.linalg.norm(b - a)


def keel_axis(sy):
    a, b = KEELS[sy]; return a, (b - a) / np.linalg.norm(b - a)


def hub_deck_bodies():
    feet = [tuple(f) for f in R["feet_xy"]]
    ph = [tuple(p) for p in R["pi_holes"]]
    zp = C["z_pi"]
    # bay seats (hub deck v2): rails under the UBEC and the buck, inside the arm envelopes
    ub, bk = M["UBEC 5 V"].bounds, M["buck 12 V"].bounds
    rails = [box(ub[0][0], ub[1][0], y - 3, y + 3, 0, T) for y in (ub[0][1] + 3, ub[1][1] - 3)]
    rails += [box(bk[0][0], bk[1][0], y - 3, y + 3, 0, T) for y in (bk[0][1] + 3, min(bk[1][1] - 3, 17.0))]
    rails += [box(ub[1][0] - 6, ub[1][0], ub[0][1], ub[1][1], 0, T), box(bk[0][0], bk[0][0] + 6, ub[0][1], bk[1][1], 0, T)]
    # zip-tie channels 3 x 2.5 open on the plate side (the tie runs between deck and plate), 4.0 of deck above: v2's tunnels at z 2..4.5
    # left 2.0 skins across the whole deck, and the smoke (27 Oct) found the front hub cut off from the rest at a 3.2 member
    # v4: no zip-tie channels: the UBEC and the buck sit on the seat rails on 3M VHB foam tape (DEFAULT; ~576 mm² a bay against ~5 N in
    # a 35 g crash). Any channel under the bays crosses the whole deck and leaves < 6.25 of it above, which cut the front hub off from
    # the rest at the solid gate's member (v4 smoke 27 Oct: gate members capped at 4.96)
    slots = []
    # keel front hub: bosses on the keel axes, seated on the front feet (the keels' 387 N go into the plate at (96, ±17))
    bosses = [along(*keel_axis(sy), -2.0, SK["off"] + SK["l"], SK["od"] / 2) for sy in (1, -1)]
    # v7 (hub deck v6 FAILED strength: worst 2.72 in the top hit at the +y keel boss / front foot joint (94, 16, 8), 18 cm3 generated):
    # a drawn solid core round the keel bosses and the front feet (both sides), so that joint can't thin out
    # v10 (28 Oct, Spark: "a bunch of material stacked at the sockets, nothing at the rear"): the v7 seat blocks and the v9
    # bars are gone. The audit (topo/audit.py) now refuses them: one drawn body held 84 % of the drawn material with a
    # 9 mm core and 1.2 mm of room behind it. Attachments only; the room behind the sockets is opened instead (hub_env).
    seats = []
    # v9 (hub deck v8: strength 0.34 but the build's drawn test cut the +y tower off at the 2.0 erosion: its only thick joint ran through
    # the centreline, where the cage's tie tubes and two wire routes cut): drawn bars from each tower core back to its mid feet beside
    # the bays (the -y one outside the UBEC's y -18), so both sides join the deck through solid
    ub_y0 = M["UBEC 5 V"].bounds[0][1]
    braces = [k for k in ROOTS if k.startswith("L2 brace")]
    bosses += [along(*root_axis(k), -BS["wall"], BS["l"] + BS["lead"], BS["d"] / 2 + BS["wall"]) for k in braces]
    # v5 (PR 148 review): a root web Ø12 x 6.0 under every Pi post (a gusset into the body; v4's bare Ø7 posts were 99.8 % in the main
    # body at the gate's 2.0 erosion, under its 99.9 % bar); 6.0 stays under the board's underside parts (z 6.5..8)
    roots = [cyl(x, y, 0, 6.0, 6.0) for x, y in ph]
    drawn = U([cyl(x, y, 0, T, FOOT_R) for x, y in feet] + [cyl(x, y, 0, zp, 3.5) for x, y in ph] + roots + rails + bosses + seats)
    bores = [along(*keel_axis(sy), SK["off"], SK["off"] + SK["l"] + 1.0, SK["d"] / 2) for sy in (1, -1)]
    tubes = [along(*keel_axis(sy), SK["off"] + SK["l"] - 0.5, 400.0, 3.5) for sy in (1, -1)]
    bores += [along(*root_axis(k), 0.0, BS["l"] + BS["lead"] + 1.0, BS["d"] / 2) for k in braces]
    tubes += [along(*root_axis(k), BS["l"] + BS["lead"] - 0.5, 400.0, 3.5) for k in braces]
    wires = [U([man(b) for b in M["wires"].split(only_watertight=False) if len(b.faces) > 3])]   # the core's wire routes, exact tubes (a hull per
    # route would be a slab across the hub room); the flow lead crosses it   # the keel tube past the socket mouth + 0.5 (0.5 lead-in: no end slivers)
    ties = json.load(open(os.path.join(INP, "layout_result.json")))["links"]   # the cage's front core ties (parked body node)
    tie_cut = [man(trimesh.creation.cylinder(4.0, np.linalg.norm(np.subtract(b, a)), sections=24, segment=np.array([a, b], float)))
               for k, (a, b) in ties.items() if k.startswith("core tie front")]
    hub_env = box(56.0, 118.0, -34.0, 34.0, 0, 30.0)        # v10: back to x 56 (was 80) so a web can grow from the sockets to the mid feet
    plates = [man(M["top plate"])]
    bays = [box(m.bounds[0][0] - 1, m.bounds[1][0] + 1, m.bounds[0][1] - 1, m.bounds[1][1] + 1, T, Z_UNDER) for m in (M["UBEC 5 V"], M["buck 12 V"])]
    cuts = U(plates + bays + [cyl(x, y, -1, T + 1, INS[0] / 2) for x, y in feet] + tie_cut + [cyl(x, y, zp - 4.0, zp + 1, 1.1) for x, y in ph] + slots + bores + tubes + wires
             + [grown(f"arm {i}", 1.0) for i in (1, 2, 3, 4)]
             + [arm_env(a_) for a_ in (45, -45, 135, -135)])                                   # the arm session's root envelopes (u >= 58, |v| <= 21)
    # core v3 round 3: deep side webs from the tower back to the mid feet (60, ±17), (65, ±22.5) beside the bays (their columns stay
    # open for placing the UBEC and buck straight down; the arm root envelopes and grown arms trim the outer side)
    web_env = [box(58.0, 92.0, sy * 18.0, sy * 34.0, 0, 30.0) for sy in (1, -1)]
    env = U([box(-76, 100, -42, 42, 0, T), hub_env] + web_env)
    return drawn, cuts, env, feet, ph


def hub_deck_setup(h):
    drawn, cuts, env, feet, ph = hub_deck_bodies()
    p = Part("overseer_hub_deck", h, drawn, cuts, env, build="+z", material="PETG")
    foot_sup = [cyl(x, y, -0.5, 3.0, FOOT_R + 0.5) for x, y in feet]
    p.support("default", U(foot_sup))
    pi_c = M[PI_NM].centroid
    posts = [np.array([x, y, C["z_pi"]]) for x, y in ph]
    post_reg = [cyl(x, y, C["z_pi"] - 3.0, C["z_pi"] + 0.5, 4.0) for x, y in ph]
    ub, bk = M["UBEC 5 V"], M["buck 12 V"]
    bay = lambda m: box(m.bounds[0][0], m.bounds[1][0], m.bounds[0][1], m.bounds[1][1], 0, 6.5)
    walls = {sy: along(*keel_axis(sy), SK["off"], SK["off"] + SK["l"], SK["d"] / 2 + 1.0) for sy in (1, -1)}   # the bore wall (epoxy)

    walls = {k: along(*root_axis(k), 0.0 if k.startswith("L2") else SK["off"], (BS["l"] + BS["lead"]) if k.startswith("L2") else SK["off"] + SK["l"],
                      (BS["d"] if k.startswith("L2") else SK["d"]) / 2 + 1.0) for k in ROOTS}       # the bore walls (epoxy)

    def payload(name, a, cls, roots=None):
        f = rigid_split(posts, -55 * G * a, pi_c)
        reg = [(post_reg[i], f[i]) for i in range(4)] + [(bay(ub), -15 * G * a), (bay(bk), -6 * G * a)]
        for k, n_ in (roots or {}).items():
            reg.append((walls[k], root_axis(k)[1] * n_))            # + tension pulls the socket toward the head
        p.load_multi(name, reg, cls=cls, accel=-a)
    rc = RF["cases"]
    for k, d in enumerate(DIRS14):
        nm = next(c for c in rc if c.startswith(f"crash {k} "))
        payload(f"{nm} (35 g + the keel / brace forces, F)", 35 * d, "crash", rc[nm])
    if not LEAN:
        payload("flight 4 g down (+ the keel / brace forces, F)", FLIGHT["flight 4 g"], "flight", rc["flight 4 g down"])
        for nm in ("flight 2 g x", "flight 2 g y", "landing 6 g"):
            payload(nm, FLIGHT[nm], "flight")
    usb = rigid_split(posts, np.array([20.0, 0, 0]), (C["pi_x"][0] - 2, 0, C["z_pi"] + 9))           # plug pushed in at the aft ports
    p.load_multi("USB plug 20 N into the Pi's aft ports", [(post_reg[i], usb[i]) for i in range(4)], cls="misuse")
    p.load_multi("press 50 N on the Pi (assembly)", [(post_reg[i], np.array([0, 0, -12.5])) for i in range(4)], cls="misuse")
    return p


setup = hub_deck_setup
