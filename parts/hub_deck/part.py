"""Overseer hub deck (v12 setup: core v4, payload loads only, 11 cm3). Copied from nox/overseer/core_topo_v4.py with the
command-line handling removed; its inputs live in inputs/."""
import json, os
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
from topo import Part
from topo.part import man

SPEC = dict(h=1.0, gen_cm3=11.0, max_worst=0.5)
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
INS = R["insert"]; T = R["foot_h"]
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


def hub_deck_bodies():
    feet = [tuple(f) for f in R["feet_xy"]]
    ph = [tuple(p) for p in R["pi_holes"]]
    zp = C["z_pi"]
    # bay seats (v12, concept review 27 Oct: drawn is attachment only): 4 pads Ø12 x 6.5 under each bay's corners for its VHB tape
    # (~450 mm² a bay against ~9 N in a 35 g crash, E); v2..v11's continuous seat rails were load-carrying bars (7.6 cm³ drawn)
    ub, bk = M["UBEC 5 V"].bounds, M["buck 12 V"].bounds
    rails = [cyl(x, y, 0, T, 6.0) for b_ in (ub, bk) for x in (b_[0][0] + 6.0, b_[1][0] - 6.0) for y in (b_[0][1] + 6.0, b_[1][1] - 6.0)]
    slots = []
    # v5 (PR 148 review): a root web Ø12 x 6.0 under every Pi post (a gusset into the body; v4's bare Ø7 posts were 99.8 % in the main
    # body at the gate's 2.0 erosion, under its 99.9 % bar); 6.0 stays under the board's underside parts (z 6.5..8)
    roots = [cyl(x, y, 0, 6.0, 6.0) for x, y in ph]
    drawn = U([cyl(x, y, 0, T, FOOT_R) for x, y in feet] + [cyl(x, y, 0, zp, 3.5) for x, y in ph] + roots + rails)
    wires = [U([man(b) for b in M["wires"].split(only_watertight=False) if len(b.faces) > 3])]   # the core's wire routes, exact tubes (a hull per
    # route would be a slab across the deck); the flow lead crosses it
    plates = [man(M["top plate"])]
    bays = [box(m.bounds[0][0] - 1, m.bounds[1][0] + 1, m.bounds[0][1] - 1, m.bounds[1][1] + 1, T, Z_UNDER) for m in (M["UBEC 5 V"], M["buck 12 V"])]
    cuts = U(plates + bays + [cyl(x, y, -1, T + 1, INS[0] / 2) for x, y in feet] + [cyl(x, y, zp - 4.0, zp + 1, 1.1) for x, y in ph] + slots + wires
             + [grown(f"arm {i}", 1.0) for i in (1, 2, 3, 4)]
             + [arm_env(a_) for a_ in (45, -45, 135, -135)])                                   # the arm session's root envelopes (u >= 58, |v| <= 21)
    env = box(-76, 100, -42, 42, 0, T)
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
    bay = lambda m: U([cyl(x, y, T - 2.0, T + 0.5, 6.0) for x in (m.bounds[0][0] + 6.0, m.bounds[1][0] - 6.0) for y in (m.bounds[0][1] + 6.0, m.bounds[1][1] - 6.0)])   # its 4 pads' tops (VHB)

    def payload(name, a, cls):
        f = rigid_split(posts, -55 * G * a, pi_c)
        reg = [(post_reg[i], f[i]) for i in range(4)] + [(bay(ub), -(15 + 10) * G * a), (bay(bk), -(6 + 10) * G * a)]   # + the deck wires' 20 g, half on each bay (E)
        p.load_multi(name, reg, cls=cls, accel=-a)
    for k, d in enumerate(DIRS14):
        payload(f"crash {k} (35 g)", 35 * d, "crash")
    if not LEAN:
        for nm, a in FLIGHT.items():
            payload(nm, a, "flight")
    usb = rigid_split(posts, np.array([20.0, 0, 0]), (C["pi_x"][0] - 2, 0, C["z_pi"] + 9))           # plug pushed in at the aft ports
    p.load_multi("USB plug 20 N into the Pi's aft ports", [(post_reg[i], usb[i]) for i in range(4)], cls="misuse")
    p.load_multi("press 50 N on the Pi (assembly)", [(post_reg[i], np.array([0, 0, -12.5])) for i in range(4)], cls="misuse")
    return p



setup = hub_deck_setup
