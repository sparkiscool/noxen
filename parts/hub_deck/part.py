"""Overseer hub deck (v13 setup: core v4 payloads + the nose node, 14 cm3). Copied from nox/overseer/core_topo_v4.py with the
command-line handling removed; its inputs live in inputs/.
v13 (29 Oct, Spark: the two front core ties floated in the air at the deleted tower's point (114, 0, 20); they belong to the
electronics mount, and the rods may grow): the deck's nose carries a node for core tie front.1 (to cage node 7) and front.2 (to
node 17, printed into the head mount, so that rod keeps its line). Node convention as parts/cage_node_lib.py: the link's end point
in systems/overseer.cage.json is the virtual node centre, the rod stops NODE_R short of it, the coupon socket (bore 3.2 x 10, resin
grooves, lead-in, crown) runs from there toward the cage node, a 2.5 wall and 3.0 floor drawn round it. Loads: the frame model's
member forces per crash case (systems/overseer.cage_loads.json), each added to the deck's own payload case of the same direction,
plus a pull-out at the largest axial force per tie.
v14 (29 Oct): v13 FAILED at the nose, index 2.1 across layers at the front.1 collar on the left front foot in the +x crash, 1.3 at the
front.2 collar under its 411 N pull-out, and a drawn body cut off at the gate: 14 cm3 spread over the whole deck left the nose thin.
Nose collars 4.0 thick (NOSE_WALL; the 2.5 of the cage nodes sits in 3..7 cm3 of generated material each), the nose room back to
x 78 (the UBEC bay ends at 82) and out to |y| 26 over both front feet, gen 22 (the v12 deck at 11 ran at index 0.17), h 1.3 for the cells."""
import json, os
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "inputs")
from topo import Part
from topo.part import man
from parts.cage_node_lib import CAGE, socket_cuts, collar_shape, KIND, WALL, FLOOR, NODE_R

SPEC = dict(h=1.3, gen_cm3=22.0, max_worst=0.5)    # v12: h 1.0, 11 cm3; v13: the nose node lifts the grid to z 24 (h 1.0 would be 350k cells, 1.2 307k, limit 300k) and adds its collars + bracing (E)
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


# ================================================================ nose node (v13): the two front core ties end in the deck
NOSE_WALL = 4.0                                                            # v14 (29 Oct): 2.5 failed at 411 N pull-out (index 1.3 across layers)
NOSE_ENV = (78.0, 104.0, -26.0, 26.0, 0.0, 26.0)                           # x 104: the whole envelope must stay under the 180 plate (v14 at 105 + -76 = 182, refused)                           # room for the collars and their bracing to the front feet (96, ±17)
TIE_CASES = json.load(open(os.path.join(HERE, "..", "..", "systems", "overseer.cage_loads.json")))["cases"]


def nose_sockets():
    """the deck's sockets: dicts name, c (virtual node centre = the link's hub end), u (unit toward the cage node), s0, depth, bore_r,
    lining (the load region: bore wall + floor), axial (largest |N| over the frame cases)"""
    out = []
    for nm, (p, q) in CAGE["links"].items():
        if not nm.startswith("core tie front"):
            continue
        c, q = np.array(p, float), np.array(q, float); u = (q - c) / np.linalg.norm(q - c)
        k = KIND["link"]
        out.append(dict(name=nm, c=c, u=u, s0=NODE_R, depth=k["depth"], bore_r=k["bore_r"],
                        lining=along(c, u, NODE_R - FLOOR - 0.5, NODE_R + k["depth"], k["bore_r"] + 1.5),
                        axial=max(abs(cs["members"][nm]["N"]) for cs in TIE_CASES)))
    assert len(out) == 2, [s_["name"] for s_ in out]
    # the two bores clear each other by WALL at their grooves (cage_node_lib's rule), measured between the two bore axes
    a, b = out
    sa = [a["c"] + a["u"] * t for t in np.linspace(a["s0"] - FLOOR, a["s0"] + a["depth"], 30)]
    sb = [b["c"] + b["u"] * t for t in np.linspace(b["s0"] - FLOOR, b["s0"] + b["depth"], 30)]
    dmin = min(np.linalg.norm(x - y) for x in sa for y in sb)
    need = a["bore_r"] + b["bore_r"] + 2 * 0.5 + NOSE_WALL
    assert dmin >= need, f"nose sockets {dmin:.1f} mm apart, need {need:.1f}"
    return out


def nose_bodies(socks):
    """drawn collars, cuts (bores and the rod keep-out beyond the mouth) and the envelope box of the nose node"""
    collars, cuts = [], []
    for s_ in socks:
        s_end = s_["s0"] + s_["depth"]
        collars.append(collar_shape(s_["c"], s_["u"], s_["s0"] - FLOOR, s_end, s_["bore_r"], NOSE_WALL))
        cuts += socket_cuts(s_["c"], s_["u"], s_["s0"], s_["depth"], s_["bore_r"])
        cuts.append(along(s_["c"], s_["u"], s_end, s_end + 60.0, s_["bore_r"] + 0.6))        # the rod: nothing grows on it
    return collars, cuts, box(*NOSE_ENV)


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
    socks = nose_sockets()
    collars, nose_cuts, nose_env = nose_bodies(socks)
    drawn = U([cyl(x, y, 0, T, FOOT_R) for x, y in feet] + [cyl(x, y, 0, zp, 3.5) for x, y in ph] + roots + rails + collars)
    wires = [U([man(b) for b in M["wires"].split(only_watertight=False) if len(b.faces) > 3])]   # the core's wire routes, exact tubes (a hull per
    # route would be a slab across the deck); the flow lead crosses it
    plates = [man(M["top plate"])]
    bays = [box(m.bounds[0][0] - 1, m.bounds[1][0] + 1, m.bounds[0][1] - 1, m.bounds[1][1] + 1, T, Z_UNDER) for m in (M["UBEC 5 V"], M["buck 12 V"])]
    # the insert holes are cut 2.5 above the feet (v13): the nose node may grow over the front feet, and the M3 x 8 from below ends
    # up to 0.2 above a foot
    cuts = U(plates + bays + [cyl(x, y, -1, T + 2.5, INS[0] / 2) for x, y in feet] + [cyl(x, y, zp - 4.0, zp + 1, 1.1) for x, y in ph] + slots + wires + nose_cuts
             + [grown(f"arm {i}", 1.0) for i in (1, 2, 3, 4)]
             + [arm_env(a_) for a_ in (45, -45, 135, -135)])                                   # the arm session's root envelopes (u >= 58, |v| <= 21)
    env = U([box(-75, 100, -42, 42, 0, T), nose_env])                   # -75 (v14; was -76): 179 long with the nose
    # every nose bore is open and no floor is cut (the wire routes cross the deck): as cage_node_lib.verify_cuts
    for s_ in socks:
        for t in (s_["s0"] + 0.5, s_["s0"] + s_["depth"] - 0.5):
            probe = mf.Manifold.sphere(0.25, 8).translate((s_["c"] + s_["u"] * t).tolist())
            assert (cuts ^ probe).volume() > 0.9 * probe.volume(), f"{s_['name']}: bore not cut at {t:.1f}"
        probe = mf.Manifold.sphere(0.25, 8).translate((s_["c"] + s_["u"] * (s_["s0"] - FLOOR / 2)).tolist())
        assert (cuts ^ probe).volume() < 0.1 * probe.volume(), f"{s_['name']}: the socket floor is cut (a wire route?)"
    return drawn, cuts, env, feet, ph, socks


def hub_deck_setup(h):
    drawn, cuts, env, feet, ph, socks = hub_deck_bodies()
    p = Part("overseer_hub_deck", h, drawn, cuts, env, build="+z", material="PETG")
    foot_sup = [cyl(x, y, -0.5, 3.0, FOOT_R + 0.5) for x, y in feet]
    p.support("default", U(foot_sup))
    pi_c = M[PI_NM].centroid
    posts = [np.array([x, y, C["z_pi"]]) for x, y in ph]
    post_reg = [cyl(x, y, C["z_pi"] - 3.0, C["z_pi"] + 0.5, 4.0) for x, y in ph]
    ub, bk = M["UBEC 5 V"], M["buck 12 V"]
    bay = lambda m: U([cyl(x, y, T - 2.0, T + 0.5, 6.0) for x in (m.bounds[0][0] + 6.0, m.bounds[1][0] - 6.0) for y in (m.bounds[0][1] + 6.0, m.bounds[1][1] - 6.0)])   # its 4 pads' tops (VHB)

    def tie_forces(case):
        """the frame model's forces on the two nose sockets in one of its cases: N along the rod toward the cage node (tension pulls
        the deck toward the node, compression pushes into the floor), the small end shear across it"""
        out = []
        for s_ in socks:
            m = case["members"][s_["name"]]
            side = np.cross(s_["u"], [0, 0, 1.0]); side /= np.linalg.norm(side)
            out.append((s_["lining"], m["N"] * s_["u"] + m["shear_j"] * side))
        return out

    def frame_case(d):
        """the frame model case decelerating along d, or None (the model ran the 11 hull faces the CG ray leaves through)"""
        for cs in TIE_CASES:
            if cs["name"].startswith("crash") and np.allclose(cs["d"], d / np.linalg.norm(d), atol=0.02):
                return cs
        return None

    def payload(name, a, cls, extra=()):
        f = rigid_split(posts, -55 * G * a, pi_c)
        reg = [(post_reg[i], f[i]) for i in range(4)] + [(bay(ub), -(15 + 10) * G * a), (bay(bk), -(6 + 10) * G * a)]   # + the deck wires' 20 g, half on each bay (E)
        p.load_multi(name, reg + list(extra), cls=cls, accel=-a)
    for k, d in enumerate(DIRS14):
        cs = frame_case(d)
        payload(f"crash {k} (35 g)" + (" + frame ties" if cs else ""), 35 * d, "crash", tie_forces(cs) if cs else ())
    if not LEAN:
        land = [cs for cs in TIE_CASES if cs["name"].startswith("landing")]
        for nm, a in FLIGHT.items():
            payload(nm, a, "flight", tie_forces(land[0]) if (nm.startswith("landing") and land) else ())
    for s_ in socks:                                                                       # the glue line: pull-out at the largest axial force
        p.load_multi(f"frame {s_['name']} axial {s_['axial']:.0f} N (pull-out)", [(s_["lining"], s_["axial"] * s_["u"])], cls="crash")
    usb = rigid_split(posts, np.array([20.0, 0, 0]), (C["pi_x"][0] - 2, 0, C["z_pi"] + 9))           # plug pushed in at the aft ports
    p.load_multi("USB plug 20 N into the Pi's aft ports", [(post_reg[i], usb[i]) for i in range(4)], cls="misuse")
    p.load_multi("press 50 N on the Pi (assembly)", [(post_reg[i], np.array([0, 0, -12.5])) for i in range(4)], cls="misuse")
    return p



setup = hub_deck_setup
