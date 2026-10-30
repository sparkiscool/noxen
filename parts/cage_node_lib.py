"""Overseer cage node setups, one generator for all (28 Oct, Spark: "nodes dont look like topology, sharp corners, not organic").

Drawn: ONLY the socket walls and floors (a collar of 2.0 mm round each bore, 3.0 mm floor behind it) and, on the six base nodes,
a small flat PETG landing face at the node's lowest point. Everything joining them is generated inside a compact envelope
(the sockets' box + 8 mm, cut to a ball). The tubes/rods themselves are cuts, through the whole envelope. (28 Oct: drawn arms from the
centre to every collar were added when the solid gate "split every collar off", and removed when that turned out to be voxel dust in the gate.)

Members of node n come from systems/overseer.cage.json: the cage tubes (rods) that end at n and the named links whose endpoint
is n. Socket axis = the direction from the node centre to the member's far end; the bore starts node_r_mm (10.15) from the node
centre (where the cage model ends the member), 15 deep for a 6/4 tube (bore 6.2), 10 deep for a 3 mm rod (bore 3.2), 3.0 floor;
every bore as the coupon socket (socket_cuts: resin grooves, lead-in, teardrop crown toward the build direction; 29 Oct).
`check_members` fails the setup when a member of the node has no socket, the far node lacks the matching socket on the same
axis, the frame model has loads for a member that is not there (or none for one that is), or the built cuts do not hold the bore.

Loads (drone frame, N; frame model F = node_loads, the 35 g crash set of runs/overseer_frame/v1):
  per member, its axial force as a pull-out along the socket axis and its side force across it as separate cases ("crash"),
  the longest tube pushed into its socket as a combined axial + side crash case, and on a base node landing 6 g on the face.
Supports: the sockets of the two tube members with the largest frame-model axial force (the stiffest load paths: the ones
the rest of the cage holds the node by); those sockets' own cases are not applied (a force on a fixed region is nothing).
"""
import json, os
import numpy as np
from topo import Part
from topo.part import man

HERE = os.path.dirname(os.path.abspath(__file__))
CAGE = json.load(open(os.path.join(HERE, "..", "systems", "overseer.cage.json")))
NODES = np.array(CAGE["nodes"], float)
NODE_R = float(CAGE["node_r_mm"])                     # 10.15: where a bore starts unless a neighbour forces it further out (SOCKET_START)
BASE = {0, 1, 2, 3, 4, 5}                             # the base loop, z -79.9
WALL, FLOOR, MARGIN = 2.5, 3.0, 8.0                   # wall 2.5 (29 Oct, was 2.0): 2.0 is left at the resin grooves
KIND = {"tube": dict(bore_r=3.1, depth=15.0), "link": dict(bore_r=1.6, depth=10.0)}   # bore 6.2 / 3.2
GROOVE_D, GROOVE_H, LEAD_IN = 0.5, 1.5, 1.0            # the coupon socket (socket_cuts below): resin grooves, lead-in
CROWN_MIN_SIN = 0.3                                    # a bore within ~17 deg of vertical prints without a crown
LAND_G, DRONE_G = 6.0, 3405.0                         # landing case 6 g; drone mass 3405 g (E, layout)
LAND_FEET = 3                                         # a flat landing may rest on 3 of the 6 feet (E)
FOOT_R, FOOT_T = 8.0, 2.4                             # landing face: Ø16 x 2.4, bottom 2.0 under the node sphere's lowest point
GN = 9.81e-3


def _raw_members(n):
    c = NODES[n]
    out = []
    for a, b in CAGE["rods"]:
        if n in (a, b):
            o = b if a == n else a
            out.append(dict(name=f"tube {a}-{b}", kind="tube", far=NODES[o], other=o))
    for nm, (p, q) in CAGE["links"].items():
        for e, f in ((p, q), (q, p)):
            if np.linalg.norm(np.array(e) - c) < 0.5:
                d = np.linalg.norm(NODES - np.array(f, float), axis=1); o = int(np.argmin(d)) if d.min() < 0.6 else None
                # head rods +1 / -1 are 6/4 tubes (dimensions.md Sensor head supports, Spark 27 Oct; the head mount bores them
                # 6.2 x 15): nodes 15 / 16 bored them 3.2 as rods until 29 Oct
                out.append(dict(name=nm, kind="tube" if nm.startswith("head rod") else "link", far=np.array(f, float), other=o))
    for m in out:
        d = m["far"] - c
        m["len"] = float(np.linalg.norm(d)); m["u"] = d / m["len"]
        m.update(KIND[m["kind"]])
    return out


def _socket_starts():
    """where each socket's bore starts, per (node, member): NODE_R, or further out where two members meet at a small angle, so
    that every bore clears its neighbours by WALL (Spark 29 Oct: no carbon touching carbon inside a node; the v5/v6 nodes had
    merged bores at every node from 0 to 19, members 20..36 deg apart). The rod is cut shorter by the same amount."""
    st = {}
    for n in range(len(NODES)):
        ms = _raw_members(n)
        for m in ms:
            s0 = NODE_R
            for o in ms:
                if o is m:
                    continue
                t = float(np.arccos(np.clip(m["u"] @ o["u"], -1.0, 1.0)))
                if t < 1e-6:
                    continue
                s0 = max(s0, (m["bore_r"] + o["bore_r"] + 2 * GROOVE_D + WALL) / (2.0 * np.sin(t / 2.0)))   # at the grooves too
            st[(n, m["name"])] = float(np.ceil(s0 * 10.0) / 10.0)          # up to the next 0.1 mm: the check below is exact
    return st


SOCKET_START = _socket_starts()


def members(n):
    """sockets of node n: dicts name, kind, far (far end point), u (unit, node -> far end), bore_r, depth, axial, side, s0 (where the
    bore starts, mm from the centre), s0_far (the same at the far end, a cage node)."""
    out = _raw_members(n)
    loads = CAGE["node_loads"].get(str(n), {})
    for m in out:
        m["s0"] = SOCKET_START[(n, m["name"])]
        m["s0_far"] = SOCKET_START.get((m["other"], m["name"]), NODE_R)
        m["axial"] = loads.get(m["name"], {}).get("axial")
        m["side"] = loads.get(m["name"], {}).get("side")
    return out


def along(c, u, s0, s1, r, n=40):
    """cylinder of radius r on the axis c + u s, s in s0..s1 (exact manifold)"""
    import trimesh
    return man(trimesh.creation.cylinder(r, s1 - s0, sections=n, segment=np.vstack([c + u * s0, c + u * s1])))


def U(ms):
    import manifold3d as mf
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)


# ---- the epoxied socket, as the coupons that fit (parts/rod_coupon_6mm, rod_coupon_3mm; Spark 29 Oct: "the one I said worked in
# the coupon", with the resin reservoir): the bore (6.2 / 3.2), two resin grooves 0.5 deep x 1.5 tall at a third and two thirds
# of the depth, a 1.0 x 45 deg lead-in at the mouth; and, for printing, a teardrop crown on the overhanging side of every bore
# that is not near-vertical on the build plate, so no support is ever printed inside a hole (Spark 29 Oct). The coupon's vent
# hole through the floor is not drawn: in a node every floor backs onto the body; a notch filed in the rod end vents instead.


def _prism(poly, c, u, s0, s1, e1, e2):
    """the 2D polygon poly (in the e1, e2 plane) extruded along u from s0 to s1"""
    import manifold3d as mf
    cs = mf.CrossSection([[(float(x), float(y)) for x, y in poly]])
    m = mf.Manifold.extrude(cs, float(s1 - s0))                       # along local z from 0 to s1 - s0
    R = np.column_stack([e1, e2, u]); t = c + u * s0
    return m.transform(np.column_stack([R, t]).astype(np.float64))


def socket_cuts(c, u, s0, depth, r, build=(0.0, 0.0, 1.0), beyond=1.0):
    """cuts of one socket whose bore bottom is at c + u s0 and mouth at c + u (s0 + depth), bore radius r: the bore (to beyond past
    the mouth), two resin grooves, the lead-in cone, and the teardrop crown toward the build direction where the bore overhangs"""
    u = np.asarray(u, float) / np.linalg.norm(u); z = np.asarray(build, float)
    s_end = s0 + depth
    cuts = [along(c, u, s0, s_end + beyond, r)]
    for f in (1.0 / 3.0, 2.0 / 3.0):
        sg = s0 + f * depth - 0.5 * GROOVE_H
        cuts.append(along(c, u, sg, sg + GROOVE_H, r + GROOVE_D))
    cuts.append(cone(c, u, s_end - LEAD_IN, s_end + beyond, r, r + LEAD_IN + beyond))
    d = crown_dir(u, build)
    if d is not None:                                                  # overhanging bore: a teardrop, apex r*sqrt(2) up the overhang side
        e1 = d; e2 = np.cross(u, e1)
        a = np.linspace(np.pi / 4, 2 * np.pi - np.pi / 4, 40)        # the circle from +45 deg round the bottom to -45 deg
        poly = [(r * np.cos(t), r * np.sin(t)) for t in a] + [(r * np.sqrt(2.0), 0.0)]
        cuts.append(_prism(poly, c, u, s0, s_end + beyond, e1, e2))
    return cuts


def crown_dir(u, build=(0.0, 0.0, 1.0)):
    """unit vector of the teardrop crown for a bore along u, or None when the bore is near-vertical on the plate"""
    u = np.asarray(u, float) / np.linalg.norm(u); z = np.asarray(build, float); d = z - (z @ u) * u
    return None if np.linalg.norm(d) <= CROWN_MIN_SIN else d / np.linalg.norm(d)


def collar_shape(c, u, s0, s1, r, wall, build=(0.0, 0.0, 1.0)):
    """the drawn collar round a socket: a cylinder of r + wall, grown to the same teardrop as the bore's crown (+ wall) where the
    bore has one, so the wall is kept all round the crown too"""
    u = np.asarray(u, float) / np.linalg.norm(u); ro = r + wall
    d = crown_dir(u, build)
    if d is None:
        return along(c, u, s0, s1, ro)
    e1 = d; e2 = np.cross(u, e1)
    a = np.linspace(np.pi / 4, 2 * np.pi - np.pi / 4, 40)
    poly = [(ro * np.cos(t), ro * np.sin(t)) for t in a] + [(ro * np.sqrt(2.0), 0.0)]
    return _prism(poly, c, u, s0, s1, e1, e2)


def cone(c, u, s0, s1, r0, r1, n=40):
    """frustum on the axis c + u s from radius r0 at s0 to r1 at s1"""
    import manifold3d as mf
    u = np.asarray(u, float) / np.linalg.norm(u)
    m = mf.Manifold.cylinder(float(s1 - s0), float(r0), float(r1), n)
    z = np.array([0, 0, 1.0]); v = np.cross(z, u); sn = np.linalg.norm(v); cs = float(z @ u)
    if sn < 1e-9:
        R = np.eye(3) if cs > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        v /= sn; K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); R = np.eye(3) + sn * K + (1 - cs) * K @ K
    return m.transform(np.column_stack([R, c + u * s0]).astype(np.float64))


def check_members(n, log=print):
    """every member of node n lands in a socket of n: same axis as the member, bore from node_r_mm out, and the member's far
    end has the matching socket on the same line (tubes). Raises ValueError listing what is wrong."""
    c, ms, bad = NODES[n], members(n), []
    names = {m["name"] for m in ms}
    loads = set(CAGE["node_loads"].get(str(n), {}))
    if names != loads:
        bad.append(f"frame model members differ: no load for {sorted(names - loads)}, no member for {sorted(loads - names)}")
    for m in ms:
        if m["kind"] == "tube":
            if m["other"] is None:                    # the far end is a printed part, not a node (head rods +1 / -1 end in the
                continue                              # head mount, which bores them itself); 29 Oct: node 15 crashed here
            back = [k for k in members(m["other"]) if k["name"] == m["name"]]
            if len(back) != 1:
                bad.append(f"{m['name']}: no socket at node {m['other']}")
                continue
            if not np.allclose(back[0]["u"], -m["u"], atol=1e-9):
                bad.append(f"{m['name']}: the two sockets are not on one axis")
            if m["len"] - m["s0"] - m["s0_far"] < 2 * m["depth"] - 1e-6:
                log(f"  note: {m['name']} is {m['len']:.0f} long: its two {m['depth']} mm bores nearly meet")
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            t = np.arccos(np.clip(a["u"] @ b["u"], -1, 1)); r = max(a["s0"], b["s0"])
            if 2 * r * np.sin(t / 2) < a["bore_r"] + b["bore_r"] + 2 * GROOVE_D + WALL - 1e-6:
                bad.append(f"{a['name']} / {b['name']}: {np.degrees(t):.0f} deg apart, bores overlap at {r:.1f} mm (carbon would touch carbon)")
    far = [f"{m['name']} from {m['s0']:.1f}" for m in ms if m["s0"] > NODE_R + 1e-6]
    if far:
        log(f"  sockets moved out so no bores overlap: {', '.join(far)} mm (others from {NODE_R})")
    if bad:
        raise ValueError(f"node {n}: " + "; ".join(bad))
    return ms


def verify_cuts(n, ms, cuts, log=print):
    """the built cuts hold every bore: sample points along each socket axis are cut, the floor behind is not"""
    import manifold3d as mf
    c = NODES[n]
    def inside(p):
        t = mf.Manifold.sphere(0.25, 8).translate(p.tolist())
        return (cuts ^ t).volume() > 0.9 * t.volume()
    for m in ms:
        for s in (m["s0"] + 0.5, m["s0"] + m["depth"] / 2, m["s0"] + m["depth"] - 0.5):
            if not inside(c + m["u"] * s):
                raise ValueError(f"node {n}: {m['name']} bore not cut at {s:.1f} mm from the node centre")
        if inside(c + m["u"] * (m["s0"] - FLOOR / 2)):
            raise ValueError(f"node {n}: {m['name']} floor is cut")
    log(f"  members check: {len(ms)} sockets, every member lands in a bore of node {n}")


def build(n, h, log=print, extra=None):
    """extra(c, ms) -> dict(drawn=[...], cuts=[...], env=[...], loads=fn(p)): what a node carries besides its sockets (node 21: the camera)"""
    ms = check_members(n, log)
    ex = extra(NODES[n], ms) if extra else {}
    c = NODES[n]
    base = n in BASE
    # ---- drawn: collars (wall + floor) and the landing face; cuts: bores, and a keep-out round each member beyond its collar
    collars, bores, keep = [], [], []
    pts = []
    for m in ms:
        ro = m["bore_r"] + WALL
        s_end = m["s0"] + m["depth"]
        collars.append(collar_shape(c, m["u"], m["s0"] - FLOOR, s_end, m["bore_r"], WALL))
        bores += socket_cuts(c, m["u"], m["s0"], m["depth"], m["bore_r"])     # bore, resin grooves, lead-in, teardrop crown
        keep.append(along(c, m["u"], s_end, s_end + 60.0, m["bore_r"] + 0.6))     # the member itself, and room to wrap it: no growth
        m["ro"], m["s_end"] = ro, s_end
        for s in (m["s0"] - FLOOR, s_end):
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                e1 = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
                e2 = np.cross(m["u"], e1)
                pts.append(c + m["u"] * s + ro * (np.cos(a) * e1 + np.sin(a) * e2))
    pts = np.array(pts)
    lo, hi = pts.min(0) - MARGIN, pts.max(0) + MARGIN
    foot = None
    if base:
        zb = c[2] - NODE_R - WALL                                # landing face bottom (z -92.05 on the base loop)
        foot = along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb, zb + FOOT_T, FOOT_R)
        lo[2] = min(lo[2], zb - 0.5)
    import manifold3d as mf
    box = mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist())
    ball = mf.Manifold.sphere(max(np.linalg.norm(pts - c, axis=1)) + MARGIN * 0.75, 48).translate(c.tolist())
    env = box ^ ball
    drawn = U(collars + ([foot] if base else []) + ex.get("drawn", []))
    cuts = U(bores + keep + ex.get("cuts", []))
    if ex.get("env"):
        env = U([env] + ex["env"])
    verify_cuts(n, ms, cuts, log)
    p = Part(f"overseer_cage_node_{n}", h, drawn, cuts, env, build="+z", material="PETG")
    # ---- supports: the sockets of the two tubes with the largest frame-model axial force
    tubes = sorted([m for m in ms if m["kind"] == "tube"] or ms, key=lambda m: -m["axial"])     # a node on rods only (21): its rods hold it
    sup = tubes[:2]
    lining = lambda m, grow=1.0: along(c, m["u"], m["s0"] - FLOOR - 0.5, m["s_end"], m["bore_r"] + grow)   # bore wall + floor
    p.support("default", U([lining(m, 1.4) for m in sup]))
    held = {m["name"] for m in sup}
    log(f"  supports: {sorted(held)}")
    # ---- loads
    def push(m, sign, f):                                      # sign +1 pull-out along u, -1 push into the socket
        return (lining(m), sign * f * m["u"])
    for m in ms:
        if m["name"] in held:
            continue
        p.load_multi(f"frame {m['name']} axial {m['axial']:.0f} N (pull-out)", [push(m, 1, m["axial"])], cls="crash")
        side = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); side /= np.linalg.norm(side)
        p.load_multi(f"frame {m['name']} side {m['side']:.1f} N", [(lining(m), m["side"] * side)], cls="crash")
    L = max(tubes, key=lambda m: m["len"])
    if L["name"] not in held:
        sd = np.cross(L["u"], [0, 0, 1.0] if abs(L["u"][2]) < 0.9 else [1.0, 0, 0]); sd /= np.linalg.norm(sd)
        p.load_multi(f"crash 35 g through the longest tube {L['name']} ({L['len']:.0f} mm), axial + side",
                     [(lining(L), -L["axial"] * L["u"] + L["side"] * sd)], cls="crash")
    if base:
        face = along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb - 0.5, zb + 1.0, FOOT_R + 0.5)
        fz = LAND_G * DRONE_G * GN / LAND_FEET
        p.load_multi(f"landing 6 g on the foot ({fz:.0f} N up, 3 of 6 feet)", [(face, np.array([0, 0, fz]))], cls="flight")
    if ex.get("loads"):
        ex["loads"](p)
    return p


def make(n, extra=None):
    return lambda h: build(n, h, extra=extra)
