"""Overseer cage node setups, one generator for all (28 Oct, Spark: "nodes dont look like topology, sharp corners, not organic").

Drawn: ONLY the socket walls and floors (a collar of 2.0 mm round each bore, 3.0 mm floor behind it) and, on the six base nodes,
a small flat PETG landing face at the node's lowest point. Everything joining them is generated inside a compact envelope
(the sockets' box + 8 mm, cut to a ball). The tubes/rods themselves are cuts, through the whole envelope. Drawn gusset (28 Oct, v6 runs
split every collar off in the solid gate): O7 arms from the centre to each collar root and a stem to the foot, one drawn body.

Members of node n come from systems/overseer.cage.json: the cage tubes (rods) that end at n and the named links whose endpoint
is n. Socket axis = the direction from the node centre to the member's far end; the bore starts node_r_mm (10.15) from the node
centre (where the cage model ends the member), 15 deep for a 6/4 tube (bore 6.2), 10 deep for a 3 mm rod (bore 3.2), 3.0 floor.
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
NODE_R = float(CAGE["node_r_mm"])                     # 10.15: where the bores start
BASE = {0, 1, 2, 3, 4, 5}                             # the base loop, z -79.9
WALL, FLOOR, MARGIN = 2.0, 3.0, 8.0
ARM_R = 2.75                                          # drawn gusset: O5.5 arms from the node centre to every collar root, one body (a 2 mm gate erosion leaves a O1.5 core;
                                                      # O7 made the arms' junction a 9 mm core, over the audit's 8 mm block limit)
KIND = {"tube": dict(bore_r=3.1, depth=15.0), "link": dict(bore_r=1.6, depth=10.0)}   # bore 6.2 / 3.2
LAND_G, DRONE_G = 6.0, 3405.0                         # landing case 6 g; drone mass 3405 g (E, layout)
LAND_FEET = 3                                         # a flat landing may rest on 3 of the 6 feet (E)
FOOT_R, FOOT_T = 8.0, 2.4                             # landing face: Ø16 x 2.4, bottom 2.0 under the node sphere's lowest point
GN = 9.81e-3


def members(n):
    """sockets of node n: dicts name, kind, far (far end point), u (unit, node -> far end), bore_r, depth, axial, side."""
    c = NODES[n]
    out = []
    for a, b in CAGE["rods"]:
        if n in (a, b):
            o = b if a == n else a
            out.append(dict(name=f"tube {a}-{b}", kind="tube", far=NODES[o], other=o))
    for nm, (p, q) in CAGE["links"].items():
        for e, f in ((p, q), (q, p)):
            if np.linalg.norm(np.array(e) - c) < 0.5:
                out.append(dict(name=nm, kind="link", far=np.array(f, float), other=None))
    loads = CAGE["node_loads"].get(str(n), {})
    for m in out:
        d = m["far"] - c
        m["len"] = float(np.linalg.norm(d)); m["u"] = d / m["len"]
        m.update(KIND[m["kind"]])
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
            back = [k for k in members(m["other"]) if k["name"] == m["name"]]
            if len(back) != 1:
                bad.append(f"{m['name']}: no socket at node {m['other']}")
                continue
            if not np.allclose(back[0]["u"], -m["u"], atol=1e-9):
                bad.append(f"{m['name']}: the two sockets are not on one axis")
            if m["len"] - 2 * NODE_R < 2 * m["depth"] - 1e-6:
                log(f"  note: {m['name']} is {m['len']:.0f} long: its two {m['depth']} mm bores nearly meet")
    ang = [(np.degrees(np.arccos(np.clip(a["u"] @ b["u"], -1, 1))), a["name"], b["name"]) for i, a in enumerate(ms) for b in ms[i + 1:]]
    for t, a, b in ang:
        if 2 * NODE_R * np.sin(np.radians(t) / 2) < 6.2:
            log(f"  WARNING {a} / {b}: {t:.0f} deg apart, their bores overlap at the node (the members would touch)")
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
        for s in (NODE_R + 0.5, NODE_R + m["depth"] / 2, NODE_R + m["depth"] - 0.5):
            if not inside(c + m["u"] * s):
                raise ValueError(f"node {n}: {m['name']} bore not cut at {s:.1f} mm from the node centre")
        if inside(c + m["u"] * (NODE_R - FLOOR / 2)):
            raise ValueError(f"node {n}: {m['name']} floor is cut")
    log(f"  members check: {len(ms)} sockets, every member lands in a bore of node {n} (bore from {NODE_R} mm)")


def build(n, h, log=print):
    ms = check_members(n, log)
    c = NODES[n]
    base = n in BASE
    # ---- drawn: collars (wall + floor) and the landing face; cuts: bores, and a keep-out round each member beyond its collar
    collars, bores, keep = [], [], []
    pts = []
    for m in ms:
        ro = m["bore_r"] + WALL
        s_end = NODE_R + m["depth"]
        collars.append(along(c, m["u"], NODE_R - FLOOR, s_end, ro))
        bores.append(along(c, m["u"], NODE_R, s_end + 1.0, m["bore_r"]))
        keep.append(along(c, m["u"], s_end, s_end + 60.0, m["bore_r"] + 0.6))     # the member itself, and room to wrap it: no growth
        m["ro"], m["s_end"] = ro, s_end
        for s in (NODE_R - FLOOR, s_end):
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                e1 = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
                e2 = np.cross(m["u"], e1)
                pts.append(c + m["u"] * s + ro * (np.cos(a) * e1 + np.sin(a) * e2))
    pts = np.array(pts)
    lo, hi = pts.min(0) - MARGIN, pts.max(0) + MARGIN
    arms = [along(c, m["u"], 0.0, NODE_R - FLOOR + 0.5, ARM_R) for m in ms]       # gusset: every collar joined to the centre
    foot = None
    if base:
        zb = c[2] - NODE_R - WALL                                # landing face bottom (z -92.05 on the base loop)
        foot = along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb, zb + FOOT_T, FOOT_R)
        arms.append(along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb + FOOT_T - 0.5, c[2], ARM_R))   # stem to the foot
        lo[2] = min(lo[2], zb - 0.5)
    import manifold3d as mf
    box = mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist())
    ball = mf.Manifold.sphere(max(np.linalg.norm(pts - c, axis=1)) + MARGIN * 0.75, 48).translate(c.tolist())
    env = box ^ ball
    drawn = U(collars + arms + ([foot] if base else []))
    cuts = U(bores + keep)
    verify_cuts(n, ms, cuts, log)
    p = Part(f"overseer_cage_node_{n}", h, drawn, cuts, env, build="+z", material="PETG")
    # ---- supports: the sockets of the two tubes with the largest frame-model axial force
    tubes = sorted([m for m in ms if m["kind"] == "tube"], key=lambda m: -m["axial"])
    sup = tubes[:2]
    lining = lambda m, grow=1.0: along(c, m["u"], NODE_R - FLOOR - 0.5, m["s_end"], m["bore_r"] + grow)   # bore wall + floor
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
    return p


def make(n):
    return lambda h: build(n, h)
