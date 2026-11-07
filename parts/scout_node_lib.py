"""Scout cage node setups, one generator for all 25 printed nodes (4 Nov, Spark: "generate the nodes for the cage"), after the
Overseer's parts/cage_node_lib.py (28 Oct: "nodes dont look like topology, sharp corners, not organic").

Drawn: ONLY the socket walls and floors (a collar of 2.5 round each bore, 3.0 floor behind it) and, on the six skid nodes, a flat
landing face under the node (the TPU strip's seat). Everything joining them is generated inside a compact envelope (the sockets'
box + 8 mm, cut to a ball). The rods themselves are cuts through the whole envelope, and so is the shared funnel swept over the whole
descent (funnel_sweep): a skid or rail node may not grow toward the funnel's walls.

Members of node k come from tools/scout_cage.py: the cage rods (every one 3.0 solid carbon: bore 3.2, 10 deep, 3.0 floor) and the
ties that end at k. Socket axis = the direction from the node centre to the member's far end; the bore starts NODE_R (7, the
cage model's sphere) from the centre, or further out where two members meet at a small angle so every bore clears its neighbours
by WALL (the rod is cut shorter by the same amount; check_members is exact). Every bore as the coupon socket (socket_cuts: resin
grooves, lead-in, teardrop crown toward the build direction). NB and TB are printed into the nose and tail units, not here.

Loads (drone frame, N; frame model F = systems/scout.cage_loads.json node_loads, tools/scout_cage_loads.py): per member its axial
force as a pull-out along the socket axis and its side force across it as separate cases ("crash"), the longest rod pushed into its
socket as a combined case, and on a skid node landing 6 g on its face (the drone's 762 g on 3 of the 6 skid nodes, E).
Supports: the sockets of the two members with the largest frame-model axial force (the stiffest load paths).
Run: parts/scout_cage_node_<key>/part.py -> setup(h) = make("<key>")."""
import json, os, sys
import numpy as np
from topo import Part
from topo.part import man
from parts.cage_node_lib import socket_cuts, collar_shape, along, U, GROOVE_D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import scout_cage as sc                                   # noqa: E402

LOADS = json.load(open(os.path.join(ROOT, "systems", "scout.cage_loads.json")))["node_loads"]
P = {**sc.NODES, **sc.TIE_ROOTS}
KEYS = [k for k in sc.NODES if k not in sc.PRINTED]      # the 25 printed nodes, in sc.GROUPS printed structures (a pair closer than sc.MERGE_MM is one part)
BRIDGE_R = 6.0                                            # the room round a printed bridge between two merged nodes (O12, the engine keeps what the rod's load needs)
NODE_R = 7.0                                              # the cage model's node sphere: where a bore starts unless a neighbour forces it out
MAX_START = 25.0                                          # a bore pushed further out than this by a near-parallel neighbour makes the node a fork (5 Nov: the arm and yoke ties met
#   at each rail corner 2..13 deg apart, bores from 39..85 mm, nodes of 290k..586k cells; the yoke ties now end at the rail mid nodes)
WALL, FLOOR, STRUT_M = 2.0, 2.0, 2.5                        # 5 Nov (Spark: the nodes are overweight): 2.5 / 3.0 drew 119 g of collars over 25 nodes, 2.0 / 2.0 draws 83 (the 2 mm floor); STRUT_M: the room round each collar
ROD_R, ROD_GAP = 1.5, 1.0                                 # a passing rod's radius and the least gap to a collar's surface
BORE_R, DEPTH = 1.6, 10.0                                 # 3.0 rod: bore 3.2, 10 deep (coupons v2)
SKID = {"K0", "K1", "K2", "K3", "KM0", "KM1"}             # the skid nodes: a landing face under each (the TPU strip's seat)
FOOT_R, FOOT_T = 8.0, 3.0                                  # 3.0: the audit refuses drawn bodies under 2.5 (the Overseer feet are 2.4)
LAND_G, DRONE_G, LAND_FEET = 6.0, 762.0, 3
GN = 9.81e-3


def _raw_members(k, bridges=False):
    """the rods and ties that end at node k and get a socket there: the bridges (sc.BRIDGES, the rod to a node merged into the same
    printed structure) are printed, not socketed, but they still take their room in the hub (bridges=True: the socket-start rule
    counts them, so a bridged node's sockets sit where they did; 5 Nov: with P0-R0 bridged, P0-P1's bore came in from 17 to 8.5 and
    its collar's mouth landed in the K0-R0 brace rod)"""
    c = np.array(P[k], float); out = []
    for a, b in (sc.RODS if bridges else sc.RODS_CARBON) + sc.TIES:
        if k in (a, b):
            o = b if a == k else a
            out.append(dict(name=f"{a}-{b}", far=np.array(P[o], float), other=o if o in sc.NODES else None))
    for m in out:
        d = m["far"] - c; m["len"] = float(np.linalg.norm(d)); m["u"] = d / m["len"]; m["bore_r"], m["depth"] = BORE_R, DEPTH
    return out


def _seg_dist(a0, a1, b0, b1):
    """the least distance between two segments"""
    u, v = a1 - a0, b1 - b0; A, C = u @ u, v @ v
    best = 1e9
    for t in np.linspace(0, 1, 41):                         # sample one segment, project onto the other (exact enough at 1/40 of a rod)
        q = b0 + t * v; s_ = np.clip((q - a0) @ u / A, 0, 1)
        best = min(best, float(np.linalg.norm(a0 + s_ * u - q)))
    for s_ in np.linspace(0, 1, 41):
        q = a0 + s_ * u; t = np.clip((q - b0) @ v / C, 0, 1)
        best = min(best, float(np.linalg.norm(b0 + t * v - q)))
    return best


def _socket_starts():
    st = {}
    for k in sc.NODES:
        ms = _raw_members(k, bridges=True)
        for m in ms:
            s0 = NODE_R
            for o in ms:
                if o is m:
                    continue
                t = float(np.arccos(np.clip(m["u"] @ o["u"], -1.0, 1.0)))
                if t < 1e-6:
                    continue
                s0 = max(s0, (m["bore_r"] + o["bore_r"] + 2 * GROOVE_D + WALL) / (2.0 * np.sin(t / 2.0)))
            st[(k, m["name"])] = float(np.ceil(s0 * 10.0) / 10.0)
    return st


SOCKET_START = _socket_starts()


def members(k):
    out = _raw_members(k); loads = LOADS.get(k, {})
    for m in out:
        m["s0"] = SOCKET_START[(k, m["name"])]
        m["s0_far"] = SOCKET_START.get((m["other"], m["name"]), NODE_R) if m["other"] else NODE_R
        m["axial"] = loads.get(m["name"], {}).get("axial"); m["side"] = loads.get(m["name"], {}).get("side")
    return out


def check_members(k, log=print):
    c, ms, bad = np.array(P[k], float), members(k), []
    bridged = {f"{a}-{b}" for a, b in sc.BRIDGES}
    names = {m["name"] for m in ms}; loads = set(LOADS.get(k, {})) - bridged
    if names != loads:
        bad.append(f"frame model members differ: no load for {sorted(names - loads)}, no member for {sorted(loads - names)}")
    for m in ms:
        if m["other"] is None:
            continue                                      # a tie: its far end is a clip or a unit, which bores it itself
        back = [x for x in members(m["other"]) if x["name"] == m["name"]]
        if len(back) != 1:
            bad.append(f"{m['name']}: no socket at node {m['other']}"); continue
        if not np.allclose(back[0]["u"], -m["u"], atol=1e-9):
            bad.append(f"{m['name']}: the two sockets are not on one axis")
        if m["len"] - m["s0"] - m["s0_far"] < 2 * m["depth"] - 1e-6:
            log(f"  note: {m['name']} is {m['len']:.0f} long: its two {m['depth']} mm bores nearly meet")
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            t = np.arccos(np.clip(a["u"] @ b["u"], -1, 1)); r = max(a["s0"], b["s0"])
            if 2 * r * np.sin(t / 2) < a["bore_r"] + b["bore_r"] + 2 * GROOVE_D + WALL - 1e-6:
                bad.append(f"{a['name']} / {b['name']}: {np.degrees(t):.0f} deg apart, bores overlap at {r:.1f} mm")
    for m in ms:                                              # a rod that passes this node must clear every collar (5 Nov: K0-R0 through P0-P1's mouth)
        a0, a1 = c + m["u"] * (m["s0"] - FLOOR), c + m["u"] * (m["s0"] + m["depth"])
        for i, j in sc.RODS + sc.TIES:
            if k in (i, j):
                continue
            gap = _seg_dist(a0, a1, np.array(P[i], float), np.array(P[j], float)) - (m["bore_r"] + WALL) - ROD_R
            if gap < ROD_GAP:
                bad.append(f"{m['name']}: the rod {i}-{j} passes {gap:.1f} mm from its collar (need {ROD_GAP}): move the socket out or re-route the rod")
    far = [f"{m['name']} from {m['s0']:.1f}" for m in ms if m["s0"] > NODE_R + 1e-6]
    if far:
        log(f"  sockets moved out so no bores overlap: {', '.join(far)} mm (others from {NODE_R})")
    for m in ms:
        if m["s0"] > MAX_START:
            bad.append(f"{m['name']}: its bore starts {m['s0']:.0f} mm out to clear a near-parallel neighbour (limit {MAX_START:.0f}): a fork, not a node; retarget one of the two members")
    if bad:
        raise ValueError(f"node {k}: " + "; ".join(bad))
    return ms


def verify_cuts(k, ms, cuts, log=print):
    import manifold3d as mf
    c = np.array(P[k], float)
    def inside(p):
        t = mf.Manifold.sphere(0.25, 8).translate(p.tolist()); return (cuts ^ t).volume() > 0.9 * t.volume()
    for m in ms:
        for s in (m["s0"] + 0.5, m["s0"] + m["depth"] / 2, m["s0"] + m["depth"] - 0.5):
            if not inside(c + m["u"] * s):
                raise ValueError(f"node {k}: {m['name']} bore not cut at {s:.1f} mm from the node centre")
        if inside(c + m["u"] * (m["s0"] - FLOOR / 2)):
            raise ValueError(f"node {k}: {m['name']} floor is cut")
    log(f"  members check: {len(ms)} sockets, every member lands in a bore of node {k}")


_SWEEP = {}


def funnel_sweep():
    """the shared funnel as the drone sees it over the whole descent (tools/seat_check.descent_states: the capture limits, heights every
    3..5), each state also shifted +-1.5 in x and y for margin, all unioned: a keep-out no node may grow into (4 Nov: the first descent
    gate on the node placeholders had the skid nodes' boxes in the funnel's y walls at the -y capture limit)"""
    if "m" not in _SWEEP:
        import manifold3d as mf, trimesh, seat_check as skc
        fun = man(trimesh.load(os.path.join(ROOT, "parts", "box_funnel", "part.stl")))
        copies = []
        for lab, ox, oy, hh in skc.descent_states("scout"):
            for dx, dy in ((0, 0), (1.5, 0), (-1.5, 0), (0, 1.5), (0, -1.5)):
                copies.append(fun.translate([-ox + dx, -oy + dy, -hh]))
        _SWEEP["m"] = mf.Manifold.batch_boolean(copies, mf.OpType.Add)
    return _SWEEP["m"]


def build(group, h, log=print):
    """one printed structure: the node(s) of group (a key, or sc.GROUPS' tuple of keys bridged together), each with its collars, struts and
    hub, a printed bridge's room between merged nodes, a landing face under a skid node"""
    import manifold3d as mf
    group = (group,) if isinstance(group, str) else tuple(group)
    collars, bores, keep, struts, drawn_extra, allm, feet = [], [], [], [], [], [], {}
    for k in group:
        ms = check_members(k, log); c = np.array(P[k], float)
        for m in ms:
            ro = m["bore_r"] + WALL; s_end = m["s0"] + m["depth"]
            collars.append(collar_shape(c, m["u"], m["s0"] - FLOOR, s_end, m["bore_r"], WALL))
            bores += socket_cuts(c, m["u"], m["s0"], m["depth"], m["bore_r"])
            keep.append(along(c, m["u"], s_end, s_end + 60.0, m["bore_r"] + 0.6))
            m["ro"], m["s_end"], m["node"], m["c"] = ro, s_end, k, c
            allm.append(m)
        # the envelope: a strut round each member from the node's centre to its collar's mouth (STRUT_M of room round the collar) and a
        # hub ball at the centre, so the engine only decides how much of each strut to keep: a node, not a web (5 Nov, Spark: run 1's
        # nodes, grown in a ball to 56 mm across, were spindly blobs of webs between far-apart collars)
        struts += [along(c, m["u"], 0.0, m["s_end"], m["ro"] + STRUT_M) for m in ms]
        struts.append(mf.Manifold.sphere(NODE_R + WALL + STRUT_M, 48).translate(c.tolist()))
        if k in SKID:
            zb = c[2] - NODE_R - WALL
            drawn_extra.append(along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb, zb + FOOT_T, FOOT_R))
            struts.append(along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb, c[2], FOOT_R + STRUT_M))   # from the foot's face up: no room under the landing face (load_room)
            feet[k] = zb
    for a, b in sc.BRIDGES:                                                     # the printed bridge between two merged nodes: its room, the whole span
        if a in group and b in group:
            ca, cb = np.array(P[a], float), np.array(P[b], float); d = cb - ca; L = float(np.linalg.norm(d))
            struts.append(along(ca, d / L, 0.0, L, BRIDGE_R))
    env = U(struts)
    drawn = U(collars + drawn_extra); cuts = U(bores + keep + [funnel_sweep()])   # the funnel's sweep: no growth toward it
    for k in group:
        verify_cuts(k, [m for m in allm if m["node"] == k], cuts, log)
    p = Part(sc.group_id(group), h, drawn, cuts, env, build="+z", material="PETG")
    sup = sorted(allm, key=lambda m: -m["axial"])[:2]
    lining = lambda m, grow=1.0: along(m["c"], m["u"], m["s0"] - FLOOR - 0.5, m["s_end"], m["bore_r"] + grow)
    p.support("default", U([lining(m, 1.4) for m in sup])); held = {(m["node"], m["name"]) for m in sup}
    log(f"  supports: {sorted(n for _, n in held)}")
    for m in allm:
        if (m["node"], m["name"]) in held:
            continue
        p.load_multi(f"frame {m['name']} axial {m['axial']:.0f} N (pull-out)", [(lining(m), m["axial"] * m["u"])], cls="crash")
        side = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); side /= np.linalg.norm(side)
        p.load_multi(f"frame {m['name']} side {m['side']:.1f} N", [(lining(m), m["side"] * side)], cls="crash")
    L = max(allm, key=lambda m: m["len"])
    if (L["node"], L["name"]) not in held:
        sd = np.cross(L["u"], [0, 0, 1.0] if abs(L["u"][2]) < 0.9 else [1.0, 0, 0]); sd /= np.linalg.norm(sd)
        p.load_multi(f"crash 35 g through the longest rod {L['name']} ({L['len']:.0f} mm), axial + side", [(lining(L), -L["axial"] * L["u"] + L["side"] * sd)], cls="crash")
    for k, zb in feet.items():
        c = np.array(P[k], float)
        face = along(np.array([c[0], c[1], 0.0]), np.array([0, 0, 1.0]), zb, zb + 1.0, FOOT_R + 0.5)   # from the face up: the cell layer under the face is envelope skin, design room to the audit (load_room)
        fz = LAND_G * DRONE_G * GN / LAND_FEET
        p.load_multi(f"landing 6 g on the skid face of {k} ({fz:.0f} N up, 3 of 6 skid nodes)", [(face, np.array([0, 0, fz]))], cls="flight")
    return p


def make(group):
    return lambda h: build(group, h)


SPEC = dict(h=1.0, gen_cm3=2.0, max_worst=0.5)            # h 1.0 as the Overseer nodes (a node ball ~60 across is 262k cells at 1.0, 512k at 0.8); 2 cm3 (5 Nov, Spark: overweight; at 4 the 23 run-1 nodes were 8.3..10.5 g with indices 0.01..0.16, so half the volume keeps every index under 0.35)
SPEC_PAIR = dict(SPEC, h=1.2, gen_cm3=5.0)                # a merged pair: two nodes' 2 cm3 and a printed bridge's worth (O8 x 40 = 2 cm3, E); h 1.2: a 57 mm pair is 230k cells at 1.0 (the tail unit's 163k took 5.6 of the runner's 7 GB), 133k at 1.2
