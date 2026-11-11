"""Overseer cage node setups, one generator for all (28 Oct, Spark: "nodes dont look like topology, sharp corners, not organic").

Drawn: ONLY the socket walls and floors (a collar of 2.0 mm round each bore, 3.0 mm floor behind it) and, on the six base nodes,
the boot stud at the node's lowest point (v7, below). Everything joining them is generated inside a compact envelope
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
Base (foot) nodes 0-5, v7 (11 Nov, Spark: "the foot nodes broke again ... it pretty much just snaps everywhere", "a full in depth design"):
  Root cause: the nodes on the drone are the 29-31 Oct prints (PETG, 12..17 g, gen 4.5..6 cm3), sized before any ground-hit case
  existed: worst index 0.16..0.30 at the 67 N landing and the frame loads. A 0.6 m drop onto one node is 9.7 J into a rigid node on an
  87 N/mm cage: 1300 N, 19x the landing they were sized for (node 5: 0.26 at 67 N -> 5.0 at 1300 N; printed PETG breaks near 0.30),
  with nothing between the node and the ground to give stroke. They are overloaded everywhere at once, so they snap everywhere; no
  one socket, port or layer line is the cause. Compared (parts/overseer_tpu_foot/make.py has the numbers):
    full TPU 95A node (E 8 MPa): epoxy does not hold in it, a 285..490 N pull-out sits on a rubber collar and the base ring's joints
      become hinges (the funnel seat floats); 68D (E 600..1190, 11 MPa across layers) is weaker than PCTG at the sockets and
      still rigid: no stroke, the same 1300 N. A PCTG core in a TPU shell: 3 mm of skin is no stroke either (9.7 J needs ~15 mm).
    the printed nodes + the v3 boots: the boot caps the hit near 560 N, still 8x what those nodes carry.
    chosen: a PCTG node that passes the 1300 N hit on its own (the gate, max_worst as before) carrying a replaceable 68D crush
      boot on a drawn stud, which caps the hit near 560 N (760 on a 40 % hard spool): two prints, because the boot's cells must
      print on their side and the node on its foot (an AMS co-print would put one of them in the wrong layer direction).
  Drawn besides the sockets, the same on all six so one boot fits every node: the boot stud, a disc O30 x 3.5 (the bed face; the
  landing face of a node without its boot) on a O24 neck 3.5 tall, under a O37 x 2.5 shoulder, with a O14 x 5 relief pocket up from the bed face. The boot's cup snaps over the disc,
  its lip and zip tie sit round the neck, and its RIM bears on the shoulder (0.4 of air under the disc): a first setup that
  carried the boot on the disc read 1.02 filled solid in the skid case, the skid's moment crossing the layers of the neck (room_full
  audit, 11 Nov); through the shoulder it enters the body on a O30..37 ring. Nothing else may grow under the collars.
  Sockets one node length back (Spark cuts the glued rods flush at the old collar mouths): each bore's floor is FOOT_GAP inside the
  PRINTED node's mouth (that lib's socket start, wall 2.5, + its 15 / 10 depth) and FOOT_DEEP deeper than a cage socket, so a hand
  cut up to 2.5 mm outside the old mouth still has the full 15 / 10 in the bore.
  Cases on the foot: landing 6 g and the boot-capped hit with a skid (BOOT_N up on the shoulder + BOOT_MU of it sideways at the
  sole, BOOT_LEVER under the shoulder at the peak, along x and along y; mu E: the boot is a lever the bare foot was not), both
  through the shoulder; and the 1300 N hit on the bare disc (the boot lost or spent), the 8 Nov case.
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
WALL, FLOOR, MARGIN = 3.0, 3.0, 8.0                   # wall 3.0 (7 Nov: 2.5 beyond the 0.5 injection annulus, as 2.5 was beyond nothing before); 2.5 at the resin grooves
KIND = {"tube": dict(bore_r=3.1, depth=15.0), "link": dict(bore_r=1.6, depth=10.0)}   # bore 6.2 / 3.2
GROOVE_D, GROOVE_H, LEAD_IN = 0.5, 1.5, 1.0            # the coupon socket (socket_cuts below): resin grooves, lead-in
POCKETS, POCKET_D, POCKET_DEG = 3, 0.3, 60.0                # resin pockets (Spark 7 Nov: "the inner pattern where the resin can make the best contact with the rod"):
#   three 0.3 deep, 60 deg wide channels along the bore from above the floor out of the mouth, between three 60 deg lands of the plain bore that
#   centre the rod; the port feeds the first pocket, the two grooves join them round the rod, the resin shows at the mouth through them
PORT_R, PORT_S, PORT_CLEAR_DEG = 1.25, 1.5, 30.0        # resin injection port (Spark 5 Nov: "nodes need to be resin injectable"; 8 Nov: O2.5 for the one 14 G tip, O2.11): through the collar wall PORT_S above the floor, out between the members (none within PORT_CLEAR_DEG of it: at 40 the 6..8-member Scout nodes had none, their freest 30..38), so the resin fills from the floor and shows at the mouth
CROWN_MIN_SIN = 0.3                                    # a bore within ~17 deg of vertical prints without a crown
LAND_G, DRONE_G = 6.0, 3405.0                         # landing case 6 g; drone mass 3405 g (E, layout)
LAND_FEET = 3                                         # a flat landing may rest on 3 of the 6 feet (E)
HIT_N = 1300.0                                        # ground hit on one base node (8 Nov, Spark: the rear base node broke in three in testing): 1.65 kg from 0.6 m, 9.6 J, stopped in ~15 mm of rod flex, peak ~2x the 640 N average
FOOT_R, FOOT_T, NECK_R, NECK_T, FOOT_CH = 15.0, 3.5, 12.0, 3.5, 0.6      # v7 boot stud: disc O30 x 3.5 on a O24 neck 3.5 tall (the neck carries the skid moment at ~9 MPa; O20 was 15)
FOOT_GAP, FOOT_DEEP, PRINTED_WALL = 0.5, 2.5, 2.5          # v7 sockets: floor 0.5 inside the printed node's collar mouth, 2.5 deeper than 15 / 10; the printed lib's wall (its socket starts)
POCKET_R, POCKET_H = 7.0, 5.0                              # a O14 x 5 relief pocket up the stud from the bed face: no wall of the stud over 5 thick
SHOULDER_R, SHOULDER_T, RIM_R = 18.5, 2.5, (15.0, 18.0)    # the shoulder the boot's rim bears on (rim: the cup wall, 3.0 from the disc)
BOOT_N, BOOT_MU, BOOT_LEVER = 800.0, 0.5, 29.0             # the boot-capped hit (overseer_tpu_foot v4 model on a 40 % hard spool: 763 N at v3's block, x 36 / 34.5 deep, E), friction at the sole (E), the sole under the shoulder at that peak (7 of stud + 31.4 of boot - 9.7 of stroke)
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


def _socket_starts(wall=WALL, foot=True):
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
                s0 = max(s0, (m["bore_r"] + o["bore_r"] + 2 * GROOVE_D + wall) / (2.0 * np.sin(t / 2.0)))   # at the grooves too
            st[(n, m["name"])] = float(np.ceil(s0 * 10.0) / 10.0)          # up to the next 0.1 mm: the check below is exact
    if foot:                                                               # v7 base nodes: one node length back (docstring)
        for (n, nm), s in _socket_starts(PRINTED_WALL, False).items():
            if n in BASE:
                st[(n, nm)] = round(s + KIND["tube" if nm.startswith("tube") else "link"]["depth"] - FOOT_GAP, 2)
    return st


def printed_mouth(n, name):
    """mm from the centre of base node n to the PRINTED node's collar mouth on this member: where Spark cuts the rod"""
    return SOCKET_START[(n, name)] + FOOT_GAP


SOCKET_START = _socket_starts()


def members(n):
    """sockets of node n: dicts name, kind, far (far end point), u (unit, node -> far end), bore_r, depth, axial, side, s0 (where the
    bore starts, mm from the centre), s0_far (the same at the far end, a cage node)."""
    out = _raw_members(n)
    loads = CAGE["node_loads"].get(str(n), {})
    for m in out:
        if n in BASE:
            m["depth"] = m["depth"] + FOOT_DEEP
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
    pts = [(float(x), float(y)) for x, y in poly]
    if sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) < 0:   # manifold wants the outline counter-clockwise (7 Nov: a clockwise pocket sector extruded to nothing)
        pts = pts[::-1]
    cs = mf.CrossSection([pts])
    m = mf.Manifold.extrude(cs, float(s1 - s0))                       # along local z from 0 to s1 - s0
    R = np.column_stack([e1, e2, u]); t = c + u * s0
    return m.transform(np.column_stack([R, t]).astype(np.float64))


def socket_cuts(c, u, s0, depth, r, build=(0.0, 0.0, 1.0), beyond=1.0, pockets=0, pocket_dir=None):
    """cuts of one socket whose bore bottom is at c + u s0 and mouth at c + u (s0 + depth), bore radius r: the bore (to beyond past
    the mouth), two resin grooves, the lead-in cone, the teardrop crown toward the build direction where the bore overhangs, and
    (pockets > 0) that many POCKET_D deep resin channels of POCKET_DEG along the bore from PORT_S - 0.5 above the floor out of the
    mouth, the first centred on pocket_dir (the injection port's direction) so the port feeds it"""
    u = np.asarray(u, float) / np.linalg.norm(u); z = np.asarray(build, float)
    s_end = s0 + depth
    cuts = [along(c, u, s0, s_end + beyond, r)]
    if pockets:
        d = np.asarray(pocket_dir, float) if pocket_dir is not None else np.cross(u, [0, 0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0, 0])
        e1 = d - (d @ u) * u; e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        half = np.radians(POCKET_DEG / 2)
        for k in range(pockets):
            a0 = 2 * np.pi * k / pockets
            arc = np.linspace(a0 - half, a0 + half, 9)
            poly = [((r - 0.2) * np.cos(t), (r - 0.2) * np.sin(t)) for t in arc] + [((r + POCKET_D) * np.cos(t), (r + POCKET_D) * np.sin(t)) for t in arc[::-1]]
            cuts.append(_prism(poly, c, u, s0 + PORT_S - 0.5, s_end + beyond, e1, e2))
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


def seg_dist(a0, a1, b0, b1, n=41):
    """the least distance between two segments (sampled, 1/40 of each)"""
    a0, a1, b0, b1 = (np.asarray(x, float) for x in (a0, a1, b0, b1))
    u, v = a1 - a0, b1 - b0; A, C = max(u @ u, 1e-12), max(v @ v, 1e-12); best = 1e9
    for t in np.linspace(0, 1, n):
        q = b0 + t * v; s_ = np.clip((q - a0) @ u / A, 0, 1); best = min(best, float(np.linalg.norm(a0 + s_ * u - q)))
        q = a0 + t * u; t2 = np.clip((q - b0) @ v / C, 0, 1); best = min(best, float(np.linalg.norm(b0 + t2 * v - q)))
    return best


def port_dir(u, prefer, others=(), blockers=(), origin=None, length=40.0, clear_r=0.0, log=None):
    """the direction of a socket's injection port: perpendicular to the bore axis u, as close to `prefer` as the room allows: no other
    member axis (others, unit vectors) within PORT_CLEAR_DEG of it, and its channel (from origin, length) clear of every blocker
    segment ((p0, p1, radius): other collars) by clear_r. Falls back to the freest direction and says so."""
    u = np.asarray(u, float) / np.linalg.norm(u); pref = np.asarray(prefer, float)
    e1 = pref - (pref @ u) * u
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(u, [0, 0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0, 0])
    e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    best, best_free = None, (-1e9, None)
    for a in np.radians(np.arange(0, 360, 10)):
        d = np.cos(a) * e1 + np.sin(a) * e2
        free = min([np.degrees(np.arccos(np.clip(abs(d @ np.asarray(o, float)), -1, 1))) for o in others] + [90.0])
        ok = free >= PORT_CLEAR_DEG
        if ok and origin is not None:
            o = np.asarray(origin, float)
            for p0, p1, r in blockers:
                if seg_dist(o, o + d * length, p0, p1) < r + clear_r:
                    ok = False; break
        score = d @ pref
        if ok and (best is None or score > best[0]):
            best = (score, d)
        if free > best_free[0]:
            best_free = (free, d)
    if best is None:
        if log:
            log(f"  port: no direction clears the members by {PORT_CLEAR_DEG:.0f} deg; the freest ({best_free[0]:.0f} deg) taken")
        return best_free[1]
    return best[1]


def port_cut(c, u, s0, d, length=40.0):
    """the injection port: a O2 PORT_R channel from the bore axis at s0 + PORT_S out along d through the whole envelope"""
    u = np.asarray(u, float) / np.linalg.norm(u)
    return along(np.asarray(c, float) + u * (s0 + PORT_S), np.asarray(d, float), 0.0, length, PORT_R)


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


from parts.socket_inject import injectable_socket_cuts


def inject_port(c, m, ms, log=print, avoid=()):
    """the direction the socket's fill port and vent leave the node: the crown side if that line clears every other member's collar
    by 1 mm out to 40 mm, else the first clear direction round the socket axis (45 deg steps); the crown side with a warning if none"""
    from parts.socket_inject import FEED_H, PORT_R, GAP
    u = m["u"]; base = crown_dir(u)
    if base is None:
        base = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    base = base - (base @ u) * u; base /= np.linalg.norm(base); e2 = np.cross(u, base)
    starts = [c + u * (m["s0"] + FEED_H / 2), c + u * (m["s0"] + m["depth"] - (3.0 if m["bore_r"] > 2.5 else 2.0) - 0.75)]
    def clear(e, margin):
        if avoid:                                                                # a node's extra bodies and cuts (node 21: the camera)
            for p0 in starts:
                hole = along(p0, e, 0.0, m["bore_r"] + 40.0, PORT_R + 2.0, 16)
                for body in avoid:
                    if (hole ^ body).volume() > 1e-6:
                        return False
        for o in ms:
            if o is m:
                continue
            A = c + o["u"] * (o["s0"] - FLOOR); B = c + o["u"] * (o["s0"] + o["depth"] + 60.0); AB = B - A
            for p0 in starts:
                P = p0[None, :] + e[None, :] * np.linspace(0.0, m["bore_r"] + 40.0, 60)[:, None]
                t = np.clip(((P - A) @ AB) / (AB @ AB), 0, 1); d = np.linalg.norm(P - (A + t[:, None] * AB), axis=1)
                if d.min() < o["bore_r"] + margin:
                    return False
        return True
    dirs = [np.cos(np.radians(22.5 * k)) * base + np.sin(np.radians(22.5 * k)) * e2 for k in range(16)]
    for margin in (WALL + PORT_R + 2.0, GAP + PORT_R + 2.0):   # the hole's wall 2 mm off the neighbour's collar; else 2 mm off its annulus
        for e in dirs:
            if clear(e, margin):
                if margin < WALL + PORT_R + 2.0:
                    log(f"  inject port for {m['name']}: passes through a neighbour's collar wall (2 mm off its annulus)")
                return e
    log(f"  inject port for {m['name']}: no clear direction, using the crown side")
    return base


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
        bores += injectable_socket_cuts(c, m["u"], m["s0"], m["depth"], m["bore_r"],
                                        port=inject_port(c, m, ms, log, avoid=ex.get("drawn", []) + ex.get("cuts", [])))   # bore, grooves, lead-in,
        #   crown + the resin injection pattern (parts/socket_inject.py: annulus with 3 lands, feed ring, O2 fill port, O1.5 vent; Spark 7 Nov)
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
        zb = c[2] - NODE_R - 3.0                                 # landing face bottom (z -93.05 on the base loop, as v6)
        axis = np.array([c[0], c[1], 0.0]); up = np.array([0, 0, 1.0])
        foot = cone(axis + up * zb, up, 0.0, FOOT_CH, FOOT_R - FOOT_CH, FOOT_R, 64) + along(axis, up, zb + FOOT_CH - 0.05, zb + FOOT_T, FOOT_R, 64) \
            + along(axis, up, zb + FOOT_T - 0.1, zb + FOOT_T + NECK_T + 0.1, NECK_R, 64) \
            + along(axis, up, zb + FOOT_T + NECK_T, zb + FOOT_T + NECK_T + SHOULDER_T, SHOULDER_R, 64)   # the bed chamfer leads the boot's lip on
        lo[2] = zb + FOOT_T + NECK_T                            # no room under the collars: the boot's cup and tie sit round the stud
    import manifold3d as mf
    box = mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist())
    ball = mf.Manifold.sphere(max(np.linalg.norm(pts - c, axis=1)) + MARGIN * 0.75, 48).translate(c.tolist())
    env = box ^ ball
    if base:                                                    # v7: sockets 25..37 out, so the box is mostly air: room is the hull of the collars grown by MARGIN (the
        hp = []                                                 # load paths between the sockets and the stud), cut off under the collars, plus the stud
        for m in ms:
            e1 = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(m["u"], e1)
            for s_ in (m["s0"] - FLOOR, m["s_end"]):
                hp += [c + m["u"] * s_ + (m["ro"] + MARGIN) * (np.cos(a_) * e1 + np.sin(a_) * e2) for a_ in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        hp += [c + (NODE_R + MARGIN) * np.array([np.cos(a_) * np.cos(b_), np.sin(a_) * np.cos(b_), np.sin(b_)]) for a_ in np.linspace(0, 2 * np.pi, 12, endpoint=False) for b_ in (-0.6, 0.0, 0.6, 1.2)]
        env = (env ^ mf.Manifold.hull_points(np.array(hp).tolist())) + foot
    drawn = U(collars + ([foot] if base else []) + ex.get("drawn", []))
    if base:
        bores.append(along(axis, up, zb - 1.0, zb + POCKET_H, POCKET_R, 48))   # the stud's relief pocket (drawn_blocks audit, 11 Nov: the solid stud was an 8..10 mm core)
    cuts = U(bores + keep + ex.get("cuts", []))
    if ex.get("env"):
        env = U([env] + ex["env"])
    verify_cuts(n, ms, cuts, log)
    p = Part(f"overseer_cage_node_{n}", h, drawn, cuts, env, build="+z", material="PCTG")
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
        face = along(axis, up, zb, zb + 3.0, FOOT_R - 1.0) - along(axis, up, zb - 1.0, zb + 4.0, POCKET_R + 0.5, 48)   # the bare disc, round its pocket
        zs = zb + FOOT_T + NECK_T                                # the shoulder's underside
        ring = along(axis, up, zs, zs + SHOULDER_T, RIM_R[1], 64) - along(axis, up, zs - 1.0, zs + SHOULDER_T + 1.0, RIM_R[0], 64)   # where the boot's rim bears
        fz = LAND_G * DRONE_G * GN / LAND_FEET
        p.load_multi(f"landing 6 g on the foot ({fz:.0f} N up, 3 of 6 feet)", [(ring, np.array([0, 0, fz]))], cls="flight")
        p.load_multi(f"ground hit {HIT_N:.0f} N on the bare disc (a 0.6 m drop onto this node alone, 8 Nov; the boot lost or spent)", [(face, np.array([0, 0, HIT_N]))], cls="crash")
        Fh = BOOT_MU * BOOT_N; a = 0.5 * (RIM_R[0] + RIM_R[1])
        P = Fh * (BOOT_LEVER - (FOOT_T / 2 + NECK_T)) / (2 * a)    # the skid's moment about the shoulder, less what the side force applied at the disc already carries, as a couple on the rim
        for nm, d in (("x", np.array([1.0, 0, 0])), ("y", np.array([0, 1.0, 0]))):
            pad = lambda sg: ring ^ along(axis + sg * a * d, up, zs - 1.0, zs + SHOULDER_T + 1.0, 5.0)
            p.load_multi(f"boot-capped hit {BOOT_N:.0f} N with a skid along {nm} ({Fh:.0f} N at the sole, {BOOT_LEVER:.0f} under the shoulder)",
                         [(ring, np.array([0, 0, BOOT_N])), (face, Fh * d), (pad(1), np.array([0, 0, P])), (pad(-1), np.array([0, 0, -P]))], cls="crash")
    if ex.get("loads"):
        ex["loads"](p)
    return p


def make(n, extra=None):
    return lambda h: build(n, h, extra=extra)
