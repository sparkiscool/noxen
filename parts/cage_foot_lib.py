"""Overseer base (foot) cage nodes 0-5, v7: the setup of parts/overseer_cage_node_{2,3,4,5} (3 mirrors for node 1, 4 for node 0).
The sockets, ports and member data are parts/cage_node_lib.py's; this file holds what the base nodes do differently. It is its own
file because cage_node_lib.py is in the identity of every cage and Scout node: an edit there reruns whichever of them is stale.

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
  Drawn besides the sockets, the same on all six so one boot fits every node: the boot stud, a disc O30 x 3.0 (the bed face, a O8 x 3
  relief pocket in it; the landing face of a node without its boot) on a O28 neck 3.5 tall, under a O25..37 x 2.5 shoulder ring. The boot's
  cup snaps over the disc, its lip and zip tie sit round the neck, and its RIM bears on the shoulder (0.4 of air under the disc), so
  a sideways hit on the boot enters the body on a O30..36 ring, not through the neck. Nothing else may grow under the collars.
  Sockets one node length back (Spark cuts the glued rods flush at the old collar mouths): each bore's floor is FOOT_GAP inside the
  PRINTED node's mouth (that lib's socket start, wall 2.5, + its 15 / 10 depth) and FOOT_DEEP deeper than a cage socket, so a hand
  cut up to 2.5 mm outside the old mouth still has the full 15 / 10 in the bore.
  Cases on the foot: landing 6 g through the shoulder, and the 1300 N hit on the bare disc (the boot lost or spent), the 8 Nov
  case; for these the node is held at every socket (support "sockets"): the frame model has no one-node drop case, so the sockets'
  reactions are the stiff bound. MISSING: a sideways (skid) hit through the boot. It needs a friction number and the boot's
  measured plateau (the drop test in overseer_tpu_foot/make.py); with mu 0.5 guessed, the boot rocks onto ~80 deg of its rim.
Supports: the frame cases as cage_node_lib (the two tubes with the largest axial force); the foot cases every socket.
"""
import numpy as np
from parts.cage_node_lib import (NODES, CAGE, NODE_R, BASE, WALL, FLOOR, MARGIN, KIND, GROOVE_D, LAND_G, DRONE_G, LAND_FEET, HIT_N, GN,
                                 along, U, cone, collar_shape, inject_port, injectable_socket_cuts, verify_cuts, check_members,
                                 _raw_members)
from parts.cage_node_lib import members as _members
from topo import Part

FOOT_R, FOOT_T, NECK_R, NECK_T, FOOT_CH = 15.0, 3.0, 14.0, 3.5, 0.6      # v7 boot stud: disc O30 x 3.0 on a O28 neck 3.5 tall (O24 with a O14 x 5 pocket read 0.25 filled solid in the 1300 N hit on node 4, limit 0.2: 4.4 MPa of plain bearing across the layers)
FOOT_GAP, FOOT_DEEP, PRINTED_WALL = 0.5, 2.5, 2.5          # v7 sockets: floor 0.5 inside the printed node's collar mouth, 2.5 deeper than 15 / 10; the printed lib's wall (its socket starts)
POCKET_R, POCKET_H = 4.0, 3.0                              # a O8 x 3 relief pocket in the disc: the stud above it is 6.5 tall (drawn_blocks audit: solid, it was an 8..10 mm core)
SHOULDER_R, SHOULDER_T, RIM_R = 18.5, 2.5, (15.0, 18.0)    # the shoulder the boot's rim bears on (rim: the cup wall, 3.0 from the disc)


def _printed_starts():
    """the PRINTED nodes' socket starts: cage_node_lib's rule at the wall it had then (2.5; 3.0 since 7 Nov)"""
    st = {}
    for n in BASE:
        ms = _raw_members(n)
        for m in ms:
            s0 = NODE_R
            for o in ms:
                if o is m:
                    continue
                t = float(np.arccos(np.clip(m["u"] @ o["u"], -1.0, 1.0)))
                if t < 1e-6:
                    continue
                s0 = max(s0, (m["bore_r"] + o["bore_r"] + 2 * GROOVE_D + PRINTED_WALL) / (2.0 * np.sin(t / 2.0)))
            st[(n, m["name"])] = float(np.ceil(s0 * 10.0) / 10.0)
    return st


_PRINTED = _printed_starts()


def printed_mouth(n, name):
    """mm from the centre of base node n to the PRINTED node's collar mouth on this member: where Spark cuts the rod"""
    return _PRINTED[(n, name)] + KIND["tube" if name.startswith("tube") else "link"]["depth"]


def members(n):
    """cage_node_lib.members with the v7 sockets: floor FOOT_GAP inside the printed mouth, FOOT_DEEP deeper"""
    check_members(n, log=lambda *a: None)
    ms = _members(n)
    for m in ms:
        m["s0"] = round(printed_mouth(n, m["name"]) - FOOT_GAP, 2); m["depth"] = m["depth"] + FOOT_DEEP
    return ms


def build(n, h, log=print):
    ms = members(n); ex = {}
    assert n in BASE
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
            + along(axis, up, zb + FOOT_T - 0.1, zb + FOOT_T + NECK_T + 0.3, NECK_R, 64) \
            + (along(axis, up, zb + FOOT_T + NECK_T, zb + FOOT_T + NECK_T + SHOULDER_T, SHOULDER_R, 64) - along(axis, up, zb, zb + 20.0, NECK_R - 1.5, 64))   # the shoulder is a ring: over the neck is design room (drawn_blocks audit: disc + neck + a full shoulder was a 9.5 mm core)   # the bed chamfer leads the boot's lip on
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
        face = along(axis, up, zb, zb + 2.5, FOOT_R - 1.0) - along(axis, up, zb - 1.0, zb + 4.0, POCKET_R + 0.5, 48)   # the bare disc, round its pocket
        zs = zb + FOOT_T + NECK_T                                # the shoulder's underside
        ring = along(axis, up, zs, zs + SHOULDER_T, RIM_R[1], 64) - along(axis, up, zs - 1.0, zs + SHOULDER_T + 1.0, RIM_R[0], 64)   # where the boot's rim bears
        p.support("sockets", U([lining(m, 1.4) for m in ms]))    # the foot cases: the node is held by every rod glued into it (v6 held it by two
        #   sockets 10 mm from the centre; with the bores 25..37 out that clamp alone read 0.25..0.35 with the room filled solid, room_full audit 11 Nov)
        fz = LAND_G * DRONE_G * GN / LAND_FEET
        p.load_multi(f"landing 6 g on the foot ({fz:.0f} N up, 3 of 6 feet)", [(ring, np.array([0, 0, fz]))], cls="flight", support="sockets")
        p.load_multi(f"ground hit {HIT_N:.0f} N on the bare disc (a 0.6 m drop onto this node alone, 8 Nov; the boot lost or spent)", [(face, np.array([0, 0, HIT_N]))], cls="crash", support="sockets")
    if ex.get("loads"):
        ex["loads"](p)
    return p



def make(n):
    return lambda h: build(n, h)
