"""Overseer base (foot) cage nodes 0-5, v8: the setup of parts/overseer_cage_node_{2,3,4,5} (3 mirrors for node 1, 4 for node 0).
The sockets, ports and member data are parts/cage_node_lib.py's; this file holds what the base nodes do differently. It is its own
file because cage_node_lib.py is in the identity of every cage and Scout node: an edit there reruns whichever of them is stale.

v8 (11 Nov, Spark on v7's stud + 68D boot: "I dont want this stupid design with the tpu block under, i want a whole redesign it needs
to look good and not be tall needs to still fit the funnel, it needs to not crack like the other ones before as well"):
  Why the printed nodes cracked (v7's finding, kept): they are the 29-31 Oct prints (PETG, 12..17 g, gen 4.5..6 cm3), sized before any
  ground-hit case existed: worst index 0.16..0.30 at the 67 N landing. A 0.6 m drop onto one node is 9.7 J into a rigid node on an
  87 N/mm cage: 1300 N, 19x that (node 5: 0.26 at 67 N -> 5.0 at 1300 N; printed PETG breaks near 0.30). Overloaded everywhere at
  once, so they snapped everywhere; and their bores started 10 mm from the centre, so the only material between six sockets was
  2.5 mm webs with a resin groove, a port and a bore corner in each: every crack had a notch to start from.
  The design: ONE PCTG body, no boot, no stud, no shoulder, nothing under it. The node itself carries the full 1300 N at its gate
  (v7's bodies already did, 0.11..0.28, with the boot off), in PCTG (10-20x PETG's impact toughness).
    Low: the landing face is the body's own flat belly at FOOT_DROP 9.1 under the loop axis (the tube collars' underside + one
      wall), 4.05 less than the printed nodes' 13.15 and 31 less than v7 with its boot. The collars stay 3.0 off the ground.
    One body: the room is a hub over the sole and one tapered arm to each socket, its floor the ground plane, so the belly, the
      arms and the collars are one generated surface and the print stands on the belly (the largest flat face).
    No notches in the load path: the sockets are one node length back (25..37 mm out), so the hub between them is solid body,
      not webs; in the hit the arms bend with tension along their underside at the collar roots, in the layers (the arms lie in
      the bed plane), and the ports leave on the crown side (the compression side); nothing drawn is under 3.0; the generated
      skin is filleted 2.5 into every collar.
    No TPU pad: 3 mm of 95A is no stroke (9.7 J needs ~15 mm) and it would be a second part glued to the bed face.
  Drawn besides the sockets: the sole, a O30 x 3.0 flat in the belly (0.6 chamfer on the bed edge): 1.8 MPa of bearing in the hit.
  Sockets one node length back (Spark cuts the glued rods flush at the old collar mouths): each bore's floor is FOOT_GAP inside the
  PRINTED node's mouth (that lib's socket start, wall 2.5, + its 15 / 10 depth) and FOOT_DEEP deeper than a cage socket, so a hand
  cut up to 2.5 mm outside the old mouth still has the full 15 / 10 in the bore.
  Cases on the foot: landing 6 g and the 1300 N hit, both on the sole; for these the node is held at every socket (support
  "sockets"): the frame model has no one-node drop case, so the sockets' reactions are the stiff bound. MISSING: a hit on the
  sole's edge with the drone tilted, and a sideways (skid) hit: they need the tilt and a friction number nobody has measured.
Supports: the frame cases as cage_node_lib (the two tubes with the largest axial force); the foot cases every socket.
"""
import numpy as np
from parts.cage_node_lib import (NODES, CAGE, NODE_R, BASE, WALL, FLOOR, MARGIN, KIND, GROOVE_D, LAND_G, DRONE_G, LAND_FEET, HIT_N, GN,
                                 along, U, cone, collar_shape, inject_port, injectable_socket_cuts, verify_cuts, check_members,
                                 _raw_members)
from parts.cage_node_lib import members as _members
from topo import Part

FOOT_R, FOOT_T, FOOT_CH = 15.0, 3.0, 0.6                   # v8 sole: O30 x 3.0 in the belly (O24 read 4.4 MPa of plain bearing across the layers in the 1300 N hit on node 4, v7)
FOOT_DROP = KIND["tube"]["bore_r"] + 2 * WALL              # the belly 9.1 under the loop axis: one wall under the tube collars (the printed nodes: NODE_R + 3.0 = 13.15)
FOOT_GAP, FOOT_DEEP, PRINTED_WALL = 0.5, 2.5, 2.5          # sockets: floor 0.5 inside the printed node's collar mouth, 2.5 deeper than 15 / 10; the printed lib's wall (its socket starts)


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
        zb = c[2] - FOOT_DROP                                    # the belly: the landing face and the bed face (z -89.0 on the base loop; printed -93.05)
        axis = np.array([c[0], c[1], 0.0]); up = np.array([0, 0, 1.0])
        foot = cone(axis + up * zb, up, 0.0, FOOT_CH, FOOT_R - FOOT_CH, FOOT_R, 64) + along(axis, up, zb + FOOT_CH - 0.05, zb + FOOT_T, FOOT_R, 64)
        lo[2] = zb                                              # the room's floor is the ground plane: nothing under the belly
    import manifold3d as mf
    box = mf.Manifold.cube((hi - lo).tolist()).translate(lo.tolist())
    ball = mf.Manifold.sphere(max(np.linalg.norm(pts - c, axis=1)) + MARGIN * 0.75, 48).translate(c.tolist())
    env = box ^ ball
    if base:                                                    # sockets 25..37 out, so the box is mostly air. Room is a HUB over the sole and one tapered arm from it
        #   to each socket (the hull of the hub with that collar grown by MARGIN), cut off at the ground plane. Not the hull of all the collars: in that room the
        #   v7 layout joined the sockets to each other past the centre and hung the foot off the body's edge, backed on one side. Every load path crosses the hub.
        hub = [c + (NODE_R + MARGIN) * np.array([np.cos(a_) * np.cos(b_), np.sin(a_) * np.cos(b_), np.sin(b_)]) for a_ in np.linspace(0, 2 * np.pi, 16, endpoint=False) for b_ in (-0.6, 0.0, 0.6, 1.2)]
        hub += [axis + up * z_ + (FOOT_R + 3.0) * np.array([np.cos(a_), np.sin(a_), 0.0]) for a_ in np.linspace(0, 2 * np.pi, 16, endpoint=False) for z_ in (zb, zb + 2 * MARGIN)]
        arms = []
        for m in ms:
            e1 = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(m["u"], e1)
            sock = [c + m["u"] * s_ + (m["ro"] + MARGIN) * (np.cos(a_) * e1 + np.sin(a_) * e2) for s_ in (m["s0"] - FLOOR, m["s_end"]) for a_ in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
            arms.append(mf.Manifold.hull_points(np.array(hub + sock).tolist()))
        env = env ^ U(arms)
    drawn = U(collars + ([foot] if base else []) + ex.get("drawn", []))
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
        face = along(axis, up, zb, zb + 2.5, FOOT_R - 1.0)       # inside the drawn 3.0 sole
        p.support("sockets", U([lining(m, 1.4) for m in ms]))    # the foot cases: the node is held by every rod glued into it
        fz = LAND_G * DRONE_G * GN / LAND_FEET
        p.load_multi(f"landing 6 g on the foot ({fz:.0f} N up, 3 of 6 feet)", [(face, np.array([0, 0, fz]))], cls="flight", support="sockets")
        p.load_multi(f"ground hit {HIT_N:.0f} N on the foot (a 0.6 m drop onto this node alone, 8 Nov)", [(face, np.array([0, 0, HIT_N]))], cls="crash", support="sockets")
    if ex.get("loads"):
        ex["loads"](p)
    return p



def make(n):
    return lambda h: build(n, h)
