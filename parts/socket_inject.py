"""The injectable socket (Spark 3 Nov: every node socket resin-injectable after the rod is home). On test in parts/inject_coupon;
once Spark accepts it, cage_node_lib.socket_cuts adds these cuts to every socket and the parts are recut (tools/recut.py), not rerun.
Kept out of cage_node_lib until then so no part's setup changes.
A GAP annulus round the rod from the floor to `lip` short of the mouth, held concentric by 3 centring lands at the bore radius (one
opposite the port); a FEED_H ring at the floor that the O2.0 fill port (an 18 G / 16 G blunt needle) enters, so the resin starts at the
bottom and rises round the rod; a O1.5 vent at the top of the annulus on the port side (resin shows there when the socket is full);
the lip at the bore radius keeps it in. The ports are cut 40 out, through whatever is round them."""
import numpy as np
from parts.cage_node_lib import along, _prism, crown_dir, socket_cuts

GAP, LAND_W, FEED_H, PORT_R, VENT_R = 0.5, 1.2, 2.5, 1.25, 0.75   # 8 Nov (Spark: one tip for every socket on both drones): 14 G blunt tip O2.11 -> port O2.5, feed ring 2.5 tall so the port opens fully into it; gap 0.5 (7 Nov, was 0.3: one export cell wide, 6 of 13 nodes came out not watertight)


def inject_cuts(c, u, s0, depth, r, port, lip):
    u = np.asarray(u, float) / np.linalg.norm(u); e1 = np.asarray(port, float); e1 = e1 - (e1 @ u) * u; e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1); s_top = s0 + depth - lip; ro = r + GAP; a_l = LAND_W / 2 / r; out = []
    for k in range(3):                                                  # the annulus less 3 lands at 60, 180, 300 deg (180 = opposite the port)
        t = np.linspace(np.radians(60 + 120 * k) + a_l, np.radians(180 + 120 * k) - a_l, 16)
        poly = [(ro * np.cos(x), ro * np.sin(x)) for x in t] + [(r * 0.98 * np.cos(x), r * 0.98 * np.sin(x)) for x in t[::-1]]
        out.append(_prism(poly, c, u, s0 + FEED_H / 2, s_top, e1, e2))     # from inside the feed ring: no face on the floor plane
    out.append(along(c, u, s0 - 0.5, s0 + FEED_H, ro))                 # feed ring at the floor, 0.5 into it: its end face off the bore's floor plane (7 Nov: coincident faces left the export non-manifold at the floor centre)
    out.append(along(c + u * (s0 + FEED_H / 2), e1, 0.0, r + 40.0, PORT_R, 24))   # fill port, needle side
    out.append(along(c + u * (s_top - VENT_R), e1, 0.0, r + 40.0, VENT_R, 24))    # vent, same side, top of the annulus
    return out


def injectable_socket_cuts(c, u, s0, depth, r, build=(0.0, 0.0, 1.0), port=None):
    """socket_cuts + the injection pattern; port default: the crown side (up on the plate)"""
    u = np.asarray(u, float) / np.linalg.norm(u)
    if port is None:
        port = crown_dir(u, build)
        if port is None:
            port = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
    return socket_cuts(c, u, s0, depth, r, build) + inject_cuts(c, u, s0, depth, r, port, lip=3.0 if r > 2.5 else 2.0)
