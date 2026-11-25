"""Projector tilt stand, OPTIMISE (Spark 24 Nov: asked for the topology run himself, before any bench test of the drawn wedge in make.py).
Same wedge as make.py (20 deg, 130 wide, 130 along the slope), same frame (x ground, y up, z the width = the build axis: it prints on its
side, every layer a triangle, +z). Drawn: the deck plate 4 thick with its four 45 deg pointed slots (cuts), the full-width stop lip at the
low edge, the two base rails (the footprint, 14 wide, 4 thick), the back wall 4 thick at the high end. Cuts: the slots and the vent window (the triangle's inside shrunk 16: the
projector's side vents and ports, and the cables, stay open through the whole width). Envelope: the wedge. Generated: the 16 mm ring
strips under the deck, along the base and the back wall. Supports: the rails' undersides (the table). Loads: the projector's 640 g on the
deck (6.3 N), a hand's 20 N press on it, a 20 N shove down the slope on the lip, a 15 N shove along the width, a 15 N push at the high
end toward the low end. All cases are loads the stand sees; nothing guessed beyond the projector's mass (C, listing) and the pushes (E).
Print: build +z, no supports, nothing hangs: the slots' ends are 45 deg points."""
import math
import numpy as np
import manifold3d as mf
from topo.part import Part

TILT = math.radians(20.0)
W, SL, H0 = 130.0, 130.0, 12.0
DECK_T, RING = 4.0, 16.0
LIP_H, LIP_T = 10.0, 3.0
RAIL_W, RAIL_T = 14.0, 4.0
c, s = math.cos(TILT), math.sin(TILT)
LX = SL * c
HH = H0 + SL * s
G_PROJ = 6.28                     # 640 g (C)


def prism(pts, z0, z1):
    a = np.array(pts, float)
    if np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]) < 0:
        a = a[::-1]                                                       # counter-clockwise, or the cross-section is empty
    cs = mf.CrossSection([a])
    return mf.Manifold.extrude(cs, z1 - z0).translate([0, 0, z0])


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def _offset(pts, d):
    return mf.CrossSection([np.array(pts, float)]).offset(d, mf.JoinType.Miter, 2.0)


BODY = [(0, 0), (LX, 0), (LX, HH), (0, H0)]
U = np.array([c, s]); N = np.array([-s, c]); P0 = np.array([0.0, H0])


def _slot(k):
    s0 = 14.0 + k * 28.0
    pts = [(0, 27), (7, 20), (14, 27), (14, W - 27), (7, W - 20), (0, W - 27)]
    cut = prism(pts, 0, DECK_T + 2.0)                                  # local x along the slope, y the width, z the depth
    # local x -> u, local y -> z, local z -> -n (down through the deck)
    M = np.array([[c, 0, s, P0[0] + s0 * c - s], [s, 0, -c, P0[1] + s0 * s + c], [0, 1, 0, 0]], float)
    return cut.transform(M)


def bodies():
    ring_in = _offset(BODY, -RING)
    env = mf.Manifold.extrude(mf.CrossSection([np.array(BODY, float)]), W)
    vent = mf.Manifold.extrude(ring_in, W + 2.0).translate([0, 0, -1.0])
    # drawn deck plate: the wedge's top strip, DECK_T thick below the deck line
    deck_line = [P0, P0 + SL * U]
    deck = prism([tuple(P0), tuple(P0 + SL * U), tuple(P0 + SL * U - DECK_T * N), tuple(P0 - DECK_T * N)], 0, W)
    lip = prism([tuple(P0 - 1.0 * U), tuple(P0 + LIP_T * U), tuple(P0 + LIP_T * U + LIP_H * N), tuple(P0 - 1.0 * U + LIP_H * N)], 0, W)
    lip = lip + prism([tuple(P0 - 1.0 * U), tuple(P0 + LIP_T * U), tuple(P0 + LIP_T * U - 3 * N), tuple(P0 - 1.0 * U - 3 * N)], 0, W)
    rails = [box(0, LX, 0, RAIL_T, z0, z1) for z0, z1 in ((0, RAIL_W), (W - RAIL_W, W))]
    slots = None
    for k in range(4):
        sl = _slot(k)
        slots = sl if slots is None else slots + sl
    back = box(LX - 4.0, LX, 0, HH, 0, W)
    drawn = deck + lip + rails[0] + rails[1] + back
    drawn = drawn - slots
    cuts = vent + slots
    return dict(drawn=drawn, cuts=cuts, env=env + lip, deck=deck, lip=lip, rails=rails, back=back)


def setup(h):
    B = bodies()
    p = Part("projector_tilt_stand", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PCTG")
    sup = box(-1, LX + 1, -1, 1.5 * h, -1, W + 1)
    p.support("default", sup)
    deck = B["deck"] - sum([_slot(k) for k in range(4)], mf.Manifold())
    up = (0, -1.0)                                                      # gravity: -y
    p.load("projector 640 g on the deck", deck, (0, -G_PROJ, 0), cls="flight")
    p.load("hand presses the projector 20 N", deck, (0, -20.0, 0), cls="misuse")
    p.load("projector slides: 20 N down the slope on the lip", B["lip"], (-20.0 * c, -20.0 * s, 0), cls="misuse")
    p.load("shove along the width 15 N on the lip", B["lip"], (0, 0, 15.0), cls="misuse")
    p.load("pushed at the high end 15 N toward the low end", B["back"], (-15.0, 0, 0), cls="misuse")
    return p


SPEC = dict(h=1.7, gen_cm3=120.0, max_worst=0.5, settings=dict(build_pitch=0.40))   # 24 Nov: 0.3 gave 2.07 M faces, 15.1 GB (over 14); 0.40 is ~1.2 M faces, ~10 GB
