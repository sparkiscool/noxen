"""Overseer battery holder v1 (3 Nov, Spark: "a battery holder for our Zeee 5200 battery, screws into the 4 screws at the bottom,
no strap, light, clears the bottom"). One PETG piece under the base, on its 4 M3 x 5.7 inserts (dimensions.md Base: 80 x 50 at
(-60, +-25), (20, +-25), faces on z -8.1). The pack (Zeee 4S 5200, 138 x 47 x 48, ~475 g, Amazon listing, E until measured) slides
in from the tail, leads aft, against a front stop, and a snap tab in the floor drops behind its rear face: no strap. Out: press the
tab end down, slide aft. Pack centre x -20 (the base's pack design point), top under the 4 pads.
Drawn: the 4 bolt pads (O11 x 6, M3 through, head recessed from below), small contact pads on the pack's bottom, sides and front
(2.0), the snap tab (2.4 x 12, 38 long: 30 free + 8 root in the floor, 2.5 hook, ramp aft). Cut: the pack (0.5 a side) and its slide-out path aft through the
envelope (less the hook), the tab's free slots and its 3.5 deflection room, the M3 shanks and a O7 screwdriver path below each pad.
Generated: everything between. Supports: the pad tops on the base. Loads: the pack 480 g (E) at its centre, each case's pack force
split by direction onto the faces that hold it (floor, pads, sides, front stop, tab): crash 35 g in 14 directions, flight, landing."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from topo import Part
from parts.cage_node_lib import U

SPEC = dict(h=1.5, gen_cm3=22.0, max_worst=0.5)
G = 9.81e-3
Z_MOUNT = -8.1
BOLTS = [(-60.0, 25.0), (-60.0, -25.0), (20.0, 25.0), (20.0, -25.0)]
PAD_R, PAD_T, SHANK_R, HEAD_R, HEAD_D = 5.5, 6.0, 1.6, 3.4, 3.2
PACK = (138.0, 47.0, 48.0); CLR = 0.5; PACK_G = 480.0
X0, X1 = -20.0 - PACK[0] / 2 - CLR, -20.0 + PACK[0] / 2 + CLR      # cavity x -89.5..49.5
YW = PACK[1] / 2 + CLR                                              # cavity y +-24
ZT = Z_MOUNT - PAD_T - CLR; ZB = ZT - PACK[2] - 2 * CLR             # cavity z -63.6..-14.6
PACK_CG = np.array([-20.0, 0.0, (ZT + ZB) / 2])
TAB_W, TAB_T, TAB_L, TAB_FREE, HOOK = 12.0, 2.4, 38.0, 30.0, 2.5   # free 30 (1.0 % strain at the 2.5 hook), root 8 in the generated floor
PAD = 2.0
DIRS14 = [np.array(v, float) / np.linalg.norm(v) for v in
          [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
FLIGHT = {"flight 4 g": np.array([0, 0, 4.0]), "flight 2 g x": np.array([2.0, 0, 1]), "flight 2 g y": np.array([0, 2.0, 1]), "landing 6 g": np.array([0, 0, 6.0])}


def box(x0, x1, y0, y1, z0, z1):
    import manifold3d as mf
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl(x, y, z0, z1, r, n=40):
    import manifold3d as mf
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def hook():
    """the tab's hook: a wedge across y, vertical catch face at x X0 - 0.5 (behind the pack's rear face), ramp down aft"""
    import manifold3d as mf
    cs = mf.CrossSection([[(0.0, 0.0), (HOOK, 0.0), (0.0, HOOK)]])                # in (x back from the catch face, z up)
    m = mf.Manifold.extrude(cs, TAB_W).translate([0, 0, -TAB_W / 2])            # local z = y
    m = m.transform(np.array([[-1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0]]))    # x -> -x, y <- local z, z <- local y
    return m.translate([X0 - 0.5, 0.0, ZB])


def bodies():
    tab_x0 = X0 - 0.5 - HOOK                                           # tab end, aft
    pads = [cyl(x, y, Z_MOUNT - PAD_T, Z_MOUNT, PAD_R) for x, y in BOLTS]
    floor = [box(x - 8, x + 8, s * 7.0, s * 23.0, ZB - PAD, ZB) if s > 0 else box(x - 8, x + 8, -23.0, -7.0, ZB - PAD, ZB)
             for x in (-75.0, 35.0) for s in (1, -1)]
    sides = [box(x - 8, x + 8, YW, YW + PAD, PACK_CG[2] - 8, PACK_CG[2] + 8) for x in (-75.0, 35.0)] + \
            [box(x - 8, x + 8, -YW - PAD, -YW, PACK_CG[2] - 8, PACK_CG[2] + 8) for x in (-75.0, 35.0)]
    front = [box(X1, X1 + PAD, -10.0, 10.0, PACK_CG[2] - 10, PACK_CG[2] + 10)]
    tab = [box(tab_x0, tab_x0 + TAB_L, -TAB_W / 2, TAB_W / 2, ZB - TAB_T, ZB), hook()]
    cuts = [box(X0, X1, -YW, YW, ZB, ZT),                              # the pack
            box(X0 - 60.0, X0, -YW, YW, ZB, ZT + 0.01) - box(X0 - 0.5 - HOOK - 0.01, X0 - 0.49, -TAB_W / 2 - 0.01, TAB_W / 2 + 0.01, ZB - 0.01, ZB + HOOK + 0.01),
            box(tab_x0 - 1.0, tab_x0 + TAB_FREE, TAB_W / 2, TAB_W / 2 + 1.0, ZB - 20.0, ZB),      # the tab's free slots
            box(tab_x0 - 1.0, tab_x0 + TAB_FREE, -TAB_W / 2 - 1.0, -TAB_W / 2, ZB - 20.0, ZB),
            box(tab_x0 - 1.0, tab_x0 + TAB_FREE, -TAB_W / 2 - 1.0, TAB_W / 2 + 1.0, ZB - 20.0, ZB - TAB_T),   # its deflection room
            box(tab_x0 - 20.0, tab_x0, -TAB_W / 2 - 1.0, TAB_W / 2 + 1.0, ZB - 20.0, ZB)]      # nothing aft of the tab end (thumb)
    cuts += [cyl(x, y, Z_MOUNT - 80, Z_MOUNT + 1, SHANK_R) for x, y in BOLTS]
    cuts += [cyl(x, y, Z_MOUNT - 80, Z_MOUNT - PAD_T + HEAD_D, HEAD_R + 0.1) for x, y in BOLTS]   # head recess + driver path
    env = box(X0 - 4.0, X1 + 7.0, -YW - 8.0, YW + 8.0, ZB - 8.0, Z_MOUNT)
    drawn = pads + floor + sides + front + tab
    return U(drawn), U(cuts), env, dict(pads=pads, floor=floor, sides=sides, front=front, tab=tab)


def setup(h):
    drawn, cuts, env, R = bodies()
    p = Part("overseer_battery_holder", h, drawn, cuts, env, build="+z", material="PETG")
    p.support("default", U([cyl(x, y, Z_MOUNT - 0.6, Z_MOUNT + 0.5, PAD_R + 0.5) for x, y in BOLTS]))
    e = PAD + 1.5                                                       # contact regions: each drawn pad and 1.5 behind it (a 0.6 layer is empty at a coarse h)
    reg = {"-z": U([box(x - 8, x + 8, s0, s1, ZB - e, ZB) for x in (-75.0, 35.0) for s0, s1 in ((7.0, 23.0), (-23.0, -7.0))]),
           "+z": U([cyl(x, y, Z_MOUNT - PAD_T - 0.01, Z_MOUNT - PAD_T + 2.0, PAD_R) for x, y in BOLTS]),
           "+y": U([box(x - 8, x + 8, YW, YW + e, PACK_CG[2] - 8, PACK_CG[2] + 8) for x in (-75.0, 35.0)]),
           "-y": U([box(x - 8, x + 8, -YW - e, -YW, PACK_CG[2] - 8, PACK_CG[2] + 8) for x in (-75.0, 35.0)]),
           "+x": box(X1, X1 + e, -10.0, 10.0, PACK_CG[2] - 10, PACK_CG[2] + 10),
           "-x": box(X0 - 0.5 - HOOK - 0.5, X0 - 0.49, -TAB_W / 2, TAB_W / 2, ZB - TAB_T, ZB + HOOK)}   # the hook and the tab end under it

    def pack(name, a, cls):
        F = -PACK_G * G * np.asarray(a, float)                          # the pack's push on the holder
        parts = []
        for k, ax in enumerate("xyz"):
            if abs(F[k]) > 1e-6:
                v = np.zeros(3); v[k] = F[k]; parts.append((reg[("+" if F[k] > 0 else "-") + ax], v))
        p.load_multi(name, parts, cls=cls, accel=-np.asarray(a, float))
    for k, d in enumerate(DIRS14):
        pack(f"crash {k} (35 g, pack {PACK_G:.0f} g)", 35 * d, "crash")
    for nm, a in FLIGHT.items():
        pack(nm, a, "flight")
    return p
