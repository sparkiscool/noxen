"""Scout v4 probe coupon (12 Nov, Spark: "only have me do what is absolutely necessary; also if we are testing with a coupon we can find
what works that way as well"): every probe number nobody can look up, in three sizes, each with its embossed label, so the answer is a
line like "D2 B1 A3 G2 M1 T2 C2". The choices live in tools/scout_v4_shape.COUPON; the answer goes in COUPON_PICK (one edit), then
the drawn makes (spool, pinion, roller, bench turret) and the flight parts (turret, head frame, cradle) rerun from it.
Drawn: python parts/scout_v4_probe_coupon/make.py writes print/scout_v4_probe_coupon.stl (and part.stl, off the drone, for the record).
  D1-3  the pinion's D bore on the N20's shaft (3.0 / 2.5, 3.1 / 2.6, 3.2 / 2.7 across the flat): the one that presses on firm by hand
  B1-3  the MR63ZZ's seat (6.00, 6.05, 6.10, 2.7 deep): the one the bearing presses into with a thumb and stays in when turned over
  A1-3  the spool's bore on an M3 (3.3, 3.4, 3.5): the smallest an M3 screw turns in freely with no wobble
  G1-3  the gear pair: a O3 peg for the spool and a D peg for the pinion, centres 20.8 / 20.9 / 21.0: the first that turns smoothly
  M1-3  mandrels r 12.6 / 14.0 / 15.3 (0.9 of hubs r 14 / 15.5 / 17): a 10 cm scrap of blade wrapped once round it, the curved-in
        (marked) face OUT, held 30 s: the smallest that leaves no permanent curl (the coil's worst stress is opposite sense)
  T1-3  the throat, 16 tall, from the flat stack at the bottom to the lens at the top, for cups R 25.4 / 15 / 12.7 (1.8 / 3.4 / 4.3 deep):
        the two blades cups facing with the ribbon between, pushed up from the flat end: the narrowest they pass without binding
  C1-3  gaps 18.6 / 20.6 / 22.6 between ribs 5 tall (cases 18 / 20 / 22 deep, 0.6 clear): the narrowest the camera case drops into front to back
A lattice of 2.5 pads and rails, every feature on its own pad. Print: flat, no supports, 0.2 layers, PCTG, ~50 min E."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import S, cyl, extrude, union, minus, save  # noqa: E402
import numpy as np, trimesh
from shapely.geometry import Point, Polygon, box as sbox
from shapely.ops import unary_union
from shapely import affinity
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

T = 2.0                                   # the plate (the gate's 2.0)
EMB = 0.6                                 # the labels stand 0.6 proud
FONT = FontProperties(family="DejaVu Sans", weight="bold")
C = S.COUPON


def text2d(s, x, y, size=4.5):
    """the label's outline (shapely), its centre at (x, y): matplotlib's glyph rings, composed even-odd"""
    tp = TextPath((0, 0), s, size=size, prop=FONT)
    rings = [Polygon(p).buffer(0) for p in tp.to_polygons() if len(p) >= 3]
    g = Polygon()
    for r in sorted(rings, key=lambda q: -q.area):
        g = g.symmetric_difference(r)
    b = g.bounds
    return affinity.translate(g, x - (b[0] + b[2]) / 2, y - (b[1] + b[3]) / 2)


pads, add, cut, labels = [], [], [], []


def label(s, x, y):
    t = text2d(s, x, y); labels.append(t); pads.append(t.envelope.buffer(1.5, join_style=2))


# row 1 (y 0): D, B, A
for k, (d, f) in enumerate(C["D"]):                                               # D bores in O9 bosses 5 tall
    x = 12 * k; add.append(cyl(4.5, T, T + 5, x, 0)); pads.append(Point(x, 0).buffer(7.0))
    r = d / 2; cut.append(extrude(Point(x, 0).buffer(r, 48).intersection(sbox(x - 2, -2, x - r + f, 2)), -1, T + 6))
    label(f"D{k + 1}", x, -10)
for k, d in enumerate(C["B"]):                                                    # bearing seats in O10.1 bosses
    x = 40 + 13 * k; add.append(cyl(5.05, T, T + 3.5, x, 0)); pads.append(Point(x, 0).buffer(7.5))
    cut += [cyl(d / 2, T + 0.8, T + 4, x, 0, n=64), cyl(2.0, -1, T + 4, x, 0)]
    label(f"B{k + 1}", x, -10)
for k, d in enumerate(C["A"]):                                                    # axle bores in O8 bosses
    x = 82 + 12 * k; add.append(cyl(4.0, T, T + 5, x, 0)); pads.append(Point(x, 0).buffer(6.5))
    cut.append(cyl(d / 2, -1, T + 6, x, 0, n=48))
    label(f"A{k + 1}", x, -10)
pads.append(sbox(-6, -4, 112, 4))                                                 # row 1's rail
# row 2 (y 30): the gear pair on three centres (the spool's peg, its bore A, 14 of its 26; the pinion's D peg 3.0 / 2.5)
GX = (8.0, 46.0, 84.0); GY = 30.0
for k, c in enumerate(C["G"]):
    x = GX[k]
    add.append(cyl(1.5, T, T + 14, x, GY, n=32))
    add.append(extrude(Point(x, GY + c).buffer(1.5, 32).intersection(sbox(x - 2, GY + c - 2, x - 1.5 + 2.5, GY + c + 2)), T, T + 5))
    pads += [Point(x, GY).buffer(6.5), Point(x, GY + c).buffer(5.0), sbox(x - 3, GY, x + 3, GY + c)]
    label(f"G{k + 1}", x + 9.5, GY + c)
pads.append(sbox(-1, GY - 3, 93, GY + 3)); pads += [sbox(x - 3, 0, x + 3, GY) for x in GX]
# row 3 (y 84): the mandrels, 2.0 tubes 21 tall (the blade's 19 + 2), 240 deg arcs on ring pads, the label below each
MY = 84.0; MX = (14.0, 50.0, 88.0)
for k, hub in enumerate(C["M"]):
    x = MX[k]; rm = 0.9 * hub
    ring = Point(x, MY).buffer(rm, 128).difference(Point(x, MY).buffer(rm - 2.0, 128))
    wedge = Polygon([(x, MY)] + [(x + 40 * math.cos(math.radians(a)), MY + 40 * math.sin(math.radians(a))) for a in np.linspace(150, 390, 25)])
    add.append(extrude(ring.intersection(wedge), T - 0.01, T + 21))                # a 240 deg arc, open to +y: the scrap wraps over it
    pads.append(Point(x, MY).buffer(rm + 1.5, 128).difference(Point(x, MY).buffer(rm - 3.5, 128)).intersection(wedge.buffer(3.0)))
    label(f"M{k + 1}", x, MY - rm - 6.0)
    pads.append(sbox(x - 3, GY, x + 3, MY - rm + 1))
pads.append(sbox(MX[0], MY - 3, MX[2], MY + 3))
# right column (x 140): the throats, 16 tall: the flat stack's slot at the bottom, the lens at the top (the turret's throat, shortened)
TX = 140.0; TYS = (0.0, 26.0, 52.0); hw = S.BLADE_W / 2 + 0.5; H = 16.0
sl = S.STACK / 2 + 0.3
for k, R in enumerate(C["T"]):
    y = TYS[k]; lh = R - math.sqrt(R ** 2 - (S.BLADE_W / 2) ** 2) + S.BLADE_T + S.CABLE_T / 2 + 0.4
    ow = lh + 2.2; wx = hw + 2.2                                                  # walls 2.2 all round the path (a block's thick foot made a blob on a neck)
    block = trimesh.convex.convex_hull(np.array([[TX + sx * wx, y + sy * (sl + 2.2), z] for sx in (-1, 1) for sy in (-1, 1) for z in (T - 0.01, T + 2.0)] +
                                                [[TX + sx * wx, y + sy * ow, T + H] for sx in (-1, 1) for sy in (-1, 1)]))
    path = trimesh.convex.convex_hull(np.array([[TX + sx * hw, y + sy * sl, -1.0] for sx in (-1, 1) for sy in (-1, 1)] +
                                               [[TX + sx * hw, y + sy * sl, T + 2.0] for sx in (-1, 1) for sy in (-1, 1)] +
                                               [[TX + sx * hw, y + sy * lh, T + H + 0.5] for sx in (-1, 1) for sy in (-1, 1)]))
    add.append(minus(block, [path])); cut.append(path)
    pads.append(sbox(TX - wx - 1.0, y - ow - 1.0, TX + wx + 1.0, y + ow + 1.0))
    label(f"T{k + 1}", TX - hw - 9.5, y)
    pads.append(sbox(TX - hw - 10.0, y - 2.0, TX - wx, y + 2.0))
pads.append(sbox(TX - 3, TYS[0], TX + 3, TYS[2])); pads.append(sbox(106, -4, TX, 4))
# right column, lower (from y 76): the camera case's gauge, ribs 12 x 2.5, 5 tall; the case stands on edge in a gap, front to back across it
CY0 = 76.0; y = CY0; ribs = []
for k, d in enumerate(C["C"]):
    ribs.append(y); g = d + 0.6
    label(f"C{k + 1}", TX - 15.0, y + 2.5 + g / 2)
    y += 2.5 + g
ribs.append(y)
for yr in ribs:
    add.append(extrude(sbox(TX - 6, yr, TX + 6, yr + 2.5), T, T + 5)); pads.append(sbox(TX - 8, yr - 1, TX + 8, yr + 3.5))
pads.append(sbox(TX - 3, ribs[0], TX + 3, ribs[-1] + 2.5)); pads.append(sbox(TX - 18, ribs[0], TX - 12, ribs[-1] + 2.5))
pads.append(sbox(TX - 3, TYS[2], TX + 3, CY0)); pads.append(sbox(MX[2], MY - 3, TX, MY + 3))

foot = unary_union(pads).buffer(0.5).buffer(-0.5)
if foot.geom_type != "Polygon":
    raise SystemExit(f"the coupon's plate is in {len(foot.geoms)} pieces")
plate = extrude(foot, 0, T)
emb = [extrude(g, T - 0.01, T + EMB) for t in labels for g in getattr(t, "geoms", [t])]
cp = minus(union([plate] + add + emb), cut)
cp.apply_translation(-cp.bounds[0] * [1, 1, 0])
assert max(cp.extents[:2]) <= 178.0, cp.extents
save("scout_v4_probe_coupon", cp, cp.copy().apply_translation([0, -400, 0]),
     f"flat, no supports; {cp.extents[0]:.0f} x {cp.extents[1]:.0f}; ~{cp.volume / 1000 / 0.7 + 5:.0f} min E (0.7 cm3/min, the 10 Nov coupon's 17 cm3 in ~25, + 5 for the tall arcs' travel)")
