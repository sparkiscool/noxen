"""Scout v4 probe head coupon (15 Nov, PROTOTYPE fits: Spark couldn't fit the head frame and the cradle together, the camera had no way
into the sleeve, a tab post snapped). Every head fit no listing gives, in three sizes with embossed labels: the answer is one line,
e.g. "C2 P2 H1 K2". The picks go into parts/scout_v4_head_lib.py (one edit), then both make.py rerun. Nothing to measure.
Drawn: python parts/scout_v4_head_coupon/make.py writes print/scout_v4_head_coupon.stl and part.stl (off the drone).
  C1-3  the cradle's sleeve: square frames 36.6 / 36.9 / 37.2 inside (0.30 / 0.45 / 0.60 a side round the 36 case), 6 tall:
        drop the camera case in, front first: the snuggest it slides into and out of without force (C2 is drawn now)
  P1-3  the SG90 tab screw: a column as the ring's (6.35 x 11.5 on a 6 mm root, 2.5 over the pilot), a horizontal pilot O1.5 / 1.7 / 1.9
        (P2 drawn) 11.5 deep: drive one of the SG90's mounting self-tappers fully in and out of each: the smallest it goes into all
        the way without cracking the column, and grips
  H1-3  the frame to cradle faces (the cradle between the idler bracket and the horn): slots 10.3 / 10.5 / 10.7 wide (0.3 / 0.5 /
        0.7 over), 8 deep, and the loose 10.0 tongue printed beside them (its 10 is along the print's z, as the cradle's span is):
        the tightest slot the tongue drops down freely (H2 drawn)
  K1-3  the horn's channel: 1.0 deep, 7.4 / 7.6 / 7.8 wide (K2 drawn), open at both ends: slide the SG90's single-arm horn along it
        hub first, arm face down: the narrowest it slides along without force
Print: flat as laid (the plate down), no supports, PCTG 0.2 mm, 4 walls, ~1 h 45 E; the H tongue is its own file
(print/scout_v4_head_coupon_tongue.stl, 20 x 8 x 10, flat, ~10 min)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "parts")); sys.path.insert(0, ROOT)
import numpy as np, trimesh                                                                     # noqa: E402
from shapely.geometry import Polygon, box as sbox                                              # noqa: E402
from shapely import affinity                                                                    # noqa: E402
from matplotlib.textpath import TextPath                                                        # noqa: E402
from matplotlib.font_manager import FontProperties                                              # noqa: E402
from scout_v4_probe_mech_lib import extrude, union, minus, save, PCTG                            # noqa: E402

T, EMB = 3.0, 0.6
FONT = FontProperties(family="DejaVu Sans", weight="bold")
COUPON = dict(C=(36.6, 36.9, 37.2), P=(1.5, 1.7, 1.9), H=(10.3, 10.5, 10.7), K=(7.4, 7.6, 7.8))


def text2d(s, x, y, size=4.5):
    tp = TextPath((0, 0), s, size=size, prop=FONT)
    g = Polygon()
    for r in sorted([Polygon(p).buffer(0) for p in tp.to_polygons() if len(p) >= 3], key=lambda q: -q.area):
        g = g.symmetric_difference(r)
    b = g.bounds
    return affinity.translate(g, x - (b[0] + b[2]) / 2, y - (b[1] + b[3]) / 2)


def bx(x0, x1, y0, y1, z0, z1):
    return trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]])


add, cut, labels, plate = [], [], [], []


def label(s, x, y):
    t = text2d(s, x, y); labels.append(t); plate.append(t.envelope.buffer(1.5, join_style=2))


# row C (y 0..): three square frames, 2.2 walls, 6 tall, each on its own ring of plate
for k, s in enumerate(COUPON["C"]):
    x0 = 45.0 * k; o = s + 4.4
    ring = sbox(x0, 0, x0 + o, o).difference(sbox(x0 + 2.2, 2.2, x0 + 2.2 + s, 2.2 + s))
    add.append(extrude(ring, T - 0.01, T + 6)); plate.append(ring)
    label(f"C{k + 1}", x0 + o / 2, -5.0)
plate.append(sbox(0, -4, 131, 4)); plate += [sbox(x - 4, 0, x + 4, 40) for x in (2.0, 129.0)]                                                                # row C's rail
# row P (y 50): the columns, a horizontal pilot along y at 12 over the plate
PY = 50.0
for k, d in enumerate(COUPON["P"]):
    x = 8.0 + 19.0 * k
    add.append(bx(x - 3.175, x + 3.175, PY, PY + 11.5, T - 0.01, T + 13.5)); plate.append(sbox(x - 8, PY - 4, x + 8, PY + 15.5))   # the ring's column section
    add.append(bx(x - 8, x + 8, PY - 4, PY + 15.5, T - 0.01, 6.0))                                    # on a 6 mm pedestal (its root)
    cut.append(trimesh.creation.cylinder(radius=d / 2, segment=[(x, PY - 1, T + 10), (x, PY + 13, T + 10)], sections=32))
    label(f"P{k + 1}", x, PY - 7.5)
plate.append(sbox(0, 36, 131, 44)); plate += [sbox(x - 4, 40, x + 4, PY - 3) for x in (8.0, 27.0, 46.0)]
# row H (y 50, x 62..): slots 8 deep in 3-wall blocks 20 long
for k, w in enumerate(COUPON["H"]):
    x = 62.0 + 22.0 * k
    add.append(bx(x, x + w + 6, PY, PY + 20, T - 0.01, T + 8)); cut.append(bx(x + 3, x + 3 + w, PY - 1, PY + 21, T, T + 9))
    plate.append(sbox(x - 1, PY - 1, x + w + 7, PY + 21))
    label(f"H{k + 1}", x + w / 2 + 3, PY - 5.0)
plate += [sbox(x - 4, 40, x + 4, PY) for x in (68.0, 90.0, 112.0)]
# row K (y 80): channel blocks 14 wide, 3 thick over the plate, the channel along y
KY = 80.0
for k, w in enumerate(COUPON["K"]):
    x = 8.0 + 22.0 * k
    add.append(bx(x - 7, x + 7, KY, KY + 24, T - 0.01, T + 3)); cut.append(bx(x - w / 2, x + w / 2, KY - 1, KY + 25, T + 2, T + 4))
    plate.append(sbox(x - 8, KY - 1, x + 8, KY + 25))
    label(f"K{k + 1}", x, KY - 5.0)
plate.append(sbox(0, 70, 60, 78)); plate += [sbox(-4, 40, 4, 74)] + [sbox(x - 4, 74, x + 4, KY) for x in (8.0, 30.0, 52.0)]
from shapely.ops import unary_union                                                             # noqa: E402
plate += [sbox(-4, 36, 137, 74), sbox(-4, 70, 62, 106)]                                          # rows P / H and K on one continuous plate each (no thin rails)
pl = unary_union(plate); base = union([extrude(g, 0.0, T) for g in getattr(pl, "geoms", [pl])])
lab = [extrude(g, T - 0.01, T + EMB) for t in labels for g in getattr(t, "geoms", [t])]
coupon = minus(union([base] + add + lab), cut)
whole = coupon
save("scout_v4_head_coupon", whole, whole, "flat, no supports, ~1 h 45 (and its H tongue)")
tongue = bx(0.0, 20.0, 0.0, 8.0, 0.0, 10.0)                                                       # the H tongue, its own print: 10.0 along z
t_lab = [extrude(g, 10.0 - 0.01, 10.0 + EMB) for g in getattr(text2d("H", 10.0, 4.0, size=4.0), "geoms", [text2d("H", 10.0, 4.0, size=4.0)])]
tg = union([tongue] + t_lab); assert tg.is_watertight and tg.is_volume
tg.export(os.path.join(ROOT, "print", "scout_v4_head_coupon_tongue.stl"))                       # its own print file only (as the roller's saddles)
import subprocess                                                                               # noqa: E402
sc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solid_check.py"), os.path.join(ROOT, "print", "scout_v4_head_coupon_tongue.stl")], capture_output=True, text=True)
print(f"scout_v4_head_coupon_tongue: {tg.volume / 1000 * PCTG:.1f} g, solid check {'PASS' if sc.returncode == 0 else 'FAIL'}; flat, no supports, ~10 min")
from topo.build import render                                                                   # noqa: E402
render(whole, os.path.join(HERE, "preview.png"), title="scout_v4_head_coupon", info=f"{whole.volume / 1000 * PCTG:.1f} g | C P H K, three sizes each | flat, no supports")
