"""Scout v4 probe fit coupon (10 Nov bench): every uncertain fit of the mechanism in three sizes, and the gear pair on three centres.
Drawn: python parts/scout_v4_probe_coupon/make.py writes print/scout_v4_probe_coupon.stl (and part.stl, off the drone, for the record).
A 2.5 plate, labelled by position (left to right = small, nominal, large):
  row 1  the pinion's D bore on the N20's shaft:          3.0 / 2.5,  3.1 / 2.6 (drawn in the pinion),  3.2 / 2.7 (across the flat)
  row 2  the MR63ZZ's seat (6.0 race):                     6.00,  6.05 (drawn in the roller),  6.10, each 2.7 deep
  row 3  the spool's bore on the M3 axle:                  3.3,  3.4 (drawn in the spool),  3.5
  row 4  the gear pair: a O3.0 peg for the spool's bore and a D peg (3.0 / 2.5) for the pinion, centres 20.8 (drawn), 20.9, 21.0:
         set the printed spool and pinion on each pair and spin; the first pair that turns smoothly with no tight spot is the
         centre distance; if it is not 20.8, the turret's pinion seat moves (tell the probe stream: one number)
Print: flat, no supports, 0.16 layers, ~25 min."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import cyl, extrude, union, minus, save  # noqa: E402
import trimesh
from shapely.geometry import Point, box as sbox

T = 2.5
plate = extrude(sbox(-6, -6, 126, 40), 0, T)
add, cut = [], []
for k, (d, f) in enumerate(((3.0, 2.5), (3.1, 2.6), (3.2, 2.7))):              # row 1, D bores in O9 bosses 5 tall
    x = 6 + 12 * k; add.append(cyl(4.5, T, T + 5, x, 0))
    r = d / 2; cut.append(extrude(Point(x, 0).buffer(r, 48).intersection(sbox(x - 2, -2, x - r + f, 2)), -1, T + 6))
for k, d in enumerate((6.0, 6.05, 6.1)):                                        # row 2, bearing seats in O10.1 bosses
    x = 6 + 13 * k; add.append(cyl(5.05, T, T + 3.5, x, 12))
    cut += [cyl(d / 2, T + 0.8, T + 4, x, 12, n=64), cyl(2.0, -1, T + 4, x, 12)]
for k, d in enumerate((3.3, 3.4, 3.5)):                                         # row 3, axle bores in O8 bosses
    x = 6 + 12 * k; add.append(cyl(4.0, T, T + 5, x, 24)); cut.append(cyl(d / 2, -1, T + 6, x, 24, n=48))
for k, c in enumerate((20.8, 20.9, 21.0)):                                      # row 4: the pair's pegs, on the plate's right half
    x = 55 + 26 * k; y0 = 0.0
    add.append(cyl(1.5, T, T + 14, x, y0 + 0.0, n=32))                           # the spool's peg (its bore 3.4, 26 long: 14 of it)
    add.append(extrude(Point(x, y0 + c).buffer(1.5, 32).intersection(sbox(x - 2, y0 + c - 2, x - 1.5 + 2.5, y0 + c + 2)), T, T + 5))   # the pinion's D peg
cp = minus(union([plate] + add), cut)
save("scout_v4_probe_coupon", cp, cp.copy().apply_translation([0, -400, 0]), "flat, ~25 min")
