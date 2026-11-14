"""Scout v4 probe pinion, 12 T module 0.8, x +0.35 (no undercut), on the N20's 3 mm D shaft (10 Nov bench); the pair on the coupon's G2 20.9 centres.
Drawn: python parts/scout_v4_probe_pinion/make.py writes part.stl (drone frame) and print/scout_v4_probe_pinion.stl.
  teeth y 11.7..14.7 (tools/scout_v4_shape.GEAR), a hub boss O7 to 15.4 (3.7 of grip on the shaft, clear of the tip bearing at 15.7)
  D bore 3.3 / 2.8 across the flat (14 Nov coupon: D1, D2 would not go on the 3.0 / 2.5 shaft, D3 3.2 / 2.7 too hard to press home;
  one step past D3): press on, a drop of CA; root r 4.08, so 2.4 of wall round the bore
Print: the hub boss up, teeth face down, no supports, 0.12 layers for the teeth; print 2 (one spare: it is the part that strips)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import S, gear_2d, extrude, cyl, union, minus, to_drone, save, SHIFT  # noqa: E402
import trimesh
from shapely.geometry import Point, box as sbox

G0, G1 = S.GEAR["y"]
pz = S.SPOOL["c"][1] - S.GEAR["c"]
body = union([extrude(gear_2d(S.GEAR["pinion_t"], SHIFT["pinion"]), G0, G1), cyl(3.5, G1 - 0.1, G1 + 0.7)])
dd, df = S.pick("D")                                                              # the coupon's D: diameter, across the flat
dbore = extrude(Point(0, 0).buffer(dd / 2, 48).intersection(sbox(-2, -2, df - dd / 2, 2)), G0 - 1, G1 + 2)
pin = minus(body, [dbore])
pr = pin.copy(); pr.apply_translation([0, 0, -G0])
save("scout_v4_probe_pinion", pr, to_drone(pin, S.SPOOL["c"][0], pz), "teeth down, ~10 min, print 2")
