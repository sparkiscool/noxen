"""Scout v4 probe pinion, 12 T module 0.8, x +0.35 (no undercut; the pair keeps 20.8 centres), on the N20's 3 mm D shaft (10 Nov bench).
Drawn: python parts/scout_v4_probe_pinion/make.py writes part.stl (drone frame) and print/scout_v4_probe_pinion.stl.
  teeth y 11.7..14.7 (tools/scout_v4_shape.GEAR), a hub boss O7 to 15.4 (3.7 of grip on the shaft, clear of the tip bearing at 15.7)
  D bore 3.1 / 2.6 across the flat (the shaft 3.0 / 2.5): press on, a drop of CA; root r 4.08, so 2.5 of wall round the bore
Print: the hub boss up, teeth face down, no supports, 0.12 layers for the teeth; print 2 (one spare: it is the part that strips)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import S, gear_2d, extrude, cyl, union, minus, to_drone, save, SHIFT  # noqa: E402
import trimesh
from shapely.geometry import Point, box as sbox

G0, G1 = S.GEAR["y"]
pz = S.SPOOL["c"][1] - (S.GEAR["spool_t"] + S.GEAR["pinion_t"]) * S.GEAR["m"] / 2
body = union([extrude(gear_2d(S.GEAR["pinion_t"], SHIFT["pinion"]), G0, G1), cyl(3.5, G1 - 0.1, G1 + 0.7)])
dbore = extrude(Point(0, 0).buffer(1.55, 48).intersection(sbox(-2, -2, 1.05, 2)), G0 - 1, G1 + 2)
pin = minus(body, [dbore])
pr = pin.copy(); pr.apply_translation([0, 0, -G0])
save("scout_v4_probe_pinion", pr, to_drone(pin, S.SPOOL["c"][0], pz), "teeth down, ~10 min, print 2")
