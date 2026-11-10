"""Scout v4 probe bench base (10 Nov, Spark: "start printing the parts needed for the bench test"): stands the printed turret as the
deck does, for the mechanism test on a table. Drawn: python parts/scout_v4_probe_bench_base/make.py writes print/scout_v4_probe_bench_base.stl.
  four O10 posts at the turret's floor bolts (parts/scout_v4_units_lib.TURRET_BOLTS), their tops on the deck's top (z 24.5), each with
  an M3 x 5.7 heat-set insert (hole 4.0 x 6.0): the turret bolts down into them with 4 x M3 x 10 from above, as on the deck
  the N20 hangs below the turret's floor (z 14.2 up): the plate is open under it; the 40 T gear clears the plate by 7.7
  a 4 mm plate 130 x 70 with two clamp tabs at its ends (C-clamps to the bench, or double-sided tape as the saw guide)
Print: plate down, no supports, 0.2 layers, 4 walls, 30 % infill, ~1 h 40 E."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scout_v4_probe_mech_lib import S, cyl, extrude, union, minus, save  # noqa: E402
from shapely.geometry import box as sbox
from parts.scout_v4_units_lib import TURRET_BOLTS, DZ1

Z0, T = 12.0, 4.0
cx = S.SPOOL["c"][0]; g0 = S.N20_FACE
plate = extrude(sbox(cx - 65, -35, cx + 65, 33), Z0, Z0 + T)
posts = [cyl(5.0, Z0 + T - 0.01, DZ1, x, y) for x, y in TURRET_BOLTS]
cuts = [cyl(2.0, DZ1 - 6.0, DZ1 + 1, x, y, n=48) for x, y in TURRET_BOLTS]
cuts.append(extrude(sbox(cx - 8.5, g0 - 30.0, cx + 8.5, g0 + 11.0), Z0 - 1, Z0 + T + 1))    # open under the N20 (can, gearbox, leads) and the bench turret's block round its pinion
cuts += [extrude(sbox(x - 3.0, -12.0, x + 3.0, 12.0), Z0 - 1, Z0 + T + 1) for x in (cx - 50, cx + 50)]   # the clamps' slots in the tabs
base = minus(union([plate] + posts), cuts)
pr = base.copy(); pr.apply_translation([0, 0, -Z0])
save("scout_v4_probe_bench_base", pr, base, "plate down, ~1 h 40")
