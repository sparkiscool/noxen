"""Scout v4 probe pressure roller tube (10 Nov bench; tools/scout_v4_shape.ROLLER, tools/probe_mast_sim.py blossom).
Drawn: python parts/scout_v4_probe_roller/make.py writes part.stl (drone frame, on the full coil) and print/scout_v4_probe_roller.stl.
  14 Nov (Spark: no bearings, the coupon's B dropped): an O8 x 19 tube (the blades' width) turning straight on its M3 x 40 axle in a
  plain bore sized like the spool's (the coupon's A1, 3.3: slides on free), 2.35 of wall; PCTG on steel, mu ~0.25 (E): its drag is
  tools/scout_v4_shape.roller_drag(), ~20 N mm at the spool against the N20's 417..612. The axle's ends ride the turret's radial slots,
  two 4 x 0.5 x 15 springs (uxcell B076LQCBR1) pushing them: 12 N on the full coil, 7.8 empty (E). A drop of light oil on the axle.
Print: on end, axis vertical (a round bore), no supports, 0.12 layers."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import S, cyl, minus, to_drone, save  # noqa: E402

R, W = S.ROLLER["r"], S.ROLLER["w"]
tube = minus(cyl(R, -W / 2, W / 2), [cyl(S.ROLLER["bore"] / 2, -W / 2 - 1, W / 2 + 1, n=48)])   # the coupon's A1
rr = S.SPOOL["hub"] + S.COIL + R; a = math.radians(S.ROLLER["ang"]); cx, cz = S.SPOOL["c"]
pr = tube.copy(); pr.apply_translation([0, 0, W / 2])
save("scout_v4_probe_roller", pr, to_drone(tube, cx + rr * math.cos(a), cz + rr * math.sin(a)), "on end, ~10 min")
