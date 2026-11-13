"""Scout v4 probe pressure roller tube (10 Nov bench; tools/scout_v4_shape.ROLLER, tools/probe_mast_sim.py blossom).
Drawn: python parts/scout_v4_probe_roller/make.py writes part.stl (drone frame, on the full coil) and print/scout_v4_probe_roller.stl.
  O10.1 x 19 (the blades' width), an MR63ZZ (3 x 6 x 2.5, uxcell B075CLH2WM) pressed into a 6.05 x 2.7 seat at each end (2.0 of wall
  round it), O4 through between them; it turns on an M3 x 40 axle whose ends ride the turret's radial slots, two 4 x 0.5 x 15 springs
  (uxcell B076LQCBR1) pushing them: 12 N on the full coil, 7.8 empty (E)
Print: on end, axis vertical (round seats), no supports, 0.12 layers; press the bearings with a vice and a socket on the outer race."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import S, cyl, minus, to_drone, save  # noqa: E402

R, W = S.ROLLER["r"], S.ROLLER["w"]
sr = S.pick("B") / 2                                                              # the coupon's B
tube = minus(cyl(R, -W / 2, W / 2), [cyl(sr, -W / 2 - 1, -W / 2 + 2.7, n=64), cyl(sr, W / 2 - 2.7, W / 2 + 1, n=64), cyl(2.0, -W / 2, W / 2, n=48)])
rr = S.SPOOL["hub"] + S.COIL + R; a = math.radians(S.ROLLER["ang"]); cx, cz = S.SPOOL["c"]
pr = tube.copy(); pr.apply_translation([0, 0, W / 2])
save("scout_v4_probe_roller", pr, to_drone(tube, cx + rr * math.cos(a), cz + rr * math.sin(a)), "on end, ~15 min")
