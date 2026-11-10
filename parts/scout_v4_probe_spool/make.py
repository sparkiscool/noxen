"""Scout v4 probe spool with its 40 T gear, one print (10 Nov bench; tools/scout_v4_shape.SPOOL / GEAR, tools/probe_mast_sim.py).
Drawn, not topology: python parts/scout_v4_probe_spool/make.py writes parts/scout_v4_probe_spool/part.stl (drone frame) and
print/scout_v4_probe_spool.stl (print pose).
  hub r 14 between two O31.8 flanges 1.9 thick, 19.6 apart (the 19 mm blades, 0.3 a side); the 40 T gear (module 0.8, x -0.35,
  0.15 backlash; parts/scout_v4_probe_mech_lib.py) 3 wide outside the +y flange; the drum's hub r 4 outside the -y flange, where
  the ribbon's loose coil starts (the turret's drum, r 11, 3.5 turns of room for 1.6)
  the stack's clamp: one slot 1.2 wide (B 0.13 + the ribbon 0.6 + A 0.13 and 0.34 for the epoxy) 8 deep into the hub at 45 deg to
  its surface, backwards from the winding, across the full 19.6: the three ends go in together (B on the hub's side), epoxy
  (scuff the paint); the ribbon leaves the stack at the slot's end, radially in to r 5 and along the axis out of the drum hub's end
  bore 3.4: the spool turns on the M3 x 40 axle (fixed in the turret's bosses, a nyloc on the -y side)
Print: +y face (the gear) down, no supports (the -y flange overhangs the hub by 1.9: bridged), 0.16 layers, 4 walls, 100 % infill."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import S, gear_2d, extrude, cyl, union, minus, to_drone, save, SHIFT  # noqa: E402
import numpy as np, trimesh
from shapely.geometry import Polygon

HUB, FL, Y0, Y1 = S.SPOOL["hub"], S.SPOOL["r"], S.SPOOL["y0"], S.SPOOL["y1"]
GAP = S.BLADE_W + 0.6
G0, G1 = S.GEAR["y"]
DRUM0 = S.DRUM["y0"] + 0.5
SLOT_W, SLOT_L, SLOT_A = 1.2, 8.0, math.radians(45.0)

parts = [cyl(FL, Y0, -GAP / 2), cyl(HUB, -GAP / 2, GAP / 2), cyl(FL, GAP / 2, Y1),
         extrude(gear_2d(S.GEAR["spool_t"], SHIFT["spool"]), G0, G1), cyl(4.0, DRUM0, Y0)]
body = union(parts)
# the clamp slot at local angle 0 (drone: the +x side at the axis's height); the coil winds clockwise in local xy (drone xz counter-clockwise)
P = np.array([HUB, 0.0]); d = np.array([-math.sin(SLOT_A), math.cos(SLOT_A)]); n = np.array([d[1], -d[0]])
q = [P + n * 0.3, P - n * (SLOT_W - 0.3), P - n * (SLOT_W - 0.3) + d * SLOT_L, P + n * 0.3 + d * SLOT_L]
slot = extrude(Polygon([tuple(v) for v in q]).buffer(0), -GAP / 2 - 0.01, GAP / 2 + 0.01)
Q = P + d * (SLOT_L - 1.0); u = Q / np.linalg.norm(Q); w = np.array([-u[1], u[0]])
rad = [5.0 * u - w * 0.7, 5.0 * u + w * 0.7, Q + w * 0.7, Q - w * 0.7]
radial = extrude(Polygon([tuple(v) for v in rad]).buffer(0), -2.0, 2.0)                     # the ribbon in toward the axis, 4 tall (its width)
ax = [u * 4.6 - w * 2.1, u * 4.6 + w * 2.1, u * 6.0 + w * 2.1, u * 6.0 - w * 2.1]
axial = extrude(Polygon([tuple(v) for v in ax]).buffer(0), DRUM0 - 0.01, 2.0)               # then along the axis out of the drum hub's end
bore = cyl(1.7, DRUM0 - 1, G1 + 1, n=48)
spool = minus(body, [slot, radial, axial, bore])
pr = spool.copy(); pr.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])); pr.apply_translation([0, 0, G1])   # gear face down on the bed
save("scout_v4_probe_spool", pr, to_drone(spool, *S.SPOOL["c"]), "gear face down, no supports, ~1 h 10 E")
