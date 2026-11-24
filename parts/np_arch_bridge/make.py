"""Non-planar break-test specimen (Spark 23 Nov: a real load-carrying arch, curved layers going deep through the part along the load path, not a tab).
Drawn: python parts/np_arch_bridge/make.py writes parts/np_arch_bridge/part.stl and print/np_arch_bridge.stl. CONCEPT/PROTOTYPE of a test, not a flight part.

Shape: a segmental arch bridge, L 120 x W 30, two flat feet on the bed. The top (extrados) is a circular arc R 400 (sagitta 4.5, never steeper than 8.6 deg: inside the
A1 mini's 10 deg design slope, tools/nonplanar.py); a through tunnel L 80 under it (open both sides along W, nothing to support inside, hand-reachable) whose roof
(intrados) is the same arc 7.0 mm lower, so the load ring is a constant 7.0 mm thick curved beam 80 mm long that rises to the crown and comes down into two solid
feet 20 mm long. Conform bends the layers of the whole ring with that arc, so every layer in the ring runs along the thrust line from foot to foot, through the full
7 mm thickness, not only in a skin; the planar print has flat layers that cross the arc at the crown (stair-stepped top, layer lines across the load path).
Feet: 12.0 high at the ends, crown 16.5 above the bed; tunnel floor is the bed, tunnel height 7.5 at the springings, 9.5 at the crown.
Test (identical for both): span 100 mm on two 10 mm round supports under the feet, 10 mm roller on the crown, load to failure. Hand calc (E): ring section
W 30 x t 7.0, Z = W t^2 / 6 = 245 mm3; as a curved beam in plain bending M = F L / 4 -> F = 4 sigma Z / L = 245 N at 25 MPa working (PCTG Xt 46), a lower bound: the
arch thrust carries part of it as compression, so the failure load should be higher; the planar one should fail at a layer line under the stair-stepped crown first.
Print twice from the same plate settings: planar, and non-planar (conform+wave). Nothing here is a design claim.
Fit table: no holes, bores or sockets; the tunnel is a clear opening, not a fit."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shapely.geometry import Polygon
from scout_v4_probe_mech_lib import extrude, save  # noqa: E402

L, W = 120.0, 30.0
R = 400.0
T_END = 12.0          # top height at the feet' outer ends
RING = 7.0            # constant ring thickness (top arc to roof arc)
TUN = 40.0            # half-length of the tunnel


def top(x):
    return T_END + (math.sqrt(R * R - x * x) - math.sqrt(R * R - (L / 2) ** 2))


SAG = top(0) - T_END
xs = np.linspace(-L / 2, L / 2, 241)
outer = [(x, top(x)) for x in xs]
xt = np.linspace(-TUN, TUN, 161)
roof = [(x, top(x) - RING) for x in xt]
prof = Polygon([(-L / 2, 0.0), (-TUN, 0.0)] + roof + [(TUN, 0.0), (L / 2, 0.0)] + outer[::-1])
assert prof.is_valid and len(prof.interiors) == 0, "profile must be one clean outline"
m = extrude(prof, 0.0, W)
m.apply_transform(__import__("trimesh").transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
m.apply_translation([0, W / 2, 0])
m.apply_translation([0, 0, -m.bounds[0][2]])
save("np_arch_bridge", m, m.copy(), f"L {L:.0f} W {W:.0f}, crown {top(0):.1f}, feet {T_END:.0f}, ring {RING:.0f}, tunnel {2*TUN:.0f} long; flat feet down, no support; print one planar, one non-planar")
