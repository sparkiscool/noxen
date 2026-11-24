"""Non-planar test coupon (Spark 23 Nov: a gently curved part that shows the non-planar difference and can be loaded to failure; arm-cover-like).
Drawn: python parts/np_arch_coupon/make.py writes parts/np_arch_coupon/part.stl and print/np_arch_coupon.stl. CONCEPT/PROTOTYPE of a test, not a flight part.

Shape: a flat-bottomed bar L 120 x W 30 whose top is a circular arch (radius R 400, sagitta 4.5 mm over the half-span 60, so the top is never steeper than
8.6 deg at the ends: inside the A1 mini's 10 deg design slope, tools/nonplanar.py). Thickness 6.0 at the ends, 10.5 at the crown. Flat face down, no overhang,
no support, nothing to remove. Print it twice from the same plate settings: planar (stair-stepped top, layers flat) and non-planar (conform: the layers of the
upper part follow the arch, the top skin is one continuous line along the surface).
Test: 3-point bend, span 100 mm, a 10 mm roller on the crown, load to failure, same for both. Hand calc (E): sigma = 3 F L / (2 W t^2) at the crown, t 10.5:
Z = W t^2 / 6 = 551 mm3, M = F L / 4; PCTG layers along the bend, 25 MPa working (46 Xt) -> F = 4 sigma Z / L = 550 N expected order; the planar one fails
at a layer line under the stair steps first, the conform one should carry more or fail later (that is the test; nothing here is a design claim).
Fit table: no holes, bores or sockets (nothing to fit)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shapely.geometry import Polygon
from scout_v4_probe_mech_lib import extrude, save  # noqa: E402

L, W = 120.0, 30.0
R = 400.0
T_END = 6.0
SAG = R - math.sqrt(R * R - (L / 2) ** 2)         # 4.5 mm
xs = np.linspace(-L / 2, L / 2, 241)
top = [(x, T_END + (math.sqrt(R * R - x * x) - math.sqrt(R * R - (L / 2) ** 2))) for x in xs]
prof = Polygon([(-L / 2, 0.0), (L / 2, 0.0)] + top[::-1])
m = extrude(prof, 0.0, W)                          # profile in x, y; extruded along z
m.apply_transform(trimesh_rot := __import__("trimesh").transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))   # profile y -> z, width along -y
m.apply_translation([0, W / 2, 0])
m.apply_translation([0, 0, -m.bounds[0][2]])
save("np_arch_coupon", m, m.copy(), f"L {L:.0f} W {W:.0f}, crown {T_END + SAG:.1f} thick, ends {T_END:.0f}; flat face down, no support; print one planar, one non-planar")
