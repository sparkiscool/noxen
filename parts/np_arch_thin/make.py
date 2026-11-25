"""Non-planar hand-break specimen (Spark 24 Nov: the solid bridge came out looking flat and was unbreakable; make it thin, with a clear weak section loaded
across the layers, so the flat and the non-planar print can be told apart by breaking them). Replaces np_arch_bridge as the comparison. CONCEPT test, not a flight part.

Shape: a thin arched band, L 120 x W 12, two solid feet 20 long and 8 high at their outer ends, flat on the bed. The top is a circular arc R 400 (sagitta 4.5,
never steeper than 8.6 deg: inside the A1 mini's 10 deg design slope, tools/nonplanar.py). Under it a tunnel L 80 whose roof is the same arc 3.6 mm lower: the weak
section is a 3.6 mm thick, 12 wide, 80 long curved band (feet 8+ mm thick, so it breaks in the band, never in a foot). Tunnel open on both sides, supports reach by hand.
Why it separates the two prints: a flat layer meets a band sloped up to 8.6 deg over only 3.6 / tan(8.6) = 24 mm of its length, so in the planar print the tension along the band
has to hop from one layer to the next every ~24 mm (about 18 layer lines under the crown, stair-steps on both faces); the non-planar layers follow the arc foot to foot, so the
band is one unbroken stack. Hand calc (E): Z = W t^2 / 6 = 12 * 3.6^2 / 6 = 25.9 mm3; feet on two rests 80 mm apart, push on the crown: F = 4 sigma Z / L = 4 * 25 * 25.9 / 80 = 32 N
(about 3.2 kg) if layer lines govern (planar), 4 * 46 * 25.9 / 80 = 60 N at PCTG's own strength (non-planar); a hand pushes 100 N. The feet rest on supports, nothing holds them
sideways (an arch held at its feet would carry the push as thrust and not break). Print twice from the same plate settings: planar, and non-planar (conform+wave). Nothing is claimed.
Fit table: no holes, bores or sockets; the tunnel is a clear opening, not a fit. Minimum wall 3.6 mm (2.4 failed the solid check as a neck under 3.0)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shapely.geometry import Polygon
from scout_v4_probe_mech_lib import extrude, save  # noqa: E402

L, W = 120.0, 12.0
R = 400.0
T_END = 8.0           # top height at the feet' outer ends
RING = 3.6            # the weak band: constant thickness (top arc to roof arc)
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
save("np_arch_thin", m, m.copy(), f"L {L:.0f} W {W:.0f}, crown {top(0):.1f}, feet {T_END:.0f}, band {RING}, tunnel {2*TUN:.0f} long; flat feet down; print one planar, one non-planar, break by hand")
