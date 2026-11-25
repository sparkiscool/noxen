"""Non-planar hand-snap specimen v2 (Spark 24 Nov: the last pair could not be broken by hand; this one must snap, and show the curved layers). CONCEPT test, not a flight part.

Shape: a long arched band, L 140 x W 12, two solid feet 20 long and 8 high at the outer ends, flat on the bed. The top is a circular arc R 500 (sagitta 4.9, 8.0 deg at the
feet: inside the A1 mini 10 deg design slope and its 5 mm relief; R 410 with sagitta 6.0 failed the layer gate, 0.4 mm layers). Under it a tunnel L 100 whose roof is the same arc 3.6 mm lower: the weak
section is a 3.6 mm thick, 12 wide, 100 long curved band (feet 8+ mm, so it breaks in the band). Tunnel open on both sides, supports reach by hand.
Why it separates the two prints: a flat layer meets a band sloped up to 8.0 deg over only 3.6 / tan(8.0) = 26 mm of its length, so the planar print shows ~25 stair steps on the
top (4.9 mm / 0.2 layer) and the tension along the band must hop from layer to layer every ~26 mm; the non-planar layers follow the arc foot to foot: smooth unbroken top lines, the
band one continuous stack. Infill: the band is 3.6 thick, 3 walls (2.5 mm) leave ~1 mm core, so infill setting hardly changes it; 15 % was tried and made 0.4 mm internal-bridge layers the non-planar gate refuses, so 100 %.
Hand calc (E): Z = W t^2 / 6 = 12 * 3.6^2 / 6 = 25.9 mm3; feet on two rests 100 mm apart, push on the crown: F = 4 sigma Z / L = 4 * 25 * 25.9 / 100 = 26 N (2.6 kg) if the layer lines
govern (planar), 4 * 46 * 25.9 / 100 = 48 N (4.8 kg) at PCTG's own strength (non-planar); a hand pushes 100 N. Nothing is claimed.
Fit table: no holes, bores or sockets. Minimum wall 3.6 mm (3.0 and 3.2 failed the solid check as a neck)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from shapely.geometry import Polygon
from scout_v4_probe_mech_lib import extrude, save  # noqa: E402

L, W = 140.0, 12.0
R = 500.0
T_END = 8.0           # top height at the feet' outer ends
RING = 3.6            # the weak band: constant thickness (top arc to roof arc)
TUN = 50.0            # half-length of the tunnel


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
save("np_arch_snap", m, m.copy(), f"L {L:.0f} W {W:.0f}, crown {top(0):.1f}, feet {T_END:.0f}, band {RING}, tunnel {2*TUN:.0f} long, 8.0 deg; flat feet down; print one planar, one non-planar, snap by hand")
