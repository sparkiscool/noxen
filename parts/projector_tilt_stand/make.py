"""Projector tilt stand: holds the S2-TEK / HOMPOW mini projector (Amazon B0F43Q5B9K; Windows reports the projector's display as "S2-TEK") tilted up 25 deg (Spark 24 Nov).
Drawn: python parts/projector_tilt_stand/make.py writes print/projector_tilt_stand.stl (and the viewing mesh and preview).
Projector (C from the listing, same as parts/projector_stand): 124.5 x 124.5 x 53.3 mm, 640 g, lens ring on one face (the HIGH end here), side vent grille and ports on side faces, speaker grille on top.
One part, one wedge: a hollow triangular frame (6 mm strips: base, back wall at the high end, 25 deg deck) 130 wide, 130 along the slope.
Everything near the projector stays open: the frame is open at both sides (side vents and ports), the deck has four 14 mm slots (45 deg pointed ends) (a bottom vent or foot is never covered), the stop lip runs the full width
at the LOW (rear, away from the lens) edge, where the projector would slide (a gap in it would be an unsupported overhang when printed on its side); cables leave through the open sides. The lens end is free.
Hand calc: deck load 6.4 N spread on four 16 mm bars across 130: bending of a 6 x 16 bar over 90 mm = 6.4/4 N x 90 / 4 = 36 Nmm against 96 mm3 section: 0.4 MPa (PCTG ~45): margin >100x.
Slide: the 25 deg slope needs mu > tan 25 = 0.47 to hold on friction alone (rubber feet on PCTG ~0.6, but not trusted): the lip, 10 mm tall, takes the whole 2.7 N downslope component.
Stability: CG 40 mm above the base, 62 mm from the low edge to the CG: tips at atan(60/40) = 56 deg; base 130 x 118 is wider than the projector.
Print: ON ITS SIDE (a triangular cheek face down, 130 tall in z): every layer is the same outline, so no supports and no overhangs; the deck slots bridge 14 mm only
(layer 0.2, 4 walls, 15 % infill, PCTG). Fits the A1 mini plate (118 x 77 footprint x 130)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scout_v4_probe_mech_lib import extrude, boolean, save  # noqa: E402
from shapely.geometry import Polygon, box
from shapely import affinity
import numpy as np
import trimesh

TILT = math.radians(25.0)
W = 130.0                      # width (z as printed)
SL = 130.0                     # deck length along the slope (projector 124.5)
T = 6.0                        # strip thickness
H0 = 12.0                      # height of the low end
LIP_H, LIP_T = 10.0, 3.0
c, s = math.cos(TILT), math.sin(TILT)
LX = SL * c                    # ground length
HH = H0 + SL * s               # height of the high end

body = Polygon([(0, 0), (LX, 0), (LX, HH), (0, H0)])
window = body.buffer(-T, join_style=2)
frame = body.difference(window)
# lip: a rectangle standing on the deck at its low end (deck frame: u along the slope, n normal to it)
u = np.array([c, s]); n = np.array([-s, c]); P0 = np.array([0.0, H0])
lip = Polygon([P0 - 1.0 * u, P0 + LIP_T * u, P0 + LIP_T * u + LIP_H * n, P0 - 1.0 * u + LIP_H * n]).union(
    Polygon([P0 - 1.0 * u, P0 + LIP_T * u, P0 + LIP_T * u - 3 * n, P0 - 1.0 * u - 3 * n]))
lip = lip.intersection(box(-20, 0, LX + 5, HH + 40))
solid = boolean(extrude(frame, 0, W), extrude(lip, 0, W), "+")
# deck slots: through the deck strip (normal direction), z 20..110, 14 wide along the slope, bars 16 between (first bar 16 from the low edge's lip)
# slots cut through the deck strip's thickness (normal direction), z 20..110, 14 along the slope, 16 mm bars between
for k in range(4):                                  # slot ends are 45 deg points (no flat ceiling to bridge: nothing needs support)
    s0 = 14.0 + k * 28.0
    pts = [(0, 27), (7, 20), (14, 27), (14, W - 27), (7, W - 20), (0, W - 27)]
    cut = extrude(Polygon(pts), 0, T + 2.0)
    M = np.eye(4); M[:3, 0] = [c, s, 0]; M[:3, 1] = [0, 0, 1]; M[:3, 2] = [s, -c, 0]
    M[:3, 3] = [P0[0] + s0 * c - s, P0[1] + s0 * s + c, 0]
    cut.apply_transform(M)
    solid = boolean(solid, cut, "-")
view = solid.copy()
view.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])); view.apply_translation([0, W / 2, 0])
save("projector_tilt_stand", solid, view, "on its side, no supports; tilt 25 deg, lip at the low edge, deck 4 slots")
