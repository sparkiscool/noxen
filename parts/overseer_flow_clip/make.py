"""Flow clip v2 (Spark 30 Oct 1 am: v1's plug gap too small and the lens window misses the sensors; "get the actual
dimensions of the sensor online"). Drawn only: python parts/overseer_flow_clip/make.py writes part.stl.

MTF-02P from MicoAir's dimension drawing (micoair.com/optical_range_sensor_mtf-02p, read 30 Oct): board 21.6 x 16, 6.5 tall
over all (board ~1.0, the ToF module is the tallest at 5.5 above the board face, the flow lens ~4.5, the SH1.0-4P socket
~3.2 and it overhangs the short edge by ~1). On the component face, with the socket edge to the right and the long edge
horizontal: ToF module 6.8 x 6 at the top-left, its two windows 5.1 and 8.5 from the left edge and 3.7 from the top edge;
the flow lens 6.8 from the left edge and 10.5 from the top edge (Ø5 module 7 x 7); the socket centred on the right edge.

In the drone (x fwd, y left, z up, z 0 = bottom plate top): board back taped to the plate underside z -2.1, component face
down, long axis along x, socket edge to the tail (-x) so the plug leaves under the plate toward the deck. Seen from below
with the nose to the left that is the drawing with image-up = +y: ToF at y +4.3, windows at x nose-5.1 / nose-8.5; lens at
y -2.5, x nose-6.8. PX4: the board is yawed 90 deg from the default (socket right), set SENS_FLOW_ROT accordingly.

Clip: pocket 21.6+0.6 x 16+0.6, centre (94, 0), walls 2.5, floor top 0.3 under the tallest component (z -8.9), floor 3.0;
tail wall notched 9 wide over its full height for the socket overhang, the plug and its wires; one window through the floor
from 2.5 to 12.5 from the nose end over the full pocket width (both ToF windows and the lens with the 42 deg flow cone
clear at the floor's bottom face); flanges 3.0 at (96, +-17) with Ø3.5 for the deck's two front M3 x 10 (as v1)."""
import numpy as np, trimesh, manifold3d as mf

BOARD_L, BOARD_W, CLR = 21.6, 16.0, 0.3
CX, CY = 94.0, 0.0
Z_TOP = -2.1                               # plate underside = board back
BOARD_T, COMP_H = 1.0, 5.5                 # ToF module height above the board face
Z_FLOOR_TOP = Z_TOP - BOARD_T - COMP_H - 0.3   # -8.9
FLOOR, WALL = 3.0, 2.5
Z_BOT = Z_FLOOR_TOP - FLOOR                # -11.9
NOTCH_W = 9.0
WIN_FROM_NOSE, WIN_TO_NOSE = 2.5, 12.5
FLANGE = [(96.0, 17.0), (96.0, -17.0)]; FLANGE_T, FLANGE_HOLE_R, FLANGE_W = 3.0, 1.75, 8.0

def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl(x, y, z0, z1, r):
    return mf.Manifold.cylinder(z1 - z0, r, r, 64).translate([x, y, z0])

px0, px1 = CX - BOARD_L / 2 - CLR, CX + BOARD_L / 2 + CLR      # pocket (tail .. nose)
py0, py1 = CY - BOARD_W / 2 - CLR, CY + BOARD_W / 2 + CLR
body = box(px0 - WALL, px1 + WALL, py0 - WALL, py1 + WALL, Z_BOT, Z_TOP)
for fx, fy in FLANGE:                                             # flange lobes joined to the body
    body += box(fx - FLANGE_W / 2, fx + FLANGE_W / 2, min(fy, 0), max(fy, 0), Z_TOP - FLANGE_T, Z_TOP)
    body += cyl(fx, fy, Z_TOP - FLANGE_T, Z_TOP, FLANGE_W / 2)
cuts = [box(px0, px1, py0, py1, Z_FLOOR_TOP, Z_TOP + 1),                                  # pocket, open to the plate
        box(px0 - WALL - 1, px0 + 0.01, -NOTCH_W / 2, NOTCH_W / 2, Z_FLOOR_TOP, Z_TOP + 1),  # plug notch in the tail wall
        box(px1 - WIN_TO_NOSE, px1 - WIN_FROM_NOSE, py0, py1, Z_BOT - 1, Z_FLOOR_TOP + 1)]   # window through the floor
cuts += [cyl(fx, fy, Z_BOT - 1, Z_TOP + 1, FLANGE_HOLE_R) for fx, fy in FLANGE]
part = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
mesh = part.to_mesh()
tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
assert tm.is_watertight and tm.is_volume
tm.export(__file__.replace("make.py", "part.stl"))
nose = CX + BOARD_L / 2
print(f"flow clip v2: {tm.volume / 1000:.2f} cm3, {tm.volume / 1000 * 1.27:.1f} g PETG, bounds {np.round(tm.bounds, 1).tolist()}, {len(tm.faces)} faces")
print(f"  sensors (drone xyz): lens ({nose - 6.8:.1f}, {CY - 2.5:.1f}), ToF windows ({nose - 5.1:.1f} / {nose - 8.5:.1f}, {CY + 4.3:.1f});"
      f" window x {px1 - WIN_TO_NOSE:.1f}..{px1 - WIN_FROM_NOSE:.1f}, y {py0:.1f}..{py1:.1f}; plug notch at x {px0:.1f}, y +-{NOTCH_W / 2}, z {Z_FLOOR_TOP}..{Z_TOP}")
