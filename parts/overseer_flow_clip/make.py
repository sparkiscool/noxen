"""Flow clip v3 (Spark 31 Oct: the sensor sits flat on its back; flat face to the bottom plate, open face to the ground).
Drawn only: python parts/overseer_flow_clip/make.py writes part.stl.

MTF-02P from MicoAir's dimension drawing (micoair.com/optical_range_sensor_mtf-02p, read 30 Oct): board 21.6 x 16, 6.5 tall
over all (board ~1.0, the ToF module the tallest at 5.5 above the board face, the flow lens ~4.5, the SH1.0-4P socket ~3.2,
overhanging the short edge by ~1). Placement as v2: long axis along x, component face down, socket edge to the tail (-x);
ToF at y +4.3, windows at x nose-5.1 / nose-8.5; lens at y -2.5, x nose-6.8 (SENS_FLOW_ROT as v2).

Clip: a flat 2.5 roof against the plate underside (z -2.1), the board back taped flat on its underside (z -4.6); walls 2.5
round a 22.2 x 16.6 pocket down to 0.5 past the ToF module (z -11.6); the whole bottom open (no floor, nothing in the lens
or ToF view); tail wall notched 9 wide for the socket overhang, plug and wires; flanges 3.0 at (96, +-17), O3.5 for the deck's
two front M3 x 10 (as v1, v2)."""
import numpy as np, trimesh, manifold3d as mf

BOARD_L, BOARD_W, CLR = 21.6, 16.0, 0.3
CX, CY = 94.0, 0.0
Z_TOP = -2.1                               # plate underside = board back
BOARD_T, COMP_H = 1.0, 5.5                 # ToF module height above the board face
ROOF, WALL = 2.5, 2.5
Z_BACK = Z_TOP - ROOF                      # -4.6 board back, taped to the roof
Z_BOT = Z_BACK - BOARD_T - COMP_H - 0.5    # -11.6 open bottom
NOTCH_W = 9.0
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
cuts = [box(px0, px1, py0, py1, Z_BOT - 1, Z_BACK),                                       # pocket, open to the ground
        box(px0 - WALL - 1, px0 + 0.01, -NOTCH_W / 2, NOTCH_W / 2, Z_BOT - 1, Z_BACK)]          # plug notch in the tail wall
cuts += [cyl(fx, fy, Z_BOT - 1, Z_TOP + 1, FLANGE_HOLE_R) for fx, fy in FLANGE]
part = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
mesh = part.to_mesh()
tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
assert tm.is_watertight and tm.is_volume
tm.export(__file__.replace("make.py", "part.stl"))
print(f"flow clip v3: {tm.volume / 1000:.2f} cm3, {tm.volume / 1000 * 1.27:.1f} g PETG, bounds {np.round(tm.bounds, 1).tolist()}; board back z {Z_BACK}, open bottom z {Z_BOT}")
