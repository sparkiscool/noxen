"""FC mount v1 (Spark 30 Oct 9 pm: one of the FC stack's bolts does not go into the top plate). Drawn only:
python parts/overseer_fc_mount/make.py writes part.stl.

Measured on parts/top_plate/part.stl (z 45..46.6): slots 3 wide on -X (x -37..-13), +-Y (r 13..37) and +X (r 23.8..36.8);
centre cutout x 8..22, y +-14.5. The stack (MicoAir743-Lite on the 55A 4-in-1, 30.5 x 30.5, turned 45 deg) puts its holes
on the axes at r 21.57: three fall in the slots, the +X one (x 20.1..23.1) on the 1.7 web between the cutout and the +X
slot, so that bolt cannot pass. The stack keeps its place and pose (SENS_BOARD_ROT yaw 45 unchanged) and moves onto this
ring instead: 4 bosses on the stack holes take M3 x 5.7 heat-set inserts from the top (O4.0, as the base), the ring bolts
to the plate through the 4 slots at r 32 with M3 x 8 heads flush in counterbores and plain M3 nuts + threadlocker under the
plate. Open centre r 17 for the wires. Stack bolts: M3 x 20 down through FC (O4 grommets) and ESC into the inserts. The
stack rises 5 mm (ESC bottom z 55, was 50); nothing above it within r 40 (checked against the assembly 30 Oct). -X tab ends
r 37.5, 0.5 short of the ESC cap. Prints flat face down."""
import numpy as np, trimesh, manifold3d as mf

Z0 = 46.6                                   # top plate top face
T_RING, Z_BOSS = 6.0, 55.0                  # ring 46.6..52.6, insert bosses to z 55 (ESC seat)
R_IN, R_OUT = 17.0, 26.0
R_STACK = 30.5 / 2 * np.sqrt(2)             # 21.57, on the axes
BOSS_R, INSERT_R, INSERT_D = 4.5, 2.0, 5.7
R_BOLT, TAB_W, TAB_END = 32.0, 11.0, 37.5
HOLE_R, CB_R, CB_D = 1.6, 3.4, 3.3          # M3 clearance, O6.8 x 3.3 counterbore (socket head 5.5 x 3.0)
AXES = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def cyl(x, y, z0, z1, r, n=64):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])

def tab(ux, uy):
    t = mf.Manifold.cube([TAB_END - R_IN, TAB_W, T_RING]).translate([R_IN, -TAB_W / 2, Z0])
    return t.rotate([0, 0, float(np.degrees(np.arctan2(uy, ux)))])

body = cyl(0, 0, Z0, Z0 + T_RING, R_OUT, 128) - cyl(0, 0, Z0 - 1, Z0 + T_RING + 1, R_IN, 128)
for ux, uy in AXES:
    body += tab(ux, uy) - cyl(0, 0, Z0 - 1, Z0 + T_RING + 1, R_IN, 128)
    body += cyl(ux * R_STACK, uy * R_STACK, Z0, Z_BOSS, BOSS_R)
cuts = []
for ux, uy in AXES:
    sx, sy, bx, by = ux * R_STACK, uy * R_STACK, ux * R_BOLT, uy * R_BOLT
    cuts += [cyl(sx, sy, Z_BOSS - INSERT_D, Z_BOSS + 1, INSERT_R),                        # insert, blind (floor 2.7)
             cyl(bx, by, Z0 - 1, Z0 + T_RING + 1, HOLE_R), cyl(bx, by, Z0 + T_RING - CB_D, Z0 + T_RING + 1, CB_R)]
part = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
mesh = part.to_mesh()
tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
assert tm.is_watertight and tm.is_volume
tm.export(__file__.replace("make.py", "part.stl"))
print(f"fc mount v1: {tm.volume / 1000:.2f} cm3, {tm.volume / 1000 * 1.27:.1f} g PETG, bounds {np.round(tm.bounds, 1).tolist()}")
