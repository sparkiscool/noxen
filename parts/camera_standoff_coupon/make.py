"""Camera standoff coupon (Spark 29 Oct: the 10 mm O7.6 -> O11 cones do not clear the board; before changing the head mount and
node 21, print the candidates and try the DECXIN 2633V2 on them). Drawn only, no topology run: python parts/camera_standoff_coupon/make.py
writes part.stl. One 3 mm strip, three 28 x 28 patterns (dimensions.md Camera: M2.5 on 28 x 28), standoffs 5 tall (half the 10):
  A (1 pip)  O5.6, insert hole O3.2 x 5.7 (1.2 wall: the thinnest that still takes the M2.5 heat-set insert, OD 3.5)
  B (2 pips) O6.0, hole O2.1 x 6: the M2.5 screw taps into the PETG, no insert (1.95 wall)
  C (3 pips) O7.6, insert hole O3.2 x 5.7 (today's 2.2 wall, only shorter)
The pips are O1.5 holes through the plate at each zone's corner. Board 38 x 38 centred on its pattern: zones 44 wide."""
import numpy as np, trimesh, manifold3d as mf

T, Z, H = 3.0, 44.0, 5.0                      # plate thickness, zone pitch, standoff height
PAT = 28.0
VARIANTS = [("A", 5.6 / 2, 3.2 / 2, 5.7), ("B", 6.0 / 2, 2.1 / 2, 6.0), ("C", 7.6 / 2, 3.2 / 2, 5.7)]

def cyl(r, h, x, y, z0, n=48):
    return mf.Manifold.cylinder(h, r, r, n).translate([x, y, z0])

plate = mf.Manifold.cube([Z * len(VARIANTS), Z, T]).translate([0, -Z / 2, 0])
solid, cuts = plate, []
for i, (nm, r_out, r_hole, depth) in enumerate(VARIANTS):
    cx = Z * (i + 0.5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * PAT / 2, sy * PAT / 2
            solid += cyl(r_out, H, x, y, T)
            cuts.append(cyl(r_hole, depth + 0.01, x, y, T + H - depth))
    for k in range(i + 1):                                            # pips: 1, 2, 3 holes along the zone's -y edge
        cuts.append(cyl(0.75, T + 2, cx - 6 + 4 * k, -Z / 2 + 3, -1))
body = solid - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
mesh = body.to_mesh()
tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
assert tm.is_watertight and tm.is_volume
tm.export(__file__.replace("make.py", "part.stl"))
print(f"camera_standoff_coupon: {tm.volume / 1000:.1f} cm3, {tm.volume / 1000 * 1.27:.1f} g PETG, {np.round(tm.extents, 1)} mm, "
      f"{len(tm.faces)} faces, watertight {tm.is_watertight}")
