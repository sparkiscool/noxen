"""Rod saw guide (Spark 29 Oct: "something that can cut 6 mm and 3 mm rods with my saw but with a guide so I'm always making
straight cuts"). Drawn only, no topology run: python parts/rod_saw_guide/make.py writes part.stl. A PETG mitre block:
  - two open-top channels along x, full length: O6 rod channel 6.4 wide, 8.4 deep; O3 rod channel 3.4 wide, 5.4 deep
    (round bottoms; the rod sits 2 mm below the top face, so the blade is guided above the rod as well as beside it)
  - one saw slot across y at mid-length, KERF 1.0 wide, from the top down to 3 mm under the deeper channel: a hacksaw or
    junior hacksaw blade (0.6-0.8 thick) runs in it; over the 30 mm guide height the play is under 1 degree
  - a fence on each side of the slot, so the block can be held in a vice by its base without squeezing the channels
Use: mark the rod from the cut list (systems/overseer.cage.json rod_cuts_mm), lay it in its channel with the mark in the
slot, hold it down with a thumb, saw. 32 TPI blade, wrap the mark in tape against splinters, cut wet or masked (carbon dust).
Prints flat face down, channels up, no supports. If the saw does not fit the 1.0 slot, change KERF and print again."""
import numpy as np, trimesh, manifold3d as mf

L, W, H = 60.0, 34.0, 30.0                    # block
KERF = 1.0                                    # saw slot width
CH = [(-8.0, 6.4, 8.4), (8.0, 3.4, 5.4)]      # (y centre, width, depth) of the O6 and O3 channels
SLOT_BOTTOM = H - 8.4 - 3.0                   # the slot reaches 3 mm under the deeper channel
BASE = 6.0                                    # the vice flange: 6 mm proud each side, 8 tall

def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

solid = box(0, L, -W / 2, W / 2, 0, H) + box(0, L, -W / 2 - BASE, W / 2 + BASE, 0, 8.0)
cuts = []
for yc, w, d in CH:
    r = w / 2
    cuts.append(box(-1, L + 1, yc - r, yc + r, H - d + r, H + 1))                              # straight walls
    cuts.append(mf.Manifold.cylinder(L + 2, r, r, 64).rotate([0, 90, 0]).translate([-1, yc, H - d + r]))  # round bottom
cuts.append(box(L / 2 - KERF / 2, L / 2 + KERF / 2, -W / 2 - BASE - 1, W / 2 + BASE + 1, SLOT_BOTTOM, H + 1))
body = solid - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
mesh = body.to_mesh()
tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
assert tm.is_watertight and tm.is_volume
tm.export(__file__.replace("make.py", "part.stl"))
print(f"rod_saw_guide: {tm.volume / 1000:.1f} cm3, {tm.volume / 1000 * 1.27:.1f} g PETG, {np.round(tm.extents, 1)} mm, "
      f"{len(tm.faces)} faces, watertight {tm.is_watertight}")
