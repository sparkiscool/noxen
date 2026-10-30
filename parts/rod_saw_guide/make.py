"""Rod saw guide v4.1, one piece, flat bottom (Spark 30 Oct 1 am: the best tool, researched; no clamp; "it needs to be one piece").
Drawn only: python parts/rod_saw_guide/make.py writes part.stl.

Research (30 Oct; clearwatercomposites.com, carbonfibertubes.net, ridegearlab.com, GTA forum): carbon splinters when one
side of the tube moves before the cut is through, so the rod is held on both sides of the blade; never in a metal vice (the
thin wall cracks); an abrasive (carbide / diamond grit) blade beats teeth, a 32 TPI hacksaw is the fallback; tape over the
mark, light pressure, a quarter turn at a time; deburr at 45 deg with a file.

One PETG block, 100 x 34 x 30 on a 60 x 8 base, nothing to clamp, nothing to assemble:
  - the O6 channel (6.4 wide, 8.4 deep) at y -8 and the O3 channel (3.4 x 5.4) at y +8 along the length, round bottoms, the
    rod top 2 under the face
  - two 14 mm clip zones near the ends: the channel mouth 0.5 narrower than the rod with a 45 deg lead-in, the walls beside
    it relieved into 2.0 mm fingers (1.0 slots, 10 deep): the rod snaps in and is gripped on both sides of the blade
  - the saw slot, KERF wide, across the middle from the top to 3 under the O6 channel (a 32 TPI hacksaw blade, 0.6-0.8
    thick, runs in a 1.0 slot; change KERF and reprint for a razor saw (0.6) or a carbide-grit blade (1.5)); the slot
    wears with use, reprint the block when the cut starts to wander
  - a tick every 10 mm along both top edges, measured from the blade (deep tick every 50): short pieces are set against the
    ticks, long ones from a tape mark laid in the slot
  - a flat bottom: Spark tapes it to the bench with double-sided tape (30 Oct, instead of rubber feet)
Prints base down, no supports, 100 % infill. Use: tape the rod at the mark, press it into the clips with the mark in the
slot, saw with light strokes turning the rod a quarter at a time, file the end at 45 deg. Dust: cut wet or masked."""
import os, numpy as np, trimesh, manifold3d as mf

L, W, H = 100.0, 34.0, 30.0
BASE_W, BASE_H = 60.0, 8.0
CH = [(-8.0, 6.4, 8.4), (8.0, 3.4, 5.4)]                   # (y, width, depth): O6 and O3 channels
KERF = 1.0
SLOT_BOTTOM = H - 8.4 - 3.0
CLIP_X, GRIP, FINGER, RELIEF, RELIEF_D = ((8.0, 22.0), (L - 22.0, L - 8.0)), 0.5, 2.0, 1.0, 10.0

def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def seat(x0, x1, yc, w, d):
    r = w / 2
    return mf.Manifold.cylinder(x1 - x0, r, r, 64).rotate([0, 90, 0]).translate([x0, yc, H - d + r])

body = box(0, L, -W / 2, W / 2, 0, H) + box(0, L, -BASE_W / 2, BASE_W / 2, 0, BASE_H)
cuts = []
free = [(-1, CLIP_X[0][0]), (CLIP_X[0][1], CLIP_X[1][0]), (CLIP_X[1][1], L + 1)]
for yc, w, d in CH:
    r = w / 2
    for x0, x1 in free:                                                          # plain channel between the clips
        cuts += [seat(x0, x1, yc, w, d), box(x0, x1, yc - r, yc + r, H - d + r, H + 1)]
    for x0, x1 in CLIP_X:                                                        # clip zone
        mouth = (w - 0.4 - GRIP) / 2                                             # half-width of the mouth: rod - 0.5
        cuts += [seat(x0, x1, yc, w, d), box(x0, x1, yc - mouth, yc + mouth, H - d + r, H + 1)]
        cuts.append(mf.Manifold.hull_points([[x, yc + sg * (mouth + (1.0 if top else 0.0)), H - (0.0 if top else 1.0) + (1.0 if top else 0.0)]
                                             for x in (x0, x1) for sg in (-1, 1) for top in (False, True)]))   # 45 deg lead-in
        for sg in (-1, 1):                                                       # relief slots: 2.0 fingers beside the mouth
            ya = yc + sg * (mouth + FINGER); yb = ya + sg * RELIEF
            cuts.append(box(x0, x1, min(ya, yb), max(ya, yb), H - RELIEF_D, H + 1))
cuts.append(box(L / 2 - KERF / 2, L / 2 + KERF / 2, -BASE_W / 2 - 1, BASE_W / 2 + 1, SLOT_BOTTOM, H + 1))   # saw slot
for k in range(1, int(L / 2 / 10) + 1):                                          # ticks from the blade, both edges
    deep = (10 * k) % 50 == 0
    for x in (L / 2 - 10 * k, L / 2 + 10 * k):
        for sg in (-1, 1):
            cuts.append(box(x - 0.4, x + 0.4, min(sg * W / 2, sg * (W / 2 - (6.0 if deep else 3.0))), max(sg * W / 2, sg * (W / 2 - (6.0 if deep else 3.0))), H - 0.6, H + 1))
part = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
m = part.to_mesh()
tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
assert tm.is_watertight and tm.is_volume
tm.export(os.path.join(os.path.dirname(os.path.abspath(__file__)), "part.stl"))
print(f"rod_saw_guide v4.1: {tm.volume / 1000:.1f} cm3, {tm.volume / 1000 * 1.27:.0f} g PETG, {np.round(tm.extents, 1).tolist()} mm, {len(tm.faces)} faces, "
      f"slot {KERF} wide at x {L / 2}, clips at {CLIP_X}")
