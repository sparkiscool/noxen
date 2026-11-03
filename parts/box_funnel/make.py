"""The shared landing funnel, drawn (3 Nov; v3 after the seat-envelope study and the exact clearance check, tools/seat_envelope.py
and tools/seat_check.py, both drones seated with their real meshes).
Both drones land at one height with the battery over the box's hole, centred by shape (Spark 3 Nov). The one feature both carry
at the same place is the latch receiver, but the two differ in length: Scout v3 105.6 x 65.5, Overseer v2 about 184 x 67 (the
swap arm's bay envelope, E). So the seat is the OVERSEER receiver's outline and the Scout carries a belly skirt of that outline round
its own receiver (parts/scout_belly_skirt). Both are 66 wide with their bottoms at seat z -44.3.
Seat frame = drone coordinates of the seated Scout (x nose, y left, z up); the box's SEAT_TO_W puts it in W.
What the Overseer allows with no change to it (seat_check): its base's side rails fill |y| 60..80 from h 31 up, and its flow clip
hangs under the nose at x 110..138 from h 42 up. So the two y faces stop at h 28 (y capture +-20) and the two x faces at h 40
(x capture +-28), all at 55 deg from the seat ledge; the mouth is 240 x 106 at the top of the y faces. The base passes 2.7 over the
y walls, the flow clip 2 over the x walls' rim. With base v8 chamfered to |y| <= 33.4 + h / 1.428 - 2 for h 31..50 and the flow
clip 17 aft, the faces could all rise to h 50 and capture +-35 (the alternative, not drawn).
Legs: four 12 square under the x walls' ends (x -115 / +118, y +-40) down to the deck (W Z 66 = seat z -89.3), merging into the
walls' tops; the Overseer's loop diagonals and bay ties pass 15 clear of them at its capture limits and the Scout cage keeps
every member above the wall tops (tools/seat_check.py --descent: 0 crossings in every state for both drones). Wall 3.0, PETG. Not drawn: the hold-down against the 60 N insert, UHMW tape on the faces.
Run: python parts/box_funnel/make.py
"""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
SEAT_Z, FACE_DEG, WALL = -44.3, 55.0, 3.0            # seat frame; the seat's z 0 is W Z 215.3 (3 Nov: +60, the head extends 60 more at the drone slot)
H_X, H_Y = 40.0, 28.0                             # the x (nose / tail) faces and the y faces end here, above the seat
CX = 1.5                                          # the receiver centre x (the catch centre) in the seat frame
SEAT = (CX - 92.0, CX + 92.0, -33.4, 33.4)        # the Overseer receiver v2 outline (E) = the Scout's belly skirt
HOLE = (CX - 85.0, CX + 85.0, -25.0, 25.0)        # both cartridges and the swap head pass (Overseer cartridge width CHECK)
RIM, LEG, SKIRT = 0.0, 12.0, 8.0
ZW = 220.3                                        # seat z 0 in W (the catch's 155.3 + 65, 3 Nov)
BEAM_Y = 209.0                                    # the beams reach the side sheets (W X +-206..209)
NOSE_BEAM = (112.0, 124.0, 150.0 - ZW, 162.0 - ZW)   # seat x, W Z 150..162: 8 under the Overseer's nose diagonals (bottom 170), outside the pack sweep (Y 279..291)
TAIL_BEAM = (-121.0, -109.0, 155.0 - ZW, 163.0 - ZW)  # seat x, W Z 155..163 in the rails' band (tips 151): 4 over them, 7 under the Overseer's tail diagonals
END_PAD_Y = 200.0                                 # the beam ends drop to the sheet tops (W Z 145) over |y| 200..209
BUTTRESS_TOP, BUTTRESS_Y = -14.0, 40.0            # the slab from the tail beam up behind the tail wall (its outer face at seat x -115 is at h 30), |y| 40: at 48 an Overseer tube grazed its top corner (23 mm3) at the +x+y capture limit, h 30
CORNER_Y = 48.0                                   # the x walls' top corners (h 28..40) stop at |y| 48: the Overseer's loop diagonals pass 15 outside them at its capture limits at its capture limits; the receiver still bears on them over |y| <= 48
LEG_Y = 40.0                                       # the nose legs stand on the nose beam under the nose wall's end (|y| 40)      # no rim on the x walls (3 Nov: the compact Scout cage's nose rods pass 3 over them)
TAN = np.tan(np.radians(FACE_DEG))


def man(tri):
    return mf.Manifold(mf.Mesh(np.asarray(tri.vertices, np.float32), np.asarray(tri.faces, np.uint32)))
def hull(pts):
    return man(trimesh.convex.convex_hull(np.array(pts, float)))
def frustum(x0, x1, y0, y1, z0, z1, grow):
    return hull([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0 - grow, y0 - grow, z1), (x1 + grow, y0 - grow, z1), (x1 + grow, y1 + grow, z1), (x0 - grow, y1 + grow, z1)])
def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def funnel():
    x0, x1, y0, y1 = SEAT; z0 = SEAT_Z
    gx, gy = H_X / TAN, H_Y / TAN                                                      # 28.0 and 19.6
    w = WALL / np.sin(np.radians(FACE_DEG))
    inner = frustum(x0, x1, y0, y1, z0, z0 + H_X + 1, (H_X + 1) / TAN)
    outer = frustum(x0 - w, x1 + w, y0 - w, y1 + w, z0 - WALL, z0 + H_X, gx)
    yw = 33.4 + gy + w                                                                  # the y walls' outer face at h 28
    top_cut = box(x0 - gy + 0.5, x1 + gy - 0.5, -200, 200, z0 + H_Y, z0 + H_X + 2)        # the y walls above h 28 go; the x walls (beyond the inner x at h 28) stay
    ox0, ox1 = x0 - gx - w, x1 + gx + w                                                 # the x walls' outer faces at h 40
    rim = [box(ox0 - RIM, ox0 + 2.0, -yw - 4.0, yw + 4.0, z0 + H_X - 4.0, z0 + H_X), box(ox1 - 2.0, ox1 + RIM, -yw - 4.0, yw + 4.0, z0 + H_X - 4.0, z0 + H_X)] if RIM else []
    skirt = box(HOLE[0] - WALL, HOLE[1] + WALL, HOLE[2] - WALL, HOLE[3] + WALL, z0 - SKIRT, z0)
    legs = [box(ox1 - LEG - 1.0, ox1 - 1.0, cy - LEG / 2, cy + LEG / 2, NOSE_BEAM[3], z0 + H_X) for cy in (-LEG_Y, LEG_Y)]   # nose legs, inside the nose wall's footprint at the top
    legs.append(box(NOSE_BEAM[0], NOSE_BEAM[1], -BEAM_Y, BEAM_Y, NOSE_BEAM[2], NOSE_BEAM[3]))                                 # the nose beam across the box
    legs.append(box(TAIL_BEAM[0], TAIL_BEAM[1], -BEAM_Y, BEAM_Y, TAIL_BEAM[2], TAIL_BEAM[3]))                                 # the tail beam across the box
    for b in (NOSE_BEAM, TAIL_BEAM):                                                                                           # the beam ends sit on the sheet tops
        for sy in (-1, 1):
            legs.append(box(b[0], b[1], min(sy * END_PAD_Y, sy * BEAM_Y), max(sy * END_PAD_Y, sy * BEAM_Y), 145.0 - ZW, b[2]))
    legs.append(box(TAIL_BEAM[0], TAIL_BEAM[1], -BUTTRESS_Y, BUTTRESS_Y, TAIL_BEAM[3], BUTTRESS_TOP))                                     # the buttress from the tail beam up behind the tail wall
    hole = box(*HOLE, z0 - SKIRT - 8.0, z0 + H_X + 2.0)
    corner_cut = [box(x0 - 200, x1 + 200, s_ * CORNER_Y, s_ * 300, z0 + H_Y, z0 + H_X + 2) if s_ > 0 else box(x0 - 200, x1 + 200, -300, -CORNER_Y, z0 + H_Y, z0 + H_X + 2) for s_ in (-1, 1)]
    body = (outer - top_cut - mf.Manifold.batch_boolean(corner_cut, mf.OpType.Add)) + mf.Manifold.batch_boolean(rim + [skirt] + legs, mf.OpType.Add)
    return body - inner - hole


if __name__ == "__main__":
    m = funnel().to_mesh()
    t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    t.export(os.path.join(HERE, "part.stl"))
    print(f"funnel v3: x faces to h {H_X:.0f} (capture +-{H_X / TAN:.0f}), y faces to h {H_Y:.0f} (+-{H_Y / TAN:.0f}); extents {t.extents.round(1).tolist()}, "
          f"min {t.bounds[0].round(1).tolist()} max {t.bounds[1].round(1).tolist()}, {t.volume * 1.27e-3:.0f} g PETG, watertight {t.is_watertight}")
