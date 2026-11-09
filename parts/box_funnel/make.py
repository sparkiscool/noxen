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
Moving end walls (9 Nov, Spark: the Scout carries as little as possible, the box takes the rest): the nose and tail walls are two
carriages, cut from the funnel along its corner hips and between the side walls' faces, that slide along x on the side ledges. Setting
"overseer" = the 184 seat as above; "scout" = each wall 39.2 in, the seat is the Scout receiver v3's own 105.6 x 65.5 outline (x -51.35
..54.25), so the Scout needs no runners. The frame (side walls, ledges, skirt, beams, buttress, legs) stays. funnel(setting) is the
frame plus both carriages at that setting; part.stl is the frame, the carriages are parts/box_funnel_wall_nose and _tail at the
Overseer setting. Drive (drawn as placeholders): one Tr8 lead screw per side under the side walls' overhang, opposite-hand halves so
one turn moves both walls; each wall reaches the two screws by tabs through slots low in the side walls; a NEMA 8 per side geared to
the screw's middle; a switch at each wall's two positions. The box lands a drone only with its setting confirmed, and falls back to the
Overseer setting, where a Scout seats loose in x and nothing collides. Both drones land on TPU strips flush in the side ledges, and
seat sense and keep-alive come through four spring probes in the ledges at (1.5 +- 30, +-29.5): each drone carries a bare pad at each.
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
BUTTRESS_TOP, BUTTRESS_Y = -20.0, 40.0            # the slab from the tail beam up behind the tail wall, top at h 24.3 where the wall's inner face is at seat x -107.5 (at -14 its inner edge x -109 stood 2.5 inside the wall at h 30 and the Scout's skirt grazed it between the descent check's states), |y| 40: at 48 an Overseer tube grazed its top corner (23 mm3) at the +x+y capture limit, h 30
END_CUT_X, END_CUT_Y, END_CUT_H = 100.0, 50.0, 24.0   # the y walls' ends (|x| > 100) stop at h 24 outside |y| 50: the Overseer's loop corner grazed the tail end's top (h 25..28, 20 mm3) at the diagonal capture limit at h 30
CORNER_Y = 45.0                                   # the x walls' top corners (h 28..40) stop at |y| 45 (at 48 an Overseer tube grazed the tail corner 1 mm3 at the diagonal capture limit, h 40) at its capture limits; the receiver still bears on them over |y| <= 48
LEG_Y = 40.0                                       # the nose legs stand on the nose beam under the nose wall's end (|y| 40)      # no rim on the x walls (3 Nov: the compact Scout cage's nose rods pass 3 over them)
TAN = np.tan(np.radians(FACE_DEG))
SETTINGS = {"overseer": 0.0, "scout": 39.2}           # how far each end wall moves in from the 184 seat: the Scout receiver v3 is 105.6 long (x -51.35..54.25, centre 1.45)


def man(tri):
    return mf.Manifold(mf.Mesh(np.asarray(tri.vertices, np.float32), np.asarray(tri.faces, np.uint32)))
def hull(pts):
    return man(trimesh.convex.convex_hull(np.array(pts, float)))
def frustum(x0, x1, y0, y1, z0, z1, grow):
    return hull([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0 - grow, y0 - grow, z1), (x1 + grow, y0 - grow, z1), (x1 + grow, y1 + grow, z1), (x0 - grow, y1 + grow, z1)])
def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def body():
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
    end_cut = [box(min(sx * END_CUT_X, sx * 300), max(sx * END_CUT_X, sx * 300), min(sy * END_CUT_Y, sy * 300), max(sy * END_CUT_Y, sy * 300), z0 + END_CUT_H, z0 + H_X + 2) for sx in (-1, 1) for sy in (-1, 1)]   # the y walls' top ends
    corner_cut = [box(x0 - 200, x1 + 200, s_ * CORNER_Y, s_ * 300, z0 + H_Y, z0 + H_X + 2) if s_ > 0 else box(x0 - 200, x1 + 200, -300, -CORNER_Y, z0 + H_Y, z0 + H_X + 2) for s_ in (-1, 1)]
    body = (outer - top_cut - mf.Manifold.batch_boolean(corner_cut + end_cut, mf.OpType.Add)) + mf.Manifold.batch_boolean(rim + [skirt] + legs, mf.OpType.Add)
    return body - inner - hole


def carriage_home(sx):
    """the nose (sx +1) or tail (sx -1) wall as drawn in body(), at the Overseer setting: the shell beyond the corner hips
    (|x - xend| >= |y| - 33.4), between the side walls' inner faces, from the seat up"""
    x0, x1, y0, y1 = SEAT; z0 = SEAT_Z + 0.01; xe = x1 if sx > 0 else x0; zt = SEAT_Z + H_X + 2.0; X = 120.0   # 0.01 over the seat plane: no coplanar slivers
    wedge = hull([(xe - sx * 33.4, 0.0, z0), (xe - sx * 33.4, 0.0, zt), (xe + sx * X, X + 33.4, z0), (xe + sx * X, -X - 33.4, z0),
                  (xe + sx * X, X + 33.4, zt), (xe + sx * X, -X - 33.4, zt)])
    g = (zt - z0) / TAN
    trough = hull([(xe - sx * 40, y0, z0), (xe - sx * 40, y1, z0), (xe + sx * X, y0, z0), (xe + sx * X, y1, z0),
                   (xe - sx * 40, y0 - g, zt), (xe - sx * 40, y1 + g, zt), (xe + sx * X, y0 - g, zt), (xe + sx * X, y1 + g, zt)])
    return body() ^ wedge ^ trough


TAB_H = (4.0, 8.0)                                # each wall reaches its nuts by two tabs, h 4..8 over the seat, through a slot in each side wall (h 3.5..8.5)
SCREW_Y, SCREW_H, SCREW_R = 54.0, 6.0, 4.0        # one Tr8 lead screw per side, under the side walls' overhang, opposite-hand halves: one turn moves both walls in or out
NUT_Y = (49.0, 59.0)                              # the nut blocks, outside the side walls' outer faces
MOTOR = (-15.0, 15.0, 62.0, 82.0, 6.0, 26.0)      # x, |y|, h: a NEMA 8 per side beside the screw's middle, geared to it; under the Overseer base's side rails (h 31) and over the 60 / 120 deg bays
STRIP_Y, STRIP_D = (25.5, 33.0), 1.5              # TPU strips flush in the side ledges, for the landing impact (both drones land on them)
PROBES = [(1.5 + dx, sy * 29.5) for dx in (-30.0, 30.0) for sy in (-1, 1)]   # keep-alive and seat-sense spring probes in the ledges: both drones carry a bare pad at each
BRACKET_Y = (26.0, 46.0)                          # the frame's ties from the ledges out to the beams, |y| 26..46 (the swap head rises at the tail end within |y| 20), under the seat plane


def frame():
    """the funnel without its end walls: the side walls, ledges and skirt tied to the nose and tail beams by four brackets under the
    seat plane (the walls slide over them); the nose legs, which only held the nose wall, go"""
    x0, x1 = SEAT[0], SEAT[1]; z0 = SEAT_Z; w = WALL / np.sin(np.radians(FACE_DEG))
    f = body() - carriage_home(1) - carriage_home(-1)
    f = f - box(NOSE_BEAM[0] - 1.0, NOSE_BEAM[1] + 2.0, -LEG_Y - LEG, LEG_Y + LEG, NOSE_BEAM[3], z0 + H_X + 1)        # the nose legs' remains
    ties = []
    for sy in (-1, 1):
        ya, yb = sorted((sy * BRACKET_Y[0], sy * BRACKET_Y[1]))
        ties.append(box(x1 + w - 4.0, NOSE_BEAM[1], ya, yb, NOSE_BEAM[3] - 1.0, z0 - 1.5))
        ties.append(box(TAIL_BEAM[0], x0 - w + 4.0, ya, yb, TAIL_BEAM[3] - 1.0, z0 - 1.5))
    xs = [tab_x(1), tab_x(-1)]
    for sy in (-1, 1):                                                                              # bearing blocks at the screws' ends, on the side walls' outer faces
        ya, yb = sorted((sy * 44.0, sy * (SCREW_Y + 6.0)))
        ties.append(box(xs[0][1] + 14.0, xs[0][1] + 20.0, ya, yb, NOSE_BEAM[3] - 1.0, z0 + SCREW_H + 6.0))     # down to the nose beam
        ties.append(box(xs[1][0] - 20.0, xs[1][0] - 14.0, ya, yb, TAIL_BEAM[3] - 1.0, z0 + SCREW_H + 6.0))     # down to the tail beam
    f = f + mf.Manifold.batch_boolean(ties, mf.OpType.Add)
    cuts = []
    for sx in (1, -1):                                                                              # the tabs' slots through the side walls, over the whole travel
        xa, xb = tab_x(sx); lo, hi = (xa - SETTINGS["scout"] - 1.0, xb + 1.0) if sx > 0 else (xa - 1.0, xb + SETTINGS["scout"] + 1.0)
        for sy in (-1, 1):
            ya, yb = sorted((sy * 33.0, sy * (NUT_Y[0] - 0.5)))
            cuts.append(box(lo, hi, ya, yb, z0 + TAB_H[0] - 0.5, z0 + TAB_H[1] + 0.5))
    for sy in (-1, 1):                                                                              # the strips' pockets
        ya, yb = sorted((sy * STRIP_Y[0], sy * STRIP_Y[1])); cuts.append(box(SEAT[0] + 1.0, SEAT[1] - 1.0, ya, yb, z0 - STRIP_D, z0 + 0.01))
    cuts += [mf.Manifold.cylinder(30.0, 1.7, 1.7, 16).translate([x, y, z0 - 20.0]) for x, y in PROBES]
    f = f - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
    keep = [m for m in f.decompose() if m.volume() > 1.0]
    return mf.Manifold.batch_boolean(keep, mf.OpType.Add)


def tab_x(sx):
    """the x span the wall's tabs cover at the Overseer setting: the wall's face at h 4 to its back at h 8 (the tab leans with the wall)"""
    xe = SEAT[1] if sx > 0 else SEAT[0]; w = WALL / np.sin(np.radians(FACE_DEG))
    a, b = xe + sx * (TAB_H[0] / TAN + 0.3), xe + sx * (TAB_H[1] / TAN + w - 0.3)
    return min(a, b), max(a, b)


def nuts(sx):
    """the wall's two tabs, leaning with the wall, out through the side walls, and its two nut blocks bored for the screws (they move
    with the wall)"""
    z0 = SEAT_Z; xe = SEAT[1] if sx > 0 else SEAT[0]; w = WALL / np.sin(np.radians(FACE_DEG)); out = []
    xa, xb = tab_x(sx); xm = (xa + xb) / 2
    for sy in (-1, 1):
        y0_, y1_ = sorted((sy * 34.0, sy * (NUT_Y[0] + 1.0)))
        pts = []
        for h in TAB_H:
            f, b = xe + sx * (h / TAN + 0.3), xe + sx * (h / TAN + w - 0.3)
            for x in (f, b):
                for y in (y0_, y1_):
                    pts.append((x, y, z0 + h))
        out.append(hull(pts))
        ya, yb = sorted((sy * NUT_Y[0], sy * NUT_Y[1]))
        out.append(box(xm - 6.0, xm + 6.0, ya, yb, z0 + SCREW_H - 5.5, z0 + SCREW_H + 5.5))
    bore = [mf.Manifold.cylinder(40.0, SCREW_R + 0.2, SCREW_R + 0.2, 24).rotate([0, 90, 0]).translate([xm - 20.0, sy * SCREW_Y, z0 + SCREW_H]) for sy in (-1, 1)]
    return mf.Manifold.batch_boolean(out, mf.OpType.Add) - mf.Manifold.batch_boolean(bore, mf.OpType.Add)


def carriage(sx, setting="overseer"):
    """the wall with its tabs and nut blocks, at a setting"""
    return (carriage_home(sx) + nuts(sx)).translate([-sx * SETTINGS[setting], 0.0, 0.0])


def drive():
    """the fixed, bought side of the drive, as solids the drones must clear: both screws, both motors with their gears, the four
    position switches (each wall in / out), the probes' bodies under the ledges"""
    z0 = SEAT_Z; out = []
    xs = [tab_x(1), tab_x(-1)]; x_lo = xs[1][0] - 14.0; x_hi = xs[0][1] + 14.0
    for sy in (-1, 1):
        out.append(mf.Manifold.cylinder(x_hi - x_lo, SCREW_R, SCREW_R, 24).rotate([0, 90, 0]).translate([x_lo, sy * SCREW_Y, z0 + SCREW_H]))
        ya, yb = sorted((sy * MOTOR[2], sy * MOTOR[3])); out.append(box(MOTOR[0], MOTOR[1], ya, yb, z0 + MOTOR[4], z0 + MOTOR[5]))
        ya, yb = sorted((sy * (SCREW_Y - 4.0), sy * MOTOR[3])); out.append(box(-6.0, 6.0, ya, yb, z0 + SCREW_H - 4.0, z0 + MOTOR[5] - 2.0))   # the gear pair
    for sx in (1, -1):                                                                                     # switches at each nut's two ends, on the +y screw
        xa, xb = tab_x(sx)
        for dx in (0.0, -sx * SETTINGS["scout"]):
            xm = (xa + xb) / 2; xc = xm + sx * 10.0 + dx if dx == 0.0 else xm - sx * 10.0 + dx       # just beyond the nut block (xm +- 6) at each end of its travel
            out.append(box(xc - 3.0, xc + 3.0, SCREW_Y + 5.0, SCREW_Y + 15.0, z0 + SCREW_H - 6.0, z0 + SCREW_H + 6.0))
    for x, y in PROBES:
        out.append(mf.Manifold.cylinder(14.0, 1.5, 1.5, 16).translate([x, y, z0 - 14.0]))
    return mf.Manifold.batch_boolean(out, mf.OpType.Add)


def strips():
    """the two TPU strips, flush in the ledges, with the probes' holes"""
    z0 = SEAT_Z; out = []
    for sy in (-1, 1):
        ya, yb = sorted((sy * STRIP_Y[0], sy * STRIP_Y[1])); out.append(box(SEAT[0] + 1.0, SEAT[1] - 1.0, ya, yb, z0 - STRIP_D, z0))
    holes = [mf.Manifold.cylinder(20.0, 1.7, 1.7, 16).translate([x, y, z0 - 10.0]) for x, y in PROBES]
    return mf.Manifold.batch_boolean(out, mf.OpType.Add) - mf.Manifold.batch_boolean(holes, mf.OpType.Add)


def funnel(setting="overseer"):
    """everything the drones must clear, at a setting ("overseer" = the 184 seat, "scout" = the walls 39.2 in): the frame, both
    walls with their nuts, the strips, the drive"""
    return frame() + carriage(1, setting) + carriage(-1, setting) + strips() + drive()


def stl(m, path):
    mm = m.to_mesh(); t = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
    os.makedirs(os.path.dirname(path), exist_ok=True); t.export(path); return t


if __name__ == "__main__":
    PARTS = os.path.dirname(HERE)
    stl(carriage(1), os.path.join(PARTS, "box_funnel_wall_nose", "part.stl"))
    stl(carriage(-1), os.path.join(PARTS, "box_funnel_wall_tail", "part.stl"))
    stl(strips(), os.path.join(PARTS, "box_funnel_strips", "part.stl"))
    stl(drive(), os.path.join(PARTS, "box_funnel_drive", "part.stl"))
    for st in SETTINGS:                                                                             # the walls never touch the frame or the drive
        for sx in (1, -1):
            v = (carriage(sx, st) ^ (frame() + drive())).volume()
            assert v < 1.0, f"the {'nose' if sx > 0 else 'tail'} wall at the {st} setting crosses the frame or drive by {v:.0f} mm3"
    t = stl(frame(), os.path.join(HERE, "part.stl"))
    print(f"frame {t.volume * 1.27e-3:.0f} g in {len(frame().decompose())} piece(s), walls with their nuts {carriage(1).volume() * 1.27e-3:.1f} + {carriage(-1).volume() * 1.27e-3:.1f} g, strips {strips().volume() * 1.21e-3:.1f} g TPU; walls in for the Scout by {SETTINGS['scout']}")
    print(f"funnel v3: x faces to h {H_X:.0f} (capture +-{H_X / TAN:.0f}), y faces to h {H_Y:.0f} (+-{H_Y / TAN:.0f}); extents {t.extents.round(1).tolist()}, "
          f"min {t.bounds[0].round(1).tolist()} max {t.bounds[1].round(1).tolist()}, {t.volume * 1.27e-3:.0f} g PETG, watertight {t.is_watertight}")
