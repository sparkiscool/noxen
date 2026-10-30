"""Rod saw guide v3 (Spark 30 Oct 1 am: "make the best tool you can, do some research on it"; then "can we make it with no
clamp": no vice, no band, no screws). Drawn only:
python parts/rod_saw_guide/make.py writes print/*.stl (one file per piece) and part.stl (the set assembled, for the page).

Research (30 Oct; clearwatercomposites.com, carbonfibertubes.net, ridegearlab.com, GTA forum): carbon splinters when one side of
the tube moves before the cut is through, so the rod is held on both sides of the blade; never in a metal vice (the thin wall
cracks), so the jig holds the rod and the vice holds the jig; an abrasive (diamond / carbide grit) blade beats teeth, a 32 TPI
hacksaw is the fallback; tape over the mark, light pressure, turn the rod a quarter at a time; deburr at 45 deg with a file.
Jig practice (MakerWorld tube jigs, jewellers' mitre jigs): a replaceable kerf guide, because the blade wears the slot, and a
sliding length stop locked by a screw for repeat cuts.

The set (PETG; body and inserts at 100 % infill, the rest 4 walls). No clamp anywhere:
  body     90 x 34 x 30 on a 60-wide, 8-tall base that stands on the bench on four stick-on rubber feet (O10 recesses): the
           O6 channel (6.4 wide, 8.4 deep) at y -8 and the O3 channel (3.4 x 5.4) at y +8 along the length, round bottoms,
           rod top 2 under the face. Two 14 mm clip zones (x 8..22 and 68..82) hold the rod instead of a clamp: there the
           channel mouth is 0.5 narrower than the rod and the walls beside it are relieved into 2.0 mm fingers (1.0 relief
           slots, 10 deep) so the rod snaps in and is gripped on both sides of the blade; a 12-wide pocket at mid-length for
           the kerf insert (full width, down to the base); a 10 x 8 socket in the -x end face for the stop rail.
  insert_k06 / k10 / k15   12 x 34 x 22 blocks with the same channels and a blade slot 0.6 / 1.0 / 1.5 wide, from the top to
           3 under the deeper channel: razor saw or junior hacksaw / 32 TPI hacksaw / carbide-grit blade. Print the one your
           blade fits; reprint when the slot wears.
  rail     150 x 20 x 12, tongue at one end, socket at the other (chains: 3 rails reach every cut on the list, 79..387 mm);
           ticks every 10 mm, a deep tick every 50; the first tick is 50 mm from the blade.
  stop     a U over the rail with a 34 x 30 face at the rod heights; a printed wedge (6 deg) drops into a slot in the U's
           top beside the rail and jams it, no screw. Length = ticks: the face on the n-th deep tick of rail k = 50 n + 150 (k - 1).
Use: slide the stop to the length, push the wedge in; rod against the stop, press it into the clips, tape at the slot; saw with
light strokes, quarter turns; file the end at 45 deg. Dust: cut wet or masked.
"""
import os, numpy as np, trimesh, manifold3d as mf

BODY_L, BODY_W, BODY_H = 90.0, 34.0, 30.0
FLANGE_W, FLANGE_H = 60.0, 8.0
CLIP_X, CLIP_GRIP, FINGER, RELIEF, RELIEF_D = ((8.0, 22.0), (68.0, 82.0)), 0.5, 2.0, 1.0, 10.0
FEET = [(12.0, 22.0), (78.0, 22.0), (12.0, -22.0), (78.0, -22.0)]; FOOT_R, FOOT_D = 5.25, 1.0
CH = [(-8.0, 6.4, 8.4), (8.0, 3.4, 5.4)]            # (y, width, depth): O6 and O3 channels
POCKET_W, CLR = 12.0, 0.15
SLOT_BOTTOM = BODY_H - 8.4 - 3.0
KERFS = {"k06": 0.6, "k10": 1.0, "k15": 1.5}
RAIL_L, RAIL_W, RAIL_H = 150.0, 20.0, 12.0
TONGUE = (10.0, 8.0, 10.0)                           # width (y), height (z), length (x)
STOP_T, STOP_FACE_T = 4.0, 6.0                       # U wall, stop face thickness
WEDGE_W, WEDGE_T, WEDGE_ANG = 6.0, 3.0, 6.0          # wedge: width along the rail, thickness at its thin end, taper

def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def channel_cuts(x0, x1):
    cs = []
    for yc, w, d in CH:
        r = w / 2
        cs.append(box(x0, x1, yc - r, yc + r, BODY_H - d + r, BODY_H + 1))
        cs.append(mf.Manifold.cylinder(x1 - x0, r, r, 64).rotate([0, 90, 0]).translate([x0, yc, BODY_H - d + r]))
    return cs
def tm(man):
    m = man.to_mesh(); t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
    assert t.is_watertight and t.is_volume, "not a solid"
    return t

# body
px0, px1 = BODY_L / 2 - POCKET_W / 2, BODY_L / 2 + POCKET_W / 2
body = box(0, BODY_L, -BODY_W / 2, BODY_W / 2, 0, BODY_H) + box(0, BODY_L, -FLANGE_W / 2, FLANGE_W / 2, 0, FLANGE_H)
cuts = []
zones = [(-1, CLIP_X[0][0]), (CLIP_X[0][1], CLIP_X[1][0]), (CLIP_X[1][1], BODY_L + 1)]                   # free channel between the clips
for x0, x1 in zones:
    cuts += channel_cuts(x0, x1)
for x0, x1 in CLIP_X:                                                                                   # clip zones: narrow mouth, round seat, relieved fingers
    for yc, w, d in CH:
        r = w / 2; rod_r = (w - 0.4) / 2; mouth = (w - 0.4 - CLIP_GRIP) / 2
        cuts.append(mf.Manifold.cylinder(x1 - x0, r, r, 64).rotate([0, 90, 0]).translate([x0, yc, BODY_H - d + r]))   # same seat
        cuts.append(box(x0, x1, yc - mouth, yc + mouth, BODY_H - d + r, BODY_H + 1))                              # narrower mouth
        cuts.append(mf.Manifold.hull_points([[x, yc + sg * (mouth + (1.0 if top else 0.0)), BODY_H - (0.0 if top else 1.0) + (1.0 if top else 0.0)]
                                             for x in (x0, x1) for sg in (-1, 1) for top in (False, True)]))   # 45 deg lead-in at the mouth
        for sgn in (-1, 1):                                                                                      # relief slots -> 2.0 fingers
            ya = yc + sgn * (mouth + FINGER); yb = ya + sgn * RELIEF
            cuts.append(box(x0, x1, min(ya, yb), max(ya, yb), BODY_H - RELIEF_D, BODY_H + 1))
cuts.append(box(px0 - CLR, px1 + CLR, -BODY_W / 2 - 1, BODY_W / 2 + 1, FLANGE_H, BODY_H + 1))          # insert pocket
cuts += [mf.Manifold.cylinder(FOOT_D + 0.01, FOOT_R, FOOT_R, 48).translate([fx, fy, -0.01]) for fx, fy in FEET]   # rubber feet recesses
tw, th, tl = TONGUE
cuts.append(box(-1, tl + CLR, -tw / 2 - CLR, tw / 2 + CLR, 2.0, 2.0 + th + 2 * CLR))                   # rail socket in the -x end
BODY = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)

# kerf inserts
INSERTS = {}
for nm, k in KERFS.items():
    ins = box(px0, px1, -BODY_W / 2, BODY_W / 2, FLANGE_H + CLR, BODY_H)
    c = channel_cuts(px0 - 1, px1 + 1) + [box(BODY_L / 2 - k / 2, BODY_L / 2 + k / 2, -BODY_W / 2 - 1, BODY_W / 2 + 1, SLOT_BOTTOM, BODY_H + 1)]
    INSERTS[nm] = ins - mf.Manifold.batch_boolean(c, mf.OpType.Add)

# rail: lies along -x from the body; local coords x 0 (body end) .. -RAIL_L; tongue at x 0 (into the body or the previous rail), socket at the far end
rail = box(-RAIL_L, 0, -RAIL_W / 2, RAIL_W / 2, 0, RAIL_H) + box(0, tl - CLR, -tw / 2 + CLR, tw / 2 - CLR, 2.0 + CLR, 2.0 + th - CLR)
rc = [box(-RAIL_L - 1, -RAIL_L + tl + CLR, -tw / 2 - CLR, tw / 2 + CLR, 2.0, 2.0 + th + 2 * CLR)]
for p in range(5, int(RAIL_L), 10):                                                                    # ticks: L = 45 + p from the blade
    deep = (45 + p) % 50 == 0
    rc.append(box(-p - 0.4, -p + 0.4, -RAIL_W / 2 - 1 if deep else -3.0, RAIL_W / 2 + 1 if deep else 3.0, RAIL_H - (1.0 if deep else 0.6), RAIL_H + 1))
RAIL = rail - mf.Manifold.batch_boolean(rc, mf.OpType.Add)

# stop: a U over the rail plus the face; its -x side is the face the rod hits (the rod comes from +x)
sw = RAIL_W + 2 * CLR + 2 * STOP_T
stop = box(0, STOP_FACE_T + 12.0, -sw / 2, sw / 2, 0.5, RAIL_H + CLR + 6.0)                               # U block over the rail
stop += box(0, STOP_FACE_T, -BODY_W / 2, BODY_W / 2, 0.5, BODY_H)                                        # the face, rod height
# the wedge drops through a slot in the U's top, beside the rail's +y side, and jams between the rail and the U wall: the slot
# is the wedge's thick end + clearance, the U wall there is thinned to leave room for the wedge's taper
wx0, wx1 = STOP_FACE_T + 3.0, STOP_FACE_T + 3.0 + WEDGE_W
wedge_h = RAIL_H - 1.0
wedge_thick = WEDGE_T + wedge_h * np.tan(np.radians(WEDGE_ANG))
sc = [box(-1, STOP_FACE_T + 13, -RAIL_W / 2 - CLR, RAIL_W / 2 + CLR, -1, RAIL_H + CLR),                   # rail passage
      box(wx0 - CLR, wx1 + CLR, RAIL_W / 2 + CLR - 0.01, RAIL_W / 2 + CLR + wedge_thick + 2 * CLR, 0.5, RAIL_H + CLR + 7)]   # wedge slot, open at the top
stop += box(wx0 - CLR - 2.0, wx1 + CLR + 2.0, RAIL_W / 2 + CLR, RAIL_W / 2 + CLR + wedge_thick + 2 * CLR + 2.0, 0.5, RAIL_H + CLR + 6.0)  # wall round the slot
STOP = stop - mf.Manifold.batch_boolean(sc, mf.OpType.Add)
# the wedge: thickness WEDGE_T at the bottom growing with the taper, a 3 mm head on top to push and pull
pts = []
for x in (0.0, WEDGE_W):
    pts += [[x, 0.0, 0.0], [x, WEDGE_T, 0.0], [x, 0.0, wedge_h], [x, wedge_thick, wedge_h]]
WEDGE = mf.Manifold.hull_points(pts) + box(-1.0, WEDGE_W + 1.0, 0.0, wedge_thick, wedge_h, wedge_h + 3.0)

here = os.path.dirname(os.path.abspath(__file__)); out = os.path.join(here, "print")
pieces = {"body": BODY, "rail": RAIL, "stop": STOP, "wedge": WEDGE, **{f"insert_{k}": v for k, v in INSERTS.items()}}
total = 0.0
for nm, man in pieces.items():
    t = tm(man); t.export(os.path.join(out, f"{nm}.stl")); g = t.volume / 1000 * 1.27; total += g if not nm.startswith("insert") else 0
    print(f"  {nm:11s} {t.volume / 1000:6.2f} cm3 {g:5.1f} g  {np.round(t.extents, 1).tolist()}  faces {len(t.faces)}")
# the assembled set for the page: body, insert k10, two rails, the stop at 150 mm from the blade
asm = BODY + INSERTS["k10"] + RAIL + RAIL.translate([-RAIL_L, 0, 0]) + STOP.translate([-(150 - 45) - STOP_FACE_T, 0, 0]) + WEDGE.translate([-(150 - 45) - STOP_FACE_T + wx0, RAIL_W / 2 + CLR, 1.0])
t = tm(asm); t.export(os.path.join(here, "part.stl"))
print(f"rod_saw_guide set: body + 1 insert + 2 rails + stop ~{total + 2 * 0 + 8:.0f} g; assembled preview {np.round(t.extents, 1).tolist()}, {len(t.faces)} faces")
