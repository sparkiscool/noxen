"""Desktop dual-board case, UPRIGHT SUITCASE (Spark 25 Nov: it stands on its end like a standing suitcase the whole time it runs; the Kootek pad's fans
sit flat in the base blowing UP; both boards stand vertical, side by side and parallel to the airflow with a gap between them (each board's hot
side facing the gap), so air rises along both like a chimney and leaves through the lid and the upper end walls; the carry handle is on the TOP,
the base it stands on is the opposite end; no fold-out legs; the box is fitted to the parts, no wasted space). Flat layers, A1 mini (every tile
<= 180), PCTG (TPU 95A foot pads), no supports.
Frame: x right (width), y back (depth), z up. Closed 372 x 126 x 299 (+ 22 door on the front, + 11 feet, + 55 handle on top).
Kootek pad (Amazon B01469DJLM, owned; mesh top and clips dropped): 1 x 120x120x25 fan + 4 x 70x70x15 fans, all reused.
Boards TIMI A7S + second board (E, outlines not published): 330 x 220 board, 27 deep with its cooler, standing on its 330 edge.
Fit: inner width 366 = the 15.6 in panel's 360 + 6 (the door is as wide as the case); inner depth 120 = the 120 fan; board A at y 6..33, board B at
y 93..120, hot sides facing, gap y 33..93 = 60 (the chimney); inner height 245 = 8 standoff + 220 board + 17 under the lid (70 extractor 15 thick).
Stack, bottom up (z mm): TPU pad 3 + rail 8 (feet 11) | D0 floor 3 | base 34 (the 120 + three 70 fans stand on D0 and blow up; rear and end walls
slotted: intake; D0 bores under each fan: intake from below, the feet lift it 11) | D1 deck 3 (fan bores) | bay 245 (two boards; end walls
slotted over their upper 100: exhaust) | lid 3 (slots, the fifth 70 as an extractor over the gap) | handle on a 3 doubler. Front of the bay: a solid wall, then the door.
Door (the display, on the FRONT face): bezel 3 + cavity 13 + back plate 6 = 22, 372 x 288, 15.6 in panel 360 x 223 x 3 + driver board in the cavity,
two friction hinges on its left vertical edge, two toggle latches on the right: it swings out and opens ~175 deg to stand beside the case with the
screen toward you. Air: intake bottom and rear of the base (hot air never comes back: the exhaust is 250 mm up), exhaust up through the lid and the upper
end walls; the front (door) and the rear wall of the bay are solid. Exhaust area lid ~7500 + ends 2 x 5600 = 18700 mm2 against the gap 60 x 330 = 19800
(E); intake D0 bores 16300 + base slots. Keep 100 mm behind and 60 mm beside for the base intake (E, as the tower note).
Joints: wall tiles screw to M3 L-brackets (bought, 20 x 20) at each corner of each layer, one M3 x 8 + nut through each flange; four M3 x 300 threaded rods
at (12,12) etc., nuts both sides of D0, D1 and the lid, clamp and hang the stack; the two foot rails are clamped by the rods and two M3 x 12 into inserts.
Hand calcs (E): carried weight <= 7 kg (tiles 3.3 + two boards + panel + fans, E) x 3 dynamic = 210 N, the case hangs from its handle (the pull is along the legs). Handle bar 24 x 14 on a 110 span:
M = 210 x 110 / 4 = 5775 N mm, Z = 24 x 14^2 / 6 = 784 mm3: 7.4 MPa (PCTG 45 E, layers flat so the stress is along them): margin 6. Legs 25 x 24 = 600 mm2:
105 N each = 0.18 MPa. Each foot is held by 2 M3 x 12 into M3 heat-set inserts (fit table: hole 4.0 x 6) in a 12 deep block, the screw heads inside the lid
through a 3 lid + 3 doubler (washer d7); the bar is 16 off the bolt centre line: insert loads 52 +- 1680 / 40 = 94 / 10 N against ~300 N pull-out: margin 3.2;
head bearing 94 / 29 mm2 = 3.3 MPa, punching 0.7 MPa: margin > 10. The lid hangs the stack on 4 rods (52 N each, M3 rod 2 kN, nut washer 1.3 MPa),
each deck on its nut pair. Door 2.2 kg (E) = 22 N on two hinges, four screws of 3 mm tiles with washers: 17 N each, margin > 5. Standing: 7 kg
with the door open at 175 deg, centre of gravity x 69 (rails reach x -150..372: margin 219), y 35 (front rail edge y 5: margin 30, tips at ~11 deg
forward: do not lean on the open door). Boards' mounts wait for their measuring (E).
Panel: 15.6 in laptop panel 360 x 223 x 3 + driver board ~105 x 55 x 8 (E; the chosen panel's model sets the bezel window and cavity).
Bought (Amazon searches, exact listings picked when the panel is chosen): 15.6 in eDP panel + HDMI/USB-C driver board; M3 L-brackets 20 mm;
M3 x 300 rods; 2 friction hinges; 2 mini toggle latches; M3 heat-set inserts; M3 nuts and washers.
Run: python parts/desktop_dual_board_case/make.py -> part.stl (open view), preview.png (open), preview_closed.png, print/desktop_dual_board_case_<tile>.stl"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import extrude, boolean, ROOT  # noqa: E402
from shapely.geometry import box, Point
from shapely import affinity
import numpy as np, trimesh

T = 3.0
X, Y = 372.0, 126.0                  # outer footprint (inner 366 x 120)
IX, IY = X - 2 * T, Y - 2 * T
HB, HL = 34.0, 245.0                 # base wall height, bay wall height
FEET_RAIL, FEET_TPU = 8.0, 3.0
FEET = FEET_RAIL + FEET_TPU
M3 = 3.4
INS_D, INS_DEPTH = 4.0, 6.0          # M3 heat-set insert hole (fit table)
tm = trimesh.transformations

Z_D0 = FEET
Z_BW = Z_D0 + T
Z_D1 = Z_BW + HB
Z_B = Z_D1 + T                       # bay walls start
Z_LID = Z_B + HL
Z_TOP = Z_LID + T
D_BEZ, D_CAV, D_BACK = 3.0, 13.0, 6.0
D_TOT = D_BEZ + D_CAV + D_BACK
DOOR_H = Z_TOP - FEET
RODS = [(12.0, 12.0), (X - 12.0, 12.0), (12.0, Y - 12.0), (X - 12.0, Y - 12.0)]
FANY = Y / 2
FAN120 = (T + 60.0, FANY)
FAN70 = [(T + 60.0 + 60.0 + 8.0 + 35.0 + 78.0 * k, FANY) for k in range(3)]
FAN70_TOP = (X / 2, FANY)


def holes4(cx, cy, pitch, d=4.3):
    return [Point(cx + sx * pitch / 2, cy + sy * pitch / 2).buffer(d / 2, 24) for sx in (-1, 1) for sy in (-1, 1)]


def cut(p, geoms):
    for g in geoms:
        p = p.difference(g)
    return p


def fan_cuts(cx, cy, fan):
    bore, pitch = (113.0, 105.0) if fan == 120 else (66.0, 61.5)
    return [Point(cx, cy).buffer(bore / 2, 64)] + holes4(cx, cy, pitch)


def rod_holes():
    return [Point(x, y).buffer(M3 / 2, 24) for x, y in RODS]


def slots_u(lo, hi, v0, v1, sw=8.0, pitch=16.0):
    out, x = [], lo
    while x + sw <= hi:
        out.append(box(x, v0, x + sw, v1)); x += pitch
    return out


def slice_tiles(poly, nx, ny, w, h):
    out = []
    for i in range(nx):
        for j in range(ny):
            t = poly.intersection(box(i * w / nx, j * h / ny, (i + 1) * w / nx, (j + 1) * h / ny))
            out.append(((i, j), affinity.translate(t, -i * w / nx, -j * h / ny), (w / nx, h / ny)))
    return out


# ---- decks (x 0..X, y 0..Y)
decks = {
    "d0": cut(box(0, 0, X, Y), rod_holes() + [Point(*FAN120).buffer(50, 64)] + [Point(*f).buffer(30, 64) for f in FAN70]),   # intake bores
    "d1": cut(box(0, 0, X, Y), rod_holes() + fan_cuts(*FAN120, 120) + sum([fan_cuts(*f, 70) for f in FAN70], [])),          # the fans blow up through here
}
# ---- lid: slots over the two sides of the handle, the extractor over the gap; solid under the handle feet
HAND_X = [X / 2 - 67.5, X / 2 + 67.5]                 # foot centres
HAND_Y0 = 35.0                                        # foot block y 35..91 (bar y 35..59)
HOLES_H = [(hx, HAND_Y0 + hz) for hx in HAND_X for hz in (8.0, 48.0)]
lid_slots = slots_u(24, 100, 16, Y - 16) + slots_u(272, 348, 16, Y - 16)
lid = cut(box(0, 0, X, Y), rod_holes() + fan_cuts(*FAN70_TOP, 70) + lid_slots + [Point(*p).buffer(M3 / 2, 24) for p in HOLES_H])
doublers = []
for hx in HAND_X:
    d = box(hx - 20, HAND_Y0 - 4, hx + 20, HAND_Y0 + 60)
    d = cut(d, [Point(px, py).buffer(M3 / 2, 24) for px, py in HOLES_H if abs(px - hx) < 1])
    doublers.append(affinity.translate(d, -(hx - 20), -(HAND_Y0 - 4)))

# ---- walls: (length, height, lead, extra, slots list)
def wall(length, height, lead, slots=()):
    p = box(0, 0, length, height)
    vs = [height / 2] if height < 60 else [height * 0.25, height * 0.75]
    br = [Point(u, v).buffer(M3 / 2, 24) for u in (lead, length - lead) for v in vs]
    return cut(p, br + list(slots))


WALLS = {
    "base_front": (X, HB, 13.0, ()), "base_rear": (X, HB, 13.0, slots_u(18, X - 18, 8, HB - 8)),
    "base_end": (IY, HB, 10.0, slots_u(18, IY - 18, 8, HB - 8)),
    "bay_front": (X, HL, 13.0, ()), "bay_rear": (X, HL, 13.0, ()),
    "bay_end": (IY, HL, 10.0, slots_u(18, IY - 18, 135.0, HL - 12.0)),
}
wall_poly = {k: wall(*v) for k, v in WALLS.items()}

# ---- the door (the display, in its own frame: u along x, v up from the door's bottom)
PW, PH = 360.0, 223.0
c = (X / 2, DOOR_H / 2)
hinge_h = [Point(x, y).buffer(M3 / 2, 24) for x in (8.0,) for y in (40.0, DOOR_H - 40.0)]
bezel = cut(box(0, 0, X, DOOR_H), [box(c[0] - 173, c[1] - 97, c[0] + 173, c[1] + 97)])
cavity = cut(box(0, 0, X, DOOR_H), [box(c[0] - PW / 2 - 2, c[1] - PH / 2 - 2, c[0] + PW / 2 + 2, c[1] + PH / 2 + 2)])
back = box(0, 0, X, DOOR_H)

# ---- the handle: printed flat on its x-y face (hx out of the lid, hy along the bar, hz across the lid); bar 160 x 24 x 14, legs 25 wide
HW, HX_LEG, HX_BAR0, HX_BAR1 = 160.0, 12.0, 38.0, 52.0
HZ_BAR, HZ_FOOT = 24.0, 56.0
arms = box(HX_LEG, 0, HX_BAR1, 25).union(box(HX_LEG, HW - 25, HX_BAR1, HW)).union(box(HX_BAR0, 0, HX_BAR1, HW))
handle = extrude(arms, 0, HZ_BAR)
for y0 in (0.0, HW - 25):
    handle = boolean(handle, extrude(box(0, y0, HX_LEG + 1.0, y0 + 25), 0, HZ_FOOT), "+")
bores = []
for ly in (12.5, HW - 12.5):
    for lz in (8.0, 48.0):
        cyl = trimesh.creation.cylinder(radius=INS_D / 2, height=INS_DEPTH + 1.0, sections=48)
        cyl.apply_transform(tm.rotation_matrix(np.pi / 2, [0, 1, 0]))
        cyl.apply_translation([(INS_DEPTH - 1.0) / 2, ly, lz]); bores.append(cyl)
for b in bores:
    handle = boolean(handle, b, "-")

# ---- foot rails (along x under D0, y 5..45 and 81..121; the left reaches 150 out as an outrigger for the open door)
RAIL_W = 40.0
RAIL_Y = [5.0, Y - 5.0 - RAIL_W]
RAIL_SEGS = [(-150.0, 30.0), (30.0, 192.0), (192.0, X)]
rails = []                                            # (x0, y0, polygon pad/rail) in model coordinates
for (x0, x1) in RAIL_SEGS:
    for y0 in RAIL_Y:
        holes = [(rx, ry) for rx, ry in RODS if x0 <= rx <= x1 and y0 <= ry <= y0 + RAIL_W]
        if x0 == 30.0:
            holes += [(x, y0 + 7.0 if y0 == RAIL_Y[0] else y0 + RAIL_W - 7.0) for x in (140.0, 180.0)]
        rails.append((x0, y0, x1 - x0, holes))

# ---- tile everything, dedupe, export
os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
for old in glob.glob(os.path.join(ROOT, "print", "desktop_dual_board_case_*.stl")):
    os.remove(old)
unique, mass = {}, 0.0


def register(prefix, tiles, thick):
    names = []
    for (i, j), p, (w, h) in tiles:
        g = p if p.geom_type == "Polygon" else max(p.geoms, key=lambda q: q.area)
        sig = (prefix.rsplit("_", 1)[0], round(p.area, 1), round(w, 1), round(h, 1), thick,
               tuple(sorted((round(x, 1), round(y, 1)) for x, y in np.array(g.exterior.coords)))[:40],
               tuple(sorted((round(r.centroid.x, 1), round(r.centroid.y, 1)) for r in g.interiors)))
        if sig not in unique:
            nm, k = f"{prefix}_{i}{j}", 1
            while any(v[0] == nm for v in unique.values()):
                nm = f"{prefix}_{i}{j}_{k}"; k += 1
            unique[sig] = [nm, p, thick, 0]
        unique[sig][3] += 1
        names.append(unique[sig][0])
    return names


tile_of = {}
for k, p in decks.items():
    tile_of[k] = slice_tiles(p, 3, 1, X, Y)
tile_of["lid"] = slice_tiles(lid, 3, 1, X, Y)
for k, p in wall_poly.items():
    length, height = WALLS[k][0], WALLS[k][1]
    tile_of[k] = slice_tiles(p, 3 if length == X else 1, 2 if height > 180 else 1, length, height)
for k, p in (("bezel", bezel), ("cavity", cavity), ("back", back)):
    tile_of[k] = slice_tiles(p, 3, 2, X, DOOR_H)
for k, t in tile_of.items():
    register(k, t, {"cavity": D_CAV, "back": D_BACK}.get(k, T))
for n, d in enumerate(doublers):
    register("doubler", [((0, 0), d, (40, 64))], T)
for x0, y0, w, holes in rails:
    p = box(0, 0, w, RAIL_W)
    p = cut(p, [Point(hx - x0, (hy - y0)).buffer(M3 / 2, 24) for hx, hy in holes])
    register("rail", [((0, 0), p, (w, RAIL_W))], FEET_RAIL)
    register("tpu_pad", [((0, 0), box(0, 0, w, RAIL_W), (w, RAIL_W))], FEET_TPU)

total = sum(v[3] for v in unique.values()) + 1
for sig, (name, p, th, cnt) in unique.items():
    m = extrude(p, 0, th)
    assert m.is_watertight and max(m.extents[:2]) <= 180.01, (name, m.extents)
    m.export(os.path.join(ROOT, "print", f"desktop_dual_board_case_{name}.stl"))
    mass += m.volume / 1000 * (1.21 if name.startswith("tpu") else 1.23) * cnt
    print(name, "x", cnt, np.round(m.extents, 1).tolist())
assert handle.is_watertight and max(handle.extents) <= 180.01, handle.extents
handle.export(os.path.join(ROOT, "print", "desktop_dual_board_case_handle.stl"))
mass += handle.volume / 1000 * 1.23
print("handle x 1", np.round(handle.extents, 1).tolist())
print(f"{total} printed tiles, {mass:.0f} g (PCTG 1.23, TPU 1.21)")


# ---- assembled views
def place(poly, th, M):
    m = extrude(poly, 0, th); m.apply_transform(M); return m


def Rx(a): return tm.rotation_matrix(a, [1, 0, 0])
def Rz(a): return tm.rotation_matrix(a, [0, 0, 1])
def Tr(x, y, z): return tm.translation_matrix([x, y, z])


def tiles_of(key, M_of):
    """M_of(u0, v0) -> transform for the tile whose lower-left corner is at (u0, v0) of the polygon"""
    ms = []
    n = len(tile_of[key])
    for (i, j), p, (w, h) in tile_of[key]:
        ms.append(place(p, THICK.get(key, T), M_of(i * w, j * h)))
    return ms


THICK = {"cavity": D_CAV, "back": D_BACK}


def build(open_state):
    ps = []
    for k, z in (("d0", Z_D0), ("d1", Z_D1)):
        ps += tiles_of(k, lambda u, v, z=z: Tr(u, 0, z))
    ps += tiles_of("lid", lambda u, v: Tr(u, 0, Z_LID))
    for lay, z0 in (("base", Z_BW), ("bay", Z_B)):
        ps += tiles_of(f"{lay}_front", lambda u, v, z0=z0: Tr(u, T, z0 + v) @ Rx(np.pi / 2))
        ps += tiles_of(f"{lay}_rear", lambda u, v, z0=z0: Tr(u, Y, z0 + v) @ Rx(np.pi / 2))
        for x0 in (0.0, X - T):
            ps += tiles_of(f"{lay}_end", lambda u, v, z0=z0, x0=x0: Tr(x0, T + u, z0 + v) @ Rz(np.pi / 2) @ Rx(np.pi / 2))
    for d, hx in zip(doublers, HAND_X):
        ps.append(place(d, T, Tr(hx - 20, HAND_Y0 - 4, Z_TOP)))
    h = handle.copy(); h.apply_transform(np.array([[0, 1, 0, X / 2 - HW / 2], [0, 0, 1, HAND_Y0], [1, 0, 0, Z_TOP + T], [0, 0, 0, 1]], float))
    ps.append(h)
    # fans (reused Kootek pad), rods
    for fx, fy in [FAN120]:
        f = trimesh.creation.box(extents=[120, 120, 25]); f.apply_translation([fx, fy, Z_BW + 12.5]); ps.append(f)
    for fx, fy in FAN70:
        f = trimesh.creation.box(extents=[70, 70, 15]); f.apply_translation([fx, fy, Z_BW + 7.5]); ps.append(f)
    f = trimesh.creation.box(extents=[70, 70, 15]); f.apply_translation([FAN70_TOP[0], FANY, Z_LID - 7.5]); ps.append(f)
    for x, y in RODS:
        r = trimesh.creation.cylinder(radius=1.5, height=Z_TOP + 8 - Z_D0, sections=12)
        r.apply_translation([x, y, (Z_TOP + 8 + Z_D0) / 2]); ps.append(r)
    # boards: 330 x 27 x 220, hot side toward the gap
    for y0 in (6.0, Y - 6.0 - 27.0):
        b = trimesh.creation.box(extents=[330, 27, 220]); b.apply_translation([X / 2, y0 + 13.5, Z_B + 8 + 110]); ps.append(b)
    # foot rails + TPU pads
    for x0, y0, w, holes in rails:
        r = trimesh.creation.box(extents=[w, RAIL_W, FEET_RAIL]); r.apply_translation([x0 + w / 2, y0 + RAIL_W / 2, FEET_TPU + FEET_RAIL / 2]); ps.append(r)
        r = trimesh.creation.box(extents=[w, RAIL_W, FEET_TPU]); r.apply_translation([x0 + w / 2, y0 + RAIL_W / 2, FEET_TPU / 2]); ps.append(r)
    # door: faces down onto the front (y<0), hinged on its left vertical edge (x 0, y 0)
    door = []
    for key, th, y_front in (("bezel", D_BEZ, 0.0), ("cavity", D_CAV, -D_BEZ), ("back", D_BACK, -D_BEZ - D_CAV)):
        door += tiles_of(key, lambda u, v, yf=y_front: Tr(u, yf, FEET + v) @ Rx(np.pi / 2))
    pan = trimesh.creation.box(extents=[PW, 3, PH]); pan.apply_translation([c[0], -D_BEZ - 1.5, FEET + c[1]]); door.append(pan)
    drv = trimesh.creation.box(extents=[105, 8, 55]); drv.apply_translation([c[0] + 100, -D_BEZ - 3 - 4, FEET + c[1] - 80]); door.append(drv)
    dm = trimesh.util.concatenate(door)
    if open_state:
        dm.apply_transform(Rz(np.deg2rad(-175)))
    ps.append(dm)
    return trimesh.util.concatenate(ps)


open_m, closed_m = build(True), build(False)
d = os.path.join(ROOT, "parts", "desktop_dual_board_case")
open_m.export(os.path.join(d, "part.stl"))
print("open", np.round(open_m.extents, 1).tolist(), "closed", np.round(closed_m.extents, 1).tolist())
sys.path.insert(0, ROOT)
from topo.build import render
render(open_m, os.path.join(d, "preview.png"), title=f"desktop_dual_board_case upright suitcase, OPEN ({total} flat tiles)",
       info="STANDING: fans flat in the base blow up | two boards upright, hot sides facing a 60 mm chimney | exhaust lid + upper end walls | handle on top | door (display) open 175 deg")
render(closed_m, os.path.join(d, "preview_closed.png"), title="desktop_dual_board_case upright suitcase, CLOSED (carry)",
       info=f"{X:.0f} x {Y:.0f} x {Z_TOP:.0f} mm + {D_TOT:.0f} door + {FEET:.0f} feet + 55 handle on top | rails and TPU pads | door latched on the front")
