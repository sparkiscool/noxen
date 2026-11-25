"""Desktop dual-board case, UPRIGHT SUITCASE (Spark 25 Nov: it stands on its end like a standing suitcase the whole time it runs; the Kootek pad's fans
sit flat in the base blowing UP; both boards stand vertical, side by side and parallel to the airflow with a gap between them (each board's hot
side facing the gap), so air rises along both like a chimney and leaves through the lid and the upper end walls; the whole TOP is an exhaust grille; the carry handle is on a SHORT END wall (carried hanging lengthwise, like an elongated suitcase);
the base it stands on is the opposite end; no fold-out legs; the box is fitted to the parts, no wasted space). Flat layers, A1 mini (every tile
<= 180), PCTG (TPU 95A foot pads), no supports.
Frame: x right (width), y back (depth), z up. Closed 372 x 126 x 299 (+ 11 feet, + 55 handle out of the +x end wall).
Kootek pad (Amazon B01469DJLM, owned; mesh top and clips dropped): 1 x 120x120x25 fan + 4 x 70x70x15 fans, all reused.
Boards TIMI A7S + second board (E, outlines not published): 330 x 220 board, 27 deep with its cooler, standing on its 330 edge.
Fit: inner width 366 = set by the fan row (120 + three 70 + gaps + walls; the boards are 330, no panel now); inner depth 120 = the 120 fan; board A at y 6..33, board B at
y 93..120, hot sides facing, gap y 33..93 = 60 (the chimney); inner height 245 = 8 standoff + 220 board + 17 under the lid (70 extractor 15 thick).
Stack, bottom up (z mm): TPU pad 3 + rail 8 (feet 11) | D0 floor 3 | base 34 (the 120 + three 70 fans stand on D0 and blow up; rear and end walls
slotted: intake; D0 bores under each fan: intake from below, the feet lift it 11) | D1 deck 3 (fan bores) | bay 245 (two boards; end walls
slotted over their upper 100: exhaust) | lid 3 (slots, the fifth 70 as an extractor over the gap) | the whole lid a grille. Handle (grip along y, centred, z 118..174 about the centre of gravity z ~145) on the +x end wall on two 3 doublers. Front of the bay: a solid wall.
Front: a plain solid wall of 3 mm tiles (no display, no panel, no driver board: Spark plugs into other monitors); the bay front tiles unscrew from their
L-brackets for the boards' access. Air: intake bottom and rear of the base (hot air never comes back: the exhaust is 250 mm up), exhaust up through the lid and the upper
end walls; the front and the rear wall of the bay are solid. Exhaust area lid ~19000 + ends 2 x 5600 = 30000 mm2 against the gap 60 x 330 = 19800
(E); intake D0 bores 16300 + base slots. Keep 100 mm behind and 60 mm beside for the base intake (E, as the tower note).
Joints: wall tiles screw to M3 L-brackets (bought, 20 x 20) at each corner of each layer, one M3 x 8 + nut through each flange; four M3 x 300 threaded rods
at (12,12) etc., nuts both sides of D0, D1 and the lid, clamp and hang the stack; the two foot rails are clamped by the rods and two M3 x 12 into inserts.
Hand calcs (E): carried weight <= 7 kg (tiles 1.8 + two boards + fans, E) x 3 dynamic = 210 N, carried hanging lengthwise: the +x SHORT end wall is up and the case hangs from
the handle (the pull is along the legs, through the end wall). Handle bar 24 x 14 on an 80 clear span: M = 210 x 80 / 4 = 4200 N mm, Z = 24 x 14^2 / 6 = 784 mm3:
5.4 MPa (PCTG 45 E, layers flat so the stress is along them): margin 8. Legs 15 x 24 = 360 mm2: 105 N each = 0.29 MPa. Each foot is held by 2 M3 x 12 into M3
heat-set inserts (fit table: hole 4.0 x 6) in a 12 deep block, the screw heads (pan head + washer d7, 2.5 high) inside the end wall through 3 wall + 3 doubler, clear of
the boards (end gap 6): bar 16 off the bolt centre line: insert loads 52 +- 1680 / 40 = 94 / 10 N against ~300 N pull-out: margin 3.2; head bearing 94 / 29 mm2 =
3.3 MPa, punching 0.7 MPa: margin > 10. Reinforcement: a 120 x 122 x 3 end tile loaded by a leg 105 N, simply supported 100 mm, effective width 60: 3 mm alone M = 2625 N mm,
Z = 60 x 9 / 6 = 90: 29 MPa (margin 1.5, not enough); with the 3 mm doubler outside (6 mm, Z = 360) 7.3 MPa: margin 6; the tile hangs from its 4 bracket screws (53 N each,
margin > 10). The handle at z 118..174 is about the centre of gravity (z ~145 E): it hangs level (offset < 15 mm) and sits below the end wall's exhaust slots (z > 186).
The lid hangs the stack on 4 rods (52 N each, M3 rod 2 kN, nut washer 1.3 MPa), each deck on its nut pair. Standing: 7 kg, centre of gravity y ~63 (rails y 5..45 and
81..121: margin 58 to the front rail edge); the handle sticks 55 out of the +x end (leave 60 beside). Boards' mounts wait for their measuring (E).
Cooling (research in NOTE.md): fans low and pushing, exhaust high, the whole lid a grille, hot and cold apart, filter sheet on the base slots.
Bought (Amazon searches): M3 L-brackets 20 mm;
M3 x 300 rods; M3 heat-set inserts; M3 nuts and washers.
Run: python parts/desktop_dual_board_case/make.py -> part.stl, preview.png, preview_closed.png, print/desktop_dual_board_case_<tile>.stl"""
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
# ---- lid: the whole top is the exhaust (slots across the full width, the fifth 70 as an extractor over the chimney); no handle on it
FAN_ZONE = (X / 2 - 46, X / 2 + 46)
lid_slots = [g for g in slots_u(16, X - 12, 16, Y - 16, sw=6.0) if not (g.bounds[2] > FAN_ZONE[0] and g.bounds[0] < FAN_ZONE[1])]
lid = cut(box(0, 0, X, Y), rod_holes() + fan_cuts(*FAN70_TOP, 70) + lid_slots)
# ---- the handle sits on the +x SHORT END wall (the smallest face: the case is carried hanging lengthwise, 372 long vertical, like an elongated suitcase):
# grip along y (centred), feet at y HAND_Y, z Z0H..Z0H+56 (below the end wall's exhaust slots, z < 186, about the centre of gravity)
HW, LEG = 110.0, 15.0
HAND_Y = [Y / 2 - HW / 2 + LEG / 2, Y / 2 + HW / 2 - LEG / 2]        # foot centres 15.5, 110.5
Z0H = 118.0
HOLES_H = [(hy, Z0H + hz) for hy in HAND_Y for hz in (8.0, 48.0)]      # (y, z) on the end wall
doublers = []
for hy in HAND_Y:
    d = box(hy - 12, Z0H - 2, hy + 12, Z0H + 58)
    d = cut(d, [Point(py, pz).buffer(M3 / 2, 24) for py, pz in HOLES_H if abs(py - hy) < 1])
    doublers.append(affinity.translate(d, -(hy - 12), -(Z0H - 2)))

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
    "bay_endh": (IY, HL, 10.0, slots_u(18, IY - 18, 135.0, HL - 12.0) + [Point(py - T, pz - Z_B).buffer(M3 / 2, 24) for py, pz in HOLES_H]),
}
wall_poly = {k: wall(*v) for k, v in WALLS.items()}

# ---- the handle: printed flat on its x-y face (hx out of the lid, hy along the bar, hz across the lid); bar 110 x 24 x 14, legs 15 wide
HX_LEG, HX_BAR0, HX_BAR1 = 12.0, 38.0, 52.0
HZ_BAR, HZ_FOOT = 24.0, 56.0
arms = box(HX_LEG, 0, HX_BAR1, LEG).union(box(HX_LEG, HW - LEG, HX_BAR1, HW)).union(box(HX_BAR0, 0, HX_BAR1, HW))
handle = extrude(arms, 0, HZ_BAR)
for y0 in (0.0, HW - LEG):
    handle = boolean(handle, extrude(box(0, y0, HX_LEG + 1.0, y0 + LEG), 0, HZ_FOOT), "+")
bores = []
for ly in (LEG / 2, HW - LEG / 2):
    for lz in (8.0, 48.0):
        cyl = trimesh.creation.cylinder(radius=INS_D / 2, height=INS_DEPTH + 1.0, sections=48)
        cyl.apply_transform(tm.rotation_matrix(np.pi / 2, [0, 1, 0]))
        cyl.apply_translation([(INS_DEPTH - 1.0) / 2, ly, lz]); bores.append(cyl)
for b in bores:
    handle = boolean(handle, b, "-")

# ---- foot rails (along x under D0, y 5..45 and 81..121)
RAIL_W = 40.0
RAIL_Y = [(5.0, RAIL_W), (Y - 5.0 - RAIL_W, RAIL_W)]
RAIL_SEGS = [(0.0, 130.0), (130.0, 252.0), (252.0, X)]
rails = []                                            # (x0, y0, polygon pad/rail) in model coordinates
for (x0, x1) in RAIL_SEGS:
    for y0, rw in RAIL_Y:
        holes = [(rx, ry) for rx, ry in RODS if x0 <= rx <= x1 and y0 <= ry <= y0 + rw]
        if x0 == 130.0:
            holes += [(x, y0 + 7.0 if y0 < 60 else y0 + rw - 7.0) for x in (140.0, 180.0)]
        rails.append((x0, y0, x1 - x0, rw, holes))

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
for k, t in tile_of.items():
    register(k, t, T)
for n, d in enumerate(doublers):
    register("doubler", [((0, 0), d, (24, 60))], T)
for x0, y0, w, rw, holes in rails:
    p = box(0, 0, w, rw)
    p = cut(p, [Point(hx - x0, (hy - y0)).buffer(M3 / 2, 24) for hx, hy in holes])
    register("rail", [((0, 0), p, (w, rw))], FEET_RAIL)
    register("tpu_pad", [((0, 0), box(0, 0, w, rw), (w, rw))], FEET_TPU)

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


THICK = {}


def build():
    ps = []
    for k, z in (("d0", Z_D0), ("d1", Z_D1)):
        ps += tiles_of(k, lambda u, v, z=z: Tr(u, 0, z))
    ps += tiles_of("lid", lambda u, v: Tr(u, 0, Z_LID))
    for lay, z0 in (("base", Z_BW), ("bay", Z_B)):
        ps += tiles_of(f"{lay}_front", lambda u, v, z0=z0: Tr(u, T, z0 + v) @ Rx(np.pi / 2))
        ps += tiles_of(f"{lay}_rear", lambda u, v, z0=z0: Tr(u, Y, z0 + v) @ Rx(np.pi / 2))
        for x0 in (0.0, X - T):
            kk = "bay_endh" if (lay == "bay" and x0 > 0) else f"{lay}_end"
            ps += tiles_of(kk, lambda u, v, z0=z0, x0=x0: Tr(x0, T + u, z0 + v) @ Rz(np.pi / 2) @ Rx(np.pi / 2))
    for d, hy in zip(doublers, HAND_Y):
        ps.append(place(d, T, Tr(X, hy - 12, Z0H - 2) @ Rz(np.pi / 2) @ Rx(np.pi / 2)))
    h = handle.copy(); h.apply_transform(np.array([[1, 0, 0, X + T], [0, 1, 0, Y / 2 - HW / 2], [0, 0, 1, Z0H], [0, 0, 0, 1]], float))
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
    for x0, y0, w, rw, holes in rails:
        r = trimesh.creation.box(extents=[w, rw, FEET_RAIL]); r.apply_translation([x0 + w / 2, y0 + rw / 2, FEET_TPU + FEET_RAIL / 2]); ps.append(r)
        r = trimesh.creation.box(extents=[w, rw, FEET_TPU]); r.apply_translation([x0 + w / 2, y0 + rw / 2, FEET_TPU / 2]); ps.append(r)
    return trimesh.util.concatenate(ps)


closed_m = build()
d = os.path.join(ROOT, "parts", "desktop_dual_board_case")
closed_m.export(os.path.join(d, "part.stl"))
print("closed", np.round(closed_m.extents, 1).tolist())
sys.path.insert(0, ROOT)
from topo.build import render
render(closed_m, os.path.join(d, "preview.png"), title=f"desktop_dual_board_case upright suitcase, no screen ({total} flat tiles)",
       info="STANDING: fans flat in the base blow up | two boards upright, hot sides facing a 60 mm chimney | whole top an exhaust grille + upper end walls | handle on a short end wall | plain front, no display")
render(closed_m, os.path.join(d, "preview_closed.png"), title="desktop_dual_board_case upright suitcase, CLOSED (carry)",
       info=f"{X:.0f} x {Y:.0f} x {Z_TOP:.0f} mm + {FEET:.0f} feet + 55 handle on the short end wall | rails and TPU pads | plain front")
