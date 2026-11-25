"""Desktop dual-board case, SUITCASE (Spark 24 Nov: a cooling fan base at the bottom that is also the stand, the two laptop boards stacked on it,
a monitor at the very top (laptop screen panel + driver board, USB or HDMI); the stand folds in and the whole unit closes up and carries like a
suitcase, with a carry handle on top). Flat layers, A1 mini (every tile <= 180), PCTG, no supports.
Kootek pad (Amazon B01469DJLM, owned; its mesh top and clips dropped): 1 x 120x120x25 fan + 4 x 70x70x15 fans, reused.
Stack, bottom up (z mm, closed): leg plates folded under (3) | base D0 floor 3 | fan base 34 (walls slotted: intake) | D1 deck 3 (fan bores)
| bay A 75 | D2 3 | bay B 75 | D3 lid 3 | display lid: bezel 3 + driver cavity 13 + back plate 6 (two 3 mm plates) | handle on the back plate.
Closed 400 x 300 x 215 + 45 handle; open: legs down 29, display leans 10 deg back from vertical on the rear edge.
Air: base fans stand on D0 and blow UP through D1 into bay A (120 at x 100, 70 at x 225); intake from below (D0 bores, legs lift 29 mm) and
the base's slotted walls. Bay A also has a 70 in its front wall; bay B has two 70s in its front wall (x 130, 270). Both bays exhaust through
slotted rear and end walls, so each board sees its own fresh air (no board downstream of the other). 5 fans total, all used. Keep 100 mm behind
(hot air off the wall, E, as the tower note) and the display, a hot-air-free lid, above.
Joints: wall tiles screw to M3 L-brackets (bought, 20 x 20) at each corner of each layer, one M3 x 8 + nut through each flange; the stack is
clamped by four M3 x 200 threaded rods (hardware) with nuts, through D0..D3 at (30,30) etc. (corner keep-out 40 x 40 for the boards).
Display lid: two bought butt/friction hinges on the rear edge (it holds its angle), two bought toggle latches on the front edge pull it onto
the lid. Legs: four 160 x 32 plates, two per long edge, on bought small hinges; folded flat under D0, down 90 deg when deployed (a stop is the
hinge's own open limit).
Hand calcs (E): carried weight <= 6 kg x 3 dynamic = 180 N. Handle legs 12 x 25 mm each: 180 / 600 mm2 = 0.3 MPa. Handle on the 6 mm back plate
by 4 M3 x 16 into heat-set inserts (bore per fit table row 'insert'): 45 N each against ~300 N pull-out: margin 6. Whole stack hangs from
back plate -> 2 hinges + 2 latches (4 screws each into 3 mm tiles with washers): 45 N per screw, margin > 5. Leg plate: 60 N / 4 legs, 15 N
each on a 3 x 160 edge: crush 0.03 MPa, buckling of a 3 x 32 strip under 15 N is far below its Euler load (~700 N).
Boards TIMI A7S + second board: outline, holes, cooler position not published, fan bores centred by estimate (E): board mounts wait for measuring.
Panel: 15.6 in laptop panel 360 x 223 x 3 + driver board ~105 x 55 x 8 (E; the chosen panel's model sets the bezel window and cavity).
Bought (Amazon searches, exact listings picked when the panel is chosen): 15.6 in eDP panel + HDMI/USB-C driver board; M3 L-brackets 20 mm;
M3 x 200 rods; 2 torque hinges; 2 mini toggle latches; 4 leg hinges; M3 heat-set inserts.
Run: python parts/desktop_dual_board_case/make.py -> part.stl (open view), preview.png (open), preview_closed.png, print/desktop_dual_board_case_<tile>.stl"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import extrude, ROOT  # noqa: E402
from shapely.geometry import box, Point
from shapely import affinity
import numpy as np, trimesh

T = 3.0
L, W = 400.0, 300.0                 # outer footprint
HB, HL = 34.0, 75.0                 # base wall height, bay wall height
LEG_H, LEG_L = 32.0, 160.0
M3 = 3.4
tm = trimesh.transformations

# z levels (closed, bottom of folded leg = 0)
Z_D0 = T
Z_BW = Z_D0 + T            # base walls start
Z_D1 = Z_BW + HB
Z_A = Z_D1 + T
Z_D2 = Z_A + HL
Z_B = Z_D2 + T
Z_D3 = Z_B + HL
Z_DISP = Z_D3 + T
D_BEZ, D_CAV, D_BACK = 3.0, 13.0, 6.0
D_TOT = D_BEZ + D_CAV + D_BACK


def holes4(cx, cy, pitch, d=4.3):
    return [Point(cx + sx * pitch / 2, cy + sy * pitch / 2).buffer(d / 2, 24) for sx in (-1, 1) for sy in (-1, 1)]


def cut(p, geoms):
    for g in geoms:
        p = p.difference(g)
    return p


def fan_cuts(cx, cy, fan):
    bore, pitch = (113.0, 105.0) if fan == 120 else (66.0, 61.5)
    return [Point(cx, cy).buffer(bore / 2, 64)] + holes4(cx, cy, pitch)


def slot_cuts(length, height, margin=8.0, sw=8.0, pitch=16.0, x_from=18.0, x_to=None, skip=()):
    x_to = length - 18.0 if x_to is None else x_to
    out, x = [], x_from
    while x + sw <= x_to:
        if not any(a <= x + sw / 2 <= b for a, b in skip):
            out.append(box(x, margin, x + sw, height - margin))
        x += pitch
    return out


def deck(openings):
    p = box(0, 0, L, W)
    rods = [Point(x, y).buffer(M3 / 2, 24) for x in (30, L - 30) for y in (30, W - 30)]
    return cut(p, rods + openings)


def wall(length, height, lead, extra=(), slots=True, skip_slots=()):
    p = box(0, 0, length, height)
    br = [Point(u, height / 2).buffer(M3 / 2, 24) for u in (lead, length - lead)]
    sl = slot_cuts(length, height, skip=skip_slots) if slots else []
    return cut(p, br + sl + list(extra))


def slice_tiles(poly, nx, ny, w, h):
    out = []
    for i in range(nx):
        for j in range(ny):
            t = poly.intersection(box(i * w / nx, j * h / ny, (i + 1) * w / nx, (j + 1) * h / ny))
            out.append(((i, j), affinity.translate(t, -i * w / nx, -j * h / ny), (w / nx, h / ny)))
    return out


# ---- the full polygons per layer (deck coordinates x 0..L, y 0..W; wall coordinates u along, v up)
FAN120 = (100.0, W / 2)
FAN70A = (225.0, W / 2)
decks = {
    "d0": deck([Point(*FAN120).buffer(50, 64), Point(*FAN70A).buffer(30, 64)]),            # intake bores under the base fans
    "d1": deck(fan_cuts(*FAN120, 120) + fan_cuts(*FAN70A, 70)),                             # the fans blow up through here
    "d2": deck([]),
    "d3": deck([]),
}
for k in ("d0", "d1", "d2", "d3"):
    pass
# every wall: front/rear span the full L; end walls span the inner 294. Bracket flange hole 13 / 10 from the corner.
WALLS = {  # key: (length, height, lead, extra cuts, slots?, skip) per face
    "base_front": (L, HB, 13.0, [], True, ()), "base_rear": (L, HB, 13.0, [], True, ()), "base_end": (W - 2 * T, HB, 10.0, [], True, ()),
    "a_front": (L, HL, 13.0, fan_cuts(200.0, HL / 2, 70), False, ()), "a_rear": (L, HL, 13.0, [], True, ()), "a_end": (W - 2 * T, HL, 10.0, [], True, ()),
    "b_front": (L, HL, 13.0, fan_cuts(130.0, HL / 2, 70) + fan_cuts(270.0, HL / 2, 70), False, ()), "b_rear": (L, HL, 13.0, [], True, ()),
    "b_end": (W - 2 * T, HL, 10.0, [], True, ()),
}
wall_poly = {k: wall(a, b, c, e, s, sk) for k, (a, b, c, e, s, sk) in WALLS.items()}


# ---- display lid plates (x 0..L, y 0..W, in the lid's own frame; bezel faces down onto the lid when closed)
PW, PH = 360.0, 223.0                                      # panel (E)
c = (L / 2, W / 2)
bezel = cut(box(0, 0, L, W), [box(c[0] - 173, c[1] - 97, c[0] + 173, c[1] + 97)] +
            [Point(x, y).buffer(M3 / 2, 24) for x in (30, L - 30) for y in (30, W - 30)])      # 346 x 194 window (15.6 in view area)
cavity = cut(box(0, 0, L, W), [box(c[0] - PW / 2 - 2, c[1] - PH / 2 - 2, c[0] + PW / 2 + 2, c[1] + PH / 2 + 2)] +
             [Point(x, y).buffer(M3 / 2, 24) for x in (30, L - 30) for y in (30, W - 30)])
HOLE_H = [(80.0, 150.0), (L - 80.0, 150.0)]                # handle feet inserts: 4 pts (see below), back plate
handle_feet = [(c[0] - 55 - 10, c[1] - 0), (c[0] + 55 + 10, c[1] - 0)]
back = cut(box(0, 0, L, W), [Point(x, y).buffer(M3 / 2, 24) for x in (30, L - 30) for y in (30, W - 30)] +
           [Point(fx + dx, c[1] + dy).buffer(M3 / 2, 24) for fx, _ in handle_feet for dx in (0,) for dy in (-8, 8)])

# ---- the handle (flat in x-z, thickness 12): 160 wide, 45 tall, legs 25 wide, grip 13 thick
HW, HH, HT = 160.0, 45.0, 12.0
handle = cut(box(0, 0, HW, HH), [box(25, -1, HW - 25, 32)])
handle = handle.difference(box(0, 0, 0.01, 0.01))

legp = box(0, 0, LEG_L, LEG_H)

# ---- tile everything, dedupe, export
os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
for old in glob.glob(os.path.join(ROOT, "print", "desktop_dual_board_case_*.stl")):
    os.remove(old)
unique, mass = {}, 0.0


def register(prefix, tiles, thick):
    """tiles: [((i,j), poly, (w,h))] -> list of (name, poly) placed; STL once per unique shape"""
    names = []
    for (i, j), p, (w, h) in tiles:
        key = (round(p.area, 1), round(w, 1), round(h, 1), tuple(np.round(p.bounds, 1)), prefix.rsplit("_", 1)[0], thick)
        # dedupe only identical geometry (area + a probe of its vertices)
        sig = (key, tuple(sorted((round(x, 1), round(y, 1)) for x, y in (np.array(p.exterior.coords) if p.geom_type == "Polygon" else np.array(max(p.geoms, key=lambda g: g.area).exterior.coords))))[:40])
        if sig not in unique:
            unique[sig] = [f"{prefix}_{i}{j}", p, thick, 0]
        unique[sig][3] += 1
        names.append((unique[sig][0], p))
    return names


tile_of = {}
for k, p in decks.items():
    tile_of[k] = slice_tiles(p, 3, 2, L, W)
for k, p in wall_poly.items():
    length, height = WALLS[k][0], WALLS[k][1]
    tile_of[k] = slice_tiles(p, 3 if length == L else 2, 1, length, height)
tile_of["bezel"] = slice_tiles(bezel, 3, 2, L, W)
tile_of["cavity"] = slice_tiles(cavity, 3, 2, L, W)
tile_of["back"] = slice_tiles(back, 3, 2, L, W)
for k, t in tile_of.items():
    thick = {"cavity": D_CAV}.get(k, T)
    register(k, t, thick)
unique[("handle",)] = ["handle", handle, HT, 2]
unique[("leg",)] = ["leg_plate", legp, T, 4]

total = 0
for sig, (name, p, th, cnt) in unique.items():
    m = extrude(p, 0, th)
    assert m.is_watertight and max(m.extents[:2]) <= 180.01, (name, m.extents)
    m.export(os.path.join(ROOT, "print", f"desktop_dual_board_case_{name}.stl"))
    mass += m.volume / 1000 * 1.23 * cnt
    total += cnt
    print(name, "x", cnt, np.round(m.extents, 1).tolist())
print(f"{total} printed tiles, {mass:.0f} g (PCTG 1.23)")


# ---- assembled views
def place(poly, th, M):
    m = extrude(poly, 0, th); m.apply_transform(M); return m


def Rx(a): return tm.rotation_matrix(a, [1, 0, 0])
def Rz(a): return tm.rotation_matrix(a, [0, 0, 1])
def Tr(x, y, z): return tm.translation_matrix([x, y, z])


def wall_ms(poly_key, kind, z0, x0=0.0):
    """kind front/rear/end_l/end_r for a wall made of its tiles"""
    ms = []
    n = len(tile_of[poly_key]); length = WALLS[poly_key][0]
    for (i, j), p, (w, h) in tile_of[poly_key]:
        u = i * length / n
        if kind == "front":
            M = Tr(u, T, z0) @ Rx(np.pi / 2)
        elif kind == "rear":
            M = Tr(u, W, z0) @ Rx(np.pi / 2)
        else:
            x = 0.0 if kind == "end_l" else L - T
            M = Tr(x, T + u, z0) @ Rz(np.pi / 2) @ Rx(np.pi / 2)
        ms.append(place(p, T, M))
    return ms


def build(open_state):
    lift = LEG_H - T if open_state else 0.0
    ps = []
    for k, z in (("d0", Z_D0), ("d1", Z_D1), ("d2", Z_D2), ("d3", Z_D3)):
        for (i, j), p, (w, h) in tile_of[k]:
            ps.append(place(p, T, Tr(i * L / 3, j * W / 2, z + lift)))
    for lay, z in (("base", Z_BW), ("a", Z_A), ("b", Z_B)):
        for kind, nm in (("front", "front"), ("rear", "rear"), ("end_l", "end"), ("end_r", "end")):
            ps += [m.apply_transform(Tr(0, 0, lift)) or m for m in wall_ms(f"{lay}_{nm}", kind, z)]
    # legs: two per long edge, folded under D0 (z 0..3) or hanging down from the hinge line (y edge, z Z_D0)
    for y_edge in (0.0, W):
        for lx in (30.0, L - 30.0 - LEG_L):
            if open_state:
                if y_edge == 0.0:
                    M = Tr(lx, 0, Z_D0 + lift) @ Rx(-np.pi / 2)
                else:
                    M = Tr(lx, W, Z_D0 + lift) @ Rx(-np.pi / 2) @ Tr(0, 0, 0)
                ps.append(place(legp, T, M @ Tr(0, 0, -T if y_edge == W else 0)))
            else:
                y0 = 0.0 if y_edge == 0.0 else W - LEG_H
                ps.append(place(legp, T, Tr(lx, y0, 0)))
    # display lid + handle: built in the lid frame (closed: bezel on the lid), then hinged about the rear top edge
    lid = []
    zz = Z_DISP + lift
    for k, th, z in (("bezel", T, zz), ("cavity", D_CAV, zz + D_BEZ), ("back", D_BACK, zz + D_BEZ + D_CAV)):
        for (i, j), p, (w, h) in tile_of[k]:
            lid.append(place(p, th, Tr(i * L / 3, j * W / 2, z)))
    pan = trimesh.creation.box(extents=[PW, PH, 3]); pan.apply_translation([c[0], c[1], zz + D_BEZ + 1.5]); lid.append(pan)
    drv = trimesh.creation.box(extents=[105, 55, 8]); drv.apply_translation([c[0] + 100, c[1] - 60, zz + D_BEZ + 3 + 4]); lid.append(drv)
    top = zz + D_TOT
    for fx in (c[0] - HW / 2, ):
        h = extrude(handle, 0, HT); h.apply_transform(Tr(fx, c[1] + HT / 2, top) @ Rx(np.pi / 2)); lid.append(h)
    lm = trimesh.util.concatenate(lid)
    if open_state:
        hinge_z = Z_D3 + T + lift
        lm.apply_transform(Tr(0, W, hinge_z) @ Rx(np.deg2rad(-100)) @ Tr(0, -W, -hinge_z))
    ps.append(lm)
    # rods
    for x in (30, L - 30):
        for y in (30, W - 30):
            r = trimesh.creation.cylinder(radius=1.5, height=Z_D3 + T + 8 + lift - 0.0, sections=12)
            r.apply_translation([x, y, (Z_D3 + T + 8 + lift) / 2]); ps.append(r)
    return trimesh.util.concatenate(ps)


open_m, closed_m = build(True), build(False)
d = os.path.join(ROOT, "parts", "desktop_dual_board_case")
open_m.export(os.path.join(d, "part.stl"))
print("open", np.round(open_m.extents, 1).tolist(), "closed", np.round(closed_m.extents, 1).tolist())
sys.path.insert(0, ROOT)
from topo.build import render
render(open_m, os.path.join(d, "preview.png"), title=f"desktop_dual_board_case suitcase, OPEN ({total} flat tiles)",
       info="fan base = stand (legs down) | board A | board B | display on the rear edge | handle on its back | all tiles <= 180, flat")
render(closed_m, os.path.join(d, "preview_closed.png"), title="desktop_dual_board_case suitcase, CLOSED (carry)",
       info=f"{L:.0f} x {W:.0f} x {Z_DISP + D_TOT:.0f} mm + 45 handle | legs folded under | display lid latched on top")
