"""Desktop dual-board case, UPRIGHT TOWER (Spark 24 Nov: stands vertical, take the Kootek pad apart and put its fans where they cool best, one side
against a wall, maybe a corner: exhaust never toward a wall, intake and exhaust well apart, count the hot air coming back off the wall).
Kootek pad (Amazon B01469DJLM, 380 x 300 x 35): one 120 x 120 x 25 fan + four 70 x 70 x 15 fans (E: ~40 CFM and ~12 CFM free air; no published figure).
Two identical bays side by side, each inside 270 (x) x 120 (y) x 402 (z): a board (TIMI A7S TM2037; bay B the same size for the MSI GF65 Thin class
guess, E, neither board's outline/holes/cooler published) stands on its long edge (~385 tall), cooler face toward the FRONT, its exhaust edge UP.
Air path: front fan wall (120 + 70 mm, bottom row, blowing straight into the cooler's intake face) -> board -> chimney -> lid (slots + one 70 mm
extractor per bay). Only the lid exhausts: hot air rises, never goes sideways, so no wall can be in its way. The rear and both ends are solid.
Wall rule: REAR (solid) and either END face a wall or corner; the FRONT stays open (intake). Exhaust (lid, z 408) is 408 - 134 = 274 mm above the
intake row's top edge and the intake is on the front face, the exhaust over the top.
Hot air off the wall (E, printed below): buoyant exhaust climbs the wall it is near, so it needs room; a 100 mm rear gap and 60 mm end gap keep
re-ingestion near 5 % (inlet +<1 K), under 50 mm it climbs the wall and falls forward over the lid edge (up to ~15 %).
Fans: bay A front 120 (pad's) + 70, lid 70; bay B front 70 + lid 70, its 120 position waits for a bought 120 x 25 fan (the 4 pad 70s are all used).
Power: one 14 mm port low in the rear at each outer end. Feet: two 180 x 180 base plates (TPU 95A pads under) stop it tipping.
Corner/edge joints come after boards are measured (E: M3 corner brackets, none drawn yet). Sections: panel 3 mm, spans <= 137 mm (bending is not
the limit; a panel is a skin; the plates carry the 5 kg tower at 10 N per fan).
Run: python parts/desktop_dual_board_case/make.py -> part.stl (assembled view), preview.png, one flat panel STL per unique panel in
print/desktop_dual_board_case_<panel>.stl (3 mm, every panel <= 180 mm, PCTG, flat, no supports)."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import extrude, ROOT  # noqa: E402
from shapely.geometry import box, Point
import numpy as np, trimesh

T = 3.0                        # panel thickness
BX, BY, BZ = 270.0, 120.0, 402.0   # one bay inside
X = 2 * BX + 3 * T             # 549 outer
Y = BY + 2 * T                 # 126
Z = BZ + 2 * T                 # 408
CW = X / 4                     # front/rear/floor/lid tile width 137.25
RH = BZ / 3                    # front/rear tile height 134
FOOT = 180.0


def holes4(cx, cy, pitch, d=4.3):
    return [Point(cx + sx * pitch / 2, cy + sy * pitch / 2).buffer(d / 2, 24) for sx in (-1, 1) for sy in (-1, 1)]


def fan_tile(fan):                       # front tile with a fan opening (120: 113 bore, 105 holes; 70: 66 bore, 61.5 holes) centred
    p = box(0, 0, CW, RH)
    bore, pitch = (113.0, 105.0) if fan == 120 else (66.0, 61.5)
    p = p.difference(Point(CW / 2, RH / 2).buffer(bore / 2, 64))
    for h in holes4(CW / 2, RH / 2, pitch):
        p = p.difference(h)
    return p


def rear_tile(port=None):
    p = box(0, 0, CW, RH)
    if port is not None:
        p = p.difference(Point(port, 20).buffer(7.0, 32))
    return p


def slots(w, h, n=7, sw=8.0, pitch=17.0, margin=14.0, skip=None):   # slots along the deep (y) side
    p = box(0, 0, w, h)
    x0 = (w - ((n - 1) * pitch + sw)) / 2
    for k in range(n):
        x = x0 + k * pitch
        p = p.difference(box(x, margin, x + sw, h - margin))
    return p


def lid_fan_tile():                      # 70 mm extractor over a slotted half
    p = box(0, 0, CW, Y)
    p = p.difference(Point(CW / 2, Y / 2).buffer(33.0, 48))
    for h in holes4(CW / 2, Y / 2, 61.5):
        p = p.difference(h)
    return p


panels = {  # name: (polygon, count)
    "floor_tile": (box(0, 0, CW, Y), 4),
    "lid_tile_slots": (slots(CW, Y), 2),
    "lid_tile_fan70": (lid_fan_tile(), 2),
    "front_tile": (box(0, 0, CW, RH), 2 * 4 - 4 + 4 + 0),   # filled below
    "front_tile_fan120": (fan_tile(120), 2),
    "front_tile_fan70": (fan_tile(70), 2),
    "rear_tile": (rear_tile(), 10),
    "rear_tile_port_left": (rear_tile(13.0 + 7), 1),
    "rear_tile_port_right": (rear_tile(CW - 20.0), 1),
    "side_tile": (box(0, 0, BY, RH), 9),
}
panels["front_tile"] = (box(0, 0, CW, RH), 8)               # 12 front tiles - 4 fan tiles
total = sum(c for _, c in panels.values())

os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
for old in glob.glob(os.path.join(ROOT, "print", "desktop_dual_board_case_*.stl")):
    os.remove(old)
mass = 0.0
for n, (poly, cnt) in panels.items():
    m = extrude(poly, 0, T)
    assert m.is_watertight and max(m.extents) <= 180, n
    m.export(os.path.join(ROOT, "print", f"desktop_dual_board_case_{n}.stl"))
    g = m.volume / 1000 * 1.27
    mass += g * cnt
    print(n, "x", cnt, np.round(m.extents, 1).tolist(), f"{g:.1f} g")
foot = extrude(box(0, 0, FOOT, FOOT), 0, T)
foot.export(os.path.join(ROOT, "print", "desktop_dual_board_case_foot_plate.stl"))
mass += 2 * foot.volume / 1000 * 1.27
total += 2
print("foot_plate x 2", np.round(foot.extents, 1).tolist(), f"total {total} panels, {mass:.0f} g")

# ---- assembled view: x across, y depth (front at y=0), z up; body sits on the foot plates (z = T)
tm = trimesh.transformations
Z0 = T


def place(poly, M):
    m = extrude(poly, 0, T); m.apply_transform(M); return m


parts = []
for cx in (X * 0.17, X * 0.83 - FOOT):                     # foot plates, centred in y on the body
    parts.append(place(box(0, 0, FOOT, FOOT), tm.translation_matrix([cx, (Y - FOOT) / 2, 0])))
for i in range(4):
    parts.append(place(panels["floor_tile"][0], tm.translation_matrix([i * CW, 0, Z0])))
    lid = "lid_tile_fan70" if i in (0, 2) else "lid_tile_slots"
    parts.append(place(panels[lid][0], tm.translation_matrix([i * CW, 0, Z0 + Z - T])))
fan_row = {0: "front_tile_fan120", 1: "front_tile_fan70", 2: "front_tile_fan120", 3: "front_tile_fan70"}
rear_row = {0: "rear_tile_port_left", 3: "rear_tile_port_right"}
for i in range(4):
    for r in range(3):
        fk = fan_row[i] if r == 0 else "front_tile"
        rk = rear_row.get(i, "rear_tile") if r == 0 else "rear_tile"
        for yy, key in ((0.0, fk), (Y - T, rk)):
            M = tm.translation_matrix([i * CW, yy + T, Z0 + T + r * RH]) @ tm.rotation_matrix(np.pi / 2, [1, 0, 0])
            parts.append(place(panels[key][0], M))
for x in (0.0, T + BX, X - T):                            # left end, divider, right end
    for r in range(3):
        M = (tm.translation_matrix([x, T, Z0 + T + r * RH]) @ tm.rotation_matrix(np.pi / 2, [0, 0, 1])
             @ tm.rotation_matrix(np.pi / 2, [1, 0, 0]))
        parts.append(place(panels["side_tile"][0], M))
# the pad's fans, drawn as dark blocks on the front (visual only): 120 and 70 per the layout
asm = trimesh.util.concatenate(parts)
d = os.path.join(ROOT, "parts", "desktop_dual_board_case"); asm.export(os.path.join(d, "part.stl"))
print("assembled", np.round(asm.extents, 1).tolist())

# ---- hot-air-off-the-wall estimate (E)
Q = 100.0                                   # W per board, gutted gaming board sustained (E)
cfm = {"A": 0.7 * (40 + 12), "B": 0.7 * 12}  # effective flow; bay B has only a 70 until the 120 is bought
for b, c in cfm.items():
    v = c * 0.000472
    print(f"bay {b}: {c:.0f} CFM effective, exhaust mean +{Q/(1.2*1005*v):.0f} K over room")
print("rear gap 100 mm / end gap 60 mm -> re-ingestion ~5 %; under 50 mm ~15 % (E)")
sys.path.insert(0, ROOT)
from topo.build import render
render(asm, os.path.join(d, "preview.png"), title=f"desktop_dual_board_case (upright tower, {total} flat panels)",
       info=f"{X:.0f} x {Y:.0f} x {Z+T:.0f} mm | front fan wall in, lid out | rear + ends solid to the wall | 100 mm rear gap")
