"""Desktop dual-board case, REDESIGN (Spark 24 Nov): two gutted laptop boards side by side, built around his Kootek cooling pad (Amazon B01469DJLM, ordered 24 Nov 2026;
listing 380 x 300 x 35 mm, one 120 x 120 x 25 fan + four 70 x 70 x 15 fans, USB powered; airflow not published). Each bay is a pad seat: the pad lies on the floor, the board sits on it
exactly as a laptop sits on a pad, so the pad's five fans blow up into the cooler's bottom intake; the pad pulls its own air through the slotted floor. The cooler's rear exhaust leaves through
a slotted rear wall; the lid is slotted for heat rise. No extra case fans (the pad is the cooling); bay B is a second pad seat of the same size until he buys one.
Bay inside 385 x 304 x 90: pad 35 + board stack about 30 (E: 15.6 in gaming chassis about 360 x 250 mm, a gutted board and cooler about 25-30 mm) + headroom.
TIMI A7S (TM2037) board outline, holes and cooler layout: not published anywhere found (E, not measured); the second board is assumed the same until named.
Power: one 14 mm cable port in the rear wall at each OUTER end (one per board's brick). Interior mounts and corner joints come after the boards are measured.
Run: python parts/desktop_dual_board_case/make.py -> part.stl (assembled view), preview.png, one flat panel STL per unique panel in print/desktop_dual_board_case_<panel>.stl (3 mm, every panel <= 180 mm, PCTG, flat, no supports)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scout_v4_probe_mech_lib import extrude, ROOT  # noqa: E402
from shapely.geometry import box, Point
import numpy as np, trimesh

T = 3.0                        # panel thickness
TW, TD = 130.0, 155.0          # floor/lid tile (3 x 2 per bay)
W, D = 6 * TW, 2 * TD          # outer 780 x 310
H = 90.0                       # wall height
SIDE_W = (D - 2 * T) / 2       # side/divider tile width


def grille(w, h):
    p = box(0, 0, w, h)
    for k in range(5):
        x = 12 + k * 22.0
        p = p.difference(box(x, 12, x + 12, h - 12))
    return p


def vent_wall(port=None):
    p = box(0, 0, TW, H)
    for k in range(6):
        y = 12 + k * 12.5
        p = p.difference(box(12, y, TW - 12, y + 6))
    if port is not None:                                   # separate power inlet: round 14 mm port low in the outer corner
        p = p.difference(Point(port, 20).buffer(7.0, 32))
    return p


panels = {  # name: (polygon, count)
    "floor_tile": (grille(TW, TD), 12),
    "lid_tile": (grille(TW, TD), 12),
    "front_tile": (vent_wall(), 6),
    "rear_tile": (vent_wall(), 4),
    "rear_tile_port_left": (vent_wall(port=13.0), 1),
    "rear_tile_port_right": (vent_wall(port=TW - 13.0), 1),
    "side_tile": (box(0, 0, SIDE_W, H), 4),
    "divider_tile": (box(0, 0, SIDE_W, H), 2),
}
os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
for n, (poly, cnt) in panels.items():
    m = extrude(poly, 0, T)
    assert m.is_watertight and max(m.extents) <= 180, n
    m.export(os.path.join(ROOT, "print", f"desktop_dual_board_case_{n}.stl"))
    print(n, "x", cnt, np.round(m.extents, 1).tolist(), f"{m.volume/1000*1.27:.1f} g")


def place(poly, M):
    m = extrude(poly, 0, T); m.apply_transform(M); return m


tm = trimesh.transformations
parts = []
for i in range(6):
    for j in range(2):
        for z, poly in ((0.0, panels["floor_tile"][0]), (H + T, panels["lid_tile"][0])):
            parts.append(place(poly, tm.translation_matrix([i * TW, j * TD, z])))
rear = ["rear_tile_port_left"] + ["rear_tile"] * 4 + ["rear_tile_port_right"]
for i in range(6):
    for yy, key in ((0.0, "front_tile"), (D - T, rear[i])):
        M = tm.translation_matrix([i * TW, yy + T, T]) @ tm.rotation_matrix(np.pi / 2, [1, 0, 0])
        parts.append(place(panels[key][0], M))
for x in (0.0, W - T, W / 2 - T / 2):
    for j in range(2):
        poly = panels["side_tile" if x != W / 2 - T / 2 else "divider_tile"][0]
        M = tm.translation_matrix([x, T + j * SIDE_W, T]) @ tm.rotation_matrix(np.pi / 2, [0, 0, 1]) @ tm.rotation_matrix(np.pi / 2, [1, 0, 0])
        parts.append(place(poly, M))
asm = trimesh.util.concatenate(parts)
d = os.path.join(ROOT, "parts", "desktop_dual_board_case"); asm.export(os.path.join(d, "part.stl"))
print("assembled", np.round(asm.extents, 1).tolist())
sys.path.insert(0, ROOT)
from topo.build import render
render(asm, os.path.join(d, "preview.png"), title="desktop_dual_board_case (2 Kootek-pad bays, 42 flat panels)", info=f"{W:.0f} x {D:.0f} x {H+2*T:.0f} mm | pad = main cooling | slotted rear exhaust | 2 power ports")
