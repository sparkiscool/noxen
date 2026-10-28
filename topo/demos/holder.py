"""The old engine's benchmark holder (noxopt_bench.holder + noxopt_demo.py bodies), same features and load set:
60 x 40 x 30 mm, 4 bolt bosses (bottom corners), pack pad (top), 2 hook roots (sides), camera pad (front), battery
channel keep-out; supports: bolts / grab (hooks) / pad; 28 cases (14 misuse directions at 25 g with the pack, flight,
landing, hand and retention loads, catcher grab, pad push). Units mm, N.

Changes for the solid gate (23 Oct, organic test part):
  * bolt bosses: 6 x 6 mm squares round the 3.2 mm holes left 1.4 mm walls, which the gate's 0.9 mm erosion cut off
    (topo-v1: 4 boss cores of 3.4 mm3 split off, FAIL; the self-test smoke run never joined one corner boss). Now
    11.5 mm round bosses, 6 mm tall, centred 0.75 mm toward the part on x and y from the bolt and cut flat by the
    envelope's outer faces (the corner bolts sit 5 mm from both edges): 4.7 mm wall toward the part, 3.4 mm to the
    envelope faces, so the boss keeps one core at the gate's 2.0 mm erosion and it sits where the legs fuse on.
    10 mm round (3.4 mm walls, organic-v1 to v4) left single-voxel cores there (v4: share 0.996 to 0.999, FAIL);
    12 mm round on the bolt (organic-v5, v6) left a 64-voxel core lobe toward the envelope corner (v6: share 0.96).
  * keep-out under the pack round the pad: starts at z 29 (was 26), 1 mm clearance under the pack instead of 4 (DEFAULT).
    The pad sat on a 4 mm layer between that keep-out and the battery channel (top z 22), so every member out of the
    pad was a plate at most 4 mm thick (3 mm as built: the build holds 0.5 mm off each keep-out), which the gate's
    2.0 mm erosion removes (organic-v1: worst index 0.65 in that layer, gate FAIL). z 27 (organic-v2, v3) built a
    4 mm plate, still cut there: the eroded part split at the pad and every other body read as loose. z 28 (6 mm,
    organic-v4, v5) builds 5 mm, but the optimiser's own eroded design needs plates of 5.7 mm at 6.25 mm members, so
    every case through the pad read 1e4 to 2e5 x its solid compliance and v5 lost the link to the right bosses (2
    bodies). A 7 mm layer clears both (6 mm built, 2 mm core at the gate's erosion).
  * battery channel: its two top long edges rounded r 3 mm (about a pack's own edge radius), approved by Spark
    23 Oct. In organic-v1 to v10 the worst index sat where each hook strut bends 90 deg over that sharp edge
    (28.5, 7.5 / 32.5, 22.5); v7 11.73 g / 0.569 and v10 12.50 g / 0.504 each missed one target.
  Load set, supports (the 6 x 6 seats) and envelope unchanged."""
import numpy as np
import manifold3d as mf
from ..part import Part

BOLTS = [(x, y) for x in (5, 55) for y in (5, 35)]
M_PACK = 0.165                                   # kg on the pad
BOSS_R = 5.75                                    # bolt boss radius, centre BOSS_OFF toward the part on x and y:
BOSS_OFF = 0.75                                  # 4.7 mm wall toward the part, 3.4 mm to the envelope faces
PACK_CLEAR_Z = 29.0                              # keep-out under the pack round the pad starts here (was 26)
CHANNEL_R = 3.0                                  # battery channel top long edges rounded (was sharp)


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl(x, y, z0, z1, r, n=48):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def channel():
    """battery channel keep-out x 10..50, y 8..32, z 6..22, its two top long edges rounded r CHANNEL_R"""
    r = CHANNEL_R
    ch = box(10, 50, 8, 32, 6, 22 - r) + box(10, 50, 8 + r, 32 - r, 6, 22)
    for y in (8 + r, 32 - r):
        ch += mf.Manifold.cylinder(40, r, r, 48).rotate([0, 90, 0]).translate([10, y, 22 - r])
    return ch


def bodies():
    pad = box(22, 38, 14, 26, 26, 30)
    hookL = box(26, 34, 0, 3, 14, 24)
    hookR = box(26, 34, 37, 40, 14, 24)
    cam = box(56, 60, 15, 25, 8, 18)
    bosses = [cyl(x + BOSS_OFF * np.sign(30 - x), y + BOSS_OFF * np.sign(20 - y), 0, 6, BOSS_R) ^ box(0, 60, 0, 40, 0, 30)
              for x, y in BOLTS]
    drawn = pad + hookL + hookR + cam
    for b in bosses:
        drawn += b
    cuts = channel() + (box(8, 52, 6, 34, PACK_CLEAR_Z, 30.5) - box(22, 38, 14, 26, 26, 30.5))
    for x, y in BOLTS:
        cuts += cyl(x, y, -1, 31, 1.6) + cyl(x, y, 6, 31, 3.0)
    env = box(0, 60, 0, 40, 0, 30)
    return dict(pad=pad, hookL=hookL, hookR=hookR, cam=cam, bosses=bosses, drawn=drawn, cuts=cuts, env=env)


def setup(h):
    B = bodies()
    p = Part("holder", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG")
    seats = None
    for x, y in BOLTS:
        s = box(x - 3, x + 3, y - 3, y + 3, -1.0, 1.5 * h)
        seats = s if seats is None else seats + s
    p.support("bolts", seats)
    p.support("grab", B["hookL"] + B["hookR"])
    p.support("pad", B["pad"])
    g = 9.81
    dirs = [np.array(v, float) for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    dirs += [np.array((sx, sy, sz), float) / np.sqrt(3) for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    for k, d in enumerate(dirs):
        p.load(f"misuse {k}", B["pad"], M_PACK * g * 25 * d, cls="misuse", support="bolts", accel=25 * d)
    p.load("flight", B["pad"], M_PACK * g * np.array([0, 2.0, -4.0]), cls="flight", support="bolts",
           accel=np.array([0, 2.0, -4.0]))
    p.load("landing 6g", B["pad"], M_PACK * g * np.array([0, 0, -6.0]), cls="misuse", support="bolts",
           accel=np.array([0, 0, -6.0]))
    for nm, m, v in [("pinch L", "hookL", (0, 20, 0)), ("pinch R", "hookR", (0, -20, 0)),
                     ("retain L", "hookL", (0, 0, 20)), ("retain R", "hookR", (0, 0, 20)),
                     ("poke cam x", "cam", (-5, 0, 0)), ("poke cam z", "cam", (0, 0, -5)),
                     ("poke cam y", "cam", (0, 5, 0)), ("preload", "pad", (0, 0, 12))]:
        p.load(nm, B[m], np.array(v, float), cls="misuse", support="bolts")
    for nm, v in [("grab lift", (0, 0, -M_PACK * g * 6)), ("grab side", (0, M_PACK * g * 3, 0))]:
        v = np.array(v, float)
        p.load(nm, B["pad"], v, cls="misuse", support="grab", accel=v / (M_PACK * g))
    for nm, v in [("pad push", (0, 0, -15)), ("pad shear", (10, 0, 0))]:
        p.load(nm, B["bosses"][0] + B["bosses"][3], np.array(v, float), cls="misuse", support="pad")
    return p
