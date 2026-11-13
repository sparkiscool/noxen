"""Swap arm v8: geometry and kinematics shared by parts/box_arm8/make.py and the three part.py setups (hub, carriage, lift).

Arm frame = the box W frame with head A at the drone slot: x across (tangential), y along the axle (W Y), z = u radial (W Z).
Head B is head A turned 180 deg about the axle (v7 had them 90 apart: see make.py). Everything is drawn at park (s 0).
A head has two stages on one lead screw: the LIFT (block on two o 6 steel rods, LM6UU) carries the boom and the grab unit and
is what the screw's nut drives, travel s 0..117.8; the CARRIAGE (three V wheels on the v7 rail) carries the rods and is picked up
by the lift for the last part: carriage travel = max(0, s - T2). A return spring holds the carriage down against the lift's stop.
Tab frame (the latch's cartridge coordinates about the tab line): x across, y up into the cartridge (0 = its bottom), z along the pack.
"""
import importlib.util, math, os
import numpy as np
import manifold3d as mf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


_m7 = None
def latch():
    global _m7
    if _m7 is None: _m7 = _load("latch7_make", "parts/latch7/make.py")
    return _m7


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl(axis, a, b, r, t0, t1, n=40):
    """axis z: centre (x a, y b); axis x: (y a, z b); axis y: (x a, z b)"""
    c = mf.Manifold.cylinder(t1 - t0, r, r, n)
    if axis == "z": return c.translate([a, b, t0])
    if axis == "x": return c.rotate([0, 90, 0]).translate([t0, a, b])
    return c.rotate([-90, 0, 0]).translate([a, t0, b])
def hullp(pts): return mf.Manifold.hull_points(np.asarray(pts, np.float64))
def U(*ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else (ms[0] if ms else mf.Manifold())
def vol(m): return 0.0 if m.is_empty() else m.volume()
def rot180(m): return m.rotate([0, 180, 0])                     # head B from head A


# ------------------------------------------------------------------ the layout (C unless tagged)
Y_C = 167.1                    # grip line = each cartridge's tab line = the funnel centre (box frame)
R_BAY = 129.0                  # bay receivers' bottom radius. v7's 113 does not take latch v7's receivers: neighbours at 30 deg overlap 623 mm3 (make.py ring())
R_DRONE = 176.0                # the seated drone's receiver bottom (seat z -44.3 at W Z 220.3)
CART_DROP = 0.2                # a seated cartridge's bottom is 0.2 under its receiver's
G_PARK = 60.0                  # pad plane (= a carried cartridge's bottom) at park: a carried Overseer's top at 118.6, 8.4 under the bay packs
S_BAY = R_BAY - CART_DROP - G_PARK        # 66.8
S_DRONE = R_DRONE - CART_DROP - G_PARK    # 115.8
S_OVER = 1.5                   # commanded past the seat; the unit's float springs take it
T2 = 60.0                      # the lift's travel on the carriage
Z0 = 32.0                      # carriage plate's inner end at park
CAR_L = 70.0                   # carriage plate length
PLATE_Y = (68.75, 73.8)        # carriage plate: 1.0 under the bay receivers (74.8)
WHEELS = [(22.3, 4.0), (22.3, 48.0), (-22.3, 26.0)]     # x, carriage coordinate; the third on the eccentric bush. Top rim at the drone 153.8: under the funnel's tail beam (155)
RAIL = (27.5, 154.0)
RODS_X, ROD_Y = (44.0, 69.0, 94.0), 64.9                     # the lift's three o 6 rods (two leave the LM6UU at 0.63 of their static rating in a plain insert), side by side beside the rail: over the funnel's tail beam (y 58.1), under the bays' closed ends (74.8)
BLOCK = dict(x=(36.5, 101.5), y=(56.0, 73.8), u=(-14.0, 36.0))      # the lift block at park: 50 long, two LM6UU on each rod (29 between the rows), the nut between the rods
SCREW = (56.5, 47.5)           # screw axis (x, y): under the rods, in a slot in the hub's plate (between the rods it cut both rod clamps off from the wheels); T8 x 2, the catch's screw
MOTOR = dict(x=(35.35, 77.65), y=(26.35, 68.65), u=(-92.5, -44.5))    # the head stepper lying through the plate's plane; far corner r 120.7: inside funnel v5's tail arms (r 123.6 at y 20..36)
NUT_U = (6.0, 21.0)            # the nut in a lug under the block (through the plate's slot); flange 3.5 over it, filed to 15 x 18
LUG = dict(x=(48.3, 64.7), y=(39.5, 58.0), u=(3.0, 27.5))
NECK = dict(x=(29.3, 36.6), y=(68.5, 73.8), u=(-14.0, 11.0))        # block to the wall, between the wheel rims and the bays' closed ends
WALL = dict(x=(29.3, 45.0), y=(68.5, 75.0), u=(-14.0, 11.0))        # up past the carriage plate's edge (x 28) and over the block
WALL2 = dict(x=(29.3, 70.0), y=(79.1, 96.0), u=(-1.0, 11.0))         # past y 79.5 the other head's cross-over is 2 below
RISER = dict(x=(29.3, 70.0), y=(73.7, 79.2), u=(-14.0, 11.0))
DEEP = dict(x=(34.6, 70.0), y=(74.0, 96.0), u=(-27.0, 11.0))            # beyond the other head's reach (x 33.5) the riser and cross-over run 27 below the axle line: the joint's section         # up over the block's whole inner width: the section the 6.6 N m lever needs (it read 3.1 through a 16 x 12 wall)
XARM = dict(x=(-33.5, 70.0), y=(79.2, 96.0), u=(3.0, 31.0))         # 29 x 28: the far boom's moment crosses it in torsion          # the cross-over, above the carriage plate (73.8)
BOOMS = [dict(x=(sx * 26.3, sx * 33.5) if sx > 0 else (sx * 33.5, sx * 26.3), y=(80.0, Y_C + 27.0), u=(3.0, 31.0)) for sx in (-1, 1)]   # one each side of the grab unit; plain PCTG: the head's pitch give (about 3 N m / deg, E) is wanted
TIE_U = (1.0, 9.6)             # two ties across the booms under the grab unit: the float bolts pass up through them, heads in their undersides
FLOAT_Z = (-45.0, 20.0)        # the float bolts' stations along the pack, from the tab line
FLOAT_X = 15.0
PLATE = (50.0, 54.0, 88.0)     # hub plate y0, y1, radius
F_INSERT = dict(scout=57.0 + 2.2, overseer=57.0 + 7.5)             # plug (E) + pack weight
F_NEED = 1.5 * F_INSERT["overseer"]                                 # 96.8 N: what the latch asks of the head
F_LIMIT = 118.0                # the float springs solid-ish at S_OVER + 1: the most the head can put on a cartridge (E)
F_PULL = 57.0 + 7.5
M_GUIDED = 2.0                 # N m: v7's frame model: the arm's pitch stiffness at the pack 4.51 N m / deg x the guides' 0.38 (Overseer) .. 0.47 deg (Scout) (parts/box_arm/ref/arm_v7_system.json xt60_mate, xt60_guided); v8's own is not modelled
AXLE_STALL_NM = 0.59 * 150 / 16     # the axle stepper's stall through the 150T ring and its 16T (v7: 2.95 at 5:1)
M_FLOAT = 3.0                  # N m: the float's pitch rate (4 x 5.76 N/mm x 32.5^2 = 24 N m / rad) over its 5.3 deg reach + the preload's 0.7


def arm_T(g=G_PARK):
    """tab frame -> arm frame with the pad plane at radius g"""
    return np.array([[-1.0, 0, 0, 0], [0, 0, 1.0, Y_C], [0, 1.0, 0, g]])


def carriage_s(s): return max(0.0, s - T2)


def Ry(phi_deg):
    """arm frame -> W with head A's bar at phi (90 = up, the drone slot)"""
    t = math.radians(90.0 - phi_deg); c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])


def lift_T(s):
    T = np.eye(4); T[2, 3] = s; return T


# ------------------------------------------------------------------ hub side (head A; rot180 for B)
def rail():
    u0, u1 = RAIL
    prof = [(-9, 54), (9, 54), (9, 59.75), (12, 62.75), (9, 65.75), (9, 66.75), (-9, 66.75), (-9, 65.75), (-12, 62.75), (-9, 59.75)]
    body = U(box(-9, 9, 54, 66.75, u0, u1), hullp([[9, 59.75, z] for z in (u0, u1)] + [[11.5, yy, z] for z in (u0 + 2.5, u1 - 2.5) for yy in (62.25, 63.25)] + [[9, 65.75, z] for z in (u0, u1)]),
             hullp([[-9, 59.75, z] for z in (u0, u1)] + [[-11.5, yy, z] for z in (u0 + 2.5, u1 - 2.5) for yy in (62.25, 63.25)] + [[-9, 65.75, z] for z in (u0, u1)]),
             box(-9, 9, 38, 54.01, PLATE[2] + 1.0, u1),                                       # the keel beyond the plate
             )                                                                                # no park stop on the rail: the screw sets park, and the carriage's tail plate passes over the rail's end
    for u in RAIL_BOLTS:
        body = body - cyl("y", 0, u, 2.75, 50, 70) - cyl("y", 0, u, 4.6, 61.0, 70)            # M5 x 10 socket head into the hub's insert
    return body - box(-2.5, 2.5, 62.75, 67, u0 + 3.5, u1 + 1)                                 # the lift's return spring lies in the channel
RAIL_BOLTS = (40.0, 78.0)


def screw_drive():
    """bought: stepper, rigid coupler, T8 x 2 screw, and two F8-16M thrust bearings either side of a wall on the plate, a set collar outside them:
    push and pull both go into the hub, not the stepper"""
    x, y = SCREW; m = MOTOR
    ch = 4.0                                                                    # the 17HS19's body corners are chamfered (about 4.5; drawn 4.0)
    body = hullp([[xx, yy, uu] for uu in m["u"] for (xx, yy) in ((m["x"][0] + ch, m["y"][0]), (m["x"][1] - ch, m["y"][0]), (m["x"][1], m["y"][0] + ch), (m["x"][1], m["y"][1] - ch),
                                                                  (m["x"][1] - ch, m["y"][1]), (m["x"][0] + ch, m["y"][1]), (m["x"][0], m["y"][1] - ch), (m["x"][0], m["y"][0] + ch))])
    return dict(motor=U(body, cyl("z", x, y, 11.0, m["u"][1], m["u"][1] + 2.0), cyl("z", x, y, 2.5, m["u"][1], m["u"][1] + 24.0)),
                coupler=cyl("z", x, y, 7.0, -41.3, -16.7),
                thrust=U(cyl("z", x, y, 8.0, -16.5, -11.5), cyl("z", x, y, 8.0, -7.5, -2.5), cyl("z", x, y, 6.5, -2.5, 2.5)) - cyl("z", x, y, 4.0, -18, 3),
                screw=cyl("z", x, y, 4.0, -30.0, 141.0))
MOTOR_WALL = dict(x=(35.35, 77.65), y=(26.0, 69.0), u=(-44.5, -41.5))    # the stepper's face plate (31 mm pattern), through the plate's plane
THRUST_WALL = dict(x=(46.5, 66.5), y=(36.0, 55.6), u=(-11.5, -7.5))      # the screw through it between the two F8-16M; a 5 mm lock ring outside the outer one, the coupler inside the inner


# ------------------------------------------------------------------ carriage (park)
def wheel(x, c):
    """printed V wheel on two 625ZZ: o 24 x 10, 90 deg groove, apex r 10.25 (v7's wheel v1)"""
    cone = lambda t0, r0, r1: mf.Manifold.cylinder(1.75, r0, r1, 48).rotate([-90, 0, 0]).translate([x, t0, Z0 + c])
    w = U(cyl("y", x, Z0 + c, 12.0, 57.75, 61.01, 48), cone(61.0, 12.0, 10.25), cone(62.75, 10.25, 12.0), cyl("y", x, Z0 + c, 12.0, 64.49, 67.75, 48))
    return w - cyl("y", x, Z0 + c, 8.0, 57, 62.75) - cyl("y", x, Z0 + c, 2.6, 57, 68)        # two 625ZZ (o 16 x 5) from one side, a 5 mm web behind them
C_BOT = -52.0                  # the rods' lower end, carriage coordinate
def carriage_bodies():
    """the drawn attachments: three wheel bosses, then a collar round each rod's end (top three, bottom three)"""
    b = [cyl("y", x, Z0 + c, 6.8, 67.75 + 1.0, 78.6) for x, c in WHEELS]                 # M5 axle bosses, the plate's full 9.85
    b += [box(x - 6.0, x + 6.0, 58.4, 73.8, Z0 + CAR_L - 6.0, Z0 + CAR_L) for x in RODS_X]
    b += [box(x - 6.0, x + 6.0, 58.4, 73.8, Z0 + C_BOT, Z0 + C_BOT + 6.0) for x in RODS_X]
    return b
def carriage_env():
    top = Z0 + CAR_L
    return U(box(-28.0, 28.0, PLATE_Y[0], 78.6, Z0, top), box(-28.0, 116.0, PLATE_Y[0], PLATE_Y[1], top - 6.0, top),   # the wheel plate 9.85 thick: the lift's cross-over passes at 79.2
             box(28.2, 116.0, 58.4, 73.8, top - 6.0, top),
             box(0.0, 36.3, PLATE_Y[0], 78.6, top - 30.0, top), box(36.3, 62.0, 74.3, 78.6, top - 30.0, top), box(28.0, 62.0, 74.3, 88.0, top - 10.0, top), box(28.0, 62.0, 73.7, 88.0, top - 6.0, top),
             box(50.0, 108.0, 58.4, 73.8, top - 6.0, top + 8.0),                                           # beyond x 50 the funnel (v4) leaves room over the clamps
             box(102.2, 116.0, 58.4, 73.8, Z0 + C_BOT - 20.0, top),                                         # the outer spine: 14 wide, the lower clamp's way to the wheels
             box(36.5, 116.0, 58.4, 73.8, Z0 + C_BOT - 20.0, Z0 + C_BOT + 6.0),                            # under and round the lower collars, down to the stepper's wall
             box(13.0, 36.6, 54.6, 68.3, Z0 + C_BOT - 6.0, Z0 + C_BOT + 6.0),                               # the lower clamp's short way to the wheels: inward under the lift's neck,
             box(13.0, 28.0, 54.6, 78.6, Z0 + C_BOT - 6.0, Z0 + C_BOT + 6.0), box(2.0, 28.0, PLATE_Y[0], 78.6, Z0 + C_BOT - 6.0, Z0 + 0.01))   # up, and along a tail plate (this head's half of the centre) to the wheel plate
def carriage_cuts():
    c = [cyl("y", x, Z0 + cc, 2.6, 50, 80) for x, cc in WHEELS] + [cyl("z", x, ROD_Y, 3.05, Z0 + C_BOT - 5, Z0 + CAR_L + 0.01) for x in RODS_X]
    return U(*c)
def rods():
    return U(*[cyl("z", x, ROD_Y, 3.0, Z0 + C_BOT, Z0 + CAR_L) for x in RODS_X])


# ------------------------------------------------------------------ lift (park)
def lm_seats():
    """six LM6UU (o 12 x 19) in the block: two on each rod, 29 between the rows"""
    u0 = BLOCK["u"][0]
    return [(x, u0 + a, u0 + a + 19.0) for x in RODS_X for a in (1.5, 29.5)]
def _b(d): return box(d["x"][0], d["x"][1], d["y"][0], d["y"][1], d["u"][0], d["u"][1])
def lift_block(): return U(_b(BLOCK), _b(LUG))
def float_pts():
    return [(sx * FLOAT_X, Y_C + z) for sx in (-1, 1) for z in FLOAT_Z]
def ties():
    return [box(-33.5, 33.5, Y_C + z - 7.0, Y_C + z + 7.0, TIE_U[0], TIE_U[1]) for z in FLOAT_Z]
def lift_cuts():
    b = BLOCK; x, y = SCREW
    c = [cyl("z", xx, ROD_Y, 6.05, a, e) for xx, a, e in lm_seats()] + [cyl("z", xx, ROD_Y, 3.6, b["u"][0] - 1, b["u"][1] + 1) for xx in RODS_X]
    c += [cyl("z", x, y, 5.2, LUG["u"][0] - 1, LUG["u"][1] + 1), box(x - 9.2, x + 9.2, LUG["y"][0] - 1, y + 7.6, NUT_U[1], NUT_U[1] + 3.7)]   # the nut's body and its flange slot, in the lug
    c += [U(cyl("z", px, py, 5.0, TIE_U[0] - 1, 30), cyl("z", px, py, 7.6, TIE_U[0] - 1, TIE_U[0] + 3.0)) for px, py in float_pts()]         # o 10 round each o 5 shoulder (+-2.5 of float), head and washer 3 into the underside
    return U(*c)
def deep_wedge():
    """DEEP, tapered: its depth grows from the cross-over's own at the boom side to the full 27 below the axle line by x 58, and runs out toward the booms along y"""
    d = DEEP; pts = [[d["x"][0], y, u] for y in (d["y"][0], XARM["y"][1]) for u in (-1.0, d["u"][1])]
    pts += [[x, y, u] for x in (58.0, d["x"][1]) for y in (d["y"][0], 92.0) for u in d["u"]] + [[x, XARM["y"][1], u] for x in (58.0, d["x"][1]) for u in (3.0, d["u"][1])]
    return hullp(pts)
def lift_env():
    return U(lift_block(), _b(NECK), _b(WALL), _b(RISER), deep_wedge(), _b(WALL2), _b(XARM), box(-33.5, 33.5, XARM["y"][1] - 0.01, Y_C + FLOAT_Z[0] - 6.9, XARM["u"][0], XARM["u"][1]),
             hullp([[x, y, u] for (x, y) in ((33.4, XARM["y"][1] - 0.01), (XARM["x"][1], XARM["y"][1] - 0.01), (33.4, Y_C + FLOAT_Z[0] - 6.9)) for u in XARM["u"]]), *[_b(o) for o in BOOMS], *ties())   # no step from the cross-over to the boom plate: a taper
def lift_bought():
    x, y = SCREW
    return dict(lm6uu=U(*[cyl("z", xx, ROD_Y, 6.0, a, e) - cyl("z", xx, ROD_Y, 3.1, a - 1, e + 1) for xx, a, e in lm_seats()]),
                nut=(U(cyl("z", x, y, 5.1, NUT_U[0], NUT_U[1]), cyl("z", x, y, 11.0, NUT_U[1], NUT_U[1] + 3.5) ^ box(x - 9.0, x + 9.0, y - 7.5, y + 7.5, 0, 60)) - cyl("z", x, y, 4.0, NUT_U[0] - 1, NUT_U[1] + 5)))


# ------------------------------------------------------------------ grab unit (tab frame; y 0 = the cartridge's bottom)
SEAT_X = 29.4                  # blade springs' seats
UNIT_BASE = (-38.5, -36.5)     # base plate (y), between the booms
CAM_Y = -24.3                  # servo and cam axis: the servo's top 14.3 under the pad plane (the funnel's ties stand at |x| 26..46 from 13.8 under it)
ROD_YT = -19.0                 # blade rods
PILOT = dict(x=(10.6, 16.7), z=(17.4, 22.3), top=5.0, lead=5.0, taper=3.0)    # two sprung pilots in the free corners of the recess
K_BLADE = 2 * 1.59             # N/mm per blade: two o 6 x 0.7 x 25 springs (E from the wire and coil count)
K_FLOAT, P_FLOAT = 4 * 5.76, 4 * 18.0     # four o 8 x 1.0 x 15 springs (5 active coils): N/mm, preload N (E)


def tabz(m, g): return m.translate([0, 0, -g["Z_T"]])
def unit_deck():
    """the latch's heel for both packs on one head: decks, hall posts, pads (TPU)"""
    m7 = latch(); decks, pads = [], []
    for pk in ("scout", "overseer"):
        g = packs()[pk]["g"]; m7.set_pack(g); d, p = m7.heel(g)
        decks.append(tabz(d, g)); pads += [tabz(q, g) for q in p]
    return U(*decks), pads
_P = {}
def packs():
    if not _P:
        m7 = latch()
        for pk in ("scout", "overseer"): _P[pk] = m7.build(pk)
    return _P
def blade(side, face):
    m7 = latch(); g = packs()["scout"]["g"]
    b = tabz(m7.blade(g, face, 1), g)
    b = U(b, *[box(face, face + 7.5, ROD_YT - 4.0, ROD_YT + 4.0, sz * 8.0 if sz > 0 else -16.5, sz * 16.5 if sz > 0 else -8.0) for sz in (-1, 1)],
          box(face, face + 4.7, -31.0, -19.9, -7.0, 7.0))                                                # the tail the cam bears on
    b = b - U(*[cyl("x", ROD_YT, z, 1.65, face - 1, face + 9) for z in (-12.0, 12.0)])
    return b if side > 0 else b.mirror([1, 0, 0])
def faces():
    m7 = latch(); return dict(open=m7.m6.BLADE_OPEN, grip=m7.bl.TAB_OUT - m7.m6.R0, limit=m7.bl.TAB_OUT - m7.bl.TRAVEL)
def pilot(sx, sz, ext=0.0):
    p = PILOT; x0, x1 = p["x"]; z0, z1 = p["z"]; t = p["taper"]; top = p["top"] + ext
    tip = hullp([[x, y, z] for (x, z) in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)) for y in (-2.0 + ext, top - t)] + [[x, top, z] for (x, z) in ((x0, z0), (x1 - t, z0), (x1 - t, z1 - t), (x0, z1 - t))])
    stem = U(box(x0, x1, -21.0 + ext, -6.5 + ext, STEM_Z[0], STEM_Z[1]), box(x0, x1, -9.0 + ext, -1.99 + ext, z0, STEM_Z[1]))   # below the deck the stem stands beyond the blade's end
    m = U(tip, stem)
    if sx < 0: m = m.mirror([1, 0, 0])
    if sz < 0: m = m.mirror([0, 0, 1])
    return m
STEM_Z = (19.5, 24.0)
PILOTS = ((1, -1), (-1, 1))    # the hall posts stand in the other two corners
def unit_body():
    """the printed housing: deck (both packs' pad seats, the hall posts), side walls with the blade rods and spring seats over the booms,
    the trough between the booms with the servo and the cam, four float ears, the pilots' guides"""
    deck, pads = unit_deck(); y0, y1 = UNIT_BASE
    walls = [U(box(*sorted((sx * SEAT_X, sx * 33.5)), -26.0, -14.5, -28.0, 27.0), box(*sorted((sx * 23.0, sx * 33.5)), -15.9, -13.9, -28.0, 27.0),
               box(*sorted((sx * 23.0, sx * 25.7)), -14.0, -5.9, -20.0, 20.0), box(*sorted((sx * 23.0, sx * 25.8)), y1, -24.0, -28.0, 27.0),
               *[box(*sorted((sx * 23.0, sx * 33.5)), -26.0, -14.5, z0, z1) for z0, z1 in ((-28.0, -25.0), (23.0, 27.0))]) for sx in (-1, 1)]
    base = box(-25.8, 25.8, y0, y1, -47.0, 71.5)
    cross = [U(box(-25.8, 25.8, y1 - 0.01, -14.5, z0, z1), box(-25.7, 25.7, -14.6, -5.9, z0, z1)) for z0, z1 in ((-28.0, -25.0), (23.0, 27.0), (68.5, 71.5))]
    blocks = [box(-25.8, 25.8, y1 - 0.01, -19.0, -47.0, -28.0), box(-25.8, -23.0, y1 - 0.01, -14.5, 20.0, 71.5)]
    guides = []
    for sx, sz in PILOTS:
        gd = box(8.2, 19.1, -22.0, -5.9, 17.3, 27.9)
        if sz < 0: gd = gd.mirror([0, 0, 1])
        if sx < 0: gd = gd.mirror([1, 0, 0])
        guides.append(gd)
    bbs = [q.bounding_box() for q in pads]
    jacks = [box(b[0] - 1, b[3] + 1, -10.0, -5.9, b[2] - 1, b[5] + 1) for b in bbs]                        # a boss under each pad for its M3 jack screw
    m = U(deck, *walls, base, *cross, *blocks, *guides, *jacks, *[cyl("y", sx * FLOAT_X, z, 5.0, y1 - 0.01, y1 + 4.5) for sx in (-1, 1) for z in FLOAT_Z])
    cut = [cyl("x", ROD_YT, z, 1.6, -40, 40) for z in (-12.0, 12.0)]                                       # o 3 carbon rods (the blades slide on them)
    cut += [cyl("z", 0.0, CAM_Y, 6.5, 22.0, 28.0)]                                                         # the servo's boss through its wall
    cut += [U(box(*sorted((sx * 10.2, sx * 17.1)), -40, -5.0, *sorted((sz * (STEM_Z[0] - 0.4), sz * (STEM_Z[1] + 0.4)))), box(*sorted((sx * 10.2, sx * 17.1)), -9.5, 0.0, *sorted((sz * 17.0, sz * (STEM_Z[1] + 0.4))))) for sx, sz in PILOTS]
    cut += [cyl("y", sx * FLOAT_X, z, 2.8, y0 - 1, y1 + 4.6) for sx in (-1, 1) for z in FLOAT_Z]              # M4 heat-set inserts for the float bolts, from below
    cut += [box(*sorted((sx * 11.3, sx * 14.7)), 3.8, 5.7, *sorted((sz * 18.7, sz * 21.5))) for sx, sz in ((1, 1), (-1, -1))]   # a pocket in each post for its hall sensor
    cut += [cyl("y", (b[0] + b[3]) / 2, (b[2] + b[5]) / 2, 1.25, -45, -5.5) for b in bbs]                  # M3 jack screws, tapped in the print
    cut += [box(-10.2, 30.2, CAM_Y - 10.2, CAM_Y + 10.2, 27.0, 68.5)]                                      # the servo
    cut += [box(-13.0, 13.0, y1, -5.0, -8.0, 8.0)]                                                         # the cam's sweep
    return m - U(*cut), pads
def unit_bought(state="grip"):
    f = faces()[state]
    paddle = box(-4.0, 4.0, CAM_Y - 11.25, CAM_Y + 11.25, -6.0, 6.0) if state != "open" else box(-11.25, 11.25, CAM_Y - 4.0, CAM_Y + 4.0, -6.0, 6.0)
    fb = [(sx * FLOAT_X, z) for sx in (-1, 1) for z in FLOAT_Z]
    out = dict(servo=U(box(-10.0, 30.0, CAM_Y - 10.0, CAM_Y + 10.0, 28.0, 68.4), cyl("z", 0.0, CAM_Y, 6.0, 22.0, 28.0), cyl("z", 0.0, CAM_Y, 3.5, 6.0, 22.0), cyl("z", 0.0, CAM_Y, 7.5, 18.5, 22.0)),
               paddle=U(paddle, cyl("z", 0.0, CAM_Y, 4.0, -6.0, 6.0)),
               blade_rods=U(*[cyl("x", ROD_YT, z, 1.5, -33.0, 33.0) for z in (-12.0, 12.0)]),
               blade_springs=U(*[cyl("x", ROD_YT, z, 3.0, *sorted((sx * (f + 7.5), sx * SEAT_X))) - cyl("x", ROD_YT, z, 2.3, -40, 40) for sx in (-1, 1) for z in (-12.0, 12.0)]),
               halls=U(*[box(*sorted((sx * 11.5, sx * 14.5)), 4.0, 5.5, *sorted((sz * 18.9, sz * 21.3))) for sx, sz in ((1, 1), (-1, -1))]),
               float_bolts=U(*[U(cyl("y", x, z, 2.5, -54.0, -33.0), cyl("y", x, z, 7.5, -55.2, -54.0), cyl("y", x, z, 4.2, -57.0, -55.2)) for x, z in fb]),
               float_springs=U(*[cyl("y", x, z, 4.0, -50.4, -38.5) - cyl("y", x, z, 3.0, -51, -38) for x, z in fb]))
    return out


# ------------------------------------------------------------------ statics: what the lift, the carriage and the hub see (N, mm; arm frame)
def lift_reactions(F, P, s):
    """a force F at point P on the lift (arm frame, lift at park) -> forces at the four LM seats (no force along u: the nut takes it)
    and at the nut. Least squares over the four seats' x and y forces."""
    seats = [np.array([x, ROD_Y, (a + e) / 2]) for x, a, e in lm_seats()]
    nut = np.array([SCREW[0], SCREW[1], sum(NUT_U) / 2])
    F = np.asarray(F, float); P = np.asarray(P, float)
    n = len(seats); A = np.zeros((6, 2 * n + 1)); b = np.concatenate([F, np.cross(P, F)])
    for i, q in enumerate(seats):
        for j, e in enumerate((np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))):
            A[:3, 2 * i + j] = e; A[3:, 2 * i + j] = np.cross(q, e)
    e = np.array([0, 0, 1.0]); A[:3, 2 * n] = e; A[3:, 2 * n] = np.cross(nut, e)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return [(seats[i], np.array([x[2 * i], x[2 * i + 1], 0.0])) for i in range(n)], (nut, np.array([0, 0, x[2 * n]]))


def cases(pack="overseer"):
    """the head's load cases at the grab unit (force on the unit from the cartridge, point of action), each its own:
    name, class, F (arm frame), point, s"""
    m = dict(scout=0.222, overseer=0.766)[pack] + 0.25                 # pack + grab unit, kg (unit E)
    pc = np.array([0.0, Y_C, G_PARK]); cg = np.array([0.0, Y_C, G_PARK + 28.0])
    pp = plug_point(pack); w = 9.81 * (m - 0.25); Fi = 57.0 + w
    # guided (in the receiver): the cartridge can pitch only by the receiver's gaps (0.2 to 0.4 deg), so the plug's lever reaches the head as M_GUIDED at most
    off = lambda F_: np.array([0.0, 1000.0 * M_GUIDED / F_, 0.0])
    out = [("insert at the drone, %s, in the receiver's guides: %.1f N + %.1f N m" % (pack, Fi, M_GUIDED), "flight", [0, 0, -Fi], pc + off(Fi), S_DRONE),
           ("pull-out at the drone, %s, in the receiver's guides: %.1f N + %.1f N m" % (pack, Fi, M_GUIDED), "flight", [0, 0, Fi], pc + off(Fi), S_DRONE),
           ("pull-out, %s, the cartridge free of the guides (bound): the plug's 57 N at the lead end" % pack, "misuse", [0, 0, Fi], (pp * 57.0 + pc * w) / Fi, S_DRONE),
           ("insert at the float limit, %s: %.0f N, the float pitched to its reach (%.1f N m)" % (pack, F_LIMIT, M_FLOAT), "misuse", [0, 0, -F_LIMIT], pc + [0, 1000.0 * M_FLOAT / F_LIMIT, 0], S_DRONE),
           ("insert off square, %s: the limit + 43.5 N across at the mouth" % pack, "misuse", [43.5, 0, -F_LIMIT], pc + [0, 0, 45.0], S_DRONE)]
    if pack == "overseer":
        Fax = 1000.0 * AXLE_STALL_NM / (R_BAY - CART_DROP)
        out.append(("axle stall, head held in a bay: %.2f N m at the hub = %.0f N across at the cartridge" % (AXLE_STALL_NM, Fax), "misuse", [Fax, 0, 0], pc + [0, 0, 30.0], S_BAY))
        Ed = (m - 0.25) * 9.81 * 0.050; Fd = min((P_FLOAT ** 2 + 2 * K_FLOAT * 1000 * Ed) ** 0.5, 2 * F_LIMIT)
        out.append(("dropped pack: an Overseer falls 50 onto the head (float springs take %.2f J)" % Ed, "misuse", [0, 0, -Fd], pc, S_DRONE))
        out.append(("box handling 3 g down + 2 g side, carrying (v7's case)", "misuse", list(9.81 * m * np.array([2.0, 0, -3.0])), cg, 0.0))
    for nm, d in (("down the bar", [0, 0, -1]), ("across", [1, 0, 0]), ("along the axle", [0, -1, 0])):
        out.append(("carrying %s at full reach, gravity %s" % (pack, nm), "flight", list(9.81 * m * np.array(d, float)), cg, S_DRONE))
        out.append(("handling 3 g %s" % nm, "misuse", list(3 * 9.81 * m * np.array(d, float)), cg, 0.0))
    return out


def distribute(F, P, pts):
    """a force F at P as forces at pts (same total force and moment, least norm)"""
    F = np.asarray(F, float); P = np.asarray(P, float); n = len(pts)
    A = np.zeros((6, 3 * n)); b = np.concatenate([F, np.cross(P, F)])
    for i, q in enumerate(pts):
        for j in range(3):
            e = np.zeros(3); e[j] = 1.0; A[:3, 3 * i + j] = e; A[3:, 3 * i + j] = np.cross(np.asarray(q, float), e)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return [x[3 * i:3 * i + 3] for i in range(n)]


def plug_point(pack):
    """where the XT60's force acts on the head (arm frame, park): the plug is at the pack's lead end, the far end from the rail"""
    g = packs()[pack]["g"]; return np.array([0.0, Y_C + g["Z_X"] - g["Z_T"], G_PARK])


def wheel_loads(F, P, s):
    """a head load -> the three V wheels (v7's joint model: each wheel takes a force across the rail and one along the axle; the screw takes the bar direction).
    Returns [(across N, axial N)] and v7's two indices: flank Hertz over 1.67 x 55 MPa (E_PCTG for E_PETG), 625ZZ axial over C0 670 N."""
    F = np.asarray(F, float); P = np.asarray(P, float) + [0, 0, s]; z0 = Z0 + carriage_s(s)
    pts = [np.array([x, 62.75, z0 + c]) for x, c in WHEELS]; scr = np.array([SCREW[0], SCREW[1], P[2]])
    A = np.zeros((6, 7)); b = np.concatenate([F, np.cross(P, F)])
    for i, q in enumerate(pts):
        for j, e in enumerate((np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))):
            A[:3, 2 * i + j] = e; A[3:, 2 * i + j] = np.cross(q, e)
    e = np.array([0, 0, 1.0]); A[:3, 6] = e; A[3:, 6] = np.cross(scr, e)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    out = [(float(x[2 * i]), float(x[2 * i + 1])) for i in range(3)]
    E = 1900.0; Es = E / (2 * (1 - 0.4 ** 2)); band = (12.0 - (22.3 - 11.5)) * 2 ** 0.5; Rw = (12.0 + 10.5) / 2 * 2 ** 0.5; pre = 10.0
    hz = 0.0
    for fx, fy in out:
        Fn = (abs(fx) + pre + abs(fy)) / 2 ** 0.5; a_ = (4 * Fn * Rw / (np.pi * band * Es)) ** 0.5; hz = max(hz, 2 * Fn / (np.pi * a_ * band) / (1.67 * 55.0))
    return out, round(float(hz), 3), round(max(abs(fy) for _, fy in out) / 670.0, 3)
