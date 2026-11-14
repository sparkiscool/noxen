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
T2 = 54.1                      # the lift's travel on the carriage: its MGN block from park to the stop over the rail's top (MGN_U[1] - BLOCK_U[1])
Z0 = 32.0                      # carriage plate's inner end at park
CAR_L = 69.5                   # the carriage's top at park: Z0 + CAR_L = 101.5 (158.8 at the drone, 110.3 in a bay: the neighbour bay's receiver is at 110.5 over x 66.6)
PLATE_Y = (68.75, 78.6)        # the wheel plate: 1.0 in front of the wheels, 0.5 behind a carried Overseer's end cap (79.1). It stays inside the bays' radius, so their closed ends (75.8) do not bound it
WHEEL_R = 13.0                 # V wheel rim radius (v7: 12.0): 2.2 of flank on the ridge instead of 1.2
WHEELS = [(22.3, -18.0), (22.3, 46.0), (-22.3, 14.0)]    # x, carriage coordinate; the third on the eccentric bush. 64 between the pair that takes the pitch couple (was 44); the low one's rim 1.0 from the axle line
RAIL = (10.0, 154.0)
# 13 Nov, the lift's guide is one bought MGN12H block on a 100 mm MGN12 rail (uncut) bolted along the carriage's spine, in front of the wheel plate and beside the carried pack (x 38..50),
# under the bays' radius instead of behind their closed ends. The rail takes pitch, roll and yaw (two rods 14 apart could not take yaw); the load's way to the wheels is straight:
# yoke -> block -> rail -> spine beside the wheel pair. v8's first layout went 70 out to three rods and back through a 5 mm plate.
MGN = dict(rail_w=12.0, rail_h=8.0, block_w=27.0, block_l=45.4, H=13.0, H1=3.0, pattern=20.0, pitch=25.0, first=12.5, C0=5880.0, MR=38.2, MP=36.3, MY=36.3)   # HIWIN MGN12H catalogue (E for the listing's own make)
SPINE = dict(x=(36.3, 50.0), y=(55.0, 78.6))                 # the carriage's spine beside the wheel pair, through the wheels' depth to the rail's seat: 23.6 deep
RAIL_X = 44.0                  # the MGN rail's centre line, on the spine
MGN_U = (-7.9, 92.1)           # the rail at park
BLOCK_U = (-7.4, 38.0)         # the MGN block at park: behind the yoke. At the stop its top is at 92.1; the carriage's top (95.1) at the drone 158.3, in a bay 109.8
SCREW = (59.0, 62.5)           # screw axis (x, y): in front of the hub's plate (no slot in it), beside the spine; 104.6 along the axle from the tab line
MOTOR = dict(x=(37.85, 80.15), y=(41.35, 83.65), u=(-92.5, -44.5))    # the head stepper lying through the plate's plane; far corner r 122.4: inside the bay receivers (129)
NUT_U = (6.0, 21.0)            # the nut in a lug on the lift's arm; flange 3.5 over it, its two sides filed off to the body (15 x 10.4): it holds along the axle side only
LUG = dict(x=(51.3, 66.7), y=(55.4, 71.5), u=(3.0, 27.5))
ARM = dict(x=(58.2, 66.0), y=(71.49, 91.7), u=(-5.0, 40.0))           # from the nut's lug past the MGN block to the yoke's plate: 45 deep along the bar (in a bay the neighbour's cartridge is at 110.6 over x 64: 41.8 at park)
YOKE = dict(x=(-26.5, 27.6), y=(82.0, 112.0), u=(2.1, 43.0))         # across the head from the far socket to the MGN block's side, behind the Scout bays beside the drone (their receivers start at y 108.5)
PLATE_L = dict(x=(27.5, 66.0), y=(91.6, 108.0), u=(-43.0, 43.0))      # the yoke's plate on the MGN block's face (four M3), 16.4 thick, and on down past the block
BOOMS = [(0.0, 7.6), (-21.0, 7.6), (21.0, 7.6)]                      # three o 6 carbon rods (x, u) side by side under the grab unit's base, the float bolts between them. They carry the lever in bending:
#                                55 of rod from the socket mouths to the tab line. Nothing of the booms stands over 13.1: beside the drone the neighbour bays' receivers come down to 14.4 (park height) at |x| 36
SOCKET = (84.0, 112.0)         # socket floor, mouth (y): 28 deep, bore 6.2
COLLAR_R = 5.5                 # 2.1 of wall outside the 0.3 resin pockets
BOOM_END = 199.0               # 117 long
TIE_U = (1.0, 9.6)             # two ties across the booms under the grab unit: the float bolts screw down into them
TIE_HALF = 8.0
FLOAT_Z = (-45.0, 20.0)        # the float bolts' stations along the pack, from the tab line
FLOAT_X = 11.0                 # was 15: between the centre boom rod and the side ones
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
    body = U(box(-9, 9, 54, 66.75, u0, u1), hullp([[9, 59.75, z] for z in (u0, u1)] + [[11.5, yy, z] for z in (u0 + 2.5, u1 - 2.5) for yy in (62.25, 63.25)] + [[9, 65.75, z] for z in (u0, u1)]),
             hullp([[-9, 59.75, z] for z in (u0, u1)] + [[-11.5, yy, z] for z in (u0 + 2.5, u1 - 2.5) for yy in (62.25, 63.25)] + [[-9, 65.75, z] for z in (u0, u1)]),
             box(-9, 9, 38, 54.01, PLATE[2] + 1.0, u1),                                       # the keel beyond the plate
             )                                                                                # no park stop on the rail: the screw sets park
    for u in RAIL_BOLTS:
        body = body - cyl("y", 0, u, 2.75, 50, 70) - cyl("y", 0, u, 4.6, 61.0, 70)            # M5 x 10 socket head into the hub's insert
    return body - box(-2.5, 2.5, 62.75, 67, u0 + 3.5, u1 + 1)                                 # the lift's return spring lies in the channel
RAIL_BOLTS = (40.0, 78.0)
SCREW_TOP = 145.0


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
                screw=cyl("z", x, y, 4.0, -30.0, SCREW_TOP))
MOTOR_WALL = dict(x=(37.85, 80.15), y=(41.0, 84.0), u=(-44.5, -41.5))    # the stepper's face plate (31 mm pattern), through the plate's plane
THRUST_WALL = dict(x=(50.5, 71.0), y=(53.9, 71.0), u=(-11.5, -7.5))      # the screw through it between the two F8-16M, standing on the plate's front face; a 5 mm lock ring outside the outer one


# ------------------------------------------------------------------ carriage (park)
def wheel(x, c):
    """printed V wheel on two 625ZZ: o 26 x 10, 90 deg groove, apex r 10.25 (v7's wheel v1 was o 24: 1.2 of flank on the ridge; this has 2.2)"""
    R = WHEEL_R; d = R - 10.25
    cone = lambda t0, r0, r1: mf.Manifold.cylinder(d, r0, r1, 48).rotate([-90, 0, 0]).translate([x, t0, Z0 + c])
    w = U(cyl("y", x, Z0 + c, R, 57.75, 62.76 - d, 48), cone(62.75 - d, R, 10.25), cone(62.75, 10.25, R), cyl("y", x, Z0 + c, R, 62.74 + d, 67.75, 48))
    return w - cyl("y", x, Z0 + c, 8.0, 57, 62.75) - cyl("y", x, Z0 + c, 2.6, 57, 68)        # two 625ZZ (o 16 x 5) from one side, a 5 mm web behind them
def mgn_bolts():
    """the rail's four bolt stations on the spine (u): 25 pitch from 12.5 in"""
    return [MGN_U[0] + MGN["first"] + MGN["pitch"] * k for k in range(4)]
def carriage_bodies():
    """the drawn attachments: three wheel bosses, then the MGN rail's seat on the spine"""
    b = [cyl("y", x, Z0 + c, 10.0, PLATE_Y[0], PLATE_Y[1]) for x, c in WHEELS]                 # M5 axle bosses, the plate's full depth
    b += [box(RAIL_X - 6.0, RAIL_X + 6.0, SPINE["y"][1] - 3.0, SPINE["y"][1], MGN_U[0], MGN_U[1])]
    b += [box(SPINE["x"][1] - 0.01, STOP["x1"], LUG["y"][0], SPINE["y"][1], STOP["u"], STOP["u"] + STOP["t"])]   # the pick-up stop: a tab on the spine's side over the lift's nut lug
    return b
# 14 Nov (print kit): the stop was room over the rail's top, 12.8 past the rail seat's plane. The 13 Nov run kept 3.3 of it (the block passes it: no pick-up, and the
# block runs off the rail's end), and with it past the seat no face of the carriage lay on the bed (1 mm2 of first layer, supports on the rail seat). Now the lift's
# nut lug meets a drawn tab on the spine's side when the block's top reaches the rail's end, and everything ends on the rail seat's plane: that plane is the bed.
STOP = dict(u=LUG["u"][1] + T2, t=3.0, x1=54.0)     # 1.0 from the screw (o 8 at x 59); 2.7 x 16.1 of the lug's top under it
CAR_TOP = MGN_U[1] + 3.0
def carriage_env():
    s = SPINE; w3 = Z0 + WHEELS[2][1]
    return U(box(12.0, s["x"][0] + 0.01, PLATE_Y[0], PLATE_Y[1], 2.0, CAR_TOP),                               # the wheel pair's column of plate, 9.85 thick
             box(12.5, s["x"][0] + 0.01, 58.0, PLATE_Y[0] + 0.01, Z0 + WHEELS[0][1] + WHEEL_R + 2.5, Z0 + WHEELS[1][1] - WHEEL_R - 2.5),   # a rib behind the plate between the wheel pair, 10.75 deep, 1.0 from the V rail's ridge
             #                                                                                                  (13 Nov run: 0.53 where the plate bends between the low wheel's axle and the spine; it was 2.4 on the front, past the bed plane)
             box(-30.5, 12.01, PLATE_Y[0], PLATE_Y[1], w3 - 22.0, w3 + 22.0),                                 # and a strap across the rail to the third wheel, 44 wide
             box(s["x"][0], s["x"][1], s["y"][0], s["y"][1], MGN_U[0], CAR_TOP))                                 # the spine: the MGN rail on its front, the wheel pair beside it
def carriage_cuts():
    c = [cyl("y", x, Z0 + cc, 2.6, 50, 82) for x, cc in WHEELS] + [cyl("y", RAIL_X, u, 2.0, SPINE["y"][1] - 7.0, SPINE["y"][1] + 1) for u in mgn_bolts()]    # M3 heat-set inserts under the rail
    return U(*c)
def mgn_rail():
    return box(RAIL_X - 6.0, RAIL_X + 6.0, SPINE["y"][1], SPINE["y"][1] + MGN["rail_h"], MGN_U[0], MGN_U[1])
def mgn_block():
    y0 = SPINE["y"][1]
    return box(RAIL_X - 13.5, RAIL_X + 13.5, y0 + MGN["H1"], y0 + MGN["H"], BLOCK_U[0], BLOCK_U[1]) - box(RAIL_X - 6.2, RAIL_X + 6.2, y0, y0 + MGN["rail_h"] + 0.2, BLOCK_U[0] - 1, BLOCK_U[1] + 1)


# ------------------------------------------------------------------ lift (park)
def _b(d): return box(d["x"][0], d["x"][1], d["y"][0], d["y"][1], d["u"][0], d["u"][1])
def float_pts():
    return [(sx * FLOAT_X, Y_C + z) for sx in (-1, 1) for z in FLOAT_Z]
def boom_axes(): return list(BOOMS)
def booms():
    return U(*[cyl("y", x, u, 3.0, SOCKET[0], BOOM_END) for x, u in boom_axes()])
def tie(z):
    """one tie (printed, the same piece at both stations): a bar under the grab unit bonded on the three rods, the two float bolts' inserts between them"""
    y0, y1 = Y_C + z - TIE_HALF, Y_C + z + TIE_HALF; xo = max(abs(x) for x, _ in BOOMS) + COLLAR_R
    m = U(box(-xo, xo, y0, y1, TIE_U[0], TIE_U[1]), *[box(x - COLLAR_R, x + COLLAR_R, y0, y1, TIE_U[1] - 0.01, u + COLLAR_R) for x, u in boom_axes()])    # the springs' seats at 9.6, a ridge over each rod
    m = m - U(*[cyl("y", x, u, 3.1, y0 - 1, y1 + 1) for x, u in boom_axes()])
    m = m - U(*[cyl("z", sx * FLOAT_X, Y_C + z, 2.8, TIE_U[0] + 2.0, 30) for sx in (-1, 1)])               # M4 heat-set inserts from above
    m = m - U(*[cyl("z", x, (y0 + y1) / 2, 1.25, 0.0, u) for x, u in boom_axes()])                         # a resin port into each bore from underneath
    return m
def ties(): return [tie(z) for z in FLOAT_Z]
def sockets():
    """the yoke's socket collars (drawn): o 11 round each bore, from the floor's back to the mouth"""
    return [cyl("y", x, u, COLLAR_R, SOCKET[0] - 2.0, SOCKET[1]) for x, u in boom_axes()]
def mount_pad():
    """the yoke's seat on the MGN block's face (drawn)"""
    return box(RAIL_X - 13.5, RAIL_X + 13.5, PLATE_L["y"][0], PLATE_L["y"][0] + 3.0, BLOCK_U[0], BLOCK_U[1])
def nut_seat():
    x, y = SCREW; g = LUG
    return box(g["x"][0], g["x"][1], g["y"][0], y + 7.7, g["u"][0], g["u"][1])
def mount_bolts():
    uc = sum(BLOCK_U) / 2; h = MGN["pattern"] / 2
    return [(RAIL_X + sx * h, uc + su * h) for sx in (-1, 1) for su in (-1, 1)]
def lift_cuts():
    import sys
    if ROOT not in sys.path: sys.path.insert(0, ROOT)
    from parts import cage_node_lib as cn
    x, y = SCREW
    c = [cyl("z", x, y, 5.2, LUG["u"][0] - 1, LUG["u"][1] + 1), box(x - 5.2, x + 5.2, LUG["y"][0] - 1, y + 7.7, NUT_U[1], NUT_U[1] + 3.7)]   # the nut's body and its flange slot, in the lug
    c += [U(cyl("y", bx, bu, 1.7, PLATE_L["y"][0] - 5.0, 120), cyl("y", bx, bu, 3.2, PLATE_L["y"][0] + 6.0, 120)) for bx, bu in mount_bolts()]                # four M3 into the MGN block, heads sunk to 6 from the block
    for bx, bu in boom_axes():                                                                              # house sockets: bore 6.2, three resin pockets, the port out of the collar's underside
        c += cn.socket_cuts([bx, SOCKET[0], bu], [0, 1.0, 0], 0.0, SOCKET[1] - SOCKET[0], 3.1, build=(0.0, 0.0, 1.0), pockets=3, pocket_dir=[0, 0, -1.0]) + [cn.port_cut([bx, SOCKET[0], bu], [0, 1.0, 0], 0.0, [0, 0, -1.0], length=8.0)]
    return U(*c)
def lift_env():
    k = PLATE_L
    plate = U(_b(k), box(k["x"][0], 35.4, k["y"][1] - 0.01, YOKE["y"][1], k["u"][0], k["u"][1]))            # inside x 35.5 (the Scout bays beside the drone start there) the plate is 20.4 thick, as the yoke
    y = YOKE; bk = hullp([[x_, y_, u_] for (x_, y_) in ((BOOMS[2][0] - COLLAR_R, y["y"][0] - 0.01), (y["x"][1] + 0.01, y["y"][0] - 0.01), (y["x"][1] + 0.01, k["y"][0])) for u_ in (-100.0, 100.0)])
    # the yoke's back runs out to the plate's back face over its last 12: no inside corner where the 30 deep yoke meets the 20 deep plate beside the MGN block (two runs peaked there)
    return U(_b(YOKE) - bk, plate - box(0, 27.51 + 2.0, 0, 200, -100, 0.0) - box(RAIL_X + 13.6, 100, 0, 200, ARM["u"][1], 100), _b(ARM), _b(LUG), *sockets())      # below the axle line the plate starts at x 29.5: the other head's yoke ends at 26.5
def lift_bought():
    x, y = SCREW
    return dict(mgn_block=mgn_block(),
                nut=(U(cyl("z", x, y, 5.1, NUT_U[0], NUT_U[1]), cyl("z", x, y, 11.0, NUT_U[1], NUT_U[1] + 3.5) ^ box(x - 5.1, x + 5.1, y - 7.5, y + 7.5, 0, 60)) - cyl("z", x, y, 4.0, NUT_U[0] - 1, NUT_U[1] + 5)),
                booms=booms())


# ------------------------------------------------------------------ grab unit (tab frame; y 0 = the cartridge's bottom)
SEAT_X = 29.4                  # blade springs' seats
UNIT_BASE = (-38.5, -36.5)     # base plate (y), between the booms
CAM_Y = -24.3                  # servo and cam axis: the servo's top 14.3 under the pad plane (the funnel's ties stand at |x| 26..46 from 13.8 under it)
ROD_YT = -19.0                 # blade rods
PILOT = dict(x=(10.6, 16.7), z=(17.4, 22.3), top=5.0, lead=5.0, taper=3.0)    # two sprung pilots in the free corners of the recess
K_BLADE = 2 * 1.59             # N/mm per blade: two o 6 x 0.7 x 25 springs (E from the wire and coil count)
K_FLOAT, P_FLOAT = 4 * 5.76, 4 * 18.0     # four o 8 x 1.0 x 15 springs (5 active coils): N/mm, preload N (E)


def tabz(m, g): return m.translate([0, 0, -g["Z_T"]])
def turned(m):
    """the unit is drawn in the head's own frame (x -> -arm x, z -> arm y - Y_C: arm_T); since 13 Nov the drones sit half a turn round (each pack's
    plug at the hub end), so what the latch defines on the cartridge (heel, pads, posts, blades) is turned half a turn about the bar in it"""
    return m.rotate([0, 180, 0])
def unit_deck():
    """the latch's heel for both packs on one head: decks, hall posts, pads (TPU)"""
    m7 = latch(); decks, pads = [], []
    for pk in ("scout", "overseer"):
        g = packs()[pk]["g"]; m7.set_pack(g); d, p = m7.heel(g)
        decks.append(turned(tabz(d, g))); pads += [turned(tabz(q, g)) for q in p]
    return U(*decks), pads
_P = {}
def packs():
    if not _P:
        m7 = latch()
        for pk in ("scout", "overseer"): _P[pk] = m7.build(pk)
    return _P
def blade(side, face):
    m7 = latch(); g = packs()["scout"]["g"]
    b = turned(tabz(m7.blade(g, face, -1), g))
    b = U(b, *[box(face, face + 7.5, ROD_YT - 4.0, ROD_YT + 4.0, sz * 8.0 if sz > 0 else -16.5, sz * 16.5 if sz > 0 else -8.0) for sz in (-1, 1)],
          box(face, face + 4.7, -31.0, -19.9, -7.0, 7.0))                                                # the tail the cam bears on
    b = b - U(*[cyl("x", ROD_YT, z, 1.65, face - 1, face + 9) for z in (-12.0, 12.0)])
    return b if side > 0 else b.mirror([1, 0, 0])
def faces():
    m7 = latch(); return dict(open=m7.m6.BLADE_OPEN, grip=m7.bl.TAB_OUT - m7.m6.R0, limit=m7.bl.TAB_OUT - m7.bl.TRAVEL)
def pilot(sx, sz, ext=0.0):
    """a sprung pilot. 14 Nov (print kit): as drawn it had nothing to keep it in, no seat for its spring, and its stem's slot ran under the Scout pads beside it.
    Now: the tip as before; under it a neck 3.8 wide through the deck; under that the stem, as wide as the tip, whose top is the shoulder a o 3 carbon pin (along z,
    PILOT_PIN, pushed in from the blade bay; the blade's end then keeps it in) holds: up-stop at rest, PILOT_TRAVEL down onto two o 3 springs side by side on the
    pocket's floor (PILOT_FLOOR, over the float bolt's head room). Nothing of it is wider than the tip, so it drops in from above and the deck beside it is whole."""
    p = PILOT; x0, x1 = p["x"]; z0, z1 = p["z"]; t = p["taper"]; top = p["top"] + ext
    tip = hullp([[x, y, z] for (x, z) in ((x0, z0), (x1, z0), (x1, z1), (x0, z1)) for y in (-2.0 + ext, top - t)] + [[x, top, z] for (x, z) in ((x0, z0), (x1 - t, z0), (x1 - t, z1 - t), (x0, z1 - t))])
    stem = U(box(x0, PILOT_PIN[0] - 1.7, PILOT_STEM[1] - 0.01 + ext, -1.99 + ext, z0, z1), box(x0, x1, PILOT_STEM[0] + ext, PILOT_STEM[1] + ext, STEM_Z[0], z1))   # below the deck the stem stands beyond the blade's end
    m = U(tip, stem)
    if sx < 0: m = m.mirror([1, 0, 0])
    if sz < 0: m = m.mirror([0, 0, 1])
    return m
def pilot_pin(sx, sz):
    return cyl("z", sx * PILOT_PIN[0], PILOT_PIN[1], 1.5, *sorted((sz * 17.3, sz * 26.5)))
STEM_Z = (19.5, 22.3)
PILOT_STEM = (-17.5, -12.0)    # the stem under the pin's level; its top is the shoulder the pin holds
PILOT_PIN = (16.1, -10.5)      # |x|, y of the o 3 x 9 carbon pin
PILOT_FLOOR = -28.0            # spring pocket floor: 10.5 of spring at rest, 5.5 with the pilot flush
PILOT_TRAVEL = 5.0
PAD_T = 2.4                    # the heel pads, nominal: glued in 0.4 deep seats in the deck, picked from 2.0 / 2.2 / 2.4 / 2.6 / 2.8 at set-up (they were on M3 jack screws whose
PAD_STEPS = (2.0, 2.2, 2.4, 2.6, 2.8)   # holes ended under the deck, and two of the Scout's four stood over the pilots' slots)
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
        gd = box(8.2, 21.1, PILOT_FLOOR - 2.0, -5.9, 17.3, 27.9)
        if sz < 0: gd = gd.mirror([0, 0, 1])
        if sx < 0: gd = gd.mirror([1, 0, 0])
        guides.append(gd)
    bbs = [q.bounding_box() for q in pads]
    pads = [box(b[0], b[3], -PAD_T, 0.0, b[2], b[5]) for b in bbs]                                         # nominal pads, 0.4 into their seats
    m = U(deck, *walls, base, *cross, *blocks, *guides)
    cut = [cyl("x", ROD_YT, z, 1.6, -40, 40) for z in (-12.0, 12.0)]                                       # o 3 carbon rods (the blades slide on them)
    cut += [cyl("z", 0.0, CAM_Y, 6.5, 22.0, 28.0)]                                                         # the servo's boss through its wall
    cut += [U(box(*sorted((sx * 10.2, sx * 17.1)), PILOT_FLOOR, 0.0, *sorted((sz * (STEM_Z[0] - 0.4), sz * (STEM_Z[1] + 0.4)))),                  # stem and spring pocket, blind
              box(*sorted((sx * 10.2, sx * 17.1)), PILOT_STEM[1] - PILOT_TRAVEL - 0.5, 0.0, *sorted((sz * 17.0, sz * (STEM_Z[1] + 0.4)))),
              cyl("z", sx * PILOT_PIN[0], PILOT_PIN[1], 1.55, *sorted((sz * 16.0, sz * 26.5)))) for sx, sz in PILOTS]
    cut += [U(cyl("y", sx * FLOAT_X, z, 5.0, y0 - 1, y1 + 1), cyl("y", sx * FLOAT_X, z, 8.0, y1, y1 + 6.5)) for sx in (-1, 1) for z in FLOAT_Z]   # o 10 round each o 5 shoulder (+-2.5 of float); room for the head and its washer
    cut += [box(*sorted((sx * 11.3, sx * 14.7)), 3.8, 5.7, *sorted((sz * 18.7, sz * 21.5))) for sx, sz in ((1, 1), (-1, -1))]   # a pocket in each post for its hall sensor
    cut += [box(b[0] - 0.2, b[3] + 0.2, -PAD_T, 0.0, b[2] - 0.2, b[5] + 0.2) for b in bbs]                 # the pads' seats
    cut += [box(-10.2, 30.2, CAM_Y - 10.2, CAM_Y + 10.2, 27.0, 68.5)]                                      # the servo
    cut += [box(-13.0, 13.0, y1, -5.0, -10.5, 8.0)]                                                        # the cam's sweep, and 4.5 behind it: the paddle is lowered in there and slid onto the spline
    cut += [box(-6.5, 6.5, CAM_Y, 1.0, 7.9, 27.01), cyl("z", 0.0, CAM_Y, 3.2, -48.0, -10.0)]               # the paddle's nose down into the wall's hole from above; its M3 x 35 and the driver in from the hub end
    # 14 Nov (print kit): the servo could not be put in or held. It slides in from the open +x side, so its boss and spline need a slot through the
    # wall out to that side; the inner ear screws from behind (two M3 x 8, tapped in the print) to a tab between the side wall and the servo's end,
    # the driver through two holes in the end bar. The outer ear stays free: a tab there would stand in the servo's way in.
    e = SERVO_EAR
    cut += [box(0.0, 34.0, CAM_Y - 6.5, CAM_Y + 6.5, 22.0, 28.0)]
    tab = box(-23.01, -10.6, y1 - 0.01, CAM_Y + e["dy"] + 3.5, e["z"] - 4.0, e["z"]) - U(*[cyl("z", e["x"][0], CAM_Y + sy * e["dy"], 1.25, e["z"] - 5, e["z"] + 1) for sy in (-1, 1)])
    cut += [cyl("z", e["x"][0], CAM_Y + sy * e["dy"], 2.5, 68.0, 72.0) for sy in (-1, 1)]
    return (m + tab) - U(*cut), pads
SERVO_EAR = dict(x=(10.0 - 24.75, 10.0 + 24.75), dy=5.0, z=68.4 - 27.7 - 2.5, t=2.5, span=54.5)       # DS3218 (systems/dimensions.md: ears 4 x o 4.5 on 49.5 x 10, 54.5 over them, underside 27.7 above the base; thickness 2.5 E)
SPLINE = dict(r=2.98, tip=22.0, case=28.0)     # 25T, o 5.96; its tip 6.0 out of the case top (E: boss and spline height are not in the sheet, measure)
def paddle(state="grip"):
    """the cam as it is printed (14 Nov; it was a brick with nothing to hold it on the servo): the bar between the blades' tails, a o 8 stem, a o 11 nose that enters
    the wall's o 13 hole and takes the 25T spline in a o 5.8 bore it cuts its own teeth in (E), drawn down on the spline by one M3 x 35 from the cam's end. o 10 round the
    screw in the bar: 3.3 of wall each side (the gate reads less as a neck); 0.4 from the blades' tails at their limit."""
    bar = box(-4.0, 4.0, -11.25, 11.25, -6.0, 6.0) if state != "open" else box(-11.25, 11.25, -4.0, 4.0, -6.0, 6.0)
    m = U(bar, cyl("z", 0, 0, 5.0, -6.0, 6.0), cyl("z", 0, 0, 4.0, 5.9, 14.5), mf.Manifold.cylinder(1.5, 4.0, 5.5, 48).translate([0, 0, 14.5]), cyl("z", 0, 0, 5.5, 16.0, SPLINE["case"] - 2.0))
    m = m - cyl("z", 0, 0, 2.9, SPLINE["tip"] - 1.5, 30) - cyl("z", 0, 0, 1.7, -7, 30)
    return m.translate([0, CAM_Y, 0])
def unit_bought(state="grip"):
    f = faces()[state]
    e = SERVO_EAR
    fb = [(sx * FLOAT_X, z) for sx in (-1, 1) for z in FLOAT_Z]
    out = dict(servo=U(box(-10.0, 30.0, CAM_Y - 10.0, CAM_Y + 10.0, 28.0, 68.4), cyl("z", 0.0, CAM_Y, 6.0, SPLINE["case"] - 2.0, 28.0), cyl("z", 0.0, CAM_Y, SPLINE["r"] - 0.1, SPLINE["tip"], 28.0),
                        box(10.0 - e["span"] / 2, 10.0 + e["span"] / 2, CAM_Y - 9.0, CAM_Y + 9.0, e["z"], e["z"] + e["t"])),      # with its ears, boss (2 high, E) and spline
               paddle=paddle(state),
               blade_rods=U(*[cyl("x", ROD_YT, z, 1.5, -33.0, 33.0) for z in (-12.0, 12.0)]),
               blade_springs=U(*[cyl("x", ROD_YT, z, 3.0, *sorted((sx * (f + 7.5), sx * SEAT_X))) - cyl("x", ROD_YT, z, 2.3, -40, 40) for sx in (-1, 1) for z in (-12.0, 12.0)]),
               halls=U(*[box(*sorted((sx * 11.5, sx * 14.5)), 4.0, 5.5, *sorted((sz * 18.9, sz * 21.3))) for sx, sz in ((1, 1), (-1, -1))]),
               float_bolts=U(*[U(cyl("y", x, z, 2.5, -56.0, -36.5), cyl("y", x, z, 7.5, -36.5, -35.3), cyl("y", x, z, 4.2, -35.3, -31.8)) for x, z in fb]),
               float_springs=U(*[cyl("y", x, z, 4.0, -50.4, -38.5) - cyl("y", x, z, 3.0, -51, -38) for x, z in fb]))
    return out


# ------------------------------------------------------------------ statics: what the lift, the carriage and the hub see (N, mm; arm frame)
def mgn_wrench(F, P):
    """a force F at point P on the lift (arm frame, lift at park) -> what its MGN block passes to the rail: the force across the rail (x, y; the nut takes the bar direction)
    and the moment about the block's centre (N mm), and the block's static index: |M pitch| / MP + |M yaw| / MY + |M roll| / MR + |F| / C0 (rail along u, mounting face normal y)"""
    F = np.asarray(F, float); P = np.asarray(P, float)
    C = np.array([RAIL_X, SPINE["y"][1] + MGN["rail_h"] / 2, sum(BLOCK_U) / 2]); N = np.array([SCREW[0], SCREW[1], sum(NUT_U) / 2])
    R = np.array([F[0], F[1], 0.0]); Fn = np.array([0, 0, -F[2]])
    M = np.cross(P - C, F) + np.cross(N - C, Fn)
    idx = abs(M[0]) / (1000 * MGN["MP"]) + abs(M[1]) / (1000 * MGN["MY"]) + abs(M[2]) / (1000 * MGN["MR"]) + float(np.linalg.norm(R)) / MGN["C0"]
    return R, M, C, float(idx)


def cases(pack="overseer"):
    """the head's load cases at the grab unit (force on the unit from the cartridge, point of action), each its own:
    name, class, F (arm frame), point, s"""
    m = dict(scout=0.222, overseer=0.766)[pack] + 0.25                 # pack + grab unit, kg (unit E)
    pc = np.array([0.0, Y_C, G_PARK]); cg = np.array([0.0, Y_C, G_PARK + 28.0])
    pp = plug_point(pack); w = 9.81 * (m - 0.25); Fi = 57.0 + w
    # in the receiver's guides the cartridge stays square, so the head's force acts where the head applies it, not at the plug (the guides take the couple between):
    # pushing, through the float springs (centre FLOAT_Z's middle, + the float's pitch rate over the guides' 0.47 deg = 0.2 N m); pulling, at the blades = the tab line
    # (the unit hangs rigid on the float bolts' heads). Turned pose (13 Nov): the plug is on the hub side of both, so the guides' couple can only shorten the lever; none counted.
    pf = pc + [0.0, sum(FLOAT_Z) / 2, 0.0]; off = np.array([0.0, 1000.0 * 0.2 / Fi, 0.0])
    out = [("insert at the drone, %s, in the receiver's guides: %.1f N through the float springs" % (pack, Fi), "flight", [0, 0, -Fi], pf + off, S_DRONE),
           ("pull-out at the drone, %s, in the receiver's guides: %.1f N at the blades" % (pack, Fi), "flight", [0, 0, Fi], pc, S_DRONE),
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
    """where the XT60's force acts (arm frame, park): at the pack's lead end, which the turned pose (13 Nov) puts at the hub end"""
    g = packs()[pack]["g"]; return np.array([0.0, Y_C - (g["Z_X"] - g["Z_T"]), G_PARK])


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
    E = 1900.0; Es = E / (2 * (1 - 0.4 ** 2)); band = (WHEEL_R - (22.3 - 11.5)) * 2 ** 0.5; Rw = (WHEEL_R + 10.8) / 2 * 2 ** 0.5; pre = 10.0
    hz = 0.0
    for fx, fy in out:
        Fn = (abs(fx) + pre + abs(fy)) / 2 ** 0.5; a_ = (4 * Fn * Rw / (np.pi * band * Es)) ** 0.5; hz = max(hz, 2 * Fn / (np.pi * a_ * band) / (1.67 * 55.0))
    return out, round(float(hz), 3), round(max(abs(fy) for _, fy in out) / 670.0, 3)
