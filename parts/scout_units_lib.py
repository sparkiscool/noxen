"""The Scout's printed inside parts (4 Nov, Spark: "design inside parts that hold everything"): one library, three setups.

  scout_nose_unit   the nose yoke (plate to the receiver's nose inserts), the nose runner (the Scout's seat in the shared funnel,
                    with a window the flow sensor looks down through), the camera boom (four M2 posts behind the board), cage node
                    NB printed in, and the two nose tie sockets (C0 / C3 -> B0 / B3)
  scout_tail_unit   the tail yoke (plate to the receiver's tail inserts), the tail runner, the electronics tray (Radxa Zero 3W on
                    four M2.5 posts, UBEC and XR2 on tape lands), cage node TB printed in, the two tail tie sockets (C1 / C2)
  scout_arm_clip_k  a clip under each arm tip on the motor's four M2 screws with the socket of tie A -> B (k = PX4 motor 1..4)

Drawn = attachments only (CLAUDE.md): bolt pads with M3 x 5.7 inserts, receiver bosses, the seat rims and 55 deg lead-ins the
funnel meets, tape lands, M2 / M2.5 posts, socket collars. Everything between them is generated. Keep-outs are cuts: the receiver
and its clearance, the boards with their plugs and cables, the camera body, plugs and view, the prop discs grown by the 12 mm
cage rule (r 60.45, z 25.2..57.2), the rods beyond their sockets.
Frame: drone (x nose, y left, z up; plate underside z 18.5). Geometry C from systems/dimensions.md and tools/scout_meshes.py; rod
forces F from systems/scout.cage_loads.json (tools/scout_cage_loads.py); masses E.
Receiver v3 (latch): x -51.3..54.3, |y| 32.8, z -44.3..9.6, 4 M3 inserts in its top at (-37.65, +-26.55), (48.28, +-26.55): the
units carry it by bolts up from below (the receiver's bolt keys) into inserts in their bosses. Receiver + cartridge + pack ~310 g
(E), the nose and tail pairs each loaded with 155 g (E, the heavier share).
"""
import json, os
import numpy as np, trimesh, manifold3d as mf
from topo.part import man
from parts.cage_node_lib import socket_cuts, collar_shape, cone, U

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOADS = json.load(open(os.path.join(ROOT, "systems", "scout.cage_loads.json")))
NL = LOADS["node_loads"]
G = 9.81e-3

# ---- the frame, the receiver, the plate -------------------------------------------------------------------------------
PLATE_Z = 18.5                                              # plate underside
RX0, RX1, RY, RZ0, RZ1 = -51.3, 54.3, 32.8, -44.3, 9.6       # receiver v3 (C)
INS_NOSE = [(48.28, 26.55), (48.28, -26.55)]                 # receiver inserts the nose yoke bolts to (C)
INS_TAIL = [(-37.65, 26.55), (-37.65, -26.55)]
REC_SET_G, REC_SET_CG = 155.0, np.array([1.5, 0.0, -20.0])  # the half each yoke carries (E), the set's CG (E)
BOLT_NOSE = [(36.0, 0.0), (19.93, 0.0), (1.9, 18.03), (1.9, -18.03), (3.9, 31.0), (3.9, -31.0)]   # M3 down through the plate into inserts, all Bee35 holes (tools/bee35_plate.HOLES): the
#   nose bolt, the FC's 25.5 diamond (the FC sits on top of the plate since 4 Nov, its standoff bolts go on through the plate), the (3.9, +-31) pair
BOLT_TAIL = [(-16.13, 0.0), (-55.0, 13.0), (-55.0, -13.0)]   # the FC diamond's tail hole (Bee35) and one drilled pair at the plate's tail end (at (-55, +-20) the pads were sheared by the UBEC lead clearance, run 1)
PAD_R, PAD_T, INS_R, INS_D = 5.0, 6.0, 2.0, 6.0               # pad O10 x 6 under the plate, M3 x 5.7 insert hole O4 x 6
BOSS_R = 5.0                                                 # receiver boss O10, z 9.6..18.5, insert hole from below
YOKE_SPLIT_X = -8.0                                          # the nose unit's yoke runs x -8..54.3 over the receiver, the tail unit's -51.3..-8 (the FC is on top of the plate since 4 Nov)
# ---- the funnel seat: runners to the Overseer receiver's 184 x 66.8 outline at z -44.3 (parts/box_funnel/make.py, parts/scout_belly_skirt)
CX, SEAT_L, SEAT_W, SEAT_Z, RUN_H = 1.5, 184.0, 66.8, -44.3, 20.0
HOLE_X, HOLE_Y = 85.0, 25.0                                  # the funnel's hole 170 x 50: the seat ring is what bears
RIM_T, CHAMFER, FACE_DEG = 3.0, 4.0, 55.0
# ---- camera (tools/scout_meshes.camera, CAMERA_LENS / CAMERA_TILT): lens (128, 0, 32) looking 30 deg up, board back 12 behind the lens, M2 on 28 x 28
LENS, TILT = np.array([130.0, 0.0, 38.0]), np.radians(30.0)             # 30 deg up (4 Nov, Spark: close-ups and ceilings); tools/scout_meshes.CAMERA_LENS / CAMERA_TILT
CAM_F, CAM_U, CAM_R = np.array([np.cos(TILT), 0, np.sin(TILT)]), np.array([-np.sin(TILT), 0, np.cos(TILT)]), np.array([0, 1.0, 0])
BOARD_BACK = LENS - 12.0 * CAM_F                                          # the board's back face centre
CAM_SQ, POST_L, POST_R, CAM_INS_R, CAM_INS_D, PLUG_BACK = 14.0, 10.0, 2.8, 1.6, 5.7, 8.0
CAM_G, CAM_CG = 10.0, LENS - 8.9 * CAM_F
HFOV, VFOV = 44.5, 36.5
# ---- flow sensor MTF-02P in the nose runner's window (tools/scout_meshes.FLOW_AT (70, 0, -24.5): lens; board 21.6 x 16, 6.5 tall)
FLOW_C, FLOW_LENS_Z = (70.0, 0.0), -24.5
FLOW_BOARD = (21.6, 16.0); FLOW_TOP = FLOW_LENS_Z + 6.1       # board top face: taped to the land above it
WINDOW = (57.0, 83.0, 10.0)                                  # x0, x1, |y|: the runner's window under the sensor
# ---- tail electronics, on edge / flat in the tail unit's bay over the tail runner (4 Nov, Spark: the electronics into the printed base; then
#   "confirm the mounting is accessible": the Radxa's four M2.5 screws come in from the open tail end, so the UBEC lies flat under the board
#   instead of standing behind it; tools/scout_meshes)
#   Radxa Zero 3W: PCB x -80..-78.4, components to -x (to -83.2), 65 along y, 30 along z (-15..15), ports edge up (plugs to z 28 between
#   the arms), SD end at -y; its STEP holes (3.55, 3.60), (3.60, 26.45), (61.40, 3.60), (61.40, 26.50) -> drone (y -32.5 + X, z 15 - Y),
#   on O8 posts along +x with M2.5 x 4 inserts; a driver column along -x from each hole is a cut
RADXA_X = -78.4                                              # the PCB's back face (the posts' seat), posts to +x
RADXA_HOLES = [(-32.5 + X, 15.0 - Y) for X, Y in ((3.55, 3.60), (3.60, 26.45), (61.40, 3.60), (61.40, 26.50))]   # (y, z)
RPOST_R, RPOST_L, RINS_R, RINS_D = 4.0, 5.0, 1.75, 4.0       # O8 post 5 long behind the PCB, M2.5 x 4 insert (E)
DRIVER_R, DRIVER_L = 4.0, 40.0                               # the screwdriver's way in: O8 from the hole out along -x, 40 long
RADXA_CUT = (-83.2 - 0.5, RADXA_X + 0.5, -33.0, 33.0, -15.5, 15.5)
RADXA_PLUGS = (-85.0, -76.0, -24.0, 27.0, 15.0, 28.0)        # USB-C x2 and micro HDMI plugs up from the ports edge, between the arms
RADXA_SD = (-79.0, -74.5, -46.0, -32.0, -8.5, 7.5)           # the microSD card out the -y end
UBEC = (-94.0, -77.0, -33.0, 28.0, -24.3, -16.3)             # flat on the runner under the Radxa (its bottom edge at -15), 61 along y, 17 along x, 8 tall; leads out the +-y ends
XR2 = (-78.0, -74.0, -8.0, 8.0, -12.0, 0.0)                  # board x -75..-74 on its land (x -74..-70), components to -x (to -78, 1.4 clear of the posts' tips); antenna along +y
XR2_ANT = (-77.0, -75.0, 6.0, 27.0, -3.0, -1.0)
LAND_T = 4.0                                                 # the UBEC's and XR2's tape lands: 4 thick plates (2 mm lands left blobs on necks, run 1)
LEAD_CH = (8.0, 5.0, 6.0)                                    # lead channel: width (x here), height (z), length out of each UBEC end
BOARDS_G, BOARDS_CG = 32.0, np.array([-82.0, 0.0, -6.0])    # Radxa 10 + UBEC 15 + XR2 1 + cables 6 (E)
# ---- sockets (cage_node_lib: 3.0 rod, bore 3.2 x 10, floor 3, wall 2.5; the bore bottom 3.0 from the node centre, as the Overseer's node 17)
BORE_R, DEPTH, FLOOR, WALL = 1.6, 10.0, 3.0, 2.5
# ---- prop keep-out: the cage rule for anything printed too (12 beyond the swept discs)
MOTORS = [(54.1, 54.1), (-54.1, 54.1), (-54.1, -54.1), (54.1, -54.1)]
DISC_R, DISC_Z = 44.45 + 4.0 + 12.0, (37.2 - 12.0, 45.2 + 12.0)
DIRS14 = [np.array(v, float) / np.linalg.norm(v) for v in
          [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
FLIGHT = {"flight 4 g": np.array([0, 0, 4.0]), "flight 2 g x": np.array([2.0, 0, 1]), "flight 2 g y": np.array([0, 2.0, 1]), "landing 6 g": np.array([0, 0, 6.0])}
CATCH_G, DRONE_G = 35.0, 765.0                               # a hard landing into the funnel: 35 g of the drone on the seat (E), half a runner's share


def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl(x, y, z0, z1, r, n=40):
    z0, z1 = sorted((z0, z1))
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def along(q0, q1, r, n=32):
    return man(trimesh.creation.cylinder(r, float(np.linalg.norm(np.subtract(q1, q0))), sections=n, segment=np.vstack([q0, q1])))


def hull(pts):
    return man(trimesh.convex.convex_hull(np.asarray(pts, float)))


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def rigid_split(pts, F, P):
    """least-norm forces at pts carrying force F at point P (force and moment)"""
    pts = np.asarray(pts, float); c = pts.mean(0); n = len(pts); Am = np.zeros((6, 3 * n))
    for i, q in enumerate(pts - c):
        Am[:3, 3 * i:3 * i + 3] = np.eye(3); Am[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
    return (np.linalg.pinv(Am) @ np.r_[F, np.cross(np.asarray(P, float) - c, F)]).reshape(n, 3)


# ---- sockets --------------------------------------------------------------------------------------------------------------
class Sock:
    """a 3.0 rod socket at a printed node: centre c, rod toward `far`; bore bottom FLOOR from c, mouth FLOOR + DEPTH"""
    def __init__(self, name, c, far):
        self.name, self.c = name, np.asarray(c, float); self.d = unit(np.asarray(far, float) - self.c)
        self.t_mouth = FLOOR + DEPTH

    def pt(self, t):
        return self.c + self.d * t

    def collar(self):
        return collar_shape(self.c, self.d, 0.0, self.t_mouth, BORE_R, WALL)

    def bores(self):
        return socket_cuts(self.pt(FLOOR), self.d, 0.0, DEPTH, BORE_R)

    def clear(self, length=300.0):
        return along(self.pt(self.t_mouth + 0.5), self.pt(self.t_mouth + length), BORE_R + 0.9)

    def region(self, extra=0.4):
        return along(self.pt(FLOOR), self.pt(self.t_mouth - 1.0), BORE_R + WALL + extra)

    def side(self):
        s = np.cross(self.d, [0, 0, 1.0])
        return unit(s) if np.linalg.norm(s) > 1e-6 else np.array([1.0, 0, 0])


def socket_cases(p, socks, support, log_prefix=""):
    """the frame model's member forces at each socket (F): tension pulls the unit toward the member's far end, compression pushes
    in, plus the side force across the member; each member alone"""
    for s in socks:
        ld = NL.get(s.name.split()[0], {}).get(s.name.split()[1]) if " " in s.name else None
        if ld is None:
            raise KeyError(f"no frame-model load for socket {s.name}")
        for sgn, lab in ((1, "tension"), (-1, "compression")):
            F = sgn * ld["axial"] * s.d + ld["side"] * s.side()
            p.load_multi(f"{log_prefix}{s.name.split()[1]} {lab} {ld['axial']:.0f} N + side {ld['side']:.0f} N", [(s.region(), F)], cls="crash", support=support)


def check_collars(socks, cuts, keep=0.97):
    """every socket collar must stand clear of the keep-outs: a collar sheared by a cut prints as a sliver the gate refuses (4 Nov run 1:
    the arm clips' collars under the plate, the tail unit's under the UBEC lead clearance)"""
    for s in socks:
        c = s.collar(); v0 = float(c.volume()); v1 = float((c - cuts).volume())
        if v1 < keep * v0:
            raise ValueError(f"socket {s.name}: its collar loses {1 - v1 / v0:.0%} of itself to a keep-out cut; move the root or the cut")


def node_socks(node, members, nodes):
    """sockets of a printed cage node: node = its key, members = the far node keys, nodes = tools/scout_cage.NODES + TIE_ROOTS"""
    return [Sock(f"{node} {node}-{m}", nodes[node], nodes[m]) for m in members]


def cage_nodes():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import scout_cage as sc
    return {**sc.NODES, **sc.TIE_ROOTS}, sc.RODS, sc.TIES


# ---- shared bodies ------------------------------------------------------------------------------------------------------------
def plate_pads(bolts):
    """O10 x 6 pads under the plate with the M3 insert hole from the top (the plate's bolt comes down into it)"""
    drawn = [cyl(x, y, PLATE_Z - PAD_T, PLATE_Z, PAD_R) for x, y in bolts]
    cuts = [cyl(x, y, PLATE_Z - INS_D, PLATE_Z + 1.0, INS_R) for x, y in bolts]
    regs = [cyl(x, y, PLATE_Z - 1.5, PLATE_Z + 0.3, PAD_R) for x, y in bolts]
    return drawn, cuts, regs


def receiver_bosses(ins):
    """O10 bosses from the receiver's top (z 9.6) to the plate underside, M3 insert hole up from below (the receiver's bolt keys)"""
    drawn = [cyl(x, y, RZ1, PLATE_Z, BOSS_R) for x, y in ins]
    cuts = [cyl(x, y, RZ1 - 1.0, RZ1 + INS_D, INS_R) for x, y in ins]
    regs = [cyl(x, y, RZ1 - 0.3, RZ1 + 2.0, BOSS_R) for x, y in ins]
    return drawn, cuts, regs


def receiver_cut():
    return box(RX0 - 0.5, RX1 + 0.5, -RY - 0.5, RY + 0.5, RZ0 - 1.0, RZ1)


def runner(x0, x1, nose=True):
    """the seat: a 3 mm rim along the runner's outer bottom edges over the funnel's seat ring (the end band and the side bands, the
    hole 170 x 50 under the rest), its outer bottom edges chamfered 55 deg (the funnel's faces), the inside open for the web.
    Returns drawn (rim), the whole runner block (envelope), the seat regions (bottom face, the end face, the side faces)"""
    W2 = SEAT_W / 2; z0, z1 = SEAT_Z, SEAT_Z + RIM_T
    xe0, xe1 = (CX + HOLE_X, x1) if nose else (x0, CX - HOLE_X)           # the end band of the seat ring
    rim = [box(x0, x1, -W2, -HOLE_Y, z0, z1), box(x0, x1, HOLE_Y, W2, z0, z1), box(xe0, xe1, -W2, W2, z0, z1)]
    # the 55 deg lead-in on the outer bottom edges: a chamfer CHAMFER wide, cut off the rim (and off the block)
    t = np.tan(np.radians(FACE_DEG)); ch = CHAMFER
    wedges = []
    for s in (-1, 1):                                                          # side edges along x
        pts = [(x0 - 1, s * W2, z0 - 1), (x1 + 1, s * W2, z0 - 1), (x0 - 1, s * (W2 - ch), z0 - 1), (x1 + 1, s * (W2 - ch), z0 - 1),
               (x0 - 1, s * (W2 + 1), z0 + ch * t + 1), (x1 + 1, s * (W2 + 1), z0 + ch * t + 1), (x0 - 1, s * W2, z0 + ch * t), (x1 + 1, s * W2, z0 + ch * t)]
        wedges.append(hull(pts))
    xe = x1 if nose else x0; se = 1 if nose else -1                            # the end edge along y
    pts = [(xe, -W2 - 1, z0 - 1), (xe, W2 + 1, z0 - 1), (xe - se * ch, -W2 - 1, z0 - 1), (xe - se * ch, W2 + 1, z0 - 1),
           (xe + se, -W2 - 1, z0 + ch * t + 1), (xe + se, W2 + 1, z0 + ch * t + 1), (xe, -W2 - 1, z0 + ch * t), (xe, W2 + 1, z0 + ch * t)]
    wedges.append(hull(pts))
    wedge = U(wedges)
    drawn = U(rim) - wedge
    block = box(x0, x1, -W2, W2, SEAT_Z, SEAT_Z + RUN_H) - wedge
    regs = dict(bottom=box(xe0 if nose else x0, x1 if nose else xe1, -W2, W2, z0 - 0.3, z0 + 1.0),
                end=box(xe - se * (ch + 1.5), xe + 1, -W2, W2, z0 - 0.3, z0 + ch * t + 1),
                sides=U([box(x0, x1, s * (W2 - ch - 1.5), s * (W2 + 1), z0 - 0.3, z0 + ch * t + 1) for s in (-1, 1)]))
    return drawn, block, regs, wedge


def disc_cuts():
    return [cyl(mx, my, DISC_Z[0], DISC_Z[1], DISC_R, 96) for mx, my in MOTORS]


def payload_cases(p, name_prefix, pts, regs, mass_g, cg, support, cls_crash="crash"):
    """a carried mass on its attachment points: 35 g crash in 14 directions, flight and landing; forces by rigid_split"""
    for k, d in enumerate(DIRS14):
        a = 35.0 * d
        f = rigid_split(pts, -mass_g * G * a, cg)
        p.load_multi(f"{name_prefix} crash {k} (35 g {np.round(d, 2).tolist()})", [(regs[i], f[i]) for i in range(len(pts))], cls=cls_crash, accel=-a, support=support)
    for nm, a in FLIGHT.items():
        f = rigid_split(pts, -mass_g * G * a, cg)
        p.load_multi(f"{name_prefix} {nm}", [(regs[i], f[i]) for i in range(len(pts))], cls="flight", accel=-a, support=support)


def catch_cases(p, regs, nose, support):
    """a hard landing into the funnel: the drone's 35 g on the seat, this runner's share 0.42 (its band of the ring, E): up on the
    bottom band, and the funnel's end face / side faces pushing the lead-ins at 55 deg"""
    F = CATCH_G * DRONE_G * G * 0.42
    c, s = np.cos(np.radians(FACE_DEG)), np.sin(np.radians(FACE_DEG))
    p.load("funnel catch 35 g: seat up", regs["bottom"], (0, 0, F), cls="crash", support=support)
    sx = -1 if nose else 1
    p.load("funnel catch 35 g: end face", regs["end"], (sx * F * c, 0, F * s), cls="crash", support=support)
    p.load("funnel catch 35 g: side faces", regs["sides"], (0, 0, F * s), cls="crash", support=support)


def preview(part, name):
    """the drawn bodies as parts/<name>/part.stl until the engine's run replaces it (the page shows the attachments)"""
    m = part.drawn_m.to_mesh()
    t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    d = os.path.join(HERE, name); os.makedirs(d, exist_ok=True); t.export(os.path.join(d, "part.stl"))
    return t


def export_env(part, name):
    m = part.env_m.to_mesh()
    t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    t.export(os.path.join(HERE, name, "envelope.stl"))
    return t


# ---- arm clips -------------------------------------------------------------------------------------------------------------------
CLIP = {1: (1, -1, "A3", "B3"), 2: (-1, 1, "A1", "B1"), 3: (1, 1, "A0", "B0"), 4: (-1, -1, "A2", "B2")}   # PX4 motor k: arm signs, tie
CLIP_Z0, CLIP_T, M2_R, M2_PAD_R, SHAFT_R = PLATE_Z - 8.0, 8.0, 1.1, 3.0, 3.0   # 8 thick (4 Nov run 1: at 4 the tie collar's top, root z 17, was sheared by the plate keep-out to a 1.1 sliver; roots now z 13.5)


def arm_clip_setup(k, h):
    """the clip under arm k's tip: 4 pads on the motor's M2 screws (12 x 12 along x / y, M2 x 10 through clip + plate into the
    motor), the shaft clearance, the socket collar of tie A -> B hanging under the arm tip"""
    sx, sy, a, b = CLIP[k]
    nodes, rods, ties = cage_nodes()
    mx, my = sx * 54.1, sy * 54.1
    holes = [(mx + 8.49 * c, my + 8.49 * s) for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    drawn = [cyl(x, y, CLIP_Z0, PLATE_Z, M2_PAD_R) for x, y in holes]
    cuts = [cyl(x, y, CLIP_Z0 - 1.0, PLATE_Z + 1.0, M2_R) for x, y in holes] + [cyl(mx, my, CLIP_Z0 - 1.0, PLATE_Z + 1.0, SHAFT_R)]
    cuts.append(box(-70, 67, -70, 70, PLATE_Z, PLATE_Z + 4.0))
    cuts.append(hull([(mx + sx * 30, my + sy * 30, PLATE_Z - 0.5), (mx - sx * 20, my + sy * 30, PLATE_Z - 0.5), (mx + sx * 30, my - sy * 20, PLATE_Z - 0.5),
                      (mx + sx * 30, my + sy * 30, PLATE_Z + 4), (mx - sx * 20, my + sy * 30, PLATE_Z + 4), (mx + sx * 30, my - sy * 20, PLATE_Z + 4)]))
    s = Sock(f"{a} {a}-{b}", nodes[a], nodes[b])
    check_collars([s], U(cuts))
    drawn.append(s.collar()); cuts += s.bores(); cuts.append(s.clear())
    env = U([box(mx - 14, mx + 14, my - 14, my + 14, CLIP_Z0, PLATE_Z), along(s.pt(-8.0), s.pt(s.t_mouth), BORE_R + WALL + 4.0),
             hull([(mx, my, CLIP_Z0), (mx, my, PLATE_Z), s.pt(-8.0) + [0, 0, -6], s.pt(-8.0) + [0, 0, 2], s.pt(s.t_mouth) + [0, 0, -6], s.pt(s.t_mouth) + [0, 0, 2]])])
    from topo import Part
    p = Part(f"scout_arm_clip_{k}", h, U(drawn), U(cuts), env, build="+z", material="PETG")
    p.support("default", U([cyl(x, y, PLATE_Z - 1.5, PLATE_Z + 0.3, M2_PAD_R) for x, y in holes]))
    socket_cases(p, [s], "default")
    return p
