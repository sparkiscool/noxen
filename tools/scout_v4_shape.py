"""Scout v4, the shape design (8 Nov, Spark: "we are going to start on scout v4, should have new props meaning we are no longer using
the bee35 plate and will be printing everything ourselves with pctg, we also need to make sure it still fits inside the landing
funnel for the battery swapper, lets first create a shape design for that"). A layout of envelopes and bought parts, not engine
setups: what sits where, how big the printed frame is, and that it seats in the shared funnel. Drone frame = the seat frame of
tools/seat_envelope.py (x nose, y left, z up; the receiver's bottom at z -44.3 on the funnel's ledge).

What sets the shape, and only this:
  props    Gemfan Hurricane SL 5125, 5.1 in (O129.4, r 64.7; systems/dimensions.md A, Spark 8 Nov) on the same EMAX ECO II 2004
           motors, hubs at (+-74.3, +-74.3): the 3520's 19.2 tip gap kept between neighbours (2 x 74.3 - 129.4), wheelbase 210
  frame    printed PCTG, no bought plate: a deck where the Bee35 was (underside z 18.5, the units bolt up into it as they did into
           the plate), 6 thick (E, until the engine sizes it), four arms out to motor pods at the hubs; the motors on the pods, the
           prop plane DECK_TOP + 19.2 (the 2004's listing height, as on v3)
  battery  unchanged, so the swapper takes the v3 pack: the Overseer receiver v3 under the deck, the cartridge v6 and the 4S 1500
           in it, at the v3 place in the seat frame (x -51.3..54.3, |y| 32.8, z -44.3..9.6)
  funnel   the shared funnel seats the Scout on its receiver alone (9 Nov, Spark: "seat the scout on the receiver alone"; the box
           session moves the funnel's end walls in): below z -4.3 nothing outside the 55 deg faces rising from the receiver's outline
           (zmin_walls_in + 2). So no runners or keel: the base is the yokes over the receiver, a nose bay (the FC), the flow sensor's
           pod, a tail tray (Radxa, UBEC, XR2) and the nose stub; four TPU feet on the receiver's bottom corners are the foot on flat
           ground and four bare pads meet the box's keep-alive probes. Gate: tools/seat_check.py scout_v4 --descent (walls in)
  camera   ONE camera (9 Nov, Spark: the probe is the front camera too; then colour, "we need colored camera": the Arducam 1080P
           Low Light USB camera, IMX291, 100 deg lens, B0200; the 160 deg fisheye B0202 dropped, "we can take our time turning"):
           1920 x 1080 colour, MJPG 30 fps over USB 2.0 (UVC: the Radxa Zero 3W's Linux takes it with no driver), rolling shutter
           (the scan holds still for each frame), 100 deg diagonal M12 lens (~92 x 61, E), IR-cut filter, white LEDs; board 38 x 38,
           holes 28 x 28 (listing), M2 like the DECXIN's posts. On an SG90 tilt servo, parked on the
           probe's throat, it is the front camera 15 deg up (-15.5..+45.5, as low as the old view; clear from 10 deg), its frame clear of every rod and disc; rolled out it is the ceiling probe
  probe    the mast is two 10 mm steel tape blades with their curves facing (a lens-shaped tube, the spacecraft bi-STEM idea: each
           braces the other's easy fold), 0.3 m out; both blades and the 8-way flat cable between them wind on one spool (axis y,
           O34) under the roof where the FC sat; an N20 300 rpm with encoder lies in a pocket in the deck, under the props' keep-out
           band, and turns the spool through a printed 15 / 40 T pair (module 0.8); the flat cable reaches the fixed side through a
           300 mm clock-spring spiral in a drum beside the spool (no slip ring); the blades leave the spool's +x side through a 30
           mm throat that forms the lens, up beside the roof's middle node (a ring, r 22..34), the mast's axis at x 17.5
  flow     MTF-02P in the nose runner's window looking down, as v3 (70, 0, -24.5)
  tail     Radxa Zero 3W, UBEC, XR2 in the tail unit's bay over the tail runner, as v3; the FC on top of the deck at the centre
  cage     a half cage of 3.0 carbon rods only (Spark 8 Nov: keep the 3 mm, the cage's structure first, nodes later). Every rod
           12 from the swept discs (r 64.7 + 4, z prop plane +- 4): the octagon ring tangent to that keep-out (flats at |x|, |y|
           155, chamfers on |x| + |y| = 262.7) at z 61 (13 over the sweep); a FLAT roof in the ring's plane: a spine on y 0 and a
           hoop on x 0 (never over a disc) with the nose and tail triangles from RF / RB at x +-130 (at x 90 they crossed the front
           discs); a V up to the ring at each end and a fan at each side from four bottom nodes at z 12 in the flats' planes (every
           point there 12 from the discs). How the cage holds the core, with no long printed outriggers (8 Nov, Spark: "the solids
           that extend out do not seem optimal at all"): rods rising past the props can only stand in the flats' planes, and rods
           from the body out to them only in the strips between the discs (x 0 and y 0), so each side node S0 / S1 hangs on a rod
           triangle on x 0 to two sockets on a small lug on the deck's side (0, +-40, z 22 and -10, beside the receiver), the tail
           node TB on a triangle on y 0 to the deck's tail end (-60, 0, 21) and the tail runner's end (-90.5, 0, -2), and the nose
           node NB sits on a stub of the base out to the front flat (a rod triangle there put one rod at 0.84 x Euler). Why (frame model, frame_check): on ties under the arm tips (v21's way) the 35 g load went through level rods in
           bending (31 mm, 2 rods over Euler); a ridge 12 over the ring took the upside-down hit alone (1.3 x Euler); the four
           printed outriggers of the first try passed (1.0 mm) but put 115 mm spars and a tail boom on the frame, modelled rigid.
           Bottom open, the Scout lands on the runners
Writes parts/scout_v4_<part>/part.stl (one folder per v4 part, build_site: a folder belongs to one drone), systems/scout_v4.assembly.json
and prints the checks. Run: python tools/scout_v4_shape.py
"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# ---- props, motors, frame ---------------------------------------------------------------------------------------------------
PROP_D, HUB = 129.4, 74.3                       # Gemfan 5125 (dimensions.md A), hubs at (+-HUB, +-HUB)
PROP_R = PROP_D / 2
MOTORS = [(HUB, -HUB), (-HUB, HUB), (HUB, HUB), (-HUB, -HUB)]   # PX4 quad X: motor 1 front right, 2 rear left, 3 front left, 4 rear right
DECK_Z0, DECK_T = 18.5, 6.0                     # the deck's underside where the Bee35's was; 6 thick (E)
DECK_TOP = DECK_Z0 + DECK_T
MOTOR_H = 19.2                                   # ECO II 2004: arm top to the prop hub face (listing, E; dimensions.md C)
Z_PROP = DECK_TOP + MOTOR_H                     # 43.7
R_KEEP, Z_SWEEP = PROP_R + 4.0, 4.0             # the disc grown 4 and swept +-4 (the cage rule's disc, as v3)
KEEP = 12.0                                      # every rod 12 from the swept disc
DECK = (-60.0, 60.0, 40.0)                       # the deck's body: x0, x1, |y| (E), arms from its corners to the pods
ARM_W, POD_R = 18.0, 15.0                        # arm width, motor pod radius (the 2004's base O27 + 1.5)
# ---- the v3 base and the battery, unchanged ---------------------------------------------------------------------------------
SEAT_Z, CX, SEAT_L, SEAT_W, RUN_H, CH = -44.3, 1.5, 184.0, 66.8, 20.0, 4.0
# ---- camera, cage --------------------------------------------------------------------------------------------------------------
TILT = math.radians(15.0)                         # the parked default: 15 deg up (the 61 deg tall view clears the nose ring and RF from 10 deg; the servo tilts -20..+90)
HFOV, VFOV = math.radians(92.0 / 2), math.radians(61.0 / 2)   # Arducam 1080P Low Light USB (IMX291, 100 deg D M12 lens, B0200): ~92 x 61 deg on 16:9 (E from the diagonal)
CAM_BOARD = dict(c=(22.0, 104.0), side=38.0, t=6.0, lens_r=7.0, lens_l=14.0)   # its board (38 x 38, holes 28 x 28, listing; M12 holder and lens 14 E) on the tilt cradle, centre (x, z), facing forward TILT up
LENS = np.array([CAM_BOARD["c"][0] + (CAM_BOARD["t"] / 2 + CAM_BOARD["lens_l"]) * math.cos(TILT), 0.0, CAM_BOARD["c"][1] + (CAM_BOARD["t"] / 2 + CAM_BOARD["lens_l"]) * math.sin(TILT)])
# the probe's turret (9 Nov, Spark: "do the double blade ... make the design for the mast itself fully"): under the roof, where the FC sat
SPOOL = dict(c=(0.0, 43.0), r=17.0, y0=-7.0, y1=7.0)              # the spool: axis along y at x 0, z 43; hub O30, two blades + the flat cable wound to ~O33, flanges O34
GEAR = dict(spool_t=40, pinion_t=15, m=0.8, y=(7.0, 10.0))        # printed spur pair, module 0.8: 40 T on the spool's +y flange, 15 T on the motor; centres 22 apart
DRUM = dict(r=13.0, y0=-17.0, y1=-7.0)                           # the clock-spring drum on the spool's -y side (a 300 mm flat cable spiral, ~6 turns of room for ~4 used)
MOTOR = dict(x=6.0, y0=10.0, y1=46.0, z0=15.0, z1=27.0)          # N20 300 rpm with encoder, along y in a pocket in the deck, under the props' keep-out band (z <= 27.7)
THROAT = dict(x0=14.0, x1=21.0, y=7.0, z0=43.0, z1=73.0)          # the two blades leave the spool's +x side going up through a 30 mm guide that forms them into a lens
MAST_X = 17.5                                                     # the mast's axis (x, 0): beside the roof's middle node, inside its ring
CAM_F, CAM_U, CAM_R = np.array([math.cos(TILT), 0, math.sin(TILT)]), np.array([-math.sin(TILT), 0, math.cos(TILT)]), np.array([0, 1.0, 0])
FLAT = HUB + R_KEEP + KEEP                       # 155.0: the ring's flats
CH_LINE = 2 * HUB + (R_KEEP + KEEP) * math.sqrt(2)   # 262.7: the corner chamfers |x| + |y| = CH_LINE
ZR, ZB, XR = 61.0, 12.0, 130.0                    # ring and flat roof, the bottom nodes, the roof's nose / tail apex
ZR = max(ZR, Z_PROP + Z_SWEEP + 13.0)
YC = CH_LINE - FLAT                              # 107.7: where a flat meets a chamfer
NODES = {
    "R0": (FLAT, YC, ZR), "P1": (YC, FLAT, ZR), "PM0": (0.0, FLAT, ZR), "P2": (-YC, FLAT, ZR), "P3": (-FLAT, YC, ZR),       # the ring (left half, nose -> tail)
    "P4": (-FLAT, -YC, ZR), "P5": (-YC, -FLAT, ZR), "PM1": (0.0, -FLAT, ZR), "P6": (YC, -FLAT, ZR), "R7": (FLAT, -YC, ZR),   # right half
    "RF": (XR, 0, ZR), "RM": (0.0, 0, ZR), "RB": (-XR, 0, ZR),                                                              # the flat roof's spine
    "TB": (-FLAT, 0, ZB), "S0": (0.0, FLAT, ZB), "S1": (0.0, -FLAT, ZB),                                                    # the bottom nodes in the flats' planes
    "NB": (FLAT, 0, ZB),                                                                                                     # on the nose stub (PRINTED)
    "DS0": (0.0, 40.0, 22.0), "BS0": (0.0, 40.0, -10.0), "DS1": (0.0, -40.0, 22.0), "BS1": (0.0, -40.0, -10.0),            # sockets on the deck's side lugs (PRINTED)
    "DT": (-60.0, 0, 21.0), "BT": (CX - SEAT_L / 2, 0, 5.0),                                                                # sockets on the deck's tail end and the tail tray's end, its socket's bottom over the moved tail wall's top + 2 (PRINTED)
}
PRINTED = ("NB", "DS0", "BS0", "DS1", "BS1", "DT", "BT")
ROOTS = {}
RODS = [("R0", "P1"), ("P1", "PM0"), ("PM0", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5"), ("P5", "PM1"), ("PM1", "P6"), ("P6", "R7"), ("R7", "R0"),   # the ring
        ("RF", "R0"), ("RF", "R7"), ("RF", "RM"), ("RM", "RB"), ("RB", "P3"), ("RB", "P4"), ("RM", "PM0"), ("RM", "PM1"),                                 # the flat roof: nose and tail triangles, spine, hoop
        ("NB", "R0"), ("NB", "R7"), ("TB", "P3"), ("TB", "P4"),                                                                                            # the end Vs (the camera's window stays clear between them)
        ("S0", "P1"), ("S0", "PM0"), ("S0", "P2"), ("S1", "P6"), ("S1", "PM1"), ("S1", "P5"),                                                              # the side fans
        ("DS0", "S0"), ("BS0", "S0"), ("DS1", "S1"), ("BS1", "S1"), ("DT", "TB"), ("BT", "TB")]                                                            # the triangles that hold the bottom nodes, on x 0 and y 0
TIES = []
LUG = (8.0, 36.0, 44.0, -14.0)                   # the deck's side lugs: |x| 8, |y| 36..44 (3.2 outside the receiver), down to z -14 (over the funnel's y walls, -16.3)
ROD_G_MM, NODE_G = 0.011, 6.0                    # 3.0 carbon rod; a printed PCTG node (v21 run: 5..13 g, E)
PCTG = 1.23e-3                                   # g / mm3


def P(k):
    return np.array({**NODES, **ROOTS}[k], float)


def disc_dist(p):
    best = 1e9
    for mx, my in MOTORS:
        dr = max(0.0, math.hypot(p[0] - mx, p[1] - my) - R_KEEP)
        dz = max(0.0, (Z_PROP - Z_SWEEP) - p[2], p[2] - (Z_PROP + Z_SWEEP))
        best = min(best, math.hypot(dr, dz))
    return best


def in_view(p, lens=LENS):
    d = np.asarray(p, float) - lens; f, r, u = d @ CAM_F, d @ CAM_R, d @ CAM_U
    return f > 0 and abs(math.atan2(r, f)) < HFOV - math.radians(2) * 0 and abs(math.atan2(u, f)) < VFOV


def seg(a, b, n=80):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def box(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0]); m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]); return m


def cyl(x, y, z0, z1, r, n=48):
    m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=n); m.apply_translation([x, y, (z0 + z1) / 2]); return m


def rod(a, b, r=1.5):
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(a, float), np.asarray(b, float)], sections=12)


def union(ms):
    import manifold3d as mf
    out = None
    for m in ms:
        mm = mf.Manifold(mf.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))
        out = mm if out is None else out + mm
    t = out.to_mesh(); return trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=False)


def frame():
    """the printed deck: the body plate, four arms from its corners to the pods, the pods, a lug down each side for the side
    triangles' sockets (an envelope: the engine sizes it)"""
    x0, x1, yb = DECK
    parts = [box(x0, x1, -yb, yb, DECK_Z0, DECK_TOP)]
    for mx, my in MOTORS:
        a = np.array([np.sign(mx) * 30.0, np.sign(my) * 30.0]); b = np.array([mx, my]); d = b - a; L = float(np.linalg.norm(d))
        arm = box(-L / 2, L / 2, -ARM_W / 2, ARM_W / 2, DECK_Z0, DECK_TOP)
        T = np.eye(4); c, s = d / L
        T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = (a + b) / 2
        arm.apply_transform(T); parts.append(arm)
        parts.append(cyl(mx, my, DECK_Z0, DECK_TOP, POD_R))
    lx, ly0, ly1, lz = LUG
    for s in (1, -1):                                                # the side lugs: the two sockets of each side triangle
        parts.append(box(-lx, lx, *sorted((s * ly0, s * ly1)), lz, DECK_Z0 + 0.5))
    return union(parts)


def zmin_walls_in(x):
    """the funnel with its end walls in (the box session, 9 Nov; Spark: "seat the scout on the receiver alone"): below z -4.3 nothing
    outside the 55 deg faces that rise from the receiver's outline; the lowest z a part may reach at x (add 2 margin)"""
    d = abs(x - 1.45) - 52.8
    return SEAT_Z if d <= 0 else min(-4.3, SEAT_Z + 1.428 * d)


def base():
    """v4 base on the receiver-only seat (9 Nov): no runners, no keel. The yokes over the receiver (its top z 9.6 to the deck's
    underside 18.5, where the v3 units bolted it up), a nose bay for the FC (x 54.3..93, |y| 22, z -2.3..18.5), the flow sensor's pod
    under it (x 54.3..70.5: its bottom at -19.1, the rule's -21.1 + 2), a tail tray for the Radxa, UBEC and XR2 (x -90.5..-51.3, z -2.3
    up), the stub out to the cage's nose node NB; every bottom on zmin_walls_in + 2"""
    rec = trimesh.load(os.path.join(ROOT, "parts", "box_ctx_scout_receiver_v3", "part.stl"), force="mesh")
    lo, hi = rec.bounds
    parts = [box(lo[0], hi[0], -34.0, 34.0, hi[2], DECK_Z0),                         # the yokes over the receiver
             box(hi[0], 93.0, -22.0, 22.0, -2.3, DECK_Z0),                           # the nose bay
             box(hi[0], 70.5, -12.0, 12.0, -19.1, -2.3),                             # the flow sensor's pod (open below: its window)
             box(CX - SEAT_L / 2, lo[0], -34.0, 34.0, -2.3, DECK_Z0),                # the tail tray
             box(92.0, FLAT + 6.0, -6.0, 6.0, 4.0, 14.0)]                            # the stub out to NB
    return union(parts)


def place(src, rot=None, lo=None, yc=0.0):
    """a bought part's mesh, turned (axis, deg) about its centre, then moved so its bounds' low corner is lo (x, z) and centred on yc"""
    m = trimesh.load(os.path.join(ROOT, "parts", src, "part.stl"), force="mesh")
    if rot:
        c = m.bounds.mean(0); m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rot[1]), rot[0], c))
    b = m.bounds
    m.apply_translation([lo[0] - b[0][0], yc - (b[0][1] + b[1][1]) / 2, lo[1] - b[0][2]])
    return m


def pads():
    """on the receiver's bottom (the box session's numbers, 9 Nov): four TPU feet at its corners in |y| 21..23.5 (clear of the ledges'
    skirt at 25 and the cartridge's path), reaching z -47.5 so they are the foot on flat ground, and four bare contact pads O6 at
    (1.5 +- 30, +-29.5) for the box's keep-alive and seat probes in the ledges"""
    feet = [box(x0, x0 + 8.0, *sorted((s * 21.0, s * 23.5)), -47.5, SEAT_Z) for x0 in (-50.0, 45.0) for s in (1, -1)]
    contacts = [cyl(1.5 + dx, s * 29.5, SEAT_Z - 0.2, SEAT_Z, 3.0, 24) for dx in (-30.0, 30.0) for s in (1, -1)]
    return trimesh.util.concatenate(feet), trimesh.util.concatenate(contacts)


def prop(mx, my):
    """the swept disc (0.8 thick) and a 3 blade sketch, the hub"""
    disc = cyl(mx, my, Z_PROP - 0.4, Z_PROP + 0.4, PROP_R, 96)
    blades = []
    for k in range(3):
        a = math.radians(120 * k + (30 if mx * my > 0 else -30))
        b = box(0, PROP_R - 1, -6.0, 6.0, -1.2, 1.2); b.apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1])); b.apply_translation([mx, my, Z_PROP])
        blades.append(b)
    return disc, union(blades + [cyl(mx, my, Z_PROP - 3, Z_PROP + 3, 7.0)])


def moved(src, dx=0.0, dy=0.0, dz=0.0):
    m = trimesh.load(os.path.join(ROOT, "parts", src, "part.stl"), force="mesh"); m.apply_translation([dx, dy, dz]); return m


def probe():
    """the ceiling probe parked (9 Nov). Turret, under the roof's middle node where the FC sat: the spool (axis y, O34), its 40 T gear,
    the clock-spring drum, the N20 in a pocket in the deck driving the gear through a 15 T pinion, the 30 mm throat the two blades
    leave through (forming their lens) beside the node, and the housing round them. Head, on the blades' tip parked on the throat: the
    clip, the SG90 on its side, the cradle with the camera board (a TPU guard on its rim), the LEDs and the VL53L0X"""
    (cx, cz), r = SPOOL["c"], SPOOL["r"]
    spool = trimesh.creation.cylinder(radius=r, segment=[(cx, SPOOL["y0"], cz), (cx, SPOOL["y1"], cz)], sections=48)
    gear = trimesh.creation.cylinder(radius=GEAR["spool_t"] * GEAR["m"] / 2 + GEAR["m"], segment=[(cx, GEAR["y"][0], cz), (cx, GEAR["y"][1], cz)], sections=48)
    pz = cz - (GEAR["spool_t"] + GEAR["pinion_t"]) * GEAR["m"] / 2
    pinion = trimesh.creation.cylinder(radius=GEAR["pinion_t"] * GEAR["m"] / 2 + GEAR["m"], segment=[(cx, GEAR["y"][0], pz), (cx, GEAR["y"][1], pz)], sections=24)
    drum = trimesh.creation.cylinder(radius=DRUM["r"], segment=[(cx, DRUM["y0"], cz), (cx, DRUM["y1"], cz)], sections=48)
    motor = box(-MOTOR["x"], MOTOR["x"], MOTOR["y0"], MOTOR["y1"], MOTOR["z0"], MOTOR["z1"])
    t = THROAT
    throat = box(t["x0"], t["x1"], -t["y"], t["y"], t["z0"], t["z1"])
    turret = trimesh.util.concatenate([spool, gear, pinion, drum, motor, throat])
    hz = t["z1"]
    cb = CAM_BOARD; (bx, bz) = cb["c"]
    board = box(-cb["t"] / 2, cb["t"] / 2, -cb["side"] / 2, cb["side"] / 2, -cb["side"] / 2, cb["side"] / 2)
    lens = trimesh.creation.cylinder(radius=cb["lens_r"], segment=[(cb["t"] / 2, 0, 0), (cb["t"] / 2 + cb["lens_l"], 0, 0)], sections=32)
    cam = trimesh.util.concatenate([board, lens]); cam.apply_transform(trimesh.transformations.rotation_matrix(-TILT, [0, 1, 0])); cam.apply_translation([bx, 0, bz])
    head = [box(t["x0"], t["x1"] + 1.0, -6.0, 6.0, hz, hz + 3.0),                   # the clip on the two blades' tip
            box(8.0, 30.0, -12.0, 12.0, hz + 3.0, hz + 15.0),                         # SG90 on its side, shaft along y
            box(10.0, 26.0, -20.0, 20.0, hz + 15.0, bz - 14.0),                       # the cradle up to the board
            cam,                                                                      # the camera board facing forward, TILT up
            box(bx + 2.0, bx + 6.0, 20.0, 26.0, bz - 4.0, bz + 4.0)]                  # LEDs and the VL53L0X beside the board
    return turret, trimesh.util.concatenate(head)


MAST_STUB = 1.0                                   # the mast's mesh: 1 mm of the two blades above the throat; the page's timeline
                                                  # (tools/scout_v4_timeline.py) stretches it to the length rolled out and lifts the head


def mast():
    """the two blades as they leave the throat, curves facing (a lens 10 wide, each blade bowed 2 out: thickness drawn 0.4 to be seen)"""
    from shapely.geometry import Polygon
    xs = np.linspace(-5.0, 5.0, 17); bow = 2.0 * (1 - (xs / 5.0) ** 2)
    out = []
    for sg in (1, -1):
        ring = [(MAST_X + x, sg * y) for x, y in zip(xs, bow)] + [(MAST_X + x, sg * (y - 0.4)) for x, y in zip(xs[::-1], bow[::-1])]
        m = trimesh.creation.extrude_polygon(Polygon(ring).buffer(0), MAST_STUB); m.apply_translation([0, 0, THROAT["z1"]])
        out.append(m)
    return out


HUB_RING = 34.0                                   # the roof's middle node RM is a ring (r 22..34) round the probe's throat: its four rods end in its sides at r 34


def cage():
    def cut(i, j):                           # each rod stops 3 inside its nodes' spheres (two in-line rods through a node must not share an end
        a, b = P(i), P(j); u = (b - a) / np.linalg.norm(b - a)   # ring: the page's mesh cleanup welds them and drops the body, 7 Nov), and at
        bi = HUB_RING if i == "RM" else 3.0                      # the ring's side at RM
        bj = HUB_RING if j == "RM" else 3.0
        return a + u * bi, b - u * bj
    rods = [rod(*cut(i, j)) for i, j in RODS + TIES]
    nodes = [trimesh.creation.icosphere(subdivisions=1, radius=5.0).apply_translation(P(k)) for k in NODES if k not in PRINTED and k != "RM"]
    nodes += [trimesh.creation.icosphere(subdivisions=1, radius=7.0).apply_translation(P(k)) for k in PRINTED]   # the sockets printed into the frame and the base
    ring = trimesh.creation.annulus(r_min=22.0, r_max=HUB_RING, height=8.0); ring.apply_translation(P("RM"))
    nodes.append(ring)                                                                   # RM: the ring node the turret's throat passes through
    return trimesh.util.concatenate(rods), trimesh.util.concatenate(nodes)


def frame_check(core_g):
    """the Scout's frame model (tools/scout_cage_loads.py, the standard cases: 35 g crash on the hull face the CG ray leaves through, 14
    directions, landing and rough landing 6 g on the runners) on these tables: the core one rigid body at the origin with the
    sockets printed into the frame and the base and the runner corners stiff to it"""
    import io, contextlib, scout_cage as sc, scout_cage_loads as sl
    sc.NODES, sc.RODS, sc.TIES, sc.TIE_ROOTS = dict(NODES), list(RODS), list(TIES), dict(ROOTS)
    sc.PRINTED = {k: "frame" for k in PRINTED}
    sc.BASE = {"G0": (CX + SEAT_L / 2, SEAT_W / 2, SEAT_Z), "G1": (CX - SEAT_L / 2, SEAT_W / 2, SEAT_Z), "G2": (CX - SEAT_L / 2, -SEAT_W / 2, SEAT_Z), "G3": (CX + SEAT_L / 2, -SEAT_W / 2, SEAT_Z)}
    sc.BRIDGES = []; sc.GROUPS = sc._groups(); sc.GROUP_OF = {k: g for g in sc.GROUPS for k in g}; sc.RODS_CARBON = list(RODS)
    sl.core_mass = lambda: (core_g, np.zeros(3))
    with contextlib.redirect_stdout(io.StringIO()):
        s_, rec = sl.run(nodes=sc.NODES, rods=sc.RODS, ties=sc.TIES, dry=True, log=lambda *a: print(*a))
    w = max(rec["cases"], key=lambda r: r["deflection_mm"])
    eu = max(rec["rods"].items(), key=lambda kv: kv[1]["euler"]); mo = max(rec["rods"].items(), key=lambda kv: kv[1]["moment"])
    return dict(mode_hz=rec["modes_hz"][0], deflection=(w["deflection_mm"], w["name"]), euler=(eu[1]["euler"], eu[0]), moment=(mo[1]["moment"], mo[0]),
                pull=max(v["tension"] for v in rec["rods"].values()))


def checks(meshes):
    out = {}
    worst = (1e9, "")
    for nm, m in meshes.items():
        if nm in ("box_ctx_scout_receiver_v3", "box_ctx_scout_cartridge_v6", "pack_4s_1500", "pack_xt60_plug", "receiver_feet", "receiver_contacts") or nm.startswith(("prop_", "blades_")):
            continue
        v = np.asarray(m.vertices); v = v[v[:, 2] < -4.3]
        for p in v[::3]:
            d = p[2] - (zmin_walls_in(p[0]) + 2.0)
            if d < worst[0]:
                worst = (round(float(d), 1), f"{nm} at x {p[0]:.0f}, z {p[2]:.1f}")
    out["walls-in funnel"] = (("clear by %s" % worst[0]) if worst[0] >= 0 else f"CROSSES by {-worst[0]}", worst[1] or "nothing below z -4.3")
    gaps = [math.dist(MOTORS[a], MOTORS[b]) - PROP_D for a in range(4) for b in range(a + 1, 4)]
    out["tip gap"] = (round(min(gaps), 1), "neighbour props, tip to tip (3520 on v3: 19.1)")
    worst = (1e9, "")
    for i, j in RODS + TIES:
        pts = seg(P(i), P(j))
        d = min(disc_dist(p) for p in pts)
        if d < worst[0]:
            worst = (round(d, 1), f"{i}-{j}")
    out["rods to discs"] = (worst[0], f"{worst[1]} (need {KEEP:.0f})")
    blk = [f"{i}-{j}" for i, j in RODS + TIES if any(in_view(p) for p in seg(P(i), P(j)))]
    out["camera view"] = ("clear" if not blk else "BLOCKED by " + ", ".join(blk), f"the probe head parked: {math.degrees(2 * HFOV):.0f} x {math.degrees(2 * VFOV):.0f} deg from lens ({LENS[0]:.0f}, 0, {LENS[2]:.0f}), {math.degrees(TILT):.0f} deg up")
    disc_pts = [np.array([mx + PROP_R * math.cos(a), my + PROP_R * math.sin(a), Z_PROP]) for mx, my in MOTORS for a in np.linspace(0, 2 * math.pi, 90)]
    out["props in view"] = ("none" if not any(in_view(p) for p in disc_pts) else "IN VIEW", "the disc rims")
    for nm in ("probe_head", "probe_turret"):
        v = np.asarray(meshes[nm].vertices)
        out[f"{nm} to discs"] = (round(min(disc_dist(p) for p in v), 1), "need 12")
    for nm in ("frame", "base"):
        v = np.asarray(meshes[nm].vertices); top = v[:, 2].max()
        out[f"{nm} to discs"] = (round(min(disc_dist(p) for p in v if p[2] > DECK_Z0 - 7), 1), f"top z {top:.1f}; the props sweep down to {Z_PROP - Z_SWEEP:.1f}")
    return out


def write():
    meshes = {}
    meshes["frame"] = frame()
    meshes["base"] = base()
    for k, (mx, my) in enumerate(MOTORS, 1):
        sx, sy = np.sign(mx), np.sign(my)
        meshes[f"motor_{k}"] = moved("scout_motor_" + str(k), sx * (HUB - 54.1), sy * (HUB - 54.1), DECK_TOP - 22.0)
        meshes[f"prop_{k}"], meshes[f"blades_{k}"] = prop(mx, my)
    meshes["probe_turret"], meshes["probe_head"] = probe()
    meshes["probe_mast_a"], meshes["probe_mast_b"] = mast()
    meshes["fc"] = place("fc", lo=(56.0, 1.0))                                              # the FC in the nose bay (its place on the deck is the turret's)
    meshes["scout_flow_mtf_02p"] = place("scout_flow_mtf_02p", rot=([0, 0, 1], 90), lo=(54.5, -19.1))   # turned 90 deg, in the pod: outer edge x 70.5
    meshes["ubec"] = place("ubec", lo=(-90.0, -1.3), yc=-2.5)                                # the tail tray, flat
    meshes["radxa_zero_3w"] = place("radxa_zero_3w", rot=([0, 1, 0], 90), lo=(-86.0, 8.5))   # flat over the UBEC (was on edge)
    meshes["xr2"] = place("xr2", rot=([0, 1, 0], 90), lo=(-70.0, 0.0))
    for src in ("pack_4s_1500", "pack_xt60_plug", "box_ctx_scout_receiver_v3", "box_ctx_scout_cartridge_v6"):
        meshes[src] = moved(src)
    meshes["receiver_feet"], meshes["receiver_contacts"] = pads()
    meshes["cage_rods"], meshes["cage_nodes"] = cage()
    for nm, m in meshes.items():
        d = os.path.join(ROOT, "parts", f"scout_v4_{nm.replace('box_ctx_scout_', '').replace('scout_', '')}"); os.makedirs(d, exist_ok=True)
        m.export(os.path.join(d, "part.stl"))
    return meshes


def entries(meshes):
    fid = lambda nm: f"scout_v4_{nm.replace('box_ctx_scout_', '').replace('scout_', '')}"
    rows = []
    def add(nm, name, role, group, color, mass, tag, note, material=None, opacity=None):
        e = dict(id=fid(nm), name=name, role=role, group=group, color=color, mesh=f"parts/{fid(nm)}/part.stl", mass_g=round(mass, 1), tag=tag, owner="scout v4", version="shape 8 Nov", source="tools/scout_v4_shape.py", note=note)
        if material:
            e["material"] = material
        if opacity is not None:
            e["opacity"] = opacity
        rows.append(e)
    fv = abs(meshes["frame"].volume); bv = abs(meshes["base"].volume)
    add("frame", "frame v4 room (printed PCTG deck, arms and motor pods: the design envelope)", "placeholder", "frame", "#d9d2c3", fv * PCTG * 0.45, "E",
        f"replaces the Bee35 plate: deck x {DECK[0]:.0f}..{DECK[1]:.0f}, |y| {DECK[2]:.0f}, z {DECK_Z0}..{DECK_TOP} (the plate's underside kept, the units bolt up into it), arms {ARM_W:.0f} wide to O{2 * POD_R:.0f} pods at (+-{HUB}, +-{HUB}), a lug down each side to z {LUG[3]:.0f} for the cage's side triangles, a socket at the tail end; mass at 45 % of the envelope (E, the engine sizes it); printed in PCTG", None, 0.85)
    add("base", "base v4 room (yokes over the receiver, nose bay, flow pod, tail tray, nose stub)", "placeholder", "base", "#c9c2b5", bv * PCTG * 0.30, "E",
        "the receiver-only seat (9 Nov, Spark: \"seat the scout on the receiver alone\"; the funnel's end walls move in for the Scout): no runners or keel. Yokes over the receiver (its top z 9.6 to the deck), the nose bay for the FC, the flow sensor's pod, the tail tray for the Radxa, UBEC and XR2, the stub out to NB; every bottom on the walls-in rule + 2 (zmin_walls_in); mass at 30 % of the envelope (E); printed in PCTG", None, 0.55)
    for k in range(1, 5):
        add(f"motor_{k}", f"motor {k} (EMAX ECO II 2004)", "bought", "motors", "#8b8f96", 18.0, "listing", f"on its pod at (+-{HUB}, +-{HUB}), base z {DECK_TOP}")
        add(f"prop_{k}", f"prop {k} disc (Gemfan 5125, O{PROP_D}, swept)", "bought", "props", "#9fd3ff", 0.0, "listing", f"prop plane z {Z_PROP:.1f}", opacity=0.18)
        add(f"blades_{k}", f"prop {k} (Gemfan Hurricane SL 5125, 3 blade)", "bought", "props", "#3b6ea8", 2.8, "listing", "a sketch of the three blades")
    add("probe_head", "front camera and probe head (Arducam 1080P Low Light colour USB, IMX291 100 deg, on an SG90 tilt servo, white LEDs, VL53L0X)", "bought", "camera", "#1d5e3a", 28.0, "E",
        f"parked on the throat it is the front camera, lens ({LENS[0]:.0f}, 0, {LENS[2]:.0f}), {math.degrees(TILT):.0f} deg up, 1920 x 1080 colour at 30 fps, ~92 x 61 deg; rolled out on the two blades it is the ceiling probe. Camera board and lens 14 E, SG90 9, VL53L0X 0.7, LEDs 0.3, cradle and clip 2.5, TPU rim guard 1.5")
    add("probe_turret", "ceiling probe turret (two-blade 0.3 m mast, spool, gears, clock-spring drum, N20 in the deck, throat)", "placeholder", "camera", "#f2b705", 36.4, "E",
        "two 10 mm steel blades 0.4 m 7.6, flat cable 2 x 300 mm 2, spool with its 40 T gear 5, clock-spring drum 2, housing and throat 6 (PCTG), N20 300 rpm with encoder 10.5, 15 T pinion 0.3, DRV8833 1, TPU lip 1, wires 1 (E)")
    for k, col in (("a", "#e8c547"), ("b", "#b8902a")):
        add(f"probe_mast_{k}", f"probe mast, blade {k.upper()} (10 mm steel tape, curves facing)", "placeholder", "camera", col, 0.0, "E",
            "drawn 1 mm long above the throat; the page's timeline rolls it out to 300 mm and back (tools/scout_v4_timeline.py). Its mass is in the turret's")
    add("fc", "FC MicoAir743 AIO (in the nose bay)", "bought", "electronics", "#1f6f3f", 10.0, "P", "36 x 36 at x 56..92 under the deck (its place at the deck's centre is the turret's); set its IMU offset in PX4 (EKF2_IMU_POS_X ~ +0.074 m)")
    for src, nm, g, note in (("ubec", "UBEC", 15.0, "the tail tray, flat, bottom z -1.3"), ("xr2", "XR2", 1.0, "the tail tray, flat"), ("radxa_zero_3w", "Radxa Zero 3W", 10.0, "the tail tray, flat over the UBEC (was on edge)"),
                             ("scout_flow_mtf_02p", "flow MTF-02P", 1.5, "the nose pod, turned 90 deg (outer edge x 70.5, bottom -19.1): set SENS_FLOW_ROT for the turn")):
        add(src, nm, "bought", "electronics", "#2b5d8a", g, "P", note)
    add("pack_4s_1500", "pack 4S 1500 (in the cartridge)", "protected", "battery", "#c0392b", 165.0, "P", "the v3 pack, unchanged: the swapper's")
    add("pack_xt60_plug", "pack XT60 plug", "bought", "battery", "#e2b007", 10.0, "P", "as v3")
    add("box_ctx_scout_receiver_v3", "receiver v3 (the Overseer's latch receiver)", "structure", "battery", "#7f8c8d", 60.0, "E", "unchanged, at the v3 place: the funnel seat and the swap arm see the same belly")
    add("box_ctx_scout_cartridge_v6", "cartridge v6", "structure", "battery", "#95a5a6", 30.0, "E", "unchanged")
    add("receiver_feet", "receiver TPU feet (4, at its bottom corners)", "structure", "battery", "#333333", 1.0, "E", "8 x 2.5 at |y| 21..23.5, to z -47.5: the Scout's foot on flat ground, clear of the box's ledges and swap head (the box session, 9 Nov); TPU 95A")
    add("receiver_contacts", "receiver contact pads (4, keep-alive and seat sensing)", "structure", "battery", "#c48a3a", 0.5, "E", "bare O6 pads at (1.5 +- 30, +-29.5) on the receiver's bottom rim, against the box's probes in the ledges")
    L = sum(float(np.linalg.norm(P(i) - P(j))) for i, j in RODS + TIES)
    add("cage_rods", "half cage v4, rods (3 mm carbon: the ring, the flat roof, end Vs and side fans)", "cage", "cage", "#2b2b2b", L * ROD_G_MM, "E",
        f"{len(RODS)} rods, {L / 1000:.2f} m, every one 12 from the swept discs and none over or under a disc: ring at z {ZR:.0f} with flats at {FLAT:.0f} (310 across), a flat roof (spine on y 0, hoop on x 0, nose / tail triangles), a V at each end and a fan at each side down to the bottom nodes, which hang on rod triangles to the deck's side lugs and tail (NB on the camera's mount)")
    add("cage_nodes", "half cage v4, nodes (spheres for now: the node design comes once the geometry is settled)", "cage", "cage", "#eef2f5", (len(NODES) - len(PRINTED)) * NODE_G, "E",
        f"{len(NODES) - len(PRINTED)} cage nodes; the larger spheres are sockets printed into the frame and the base (NB on the nose stub, two on each side lug, the deck's tail end, the tail tray's end)", opacity=0.9)
    return rows


if __name__ == "__main__":
    M = write()
    L = sum(float(np.linalg.norm(P(i) - P(j))) for i, j in RODS)
    print(f"Scout v4 shape: props {PROP_D} on hubs +-{HUB} (wheelbase {2 * HUB * math.sqrt(2):.0f}), prop plane z {Z_PROP:.1f}, deck z {DECK_Z0}..{DECK_TOP}, ring flats {FLAT:.1f}, chamfer {CH_LINE:.1f}, ring and roof z {ZR:.1f}; "
          f"cage {len(NODES) - len(PRINTED)} nodes + {len(PRINTED)} printed, {len(RODS)} rods {L / 1000:.2f} m")
    for k, (v, why) in checks(M).items():
        print(f"  {k:16} {v}  ({why})")
    core = sum(r["mass_g"] for r in entries(M) if "cage" not in r["id"])
    fc = frame_check(core)
    print(f"  frame model     first mode {fc['mode_hz']:.0f} Hz, worst {fc['deflection'][0]:.1f} mm ({fc['deflection'][1][:60]}), Euler <= {fc['euler'][0]:.2f} ({fc['euler'][1]}), "
          f"end moment <= {fc['moment'][0]:.0f} N mm ({fc['moment'][1]}; socket cap 2000), pull-out <= {fc['pull']:.0f} N (cap 940)")
    F = os.path.join(ROOT, "systems", "scout_v4.assembly.json")
    old = json.load(open(F)) if os.path.exists(F) else {}
    rows = entries(M)
    lo = np.min([m.bounds[0] for m in M.values()], 0); hi = np.max([m.bounds[1] for m in M.values()], 0)
    mass = sum(r["mass_g"] for r in rows)
    a = dict(name="Scout v4", title="Scout v4", artifact=old.get("artifact"), machine="scout_v4", order=0, readme_view=old.get("readme_view", "iso"),
             appendix=old.get("appendix", "systems/scout_v4.appendix.html"), timeline="tools/scout_v4_timeline.py:timeline", frame_timeline=True,
             note=f"Scout v4 shape design (8 Nov, Spark: new props, no Bee35, everything printed in PCTG, still seats in the landing funnel for the battery swapper): "
                  f"Gemfan 5125 props on a printed frame, the v3 base and battery unchanged, a half cage on the new discs. {lo[0]:.0f}..{hi[0]:.0f} x {lo[1]:.0f}..{hi[1]:.0f} x {lo[2]:.0f}..{hi[2]:.0f} mm, "
                  f"~{mass:.0f} g (E). tools/scout_v4_shape.py",
             mass_budget_g=None, parts=rows)
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + "\n")
    print(f"  extents         {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} x {hi[2] - lo[2]:.0f} mm; all-up ~{mass:.0f} g (E)")
    for r in sorted(rows, key=lambda r: -r["mass_g"])[:8]:
        print(f"    {r['mass_g']:6.1f} g  {r['name'][:70]}")
    print("  written: systems/scout_v4.assembly.json, parts/scout_v4_*/part.stl")
