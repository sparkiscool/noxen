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
           (zmin_walls_in + 2). So no runners or keel: the base (one body with the deck, 9 Nov) is the yokes over the receiver (the FC,
           the Radxa, the XR2, the DRV8833 under the deck), a nose block and the flow sensor's pod, a tail tray (the UBEC); four TPU feet on the receiver's bottom corners are the foot on flat
           ground and four bare pads meet the box's keep-alive probes. Gate: tools/seat_check.py scout_v4 --descent (walls in)
  camera   ONE camera (9 Nov, Spark: the probe is the front camera too, colour; bought: the Arducam 12MP autofocus USB camera,
           B0FLX58WKZ): 1/2.43" colour, IR-cut, AF 3.95 mm f/1.8 focusing 10 cm to inf (settable over UVC), 75 D / 61 H / 50 V deg,
           MJPG 4608 x 2592 at 15 fps, 1080p at 30, USB 2.0 UVC (the Radxa Zero 3W's Linux takes it with no driver); 45 g in its
           metal case, flown without it (the bare board ~32 square and ~12 g: E, measure on arrival). On an SG90 tilt servo, parked
           on the probe's throat, it is the front camera, its frame clear of every rod and disc; rolled out it is the ceiling probe
  probe    the mast is two 19 mm steel tape blades with their curves facing (a lens-shaped tube, the spacecraft bi-STEM idea: each
           braces the other's easy fold), 0.15 m out (Spark 9 Nov: as short as is safe; was 0.3), 19 mm blades (Spark 9 Nov: "as safe as possible", was 10); both blades and the 8-way flat cable between them wind on one spool (axis y,
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
FC_X, FC_Z0, UBEC_YC = (15.5, 51.5), 10.0, 0.0       # 9 Nov: the FC hangs under the deck over the receiver (x 15.5..51.5, z 10..20.5); the UBEC across the tail tray
BED = 89.3                                       # 9 Nov (Spark: deck, arms, pods and base as ONE body): the body inside |x|, |y| <= 89.3, the A1 mini's 180 less 1.4
# ---- the v3 base and the battery, unchanged ---------------------------------------------------------------------------------
BX = 1.3                                          # 9 Nov (Spark: "the center of gravity ... perfectly centered"): everything under the deck (receiver, cartridge, pack, base,
                                                  # FC, flow, tail electronics) moved along x as one group, to put the Scout's CG on the thrust centre; the funnel moves with the receiver
SEAT_Z, CX, SEAT_L, SEAT_W, RUN_H, CH = -44.3, 1.5 + BX, 184.0, 66.8, 20.0, 4.0
# ---- camera, cage --------------------------------------------------------------------------------------------------------------
TILT = math.radians(15.0)                         # the parked default: 15 deg up (the 61 deg tall view clears the nose ring and RF from 10 deg; the servo tilts -20..+90)
HFOV, VFOV = math.radians(61.0 / 2), math.radians(50.0 / 2)   # Arducam 12MP autofocus USB (B0FLX58WKZ): 75 D / 61 H / 50 V deg (listing)
PROBE_DX = 0.0                                                    # the whole probe (turret, mast, head) and RM moved along x to put the Scout's CG on the thrust centre
MAST_X = 15.1 + PROBE_DX                                          # the mast's axis (x, 0): the stack's tangent off the wound spool (r 14..15.4, mean + half the stack), inside the roof's ring node
REACH = 150.0                                                     # the mast's roll-out (9 Nov, Spark: "shorten the distance where it's as safe as possible"; was 300)
# the head (9 Nov, Spark: "show how it can nod ... a led light for the camera, build out the actual mechanism"): a clip on the blades' tip,
# a base plate, an SG90 on its side on the +y bracket driving the tilt cradle through its horn, an idler pin in the -y bracket; the cradle
# carries the camera board on four M2 posts, the lens hood, the VL53L0X over the board and two 5 V COB LED bars beside the lens
HEAD = dict(clip=14.0, plate=3.0, dx=-7.5, ay=-3.8, az=45.0, range=(-20.0, 90.0))   # tilt axis along y at x MAST_X + dx, y ay, z = throat top + az (+ the length out); dx, ay put the head's CG over the mast
CAM = dict(case=(1.5, 21.5), side=36.0, lens=(21.5, 23.0, 6.0))   # the camera IN its metal case (Spark 9 Nov: kept on, the heat spreader; bracket off): 36 square (listing), depth 20 E: measure
LENS_OFF = 23.0                                                   # the lens window ahead of the tilt axis (case depth E)
# the probe's turret (9 Nov, Spark: "do the double blade ... make the design for the mast itself fully"): under the roof, where the FC sat
BLADE_W = 19.0                                                    # tape blade width (9 Nov, Spark: "as safe as possible", 3/4 in; was 10): ~3.5x the fold moment (E)
SPOOL = dict(c=(PROBE_DX, 41.0), r=15.9, hub=14.0, y0=-11.7, y1=11.7)  # 9 Nov, the mast sims (tools/probe_mast_sim.py): hub O28 (was O20: von Mises 1574 MPa at r 10 opposite sense, 1283 at r 14), 1.6 turns of the stack (A, the 0.6 ribbon, B) to O30.8, O31.8 flanges, 23.4 wide; axis y at x 0, z 41 (its flanges 0.6 over the deck)
ROLLER = dict(ang=-30.0, r=5.05, gap=0.0, w=19.0, F=(12.0, 7.8))  # the coil's pressure roller 30 deg before the exit (the sims: with the mast pushed, nothing but a friction lock holds the outer turn;
                                                  # 8 N locks a 2.4 N push at mu 0.3): O10.1 printed tube on two MR63ZZ (10 Nov: O8 left 1.0 round their O6.05 seats; 2.0 now), an M3 axle in radial slots, two 4 x 0.5 x 15 springs, 12 N on the full coil, 7.8 empty
GEAR = dict(spool_t=40, pinion_t=12, m=0.8, y=(11.7, 14.7))      # printed spur pair, module 0.8, 40 / 12 on 20.8 centres: the N20 sits under the spool (9 Nov: outboard along +y it ran through the DS0 lug socket); ~1.5 s for the 150 mm, 3.3x its torque
N20_FACE = GEAR["y"][0] - 2.9                                    # the N20's gearbox face (y): 2.9 short of the pinion, the turret's 2.6 face plate between (9 Nov audit: 1.6 was too thin)
DRUM = dict(r=9.0, y0=-17.7, y1=-11.7)   # 10 Nov turret run 6: r 11 put the drum's ring in the head-first path (0.34); the 0.6 ribbon's 3.5 turns of room need r ~8.2                        # the drum on the spool's -y side: the camera cable's loose coil, 3.5 turns of room for the spool's ~2.2
MOTOR = dict(x=6.0, y0=-17.3, y1=9.7, z0=15.2, z1=25.2)           # N20 6 V 100 rpm (B07MZH851L; the 300 rpm stalled at ~140 N mm on the spool, the coil's own unwind is ~100 + friction): under the spool and the drum, facing +y, in a pocket through the deck
THROAT = dict(x0=MAST_X - 4.0, x1=MAST_X + 4.0, y=11.0, z0=41.0, z1=73.0)   # the 32 mm guide from the coil's tangent: the stack re-forms into the lens in it (tools/probe_mast_fe.py) and it grips the mast's base
AXIS = np.array([MAST_X + HEAD["dx"], HEAD["ay"], THROAT["z1"] + HEAD["az"]])   # the tilt axis parked
LENS = AXIS + LENS_OFF * np.array([math.cos(TILT), 0.0, math.sin(TILT)])
CAM_F, CAM_U, CAM_R = np.array([math.cos(TILT), 0, math.sin(TILT)]), np.array([-math.sin(TILT), 0, math.cos(TILT)]), np.array([0, 1.0, 0])
FLAT = HUB + R_KEEP + KEEP                       # 155.0: the ring's flats
CH_LINE = 2 * HUB + (R_KEEP + KEEP) * math.sqrt(2)   # 262.7: the corner chamfers |x| + |y| = CH_LINE
ZR, ZB, XR = 61.0, 12.0, 130.0                    # ring and flat roof, the bottom nodes, the roof's nose / tail apex
ZR = max(ZR, Z_PROP + Z_SWEEP + 13.0)
YC = CH_LINE - FLAT                              # 107.7: where a flat meets a chamfer
NODES = {
    "R0": (FLAT, YC, ZR), "P1": (YC, FLAT, ZR), "P2": (-YC, FLAT, ZR), "P3": (-FLAT, YC, ZR),                               # the ring's eight corners (left half, nose -> tail)
    "P4": (-FLAT, -YC, ZR), "P5": (-YC, -FLAT, ZR), "P6": (YC, -FLAT, ZR), "R7": (FLAT, -YC, ZR),                           # right half
    "RF": (XR, 0, ZR), "RM": (PROBE_DX, 0, ZR), "RB": (-XR, 0, ZR), "RS0": (0.0, XR, ZR), "RS1": (0.0, -XR, ZR),            # the flat roof: a cross from RM to four apexes 25 in from the ring (9 Nov: the same on all four sides)                                                              # the flat roof's spine
    "TB": (-FLAT, 0, ZB), "S0": (0.0, FLAT, ZB), "S1": (0.0, -FLAT, ZB),                                                    # the bottom nodes in the flats' planes
    "NB": (FLAT, 0, ZB),                                                                                                     # the nose node: on a rod triangle to the deck's nose end and the nose bay (9 Nov: was a solid stub of the base, there for the old front camera)
    "DS0": (0.0, 40.0, 19.9), "BS0": (0.0, 40.0, -8.8), "DS1": (0.0, -40.0, 19.9), "BS1": (0.0, -40.0, -8.8),               # 9 Nov one body (printed deck-top down): DS's collar top on the bed (24.5), BS's crown (down in print) over the funnel's y walls           # (9 Nov: 33.8 apart, so the two rods to S0 / S1 are 17 deg apart and their sockets stay a node; DS under the discs' 27.7)            # sockets on the deck's side lugs (PRINTED)
    "DN": (75.5, 0, 19.9), "BN": (75.5, 0, 2.7),                                                                           # 9 Nov one body: the collars (13 toward NB, crowns down) end at x 89.2, inside the A1 mini's 180; BN's crown (down in print) on the funnel's -2.3                                                                            # two sockets on the nose bay's front face, 62 mm rods to NB (PRINTED; DN on the deck's nose end, 95 mm, read 1.1 x Euler nose-first)
    "DT": (-75.5, 0, 19.9), "BT": (-75.5, 0, 2.7),                                                                         # 9 Nov one body: as DN / BN at the tail                                                     # both on the tail tray's end, 17 deg apart from TB (9 Nov: DT on the deck's tail end was 11 deg off BT, a fork)                                                                # sockets on the deck's tail end and the tail tray's end, its socket's bottom over the moved tail wall's top + 2 (PRINTED)
}
PRINTED = ("DN", "BN", "DS0", "BS0", "DS1", "BS1", "DT", "BT")
ROOTS = {}
TIED = ("RM",)                                    # nodes tied to the core through the probe's turret: RM's ring and the throat's top collar print as one (9 Nov: 2.1 -> ~1 mm worst, the roof's centre held)
NODE_PARTS = [k for k in NODES if k not in PRINTED and k not in TIED]               # the generated cage nodes (parts/scout_v4_node_<k>, parts/scout_v4_node_lib.py)
RODS = [("R0", "P1"), ("P2", "P3"), ("P4", "P5"), ("P6", "R7"),   # the ring, open at the nose and tail flats (9 Nov: with RM tied to the turret the two 215 mm flats only carried a head-first crash, which buckled them; dropped, -0.43 m)
        ("RF", "R0"), ("RF", "R7"), ("RF", "RM"), ("RM", "RB"), ("RB", "P3"), ("RB", "P4"), ("RM", "RS0"), ("RM", "RS1"), ("RS0", "P1"), ("RS0", "P2"), ("RS1", "P5"), ("RS1", "P6"),                                 # the flat roof: nose and tail triangles, spine, hoop
        ("NB", "R0"), ("NB", "R7"), ("TB", "P3"), ("TB", "P4"),                                                                                            # the end Vs (the camera's window stays clear between them)
        ("S0", "P1"), ("S0", "P2"), ("S1", "P6"), ("S1", "P5"),                                                              # the side fans
        ("DS0", "S0"), ("BS0", "S0"), ("DS1", "S1"), ("BS1", "S1"), ("DT", "TB"), ("BT", "TB"), ("DN", "NB"), ("BN", "NB")]                                                            # the triangles that hold the bottom nodes, on x 0 and y 0
TIES = []
LUG = (8.0, 36.0, 44.0, -15.5)                   # the deck's side lugs: |x| 8, |y| 36..44 (3.2 outside the receiver), down to z -14 (over the funnel's y walls, -16.3)
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
    parts.append(base())                                             # 9 Nov: the base is part of the same body (Spark: "design the whole thing ... as one body")
    import manifold3d as mf
    b = union(parts)
    bm = mf.Manifold(mf.Mesh(np.asarray(b.vertices, np.float32), np.asarray(b.faces, np.uint32))) ^ mf.Manifold.cube([2 * BED, 2 * BED, 200.0]).translate([-BED, -BED, -100.0])
    t = bm.to_mesh(); return trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=False)


def zmin_walls_in(x):
    """the funnel with its end walls in (the box session, 9 Nov; Spark: "seat the scout on the receiver alone"): below z -4.3 nothing
    outside the 55 deg faces that rise from the receiver's outline; the lowest z a part may reach at x (add 2 margin)"""
    d = abs(x - 1.45 - BX) - 52.8
    return SEAT_Z if d <= 0 else min(-4.3, SEAT_Z + 1.428 * d)


def base():
    """v4 base on the receiver-only seat, now the under-deck part of the one body (9 Nov). The yokes over the receiver (its top z 9.6
    to the deck), where the FC, the Radxa, the XR2 and the DRV8833 hang under the deck (the nose bay is gone: with the nose sockets'
    collars inside the A1 mini's 180 there is no room for the FC between DN and BN); the nose block (x 55.6..89.3, |y| 26) for DN /
    BN and the flow sensor's pod under it (its bottom at -19.1, the rule's -21.1 + 2); the tail tray (x -89.3..-50, z -2.3 up) for DT /
    BT and the UBEC on its floor; every bottom on zmin_walls_in + 2"""
    lo, hi = np.array([-51.3 + BX, -32.8, -44.3]), np.array([54.3 + BX, 32.8, REC_TOP])   # the receiver v3 (C), not drawn
    parts = [box(lo[0], hi[0], -DECK[2], DECK[2], hi[2], DECK_Z0 + 0.5),             # the yokes over the receiver, the deck's width
             box(hi[0], BED, -26.0, 26.0, -2.3, DECK_Z0 + 0.5),                      # the nose block
             box(hi[0], 71.8, -12.0, 12.0, -19.1, -2.0),                             # the flow sensor's pod (open below: its window)
             box(-BED, lo[0], -34.0, 34.0, -2.3, DECK_TOP),                          # the tail tray
             ]
    return union(parts)


def place(src, rot=None, lo=None, yc=0.0):
    """a bought part's mesh, turned (axis, deg) about its centre, then moved so its bounds' low corner is lo (x, z) and centred on yc"""
    m = trimesh.load(os.path.join(ROOT, "parts", src, "part.stl"), force="mesh")
    if rot:
        c = m.bounds.mean(0); m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(rot[1]), rot[0], c))
    b = m.bounds
    m.apply_translation([lo[0] - b[0][0], yc - (b[0][1] + b[1][1]) / 2, lo[1] - b[0][2]])
    return m


REC_BOLTS = [(48.28, 26.55), (48.28, -26.55), (-37.65, 26.55), (-37.65, -26.55)]   # the receiver's 4 bolt bosses (receiver v3, C), drone frame before BX
REC_TOP = 9.6                                                                       # the receiver's top: the base's yokes sit on it
BATTERY_SET = (266.5, (-3.47, -0.01, -15.93))                                       # 9 Nov (Spark: "get rid of the battery mechanism, leave 4 bolts for it to mount
                                                                                    # to"): the receiver 60, cartridge 30, pack 165, XT60 10, feet and pads 1.5 are not drawn;
                                                                                    # they fly on the 4 bolts, so their mass at their CG (x + BX) stays in the balance and frame model


def receiver_bolts():
    """the battery's receiver held like the Overseer's (9 Nov, Spark: "just like the overseer make the bottom of the battery mounted with
    4 screws"): 4 M3 up from below through the receiver's O3.4 boss holes into M3 x 5.7 heat-set inserts in the base's yokes (O4 x 6
    holes in O10 bosses, wall >= 3); the screws' length is set by the receiver's boss (M3 x 12 E: measure), their heads reached from below"""
    out = []
    for x, y in REC_BOLTS:
        out.append(cyl(x + BX, y, REC_TOP - 7.0, REC_TOP + 5.0, 1.5, 16))            # the M3 screw, 12 long, head down inside the receiver's boss
        out.append(cyl(x + BX, y, REC_TOP - 7.0, REC_TOP - 5.0, 2.75, 16))           #   its head O5.5 x 2 (low-profile button)
        out.append(cyl(x + BX, y, REC_TOP, REC_TOP + 5.7, 2.0, 16))                  # the M3 x 5.7 heat-set insert in the yoke
    return trimesh.util.concatenate(out)


def pads():
    """on the receiver's bottom (the box session's numbers, 9 Nov): four TPU feet at its corners in |y| 21..23.5 (clear of the ledges'
    skirt at 25 and the cartridge's path), reaching z -47.5 so they are the foot on flat ground, and four bare contact pads O6 at
    (1.5 +- 30, +-29.5) for the box's keep-alive and seat probes in the ledges"""
    feet = [box(x0 + BX, x0 + BX + 8.0, *sorted((s * 21.0, s * 23.5)), -47.5, SEAT_Z) for x0 in (-50.0, 45.0) for s in (1, -1)]
    contacts = [cyl(CX + dx, s * 29.5, SEAT_Z - 0.2, SEAT_Z, 3.0, 24) for dx in (-30.0, 30.0) for s in (1, -1)]
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


def _xz(poly, y0, y1):
    """a shapely polygon drawn in the (x, z) plane, extruded along y from y0 to y1"""
    m = trimesh.creation.extrude_polygon(poly, y1 - y0)
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))   # (x, y, z) -> (x, -z, y)
    m.apply_translation([0, y1, 0])
    return m


def gear(teeth, m, cx, cz, y0, y1):
    """a spur gear, module m: trapezoid teeth from the root to the tip circle"""
    from shapely.geometry import Polygon
    rp = teeth * m / 2; ra, rf = rp + m, rp - 1.25 * m; pts = []
    for k in range(teeth):
        a = 2 * math.pi * k / teeth; w = 2 * math.pi / teeth
        for da, r in ((-0.5 * w, rf), (-0.22 * w, rf), (-0.12 * w, ra), (0.12 * w, ra), (0.22 * w, rf)):
            pts.append((cx + r * math.cos(a + da), cz + r * math.sin(a + da)))
    return _xz(Polygon(pts).buffer(0), y0, y1)


def cyl_y(r, x, z, y0, y1, n=32):
    return trimesh.creation.cylinder(radius=r, segment=[(x, y0, z), (x, y1, z)], sections=n)


def head_parts(split=False):
    """the head at the identity pose = parked, tilted TILT up (9 Nov compact). Fixed to the blades' tip: clip, base plate, the idler's
    bracket (-y) and the SG90's (+y), both tight to the cradle, and the SG90. Tilting about AXIS (y): the cradle (back plate, two side
    walls gripping the case, the horn's boss, the idler: an M3 screw), the camera in its case; the LED bars stuck on the case's top and
    bottom faces (the metal case is their heat sink), so they nod with it and need no fins"""
    hz, ax, az = THROAT["z1"], AXIS[0], AXIS[2]
    pz = hz + HEAD["clip"] + HEAD["plate"]                                           # the base plate's top
    c = CAM; h = c["side"] / 2; w = h + 2.0                                           # the cradle's side walls just outside the case
    ay = HEAD["ay"]
    sv = w + 3.0                                                                      # the SG90's output face: horn 1.5 + spline 1.5 off the wall
    clip = box(MAST_X - 5.5, MAST_X + 5.5, -BLADE_W / 2 - 2.0, BLADE_W / 2 + 2.0, hz, hz + HEAD["clip"])   # the clip round the two blades' tip
    fixed = [box(min(ax - 17.0, MAST_X - 5.5), max(ax + 8.0, MAST_X + 5.5), -w - 4.0, sv + 14.0, hz + HEAD["clip"], pz),   # the base plate, over the clip, under the servo and the brackets
             box(ax - 5.0, ax + 5.0, -w - 4.0, -w - 1.5, pz, az + 2.0),              # -y bracket, the idler's
             cyl_y(4.0, ax, az, -w - 4.0, -w - 1.5),                                  #   its boss
             box(ax - 17.0, ax + 8.0, sv + 11.5, sv + 14.0, pz, az - 6.25),           # +y bracket under the SG90's tabs
             box(ax - 16.4, ax + 6.0, sv, sv + 22.8, az - 6.25, az + 6.25),           # SG90 body 22.4 x 12.5 x 22.8 (typical drawing, E), shaft on the axis, facing -y
             box(ax - 21.2, ax + 11.6, sv + 6.0, sv + 8.5, az - 6.25, az + 6.25),    #   its mounting tabs (32.8 across, DORHEA listing)
             cyl_y(2.4, ax, az, w + 1.5, sv, 16)]                                     #   the spline
    fixed = [m.apply_translation([0, ay, 0]) for m in fixed] + [clip]
    t = [box(-0.5, c["case"][0], -w, w, -h + 4.0, h - 4.0),                          # back plate (open in the middle for the cable plug)
         box(-6.0, 14.0, -w, -h, -10.0, 10.0), box(-6.0, 14.0, h, w, -10.0, 10.0),   # the two side walls gripping the case (a TPU strap over its top)
         cyl_y(7.0, 0.0, 0.0, w, w + 1.5),                                            # the SG90's horn, screwed to the +y wall
         cyl_y(1.5, 0.0, 0.0, -w - 4.0, -w, 12),                                      # idler: an M3 screw (-y) into the bracket's boss
         box(c["case"][0], c["case"][1], -h, h, -h, h),                               # the camera in its metal case
         trimesh.creation.cylinder(radius=c["lens"][2], segment=[(c["lens"][0], 0, 0), (c["lens"][1], 0, 0)], sections=32)]   # its lens window
    f = c["case"][1]
    leds = [box(f - 8.0, f, -12.5, 12.5, h, h + 1.2), box(f - 8.0, f, -12.5, 12.5, -h - 1.2, -h)]   # Clearhill 5 V COB, 25 x 8 each, on the case's top and bottom fronts
    pose = trimesh.transformations.rotation_matrix(-TILT, [0, 1, 0]); pose[:3, 3] = AXIS            # AXIS carries ay
    tilt = trimesh.util.concatenate(t); tilt.apply_transform(pose)
    led = trimesh.util.concatenate(leds); led.apply_transform(pose)
    if split:                                                                         # the page's parts: printed frame, bought SG90, printed cradle, bought camera
        frame_ = trimesh.util.concatenate(fixed[:4] + fixed[7:]); servo = trimesh.util.concatenate(fixed[4:7])
        cradle = trimesh.util.concatenate(t[:4]); cam = trimesh.util.concatenate(t[4:])
        cradle.apply_transform(pose); cam.apply_transform(pose)
        return dict(head_frame=frame_, probe_servo=servo, cradle=cradle, probe_tilt=cam, probe_leds=led)
    return trimesh.util.concatenate(fixed), tilt, led


def tilt_sweep():
    """the cradle swept over the servo's range: its least gap over the base plate (2 deg steps), and whether any of it enters the SG90
    or the idler bracket (vertex in box; the coupler and the pin, which run into them on the axis, excepted)"""
    _, tilt, led = head_parts()
    v0 = np.vstack([tilt.vertices, led.vertices]) - AXIS
    hz, ax, az = THROAT["z1"], AXIS[0], AXIS[2]; pz = hz + HEAD["clip"] + HEAD["plate"]
    worst, hit = 1e9, []
    for d in np.arange(HEAD["range"][0], HEAD["range"][1] + 0.1, 2.0):
        R = trimesh.transformations.rotation_matrix(-(math.radians(d) - TILT), [0, 1, 0])[:3, :3]
        v = v0 @ R.T + AXIS
        worst = min(worst, float((v[:, 2] - pz).min()))
        off_axis = np.hypot(v[:, 0] - ax, v[:, 2] - az) > 7.5
        w = CAM["side"] / 2 + 2.0
        yy = v[:, 1] - HEAD["ay"]
        servo = (v[:, 0] > ax - 21.2) & (v[:, 0] < ax + 11.6) & (yy > w + 3.0) & (np.abs(v[:, 2] - az) < 6.25)
        brk = (np.abs(v[:, 0] - ax) < 5.0) & (yy < -w - 1.5) & (v[:, 2] < az + 2.0)
        if (off_axis & (servo | brk)).any():
            hit.append(int(d))
    return round(worst, 1), hit


def probe():
    """the ceiling probe (9 Nov). Turret, under the roof's middle node where the FC sat: the spool (axis y) with its 40 T gear, the 15 T
    pinion on the N20 in a pocket in the deck, the clock-spring spiral that carries the flat cable to the fixed side, the 30 mm throat
    the two blades leave through (forming their lens) and the housing's two side plates; M3 screws are the spool's and the idler's axles. Returns {name: mesh}"""
    from shapely.geometry import Point, LineString, Polygon
    (cx, cz), r = SPOOL["c"], SPOOL["r"]; t = THROAT
    y0, y1 = SPOOL["y0"], SPOOL["y1"]
    spool = [cyl_y(SPOOL["hub"], cx, cz, y0 + 1.2, y1 - 1.2, 48), cyl_y(r, cx, cz, y0, y0 + 1.2, 48), cyl_y(r, cx, cz, y1 - 1.2, y1, 48),
             cyl_y(SPOOL["hub"] + 1.4, cx, cz, -BLADE_W / 2, BLADE_W / 2, 48),                     # the stack (A, the ribbon, B) wound on the hub, 1.6 turns
             gear(GEAR["spool_t"], GEAR["m"], cx, cz, *GEAR["y"]), cyl_y(1.5, cx, cz, DRUM["y0"] - 2.0, GEAR["y"][1] + 3.0, 12)]   # 40 T and its M3 axle
    pz = cz - (GEAR["spool_t"] + GEAR["pinion_t"]) * GEAR["m"] / 2
    pinion = gear(GEAR["pinion_t"], GEAR["m"], cx, pz, *GEAR["y"])
    g0 = N20_FACE                                                                     # the gearbox's face, 2.9 short of the pinion (room for the turret's 2.6 face plate)
    motor = [box(cx - 6.0, cx + 6.0, g0 - 10.0, g0, pz - 5.0, pz + 5.0),                        # Augiimor N20 6 V 300 rpm: gearbox 12 x 10 x 10 (37 total - can 17 - shaft 10), facing +y
             cyl_y(6.0, cx, pz, g0 - 27.0, g0 - 10.0, 24),                             # can O12 x 17, under the spool and the drum
             cyl_y(1.5, cx, pz, g0, g0 + 10.0, 12)]                                    # 3 mm D shaft: through the pinion, its tip in the +y plate's bearing hole
    th = np.linspace(0, 2 * math.pi * 3.5, 400); rr = 4.0 + (DRUM["r"] - 1.5 - 4.0) * th / th[-1]
    spiral = LineString([(cx + a * math.cos(b), cz + a * math.sin(b)) for a, b in zip(rr, th)]).buffer(1.1)
    drum = [_xz(spiral, DRUM["y0"] + 1.0, DRUM["y1"] - 1.0), cyl_y(4.0, cx, cz, DRUM["y0"] + 0.5, DRUM["y1"] + 0.5, 24)]           # the camera's own USB cable in a loose coil (2.5 turns of room, ~1.5 used)
    feed = [box(MAST_X + sg * 1.6 - 0.2, MAST_X + sg * 1.6 + 0.2, -BLADE_W / 2, BLADE_W / 2, cz, t["z0"] + 2.0) for sg in (1, -1)]   # the blades from the spool up into the throat
    yp, ym = GEAR["y"][1] + 0.8, DRUM["y0"] - 0.8
    rd = DRUM["r"] + 1.0; rg = max(r, (GEAR["spool_t"] / 2 + 1) * GEAR["m"]) + 0.5
    side_d = Polygon([(cx - rd, 24.5), (cx + rd, 24.5), (cx + rd, cz), (cx - rd, cz)]).union(Point(cx, cz).buffer(rd, 64))   # the -y plate: only the drum is there
    side_g = Polygon([(cx - rg, 24.5), (cx + rg, 24.5), (cx + rg, cz), (cx - rg, cz)]).union(Point(cx, cz).buffer(rg, 64))   # the +y plate: carries the axle
    housing = [_xz(side_g, yp, yp + 1.2), _xz(side_d, ym - 1.2, ym),
               box(cx - rg, t["x1"], ym - 1.2, yp + 1.2, 24.5, 26.0),                        # floor on the deck
               cyl_y(DRUM["r"] + 1.0, cx, cz, ym, DRUM["y0"], 48),                      # the drum's cover
               box(t["x0"], t["x1"], -t["y"], t["y"], t["z0"], t["z1"]),                 # the throat
               trimesh.creation.annulus(r_min=t["x1"] - cx + 1.0, r_max=22.0, height=8.0).apply_translation([cx, 0, ZR])]   # its top collar out to RM's ring: one print with the node
    rr = SPOOL["hub"] + 1.4 + ROLLER["r"]; a = math.radians(ROLLER["ang"])                         # the pressure roller on the full coil
    roller = [cyl_y(ROLLER["r"], cx + rr * math.cos(a), cz + rr * math.sin(a), -ROLLER["w"] / 2, ROLLER["w"] / 2, 24),
              cyl_y(1.5, cx + rr * math.cos(a), cz + rr * math.sin(a), -ROLLER["w"] / 2 - 3.0, ROLLER["w"] / 2 + 3.0, 12)]
    for sy in (-1, 1):                                                                # its two springs, radial, beyond the tube's ends
        e = np.array([cx + (rr + 2.0) * math.cos(a), 0, cz + (rr + 2.0) * math.sin(a)]); f = np.array([cx + (rr + 13.0) * math.cos(a), 0, cz + (rr + 13.0) * math.sin(a)])
        e[1] = f[1] = sy * (ROLLER["w"] / 2 + 1.6)
        roller.append(trimesh.creation.cylinder(radius=2.0, segment=[e, f], sections=12))
    return dict(probe_turret=trimesh.util.concatenate(housing), probe_spool=trimesh.util.concatenate(spool), probe_pinion=pinion, probe_roller=trimesh.util.concatenate(roller),
                probe_motor=trimesh.util.concatenate(motor), probe_drum=trimesh.util.concatenate(drum), probe_feed=trimesh.util.concatenate(feed),
                **head_parts(split=True))


MAST_STUB = 1.0                                   # the mast's mesh: 1 mm of the two blades above the throat; the page's timeline
                                                  # (tools/scout_v4_timeline.py) stretches it to the length rolled out and lifts the head


def mast():
    """the two blades as they leave the throat, curves facing: a lens BLADE_W wide across the drone (y), each blade bowed 2.5 out along x
    (the blades' width runs along the spool's axis; thickness drawn 0.4 to be seen)"""
    from shapely.geometry import Polygon
    hw = BLADE_W / 2; ys = np.linspace(-hw, hw, 21); bow = 2.5 * (1 - (ys / hw) ** 2)
    out = []
    for sg in (1, -1):
        ring = [(MAST_X + sg * b, y) for y, b in zip(ys, bow)] + [(MAST_X + sg * (b - 0.4), y) for y, b in zip(ys[::-1], bow[::-1])]
        m = trimesh.creation.extrude_polygon(Polygon(ring).buffer(0), MAST_STUB); m.apply_translation([0, 0, THROAT["z1"]])
        out.append(m)
    return out


HUB_RING = 34.0                                   # the roof's middle node RM is a ring (r 22..34) round the probe's throat (its corners at r 20.2): its four rods end in its sides at r 34
RM_MOUTH = {"RF": 33.0, "RB": 26.0, "RS0": 29.0, "RS1": 34.0}   # 10 Nov (the turret's run: 0.72 at RM-RB, 1.01 on 9 Nov, with every mouth at r 34): each socket's mouth as near
                                                  # as its neighbours allow: RF clear of the throat's slot (the mast at x 15.1), RB over the spool, RS0 past the 40 T gear, RS1 past the drum's end


def cage():
    def cut(i, j):                           # each rod stops 3 inside its nodes' spheres (two in-line rods through a node must not share an end
        a, b = P(i), P(j); u = (b - a) / np.linalg.norm(b - a)   # ring: the page's mesh cleanup welds them and drops the body, 7 Nov), and at
        bi = RM_MOUTH[j] if i == "RM" else 3.0                   # the turret's socket mouths at RM
        bj = RM_MOUTH[i] if j == "RM" else 3.0
        return a + u * bi, b - u * bj
    rods = [rod(*cut(i, j)) for i, j in RODS + TIES]
    nodes = [trimesh.creation.icosphere(subdivisions=1, radius=5.0).apply_translation(P(k)) for k in NODES if k not in PRINTED and k != "RM"]
    nodes += [trimesh.creation.icosphere(subdivisions=1, radius=4.6).apply_translation(P(k)) for k in PRINTED]   # the sockets printed into the frame and the base (their collars, O9.2)
    ring = trimesh.creation.annulus(r_min=22.0, r_max=HUB_RING, height=8.0); ring.apply_translation(P("RM"))
    nodes.append(ring)                                                                   # RM: the ring node the turret's throat passes through
    return trimesh.util.concatenate(rods), trimesh.util.concatenate(nodes)


def frame_check(core_g):
    """the Scout's frame model (tools/scout_cage_loads.py, the standard cases: 35 g crash on the hull face the CG ray leaves through, 14
    directions, landing and rough landing 6 g on the runners) on these tables: the core one rigid body at the origin with the
    sockets printed into the frame and the base and the runner corners stiff to it"""
    import io, contextlib, scout_cage as sc, scout_cage_loads as sl
    sc.NODES, sc.RODS, sc.TIES, sc.TIE_ROOTS = dict(NODES), list(RODS), list(TIES), dict(ROOTS)
    sc.PRINTED = {k: "frame" for k in PRINTED + TIED}
    sc.BASE = {"G0": (CX + SEAT_L / 2, SEAT_W / 2, SEAT_Z), "G1": (CX - SEAT_L / 2, SEAT_W / 2, SEAT_Z), "G2": (CX - SEAT_L / 2, -SEAT_W / 2, SEAT_Z), "G3": (CX + SEAT_L / 2, -SEAT_W / 2, SEAT_Z)}
    sc.BRIDGES = []; sc.GROUPS = sc._groups(); sc.GROUP_OF = {k: g for g in sc.GROUPS for k in g}; sc.RODS_CARBON = list(RODS)
    sl.core_mass = lambda: (core_g, np.zeros(3))
    with contextlib.redirect_stdout(io.StringIO()):
        s_, rec = sl.run(nodes=sc.NODES, rods=sc.RODS, ties=sc.TIES, dry=True, log=lambda *a: print(*a))
    w = max(rec["cases"], key=lambda r: r["deflection_mm"])
    eu = max(rec["rods"].items(), key=lambda kv: kv[1]["euler"]); mo = max(rec["rods"].items(), key=lambda kv: kv[1]["moment"])
    out = dict(mode_hz=rec["modes_hz"][0], deflection=(w["deflection_mm"], w["name"]), euler=(eu[1]["euler"], eu[0]), moment=(mo[1]["moment"], mo[0]),
               pull=max(v["tension"] for v in rec["rods"].values()))
    sc.BASE["HD"] = (float(AXIS[0]), 0.0, float(head_parts()[1].bounds[1][2]))      # the parked head's top: the Scout's highest point, rigid to the core through the throat
    with contextlib.redirect_stdout(io.StringIO()):
        s_, rh = sl.run(nodes=sc.NODES, rods=sc.RODS, ties=sc.TIES, dry=True, log=lambda *a: print(*a))
    wh = max(rh["cases"], key=lambda r: r["deflection_mm"]); muh = max(rh["rods"].items(), key=lambda kv: kv[1]["moment"]); euh = max(rh["rods"].items(), key=lambda kv: kv[1]["euler"])
    out["head_first"] = dict(deflection=wh["deflection_mm"], euler=(euh[1]["euler"], euh[0]), moment=(muh[1]["moment"], muh[0]),
                             head_N=max(-(c["reactions"].get("HD") or 0) for c in rh["cases"]))
    del sc.BASE["HD"]
    nl = {}                                                                         # the nodes' topology loads: per node and member, the worse of the standard and the head-first runs
    for r in (rec, rh):
        for k, mm in r["node_loads"].items():
            for nm, v in mm.items():
                o = nl.setdefault(k, {}).get(nm)
                if o is None or v["axial"] > o["axial"] or v["side"] > o["side"]:
                    nl[k][nm] = dict(axial=max(v["axial"], o["axial"] if o else 0), side=max(v["side"], o["side"] if o else 0), N=v["N"], case=v["case"], tag="F")
    json.dump(dict(note="Scout v4 frame model (tools/scout_v4_shape.py frame_check): per node and member the worse of the standard cases and the head-first crash", node_loads=nl),
              open(os.path.join(ROOT, "systems", "scout_v4.cage_loads.json"), "w"), indent=1)
    hn = int(math.ceil(out["head_first"]["head_N"] / 10.0) * 10)                    # the probe's setups read the head-first load from here (9 Nov: it was 230 by hand, the model gives ~250)
    json.dump(dict(note="Scout v4 frame model: the parked head's contact force in the head-first crash, rounded up to 10 N; read by the probe's turret, head frame and cradle", head_first_N=hn),
              open(os.path.join(ROOT, "systems", "scout_v4.head_load.json"), "w"), indent=1)
    return out


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
    for nm, ids in (("probe head", ("head_frame", "probe_servo", "cradle", "probe_tilt", "probe_leds")), ("probe turret", ("probe_turret", "probe_spool", "probe_pinion", "probe_motor", "probe_drum", "probe_feed", "probe_roller"))):
        v = np.vstack([np.asarray(meshes[k].vertices) for k in ids])
        out[f"{nm} to discs"] = (round(min(disc_dist(p) for p in v), 1), "need 12")
    rods_m, nodes_m = cage()
    nodes_x = trimesh.util.concatenate([trimesh.creation.icosphere(subdivisions=1, radius=7.0 if k in PRINTED else 5.0).apply_translation(P(k)) for k in NODES if k not in TIED])
    clash = []
    for k in ("probe_turret", "probe_spool", "probe_pinion", "probe_motor", "probe_drum", "probe_feed", "probe_roller", "head_frame", "probe_servo", "cradle", "probe_tilt", "probe_leds"):
        for nm, o in (("nodes", nodes_x if k == "probe_turret" else nodes_m), ("rods", rods_m)):   # the turret's collar and RM's ring are one print
            if o.contains(meshes[k].vertices).any() or (meshes[k].is_watertight and meshes[k].contains(o.vertices).any()):
                clash.append(f"{k} x cage {nm}")
    out["probe vs cage"] = ("clear" if not clash else "CLASH: " + ", ".join(clash), "every probe part against the cage's nodes, sockets and rods (parked)")
    gap, hit = tilt_sweep()
    out["head tilt sweep"] = (f"clear, {gap} over the base plate" if gap > 0 and not hit else f"HITS at {hit} deg, gap {gap}", f"the cradle, camera and LED bars over {HEAD['range'][0]:.0f}..+{HEAD['range'][1]:.0f} deg")
    for nm in ("frame",):
        v = np.asarray(meshes[nm].vertices); top = v[:, 2].max()
        out[f"{nm} to discs"] = (round(min(disc_dist(p) for p in v if p[2] > DECK_Z0 - 7), 1), f"top z {top:.1f}; the props sweep down to {Z_PROP - Z_SWEEP:.1f}")
    return out


def write():
    meshes = {}
    meshes["frame"] = frame()                                                        # 9 Nov: deck, arms, pods, lugs and the base in one body (parts/scout_v4_frame)
    for k, (mx, my) in enumerate(MOTORS, 1):
        sx, sy = np.sign(mx), np.sign(my)
        meshes[f"motor_{k}"] = moved("scout_motor_" + str(k), sx * (HUB - 54.1), sy * (HUB - 54.1), DECK_TOP - 22.0)
        meshes[f"prop_{k}"], meshes[f"blades_{k}"] = prop(mx, my)
    meshes.update(probe())
    meshes["probe_mast_a"], meshes["probe_mast_b"] = mast()
    meshes["scout_flow_mtf_02p"] = place("scout_flow_mtf_02p", rot=([0, 0, 1], 90), lo=(54.5 + BX, -19.1))   # turned 90 deg, in the pod: outer edge x 70.5
    # 9 Nov (one body, the nose bay gone): the FC and the Radxa hang under the deck over the receiver on posts (inserts in the deck),
    # the XR2 and the DRV8833 flat on tape lands beside the FC, the UBEC on the tail tray's floor across the drone
    meshes["fc"] = place("fc", lo=(FC_X[0], FC_Z0))                                  # 36 square, component side up, its posts down from the deck
    meshes["radxa_zero_3w"] = place("radxa_zero_3w", rot=([0, 1, 0], 90), lo=(-72.6, 10.4), yc=0.0)
    meshes["radxa_zero_3w"].apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1], meshes["radxa_zero_3w"].bounds.mean(0)))   # 65 along x
    meshes["radxa_zero_3w"].apply_translation([-72.6 - meshes["radxa_zero_3w"].bounds[0][0], 0, 0])
    meshes["ubec"] = place("ubec", lo=(-71.5, 0.3), yc=UBEC_YC)                        # 61 along y on the tail tray's land
    meshes["xr2"] = place("xr2", rot=([0, 0, 1], 90), lo=(4.0, 10.0), yc=28.5)        # flat, 34.5 x 13 x 4, under its land at +y
    meshes["xr2"].apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0], meshes["xr2"].bounds.mean(0)))
    meshes["xr2"].apply_translation([0, 0, 10.0 - meshes["xr2"].bounds[0][2]])
    meshes["drv8833"] = box(4.0, 22.5, -38.8, -22.8, 10.0, 13.0)                     # KOOBOOK DRV8833 18.5 x 16 (listing), ~3 thick E, flat under its land at -y
    # the battery mechanism (receiver, cartridge, pack, its feet and pads) is not part of this design: it mounts on receiver_bolts
    meshes["receiver_bolts"] = receiver_bolts()
    meshes["cage_rods"], _ = cage()
    for k in NODE_PARTS:                                                             # the 16 generated cage nodes: a sphere until the run lands
        meshes[f"node_{k.lower()}"] = trimesh.creation.icosphere(subdivisions=1, radius=6.0).apply_translation(P(k))
    for nm, m in meshes.items():
        d = os.path.join(ROOT, "parts", f"scout_v4_{nm.replace('box_ctx_scout_', '').replace('scout_', '')}"); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "result.json")) or os.path.exists(os.path.join(d, "make.py")):   # a generated part whose run has landed, or a drawn part (make.py): its part.stl is its own, never overwritten
            meshes[nm] = trimesh.load(os.path.join(d, "part.stl"), force="mesh"); continue
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
    fv = abs(meshes["frame"].volume)
    add("frame", "frame v4, one body: deck, arms, motor pods, side lugs and the base (yokes, nose block, flow pod, tail tray) (generated, parts/scout_v4_frame)", "structure", "frame", "#d9d2c3", 110.0 if not os.path.exists(os.path.join(ROOT, "parts", "scout_v4_frame", "result.json")) else json.load(open(os.path.join(ROOT, "parts", "scout_v4_frame", "result.json")))["mass_g"], "E",
        f"one print inside |x|, |y| {BED} (the A1 mini's 180), deck-top down on the bed: deck z {DECK_Z0}..{DECK_TOP} and the yokes over the receiver down to its top 9.6, arms to O28 motor pads at (+-{HUB}, +-{HUB}), "
        "the battery mechanism's 4 M3 inserts, the FC's and the Radxa's posts, the XR2's, DRV8833's, UBEC's and flow sensor's lands, the 8 rod sockets (DN BN DT BT DS0 BS0 DS1 BS1), the turret's four insert bosses; the room until the run lands; PCTG", "PCTG", 0.85)
    for k in range(1, 5):
        add(f"motor_{k}", f"motor {k} (EMAX ECO II 2004)", "bought", "motors", "#8b8f96", 18.0, "listing", f"on its pod at (+-{HUB}, +-{HUB}), base z {DECK_TOP}")
        add(f"prop_{k}", f"prop {k} disc (Gemfan 5125, O{PROP_D}, swept)", "bought", "props", "#9fd3ff", 0.0, "listing", f"prop plane z {Z_PROP:.1f}", opacity=0.18)
        add(f"blades_{k}", f"prop {k} (Gemfan Hurricane SL 5125, 3 blade)", "bought", "props", "#3b6ea8", 2.8, "listing", "a sketch of the three blades")
    add("probe_tilt", "probe camera: Arducam 12MP autofocus USB in its metal case (B0FLX58WKZ; bracket off)", "bought", "camera", "#1d5e3a", 45.0, "listing",
        f"nods with the cradle about the SG90's axis from {HEAD['range'][0]:.0f} to +{HEAD['range'][1]:.0f} deg; parked {math.degrees(TILT):.0f} deg up it is the front camera, lens ({LENS[0]:.0f}, 0, {LENS[2]:.0f}), "
        "61 x 50 deg, autofocus 10 cm to inf, 1080p 30 live, 12 MP stills; the case is its heat spreader (Spark 9 Nov); 36 square, depth 20 E: measure")
    add("cradle", "probe camera cradle (generated, parts/scout_v4_cradle)", "structure", "camera", "#3c6e57", 2.5, "E",
        "side pads, front lips and a back stop round the case, the horn's pad and the idler's insert; a TPU strap over the case (1 g); PCTG", "PCTG")
    add("probe_leds", "probe light: two 25 x 8 mm bars of Clearhill 5 V COB strip, 4000 K, CRI 90 (B0F1YPP6QS)", "bought", "camera", "#fff1b8", 0.5, "E",
        "stuck on the camera case's top and bottom fronts (the metal case is their heat sink; no fins), 25 mm is the strip's cut length; ~0.2 W a bar at 8 W/m; on the DRV8833's B channel, DC or PWM >= 20 kHz")
    add("probe_servo", "probe tilt servo: SG90 (DORHEA, B0DFKPVRC2) and its 470 uF capacitor", "bought", "camera", "#2f3b45", 10.0, "listing",
        "its shaft on the tilt axis facing -y, held by its 32.8 tabs on the head frame's +y pads; the capacitor on its lead covers its > 1 A start")
    add("head_frame", "probe head frame: the clip on the blades' tip, the SG90's pads, the idler's boss (generated, parts/scout_v4_head_frame)", "structure", "camera", "#56606a", 2.5, "E",
        "a 14 mm sleeve round the lens: blade A held by an M2 at its top, blade B's M2 in an 11 mm slot (on one spool B pays out 7.5 mm less than A over the 150: the ribbon between them, tools/probe_mast_sim.py), the ribbon's 4 mm loop; an M3 screw is the idler; PCTG", "PCTG")
    add("probe_turret", "probe turret: housing, throat, drum and the roof's middle node RM (PCTG, generated: parts/scout_v4_probe_turret)", "structure", "camera", "#f2b705", 41.2, "C",
        "side plates round the spool and the drum, the N20's cheeks, the roller's guides and spring pockets, the throat up to the head's seat, RM's rod sockets; TPU lip 1, wires 1 (E); PCTG", "PCTG")
    add("probe_spool", "probe spool with its 40 T gear (drawn: parts/scout_v4_probe_spool/make.py)", "structure", "camera", "#d08a1e", 21.0, "C",
        f"hub O28 between O31.8 flanges 19.6 apart, the 40 T gear (module 0.8, x -0.35, generated involute) outside the +y one, the drum's hub O8 outside the -y one; the stack (blade A outside, the 0.6 ribbon, blade B on the hub) wound 1.6 turns to O30.8 for the {REACH:.0f} mm; its three ends epoxied in one 1.2 slot at 45 deg into the hub, the ribbon on through the hub to the drum; turns on an M3 x 40; PCTG, gear face down", "PCTG")
    add("probe_pinion", "probe pinion, 12 T, on the N20's 3 mm D shaft (drawn: parts/scout_v4_probe_pinion/make.py)", "structure", "camera", "#a85f10", 0.3, "C", "x +0.35 (no undercut), centres 20.8 from the spool's; D bore 3.1 / 2.6 pressed on with CA; PCTG", "PCTG")
    add("probe_motor", "N20 6 V 100 rpm gear motor (B07MZH851L; replaces the 300 rpm B0BZDC951L)", "bought", "camera", "#8b8f96", 10.0, "E",
        "the same body: 37 long (can 17 x O12, 3 mm D shaft 10); under the spool and the drum facing +y, in a pocket through the deck; the coil's own unwind is ~100 N mm on the spool and winding in needs up to ~170 (sims): the 300 rpm stalls at ~140 through 40 / 12, the 100 rpm at ~420 (E); ~4.5 s for the 150; parked: the DRV8833 in brake, the head seated (the coil's push is ~35 N mm at the gearbox)")
    add("probe_drum", "the head's cable, coiled loosely in the drum", "bought", "camera", "#c96f2d", 3.0, "E",
        "a flat ribbon of 6 x 30 AWG PTFE wire (Remington 30KITSTRPTFE, B0722LL645; 0.6 thick, 3.3 wide, kapton-bound; the 26 AWG pair made it 1.2 and the blades' fight 17 mm): 5 V, GND, USB D+ / D- (side by side), the servo's signal, the LED bars' minus; the 470 uF at the servo covers its start; a JST ZH 1.5 mm 4-pin lead into the camera; up inside the lens on A, wound between the blades, its spool end through the hub into the drum's loose coil (3.5 turns of room for 1.6); the USB pair to the USB-C breakout on the Radxa")
    add("probe_feed", "the two blades from the spool up into the throat", "bought", "camera", "#b8902a", 8.8, "E",
        "two 19 mm (3/4 in) steel tape strips, 0.2 m each with 0.15 m used, cut from the Topzone 16 ft tape (B00ZMHXV4I; its case and spring not used), curves facing (~22 g/m, E); spool ends set in printed clamp slots with epoxy, tip ends with an O2.2 hole each for the head's M2s (a carbide bit or a punch: hardened steel); their mass is all here")
    add("probe_roller", "the coil's pressure roller (drawn: parts/scout_v4_probe_roller/make.py): a printed O10.1 x 19 tube on two MR63ZZ (uxcell B075CLH2WM), an M3 axle, two 4 x 0.5 x 15 springs (uxcell B076LQCBR1)", "bought", "camera", "#7a8590", 2.5, "E",
        "30 deg before the exit, its axle's ends in radial slots of the turret's walls, the two springs pushing it on the coil: 12 N full, 7.8 N empty (E: the springs' rate ~1.5 N/mm); the sims: a coil whose mast is pushed out lifts its outer turn (blossoms) unless something locks it; the roller's friction locks a push of up to 2.4 N (mu 0.3)")
    for k, col in (("a", "#e8c547"), ("b", "#b8902a")):
        add(f"probe_mast_{k}", f"probe mast, blade {k.upper()} (10 mm steel tape, curves facing)", "placeholder", "camera", col, 0.0, "E",
            f"drawn 1 mm long above the throat; the page's timeline rolls it out to {REACH:.0f} mm and back (tools/scout_v4_timeline.py). Its mass is in the feed's")
    add("fc", "FC MicoAir743 AIO (under the deck, over the receiver)", "bought", "electronics", "#1f6f3f", 10.0, "P", f"36 x 36 at x {FC_X[0]}..{FC_X[1]}, z {FC_Z0}..{FC_Z0 + 10.5}, component side up on 4 M3 posts down from the deck (grommets; heads from below with the receiver off); set its IMU offset in PX4 (EKF2_IMU_POS_X ~ +{(FC_X[0] + FC_X[1]) / 2000:.3f} m)")
    for src, nm, g, note in (("ubec", "UBEC", 15.0, "on the tail tray's tape land, across the drone (x -71.5..-54.5, |y| 30.5), under the Radxa"), ("xr2", "XR2", 1.0, "flat under its tape land beside the FC (+y)"),
                             ("drv8833", "DRV8833 (KOOBOOK, B07S7Z8CFL)", 1.0, "flat under its tape land beside the FC (-y): channel A the probe's N20, B its LED bars"),
                             ("radxa_zero_3w", "Radxa Zero 3W", 10.0, "under the deck behind the turret's N20 on four M2.5 posts (inserts in the deck), 65 along x, its USB-C plugs out the -y edge"),
                             ("scout_flow_mtf_02p", "flow MTF-02P", 1.5, "the nose pod, turned 90 deg (outer edge x 70.5, bottom -19.1): set SENS_FLOW_ROT for the turn")):
        add(src, nm, "bought", "electronics", "#2b5d8a", g, "P", note)
    add("receiver_bolts", "battery mount: 4 x M3 up from below into M3 x 5.7 heat-set inserts in the base's yokes (as the Overseer's)", "bought", "battery", "#5a5f66", 4.0, "E",
        f"at ({REC_BOLTS[0][0] + BX:.1f}, +-26.55) and ({REC_BOLTS[2][0] + BX:.1f}, +-26.55) through the receiver's O3.4 boss holes; M3 x 12 E (the boss's depth: measure); the inserts in O10 bosses of the yokes (wall >= 3), the heads reached from below with the cartridge out")
    L = sum(float(np.linalg.norm(P(i) - P(j))) for i, j in RODS + TIES)
    add("cage_rods", "half cage v4, rods (3 mm carbon: the ring, the flat roof, end Vs and side fans)", "cage", "cage", "#2b2b2b", L * ROD_G_MM, "E",
        f"{len(RODS)} rods, {L / 1000:.2f} m, every one 12 from the swept discs and none over or under a disc: ring at z {ZR:.0f} with flats at {FLAT:.0f} (310 across), a flat roof (spine on y 0, hoop on x 0, nose / tail triangles), a V at each end and a fan at each side down to the bottom nodes, which hang on rod triangles to the deck's side lugs and tail (NB on the camera's mount)")
    for k in NODE_PARTS:
        rj = os.path.join(ROOT, "parts", f"scout_v4_node_{k.lower()}", "result.json")
        add(f"node_{k.lower()}", f"cage node {k} (generated, parts/scout_v4_node_{k.lower()})", "structure", "cage", "#eef2f5", 5.0 if not os.path.exists(rj) else json.load(open(rj))["mass_g"], "E",
            "the socket collars of every rod that ends here, the rest generated (parts/scout_v4_node_lib.py); a sphere until the run lands; PCTG", "PCTG", 0.9)
    return rows


if __name__ == "__main__":
    M = write()
    L = sum(float(np.linalg.norm(P(i) - P(j))) for i, j in RODS)
    print(f"Scout v4 shape: props {PROP_D} on hubs +-{HUB} (wheelbase {2 * HUB * math.sqrt(2):.0f}), prop plane z {Z_PROP:.1f}, deck z {DECK_Z0}..{DECK_TOP}, ring flats {FLAT:.1f}, chamfer {CH_LINE:.1f}, ring and roof z {ZR:.1f}; "
          f"cage {len(NODES) - len(PRINTED)} nodes + {len(PRINTED)} printed, {len(RODS)} rods {L / 1000:.2f} m")
    for k, (v, why) in checks(M).items():
        print(f"  {k:16} {v}  ({why})")
    core = sum(r["mass_g"] for r in entries(M) if "cage" not in r["id"]) + BATTERY_SET[0]
    fc = frame_check(core)
    print(f"  frame model     first mode {fc['mode_hz']:.0f} Hz, worst {fc['deflection'][0]:.1f} mm ({fc['deflection'][1][:60]}), Euler <= {fc['euler'][0]:.2f} ({fc['euler'][1]}), "
          f"end moment <= {fc['moment'][0]:.0f} N mm ({fc['moment'][1]}; socket cap 2000), pull-out <= {fc['pull']:.0f} N (cap 940)")
    hf = fc["head_first"]
    print(f"  head-first      the parked head as a contact (35 g onto it at an angle): worst {hf['deflection']:.1f} mm, Euler <= {hf['euler'][0]:.2f} ({hf['euler'][1]}), "
          f"end moment <= {hf['moment'][0]:.0f} N mm ({hf['moment'][1]}), up to {hf['head_N']:.0f} N into the head, throat and turret")
    F = os.path.join(ROOT, "systems", "scout_v4.assembly.json")
    old = json.load(open(F)) if os.path.exists(F) else {}
    rows = entries(M)
    lo = np.min([m.bounds[0] for m in M.values()], 0); hi = np.max([m.bounds[1] for m in M.values()], 0)
    drawn = sum(r["mass_g"] for r in rows); mass = drawn + BATTERY_SET[0]
    S_ = np.zeros(3)
    by_id = {f"scout_v4_{nm.replace('box_ctx_scout_', '').replace('scout_', '')}": m for nm, m in M.items()}
    for r in rows:
        m = by_id[r["id"]]
        c = m.center_mass if (m.is_watertight and abs(m.volume) > 1e-6) else m.bounds.mean(0)
        S_ += r["mass_g"] * np.asarray(c)
    bc = np.array(BATTERY_SET[1]) + [BX, 0, 0]; S_ += BATTERY_SET[0] * bc; cg = S_ / mass
    a = dict(name="Scout v4", title="Scout v4", artifact=old.get("artifact"), machine="scout_v4", order=0, readme_view=old.get("readme_view", "iso"),
             appendix=old.get("appendix", "systems/scout_v4.appendix.html"), timeline="tools/scout_v4_timeline.py:timeline", frame_timeline=True,
             views={"probe": [[330.0, -380.0, 330.0], [22.0, 0.0, 185.0]], "turret": [[90.0, -140.0, 70.0], [2.0, 0.0, 40.0]]},
             note=f"Scout v4 shape design (8 Nov, Spark: new props, no Bee35, everything printed in PCTG, still seats in the landing funnel for the battery swapper): "
                  f"Gemfan 5125 props on a printed frame, a half cage on the new discs; the battery mechanism (receiver, cartridge, pack, {BATTERY_SET[0]:.0f} g) not drawn: it mounts on 4 M3 bolts. {lo[0]:.0f}..{hi[0]:.0f} x {lo[1]:.0f}..{hi[1]:.0f} x {lo[2]:.0f}..{hi[2]:.0f} mm, "
                  f"~{mass:.0f} g (E). tools/scout_v4_shape.py",
             mass_budget_g=None, parts=rows)
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + "\n")
    print(f"  extents         {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} x {hi[2] - lo[2]:.0f} mm; all-up ~{mass:.0f} g (E): drawn {drawn:.0f} + the battery set {BATTERY_SET[0]:.0f} on its 4 bolts")
    print(f"  balance         CG ({cg[0]:+.2f}, {cg[1]:+.2f}, {cg[2]:.1f}) mm; the thrust centre (0, 0): {math.hypot(cg[0], cg[1]):.2f} off (masses E)")
    for r in sorted(rows, key=lambda r: -r["mass_g"])[:8]:
        print(f"    {r['mass_g']:6.1f} g  {r['name'][:70]}")
    print("  written: systems/scout_v4.assembly.json, parts/scout_v4_*/part.stl")
