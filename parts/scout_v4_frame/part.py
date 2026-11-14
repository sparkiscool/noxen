"""Scout v4 frame, ONE body (9 Nov, Spark: "design the whole thing including the thing the battery latch bolts onto as one body and make
sure the motor isnt going to fracture and snap off like it did on the overseer arms"): the deck, the four arms, the motor pods, the side
lugs and the base (the yokes the battery mechanism's 4 M3 bolt up into, the nose block, the flow sensor's pod, the tail tray) in one
PCTG print. It replaces scout_v4_frame + scout_v4_base_nose + scout_v4_base_tail (and their 12 deck bolts).
Fit: inside |x|, |y| <= 89.3 (tools/scout_v4_shape.BED, the A1 mini's 180 less 1.4): the pods' room r 15 at +-74.3, and the nose and
tail sockets moved in to x +-75.5 so their collars (13 toward NB / TB) end at 89.2. No nose bay: the FC, the Radxa, the XR2 and the DRV8833
moved under the deck over the receiver (tools/scout_v4_shape.py write()).
Print: deck-top down on the bed (build -z): the deck top, the four motor pads' tops and the turret bosses are one flat face at z 24.5.
So each pad and its arm are printed in the same layers: thrust, torque and the arm's bending all run in-layer from the pad into the arm
(the Overseer v9 arms broke under thrust at 0.30 printed flat, where the pad sat on a tall web: the joint under it was loaded across the
layers; here there is no layer seam between a pad and its arm). The drawn pad is the deck's 6 thick (O28, M2 x 8.5 that come with the
motor: 6 + 2.5 into its base), with room under it to z 8 (z 2 at the arm's root) for the engine to gusset it in the same layers.
Drawn (attachments only): the motor pads (4 x M2 on 12 x 12, O8 centre), the battery mechanism's 4 O10 insert bosses (M3 x 5.7 up from
below, on the receiver's top z 9.6), the turret's 4 O10 insert bosses (M3 from the top), the FC's 4 O8 posts (M3 inserts, the FC on
grommets under them, component side up) and the Radxa's 4 O8 posts (M2.5 inserts), the tape lands of the UBEC (over it, in its pod under the tail tray),
the XR2 and the DRV8833 (over them) and the flow sensor (over it, open below), the 8 rod sockets (DN BN DT BT DS0 BS0 DS1 BS1).
Cuts: the receiver's box under the yokes, the boards and their plugs, the props' discs + 12, the motors and their screws' drivers
from below, the turret above the deck, its N20's pocket and its gear's slot, the bolt holes, the rods beyond the mouths.
Support: the battery mechanism's 4 bolts (the receiver, the cartridge and the pack, 266.5 g, hang there; everything else is carried
to them).
Loads (each its own case), the motors from EMAX's ECO II 2004 table (dimensions.md: KV open, sized for the 3000KV, the highest thrust on
4S: 1090 g at 30.2 A with the Gemfan SL5125 at 16 V; x (16.8 / 16)^2 = 1.20 kg = 11.8 N on a full pack):
  flight, per motor: full thrust x 1.5 (DYN: punch-outs reach full thrust in ~50 ms and the ESC's rpm overshoot and the 3-blade's
    blade-pass at 1.4 kHz ride on it; with the flight class's own 4.0 the pad is checked at 6 x the full thrust) with the torque
    reaction at peak current (Kt = 60 / (2 pi KV) = 3.2 mN m / A, 40 A bursts = 0.13 N m) both ways, each with a nicked prop's rotating
    imbalance (0.02 g lost at r 60 at 28,700 rpm: m e w^2 = 11 N in the prop plane) along and across the arm;
  crash, per motor: a prop strike (the rotor stopped on a cage rod: the SL5125's polycarbonate blade root carries ~0.6 N m before it
    breaks, so 1.0 N m about the shaft with the 17 N blow at the blade's r 60 in the prop plane, both ways) with full thrust;
  the whole drone in 35 g crashes (14 directions) and flight 4 g / 2 g x / 2 g y / landing 6 g: every carried mass at its CG on its own
    attachments at once (motors + props, the probe, FC, Radxa, UBEC, XR2, DRV8833, flow), the body's own inertia with them;
  the ground hit on the receiver's feet (2 m g h / 15 mm from 0.6 m: 80 g on everything the 4 bolts carry);
  the head-first crash (the parked head is the highest point: 261 N down the throat, the frame model);
  the 8 sockets' frame-model forces (systems/scout_v4.cage_loads.json).
The motor's forces enter through its 4 screws (each a column of the pad round its screw: the head under the pad, the motor's base on
it), split rigidly from where they act (the prop plane for thrust, imbalance and the strike's blow)."""
import os
import numpy as np, trimesh
from topo import Part
from topo.part import man
from parts.scout_v4_units_lib import (S, ASM, box, cyl, along, U, unit, socks_of, socket_cases, socket_bodies, disc_cuts, region_top, region_bot,
                                      check_collars, rigid_split, SockDown, SockDown6, BraceSock, TURRET_BOLTS, REC_BOLTS, DZ0, DZ1, DRONE_G, G, HIT_N, M3_INS, M3_CLR,
                                      M25_INS, BOSS_R, DIRS14, FLIGHT)

SPEC = dict(h=2.0, gen_cm3=130.0, max_worst=0.3)
# 14 Nov, the fold of the bolt forks (rounds 7 / 7b, run of 8f97d87, scout_v4_frame_vl): 189.3 g, solid PASS, worst 0.211 (6 g landing on the
# nose bolts only), every motor case <= 0.20 in-layer / 0.17 across; latch v7's keep-out cut, the flow sensor forward, the boards on the grid.
# Rounds 1-6 (12-13 Nov) on the D / B cage: 0.46-0.83 from 150 g to 300 g, BT-TB's 550 N climb over the receiver's tail edge the floor   # E: the old frame's 36 + the units' 24 + 20, less the 12 deck bolts' pairs of bosses and the nose bay; PCTG at the nodes' 0.3
# 10 Nov (the NODES stream's socket-depth rule, units_lib.socket_depth): DN / DT 13.5 and BN / BT 11.5 deep (their 645 / 528 N), taken
# behind the standard floor (the mouths stay inside the 180; those 4 rods 3.5 / 1.5 longer), DS / BS 7.0 (their mouths 3 nearer).
# 10 Nov, run 186 (60 cm3, 18 % of the room): FAIL 1.515 (BN-NB 528 N), 117 g; the result read what the room filled solid thinned
# uniformly to the budget predicted (0.261 / 0.177 = 1.48): this many separate load paths leave the layout little to gain over
# uniform. Budget 60 -> 120 and the nose block |y| 26 -> 34 as the tail tray (the BN socket's room: 100 alone thinned to ~1.0), arms' room 26 -> 32 wide (motor 1's root 1.25 in-layer), the lugs' room x +-20 -> +-28
# (BS1-S1 1.28 across the layers: the lug bends as a column), the FC posts' clear ring r 6.5 -> 7.5 (one post off the main body).
# 11 Nov, run of a093523 (solid check PASS, 222.9 g): FAIL 0.821 ACROSS at BS0-S0 (175 N out along y at z -8.8, 33 under the deck: the lug
# x +-28 could only hang from the deck and bend across its layers), motors 0.35-0.65 in-layer, 0.25-0.55 across; the beta-4 check read
# 0.54 and the room thinned uniformly ~0.56: the layout gains nothing over uniform here. The lugs' wedge runs the receiver's length
# (a plate at the socket's height carries the pull along x in its layers to the nose and tail) and the budget 140 -> 180.
# Run of 071cc14: 274.1 g, solid PASS, FAIL 0.481 (motor 4's root; sockets <= 0.41 across, 0.40 in-layer): arms' root room 32 -> 48, 180 -> 200.
POD_R, POD_T, POD_ROOM = 14.0, 8.0, 15.0             # 11 Nov: pad 6 -> 8 (motors 0.29-0.53 in-layer, 0.33-0.39 across in run a25e31d); M2 x 10 screws
ARM_W0, ARM_W1 = 48.0, 30.0                         # the arms' room across, at the root and at the pod. 11 Nov, run of 071cc14 (274.1 g, 0.481): every
                                                   # pod's flight case worst at its root by the deck's corner (0.29-0.48 in-layer), each arm one ~10 wide bar
                                                   # in a 32 room: 48 at the root (was 32) lets the root spread into a V in its layers
POD_Z0 = DZ1 - 12.0                                 # 12 Nov (the design chat: <= 160 g, every fail by geometry): the arms' room a 12 deep band in the deck's layers,
                                                   # root to pod (was down to z 2 / 8): thrust bends a plate in its layers (a 10 x 34 section reads ~0.1 by hand), no
                                                   # web hung under the pad (run of 4b95423: motor 3 0.386 ACROSS at z 13.7 under its pad)
ARM_Z0 = POD_Z0 = DZ1 - 16.0                        # 13 Nov (variant vb of the round va / vb / vc): a flat 16 deep band, root to pod: 159.0 g, solid PASS, worst 0.59
                                                   # (BS0 compression across), every motor case <= 0.44, the engine's hold kept; the 12 band (va) 0.81 and the haunch at
                                                   # gen 130 (vc) 0.46 at 196 g. Was a haunch to z 6 at the root (0.83 across where its bottom chord met the body)
LATCH_V7 = True                                    # 14 Nov (the swap design chat): latch v7's Scout set (parts/latch7/result.json "scout_keepout": its receiver
                                                   # room, carrier and cap, the cartridge's path down, the lowered plug's tabs and its leads out the lead end) as
                                                   # the frame's keep-out in place of the receiver v3 box; the receiver's room x -48.3..63.1
FLOW_LO = None                                     # the flow sensor's low corner (x, z) when moved (it is 16 x 22.6 x 6.5 turned): None = its placed mesh
LANDING_ONE_END = True                             # the box's keys can hold one end only while a Scout slides down an end wall: 6 g on the nose bolts / the tail bolts
BRACES, BRACE_IN, BRACE_LO, BRACE_ROOM = False, 20.0, (42.0, 42.0, -8.0), 9.6   # the under-arm braces (off by default): the upper socket BRACE_IN
                                                   # inboard of the motor along the arm at the room's bottom, the lower at (+-x, +-y, z); room r round the rod
END_NODES, SIDE_NODES, TUBE_NODES = ("BN0", "BN1", "BT0", "BT1"), ("SA0", "SB0", "SA1", "SB1"), ()   # the frame's end sockets; those holding a 6 mm tube (6.2 bore, units_lib.SockDown6)
REC_SUPPORT = "insert"                              # 13 Nov round 5 (insert at 170 / 185 / 200: 246.7 / 265.5 / 284.2 g, 0.306 / 0.308 / 0.281)
                                                   # the battery bolts' support: "face" the boss's clamped bottom 1.5; "insert" with the insert's 6 mm
                                                   # bore lining (13 Nov: BT-TB's floor sits 3.6 over the face region's top, by the tail boss)
SIDE_Y1, FRONT_Z0 = 40.0, None                      # the rooms beside the receiver: out to |y| SIDE_Y1 (within the funnel's 55 deg profile); FRONT_Z0: their front
                                                   # part (x 20 to the nose block) down to that z (None: not)
                                                   # (12 Nov smoke: room filled solid 0.185 at motor 4's root with the 12 band)
PAD_T = 2.5                                         # the drawn motor seat: the pad's top skin (the motor's base on it); the 4 screw bosses the pad's depth
LUG_Y1 = 64.0                                      # the lugs' room: out to y 64 at the deck (9 Nov), along the receiver's whole length (11 Nov)
# the motor (EMAX ECO II 2004 3000KV, the table above; dimensions.md open item 19: the KV)
T_MAX, DYN, Q_PEAK, F_IMB, Q_STRIKE, F_STRIKE = 11.8, 1.5, 130.0, 11.0, 1000.0, 17.0   # N, -, N mm, N, N mm, N
M2_PATTERN = 6.0                                   # 12 x 12, 4 x M2
ROOT = S.ROOT


def bounds(pid):
    return trimesh.load(os.path.join(ROOT, "parts", pid, "part.stl"), force="mesh").bounds


def probe_mass_cg():
    """the probe's mass (assembly rows) at its CG from the shape design (tools/scout_v4_shape.probe(), mast()), not from the parts'
    generated meshes: a probe part's result landing must not change this setup (9 Nov: run 180 was superseded by the cradle's)"""
    m = dict(S.probe()); m["probe_mast_a"], m["probe_mast_b"] = S.mast()
    g, s = 0.0, np.zeros(3)
    for k, t in m.items():
        pid = "scout_v4_" + k
        if pid not in ASM:
            continue
        mg = ASM[pid]["mass_g"]
        c = t.center_mass if (t.is_watertight and abs(t.volume) > 1e-6) else t.bounds.mean(0)
        g += mg; s += mg * np.asarray(c)
    return g, s / g


def hull_pts(pts):
    return man(trimesh.convex.convex_hull(np.array(pts)))


def funnel_room():
    """the room the walls-in funnel leaves (tools/scout_v4_shape.zmin_walls_in + 2), in 1 mm steps along x"""
    xs = np.arange(-S.BED - 1.0, S.BED + 1.0, 1.0)
    return U([box(x, x + 1.0, -S.BED - 1, S.BED + 1, min(S.zmin_walls_in(x), S.zmin_walls_in(x + 1.0)) + 2.0, DZ1 + 1.0) for x in xs])


def latch_keepout():
    """latch v7's Scout keep-out meshes (drone frame) and its receiver room's bounds"""
    k = __import__("json").load(open(os.path.join(ROOT, "parts", "latch7", "result.json")))["scout_keepout"]
    ms = [man(trimesh.load(os.path.join(ROOT, v["mesh"]), force="mesh")) for n, v in k["items"].items()]
    return U(ms), k["items"]["receiver_room"]["lo"], k["items"]["receiver_room"]["hi"], k["items"]["leads_out_E"]


def flow_bounds():
    if FLOW_LO is None:
        return bounds("scout_v4_flow_mtf_02p")
    x, z = FLOW_LO
    return np.array([[x, -11.3, z], [x + 16.0, 11.3, z + 6.5]])


def grid_snap(lo, hi, bb, h):
    """[lo, hi] grown out to the run's cell boundaries (topo.Part's grid: pad 1, a multiple of 8 cells, centred on the room's box), per axis
    (14 Nov: a cut face part-way into a cell leaves the layout's last cell and the exact cut 0.3-1.7 apart, a stair-stepped torn web at the
    window over the boards, the design chat on vh)"""
    blo, bhi = np.asarray(bb[:3], float), np.asarray(bb[3:], float)
    n = np.ceil((bhi - blo) / h).astype(int) + 2; n = (np.ceil(n / 8) * 8).astype(int)
    o = 0.5 * (blo + bhi) - n * h / 2.0
    return o + np.floor((np.asarray(lo) - o) / h) * h, o + np.ceil((np.asarray(hi) - o) / h) * h


def screw_pts(mx, my, z):
    return [(mx + dx, my + dy, z) for dx in (-M2_PATTERN, M2_PATTERN) for dy in (-M2_PATTERN, M2_PATTERN)]


def bodies():
    x0, x1, yb = S.DECK
    rx0, rx1 = -51.3 + S.BX, 54.3 + S.BX                                             # the receiver v3 (C), not drawn: it bolts up under the yokes
    if LATCH_V7:
        ko, klo, khi, leads = latch_keepout(); rx0, rx1 = klo[0], khi[0]
    lb = flow_bounds()
    lx, ly0, ly1, lz = S.LUG
    yz = [(-ly0, lz), (ly0, lz), (ly1, lz), (ly1, lz + 1.2), (LUG_Y1, lz + 1.2 + 1.43 * (LUG_Y1 - ly1)), (LUG_Y1, DZ1), (-ly0, DZ1)]
    side_clip = U([hull_pts([(x, s_ * y, z) for x in (-S.BED, S.BED) for y, z in yz]) for s_ in (1, -1)])   # the side sockets' rooms stay inside the
    # lugs' old profile: under the discs' keep-out, the outer face up the funnel's 55 deg (12 Nov: the lug wedges and the belt along the receiver went
    # with BS moving to the nose block's corners, tools/scout_v4_shape.BS_X)
    env = [box(x0, x1, -yb, yb, S.REC_TOP, DZ1),                                      # the deck and the yokes over the receiver, one room
           box(-S.BED, rx0, -40.0, 40.0, -2.3, DZ1),                                  # the tail tray (DT / BT)
           box(-S.BED, rx0 + 0.5, -40.0, 40.0, -12.0, -2.0), box(rx1 - 0.5, 78.0, -40.0, 40.0, -12.0, -2.0),   # 12 Nov: the nose and tail blocks down to
           # -12 across |y| 40: BS0 / BS1 now sit in the nose block's corners (x 60, z -8.8) and pull in its layers; BT / BN turn the receiver's corners
           # in the same layers (run of 4b95423: BS1-S1 0.458 ACROSS, the lug hung from the deck mid-way along the receiver)
           box(rx1, S.BED, -40.0, 40.0, -2.3, DZ1), box(rx1, 78.0, -34.0, 34.0, -10.5, -2.0),   # the nose block (DN / BN) and its lower part to the flow pod (-10.5: into the flow land, not flush on it)
           box(rx1, lb[1][0] + 2.5, lb[0][1] - 2.5, lb[1][1] + 2.5, lb[0][2], -2.0),        # the flow sensor's pod (its bottom on the walls-in rule + 2)
           box(S.UBEC_LO[0] - 2.5, S.UBEC_LO[0] + 19.5, -33.0, 33.0, S.UBEC_LO[1], -2.0),   # 11 Nov: the UBEC's pod under the tail tray (S.UBEC_LO: off BT's path)
           (box(rx0 - 0.5, rx1 + 0.5, 33.8, SIDE_Y1, -2.3, S.REC_TOP + 0.5) ^ side_clip), (box(rx0 - 0.5, rx1 + 0.5, -SIDE_Y1, -33.8, -2.3, S.REC_TOP + 0.5) ^ side_clip)]   # 10 Nov: beside the receiver, so the end
    # sockets' loads climb to the battery bolts (z 9.6) on shallow in-layer diagonals; run 187 stopped early at 1.064 ACROSS the layers
    # (BT-TB 527 N at (-51, -33, 3.7)): with room only over the receiver and behind it, the path had to rise as a wall at its tail end.
    # 11 Nov (run of 0dd1778, the audit's parity check since): these rooms' tops flush under the deck's room (z 9.6) and the flow land
    # flush on the nose room's floor (z -10) stayed separate shells: 52 columns voxelised empty up to the deck (a 1 mm U slot round the
    # flow pod, 1.4 mm walls, two blobs on necks; the side rooms lost everything above 9.6 beside the receiver). Each now overlaps 0.5.
    for mx, my in S.MOTORS:
        a = np.array([np.sign(mx) * 30.0, np.sign(my) * 30.0, 0]); b = np.array([mx, my, 0])
        u = unit(b - a); n = np.array([-u[1], u[0], 0])
        env.append(hull_pts([p + s * n * w / 2 + [0, 0, z] for p, z0, w in ((a, ARM_Z0, ARM_W0), (b, POD_Z0, ARM_W1)) for s in (-1, 1) for z in (z0, DZ1)]))
        env.append(cyl(mx, my, POD_Z0, DZ1, POD_ROOM))
    if FRONT_Z0 is not None:
        env += [box(20.0, rx1 + 0.5, a_, b_, FRONT_Z0, -2.0) ^ side_clip for a_, b_ in ((33.8, SIDE_Y1), (-SIDE_Y1, -33.8))]
    drawn, cuts = [], []
    if LATCH_V7:                                                                      # latch v7's whole Scout set and the paths it needs
        cuts.append(ko)
        lx1, lz0, lz1 = leads["hi"][0], leads["lo"][2], leads["hi"][2]               # its 12 AWG pair leaves at x 75 (leads_out_E's end): forward,
        cuts += [box(lx1 - 6.0, lx1 + 0.5, -6.0, 6.0, lz0, 18.0), box(rx1 - 9.0, lx1 + 0.5, -6.0, 6.0, 13.0, 18.0)]   # up and back over the receiver's top to the FC stack, unclamped
    else:
        cuts.append(box(rx0 - 0.1, rx1, -33.3, 33.3, -60.0, -2.6))                   # the battery mechanism (the receiver's box) under the yokes; above
        rr = 5.0                                                                      # z -2.6 its vertical corners round r 5 (10 Nov, run 188: 0.785 at the box's
        cuts.append(hull_pts([(x, y, z) for x in (rx0 - 0.1 + rr, rx1 - rr) for y in (-33.3 + rr, 33.3 - rr) for z in (-2.7, S.REC_TOP)]))   # tail corner; the
        cuts += [cyl(x, y, -2.7, S.REC_TOP, rr) for x in (rx0 - 0.1 + rr, rx1 - rr) for y in (-33.3 + rr, 33.3 - rr)]   # receiver v3 mesh is empty there, checked)
        cuts += [box(rx0 - 0.1 + rr, rx1 - rr, -33.3, 33.3, -2.7, S.REC_TOP), box(rx0 - 0.1, rx1, -33.3 + rr, 33.3 - rr, -2.7, S.REC_TOP)]
    # ---- the motor pods: the pad in the deck's layers, 4 x M2 up from below (the heads under the pad), the bell over it
    pods, pod_regs = [], []
    for mx, my in S.MOTORS:
        pods.append(U([cyl(mx, my, DZ1 - PAD_T, DZ1, POD_R)] + [cyl(x, y, DZ1 - POD_T, DZ1, 3.5) for x, y, _ in screw_pts(mx, my, 0.0)]))   # 12 Nov: the seat
        # skin and the 4 screw bosses (O7, 2.4 round the M2's 2.2), the rest of the pad generated (was a solid O28 x 8 disc: 18 g of the 44 g drawn)
        pod_regs.append(([cyl(x, y, DZ1 - POD_T, DZ1 - POD_T + 1.5, 2.6) for x, y, _ in screw_pts(mx, my, 0.0)],    # each screw head's seat on the pad's underside
                         [cyl(x, y, DZ1 - 1.5, DZ1, 3.5) for x, y, _ in screw_pts(mx, my, 0.0)]))                 # and the motor base on the pad's top round it
        cuts.append(cyl(mx, my, ARM_Z0 - 1, DZ1 + 1, 4.0))
        for x, y, _ in screw_pts(mx, my, 0.0):
            cuts.append(cyl(x, y, ARM_Z0 - 1, DZ1 + 1, 1.1))                          # M2 clearance
            cuts.append(cyl(x, y, ARM_Z0 - 30, DZ1 - POD_T, 2.6))                     # its head and the driver from below
        cuts.append(cyl(mx, my, DZ1, DZ1 + 60, POD_R + 1.0))                         # the motor
    drawn += pods
    # ---- the battery mechanism's 4 bolts: M3 up from below into inserts in O10 bosses on the receiver's top (the support)
    rec_regs = []
    for x, y in REC_BOLTS:
        drawn.append(cyl(x, y, S.REC_TOP, DZ0, BOSS_R))
        reg = region_bot(x, y, S.REC_TOP)                                             # the clamped face (the boss's bottom 1.5 on the receiver's top)
        if REC_SUPPORT == "insert":                                                   # and the insert's length: the M3's shear goes in along its 6 mm
            reg = U([reg, cyl(x, y, S.REC_TOP, S.REC_TOP + M3_INS[1], M3_INS[0] + 1.5) - cyl(x, y, S.REC_TOP - 1, S.REC_TOP + M3_INS[1] + 1, M3_INS[0])])
        rec_regs.append(reg)
        cuts.append(cyl(x, y, S.REC_TOP - 1, S.REC_TOP + M3_INS[1], M3_INS[0]))
    # ---- the turret's bosses: M3 x 5.7 inserts from the top (PROBE's interface)
    tur, tur_regs = [], []
    for x, y in TURRET_BOLTS:
        tur.append(cyl(x, y, DZ0, DZ1, BOSS_R)); tur_regs.append(region_top(x, y, DZ1))
        cuts.append(cyl(x, y, DZ1 - M3_INS[1], DZ1 + 1, M3_INS[0])); cuts.append(cyl(x, y, DZ0 - 20, DZ1 - M3_INS[1] - 0.6, M3_CLR))
    drawn += tur
    cx, cz = S.SPOOL["c"]; g0 = S.N20_FACE; pz = cz - S.GEAR["c"]
    cuts.append(box(cx - 6.6, cx + 6.6, g0 - 27.6, g0 + 0.6, pz - 5.6, DZ1 + 1))      # the N20's pocket through the deck
    ra = (S.GEAR["spool_t"] / 2 + 1) * S.GEAR["m"]
    cuts.append(box(cx - ra - 0.8, cx + ra + 0.8, S.GEAR["y"][0] - 0.5, S.GEAR["y"][1] + 0.5, cz - ra - 0.8, DZ1 + 1))   # the 40 T gear's slot
    tb = S.probe()["probe_turret"].bounds
    cuts.append(box(tb[0][0] - 0.5, tb[1][0] + 0.5, tb[0][1] - 0.5, tb[1][1] + 0.5, DZ1, 120))   # the turret above the deck
    import parts.scout_v4_probe_turret.part as TUR                                    # 11 Nov (the PROBE stream's turret, main 722f7dc): its N20 cheeks,
    tt = TUR.bodies(); tdown = ((tt["drawn"] + tt["env"]) - tt["cuts"]) ^ box(-200, 200, -200, 200, -100, DZ1 - 0.01)   # what the turret prints there (its cuts win)   # stubs and roller guide hang below its floor pads
    cuts.append(U([tdown.translate(list(d)) for d in ((0.4, 0, 0), (-0.4, 0, 0), (0, 0.4, 0), (0, -0.4, 0), (0, 0, -0.4), (0, 0, 0))]) ^ box(-200, 200, -200, 200, -100, DZ1))   # 0.4 clear, never above the deck top (the bosses keep their height)
    cuts += disc_cuts()
    # ---- the boards (bought meshes placed by tools/scout_v4_shape.py)
    boards = []
    fb = bounds("scout_v4_fc"); fx, fy = fb[:, 0].mean(), fb[:, 1].mean(); ftop = fb[0][2] + 5.6 + 1.0   # board top (4 under it, 1.6 board) + a grommet
    posts, pts, regs = [], [], []
    for dx in (-12.75, 12.75):
        for dy in (-12.75, 12.75):                                                    # 4 posts on its 25.5 holes (MicoAir spec), down from the deck
            posts.append(cyl(fx + dx, fy + dy, ftop, DZ1, 4.0)); pts.append((fx + dx, fy + dy, ftop)); regs.append(region_bot(fx + dx, fy + dy, ftop, 4.0))
            cuts.append(cyl(fx + dx, fy + dy, ftop - 0.1, ftop + M3_INS[1], M3_INS[0]))
    drawn += posts
    fc_rings = U([cyl(x, y, ftop - 1, DZ1 + 1, 7.5) for x, y, _ in pts])
    board_cuts = [((fb[0][0] - 0.5, fb[0][1] - 0.5, fb[0][2] - 0.5), (fb[1][0] + 0.5, fb[1][1] + 0.5, ftop), None),   # the FC, under it and its grommets,
                  ((fb[0][0] - 0.5, fb[0][1] - 0.5, ftop), (fb[1][0] + 0.5, fb[1][1] + 0.5, fb[1][2] + 0.5), fc_rings)]   # its top components but not within 6.5
    # of a hole (E: the grommets' zone): each post's room runs out past the board's edge; their y faces on the grid (grid_snap, at the end)
    boards.append(("FC MicoAir743", pts, regs, ASM["scout_v4_fc"]["mass_g"], tuple(fb.mean(0))))
    rb = bounds("scout_v4_radxa_zero_3w"); rx, ry = rb[:, 0].mean(), rb[:, 1].mean(); rtop = rb[0][2] + 3.2   # its PCB's top (1.55 under it, 1.6 board)
    posts, pts, regs = [], [], []
    for dx in (-28.925, 28.925):
        for dy in (-11.425, 11.425):                                                  # 4 M2.5 posts (Radxa STEP: 57.85 x 22.85), down from the deck
            posts.append(cyl(rx + dx, ry + dy, rtop, DZ1, 3.75)); pts.append((rx + dx, ry + dy, rtop)); regs.append(region_bot(rx + dx, ry + dy, rtop, 3.75))   # 3.75: 2.0 round the insert
            cuts.append(cyl(rx + dx, ry + dy, rtop - 0.1, rtop + M25_INS[1], M25_INS[0]))
    drawn += posts
    board_cuts.append(((rb[0][0] - 0.2, rb[0][1] - 0.5, rb[0][2] - 0.5), (rb[1][0] + 2.4, rb[1][1] + 0.5, rb[1][2] + 0.5), U(posts)))   # 0.2 at its tail end (DT's collar); its front end
    # 2.4 on, into the turret's cut (13 Nov round 5: at gen 185 / 200 the engine filled the 1.9 gap between the two cuts, a 1.5-1.7 wall the gate refused)
    for xa, xb in ((rb[1][0] - 46.0, rb[1][0] - 37.1),):                         # one USB-C plug out the -y edge, the camera's (12 Nov: the board runs on the UBEC
        # through its 5 V pins, so the other port's cut, x -67..-58 across motor 4's arm root, went: room filled solid 0.270 there; it flashes on this one)
        cuts.append(box(xa - 1.0, xb + 1.0, -41.0, rb[0][1] + 1.0, rb[0][2] - 0.5, rb[0][2] + 7.0))
    boards.append(("Radxa Zero 3W", pts, regs, ASM["scout_v4_radxa_zero_3w"]["mass_g"], tuple(rb.mean(0))))
    for pid, nm, below in (("scout_v4_ubec", "UBEC", False), ("scout_v4_xr2", "XR2", False), ("scout_v4_drv8833", "DRV8833", False)):
        b = bounds(pid)                                                               # tape lands (2.6: the gate's floor) over each: the UBEC hangs under its land in its pod (11 Nov)
        if below:
            drawn.append(box(b[0][0], b[1][0], b[0][1], b[1][1], b[0][2] - 2.6, b[0][2])); reg = box(b[0][0], b[1][0], b[0][1], b[1][1], b[0][2] - 2.6, b[0][2])
        else:
            drawn.append(box(b[0][0], b[1][0], b[0][1], b[1][1], b[1][2] + 0.5, b[1][2] + 3.1)); reg = box(b[0][0], b[1][0], b[0][1], b[1][1], b[1][2] + 0.5, b[1][2] + 3.1)
        cuts.append(box(b[0][0] - 0.5, b[1][0] + 0.5, b[0][1] - 0.5, b[1][1] + 0.5, b[0][2] - (0.0 if below else 0.5), b[1][2] + 0.5))
        boards.append((nm, [tuple(b.mean(0))], [reg], ASM[pid]["mass_g"], tuple(b.mean(0))))
    ub = bounds("scout_v4_ubec")
    cuts += [box(ub[:, 0].mean() - 4, ub[:, 0].mean() + 4, ub[0][1] - 4, ub[0][1], ub[0][2] - 0.5, ub[1][2]), box(ub[:, 0].mean() - 4, ub[:, 0].mean() + 4, ub[1][1], ub[1][1] + 3, ub[0][2] - 0.5, ub[1][2])]   # its leads out its ends, in the pod's height only (11 Nov: up to the deck they cut the tray's sides through BT's path)
    drawn.append(box(lb[0][0] - 1.5, lb[1][0] + 1.5, lb[0][1] - 1.5, lb[1][1] + 1.5, lb[1][2], lb[1][2] + 2.6))   # the flow sensor's tape land over it
    cuts.append(box(lb[0][0] - 0.3, lb[1][0] + 0.3, lb[0][1] - 0.3, lb[1][1] + 0.3, lb[0][2] - 5, lb[1][2]))      # its body and window, open below
    cuts.append(box(lb[0][0] - 0.3, lb[0][0] + 4, -3, 3, lb[1][2] - 3, S.REC_TOP + 0.3))                          # its cable up the receiver's nose face
    reg = box(lb[0][0], lb[1][0], lb[0][1], lb[1][1], lb[1][2], lb[1][2] + 2.6)   # (the whole land: h 2.25)
    boards.append(("flow MTF-02P", [tuple(lb.mean(0))], [reg], ASM["scout_v4_flow_mtf_02p"]["mass_g"], tuple(lb.mean(0))))
    flow_reg = box(lb[0][0] - 1.5, lb[1][0] + 1.5, lb[0][1] - 1.5, lb[1][1] + 1.5, lb[1][2], lb[1][2] + 2.6)   # the whole land
    # ---- the 8 rod sockets (crowns down: the print's up)
    side = [s for k in SIDE_NODES for s in socks_of(k, cls=(SockDown6 if k in TUBE_NODES else SockDown))]
    ends = [s for k in END_NODES for s in socks_of(k, cls=(SockDown6 if k in TUBE_NODES else SockDown))]
    socks = side + ends
    d, c = socket_bodies(side, lambda s: np.array([1.0, 0.0, 0.0]))                      # ports out of the lugs along x (9 Nov: out the side they cut the crown, down in print)
    drawn += d; cuts += c
    d, c = socket_bodies(ends, lambda s: np.array([0.0, (np.sign(s.c[1]) if abs(s.c[1]) > 1.0 else (1.0 if s.c[2] > 10 else -1.0)), 0.0]))   # off the centre line: out its own side
    drawn += d; cuts += c
    from parts.scout_units_lib import BORE_R as _BR, WALL as _W
    env += [along(s.pt(-8.0), s.pt(s.t_mouth), getattr(s, "R", _BR) + _W + 5.0) for s in ends]       # room round each end socket and back into the body
    env += [along(s.pt(-8.0), s.pt(s.t_mouth), _BR + _W + 5.0) ^ side_clip for s in side]   # and round each side socket (DS in the deck, BS in the nose block)
    env += [along(s.pt(-8.0), np.array([s.c[0] - np.sign(s.c[0]) * 25.0, 0.0, 10.0]), 7.0) for s in ends]
    braces = []
    if BRACES:                                                                        # 13 Nov (Spark: "if adding a rod to the bottom of the motors can
        for i, (mx, my) in enumerate(S.MOTORS):                                       # prevent snapping we should do that"): per arm a 3.0 carbon rod from
            sx, sy = np.sign(mx), np.sign(my)                                         # a socket under the arm 20 inboard of the motor (clear of its 4 M2
            a = np.array([mx - sx * BRACE_IN / np.sqrt(2), my - sy * BRACE_IN / np.sqrt(2), POD_Z0])   # drivers and its O8 centre) down and in to a
            b = np.array([sx * BRACE_LO[0], sy * BRACE_LO[1], BRACE_LO[2]])          # socket beside the receiver: the arm a triangle, thrust in the rod
            up, lo = BraceSock(f"brace {i + 1}a", a, b), BraceSock(f"brace {i + 1}b", b, a)
            drawn += [up.collar(), lo.collar()]
            for s_ in (up, lo):
                cuts.append(s_.port(np.array([0.0, 0.0, -1.0]) if s_ is up else np.array([sx, sy, 0.0]) / np.sqrt(2), [lo if s_ is up else up]))
                cuts += s_.bores()
            env.append(along(up.pt(-6.0), lo.pt(-6.0), BRACE_ROOM))
            env.append(along(lo.pt(-6.0), np.array([sx * 40.0, sy * 35.0, 0.0]), 7.0))   # and from the lower socket into the room beside the receiver
            p0, p1 = up.pt(up.t_floor + 2.0), lo.pt(lo.t_floor + 2.0)                 # the rod as an insert, 2 short of each floor (a pin: the voxel rod's
            braces.append((f"brace {i + 1}", along(p0, p1, 2.0), float(np.linalg.norm(p1 - p0))))   # bending stiffness is overstated)
            socks += [up, lo]
    env = U(env) ^ box(-S.BED, S.BED, -S.BED, S.BED, -60.0, DZ1) ^ funnel_room()     # the A1 mini's 180; nothing below the walls-in funnel's line
    bb = np.asarray((env + U(drawn)).bounding_box(), float)                          # the boards' cuts with their long (y) faces on the run's grid
    for lo, hi, minus in board_cuts:
        (sl, sh) = grid_snap(lo, hi, bb, SPEC["h"]); lo2 = (lo[0], sl[1], lo[2]); hi2 = (hi[0], sh[1], hi[2])
        c_ = box(lo2[0], hi2[0], lo2[1], hi2[1], lo2[2], hi2[2]); cuts.append(c_ - minus if minus is not None else c_)
    check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    return dict(drawn=U(drawn) ^ box(-S.BED - 5, S.BED + 5, -S.BED - 5, S.BED + 5, -60.0, DZ1), cuts=U(cuts), env=env, flow_reg=flow_reg,   # the bed: DN / DT's deeper collars rise to 24.8 at their back (10 Nov)
                pod_regs=pod_regs, rec_regs=rec_regs, tur_regs=tur_regs, socks=[s_ for s_ in socks if not isinstance(s_, BraceSock)], boards=boards, braces=braces)


def whole_drone(p, masses, name, a, cls, accel=None, support="default"):
    """every carried mass at acceleration a (g, the drone's frame) on its own attachments, one case"""
    parts = []
    for nm, pts, regs, mg, cg in masses:
        f = rigid_split(pts, -mg * G * np.asarray(a, float), cg)
        parts += [(regs[i], f[i]) for i in range(len(pts))]
    p.load_multi(name, parts, cls=cls, accel=-np.asarray(a, float) if accel is None else accel, support=support)


def motor_parts(regs, mx, my, F, at, Q=0.0):
    """a force F acting at `at` and a torque Q about the motor's axis (+z), into the pad through its 4 screws: the pull away from the
    pad (+z) at each screw head's seat on its underside (the head bears up into the pad), the rest (lateral, torque, push) at the motor
    base on its top round each screw (11 Nov: a column through the pad's thickness pulled its layers apart, an across-layer artefact)"""
    pts = screw_pts(mx, my, DZ1 - POD_T / 2)
    f = rigid_split(pts, np.asarray(F, float), np.asarray(at, float))
    for i, (x, y, _) in enumerate(pts):
        r = np.array([x - mx, y - my, 0.0]); t = np.cross([0, 0, 1.0], r) / np.linalg.norm(r)
        f[i] = f[i] + Q / (4 * np.linalg.norm(r)) * t
    bot, top = regs
    out = []
    for i in range(4):
        up = max(0.0, f[i][2])
        out += [(bot[i], np.array([0.0, 0.0, up])), (top[i], f[i] - np.array([0.0, 0.0, up]))]
    return out


def setup(h):
    B = bodies()
    p = Part("scout_v4_frame", h, B["drawn"], B["cuts"], B["env"], build="-z", material="PCTG")
    p.support("default", U(B["rec_regs"]))
    for nm, reg, L in B["braces"]:                                                    # carbon 110 GPa / PCTG 2.5 = 44, the 3.0 rod's 7.07 mm2
        p.insert(nm, reg, 110000.0 / 2500.0, area_mm2=np.pi * 1.5 ** 2, length_mm=L)
    zp = S.Z_PROP
    T = T_MAX * DYN
    for i, (mx, my) in enumerate(S.MOTORS):
        regs = B["pod_regs"][i]; at = (mx, my, zp); u = unit(np.array([mx, my, 0.0])); v = np.cross([0, 0, 1.0], u)
        for qs in (1, -1):
            for lab, e in ((("across", v),) if qs > 0 else (("along", u),)):   # 10 Nov: 2 a pod, not 4 (run 189 died on the runner at ~16 GB;
                p.load_multi(f"motor {i + 1}: thrust {T:.1f} N (1.20 kg x {DYN}) + torque {qs * Q_PEAK / 1000:+.2f} N m + imbalance {F_IMB:.0f} N {lab} the arm",
                             motor_parts(regs, mx, my, (0, 0, T) + F_IMB * e, at, qs * Q_PEAK), cls="flight")
            p.load_multi(f"motor {i + 1}: prop strike {qs * Q_STRIKE / 1000:+.1f} N m + {F_STRIKE:.0f} N blow + thrust {T_MAX:.1f} N",
                         motor_parts(regs, mx, my, (0, 0, T_MAX) + qs * F_STRIKE * v, at, -qs * Q_STRIKE), cls="crash")
    masses = []
    for i, (mx, my) in enumerate(S.MOTORS):
        mg = ASM[f"scout_v4_motor_{i + 1}"]["mass_g"] + ASM[f"scout_v4_blades_{i + 1}"]["mass_g"]
        masses.append((f"motor {i + 1}", screw_pts(mx, my, DZ1 - POD_T / 2), B["pod_regs"][i][1], mg, (mx, my, DZ1 + 11.0)))
    g, cg = probe_mass_cg()
    tpts = [(x, y, DZ1) for x, y in TURRET_BOLTS]
    masses.append(("probe", tpts, B["tur_regs"], g, cg))
    masses += B["boards"]
    for k, d in enumerate(DIRS14):
        whole_drone(p, masses, f"crash {k} (35 g {np.round(d, 2).tolist()}): every carried mass", 35.0 * d, "crash")
    for nm, a in FLIGHT.items():
        whole_drone(p, masses, f"{nm}: every carried mass", a, "flight")
    if LANDING_ONE_END:                                                               # 14 Nov (the swap design chat): sliding down an end wall a Scout
        for lab, regs_ in (("nose", B["rec_regs"][:2]), ("tail", B["rec_regs"][2:])):  # rests one end on two of the box's keys until the walls back off
            p.support(lab, U(regs_))                                                  # (21.5 N a key at 6 g, through the receiver's foot blocks into the bolts)
            whole_drone(p, masses, f"landing 6 g on the {lab} bolts only (one end on the box's keys): every carried mass", FLIGHT["landing 6 g"], "flight", support=lab)
    a_hit = HIT_N / (DRONE_G * G)
    whole_drone(p, masses, f"ground hit on the receiver's feet ({a_hit:.0f} g on all the bolts carry)", (0, 0, a_hit), "crash")
    f = rigid_split(tpts, np.array([0, 0, -261.0]), np.array([S.AXIS[0], 0.0, 130.0]))
    p.load_multi("head-first crash: 261 N down the throat", [(B["tur_regs"][i], f[i]) for i in range(4)], cls="crash")
    socket_cases(p, B["socks"])
    for lab, F in (("up", (0, 0, 10.0)), ("side", (0, 10.0, 0))):                     # 10 Nov (run 188: the land on 1.8 mm necks): the frame's lowest, most
        p.load(f"handling: 10 N {lab} on the flow sensor's pod (a hand)", B["flow_reg"], F, cls="misuse")   # exposed printed feature, picked by hand
    return p
