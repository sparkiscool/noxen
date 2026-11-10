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
grommets under them, component side up) and the Radxa's 4 O8 posts (M2.5 inserts), the tape lands of the UBEC (the tail tray's floor),
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
                                      check_collars, rigid_split, SockDown, TURRET_BOLTS, REC_BOLTS, DZ0, DZ1, DRONE_G, G, HIT_N, M3_INS, M3_CLR,
                                      M25_INS, BOSS_R, DIRS14, FLIGHT)

SPEC = dict(h=2.0, gen_cm3=120.0, max_worst=0.3)   # E: the old frame's 36 + the units' 24 + 20, less the 12 deck bolts' pairs of bosses and the nose bay; PCTG at the nodes' 0.3
# 10 Nov (the NODES stream's socket-depth rule, units_lib.socket_depth): DN / DT 13.5 and BN / BT 11.5 deep (their 645 / 528 N), taken
# behind the standard floor (the mouths stay inside the 180; those 4 rods 3.5 / 1.5 longer), DS / BS 7.0 (their mouths 3 nearer).
# 10 Nov, run 186 (60 cm3, 18 % of the room): FAIL 1.515 (BN-NB 528 N), 117 g; the result read what the room filled solid thinned
# uniformly to the budget predicted (0.261 / 0.177 = 1.48): this many separate load paths leave the layout little to gain over
# uniform. Budget 60 -> 120 and the nose block |y| 26 -> 34 as the tail tray (the BN socket's room: 100 alone thinned to ~1.0), arms' room 26 -> 32 wide (motor 1's root 1.25 in-layer), the lugs' room x +-20 -> +-28
# (BS1-S1 1.28 across the layers: the lug bends as a column), the FC posts' clear ring r 6.5 -> 7.5 (one post off the main body).
POD_R, POD_T, POD_ROOM, ARM_W = 14.0, 6.0, 15.0, 32.0
ARM_Z0, POD_Z0 = 2.0, 8.0                            # the arms' room bottom at the root and at the pod (9 Nov smoke: room_full 0.33 at the rear arm's root, 12.5 deep)
LUG_X, LUG_Y1 = 28.0, 64.0                         # the lugs' room: x +-20, out to y 64 at the deck (9 Nov)
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


def screw_pts(mx, my, z):
    return [(mx + dx, my + dy, z) for dx in (-M2_PATTERN, M2_PATTERN) for dy in (-M2_PATTERN, M2_PATTERN)]


def bodies():
    x0, x1, yb = S.DECK
    rx0, rx1 = -51.3 + S.BX, 54.3 + S.BX                                             # the receiver v3 (C), not drawn: it bolts up under the yokes
    env = [box(x0, x1, -yb, yb, S.REC_TOP, DZ1),                                      # the deck and the yokes over the receiver, one room
           box(-S.BED, rx0, -34.0, 34.0, -2.3, DZ1),                                  # the tail tray (DT / BT, the UBEC)
           box(rx1, S.BED, -34.0, 34.0, -2.3, DZ1), box(rx1, 78.0, -34.0, 34.0, -10.0, -2.0),   # the nose block (DN / BN) and its lower part to the flow pod
           box(rx1, 71.8, -12.0, 12.0, -19.1, -2.0),                                 # the flow sensor's pod (its bottom on the walls-in rule + 2)
           box(rx0 - 0.5, rx1 + 0.5, 33.8, 40.0, -2.3, S.REC_TOP), box(rx0 - 0.5, rx1 + 0.5, -40.0, -33.8, -2.3, S.REC_TOP)]   # 10 Nov: beside the receiver, so the end
    # sockets' loads climb to the battery bolts (z 9.6) on shallow in-layer diagonals; run 187 stopped early at 1.064 ACROSS the layers
    # (BT-TB 527 N at (-51, -33, 3.7)): with room only over the receiver and behind it, the path had to rise as a wall at its tail end
    for mx, my in S.MOTORS:
        a = np.array([np.sign(mx) * 30.0, np.sign(my) * 30.0, 0]); b = np.array([mx, my, 0])
        u = unit(b - a); n = np.array([-u[1], u[0], 0])
        env.append(hull_pts([p + s * n * ARM_W / 2 + [0, 0, z] for p, z0 in ((a, ARM_Z0), (b, POD_Z0)) for s in (-1, 1) for z in (z0, DZ1)]))
        env.append(cyl(mx, my, POD_Z0, DZ1, POD_ROOM))
    lx, ly0, ly1, lz = S.LUG
    for s_ in (1, -1):                                                                # the side lugs' room: a wedge from the deck's side down to the
        yz = [(ly0, lz), (ly1, lz), (ly1, lz + 1.2), (LUG_Y1, lz + 1.2 + 1.43 * (LUG_Y1 - ly1)), (LUG_Y1, DZ1), (ly0, DZ1)]   # BS socket, out under the
        env.append(hull_pts([(x, s_ * y, z) for x in (-LUG_X, LUG_X) for y, z in yz]))   # discs' keep-out, its outer face up the funnel's 55 deg
    drawn, cuts = [], []
    cuts.append(box(rx0 - 0.1, rx1, -33.3, 33.3, -60.0, S.REC_TOP))                  # the battery mechanism (the receiver's box) under the yokes
    # ---- the motor pods: the pad in the deck's layers, 4 x M2 up from below (the heads under the pad), the bell over it
    pods, pod_regs = [], []
    for mx, my in S.MOTORS:
        pods.append(cyl(mx, my, DZ1 - POD_T, DZ1, POD_R))
        pod_regs.append([cyl(x, y, DZ1 - POD_T, DZ1, 2.6) for x, y, _ in screw_pts(mx, my, 0.0)])   # each screw's column of the pad
        cuts.append(cyl(mx, my, ARM_Z0 - 1, DZ1 + 1, 4.0))
        for x, y, _ in screw_pts(mx, my, 0.0):
            cuts.append(cyl(x, y, ARM_Z0 - 1, DZ1 + 1, 1.1))                          # M2 clearance
            cuts.append(cyl(x, y, ARM_Z0 - 30, DZ1 - POD_T, 2.6))                     # its head and the driver from below
        cuts.append(cyl(mx, my, DZ1, DZ1 + 60, POD_R + 1.0))                         # the motor
    drawn += pods
    # ---- the battery mechanism's 4 bolts: M3 up from below into inserts in O10 bosses on the receiver's top (the support)
    rec_regs = []
    for x, y in REC_BOLTS:
        drawn.append(cyl(x, y, S.REC_TOP, DZ0, BOSS_R)); rec_regs.append(region_bot(x, y, S.REC_TOP))
        cuts.append(cyl(x, y, S.REC_TOP - 1, S.REC_TOP + M3_INS[1], M3_INS[0]))
    # ---- the turret's bosses: M3 x 5.7 inserts from the top (PROBE's interface)
    tur, tur_regs = [], []
    for x, y in TURRET_BOLTS:
        tur.append(cyl(x, y, DZ0, DZ1, BOSS_R)); tur_regs.append(region_top(x, y, DZ1))
        cuts.append(cyl(x, y, DZ1 - M3_INS[1], DZ1 + 1, M3_INS[0])); cuts.append(cyl(x, y, DZ0 - 20, DZ1 - M3_INS[1] - 0.6, M3_CLR))
    drawn += tur
    cx, cz = S.SPOOL["c"]; g0 = S.N20_FACE; pz = cz - (S.GEAR["spool_t"] + S.GEAR["pinion_t"]) * S.GEAR["m"] / 2
    cuts.append(box(cx - 6.6, cx + 6.6, g0 - 27.6, g0 + 0.6, pz - 5.6, DZ1 + 1))      # the N20's pocket through the deck
    ra = (S.GEAR["spool_t"] / 2 + 1) * S.GEAR["m"]
    cuts.append(box(cx - ra - 0.8, cx + ra + 0.8, S.GEAR["y"][0] - 0.5, S.GEAR["y"][1] + 0.5, cz - ra - 0.8, DZ1 + 1))   # the 40 T gear's slot
    tb = S.probe()["probe_turret"].bounds
    cuts.append(box(tb[0][0] - 0.5, tb[1][0] + 0.5, tb[0][1] - 0.5, tb[1][1] + 0.5, DZ1, 120))   # the turret above the deck
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
    cuts.append(box(fb[0][0] - 0.5, fb[1][0] + 0.5, fb[0][1] - 0.5, fb[1][1] + 0.5, fb[0][2] - 0.5, ftop))   # the board, under it and its grommets
    cuts.append(box(fb[0][0] - 0.5, fb[1][0] + 0.5, fb[0][1] - 0.5, fb[1][1] + 0.5, ftop, fb[1][2] + 0.5)      # its top components, but not within 6.5 of
                - U([cyl(x, y, ftop - 1, DZ1 + 1, 7.5) for x, y, _ in pts]))                                        # a hole (E: the grommets' zone): each post's room runs out past the board's edge
    boards.append(("FC MicoAir743", pts, regs, ASM["scout_v4_fc"]["mass_g"], tuple(fb.mean(0))))
    rb = bounds("scout_v4_radxa_zero_3w"); rx, ry = rb[:, 0].mean(), rb[:, 1].mean(); rtop = rb[0][2] + 3.2   # its PCB's top (1.55 under it, 1.6 board)
    posts, pts, regs = [], [], []
    for dx in (-28.925, 28.925):
        for dy in (-11.425, 11.425):                                                  # 4 M2.5 posts (Radxa STEP: 57.85 x 22.85), down from the deck
            posts.append(cyl(rx + dx, ry + dy, rtop, DZ1, 4.0)); pts.append((rx + dx, ry + dy, rtop)); regs.append(region_bot(rx + dx, ry + dy, rtop, 4.0))
            cuts.append(cyl(rx + dx, ry + dy, rtop - 0.1, rtop + M25_INS[1], M25_INS[0]))
    drawn += posts
    cuts.append(box(rb[0][0] - 0.5, rb[1][0] + 0.5, rb[0][1] - 0.5, rb[1][1] + 0.5, rb[0][2] - 0.5, rb[1][2] + 0.5) - U(posts))
    for xa, xb in ((rb[1][0] - 58.5, rb[1][0] - 49.6), (rb[1][0] - 46.0, rb[1][0] - 37.1)):   # its two USB-C plugs out the -y edge
        cuts.append(box(xa - 1.0, xb + 1.0, -41.0, rb[0][1] + 1.0, rb[0][2] - 0.5, rb[0][2] + 7.0))
    boards.append(("Radxa Zero 3W", pts, regs, ASM["scout_v4_radxa_zero_3w"]["mass_g"], tuple(rb.mean(0))))
    for pid, nm, below in (("scout_v4_ubec", "UBEC", True), ("scout_v4_xr2", "XR2", False), ("scout_v4_drv8833", "DRV8833", False)):
        b = bounds(pid)                                                               # tape lands (2.6: the gate's floor): the UBEC on one, the XR2 and the DRV8833 under theirs
        if below:
            drawn.append(box(b[0][0], b[1][0], b[0][1], b[1][1], b[0][2] - 2.6, b[0][2])); reg = box(b[0][0], b[1][0], b[0][1], b[1][1], b[0][2] - 0.6, b[0][2])
        else:
            drawn.append(box(b[0][0], b[1][0], b[0][1], b[1][1], b[1][2] + 0.5, b[1][2] + 3.1)); reg = box(b[0][0], b[1][0], b[0][1], b[1][1], b[1][2] + 0.5, b[1][2] + 1.1)
        cuts.append(box(b[0][0] - 0.5, b[1][0] + 0.5, b[0][1] - 0.5, b[1][1] + 0.5, b[0][2] - (0.0 if below else 0.5), b[1][2] + 0.5))
        boards.append((nm, [tuple(b.mean(0))], [reg], ASM[pid]["mass_g"], tuple(b.mean(0))))
    ub = bounds("scout_v4_ubec")
    cuts += [box(ub[:, 0].mean() - 4, ub[:, 0].mean() + 4, ub[0][1] - 4, ub[0][1], ub[0][2], DZ0), box(ub[:, 0].mean() - 4, ub[:, 0].mean() + 4, ub[1][1], ub[1][1] + 3, ub[0][2], DZ0)]   # its leads out its ends
    lb = bounds("scout_v4_flow_mtf_02p")
    drawn.append(box(lb[0][0] - 1.5, lb[1][0] + 1.5, lb[0][1] - 1.5, lb[1][1] + 1.5, lb[1][2], lb[1][2] + 2.6))   # the flow sensor's tape land over it
    cuts.append(box(lb[0][0] - 0.3, lb[1][0] + 0.3, lb[0][1] - 0.3, lb[1][1] + 0.3, lb[0][2] - 5, lb[1][2]))      # its body and window, open below
    cuts.append(box(lb[0][0] - 0.3, lb[0][0] + 4, -3, 3, lb[1][2] - 3, S.REC_TOP + 2.0))                          # its cable up the receiver's nose face
    reg = box(lb[0][0], lb[1][0], lb[0][1], lb[1][1], lb[1][2], lb[1][2] + 0.6)
    boards.append(("flow MTF-02P", [tuple(lb.mean(0))], [reg], ASM["scout_v4_flow_mtf_02p"]["mass_g"], tuple(lb.mean(0))))
    # ---- the 8 rod sockets (crowns down: the print's up)
    side = [s for k in ("DS0", "BS0", "DS1", "BS1") for s in socks_of(k, cls=SockDown)]
    ends = [s for k in ("DN", "BN", "DT", "BT") for s in socks_of(k, cls=SockDown)]
    socks = side + ends
    d, c = socket_bodies(side, lambda s: np.array([1.0, 0.0, 0.0]))                      # ports out of the lugs along x (9 Nov: out the side they cut the crown, down in print)
    drawn += d; cuts += c
    d, c = socket_bodies(ends, lambda s: np.array([0.0, 1.0 if s.c[2] > 10 else -1.0, 0.0]))
    drawn += d; cuts += c
    from parts.scout_units_lib import BORE_R as _BR, WALL as _W
    env += [along(s.pt(-8.0), s.pt(s.t_mouth), _BR + _W + 5.0) for s in ends]       # room round each end socket and back into the body
    env += [along(s.pt(-8.0), np.array([s.c[0] - np.sign(s.c[0]) * 25.0, 0.0, 10.0]), 7.0) for s in ends]
    env = U(env) ^ box(-S.BED, S.BED, -S.BED, S.BED, -60.0, DZ1) ^ funnel_room()     # the A1 mini's 180; nothing below the walls-in funnel's line
    check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    return dict(drawn=U(drawn) ^ box(-S.BED - 5, S.BED + 5, -S.BED - 5, S.BED + 5, -60.0, DZ1), cuts=U(cuts), env=env,   # the bed: DN / DT's deeper collars rise to 24.8 at their back (10 Nov)
                pod_regs=pod_regs, rec_regs=rec_regs, tur_regs=tur_regs, socks=socks, boards=boards)


def whole_drone(p, masses, name, a, cls, accel=None):
    """every carried mass at acceleration a (g, the drone's frame) on its own attachments, one case"""
    parts = []
    for nm, pts, regs, mg, cg in masses:
        f = rigid_split(pts, -mg * G * np.asarray(a, float), cg)
        parts += [(regs[i], f[i]) for i in range(len(pts))]
    p.load_multi(name, parts, cls=cls, accel=-np.asarray(a, float) if accel is None else accel)


def motor_parts(regs, mx, my, F, at, Q=0.0):
    """a force F acting at `at` and a torque Q about the motor's axis (+z), into the pad through its 4 screws' columns"""
    pts = screw_pts(mx, my, DZ1 - POD_T / 2)
    f = rigid_split(pts, np.asarray(F, float), np.asarray(at, float))
    for i, (x, y, _) in enumerate(pts):
        r = np.array([x - mx, y - my, 0.0]); t = np.cross([0, 0, 1.0], r) / np.linalg.norm(r)
        f[i] = f[i] + Q / (4 * np.linalg.norm(r)) * t
    return [(regs[i], f[i]) for i in range(4)]


def setup(h):
    B = bodies()
    p = Part("scout_v4_frame", h, B["drawn"], B["cuts"], B["env"], build="-z", material="PCTG")
    p.support("default", U(B["rec_regs"]))
    zp = S.Z_PROP
    T = T_MAX * DYN
    for i, (mx, my) in enumerate(S.MOTORS):
        regs = B["pod_regs"][i]; at = (mx, my, zp); u = unit(np.array([mx, my, 0.0])); v = np.cross([0, 0, 1.0], u)
        for qs in (1, -1):
            for lab, e in (("along", u), ("across", v)):
                p.load_multi(f"motor {i + 1}: thrust {T:.1f} N (1.20 kg x {DYN}) + torque {qs * Q_PEAK / 1000:+.2f} N m + imbalance {F_IMB:.0f} N {lab} the arm",
                             motor_parts(regs, mx, my, (0, 0, T) + F_IMB * e, at, qs * Q_PEAK), cls="flight")
            p.load_multi(f"motor {i + 1}: prop strike {qs * Q_STRIKE / 1000:+.1f} N m + {F_STRIKE:.0f} N blow + thrust {T_MAX:.1f} N",
                         motor_parts(regs, mx, my, (0, 0, T_MAX) + qs * F_STRIKE * v, at, -qs * Q_STRIKE), cls="crash")
    masses = []
    for i, (mx, my) in enumerate(S.MOTORS):
        mg = ASM[f"scout_v4_motor_{i + 1}"]["mass_g"] + ASM[f"scout_v4_blades_{i + 1}"]["mass_g"]
        masses.append((f"motor {i + 1}", screw_pts(mx, my, DZ1 - POD_T / 2), B["pod_regs"][i], mg, (mx, my, DZ1 + 11.0)))
    g, cg = probe_mass_cg()
    tpts = [(x, y, DZ1) for x, y in TURRET_BOLTS]
    masses.append(("probe", tpts, B["tur_regs"], g, cg))
    masses += B["boards"]
    for k, d in enumerate(DIRS14):
        whole_drone(p, masses, f"crash {k} (35 g {np.round(d, 2).tolist()}): every carried mass", 35.0 * d, "crash")
    for nm, a in FLIGHT.items():
        whole_drone(p, masses, f"{nm}: every carried mass", a, "flight")
    a_hit = HIT_N / (DRONE_G * G)
    whole_drone(p, masses, f"ground hit on the receiver's feet ({a_hit:.0f} g on all the bolts carry)", (0, 0, a_hit), "crash")
    f = rigid_split(tpts, np.array([0, 0, -261.0]), np.array([S.AXIS[0], 0.0, 130.0]))
    p.load_multi("head-first crash: 261 N down the throat", [(B["tur_regs"][i], f[i]) for i in range(4)], cls="crash")
    socket_cases(p, B["socks"])
    return p
