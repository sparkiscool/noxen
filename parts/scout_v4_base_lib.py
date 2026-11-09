"""Scout v4 base units (9 Nov, Spark: "generate everything with topology"): the base under the deck, split at x = BX into a nose unit and
a tail unit (183 mm long in one piece; the A1 mini prints 180). Each bolts up to the deck's flat underside (M3 down from the deck's top
into its inserts) and carries:
  nose: the battery mechanism's two front bolts (M3 up into its inserts), the FC on four M3 posts (25.5 square), the MTF-02P on a tape
        land in the flow pod (open below for its window), and the nose node's two rod sockets DN / BN;
  tail: the battery mechanism's two rear bolts, the Radxa Zero 3W hung under the tail tray's roof on four M2.5 posts (57.85 x 22.85;
        under it the UBEC and the XR2 lie flat on tape lands), and the tail node's two rod sockets DT / BT.
Drawn: only those bosses, posts, lands and collars. Cuts: the boards and their plugs, the battery mechanism's volume below the yokes
(the receiver's box), the N20's pocket, the bolt holes, the rods beyond the mouths. Room: tools/scout_v4_shape.base() (every bottom on
the walls-in funnel + 2), cut at the split.
Support: the unit's four deck bolts. Loads: its share of the battery set (266.5 g, not drawn, at its CG; by the bolts' lever about it)
in 35 g crashes and flight, and its half of the ground hit up through its two battery bolts (the Overseer's rule); the boards; the rod
sockets' frame-model forces."""
import os
import numpy as np, trimesh
from topo import Part
from topo.part import man
from parts.scout_v4_units_lib import (S, ASM, box, cyl, along, U, disc_cuts, socks_of, socket_cases, socket_bodies, payload_cases, region_top, region_bot,
                                      check_collars, BASE_BOLTS, REC_BOLTS, SPLIT_X, DZ0, HIT_N, M3_INS, M3_CLR, M25_INS, BOSS_R)

ROOT = S.ROOT


def bounds(pid):
    return trimesh.load(os.path.join(ROOT, "parts", pid, "part.stl")).bounds


def bodies(unit):
    nose = unit == "nose"
    env = man(S.base())
    half = box(SPLIT_X + 0.5, 300, -100, 100, -100, 100) if nose else box(-300, SPLIT_X - 0.5, -100, 100, -100, 100)
    if nose:                                                                          # the nose bay beyond the deck's end: room over and beside the FC (its 36 square fills the bay's 44),
        env = env + box(57.0, 104.0, -30.0, 30.0, -2.3, 27.0) + box(57.0, 78.0, -30.0, 30.0, -10.0, 1.0)   # and under it down toward the flow pod (the funnel's line allows -14 at x 78)
    env = env ^ half
    drawn, cuts = [], []
    bolts = [b for b in BASE_BOLTS if (b[0] > SPLIT_X) == nose]
    top_regs = []
    for x, y in bolts:                                                                # the deck bolts: bosses up to the deck's underside, M3 x 5.7 inserts from the top
        drawn.append(cyl(x, y, DZ0 - 7.0, DZ0, BOSS_R)); top_regs.append(region_top(x, y, DZ0))
        cuts.append(cyl(x, y, DZ0 - M3_INS[1], DZ0 + 1, M3_INS[0])); cuts.append(cyl(x, y, DZ0 - 30, DZ0 - M3_INS[1] - 0.6, M3_CLR))
    rec = [b for b in REC_BOLTS if (b[0] > SPLIT_X) == nose]
    rec_regs, rec_pts = [], []
    for x, y in rec:                                                                  # the battery mechanism's bolts: M3 up from below into inserts
        drawn.append(cyl(x, y, S.REC_TOP, DZ0, BOSS_R)); rec_regs.append(region_bot(x, y, S.REC_TOP)); rec_pts.append((x, y, S.REC_TOP))   # up to the deck: 2.9 over the insert
        cuts.append(cyl(x, y, S.REC_TOP - 1, S.REC_TOP + M3_INS[1], M3_INS[0]))
    cuts.append(box(-51.4 + S.BX, 54.3 + S.BX, -33.3, 33.3, -60, S.REC_TOP))           # the battery mechanism (the receiver's box) under the yokes
    cx, cz = S.SPOOL["c"]; g0 = S.N20_FACE; pz = cz - (S.GEAR["spool_t"] + S.GEAR["pinion_t"]) * S.GEAR["m"] / 2
    cuts.append(box(cx - 6.6, cx + 6.6, g0 - 27.6, g0 + 0.6, pz - 5.6, DZ0 + 1))        # the turret's N20 in its pocket
    boards, socks = [], []
    if nose:
        fb = bounds("scout_v4_fc"); fx, fy = fb[:, 0].mean(), fb[:, 1].mean(); fz = fb[0][2]
        cuts.append(box(fb[0][0] - 0.5, fb[1][0] + 0.5, fb[0][1] - 0.5, fb[1][1] + 0.5, fz, fb[1][2] + 4.0))   # the FC and its plugs' room above it
        pts, regs = [], []
        for dx in (-12.75, 12.75):
            for dy in (-12.75, 12.75):                                               # four M3 posts (MicoAir spec 25.5 square), the soft-mount grommets on them
                drawn.append(cyl(fx + dx, fy + dy, -2.3, fz, 3.5)); cuts.append(cyl(fx + dx, fy + dy, fz - 4.0, fz + 1, 1.6))
                pts.append((fx + dx, fy + dy, fz)); regs.append(region_top(fx + dx, fy + dy, fz, 3.5))
        boards.append(("FC MicoAir743", pts, regs, ASM["scout_v4_fc"]["mass_g"], (fx, fy, (fz + fb[1][2]) / 2)))
        lb = bounds("scout_v4_flow_mtf_02p")
        drawn.append(box(lb[0][0] - 1.5, lb[1][0] + 1.5, lb[0][1] - 1.5, lb[1][1] + 1.5, lb[1][2], lb[1][2] + 2.6))   # the flow sensor's tape land over it (2.6: the gate's floor)
        cuts.append(box(lb[0][0] - 0.3, lb[1][0] + 0.3, lb[0][1] - 0.3, lb[1][1] + 0.3, lb[0][2] - 5, lb[1][2]))      # its body and window, open below
        cuts.append(box(lb[0][0] - 0.3, lb[0][0] + 4, -3, 3, lb[1][2] - 3, lb[1][2] + 12))                             # its cable up into the bay
        reg = box(lb[0][0], lb[1][0], lb[0][1], lb[1][1], lb[1][2], lb[1][2] + 0.6); c = (lb[0] + lb[1]) / 2
        boards.append(("flow MTF-02P", [tuple(c)], [reg], ASM["scout_v4_flow_mtf_02p"]["mass_g"], tuple(c)))
        for pid, nm, sy in (("scout_v4_xr2", "XR2", 1), ("scout_v4_drv8833", "DRV8833", -1)):   # on edge beside the FC: a side land behind each (tape)
            b = bounds(pid); yo = b[1][1] if sy > 0 else b[0][1]
            drawn.append(box(b[0][0], b[1][0], yo + sy * 0.5, yo + sy * 3.7, b[0][2], b[1][2]))   # 3.2 beyond the board's 0.5 clearance
            cuts.append(box(b[0][0] - 0.5, b[1][0] + 0.5, b[0][1] - 0.5, b[1][1] + 0.5, b[0][2] - 0.5, b[1][2] + 3.0))
            reg = box(b[0][0], b[1][0], yo + sy * 0.5, yo + sy * 1.1, b[0][2], b[1][2]); c = (b[0] + b[1]) / 2
            boards.append((nm, [tuple(c)], [reg], ASM[pid]["mass_g"], tuple(c)))
        socks = socks_of("DN") + socks_of("BN")
    else:
        rb_ = bounds("scout_v4_radxa_zero_3w"); rx, ry = rb_[:, 0].mean(), rb_[:, 1].mean(); rz = rb_[1][2]
        rz0 = rb_[0][2]
        cuts.append(box(rb_[0][0] - 0.5, rb_[1][0] + 0.5, rb_[0][1] - 0.5, rb_[1][1] + 0.5, rz0, rz + 0.5))      # the Radxa (its parts in the mesh's height)
        cuts.append(box(rb_[0][0] - 0.5, rb_[1][0] + 0.5, rb_[1][1], rb_[1][1] + 30, rz0, rz))                  # its USB-C plugs out the +y edge
        pts, regs = [], []
        for dx in (-11.425, 11.425):
            for dy in (-28.925, 28.925):                                             # four M2.5 posts hanging down onto its top (Radxa STEP: 57.85 x 22.85): at the rear tray's
                drawn.append(cyl(rx + dx, ry + dy, rz + 0.5, rz + 7.0, 4.0))           # floor (the funnel's -2.3) nothing under it is thicker than a member; over its corners there is room beside the UBEC
                cuts.append(cyl(rx + dx, ry + dy, rz - 1, rz + 0.5 + M25_INS[1], M25_INS[0])); cuts.append(cyl(rx + dx, ry + dy, -40, rz0, 2.6))   # the insert; the screw up from below and its driver
                pts.append((rx + dx, ry + dy, rz + 0.5)); regs.append(region_bot(rx + dx, ry + dy, rz + 0.5, 4.0))
        boards.append(("Radxa Zero 3W", pts, regs, ASM["scout_v4_radxa_zero_3w"]["mass_g"], (rx, ry, (rz0 + rz) / 2)))
        ub = bounds("scout_v4_ubec")                                                   # the UBEC is taped to the deck's underside (the frame's land): keep its room and leads clear
        cuts.append(box(ub[0][0] - 0.5, ub[1][0] + 0.5, ub[0][1] - 0.5, ub[1][1] + 0.5, ub[0][2] - 0.5, DZ0 + 0.1))
        cuts += [box(ub[0][0] - 8, ub[0][0], -4, 4, ub[0][2], DZ0), box(ub[1][0], ub[1][0] + 6, -4, 4, ub[0][2], DZ0)]   # its leads out its two ends (along x)
        socks = socks_of("DT") + socks_of("BT")
    d, c = socket_bodies(socks, lambda s: np.array([0.0, 1.0 if s.c[2] > 10 else -1.0, 0.0]))
    drawn += d; cuts += c
    from parts.scout_units_lib import BORE_R as _BR, WALL as _W
    env = env + U([along(s.pt(-8.0), s.pt(s.t_mouth), _BR + _W + 5.0) for s in socks] +           # room round each socket and back into the unit
                  [along(s.pt(-8.0), np.array([s.c[0] - np.sign(s.c[0]) * 25.0, 0.0, 10.0]), 7.0) for s in socks])
    lb = bounds("scout_v4_flow_mtf_02p")
    floor = box(-300, 300, -100, 100, -2.3, 200) + (box(54.3 + S.BX, 70.5 + S.BX, -12.0, 12.0, -19.1, -2.2) + box(57.0, 78.0, -30.0, 30.0, -10.0, -2.2) if nose else box(0, 0.1, 0, 0.1, 0, 0.1))
    if not nose:                                                                      # behind the deck's end the tail unit may rise to the props' keep-out (27.7) and route over the Radxa
        env = env + box(-110.0, -60.0, -34.0, 34.0, DZ0 - 1.0, 27.0)
    env = env ^ floor ^ half                                                          # nothing below the walls-in funnel's line (-2.3 past the receiver's ends; the flow pod as tools/scout_v4_shape.base())
    cuts += disc_cuts()
    check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    return dict(drawn=U(drawn), cuts=U(cuts), env=env, top_regs=top_regs, rec_regs=rec_regs, rec_pts=rec_pts, boards=boards, socks=socks)


def setup(unit, h):
    B = bodies(unit)
    p = Part(f"scout_v4_base_{unit}", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG")
    p.support("default", U(B["top_regs"]))
    g, (bx, by, bz) = S.BATTERY_SET[0], S.BATTERY_SET[1]; bcg = np.array([bx + S.BX, by, bz])
    xs = [b[0] for b in REC_BOLTS]; xn, xt = max(xs), min(xs)
    share = (bcg[0] - xt) / (xn - xt) if unit == "nose" else (xn - bcg[0]) / (xn - xt)
    payload_cases(p, f"battery set ({share:.2f} of {g:.0f} g)", B["rec_pts"], B["rec_regs"], g * share, bcg)
    for i, r in enumerate(B["rec_regs"]):
        p.load(f"ground hit: {HIT_N / 4:.0f} N up through battery bolt {i + 1}", r, (0, 0, HIT_N / 4), cls="crash")
    for nm, pts, regs, mg, cg in B["boards"]:
        payload_cases(p, nm, pts, regs, mg, cg)
    socket_cases(p, B["socks"])
    return p
