"""Scout tail unit (v1 setup, 4 Nov): ONE printed PETG piece under the plate's tail that holds the receiver's tail end, seats the
Scout in the shared funnel, carries the tail electronics in a bay over the runner (Radxa Zero 3W on edge, UBEC flat under it, XR2; 4 Nov, Spark: into the printed base, every
screw reachable: the Radxa's four from the open tail end),
and is cage node TB with the two tail tie
sockets (the frame model: without a stiff tail link a 35 g tail-first wall hit racked the cage 17 mm).
Drawn (attachments only): 3 plate bolt pads (M3 x 5.7 inserts: the FC diamond's tail hole, a drilled pair at (-55, +-13)), 2 receiver bosses,
the tail runner's seat rim and 55 deg lead-ins, 4 Radxa posts (O8 along +x behind the PCB, M2.5 x 4 inserts), the UBEC's and XR2's
4 mm tape lands, the socket collars of TB (rods to B1, B2, P3, P4) and of ties C1 / C2. Cuts: the receiver and its clearance, the plate,
the boards with their plugs (USB-C / HDMI out the right edge, the SD card out the tail, the UBEC's leads, the XR2's antenna), the
rods beyond their mouths, the insert holes. Loads: the receiver set's tail share (155 g E) and the boards (32 g E) in 35 g crashes
(14 directions) and flight; a 35 g landing into the funnel on the seat and the lead-ins; every socket's frame-model force (F).
Support: the plate pads. Geometry and numbers: parts/scout_units_lib.py."""
import numpy as np
from topo import Part
from parts import scout_units_lib as L
from parts.scout_units_lib import box, cyl, along, U

SPEC = dict(h=2.0, gen_cm3=50.0, max_worst=0.5)      # E: tail bay mount v2 49 g + a runner. Run 2 (4 Nov): gate FAIL, one neck over the UBEC lead slot in the yoke; index 1.16 at the tray's step past the wall room in the funnel catch; 80.9 g. Run 3 (PASS, 0.27, 80.6 g): the FC on top of the plate, the yoke to x -8, the electronics on edge in the bay over the runner (no tray), the tail node in to (-112, 0, 10). Run 4: the Radxa raised to z -15..15 with the UBEC flat under it so its screws come in from the tail end (driver columns cut). h 2.0: 123k cells (287k at 1.5: the direct solver took 8.8 GB and was killed; the CI runner has 7 GB). Run 1 (4 Nov, 40 cm3): gate FAIL, two necks at the 2 mm tape lands' ends; index 2.34 in TB-B1 tension under the tail pad (-55, -20): the tray hung on a 10 wide wall slot. Run 2: lands 4, the wall room to x -66, 50 cm3


def bodies():
    nodes, rods, ties = L.cage_nodes()
    drawn, cuts, env = [], [], []
    pads, c, pad_regs = L.plate_pads(L.BOLT_TAIL); drawn += pads; cuts += c
    bosses, c, boss_regs = L.receiver_bosses(L.INS_TAIL); drawn += bosses; cuts += c
    cuts.append(L.receiver_cut())
    cuts.append(box(-70, 67, -70, 70, L.PLATE_Z, L.PLATE_Z + 4.0))                       # the plate
    rim, block, seat_regs, wedge = L.runner(L.CX - L.SEAT_L / 2, L.RX0 - 1.0, nose=False)
    drawn.append(rim); env.append(block)
    # the body: yoke, the wall down the receiver's tail face, a deck over the runner up to the tray, the tray, the TB block
    env += [box(-60.0, L.YOKE_SPLIT_X, -34.0, 34.0, L.RZ1, L.PLATE_Z), box(-62.0, -40.0, -41.0, 41.0, 7.0, L.PLATE_Z), box(-66.0, L.RX0 - 0.5, -33.4, 33.4, L.SEAT_Z, L.PLATE_Z),   # the yoke to the split at x -8, its tail corners out to |y| 41 for the tie sockets, the wall room to x -66
            box(-100.0, L.RX0 - 1.0, -33.4, 33.4, L.SEAT_Z + L.RUN_H, 0.5), box(-100.0, -66.0, -33.4, 33.4, 0.5, 18.0),   # the bay over the runner, to x -100 (the funnel's tail face is 4.5 further at h 20) and up to z 18 behind the plate's tail (the Radxa stands to z 15)
            box(-120.0, -96.0, -14.0, 14.0, -2.0, 20.0)]                                                                  # the arm to the tail node TB (-112, 0, 10)
    # the bay over the tail runner: Radxa on edge on 4 posts along +x (M2.5 inserts), its screws' driver columns from the open tail end, the
    # board, its plugs (up) and the SD card (out -y) cut; the UBEC flat on a land under it, the XR2 on a land ahead of it; leads and antenna cut
    posts, post_pts, post_regs = [], [], []
    for y, z in L.RADXA_HOLES:
        q = np.array([L.RADXA_X, y, z]); q0 = q + [L.RPOST_L, 0, 0]
        posts.append(along(q, q0, L.RPOST_R)); post_pts.append(tuple(q))
        cuts.append(along(q - [1.0, 0, 0], q + [L.RINS_D, 0, 0], L.RINS_R))
        cuts.append(along(q - [L.DRIVER_L, 0, 0], q - [1.0, 0, 0], L.DRIVER_R))          # the screwdriver's way in
        post_regs.append(along(q - [0.3, 0, 0], q + [2.0, 0, 0], L.RPOST_R))
    drawn += posts
    cuts.append(box(*L.RADXA_CUT) - U(posts))
    cuts.append(box(*L.RADXA_PLUGS)); cuts.append(box(*L.RADXA_SD))
    x0, x1, y0, y1, z0, z1 = L.UBEC
    drawn.append(box(x0, x1, y0, y1, z0 - L.LAND_T, z0)); cuts.append(box(x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5, z0, z1 + 0.5))
    cw, ch, cl = L.LEAD_CH; cx_, cz_ = (x0 + x1) / 2, (z0 + z1) / 2
    cuts += [box(cx_ - cw / 2, cx_ + cw / 2, y0 - cl, y0, cz_ - ch / 2, cz_ + ch / 2), box(cx_ - cw / 2, cx_ + cw / 2, y1, y1 + cl, cz_ - ch / 2, cz_ + ch / 2)]   # the leads out both ends
    x0, x1, y0, y1, z0, z1 = L.XR2
    drawn.append(box(x1, x1 + L.LAND_T, y0, y1, z0, z1)); cuts.append(box(x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5, z0 - 0.5, z1 + 0.5))
    cuts.append(box(*L.XR2_ANT))                                                       # the antenna, flat along +y
    cuts += L.disc_cuts()
    socks = L.node_socks("TB", ["B1", "B2", "P3", "P4"], nodes) + [L.Sock("C1 C1-B1", nodes["C1"], nodes["B1"]), L.Sock("C2 C2-B2", nodes["C2"], nodes["B2"])]
    for s in socks:
        drawn.append(s.collar()); cuts += s.bores(); cuts.append(s.clear())
        env.append(along(s.pt(-6.0), s.pt(s.t_mouth), L.BORE_R + L.WALL + 5.0))
    L.check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    return dict(drawn=U(drawn), cuts=U(cuts), env=U(env), pad_regs=pad_regs, boss_regs=boss_regs, seat_regs=seat_regs,
                post_pts=post_pts, post_regs=post_regs, socks=socks)


def setup(h):
    B = bodies()
    p = Part("scout_tail_unit", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG")
    p.support("default", U(B["pad_regs"]))
    L.payload_cases(p, "receiver set", [(x, y, L.RZ1) for x, y in L.INS_TAIL], B["boss_regs"], L.REC_SET_G, L.REC_SET_CG, "default")
    L.payload_cases(p, "boards", B["post_pts"], B["post_regs"], L.BOARDS_G, L.BOARDS_CG, "default")
    L.catch_cases(p, B["seat_regs"], False, "default")
    L.socket_cases(p, B["socks"], "default")
    return p
