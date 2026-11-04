"""Scout tail unit (v1 setup, 4 Nov): ONE printed PETG piece under the plate's tail that holds the receiver's tail end, seats the
Scout in the shared funnel, carries the tail electronics (Radxa Zero 3W, UBEC, XR2), and is cage node TB with the two tail tie
sockets (the frame model: without a stiff tail link a 35 g tail-first wall hit racked the cage 17 mm).
Drawn (attachments only): 4 plate bolt pads (M3 x 5.7 inserts, drilled holes at (-30, +-14) and (-55, +-20)), 2 receiver bosses,
the tail runner's seat rim and 55 deg lead-ins, 4 Radxa posts (O8, M2.5 x 4 inserts, z 0.5..5.9 under the PCB), the UBEC and XR2
tape lands, the socket collars of TB (rods to B1, B2, P3, P4) and of ties C1 / C2. Cuts: the receiver and its clearance, the plate,
the boards with their plugs (USB-C / HDMI out the right edge, the SD card out the tail, the UBEC's leads, the XR2's antenna), the
rods beyond their mouths, the insert holes. Loads: the receiver set's tail share (155 g E) and the boards (32 g E) in 35 g crashes
(14 directions) and flight; a 35 g landing into the funnel on the seat and the lead-ins; every socket's frame-model force (F).
Support: the plate pads. Geometry and numbers: parts/scout_units_lib.py."""
import numpy as np
from topo import Part
from parts import scout_units_lib as L
from parts.scout_units_lib import box, cyl, along, U

SPEC = dict(h=2.0, gen_cm3=40.0, max_worst=0.5)      # E: tail bay mount v2 49 g + a runner; refine after the first run. h 2.0: 123k cells (287k at 1.5: the direct solver took 8.8 GB and was killed; the CI runner has 7 GB)


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
    env += [box(-60.0, -28.5, -34.0, 34.0, L.RZ1, L.PLATE_Z), box(-62.0, L.RX0 - 0.5, -33.4, 33.4, L.SEAT_Z, L.PLATE_Z),
            box(L.CX - L.SEAT_L / 2, L.RX0 - 1.0, -33.4, 33.4, L.SEAT_Z + L.RUN_H, 0.5), box(-120.0, -52.0, -36.0, 36.0, 0.0, 7.5),
            box(-140.0, -118.0, -14.0, 14.0, 0.0, 24.0)]
    # Radxa on 4 posts (M2.5 inserts), the board and its plugs cut; SD card out the tail end; USB-C / HDMI out the -y edge
    posts, post_pts, post_regs = [], [], []
    for x, y in L.RADXA_HOLES:
        posts.append(cyl(x, y, L.RPOST_Z0, L.RADXA_PCB_BOT, L.RPOST_R)); post_pts.append((x, y, L.RADXA_PCB_BOT))
        cuts.append(cyl(x, y, L.RADXA_PCB_BOT - L.RINS_D, L.RADXA_PCB_BOT + 1.0, L.RINS_R))
        post_regs.append(cyl(x, y, L.RADXA_PCB_BOT - 2.0, L.RADXA_PCB_BOT + 0.3, L.RPOST_R))
    drawn += posts
    cuts.append(box(-118.5, -52.5, -17.0, 15.5, 2.9, L.RADXA_TOP + 0.5) - U(posts))
    cuts.append(box(-140.0, -118.0, -8.0, 8.0, 2.0, 7.0))                                 # the microSD card
    cuts.append(box(-111.0, -58.0, -40.0, -14.5, 7.0, 12.0))                               # USB-C x2 and micro HDMI plugs
    # UBEC and XR2 on tape lands, their bodies and leads cut
    x0, x1, y0, y1, z0, z1 = L.UBEC
    drawn.append(box(x0, x1, y0, y1, z0 - 2.0, z0)); cuts.append(box(x0 - 6.0, x1 + 6.0, y0 - 0.5, y1 + 0.5, z0, z1 + 1.0))
    x0, x1, y0, y1, z0, z1 = L.XR2
    drawn.append(box(x0, x1, y0, y1, z0 - 2.0, z0)); cuts.append(box(x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5, z0, z1 + 0.5))
    cuts.append(box(-135.0, x0, (y0 + y1) / 2 - 2.5, (y0 + y1) / 2 + 2.5, z0, z1))        # the antenna, flat along -x
    cuts += L.disc_cuts()
    socks = L.node_socks("TB", ["B1", "B2", "P3", "P4"], nodes) + [L.Sock("C1 C1-B1", nodes["C1"], nodes["B1"]), L.Sock("C2 C2-B2", nodes["C2"], nodes["B2"])]
    for s in socks:
        drawn.append(s.collar()); cuts += s.bores(); cuts.append(s.clear())
        env.append(along(s.pt(-6.0), s.pt(s.t_mouth), L.BORE_R + L.WALL + 5.0))
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
