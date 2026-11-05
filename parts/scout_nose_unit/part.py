"""Scout nose unit (v1 setup, 4 Nov): ONE printed PETG piece under the plate's nose that holds the receiver's nose end, seats the
Scout in the shared funnel, carries the camera (30 deg up since 4 Nov) and the flow sensor, and is cage node NB with the two nose tie sockets.
Drawn (attachments only): 6 plate bolt pads (M3 x 5.7 inserts, all Bee35 holes: the nose bolt, the FC diamond's front three, the (3.9, +-31) pair; the FC sits on top of the plate since 4 Nov and its standoff bolts go on through the plate into the yoke), 2 receiver bosses
(inserts, the receiver's bolts come up through its keys), the nose runner's seat rim and 55 deg lead-ins (what the funnel meets),
the flow sensor's tape land over its window, 4 camera M2 posts (O5.6 x 10, 1.2 wall: the one thin exception), the socket collars
of NB (rods to B0, B3, P0, P7) and of ties C0 / C3. Cuts: the receiver and its clearance, the plate, the camera body, plugs and view,
the flow board, window and lead channel, the front discs grown 12 (cage rule), the rods beyond their mouths, the insert holes.
Loads: the receiver set's nose share (155 g E) in 35 g crashes (14 directions) and flight; the camera (10 g) the same; a 35 g
landing into the funnel on the seat and the lead-ins; every socket's frame-model force (systems/scout.cage_loads.json, F).
Support: the plate pads. Build +z? No: the runner's rim must print flat: build -z is not a printer option here, so the part prints
on its plate pads' faces (+z = the pad faces down... the pads are the top): build "+z" with the seat rim up is wrong; the runner is
the largest flat face: this part prints runner-down, which is build "-z" in the engine's terms (the part's -z face on the bed).
Geometry and numbers: parts/scout_units_lib.py."""
import numpy as np
from topo import Part
from parts import scout_units_lib as L
from parts.scout_units_lib import box, cyl, along, U

SPEC = dict(h=2.2, gen_cm3=55.0, max_worst=0.5)      # E: belly mount v1 25 g + camera mount v1 22 g + a runner. Run 3 (h 2.3, 50): gate PASS 79.1 g, index 0.62 under the (36, 0) pad in the funnel catch. Run 4: h 2.2, 55 cm3, the camera 30 deg up: gate FAIL, the lower camera posts' stubs loose (room); index 0.42, 84 g. Run 5: the post block to the board's back and |y| 24: gate FAIL again, the same two loose stubs, index 0.67, 82 g. Run 6: a O10 x 4 root boss behind each post. h 2.3 since the yoke reaches x -8 (FC on top): at 2.0 the grid is 194k cells and the direct solve peaked at 6.2 GB here (the CI runner has 7 GB); at 2.3 it is 147k. Run 1 (4 Nov, 40 cm3): gate PASS 60.4 g, index 0.60 in the funnel catch (seat up) at the nose wall's top (56.5, 19, 9): the wall room was 7 wide. Run 2: the wall room to x 66, 50 cm3. h 2.0: the grid (102 x 91 x 104, padded to 8s) is 332k cells at 1.5 (refused), 229k at 1.7, 150k at 2.0: the direct solver took 8.8 GB on the tail unit's 287k cells and was killed (the CI runner has 7 GB; the hub deck runs at 184k)


def bodies():
    nodes, rods, ties = L.cage_nodes()
    drawn, cuts, env = [], [], []
    pads, c, pad_regs = L.plate_pads(L.BOLT_NOSE); drawn += pads; cuts += c
    bosses, c, boss_regs = L.receiver_bosses(L.INS_NOSE); drawn += bosses; cuts += c
    cuts.append(L.receiver_cut())
    cuts.append(box(-70, 67, -70, 70, L.PLATE_Z, L.PLATE_Z + 4.0))                       # the plate
    # the nose runner: seat rim + lead-ins drawn, the block its envelope; the flow window through it, the land over the board
    rim, block, seat_regs, wedge = L.runner(L.RX1 + 1.0, L.CX + L.SEAT_L / 2, nose=True)
    drawn.append(rim); env.append(block)
    bx, by = L.FLOW_BOARD
    cuts.append(box(L.WINDOW[0], L.WINDOW[1], -L.WINDOW[2], L.WINDOW[2], L.SEAT_Z - 1.0, L.FLOW_TOP))            # the window, up to the board's top
    land = box(L.FLOW_C[0] - bx / 2 - 1.5, L.FLOW_C[0] + bx / 2 + 1.5, -by / 2 - 1.5, by / 2 + 1.5, L.FLOW_TOP, L.FLOW_TOP + 3.0)
    drawn.append(land)
    cuts.append(box(L.FLOW_C[0] - bx / 2 - 12.0, L.FLOW_C[0] - bx / 2 + 2.0, -4.0, 4.0, L.FLOW_TOP - 7.0, L.FLOW_TOP + 3.0))   # the lead: out the tail end of the board, up the nose wall
    cuts.append(box(54.0, 62.0, -4.0, 4.0, L.FLOW_TOP - 7.0, 2.0))                                                # ... to the FC
    # the body: yoke under the plate, the wall down the receiver's nose face, a deck over the runner, the spine out to the camera
    env += [box(L.YOKE_SPLIT_X, 60.0, -34.0, 34.0, L.RZ1, L.PLATE_Z), box(L.RX1 + 0.5, 66.0, -33.4, 33.4, L.SEAT_Z, L.PLATE_Z),   # the yoke from the split at x -8 (the FC is on top of the plate: Spark 4 Nov) to the nose wall   # the wall room to x 66, the plate's nose end (run 1: index 0.60 at the wall's top in the funnel catch)
            box(L.RX1 + 1.0, L.CX + L.SEAT_L / 2, -33.4, 33.4, L.SEAT_Z + L.RUN_H, -8.0), box(60.0, 104.0, -20.0, 20.0, -8.0, 25.0),
            box(100.0, 126.0, -11.0, 11.0, 6.0, 52.0), box(99.0, 126.0, -24.0, 24.0, 6.0, 52.0),   # the neck and the block behind the tilted posts, to the board's back and |y| 24 (run 4: the lower posts' 2 mm stubs behind the plug keep-out had 1.2 of room beside them, two loose blobs); the disc cut trims it to what clears the props
            box(100.0, 130.0, -12.0, 12.0, -2.0, 22.0)]
    # camera, 30 deg up: 4 M2 posts behind the board back along -f, each on a O10 x 4 root boss behind the plug keep-out (runs 4 and 5: the
    # lower posts' 2 mm stubs beyond the 8 mm plug zone were left loose by the layout, their load under 1 N; a boss is a body the layout ties
    # and builds onto), insert hole from the front; the board + body, the plugs and the view are cuts
    F, Uu, R = L.CAM_F, L.CAM_U, L.CAM_R
    posts, roots, post_pts, post_regs = [], [], [], []
    for sy in (1, -1):
        for sz in (1, -1):
            q = L.BOARD_BACK + sy * L.CAM_SQ * R + sz * L.CAM_SQ * Uu
            q0 = q - L.POST_L * F
            posts.append(along(q0, q, L.POST_R)); post_pts.append(q0)
            roots.append(along(q0 - L.ROOT_L * F, q0 + 0.5 * F, L.ROOT_R))                    # the post's root boss, behind the plug keep-out
            cuts.append(along(q + 1.0 * F, q - L.CAM_INS_D * F, L.CAM_INS_R))
            post_regs.append(along(q0 - L.ROOT_L * F, q0 + 0.3 * F, L.ROOT_R + 0.6))
    drawn += posts + roots
    cam_box = lambda f0, f1, r_, u_: L.hull([L.BOARD_BACK + a * F + b * R + c * Uu for a in (f0, f1) for b in (-r_, r_) for c in (-u_, u_)])
    cuts.append(cam_box(-0.5, 18.0, 20.0, 20.0))                                                    # board + body + clearance
    cuts.append(cam_box(-L.PLUG_BACK, 0.0, 19.0, 19.0) - U(posts))                                   # plugs behind the board
    ha, hv = np.radians(L.HFOV + 2.0), np.radians(L.VFOV + 2.0); far = 400.0
    cuts.append(L.hull([L.LENS - 0.5 * F] + [L.LENS + far * F + sy * far * np.tan(ha) * R + sz * far * np.tan(hv) * Uu for sy in (1, -1) for sz in (1, -1)]))
    dc = U(L.disc_cuts())
    for q in posts:                                                                                  # the posts must stand clear of the front discs' keep-out
        if float((q - dc).volume()) < 0.97 * float(q.volume()):
            raise ValueError("a camera post is inside the front discs' 12 mm keep-out: move the lens forward or down")
    cuts += L.disc_cuts()
    # cage node NB and the tie sockets C0 / C3
    socks = L.node_socks("NB", ["B0", "B3", "P0", "P7"], nodes) + L.tie_socks(("C0", "C3"), nodes, ties)
    for s in socks:
        drawn.append(s.collar()); cuts += s.bores(); cuts.append(s.clear())
        env.append(along(s.pt(-6.0), s.pt(s.t_mouth), L.BORE_R + L.WALL + 5.0))
    L.check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    return dict(drawn=U(drawn), cuts=U(cuts), env=U(env), pad_regs=pad_regs, boss_regs=boss_regs, seat_regs=seat_regs,
                posts=posts, post_pts=post_pts, post_regs=post_regs, socks=socks)


def setup(h):
    B = bodies()
    p = Part("scout_nose_unit", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG", thin_ok=B["posts"])
    p.support("default", U(B["pad_regs"]))
    L.payload_cases(p, "receiver set", [(x, y, L.RZ1) for x, y in L.INS_NOSE], B["boss_regs"], L.REC_SET_G, L.REC_SET_CG, "default")
    L.payload_cases(p, "camera", B["post_pts"], B["post_regs"], L.CAM_G, L.CAM_CG, "default")
    L.catch_cases(p, B["seat_regs"], True, "default")
    L.socket_cases(p, B["socks"], "default")
    return p
