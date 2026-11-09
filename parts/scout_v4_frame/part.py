"""Scout v4 frame (9 Nov, Spark: "generate everything with topology"): the printed PCTG deck, its four arms to the motor pods and the two
side lugs, in one piece (the A1 mini's 180: pods at +-74.3 with O28 pads span 177).
Drawn (attachments only): the four motor pads (O28 x 4 under the ECO II 2004s, 4 x M2 on 12 x 12 and a O8 centre hole), the eight
base-unit bolt pads (O10 through the deck, M3 down from the top into the units' inserts), the turret's four O10 insert bosses (M3 x 5.7
from the top: its screws come down through its floor), and the side lugs' four rod sockets (DS0 / BS0 to S0, DS1 / BS1 to S1).
Cuts: the props' discs + 12, the motors, the turret over the deck, the N20's pocket and the 40 T gear's slot through the deck, the bolt
holes and their tool paths (the base bolts' heads and driver from above, the motor screws' from below), the rods beyond the mouths.
Room: the deck z 18.5..26.5 over |x| 60, |y| 40 (the base units bolt to its flat underside), the arms 24 wide z 12..26.5 out to the pods
(under the discs' keep-out at 27.7), the lugs a wedge x +-20 from y 36..64 at the deck down to y 36..44 at z -15.5.
Support: the eight base-unit pads (the battery set, the base and the electronics hang there).
Loads: per motor its thrust (the drone at 4 g, a quarter each) and its motor + prop in 35 g crashes; the probe (turret and head) on its
four bosses in 35 g crashes and flight, and the head-first crash (the parked head is the highest point: up to 230 N down the throat);
the side sockets' frame-model forces."""
import numpy as np
from topo import Part
from parts.scout_v4_units_lib import (S, ASM, box, cyl, along, U, unit, socks_of, socket_cases, socket_bodies, payload_cases, point_case,
                                      disc_cuts, region_top, region_bot, check_collars, BASE_BOLTS, TURRET_BOLTS, DZ0, DZ1, DRONE_G, G,
                                      M3_INS, M3_CLR, BOSS_R)

SPEC = dict(h=2.0, gen_cm3=36.0, max_worst=0.3)   # E: the frame room's 54 g estimate at 0.45 fill (tools/scout_v4_shape.py), PCTG at the nodes' 0.3 limit; 9 Nov +2 for the lugs' wedges
POD_R, POD_T, ARM_W, ROOM_TOP, ARM_Z0 = 14.0, 4.0, 24.0, 26.5, 12.0
LUG_X, LUG_Y1 = 20.0, 64.0                          # the lugs' room: x +-20 (was 8), out to y 64 at the deck (was 44)
PROBE = ("probe_turret", "probe_spool", "probe_pinion", "probe_motor", "probe_drum", "probe_feed", "head_frame", "probe_servo", "cradle", "probe_tilt", "probe_leds", "probe_mast_a", "probe_mast_b")


def probe_mass_cg():
    import trimesh, os
    g, s = 0.0, np.zeros(3)
    for k in PROBE:
        pid = "scout_v4_" + k; mg = ASM[pid]["mass_g"]
        t = trimesh.load(os.path.join(S.ROOT, "parts", pid, "part.stl"))
        c = t.center_mass if (t.is_watertight and abs(t.volume) > 1e-6) else t.bounds.mean(0)
        g += mg; s += mg * np.asarray(c)
    return g, s / g


def bodies():
    x0, x1, yb = S.DECK
    env = [box(x0, x1, -yb, yb, DZ0, ROOM_TOP)]
    for mx, my in S.MOTORS:
        a = np.array([np.sign(mx) * 30.0, np.sign(my) * 30.0, 0]); b = np.array([mx, my, 0])
        d = b - a; L = float(np.linalg.norm(d)); u = d / L; n = np.array([-u[1], u[0], 0])
        pts = [a + s * n * ARM_W / 2 + [0, 0, z] for s in (-1, 1) for z in (ARM_Z0, ROOM_TOP)] + [b + s * n * ARM_W / 2 + [0, 0, z] for s in (-1, 1) for z in (ARM_Z0, ROOM_TOP)]
        import trimesh
        from topo.part import man
        env.append(man(trimesh.convex.convex_hull(np.array(pts))))
        env.append(cyl(mx, my, ARM_Z0, DZ1, POD_R + 1.5))                            # 1.5 round the pad: the frame spans 179.6 (the A1 mini's 180)
    lx, ly0, ly1, lz = S.LUG
    for s_ in (1, -1):                                                                # the side lugs' room: a wedge from the deck's side down to the
        yz = [(ly0, lz), (ly1, lz), (ly1, lz + 1.2), (LUG_Y1, lz + 1.2 + 1.43 * (LUG_Y1 - ly1)), (LUG_Y1, ROOM_TOP), (ly0, ROOM_TOP)]   # BS socket, out under the
        pts = np.array([(x, s_ * y, z) for x in (-LUG_X, LUG_X) for y, z in yz])      # discs' keep-out, its outer face up the funnel's 55 deg (9 Nov: the 16 x 8
        import trimesh
        from topo.part import man
        env.append(man(trimesh.convex.convex_hull(pts)))                              # column read 2.55 in bending under BS's 134 N, 29 under the deck)
    drawn, cuts = [], []
    pods, pod_regs = [], []
    for mx, my in S.MOTORS:                                                           # the motor pads: 4 x M2 on 12 x 12 (ECO II 2004, listing), centre O8
        pods.append(cyl(mx, my, DZ1 - POD_T, DZ1, POD_R)); pod_regs.append(region_top(mx, my, DZ1, POD_R))
        cuts.append(cyl(mx, my, DZ1 - POD_T - 1, DZ1 + 1, 4.0))
        for dx in (-6.0, 6.0):
            for dy in (-6.0, 6.0):
                cuts.append(cyl(mx + dx, my + dy, ARM_Z0 - 1, DZ1 + 1, 1.1))           # M2 clearance
                cuts.append(cyl(mx + dx, my + dy, ARM_Z0 - 30, DZ1 - POD_T, 2.6))       # its head and the driver from below
        cuts.append(cyl(mx, my, DZ1, DZ1 + 60, POD_R + 1.0))                         # the motor
    drawn += pods
    base_pads, base_regs = [], []
    for x, y in BASE_BOLTS:                                                           # the base units' bolts: M3 down from the top into their inserts
        base_pads.append(cyl(x, y, DZ0, DZ1, BOSS_R)); base_regs.append(region_bot(x, y, DZ0))
        cuts.append(cyl(x, y, DZ0 - 1, DZ1 + 1, M3_CLR)); cuts.append(cyl(x, y, DZ1, DZ1 + 40, 3.3))   # clearance; the head and the driver from above
    drawn += base_pads
    tur, tur_regs = [], []
    for x, y in TURRET_BOLTS:                                                         # the turret's bosses: M3 x 5.7 inserts from the top
        tur.append(cyl(x, y, DZ0, DZ1, BOSS_R)); tur_regs.append(region_top(x, y, DZ1))
        cuts.append(cyl(x, y, DZ1 - M3_INS[1], DZ1 + 1, M3_INS[0])); cuts.append(cyl(x, y, DZ0 - 20, DZ1 - M3_INS[1] - 0.6, M3_CLR))   # insert; the screw's tip clearance below
    drawn += tur
    cx, cz = S.SPOOL["c"]; g0 = S.N20_FACE; pz = cz - (S.GEAR["spool_t"] + S.GEAR["pinion_t"]) * S.GEAR["m"] / 2
    cuts.append(box(cx - 6.6, cx + 6.6, g0 - 27.6, g0 + 0.6, pz - 5.6, ROOM_TOP + 1))      # the N20's pocket through the deck
    ra = (S.GEAR["spool_t"] / 2 + 1) * S.GEAR["m"]
    cuts.append(box(cx - ra - 0.8, cx + ra + 0.8, S.GEAR["y"][0] - 0.5, S.GEAR["y"][1] + 0.5, cz - ra - 0.8, ROOM_TOP + 1))   # the 40 T gear's slot
    tb = S.probe()["probe_turret"].bounds
    cuts.append(box(tb[0][0] - 0.5, tb[1][0] + 0.5, tb[0][1] - 0.5, tb[1][1] + 0.5, DZ1, 120))   # the turret above the deck
    cuts += disc_cuts()
    import trimesh, os
    ub = trimesh.load(os.path.join(S.ROOT, "parts", "scout_v4_ubec", "part.stl")).bounds   # the UBEC taped under the deck, lengthwise in the yokes' room
    drawn.append(box(ub[0][0], ub[1][0], ub[0][1], ub[1][1], DZ0, DZ0 + 2.6))              # its tape land in the deck (2.6: the gate's floor)
    ubec_reg = box(ub[0][0], ub[1][0], ub[0][1], ub[1][1], DZ0, DZ0 + 0.6); ubec_c = tuple((ub[0] + ub[1]) / 2)
    socks = socks_of("DS0") + socks_of("BS0") + socks_of("DS1") + socks_of("BS1")
    d, c = socket_bodies(socks, lambda s: np.array([0.0, np.sign(s.c[1]), -0.3]))       # ports out of the lugs' sides
    drawn += d; cuts += c
    check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    return dict(drawn=U(drawn), cuts=U(cuts), env=U(env), pod_regs=pod_regs, base_regs=base_regs, tur_regs=tur_regs, socks=socks, ubec=(ubec_c, ubec_reg))


def setup(h):
    B = bodies()
    p = Part("scout_v4_frame", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG")
    p.support("default", U(B["base_regs"]))
    thrust = DRONE_G * 4.0 * G / 4.0
    for i, (mx, my) in enumerate(S.MOTORS):
        p.load(f"motor {i + 1} thrust 4 g ({thrust:.1f} N)", B["pod_regs"][i], (0, 0, thrust), cls="flight")
        mg = ASM[f"scout_v4_motor_{i + 1}"]["mass_g"] + ASM[f"scout_v4_blades_{i + 1}"]["mass_g"]
        payload_cases(p, f"motor {i + 1}", [(mx, my, S.DECK_TOP)], [B["pod_regs"][i]], mg, (mx, my, S.DECK_TOP + 15.0), flight=False)
    g, cg = probe_mass_cg()
    pts = [(x, y, DZ1) for x, y in TURRET_BOLTS]
    payload_cases(p, "probe", pts, B["tur_regs"], g, cg)
    point_case(p, "head-first crash: 230 N down the throat", pts, B["tur_regs"], (0, 0, -230.0), (S.AXIS[0], 0.0, 130.0))
    payload_cases(p, "UBEC", [B["ubec"][0]], [B["ubec"][1]], ASM["scout_v4_ubec"]["mass_g"], B["ubec"][0])
    socket_cases(p, B["socks"])
    return p
