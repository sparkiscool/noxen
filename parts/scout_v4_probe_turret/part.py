"""Scout v4 probe turret (9 Nov, Spark: "generate everything with topology"): the housing under the roof's middle node that carries the
spool, the N20 and the cable drum, forms the two tape blades into the mast through the throat, and is the roof's middle node RM (its
four rod sockets: the roof's cross to RF, RB, RS0, RS1), so the roof's centre is held to the deck (tools/scout_v4_shape.py TIED).
Drawn (attachments only): four O10 floor bosses (M3 down into the deck's inserts, the driver from above beside the gear and the drum);
the spool's two axle bosses (an M3 screw is the axle); the N20's face plate (its 2 x M1.6 on 7.7) and its shaft's tip bearing; the throat's
walls (a flat slot at the spool's height opening to the blades' lens at its top, where the parked head's clip seats); the drum's wall
round the cable's coil; RM's four rod collars, their mouths at r 26..34 (tools/scout_v4_shape.RM_MOUTH); the drum's foot.
Cuts: the spool, its 40 T gear and the 12 T pinion swept; the N20; the axle; the blades' path and the mast above; the drum's cavity;
the bolts and their drivers; the props' discs + 12; the deck (z < 24.5) but the motor's pocket. (The DRV8833 is in the nose bay.)
Support: the four floor bosses. Loads: the head-first crash (230 N down the throat); the parked head (60 g) on the throat's seat in 35 g
crashes; the mast's fold-limited bending at the throat (0.3 N m, 1.5 x the bench target); the spool's tape tension at the N20's stall
(17 N each way on the axle); the turret's own spool, blades and cable (15 g) on the axle bosses in crashes; RM's rods."""
import numpy as np
from topo import Part
from parts.scout_v4_units_lib import (S, ASM, box, cyl, along, hull, U, socks_of, socket_cases, socket_bodies, payload_cases, point_case,
                                      disc_cuts, region_top, region_bot, check_collars, TURRET_BOLTS, DZ1, M3_CLR, BOSS_R)

SPEC = dict(h=1.2, gen_cm3=12.0, max_worst=0.3)  # 9 Nov: 9 cm3 read 1.01 at RM-RB with its arms out to r 34 and the budget spent; the arms now shorter, 12
CUP_R, BLADE_T, CABLE_T = 15.0, 0.13, 1.2       # the 19 mm blade's cup radius and thickness (patents: R 12.7..25.4, t 0.11..0.19; E), the 6-wire bundle
FOLD_NM = 0.3                                    # the mast's fold moment the throat must carry: 1.5 x the 0.2 N m bench target


def lens_half():
    hw = S.BLADE_W / 2; sag = CUP_R - np.sqrt(CUP_R ** 2 - hw ** 2)
    return sag + BLADE_T + CABLE_T / 2 + 0.4                              # half the lens's thickness along x, with clearance


def yz_ring(cx, cz, r0, r1, y0, y1):
    return cyl(0, 0, y0, y1, r1).rotate([-90, 0, 0]).translate([cx, 0, cz]) - cyl(0, 0, y0 - 1, y1 + 1, r0).rotate([-90, 0, 0]).translate([cx, 0, cz]) if r0 > 0 else cyl(0, 0, y0, y1, r1).rotate([-90, 0, 0]).translate([cx, 0, cz])


def ycyl(cx, cz, r, y0, y1):
    """a cylinder along y from y0 to y1 at (cx, cz)"""
    import manifold3d as mf
    return mf.Manifold.cylinder(y1 - y0, r, r, 48).rotate([-90, 0, 0]).translate([cx, y0, cz])


def bodies():
    cx, cz = S.SPOOL["c"]; r = S.SPOOL["r"]; t = S.THROAT; mx = S.MAST_X; hw = S.BLADE_W / 2 + 0.5
    gy0, gy1 = S.GEAR["y"]; ra = (S.GEAR["spool_t"] / 2 + 1) * S.GEAR["m"]; rp = (S.GEAR["pinion_t"] / 2 + 1) * S.GEAR["m"]
    pz = cz - (S.GEAR["spool_t"] + S.GEAR["pinion_t"]) * S.GEAR["m"] / 2; g0 = S.N20_FACE
    dy0, dy1, dr = S.DRUM["y0"], S.DRUM["y1"], S.DRUM["r"]
    yp, ym = gy1 + 0.8, dy0 - 0.8
    drawn, cuts, env = [], [], []
    # room: a drum-shaped housing round the spool from the deck to the roof's ring, the ring node, and the motor's pocket
    env.append(cyl(cx, 0, DZ1, 67.0, 36.0))
    env.append(box(cx - 8.0, cx + 8.0, g0 - 30.0, yp + 6.0, pz - 7.0, DZ1 + 1))
    # floor bosses
    regs = []
    for x, y in TURRET_BOLTS:
        drawn.append(cyl(x, y, DZ1, DZ1 + 4.0, BOSS_R)); regs.append(region_bot(x, y, DZ1))
        cuts.append(cyl(x, y, DZ1 - 1, DZ1 + 5, M3_CLR)); cuts.append(cyl(x, y, DZ1 + 4.0, 140, 3.3))
    # the spool's axle bosses (M3 screw axle) in the two side walls
    axle = []
    for y0, y1 in ((yp, yp + 3.0), (ym - 3.0, ym)):
        drawn.append(ycyl(cx, cz, 5.0, y0, y1)); axle.append(ycyl(cx, cz, 5.4, y0 - 0.2, y1 + 0.2))
    cuts.append(ycyl(cx, cz, 1.6, ym - 10, yp + 10))
    # the N20: face plate with 2 x M1.6 on 7.7 and the shaft hole; the shaft tip's bearing beyond the pinion
    face = box(cx - 6.5, cx + 6.5, g0 + 0.15, g0 + 2.75, pz - 5.5, pz + 5.5); drawn.append(face)   # 2.6 thick (the gate's floor 2.5)
    cuts.append(ycyl(cx, pz, 2.2, g0 - 1, g0 + 3.5)); cuts += [ycyl(cx + s * 3.85, pz, 0.9, g0 - 1, g0 + 3.5) for s in (-1, 1)]
    drawn.append(ycyl(cx, pz, 3.5, gy1 + 1.0, gy1 + 4.6)); cuts.append(ycyl(cx, pz, 1.6, gy1 + 0.5, gy1 + 5.0))   # the shaft's tip (10 from the face) in its bearing
    cuts.append(box(cx - 6.4, cx + 6.4, g0 - 10.0 - 0.3, g0 + 0.2, pz - 5.3, pz + 5.3))           # the gearbox
    cuts.append(ycyl(cx, pz, 6.3, g0 - 27.5, g0 - 10.0))                                           # the can
    # the throat: walls round the blades' path, a flat slot at the spool's height to the lens at its top (the head's clip seats in it)
    lh = lens_half(); sl = BLADE_T + CABLE_T / 2 + 0.3
    path = hull(np.array([[mx + sx * sl, sy * hw, t["z0"]] for sx in (-1, 1) for sy in (-1, 1)] + [[mx + sx * lh, sy * hw, t["z1"] + 0.5] for sx in (-1, 1) for sy in (-1, 1)]))
    walls = hull(np.array([[mx + sx * (sl + 2.0), sy * (hw + 2.0), t["z0"]] for sx in (-1, 1) for sy in (-1, 1)] + [[mx + sx * (lh + 2.0), sy * (hw + 2.0), t["z1"]] for sx in (-1, 1) for sy in (-1, 1)]))
    drawn.append(walls - path); cuts.append(path)
    cuts.append(box(mx - lh - 6.0, mx + lh + 6.0, -hw - 3.0, hw + 3.0, t["z1"], 400))               # the mast and the head above
    cuts.append(box(mx - sl - 0.2, mx + sl + 0.2, -hw, hw, cz - 2.0, t["z0"] + 0.5))                 # the blades from the spool's tangent into the throat
    # the drum round the cable's coil
    drawn.append(yz_ring(cx, cz, dr + 0.3, dr + 2.8, dy0 - 2.6, dy1))                               # 2.5 wall and a 2.6 end (the gate's floor)
    for sx in (-1, 1):                                                                              # its two legs to the floor either side of the N20's can (9 Nov: the drum
        drawn.append(box(*sorted((cx + sx * 6.9, cx + sx * 10.5)), dy0 - 2.6, dy1, DZ1, cz) - ycyl(cx, cz, dr + 0.3, dy0 - 3, dy1 + 1))   # sat 0.7 over the can: a neck)
    cuts.append(ycyl(cx, cz, dr + 0.3, dy0, dy1 + 0.5)); cuts.append(box(cx - 2, cx + 2, ym - 8, dy0 + 1, cz - 2, cz + 2))   # its cavity; the cable's way out to the Radxa
    # the spool, its gear and the pinion swept
    cuts.append(ycyl(cx, cz, r + 1.0, S.SPOOL["y0"] - 0.6, S.SPOOL["y1"] + 0.6))
    cuts.append(ycyl(cx, cz, ra + 0.8, gy0 - 0.5, gy1 + 0.5)); cuts.append(ycyl(cx, pz, rp + 0.8, gy0 - 0.5, gy1 + 0.5))
    cuts += disc_cuts()
    deck = box(-200, 200, -200, 200, -100, DZ1) - box(cx - 8.5, cx + 8.5, g0 - 31.0, yp + 7.0, pz - 8.0, DZ1 + 0.1)
    cuts.append(deck)
    # RM: the roof's middle node, its four rod collars, each mouth as near as its neighbours allow (tools/scout_v4_shape.RM_MOUTH: RF's
    # floor on the throat's wall, so collar and wall are one body; 9 Nov, r 34 left a 2 mm slab between them, a neck)
    socks = socks_of("RM", ring_r=S.RM_MOUTH)
    d, c = socket_bodies(socks, lambda s: np.array([0.0, 0.0, 1.0]))
    drawn += d; cuts += c
    check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    seat = box(mx - lh - 2.0, mx + lh + 2.0, -hw - 2.0, hw + 2.0, t["z1"] - 1.5, t["z1"] + 0.3)       # the throat's top: the parked head's seat
    return dict(drawn=U(drawn), cuts=U(cuts), env=U(env), regs=regs, axle=axle, seat=seat, socks=socks, cz=cz, cx=cx)


def setup(h):
    B = bodies()
    p = Part("scout_v4_probe_turret", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PETG")
    p.support("default", U(B["regs"]))
    t = S.THROAT; mx = S.MAST_X
    p.load("head-first crash: 230 N down the throat", B["seat"], (0, 0, -230.0), cls="crash")
    head_g = sum(ASM["scout_v4_" + k]["mass_g"] for k in ("head_frame", "probe_servo", "cradle", "probe_tilt", "probe_leds"))
    payload_cases(p, "the parked head", [(mx, 0, t["z1"])], [B["seat"]], head_g, (S.AXIS[0], S.AXIS[1], S.AXIS[2]), flight=False)
    Fm = FOLD_NM / ((t["z1"] - t["z0"]) / 1000.0)
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p.load(f"the mast's fold moment {FOLD_NM} N m at the throat ({d})", B["seat"], (Fm * d[0], Fm * d[1], 0), cls="misuse")
    for sgn, lab in ((1, "paying out"), (-1, "pulling in")):
        p.load_multi(f"tape tension at the N20's stall, {lab} (17 N)", [(B["axle"][0], (0, 0, -8.5 * sgn)), (B["axle"][1], (0, 0, -8.5 * sgn))], cls="flight")
    own = sum(ASM["scout_v4_" + k]["mass_g"] for k in ("probe_spool", "probe_drum", "probe_feed"))
    payload_cases(p, "spool, blades and cable", [(B["cx"], S.GEAR["y"][1] + 0.8, B["cz"]), (B["cx"], S.DRUM["y0"] - 0.8, B["cz"])], B["axle"], own, (B["cx"], 0, B["cz"]), flight=False)
    socket_cases(p, B["socks"])
    return p
