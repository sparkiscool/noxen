"""Scout v4 probe turret (9 Nov, Spark: "generate everything with topology"): the housing under the roof's middle node that carries the
spool, the N20, the coil's pressure roller and the cable drum, forms the two tape blades into the mast through the throat, and is the roof's middle node RM (its
four rod sockets: the roof's cross to RF, RB, RS0, RS1), so the roof's centre is held to the deck (tools/scout_v4_shape.py TIED).
Drawn (attachments only): four O10 floor bosses (M3 down into the deck's inserts, the driver from above beside the gear and the drum);
the spool's two axle bosses (an M3 screw is the axle); the pressure roller's two guides (its M3 axle's ends in radial slots, a 4 x 0.5 x 15
spring behind each: tools/probe_mast_sim.py, a pushed coil blossoms unless locked); the N20's face plate (its 2 x M1.6 on 7.7) and its shaft's tip bearing; the throat's
walls (a flat slot at the spool's height opening to the blades' lens at its top, where the parked head's clip seats); the drum's wall
round the cable's coil; RM's four rod collars, their mouths at r 26..34 (tools/scout_v4_shape.RM_MOUTH).
Cuts: the spool, its 40 T gear and the 12 T pinion swept; the roller's travel, its axle's slots, the springs; the N20; the axle; the blades' path and the mast above; the drum's cavity;
the bolts and their drivers; the props' discs + 12; the deck (z < 24.5) but the motor's pocket. (The DRV8833 is in the nose bay.)
Support: the four floor bosses. Loads: the head-first crash (systems/scout_v4.head_load.json, ~250 N down the throat); the parked head (60 g) on the throat's seat in 35 g
crashes; the mast's fold-limited bending at the throat (0.6 N m in x, 1.2 in y: the FE's fold moments); the spool's tape tension at the N20's stall
(28 N each way on the axle: the 100 rpm N20, 420 N mm); the roller's springs (12 N against the axle); the turret's own spool, blades and cable (15 g) on the axle bosses in crashes; RM's rods."""
import numpy as np
from topo import Part
from parts.scout_v4_units_lib import (S, ASM, box, cyl, along, hull, U, socks_of, socket_cases, socket_bodies, payload_cases, point_case,
                                      disc_cuts, region_top, region_bot, check_collars, TURRET_BOLTS, DZ1, M3_CLR, BOSS_R)

import json as _json, os as _os
HEAD_N = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "systems", "scout_v4.head_load.json")))["head_first_N"]   # the frame model's head-first contact force (tools/scout_v4_shape.py)
SPEC = dict(h=1.2, gen_cm3=14.0, max_worst=0.3)  # 9 Nov: 9 cm3 read 1.01 at RM-RB with its arms out to r 34 and the budget spent; 14 with the FE fold loads (1.2 N m in y) and the roller
CUP_R, BLADE_T, CABLE_T = 15.0, 0.13, 0.6       # the 19 mm blade's cup radius and thickness (patents: R 12.7..25.4, t 0.11..0.19; E), the 6 x 30 AWG PTFE ribbon
FOLD_NM = (0.6, 1.2)                             # the moment the mast can put on the throat before it folds, +-x and +-y (tools/probe_mast_fe.py: the unbonded lens folds at ~0.50 N m in x, 3.6 in y; x x 1.2, y capped at 2 x x: a bigger y moment is a crash on the head)


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
    lh_, hw_ = lens_half(), S.BLADE_W / 2 + 0.5
    env.append(box(mx - lh_ - 6.0, mx + lh_ + 6.0, -hw_ - 6.0, hw_ + 6.0, 55.0, t["z1"]))   # 10 Nov: room up the throat to its seat (the drum ended at 67, the throat's top 6 stood on drawn 2 mm walls: 0.53 in the 1.2 N m fold)
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
    lh = lens_half(); sl = BLADE_T + CABLE_T / 2 + 0.3 + 0.7                        # the stack's half and the 1.4 its tangent walks as the coil empties
    path = hull(np.array([[mx + sx * sl, sy * hw, t["z0"]] for sx in (-1, 1) for sy in (-1, 1)] + [[mx + sx * lh, sy * hw, t["z1"] + 0.5] for sx in (-1, 1) for sy in (-1, 1)]))
    walls = hull(np.array([[mx + sx * (sl + 2.0), sy * (hw + 2.0), t["z0"]] for sx in (-1, 1) for sy in (-1, 1)] + [[mx + sx * (lh + 2.0), sy * (hw + 2.0), t["z1"]] for sx in (-1, 1) for sy in (-1, 1)]))
    drawn.append(walls - path - ycyl(cx, cz, S.SPOOL["r"] + 3.0, S.SPOOL["y0"] - 3, S.SPOOL["y1"] + 3)); cuts.append(path)   # the -x wall stops 3 short of the spool's sweep (no sliver)
    cuts.append(box(mx - lh - 6.0, mx + lh + 6.0, -hw - 3.0, hw + 3.0, t["z1"], 400))               # the mast and the head above
    cuts.append(box(mx - sl - 0.2, mx + sl + 0.2, -hw, hw, cz - 2.0, t["z0"] + 0.5))                 # the blades from the spool's tangent into the throat
    # the drum round the cable's coil
    drawn.append(yz_ring(cx, cz, dr + 0.3, dr + 2.8, dy0 - 2.6, dy1))                               # 2.5 wall and a 2.6 end (the gate's floor)
    cuts.append(ycyl(cx, cz, dr + 0.3, dy0, dy1 + 0.5)); cuts.append(box(cx - 2, cx + 2, ym - 8, dy0 + 1, cz - 2, cz + 2))   # its cavity; the cable's way out to the Radxa
    # the coil's pressure roller: its axle's ends in radial slots (30 deg before the exit), a spring behind each down the slot's line
    a = np.radians(S.ROLLER["ang"]); u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]]); rw = S.ROLLER["w"] / 2
    r_full = S.SPOOL["hub"] + 1.4 + S.ROLLER["r"]; r_empty = S.SPOOL["hub"] + 0.15 + S.ROLLER["r"]
    def radial(s0, s1, hv, y0, y1):                                                 # a box along the slot's line: radial s0..s1, tangential +-hv
        pts = [[cx + q * u[0] + w * v[0], yy, cz + q * u[1] + w * v[1]] for q in (s0, s1) for w in (-hv, hv) for yy in (y0, y1)]
        return hull(np.array(pts))
    guides, seats = [], []
    for sy in (1, -1):                                                              # each guide 9 thick round the spring's O4.5 pocket: walls >= 2.25
        yc = sy * (rw + 4.0)
        guides.append(radial(17.9, r_full + 13.5, 4.5, yc - 4.5, yc + 4.5))
        cuts.append(radial(10.0, r_full + 0.6, 1.65, yc - 6.0, yc + 6.0))                         # the axle's slot (M3 x 40), open through the guide's inner end (10 Nov: a 0.7 web was left before it)
        cuts.append(radial(r_full - 0.5, r_full + 11.0, 2.25, yc - 2.25, yc + 2.25))               # the spring's pocket on the slot's line (O4, 11 long at the full coil)
        seats.append(radial(r_full + 11.0, r_full + 11.6, 2.0, yc - 2.0, yc + 2.0))
    drawn += guides
    cuts.append(radial(r_empty - S.ROLLER["r"] - 0.8, r_full + 0.8, S.ROLLER["r"] + 0.8, -rw - 0.3, rw + 0.3))   # the roller's tube and its travel
    # the spool, its gear and the pinion swept
    cuts.append(ycyl(cx, cz, r + 1.0, S.SPOOL["y0"] - 0.6, S.SPOOL["y1"] + 0.6))
    cuts.append(ycyl(cx, cz, ra + 0.8, gy0 - 0.5, gy1 + 0.5)); cuts.append(ycyl(cx, pz, rp + 0.8, gy0 - 0.5, gy1 + 0.5))
    cuts += disc_cuts()
    deck = box(-200, 200, -200, 200, -100, DZ1) - box(cx - 8.5, cx + 8.5, g0 - 31.0, yp + 7.0, pz - 8.0, DZ1 + 0.1)
    cuts.append(deck)
    # RM: the roof's middle node, its four rod collars with their mouths at r 34
    socks = [k for o, r in S.RM_MOUTH.items() for k in socks_of("RM", ring_r=r) if o in k.name.split(" ")[1].split("-")]   # each mouth at its own r (tools/scout_v4_shape.RM_MOUTH)
    d, c = socket_bodies(socks, lambda s: np.array([0.0, 0.0, 1.0]))
    drawn += d; cuts += c
    check_collars(socks, U(cuts) - U([b for s in socks for b in s.bores()]))
    seat = box(mx - lh - 2.0, mx + lh + 2.0, -hw - 2.0, hw + 2.0, t["z1"] - 1.5, t["z1"] + 0.3)       # the throat's top: the parked head's seat
    return dict(drawn=U(drawn), cuts=U(cuts), env=U(env), regs=regs, axle=axle, seat=seat, socks=socks, cz=cz, cx=cx, roller_seats=seats, roller_u=u)


def setup(h):
    B = bodies()
    p = Part("scout_v4_probe_turret", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PCTG")
    p.support("default", U(B["regs"]))
    t = S.THROAT; mx = S.MAST_X
    p.load(f"head-first crash: {HEAD_N} N down the throat", B["seat"], (0, 0, -float(HEAD_N)), cls="crash")
    head_g = sum(ASM["scout_v4_" + k]["mass_g"] for k in ("head_frame", "probe_servo", "cradle", "probe_tilt", "probe_leds"))
    payload_cases(p, "the parked head", [(mx, 0, t["z1"])], [B["seat"]], head_g, (S.AXIS[0], S.AXIS[1], S.AXIS[2]), flight=False)
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        M = FOLD_NM[0] if d[0] else FOLD_NM[1]; Fm = M / ((t["z1"] - t["z0"]) / 1000.0)
        p.load(f"the mast's fold moment {M} N m at the throat ({d})", B["seat"], (Fm * d[0], Fm * d[1], 0), cls="misuse")
    for sgn, lab in ((1, "paying out"), (-1, "pulling in")):
        p.load_multi(f"tape tension at the N20's stall, {lab} (28 N)", [(B["axle"][0], (0, 0, -14.0 * sgn)), (B["axle"][1], (0, 0, -14.0 * sgn))], cls="flight")
    u = B["roller_u"]; fu = (6.0 * u[0], 0.0, 6.0 * u[1])                           # the springs push their seats out and the coil (the axle) in: 12 N
    p.load_multi("the roller's springs on the full coil (12 N)", [(B["roller_seats"][0], fu), (B["roller_seats"][1], fu),
                 (B["axle"][0], (-fu[0], 0, -fu[2])), (B["axle"][1], (-fu[0], 0, -fu[2]))], cls="flight")
    own = sum(ASM["scout_v4_" + k]["mass_g"] for k in ("probe_spool", "probe_drum", "probe_feed"))
    payload_cases(p, "spool, blades and cable", [(B["cx"], S.GEAR["y"][1] + 0.8, B["cz"]), (B["cx"], S.DRUM["y0"] - 0.8, B["cz"])], B["axle"], own, (B["cx"], 0, B["cz"]), flight=False)
    socket_cases(p, B["socks"])
    return p
