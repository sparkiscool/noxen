"""Scout v4 probe turret (9 Nov, Spark: "generate everything with topology"): the housing under the roof's middle node that carries the
spool, the N20, the coil's pressure roller and the cable drum, forms the two tape blades into the mast through the throat, and is the roof's middle node RM (its
four rod sockets: the roof's cross to RF, RB, RS0, RS1), so the roof's centre is held to the deck (tools/scout_v4_shape.py TIED).
Drawn (attachments only): four O10 floor bosses (M3 down into the deck's inserts, the driver from above beside the gear and the drum);
the spool's two axle bosses (an M3 screw is the axle); the pressure roller's rod guides (its O3 rod's ends in radial slots at |y| 9.8..12) and its leaf pocket
on the rod's middle (a saddle and a Keystone-209-type leaf on a fixed end wall: tools/probe_mast_sim.py, a pushed coil blossoms unless locked); the N20's face plate (its 2 x M1.6 on 7.7) and its shaft's tip bearing; the throat's
walls (a flat slot at the spool's height opening to the blades' lens at its top, where the parked head's clip seats); the drum's wall
round the cable's coil; RM's four rod collars, their mouths at r 26..34 (tools/scout_v4_shape.RM_MOUTH).
Cuts: the spool, its 40 T gear and the 12 T pinion swept; the tubes' travel, the rod's slots, the leaf's pocket; the N20; the axle; the blades' path and the mast above; the drum's cavity;
the bolts and their drivers; the props' discs + 12; the deck (z < 24.5) but the motor's pocket. (The DRV8833 is in the nose bay.)
Support: the four floor bosses. Loads: the head-first crash (systems/scout_v4.head_load.json, ~250 N down the throat); the parked head (60 g) on the throat's seat in 35 g
crashes; the mast's fold-limited bending at the throat (0.6 N m in x, 1.2 in y: the FE's fold moments); the spool's tape tension at the N20's stall
(44 N each way on the axle: the 100 rpm N20's 2.2 kg cm, 612 N mm at the spool, E); the roller's leaf (40 N on the full coil, E); the turret's own spool, blades and cable (15 g) on the axle bosses in crashes; RM's rods."""
import numpy as np
from topo import Part
from parts.scout_v4_units_lib import (S, ASM, box, cyl, along, hull, U, socks_of, socket_cases, socket_bodies, payload_cases, point_case,
                                      disc_cuts, region_top, region_bot, check_collars, TURRET_BOLTS, DZ1, M3_CLR, BOSS_R)

import json as _json, os as _os
HEAD_N = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "systems", "scout_v4.head_load.json")))["head_first_N"]   # the frame model's head-first contact force (tools/scout_v4_shape.py)
SPEC = dict(h=1.2, gen_cm3=28.0, max_worst=0.3)  # 15 Nov: 23 read 0.385 (field 0.334 at the end, budget spent) at the RM-RS1 socket in compression with its 1.55 N m end moment, across the layers (room_full 0.13): 28. 10 Nov run 4: 20 read 0.34 (head-first at the -y side, 37.7 g) and skins under the spool; 23. Run 3: 14 cm3 read 0.50 at the +y roller guide (the 1.2 N m fold in y and the head-first crash share its one strut down to the floor bolts): 20. 9 Nov: 9 cm3 read 1.01 at RM-RB with its arms out to r 34 and the budget spent; 14 with the FE fold loads (1.2 N m in y) and the roller
CUP_R, BLADE_T, CABLE_T = S.CUP_R, S.BLADE_T, S.CABLE_T   # 12 Nov: the cup radius is the coupon's T, t 0.13 E, the 30 AWG PTFE ribbon 0.81 (tools/scout_v4_shape.py)
FOLD_NM = (0.6, 1.2)                             # the moment the mast can put on the throat before it folds, +-x and +-y (tools/probe_mast_fe.py: the unbonded lens folds at ~0.50 N m in x, 3.6 in y; x x 1.2, y capped at 2 x x: a bigger y moment is a crash on the head)


def lens_half():
    return S.LENS_HALF                                                     # half the lens's thickness along x, with clearance (tools/scout_v4_shape.py: the coupon's T)


def yz_ring(cx, cz, r0, r1, y0, y1):
    return cyl(0, 0, y0, y1, r1).rotate([-90, 0, 0]).translate([cx, 0, cz]) - cyl(0, 0, y0 - 1, y1 + 1, r0).rotate([-90, 0, 0]).translate([cx, 0, cz]) if r0 > 0 else cyl(0, 0, y0, y1, r1).rotate([-90, 0, 0]).translate([cx, 0, cz])


def ycyl(cx, cz, r, y0, y1):
    """a cylinder along y from y0 to y1 at (cx, cz)"""
    import manifold3d as mf
    return mf.Manifold.cylinder(y1 - y0, r, r, 48).rotate([-90, 0, 0]).translate([cx, y0, cz])


def bodies():
    cx, cz = S.SPOOL["c"]; r = S.SPOOL["r"]; t = S.THROAT; mx = S.MAST_X; hw = S.BLADE_W / 2 + 0.5
    gy0, gy1 = S.GEAR["y"]; ra = (S.GEAR["spool_t"] / 2 + 1) * S.GEAR["m"]; rp = (S.GEAR["pinion_t"] / 2 + 1) * S.GEAR["m"]
    pz = cz - S.GEAR["c"]; g0 = S.N20_FACE
    dy0, dy1, dr = S.DRUM["y0"], S.DRUM["y1"], S.DRUM["r"]
    yp, ym = gy1 + 0.8, dy0 - 0.8
    drawn, cuts, env = [], [], []
    # room: a drum-shaped housing round the spool from the deck to the roof's ring, the ring node, and the motor's pocket
    env.append(cyl(cx, 0, DZ1, 67.0, 36.0))
    env.append(box(cx - 8.0, cx + 8.0, g0 - 30.0, yp + 6.0, pz, DZ1 + 1))   # 10 Nov run 5: from the N20's axis up (below it the can left 1.7 strips: two loose bodies)
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
    drawn.append(ycyl(cx, pz, 3.5, gy1 + 1.0, gy1 + 4.6)); cuts.append(ycyl(cx, pz, 1.6, gy1 + 0.5, gy1 + 5.0))
    for sx in (-1, 1):                                                                              # 10 Nov run 5: the N20's mount drawn, its two 2.0 cheeks beside the gearbox
        drawn.append(box(*sorted((cx + sx * 6.5, cx + sx * 8.5)), g0 + 0.15, gy1 + 4.6, pz - 4.5, DZ1))   # from the face plate to the tip bearing (the pocket's room gave only 1.7 strips)   # the shaft's tip (10 from the face) in its bearing
    cuts.append(box(cx - 6.4, cx + 6.4, g0 - 10.0 - 0.3, g0 + 0.2, pz - 5.3, pz + 5.3))           # the gearbox
    cuts.append(ycyl(cx, pz, 6.3, g0 - 27.5, g0 - 10.0))                                           # the can
    # the throat: walls round the blades' path, a flat slot at the spool's height to the lens at its top (the head's clip seats in it)
    lh = lens_half(); sl = S.STACK / 2 + max(S.MAST_X - S.SPOOL["hub"] - S.STACK / 2, S.SPOOL["hub"] + S.COIL - S.STACK / 2 - S.MAST_X) + 0.3   # the stack's half, the walk of its tangent off the coil (full to empty) from the mast's axis, 0.3
    path = hull(np.array([[mx + sx * sl, sy * hw, t["z0"]] for sx in (-1, 1) for sy in (-1, 1)] + [[mx + sx * lh, sy * hw, t["z1"] + 0.5] for sx in (-1, 1) for sy in (-1, 1)]))
    walls = hull(np.array([[mx + sx * (sl + 2.0), sy * (hw + 2.0), t["z0"]] for sx in (-1, 1) for sy in (-1, 1)] + [[mx + sx * (lh + 2.0), sy * (hw + 2.0), t["z1"]] for sx in (-1, 1) for sy in (-1, 1)]))
    drawn.append(walls - path - ycyl(cx, cz, S.SPOOL["r"] + 3.0, S.SPOOL["y0"] - 3, S.SPOOL["y1"] + 3)); cuts.append(path)   # the -x wall stops 3 short of the spool's sweep (no sliver)
    cuts.append(box(mx - lh - 6.0, mx + lh + 6.0, -hw - 3.0, hw + 3.0, t["z1"], 400))               # the mast and the head above
    cuts.append(box(mx - sl - 0.2, mx + sl + 0.2, -hw, hw, cz - 2.0, t["z0"] + 0.5))                 # the blades from the spool's tangent into the throat
    # the drum round the cable's coil
    drawn.append(yz_ring(cx, cz, dr + 0.3, dr + 2.8, dy0 - 2.6, dy1))                               # 2.5 wall and a 2.6 end (10 Nov run 6: 4.0 on r 11 still peaked in it; the drum is r 9 now, the room round it generated)
    cuts.append(ycyl(cx, cz, dr + 0.3, dy0, dy1 + 0.5)); cuts.append(box(cx - 2, cx + 2, ym - 8, dy0 + 1, cz - 2, cz + 2))   # its cavity; the cable's way out to the Radxa
    # the coil's pressure roller (14 Nov, probe stream): two O8 tubes on a O3 carbon rod 15 deg before the exit, the rod's ends guided in radial
    # slots of two walls at |y| 9.8..12 (open to the spool: the rod drops in from there); on the rod's middle, between the tubes, a printed saddle
    # carrying a Keystone-209-type leaf contact (ZCZQC B0C5D3XLBK) that presses the pocket's fixed end wall (no screw: the saddle's set sets the force).
    # The old pockets at the rod's ends (|y| 12..29, out to r 39.5) stood in the props' keep-out; at |y| < 6.5 the keep-out leaves x to ~31.
    a = np.radians(S.ROLLER["ang"]); u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]]); rw = S.ROLLER["w"] / 2
    r_full = S.SPOOL["hub"] + S.COIL + S.ROLLER["r"]; r_empty = S.SPOOL["hub"] + 0.15 + S.ROLLER["r"]
    L = S.LEAF; sw = L["s_wall"]; wl = L["wall"]; ht, hy = L["ht"], L["hy"]
    def radial(s0, s1, t0, t1, y0, y1):                                             # a box along the roller's line: radial s0..s1 from the spool's axis, tangential t0..t1
        pts = [[cx + q * u[0] + w * v[0], yy, cz + q * u[1] + w * v[1]] for q in (s0, s1) for w in (t0, t1) for yy in (y0, y1)]
        return hull(np.array(pts))
    def ucirc(sc, rr, y0, y1, s_in=None):                                            # a circle r rr round the line's point sc (along y), hulled in to s_in
        th = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        pts = [[cx + (sc + rr * np.cos(t_)) * u[0] + rr * np.sin(t_) * v[0], yy, cz + (sc + rr * np.cos(t_)) * u[1] + rr * np.sin(t_) * v[1]] for t_ in th for yy in (y0, y1)]
        if s_in is not None:
            pts += [[cx + s_in * u[0] + w * v[0], yy, cz + s_in * u[1] + w * v[1]] for w in (-rr, rr) for yy in (y0, y1)]
        return hull(np.array(pts))
    s_end = r_full + 0.15                                                            # the rod's centre at its outermost (the full coil) + 0.15 (at + 0.3 the wall's far side met the keep-out)
    y0s, y1s = L["slot_y"]
    guides = []
    for sy in (1, -1):
        Y = lambda p, q: tuple(sorted((sy * p, sy * q)))
        guides.append(ucirc(s_end, 1.65 + wl, *Y(y0s, y1s), s_in=17.8))              # the rod's guide wall (2.55 thick: the audit's 2.5): 2.0 round the slot, from r 17.8 (the spool's flanges, r 15.9, sweep 16.9)
        cuts.append(ucirc(s_end, 1.65, *Y(y0s - 0.5, y1s + 1.0), s_in=15.0))          # the slot (O3.3 wide), open to the spool, through
    pk = [radial(20.0, sw + wl, ht, ht + wl, -hy - wl, hy + wl),                     # the pocket's two tangential walls (they guide the saddle; clear of the tubes' |t| < 4.8)
          radial(17.5, sw + wl, -ht - wl, -ht, -hy - wl, hy + wl),
          radial(sw, sw + wl, -ht - wl, ht + wl, -hy - wl, hy + wl)]                 # its end wall: the leaf presses it
    for sy in (1, -1):
        pk.append(radial(L["y_walls"], sw, -ht - wl, ht + wl, *sorted((sy * hy, sy * (hy + wl)))))   # its y walls, beyond the tubes' sweep (s 24.5)
    drawn += [g ^ box(-200, 200, -200, 200, DZ1, 400) for g in guides] + [U(pk) ^ box(-200, 200, -200, 200, DZ1, 400)]
    cuts.append(radial(15.0, sw, -ht, ht, -hy, hy))                                   # the pocket: the saddle and the leaf, open to the spool (they go in from there with the rod)
    seats = [radial(sw + 0.2, sw + 1.8, -4.0, 4.0, -3.5, 3.5)]                        # the end wall where the leaf presses
    env.append(radial(17.0, sw + wl + 1.0, -ht - wl - 1.0, ht + wl + 1.0, -y1s - 0.5, y1s + 0.5) ^ box(-200, 200, -200, 200, DZ1, 400))   # room round the guides and the pocket
    cuts.append(radial(r_empty - S.ROLLER["r"] - 0.8, r_full + S.ROLLER["r"] + 0.8, -S.ROLLER["r"] - 0.8, S.ROLLER["r"] + 0.8, -rw - 0.2, rw + 0.2))   # the tubes and their travel
    cuts.append(box(cx - 13.0, cx + 13.0, S.SPOOL["y0"] - 0.6, g0 - 0.3, DZ1, cz - 10.0))   # 10 Nov run 4: under the spool, between its sweep (down to z 24.1) and the N20's gearbox (up to 25.5), room left only skins 1.1-1.5 thick (the solid check): void
    # the spool, its gear and the pinion swept
    cuts.append(ycyl(cx, cz, r + 1.0, S.SPOOL["y0"] - 0.6, S.SPOOL["y1"] + 0.6))
    cuts.append(ycyl(cx, cz, ra + 0.8, gy0 - 0.5, gy1 + 0.5)); cuts.append(ycyl(cx, pz, rp + 0.8, gy0 - 0.5, gy1 + 0.5))
    cuts.append(ycyl(cx, pz, 3.5 + 0.3, gy1, gy1 + 0.9))                                            # 15 Nov (assembly check): the pinion's O7 hub boss runs to gy1 + 0.7, the sweep stopped at + 0.5
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
    Ft = round(S.SPOOL_STALL / S.SPOOL["hub"], 1)                                  # 12 Nov: the N20's stall 2.2 kg cm (E, the listings' top) through 40 / 12 on the hub: 44 N (was 28)
    for sgn, lab in ((1, "paying out"), (-1, "pulling in")):
        p.load_multi(f"tape tension at the N20's stall, {lab} ({Ft:.0f} N)", [(B["axle"][0], (0, 0, -Ft / 2 * sgn)), (B["axle"][1], (0, 0, -Ft / 2 * sgn))], cls="flight")
    u = B["roller_u"]; Fl = S.LEAF["F_case"]; fu = (Fl * u[0], 0.0, Fl * u[1])   # the leaf pushes the end wall out and the coil (the spool's axle) in
    p.load_multi(f"the roller's leaf on the full coil ({S.LEAF['F_case']:.0f} N)", [(B["roller_seats"][0], fu),
                 (B["axle"][0], (-fu[0] / 2, 0, -fu[2] / 2)), (B["axle"][1], (-fu[0] / 2, 0, -fu[2] / 2))], cls="flight")
    own = sum(ASM["scout_v4_" + k]["mass_g"] for k in ("probe_spool", "probe_drum", "probe_feed"))
    payload_cases(p, "spool, blades and cable", [(B["cx"], S.GEAR["y"][1] + 0.8, B["cz"]), (B["cx"], S.DRUM["y0"] - 0.8, B["cz"])], B["axle"], own, (B["cx"], 0, B["cz"]), flight=False)
    socket_cases(p, B["socks"])
    return p
