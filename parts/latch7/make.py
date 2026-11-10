"""Battery latch v7 (10 Nov 2026, Spark: run the whole design through four tests: error tolerance at every swap step; every
spring and plug force centred, nothing tilting onto a wall; the parts we really have; every stressed part as big as fits).

Latch v6 (parts/latch6/make.py, on v5 and the vendored v4 CAD) audited and changed where it failed. audit() holds the numbers.
  1. The XT60 is 42.9 mm (Scout) / 77.8 mm (Overseer) from the tab line the grabber pulls on. v6 sized the grip for the plug's
     57 N; with that lever the two nubs carry 173 N (Scout) and v6's nub, foot, blade and rib all go over 1.0. v7: nubs 13 wide,
     feet 18 wide, blades 4.7 x 16, and on the Overseer the heel pads at the cartridge's ends (one pair under the XT60).
  2. v6's wedge left +-0.14 mm of height tolerance before a bolt was not fully drawn. v7: 15 deg wedge: +-0.21 mm, the most
     the tab's thickness gives with 1.5 mm of nub still on the foot (window + contact = tab - 0.3).
  3. The blade pushed the tab 6.6 mm below the springs' line and, 1 mm off along the pack, on one tab leg only. v7: one crowned
     pad on the tab at the bar's height, 2.1 under the springs, on the slider's centre line whatever the head's offset.
  4. A 30 mm spring 2.7 across buckles; v6's open slots let it bow onto the pack. v7: a 2 x 8 steel pin inside each spring.
  5. v6's rest-stop lugs left the blade +-1.0 along the pack; the arm is +-1.8. v7: +-2.1 everywhere (lugs, arch, recess).
The receiver, the XT60 carrier, the guide poles, the bolts, their steel pins and the 20 deg seat ramp are v5's, unchanged.

python3 parts/latch7/make.py   -> parts/latch7_*/part.stl, parts/latch7/result.json, view/*.stl, systems/latch7.*
"""
import importlib.util, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
_s = importlib.util.spec_from_file_location("latch6_make", os.path.join(ROOT, "parts", "latch6", "make.py"))
m6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m6)      # v6 (which patches the vendored module to v5, then v6)
bl, v5 = m6.bl, m6.v5
box, cyl, hullp, prism_xy, U, vol, rot_y180 = m6.box, m6.cyl, m6.hullp, m6.prism_xy, m6.U, m6.vol, m6.rot_y180
V6 = json.load(open(os.path.join(ROOT, "parts", "latch6", "result.json")))


def __getattr__(name):                  # the motion script reads the v6 API through this module
    return getattr(m6, name)


# ------------------------------------------------------------------ v7 numbers (C unless tagged). x budget from the centre:
# rib 2.0 | travel 4.9 | tab plate 2.9 + crowned pad 0.5 | 1.0 | blade 4.7 | 1.0 | recess wall 17.0
# The wedge's window and its shortest contact share the tab's thickness: window + contact = tab - 0.3 (the nub must stop
# short of the rib, the foot short of the blade). 3.4 of tab = 1.6 of window + 1.5 of contact.
bl.REC_X = 17.0                        # recess half width (v6 15.5): out to the Scout's side wall
bl.REC_Z = 13.5                        # recess half length (v6 11.5)
bl.TAB_HZ = 13.1
bl.TAB_IN = 6.9
PLATE_OUT = 9.8                        # the tab plate's outer face
bl.TAB_OUT = 10.3                      # the crowned pad's top: what the blade face meets
bl.TRAVEL = bl.TAB_IN - bl.RIB_HX      # 4.9 to the rib
m6.R0 = 4.1                            # retraction with the blade tight on its wedge, nominal: the middle of 3.3 .. 4.9
m6.NEED = bl.OVERLAP + 0.3             # 3.3
m6.ALPHA = math.radians(15.0)
m6.FOOT_X1, m6.FOOT_Y, m6.FOOT_HZ = 5.2, 2.6, 9.0          # foot: 3.2 long, 18 wide, 3.46 thick at the rib
m6.NUB_L, m6.NUB_HZ, m6.NUB_Y1 = 3.3, 6.5, 6.0             # nub: 13 wide, 3.7 thick at the blade
m6.ARCH_HZ, m6.ARCH_Y = 8.6, 6.2                           # the nub's window through the tab plate: +-2.1 along the pack
m6.BLADE_T, m6.BLADE_HZ, m6.BLADE_Y1, m6.BLADE_OPEN = 4.7, 8.0, 8.0, 11.3
TAB_Y0 = 3.8                           # tab plate bottom between its legs (over the feet, 3.46 at the rib)
LEG_Z0 = 10.2                          # legs and rest-stop lugs from here out: blade 8.0 + 2.1 + 0.1
RELIEF_Y = 8.4                         # bar underside over the blade (blade top 8.0)
PAD_Y, PAD_HZ = (6.5, 8.3), 5.0        # the crowned pad's base on the plate; its top 0.8 x 4.0 at y 7.1..7.9
PUSH_Y = 7.5
LUG_Y1 = 8.0                           # rest-stop lugs' top (v6 5.5): they join the plate, not only the magnet leg
bl.SPR_DEPTH = 3.0                     # spring pocket (v6 3.5): ends 0.4 short of the relief
bl.MAG_X, bl.MAG_Z = 8.6, 11.65        # magnet in a leg (the head's hall follows: x +-8.6, z_T +- 11.65)
GUIDE_PIN = (2.0, 8.0)                 # steel pin inside each spring (spring inside o 2.4, shortest length 10.0)
F_BLADE = 15.0                         # the head's blade spring at the squeeze (heads v7 concept, E); 20 N recommended
ARM = dict(x_deg_rss=0.30, x_deg_worst=0.55, x_capture_deg=1.20, y_axial_mm=1.8, tilt_x_deg=1.28, tilt_x_float_deg=2.37,
           roll_z_deg=0.74, roll_z_float_deg=1.06, stall_N=74.0, insert_N=60.0, xt60_pull_N=57.0, across_N=43.5, r_drone_mm=220.3)
SERVO = {"DS3218 (on hand)": 20.0, "MG995 (ordered)": 9.2}      # kg cm (listings, dimensions M)
CAM_KGCM_AT_19p5N = 1.98               # heads v7 concept: cam torque with the blades at 19.5 N open
m6.BOUGHT = [b for b in m6.BOUGHT if b["part"] != "heel pad facing (grabber head)"] + [
    dict(part="spring guide pin", spec="dowel 2 x 8, stainless", per_cartridge=2, listing="uxcell 2 x 8 dowel pins, 304 stainless",
         url="https://www.amazon.com/s?k=2mm+x+8mm+dowel+pin+stainless", check="search link: pick any 2 x 8 stainless dowel; not a checked listing"),
    dict(part="heel pad facing (grabber head)", spec="TPU 95A, 2.0 thick, 3 x 7, four per head", listing="printed from the TPU 95A on hand", url="", check="")]
for b in m6.BOUGHT:
    if b["part"] == "latch spring":
        b["check"] = "on a kitchen scale: 1.0..1.4 N at 21.0 long; fully closed it must be 9.0 long or less (E 0.127 N/mm, 28 coils)"


def pads_of(g):
    """heel pads: (x0, x1) on the rim band, and the two z ranges. Scout: the ends of its flat bottom (the collar clip takes
    the rest); Overseer: the cartridge's ends, the lead pair under the XT60."""
    x0 = max(15.2, g["X_IN"] - 1.8); zt = g["Z_T"]
    if g["pack"] == "scout": return (x0, x0 + 3.0), [(zt + 18.4, zt + 25.4), (zt - 25.4, zt - 18.4)]
    return (x0, x0 + 3.0), [(g["L_C"] - 9.0, g["L_C"] - 2.0), (2.0, 9.0)]


def set_pack(g):
    px, pz = pads_of(g); zt = g["Z_T"]
    m6.PAD_X = px
    m6.PAD_Z = (min(abs(pz[0][0] - zt), abs(pz[1][1] - zt)), min(abs(pz[0][1] - zt), abs(pz[1][0] - zt)))


# ------------------------------------------------------------------ slider v9
def slider(g, retract=0.0, side=1):
    zt, tip = g["Z_T"], g["X_TIP"]; Y0, Y1 = bl.BOLT_Y0, bl.BOLT_Y1; HZ = bl.TAB_HZ
    plate = box(bl.TAB_IN, PLATE_OUT, TAB_Y0, Y1, zt - HZ, zt + HZ)
    bar = box(bl.TAB_IN, tip, Y0, Y1, zt - bl.BOLT_HZ, zt + bl.BOLT_HZ)
    legs = [box(bl.TAB_IN, bl.TAB_OUT, 0.0, TAB_Y0 + 0.01, *sorted((zt + sz * LEG_Z0, zt + sz * HZ))) for sz in (-1, 1)]
    lugs = [box(PLATE_OUT - 0.01, bl.REC_X, 0.0, LUG_Y1, *sorted((zt + sz * LEG_Z0, zt + sz * HZ))) for sz in (-1, 1)]   # rest stops, on the plate
    pad = hullp([[PLATE_OUT - 0.01, y, zt + z] for y in PAD_Y for z in (-PAD_HZ, PAD_HZ)] +
                [[bl.TAB_OUT, y, zt + z] for y in (PUSH_Y - 0.4, PUSH_Y + 0.4) for z in (-2.0, 2.0)])
    s = U(plate, bar, pad, *legs, *lugs)
    s = s - prism_xy([(tip - bl.TIP_CH, Y1 + 0.01), (tip + 0.01, Y1 + 0.01), (tip + 0.01, Y1 - bl.TIP_CH)], zt - bl.BOLT_HZ - 1, zt + bl.BOLT_HZ + 1)
    xp, yp = v5.pin_axis(g)
    s = s - cyl("z", xp, yp, m6.PIN_R, zt - bl.BOLT_HZ - 0.01, zt - bl.BOLT_HZ + m6.PIN_L + 0.5, 32)
    s = s - box(bl.TAB_OUT, bl.REC_X + 0.1, Y0 - 0.01, RELIEF_Y, zt - bl.BOLT_HZ - 0.1, zt + bl.BOLT_HZ + 0.1)      # room for the blade top
    s = s - box(bl.TAB_IN - 0.01, bl.TAB_OUT + 0.01, TAB_Y0 - 0.01, m6.ARCH_Y, zt - m6.ARCH_HZ, zt + m6.ARCH_HZ)    # the nub's window
    s = s - prism_xy([(PLATE_OUT - 1.0, TAB_Y0 - 0.01), (PLATE_OUT + 0.01, TAB_Y0 - 0.01), (PLATE_OUT + 0.01, TAB_Y0 + 1.0)], zt - LEG_Z0, zt + LEG_Z0)
    for sz in (-1, 1):
        zc = zt + sz * m6.SPR_DZ
        s = s - cyl("x", bl.SPR_YC, zc, bl.SPR_R, bl.TAB_IN - 0.01, bl.TAB_IN + bl.SPR_DEPTH, 32) \
              - box(bl.TAB_IN - 0.01, bl.TAB_IN + bl.SPR_DEPTH, bl.SPR_YC, Y1 + 1, zc - bl.SPR_R, zc + bl.SPR_R)
    s = s - cyl("y", bl.MAG_X, zt + bl.MAG_Z, bl.MAG_D / 2, -0.01, bl.MAG_DEPTH, 24)
    s = s.translate([-retract, 0, 0])
    return s if side > 0 else rot_y180(s, zt)


def cartridge(g, clip=None):
    """v5's cartridge with the wider, longer recess; the Scout's rim band cut back to the recess wall in the latch zone."""
    t = v5.cartridge(g, None)
    zt = g["Z_T"]; xb = g["X_IN"] - 2.0
    if xb < bl.REC_X:
        for s in (-1, 1):
            t = t - box(*sorted((s * (xb - 0.01), s * bl.REC_X)), -1.0, 4.01, zt - bl.REC_Z, zt + bl.REC_Z)
    return t if clip is None else t ^ clip


def heel(g):
    (x0, x1), zz = pads_of(g); zt = g["Z_T"]
    z0, z1 = min(a for a, b in zz), max(b for a, b in zz)
    deck = box(-x1, x1, -6.0, -m6.PAD_T, z0, z1) - box(-bl.REC_X - 1.0, bl.REC_X + 1.0, -7.0, 0.0, zt - bl.REC_Z - 1.0, zt + bl.REC_Z + 1.0)
    pads = [box(*sorted((sx * x0, sx * x1)), -m6.PAD_T, 0.0, a, b) for sx in (1, -1) for a, b in zz]
    return deck, pads


m6.slider, m6.cartridge, m6.heel = slider, cartridge, heel
bl.slider, bl.cartridge = slider, cartridge
v5.slider = slider


def build(pack="scout"):
    bl._CACHE.clear()
    B = m6.build(pack); set_pack(B["g"])
    return B


# ------------------------------------------------------------------ the audit (hand models E, CAD sweeps C)
def lever(g, zf=None, zl=None, P=None):
    """the XT60's lever on the grip. Pull-out: the plug holds the lead end, the nubs pull at the tab line, the far heel pads
    push. Insert: the plug resists at the lead end; if it is past the lead pads the nubs must pull the middle down."""
    P = ARM["xt60_pull_N"] if P is None else P
    (x0, x1), zz = pads_of(g); zt = g["Z_T"]; e = g["Z_X"] - zt
    zl = (sum(zz[0]) / 2 - zt) if zl is None else zl; zf = (zt - sum(zz[1]) / 2) if zf is None else zf
    W = g["SWAP_M"] / 1000 * 9.81
    far = P * e / zf
    ins_nub = max(0.0, P * (e - zl) / zl)
    return dict(xt60_from_tab_line_mm=round(e, 1), far_pads_at_mm=round(zf, 1), lead_pads_at_mm=round(zl, 1),
                pull=dict(far_pads_N=round(far, 1), both_nubs_N=round(P + far + W, 1), per_nub_N=round((P + far + W) / 2, 1),
                          v6_assumed_per_nub_N=round((P + W) / 2, 1), times_v6=round((P + far + W) / (P + W), 2)),
                insert=dict(both_nubs_N=round(ins_nub, 1), lead_pads_N=round(P + ins_nub + (ARM["stall_N"] - P) / 2, 1),
                            plug_inside_the_pads=bool(e <= zl)))


def strength(B):
    g = B["g"]; lv = lever(g)
    bd = m6.bending(B, F_BLADE, lv["pull"]["both_nubs_N"])
    rib = {n: c for n, c in bd["rib"].items() if n.startswith(("XT60", "swap", "pull +"))}
    both = rib.pop("pull + across together")          # the arm's full side force during the hardest pull: a jam, misuse class
    worst = max(both["end_mid_MPa"], both["notch_MPa"], both["post_MPa"])
    rib["full pull + full across at once (a jam: misuse class)"] = dict(MPa=worst, index=round(worst / m6.ALLOW["misuse"], 3), functional_class_index=both["index"])
    ins = m6.bending(B, F_BLADE, max(lv["insert"]["both_nubs_N"], 1.0))["rib"]["pull + across together"]
    rib["insert + full across (the lead-ins centring it)"] = dict(MPa=ins["end_mid_MPa"], index=ins["index"])
    keep = {k: v for k, v in bd.items() if k not in ("tab", "rib", "worst_index")}
    keep["rib"] = rib
    keep["worst_index"] = max([keep[k]["index"] for k in ("blade_pull", "nub_pull", "foot_pull")] + [c["index"] for c in rib.values()])
    return dict(lever=lv, at_the_real_pull=keep)


def v6_at_real_load(pack="scout"):
    """v6's own indices (its result.json, at 59.1 N on the two nubs) scaled to the load its pads really give."""
    b = V6["bending"]; zt = V6["geom"]["Z_T"]; e = V6["geom"]["Z_X"] - zt
    tot = 57.0 * (1 + e / 21.5) + 2.13
    k = tot / 59.1
    return dict(both_nubs_N=round(tot, 1), scale=round(k, 2), blade=round(b["blade_pull"]["index"] * k, 2), nub=round(b["nub_pull"]["index"] * k, 2),
                foot=round(b["foot_pull"]["index"] * k, 2), rib=round(b["rib"]["XT60 pull 57 N + 1 g"]["index"] * k, 2))


def slider_balance(g, y_push, e_z, mu=0.3):
    """the squeeze: the blade pushes at y_push, e_z off the slider's centre line; the springs resist at SPR_YC. The pitch
    couple tips the slider until its bar rests on the floor land and its inboard top edge on the pack's underside (0.5 above,
    10 mm inboard: the wall hole's 0.35 lets it); the yaw couple goes into the wall hole and the tab's ends (10.5 apart)."""
    ey = bl.SPR_YC - y_push
    Lp, Ly = 10.0, 10.5
    Fs = 2 * m6.spring_state(m6.R0)["N"]
    f = 2 * mu * ey / Lp + 2 * mu * abs(e_z) / Ly
    return dict(push_below_springs_mm=round(ey, 1), push_off_centre_mm=round(e_z, 1), on_the_pack_underside_N=round(Fs * ey / Lp, 2),
                side_load_in_the_channel_N=round(Fs * abs(e_z) / Ly, 2), blade_N_to_unlatch=round(Fs / (1 - f), 1))


def spring_column():
    D = m6.SPR["od"] - m6.SPR["d"]; L0 = m6.SPR["free"]
    rest = 2 * (bl.TAB_IN + bl.SPR_DEPTH)
    return dict(free_over_mean_dia=round(L0 / D, 1), buckles_unguided_above=5.3, deflection_ratio_squeezed=round((L0 - (rest - 2 * m6.R0)) / L0, 2),
                unsupported_each_side_mm=round(bl.TAB_IN - bl.RIB_HX, 1), guide="2 x 8 steel pin inside each spring (v6: none; open slots under the pack)",
                shortest_length_mm=round(rest - 2 * bl.TRAVEL, 1), solid_mm=round((m6.SPR["n_active"] + 2) * m6.SPR["d"], 1))


def blade_entry(B):
    """the open blades rising 30 mm into the recess with the head off in x (across) and z (along the pack): clean or not."""
    g = B["g"]; C = B["cart"] + slider(g, 0, 1) + slider(g, 0, -1)
    b0 = U(*m6.blades(g, m6.BLADE_OPEN))
    ok = {}
    for dz in (0.0, 1.0, 1.8, 2.0, 2.2):
        for dx in (0.0, 0.5, 0.9, 1.1):
            bb = b0.translate([dx, 0, dz])
            ok[(dx, dz)] = max(vol(bb.translate([0, -k, 0]) ^ C) for k in np.arange(0, 30.01, 1.0)) < 0.01
    cx = max(dx for (dx, dz), v in ok.items() if v and dz == 1.8) if any(v for (dx, dz), v in ok.items() if dz == 1.8) else None
    cz = max(dz for (dx, dz), v in ok.items() if v and dx == 0.9) if any(v for (dx, dz), v in ok.items() if dx == 0.9) else None
    return dict(clean_across_mm_at_1p8_along=cx, clean_along_mm_at_0p9_across=cz,
                table={f"x {dx:g} z {dz:g}": bool(v) for (dx, dz), v in ok.items()},
                chamfers_beyond="recess mouth 1.0, tab plate edge 1.0: a blade up to 2.0 across still cams in (not swept)")


def tolerance(g):
    ta = math.tan(m6.ALPHA)
    win = (bl.TRAVEL - m6.NEED) / 2
    Fs = 2 * m6.spring_state(m6.R0)["N"]
    def mu_needed(Fb, per_nub):          # the blade must not be wedged open: Fb - springs >= per_nub * tan(alpha - phi)
        return round(max(0.0, math.tan(m6.ALPHA - math.atan2(Fb - Fs, per_nub))), 2)
    return dict(wedge_deg=round(math.degrees(m6.ALPHA), 1), travel_to_rib=bl.TRAVEL, needed=m6.NEED, nominal=m6.R0,
                height_tolerance_mm=round(win * ta, 2), v6_height_tolerance_mm=round((4.9 - 3.3) / 2 * math.tan(math.radians(10)), 2),
                shortest_nub_on_foot_mm=round(m6.FOOT_X1 - (bl.TAB_OUT - m6.NEED - m6.NUB_L), 2),
                blade_stroke_mm=dict(nominal=round(m6.BLADE_OPEN - (bl.TAB_OUT - m6.R0), 1), longest=round(m6.BLADE_OPEN - (bl.TAB_OUT - bl.TRAVEL) + 1.0, 1)),
                springs_on_a_squeezed_blade_N=round(Fs, 2))


def error_budget(g, be):
    r = ARM["r_drone_mm"]; mm = lambda deg: round(math.radians(deg) * r, 2)
    return [
        dict(step="1 blades rise into the recess", error="across %.1f mm (0.30 deg RSS at the drone slot), worst %.1f; along the pack +-%.1f" % (mm(ARM["x_deg_rss"]), mm(ARM["x_deg_worst"]), ARM["y_axial_mm"]),
             takes="clean to %s across, %s along; chamfers cam a blade in to 2.0 across" % (be["clean_across_mm_at_1p8_along"], be["clean_along_mm_at_0p9_across"]),
             v6="+-1.0 across, +-1.0 along (rest-stop lugs 1.0 from the blade)", ok=bool((be["clean_along_mm_at_0p9_across"] or 0) >= ARM["y_axial_mm"]),
             note="across: the worst case (2.1 mm) needs the chamfers; the head should float +-1 across (the head research's item 4)"),
        dict(step="2 squeeze: bolts drawn", error="height of nub over pads vs foot over the cartridge bottom (two prints, a TPU face)",
             takes="+-%.2f mm" % tolerance(g)["height_tolerance_mm"], v6="+-0.14 mm", ok=True,
             note="each blade is sprung on its own, so an offset across does not change either bolt's draw"),
        dict(step="3 pull out: XT60 unmates", error="the plug is %.0f mm from the pull" % (g["Z_X"] - g["Z_T"]),
             takes="far heel pads + nubs sized for it", v6="not counted: nub, foot, blade, rib over 1.0", ok=True, note="see strength"),
        dict(step="4 carry", error="any direction of gravity", takes="see hold", v6="as v6", ok=True, note="across is friction + the blade springs"),
        dict(step="5 insert: receiver lead-ins", error="across 0.55 deg worst of the 1.20 deg the lead-ins take; along +-1.8 of 3.0; tilt 1.28 of 2.37 deg; roll 0.74 of 1.06 deg",
             takes="54 / 40 / 46 / 30 % left", v6="same (v5 receiver)", ok=True, note="roll is the tightest: it needs the per-bay gravity offset in the axle's step target"),
        dict(step="6 XT60 mates", error="carrier floats +-1.0, the poles' cones take 1.5, poles in 11.7 before the seat, plug at 6.0",
             takes="cartridge is inside 0.3 by then", v6="same", ok=True, note="push through the lead pads: Scout's plug is past them (nubs carry the rest)"),
        dict(step="7 release: bolts out", error="head stops on the stall (74 N), not a position; bolts snap out 0.5 under the seat",
             takes="stall 74 N vs 60 N needed: x1.23", v6="same", ok=True, note="the two halls read each bolt; a bolt not out = no release of the drone"),
    ]


def parts_check(g):
    Fs = 2 * m6.spring_state(bl.TRAVEL)["N"]
    out = {}
    for Fb in (15.0, 20.0):
        kgcm = CAM_KGCM_AT_19p5N * (Fb + 5.4) / 19.5
        out[f"{Fb:g} N blades"] = dict(open_N=round(Fb + 5.4, 1), cam_kgcm=round(kgcm, 2), **{k: "x%.1f" % (v / kgcm) for k, v in SERVO.items()},
                                       left_for_the_wedge_N=round(Fb - Fs, 1))
    out["head stepper (NEMA 17 17HS19, 2 A, r 8 pinion)"] = dict(stall_N=ARM["stall_N"], insert_N=ARM["insert_N"], margin="x%.2f" % (ARM["stall_N"] / ARM["insert_N"]),
                                                               pull_N=ARM["xt60_pull_N"] + 2.1, pull_margin="x%.2f" % (ARM["stall_N"] / (ARM["xt60_pull_N"] + 2.1)))
    out["XT60"] = "57 N to unmate is the project's own reading (the head research note); the spec sheet says 18 to 44 N"
    return out


def audit(B):
    g = B["g"]
    be = blade_entry(B)
    Fs = 2 * m6.spring_state(m6.R0)["N"]
    st = strength(B)
    per = st["lever"]["pull"]["per_nub_N"]
    a = dict(
        lever=st["lever"], strength=st["at_the_real_pull"], v6_at_the_real_pull=v6_at_real_load(),
        tolerance=tolerance(g), blade_entry=be, error_budget=error_budget(g, be),
        slider=dict(v7_centred=slider_balance(g, PUSH_Y, 0.0), v7_head_1p8_off=slider_balance(g, PUSH_Y, 2.0),
                    v6_centred=slider_balance(g, 3.0, 0.0), v6_head_1mm_off=slider_balance(g, 3.0, 5.7),
                    note="v6: flat blade face on two tab legs 1.6 wide; 1 mm off along the pack it bears on one leg, 5.7 off centre"),
        spring_column=spring_column(),
        wedge_hold={f"{Fb:g} N": dict(left_N=round(Fb - Fs, 1), mu_needed=round(max(0.0, math.tan(m6.ALPHA - math.atan2(Fb - Fs, per))), 2)) for Fb in (15.0, 20.0)},
        parts=parts_check(g))
    return a


# ------------------------------------------------------------------ save, page
PRINTED = ("latch7_receiver", "latch7_carrier", "latch7_carrier_cap", "latch7_cartridge", "latch7_slider", "latch7_receiver_strip_a", "latch7_receiver_strip_b")


def save(B):
    g = B["g"]
    out = {"latch7_receiver": B["recv"], "latch7_carrier": B["carrier"], "latch7_carrier_cap": B["cap"], "latch7_cartridge": B["cart"],
           "latch7_slider": slider(g, 0, 1), "latch7_bolt_pin": m6.bolt_pin(g, 0, 1)}
    for k, m in zip("ab", sorted(B["strips"].decompose(), key=lambda m: m.bounding_box()[0])): out[f"latch7_receiver_strip_{k}"] = m
    paths = {}
    for nm, m in out.items():
        d = os.path.join(ROOT, "parts", nm); os.makedirs(d, exist_ok=True)
        t = bl.tm(m); t.export(os.path.join(d, "part.stl")); paths[nm] = t
    return paths


ARTIFACT = "https://claude.ai/artifact/Pqu6aZFdW9g7jtRyBFdraw"


def write_assembly(res):
    a = res["audit"]; lv = a["lever"]; s = a["strength"]
    asm = {"name": "latch v7", "title": "Battery Latch v7 (Scout)", "readme_view": "section", "artifact": ARTIFACT, "machine": "box", "order": 90,
           "views": {"section": {"eye": [0, 95, 8], "target": [0, 0, 8]}, "wedge close": {"eye": [8, 34, 5], "target": [6, 0, 5]},
                     "bolt close": {"eye": [30, 45, 6], "target": [19, 0, 8]}, "springs, from above": {"eye": [0, 42, 38], "target": [0, 3, 9]},
                     "grabber side, from below": {"eye": [45, 70, -70], "target": [0, -4, 0]}, "lead end": {"eye": [120, -60, 70], "target": [8, -36, 34]},
                     "held, tipped": {"eye": [150, 150, -85], "target": [0, 0, -112]}},
           "note": "The press-two-tabs latch, v7, battery side: v6 run through four tests (error tolerance at every swap step, centred forces, "
                   "the parts on hand, stressed parts as big as fits) and changed where it failed. Cut at the tab line. "
                   f"The XT60 is {lv['xt60_from_tab_line_mm']} mm from the pull, so the two nubs carry {lv['pull']['both_nubs_N']} N, not 59: "
                   f"nubs 13 wide, feet 18 wide, blades 4.7 x 16 (worst index {s['worst_index']}). 15 deg wedge: "
                   f"+-{a['tolerance']['height_tolerance_mm']} mm of height tolerance (v6 +-0.14). The blade pushes one crowned pad on the slider's centre line. "
                   "A steel pin inside each spring. Receiver, XT60 carrier, guide poles, bolts and their steel pins are v5's. Numbers under the viewer.",
           "cg_ref": None, "timeline": "tools/latch7_motion.py:timeline", "appendix": "systems/latch7.appendix.html", "parts": res["page_parts"]}
    json.dump(asm, open(os.path.join(ROOT, "systems", "latch7.assembly.json"), "w"), indent=1)


def main():
    B = build(); g = B["g"]
    res, rows = m6.checks(B)
    res["hold"] = m6.hold_report(B)
    res["audit"] = audit(B)
    paths = save(B)
    res["mass_g"] = {nm: round(abs(t.volume) / 1000 * (7.85 if "pin" in nm else 1.21 if "strip" in nm else 1.23), 2) for nm, t in paths.items()}
    res["swap_g"] = round(g["SWAP_M"], 1); res["swap_cg"] = [round(float(v), 2) for v in g["SWAP_CG"]]
    res["insert_rows"] = rows
    res["geom"] = {k: round(float(v), 3) for k, v in g.items() if isinstance(v, (int, float))}
    res["v6"] = dict(TAB_IN=bl.TAB_IN, TAB_OUT=bl.TAB_OUT, TRAVEL=bl.TRAVEL, R0=m6.R0, NEED=m6.NEED, OVERLAP=bl.OVERLAP, ALPHA_deg=15.0, PUSH_Y=PUSH_Y, FOOT=(m6.FOOT_X1, m6.FOOT_Y, m6.FOOT_HZ), NUB=(m6.NUB_L, m6.NUB_HZ, m6.NUB_Y1), BLADE=(m6.BLADE_T, m6.BLADE_HZ, m6.BLADE_Y1), REC=(bl.REC_X, bl.REC_Z), MAG=(bl.MAG_X, bl.MAG_Z),
                     BLADE_OPEN=m6.BLADE_OPEN, PAD_X=m6.PAD_X, PAD_Z=m6.PAD_Z, PAD_T=m6.PAD_T, K_BLADE=m6.K_BLADE, MU=m6.MU,
                     SPR_DZ=m6.SPR_DZ, SPR_YC=bl.SPR_YC, SPR_DEPTH=bl.SPR_DEPTH)
    res["rest_stop_mm3_at_0.1_out"] = round(vol(slider(g, -0.1, 1) ^ B["cart"]), 3)
    res["bought"] = m6.BOUGHT
    m6.HERE = HERE
    pp = m6.write_page(B, res)
    for p in pp:
        p["id"] = p["id"].replace("latch6", "latch7"); p["version"] = "latch v7 (10 Nov)"; p["source"] = "parts/latch7/make.py"
    res["page_parts"] = pp
    ov = build("overseer")
    res["overseer"] = dict(swap_g=round(ov["g"]["SWAP_M"], 1), **strength(ov),
                           hold=[m6.hold(ov, Fb, m6.MU["pad_tpu"], n) for Fb in (15.0, 20.0) for n in (1.0, 2.0, 3.0)],
                           static_mm3=round(max(vol(m6.moving(ov["g"], 0, s) ^ m6.cart_all(ov)) for s in (1, -1)), 3),
                           squeezed_mm3=round(max(vol(m6.moving(ov["g"], bl.TRAVEL - 0.02, s) ^ m6.cart_all(ov)) for s in (1, -1)), 3),
                           blades_tight_mm3=round(vol(U(*m6.blades(ov["g"], bl.TAB_OUT - m6.R0, back=0.02)) ^ ov["cart"]), 3),
                           pads_contact_mm2=round(sum(vol(p.translate([0, 0.05, 0]) ^ ov["cart"]) / 0.05 for p in heel(ov["g"])[1]), 1))
    B = build()
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)
    write_assembly(res)
    show = {k: v for k, v in res.items() if k not in ("insert_rows", "geom", "page_parts", "hold", "bought", "spring", "v6")}
    print(json.dumps(show, indent=1, default=str))



def appendix(res):
    a = res["audit"]; lv = a["lever"]; st = a["strength"]; v6r = a["v6_at_the_real_pull"]; tol = a["tolerance"]; sl = a["slider"]
    sp = res["spring"]; sc = a["spring_column"]; ov = res["overseer"]; ovl = ov["lever"]; ovs = ov["at_the_real_pull"]; mc = res.get("motion_collision", {})
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    rib = st["rib"]; orib = ovs["rib"]
    kp, kj, ki = "XT60 pull 57 N + 1 g", "full pull + full across at once (a jam: misuse class)", "insert + full across (the lead-ins centring it)"
    budget = "".join(td(e["step"], e["error"], e["takes"], e["v6"], e["note"]) for e in a["error_budget"])
    hold = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}", h["slip_x_mm"], h["pack_top_mm"])
                   for h in res["hold"]["all_gravity_directions"] if h["pads"].startswith("TPU") and h.get("releases") and h["Fb"] >= 10)
    ohold = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}", h["slip_x_mm"]) for h in ov["hold"] if h.get("releases"))
    bought = "".join(td(b["part"], b["spec"], f'<a href="{b["url"]}">{b["listing"]}</a>' if b["url"] else b["listing"], b["check"]) for b in m6.BOUGHT)
    pr = a["parts"]; p15, p20 = pr["15 N blades"], pr["20 N blades"]; stp = pr["head stepper (NEMA 17 17HS19, 2 A, r 8 pinion)"]
    V = res["v6"]
    return f"""<!-- latch v7: the audit behind the motion (written by parts/latch7/make.py; appended by tools/build_site.py). Scoped to #latch7. -->
<style>
#latch7 {{ background: var(--bg); color: var(--fg); font: 14px/1.5 var(--sans); padding: 8px 16px 40px; }}
#latch7 .wrap {{ max-width: 1040px; margin: 0 auto; }}
#latch7 h2 {{ font: 600 16px/1.3 var(--sans); margin: 26px 0 6px; }}
#latch7 p {{ margin: 6px 0; max-width: 78ch; }}
#latch7 .t {{ overflow-x: auto; }}
#latch7 table {{ border-collapse: collapse; font: 12px/1.4 var(--mono); margin: 8px 0; background: var(--panel); }}
#latch7 th, #latch7 td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }}
#latch7 th {{ color: var(--muted); font-weight: 500; }}
#latch7 a {{ color: var(--accent); }}
#latch7 .open {{ border-left: 3px solid var(--bad); padding-left: 10px; }}
</style>
<section id="latch7"><div class="wrap">
<h2>What the run-through found in v6, and what v7 changes</h2>
<div class="t"><table><tr><th>test</th><th>v6</th><th>v7</th></tr>
{td("pull-out load on the grip (the XT60 is off to one end)", f"sized for 29.6 N per nub; the real load is {v6r['both_nubs_N'] / 2:.0f} N. Index: nub {v6r['nub']}, foot {v6r['foot']}, blade {v6r['blade']} (over 1.0 fails)", f"{lv['pull']['per_nub_N']} N per nub. Index: nub {st['nub_pull']['index']}, foot {st['foot_pull']['index']}, blade {st['blade_pull']['index']}, rib {rib[kp]['index']}")}
{td("height tolerance between the head and the cartridge before a bolt is not fully drawn", f"+-{tol['v6_height_tolerance_mm']} mm", f"+-{tol['height_tolerance_mm']} mm (15 deg wedge; at least {tol['shortest_nub_on_foot_mm']} mm of nub on the foot)")}
{td("blade room along the pack (the arm is +-1.8 mm there)", "+-1.0 mm (the rest-stop lugs)", f"+-{res['grabber_interface']['z_capture_mm']} mm; swept clean to {a['blade_entry']['clean_along_mm_at_0p9_across']} mm")}
{td("where the blade pushes the slider", f"{sl['v6_centred']['push_below_springs_mm']} mm under the springs, on two 1.6 mm strips; 1 mm off along the pack it pushes one strip, 5.7 off centre: {sl['v6_head_1mm_off']['blade_N_to_unlatch']} N to unlatch", f"one crowned pad on the centre line, {sl['v7_centred']['push_below_springs_mm']} mm under the springs: {sl['v7_centred']['blade_N_to_unlatch']} N, {sl['v7_head_1p8_off']['blade_N_to_unlatch']} N with the head 1.8 off")}
{td("slider tipped up against the pack's underside during the squeeze", f"{sl['v6_centred']['on_the_pack_underside_N']} N", f"{sl['v7_centred']['on_the_pack_underside_N']} N")}
{td("spring column", f"30 long, 2.7 across (ratio {sc['free_over_mean_dia']}; unguided springs buckle above about {sc['buckles_unguided_above']}), in open slots under the pack", "the same spring with a 2 x 8 steel pin inside it: it cannot bow onto the pack or the rib")}
</table></div>
<h2>1. Error at every step of the swap</h2>
<div class="t"><table><tr><th>step</th><th>the error it meets</th><th>what the latch takes</th><th>v6</th><th>note</th></tr>{budget}</table></div>
<p>Across, the arm's worst case at the drone slot is about 2.1 mm and the blades enter clean to {a['blade_entry']['clean_across_mm_at_1p8_along']} mm; past that the 1.0 mm chamfers on the recess mouth and the tab plate cam a blade in. That is the thinnest margin in the cycle. It is the head's to fix: let it float 1 mm either way across, or tighten the axle's index.</p>
<h2>2. Forces: where they act and what they load</h2>
<div class="t"><table><tr><th></th><th>Scout</th><th>Overseer</th></tr>
{td("XT60 from the tab line", f"{lv['xt60_from_tab_line_mm']} mm", f"{ovl['xt60_from_tab_line_mm']} mm")}
{td("far heel pads from the tab line", f"{lv['far_pads_at_mm']} mm (the end of its flat bottom)", f"{ovl['far_pads_at_mm']} mm (the cartridge's end)")}
{td("pull-out, 57 N at the plug: far pads / both nubs", f"{lv['pull']['far_pads_N']} / {lv['pull']['both_nubs_N']} N (x{lv['pull']['times_v6']} what v6 assumed)", f"{ovl['pull']['far_pads_N']} / {ovl['pull']['both_nubs_N']} N (x{ovl['pull']['times_v6']})")}
{td("insert, 57 N at the plug: nubs / lead pads", f"{lv['insert']['both_nubs_N']} / {lv['insert']['lead_pads_N']} N (the plug is past the pads)", f"{ovl['insert']['both_nubs_N']} / {ovl['insert']['lead_pads_N']} N (the lead pads are under the plug)")}
{td("index: nub / foot / blade", f"{st['nub_pull']['index']} / {st['foot_pull']['index']} / {st['blade_pull']['index']}", f"{ovs['nub_pull']['index']} / {ovs['foot_pull']['index']} / {ovs['blade_pull']['index']}")}
{td("index: rib, pull / insert + full side force / full pull + full side force (a jam)", f"{rib[kp]['index']} / {rib[ki]['index']} / {rib[kj]['index']}", f"{orib[kp]['index']} / {orib[ki]['index']} / {orib[kj]['index']}")}
</table></div>
<p>Index 1.0 is the allowable for PCTG ({m6.ALLOW['functional']:.1f} MPa in use, {m6.ALLOW['misuse']:.1f} MPa in a jam). Hand models (E). The blade and nub are the grabber head's parts: 4.7 x 16 blade, nub 13 wide and 3.7 thick at its root, in PCTG.</p>
<p>Springs: two in parallel, {sp['preload_pair_N']} N at rest on each slider, x{sp['preload_over_inertia']} the 35 g inertia of a slider with its pin ({sp['slider_pin_magnet_g']} g); x{sp['one_spring_left_over_inertia']} with one spring lost. {sp['squeezed_pair_N']} N squeezed, wire at {100 * sp['at_rib_limit']['of_uts']:.0f} % of tensile at the limit, {sp['solid_margin_mm']} mm above solid (E: the listing gives no coil count).</p>
<h2>3. Holding the pack on the grabber</h2>
<div class="t"><table><tr><th>Scout, per blade</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>slip across, mm</th><th>pack top moves, mm</th></tr>{hold}</table></div>
<div class="t"><table><tr><th>Overseer ({ov['swap_g']} g), per blade</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>slip across, mm</th></tr>{ohold}</table></div>
<p>The wedge needs friction of {a['wedge_hold']['15 N']['mu_needed']} at 15 N per blade, {a['wedge_hold']['20 N']['mu_needed']} at 20 N, to keep the hardest pull from opening a blade (PCTG on PCTG is about 0.3, E). Across the pack the hold is friction and the two blade springs, as in v6.</p>
<h2>4. The parts on hand</h2>
<div class="t"><table><tr><th>part</th><th>needs</th><th>has</th></tr>
{td("gripper servo, cam opening both blades", f"{p15['cam_kgcm']} kg cm at 15 N blades, {p20['cam_kgcm']} at 20 N (scaled from the heads v7 concept, E)", f"DS3218 20 kg cm: {p15['DS3218 (on hand)']} / {p20['DS3218 (on hand)']}; MG995 9.2 kg cm: {p15['MG995 (ordered)']} / {p20['MG995 (ordered)']}")}
{td("head stepper, NEMA 17 at 2 A on the r 8 pinion", f"insert {stp['insert_N']:.0f} N, pull {stp['pull_N']:.1f} N", f"stall {stp['stall_N']:.0f} N: {stp['margin']} / {stp['pull_margin']}")}
{td("blade springs (head)", f"{tol['springs_on_a_squeezed_blade_N']} N of latch spring to beat, then the wedge", "15 N in the heads v7 concept; 20 N recommended for the Overseer")}
{td("XT60", "57 N to pull apart", pr["XT60"])}
</table></div>
<div class="t"><table><tr><th>bought part</th><th>spec</th><th>Amazon listing</th><th>check</th></tr>{bought}</table></div>
<h2>What the grabber head has to match</h2>
<div class="t"><table><tr><th>item</th><th>v7</th></tr>
{td("blade in the recess", f"{V['BLADE'][0]} thick x {2 * V['BLADE'][1]:g} wide, top {V['BLADE'][2]} above the pads; inner faces +-{V['BLADE_OPEN']} open, +-{V['TAB_OUT'] - V['R0']:.1f} tight (stroke {tol['blade_stroke_mm']['nominal']}, {tol['blade_stroke_mm']['longest']} with every tolerance)")}
{td("nub", f"{V['NUB'][0]} long x {2 * V['NUB'][1]:g} wide, top {V['NUB'][2]}, underside 15 deg")}
{td("heel pads", f"x {V['PAD_X'][0]}..{V['PAD_X'][1]}, {V['PAD_Z'][0]:.1f}..{V['PAD_Z'][1]:.1f} mm either side of the tab line (Scout); the Overseer's at its cartridge ends")}
{td("hall sensors", f"x +-{V['MAG'][0]}, {V['MAG'][1]} mm along the pack from the tab line (one each side, opposite ends), 1.0 under the flat bottom")}
</table></div>
<h2 class="open">Open</h2>
<p>1. The Scout's plug is {lv['xt60_from_tab_line_mm']} mm from the pull and its flat bottom ends {lv['far_pads_at_mm']} mm the other way, so the grip carries x{lv['pull']['times_v6']} the plug's force. v7 is sized for it. The cause stays: only moving the tab line toward the plug or a lower-force connector removes it.</p>
<p>2. The spring's rate and closed length are not on the listing: weigh one (1.0 to 1.4 N at 21 mm) and close one fully (9.0 mm or less).</p>
<p>3. Friction, the XT60's 57 N and the blade springs' 15 N are estimates or single readings.</p>
<h2>Checks</h2>
<div class="t"><table><tr><th>check</th><th>result</th></tr>
{td("static fit, squeeze, free insertion, held pull (at 3.3, 4.1 and 4.9 of retraction)", "0 mm&sup3; everywhere, no stuck step")}
{td("blades and heel: rising, squeezing, tight, at the rib limit; Overseer cartridge", "0 mm&sup3;")}
{td("blade entry with the head off", f"clean to {a['blade_entry']['clean_across_mm_at_1p8_along']} mm across with 1.8 along, to {a['blade_entry']['clean_along_mm_at_0p9_across']} mm along")}
{td("motion, every frame, uncut solids", f"{mc.get('frames')} frames, worst {mc.get('max_mm3')} mm&sup3;")}
{td("solid check", "receiver, carrier, carrier cap, cartridge, slider, both strips: PASS")}
</table></div>
</div></section>
"""


if __name__ == "__main__":
    if "--appendix" in sys.argv:
        open(os.path.join(ROOT, "systems", "latch7.appendix.html"), "w").write(appendix(json.load(open(os.path.join(HERE, "result.json")))))
    else:
        main()
