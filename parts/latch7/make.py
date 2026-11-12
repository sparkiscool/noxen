"""Battery latch v7 (10 Nov 2026, Spark: run the whole design through four tests: error tolerance at every swap step; every
spring and plug force centred, nothing tilting onto a wall; the parts we really have; every stressed part as big as fits).

Latch v6 (parts/latch6/make.py, on v5 and the vendored v4 CAD) audited and changed where it failed. audit() holds the numbers.
  1. The XT60 is 42.9 mm (Scout) / 77.8 mm (Overseer) from the tab line the grabber pulls on. v6 sized the grip for the plug's
     57 N; with that lever the two nubs carry 173 N (Scout) and v6's nub, foot and blade all go over 1.0. v7: nubs 13 wide,
     feet 18 wide, blades 4.7 x 16, and on the Overseer the heel pads at the cartridge's ends (one pair under the XT60).
  2. v6's wedge left +-0.14 mm of height tolerance before a bolt was not fully drawn. v7: 15 deg wedge: +-0.21 mm, the most
     the tab's thickness gives with 1.5 mm of nub still on the foot (window + contact = tab - 0.3).
  3. The blade pushed the tab 6.1 mm below the springs' line on two 1.6 mm strips; 1 mm off along the pack the push is 3.4 off centre. v7: one crowned
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
GUIDE_PIN = (2.0, 8.0)                 # pin inside each spring (spring inside o 2.4, shortest length 10.0): printed PCTG, nothing bought
bl.BOLT_HZ = 11.0                      # bolt bar 22 wide (v6 18): it takes the same 3 x 20 dowel as the guide poles (Spark 10 Nov: one dowel size)
v5.PIN_L = m6.PIN_L = 20.0
F_BLADE = 15.0                         # the head's blade spring at the squeeze (heads v7 concept, E); 20 N recommended
ARM = dict(x_deg_rss=0.30, x_deg_worst=0.55, x_capture_deg=1.20, y_axial_mm=1.8, tilt_x_deg=1.28, tilt_x_float_deg=2.37,
           roll_z_deg=0.74, roll_z_float_deg=1.06, stall_N=74.0, insert_N=60.0, xt60_pull_N=57.0, across_N=43.5, r_drone_mm=220.3)
SERVO = {"DS3218 (on hand)": 20.0, "MG995 (ordered)": 9.2}      # kg cm (listings, dimensions M)
CAM_KGCM_AT_19p5N = 1.98               # heads v7 concept: cam torque with the blades at 19.5 N open
DOWEL = dict(part="dowel pin (bolt pins and guide poles)", spec="3 x 20, stainless, DIN 7", per_cartridge=2, per_receiver=2,
             listing="M3 x 20 dowel pins A2-70 DIN 7", url="https://www.amazon.com/Pieces-M3x20mm-Metric-Dowel-Stainless/dp/B09X5ZC9B6",
             check="one size for both jobs; 50 pcs: https://www.amazon.com/Dowel-Pin-Std-M3x20mm-PK50/dp/B007IA6NTC")
m6.BOUGHT = [dict(b, check="on a kitchen scale: 1.0..1.4 N at 19.8 long; fully closed it must be 9.0 long or less (E 0.127 N/mm, 28 coils)") for b in m6.BOUGHT if b["part"] == "latch spring"] + [DOWEL] + \
    [b for b in m6.BOUGHT if b["part"] == "tab magnet"] + [
    dict(part="XT60 pair", spec="Amass XT60W, male + female (Spark 10 Nov)", per_cartridge=1, listing="ELFCULB 10 pairs Amass XT60W",
         url="https://www.amazon.com/dp/B0DZ288QD2", check="the listing says over 100 cycles and 35 A; the receiver's carrier is drawn for the XT60E1-M panel mount: measure this plug's body before printing a carrier"),
    dict(part="spring guide pin", spec="o 2.0 x 8, printed PCTG", per_cartridge=2, listing="printed", url="", check=""),
    dict(part="heel pad facing (grabber head)", spec="TPU 95A, 2.0 thick, 3 x 7, four per head", listing="printed from the TPU 95A on hand", url="", check="")]


# ------------------------------------------------------------------ the stronger module (10 Nov, Spark: "I don't think the latches or the fingers
# are strong enough to hold the larger battery but both need to be the same so make both stronger"). The Scout has no width to give,
# both packs have length: everything grows along the pack. Bolt bar 42 wide on two 3 x 20 pins end to end (was 22 on one); blade 34
# wide with two hooks 25 apart (was 16 with one), so the pack cannot rock on the grabber; four springs (was two) for the heavier
# bolt; the rib's feet run wall to wall and its ends are gusseted; 25 N per blade asked of the head (was 15).
bl.BOLT_HZ = 21.0
v5.PIN_L = m6.PIN_L = 40.0             # two 3 x 20 dowels end to end in one blind bore
bl.TAB_HZ, bl.REC_Z = 22.2, 22.6
m6.BLADE_HZ = 17.0
NUB_ZC = 12.5; m6.NUB_HZ = 4.5         # two hooks per blade, 9 wide, at z_T +- 12.5
WIN = (NUB_ZC - m6.NUB_HZ - 2.1, NUB_ZC + m6.NUB_HZ + 2.1)      # their windows through the tab plate (+-2.1 along the pack)
LEG_Z0 = m6.BLADE_HZ + 2.2             # rest-stop lugs from here out
LUG_X0 = m6.FOOT_X1 + bl.TRAVEL + 0.3  # the lugs' low part starts here: clear of the feet at full travel
m6.FOOT_HZ = bl.REC_Z                  # feet wall to wall
# Installable (12 Nov, the build check: with the rib part of the cartridge a bolt with its tab plate could move 4.9 in and needed
# 6.5 to leave its wall hole, so it could not be put in at all, from above or below; true since the rib was added). Each bolt
# is now a plain bar the size of its wall hole: it goes in from OUTSIDE, with the pack in or out. What used to be on the tab plate:
#   rest stop  -> one printed stop bar dropped in from above, keyed in a notch in the rib's top; its T ends sit in a chamber in
#                 each bolt's top and stop the bolts at rest. It also keeps the bolts' inner ends down off the pack.
#   springs    -> four per bolt in long slots along the bolt, bearing on the rib's faces (the rib has no spring notch at all)
#   push pad   -> on a boss under the bolt's inner end, at the blade's height; the wall hole's floor is notched for it
#   magnet     -> in the bolt's underside; the head reads it from a short post
# The rib stays part of the cartridge, fixed at both ends, gusseted, feet wall to wall. A shelf in each recess corner carries the
# bolts' inner ends.
SPRINGS_Z = (-16.5, -11.0, 11.0, 16.5)  # four springs per bolt, eight per cartridge
SPR_Y = 9.9                            # their axis (slot bottom 8.3)
SLOT_X1 = 16.6                         # each slot's outer end with the bolt at rest: spring 14.6 long at rest, 9.7 at full travel
GUS, GUS_Y0 = 4.0, 8.5                 # rib end gussets: 4 x 4 in plan, y 8.5..11.5 (the bolts' inner corners are cut to pass)
TONGUE_HZ, BLADE_TOP = 5.0, 6.4        # blade top 6.4 all along, under the bolts
m6.BLADE_Y1 = 6.4; PUSH_Y = 5.8; PAD_Y = (5.4, 6.2)
BOSS = dict(hz=5.0, y0=4.8)            # the push boss under each bolt's inner end (it passes a notch in the wall hole's floor): the blade's flat face pushes its crowned pad
ST = dict(hz=1.0, y0=9.4, t_hz=2.5, t_len=2.0, key_hx=0.8, key_hz=2.5, xs=9.0)   # stop bar: body 2 wide, 2.1 deep in the tops; T ends 5 x 2; key 1.6 x 5 in the rib; chamber shoulder at x 9.0
SHELF = dict(x1=8.6, y=(4.4, 6.65), z0=19.4)       # corner shelves under the bolts' inner ends (bolt underside 6.8)
bl.MAG_X, bl.MAG_Z = 13.0, 19.7        # magnet up in the bolt's underside; the head's hall on a post to y 5.6 under it
F_BLADE = 25.0
for _b in m6.BOUGHT:
    if _b["part"].startswith("latch spring"):
        _b["per_cartridge"] = 8; _b["spec"] += "; eight per cartridge (four in each bolt)"
        _b["check"] = "on a kitchen scale: 1.7..2.2 N at 14.6 long; fully closed it must be 9.0 long or less (E 0.127 N/mm, 28 coils)"
    if _b["part"].startswith("dowel pin"): _b["per_cartridge"] = 4; _b["spec"] += "; four per cartridge (two in each bolt), two per receiver"
    if _b["part"].startswith("spring guide pin"): _b["spec"] = "o 2.0 x 7, printed PCTG; eight per cartridge"
m6.BOUGHT.append(dict(part="stop bar", spec="printed PCTG, one per cartridge, dropped in after the bolts", per_cartridge=1, listing="printed", url="", check=""))
GUIDE_PIN = (2.0, 7.0)


def _spring_state(retract):
    """one spring, rib face to its slot's end; N is scaled so that v6's 2 x N is the force of the four on one bolt."""
    k = m6.spring_rate(); D = m6.SPR["od"] - m6.SPR["d"]; C = D / m6.SPR["d"]
    L = SLOT_X1 - bl.RIB_HX - retract
    F = k * (m6.SPR["free"] - L)
    tau = 8 * F * D / (math.pi * m6.SPR["d"] ** 3) * ((4 * C - 1) / (4 * C - 4) + 0.615 / C)
    return dict(length=round(L, 2), N=round(F * len(SPRINGS_Z) / 2, 3), each_N=round(F, 3), MPa=round(tau, 0), of_uts=round(tau / m6.SPR["uts"], 3))


m6.spring_state = _spring_state


def pads_of(g):
    """heel pads: (x0, x1) and the two z ranges. Scout: under the latch zone's cross walls, at the ends of its flat bottom (the
    collar clip takes the rest); Overseer: on the rim bands at the cartridge's ends, the lead pair under the XT60."""
    zt = g["Z_T"]
    if g["pack"] == "scout": return (12.0, 18.5), [(zt + bl.REC_Z + 0.3, zt + 25.6), (zt - 25.6, zt - bl.REC_Z - 0.3)]
    x0 = max(15.2, g["X_IN"] - 1.8)
    return (x0, x0 + 3.0), [(g["L_C"] - 9.0, g["L_C"] - 2.0), (2.0, 9.0)]


def set_pack(g):
    px, pz = pads_of(g); zt = g["Z_T"]
    m6.PAD_X = px
    m6.PAD_Z = (min(abs(pz[0][0] - zt), abs(pz[1][1] - zt)), min(abs(pz[0][1] - zt), abs(pz[1][0] - zt)))


def blade(g, face, side=1, dy=0.0, back=0.0):
    """one blade with its two hook nubs, inner face at |x| = face."""
    zt = g["Z_T"]
    shank = box(face, face + m6.BLADE_T, -20.0, BLADE_TOP, zt - m6.BLADE_HZ, zt + m6.BLADE_HZ) + \
        box(face, face + m6.BLADE_T, BLADE_TOP - 0.01, m6.BLADE_Y1, zt - TONGUE_HZ, zt + TONGUE_HZ)
    xb0 = bl.TAB_OUT - m6.R0
    under = lambda x: m6.foot_top(x - face + xb0) + back
    nubs = [hullp([[x, y, zt + zc + z] for z in (-m6.NUB_HZ, m6.NUB_HZ)
                   for (x, y) in ((face - m6.NUB_L, under(face - m6.NUB_L)), (face + 0.01, under(face + 0.01)), (face + 0.01, m6.NUB_Y1), (face - m6.NUB_L, m6.NUB_Y1))])
            for zc in (-NUB_ZC, NUB_ZC)]
    b = U(shank, *nubs).translate([0, dy, 0])
    return b if side > 0 else b.mirror([1, 0, 0])


m6.blade = blade


def slider(g, retract=0.0, side=1):
    """a bolt: a plain bar the section of its wall hole (it goes in from outside)."""
    zt, tip = g["Z_T"], g["X_TIP"]; Y0, Y1 = bl.BOLT_Y0, bl.BOLT_Y1; HZ = bl.BOLT_HZ
    s = box(bl.TAB_IN, tip, Y0, Y1, zt - HZ, zt + HZ)
    s = s - prism_xy([(tip - bl.TIP_CH, Y1 + 0.01), (tip + 0.01, Y1 + 0.01), (tip + 0.01, Y1 - bl.TIP_CH)], zt - HZ - 1, zt + HZ + 1)
    xp, yp = v5.pin_axis(g)
    s = s - cyl("z", xp, yp, m6.PIN_R, zt - HZ - 0.01, zt - HZ + m6.PIN_L + 0.5, 32)
    s = s + box(bl.TAB_IN, PLATE_OUT, BOSS["y0"], Y0 + 0.01, zt - BOSS["hz"], zt + BOSS["hz"])                          # the push boss
    s = s + hullp([[PLATE_OUT - 0.01, y, zt + z] for y in (BOSS["y0"], Y0) for z in (-BOSS["hz"], BOSS["hz"])] +         # and its crowned pad
                  [[bl.TAB_OUT, y, zt + z] for y in PAD_Y for z in (-2.0, 2.0)])
    for dz in SPRINGS_Z:                                             # a spring's slot along the bolt, open to the top (the pack over it)
        zc = zt + dz
        s = s - cyl("x", SPR_Y, zc, bl.SPR_R, bl.TAB_IN - 0.01, SLOT_X1, 32) - box(bl.TAB_IN - 0.01, SLOT_X1, SPR_Y, Y1 + 1, zc - bl.SPR_R, zc + bl.SPR_R)
    c = 0.25                                                         # the stop bar's channel and chamber in the top
    s = s - box(bl.TAB_IN - 0.01, ST["xs"] + 0.01, ST["y0"], Y1 + 1, zt - ST["hz"] - c, zt + ST["hz"] + c)
    s = s - box(ST["xs"], ST["xs"] + ST["t_len"] + bl.TRAVEL + 0.4, ST["y0"], Y1 + 1, zt - ST["t_hz"] - c, zt + ST["t_hz"] + c)
    for sz in (-1, 1):                                               # the inner top corners, for the rib's gussets
        zc, zw = zt + sz * (bl.REC_Z - GUS - 0.6), zt + sz * (HZ + 1.0)
        s = s - hullp([[x, y, z] for y in (GUS_Y0 - 0.3, Y1 + 1) for (x, z) in ((bl.TAB_IN - 0.01, zc), (bl.TAB_IN - 0.01, zw), (bl.TAB_IN + GUS + 0.6, zw))])
    s = s - cyl("y", bl.MAG_X, zt + bl.MAG_Z, bl.MAG_D / 2, Y0 - 0.01, Y0 + bl.MAG_DEPTH, 24)
    s = s.translate([-retract, 0, 0])
    return s if side > 0 else rot_y180(s, zt)


def stop_bar(g):
    """the one loose piece: a flat bar across the rib's top, a T end in each bolt's chamber. Printed flat."""
    zt = g["Z_T"]; x1 = ST["xs"] + ST["t_len"]
    return U(box(-x1, x1, ST["y0"] + 0.1, bl.BOLT_Y1, zt - ST["hz"], zt + ST["hz"]),
             *[box(*sorted((sx * ST["xs"], sx * x1)), ST["y0"] + 0.1, bl.BOLT_Y1, zt - ST["t_hz"], zt + ST["t_hz"]) for sx in (-1, 1)],
             box(-ST["key_hx"], ST["key_hx"], ST["y0"] + 0.1, bl.BOLT_Y1, zt - ST["key_hz"], zt + ST["key_hz"]))


def centre_rib(g):
    """the rib, part of the cartridge: web 4 x 11.5 fixed into a 6 mm cross wall at each end and gusseted there, a wedge foot
    along each side from wall to wall (the hooks' seats), a shelf in each corner under the bolts' inner ends, and in its top
    the stop bar's notch and key pocket. No spring notch."""
    zt = g["Z_T"]; RZ = bl.REC_Z; c = 0.15
    web = box(-bl.RIB_HX, bl.RIB_HX, 0.0, bl.RIB_Y1, zt - RZ - 0.01, zt + RZ + 0.01)
    feet = [hullp([[s * x, y, zt + z] for z in (-RZ, RZ)
                   for (x, y) in ((bl.RIB_HX - 0.5, 0.0), (m6.FOOT_X1, 0.0), (m6.FOOT_X1, m6.FOOT_Y), (bl.RIB_HX - 0.5, m6.foot_top(bl.RIB_HX - 0.5)))]) for s in (-1, 1)]
    gus = [hullp([[sx * x, y, zt + sg * z] for y in (GUS_Y0, bl.RIB_Y1) for (x, z) in ((bl.RIB_HX - 0.1, RZ + 0.01), (bl.RIB_HX + GUS, RZ + 0.01), (bl.RIB_HX - 0.1, RZ - GUS))])
           for sx in (-1, 1) for sg in (-1, 1)]
    shelves = [box(*sorted((sx * (bl.RIB_HX - 0.1), sx * SHELF["x1"])), SHELF["y"][0], SHELF["y"][1], *sorted((zt + sg * SHELF["z0"], zt + sg * (RZ + 0.01))))
               for sx in (-1, 1) for sg in (-1, 1)]
    walls = [box(-g["X_OUT"], g["X_OUT"], 0.0, bl.Y_FL1 - 0.01, *sorted((zt + sg * RZ, zt + sg * (RZ + bl.RIB_POST)))) for sg in (-1, 1)]
    r = U(web, *feet, *gus, *shelves, *walls)
    r = r - box(-bl.RIB_HX - 1, bl.RIB_HX + 1, ST["y0"], bl.RIB_Y1 + 1, zt - ST["hz"] - c, zt + ST["hz"] + c)            # the stop bar's notch
    return r - box(-ST["key_hx"] - c, ST["key_hx"] + c, ST["y0"], bl.RIB_Y1 + 1, zt - ST["key_hz"] - c, zt + ST["key_hz"] + c)   # and its key pocket


def cartridge(g, clip=None):
    """v5's cartridge with the wider, longer recess; the Scout's rim band cut back to the recess wall in the latch zone."""
    zt = g["Z_T"]; xb = g["X_IN"] - 2.0
    t = v5.cartridge(g, None)
    for sx in (-1, 1):                                 # a notch in each wall hole's floor for the bolt's push boss to pass
        t = t - box(*sorted((sx * (bl.REC_X - 1.01), sx * (g["X_OUT"] + 1.5))), BOSS["y0"] - 0.4, bl.BOLT_Y0, zt - BOSS["hz"] - 0.4, zt + BOSS["hz"] + 0.4)
    if xb < bl.REC_X:
        for s in (-1, 1):
            t = t - box(*sorted((s * (xb - 0.01), s * bl.REC_X)), -1.0, 4.01, zt - bl.REC_Z, zt + bl.REC_Z)
    return t if clip is None else t ^ clip


def heel(g):
    (x0, x1), zz = pads_of(g); zt = g["Z_T"]
    z0, z1 = min(a for a, b in zz), max(b for a, b in zz)
    deck = box(-x1, x1, -6.0, -m6.PAD_T, z0, z1) - box(-bl.REC_X - 1.0, bl.REC_X + 1.0, -7.0, 0.0, zt - m6.BLADE_HZ - 2.6, zt + m6.BLADE_HZ + 2.6)
    pads = [box(*sorted((sx * x0, sx * x1)), -m6.PAD_T, 0.0, a, b) for sx in (1, -1) for a, b in zz]
    for sx, sz in ((1, 1), (-1, -1)):                 # a hall post under each bolt's magnet (magnet at rest; the post stands beyond the blade, in the recess corner)
        deck = deck + box(*sorted((sx * (bl.MAG_X - 2.0), sx * (bl.MAG_X + 2.0))), -6.0, 5.6, *sorted((zt + sz * (bl.MAG_Z - 1.2), zt + sz * (bl.MAG_Z + 2.0))))
    return deck, pads


m6.cart_all = lambda B: U(B["cart"], B["stop"], B["female"], *B.get("cstrips", []))
m6.VIEW = m6.VIEW + [("stop_bar", m6.G_L, "#f2d14e", None, False, "structure", "PCTG")] + [(f"spring_{k}", m6.G_L, "#c9783f", None, False, "bought", None) for k in "cdefgh"] + [(f"spring_pin_{k}", m6.G_L, "#e3e7ec", None, False, "structure", None) for k in "abcdefgh"]
_view_sources = m6.view_sources


def view_sources(B):
    src = _view_sources(B); g = B["g"]
    src["stop_bar"] = B["stop"]
    for k, (sx, dz) in zip("abcdefgh", [(sx_, dz_) for sx_ in (1, -1) for dz_ in SPRINGS_Z]):   # the guide pin inside each spring
        src[f"spring_pin_{k}"] = cyl("x", SPR_Y, g["Z_T"] + dz, GUIDE_PIN[0] / 2, *sorted((sx * 2.5, sx * (2.5 + GUIDE_PIN[1]))), 20)
    return src


_view_meshes = m6.view_meshes


def view_meshes(B):
    import trimesh
    out = _view_meshes(B); g = B["g"]; M = m6.page_matrix(g["Z_T"])
    L = SLOT_X1 - bl.RIB_HX; turns = m6.SPR["n_active"] + 2; rm = (m6.SPR["od"] - m6.SPR["d"]) / 2; n = turns * 12
    for k, (sx, dz) in zip("abcdefgh", [(sx_, dz_) for sx_ in (1, -1) for dz_ in SPRINGS_Z]):   # a and b on the +x bolt, c and d on the -x one
        pts = np.array([[sx * (bl.RIB_HX + L * i / n), SPR_Y + rm * math.cos(2 * math.pi * turns * i / n), g["Z_T"] + dz + rm * math.sin(2 * math.pi * turns * i / n)] for i in range(n + 1)])
        sp = trimesh.util.concatenate([trimesh.creation.cylinder(radius=m6.SPR["d"] / 2, segment=[p, q], sections=6) for p, q in zip(pts[:-1], pts[1:])])
        sp.apply_transform(M); out[f"spring_{k}"] = sp
    return out


m6.view_meshes = view_meshes
m6.view_sources = view_sources
m6.slider, m6.cartridge, m6.heel, m6.centre_rib = slider, cartridge, heel, centre_rib
bl.slider, bl.cartridge, bl.centre_rib = slider, cartridge, centre_rib
v5.slider = slider


FOOT_PLAY = 0.3                        # each Scout foot's outer end to its end wall's foot
FOOT_PAD = dict(x=(29.2, 34.0), y1=4.0, L=6.0)      # foot pads on the funnel's side ledges (|y| 28.5..34.4): the same +-34.0 as the Overseer's feet
FOOT_ANCHOR = dict(x=(26.0, 32.0), y=(13.5, 19.0), z=((84.5, 91.5), (-5.5, 1.5)))   # in the receiver's four corner bosses (nose, tail)


def scout_feet(B):
    """four landing feet on the Scout receiver (10 Nov, Spark: the battery must fit the funnel). Its keel hangs in the funnel's
    hole and its clipped ends never reach the end walls, so nothing held it along x. Each foot is a pad on a side ledge, its outer
    end 0.3 from an end wall's foot, on a leg down from a corner boss; the Scout then sits at its drawn height, held along x and y."""
    g = B["g"]; fm = _funnel(); zt = g["Z_T"]
    xn = fm.SEAT[1] - fm.SETTINGS["scout"] - FOOT_PLAY; xt = fm.SEAT[0] + fm.SETTINGS["scout"] + FOOT_PLAY      # drone x of the pads' outer ends
    legs = []
    for sx in (1, -1):
        for zo, za in ((xn + zt, FOOT_ANCHOR["z"][0]), (xt + zt, FOOT_ANCHOR["z"][1])):
            zp = sorted((zo, zo - math.copysign(FOOT_PAD["L"], zo - zt)))
            pts = [[sx * x, y, z] for x in FOOT_PAD["x"] for y in (bl.Y_RB, FOOT_PAD["y1"]) for z in zp] + \
                  [[sx * x, y, z] for x in FOOT_ANCHOR["x"] for y in FOOT_ANCHOR["y"] for z in za]
            legs.append(hullp(pts))
    legs = U(*legs)
    for sk in B.get("rskids", []):                                   # 0.4 clear of the TPU skids
        for dx, dy in ((0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4), (0, 0)):
            legs = legs - sk.translate([dx, dy, 0])
    return legs


def build(pack="scout"):
    bl._CACHE.clear()
    B = m6.build(pack); set_pack(B["g"])
    B["stop"] = stop_bar(B["g"])
    if pack == "scout":
        B["feet"] = scout_feet(B); B["recv"] = B["recv"] + B["feet"]
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


PREV = dict(scout=dict(nub=0.64, foot=0.50, blade=0.59, rib=0.67, spring=2.23, hold3g=(2.7, 3.13)),      # the module before this pass (commit dacc1cd..9ea38ff),
            overseer=dict(nub=0.45, foot=0.35, blade=0.42, rib=0.47, spring=1.74, hold3g=(0.81, 0.65)))  # hold at 15 N per blade, 3 g


def strength(B):
    """hand models (E) of the stronger module at the real pull: four hooks (two per blade), feet wall to wall, the rib a T
    section with gusseted ends, loaded by two hook pairs 10.1 from its ends; and the bolts on their two pins in a 35 g crash."""
    g = B["g"]; lv = lever(g); P = lv["pull"]["both_nubs_N"]; A = m6.ALLOW; ta = math.tan(m6.ALPHA)
    xb0 = bl.TAB_OUT - m6.R0; idx = lambda mpa, cls="functional": round(mpa / A[cls], 3)
    wn = 2 * m6.NUB_HZ; tn = m6.NUB_Y1 - m6.foot_top(xb0); ln = xb0 - (xb0 - m6.NUB_L + m6.FOOT_X1) / 2
    s_n = (P / 4) * ln / (wn * tn ** 2 / 6)
    hf = m6.foot_top(bl.RIB_HX); lf = (xb0 - m6.NUB_L + m6.FOOT_X1) / 2 - bl.RIB_HX; wf = wn + 2 * (m6.FOOT_X1 - bl.RIB_HX)
    s_f = (P / 4) * lf / (wf * hf ** 2 / 6)
    wb = 2 * m6.BLADE_HZ; s_b = (P / 2) * (ln + m6.BLADE_T / 2) / (wb * m6.BLADE_T ** 2 / 6)
    # rib (part of the cartridge, fixed at both ends): web + a foot flange each side, a gusset each side at the top of its ends;
    # loaded by two hook pairs a from its ends. No notch except the stop bar's at mid span, where the moment is small.
    t, h = 2 * bl.RIB_HX, bl.RIB_Y1; fw = 2 * (m6.FOOT_X1 - bl.RIB_HX); fh = (m6.FOOT_Y + hf) / 2; gh = bl.RIB_Y1 - GUS_Y0
    def sec(gus, top=h):
        parts = [(t * top, top / 2, t * top ** 3 / 12), (fw * fh, fh / 2, fw * fh ** 3 / 12)] + ([(2 * GUS * gh, h - gh / 2, 2 * GUS * gh ** 3 / 12)] if gus else [])
        Ar = sum(a for a, c, i in parts); yc = sum(a * c for a, c, i in parts) / Ar
        return yc, sum(i + a * (c - yc) ** 2 for a, c, i in parts), top
    L = 2 * bl.REC_Z; a_ = bl.REC_Z - NUB_ZC
    def rib_y(Fy):
        Me = (Fy / 2) * a_ * (L - a_) / L; Mm = (Fy / 2) * a_ - Me
        yc, I, top = sec(True); s_end = Me * (top - yc) / I               # the ends: top in tension, gussets in
        yc, I, top = sec(False, ST["y0"]); s_mid = Mm * yc / I            # mid span, at the stop bar's notch: bottom in tension
        return max(s_end, s_mid)
    def rib_x(Fx):
        Iy = h * t ** 3 / 12 + fh * ((t + fw) ** 3 - t ** 3) / 12 + gh * ((t + 2 * GUS) ** 3 - t ** 3) / 12
        return (Fx * L / 8) * (t / 2 + GUS) / Iy
    ins = lv["insert"]["both_nubs_N"]
    xw = (P / 2) * (2 * bl.REC_X) / 4 / (bl.RIB_POST * bl.Y_FL1 ** 2 / 6)   # a cross wall as a beam between the side walls, a rib end at its middle (E)
    rib = {"XT60 pull 57 N + 1 g": dict(MPa=round(rib_y(P), 2), index=idx(rib_y(P))),
           "cross wall carrying a rib end": dict(MPa=round(xw, 2), index=idx(xw)),
           "swap across 43.5 N": dict(MPa=round(rib_x(ARM["across_N"]), 2), index=idx(rib_x(ARM["across_N"]))),
           "insert + full across (the lead-ins centring it)": dict(MPa=round(rib_y(ins) + rib_x(ARM["across_N"]), 2), index=idx(rib_y(ins) + rib_x(ARM["across_N"]))),
           "full pull + full across at once (a jam: misuse class)": dict(MPa=round(rib_y(P) + rib_x(ARM["across_N"]), 2), index=idx(rib_y(P) + rib_x(ARM["across_N"]), "misuse"))}
    # the bolts: a 35 g crash on the swap mass, half on each bolt; the pins' line on the receiver's PCTG ramp (Hertz)
    Fc = g["SWAP_M"] / 1000 * 35 * 9.81 / 2; Es = m6.PCTG.E / (1 - 0.38 ** 2)
    hz = lambda F, Lp: math.sqrt((F / Lp) * Es / (math.pi * m6.PIN_R))
    W1 = g["SWAP_M"] / 1000 * 9.81 / 2
    bolt = dict(crash_N_per_bolt=round(Fc, 0), pin_line_mm=m6.PIN_L, contact_MPa=dict(crash_35g=round(hz(Fc, m6.PIN_L), 0), flight_3g=round(hz(3 * W1, m6.PIN_L), 0),
                                                                                          one_pin_20mm_crash=round(hz(Fc, 20.0), 0)),
                pctg_strength_MPa=round(m6.PCTG.Xt, 0), bar_shear_MPa=round(Fc / (2 * bl.BOLT_HZ * bl.BOLT_T), 2),
                over_the_pin_bore_MPa=round(Fc / (m6.PIN_L * (bl.BOLT_T - 2 * m6.PIN_R + m6.PIN_DOWN)), 2),
                index=round(max(Fc / (2 * bl.BOLT_HZ * bl.BOLT_T), Fc / (m6.PIN_L * (bl.BOLT_T - 2 * m6.PIN_R + m6.PIN_DOWN))) / (0.58 * A["misuse"]), 3))
    out = dict(blade_pull=dict(N=round(P / 2, 1), MPa=round(s_b, 2), index=idx(s_b)), nub_pull=dict(N=round(P / 4, 1), MPa=round(s_n, 2), index=idx(s_n), root_mm=round(tn, 2)),
               foot_pull=dict(N=round(P / 4, 1), MPa=round(s_f, 2), index=idx(s_f)), rib=rib, bolt=bolt)
    out["worst_index"] = max([out[k]["index"] for k in ("blade_pull", "nub_pull", "foot_pull")] + [c["index"] for c in rib.values()] + [bolt["index"]])
    return dict(lever=lv, at_the_real_pull=out)


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
    ey = SPR_Y - y_push
    Lp, Ly = 10.0, 10.5
    Fs = 2 * m6.spring_state(m6.R0)["N"]
    f = 2 * mu * ey / Lp + 2 * mu * abs(e_z) / Ly
    return dict(push_below_springs_mm=round(ey, 1), push_off_centre_mm=round(e_z, 1), on_the_pack_underside_N=round(Fs * ey / Lp, 2),
                side_load_in_the_channel_N=round(Fs * abs(e_z) / Ly, 2), blade_N_to_unlatch=round(Fs / (1 - f), 1))


def spring_column():
    D = m6.SPR["od"] - m6.SPR["d"]; L0 = m6.SPR["free"]
    rest = SLOT_X1 - bl.RIB_HX
    return dict(free_over_mean_dia=round(L0 / D, 1), buckles_unguided_above=5.3, deflection_ratio_squeezed=round((L0 - (rest - m6.R0)) / L0, 2),
                unsupported_each_side_mm=round(bl.TAB_IN - bl.RIB_HX, 1), guide="its slot along the bolt, with a 2 x 7 printed pin inside (v6: none; open slots under the pack)",
                shortest_length_mm=round(rest - bl.TRAVEL, 1), solid_mm=round((m6.SPR["n_active"] + 2) * m6.SPR["d"], 1))


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
             takes="far heel pads + nubs sized for it", v6="not counted: nub, foot, blade over 1.0", ok=True, note="see strength"),
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
    for Fb in (20.0, 25.0):
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
                    v6_centred=slider_balance(g, 3.5, 0.0), v6_head_1mm_off=slider_balance(g, 3.5, 3.4),
                    note="v6: flat blade face on two tab legs 1.6 wide and a 0.4 strip over the arch (centroid y 3.5); 1 mm off along the pack the contact is 2.6 / 0.6 wide on the legs: 3.4 off centre; past 1.0 the blade meets a rest-stop lug"),
        spring_column=spring_column(),
        wedge_hold={f"{Fb:g} N": dict(left_N=round(Fb - Fs, 1), mu_needed=round(max(0.0, math.tan(m6.ALPHA - math.atan2(Fb - Fs, per))), 2)) for Fb in (20.0, 25.0)},
        parts=parts_check(g))
    return a



# ------------------------------------------------------------------ fit in the landing funnel (parts/box_funnel/make.py, seat frame)
def to_seat(B):
    """cartridge coords -> the funnel's seat frame (x nose = the lead end, y left, z up): the receiver's bottom at seat z -44.3
    and the receiver (with its skids) centred on the catch centre x, as the funnel's seat is drawn for each drone."""
    fm = _funnel()
    if B["g"]["pack"] == "scout":                         # the seat frame is the seated Scout's drone frame: the tab line at x 0
        return np.array([[0, 0, 1, -B["g"]["Z_T"]], [1, 0, 0, 0], [0, 1, 0, fm.SEAT_Z - bl.Y_RB]])
    R = m6.receiver_all(B); zc = [R.bounding_box()[2], R.bounding_box()[5]]
    return np.array([[0, 0, 1, fm.CX - sum(zc) / 2], [1, 0, 0, 0], [0, 1, 0, fm.SEAT_Z - bl.Y_RB]])


def _funnel():
    sp = importlib.util.spec_from_file_location("box_funnel_make", os.path.join(ROOT, "parts", "box_funnel", "make.py"))
    fm = importlib.util.module_from_spec(sp); sp.loader.exec_module(fm); return fm


def funnel_fit(B, step=4.0):
    g = B["g"]; pack = g["pack"]; setting = "scout" if pack == "scout" else "overseer"
    fm = _funnel(); F = fm.funnel(setting); T = to_seat(B)
    sys.path.insert(0, os.path.join(ROOT, "tools")); import seat_check
    R = m6.receiver_all(B).transform(T)
    Cs = (m6.cart_all(B) + m6.moving(g, 0, 1) + m6.moving(g, 0, -1) + bl.pack_body(g)).transform(T)
    Cr = (m6.cart_all(B) + m6.moving(g, m6.R0, 1) + m6.moving(g, m6.R0, -1) + bl.pack_body(g)).transform(T)
    deck, pads = heel(g)
    H = U(deck, *pads, *m6.blades(g, bl.TAB_OUT - m6.R0, back=0.02)).transform(T)
    # where it comes to rest: lowered from 2 mm up until more than 0.5 mm3 of it is in the funnel
    lo, hi = -12.0, 2.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if vol(R.translate([0, 0, mid]) ^ F) > 0.5: lo = mid
        else: hi = mid
    rest = round(hi, 2)
    touch = R.translate([0, 0, rest - 0.3]) ^ F; tb = touch.bounding_box()
    R, Cs, Cr, H = (m.translate([0, 0, rest]) for m in (R, Cs, Cr, H))
    up = lambda m, k=0.05: m.translate([0, 0, k])
    rb, cb = R.bounding_box(), Cr.bounding_box()
    x0, x1, y0, y1 = fm.SEAT; d = fm.SETTINGS[setting]; seat = (x0 + d, x1 - d, y0, y1); hole = fm.HOLE
    out = dict(setting=setting,
               receiver_box=[round(v, 2) for v in (rb[0], rb[3], rb[1], rb[4])], seat_opening=[round(v, 2) for v in seat],
               seat_gap_mm=dict(nose=round(seat[1] - rb[3], 2), tail=round(rb[0] - seat[0], 2), side=round(seat[3] - rb[4], 2)),
               cartridge_box=[round(v, 2) for v in (cb[0], cb[3], cb[1], cb[4])], hole=[round(v, 2) for v in hole],
               hole_gap_mm=dict(nose=round(hole[1] - cb[3], 2), tail=round(cb[0] - hole[0], 2), side=round(hole[3] - cb[4], 2)),
               seated_mm3=dict(receiver=round(vol(up(R) ^ F), 2), cartridge=round(vol(up(Cs) ^ F), 2)),
               rests_at_mm=rest, tab_line_at_seat_x=round(float(T[0, 3] + g["Z_T"]), 2), touches_at=dict(x=[round(tb[0], 1), round(tb[3], 1)], y_abs=[round(min(abs(tb[1]), abs(tb[4])), 1), round(max(abs(tb[1]), abs(tb[4])), 1)], z=[round(tb[2], 1), round(tb[5], 1)]),
               pressed_0p3_in_mm3=round(vol(touch), 1))
    worst = (0.0, None)
    for lab, ox, oy, h in seat_check.descent_states("scout_v4" if pack == "scout" else "overseer")[1:]:
        v = vol((R + Cs).translate([ox, oy, h]) ^ F)
        if v > worst[0]: worst = (v, lab)
    out["landing_mm3"] = dict(worst=round(worst[0], 2), at=worst[1], states=len(seat_check.descent_states("overseer")) - 1)
    wc = wh = 0.0; atc = None
    for k in np.arange(0.0, 160.01, step):
        v = vol(up(Cr).translate([0, 0, -k]) ^ F)
        if v > wc: wc, atc = v, float(k)
        wh = max(wh, vol(H.translate([0, 0, -k]) ^ F))
    out["swap_path_mm3"] = dict(cartridge=round(wc, 2), at_mm_down=atc, grabber_heel_and_blades=round(wh, 2))
    def play(dx, dy):                                        # how far the seated set can slide before the funnel stops it (to 6 mm)
        lo, hi = 0.0, 6.0
        if vol(up(R, 0.1).translate([hi * dx, hi * dy, 0]) ^ F) < 0.5: return None
        for _ in range(8):
            mid = (lo + hi) / 2
            if vol(up(R, 0.1).translate([mid * dx, mid * dy, 0]) ^ F) > 0.5: hi = mid
            else: lo = mid
        return round(lo, 2)
    out["free_play_mm"] = dict(to_nose=play(1, 0), to_tail=play(-1, 0), across=play(0, 1))
    out["ok"] = bool(max(out["seated_mm3"].values()) < 0.5 and worst[0] < 0.5 and wc < 0.5 and wh < 0.5)
    return out

# ------------------------------------------------------------------ save, page
OUT = {"scout": ("latch7", "latch7_"), "overseer": ("latch7_overseer", "latch7_ov_")}     # result folder, part folder prefix


def save(B):
    g = B["g"]; pre = OUT[g["pack"]][1]
    out = {pre + "receiver": B["recv"], pre + "carrier": B["carrier"], pre + "carrier_cap": B["cap"], pre + "cartridge": B["cart"],
           pre + "slider": slider(g, 0, 1), pre + "bolt_pin": m6.bolt_pin(g, 0, 1), pre + "stop_bar": B["stop"]}
    for k, m in zip("ab", sorted(B["strips"].decompose(), key=lambda m: m.bounding_box()[0])): out[f"{pre}receiver_strip_{k}"] = m
    paths = {}
    for nm, m in out.items():
        d = os.path.join(ROOT, "parts", nm); os.makedirs(d, exist_ok=True)
        t = bl.tm(m); t.export(os.path.join(d, "part.stl")); paths[nm] = t
    return paths


ARTIFACT = "https://claude.ai/artifact/Pqu6aZFdW9g7jtRyBFdraw"


ARTIFACT_OV = "https://claude.ai/artifact/2pPEEK9g13nZJLSdbmSdKr"


def write_assembly(res, pack="scout"):
    a = res["audit"]; lv = a["lever"]; st = a["strength"]; f = res["funnel"]; k = 1.0 if pack == "scout" else 1.7
    nm = OUT[pack][0]; ey = lambda *v: [round(x * k, 1) for x in v]
    asm = {"name": "latch v7" if pack == "scout" else "latch v7 overseer", "title": f"Battery Latch v7 ({pack.capitalize()})", "readme_view": "section",
           "artifact": ARTIFACT if pack == "scout" else (ARTIFACT_OV if ARTIFACT_OV != "TBD" else None), "machine": "box", "order": 90 if pack == "scout" else 91,
           "views": {"section": {"eye": ey(0, 95, 8), "target": [0, 0, 8]}, "wedge close": {"eye": [8, 34, 5], "target": [6, 0, 5]},
                     "bolt close": {"eye": [res["geom"]["RF"] + 10, 45, 6], "target": [res["geom"]["RF"] - 1, 0, 8]}, "springs, from above": {"eye": [0, 42, 38], "target": [0, 3, 9]},
                     "grabber side, from below": {"eye": ey(45, 70, -70), "target": [0, -4, 0]}, "lead end": {"eye": ey(120, -60, 70), "target": [8, round(-36 * k * 1.2), 34]},
                     "held, tipped": {"eye": ey(150, 150, -85), "target": [0, 0, -112 if pack == "scout" else -150]}},
           "note": f"The press-two-tabs latch, v7, on the {pack.capitalize()} pack ({res['swap_g']} g with its cartridge). The same latch module as the "
                   f"{'Overseer' if pack == 'scout' else 'Scout'}'s: same slider, springs, 3 x 20 dowels and grabber blades. Cut at the tab line. "
                   f"The XT60 is {lv['xt60_from_tab_line_mm']} mm from the pull, so the two nubs carry {lv['pull']['both_nubs_N']} N (worst index {st['worst_index']}). "
                   f"In the landing funnel: {'clear seated, landing and on the swap path' if f['ok'] else 'NOT clear, see the numbers'}; "
                   f"the cartridge passes the hole with {f['hole_gap_mm']['side']} mm a side. Numbers under the viewer.",
           "cg_ref": None, "timeline": f"tools/{nm}_motion.py:timeline", "appendix": f"systems/{nm}.appendix.html", "parts": res["page_parts"]}
    json.dump(asm, open(os.path.join(ROOT, "systems", f"{nm}.assembly.json"), "w"), indent=1)


def main(pack="scout"):
    nm = OUT[pack][0]; out_dir = os.path.join(ROOT, "parts", nm); os.makedirs(out_dir, exist_ok=True)
    B = build(pack); g = B["g"]
    res, rows = m6.checks(B)
    res["pack"] = pack
    res["hold"] = m6.hold_report(B)
    res["hold_strong"] = [m6.hold(B, Fb, m6.MU["pad_tpu"], n) for Fb in (20.0, 25.0) for n in (1.0, 2.0, 3.0)]
    res["hold"]["all_gravity_directions"] += [dict(h, pads="TPU-faced pads") for h in res["hold_strong"]]      # the motion's readouts use the 25 N row
    res["audit"] = audit(B)
    res["funnel"] = funnel_fit(B)
    res["build"] = build_check(B)
    if pack == "scout":
        W = (g["SWAP_M"] + 510.0) / 1000 * 9.81            # the Scout's weight without its pack, 510 g (E, ~730 g all-up)
        Ff = max((ARM["xt60_pull_N"] + W) / 4, 3 * W / 2)      # per foot: the XT60 pull-out on four, or a 3 g landing on two
        lever = 7.5; Z = 6.0 * 7.0 ** 2 / 6                    # pad centre to the anchor, the leg's 6 x 7 root
        res["scout_feet"] = dict(count=4, added_g=round(vol(B["feet"]) / 1000 * 1.23, 1), pad_mm=[4.8, FOOT_PAD["L"]], end_play_mm=FOOT_PLAY,
                                 per_foot_N=round(Ff, 1), root_MPa=round(Ff * lever / Z, 2), index=round(Ff * lever / Z / m6.ALLOW["functional"], 2))
    paths = save(B)
    res["mass_g"] = {n: round(abs(t.volume) / 1000 * (7.85 if "pin" in n else 1.21 if "strip" in n else 1.23), 2) for n, t in paths.items()}
    res["swap_g"] = round(g["SWAP_M"], 1); res["swap_cg"] = [round(float(v), 2) for v in g["SWAP_CG"]]
    res["insert_rows"] = rows
    res["geom"] = {k: round(float(v), 3) for k, v in g.items() if isinstance(v, (int, float))}
    res["v6"] = dict(TAB_IN=bl.TAB_IN, TAB_OUT=bl.TAB_OUT, TRAVEL=bl.TRAVEL, R0=m6.R0, NEED=m6.NEED, OVERLAP=bl.OVERLAP, ALPHA_deg=15.0, PUSH_Y=PUSH_Y,
                     FOOT=(m6.FOOT_X1, m6.FOOT_Y, m6.FOOT_HZ), NUB=(m6.NUB_L, m6.NUB_HZ, m6.NUB_Y1), BLADE=(m6.BLADE_T, m6.BLADE_HZ, m6.BLADE_Y1),
                     REC=(bl.REC_X, bl.REC_Z), MAG=(bl.MAG_X, bl.MAG_Z), BLADE_OPEN=m6.BLADE_OPEN, PAD_X=m6.PAD_X, PAD_Z=m6.PAD_Z, PAD_T=m6.PAD_T,
                     K_BLADE=m6.K_BLADE, MU=m6.MU, SPRINGS_Z=SPRINGS_Z, SPR_Y=SPR_Y, SLOT_X1=SLOT_X1)
    res["rest_stop_mm3_at_0.3_out"] = round(vol(slider(g, -0.3, 1) ^ B["stop"]), 3)      # > 0: the stop bar stops a bolt pushed 0.3 past rest
    res["stop_bar_mm3"] = dict(vs_cartridge=round(vol(B["stop"] ^ B["cart"]), 3), vs_bolts_at_rest=round(vol(B["stop"] ^ (slider(g, 0, 1) + slider(g, 0, -1))), 3),
                               vs_bolts_fully_in=round(vol(B["stop"] ^ (slider(g, bl.TRAVEL - 0.02, 1) + slider(g, bl.TRAVEL - 0.02, -1))), 3))
    res["bought"] = m6.BOUGHT
    m6.HERE = out_dir
    pp = m6.write_page(B, res)
    for p in pp:
        p["id"] = p["id"].replace("latch6", nm); p["version"] = "latch v7 (10 Nov)"; p["source"] = "parts/latch7/make.py"
    res["page_parts"] = pp
    if pack == "scout":
        ov = build("overseer")
        res["overseer"] = dict(swap_g=round(ov["g"]["SWAP_M"], 1), spring_over_inertia=m6.spring_report(ov)["preload_over_inertia"], **strength(ov),
                               hold=[m6.hold(ov, Fb, m6.MU["pad_tpu"], n) for Fb in (20.0, 25.0) for n in (1.0, 2.0, 3.0)])
        build(pack)
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    write_assembly(res, pack)
    print(pack, "funnel", json.dumps(res["funnel"]), "static", res["static_fit_mm3"], res["squeeze_mm3"], "stuck", res["insert_stuck"], "pull", res["gripped_pull_mm3"],
          "blades", res["grabber_interface"]["squeezing_mm3"], res["grabber_interface"]["rising_open_vs_cart_and_tabs_mm3"], "worst", res["audit"]["strength"]["worst_index"])


def appendix(res):
    a = res["audit"]; lv = a["lever"]; st = a["strength"]; v6r = a["v6_at_the_real_pull"]; tol = a["tolerance"]; sl = a["slider"]
    sp = res["spring"]; sc = a["spring_column"]; ov = res["overseer"]; ovl = ov["lever"]; ovs = ov["at_the_real_pull"]; mc = res.get("motion_collision", {})
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    rib = st["rib"]; orib = ovs["rib"]
    kp, kj, ki = "XT60 pull 57 N + 1 g", "full pull + full across at once (a jam: misuse class)", "insert + full across (the lead-ins centring it)"
    budget = "".join(td(e["step"], e["error"], e["takes"], e["v6"], e["note"]) for e in a["error_budget"])
    hold = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}", h["slip_x_mm"], h["pack_top_mm"])
                   for h in res["hold_strong"] if h.get("releases"))
    ohold = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}", h["slip_x_mm"]) for h in ov["hold"] if h.get("releases"))
    bought = "".join(td(b["part"], b["spec"], f'<a href="{b["url"]}">{b["listing"]}</a>' if b["url"] else b["listing"], b["check"]) for b in m6.BOUGHT)
    pr = a["parts"]; p15, p20 = pr["20 N blades"], pr["25 N blades"]; stp = pr["head stepper (NEMA 17 17HS19, 2 A, r 8 pinion)"]
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
<!--STRONG-->
<h2>What the run-through found in v6, and what v7 changes</h2>
<div class="t"><table><tr><th>test</th><th>v6</th><th>v7</th></tr>
{td("pull-out load on the grip (the XT60 is off to one end)", f"sized for 29.6 N per nub; the real load is {v6r['both_nubs_N'] / 2:.0f} N. Index: nub {v6r['nub']}, foot {v6r['foot']}, blade {v6r['blade']} (over 1.0 fails)", f"{lv['pull']['per_nub_N']} N per nub. Index: nub {st['nub_pull']['index']}, foot {st['foot_pull']['index']}, blade {st['blade_pull']['index']}, rib {rib[kp]['index']}")}
{td("height tolerance between the head and the cartridge before a bolt is not fully drawn", f"+-{tol['v6_height_tolerance_mm']} mm", f"+-{tol['height_tolerance_mm']} mm (15 deg wedge; at least {tol['shortest_nub_on_foot_mm']} mm of nub on the foot)")}
{td("blade room along the pack (the arm is +-1.8 mm there)", "+-1.0 mm (the rest-stop lugs)", f"+-{res['grabber_interface']['z_capture_mm']} mm; swept clean to {a['blade_entry']['clean_along_mm_at_0p9_across']} mm")}
{td("where the blade pushes the slider", f"{sl['v6_centred']['push_below_springs_mm']} mm under the springs, on two 1.6 mm strips: {sl['v6_centred']['blade_N_to_unlatch']} N to unlatch; 1 mm off along the pack the push is {sl['v6_head_1mm_off']['push_off_centre_mm']} mm off centre: {sl['v6_head_1mm_off']['blade_N_to_unlatch']} N", f"one crowned pad on the centre line, {sl['v7_centred']['push_below_springs_mm']} mm under the springs: {sl['v7_centred']['blade_N_to_unlatch']} N, {sl['v7_head_1p8_off']['blade_N_to_unlatch']} N with the head 1.8 off")}
{td("slider tipped up against the pack's underside during the squeeze", f"{sl['v6_centred']['on_the_pack_underside_N']} N", f"{sl['v7_centred']['on_the_pack_underside_N']} N")}
{td("spring column", f"30 long, 2.7 across (ratio {sc['free_over_mean_dia']}; unguided springs buckle above about {sc['buckles_unguided_above']}), in open slots under the pack", "the same spring with a printed 2 x 8 pin inside it: it cannot bow onto the pack or the rib")}
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
<p>Springs: four on each bolt, in slots along it, bearing on the rib: {sp['preload_pair_N']} N at rest, x{sp['preload_over_inertia']} the 35 g inertia of a bolt with its two pins ({sp['slider_pin_magnet_g']} g); x{sp['one_spring_left_over_inertia']} with two of the four lost. {sp['squeezed_pair_N']} N squeezed, wire at {100 * sp['at_rib_limit']['of_uts']:.0f} % of tensile at the limit, {sp['solid_margin_mm']} mm above solid (E: the listing gives no coil count).</p>
<h2>3. Holding the pack on the grabber</h2>
<div class="t"><table><tr><th>Scout, per blade</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>slip across, mm</th><th>pack top moves, mm</th></tr>{hold}</table></div>
<div class="t"><table><tr><th>Overseer ({ov['swap_g']} g), per blade</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>slip across, mm</th></tr>{ohold}</table></div>
<p>The wedge needs friction of {a['wedge_hold']['20 N']['mu_needed']} at 20 N per blade, {a['wedge_hold']['25 N']['mu_needed']} at 25 N, to keep the hardest pull from opening a blade (PCTG on PCTG is about 0.3, E). Across the pack the hold is friction and the two blade springs, as in v6.</p>
<!--FIT-->
<!--BUILD-->
<h2>4. The parts on hand</h2>
<p>Printed parts: PCTG, the project's rule since 8 Nov (about 10 to 20 times PETG's impact toughness at 90 % of its stiffness; the PETG nodes and arms broke in testing). Heel pads and strips: TPU 95A. One dowel size, 3 x 20, does the bolt pins and the guide poles; the spring guide pins are printed.</p>
<div class="t"><table><tr><th>part</th><th>needs</th><th>has</th></tr>
{td("gripper servo, cam opening both blades", f"{p15['cam_kgcm']} kg cm at 20 N blades, {p20['cam_kgcm']} at 25 N (scaled from the heads v7 concept, E)", f"DS3218 20 kg cm: {p15['DS3218 (on hand)']} / {p20['DS3218 (on hand)']}; MG995 9.2 kg cm: {p15['MG995 (ordered)']} / {p20['MG995 (ordered)']}")}
{td("head stepper, NEMA 17 at 2 A on the r 8 pinion", f"insert {stp['insert_N']:.0f} N, pull {stp['pull_N']:.1f} N", f"stall {stp['stall_N']:.0f} N: {stp['margin']} / {stp['pull_margin']}")}
{td("blade springs (head)", f"{tol['springs_on_a_squeezed_blade_N']} N of latch spring to beat, then the wedge", "15 N in the heads v7 concept; this latch asks for 25 N (four latch springs to beat, and the hold)")}
{td("XT60", "57 N to pull apart", pr["XT60"])}
</table></div>
<div class="t"><table><tr><th>bought part</th><th>spec</th><th>Amazon listing</th><th>check</th></tr>{bought}</table></div>
<h2>What the grabber head has to match</h2>
<div class="t"><table><tr><th>item</th><th>v7</th></tr>
{td("blade in the recess", f"{V['BLADE'][0]} thick x {2 * V['BLADE'][1]:g} wide, top {V['BLADE'][2]} above the pads; inner faces +-{V['BLADE_OPEN']} open, +-{V['TAB_OUT'] - V['R0']:.1f} tight (stroke {tol['blade_stroke_mm']['nominal']}, {tol['blade_stroke_mm']['longest']} with every tolerance)")}
{td("nub", f"two on each blade, {2 * NUB_ZC:g} apart: {V['NUB'][0]} long x {2 * V['NUB'][1]:g} wide, top {V['NUB'][2]}, underside 15 deg")}
{td("heel pads", f"x {V['PAD_X'][0]}..{V['PAD_X'][1]}, {V['PAD_Z'][0]:.1f}..{V['PAD_Z'][1]:.1f} mm either side of the tab line (Scout); the Overseer's at its cartridge ends")}
{td("hall sensors", f"on a post to 5.6 mm above the pads, under each bolt's magnet: x +-{V['MAG'][0]}, {V['MAG'][1]} mm along the pack from the tab line (one each side, opposite ends)")}
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




def strong_section(rs, ro):
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    ss, so = rs["audit"]["strength"], ro["audit"]["strength"]; ps, po = PREV["scout"], PREV["overseer"]
    ix = lambda st: f"{st['nub_pull']['index']} / {st['foot_pull']['index']} / {st['blade_pull']['index']} / {st['rib']['XT60 pull 57 N + 1 g']['index']}"
    h3 = lambda r, Fb: next(f"x{h['margin_pads_stay_down']} / x{h['margin_no_slip']}" for h in r["hold_strong"] if h["Fb"] == Fb and h["n_g"] == 3.0)
    bo = so["bolt"]["contact_MPa"]
    return f"""<h2>Made stronger for the larger pack (the same parts on both)</h2>
<p>The Scout has no width to give, both packs have length, so the latch and the fingers grew along the pack.</p>
<div class="t"><table><tr><th></th><th>before</th><th>now</th></tr>
{td("bolt", "22 mm wide on one 3 x 20 steel pin, with a tab plate that could not be put in", "a plain bar 42 mm wide on two 3 x 20 pins end to end; slides in from outside")}
{td("pin on the receiver's ramp in a 35 g crash, Overseer (PCTG strength " + str(so['bolt']['pctg_strength_MPa']) + " MPa)", f"{bo['one_pin_20mm_crash']:.0f} MPa", f"{bo['crash_35g']:.0f} MPa ({bo['flight_3g']:.0f} MPa at 3 g in flight)")}
{td("finger", "blade 16 mm wide, one hook 13 wide", "blade 34 mm wide, two hooks 9 wide, 25 mm apart")}
{td("rib that the hooks pull on", "4 x 11.5, feet 18 long, two spring notches", "feet wall to wall, ends gusseted, no spring notch")}
{td("springs per cartridge / over the 35 g inertia of a bolt: Scout, Overseer", f"2 / x{ps['spring']}, x{po['spring']}", f"8 / x{rs['spring']['preload_over_inertia']}, x{ro['spring']['preload_over_inertia']}")}
{td("index nub / foot / blade / rib at the XT60 pull-out: Scout", f"{ps['nub']} / {ps['foot']} / {ps['blade']} / {ps['rib']}", ix(ss))}
{td("the same: Overseer", f"{po['nub']} / {po['foot']} / {po['blade']} / {po['rib']}", ix(so))}
{td("worst index anywhere: Scout, Overseer", "0.72, 0.60", f"{ss['worst_index']}, {so['worst_index']}")}
{td("blade force asked of the head", "15 N", "25 N (four latch springs on each bolt to beat; DS3218 " + rs['audit']['parts']['25 N blades']['DS3218 (on hand)'] + ", MG995 " + rs['audit']['parts']['25 N blades']['MG995 (ordered)'] + ")")}
{td("hold at 3 g, pads stay down / no slip: Scout, Overseer", f"x{ps['hold3g'][0]} / x{ps['hold3g'][1]}, x{po['hold3g'][0]} / x{po['hold3g'][1]} (at 15 N)", f"{h3(rs, 25.0)}, {h3(ro, 25.0)} (at 25 N)")}
{td("bought parts per cartridge", "2 springs, 2 dowels, 2 magnets", "8 springs, 4 dowels, 2 magnets (the same spring and dowel), and one printed stop bar")}
{td("slider with its pins / cartridge, Scout", "3.4 g / 39.5 g", f"{rs['spring']['slider_pin_magnet_g']} g / {rs['mass_g']['latch7_cartridge']} g")}
</table></div>
<p>Below x1 on "pads stay down" the pack does not come off: the hooks hold it positively and lock on their wedges; it only stops being pressed tight. Hand models (E); the receiver's wider bolt pockets have not been through FE.</p>
"""


def build_section(rs, ro):
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    bs, bo = rs["build"], ro["build"]
    pr = "".join(td(k, f"{v['size_mm'][0]:g} x {v['size_mm'][1]:g} x {v['size_mm'][2]:g}", ("yes, on the diagonal" if v.get("on_the_diagonal") else "yes") if v["fits_bed"] else "NO", f"{v['bed_mm2']:.0f}", f"{v['support_mm2']:.0f}",
                    f"{bo['printed'][k]['size_mm'][0]:g} x {bo['printed'][k]['size_mm'][1]:g} x {bo['printed'][k]['size_mm'][2]:g}", (("yes, on the diagonal, %.1f to spare" % bo['printed'][k]["diag_spare_mm"]) if bo['printed'][k].get("on_the_diagonal") else "yes") if bo['printed'][k]["fits_bed"] else "NO", f"{bo['printed'][k]['support_mm2']:.0f}")
                 for k, v in bs["printed"].items())
    st = "".join(td(a["step"], a["how"], "ok" if a["ok"] else "BLOCKED", ("%.2f" % a["mm3"]) if a["mm3"] is not None else "", ("%.2f" % b["mm3"]) if b["mm3"] is not None else "")
                 for a, b in zip(bs["install"], bo["install"]))
    return f"""<h2>Can it be built and put together</h2>
<p class="open">The check found that the bolt could not be put into the cartridge: with its tab plate it had to move 6.5 mm inward to leave its wall hole and the rib stopped it at 4.9, from above or below. That was true of every version since the rib was added. Each bolt is now a plain bar that slides in from outside; a small printed stop bar, dropped in across the rib, holds both at rest.</p>
<div class="t"><table><tr><th>assembly step</th><th>how it was checked</th><th></th><th>Scout overlap, mm&sup3;</th><th>Overseer overlap, mm&sup3;</th></tr>{st}</table></div>
<p>{bs['serviceable'].capitalize()}.</p>
<div class="t"><table><tr><th>printed part</th><th>Scout size, mm</th><th>fits the 180 bed</th><th>on the bed, mm&sup2;</th><th>needs support, mm&sup2;</th><th>Overseer size, mm</th><th>fits</th><th>needs support, mm&sup2;</th></tr>{pr}</table></div>
<p>Each part in the flat-down pose with the least overhang past 45 degrees. Support is allowed (tree supports); every part passes the solid check (no wall under 2 mm, no neck). Not checked here: the receiver's inner assembly (plug carrier, cap, XT60, guide poles) and the pack's plug into its pocket, which are unchanged from v5; the grabber head, which is not drawn.</p>
"""

def fit_section(rs, ro):
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    fs, fo = rs["funnel"], ro["funnel"]; sf = rs["scout_feet"]
    pl = lambda f: " / ".join("over 6" if f["free_play_mm"][k] is None else str(f["free_play_mm"][k]) for k in ("to_nose", "to_tail", "across"))
    ok = lambda f: "clear" if f["ok"] else "NOT clear"
    return f"""<h2>Fit in the landing funnel (both packs)</h2>
<div class="t"><table><tr><th></th><th>Scout set</th><th>Overseer set</th></tr>
{td("end walls", "in 39.2 mm", "out (the 184 seat)")}
{td("comes to rest, from its drawn height", f"{fs['rests_at_mm'] + 0.0:.1f} mm, on its four feet on the side ledges ({fs['pressed_0p3_in_mm3'] / 0.3:.0f} mm&sup2;)", f"{fo['rests_at_mm']} mm, on the side ledges ({fo['pressed_0p3_in_mm3'] / 0.3:.0f} mm&sup2;)")}
{td("free play once seated: to the nose / to the tail / across, mm", pl(fs), pl(fo))}
{td("cartridge through the hole: nose / tail / each side, mm", f"{fs['hole_gap_mm']['nose']} / {fs['hole_gap_mm']['tail']} / {fs['hole_gap_mm']['side']}", f"{fo['hole_gap_mm']['nose']} / {fo['hole_gap_mm']['tail']} / {fo['hole_gap_mm']['side']}")}
{td("overlap seated / worst of 65 landing states / cartridge going down 160 mm / grabber heel and blades, mm&sup3;", f"{max(fs['seated_mm3'].values())} / {fs['landing_mm3']['worst']} / {fs['swap_path_mm3']['cartridge']} / {fs['swap_path_mm3']['grabber_heel_and_blades']}", f"{max(fo['seated_mm3'].values())} / {fo['landing_mm3']['worst']} / {fo['swap_path_mm3']['cartridge']} / {fo['swap_path_mm3']['grabber_heel_and_blades']}")}
{td("tab line along the arm's axis (seat x)", f"{fs['tab_line_at_seat_x']} mm", f"{fo['tab_line_at_seat_x']} mm")}
{td("verdict", ok(fs), ok(fo))}
</table></div>
<p>The funnel was drawn round estimated outlines. Against the real latch sets its hole (170 x 50) was too small for the Overseer cartridge (174 x 55) and its seat (66.8 wide) too narrow for the Overseer receiver (68.0). Now: hole 177 x 57, seat 68.8 wide. The whole Scout v4 and Overseer still clear it: 0 crossings seated and in all 65 landing states each (tools/seat_check.py).</p>
<p>The Scout's receiver is a 51 mm wide keel with ends clipped at 45 degrees: it hung in the hole and nothing held it along its length (over 6 mm of play each way). It now has four landing feet, one on a leg down from each corner boss: pads {sf['pad_mm'][0]} x {sf['pad_mm'][1]} mm on the funnel's side ledges, their outer ends {sf['end_play_mm']} mm from the end walls' feet, the same 68 mm across as the Overseer's. {sf['added_g']} g in all, part of the receiver print. At {sf['per_foot_N']} N a foot (the XT60 pull-out on four feet, or a 3 g landing on two) a leg's root is at index {sf['index']}. They clear the Scout v4's own parts, its TPU skids, the cartridge's path and the grabber. Both tab lines sit at the same place along the arm's axis ({abs(fo['tab_line_at_seat_x'] - fs['tab_line_at_seat_x']):.1f} mm apart).</p>
"""


def appendix_overseer(ro, rs):
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    st = ro["audit"]["strength"]; lv = ro["audit"]["lever"]; sp = ro["spring"]
    hold = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}", h["slip_x_mm"]) for h in rs["overseer"]["hold"] if h.get("releases"))
    bought = "".join(td(b["part"], b["spec"], f'<a href="{b["url"]}">{b["listing"]}</a>' if b["url"] else b["listing"], b["check"]) for b in ro["bought"])
    css = appendix(rs).split("</style>")[0].replace("#latch7", "#latch7ov") + "</style>"
    return css + f"""
<section id="latch7ov"><div class="wrap">
<h2>The same latch on the Overseer pack</h2>
<p>One latch module for both packs: the same slider, springs, 3 x 20 dowels, magnets and grabber blades. Only the cartridge and the receiver are sized to the pack ({ro['swap_g']} g with its cartridge). The audit behind the design is on the Scout page: <a href="{ARTIFACT}">Battery Latch v7 (Scout)</a>.</p>
<div class="t"><table><tr><th>pull-out, 57 N at the plug</th><th></th></tr>
{td("XT60 from the tab line / far heel pads", f"{lv['xt60_from_tab_line_mm']} / {lv['far_pads_at_mm']} mm")}
{td("both nubs / far pads", f"{lv['pull']['both_nubs_N']} / {lv['pull']['far_pads_N']} N")}
{td("index: nub / foot / blade / worst", f"{st['nub_pull']['index']} / {st['foot_pull']['index']} / {st['blade_pull']['index']} / {st['worst_index']}")}
{td("springs at rest over the 35 g inertia of a slider", f"x{sp['preload_over_inertia']} (x{sp['one_spring_left_over_inertia']} with one lost)")}
</table></div>
{fit_section(rs, ro)}
<h2>Holding it on the grabber</h2>
<div class="t"><table><tr><th>per blade</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>slip across, mm</th></tr>{hold}</table></div>
<h2>Bought parts</h2>
<div class="t"><table><tr><th>part</th><th>spec</th><th>Amazon listing</th><th>check</th></tr>{bought}</table></div>
<h2>Checks</h2>
<p>Static fit, squeeze, free insertion, held pull, blades and heel: 0 mm&sup3;. Motion, every frame: {ro.get('motion_collision', {}).get('frames')} frames, worst {ro.get('motion_collision', {}).get('max_mm3')} mm&sup3;. Solid check: receiver, carrier, cap, cartridge, slider, strips PASS.</p>
</div></section>
"""



# ------------------------------------------------------------------ can it be built and put together (12 Nov, Spark)
BED = 180.0                            # Bambu A1 mini


def printability(m):
    """the best of the six flat-down poses on the A1 mini: least area that overhangs past 45 deg (needs support), then most bed contact."""
    t = bl.tm(m); n = t.face_normals; ar = t.area_faces; c = t.triangles_center
    best = None
    for ax in range(3):
        for sg in (1, -1):
            down = np.zeros(3); down[ax] = sg                     # this direction of the part goes onto the bed
            h = c @ down; hmax = float((t.vertices @ down).max())
            dn = n @ down
            on_bed = (dn > 0.999) & (h > hmax - 0.05)
            over = (dn > math.cos(math.radians(45.0))) & ~on_bed
            ext = t.extents; hgt = float(ext[ax]); a_, b_ = sorted((float(ext[(ax + 1) % 3]), float(ext[(ax + 2) % 3])), reverse=True)
            diag = a_ > BED and (a_ + b_) / math.sqrt(2) <= BED               # longer than the bed's side but fits turned 45 deg on it
            fits = bool(hgt <= BED and (a_ <= BED or diag))
            row = dict(down="%s%s" % ("+" if sg > 0 else "-", "xyz"[ax]), bed_mm2=round(float(ar[on_bed].sum()), 0), support_mm2=round(float(ar[over].sum()), 0),
                       size_mm=[round(float(v), 1) for v in ext], fits_bed=fits, on_the_diagonal=bool(diag))
            row["diag_spare_mm"] = round(BED - (a_ + b_) / math.sqrt(2), 1) if diag else None
            if best is None or (not row["fits_bed"], row["support_mm2"], -row["bed_mm2"]) < (not best["fits_bed"], best["support_mm2"], -best["bed_mm2"]): best = row
    return best


def build_check(B):
    g = B["g"]; pre = OUT[g["pack"]][1]
    printed = {"receiver": B["recv"], "carrier": B["carrier"], "carrier cap": B["cap"], "cartridge": B["cart"], "bolt": slider(g, 0, 1), "stop bar": B["stop"]}
    for k, mm in zip("ab", sorted(B["strips"].decompose(), key=lambda m: m.bounding_box()[0])): printed[f"receiver strip {k} (TPU)"] = mm
    out = dict(printed={k: printability(m) for k, m in printed.items()})
    cart = B["cart"] + B["female"]; pack = bl.pack_body(g)
    bolts = m6.moving(g, 0, 1) + m6.moving(g, 0, -1)
    sweep = lambda body, d, fixed, n=60, step=0.5: round(max(vol(body.translate([d[0] * k * step, d[1] * k * step, d[2] * k * step]) ^ fixed) for k in range(n + 1)), 3)
    steps = []
    steps.append(dict(step="1 two dowels into each bolt's bore, a magnet into its underside", how="by hand, before the bolt goes in; the bore opens on the bolt's side", mm3=None, ok=True))
    v = sweep(m6.moving(g, 0, 1), (1, 0, 0), cart + m6.moving(g, 0, -1))
    v2 = sweep(m6.moving(g, 0, 1), (1, 0, 0), cart + m6.moving(g, 0, -1) + pack)
    steps.append(dict(step="2 each bolt in through its wall hole, from outside", how="straight in, 30 mm swept; also with the pack already in", mm3=max(v, v2), ok=bool(max(v, v2) < 0.05)))
    v = sweep(B["stop"], (0, 1, 0), B["cart"] + slider(g, 0, 1) + slider(g, 0, -1), n=40)
    steps.append(dict(step="3 stop bar down across the rib, into both bolts", how="straight down from above, 20 mm swept, bolts held at rest", mm3=v, ok=bool(v < 0.05)))
    Lr = SLOT_X1 - bl.RIB_HX
    steps.append(dict(step="4 eight springs, each with its guide pin, into the bolts' slots", how="squeezed to %.1f mm and dropped in from above (slot %.1f wide, spring %.1f)" % (Lr, 2 * bl.SPR_R, m6.SPR["od"]),
                      mm3=None, ok=bool(2 * bl.SPR_R > m6.SPR["od"] and Lr > (m6.SPR["n_active"] + 2) * m6.SPR["d"])))
    v = sweep(pack, (0, 1, 0), B["cart"] + B["stop"] + bolts, n=40)
    steps.append(dict(step="5 pack down into the cartridge (over the bolts, springs and stop bar)", how="straight down, 20 mm swept (the 0.6 keeper lips snap over it, not counted)", mm3=v, ok=True, note="lips overlap %.1f mm3 by design" % v))
    R = m6.receiver_all(B)
    v = sweep(m6.cart_all(B) + m6.moving(g, bl.OVERLAP + 0.3, 1) + m6.moving(g, bl.OVERLAP + 0.3, -1) + pack, (0, -1, 0), R, n=80)
    steps.append(dict(step="6 cartridge up into the receiver", how="bolts held in, 40 mm swept; pushed in alone the lead-ins cam the bolts (no stuck step)", mm3=v, ok=bool(v < 0.05)))
    out["install"] = steps
    out["serviceable"] = "a bolt comes out the way it went in once the pack, springs and stop bar are lifted out; nothing is glued"
    out["ok"] = bool(all(s_["ok"] for s_ in steps) and all(p_["fits_bed"] for p_ in out["printed"].values()))
    return out

BOTH_X = {"scout": 45.0, "overseer": -45.0}       # page x (the section view looks from +y, so +x is on the left) of each set on the one page (tools/latch7_both_motion.py OFF)


def write_both():
    """one page with both packs' latch sets side by side (Spark 10 Nov: both packs in one artifact), at the latch's artifact;
    the two single-pack assemblies are archived (their results and motions are this page's sources)."""
    parts = []; res = {}
    for pk in ("scout", "overseer"):
        nm = OUT[pk][0]; res[pk] = json.load(open(os.path.join(ROOT, "parts", nm, "result.json")))
        T = np.eye(4); T[0, 3] = BOTH_X[pk]
        for p in res[pk]["page_parts"]:
            parts.append(dict(p, id=p["id"] + "_both", name=f"{pk}: {p['name']}", group=f"{pk.capitalize()}: {p['group']}", pose=T.tolist()))
        f = os.path.join(ROOT, "systems", f"{nm}.assembly.json"); a = json.load(open(f)); a["artifact"] = None; a["archived"] = True
        json.dump(a, open(f, "w"), indent=1)
    rs, ro = res["scout"], res["overseer"]
    asm = {"name": "latch v7 both", "title": "Battery Latch v7 (Scout and Overseer)", "readme_view": "section", "artifact": ARTIFACT, "machine": "box", "order": 90,
           "views": {"section": {"eye": [0, 215, 12], "target": [0, 0, 4]}, "below": {"eye": [95, 170, -150], "target": [0, -4, -6]},
                     "scout": {"eye": [BOTH_X["scout"], 95, 8], "target": [BOTH_X["scout"], 0, 8]},
                     "overseer": {"eye": [BOTH_X["overseer"], 120, 8], "target": [BOTH_X["overseer"], 0, 8]},
                     "wedge": {"eye": [BOTH_X["scout"] + 8, 34, 5], "target": [BOTH_X["scout"] + 6, 0, 5]},
                     "tipped": {"eye": [280, 280, -150], "target": [0, 0, -135]}},
           "note": "The press-two-tabs latch, v7, on both packs, side by side: the Scout's on the left, the Overseer's on the right. One latch module: the same slider, "
                   "springs, 3 x 20 dowels, magnets and grabber blades; only the cartridge and the receiver are sized to the pack. Each is cut at its tab line. "
                   "Made stronger on 10 Nov: bolts 42 wide on two steel pins each, blades 34 wide with two hooks each, four springs. "
                   f"Scout {rs['swap_g']} g with its cartridge, Overseer {ro['swap_g']} g. Both fit their landing funnel: clear seated, in 65 landing states and on the swap path; "
                   f"the Scout receiver stands on four landing feet. Worst strength index {rs['audit']['strength']['worst_index']} (Scout), {ro['audit']['strength']['worst_index']} (Overseer). Numbers under the viewer.",
           "cg_ref": None, "timeline": "tools/latch7_both_motion.py:timeline", "appendix": "systems/latch7.appendix.html", "parts": parts}
    json.dump(asm, open(os.path.join(ROOT, "systems", "latch7_both.assembly.json"), "w"), indent=1)
    print("latch7_both:", len(parts), "parts")

if __name__ == "__main__":
    pk = "overseer" if "--overseer" in sys.argv else "scout"
    if "--fit" in sys.argv:
        for q in ("scout", "overseer"):
            print(q, json.dumps(funnel_fit(build(q)), indent=1))
    elif "--both" in sys.argv:
        write_both()
    elif "--buildcheck" in sys.argv:
        for q in ("scout", "overseer"):
            f = os.path.join(ROOT, "parts", OUT[q][0], "result.json"); res = json.load(open(f)); res["build"] = build_check(build(q)); json.dump(res, open(f, "w"), indent=1)
            print(q, json.dumps(res["build"]))
    elif "--refit" in sys.argv:                            # the funnel changed, the latch did not: redo the fit and the page text only
        for q in ("scout", "overseer"):
            f = os.path.join(ROOT, "parts", OUT[q][0], "result.json"); res = json.load(open(f))
            res["funnel"] = funnel_fit(build(q)); json.dump(res, open(f, "w"), indent=1); write_assembly(res, q)
            print(q, {k: res["funnel"][k] for k in ("rests_at_mm", "tab_line_at_seat_x", "free_play_mm", "hole_gap_mm", "seated_mm3", "landing_mm3", "swap_path_mm3", "ok")})
    elif "--appendix" in sys.argv:
        rs = json.load(open(os.path.join(ROOT, "parts", "latch7", "result.json"))); ro = json.load(open(os.path.join(ROOT, "parts", "latch7_overseer", "result.json")))
        old = open(os.path.join(ROOT, "systems", "latch6.appendix.html")).read()      # v6's page section, kept under v7's, folded
        keep = ('<details style="max-width:1040px;margin:0 auto;padding:0 16px 40px;font:14px/1.5 var(--sans);color:var(--fg)"><summary>Latch v6\'s own numbers '
                '(superseded: its pull-out loads left out the XT60\'s lever, see the first table above)</summary>' + old + '</details>\n')
        open(os.path.join(ROOT, "systems", "latch7.appendix.html"), "w").write(appendix(rs).replace("<!--FIT-->", fit_section(rs, ro)).replace("<!--STRONG-->", strong_section(rs, ro)).replace("<!--BUILD-->", build_section(rs, ro)) + keep)
        open(os.path.join(ROOT, "systems", "latch7_overseer.appendix.html"), "w").write(appendix_overseer(ro, rs))
    else:
        main(pk)
