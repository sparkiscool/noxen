# Vendored unchanged from REPO1 (jy202108g-bot/repo1 @ 71dd86b, nox/box/battery_latch.py, 27 Oct 2026): latch v4 /
# receiver v2-v3 / cartridge v6, the press-two-tabs latch. parts/latch5/make.py builds latch v5 on top of it; only build()
# and the geometry functions are used here (main() points at REPO1 paths and is never run from noxen).
"""Battery latch v1 (24 Oct 2026, Spark: "design the new latch system for all batteries so the grabber can follow it").

One latch + cartridge standard for every pack: Scout 4S 1500 (70 x 32 x 33, 165 g P) and Overseer 4S 6000 (148 x 48 x 45 P,
650 g E). Concept (nox/box/battery-latch.md, scored): squeeze latch v5 = the v2 squeeze latch kept (two sliders and one
pen spring in the cartridge, passive pockets in the receiver), made lower, sized for the Overseer and the XT60 pull, and
given one grip interface on both cartridges. The grabber follows it: one finger squeeze grips the tabs and pulls both
bolts in (unlatched, held); opening the fingers lets the spring throw the bolts into the pockets (latched). The insert
stroke alone also latches: the receiver feet's 30 deg lead-ins cam a free bolt in and it snaps out at the seat.

Cartridge coords c (as Scout bottom layout v1, `nox/scout/v3/bottom.md`): x across the pack (= -drone y), y up the install
axis (0 = flat bottom = ring r 100 in a bay), z along the pack (0 = closed end). Tab line z_T = the swap assembly's CG
(pack + cartridge + sliders + plug), so the gripper holds the pack at its CG and the drone puts it on the thrust axis.
Every receiver number is in seated cartridge coords. The Scout drone frame stand-in: d = (z - z_T, x, y - 44.5): the XT60
(lead) end at the arm side of the wheel (-Y) = the drone nose.

python3 nox/box/battery_latch.py            -> parts, sweeps, hand calc, system manifest, runs/battery_latch/v1/
from battery_latch import latch_assembly; latch_assembly("scout", "latched") -> {name: manifold3d.Manifold} (c coords)
"""
import json, math, os, sys
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "nox")]
OUT = os.path.join(ROOT, "runs", "battery_latch", "v4")         # v4: + tail skids; v3: cartridge v6 + skids; v2: receiver v2


# ======================================================================== helpers (scout_bottom.py style)
def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl(axis, a, b, r, t0, t1, n=40):
    c = mf.Manifold.cylinder(t1 - t0, r, r, n)
    if axis == "z": return c.translate([a, b, t0])
    if axis == "x": return c.rotate([0, 90, 0]).translate([t0, a, b])
    return c.rotate([-90, 0, 0]).translate([a, t0, b])
def hullp(pts): return mf.Manifold.hull_points(np.asarray(pts, np.float64))
def prism_xy(poly, z0, z1): return hullp([[x, y, z] for (x, y) in poly for z in (z0, z1)])
def prism_zy(poly, x0, x1): return hullp([[x, y, z] for (z, y) in poly for x in (x0, x1)])
def U(*ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else (ms[0] if ms else mf.Manifold())
def tm(m, T=None):
    me = m.to_mesh(); t = trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=True)
    if T is not None: t.apply_transform(T)
    return t
def vol(m): return 0.0 if m.is_empty() else m.volume()
def ov(a, b): return round(vol(a ^ b), 3)
def rot_y180(m, zc): return m.translate([0, 0, -zc]).rotate([0, 180, 0]).translate([0, 0, zc])
def mirx(m): return m.mirror([1, 0, 0])
RHO = {"PETG": 1.27, "TPU": 1.21}


# ======================================================================== packs (P / E)
PACKS = {
    # width (x) / height (y) / length (z), mass with its own leads and plug; plug + leads share of that mass (E)
    "scout":    dict(W=33.0, H=32.0, L=70.0, m=165.0, m_tag="P", m_plug=10.0),      # dimensions A (P); 32 tall as v4
    "overseer": dict(W=48.0, H=45.0, L=148.0, m=650.0, m_tag="E", m_plug=15.0),     # dimensions O (P size), L mass 650 E;
}                                                                                    # 45 tall, 48 across (DEFAULT: lower)

# ======================================================================== the common latch module (C; same on every cartridge)
WALL = 2.0
Y_FL1 = 12.0                      # pack bottom (v4 14.4): set by the slider top 11.5 + 0.5
REC_X = 15.5                      # finger recess half width: blades open 11..14 + 1.5 (head offset +-1.0)
REC_Z = 11.5                      # recess half length about z_T (tab +-11 + 0.5)
TAB_IN, TAB_OUT = 5.0, 10.0       # tab faces at rest (slider v6; v5 4.0); squeezed 2.0 / 7.0 on the centre rib (v5: 0 / 6)
TAB_HZ = 11.0                     # tab z +-11 (22 wide: the 13 wide blades + the magnet ear)
RIB_HX, RIB_Y1, RIB_NOTCH = 2.0, 11.5, 2.5   # cartridge v6 centre rib (heads v7 ask, 24 Oct, DEFAULT): x +-2.0 (4.0; 3.0 necked),
                                  # y 0..11.5 along the recess, notched to y 7.9 over z_T +- 2.5 for the pen spring (y 8.0..
                                  # 11.2). Each tab stops on it, so the squeezed tabs clamp the cartridge to the head and both
                                  # bolts retract 3.0 whatever the head offset (need 2.6). It carries the head's x load (the
                                  # 43.5 N across, a 3 g carry), so it is a fixed-fixed beam between two 6.0 cross walls past the
                                  # recess, frame to frame (hand_calc "rib"; review 24 Oct: 2.0 x 7.5 on 2 x 4 posts, 4.7)
RIB_POST = 6.0
TRAVEL = TAB_IN - RIB_HX          # 3.0 (v5 4.0)
BOLT_Y0, BOLT_T, BOLT_HZ = 6.5, 5.0, 9.0      # bar = bolt section: y 6.5..11.5, z +-9 (18 wide, the grip pull lever)
BOLT_Y1 = BOLT_Y0 + BOLT_T                   # 11.5
OVERLAP = 2.3                     # bolt into the pocket past the locating face (v4 3.0): a +-1.0 head offset still retracts
                                  # the worst bolt 3.0, 0.4 past the 2.6 it needs (concept review 24 Oct; 2.5 left 0.2)
UNDER = math.radians(15.0)        # bolt underside undercut (v2 10): pull-out drives it outward; at 15 deg the 1 g share alone
                                  # (wedge + friction) plus the spring holds a bolt against its own 35 g inertia (side hit)
TIP_CH = 1.0                      # 45 deg chamfer on the bolt tip's top outer edge (rides the feet lead-in)
GP_YC, GP_H, GP_W, GP_D, GP_TAPER, GP_CH = 4.0, 4.0, 8.0, 3.0, math.radians(5.0), 0.5   # grip pocket (J, + 0.5 mouth chamfer)
SPR_R, SPR_YC, SPR_DEPTH = 1.6, 9.6, 3.0      # pen spring U-slot (open top, under the pack): o 3.2, 3 deep per tab
                                              # (slider v6: gap at rest 16 as v5's 2 x 4 + 2 x 4, 10 squeezed)
MAG_D, MAG_DEPTH, MAG_X, MAG_Z = 2.0, 2.2, 7.0, 8.0   # o 2 x 2 N52 magnet, press-fit up into the tab bottom (head hall reads it)
CH_CLR = 0.35                     # slider in its channel, per side
TAB_CH = 1.0                      # 45 deg chamfer on the tab's bottom outer edge (a blade 1.0 inboard cams it in)
REC_CH = 1.0                      # 45 deg chamfer on the recess mouth's outer edges


def geom(pack):
    """every pack-dependent number (C), cartridge coords."""
    P = PACKS[pack]
    g = dict(pack=pack, **P)
    g["X_IN"] = P["W"] / 2 + 0.5
    g["X_OUT"] = g["X_IN"] + WALL
    g["X_RAIL"] = g["X_OUT"] + 1.0
    g["RF"] = g["X_RAIL"] + 0.3                         # receiver locating face
    g["X_TIP"] = g["RF"] + OVERLAP
    g["Y_LIP"] = Y_FL1 + P["H"] + 1.0                   # keeper lips 0.6, pack 1.0 under them
    g["Y_TOP"] = g["Y_LIP"] + 0.6
    g["RAIL_Y0"] = g["Y_TOP"] - 10.0
    g["Z_P0"] = 2.0
    g["Z_P1"] = g["Z_P0"] + P["L"] + 1.0
    g["Z_STOP"] = g["Z_P1"] + 3.0
    g["Z_CW0"], g["Z_CW1"] = g["Z_P1"] + 6.6, g["Z_P1"] + 8.6
    g["Z_X"] = g["Z_P1"] + 12.8                         # plug axis (v4 offsets from the pack end)
    g["Z_EW0"] = g["Z_P1"] + 17.0
    g["L_C"] = g["Z_P1"] + 20.0
    g["Y_XF_FACE"] = g["Y_TOP"] - 2.0
    g["Y_XF_REAR"] = g["Y_XF_FACE"] - 16.0
    g["Y_SH0"], g["Y_SH1"] = g["Y_XF_REAR"] - 2.5, g["Y_XF_REAR"]
    g["Y_PW1"] = g["Y_XF_REAR"] + 10.0
    g["Y_ZONE"] = g["Y_TOP"] - 3.8                      # cartridge top in the plug zone (under the receiver's XT60 lips)
    g["FLOOR_T"] = 2.0 if P["W"] < 40 else 3.0          # floor plate under the pack (35 g crash, hand calc)
    g["Z_T"] = (g["Z_P0"] + g["Z_P1"]) / 2              # replaced by the CG solve in place_tab_line()
    return g


# ======================================================================== slider (one part, both sides: 180 deg about y)
def slider(g, retract=0.0, side=1):
    """slider at rest (retract 0) or pulled in by `retract`; side +1 = x > 0 (the -x one is the same part turned 180 about y)."""
    zt, tip, rf = g["Z_T"], g["X_TIP"], g["RF"]
    tab = box(TAB_IN, TAB_OUT, 0.0, BOLT_Y1, zt - TAB_HZ, zt + TAB_HZ)
    bar = box(TAB_IN, tip, BOLT_Y0, BOLT_Y1, zt - BOLT_HZ, zt + BOLT_HZ)
    drop = OVERLAP * math.tan(UNDER)
    under = prism_xy([(rf, BOLT_Y0), (tip, BOLT_Y0 - drop), (tip, BOLT_Y0)], zt - BOLT_HZ, zt + BOLT_HZ)
    cham = prism_xy([(tip - TIP_CH, BOLT_Y1 + 0.01), (tip + 0.01, BOLT_Y1 + 0.01), (tip + 0.01, BOLT_Y1 - TIP_CH)],
                    zt - BOLT_HZ - 1, zt + BOLT_HZ + 1)
    s = U(tab, bar, under) - cham
    # tab bottom outer edge chamfer (a blade that lands 1.0 inboard cams the tab in, harmless)
    s = s - prism_xy([(TAB_OUT - TAB_CH, -0.01), (TAB_OUT + 0.01, -0.01), (TAB_OUT + 0.01, TAB_CH)], zt - TAB_HZ - 1, zt + TAB_HZ + 1)
    # grip pocket: 8 (z) x 4 (y) mouth on the outer face, 10 deg included taper, 3 deep, centred y 4, 0.5 mouth chamfer
    h1 = GP_H - 2 * GP_D * math.tan(GP_TAPER); w1 = GP_W - 2 * GP_D * math.tan(GP_TAPER)
    gp = hullp([[TAB_OUT + 0.01, GP_YC + sy * GP_H / 2, zt + sz * GP_W / 2] for sy in (-1, 1) for sz in (-1, 1)] +
               [[TAB_OUT - GP_D, GP_YC + sy * h1 / 2, zt + sz * w1 / 2] for sy in (-1, 1) for sz in (-1, 1)])
    gc = hullp([[TAB_OUT + 0.01, GP_YC + sy * (GP_H / 2 + GP_CH), zt + sz * (GP_W / 2 + GP_CH)] for sy in (-1, 1) for sz in (-1, 1)] +
               [[TAB_OUT - GP_CH, GP_YC + sy * GP_H / 2, zt + sz * GP_W / 2] for sy in (-1, 1) for sz in (-1, 1)])
    s = s - gp - gc
    # pen spring U-slot from the inner face, open to the top (the pack 0.5 above keeps the spring in)
    s = s - cyl("x", SPR_YC, zt, SPR_R, TAB_IN - 0.01, TAB_IN + SPR_DEPTH, 32) \
          - box(TAB_IN - 0.01, TAB_IN + SPR_DEPTH, SPR_YC, BOLT_Y1 + 1, zt - SPR_R, zt + SPR_R)
    # magnet seat in the tab bottom, on the ear beyond the blade (z_T + 8)
    s = s - cyl("y", MAG_X, zt + MAG_Z, MAG_D / 2, -0.01, MAG_DEPTH, 24)
    s = s.translate([-retract, 0, 0])
    return s if side > 0 else rot_y180(s, zt)


# ======================================================================== cartridge v5
def cartridge(g, clip=None):
    """cartridge v5 tray (one print, closed end down): walls, rails, floor, latch frames with the channels, recess, lips,
    windows, lead end with the XT60 plug pocket, shelf, set screw boss. clip: a manifold the tray is intersected with."""
    xi, xo, xr, rf = g["X_IN"], g["X_OUT"], g["X_RAIL"], g["RF"]
    yt, lc, zt = g["Y_TOP"], g["L_C"], g["Z_T"]
    t = box(-xo, xo, 0, yt, 0, lc)
    for s in (-1, 1):                                                        # guide rails (x datum), top chamfer 1.0
        a0, a1 = sorted((s * (xo - 0.01), s * xr))
        t = t + box(a0, a1, g["RAIL_Y0"], yt, 0, lc)
        t = t - prism_xy([(s * (xr - 1.0), yt + 0.01), (s * (xr + 0.01), yt + 0.01), (s * (xr + 0.01), yt - 1.0)], -1, lc + 1)
    fl0 = Y_FL1 - g["FLOOR_T"]
    t = t - box(-xi, xi, Y_FL1, yt + 1, g["Z_P0"], g["Z_P1"])               # pack cavity, open top
    t = t - box(-xi, xi, -1, fl0, g["Z_P0"], g["Z_P1"])                     # open under the floor plate
    t = t - box(-(xi - 2.0), xi - 2.0, -1, 4.01, g["Z_P0"], lc - 4.0)       # rim band 4 wide at the walls stays
    # latch zone: frames x REC_X..X_OUT up to the pack, recess between them open to the bottom
    t = t + box(-xo, -REC_X, 0, Y_FL1, zt - REC_Z, zt + REC_Z) + box(REC_X, xo, 0, Y_FL1, zt - REC_Z, zt + REC_Z)
    t = t - box(-REC_X, REC_X, -1, Y_FL1 + 0.01, zt - REC_Z, zt + REC_Z)
    drop = OVERLAP * math.tan(UNDER)
    for s in (-1, 1):
        t = t - prism_xy([(s * (REC_X - 0.01), -0.01), (s * (REC_X + REC_CH), -0.01), (s * (REC_X - 0.01), REC_CH)],
                         zt - REC_Z - 1, zt + REC_Z + 1)
        # channel: open to the pack inside the walls, a closed hole through the wall (its top edge carries the pull-out)
        a0, a1 = sorted((s * (REC_X - 1), s * (xi + 0.01)))
        t = t - box(a0, a1, BOLT_Y0 - CH_CLR, Y_FL1 + 1, zt - BOLT_HZ - CH_CLR, zt + BOLT_HZ + CH_CLR)
        a0, a1 = sorted((s * (REC_X - 1), s * (xo + 1)))
        t = t - box(a0, a1, BOLT_Y0 - CH_CLR, BOLT_Y1 + CH_CLR, zt - BOLT_HZ - CH_CLR, zt + BOLT_HZ + CH_CLR)
        # floor drop for the retracted undercut (it passes x rf - TRAVEL .. tip - TRAVEL)
        a0, a1 = sorted((s * (rf - TRAVEL - 0.5), s * (xo + 1)))
        t = t - box(a0, a1, BOLT_Y0 - CH_CLR - drop, BOLT_Y0 - CH_CLR + 0.01, zt - BOLT_HZ - CH_CLR, zt + BOLT_HZ + CH_CLR)
    for s in (-1, 1):                                                        # pack keeper lips
        t = t + prism_xy([(s * xi, g["Y_LIP"] - 0.6), (s * (xi - 0.6), g["Y_LIP"]), (s * (xi - 0.6), yt), (s * xi, yt)],
                         g["Z_P0"] + 4, g["Z_P1"] - 4)
    # long-wall windows (DEFAULT, mass): rows under and over the floor, 4 ribs, clear of the latch zone by 4
    for (wy0, wy1) in ((4.0, fl0 - 0.01 if fl0 - 4.0 >= 4 else 4.0), (Y_FL1 + 2.0, g["RAIL_Y0"] - 2.0)):
      if wy1 - wy0 >= 4:
        for (z0, z1) in ((g["Z_P0"] + 5, zt - REC_Z - 4), (zt + REC_Z + 4, g["Z_P1"] - 5)):
            n = max(1, int((z1 - z0) // 30))
            dz = (z1 - z0) / n
            for k in range(n):
                a, b = z0 + k * dz + (4.0 if k else 0), z0 + (k + 1) * dz
                if b - a > 8:
                    t = t - box(-xo - 2, -xi + 0.01, wy0, wy1, a, b) - box(xi - 0.01, xo + 2, wy0, wy1, a, b)
    # ---- lead end: end stop, lead loop, cross wall, plug pocket, shelf, end wall, set screws (v4 layout)
    t = t - box(-16.85, 16.85, g["Y_ZONE"], yt + 1, g["Z_P1"], lc + 1)
    t = t - box(-xi, xi, 4.0, g["Y_ZONE"] + 0.01, g["Z_P1"], g["Z_EW0"])
    t = t - box(-xi, xi, -1, 4.01, g["Z_P1"], lc - 4.0)
    t = t + box(-xi - 0.01, xi + 0.01, fl0, Y_FL1 + 5.0, g["Z_P1"] - 0.01, g["Z_STOP"])
    t = t + box(-xi - 0.01, xi + 0.01, g["Y_SH0"], g["Y_PW1"], g["Z_CW0"], g["Z_CW1"])
    t = t + box(-xi - 0.01, xi + 0.01, g["Y_SH0"], g["Y_SH1"], g["Z_CW0"], g["Z_EW0"])
    px, pz = XF_W / 2 + P_CLR, XF_D / 2 + P_CLR
    t = t + box(-px - 2.0, px + 2.0, g["Y_SH1"], g["Y_PW1"], g["Z_CW1"] - 0.01, g["Z_EW0"] + 0.01)
    t = t - box(-px, px, g["Y_SH1"], g["Y_PW1"] + 1, g["Z_X"] - pz, g["Z_X"] + pz)
    for sz in (-1, 1):                                                       # D-key: the two chamfered corners at x +
        z_a = g["Z_X"] + sz * pz
        t = t + hullp([[px - XF_KEY, y, z_a] for y in (g["Y_SH1"], g["Y_PW1"])] + [[px + 0.01, y, z_a] for y in (g["Y_SH1"], g["Y_PW1"])] +
                      [[px + 0.01, y, z_a - sz * XF_KEY] for y in (g["Y_SH1"], g["Y_PW1"])])
    t = t - box(-6.0, 6.0, g["Y_SH0"] - 1, g["Y_SH1"] + 0.01, g["Z_X"] - 2.6, g["Z_X"] + 2.6)   # wire slot
    t = t + box(-xi - 0.01, xi + 0.01, 0, g["Y_ZONE"], g["Z_EW0"], lc)       # end wall
    ss_y = g["Y_XF_REAR"] + 2.0
    t = t + box(-8.25, 8.25, ss_y - 4.25, ss_y + 4.25, lc - 0.01, lc + 3.0)  # set screw boss, 3.0 proud
    for sx in (-1, 1):
        t = t - cyl("z", sx * 4.0, ss_y, 1.25, g["Z_EW0"] - 0.5, lc + 4.0, 24)
    t = t + box(-xo, xo, 0, 4.0, 0, g["Z_P0"] + 2.0) + box(-xo, xo, 0, 4.0, lc - 4.0, lc)
    t = t + box(-xo, -(xi - 2.0), 0, 4.0, 0, lc) + box(xi - 2.0, xo, 0, 4.0, 0, lc)
    t = t + centre_rib(g)
    return t if clip is None else t ^ clip

def centre_rib(g):
    """cartridge v6 centre rib: the tab stop along the recess (notched for the pen spring), on a 6 x 6 post at each end
    past the recess rising into the floor plate, gusseted into it."""
    zt = g["Z_T"]
    rib = box(-RIB_HX, RIB_HX, 0.0, RIB_Y1, zt - REC_Z - 0.01, zt + REC_Z + 0.01) - \
        box(-RIB_HX - 1, RIB_HX + 1, SPR_YC - SPR_R - 0.1, RIB_Y1 + 1, zt - RIB_NOTCH, zt + RIB_NOTCH)
    parts = [rib]
    for sg in (-1, 1):                     # a 6.0 cross wall at each end of the recess, frame to frame (side wall to side
        z0, z1 = sorted((zt + sg * REC_Z, zt + sg * (REC_Z + RIB_POST)))     # wall) under the pack: the rib is fixed into
        parts.append(box(-g["X_OUT"], g["X_OUT"], 0.0, Y_FL1 - 0.01, z0, z1))  # the cartridge's sides, not the 2.0 floor
    return U(*parts)

# ---- TPU edge strips (Scout v3 ask 24 Oct, its system pass `nox/scout/v3/layout_v2.py latch_skids()`: cage role, E;
# drone frame limits, >= 1.0 inside the funnel planes, clear of the 4 bolt keys): L sections bonded on (flexible contact
# adhesive), the foot under the edge, the leg on the outer face
CSTRIP = dict(x0=19.2, x1=23.2, y0=-2.0, leg=(19.0, 21.0, 3.0), dz=32.0)      # cartridge v6: under its bottom long edges
NOSE_BAR = 3.5                                     # o 3.5 (spec), along the 45 deg lower nose chamfer
NOSE_PTS = [(59.2, 34.0, -16.0), (35.2, 34.0, -40.0)]   # drone frame (+y side; mirrored): Scout spec (y 31), its axis 1.0 off the
                                                   # 45 deg chamfer (a 2.9 flat bonds and prints), above z -40 only (keel)
NOSE_PLATE = dict(t=2.2, y=27.0, z0=-22.0, z1=-6.0)    # drone frame (spec y +-27), 2.2 on the lead end wall's face
TAIL_PTS = [(-49.95, 34.0, -25.25), (-35.2, 34.0, -40.0)]   # tail bars on the same chamfer line, from the tail wall face;
                                                   # y 34 (spec 31): outboard of the bolt rings (r 6.2 at y 26.55), at 31 / 31.5 they left 0.5 skins
EAR = dict(y0=26.0, y1=34.5, dz=(-27.0, -12.0))   # each plate's ears out to the bar line, so plate + bars are one print
TAIL_PLATE = dict(t=2.2, y=27.0, z0=-31.5, z1=-6.0, key_r=4.5)   # drone frame (Scout v3 ask 24 Oct, no-keel cage): 2.2 on
                                                   # the tail end wall's face and 45 deg chamfer, o 9 round each bolt key;
                                                   # down to z -31.5 (below it the side bars and strips cover)
SKID_FUNNEL = 1.0                                  # every receiver skid >= 1.0 inside the funnel planes at +-5 yaw / +-2 tilt
RSKID_NAMES = ("nose", "tail")    # each: the end plate + its two bars + the ears joining them, one TPU print bonded on
NARROW = dict(x=25.5, y=6.5)    # receiver v3 (Scout, keel back 24 Oct): |x| <= 25.5 below y 6.5 (drone z -38), clear of the
                                # keel nodes; the collar clip already held the columns inside 25.9 there, so this shaves
                                # the pocket blocks' outer 0.4 for y 5.75..6.5 (wall 2.6 over that band)
def key_lines(g, r):
    """the receiver's bolt key lines, drone (x, y), vertical: from its mount points (Scout v2/v3: x -43.85 / 43.25)."""
    return [(bz - g["Z_T"], bx) for (bx, bz) in r["MOUNT"]]
KEEL_NODES = [(x, s * 29.6, -47.5) for x in (-28.0, -10.0, 28.0) for s in (1, -1)] + [(0.0, -29.6, -47.5)]
KEEL_R = 4.0                    # Scout v3 keel (24 Oct): printed nodes r 4 at these points (+y at x 0 is in the flow view), the
                                # rail between them at y +-29.6, z -47.5, x -28..28; keep 1.0 off every node

def cart_strips(g):
    """cartridge v6 TPU strips (Scout only): foot x 19.2..23.2 under the flat bottom (y -2..0, 2.0 proud of the sliders),
    leg on the wall's outer face x 19..21, y 0..3 (0.68 off the receiver's lead-in at the seat); z = z_T +- 32."""
    zt, c = g["Z_T"], CSTRIP
    out = []
    for s in (-1, 1):
        def X(a, b): return sorted((s * a, s * b))
        out.append(U(box(*X(c["x0"], c["x1"]), c["y0"], 0.0, zt - c["dz"], zt + c["dz"]),
                     box(*X(c["leg"][0], c["leg"][1]), c["y0"], c["leg"][2], zt - c["dz"], zt + c["dz"])))
    return out

def _trim_skids(sk, recv, clip=None):
    """skids clear of the receiver and inside the funnel clip, in RSKID_NAMES order; each bar is bonded on its flat to the
    receiver's chamfer (trimmed by the plate it butts against; each bonded on)."""
    sk = [(m - recv) if clip is None else (m ^ clip) - recv for m in sk]
    plate, bp, bn, tail, tbp, tbn = sk
    big = lambda m: max(m.decompose(), key=lambda p: p.volume())     # drop the boolean's zero-volume shards
    zo, zt_ = plate.bounding_box()[5], tail.bounding_box()[2]       # plate outer faces: the bars end flush (bed face)
    nose = big(U(plate, bp, bn) ^ box(-500, 500, -500, 500, -500, zo))
    return [nose, big(U(tail, tbp, tbn) ^ box(-500, 500, -500, 500, zt_, 500))]

def recv_skids(g, r, moving=()):
    """receiver TPU skids (Scout only), cartridge coords: [nose plate, nose bar +x, nose bar -x, tail plate, tail bar +x,
    tail bar -x]. `moving`: the swap parts; the tail plate keeps out of their path (down, -y)."""
    T = T_dc(g); Ti = np.linalg.inv(T)
    to_c = lambda P: (Ti @ np.array([*P, 1.0]))[:3]
    zt = g["Z_T"]
    nose = r["Z_E1"] + r["END_T"]                                   # lead end wall face, cartridge z
    p = NOSE_PLATE
    yc0, yc1 = p["z0"] - SCOUT_Z0, p["z1"] - SCOUT_Z0               # drone z -> cartridge y
    plate = box(-p["y"], p["y"], yc0, yc1, nose, nose + p["t"])
    ey = (EAR["dz"][0] - SCOUT_Z0, EAR["dz"][1] - SCOUT_Z0)
    def ears(z0, z1):
        return U(*[box(*sorted((s * EAR["y0"], s * EAR["y1"])), *ey, z0, z1) for s in (-1, 1)])
    sph = mf.Manifold.sphere(NOSE_BAR / 2, 24)
    def bars_on(PTS):
        out = []
        for s in (1, -1):
            pts = [to_c((x, s * y, z)) for (x, y, z) in PTS]
            out.append(U(*[mf.Manifold.batch_hull([sph.translate(a.tolist()), sph.translate(b.tolist())])
                           for a, b in zip(pts[:-1], pts[1:])]))                 # round bars (capsules), spec o 3.5
        return out
    bars = bars_on(NOSE_PTS)
    tp = TAIL_PLATE                                                 # tail: wall face + chamfer, 2.2 normal to each
    tail = r["Z_E0"] - r["END_T"]                                   # tail end wall face, cartridge z
    ys = (tp["z0"] - SCOUT_Z0, tp["z1"] - SCOUT_Z0)
    outer = collar_clip(g, SCOUT_INSET - tp["t"] * math.sqrt(2))   # 2.2 normal to the 45 deg chamfer (receiver at SCOUT_INSET)
    shell = U((box(-tp["y"], tp["y"], *ys, tail - tp["t"], g["Z_T"]) ^ outer) - collar_clip(g, SCOUT_INSET),
              box(-tp["y"], tp["y"], *ys, tail - tp["t"], tail) ^ outer, ears(tail - tp["t"], tail))
    for kx, ky in key_lines(g, r):                                  # bolt keys stay open (driver from below)
        if kx < 0:
            c = to_c((kx, ky, 0.0))
            shell = shell - cyl("y", c[0], c[2], tp["key_r"], -100, 200)
    for m in moving:                                                # the cartridge's path out (-y), 0.5 clear
        P = tm(m).vertices
        Q = np.vstack([P + np.array([dx, dy, dz]) for dx in (-0.5, 0.5) for dz in (-0.5, 0.5) for dy in (0.5, -120.0)])
        shell = shell - hullp(Q)
    return [U(plate, ears(nose, nose + p["t"]))] + bars + [shell] + bars_on(TAIL_PTS)

def skid_inset(g, r, recv):
    """smallest collar-clip inset (0.25 steps) that leaves every receiver skid >= SKID_FUNNEL inside the funnel planes at
    +-5 yaw / +-2 tilt (collar_margin), and that inset."""
    T = T_dc(g)
    raw = recv_skids(g, r)
    for inset in np.arange(SKID_FUNNEL, 12.01, 0.25):
        sk = [m ^ collar_clip(g, inset) for m in raw]
        P = np.vstack([tm(m, T).vertices for m in sk if not m.is_empty()])
        if collar_margin(P) >= SKID_FUNNEL:
            return float(inset)
    return None

def skid_report(g, r, sk):
    """per skid: funnel margin (mm, +-5 yaw / +-2 tilt) and the least distance to the 4 bolt key lines (drone xy)."""
    T = T_dc(g); out = {}
    for nm, m in zip(RSKID_NAMES, sk):
        P = tm(m, T).vertices
        q = trimesh.proximity.ProximityQuery(tm(m, T))      # the true least distance from each vertical key line
        key = min(float(np.abs(q.signed_distance(np.array([(kx, ky, z) for z in np.arange(-60.0, 10.01, 0.25)]))).min())
                  for kx, ky in key_lines(g, r))
        out[nm] = dict(funnel_margin_mm=round(collar_margin(P), 2), key_line_mm=round(key, 2))
    return out


# ======================================================================== XT60 (bought; v4 numbers, spec / E)
XF_W, XF_D, XF_H, XF_KEY, P_CLR = 15.5, 8.1, 16.0, 1.5, 0.15
XM_FW, XM_FD, XM_FT = 27.1, 12.1, 3.0
XM_IW, XM_ID, XM_IL = 13.3, 5.9, 7.5
XM_RW, XM_RD, XM_RL = 15.5, 8.1, 6.2
FLOAT, AX_PLAY = 1.0, 0.5

def xt60_female(g, dy=0.0):
    zx, f, r = g["Z_X"], g["Y_XF_FACE"], g["Y_XF_REAR"]
    h = box(-XF_W / 2, XF_W / 2, r, f, zx - XF_D / 2, zx + XF_D / 2)
    for sz in (-1, 1):
        z_a = zx + sz * XF_D / 2
        h = h - hullp([[XF_W / 2 - XF_KEY, y, z_a + sz * 0.01] for y in (r - 1, f + 1)] + [[XF_W / 2 + 0.01, y, z_a + sz * 0.01] for y in (r - 1, f + 1)] +
                      [[XF_W / 2 + 0.01, y, z_a - sz * XF_KEY] for y in (r - 1, f + 1)])
    h = h - box(-6.65, 6.65, f - XM_IL, f + 1, zx - 2.95, zx + 2.95)
    w = U(*[cyl("y", sx * 3.6, zx, 1.8, r - 8.0, r + 0.5) for sx in (-1, 1)])
    return (h + w).translate([0, dy, 0])

def xt60_male(g):
    zx, ym = g["Z_X"], g["Y_XF_FACE"] + 1.5
    f = box(-XM_FW / 2, XM_FW / 2, ym, ym + XM_FT, zx - XM_FD / 2, zx + XM_FD / 2)
    f = f - U(*[cyl("y", s * 10.25, zx, 1.6, ym - 1, ym + XM_FT + 1, 20) for s in (-1, 1)])
    i = box(-XM_IW / 2, XM_IW / 2, ym - XM_IL, ym, zx - XM_ID / 2, zx + XM_ID / 2)
    r = box(-XM_RW / 2, XM_RW / 2, ym + XM_FT, ym + XM_FT + XM_RL, zx - XM_RD / 2, zx + XM_RD / 2)
    w = U(*[cyl("y", sx * 3.6, zx, 1.8, ym + XM_FT + XM_RL - 0.5, ym + XM_FT + XM_RL + 5) for sx in (-1, 1)])
    return f + i + r + w


# ======================================================================== receiver latch block v1 (one design: drone and every bay)
Y_RB = 0.2                         # block bottom 0.2 over the flat bottom (bays: 1.7 to the heads' r 98.5)
CAP, LEAD = 3.0, math.radians(30.0)
HCAP = CAP / math.tan(LEAD)        # 5.20
FOOT_T, DECK_T = 4.0, 3.0
END_T_BY = {"scout": 4.0, "overseer": 4.0}       # end plates; 10 on the Overseer passed FE but made it 191.6 long (> 180 bed)
END_RIB_BY = {"scout": False, "overseer": False}   # deck ends raised to the mount plane (Overseer +-z crash, FE v1)
DECK_END_BY = {"scout": 6.0, "overseer": 8.0}  # deck kept solid this far in from each pack end (FE v1: the end plate
                                                # bends the deck strip beside the window)
LEG_L_BY = {"scout": 16.0, "overseer": 24.0}    # side frame end legs, z (FE v1: 50 N on a foot)
MID_MOUNT = {"scout": False, "overseer": True}   # bosses at the tab line too: the Overseer's 257 N pull-out goes pocket ->
                                                  # leg -> boss, not along a 163 span (concept review 24 Oct); the Scout's
                                                  # tab-line boss would sit under the FC corner (1.9, +-25.5)
PLATE_T_BY = {"scout": 2.5, "overseer": 3.0}   # side plate: root bending in the 35 g side hit (hand calc)
PK_GAP = 0.3                       # pocket floor over the lead-in top (v4 0.55)
REST_GAP = round((OVERLAP + 0.3) * math.tan(UNDER) + 0.05, 2)   # bolt underside over the floor at the seat: >= the
                                   # floor's rise over the retract path (2.6 tan 15 = 0.70), so a seated bolt slides in free
PK_WALL = 3.0
BOSS_R, INS_D, INS_DEPTH = 5.0, 4.0, 6.7      # boss o 10; INS_* kept for the boss height (8.7 under the deck top)
BOLT_D = 3.4                                   # M3 clearance through each boss: the bolts come up from below into M3 heat-set
                                               # inserts in the partner (drone mount, sled, bay body); from above the frame
                                               # plate and props block the key (Scout v3 refit access check, 24 Oct)
LID_T_BY = {"scout": 4.0, "overseer": 5.7}   # deck top to the mount plane; the Overseer's bosses sit on the deck
                                            # (narrow receiver), so its M3 insert hole (6.7) needs 5.7 to keep a 2.0 floor
STRIP_T, STRIP_REC = 2.4, 0.9

def rgeom(g):
    r = dict(g)
    rf = g["RF"]
    r["PLATE_T"] = PLATE_T_BY[g["pack"]]
    r["XW"] = rf + FOOT_T                                   # plates + feet half width
    r["FOOT_Y1"] = Y_RB + HCAP + 2.0
    r["PK_Y0"] = BOLT_Y0 - REST_GAP                         # pocket floor at the mouth (5.95)
    assert r["PK_Y0"] >= Y_RB + HCAP + PK_GAP - 1e-6, "lead-in runs into the pocket floor"
    r["PK_Y1"] = BOLT_Y1 + CH_CLR                           # pocket ceiling
    r["PK_D"] = OVERLAP + 0.3                               # pocket depth: tip 0.3 off the back wall
    r["PK_HZ"] = BOLT_HZ + 0.8
    r["Y_DK0"] = g["Y_TOP"] + STRIP_T - STRIP_REC           # deck underside (strip face = cartridge top)
    r["Y_DK1"] = r["Y_DK0"] + DECK_T
    r["LID_T"] = LID_T_BY[g["pack"]]
    r["Y_M"] = r["Y_DK1"] + r["LID_T"]                      # mounting plane = lid top = boss tops
    r["Z_E0"], r["Z_E1"] = -END_GAP, g["L_C"] + END_GAP
    r["Y_END0"] = r["FOOT_Y1"] + 1.0 if GUIDE else r["RAIL_Y0"]      # bottom of the end locating faces (pitch guidance)
    r["BX"] = rf + 5.3 if g["pack"] == "scout" else r["XW"] - BOSS_R   # boss / lid bolt x: Scout outboard (v4 25.6, its
    # drone interface); Overseer on the deck inside the feet, so the receiver is no wider than its feet (63.6, was 76.2):
    # its bays alternate with Scout bays at R 113 (arm v7)
    r["END_T"] = END_T_BY[g["pack"]]
    zs_m = [r["Z_E0"] - r["END_T"] + BOSS_R, g["Z_X"]] + ([g["Z_T"]] if MID_MOUNT[g["pack"]] else [])
    r["MOUNT"] = [(sx * r["BX"], z) for sx in (-1, 1) for z in zs_m]   # closed end flush
    # with the end face; at the lead end the two lid bolts are the mount bolts (through partner + lid into the deck inserts)
    r["Y_XM"] = g["Y_XF_FACE"] + 1.5
    r["PKX_XW"], r["PKX_ZW"] = XM_FW / 2 + FLOAT, XM_FD / 2 + FLOAT
    r["Y_PCEIL"] = r["Y_XM"] + XM_FT + AX_PLAY
    r["LIP_Y0"] = r["Y_XM"] - 2.0
    return r

def pocket_cut(r):
    zt, rf = r["Z_T"], r["RF"]
    z0, z1 = zt - r["PK_HZ"], zt + r["PK_HZ"]
    yb = r["PK_Y0"] - r["PK_D"] * math.tan(UNDER)
    pk = hullp([[rf - 0.02, y, z] for y in (r["PK_Y0"], r["PK_Y1"]) for z in (z0, z1)] +
               [[rf + r["PK_D"], y, z] for y in (yb, r["PK_Y1"]) for z in (z0, z1)])
    ch = prism_xy([(rf - 0.02, r["PK_Y1"] + 1.0), (rf - 0.02, r["PK_Y1"] - 0.01), (rf + 1.0, r["PK_Y1"] - 0.01)], z0, z1)
    return pk + ch, yb

def receiver(r, clip=None):
    """receiver latch block v1, +x half drawn then mirrored (the XT60 parts are symmetric in x but for nothing)."""
    rf, xw, zt = r["RF"], r["XW"], r["Z_T"]
    END_T = r["END_T"]
    z0, z1 = r["Z_E0"] - END_T, r["Z_E1"] + END_T
    parts = []
    for s in (-1, 1):
        def X(a, b): return sorted((s * a, s * b))
        a0, a1 = X(rf, rf + r["PLATE_T"]); parts.append(box(a0, a1, Y_RB + HCAP, r["Y_DK1"], z0, z1))          # side plate
        poly = [(rf + CAP, Y_RB), (xw, Y_RB), (xw, r["FOOT_Y1"]), (rf, r["FOOT_Y1"]), (rf, Y_RB + HCAP)]
        parts.append(prism_xy([(s * x, y) for x, y in poly], z0, z1))                                      # foot, 30 deg lead-in
        pk, yb = pocket_cut(r)
        a0, a1 = X(rf, rf + r["PK_D"] + PK_WALL)
        parts.append(box(a0, a1, Y_RB, r["PK_Y1"] + PK_WALL, zt - r["PK_HZ"] - PK_WALL, zt + r["PK_HZ"] + PK_WALL))
        # leg rib: pocket block to the deck, 6 x 6 post outside the plate at the tab line (pull-out path)
        a0, a1 = X(rf, rf + r["PK_D"] + PK_WALL); parts.append(box(a0, a1, Y_RB, r["Y_DK1"], zt - 3.0, zt + 3.0))
    for zf, sg in ((r["Z_E0"], -1), (r["Z_E1"], 1)):                                                      # end plates
        y0 = r["RAIL_Y0"]
        pts = [(zf + sg * CAP, y0), (zf + sg * END_T, y0), (zf + sg * END_T, r["Y_DK1"]), (zf, r["Y_DK1"]), (zf, y0 + HCAP)]
        parts.append(prism_zy(pts, -xw, xw))
        if GUIDE:                                       # corner guide posts: the end faces run down to Y_END0 where the
            yg = r["Y_END0"]                            # cartridge's corners bear (end_relief), 30 deg lead-in at the bottom
            pg = [(zf + sg * CAP, yg), (zf + sg * END_T, yg), (zf + sg * END_T, r["Y_DK1"]), (zf, r["Y_DK1"]), (zf, yg + HCAP)]
            for s in (-1, 1):
                a0, a1 = sorted((s * (r["X_IN"] - 1.0), s * xw))
                parts.append(prism_zy(pg, a0, a1))
    parts.append(box(-xw, xw, r["Y_DK0"], r["Y_DK1"], z0, z1))
    if END_RIB_BY[r["pack"]]:                                                                               # end ribs
        de = DECK_END_BY[r["pack"]]
        parts.append(box(-xw, xw, r["Y_DK1"] - 0.01, r["Y_M"], z0, r["Z_P0"] + de))
        parts.append(box(-xw, xw, r["Y_DK1"] - 0.01, r["Y_M"], r["Z_P1"] - de, r["Z_X"] - 10.5))
    # XT60E1-M float pocket body + lid bolt bosses; mount bosses
    zx = r["Z_X"]
    parts.append(box(-r["PKX_XW"] - 2.0, r["PKX_XW"] + 2.0, r["LIP_Y0"], r["Y_DK1"], zx - r["PKX_ZW"] - 2.0, zx + r["PKX_ZW"] + 2.0))
    for (bx, bz) in r["MOUNT"]:
        top = r["Y_DK1"] if abs(bz - zx) < 1e-6 else r["Y_M"]           # lead-end bosses stop under the lid
        bot = r["Y_DK1"] - INS_DEPTH - 2.0
        parts.append(cyl("y", bx, bz, BOSS_R, bot, top))
        parts.append(box(min(bx, np.sign(bx) * rf), max(bx, np.sign(bx) * rf), bot, r["Y_DK1"], bz - 3, bz + 3))
    R = U(*parts)
    # ---- cuts
    cuts = []
    for s in (-1, 1):
        pk, _ = pocket_cut(r); cuts.append(pk if s > 0 else mirx(pk))
        a0, a1 = sorted((s * (r["X_IN"] - 1.6), s * rf))
        cuts.append(box(a0, a1, r["Y_DK0"] - 1, r["Y_DK0"] + STRIP_REC, r["Z_P0"] - 1, r["Z_P1"] - 1))   # strip recesses
    wx = r["X_IN"] - 2.5
    cuts.append(box(-wx, wx, r["Y_DK0"] - 1, r["Y_DK1"] + 1, r["Z_P0"] + DECK_END_BY[r["pack"]], r["Z_P1"] - DECK_END_BY[r["pack"]]))   # deck window
    cuts.append(box(-r["PKX_XW"], r["PKX_XW"], r["Y_XM"], r["Y_DK1"] + 10, zx - r["PKX_ZW"], zx + r["PKX_ZW"]))
    cuts.append(box(-8.5, 8.5, 0.0, r["Y_XM"] + 0.01, zx - 4.8, zx + 4.8))                                   # plug / male path
    for (bx, bz) in r["MOUNT"]:
        top = r["Y_DK1"] if abs(bz - zx) < 1e-6 else r["Y_M"]
        cuts.append(cyl("y", bx, bz, BOLT_D / 2, -5.0, top + 1))                                       # bolt from below
    # side plate windows (DEFAULT, mass): between the latch leg and the ends, y foot + 3 .. rail - 2
    # side frame (DEFAULT, mass): a top band (rails' locating face) full length and three legs down to the feet per side
    # (both ends and the latch leg); the feet lead-ins at those stations capture the rigid cartridge
    wy0, wy1 = (r["FOOT_Y1"] if WINDOWS == "truss" else Y_RB - 1.0), r["RAIL_Y0"] - 2.0     # truss: the feet stay a bottom chord
    for (za, zb) in ((r["Z_E0"] + LEG_L_BY[r["pack"]], zt - r["PK_HZ"] - PK_WALL - 2), (zt + r["PK_HZ"] + PK_WALL + 2, r["Z_E1"] - LEG_L_BY[r["pack"]])):
        if zb - za > 8:
            for s in (-1, 1):
                a0, a1 = sorted((s * (rf - 0.5), s * (xw + 1)))
                if WINDOWS == "box":
                    cuts.append(box(a0, a1, wy0, wy1, za, zb))
                    continue
                # truss (guide moment, arm PR 79): one diagonal strut STRUT wide per window, from its top end-side
                # corner to its bottom tab-side corner, so the side plate carries the couple in-plane
                W_, H_ = zb - za, wy1 - wy0
                L_ = math.hypot(W_, H_); dz_, dy_ = STRUT / 2 * L_ / H_, STRUT / 2 * L_ / W_
                end_side = za if za < zt else zb
                tab_side = zb if za < zt else za
                sg_ = 1 if tab_side > end_side else -1
                t1 = [(end_side, wy1), (tab_side - sg_ * dz_, wy1), (end_side, wy0 + dy_)]     # diagonal: end-leg bottom
                t2 = [(tab_side, wy0), (end_side + sg_ * dz_, wy0), (tab_side, wy1 - dy_)]     # to the tab-side top band
                for tri in (t1, t2):
                    if abs(tri[1][0] - tri[0][0]) > 3 and abs(tri[2][1] - tri[0][1]) > 3:
                        cuts.append(prism_zy(tri, a0, a1))
    R = R - U(*cuts) - cart_path(r) - end_relief(r)
    return R if clip is None else R ^ clip

END_GAP = 0.3                      # end faces to the cartridge's ends (v1; receiver v2: END_GAP2_BY)
WINDOWS = "box"                    # side plate windows: "box" (open) or "truss" (a diagonal strut each, option B)
STRUT = 6.0
GUIDE = False                      # pitch guidance (hub ask 24 Oct, arm PR 79). DEFAULT off: the XT60E1-M's float carrier
                                   # tilts +-2.37 deg in pitch (0.5 play over its 12.1 flange depth; the +-1.06 is roll over
                                   # 27.1), which covers the arm's free mate (1.33 / 1.14 deg with RSS). Option B (True, with
                                   # END_GAP 0.15, WINDOWS "truss"): corner guide posts, pitch at first contact 0.47 / 0.38 deg;
                                   # the Overseer carries the arm's 3.5 / 6.1 N m (index 0.49 / 0.54, +12 g), the Scout's
                                   # clipped posts cannot (0.85 / 0.97 at +19.5 g)
END_RELIEF = 0.5                   # end plate faces set back over |x| < X_IN - 1: the cartridge's end bears at its corners
                                   # (its side walls), so a +-z crash goes into the side frames, not the deck beside the
                                   # XT60 pocket (FE v1: Overseer +z 1.56 -> see result.json)

# ======================================================================== receiver latch block v2 (24 Oct: Spark "the latch
# for the charging bays ... dosent seem to hold in repetition and seems to bend since its just thin plastic"; the swap load
# set: 74 N stall insert, 57 N pull, 3.5 N m pitch guide, 43.5 N across, battery_latch_topo.SWAP). v1 failed it (guide
# index 3.6 / 1.7, 43.5 N across bent the feet 5.4 / 4.4 mm). v2 = a closed channel: solid side walls from the feet to the
# mount plane (no windows), a full deck, end walls to the mount plane with corner guide posts (GUIDE), cross ribs; the
# XT60E1-M float pocket leaves the block for a replaceable XT60 carrier (base + cap) clamped by the two lead-end bolts.
RECV_V = 2
FE_H = {"scout": 1.0, "overseer": 1.5}          # FE voxel size: the Overseer's solid v2 at 1.0 runs > 2 h (iterative)
RECV_BUILD = "+y"                               # print deck down (FE: +z put the guide posts' pull across layers, 1.11)
PROBE_EXEMPT_DEG = 40.0                         # the swap set round the swap opening is exempt from the edge probes to
                                                # this cone (DEFAULT, as the Scout's system_check; was 35. Alt: no
                                                # exemption, the receiver >= 3 inside the cage's bottom plane)
GUIDE2_BY = {"scout": False, "overseer": False}  # corner guide posts (pitch at the XT60's first contact). Off for both
                                                # (arm, 24 Oct: about the cartridge x its free mate is 1.23 / 1.03 deg
                                                # inside the float's +-2.37, so no posts and no guide couple). Scout off
                                                # (DEFAULT): the collar planes clip its posts to a 13 mm lever, ~265 N a
                                                # post at 3.5 N m every swap; its XT60 float (+-2.37 deg) takes the arm's
                                                # 1.33 deg and the seat's couple goes to the stops and bolts (review 24 Oct)
END_GAP2_BY = {"scout": 0.3, "overseer": 0.3}   # end faces to the cartridge's ends (0.15 with posts)
W2_BY = {"scout": 5.0, "overseer": 5.0}         # side wall = foot width: 5.0 leaves 2.0 under the 3.0 lead-in capture (v1: 2.5 / 3.0 plates over 4.0 feet with a 1.0 tip, windows)
DECK2_BY = {"scout": 3.0, "overseer": 3.0}
END2_BY = {"scout": 6.0, "overseer": 6.0}       # end walls: 6.0 leaves 3.0 below the rails behind the 3.0 capture and 3.3 of the lead-end wall past the carrier notch (deck-down print: the Overseer fits the 180 bed on its diagonal)
RIB2 = 4.0                                      # cross ribs deck -> mount plane (tab line, closed end of the carrier)
CARRIER_HZ = 10.0                               # carrier ears / cap half length in z (v1 lid +-10)
FIT = 0.2                                       # carrier in its notch, per side
GUSSET = 6.0                                    # lead-end window corner gussets
WIN2_BY = {"scout": False, "overseer": False}   # side wall truss windows: off (FE v2, Overseer: 43.5 N across moved the truss 1.15 mm, past the XT60 float, solid 0.79; truss 34 g lighter)
STRUT2 = 8.0
GUSSET2 = 0.0                                   # end-wall corner gussets where there are no posts (FE: 8.0 moved the
                                                # across peak to the gusset tip, 1.31; off)
FC_RELIEF = {"scout": (-5.0, 9.0)}             # z about the tab line (Scout v3 FC footprint, drone x -3.3..7.1, + 1.0)
HEAD_Y = 24.5                                   # Scout bolt head seat (the flat; the head fills y HEAD_Y-3..HEAD_Y below
                                                # it): the 3.0 ring (r 6.2) round that band whole inside the collar clip
                                                # at both ends (the nose pair needs HEAD_Y >= 23.65 at inset 11.5; v2's
                                                # 14.0 left the seats 44-52 %, PR 109 review); M3 x 35 (DEFAULT, E)
HEAD_R = 3.2                                    # M3 socket head o 5.5 + 0.45 a side

def rgeom2(g):
    r = rgeom(g)
    p = g["pack"]
    r["END_T"] = END2_BY[p]
    r["XW"] = g["RF"] + W2_BY[p]
    r["Y_DK1"] = r["Y_DK0"] + DECK2_BY[p]
    r["GUIDE"] = GUIDE2_BY[p]
    r["Z_E0"], r["Z_E1"] = -END_GAP2_BY[p], g["L_C"] + END_GAP2_BY[p]
    r["Y_END0"] = r["FOOT_Y1"] + 1.0 if r["GUIDE"] else r["RAIL_Y0"]
    if p != "scout":
        r["BX"] = r["XW"] - BOSS_R
    else:                                          # the counterbore's 3.0 ring clear of the cartridge path (x 20.35):
        r["BX"] = round(g["RF"] + 0.05 + HEAD_R + 3.0, 2)   # 26.6 (v1/v2 25.6 left 2.05 there, PR 109 review)
    e0 = HEAD_R + 3.0 if p == "scout" else BOSS_R   # Scout: the head seat's 3.0 ring stays inside the tail end wall (z -0.1)
    zs_m = [r["Z_E0"] - r["END_T"] + e0, g["Z_X"]] + ([g["Z_T"]] if MID_MOUNT[p] else [])
    r["MOUNT"] = [(sx * r["BX"], z) for sx in (-1, 1) for z in zs_m]
    return r

def carrier_zone(r, fit=0.0):
    """the XT60 carrier's envelope (base + cap), cartridge coords; fit grows it (the body's notch)."""
    zx, bx = r["Z_X"], r["BX"]
    px, pz = r["PKX_XW"] + 2.0 + fit, r["PKX_ZW"] + 2.0 + fit
    ex = bx + BOSS_R + fit
    return U(box(-px, px, r["LIP_Y0"] - fit, r["Y_M"] + 1.0, zx - pz, zx + pz),
             box(-ex, ex, r["Y_DK0"] - fit, r["Y_M"] + 1.0, zx - CARRIER_HZ - fit, zx + CARRIER_HZ + fit))

def receiver_v2(r, clip=None):
    """receiver latch block v2 (body): +x half drawn, mirrored; cartridge coords."""
    rf, xw, zt, ym = r["RF"], r["XW"], r["Z_T"], r["Y_M"]
    END_T = r["END_T"]
    z0, z1 = r["Z_E0"] - END_T, r["Z_E1"] + END_T
    parts = []
    for s in (-1, 1):
        def X(a, b): return sorted((s * a, s * b))
        a0, a1 = X(rf, xw); parts.append(box(a0, a1, Y_RB, ym, z0, z1))                                  # side wall
        a0, a1 = X(rf, rf + r["PK_D"] + PK_WALL)                                                          # pocket block
        parts.append(box(a0, a1, Y_RB, r["PK_Y1"] + PK_WALL, zt - r["PK_HZ"] - PK_WALL, zt + r["PK_HZ"] + PK_WALL))
    parts.append(box(-xw, xw, r["Y_DK0"], r["Y_DK1"], z0, z1))                                             # deck
    for zf, sg in ((r["Z_E0"], -1), (r["Z_E1"], 1)):                                                      # end walls
        y0 = r["RAIL_Y0"]
        pts = [(zf + sg * CAP, y0), (zf + sg * END_T, y0), (zf + sg * END_T, ym), (zf, ym), (zf, y0 + HCAP)]
        parts.append(prism_zy(pts, -xw, xw))
        yg = r["Y_END0"]                                                                                   # corner guide posts
        pg = [(zf + sg * CAP, yg), (zf + sg * END_T, yg), (zf + sg * END_T, ym), (zf, ym), (zf, yg + HCAP)]
        if not r["GUIDE"]:                 # no posts: the end wall runs on down to the feet set back CAP + gap from the
            parts.append(prism_zy([(zf + sg * CAP, Y_RB), (zf + sg * END_T, Y_RB), (zf + sg * END_T, y0),   # cartridge
                                   (zf + sg * CAP, y0)], -xw, xw))   # end (3.3: pitch free to ~3 deg, no guide couple;
                                           # only the top band at 0.3 locates z). FE v2: ending at the rails left the
                                           # Overseer's 43.5 N across case 1.20 mm, index 0.92, at that corner
        if not r["GUIDE"]:                 # no posts: gussets in the end wall's bottom corners at the side walls (FE v2:
            for s in (-1, 1) if GUSSET2 > 0 else ():   # the 43.5 N across case peaked in that re-entrant corner)
                xg = s * rf
                parts.append(hullp([[x, y, z] for (x, y) in ((xg, y0 - GUSSET2), (xg - s * GUSSET2, y0), (xg, y0))
                                    for z in sorted((zf + sg * HCAP / 3.0, zf + sg * END_T))]))
            continue
        parts.append(prism_zy(pg, -xw, xw))  # both ends run full width down to the posts' bottom (FE v2: a window between
                                           # the lead-end posts put the 43.5 N across peak on its corner, 1.14); the lead
                                           # end keeps only a slot for the cartridge's set-screw boss (cut below)
    parts.append(box(-xw, xw, r["Y_DK1"] - 0.01, ym, zt - RIB2 / 2, zt + RIB2 / 2))                       # tab-line rib
    zc = r["Z_X"] - CARRIER_HZ - FIT - RIB2
    parts.append(box(-xw, xw, r["Y_DK1"] - 0.01, ym, zc, zc + RIB2))                                     # carrier rib
    zx = r["Z_X"]
    for (bx, bz) in r["MOUNT"]:
        top = r["Y_DK0"] if abs(bz - zx) < 1e-6 else r["Y_M"]          # lead end: the carrier's ears sit on these
        bot = r["Y_DK1"] - INS_DEPTH - 2.0
        if abs(bx) > rf:                   # Scout (bolts outboard, v1's 25.6): a full-height column beside the wall, the
            parts.append(cyl("y", bx, bz, BOSS_R, Y_RB, top))           # head seated on a flat at HEAD_Y from below
            parts.append(cyl("y", bx, bz, HEAD_R + 3.0, Y_RB, HEAD_Y + 3.0))   # 3.0 round the head counterbore
                                           # (an access channel instead cut the wall's corner: across 0.96, FE v2)
        parts.append(cyl("y", bx, bz, BOSS_R, bot, top))
        parts.append(box(min(bx, np.sign(bx) * rf), max(bx, np.sign(bx) * rf), bot, min(top, r["Y_DK1"]), bz - 3, bz + 3))
    R = U(*parts)
    cuts = [carrier_zone(r, FIT)]
    for s in (-1, 1):
        pk, _ = pocket_cut(r); cuts.append(pk if s > 0 else mirx(pk))
        a0, a1 = sorted((s * (r["X_IN"] - 1.6), s * rf))
        cuts.append(box(a0, a1, r["Y_DK0"] - 1, r["Y_DK0"] + STRIP_REC, r["Z_P0"] - 1, r["Z_P1"] - 1))   # strip recesses
    cuts.append(box(-8.5, 8.5, 0.0, r["Y_XM"] + 0.01, zx - 4.8, zx + 4.8))                                # plug / male path
    if True:                               # lead end wall: slot for the cartridge's set-screw boss (x +-8.25, 3.0 proud),
        ss_top = r["Y_XF_REAR"] + 2.0 + 4.25 + 1.0       # 1.0 clear, open from below with a 30 deg lead-in each side
        sx = 8.25 + 1.0; z0s, z1s = r["Z_E1"] - 0.5, r["Z_E1"] + END_T + 1.0
        ym_ = r["Y_END0"] if r["GUIDE"] else Y_RB
        cuts.append(box(-sx, sx, -5.0, ss_top, z0s, z1s))
        cuts.append(prism_xy([(-sx - CAP, ym_ - 0.01), (sx + CAP, ym_ - 0.01), (sx, ym_ + HCAP), (-sx, ym_ + HCAP)], z0s, z1s))
    for (bx, bz) in r["MOUNT"]:
        cuts.append(cyl("y", bx, bz, BOLT_D / 2, -5.0, r["Y_M"] + 1))                                     # bolts from below
        if abs(bx) > rf:                   # head counterbore from below to a flat above the collar clip
            cuts.append(cyl("y", bx, bz, HEAD_R, -5.0, HEAD_Y))
    if r["pack"] in FC_RELIEF:            # the Scout's FC sits 0.1 under the mount plane over the tab line: a 1.0 relief
        a_, b_ = FC_RELIEF[r["pack"]]     # there (drone x -4..+8), the rest of the top stays on the plane (the bed face)
        cuts.append(box(-200, 200, ym - 1.0, ym + 1.0, zt + a_, zt + b_))
    R = R - U(*cuts) - cart_path(r) - end_relief(r)
    if WIN2_BY[r["pack"]]:                       # lightening (DEFAULT where FE allows): truss windows in the side walls
        wy0, wy1 = r["FOOT_Y1"] + 1.0, r["RAIL_Y0"] - 2.0          # between the feet chord and the top band
        for (za, zb) in ((r["Z_E0"] + LEG_L_BY[r["pack"]], zt - r["PK_HZ"] - PK_WALL - 2),
                         (zt + r["PK_HZ"] + PK_WALL + 2, r["Z_X"] - CARRIER_HZ - FIT - RIB2 - 2)):
            if zb - za < 12: continue
            W_, H_ = zb - za, wy1 - wy0
            L_ = math.hypot(W_, H_); dz_, dy_ = STRUT2 / 2 * L_ / H_, STRUT2 / 2 * L_ / W_
            end_side, tab_side = (za, zb) if za < zt else (zb, za)
            sg_ = 1 if tab_side > end_side else -1
            for s in (-1, 1):
                a0, a1 = sorted((s * (rf - 0.5), s * (xw + 1)))
                for tri in ([(end_side, wy1), (tab_side - sg_ * dz_, wy1), (end_side, wy0 + dy_)],
                            [(tab_side, wy0), (end_side + sg_ * dz_, wy0), (tab_side, wy1 - dy_)]):
                    R = R - prism_zy(tri, a0, a1)
    return R if clip is None else R ^ clip

def xt60_carrier(r):
    """XT60 carrier v1 (replaceable with the XT60E1-M and its pigtail): base = the float pocket with the pull lips and
    two ears on the lead-end bosses; cap = the v1 lid (the float ceiling). Both clamped by the two lead-end M3 bolts
    (from below: body boss -> base ear -> cap -> partner insert)."""
    zx, bx = r["Z_X"], r["BX"]
    px, pz = r["PKX_XW"] + 2.0, r["PKX_ZW"] + 2.0
    base = U(box(-px, px, r["LIP_Y0"], r["Y_DK1"], zx - pz, zx + pz),
             box(-(bx + BOSS_R), bx + BOSS_R, r["Y_DK0"], r["Y_DK1"], zx - CARRIER_HZ, zx + CARRIER_HZ))
    base = base - box(-r["PKX_XW"], r["PKX_XW"], r["Y_XM"], r["Y_DK1"] + 1, zx - r["PKX_ZW"], zx + r["PKX_ZW"])
    base = base - box(-8.5, 8.5, r["LIP_Y0"] - 1, r["Y_XM"] + 0.01, zx - 4.8, zx + 4.8)
    for s in (-1, 1):
        base = base - cyl("y", s * bx, zx, BOLT_D / 2, r["Y_DK0"] - 1, r["Y_DK1"] + 1, 24)
    return base, lid(r)


def end_relief(r):
    xr = r["X_IN"] - 1.0
    return U(box(-xr, xr, r["RAIL_Y0"] - 1, r["Y_DK0"] - 0.01, r["Z_E0"] - END_RELIEF, r["Z_E0"] + 0.01),
             box(-xr, xr, r["RAIL_Y0"] - 1, r["Y_DK0"] - 0.01, r["Z_E1"] - 0.01, r["Z_E1"] + END_RELIEF))

def lid(r):
    zx, bx = r["Z_X"], r["BX"]
    hz = max(BOSS_R + 1.0, 10.0)                                            # v4 lid: +-10 (3.0 bands neck-fail)
    l = box(-(bx + BOSS_R), bx + BOSS_R, r["Y_DK1"], r["Y_M"], zx - hz, zx + hz)
    for s in (-1, 1):
        a0, a1 = sorted((s * 9.26, s * (r["PKX_XW"] - 0.25)))
        l = l + box(a0, a1, r["Y_PCEIL"], r["Y_DK1"] + 0.01, zx - r["PKX_ZW"] + 0.25, zx + r["PKX_ZW"] - 0.25)
    l = l - box(-9.25, 9.25, r["Y_PCEIL"] - 1, r["Y_M"] + 1, zx - 5.55, zx + 5.55)
    for s in (-1, 1):
        l = l - cyl("y", s * bx, zx, 1.7, r["Y_DK1"] - 1, r["Y_M"] + 1, 24)
    return l

def strips(r):
    return U(*[box(*sorted((s * (r["X_IN"] - 1.55), s * (r["RF"] - 0.05))), r["Y_TOP"], r["Y_TOP"] + STRIP_T,
                   r["Z_P0"] - 0.8, r["Z_P1"] - 1.2) for s in (-1, 1)])

def cart_path(r):
    """space the cartridge (and its 3.0 capture) sweeps on the way in; nothing of the receiver in it."""
    rf, lc = r["RF"], r["L_C"]
    c = box(-rf, rf, -60, r["Y_DK0"], r["Z_E0"], r["Z_E1"])
    k = math.tan(LEAD)
    for s in (-1, 1):
        poly = [(0, -60), (rf + CAP + 62 * k, -60), (rf + CAP, Y_RB), (rf, Y_RB + HCAP), (0, Y_RB + HCAP)]
        c = c + prism_xy([(s * x, y) for x, y in poly], r["Z_E0"] - CAP - 1, r["Z_E1"] + CAP + 1)
    for zf, sg in ((r["Z_E0"], -1), (r["Z_E1"], 1)):
        y0 = r.get("Y_END0", r["RAIL_Y0"])
        poly = [(lc / 2, -60), (zf + sg * CAP, -60), (zf + sg * CAP, y0), (zf, y0 + HCAP), (lc / 2, y0 + HCAP)]
        c = c + prism_zy(poly, -rf, rf)
    return c


# ======================================================================== Scout drone frame, collar planes (Scout v3 system pass)
COLLAR_K = 84.8                     # contact faces max(|x|, |y|) = 84.8 + z (drone frame; Scout v3 system pass, C)
COLLAR_NEED = 7.0                   # receiver + cartridge >= 7.0 inside below z 20 (cage band 1.0..4.0 + 3.0 lead)
SCOUT_Z0 = -44.5                    # cartridge flat bottom at drone z -44.5 (ring r 100, as v4)
CG_D = np.array([0.0, 0.0, 4.2])    # Scout CG (E, dimensions H flow row)

def T_dc(g):
    """drone <- cartridge (4x4): d = (z - z_T, x, y + SCOUT_Z0). The XT60 (lead) end is the arm side of the wheel (-Y)
    = the drone nose (hub 24 Oct, arm v6 system pass: the XT60 mates on the arm's own pitch stiffness, 1.03 / 0.52 / 0.70
    deg there vs 2.56 / 1.71 / 3.47 at the far end, float carrier +-1.06 deg); was d = (z_T - z, -x, ...)."""
    return np.array([[0, 0, 1, -g["Z_T"]], [1, 0, 0, 0], [0, 1, 0, SCOUT_Z0], [0, 0, 0, 1.0]])

def collar_margin(P, yaw=5.0, tilt=2.0):
    """min over yaw / tilt about the CG of (84.8 + z) - max(|x|, |y|), points P (drone frame), z < 20 only."""
    P = P[P[:, 2] < 20.0]
    if not len(P): return 99.0
    worst = 99.0
    for a in (-yaw, 0, yaw):
        for tx in (-tilt, 0, tilt):
            for ty in (-tilt, 0, tilt):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                cx, sx = math.cos(math.radians(tx)), math.sin(math.radians(tx))
                cy, sy = math.cos(math.radians(ty)), math.sin(math.radians(ty))
                Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Q = (P - CG_D) @ (Rz @ Ry @ Rx).T + CG_D
                m = (COLLAR_K + Q[:, 2]) - np.maximum(np.abs(Q[:, 0]), np.abs(Q[:, 1]))
                worst = min(worst, float(m.min()))
    return worst

def collar_clip(g, inset):
    """the frustum max(|x|,|y|) <= 84.8 + z - inset (drone frame) in cartridge coords."""
    zlo, zhi = inset - COLLAR_K + 0.5, 40.0           # apex side kept positive (a negative half size flips the hull)
    pts = []
    for zz in (zlo, zhi):
        h = COLLAR_K + zz - inset
        for sx in (-1, 1):
            for sy in (-1, 1): pts.append([sx * h, sy * h, zz])
    Cd = hullp(pts)
    return Cd.transform(np.linalg.inv(T_dc(g))[:3])


def keel_report(B):
    """Scout keel clearance (C): per part, the least distance from a node's surface (r KEEL_R) and from the rail line swept
    with the node radius (x -28..28 every 0.5, conservative), drone frame; need >= 1.0."""
    g = B["g"]; T = T_dc(g)
    parts = dict(receiver=U(B["recv"], B["lid"]), cartridge_strips=U(*B.get("cstrips", [])),
                 receiver_end_pieces=U(*B.get("rskids", [])), cartridge=B["cart"], cartridge_end_skids=U(*B["skids"]))
    rail = np.array([(x, s * 29.6, -47.5) for x in np.arange(-28.0, 28.01, 0.5) for s in (1, -1)])
    out = {}
    for nm, m in parts.items():
        if m.is_empty(): continue
        q = trimesh.proximity.ProximityQuery(tm(m, T))
        dn = np.abs(q.signed_distance(np.array(KEEL_NODES))) - KEEL_R
        dr = np.abs(q.signed_distance(rail)) - KEEL_R
        out[nm] = dict(nodes_mm=round(float(dn.min()), 2), rail_swept_mm=round(float(dr.min()), 2))
    out["need_mm"] = 1.0
    return out

def head_seats(B):
    """each Scout bolt's head seat (C): the share of the head's bearing annulus (r 1.75..2.7) on the flat (sampled 0.2 above
    it), and the least share, over the head's band y HEAD_Y-3..HEAD_Y, of the 3.0 ring round the counterbore
    (r HEAD_R..HEAD_R + 3) that is receiver (rule: boss wall >= 3 round every counterbore)."""
    r = B["r"]; R_ = tm(B["recv"]); out = []
    for (bx, bz) in r["MOUNT"]:
        if abs(bx) <= r["RF"]: continue
        def share(r0, r1, y):
            P = np.array([(bx + rr * math.cos(a), y, bz + rr * math.sin(a))
                          for rr in np.linspace(r0, r1, 6) for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)])
            return round(float(R_.contains(P).mean()), 3)
        out.append(dict(at=[bx, bz], bearing=share(1.75, 2.7, HEAD_Y + 0.2),
                        ring=min(share(HEAD_R + 0.05, HEAD_R + 2.95, y) for y in np.linspace(HEAD_Y - 2.9, HEAD_Y - 0.1, 8))))
    return out


# ======================================================================== assembly API (for nox/box/swap_system.py)
_CACHE = {}

def build(pack, clip_inset=None):
    """all meshes for one pack, cartridge coords; the Scout set is clipped to the collar planes (inset from the search)."""
    key = (pack, clip_inset)
    if key in _CACHE: return _CACHE[key]
    g = place_tab_line(pack)
    r1 = rgeom(g)
    r = rgeom2(g) if RECV_V == 2 else r1
    clip = cclip = None
    skids = []
    if pack == "scout":
        inset = clip_inset if clip_inset is not None else SCOUT_INSET
        clip = collar_clip(g, inset)
        cclip = collar_clip(g, inset + SKID_T)
        band = box(-g["X_OUT"], g["X_OUT"], 0.0, g["Y_TOP"], 0.0, g["L_C"]) ^ clip
        wrap = box(-g["X_IN"] - WALL, g["X_IN"] + WALL, Y_FL1 - g["FLOOR_T"], Y_FL1 + g["H"] + WALL,     # the pack + its walls:
                   g["Z_P0"] - WALL, g["Z_P1"] + WALL)                                                  # never cut deeper
        band = band - cclip - wrap
        cclip = cclip + (wrap ^ clip)
        for z0, z1 in ((-1.0, g["L_C"] / 2), (g["L_C"] / 2, g["L_C"] + 4.0)):
            sk = band ^ box(-50, 50, -1, 100, z0, z1)
            if not sk.is_empty(): skids.append(sk)
    if RECV_V == 2:
        cb, cc = xt60_carrier(r)
        if clip is not None: cb, cc = cb ^ clip, cc ^ clip
        out_recv = receiver_v2(r, clip)
        if pack == "scout":                                          # receiver v3: narrowed under the keel
            out_recv = out_recv - U(*[box(*sorted((s * NARROW["x"], s * 200)), -50, NARROW["y"], -200, 400) for s in (-1, 1)])
        cart = cartridge(g, cclip)
        cst = cart_strips(g) if pack == "scout" else []
        moving = [cart, slider(g, 0, 1), slider(g, 0, -1), slider(g, TRAVEL, 1), slider(g, TRAVEL, -1)] + skids + cst
        out = dict(g=g, r=r, r1=r1, clip=clip, cart=cart, recv=out_recv, carrier=cb, cap=cc,
                   lid=U(cb, cc), strips=strips(r), male=xt60_male(g), skids=skids, recv_v1=receiver(r1, clip),
                   cstrips=cst,
                   rskids=_trim_skids(recv_skids(g, r, moving), out_recv, collar_clip(g, skid_inset(g, r, out_recv)))
                   if pack == "scout" else [])
        # lid = the clamped XT60 carrier (base + cap) wherever the set is checked as one; cap / carrier are the prints
    else:
        out = dict(g=g, r=r, r1=r1, clip=clip, cart=cartridge(g, cclip), recv=receiver(r, clip), lid=lid(r), strips=strips(r),
                   male=xt60_male(g), skids=skids, carrier=None, recv_v1=None)
    _CACHE[key] = out
    return out

def latch_assembly(pack="scout", state="latched"):
    """meshes in cartridge coordinates (manifold3d.Manifold). state: 'latched' (sliders at rest, bolts in the pockets),
    'gripped' (sliders squeezed 4.0 by the fingers, bolts in), 'free' (cartridge out of the receiver, sliders at rest)."""
    B = build(pack)
    g = B["g"]
    rt = 0.0 if state in ("latched", "free") else TRAVEL
    d = {"cartridge": B["cart"], "slider_px": slider(g, rt, 1), "slider_nx": slider(g, rt, -1), "pack": pack_body(g),
         **{f"cartridge_skid_{k}": m for k, m in zip(("closed", "lead"), B["skids"])},
         **{f"cartridge_strip_{k}": m for k, m in zip(("px", "nx"), B.get("cstrips", []))},
         "xt60_plug": xt60_female(g), "receiver": B["recv"], "receiver_lid": B.get("cap", B["lid"]), "receiver_strips": B["strips"],
         "xt60e1m": B["male"]}
    if B.get("carrier") is not None: d["xt60_carrier"] = B["carrier"]
    for k, m in zip(RSKID_NAMES, B.get("rskids", [])):
        d[f"receiver_skid_{k}"] = m
    if state == "free":
        for k in ("receiver", "receiver_lid", "receiver_strips", "xt60e1m", "xt60_carrier") + tuple(
                k_ for k_ in d if k_.startswith("receiver_skid_")): d.pop(k, None)
    return d

def pack_body(g):
    return box(-g["W"] / 2, g["W"] / 2, Y_FL1, Y_FL1 + g["H"], g["Z_P0"] + 0.5, g["Z_P1"] - 0.5)


# ======================================================================== tab line on the swap CG
SPRING_G, MAG_G = 0.3, 0.06          # pen spring, o 2 x 2 magnet (E)
SPRING_K, SPRING_FREE = 0.20, 20.0  # o 3 compression spring, music wire 0.3, mean o 2.7, ~20 active coils: solid ~6.6 under the
                                    # 8.0 squeezed length, 671 MPa at 2.4 N (E, DEFAULT; a stiffer o 3 spring goes solid)
MU = 0.3                            # PETG on PETG (E)

def swap_cg(g):
    """mass (g) and z of the swap assembly (pack + plug + cartridge + 2 sliders + spring + magnets), cartridge coords."""
    items = []
    m_body = g["m"] - g["m_plug"]
    items.append((m_body, np.array([0.0, Y_FL1 + g["H"] / 2, (g["Z_P0"] + g["Z_P1"]) / 2])))
    items.append((g["m_plug"], np.array([0.0, g["Y_XF_REAR"], g["Z_X"]])))
    for nm, m in (("cart", cartridge(g)), ("s1", slider(g, 0, 1)), ("s2", slider(g, 0, -1))):
        t = tm(m)
        items.append((t.volume / 1000 * RHO["PETG"], np.asarray(t.center_mass)))
    items.append((SPRING_G + 2 * MAG_G, np.array([0.0, SPR_YC, g["Z_T"]])))
    M = sum(a for a, _ in items)
    c = sum(a * b for a, b in items) / M
    return M, c, items

TAB_OFF = {"scout": 0.0, "overseer": 5.18}   # Scout kept on its swap CG (DEFAULT): at 3.71 the pack's tail-bottom corner sits
                                              # 6.4 inside the collar planes and the cartridge can no longer wrap it at 7.0  # tab line from the swap CG toward the XT60 (lead, arm-side) end: each catch
                                              # centre and tab line sit on the latch set's centre (arm v7 swap geometry, PR 79,
                                              # hub DEFAULT 24 Oct; grip Y_C 165); was Overseer + 30 (its thrust axis)

def place_tab_line(pack, it=3):
    g = geom(pack)
    for _ in range(it):
        M, c, _ = swap_cg(g)
        g["Z_T"] = round(float(c[2]) + TAB_OFF[pack], 2)
    g["SWAP_M"], g["SWAP_CG"] = M, c
    return g


# ======================================================================== checks
def insertion_sweep(B, side_off=0.0, dz=1.0, depth=50.0):
    """cartridge from `depth` below seat to the seat. Sliders free: at each step the least retraction (bisection) that
    clears the receiver = what the feet lead-ins cam them to. Returns rows (d, retract) and the worst overlap."""
    g, R = B["g"], U(B["recv"], B["lid"], B["strips"], B["male"], *B.get("rskids", []))
    C = U(B["cart"], xt60_female(g), *B.get("cstrips", []))
    rows, worst_c = [], 0.0
    d = -depth
    while d <= 1e-6:
        Cd = C.translate([side_off, d, 0])
        worst_c = max(worst_c, vol(Cd ^ R))
        rr = []
        for s in (1, -1):
            lo, hi = 0.0, TRAVEL
            if vol(slider(g, 0, s).translate([side_off, d, 0]) ^ R) < 0.01: rr.append(0.0); continue
            if vol(slider(g, hi, s).translate([side_off, d, 0]) ^ R) >= 0.01: rr.append(None); continue
            for _ in range(10):
                mid = (lo + hi) / 2
                if vol(slider(g, mid, s).translate([side_off, d, 0]) ^ R) < 0.01: hi = mid
                else: lo = mid
            rr.append(round(hi, 2))
        rows.append((round(d, 2), rr))
        d += dz
    return rows, round(worst_c, 3)

def gripped_pull_sweep(B, depth=50.0, dz=1.0):
    g, R = B["g"], B["recv"] + B["lid"] + B["strips"] + B["male"]
    A = B["cart"] + xt60_female(g) + slider(g, TRAVEL, 1) + slider(g, TRAVEL, -1)
    worst, d = 0.0, 0.0
    while d >= -depth - 1e-6:
        worst = max(worst, vol(A.translate([0, d, 0]) ^ R)); d -= dz
    return round(worst, 3)

def squeeze_sweep(B, n=9):
    """seated: sliders 0..4 in; vs cartridge, receiver, each other."""
    g = B["g"]; R = B["recv"] + B["lid"] + B["strips"] + B["male"]; C = B["cart"]
    w = dict(vs_cart=0.0, vs_recv=0.0, vs_each=0.0)
    for k in range(n):
        rt = TRAVEL * k / (n - 1) - (0.02 if k == n - 1 else 0.0)
        a, b = slider(g, rt, 1), slider(g, rt, -1)
        w["vs_cart"] = max(w["vs_cart"], vol(a ^ C), vol(b ^ C)); w["vs_recv"] = max(w["vs_recv"], vol(a ^ R), vol(b ^ R))
        w["vs_each"] = max(w["vs_each"], vol(a ^ b))
    return {k: round(v, 3) for k, v in w.items()}

def static_fit(B):
    g = B["g"]
    parts = dict(cart=B["cart"], plug=xt60_female(g), s1=slider(g, 0, 1), s2=slider(g, 0, -1), pack=pack_body(g),
                 recv=B["recv"], lid=B["lid"], strips=B["strips"], male=B["male"])
    names = list(parts); res = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = vol(parts[names[i]] ^ parts[names[j]])
            if v > 0.01: res[f"{names[i]}-{names[j]}"] = round(v, 3)
    return res

def bearing_drop(B):
    """sliders at rest: how far the cartridge drops before each bolt bears on its pocket floor (first contact, 0.05 steps)."""
    g = B["g"]; R = B["recv"]
    out = []
    for s in (1, -1):
        d = 0.0
        while d < 2.0 and vol(slider(g, 0, s).translate([0, -d, 0]) ^ R) < 0.01: d += 0.05
        out.append(round(d, 2))
    return out

def finger_model(g, open_=True, off=0.0, reach=6.0):
    """two v4 blades (3 thick, 13 wide in z) at their open (11..14) or squeezed (6..9) inner faces, head offset off in x."""
    xi = 11.0 if open_ else TAB_OUT - TRAVEL
    return U(*[box(*sorted((s * xi + off, s * (xi + 3.0) + off)), -20.0, reach, g["Z_T"] - 6.5, g["Z_T"] + 6.5) for s in (-1, 1)])

def grip_checks(B):
    """blades enter the recess open at head offset -1 / 0 / +1 (x), then squeeze; margins C."""
    g = B["g"]; C = B["cart"]
    res = {}
    for off in (-1.0, 0.0, 1.0):
        Fo = finger_model(g, True, off)
        res[f"open_vs_cart_off{off:+.0f}"] = ov(Fo, C)
        res[f"open_vs_tabs_rest_off{off:+.0f}"] = round(vol(Fo ^ (slider(g, 0, 1) + slider(g, 0, -1))), 3)
        # squeezed: each tab stops on the centre rib, so both retract TRAVEL whatever the offset (v5: TRAVEL - |off|)
        res[f"min_retract_off{off:+.0f}"] = TRAVEL
    res["retract_needed"] = OVERLAP + 0.3
    res["retract_margin_at_1mm"] = round(TRAVEL - (OVERLAP + 0.3), 2)
    Fs = finger_model(g, False, 0.0)
    res["squeezed_vs_cart"] = ov(Fs, C)
    res["squeezed_vs_sliders_squeezed"] = round(vol(Fs ^ (slider(g, TRAVEL, 1) + slider(g, TRAVEL, -1))), 3)
    return res


def asym_squeeze(B, off=1.0):
    """head offset `off` in x: the blades close about x = off, each tab stops on the centre rib (cartridge v6), so both
    sliders retract TRAVEL (v5: TRAVEL -+ off). Overlaps (mm3) and the worst bolt's tip vs the locating face (C)."""
    g = B["g"]; R = B["recv"] + B["lid"] + B["strips"] + B["male"]; C = B["cart"]
    a, b = slider(g, TRAVEL, 1), slider(g, TRAVEL, -1)
    tip_in = TRAVEL - OVERLAP                            # how far each bolt's tip sits inside the locating face
    A = C + xt60_female(g) + a + b
    pull = 0.0
    for k in range(0, 51, 2):
        pull = max(pull, vol(A.translate([0, -k, 0]) ^ R))
    return dict(off=off, retract=[TRAVEL, TRAVEL], vs_cart=round(max(vol(a ^ C), vol(b ^ C)), 3),
                vs_recv=round(max(vol(a ^ R), vol(b ^ R)), 3), vs_each=round(vol(a ^ b), 3),
                tip_inside_face_mm=round(tip_in, 2), pull_out_mm3=round(pull, 3))


def pitch_at_contact(B, d=6.0, tmax=4.0):
    """largest pitch (deg, about x through the cartridge's middle) the receiver lets the gripped cartridge take d below the
    seat (d 6.0 = the XT60 noses' first contact, the PR 44 stack), both senses; bisection on overlap with receiver + lid."""
    g = B["g"]
    R = B["recv"] + B["lid"]
    A = B["cart"] + xt60_female(g) + slider(g, TRAVEL, 1) + slider(g, TRAVEL, -1)
    c = np.array([0.0, g["Y_TOP"] / 2, g["L_C"] / 2])
    def hit(th):
        m = A.translate((-c).tolist()).rotate([th, 0, 0]).translate((c + np.array([0, -d, 0])).tolist())
        return vol(m ^ R) > 0.5
    out = {}
    for sgn in (1, -1):
        lo, hi = 0.0, tmax
        if not hit(sgn * hi): out[sgn] = f">{tmax}"; continue
        for _ in range(12):
            mid = (lo + hi) / 2
            if hit(sgn * mid): hi = mid
            else: lo = mid
        out[sgn] = round(lo, 2)
    return dict(d_below_seat=d, pitch_pos_deg=out[1], pitch_neg_deg=out[-1])


def float_tilt():
    """free tilt of the XT60E1-M on its float carrier (deg): the 0.5 axial play over the flange's depth (z, 12.1) for pitch
    about the cartridge's x (the arm's slide-tangential axis) and over its width (x, 27.1) for roll about z."""
    return dict(pitch_deg=round(math.degrees(math.atan(AX_PLAY / XM_FD)), 2), roll_deg=round(math.degrees(math.atan(AX_PLAY / XM_FW)), 2))


# ======================================================================== latched-state read (hall under the tab magnet)
BR_N52 = 1.45                          # T (E)

def mag_by(px, py, pz, mx, my, mz, d=MAG_D, h=MAG_DEPTH, n=48):
    """B_y (mT) at a point from a o d x h axially (y) magnetised N52 cylinder centred (mx, my, mz): two charged discs."""
    rr = (np.arange(n) + 0.5) / n * d / 2
    th = (np.arange(n) + 0.5) / n * 2 * math.pi
    R_, T_ = np.meshgrid(rr, th, indexing="ij"); dA = (d / 2 / n) * (2 * math.pi / n) * R_
    xs, zs = mx + R_ * np.cos(T_), mz + R_ * np.sin(T_)
    by = 0.0
    for sgn, yy in ((1, my + h / 2), (-1, my - h / 2)):
        rx, ry, rz = px - xs, py - yy, pz - zs
        r3 = (rx ** 2 + ry ** 2 + rz ** 2) ** 1.5
        by += sgn * float((ry / r3 * dA).sum())
    return BR_N52 / (4 * math.pi) * by * 1000.0

def hall_sweep(g, offs=(-1.0, -0.5, 0.0, 0.5, 1.0), gaps=(-0.7, -1.0, -1.3)):
    """the latched and half-in readings over the head's x offset (the hall rides on the head) and the hall gap: the
    worst latched reading must stay above the worst half-in one with the threshold between them."""
    rows = []
    for o in offs:
        for gy in gaps:
            h = hall_read(g, (7.0 + o, gy))
            rows.append(dict(off=o, gap=gy, latched=h["latched"], half_in=h["half_in"]))
    out = dict(rows=rows)
    for tol in (0.5, 1.0):
        sub = [r_ for r_ in rows if abs(r_["off"]) <= tol + 1e-9]
        lo = min(r_["latched"] for r_ in sub); hi = max(r_["half_in"] for r_ in sub)
        out[f"off_{tol}"] = dict(latched_min=lo, half_in_max=hi, threshold_mT=round((lo + hi) / 2, 1), separation=round(lo / hi, 1))
    return out

def hall_read(g, hall=(7.5, -1.0)):
    """one hall per tab on the head (analog, DRV5055-class, E) at x +-7.5, 1.0 under the flat bottom, z on the magnet (searched
    for the best latched / half-in ratio under +-0.5 / +-1.0 head offset and +-0.3 gap): B_y in mT for the
    tab magnet latched (x 7), half in (bolt on the plate face, retract OVERLAP + 0.3) and squeezed (x 3)."""
    zm = g["Z_T"] + MAG_Z; my = MAG_DEPTH / 2
    st = {"latched": 0.0, "half_in": OVERLAP + 0.3, "squeezed": TRAVEL}
    out = {k: round(abs(mag_by(hall[0], hall[1], zm, MAG_X - v, my, zm)), 1) for k, v in st.items()}
    out["threshold_mT"] = round((out["latched"] + out["half_in"]) / 2, 1)
    out["hall_at"] = [hall[0], hall[1], round(zm, 2)]
    return out


# ======================================================================== hand calc (E loads, dimensions L / I)
G0 = 9.81
ALLOW = dict(functional=10.8, misuse=17.3)    # MPa at index 1.0 (hook latch convention: SF 4 / 2.5 on PETG, E)

def section_ledge(r):
    """ledge + back wall section (x-y plane) under / behind the pocket: (I about its centroid, c_max), numeric strips."""
    rf = r["RF"]; xs = np.linspace(rf, rf + r["PK_D"] + PK_WALL, 241); dx = xs[1] - xs[0]
    A = Sy = 0.0; cells = []
    for x in xs[:-1] + dx / 2:
        lead = Y_RB + max(0.0, (rf + CAP - x)) / math.tan(LEAD) if x < rf + CAP else Y_RB
        if x < rf + r["PK_D"]:
            top = r["PK_Y0"] - (x - rf) * math.tan(UNDER)
        else:
            top = r["PK_Y1"] + PK_WALL
        y0 = min(lead, top); h = max(0.0, top - y0)
        cells.append((y0, top)); A += h * dx; Sy += h * dx * (y0 + top) / 2
    yc = Sy / A; I = 0.0; cmax = 0.0
    for (y0, top) in cells:
        h = top - y0
        I += dx * h ** 3 / 12 + dx * h * ((y0 + top) / 2 - yc) ** 2
        cmax = max(cmax, abs(top - yc), abs(y0 - yc))
    return I, cmax, A

def hand_calc(B):
    g, r = B["g"], B["r"]
    m = g["SWAP_M"] / 1000.0                      # kg, swap assembly
    W1 = m * G0
    out = {"mass_swap_g": round(g["SWAP_M"], 1)}
    lever = (g["RF"] - g["X_OUT"]) + OVERLAP / 2                  # wall hole to the bearing centroid
    Zb = 2 * BOLT_HZ * BOLT_T ** 2 / 6                            # bolt section modulus (18 x 5)
    Il, cl, Al = section_ledge(r)
    span = 2 * r["PK_HZ"]
    cases = {}
    def add(name, Fy_out, cls, Fx=0.0, Fz=0.0):
        """Fy_out: pull-out force (N) on the bolts; Fx lateral."""
        Fb = max(Fy_out, 0.0) / 2.0
        s_bolt = Fb * lever / Zb
        s_ledge = (Fb * span / 12.0) * cl / Il                    # fixed-fixed beam over the pocket span, L section
        idx = max(s_bolt, s_ledge) / ALLOW[cls]
        cases[name] = dict(pull_N=round(Fy_out, 1), per_bolt_N=round(Fb, 1), bolt_MPa=round(s_bolt, 2),
                           ledge_MPa=round(s_ledge, 2), cls=cls, index=round(idx, 3), Fx_N=round(Fx, 1), Fz_N=round(Fz, 1))
    add("hover 1 g", W1, "functional")
    add("flight 4 g + 2 g lateral", 4 * W1, "functional", Fx=2 * W1)
    add("catch 6 g", 6 * W1, "functional")
    add("crash 35 g -y (drop on the belly)", 35 * W1, "misuse")
    add("crash 35 g corner (-y share 35/sqrt3)", 35 * W1 / math.sqrt(3), "misuse", Fx=35 * W1 / math.sqrt(3), Fz=35 * W1 / math.sqrt(3))
    add("crash 35 g +y (top hit, into the deck)", 0.0, "misuse")
    add("crash 35 g side x (into the plates)", 0.0, "misuse", Fx=35 * W1)
    add("swap pull 57 N, bolts in (arm pulls before the squeeze, fault)", 57.0, "functional")
    add("swap stall 74 N on one bolt tip (a bolt jammed on its lead-in, fault)", 2 * 74.0, "functional")
    out["cases"] = cases
    # overturning couple in a side hit: top rails bear on the plate at y ~ rail mid, CG at pack centre -> vertical on bolts
    y_rail = g["Y_TOP"] - 5.0; y_cg = float(g["SWAP_CG"][1])
    Fx = 35 * W1
    out["side_hit_couple_per_bolt_N"] = round(Fx * abs(y_rail - y_cg) / (2 * g["RF"]), 1)
    # grip pull while the XT60 unmates: 50 N + 1 g of the swap mass, through the two tabs (bolts in, squeezed)
    Fg = 57.0 + W1; Ft = Fg / 2                                   # 57 N: the arm's hold while the XT60 unmates (24 Oct)
    lever_t = REC_X - (TAB_OUT - TRAVEL - GP_D / 2)               # frame edge to the pocket centre, squeezed
    s_tab = Ft * lever_t / Zb
    out["grip_pull"] = dict(total_N=round(Fg, 1), per_tab_N=round(Ft, 1), bar_MPa=round(s_tab, 2),
                            index=round(s_tab / ALLOW["functional"], 3), lever_mm=round(lever_t, 2))
    # grip moment about x when the tab line is off the swap CG (Overseer: 30): each tab's rib bears on its pocket's y-walls
    # at the pocket's z ends (couple arm GP_W), on ~2 x GP_D mm2 per corner (E)
    off = abs(g["Z_T"] - float(g["SWAP_CG"][2]))
    gm = {}
    for ng in (1, 3):
        Mt = W1 * ng * off / 2.0                                     # N mm per tab
        Fc = Mt / GP_W
        sc = Fc / (2.0 * GP_D)
        gm[f"{ng}g"] = dict(moment_Nm=round(2 * Mt / 1000, 3), per_tab_corner_N=round(Fc, 1), MPa=round(sc, 2),
                            index=round(sc / ALLOW["functional"], 3))
    out["grip_moment"] = dict(offset_mm=round(off, 2), **gm)
    # floor plate under the pack, 35 g, simply supported strip over the cavity width
    p = (g["m"] - g["m_plug"]) / 1000 * 35 * G0 / (g["W"] * g["L"])
    Mf = p * (2 * g["X_IN"]) ** 2 / 8; s_f = 6 * Mf / g["FLOOR_T"] ** 2
    out["floor_35g"] = dict(MPa=round(s_f, 2), index=round(s_f / ALLOW["misuse"], 3))
    # long walls as beams from the bolt holes, 35 g (half per wall, cantilever each side of the tab line)
    w_wall = 35 * W1 / 2; Lh = max(g["Z_T"], g["L_C"] - g["Z_T"])
    Mw = (w_wall / 2) * Lh / 2; Iw = WALL * g["Y_TOP"] ** 3 / 12 * 0.6                          # 0.6: windows (E)
    s_w = Mw * (g["Y_TOP"] / 2) / Iw
    out["wall_35g"] = dict(MPa=round(s_w, 2), index=round(s_w / ALLOW["misuse"], 3))
    # slider inertia in a 35 g side hit vs the spring and the friction lock of a loaded bolt (corner case)
    m_sl = tm(slider(g, 0, 1)).volume / 1000 * RHO["PETG"] / 1000
    out["slider_inertia_35g_N"] = round(m_sl * 35 * G0, 2)
    out["bolt_friction_lock_corner_N"] = round(0.3 * cases["crash 35 g corner (-y share 35/sqrt3)"]["per_bolt_N"], 1)
    # centre rib (cartridge v6): the head's x load through the squeezed tab into the rib, fixed-fixed over the recess,
    # uniform over the tab (E): ends at the full section, midspan at the spring notch; the posts as cantilevers
    def rib_case(F, cls):
        L = 2 * REC_Z; w = F / L; t = 2 * RIB_HX
        s_end = (w * L ** 2 / 12) / (RIB_Y1 * t ** 2 / 6)
        s_mid = (w * L ** 2 / 24) / ((SPR_YC - SPR_R - 0.1) * t ** 2 / 6)
        s_post = (F / 2) / (RIB_POST * Y_FL1)               # the cross wall in shear (frame to frame, E)
        sm = max(s_end, s_mid, s_post)
        return dict(F_N=round(F, 1), end_MPa=round(s_end, 2), mid_MPa=round(s_mid, 2), post_MPa=round(s_post, 2), cls=cls,
                    index=round(sm / ALLOW[cls], 3))
    out["rib"] = {"swap across 43.5 N": rib_case(43.5, "functional"),
                  "carry 3 g across (swap mass)": rib_case(3 * W1, "functional"),
                  "misuse 6 g across": rib_case(6 * W1, "misuse")}
    # spring / release
    k_spr, free = SPRING_K, SPRING_FREE
    gap_rest = 2 * TAB_IN + 2 * SPR_DEPTH
    out["spring"] = dict(rate_N_mm=k_spr, free_mm=free, rest_N=round(k_spr * (free - gap_rest), 2),
                         squeezed_N=round(k_spr * (free - (gap_rest - 2 * TRAVEL)), 2), length_squeezed_mm=gap_rest - 2 * TRAVEL)
    fr = MU * W1 / 2 + W1 / 2 * math.tan(UNDER)
    out["release_force_per_finger_N"] = round(out["spring"]["squeezed_N"] + fr, 2)
    # side hit along x (35 g): the inboard slider's inertia against the spring + its share of the 1 g hang (wedge + friction)
    hold = out["spring"]["rest_N"] + fr
    out["side_hit_slider"] = dict(inertia_N=out["slider_inertia_35g_N"], hold_N=round(hold, 2),
                                  ratio=round(hold / out["slider_inertia_35g_N"], 2),
                                  no_hang_hold_N=out["spring"]["rest_N"],
                                  no_hang_ratio=round(out["spring"]["rest_N"] / out["slider_inertia_35g_N"], 2),
                                  one_bolt_corner_index=round(2 * cases["crash 35 g corner (-y share 35/sqrt3)"]["index"], 3),
                                  note="the outboard bolt is pushed further in, so at most one bolt moves for the pulse; with no "
                                       "hang (inverted, free fall) nothing pulls the pack out, and in a corner hit the -y share's "
                                       "friction lock holds it; one bolt alone carries the corner case at the index given")
    # side hit: rails bear on the side plate near the deck; plate root bending per mm of length, and the overturning
    # couple carried by the bolts (vertical) and the deck strips
    q = 35 * W1 / g["L_C"]; Mp = q * (r["Y_DK0"] - (g["Y_TOP"] - 5.0)); s_p = 6 * Mp / r["PLATE_T"] ** 2
    out["plate_side_hit_35g"] = dict(MPa=round(s_p, 2), index=round(s_p / ALLOW["misuse"], 3))
    Fc = out["side_hit_couple_per_bolt_N"]
    s_c = Fc * lever / Zb
    out["bolt_side_hit_couple"] = dict(MPa=round(s_c, 2), index=round(s_c / ALLOW["misuse"], 3))
    out["worst_index"] = max([c["index"] for c in cases.values()] + [out["grip_pull"]["index"], out["floor_35g"]["index"], out["wall_35g"]["index"],
                                                                   out["plate_side_hit_35g"]["index"], out["bolt_side_hit_couple"]["index"]]
                            + [c["index"] for c in out["rib"].values()])
    return out


# ======================================================================== joint reactions at the 4 mount bosses (rigid deck, E)
def reactions(B, hc):
    """per case: resultant (N, N mm) at the mount plane centre and per-boss forces (rigid plate, equal shear, axial from
    the moment), cartridge coords. Loads act at the swap CG (inertia) or at the cartridge bottom / XT60 (swap)."""
    g, r = B["g"], B["r"]
    W1 = g["SWAP_M"] / 1000 * G0
    cg = np.asarray(g["SWAP_CG"], float)
    pts = np.array([[bx, r["Y_M"], bz] for (bx, bz) in r["MOUNT"]])
    ctr = pts.mean(0)
    s3 = 1 / math.sqrt(3)
    cases = {
        "hover 1 g": (np.array([0, -1, 0]) * W1, cg),
        "flight 4 g up + 2 g lateral": (np.array([2, -4, 0]) * W1, cg),
        "catch 6 g": (np.array([0, -6, 0]) * W1, cg),
        "crash 35 g -y": (np.array([0, -35, 0]) * W1, cg),
        "crash 35 g +y": (np.array([0, 35, 0]) * W1, cg),
        "crash 35 g +x": (np.array([35, 0, 0]) * W1, cg),
        "crash 35 g +z": (np.array([0, 0, 35]) * W1, cg),
        "crash 35 g corner (+x -y +z)": (np.array([35 * s3, -35 * s3, 35 * s3]) * W1, cg),
        "swap insert 60 N": (np.array([0, 60.0, 0]), np.array([0, 0, g["Z_T"]])),
        "swap pull 50 N (XT60)": (np.array([0, -50.0, 0]), np.array([0, r["Y_XM"], g["Z_X"]])),
    }
    out = {}
    for nm, (F, at) in cases.items():
        M = np.cross(at - ctr, F)
        # boss axial (y) from F_y and the moments about x and z; shear (x, z) equal share + torsion about y
        d = pts - ctr
        Iz = float((d[:, 0] ** 2).sum()); Ix = float((d[:, 2] ** 2).sum())
        Fy = F[1] / 4 + (-M[2] * d[:, 0] / Iz if Iz else 0) + (M[0] * d[:, 2] / Ix if Ix else 0)
        Jt = float((d[:, 0] ** 2 + d[:, 2] ** 2).sum())
        Fx = F[0] / 4 + (-M[1] * d[:, 2] / Jt); Fz = F[2] / 4 + (M[1] * d[:, 0] / Jt)
        out[nm] = dict(F_N=np.round(F, 1).tolist(), M_Nmm=np.round(M, 0).tolist(),
                       boss_axial_N=np.round(Fy, 1).tolist(), boss_shear_N=np.round(np.hypot(Fx, Fz), 1).tolist(),
                       max_tension_N=round(float(-Fy.min()) if Fy.min() < 0 else 0.0, 1))
    return out, pts.tolist(), ctr.tolist()


# ======================================================================== collar inset search (Scout)
SCOUT_INSET = 11.5                # the search's result (search_inset() refreshes it at run time; 11.0 before the 26.55 bolts)
SKID_T = 3.0                      # Scout cartridge ends: clipped SKID_T deeper than the set, a TPU skid fills the band back to
                                  # the set's clip, so an edge from below meets the skid, not the cartridge (Scout edge probe)

def search_inset(log=print):
    """smallest clip inset (0.5 steps) that gives every receiver + cartridge point >= 7.0 inside the collar planes."""
    global SCOUT_INSET
    g = place_tab_line("scout"); T = T_dc(g)
    best = None
    for inset in np.arange(8.0, 20.01, 0.5):
        clip = collar_clip(g, inset)
        if RECV_V == 2:                    # the current receiver (v2 body + XT60 carrier + cap)
            r = rgeom2(g); cb, cc = xt60_carrier(r); recv_m = receiver_v2(r, clip); lid_m = U(cb, cc) ^ clip
        else:
            r = rgeom(g); recv_m = receiver(r, clip); lid_m = lid(r)
        P = np.vstack([tm(cartridge(g, clip), T).vertices, tm(recv_m, T).vertices,
                       tm(slider(g, 0, 1), T).vertices, tm(slider(g, 0, -1), T).vertices, tm(lid_m, T).vertices])
        mg = collar_margin(P)
        log(f"  inset {inset:.1f}: collar margin {mg:.2f}")
        if mg >= COLLAR_NEED:
            best = (float(inset), mg); break
    SCOUT_INSET = best[0]
    _CACHE.clear()
    return best


# ======================================================================== print files, system manifest
def _rot(axis, deg):
    a = math.radians(deg); c, s_ = math.cos(a), math.sin(a)
    R = {"x": [[1, 0, 0], [0, c, -s_], [0, s_, c]], "y": [[c, 0, s_], [0, 1, 0], [-s_, 0, c]], "z": [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]}[axis]
    T = np.eye(4); T[:3, :3] = R; return T

PIECES = {   # name: (getter, material, print pose (c -> print), count, version, bed face)
    "cartridge": (lambda B: B["cart"], "PETG", np.eye(4), 1, {"scout": "v7", "overseer": "v6"}, "closed end (z 0) down"),
    "cart_strip": (lambda B: B["cstrips"][0] if B.get("cstrips") else None, "TPU", _rot("x", 90), 2, "v1",
                   "foot down (one part, both sides)"),
    "rskid_nose": (lambda B: B["rskids"][0] if B.get("rskids") else None, "TPU", _rot("y", 180), 1, "v2",
                   "plate's outer face down, the bars on tree supports"),
    "rskid_tail": (lambda B: B["rskids"][1] if B.get("rskids") else None, "TPU", np.eye(4), 1, "v1",
                   "plate's outer face down, the chamfer and bars on tree supports"),
    "slider": (lambda B: slider(B["g"], 0, 1), "PETG", _rot("x", -90), 2, "v6", "bar top (y 11.5) down: largest face, bar layers along its length"),
    "receiver": (lambda B: B["recv"], "PETG", _rot("x", -90), 1, {"scout": "v3", "overseer": "v2"},
                 "mount plane (deck top) down: layers across the install axis"),
    "xt60_carrier": (lambda B: B["carrier"], "PETG", _rot("x", -90), 1, {"scout": "v2", "overseer": "v1"}, "ear tops down (pocket open to the bed)"),
    "receiver_lid": (lambda B: B["cap"], "PETG", _rot("x", -90), 1, {"scout": "v2", "overseer": "v1"}, "top face down (the XT60 carrier's cap, = v1 lid)"),
    "receiver_strip": (lambda B: U(*[box(*sorted((B["r"]["X_IN"] - 1.55, B["r"]["RF"] - 0.05)), B["r"]["Y_TOP"], B["r"]["Y_TOP"] + STRIP_T,
                                         B["r"]["Z_P0"] - 0.8, B["r"]["Z_P1"] - 1.2)]), "TPU", _rot("x", 90), 2, "v1", "flat"),
    "skid_closed": (lambda B: B["skids"][0] if B["skids"] else None, "TPU", _rot("x", -135), 1, {"scout": "v2", "overseer": "v1"}, "outer (chamfer) face down"),
    "skid_lead": (lambda B: B["skids"][1] if B["skids"] else None, "TPU", _rot("x", -45), 1, {"scout": "v2", "overseer": "v1"}, "outer (chamfer) face down"),
}

def save_prints(B, pack, log=print):
    """export every printed piece on its bed face to runs/battery_latch/v1/stl/, solid_check, print_orient numbers."""
    from solid_check import solid_check
    import print_orient as PO
    d = os.path.join(OUT, "stl"); os.makedirs(d, exist_ok=True)
    out = {}
    for nm, (get, mat, T, n, ver, face) in PIECES.items():
        m = get(B)
        if m is None: continue
        if isinstance(ver, dict): ver = ver[pack]
        t = tm(m, T)
        if T is None:                                     # largest flat face down (the bars' bond face)
            k = int(np.argmax(t.facets_area))
            t.apply_transform(trimesh.geometry.align_vectors(t.facets_normal[k], [0, 0, -1]))
        t.apply_translation([-t.bounds[:, 0].mean(), -t.bounds[:, 1].mean(), -t.bounds[0, 2]])
        fn = f"box_battery_{nm}_{pack}_{ver}.stl"
        path = os.path.join(d, fn); t.export(path)
        ok, info = solid_check(t)
        bed, over, run = PO.analyse(t, nm)
        out[nm] = dict(file=fn, material=mat, count=n, version=ver, bed_face=face, mass_g=round(t.volume / 1000 * RHO[mat], 2),
                       extents=np.round(t.extents, 1).tolist(), solid_check="PASS" if ok else "FAIL",
                       solid_info={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in (info or {}).items()},
                       bed_mm2=round(float(bed), 0), overhang_mm2=round(float(over), 0), island_run_mm=round(float(run), 1))
        log(f"  {fn}: solid_check {'PASS' if ok else 'FAIL'}, {out[nm]['mass_g']} g, bed {bed:.0f} mm2, overhang {over:.0f} mm2")
    return out

def write_manifest(B, pack):
    """nox/system_check.py manifest (drone frame stand-in). Scout: collar-band cage stand-in, 500 g drone (dimensions L);
    Overseer: 2200 g drone, a box cage stand-in 3.0 + 3.0 outside (its seat height and collar are the Overseer's pass)."""
    g, r = B["g"], B["r"]
    T = T_dc(g)
    d = os.path.join(OUT, "manifest_" + pack); os.makedirs(d, exist_ok=True)
    parts = []
    def add(nm, m, role, mat=None, mass=None, tag="M", swap=False, cg=None, mesh_T=T):
        fn = os.path.join(d, nm + ".stl"); tm(m, mesh_T).export(fn)
        p = dict(name=nm, mesh=os.path.relpath(fn, d), role=role, tag=tag)
        if mat: p["material"] = mat
        if mass is not None: p["mass_g"] = round(float(mass), 2)
        if swap: p["swap"] = True
        if cg is not None: p["cg"] = [round(float(v), 2) for v in cg]
        parts.append(p)
    add("pack", pack_body(g), "protected", mass=g["m"] - g["m_plug"], tag=g["m_tag"], swap=True)
    add("xt60_plug", xt60_female(g), "bought", mass=g["m_plug"], tag="E", swap=True)
    add("cartridge", B["cart"], "protected", mat="PETG", swap=True)
    for k, m in zip(("px", "nx"), B.get("cstrips", [])):
        add(f"cartridge_strip_{k}", m, "cage", mat="TPU", swap=True)
    for k, m in zip(("closed", "lead"), B.get("skids") or []):         # TPU end skids: shield the clipped cartridge ends
        add(f"cartridge_skid_{k}", m, "cage", mat="TPU", swap=True)
    for k, m in zip(RSKID_NAMES, B.get("rskids", [])):
        add(f"receiver_skid_{k}", m, "cage", mat="TPU")
    add("slider_px", slider(g, 0, 1), "protected", mat="PETG", swap=True)
    add("slider_nx", slider(g, 0, -1), "protected", mat="PETG", swap=True)
    add("receiver", B["recv"], "protected", mat="PETG")
    add("receiver_lid", B["lid"], "protected", mat="PETG")
    add("receiver_strips", B["strips"], "structure", mat="TPU")
    add("xt60e1m", B["male"], "bought", mass=6.0, tag="E")
    total = 500.0 if pack == "scout" else 2200.0            # dimensions L mass budgets
    # mass of what is drawn here
    drawn = (g["m"]) + sum(vol(m) / 1000 * RHO[mat] for m, mat in ((B["cart"], "PETG"), (slider(g, 0, 1), "PETG"), (slider(g, 0, -1), "PETG"),
                                                                 (B["recv"], "PETG"), (B["lid"], "PETG"), (B["strips"], "TPU"))) + 6.0
    yb = r["Y_M"] + SCOUT_Z0
    win = tm(box(-9.25, 9.25, r["Y_M"] - 1, r["Y_M"] + 40, g["Z_X"] - 5.55, g["Z_X"] + 5.55), T)    # partner window over the
    wb = win.bounds                                                                                     # XT60E1-M rear + leads
    stand = box(-40, 40, -40, 40, yb, yb + 30) - box(wb[0][0], wb[1][0], wb[0][1], wb[1][1], wb[0][2], wb[1][2])
    add("drone_rest_standin", stand, "structure", mass=total - drawn, tag="E", cg=[0, 0, yb + 15], mesh_T=np.eye(4))
    if pack == "scout":                                      # collar band cage stand-in: outer faces 1.0 inside the planes
        zb, zt_ = -47.5, 20.0
        pts = [[sx * (COLLAR_K + z - 1.0), sy * (COLLAR_K + z - 1.0), z] for z in (zb, zt_) for sx in (-1, 1) for sy in (-1, 1)]
        cage = hullp(pts) - hullp([[sx * (COLLAR_K + z - 4.0), sy * (COLLAR_K + z - 4.0), z] for z in (zb + 3.0, zt_ + 1) for sx in (-1, 1) for sy in (-1, 1)])
        for m in list(B.get("cstrips") or []) + list(B.get("skids") or []) + list(B.get("rskids") or []):
            lo, hi = tm(m, T).bounds                         # the TPU skids are the cage there: the stand-in band clears them by 0.5
            cage = cage - box(lo[0] - 0.5, hi[0] + 0.5, lo[1] - 0.5, hi[1] + 0.5, lo[2] - 0.5, hi[2] + 0.5)
    else:
        Ad = tm(U(B["cart"], B["recv"], B["lid"]), T); lo, hi = Ad.bounds
        cage = box(lo[0] - 6, hi[0] + 6, lo[1] - 6, hi[1] + 6, lo[2] - 3.0, hi[2] + 6) - box(lo[0] - 3, hi[0] + 3, lo[1] - 3, hi[1] + 3, lo[2], hi[2] + 3)
    add("cage_standin", cage, "cage", mass=0.0, tag="E", mesh_T=np.eye(4))
    man = dict(name=f"battery_latch_{pack}", parts=parts,
               openings=[dict(axis=[0, 0, -1], half_angle_deg=20.0, min_mm=0.0)],
               contact_pairs=[["cartridge", "receiver_strips", 5.0], ["pack", "cartridge", 1.0], ["xt60_plug", "cartridge", 5.0],
                              ["xt60e1m", "receiver", 5.0], ["xt60e1m", "receiver_lid", 5.0], ["receiver_strips", "receiver", 5.0],
                              ["drone_rest_standin", "receiver_lid", 1.0], ["drone_rest_standin", "receiver", 1.0]]
                             + [[p["name"], "cartridge", 5.0] for p in parts if p["name"].startswith("cartridge_s")],
               cg_ref=[0.0, 0.0, 0.0], thrust_axis=[0, 0, 1], cg_tol_mm=3.0, cage_margin_mm=3.0, fit_tol_mm3=0.5)
    fn = os.path.join(d, "manifest.json")
    with open(fn, "w") as f: json.dump(man, f, indent=1)
    return os.path.relpath(fn, ROOT)


# ======================================================================== Overseer pack bay (nox/overseer/overseer.assembly.json)
OV_ASM = os.path.join(ROOT, "nox", "overseer", "overseer.assembly.json")
OV_Z0 = -80.0                      # cartridge flat bottom at the layout's envelope bottom (drone z, layout v2)

def overseer_fit(x_tab=-30.0, lead_aft=False, log=print):
    """the Overseer set posed in the Overseer layout (its cartridge and pack boxes replaced): flat bottom z -80, tab line
    (= swap CG) at drone x x_tab, lead (XT60) end forward = the arm side; system_check fit / balance on the whole drone. Returns the summary."""
    import system_check as sc
    B = build("overseer"); g = B["g"]
    sgn = 1.0 if lead_aft else -1.0
    T = np.array([[0, 0, -sgn, x_tab + sgn * g["Z_T"]], [-sgn, 0, 0, 0], [0, 1, 0, OV_Z0], [0, 0, 0, 1.0]])
    man = json.load(open(OV_ASM))
    base = os.path.dirname(OV_ASM)
    keep = []
    for prt in man["parts"]:
        if prt["name"] in ("cartridge", "pack"): continue
        q = dict(prt)
        if "mesh" in q: q["mesh"] = os.path.join(ROOT, q["mesh"]) if not os.path.isabs(q["mesh"]) else q["mesh"]
        keep.append(q)
    def P(nm, m, role, mat=None, mass=None, swap=False, tag="M"):
        d = dict(name=nm, trimesh=tm(m, T), role=role, tag=tag, geom_tag="C")
        if mat: d["material"] = mat
        if mass is not None: d["mass_g"] = mass
        if swap: d["swap"] = True
        return d
    mine = [P("pack", pack_body(g), "bought", mass=g["m"] - g["m_plug"], swap=True, tag="E"),
            P("xt60_plug", xt60_female(g), "bought", mass=g["m_plug"], swap=True, tag="E"),
            P("cartridge", B["cart"], "protected", "PETG", swap=True), P("slider_px", slider(g, 0, 1), "protected", "PETG", swap=True),
            P("slider_nx", slider(g, 0, -1), "protected", "PETG", swap=True), P("receiver", B["recv"], "protected", "PETG"),
            P("receiver_lid", B["lid"], "protected", "PETG"), P("receiver_strips", B["strips"], "structure", "TPU"),
            P("xt60e1m", B["male"], "bought", mass=6.0, tag="E")]
    for d in mine:
        if d["role"] == "bought" and d["name"] == "pack": d["protect"] = True
    cp = [c for c in man.get("contact_pairs", []) if "cartridge" not in c and "pack" not in c]
    cp += [["cartridge", "receiver_strips", 5.0], ["pack", "cartridge", 1.0], ["xt60_plug", "cartridge", 5.0], ["xt60e1m", "receiver", 5.0],
           ["xt60e1m", "receiver_lid", 5.0], ["receiver_strips", "receiver", 5.0], ["receiver", "bottom plate", 1.0], ["receiver_lid", "bottom plate", 1.0]]
    m2 = dict(man); m2["parts"] = keep + mine; m2["contact_pairs"] = cp
    res = sc.run(m2, n_dirs=600)
    fit_pairs = [pp for pp in res["fit"]["pairs"] if any(k in (pp.get("a"), pp.get("b")) for k in [d["name"] for d in mine])] \
        if isinstance(res.get("fit", {}).get("pairs"), list) else res.get("fit")
    return res, fit_pairs, T


# ======================================================================== Scout v3 system (latch refit snapshot + this set)
SCOUT_SYS = os.path.join(ROOT, "runs", "battery_latch", "v1", "scout_v3_refit", "scout.assembly.json")
SCOUT_SWAP = {   # Scout v3 refit part name -> this set (drone pose via T_dc)
    "receiver v1 (latch)": "recv", "receiver lid (latch)": "lid", "receiver TPU strips (latch)": "strips",
    "XT60E1-M": "male", "cartridge v5 (latch)": "cart", "slider +y (latch)": "s_px", "slider -y (latch)": "s_nx",
    "pack 4S 1500": "pack", "pack XT60 plug": "plug"}

def scout_system(log=print):
    """system_check on the Scout v3 latch refit assembly (snapshot of branch claude/gracious-darwin-ogidvp 3595381: its
    cage with the 4th keel node, its mount on this set's 4 bosses) with the latch parts swapped for the current ones, plus
    the access paths the latch owes: cartridge and pack out along the install axis, the 4 receiver bolts, the XT60 rear."""
    import system_check as sc
    man = json.load(open(SCOUT_SYS)); base = os.path.dirname(SCOUT_SYS)
    B = build("scout"); g, r = B["g"], B["r"]; T = T_dc(g)
    mesh = dict(recv=B["recv"], lid=B["lid"], strips=B["strips"], male=B["male"], cart=B["cart"], s_px=slider(g, 0, 1),
                s_nx=slider(g, 0, -1), pack=pack_body(g), plug=xt60_female(g))
    for p in man["parts"]:
        if p["name"] in SCOUT_SWAP:
            p.pop("mesh", None); p["trimesh"] = tm(mesh[SCOUT_SWAP[p["name"]]], T); p["tag"] = "M"
            if p["name"] in ("receiver v1 (latch)", "receiver lid (latch)", "cartridge v5 (latch)", "slider +y (latch)",
                             "slider -y (latch)"):
                p.pop("mass_g", None); p["material"] = "PETG"
            elif p["name"] == "receiver TPU strips (latch)":
                p.pop("mass_g", None); p["material"] = "TPU"
        elif "mesh" in p:
            p["mesh"] = os.path.join(base, p["mesh"])
    for k, sk in zip(("closed", "lead"), B["skids"]):                  # the TPU end skids: shield the clipped ends
        man["parts"].append(dict(name=f"cartridge skid {k} (latch)", trimesh=tm(sk, T), role="cage", material="TPU", swap=True,
                                 tag="M"))
        man["contact_pairs"].append([f"cartridge skid {k} (latch)", "cartridge v5 (latch)", 5.0])
    for k, sk in zip(("px", "nx"), B.get("cstrips", [])):             # cartridge v6 bottom-edge strips (Scout ask)
        man["parts"].append(dict(name=f"cartridge strip {k} (latch)", trimesh=tm(sk, T), role="cage", material="TPU", swap=True,
                                 tag="M"))
        man["contact_pairs"].append([f"cartridge strip {k} (latch)", "cartridge v5 (latch)", 5.0])
    for k, sk in zip([n.replace("_", " ") for n in RSKID_NAMES], B.get("rskids", [])):
        man["parts"].append(dict(name=f"receiver skid {k} (latch)", trimesh=tm(sk, T), role="cage", material="TPU", tag="M"))
        man["contact_pairs"].append([f"receiver skid {k} (latch)", "receiver v1 (latch)", 5.0])
        man["contact_pairs"].append([f"receiver skid {k} (latch)", "receiver lid (latch)", 5.0])

    swap_move = ["pack 4S 1500", "pack XT60 plug", "slider +y (latch)", "slider -y (latch)", "cartridge v5 (latch)",
                 "cartridge skid closed (latch)", "cartridge skid lead (latch)", "cartridge strip px (latch)",
                 "cartridge strip nx (latch)"]
    acc = [dict(name="cartridge out (swap)", part="cartridge v5 (latch)", dir=[0, 0, -1], length_mm=80, tool="part",
                ignore=swap_move + ["receiver TPU strips (latch)", "XT60E1-M"]),
           dict(name="pack out with the cartridge", part="pack 4S 1500", dir=[0, 0, -1], length_mm=80, tool="part",
                ignore=swap_move + ["receiver TPU strips (latch)", "XT60E1-M"])]
    for k, (bx, bz) in enumerate(r["MOUNT"]):
        dp = (T @ np.array([bx, r["Y_M"], bz, 1.0]))[:3]                      # bolt line: from the mount plane down
        acc.append(dict(name=f"receiver bolt {k + 1} (M3 up from below)", part="receiver v1 (latch)", **{"from": dp.round(2).tolist()},
                        dir=[0, 0, -1], length_mm=80, tool=dict(type="cylinder", d=3.0)))
    xp = (T @ np.array([0.0, r["Y_M"], g["Z_X"], 1.0]))[:3]
    acc.append(dict(name="XT60 rear and leads (up through the mount window)", part="XT60E1-M", **{"from": xp.round(2).tolist()},
                    dir=[0, 0, 1], length_mm=8.9, tool=dict(type="box", w=18.5, h=11.1, w_axis=[0, 1, 0]),
                    ignore=["XT60E1-M", "receiver lid (latch)", "receiver v1 (latch)", "mount + nose legs (Scout)"]))
    man["access"] = acc
    for o in man["openings"]:                                        # the snapshot's cartridge-set cones (35) to ours
        if o.get("skip_probes"): o["half_angle_deg"] = PROBE_EXEMPT_DEG
    for n in ("receiver v1 (latch)", "receiver lid (latch)", "receiver TPU strips (latch)", "XT60E1-M"):   # the receiver round
        man["openings"].append(dict(part=n, axis=[0, 0, -1], half_angle_deg=PROBE_EXEMPT_DEG, min_mm=0.0, skip_probes=True))  # the swap
    res = sc.run(man)                                                # opening, exempt like the cartridge set (DEFAULT)
    mine = set(SCOUT_SWAP)
    clash = [(p_["a"], p_["b"], p_["overlap_mm3"]) for p_ in res["fit"]["pairs"] if not p_["pass_"] and ({p_["a"], p_["b"]} & mine)]
    cf = res["cage_first"]["per_part"]
    log("scout v3 system (refit): " + res["summary"])
    return dict(summary=res["summary"], pass_=res.get("pass_"), clashes=clash, balance=res.get("balance"),
                access=res.get("access"), probes={k: v for k, v in (res.get("probes") or {}).items() if k != "rows"},
                cage_first={k: cf[k]["min_margin_mm"] for k in cf if k in mine})


# ======================================================================== main
def solid(mesh):
    from solid_check import solid_check
    ok, info = solid_check(mesh)
    return bool(ok), {k: (v if isinstance(v, (int, float, str, bool, list)) else str(v)) for k, v in (info or {}).items()}

PRINT_POSE = {   # c -> print frame: largest flat face down
    "cartridge": np.diag([1.0, 1, 1, 1]),                                  # replaced: closed end (z 0) down = identity
}

def export(m, path, T=None):
    t = tm(m, T); t.export(path); return t

def main():
    os.makedirs(OUT, exist_ok=True)
    log_lines = []
    def log(s): print(s); log_lines.append(s)
    res = {}
    log("collar inset search (Scout, >= 7.0 inside the planes below z 20, +-5 yaw, +-2 tilt):")
    inset, mg = search_inset(log)
    res["scout_collar"] = dict(inset=inset, margin=round(mg, 2), need=COLLAR_NEED)
    for pack in ("scout", "overseer"):
        B = build(pack); g, r = B["g"], B["r"]
        log(f"== {pack}: z_T {g['Z_T']} (swap CG), swap {g['SWAP_M']:.1f} g, cartridge {g['L_C']:.1f} x {2*g['X_RAIL']:.1f} x {g['Y_TOP']:.1f}")
        R = {}
        R["geom"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if isinstance(v, (int, float, str))}
        R["rgeom"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in g and isinstance(v, (int, float))}
        R["mount"] = [[round(a, 2) for a in p] for p in r["MOUNT"]]
        R["static_fit_mm3"] = static_fit(B)
        R["squeeze"] = squeeze_sweep(B)
        R["gripped_pull_mm3"] = gripped_pull_sweep(B)
        rows, wc = insertion_sweep(B)
        R["insert_free"] = dict(cart_vs_recv_mm3=wc, stuck=[d for d, rr in rows if None in rr],
                               retract_profile=[(d, rr) for d, rr in rows if any(x for x in rr)])
        rows1, wc1 = insertion_sweep(B, side_off=0.3, dz=2.0, depth=20.0)
        R["insert_free_off0.3"] = dict(cart_vs_recv_mm3=wc1, stuck=[d for d, rr in rows1 if None in rr])
        R["bearing_drop"] = bearing_drop(B)
        R["grip"] = grip_checks(B)
        R["asym_squeeze"] = [asym_squeeze(B, 1.0), asym_squeeze(B, -1.0)]
        R["hall"] = hall_read(g)
        R["hall_sweep"] = hall_sweep(g)
        R["pitch_at_contact"] = pitch_at_contact(B)
        R["float_tilt"] = float_tilt()
        import battery_latch_topo as BLT
        fe_fn = os.path.join(OUT, "fe", f"fe_{pack}_receiver_v{RECV_V}.json")   # written by battery_latch_topo --swap-v2
        if os.path.exists(fe_fn) and os.path.getmtime(fe_fn) > os.path.getmtime(__file__):  # (+ the |u| fields), newer
            fe = json.load(open(fe_fn))                                                 # than this script: reused
        else:
            fe = BLT.fe_swap_v(pack, RECV_V, FE_H[pack], build=RECV_BUILD, log=lambda s_: None, fields=())
            fe.pop("_fields", None)
        R["fe"] = dict(fe, crash_worst=max(v for k, v in fe["rows"].items() if k.startswith("crash")),
                       swap_worst=max(v for k, v in fe["rows"].items() if k.startswith("swap")))
        log(f"  FE receiver v{RECV_V} (h {FE_H[pack]}, build {RECV_BUILD}): worst {fe['worst']} in {fe['worst_case']}; crash worst "
            f"{R['fe']['crash_worst']}; swap worst {R['fe']['swap_worst']}; deflection {fe['deflection_mm']}")
        R["first_mode"] = BLT.first_mode(pack, FE_H[pack], log=log)
        R["plug_vs_male_seat_mm3"] = ov(xt60_female(g), B["male"])
        hc = hand_calc(B); R["hand_calc"] = hc
        rx, pts, ctr = reactions(B, hc); R["reactions"] = rx; R["mount_pts"] = pts
        # masses and bounding volume
        ms = {}
        mlist = [("cartridge", B["cart"], "PETG"), ("slider", slider(g, 0, 1), "PETG"), ("receiver", B["recv"], "PETG"),
                 ("lid", B.get("cap", B["lid"]), "PETG"), ("strips", B["strips"], "TPU")]
        if B.get("carrier") is not None: mlist.append(("xt60_carrier", B["carrier"], "PETG"))
        for nm, m, mat in mlist:
            ms[nm] = round(vol(m) / 1000 * RHO[mat], 2)
        if B["skids"]: ms["skids"] = round(sum(vol(m) for m in B["skids"]) / 1000 * RHO["TPU"], 2)
        if B.get("cstrips"): ms["cart_strips"] = round(sum(vol(m) for m in B["cstrips"]) / 1000 * RHO["TPU"], 2)
        if B.get("rskids"): ms["recv_skids"] = round(sum(vol(m) for m in B["rskids"]) / 1000 * RHO["TPU"], 2)
        R["mass_g"] = ms
        gd = r.get("GUIDE", GUIDE)
        R["pitch_guidance"] = dict(guide=gd, end_gap=round(r["Z_E1"] - g["L_C"], 2), y_end0=round(r["Y_END0"], 2),
                                  note=("corner guide posts; pitch at the XT60's first contact" if gd else
                                        "no guide posts: the XT60E1-M float takes the pitch (float_tilt)"))
        A = U(B["cart"], B["recv"], B["lid"], slider(g, 0, 1), slider(g, 0, -1))
        bb = tm(A).bounds; ext = bb[1] - bb[0]
        R["bbox"] = dict(extents=np.round(ext, 1).tolist(), volume_L=round(float(np.prod(ext)) / 1e6, 4), largest=round(float(ext.max()), 1))
        if pack == "scout":
            T = T_dc(g)
            R["collar_margin"] = round(collar_margin(np.vstack([tm(A, T).vertices])), 2)
            R["keel"] = keel_report(B)
            R["head_seats"] = head_seats(B)
        res[pack] = R
        log(json.dumps({k: R[k] for k in ("static_fit_mm3", "squeeze", "gripped_pull_mm3", "bearing_drop", "mass_g", "bbox")}))
        log(f"  insert free: cart {R['insert_free']['cart_vs_recv_mm3']} mm3, stuck {R['insert_free']['stuck']}; off 0.3 stuck {R['insert_free_off0.3']['stuck']}")
        log(f"  hand calc worst index {hc['worst_index']}; grip pull {hc['grip_pull']}")
        R["prints"] = save_prints(B, pack, log)
        if B.get("rskids"):                       # the Scout skids: funnel margin and bolt-key distance each (C)
            R["skids"] = dict(funnel_clip_inset=skid_inset(g, r, B["recv"]), need_mm=SKID_FUNNEL,
                              receiver=skid_report(g, r, B["rskids"]),
                              cartridge_strips=skid_report(g, r, [B["cstrips"][0], B["cstrips"][1], B["cstrips"][1]]) )
            R["skids"]["cartridge_strips"] = {k: v for k, v in zip(("px", "nx"), list(R["skids"]["cartridge_strips"].values())[:2])}
            Rt = tm(B["recv"])                    # bonded area: end-piece faces within 0.05 of the receiver (M)
            for nm_, m_ in zip(RSKID_NAMES, B["rskids"]):
                t_ = tm(m_); _, d_, _ = trimesh.proximity.closest_point(Rt, t_.triangles_center)
                R["skids"]["receiver"][nm_]["bond_mm2"] = round(float(t_.area_faces[d_ < 0.05].sum()), 0)
            log(f"  skids: {R['skids']}")
        R["system_manifest"] = write_manifest(B, pack)
    res["scout_system"] = scout_system(log)
    # Overseer: posed in the Overseer assembly on main, pack on the thrust axis (Overseer DEFAULT A) and 30 behind (alt B)
    man = json.load(open(OV_ASM))
    for lab, dx in (("B", TAB_OFF["overseer"] - 30.0), ("A", TAB_OFF["overseer"])):   # B: pack 30 aft of the thrust axis (Spark)
        try:
            xt = float(man["cg_ref"][0]) + dx
            ores, fp, _ = overseer_fit(x_tab=xt)
            res["overseer"][f"overseer_assembly_{lab}"] = dict(
                x_tab=xt, summary=ores["summary"], balance=ores.get("balance"),
                pairs=[(p_["a"], p_["b"], p_["overlap_mm3"]) for p_ in fp if not p_["contact"] and p_["overlap_mm3"] > 0.5])
            log(f"overseer in overseer.assembly.json ({lab}): " + ores["summary"])
        except Exception as e:                                               # the Overseer file is the Overseer chat's
            res["overseer"][f"overseer_assembly_{lab}"] = dict(error=str(e))
    with open(os.path.join(OUT, "result.json"), "w") as f: json.dump(res, f, indent=1, default=float)
    with open(os.path.join(OUT, "log.txt"), "w") as f: f.write("\n".join(log_lines) + "\n")
    return res

if __name__ == "__main__":
    main()
