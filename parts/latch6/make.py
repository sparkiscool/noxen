"""Battery latch v6 (10 Nov 2026, Spark: "the most reliable way for the latch to spring and keep itself in place", and "once
unlatched and the battery is on the pincher, it needs to hold at all angles with minimum slip and bend").

Latch v5 (parts/latch5/make.py on the vendored v4 CAD) with the battery side reworked for those two asks. The receiver, the
XT60 carrier, the guide poles, the steel bolt pins, the 20 deg seat ramp and the rest-stop lugs are v5's, unchanged.

  1. Two steel compression springs in parallel between the tabs (v5: one), at z_T +- 5 in 3.5 deep pockets, under the pack.
     Together 2.2 x the 35 g inertia of one slider + pin at rest; one alone still returns the
     latch. 5 mm either side of the tab line they also square the slider in its channel (one spring on the centre line let it
     cock). The rib's one centre notch becomes two at +- 5, near the points of zero bending of a rib loaded at its middle.
  2. The grabber grips the CARTRIDGE, not the loose tabs. v5's grip pockets were in the sliders, which float 0.35 in their
     channels and bear 9 mm outboard of the grip: a pull on them sags the tab ~1.6 mm before anything is tight. v6: a wedge
     foot on each side of the centre rib (x 2..4.8, top 10 deg, under the tab line) and a hook nub on each blade tip that
     slides over it as the blade closes. 10 deg is self-locking (tan 0.18 < mu), so the squeeze draws the cartridge down onto
     four heel pads on the grabber head (on the cartridge's rim bands, x +-16.7, z_T +- 21.5) and a pull cannot open the blades.
     The tabs only get pushed (flat blade face on the tab legs); the nub passes through an arch in each tab.
  3. Travel to the rib 4.9 (v5 3.8): the blade stops on its wedge at 4.6 nominal, not on the rib, so the tab is drawn 3.3
     (what the bolts need) even with the wedge 0.2 mm early in y; 0.2 mm late and the tab reaches the rib first (0.07 mm of
     play at the wedge, no preload). The tab is 3.1 thick (6.9..10), the magnet moves to x 8.45 (the head's hall with it).
The design chat's 30 deg draw face is not used: at 30 deg a 57 N XT60 pull opens 8-10 N blades unless friction helps
(needs mu > 0.58); and its pads at +-30 are off the Scout cartridge's flat bottom (the collar clip cuts it past +-26).

python3 parts/latch6/make.py   -> parts/latch6_*/part.stl, parts/latch6/result.json, parts/latch6/view/*.stl, the checks printed
Coordinates as v5: cartridge c (x across, y up the install axis, 0 = flat bottom, z along the pack); page P = (c.x, z_T - c.z, c.y).
"""
import importlib.util, json, math, os, sys
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "parts", "latch5"))
_spec = importlib.util.spec_from_file_location("latch5_make", os.path.join(ROOT, "parts", "latch5", "make.py"))
v5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v5)          # patches the vendored module to v5
bl = v5.bl
from battery_latch_v4 import box, cyl, hullp, prism_xy, U, vol, rot_y180
sys.path.insert(0, ROOT)
from topo.material import MATERIALS, KNOCKDOWN, SAFETY

PIN_R, PIN_L, PIN_DOWN = v5.PIN_R, v5.PIN_L, v5.PIN_DOWN
LUG_Z0, LUG_Y1 = v5.LUG_Z0, v5.LUG_Y1
G0 = 9.81
PCTG = MATERIALS["PCTG"]
ALLOW = {k: PCTG.Xt * KNOCKDOWN / SAFETY[c] for k, c in (("functional", "flight"), ("misuse", "misuse"))}   # 9.78 / 15.64 MPa

# ------------------------------------------------------------------ v6 numbers (C unless tagged)
bl.TAB_IN = 6.9                        # tab inner face at rest (v5 5.8): 4.9 to the centre rib
bl.TRAVEL = bl.TAB_IN - bl.RIB_HX      # 4.9: the limit (tab on the rib)
R0 = 4.6                               # retraction with the blade tight on its wedge, nominal
NEED = bl.OVERLAP + 0.3                # 3.3: what the bolts need
bl.MAG_X = 8.45                        # magnet (v5 7.0): 0.55 walls in the 3.1 tab; the head's hall follows to x +-8.95
SPR_DZ = 5.0                           # the two springs at z_T +- 5
bl.SPR_DEPTH = 3.5                     # pocket depth from the tab inner face (v5 3.0): the 3.1 tab and 0.4 into the bolt bar;
                                       # sets the preload: 2.2 x the 35 g inertia of the Scout's slider + pin
RIB_NOTCH = 2.0                        # each rib notch half width (spring o 3.0 + 0.5 a side)
# spring: the Amazon listing in BOUGHT (304 stainless, wire 0.3, o 3.0, free 30). Its coil count is not listed: 28 active (E,
# 1.0 pitch) until one is on a kitchen scale; accepted if one reads 1.0..1.4 N at 20.8 long
SPR = dict(d=0.3, od=3.0, free=30.0, n_active=28, G=69000.0, uts=2000.0, material="304 stainless (E)")
BOUGHT = [
    dict(part="latch spring", spec="compression, 304 stainless, wire 0.3, OD 3.0, free 30", per_cartridge=2,
         listing="PATIKIL 0.3 x 3 x 30, 10 pcs", url="https://www.amazon.com/PATIKIL-Compression-Stainless-Mechanical-Assortment/dp/B0DKCRCBV2",
         check="rate on a kitchen scale: 1.0..1.4 N at 20.8 long (E 0.127 N/mm, 28 coils)"),
    dict(part="bolt pin", spec="dowel 3 x 16, 304 stainless", per_cartridge=2,
         listing="uxcell 3 x 16 dowel pin, 304 stainless, 50 pcs", url="https://www.amazon.com/uxcell-Stainless-Support-Fasten-Elements/dp/B07MH1Q65Z",
         check="alternative, hardened and ground DIN 6325, 100 pcs: https://www.amazon.com/Dowel-Hardened-Ground-Bright-Finish/dp/B07CVM9Z2N"),
    dict(part="guide pole", spec="dowel 3 x 20, stainless, DIN 7", per_receiver=2,
         listing="M3 x 20 dowel pins A2-70 DIN 7, 12 pcs", url="https://www.amazon.com/Pieces-M3x20mm-Metric-Dowel-Stainless/dp/B09X5ZC9B6",
         check="or 50 pcs: https://www.amazon.com/Dowel-Pin-Std-M3x20mm-PK50/dp/B007IA6NTC"),
    dict(part="tab magnet", spec="o 2 x 2 N52, axial", per_cartridge=2,
         listing="tElement 2 x 2 N52 cylinder magnets, 500 pcs", url="https://us.amazon.com/Cylinder-Magnets-2x2mm-Very-Small/dp/B08YPBKNTS", check=""),
    dict(part="heel pad facing (grabber head)", spec="TPU 95A, 2.0 thick, 3 x 7, four per head", listing="printed from the TPU 95A on hand", url="", check=""),
]
ALPHA = math.radians(10.0)             # wedge: foot top and nub underside, from the cartridge's bottom plane
FOOT_X1 = 4.8                          # foot: x RIB_HX..4.8 each side of the rib
FOOT_Y = 2.0                           # foot thickness at its outer end (2.49 at the rib)
FOOT_HZ = 4.5                          # foot z_T +- 4.5
NUB_L, NUB_HZ, NUB_Y1 = 3.0, 3.0, 5.0  # nub: 3.0 inboard of the blade face, 6 wide, top y 5.0
ARCH_HZ, ARCH_Y = 4.9, 5.6             # arch through each tab for the foot and the nub (nub +-1.9 of z capture)
BLADE_T, BLADE_HZ, BLADE_Y1, BLADE_OPEN = 3.0, 6.5, 6.0, 11.0    # blades as heads v7: 3 x 13, open inner faces +-11
PAD_X, PAD_Z, PAD_T = (15.2, 18.2), (18.0, 25.0), 2.0           # heel pads: on the rim bands, TPU 95A 2.0 facing
MU = dict(wedge=0.3, wedge_low=0.2, pad_tpu=0.6, pad_pctg=0.3, slide=0.3)   # E
K_BLADE = 1.0                          # N/mm, the head's blade springs (heads v7 concept, E)
RHO = {"PCTG": 1.23, "steel": 7.85, "TPU": 1.21}


def spring_rate():
    D = SPR["od"] - SPR["d"]
    return SPR["G"] * SPR["d"] ** 4 / (8 * D ** 3 * SPR["n_active"])


def spring_state(retract):
    """one spring at a retraction of both tabs: length, force, wire stress (Wahl)."""
    k = spring_rate(); D = SPR["od"] - SPR["d"]; C = D / SPR["d"]
    L = 2 * (bl.TAB_IN + bl.SPR_DEPTH) - 2 * retract
    F = k * (SPR["free"] - L)
    tau = 8 * F * D / (math.pi * SPR["d"] ** 3) * ((4 * C - 1) / (4 * C - 4) + 0.615 / C)
    return dict(length=round(L, 2), N=round(F, 3), MPa=round(tau, 0), of_uts=round(tau / SPR["uts"], 3))


def foot_top(x):
    """y of the wedge plane at |x| (the foot's top; the seated nub's underside)."""
    return FOOT_Y + (FOOT_X1 - x) * math.tan(ALPHA)


# ------------------------------------------------------------------ slider v8 (one part, both sides)
def slider(g, retract=0.0, side=1):
    zt, tip = g["Z_T"], g["X_TIP"]
    Y0, Y1 = bl.BOLT_Y0, bl.BOLT_Y1
    tab = box(bl.TAB_IN, bl.TAB_OUT, 0.0, Y1, zt - bl.TAB_HZ, zt + bl.TAB_HZ)
    bar = box(bl.TAB_IN, tip, Y0, Y1, zt - bl.BOLT_HZ, zt + bl.BOLT_HZ)
    cham = prism_xy([(tip - bl.TIP_CH, Y1 + 0.01), (tip + 0.01, Y1 + 0.01), (tip + 0.01, Y1 - bl.TIP_CH)],
                    zt - bl.BOLT_HZ - 1, zt + bl.BOLT_HZ + 1)
    x_stop = min(g["X_IN"] - 2.0, bl.REC_X)      # the rim band's inner face (Scout 15.0) or the recess wall (Overseer 15.5)
    lugs = [box(bl.TAB_OUT - 0.01, x_stop, 0.0, LUG_Y1, *sorted((zt + sz * LUG_Z0, zt + sz * bl.TAB_HZ))) for sz in (-1, 1)]
    s = U(tab, bar, *lugs) - cham
    xp, yp = v5.pin_axis(g)
    s = s - cyl("z", xp, yp, PIN_R, zt - bl.BOLT_HZ - 0.01, zt - bl.BOLT_HZ + PIN_L + 0.5, 32)
    s = s - prism_xy([(bl.TAB_OUT - bl.TAB_CH, -0.01), (bl.TAB_OUT + 0.01, -0.01), (bl.TAB_OUT + 0.01, bl.TAB_CH)],
                     zt - bl.TAB_HZ - 1, zt + bl.TAB_HZ + 1)
    # the arch (v6): the foot and the blade's nub pass through the tab; 1.0 x 45 deg lead-in on its bottom edges
    s = s - box(bl.TAB_IN - 0.01, bl.TAB_OUT + 0.01, -0.01, ARCH_Y, zt - ARCH_HZ, zt + ARCH_HZ)
    for sz in (-1, 1):
        s = s - hullp([[x, y, zt + sz * z] for x in (bl.TAB_IN - 0.01, bl.TAB_OUT + 0.01)
                       for (y, z) in ((-0.01, ARCH_HZ - 0.01), (-0.01, ARCH_HZ + 1.0), (1.0, ARCH_HZ - 0.01))])
    # two spring pockets, open to the top (the pack 0.5 above keeps the springs in)
    for sz in (-1, 1):
        zc = zt + sz * SPR_DZ
        s = s - cyl("x", bl.SPR_YC, zc, bl.SPR_R, bl.TAB_IN - 0.01, bl.TAB_IN + bl.SPR_DEPTH, 32) \
              - box(bl.TAB_IN - 0.01, bl.TAB_IN + bl.SPR_DEPTH, bl.SPR_YC, Y1 + 1, zc - bl.SPR_R, zc + bl.SPR_R)
    s = s - cyl("y", bl.MAG_X, zt + bl.MAG_Z, bl.MAG_D / 2, -0.01, bl.MAG_DEPTH, 24)
    s = s.translate([-retract, 0, 0])
    return s if side > 0 else rot_y180(s, zt)


# ------------------------------------------------------------------ centre rib v7: two spring notches, two wedge feet
def centre_rib(g):
    zt = g["Z_T"]
    rib = box(-bl.RIB_HX, bl.RIB_HX, 0.0, bl.RIB_Y1, zt - bl.REC_Z - 0.01, zt + bl.REC_Z + 0.01)
    for sz in (-1, 1):
        rib = rib - box(-bl.RIB_HX - 1, bl.RIB_HX + 1, bl.SPR_YC - bl.SPR_R - 0.1, bl.RIB_Y1 + 1,
                        zt + sz * SPR_DZ - RIB_NOTCH, zt + sz * SPR_DZ + RIB_NOTCH)
    parts = [rib]
    for s in (-1, 1):
        parts.append(hullp([[s * x, y, zt + z] for z in (-FOOT_HZ, FOOT_HZ)
                            for (x, y) in ((bl.RIB_HX - 0.5, 0.0), (FOOT_X1, 0.0), (FOOT_X1, FOOT_Y), (bl.RIB_HX - 0.5, foot_top(bl.RIB_HX - 0.5)))]))
    for sg in (-1, 1):
        z0, z1 = sorted((zt + sg * bl.REC_Z, zt + sg * (bl.REC_Z + bl.RIB_POST)))
        parts.append(box(-g["X_OUT"], g["X_OUT"], 0.0, bl.Y_FL1 - 0.01, z0, z1))
    return U(*parts)


def cartridge(g, clip=None):
    """v5's cartridge; where the rim bands sit outboard of the heel pads (Overseer) a land under each pad pair."""
    t = v5.cartridge(g, None)
    zt = g["Z_T"]
    if g["X_IN"] - 2.0 > PAD_X[0]:
        for sx in (-1, 1):
            for sz in (-1, 1):
                z0, z1 = sorted((zt + sz * (bl.REC_Z + bl.RIB_POST - 0.01), zt + sz * (PAD_Z[1] + 1.0)))
                t = t + box(*sorted((sx * (PAD_X[0] - 1.2), sx * (g["X_IN"] - 2.0 + 0.01))), 0.0, 4.0, z0, z1)
    return t if clip is None else t ^ clip


bl.slider, bl.centre_rib, bl.cartridge = slider, centre_rib, cartridge
v5.slider = slider                      # v5.cartridge / v5.checks pieces that call slider by its module name


def bolt_pin(g, retract=0.0, side=1):
    return v5.bolt_pin(g, retract, side)


# ------------------------------------------------------------------ grabber interface: blade tips, nubs, heel pads
def blade(g, face, side=1, dy=0.0, back=0.0):
    """one blade with its hook nub, inner face at |x| = face. back: the nub underside lifted this much (collision checks)."""
    zt = g["Z_T"]
    shank = box(face, face + BLADE_T, -20.0, BLADE_Y1, zt - BLADE_HZ, zt + BLADE_HZ)
    xb0 = bl.TAB_OUT - R0                                    # the face where the nub is tight on the foot
    under = lambda x: foot_top(x - face + xb0) + back        # nub underside in this blade's own position
    nub = hullp([[x, y, zt + z] for z in (-NUB_HZ, NUB_HZ)
                 for (x, y) in ((face - NUB_L, under(face - NUB_L)), (face + 0.01, under(face + 0.01)), (face + 0.01, NUB_Y1), (face - NUB_L, NUB_Y1))])
    b = (shank + nub).translate([0, dy, 0])
    return b if side > 0 else b.mirror([1, 0, 0])


def blades(g, face, **kw):
    return [blade(g, face, 1, **kw), blade(g, face, -1, **kw)]


def heel(g):
    """the grabber head's heel as the latch needs it: four TPU-faced pads under the rim bands, on a deck that clears the
    blades. Returns (deck (PCTG, the head's to shape), [pads (TPU 95A)])."""
    zt = g["Z_T"]
    deck = box(-PAD_X[1], PAD_X[1], -6.0, -PAD_T, zt - PAD_Z[1], zt + PAD_Z[1]) \
        - box(-15.0, 15.0, -7.0, 0.0, zt - 12.0, zt + 12.0)                 # the blades' window
    pads = [box(*sorted((sx * PAD_X[0], sx * PAD_X[1])), -PAD_T, 0.0, *sorted((zt + sz * PAD_Z[0], zt + sz * PAD_Z[1])))
            for sx in (1, -1) for sz in (1, -1)]
    return deck, pads


def spring_mesh(g, sz, turns=None, wire=None, od=None):
    """one spring at its rest length on its pocket axis (a pose scales it in x)."""
    wire = wire or SPR["d"]; od = od or SPR["od"]; turns = turns or SPR["n_active"] + 2
    L = 2 * (bl.TAB_IN + bl.SPR_DEPTH)
    rm = (od - wire) / 2
    n = turns * 12
    pts = np.array([[-L / 2 + L * i / n, bl.SPR_YC + rm * math.cos(2 * math.pi * turns * i / n),
                     g["Z_T"] + sz * SPR_DZ + rm * math.sin(2 * math.pi * turns * i / n)] for i in range(n + 1)])
    segs = [trimesh.creation.cylinder(radius=wire / 2, segment=[a, b], sections=6) for a, b in zip(pts[:-1], pts[1:])]
    return trimesh.util.concatenate(segs), L


# ------------------------------------------------------------------ build
def build(pack="scout"):
    B = bl.build(pack)
    g, r = B["g"], B["r"]
    B["poles"] = v5.poles(r)
    B["female"] = bl.xt60_female(g)
    return B


def receiver_all(B):
    return U(B["recv"], B["carrier"], B["cap"], B["strips"], B["male"], *B["poles"], *B.get("rskids", []))


def cart_all(B):
    return U(B["cart"], B["female"], *B.get("cstrips", []))


def moving(g, rt, s):
    return slider(g, rt, s) + bolt_pin(g, rt, s)


# ------------------------------------------------------------------ the hold on the pincher (hand model, E)
def hold(B, Fb, mu_pad, n_g=1.0, mu_w=None, ndir=26):
    """the squeezed blades (Fb N each) on the wedge feet, the cartridge drawn onto the four heel pads; gravity x n_g in every
    direction of a sphere grid. Margins > 1 hold with no slip and every pad in contact."""
    g = B["g"]; mu_w = MU["wedge"] if mu_w is None else mu_w
    W = g["SWAP_M"] / 1000 * G0
    cg = np.asarray(g["SWAP_CG"], float); ycg, dz = float(cg[1]), float(cg[2] - g["Z_T"])
    Fs = 2 * spring_state(R0)["N"]                                  # both springs, on each blade
    Fw = Fb - Fs - MU["slide"] * 0.05                               # x force left for the wedge, per blade
    if Fw <= 0:
        return dict(Fb=Fb, releases=False, note="the blade cannot push the tab in against the springs")
    ratio = (math.cos(ALPHA) - mu_w * math.sin(ALPHA)) / (math.sin(ALPHA) + mu_w * math.cos(ALPHA))
    D = Fw * ratio; P0 = 2 * D                                      # draw per blade, preload on the pads
    px = sum(PAD_X) / 2; pz = sum(PAD_Z) / 2
    dirs = []
    for a in np.linspace(-90, 90, 7):
        for b in (np.arange(0, 360, 30) if abs(a) < 89 else [0]):
            ca = math.cos(math.radians(a)); dirs.append((ca * math.cos(math.radians(b)), math.sin(math.radians(a)), ca * math.sin(math.radians(b))))
    worst = dict(lift=(1e9, None), slip=(1e9, None), pull=(1e9, None)); slip_mm = 0.0; tilt = 0.0
    kpad = MATERIALS["TPU95A"].E * (PAD_X[1] - PAD_X[0]) * (PAD_Z[1] - PAD_Z[0]) / PAD_T if mu_pad > 0.45 else 1e5
    for d in dirs:
        F = n_g * W * np.array(d)
        Mx = ycg * F[2] - dz * F[1]; My = dz * F[0]; Mz = -ycg * F[0]
        N = P0 - F[1]                                               # pad load: a pull (+y, away from the head) unloads them
        # the pull the nubs hold with no friction at all on the wedge, and whether it is self-locking at mu_w
        pull_cap = 2 * Fw / math.tan(ALPHA)
        if F[1] > 0: worst["pull"] = min(worst["pull"], (pull_cap / F[1], d))
        if N <= 0:
            worst["lift"] = min(worst["lift"], (0.0, d)); continue
        lift = 1.0 / (abs(Mx) / (N * pz) + abs(Mz) / (N * px) + 1e-12)
        Tx = mu_pad * N; Tz = mu_pad * N + mu_w * P0
        r_yaw = math.hypot(px, pz)
        slip = 1.0 / (math.hypot(F[0] / Tx, F[2] / Tz) + abs(My) / (mu_pad * N * r_yaw) + 1e-12)
        worst["lift"] = min(worst["lift"], (lift, d)); worst["slip"] = min(worst["slip"], (slip, d))
        if abs(F[0]) > Tx: slip_mm = max(slip_mm, (abs(F[0]) - Tx) / (2 * K_BLADE))
        tilt = max(tilt, math.hypot(Mx / (4 * kpad * pz ** 2), Mz / (4 * kpad * px ** 2)))
    rd = lambda t: None if t[1] is None else [round(float(v), 2) for v in t[1]]
    return dict(Fb=Fb, n_g=n_g, releases=True, wedge_x_N=round(Fw, 2), draw_per_blade_N=round(D, 2), pad_preload_N=round(P0, 2),
                margin_pads_stay_down=round(worst["lift"][0], 2), worst_dir_lift=rd(worst["lift"]),
                margin_no_slip=round(worst["slip"][0], 2), worst_dir_slip=rd(worst["slip"]),
                margin_pull_frictionless=round(worst["pull"][0], 2),
                slip_x_mm=round(slip_mm, 2), pack_top_mm=round(tilt * (bl.Y_FL1 + g["H"]), 3))


def six_directions(B, Fb, mu_pad):
    """what carries each direction, and the load it holds before anything moves (N or N mm)."""
    g = B["g"]; W = g["SWAP_M"] / 1000 * G0
    h = hold(B, Fb, mu_pad)
    if not h["releases"]: return h
    P0, Fw = h["pad_preload_N"], h["wedge_x_N"]
    px = sum(PAD_X) / 2; pz = sum(PAD_Z) / 2
    mu_w = MU["wedge"]
    rows = {
        "+y (away from the head)": dict(path="nub on the wedge foot -> rib -> cross walls; positive, self-locking for mu > %.2f" % math.tan(ALPHA),
                                        holds_N=round(2 * Fw / math.tan(ALPHA), 1), holds_note="with no wedge friction at all; no limit at mu 0.3",
                                        preload_N=P0, slip_mm=0.0),
        "-y (onto the head)": dict(path="rim bands on the four heel pads; positive", holds_N=None, slip_mm=0.0),
        "+-x (across)": dict(path="pad friction, then the head's blade springs (each blade is sprung on its own: not positive)",
                             holds_N=round(mu_pad * P0, 1), slip_mm_per_N_beyond=round(1 / (2 * K_BLADE), 2)),
        "+-z (along the pack)": dict(path="pad + wedge friction; positive stop: the nub on the tab arch", holds_N=round((mu_pad + mu_w) * P0, 1),
                                     stop_mm=round(ARCH_HZ - NUB_HZ + (bl.REC_Z - bl.TAB_HZ), 1)),
        "pitch about x (pack long axis vertical)": dict(path="pad pairs at z_T +- %.1f, preloaded" % pz, holds_Nmm=round(P0 * pz, 0),
                                                         load_1g_Nmm=round(W * float(g["SWAP_CG"][1]), 0)),
        "roll about z": dict(path="pad pairs at x +- %.1f, preloaded" % px, holds_Nmm=round(P0 * px, 0), load_1g_Nmm=round(W * float(g["SWAP_CG"][1]), 0)),
        "yaw about y": dict(path="pad friction at r %.1f" % math.hypot(px, pz), holds_Nmm=round(mu_pad * P0 * math.hypot(px, pz), 0),
                            load_1g_Nmm=round(W * abs(float(g["SWAP_CG"][2]) - g["Z_T"]), 0)),
    }
    for k, v in rows.items():
        if v.get("holds_N") and "x" in k or "z (" in k: v["margin_1g"] = round(v["holds_N"] / W, 2)
        if v.get("holds_Nmm") is not None: v["margin_1g"] = round(v["holds_Nmm"] / max(v["load_1g_Nmm"], 1e-9), 2) if v["load_1g_Nmm"] > 0.5 else None
    return rows


def bending(B, Fb=10.0, pull=59.1):
    """blade, nub, foot, rib, tab under the squeeze and under the XT60 pull (57 N + 1 g), hand model (E)."""
    g = B["g"]; E = PCTG.E
    per = pull / 2
    out = {}
    L = 8.0                                                         # blade free length above its guide boss (heads v7: boss top y -5, E)
    I = 2 * BLADE_HZ * BLADE_T ** 3 / 12
    out["blade_squeeze"] = dict(N=Fb, tip_mm=round(Fb * L ** 3 / (3 * E * I), 3), MPa=round(Fb * L / (I / (BLADE_T / 2)), 2))
    arm = NUB_L / 2 + BLADE_T / 2                                   # the pull's lever on the blade's axis
    M = per * arm
    out["blade_pull"] = dict(N=per, opens_mm=round(M * L ** 2 / (2 * E * I), 3), nub_drops_mm=round(M * L / (E * I) * arm, 3),
                             MPa=round(M / (I / (BLADE_T / 2)), 2), index=round(M / (I / (BLADE_T / 2)) / ALLOW["functional"], 3))
    xb0 = bl.TAB_OUT - R0
    h_root = NUB_Y1 - foot_top(FOOT_X1 + 0.0 - (xb0 - xb0)) + (xb0 - FOOT_X1) * math.tan(ALPHA)      # nub thickness at the blade face
    Zn = 2 * NUB_HZ * h_root ** 2 / 6; ln = xb0 - (xb0 - NUB_L + FOOT_X1) / 2                        # face to the middle of the contact
    out["nub_pull"] = dict(N=per, lever_mm=round(ln, 2), root_mm=round(h_root, 2), MPa=round(per * ln / Zn, 2), index=round(per * ln / Zn / ALLOW["functional"], 3),
                           contact_mm2=round((FOOT_X1 - (xb0 - NUB_L)) * 2 * NUB_HZ, 1),
                           bearing_MPa=round(per / ((FOOT_X1 - (xb0 - NUB_L)) * 2 * NUB_HZ), 2))
    hf = foot_top(bl.RIB_HX); lf = (xb0 - NUB_L + FOOT_X1) / 2 - bl.RIB_HX
    Zf = 2 * FOOT_HZ * hf ** 2 / 6
    out["foot_pull"] = dict(N=per, lever_mm=round(lf, 2), root_mm=round(hf, 2), MPa=round(per * lf / Zf, 2), index=round(per * lf / Zf / ALLOW["functional"], 3),
                            tip_mm=round(per * lf ** 3 / (3 * E * (2 * FOOT_HZ * hf ** 3 / 12)), 4))
    # rib: fixed-fixed over the recess, loaded at its middle through the feet. y pull (bending about x) and the head's x load
    Lr = 2 * bl.REC_Z; t = 2 * bl.RIB_HX; hy = bl.RIB_Y1; hn = bl.SPR_YC - bl.SPR_R - 0.1
    def rib_case(Fy, Fx, cls):
        Me = lambda F: F * Lr / 8                                   # ends and middle, point load at mid span
        Mn = lambda F: abs(F / 2 * (Lr / 4 - SPR_DZ))               # at the notches (z_T +- 5; zero at L/4 = 5.75)
        s_end = Me(Fy) / (t * hy ** 2 / 6) + Me(Fx) / (hy * t ** 2 / 6)
        s_not = Mn(Fy) / (t * hn ** 2 / 6) + Mn(Fx) / (hn * t ** 2 / 6)
        s_post = math.hypot(Fx, Fy) / 2 / (bl.RIB_POST * bl.Y_FL1)
        sm = max(s_end, s_not, s_post)
        mid = Fy * Lr ** 3 / (192 * E * t * hy ** 3 / 12), Fx * Lr ** 3 / (192 * E * hy * t ** 3 / 12)
        return dict(Fy_N=round(Fy, 1), Fx_N=round(Fx, 1), end_mid_MPa=round(s_end, 2), notch_MPa=round(s_not, 2), post_MPa=round(s_post, 2),
                    mid_y_mm=round(mid[0], 4), mid_x_mm=round(mid[1], 4), cls=cls, index=round(sm / ALLOW[cls], 3))
    W = g["SWAP_M"] / 1000 * G0
    out["rib"] = {"XT60 pull 57 N + 1 g": rib_case(pull, 0.0, "functional"),
                  "swap across 43.5 N": rib_case(0.0, 43.5, "functional"),
                  "pull + across together": rib_case(pull, 43.5, "functional"),
                  "carry 3 g across": rib_case(0.0, 3 * W, "functional"),
                  "v5 one centre notch, across 43.5 N (for comparison)": dict(
                      notch_MPa=round((43.5 * Lr / 8) / (hn * t ** 2 / 6), 2), index=round((43.5 * Lr / 8) / (hn * t ** 2 / 6) / ALLOW["functional"], 3))}
    out["tab"] = dict(note="pushed only (flat blade face on the tab legs, %.1f mm2): no pull, no bending; v5's tab bar carried the pull on a %.1f lever "
                           "through 0.35 of channel play" % (2 * (BLADE_HZ - ARCH_HZ) * BLADE_Y1 + 2 * ARCH_HZ * (BLADE_Y1 - ARCH_Y), bl.REC_X - 1 - 5.2),
                      v5_sag_at_grip_mm=round(0.35 + (bl.REC_X - 1 - 5.2) * 0.7 / 5.0, 2))
    out["worst_index"] = max([out["blade_pull"]["index"], out["nub_pull"]["index"], out["foot_pull"]["index"]] +
                             [c["index"] for c in out["rib"].values() if "cls" in c])
    return out


def spring_report(B):
    g = B["g"]
    m_sl = abs(bl.tm(slider(g, 0, 1)).volume) / 1000 * RHO["PCTG"]
    m_pin = abs(bl.tm(bolt_pin(g, 0, 1)).volume) / 1000 * RHO["steel"]
    m = m_sl + m_pin + bl.MAG_G
    inertia = m / 1000 * 35 * G0
    rest, sq, lim = spring_state(0.0), spring_state(R0), spring_state(bl.TRAVEL)
    solid = (SPR["n_active"] + 2) * SPR["d"]
    out = dict(count=2, each=dict(SPR, rate_N_mm=round(spring_rate(), 4), solid_mm=round(solid, 1)),
               rest=rest, squeezed=sq, at_rib_limit=lim, solid_margin_mm=round(lim["length"] - solid, 2),
               slider_pin_magnet_g=round(m, 2), inertia_35g_N=round(inertia, 2),
               preload_pair_N=round(2 * rest["N"], 2), preload_over_inertia=round(2 * rest["N"] / inertia, 2),
               one_spring_left_over_inertia=round(rest["N"] / inertia, 2),
               squeezed_pair_N=round(2 * sq["N"], 2), stress_of_uts_at_limit=lim["of_uts"],
               v5=dict(count=1, rest_N=0.88, over_inertia=round(0.88 / inertia, 2), squeezed_N=2.4))
    # the alternative: one spring per slider, seated on the rib (each slider on its own)
    k = 0.2; free = 2 * 0 + 21.2; Lr = bl.TAB_IN - bl.RIB_HX + 10.0
    out["alternative_one_per_slider"] = dict(
        each="o 3 x 0.3, free 21.2, k 0.2 (E), tab + bar pocket 10 deep, seated on the rib face",
        rest_N=round(k * (free - Lr), 2), squeezed_N=round(k * (free - (Lr - R0)), 2),
        over_inertia=round(k * (free - Lr) / inertia, 2), one_spring_fails="that bolt has NO return force: it can sit retracted (one bolt holds the pack)",
        why_not="a broken or lost spring leaves one bolt dead instead of both at half force; each spring sits on the slider's centre line "
                "(no squaring couple); the 4.0 rib carries a standing preload on each face (PCTG creep) and its seats are 1.0 deep at most")
    out["chosen"] = "two in parallel between the tabs: either one failing leaves both bolts sprung at half force (%.2f x the 35 g inertia), " \
                    "no standing load on the rib, a squaring couple on each slider" % (rest["N"] / inertia)
    return out


def checks(B):
    g, r = B["g"], B["r"]
    R = receiver_all(B); C = cart_all(B)
    res = {}
    parts = dict(cart=B["cart"], female=B["female"], sl_p=slider(g, 0, 1), sl_n=slider(g, 0, -1),
                 pin_p=bolt_pin(g, 0, 1), pin_n=bolt_pin(g, 0, -1), pack=bl.pack_body(g), recv=B["recv"],
                 carrier=B["carrier"], cap=B["cap"], strips=B["strips"], male=B["male"], pole_p=B["poles"][0], pole_n=B["poles"][1])
    names = list(parts); sf = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = vol(parts[names[i]] ^ parts[names[j]])
            if v > 0.01 and {names[i], names[j]} not in ({"sl_p", "pin_p"}, {"sl_n", "pin_n"}, {"carrier", "pole_p"}, {"carrier", "pole_n"}):
                sf[f"{names[i]}-{names[j]}"] = round(v, 3)
    res["static_fit_mm3"] = sf
    w = dict(vs_cart=0.0, vs_recv=0.0, vs_each=0.0)
    for k in range(9):
        rt = bl.TRAVEL * k / 8 - (0.02 if k == 8 else 0.0)
        a = moving(g, rt, 1); b = moving(g, rt, -1)
        w["vs_cart"] = max(w["vs_cart"], vol(a ^ C), vol(b ^ C)); w["vs_recv"] = max(w["vs_recv"], vol(a ^ R), vol(b ^ R))
        w["vs_each"] = max(w["vs_each"], vol(a ^ b))
    res["squeeze_mm3"] = {k: round(v, 3) for k, v in w.items()}
    rows, worst_c = [], 0.0
    for d in np.arange(-40.0, 0.001, 0.5):
        worst_c = max(worst_c, vol(C.translate([0, d, 0]) ^ R))
        rr = []
        for s in (1, -1):
            S = lambda rt: moving(g, rt, s).translate([0, d, 0])
            if vol(S(0.0) ^ R) < 0.01: rr.append(0.0); continue
            if vol(S(bl.TRAVEL) ^ R) >= 0.01: rr.append(None); continue
            lo, hi = 0.0, bl.TRAVEL
            for _ in range(9):
                mid = (lo + hi) / 2
                if vol(S(mid) ^ R) < 0.01: hi = mid
                else: lo = mid
            rr.append(round(hi, 2))
        rows.append((round(float(d), 2), rr))
    res["insert_cart_vs_recv_mm3"] = round(worst_c, 3)
    res["insert_stuck"] = [d for d, rr in rows if None in rr]
    res["insert_max_cam_mm"] = max(max(x for x in rr if x is not None) for _, rr in rows)
    res["snap_out_depth_mm"] = next((-d for d, rr in rows[::-1] if max(rr) > 0.0), 0.0)
    for nm, rt in (("gripped_pull_mm3", R0), ("gripped_pull_at_min_retract_mm3", NEED), ("gripped_pull_at_rib_limit_mm3", bl.TRAVEL - 0.02)):
        A = C + moving(g, rt, 1) + moving(g, rt, -1)
        res[nm] = round(max(vol(A.translate([0, -k, 0]) ^ R) for k in np.arange(0, 40.01, 0.5)), 3)
    first, bar_first = None, None
    for d in np.arange(0.0, 2.0, 0.02):
        if first is None and vol(bolt_pin(g, 0, 1).translate([0, -d, 0]) ^ B["recv"]) > 0.001: first = round(float(d), 2)
        if bar_first is None and vol(slider(g, 0, 1).translate([0, -d, 0]) ^ B["recv"]) > 0.001: bar_first = round(float(d), 2)
    res["pull_drop_to_pin_mm"] = first
    res["pull_drop_to_bolt_pctg_mm"] = bar_first
    mu = 0.2
    res["wedge"] = dict(angle_deg=20.0, mu_steel_pctg_E=mu, outward_per_N_pull=round(math.sin(bl.UNDER) - mu * math.cos(bl.UNDER), 3))
    res["engage_depth_mm"] = dict(guide_poles=round(g["Y_ZONE"] - (r["Y_DK1"] - v5.POLE_L), 2), xt60=round(bl.XM_IL - 1.5, 2),
                                  bolts_on_lead_in=round(bl.BOLT_Y1 - bl.Y_RB, 2))
    tol_x = 0.2 / math.tan(ALPHA)
    res["travel"] = dict(to_rib=round(bl.TRAVEL, 2), wedge_tight_nominal=R0, needed=NEED, wedge_tol_mm_y=0.2, wedge_tol_mm_x=round(tol_x, 2),
                         retract_if_0p2_early=round(R0 - tol_x, 2), margin_early=round(R0 - tol_x - NEED, 2),
                         play_at_wedge_if_0p2_late_mm=round(max(0.0, R0 + tol_x - bl.TRAVEL) * math.tan(ALPHA), 3),
                         bolt_tip_in_wall_hole_at_limit_mm=round(g["X_TIP"] - bl.TRAVEL - g["X_IN"], 2))
    res["spring"] = spring_report(B)
    # ---- blades (with their hook nubs) and heel: open at rest, rising, squeezing, tight, at the rib limit
    deck, pads = heel(g)
    H = U(deck, *pads)
    S0 = slider(g, 0, 1) + slider(g, 0, -1)
    bo = U(*blades(g, BLADE_OPEN))
    gb = {}
    gb["open_vs_cart_mm3"] = round(vol(bo ^ B["cart"]), 3)
    gb["open_vs_tabs_mm3"] = round(vol(bo ^ S0), 3)
    gb["rising_open_vs_cart_and_tabs_mm3"] = round(max(vol(U(*blades(g, BLADE_OPEN, dy=-k)) ^ (B["cart"] + S0)) for k in np.arange(0, 30.01, 0.5)), 3)
    gb["heel_vs_cart_mm3"] = round(vol(H ^ C), 3)
    gb["heel_vs_receiver_seated_mm3"] = round(vol(H ^ R), 3)
    gb["heel_pad_contact_mm2"] = round(sum(vol(p.translate([0, 0.05, 0]) ^ B["cart"]) / 0.05 for p in pads), 1)
    gb["heel_pad_area_mm2"] = round(4 * (PAD_X[1] - PAD_X[0]) * (PAD_Z[1] - PAD_Z[0]), 1)
    worst = dict(vs_cart=0.0, vs_tabs=0.0, vs_each=0.0)
    for k in range(24):
        face = BLADE_OPEN - (BLADE_OPEN - (bl.TAB_OUT - R0)) * k / 23
        rt = max(0.0, bl.TAB_OUT - face)
        Bk = blades(g, face, back=0.02)
        Sk = slider(g, rt + 0.02, 1) + slider(g, rt + 0.02, -1)
        worst["vs_cart"] = max(worst["vs_cart"], vol(U(*Bk) ^ B["cart"])); worst["vs_tabs"] = max(worst["vs_tabs"], vol(U(*Bk) ^ Sk))
        worst["vs_each"] = max(worst["vs_each"], vol(Bk[0] ^ Bk[1]))
    gb["squeezing_mm3"] = {k: round(v, 3) for k, v in worst.items()}
    lim = blades(g, bl.TAB_OUT - bl.TRAVEL + 0.02, back=(bl.TRAVEL - R0) * math.tan(ALPHA) + 0.02)
    gb["at_rib_limit_vs_cart_mm3"] = round(vol(U(*lim) ^ B["cart"]), 3)
    tight = blades(g, bl.TAB_OUT - R0)
    gb["nub_on_foot_contact_mm2"] = round(vol(blade(g, bl.TAB_OUT - R0, 1, dy=-0.05) ^ B["cart"]) / 0.05, 1)
    gb["blade_face_on_tab_mm2"] = round(vol(blade(g, bl.TAB_OUT - R0 - 0.05, 1, back=0.2) ^ slider(g, R0, 1)) / 0.05, 1)
    gb["blade_stroke_mm"] = round(BLADE_OPEN - (bl.TAB_OUT - R0), 2)
    gb["blade_face_squeezed_mm"] = round(bl.TAB_OUT - R0, 2)
    gb["z_capture_mm"] = round(ARCH_HZ - NUB_HZ, 2)
    res["grabber_interface"] = gb
    return res, rows


def hold_report(B):
    out = {"six_directions_at_8N_tpu_pads": six_directions(B, 8.0, MU["pad_tpu"])}
    tab = []
    for Fb in (6.0, 8.0, 10.0, 15.0):
        for nm, mu_pad in (("TPU-faced pads", MU["pad_tpu"]), ("bare PCTG pads", MU["pad_pctg"])):
            for n in (1.0, 2.0, 3.0):
                h = hold(B, Fb, mu_pad, n); h["pads"] = nm; tab.append(h)
    out["all_gravity_directions"] = tab
    Fs = 2 * spring_state(R0)["N"]
    out["squeeze_needed_N_per_blade"] = dict(
        to_unlatch=round(Fs + 0.5, 1), note="both springs + ~0.5 slider friction (E)",
        v5_to_unlatch=2.9,
        to_hold_xt60_pull_frictionless=round(Fs + 59.1 / 2 * math.tan(ALPHA), 1),
        thirty_deg_face_to_hold_xt60_pull_frictionless=round(Fs + 59.1 / 2 * math.tan(math.radians(30)), 1),
        thirty_deg_face_self_locks_above_mu=round(math.tan(math.radians(30)), 2), ten_deg_self_locks_above_mu=round(math.tan(ALPHA), 2))
    def need(n_g, mu_pad, tgt=1.5):
        for Fb in np.arange(4.0, 60.01, 0.25):
            h = hold(B, float(Fb), mu_pad, n_g)
            if h["releases"] and min(h["margin_pads_stay_down"], h["margin_no_slip"]) >= tgt: return float(Fb)
        return None
    out["squeeze_for_margin_1p5_N_per_blade"] = {f"{n:g} g, {nm}": need(n, mu) for n in (1.0, 2.0, 3.0)
                                                 for nm, mu in (("TPU pads", MU["pad_tpu"]), ("PCTG pads", MU["pad_pctg"]))}
    return out


# ------------------------------------------------------------------ save
def tm(m):
    return bl.tm(m)


PRINTED = ("latch6_receiver", "latch6_carrier", "latch6_carrier_cap", "latch6_cartridge", "latch6_slider", "latch6_receiver_strip_a", "latch6_receiver_strip_b")


def save(B):
    g = B["g"]
    out = {"latch6_receiver": B["recv"], "latch6_carrier": B["carrier"], "latch6_carrier_cap": B["cap"],
           "latch6_cartridge": B["cart"], "latch6_slider": slider(g, 0, 1),
           "latch6_xt60_male": B["male"], "latch6_xt60_female": B["female"], "latch6_pack": bl.pack_body(g)}
    for i, p in enumerate(B["poles"]): out[f"latch6_guide_pole_{'pn'[i]}"] = p
    out["latch6_bolt_pin"] = bolt_pin(g, 0, 1)
    for k, m in zip("ab", sorted(B["strips"].decompose(), key=lambda m: m.bounding_box()[0])):   # two strips, one print each
        out[f"latch6_receiver_strip_{k}"] = m
    paths = {}
    for nm, m in out.items():
        d = os.path.join(ROOT, "parts", nm); os.makedirs(d, exist_ok=True)
        t = tm(m); t.export(os.path.join(d, "part.stl")); paths[nm] = t
    return paths


# ------------------------------------------------------------------ page meshes
def page_matrix(zt):
    return np.array([[1, 0, 0, 0], [0, 0, -1, zt], [0, 1, 0, 0], [0, 0, 0, 1.0]])


G_R, G_C, G_L, G_G = "receiver (on the drone), cut away", "cartridge (on the pack), cut away", "latch (moving)", "grabber interface (blade tips, nubs, heel pads)"
VIEW = [  # name, group, colour, opacity, cut, role, material
    ("receiver", G_R, "#9fb4cc", None, True, "structure", "PCTG"),
    ("receiver_strips", G_R, "#3a3a3a", None, True, "TPU", "TPU"),
    ("receiver_skid_nose", G_R, "#3a3a3a", None, True, "TPU", "TPU"),
    ("carrier", G_R, "#7f93ab", None, True, "structure", "PCTG"),
    ("carrier_cap", G_R, "#7f93ab", None, True, "structure", "PCTG"),
    ("xt60_male", G_R, "#e8b923", None, False, "bought", None),
    ("guide_pole_p", "guide poles (steel)", "#e3e7ec", None, False, "bought", None),
    ("guide_pole_n", "guide poles (steel)", "#e3e7ec", None, False, "bought", None),
    ("cartridge", G_C, "#4f9d8f", None, True, "structure", "PCTG"),
    ("cartridge_strip_px", G_C, "#3a3a3a", None, True, "TPU", "TPU"),
    ("cartridge_strip_nx", G_C, "#3a3a3a", None, True, "TPU", "TPU"),
    ("pack", G_C, "#555b66", 0.35, True, "bought", None),
    ("xt60_female", G_C, "#e8b923", None, False, "bought", None),
    ("slider_p", G_L, "#ff8a1f", None, True, "structure", "PCTG"),      # the +x side in section at the tab line, the -x side whole
    ("slider_n", G_L, "#ff8a1f", None, False, "structure", "PCTG"),
    ("bolt_pin_p", G_L, "#e3e7ec", None, False, "bought", None),
    ("bolt_pin_n", G_L, "#e3e7ec", None, False, "bought", None),
    ("spring_a", G_L, "#c9783f", None, False, "bought", None),
    ("spring_b", G_L, "#c9783f", None, False, "bought", None),
    ("blade_p", G_G, "#d64545", None, True, "structure", None),
    ("blade_n", G_G, "#d64545", None, False, "structure", None),
    ("heel_deck", G_G, "#a83a3a", None, True, "structure", None),
    ("heel_pads", G_G, "#2b2b2b", None, True, "TPU", "TPU"),
]


def view_sources(B):
    g = B["g"]
    src = dict(receiver=B["recv"], receiver_strips=B["strips"], carrier=B["carrier"], carrier_cap=B["cap"], xt60_male=B["male"],
               guide_pole_p=B["poles"][1], guide_pole_n=B["poles"][0], cartridge=B["cart"], pack=bl.pack_body(g),
               xt60_female=B["female"], slider_p=slider(g, 0, 1), slider_n=slider(g, 0, -1),
               bolt_pin_p=bolt_pin(g, 0, 1), bolt_pin_n=bolt_pin(g, 0, -1))
    if B.get("rskids"): src["receiver_skid_nose"] = B["rskids"][0]
    for k, m in zip(("px", "nx"), B.get("cstrips", [])): src[f"cartridge_strip_{k}"] = m
    src["blade_p"], src["blade_n"] = blades(g, BLADE_OPEN)
    deck, pads = heel(g)
    src["heel_deck"], src["heel_pads"] = deck, U(*pads)
    return src


def view_meshes(B):
    g = B["g"]; zt = g["Z_T"]
    src = view_sources(B)
    half = box(-500, 500, -500, 500, zt, 1000) - box(v5.POLE_X, 500, -500, 500, g["Z_P1"] - 3.0, 1000)
    M = page_matrix(zt)
    out = {}
    for nm, grp, col, op, cut, role, mat in VIEW:
        if nm not in src: continue
        m = src[nm] ^ half if cut else src[nm]
        if m.is_empty(): continue
        t = tm(m); t.apply_transform(M); out[nm] = t
    for nm, sz in (("spring_a", -1), ("spring_b", 1)):
        sp, L = spring_mesh(g, sz); sp.apply_transform(M); out[nm] = sp
    return out


def write_page(B, res):
    meshes = view_meshes(B)
    d = os.path.join(HERE, "view"); os.makedirs(d, exist_ok=True)
    parts = []
    for nm, grp, col, op, cut, role, mat in VIEW:
        if nm not in meshes: continue
        f = os.path.join(d, f"{nm}.stl"); meshes[nm].export(f)
        steel = nm.startswith(("bolt_pin", "guide_pole"))
        mass = None
        if (mat or steel) and not cut:
            mass = round(abs(meshes[nm].volume) / 1000 * (RHO["steel"] if steel else RHO.get(mat, 1.23)), 2)
        if nm.startswith("spring"): mass = 0.2
        parts.append(dict(id=f"latch6_view_{nm}", name=nm.replace("_", " "), role=role, group=grp, color=col, opacity=op,
                          mesh=os.path.relpath(f, ROOT), pose=None, material=mat if not steel else None, tag="C",
                          mass_g=mass, version="latch v6 (10 Nov)", source="parts/latch6/make.py"))
    return parts


def overseer_report():
    """the same module on the Overseer cartridge: what fits unchanged and the hold at its mass."""
    bl._CACHE.clear()
    try:
        B = build("overseer")
    except Exception as e:                                   # noqa
        return dict(built=False, error=repr(e)[:300])
    g = B["g"]
    out = dict(built=True, swap_g=round(g["SWAP_M"], 1), z_T=g["Z_T"], cg=[round(float(v), 1) for v in g["SWAP_CG"]])
    C = cart_all(B); R = receiver_all(B)
    sl = [moving(g, 0, 1), moving(g, 0, -1)]
    out["static_mm3"] = dict(sliders_vs_cart=round(max(vol(s ^ C) for s in sl), 3), sliders_vs_recv=round(max(vol(s ^ R) for s in sl), 3),
                             cart_vs_recv=round(vol(C ^ R), 3))
    out["squeezed_vs_cart_mm3"] = round(max(vol(moving(g, bl.TRAVEL - 0.02, s) ^ C) for s in (1, -1)), 3)
    A = C + moving(g, R0, 1) + moving(g, R0, -1)
    out["gripped_pull_mm3"] = round(max(vol(A.translate([0, -k, 0]) ^ R) for k in np.arange(0, 40.01, 1.0)), 3)
    deck, pads = heel(g)
    out["heel_pad_contact_mm2"] = round(sum(vol(p.translate([0, 0.05, 0]) ^ B["cart"]) / 0.05 for p in pads), 1)
    out["blades_tight_vs_cart_mm3"] = round(vol(U(*blades(g, bl.TAB_OUT - R0, back=0.02)) ^ B["cart"]), 3)
    out["spring"] = {k: v for k, v in spring_report(B).items() if k in ("slider_pin_magnet_g", "inertia_35g_N", "preload_over_inertia", "one_spring_left_over_inertia")}
    out["hold"] = [hold(B, Fb, MU["pad_tpu"], n) for Fb in (8.0, 10.0, 15.0, 20.0) for n in (1.0, 2.0, 3.0)]
    out["bending_at_its_pull"] = {k: v for k, v in bending(B, 10.0, 57.0 + g["SWAP_M"] / 1000 * G0).items() if k in ("nub_pull", "foot_pull", "worst_index")}
    def need(n_g, tgt=1.5):
        for Fb in np.arange(4.0, 80.01, 0.5):
            h = hold(B, float(Fb), MU["pad_tpu"], n_g)
            if h["releases"] and min(h["margin_pads_stay_down"], h["margin_no_slip"]) >= tgt: return float(Fb)
    out["squeeze_for_margin_1p5_N_per_blade"] = {f"{n:g} g": need(n) for n in (1.0, 2.0, 3.0)}
    out["rest_stop_mm3_at_0.1_out"] = round(vol(slider(g, -0.1, 1) ^ B["cart"]), 3)
    out["squeeze_at_3g_15N"] = hold(B, 15.0, MU["pad_tpu"], 3.0)
    out["mass_g"] = dict(cartridge=round(vol(B["cart"]) / 1000 * RHO["PCTG"], 1), slider=round(vol(slider(g, 0, 1)) / 1000 * RHO["PCTG"], 2))
    bl._CACHE.clear()
    return out


ARTIFACT = "https://claude.ai/artifact/Pqu6aZFdW9g7jtRyBFdraw"


def write_assembly(res):
    sp = res["spring"]; h8 = next(h for h in res["hold"]["all_gravity_directions"] if h["Fb"] == 8.0 and h["n_g"] == 1.0 and h["pads"].startswith("TPU"))
    a = {"name": "latch v6", "title": "Battery Latch v6 (Scout)", "readme_view": "section", "artifact": ARTIFACT, "machine": "box", "order": 90,
         "views": {"section": {"eye": [0, 130, 8], "target": [0, 0, 12]},
                   "wedge close": {"eye": [10, 46, 6], "target": [5, 0, 4]},
                   "bolt close": {"eye": [36, 62, 2], "target": [19, 0, 7]},
                   "springs, from above": {"eye": [0, 22, 60], "target": [0, 2, 9]},
                   "grabber side, from below": {"eye": [45, 70, -70], "target": [0, -4, 0]},
                   "lead end": {"eye": [120, -60, 70], "target": [8, -36, 34]},
                   "held, tipped": {"eye": [170, 170, -110], "target": [0, 0, -105]}},
         "note": "The press-two-tabs latch, v6, battery side only: one push latches it, one squeeze unlatches it and holds the pack on the grabber. "
                 "Cut at the tab line; the +x slider and blade are in section, the -x ones whole. "
                 f"Two steel springs between the tabs ({sp['preload_pair_N']} N at rest, {sp['preload_over_inertia']} x the 35 g inertia of a slider; "
                 f"{sp['squeezed_pair_N']} N squeezed). The grabber grips the cartridge, not the loose tabs: each blade's nub slides onto a 10 deg wedge foot "
                 "on the centre rib (self-locking) and draws the cartridge onto four heel pads; "
                 f"at 8 N per blade {h8['pad_preload_N']} N on the pads, x{h8['margin_pads_stay_down']} / x{h8['margin_no_slip']} against tipping / slip at 1 g in the worst direction. "
                 "Receiver, XT60 carrier, guide poles, steel bolt pins and the 20 deg seat ramp are v5's. Built by parts/latch6/make.py on the vendored latch v4 CAD.",
         "cg_ref": None, "timeline": "tools/latch6_motion.py:timeline", "appendix": "systems/latch6.appendix.html", "parts": res["page_parts"]}
    json.dump(a, open(os.path.join(ROOT, "systems", "latch6.assembly.json"), "w"), indent=1)


def appendix(res):
    sp, H, bd, gi, tv, ov = res["spring"], res["hold"], res["bending"], res["grabber_interface"], res["travel"], res["overseer"]
    mc = res.get("motion_collision", {})
    td = lambda *c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    rows = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}",
                      f"x{h['margin_pull_frictionless']}", h["slip_x_mm"], h["pack_top_mm"])
                   for h in H["all_gravity_directions"] if h["pads"].startswith("TPU") and h.get("releases"))
    orow = "".join(td(f"{h['Fb']:g} N", f"{h['n_g']:g} g", h["pad_preload_N"], f"x{h['margin_pads_stay_down']}", f"x{h['margin_no_slip']}", h["slip_x_mm"])
                   for h in ov.get("hold", []) if h.get("releases"))
    six = "".join(td(k, v["path"], v.get("holds_N") if v.get("holds_N") is not None else (f"{v['holds_Nmm']:.0f} N mm" if v.get("holds_Nmm") is not None else "positive"),
                     f"x{v['margin_1g']}" if v.get("margin_1g") else "") for k, v in H["six_directions_at_8N_tpu_pads"].items())
    bought = "".join(td(b["part"], b["spec"], f'<a href="{b["url"]}">{b["listing"]}</a>' if b["url"] else b["listing"], b["check"]) for b in BOUGHT)
    need = H["squeeze_needed_N_per_blade"]; n15 = H["squeeze_for_margin_1p5_N_per_blade"]
    return f"""<!-- latch v6: the numbers behind the motion (written by parts/latch6/make.py; appended by tools/build_site.py). Scoped to #latch6. -->
<style>
#latch6 {{ background: var(--bg); color: var(--fg); font: 14px/1.5 var(--sans); padding: 8px 16px 40px; }}
#latch6 .wrap {{ max-width: 1040px; margin: 0 auto; }}
#latch6 h2 {{ font: 600 16px/1.3 var(--sans); margin: 26px 0 6px; }}
#latch6 p {{ margin: 6px 0; max-width: 78ch; }}
#latch6 .t {{ overflow-x: auto; }}
#latch6 table {{ border-collapse: collapse; font: 12px/1.4 var(--mono); margin: 8px 0; background: var(--panel); }}
#latch6 th, #latch6 td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: left; vertical-align: top; }}
#latch6 th {{ color: var(--muted); font-weight: 500; }}
#latch6 a {{ color: var(--accent); }}
#latch6 .open {{ border-left: 3px solid var(--bad); padding-left: 10px; }}
</style>
<section id="latch6"><div class="wrap">
<h2>1. How it springs and stays latched</h2>
<p>Two steel compression springs in parallel between the two tabs, under the pack. Each bolt's steel pin sits on a 20 deg ramp in the receiver, so a pull seats it harder; the springs only have to beat vibration and a crash's inertia on a slider. Not printed flexures (PCTG creeps under a standing preload) and not magnets (their force falls off as the bolt moves out, the wrong way round).</p>
<div class="t"><table><tr><th></th><th>v5, one spring</th><th>v6, two springs</th><th>v6 with one spring lost</th><th>alternative: one per slider</th></tr>
{td("at rest, on each slider", "0.88 N", f"{sp['preload_pair_N']} N", f"{sp['rest']['N']} N", f"{sp['alternative_one_per_slider']['rest_N']} N")}
{td(f"over the 35 g inertia of a slider + pin ({sp['slider_pin_magnet_g']} g, {sp['inertia_35g_N']} N)", f"x{sp['v5']['over_inertia']}", f"x{sp['preload_over_inertia']}", f"x{sp['one_spring_left_over_inertia']}", f"x{sp['alternative_one_per_slider']['over_inertia']}; x0 on one bolt if its spring is lost")}
{td("squeezed, on each blade", "2.4 N", f"{sp['squeezed_pair_N']} N", "", "")}
{td("wire stress at full squeeze", "", f"{sp['at_rib_limit']['MPa']:.0f} MPa, {100 * sp['at_rib_limit']['of_uts']:.0f} % of tensile (E)", "", "")}
{td("solid height under the shortest length", "", f"{sp['solid_margin_mm']} mm spare", "", "")}
</table></div>
<p>Two between the tabs wins on reliability: either spring failing leaves both bolts sprung at half force; with one per slider a lost spring leaves one bolt dead. Set 5 mm either side of the tab line they also keep each slider square in its channel. The rib's two notches sit near its points of zero bending: rib index {bd['rib']['swap across 43.5 N']['index']} at the 43.5 N across (v5's one centre notch {bd['rib']['v5 one centre notch, across 43.5 N (for comparison)']['index']}), {bd['rib']['pull + across together']['index']} with the XT60 pull on top.</p>
<h2>2. Holding the pack on the grabber, at any angle</h2>
<p>v5 gripped pockets in the sliders. A slider floats 0.35 in its channel and bears 9 mm outboard of the grip, so a pull sagged the tab about {bd['tab']['v5_sag_at_grip_mm']} mm before anything was tight. v6 grips the cartridge: a wedge foot each side of the centre rib, a hook nub on each blade tip, four TPU-faced heel pads on the head under the cartridge's rim bands (x &plusmn;16.7, tab line &plusmn;21.5). The wedge is 10 deg: self-locking above a friction of {need['ten_deg_self_locks_above_mu']}, so the squeeze draws the cartridge onto the pads and a pull cannot open the blades. A 30 deg face would need friction above {need['thirty_deg_face_self_locks_above_mu']} or {need['thirty_deg_face_to_hold_xt60_pull_frictionless']} N per blade to hold the 57 N XT60 pull.</p>
<div class="t"><table><tr><th>direction (8 N per blade)</th><th>what carries it</th><th>holds before anything moves, N</th><th>margin at 1 g</th></tr>{six}</table></div>
<p>Every direction of gravity (26 over the sphere), Scout swap {res['swap_g']} g, CG {res['swap_cg'][1]} mm above the pads, TPU pads:</p>
<div class="t"><table><tr><th>per blade</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>pull, no wedge friction</th><th>slip across, mm</th><th>pack top moves, mm</th></tr>{rows}</table></div>
<p>Below x1 on "pads stay down" the pack does not come off: the nubs are self-locking, the hold just stops being preloaded. Across (x) is held by pad friction and then by the head's two blade springs, because each blade is sprung on its own: that is the head's mechanism, not a positive stop.</p>
<p>Bend under the 57 N XT60 pull + 1 g: blade opens {bd['blade_pull']['opens_mm']} mm (index {bd['blade_pull']['index']}), nub index {bd['nub_pull']['index']} on {bd['nub_pull']['contact_mm2']} mm&sup2;, foot {bd['foot_pull']['index']} ({bd['foot_pull']['tip_mm']} mm), rib {bd['rib']['XT60 pull 57 N + 1 g']['index']}; the tabs carry no pull at all. Blade under a 10 N squeeze: {bd['blade_squeeze']['tip_mm']} mm.</p>
<h2 class="open">3. Open number: the squeeze force per blade</h2>
<p>To unlatch: {need['to_unlatch']} N (v5 {need['v5_to_unlatch']}). For x1.5 on both hold margins, Scout, TPU pads: {n15['1 g, TPU pads']} N at 1 g, {n15['2 g, TPU pads']} N at 2 g, {n15['3 g, TPU pads']} N at 3 g (bare PCTG pads {n15['1 g, PCTG pads']} / {n15['2 g, PCTG pads']} / {n15['3 g, PCTG pads']}). 8-10 N covers the Scout. The Overseer on the same module needs {ov.get('squeeze_for_margin_1p5_N_per_blade', {}).get('1 g')} N at 1 g and {ov.get('squeeze_for_margin_1p5_N_per_blade', {}).get('2 g')} N at 2 g.</p>
<div class="t"><table><tr><th>Overseer ({ov.get('swap_g')} g)</th><th>load</th><th>on the pads, N</th><th>pads stay down</th><th>no slip</th><th>slip across, mm</th></tr>{orow}</table></div>
<h2>Checks</h2>
<div class="t"><table><tr><th>check</th><th>result</th></tr>
{td("static fit, squeeze, free insertion, held pull (at 3.3, 4.6 and 4.9 of retraction)", "0 mm&sup3; everywhere, no stuck step")}
{td("bolt travel", f"{tv['wedge_tight_nominal']} with the blade tight on its wedge, {tv['needed']} needed, {tv['to_rib']} to the rib; wedge 0.2 mm early in y: {tv['retract_if_0p2_early']}; 0.2 late: {tv['play_at_wedge_if_0p2_late_mm']} mm of play, no preload")}
{td("pull drop onto the steel pins / PCTG bolt would touch at", f"{res['pull_drop_to_pin_mm']} / {res['pull_drop_to_bolt_pctg_mm']} mm")}
{td("blades and heel: rising, squeezing, tight, at the rib limit", f"0 mm&sup3;; nub on foot {gi['nub_on_foot_contact_mm2']} mm&sup2;, blade face on tab {gi['blade_face_on_tab_mm2']} mm&sup2;, pads {gi['heel_pad_contact_mm2']} of {gi['heel_pad_area_mm2']} mm&sup2; on the rim bands, z capture &plusmn;{gi['z_capture_mm']}")}
{td("motion, every frame, uncut solids", f"{mc.get('frames')} frames, worst {mc.get('max_mm3')} mm&sup3;")}
{td("solid check", "receiver, carrier, carrier cap, cartridge, slider, both strips: PASS")}
</table></div>
<h2>Bought parts</h2>
<div class="t"><table><tr><th>part</th><th>spec</th><th>Amazon listing</th><th>check</th></tr>{bought}</table></div>
</div></section>
"""


def main():
    B = build()
    res, rows = checks(B)
    res["hold"] = hold_report(B)
    res["bending"] = bending(B)
    paths = save(B)
    g = B["g"]
    res["mass_g"] = {nm: round(abs(t.volume) / 1000 * (RHO["steel"] if ("pin" in nm or "pole" in nm) else RHO["TPU"] if "strip" in nm else RHO["PCTG"]), 2)
                     for nm, t in paths.items() if not any(k in nm for k in ("xt60", "pack"))}
    res["swap_g"] = round(g["SWAP_M"], 1)
    res["swap_cg"] = [round(float(v), 2) for v in g["SWAP_CG"]]
    res["insert_rows"] = rows
    res["geom"] = {k: round(float(v), 3) for k, v in g.items() if isinstance(v, (int, float))}
    res["rgeom"] = {k: round(float(v), 3) for k, v in B["r"].items() if isinstance(v, (int, float)) and k not in res["geom"]}
    res["v6"] = dict(TAB_IN=bl.TAB_IN, TAB_OUT=bl.TAB_OUT, TRAVEL=bl.TRAVEL, R0=R0, NEED=NEED, OVERLAP=bl.OVERLAP, ALPHA_deg=10.0, FOOT_X1=FOOT_X1,
                     FOOT_Y=FOOT_Y, FOOT_HZ=FOOT_HZ, NUB_L=NUB_L, NUB_HZ=NUB_HZ, NUB_Y1=NUB_Y1, ARCH_HZ=ARCH_HZ, ARCH_Y=ARCH_Y, SPR_DZ=SPR_DZ,
                     SPR_DEPTH=bl.SPR_DEPTH, SPR_YC=bl.SPR_YC, MAG_X=bl.MAG_X, BLADE_OPEN=BLADE_OPEN, PAD_X=PAD_X, PAD_Z=PAD_Z, PAD_T=PAD_T,
                     K_BLADE=K_BLADE, MU=MU, allow_MPa=ALLOW)
    res["rest_stop_mm3_at_0.1_out"] = round(vol(slider(g, -0.1, 1) ^ B["cart"]), 3)
    res["bought"] = BOUGHT
    res["page_parts"] = write_page(B, res)
    res["overseer"] = overseer_report()
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)
    write_assembly(res)
    spm = importlib.util.spec_from_file_location("latch6_motion", os.path.join(ROOT, "tools", "latch6_motion.py"))
    mo = importlib.util.module_from_spec(spm); spm.loader.exec_module(mo)
    bl._CACHE.clear()
    res["motion_collision"] = mo.collide()                      # every frame of the page's motion, the uncut solids
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)
    open(os.path.join(ROOT, "systems", "latch6.appendix.html"), "w").write(appendix(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("insert_rows", "geom", "rgeom", "page_parts", "hold", "overseer", "bought")}, indent=1))


if __name__ == "__main__":
    main()
