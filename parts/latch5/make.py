"""Battery latch v5 (10 Nov 2026, Spark: "Go ahead": the press-two-tabs latch kept, with the good ideas from Alex's xswap).

The latch v4 / receiver v3 / cartridge v6 (battery_latch_v4.py, vendored) with four changes:
  1. Steel bearing pins: a 3 x 16 steel dowel pressed along z into each bolt's bottom outer corner (0.5 proud under the bolt,
     flush with its tip). The pin, not the PCTG bolt, carries the pull-out, so the bolt edge cannot creep or round off.
  2. A deeper pocket on a proper ledge: overlap 2.3 -> 3.0, the pocket floor a flat shelf from the mouth, then a 20 deg seat
     ramp under the pin (v4: one 15 deg slope from the mouth, its edge 0.55 over the lead-in).
  3. Locks tighter under pull: steel on PCTG (mu ~0.2, E) on the 20 deg ramp (tan 0.36): a pull drives each pin outward,
     down the ramp, until the bolt tip sits on the pocket's back wall. A harder pull only seats it harder.
     (v4: PETG on PETG mu 0.3 on 15 deg, tan 0.27: neutral, the spring held it.)
  4. Two 3 x 20 steel guide poles beside the XT60, pressed in the receiver's XT60 carrier, entering 3.3 bores with lead-in
     cones in two new bosses at the cartridge's lead end. They engage 11.7 below the seat, the XT60 at 6.0.
Travel 3.0 -> 3.8 (tab inner face 5.0 -> 5.8, the blades close to 6.2, v5 heads went to 6.0); grip pocket 3.0 -> 2.2 deep
(2.0 wall behind it); pen spring cut to 22 free (rest 0.88 N, squeezed 2.4 N).

python3 parts/latch5/make.py   -> parts/latch5_*/part.stl, parts/latch5/result.json, the checks printed
Coordinates: cartridge c (x across, y up the install axis, 0 = flat bottom, z along the pack, 0 = closed end); the page frame
is P = (c.x, z_T - c.z, c.y): the receiver on top, the cartridge rising into it, the lead end (XT60) toward -Y.
"""
import json, math, os, sys
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import battery_latch_v4 as bl
from battery_latch_v4 import box, cyl, hullp, prism_xy, U, vol, mirx, rot_y180

# ------------------------------------------------------------------ v5 numbers (C unless tagged)
bl.OVERLAP = 3.0                       # pin outer edge past the locating face (v4 2.3)
bl.UNDER = math.radians(20.0)          # seat ramp (v4 15)
bl.TAB_IN = 5.8                        # tab inner face at rest (v4 5.0): travel to the centre rib 3.8
bl.TRAVEL = bl.TAB_IN - bl.RIB_HX      # 3.8 (needs OVERLAP + 0.3 = 3.3: margin 0.5, v4 0.4)
bl.BOLT_Y0 = 6.8                       # bolt underside (v4 6.5): the pin hangs 0.5 below it
bl.BOLT_T = bl.BOLT_Y1 - bl.BOLT_Y0    # 4.7
bl.GP_D = 2.2                          # grip pocket depth (v4 3.0): the tab is 4.2 thick now, 2.0 stays behind the pocket
bl.SPRING_FREE = 22.0                  # pen spring free length (v4 20)
PIN_R, PIN_L = 1.5, 16.0               # bolt pins: 3 x 16 steel dowel (bought)
PIN_DOWN = 0.5                         # pin below the bolt underside
SNAP = 0.45                            # pin bottom over the pocket shelf at the seat: the bolt snaps out 0.45 before it
LUG_Z0, LUG_Y1 = 7.5, 5.5             # rest-stop lugs on the tab ears: |z| 7.5..11 (blades +-6.5), y 0..5.5
RAMP_IN = 0.3                          # the seat ramp starts this far inboard of the pin's lowest point, so the pin lands
                                       # on the ramp (0.46 drop) before the shelf corner (0.48) and slides outward off it
BACK = 0.2                             # bolt tip to the pocket back wall at rest (v4 0.3; 2.0 wall left behind it)
POLE_R, POLE_L = 1.5, 20.0             # guide poles: 3 x 20 steel dowel (bought)
POLE_X = 12.5                          # |x| of the poles, beside the XT60 flange
POLE_DZ = 8.55                         # pole axis this far toward the closed end from the plug axis: tangent to the flange's
                                       # float pocket (the steel is the float stop there)
POLE_CLR = 0.15                        # pole in its cartridge bore, per side
POLE_CONE = 1.5                        # bore mouth cone (45 deg)
POLE_TIP = 0.5                         # pole tip chamfer
POLE_EMBED = 7.0                       # pressed this deep into the carrier base (LIP_Y0 .. Y_DK1)
RHO = {"PCTG": 1.23, "steel": 7.85}


def pin_axis(g):
    """bolt pin centre (x, y) at rest, +x slider."""
    return g["X_TIP"] - PIN_R, bl.BOLT_Y0 + PIN_R - PIN_DOWN


def contact(g):
    """the pin's contact with the 20 deg ramp at rest (x, y) and the ramp line through the seat."""
    xp, yp = pin_axis(g)
    return xp + PIN_R * math.sin(bl.UNDER), yp - PIN_R * math.cos(bl.UNDER)


# ------------------------------------------------------------------ slider v7 (one part, both sides)
def slider(g, retract=0.0, side=1):
    zt, tip = g["Z_T"], g["X_TIP"]
    Y0, Y1 = bl.BOLT_Y0, bl.BOLT_Y1
    tab = box(bl.TAB_IN, bl.TAB_OUT, 0.0, Y1, zt - bl.TAB_HZ, zt + bl.TAB_HZ)
    bar = box(bl.TAB_IN, tip, Y0, Y1, zt - bl.BOLT_HZ, zt + bl.BOLT_HZ)
    cham = prism_xy([(tip - bl.TIP_CH, Y1 + 0.01), (tip + 0.01, Y1 + 0.01), (tip + 0.01, Y1 - bl.TIP_CH)],
                    zt - bl.BOLT_HZ - 1, zt + bl.BOLT_HZ + 1)
    # rest stops (new in v5; v4 had none, so its spring would push a free slider 2.0 past rest, the bolt past the feet's
    # lead-in, and jam it on their bottom faces): a lug on each tab ear, beside the blades (|z| 7.5..11), bears on the
    # cartridge rim band's inner face (x 15.0, y 0..4) at rest
    x_stop = g["X_IN"] - 2.0                     # the cartridge's 4.0 rim band re-added over the recess: its face is the stop
    lugs = [box(bl.TAB_OUT - 0.01, x_stop, 0.0, LUG_Y1, *sorted((zt + sz * LUG_Z0, zt + sz * bl.TAB_HZ))) for sz in (-1, 1)]
    s = U(tab, bar, *lugs) - cham
    xp, yp = pin_axis(g)
    s = s - cyl("z", xp, yp, PIN_R, zt - bl.BOLT_HZ - 0.01, zt - bl.BOLT_HZ + PIN_L + 0.5, 32)   # blind bore from -z
    s = s - prism_xy([(bl.TAB_OUT - bl.TAB_CH, -0.01), (bl.TAB_OUT + 0.01, -0.01), (bl.TAB_OUT + 0.01, bl.TAB_CH)],
                     zt - bl.TAB_HZ - 1, zt + bl.TAB_HZ + 1)
    T = bl.GP_TAPER
    h1 = bl.GP_H - 2 * bl.GP_D * math.tan(T); w1 = bl.GP_W - 2 * bl.GP_D * math.tan(T)
    gp = hullp([[bl.TAB_OUT + 0.01, bl.GP_YC + sy * bl.GP_H / 2, zt + sz * bl.GP_W / 2] for sy in (-1, 1) for sz in (-1, 1)] +
               [[bl.TAB_OUT - bl.GP_D, bl.GP_YC + sy * h1 / 2, zt + sz * w1 / 2] for sy in (-1, 1) for sz in (-1, 1)])
    gc = hullp([[bl.TAB_OUT + 0.01, bl.GP_YC + sy * (bl.GP_H / 2 + bl.GP_CH), zt + sz * (bl.GP_W / 2 + bl.GP_CH)] for sy in (-1, 1) for sz in (-1, 1)] +
               [[bl.TAB_OUT - bl.GP_CH, bl.GP_YC + sy * bl.GP_H / 2, zt + sz * bl.GP_W / 2] for sy in (-1, 1) for sz in (-1, 1)])
    s = s - gp - gc
    s = s - cyl("x", bl.SPR_YC, zt, bl.SPR_R, bl.TAB_IN - 0.01, bl.TAB_IN + bl.SPR_DEPTH, 32) \
          - box(bl.TAB_IN - 0.01, bl.TAB_IN + bl.SPR_DEPTH, bl.SPR_YC, Y1 + 1, zt - bl.SPR_R, zt + bl.SPR_R)
    s = s - cyl("y", bl.MAG_X, zt + bl.MAG_Z, bl.MAG_D / 2, -0.01, bl.MAG_DEPTH, 24)
    s = s.translate([-retract, 0, 0])
    return s if side > 0 else rot_y180(s, zt)


def bolt_pin(g, retract=0.0, side=1):
    xp, yp = pin_axis(g)
    zt = g["Z_T"]
    p = cyl("z", xp - retract, yp, PIN_R, zt - bl.BOLT_HZ + 0.5, zt - bl.BOLT_HZ + 0.5 + PIN_L, 32)
    return p if side > 0 else rot_y180(p, zt)


# ------------------------------------------------------------------ receiver pocket: shelf + seat ramp
def pocket_cut(r):
    zt, rf = r["Z_T"], r["RF"]
    z0, z1 = zt - r["PK_HZ"], zt + r["PK_HZ"]
    cx, cy = contact(r)
    t = math.tan(bl.UNDER)
    x_r = pin_axis(r)[0] - RAMP_IN                          # ramp top = shelf height
    assert rf + 0.5 < x_r < cx, (x_r, cx)
    xb = rf + r["PK_D"]
    yb = r["PK_Y0"] - (xb - x_r) * t
    prof = [(rf - 0.02, r["PK_Y0"]), (x_r, r["PK_Y0"]), (xb, yb), (xb, r["PK_Y1"]), (rf - 0.02, r["PK_Y1"])]
    pk = U(*[prism_xy([prof[0], prof[1], prof[3], prof[4]], z0, z1), prism_xy([prof[1], prof[2], prof[3]], z0, z1)])
    ch = prism_xy([(rf - 0.02, r["PK_Y1"] + 1.0), (rf - 0.02, r["PK_Y1"] - 0.01), (rf + 1.0, r["PK_Y1"] - 0.01)], z0, z1)
    return pk + ch, yb


_rgeom = bl.rgeom
def rgeom(g):
    r = _rgeom(g)
    xp, yp = pin_axis(g)
    r["PK_Y0"] = round(yp - PIN_R - SNAP, 3)                # shelf: pin bottom 0.45 over it at the seat
    r["PK_D"] = bl.OVERLAP + BACK
    assert r["PK_Y0"] >= bl.Y_RB + bl.HCAP + bl.PK_GAP - 1e-6, "shelf runs into the lead-in"
    return r


# ------------------------------------------------------------------ guide poles
def pole_xz(g):
    return [(s * POLE_X, g["Z_X"] - POLE_DZ) for s in (-1, 1)]


def poles(r):
    """the two steel poles, fixed in the carrier base, tip chamfered."""
    out = []
    y1 = r["Y_DK1"]; y0 = y1 - POLE_L
    for (x, z) in pole_xz(r):
        body = cyl("y", x, z, POLE_R, y0 + POLE_TIP, y1, 32)
        tip = mf.Manifold.cylinder(POLE_TIP, POLE_R - POLE_TIP, POLE_R, 32).rotate([-90, 0, 0]).translate([x, y0, z])
        out.append(body + tip)
    return out


def pole_boss_zone(g, grow=0.0):
    """carrier base extension round each pole (x POLE_X +- 3.5, toward the closed end)."""
    zx = g["Z_X"]
    return [box(*sorted((s * (POLE_X - 3.5 - grow), s * (POLE_X + 3.5 + grow))), 0, 1, zx - POLE_DZ - POLE_R - 2.0 - grow, zx - 6.0)
            for s in (-1, 1)]


_carrier_zone = bl.carrier_zone
def carrier_zone(r, fit=0.0):
    z = _carrier_zone(r, fit)
    zx = r["Z_X"]
    for s in (-1, 1):
        x0, x1 = sorted((s * (POLE_X - 3.5 - fit), s * (POLE_X + 3.5 + fit)))
        z = z + box(x0, x1, r["LIP_Y0"] - fit, r["Y_DK1"] + 0.01, zx - POLE_DZ - POLE_R - 2.0 - fit, zx - 6.0)
    return z


_xt60_carrier = bl.xt60_carrier
def xt60_carrier(r):
    base, cap = _xt60_carrier(r)
    zx = r["Z_X"]
    for s in (-1, 1):
        x0, x1 = sorted((s * (POLE_X - 3.5), s * (POLE_X + 3.5)))
        ext = box(x0, x1, r["LIP_Y0"], r["Y_DK1"], zx - POLE_DZ - POLE_R - 2.0, zx - 6.0)
        ext = ext - box(-r["PKX_XW"], r["PKX_XW"], r["Y_XM"], r["Y_DK1"] + 1, zx - r["PKX_ZW"], zx + r["PKX_ZW"])  # float pocket stays
        base = base + ext
    for (x, z) in pole_xz(r):
        base = base - cyl("y", x, z, POLE_R, r["Y_DK1"] - POLE_EMBED - 0.01, r["Y_DK1"] + 0.01, 32)
    return base, cap


_cartridge = bl.cartridge
def cartridge(g, clip=None):
    t = _cartridge(g, None)
    xi = g["X_IN"]
    y_top = g["Y_ZONE"]
    for (x, z) in pole_xz(g):
        s = 1 if x > 0 else -1
        r_b = POLE_R + POLE_CLR
        x0, x1 = sorted((s * (abs(x) - r_b - 2.0), s * (xi + 0.01)))
        boss = box(x0, x1, g["Y_SH0"], y_top, g["Z_P1"], g["Z_CW1"])        # end stop to cross wall, merged to the wall
        t = t + boss
        t = t - cyl("y", x, z, r_b, g["Y_SH0"] + 2.0, y_top + 0.01, 32)      # blind bore, 2.0 floor
        t = t - mf.Manifold.cylinder(POLE_CONE, r_b, r_b + POLE_CONE, 32).rotate([-90, 0, 0]).translate([x, y_top - POLE_CONE + 0.01, z])
    # channel floor relief for the pin under the retracting bolt (pin bottom 0.5 under the bolt, the channel 0.35)
    zt = g["Z_T"]; rf = g["RF"]
    for s in (-1, 1):
        a0, a1 = sorted((s * (rf - bl.TRAVEL - PIN_R * 2 - 0.5), s * (g["X_OUT"] + 1)))
        t = t - box(a0, a1, bl.BOLT_Y0 - bl.CH_CLR - PIN_DOWN - 0.3, bl.BOLT_Y0 - bl.CH_CLR + 0.01,
                    zt - bl.BOLT_HZ - bl.CH_CLR, zt + bl.BOLT_HZ + bl.CH_CLR)
    return t if clip is None else t ^ clip


bl.slider, bl.pocket_cut, bl.rgeom, bl.carrier_zone, bl.xt60_carrier, bl.cartridge = \
    slider, pocket_cut, rgeom, carrier_zone, xt60_carrier, cartridge
bl.REST_GAP = 0.0                    # unused: rgeom2 sets the shelf from the pin


# ------------------------------------------------------------------ spring (visual, bought) and blades
def spring_mesh(g, turns=12, wire=0.3, od=3.0):
    """pen spring at its rest length, centred x 0, on the slot axis; a pose scales it in x."""
    L = 2 * (bl.TAB_IN + bl.SPR_DEPTH)
    rm = (od - wire) / 2
    n = turns * 20
    pts = np.array([[-L / 2 + L * i / n, bl.SPR_YC + rm * math.cos(2 * math.pi * turns * i / n),
                     g["Z_T"] + rm * math.sin(2 * math.pi * turns * i / n)] for i in range(n + 1)])
    segs = []
    for a, b in zip(pts[:-1], pts[1:]):
        c = trimesh.creation.cylinder(radius=wire / 2, segment=[a, b], sections=8)
        segs.append(c)
    return trimesh.util.concatenate(segs), L


def blades(g):
    """the v4 heads' two blades (3 thick, 13 wide in z), open inner faces at 11, reach 6 into the recess (finger_model)."""
    return [box(*sorted((s * 11.0, s * 14.0)), -20.0, 6.0, g["Z_T"] - 6.5, g["Z_T"] + 6.5) for s in (1, -1)]


# ------------------------------------------------------------------ build + checks
def build():
    B = bl.build("scout")
    g, r = B["g"], B["r"]
    B["poles"] = poles(r)
    B["female"] = bl.xt60_female(g)
    return B


def receiver_all(B):
    return U(B["recv"], B["carrier"], B["cap"], B["strips"], B["male"], *B["poles"], *B.get("rskids", []))


def cart_all(B):
    return U(B["cart"], B["female"], *B.get("cstrips", []))


def checks(B):
    g, r = B["g"], B["r"]
    R = receiver_all(B); C = cart_all(B)
    res = {}
    # static fit, latched
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
    # squeeze: sliders + pins 0 .. TRAVEL vs everything
    w = dict(vs_cart=0.0, vs_recv=0.0, vs_each=0.0)
    for k in range(9):
        rt = bl.TRAVEL * k / 8 - (0.02 if k == 8 else 0.0)
        a = slider(g, rt, 1) + bolt_pin(g, rt, 1); b = slider(g, rt, -1) + bolt_pin(g, rt, -1)
        w["vs_cart"] = max(w["vs_cart"], vol(a ^ C), vol(b ^ C)); w["vs_recv"] = max(w["vs_recv"], vol(a ^ R), vol(b ^ R))
        w["vs_each"] = max(w["vs_each"], vol(a ^ b))
    res["squeeze_mm3"] = {k: round(v, 3) for k, v in w.items()}
    # free insertion: the least retraction that clears the receiver at each depth (= what the lead-ins cam the bolts to)
    rows, worst_c = [], 0.0
    for d in np.arange(-40.0, 0.001, 0.5):
        worst_c = max(worst_c, vol(C.translate([0, d, 0]) ^ R))
        rr = []
        for s in (1, -1):
            S = lambda rt: (slider(g, rt, s) + bolt_pin(g, rt, s)).translate([0, d, 0])
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
    # held insert / pull: squeezed sliders, cartridge 0 .. 40 below the seat
    A = C + slider(g, bl.TRAVEL, 1) + slider(g, bl.TRAVEL, -1) + bolt_pin(g, bl.TRAVEL, 1) + bolt_pin(g, bl.TRAVEL, -1)
    res["gripped_pull_mm3"] = round(max(vol(A.translate([0, -k, 0]) ^ R) for k in np.arange(0, 40.01, 0.5)), 3)
    # pull-out: the cartridge drops until a pin bears; the PCTG bolt must not touch first
    first, bar_first = None, None
    for d in np.arange(0.0, 2.0, 0.02):
        if first is None and vol(bolt_pin(g, 0, 1).translate([0, -d, 0]) ^ B["recv"]) > 0.001: first = round(float(d), 2)
        if bar_first is None and vol(slider(g, 0, 1).translate([0, -d, 0]) ^ B["recv"]) > 0.001: bar_first = round(float(d), 2)
    res["pull_drop_to_pin_mm"] = first
    res["pull_drop_to_bolt_pctg_mm"] = bar_first
    # the wedge: pin on the 20 deg ramp, steel on PCTG
    mu = 0.2
    res["wedge"] = dict(angle_deg=20.0, mu_steel_pctg_E=mu, outward_per_N_pull=round(math.sin(bl.UNDER) - mu * math.cos(bl.UNDER), 3),
                        v4_outward_per_N_pull=round(math.sin(math.radians(15)) - 0.3 * math.cos(math.radians(15)), 3))
    # engagement order on the way in (depth below the seat where each starts)
    res["engage_depth_mm"] = dict(
        guide_poles=round(g["Y_ZONE"] - (r["Y_DK1"] - POLE_L), 2),
        xt60=round(bl.XM_IL - 1.5, 2),
        bolts_on_lead_in=round(bl.BOLT_Y1 - bl.Y_RB, 2))
    res["travel"] = dict(travel=bl.TRAVEL, needed=bl.OVERLAP + 0.3, margin=round(bl.TRAVEL - bl.OVERLAP - 0.3, 2))
    k = bl.SPRING_K; L0 = 2 * (bl.TAB_IN + bl.SPR_DEPTH); L1 = 2 * (bl.RIB_HX + bl.SPR_DEPTH)
    res["spring_N"] = dict(rest=round(k * (bl.SPRING_FREE - L0), 2), squeezed=round(k * (bl.SPRING_FREE - L1), 2),
                           rest_len=L0, squeezed_len=L1)
    # blades: open vs cartridge and tabs, squeezed vs cartridge
    Fo = U(*blades(g))
    res["blades_open_vs_cart_mm3"] = round(vol(Fo ^ B["cart"]), 3)
    res["blades_open_vs_tabs_mm3"] = round(vol(Fo ^ (slider(g, 0, 1) + slider(g, 0, -1))), 3)
    Fs = U(*[b.translate([-s * (11.0 - (bl.TAB_OUT - bl.TRAVEL)), 0, 0]) for b, s in zip(blades(g), (1, -1))])
    res["blades_squeezed_vs_cart_mm3"] = round(vol(Fs ^ B["cart"]), 3)
    return res, rows


def tm(m):
    return bl.tm(m)


def save(B):
    g = B["g"]
    out = {"latch5_receiver": B["recv"], "latch5_carrier": B["carrier"], "latch5_carrier_cap": B["cap"],
           "latch5_cartridge": B["cart"], "latch5_slider": slider(g, 0, 1),
           "latch5_xt60_male": B["male"], "latch5_xt60_female": B["female"], "latch5_pack": bl.pack_body(g)}
    for i, p in enumerate(B["poles"]): out[f"latch5_guide_pole_{'pn'[i]}"] = p
    out["latch5_bolt_pin"] = bolt_pin(g, 0, 1)
    for nm, m in zip(("latch5_receiver_strip_" + k for k in ("a", "b")), [B["strips"]] if not isinstance(B["strips"], list) else B["strips"]):
        out[nm] = m
    paths = {}
    for nm, m in out.items():
        d = os.path.join(ROOT, "parts", nm); os.makedirs(d, exist_ok=True)
        t = tm(m); t.export(os.path.join(d, "part.stl")); paths[nm] = t
    return paths


def main():
    B = build()
    res, rows = checks(B)
    paths = save(B)
    g = B["g"]
    res["mass_g"] = {nm: round(abs(t.volume) / 1000 * (RHO["steel"] if ("pin" in nm or "pole" in nm) else RHO["PCTG"]), 2)
                     for nm, t in paths.items() if not any(k in nm for k in ("xt60", "pack"))}
    res["insert_rows"] = rows
    res["geom"] = {k: round(float(v), 3) for k, v in g.items() if isinstance(v, (int, float))}
    res["rgeom"] = {k: round(float(v), 3) for k, v in B["r"].items() if isinstance(v, (int, float)) and k not in res["geom"]}
    res["rest_stop_mm3_at_0.1_out"] = round(vol(slider(g, -0.1, 1) ^ B["cart"]), 3)   # > 0: the lugs stop the slider at rest
    res["page_parts"] = write_page(B, res)
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("insert_rows", "geom", "rgeom", "page_parts")}, indent=1))


# ------------------------------------------------------------------ page meshes (page frame, containers cut at the tab line)
def page_matrix(zt):
    return np.array([[1, 0, 0, 0], [0, 0, -1, zt], [0, 1, 0, 0], [0, 0, 0, 1.0]])


VIEW = [  # name, group, colour, opacity, cut (stepped: the tab line, and the pole plane at the lead end), role, material
    ("receiver", "receiver (on the drone), cut away", "#9fb4cc", None, True, "structure", "PCTG"),
    ("receiver_strips", "receiver (on the drone), cut away", "#3a3a3a", None, True, "TPU", "TPU"),
    ("receiver_skid_nose", "receiver (on the drone), cut away", "#3a3a3a", None, True, "TPU", "TPU"),
    ("carrier", "receiver (on the drone), cut away", "#7f93ab", None, True, "structure", "PCTG"),
    ("carrier_cap", "receiver (on the drone), cut away", "#7f93ab", None, True, "structure", "PCTG"),
    ("xt60_male", "receiver (on the drone), cut away", "#e8b923", None, False, "bought", None),
    ("guide_pole_p", "guide poles (steel)", "#e3e7ec", None, False, "bought", None),
    ("guide_pole_n", "guide poles (steel)", "#e3e7ec", None, False, "bought", None),
    ("cartridge", "cartridge (on the pack), cut away", "#4f9d8f", None, True, "structure", "PCTG"),
    ("cartridge_strip_px", "cartridge (on the pack), cut away", "#3a3a3a", None, True, "TPU", "TPU"),
    ("cartridge_strip_nx", "cartridge (on the pack), cut away", "#3a3a3a", None, True, "TPU", "TPU"),
    ("pack", "cartridge (on the pack), cut away", "#555b66", 0.35, True, "bought", None),
    ("xt60_female", "cartridge (on the pack), cut away", "#e8b923", None, False, "bought", None),
    ("slider_p", "latch (moving)", "#ff8a1f", None, False, "structure", "PCTG"),
    ("slider_n", "latch (moving)", "#ff8a1f", None, False, "structure", "PCTG"),
    ("bolt_pin_p", "latch (moving)", "#e3e7ec", None, False, "bought", None),
    ("bolt_pin_n", "latch (moving)", "#e3e7ec", None, False, "bought", None),
    ("spring", "latch (moving)", "#c9783f", None, False, "bought", None),
    ("blade_p", "grabber blades (box arm head, unchanged)", "#d64545", None, False, "structure", None),
    ("blade_n", "grabber blades (box arm head, unchanged)", "#d64545", None, False, "structure", None),
]


def view_meshes(B):
    g, r = B["g"], B["r"]; zt = g["Z_T"]
    src = dict(receiver=B["recv"], receiver_strips=B["strips"], carrier=B["carrier"], carrier_cap=B["cap"], xt60_male=B["male"],
               guide_pole_p=B["poles"][1], guide_pole_n=B["poles"][0], cartridge=B["cart"], pack=bl.pack_body(g),
               xt60_female=B["female"], slider_p=slider(g, 0, 1), slider_n=slider(g, 0, -1),
               bolt_pin_p=bolt_pin(g, 0, 1), bolt_pin_n=bolt_pin(g, 0, -1))
    if B.get("rskids"): src["receiver_skid_nose"] = B["rskids"][0]
    for k, m in zip(("px", "nx"), B.get("cstrips", [])): src[f"cartridge_strip_{k}"] = m
    bp, bn = blades(g); src["blade_p"], src["blade_n"] = bp, bn
    # kept: past the tab line (the closed-end half off: the pockets in section), and at the lead end only x < POLE_X (the +x
    # pole's boss and carrier split along the pole axis, so the pole shows in its bore)
    half = box(-500, 500, -500, 500, zt, 1000) - box(POLE_X, 500, -500, 500, g["Z_P1"] - 3.0, 1000)
    M = page_matrix(zt)
    out = {}
    for nm, grp, col, op, cut, role, mat in VIEW:
        if nm not in src: continue
        m = src[nm] ^ half if cut else src[nm]
        if m.is_empty(): continue
        t = tm(m); t.apply_transform(M); out[nm] = t
    sp, L = spring_mesh(g); sp.apply_transform(M); out["spring"] = sp
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
        if mat or steel:
            vol_ = abs(meshes[nm].volume) if meshes[nm].is_volume else 0.0
            mass = round(vol_ / 1000 * (RHO["steel"] if steel else RHO.get(mat, 1.23)), 2)
        parts.append(dict(id=f"latch5_view_{nm}", name=nm.replace("_", " "), role=role, group=grp, color=col, opacity=op,
                          mesh=os.path.relpath(f, ROOT), pose=None, material=mat if not steel else None, tag="C",
                          mass_g=None if cut else mass, version="latch v5 (10 Nov)", source="parts/latch5/make.py"))
    return parts


if __name__ == "__main__":
    main()
