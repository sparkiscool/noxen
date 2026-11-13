"""Swap arm v4 (cartridge long side along the axle, bay modules): every part as a mesh in wheel coords (W), the swap-cycle timeline, the state -> transforms map, and
an exact clearance sweep (python-fcl distances on decimated meshes, manifold3d volumes on anything that touches).
python arm_kinematics.py [dt] [all|axle|gears]  -> out/arm_sweep.json  (per-state clearances, overlaps, allowed contacts)
  all   (default): every part, the axle drive as built (80T sun + spliced 6 mm loop).
  axle  : only pairs with the axle drive (80T sun, loop, 20T), no bay modules (fast).
  gears : axle drive option 1 (printed m1 gears, 111T sun : 28T, C 69.5) in place of the belt; only the gear pairs, and
          gear_slots(): every 6-wide axial window in Y 50..80 for the 111T, with what it crosses.
Axle drive v4 (17 Oct): the sun is FIXED (planetary: the axle motor rides on the arm and walks round it), so p80 / the
111T are in the "fixed" group. Hub v4 (17 Oct): the 80T sits on the fixed sun post (arm_cad.sun_post, bolted to the stub,
through the hub's Ø45.5 hole); the bearing cap and the slip ring are gone (leads twist inside the post's bore).
v4c (17 Oct 02:02): 11 universal bay modules at 30 deg, bolted to each other (lugs + dovetails), no link plates.
Hub v5 (17 Oct): the detent tower is drawn into the hub (arm_layout.HUB_DETENT); the sweep loads out/box_hub_v5.stl then.
Stand v5e / stub v3 (18 Oct, arm chat): the cycle picks and returns at the mounted bay L.DEMO_BAY (-90): A picks there, the arm
turns +90 (B under the drone), B pulls, +90 (A inserts), -270 (B to the -90 bay), +90 home; phi -180..90 (soft limits -278 / +143).
The stub's 4 M5 button heads are a part (stub_bolts)."""
import sys, os, json, math, time
import numpy as np, trimesh, manifold3d as mf
import fast_simplification as fsimp
import arm_layout as L, arm_cad as A
from arm_cad import box, cylY, tm, man, rsquare

OUT = "out"
HERE_K = os.path.dirname(os.path.abspath(__file__))
YC = L.Y_C
MODE = "all"

# ------------------------------------------------------------------ meshes (W coords, "home" placement)
def load_dec(path, target=30000):
    m = trimesh.load(path, force="mesh", process=False)
    m.merge_vertices()                      # STL triangles share no vertices: the decimator needs a connected mesh
    if len(m.faces) > target:
        v, f = fsimp.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32), target_count=target)
        m = trimesh.Trimesh(v, f, process=False)
    return m

def cartridge_local(squeeze=0.0):
    """cartridge v3 as a solid for the checks, v4 orientation (long side along Y, closed end toward -Y), at the top
    position with its bottom at Z 0: body X +-19 x Y_C-37.5..+41.5 x 48; guide rails X +-19..+-20 over Z 38..48 full
    length; recess X +-14.5 x Y_C +-7 x 7.65 open below with the two tabs (outer faces X +-(10 - squeeze)), each with the
    grip pocket on its outer face (8 along Y x 4 along Z, 3 deep, centred Z 4)."""
    y0, y1 = YC - L.Z_TABC, YC + L.CART_L - L.Z_TABC
    c = box(-L.CART_X_OUT, L.CART_X_OUT, y0, y1, 0.0, L.CART_H)
    c += box(-L.CART_W / 2, L.CART_W / 2, y0, y1, L.CART_RAIL_Y0, L.CART_H)
    c -= box(-14.5, 14.5, YC - 7.0, YC + 7.0, -1.0, 7.65)
    for sgn in (-1, 1):
        x_out = sgn * (10.0 - squeeze); x_in = sgn * (4.0 - squeeze)
        tab = box(min(x_out, x_in), max(x_out, x_in), YC - 7.0, YC + 7.0, -0.01, 7.66)
        pocket = box(min(x_out, x_out - sgn * 3.0), max(x_out, x_out - sgn * 3.0), YC - 4.15, YC + 4.15, 4.0 - 2.15, 4.0 + 2.15)
        c += tab - pocket
    return c

class Part:
    def __init__(self, name, mesh, group, color, printed=True):
        self.name, self.mesh, self.group, self.color, self.printed = name, mesh, group, color, printed

def gear_disc(xc, zc, y0, y1, r_tip, r_in=0.0, n=128):
    """a printed spur gear as its tip cylinder (teeth envelope), optional bore: conservative for clearances."""
    g = cylY(xc, zc, y0, y1, r_tip, n)
    return g - cylY(xc, zc, y0 - 1, y1 + 1, r_in, 48) if r_in > 0 else g

def build_parts_v5():
    """arm v5 (24 Oct, arm-mech): printed V-rails + 3-wheel carriages + 16T rack and pinion (arm_v5_cad / arm_v5_layout), v5 motors,
    the axle at -39 deg, the detent at detent.CHOSEN_V5. The hub is a DRAWN proxy of hub v6 (hub_parts() with the v5 motor seats and
    no v4 tower: sleeve, plate spokes, seats, posts) and each head is its drawn carriage + the v4 gripper (guide block, servo plate,
    servo, fingers, gripper pinion): lower bounds until hub v6 / heads v6 exist. Bay modules: v4i (M) at all 9 angles."""
    import arm_v5_layout as V, arm_v5_cad as C5, detent as DT
    P = {}
    M5 = dict(A=V.MOTORS_V5["A"], B=V.MOTORS_V5["B"], axle=dict(xc=53.93, zc=-43.67, rot=-39.0))
    old = (L.MOTORS, getattr(L, "HUB_DETENT", False))
    L.MOTORS = M5; L.HUB_DETENT = False
    try:
        H = A.hub_parts(); hub = H["drawn"] - H["cuts"]
    finally:
        L.MOTORS, L.HUB_DETENT = old
    P["hub"] = Part("hub v6 proxy (drawn: sleeve, plate, v5 seats, posts)", tm(hub), "arm", "#7f8c8d")
    P["stub"] = Part("stub axle v3", tm(A.stub()), "fixed", "#566573")
    P["stub_bolts"] = Part("stub M5 button heads x4", tm(A.stub_bolts()), "fixed", "#2c3e50", printed=False)
    P["post"] = Part("sun post (fixed)", tm(A.sun_post()), "fixed", "#566573")
    for nm, M in M5.items():
        P[f"motor_{nm}"] = Part(f"NEMA17 {nm}", tm(rsquare(M, L.NEMA / 2, 2.0, L.Y_PLATE0)), "arm", "#2c3e50", printed=False)
        P[f"shaft_{nm}"] = Part(f"shaft {nm}", tm(cylY(M["xc"], M["zc"], L.Y_PLATE0 - 0.01, L.Y_PLATE0 + 2.0, L.NEMA_BOSS / 2) +
                                                  cylY(M["xc"], M["zc"], L.Y_PLATE0, 74.0 if nm != "axle" else L.Y_AXBELT[1] + 2, 2.5)), "arm", "#7f8c8d", printed=False)
    Mx = M5["axle"]
    P["p20_axle"] = Part("20T axle", tm(cylY(Mx["xc"], Mx["zc"], L.Y_AXBELT[0] - 2, L.Y_AXBELT[1] + 2, L.P20_FL / 2)), "arm", "#b7950b", printed=False)
    P["p80"] = Part("80T sun (fixed)", tm(cylY(0, 0, L.Y_P80[0], L.Y_P80[1], L.P80_FL / 2) + cylY(0, 0, L.Y_P80[1], 72.0, L.P80_BOSS / 2)), "fixed", "#b7950b", printed=False)
    y0, y1 = L.Y_AXBELT
    outer = (cylY(0, 0, y0, y1, L.P80_OD / 2 + 1.2) + cylY(Mx["xc"], Mx["zc"], y0, y1, L.P20_OD / 2 + 1.2)).hull()
    inner = (cylY(0, 0, y0 - 1, y1 + 1, L.P80_OD / 2) + cylY(Mx["xc"], Mx["zc"], y0 - 1, y1 + 1, L.P20_OD / 2)).hull()
    P["belt_axle"] = Part("axle loop 122T (spliced 6 mm)", tm(outer - inner), "arm", "#2b2b2b", printed=False)
    for k, yb in enumerate(L.Y_BRG):
        P[f"brg{k}"] = Part(f"6010 bearing {k + 1}", tm(cylY(0, 0, yb, yb + L.BRG_W, L.BRG_OD / 2) - cylY(0, 0, yb - 1, yb + L.BRG_W + 1, L.BRG_ID / 2)), "fixed", "#aab4be", printed=False)
    P["railA"] = Part("rail A v1", tm(C5.rail("A")), "arm", "#95a5a6")
    P["railB"] = Part("rail B v1", tm(A.mirror_x(C5.rail("B"))), "armB", "#95a5a6")
    pin = C5.pinion16()[0]
    P["pin16_A"] = Part("16T pinion A v1", tm(pin), "arm", "#f5b041")
    P["pin16_B"] = Part("16T pinion B v1", tm(A.mirror_x(pin)), "armB", "#f5b041")
    for h in ("A", "B"):
        Cc = C5.carriage(h)
        P[f"head{h}"] = Part(f"carriage {h} (drawn proxy: plate, wing, rack web)", tm(Cc["printed"]), f"head{h}", "#d35400")
        P[f"wheels{h}"] = Part(f"V-wheels {h} v1 x3", tm(Cc["wheels"]), f"head{h}", "#e67e22")
        P[f"axles{h}"] = Part(f"axle stacks {h} (M5, shims, nylocs)", tm(Cc["stack"]), f"head{h}", "#2c3e50", printed=False)
        P[f"bush{h}"] = Part(f"eccentric bush {h} v1", tm(Cc["bush"]), f"head{h}", "#e59866")
        G = A.gripper_v4(); F = A.fingers_and_racks(h)
        g = G["drawn"] if h == "A" else A.mirror_x(G["drawn"])
        P[f"grip{h}"] = Part(f"gripper guide + servo plate {h} (v4)", tm(g), f"head{h}", "#d35400")
        P[f"fingA_{h}"] = Part(f"finger A ({h})", tm(F["A"]), f"head{h}", "#e59866")
        P[f"fingB_{h}"] = Part(f"finger B ({h})", tm(F["B"]), f"head{h}", "#e59866")
        P[f"pinion_{h}"] = Part(f"gripper pinion ({h})", tm(A.pinion_u().translate([-L.P_SH, -L.Y_C, 0])), f"head{h}", "#f5b041")
        P[f"servo_{h}"] = Part(f"servo ({h})", tm(A.servo_v4(h)), f"head{h}", "#1f2a33", printed=False)
    c = getattr(DT, "CHOSEN_V5", None)
    if c is not None:
        Dc = DT.design(c["th"], c["s"], c["a"], c["b"], c["phi_d"], c["psi"]); Sd = DT.arm_solids(Dc, np.array(c["Q"]))
        P["detent_ring"] = Part("detent ring v1", tm(DT.ring(c["th"])), "fixed", "#3d5a80")
        P["detent_arm"] = Part("detent arm + spacer + nut + hook pin", tm(Sd["arm"] + Sd["spacer"] + Sd["nut"] + Sd["hook"]), "arm", "#e8914f")
        P["detent_roller"] = Part("683ZZ roller", tm(Sd["roller"]), "arm", "#aab4be", printed=False)
        P["detent_springs"] = Part("extension springs x2", tm(Sd["springs"][0] + Sd["springs"][1]), "arm", "#c0392b", printed=False)
        P["detent_tower"] = Part("detent tower (hub v6) + anchor pin", tm(Sd["tower"] + Sd["anchor"]), "arm", "#e8914f")
    bm = A.man(load_dec("models/box_baymod_v4i_W.obj", 40000))
    for k_, a in enumerate(L.SCOUT_BAYS):
        P[f"bay{k_}"] = Part(f"bay module v4i {k_} ({a:.0f} deg)", tm(bm.rotate([0, 90.0 - a, 0])), "fixed", "#5b8ac9")
        P[f"strip{k_}"] = Part(f"TPU strip {k_}", tm(A.bay_strip().rotate([0, 90.0 - a, 0])), "fixed", "#2e4053")
    PARTS_CACHE.clear(); PARTS_CACHE.update(P)
    return P


def latch_cartridge_local(squeeze=0.0, pack=None, grip_off=None):
    """the latch standard's cartridge (battery_latch.latch_assembly via swap_system's loader: cartridge, both sliders pulled in by
    `squeeze`, pack, XT60 plug) in cartridge_local's frame: X = -x, Y = Y_C + (z - z_T), Z = y (bottom at 0); a proper rotation."""
    import swap_system as SS
    BL = SS._BL
    if BL is None: return cartridge_local(squeeze)
    pack = pack or CYCLE["pack"]; grip_off = CYCLE["grip_off"] if grip_off is None else grip_off
    b = BL.build(pack); g = b["g"]
    m = b["cart"] + BL.slider(g, squeeze, 1) + BL.slider(g, squeeze, -1) + BL.pack_body(g) + BL.xt60_female(g)
    return m.transform(latch_rot(BL, g, grip_off))


def latch_rot(BL, g, grip_off=0.0):
    """cartridge coords -> cartridge_local: latch d0cd1a1 on (TAB_OFF) the XT60 end faces the arm (-Y): X = x, Y = Y_C + Z_T - z, Z = y
    (the Overseer's 30 grip offset is in its Z_T); older latch: X = -x, Y = Y_C + (z - Z_T) + grip_off, Z = y. Proper rotations."""
    if hasattr(BL, "TAB_OFF"): return np.array([[1.0, 0, 0, 0], [0, 0, -1.0, YC + g["Z_T"]], [0, 1.0, 0, 0]])
    return np.array([[-1.0, 0, 0, 0], [0, 0, 1.0, YC - g["Z_T"] + grip_off], [0, 1.0, 0, 0]])


CYCLE = dict(pack="scout", grip_off=0.0, pick=None, ret=None)   # v7: the pack and the bays of the cycle being swept (None = DB)


CART_LOCAL = cartridge_local                   # v6: latch_cartridge_local


def build_parts_v6():
    """arm v6 (24 Oct, swap arm system worker; swap_system.py): the v5 mechanism on the compact stack (arm_v6_layout: rails on the hub
    motor plate), rails A / B v2, 16T pinions v2, detent ring v3 at detent.CHOSEN_V6, hub v6 (the topo mesh runs/hub/v6/box_hub_v6.stl
    when it exists, else hub_v6's drawn bodies), carriages drawn from the layout (heads v6 do not exist: carriage proxy + v4 gripper),
    the latch standard's cartridge (CART_LOCAL). Bay modules v4i (M) at all 9 angles."""
    import arm_v6_layout as S6
    S6.use("v6")
    import arm_v5_layout as V, arm_v5_cad as C5, detent as DT, arm_v6_cad as C6, hub_v6 as HV
    global CART_LOCAL
    CART_LOCAL = latch_cartridge_local
    P = build_parts_v5()                                                   # the shared parts (stub, motors, belt, bearings, bays, grippers)
    hp = os.path.normpath(os.path.join(HERE_K, "..", "..", "runs", "hub", "v7", "box_hub_v7.stl"))   # hub v7 (system pass); v6 runs superseded
    H = HV.hub_v6_parts()
    if os.path.exists(hp):
        P["hub"] = Part("hub v7 (topo)", load_dec(hp, 120000), "arm", "#7f8c8d")
    else:
        P["hub"] = Part("hub v7 (drawn bodies: sleeve, wall, roof, plate, seats, pads, tower)", tm(H["drawn"] - H["cuts"]), "arm", "#7f8c8d")
    P["railA"] = Part("rail A v2", tm(C6.rail_v2("A")), "arm", "#95a5a6")
    P["railB"] = Part("rail B v2", tm(A.mirror_x(C6.rail_v2("B"))), "armB", "#95a5a6")
    pin = C6.pinion_v2()[0]
    P["pin16_A"] = Part("16T pinion A v2", tm(pin), "arm", "#f5b041")
    P["pin16_B"] = Part("16T pinion B v2", tm(A.mirror_x(pin)), "armB", "#f5b041")
    for h in ("A", "B"):                                                   # carriages on the v6 stack (the layout constants are patched)
        Cc = C5.carriage(h)
        P[f"head{h}"] = Part(f"carriage {h} (drawn proxy: plate, wing, rack web)", tm(Cc["printed"]), f"head{h}", "#d35400")
        P[f"wheels{h}"] = Part(f"V-wheels {h} v1 x3", tm(Cc["wheels"]), f"head{h}", "#e67e22")
        P[f"axles{h}"] = Part(f"axle stacks {h} (M5, shims, nylocs)", tm(Cc["stack"]), f"head{h}", "#2c3e50", printed=False)
        P[f"bush{h}"] = Part(f"eccentric bush {h} v1", tm(Cc["bush"]), f"head{h}", "#e59866")
    c = DT.chosen()
    Dc = DT.design(c["th"], c["s"], c["a"], c["b"], c["phi_d"], c["psi"]); Sd = DT.arm_solids(Dc, np.array(c["Q"]))
    P["detent_ring"] = Part("detent ring v3", tm(DT.ring_v3(c["th"])), "fixed", "#3d5a80")
    P["detent_arm"] = Part("detent arm v2 + carbon roller axle + hook pin", tm(Sd["arm"] + Sd["nut"] + Sd["hook"]), "arm", "#e8914f")
    P["detent_roller"] = Part("683ZZ roller", tm(Sd["roller"]), "arm", "#aab4be", printed=False)
    P["detent_springs"] = Part("extension springs x2", tm(Sd["springs"][0] + Sd["springs"][1]), "arm", "#c0392b", printed=False)
    P["detent_tower"] = Part("detent anchor pin (the tower is hub v6's)", tm(Sd["anchor"]), "arm", "#e8914f", printed=False)
    rec = latch_receiver_local()                                          # the demo bay: the latch standard's receiver (the v4i module holds
    if rec is not None:                                                   # a v3 receiver; the next module version carries the latch's)
        di = demo_idx()
        P[f"bay{di}"] = Part("demo bay: latch receiver + lid + strips + XT60E1-M (battery_latch v1)", tm(rec.transform((rotY(90.0 - DB) @ trans(0, 0, L.R))[:3])),
                             "fixed", "#5b8ac9")
        P.pop(f"strip{di}", None)
    PARTS_CACHE.clear(); PARTS_CACHE.update(P)
    return P


def latch_receiver_local(pack="scout", grip_off=0.0, state="latched", with_pack=False):
    """the latch receiver set (seated cartridge coords) in cartridge_local's frame, bottom face of the seated cartridge at Z 0."""
    import swap_system as SS
    BL = SS._BL
    if BL is None: return None
    d = BL.latch_assembly(pack, state); g = BL.build(pack)["g"]
    m = d["receiver"] + d["receiver_lid"] + d["receiver_strips"] + d["xt60e1m"]
    if with_pack: m = m + d["cartridge"] + d["slider_px"] + d["slider_nx"] + d["pack"] + d["xt60_plug"]
    return m.transform(latch_rot(BL, g, grip_off))


def build_parts_v7(pack="scout", pick=-210.0, ret=30.0):
    """arm v7 (swap_system, hub decisions 24 Oct): the v6 parts on the v7 stack (arm_v7_layout: R 113, rails to r 151, rack tail u -63,
    strokes +13), the cartridge of `pack` (the Overseer gripped 30 toward its closed end), and every one of the 9 bays at R 113 as its latch
    receiver; every bay but the pick and return bays also holds a seated cartridge (what a turning arm must clear). v4i modules out."""
    import arm_v7_layout as S7
    S7.use()
    global CART_LOCAL, YC
    YC = L.Y_C                                                              # v7: the common grip Y_C (arm_v7_layout)
    CYCLE.update(pack=pack, grip_off=S7.OV_GRIP_OFFSET if pack == "overseer" else 0.0, pick=pick, ret=ret)
    P = build_parts_v6()
    for nm in [n for n in P if n.startswith(("bay", "strip"))]: P.pop(nm)
    CART_LOCAL = latch_cartridge_local
    import arm_v7_cad as C7                                                # rails v3 (r 151, tie sockets) and the carbon tie rods
    P["railA"] = Part("rail A v3", tm(C7.rail_v3("A")), "arm", "#95a5a6")
    P["railB"] = Part("rail B v3", tm(A.mirror_x(C7.rail_v3("B"))), "armB", "#95a5a6")
    P["tiesA"] = Part("carbon tie rods A (2 x 3.0)", tm(C7.tie_rods("A")), "arm", "#222222", printed=False)
    P["tiesB"] = Part("carbon tie rods B (2 x 3.0)", tm(A.mirror_x(C7.tie_rods("B"))), "armB", "#222222", printed=False)
    ALLOWED.update({("tiesA", "railA"), ("tiesB", "railB"), ("tiesA", "hub"), ("tiesB", "hub")})
    for kind in ("scout", "overseer"):
        go = S7.OV_GRIP_OFFSET if kind == "overseer" else 0.0
        rec = latch_receiver_local(kind, go); full = latch_receiver_local(kind, go, with_pack=True)
        for a in S7.BAYS[kind]:
            T = (rotY(90.0 - a) @ trans(0, 0, L.R))[:3]
            nm = f"rec{int(a)}"
            seated = a not in (pick, ret)
            P[nm] = Part(f"{kind} bay at {a:g} deg: latch receiver" + (" + a seated cartridge" if seated else ""), tm((full if seated else rec).transform(T)),
                         "fixed", "#5b8ac9" if kind == "scout" else "#9955bb")
    PARTS_CACHE.clear(); PARTS_CACHE.update(P)
    return P


def keys_v7(pick, ret, zbay=None, zdr=None, zdri=None, zw=None):
    """the v5 swap cycle for any pick and return bay: A picks at phi = pick, B under the drone at phi 0, A inserts at phi 90, B returns the
    dead pack at phi = ret - 90, home at phi = pick. Strokes from arm_layout (v7-patched)."""
    zbay = zbay or L.Z0_BAY; zdr = zdr or L.Z0_DRONE; zw = zw or L.Z0_B_WAIT; zdri = zdri or (zdr + 6.0)
    zs = zbay - 12.0; rb = ret - 90.0
    return [(0.0, pick, ZP, ZP, 4, 4), (1.0, pick, ZP, ZP, -1, -1), (2.5, pick, zbay, ZP, -1, -1), (3.0, pick, zbay, ZP, 4, -1), (4.5, pick, ZP, ZP, 4, -1),
            (5.5, 0.0, ZP, ZP, 4, -1), (6.5, 0.0, ZP, zw, 4, -1), (7.0, 0.0, ZP, zdr, 4, -1), (7.5, 0.0, ZP, zdr, 4, 4),
            (9.0, 0.0, ZP, ZP, 4, 4), (10.0, 90.0, ZP, ZP, 4, 4), (10.9, 90.0, zdri - 12.0, ZP, 4, 4), (11.5, 90.0, zdri, ZP, 4, 4),
            (12.0, 90.0, zdri, ZP, -1, 4), (13.5, 90.0, ZP, ZP, -1, 4), (14.0, 90.0, ZP, ZP, 4, 4), (17.0, rb, ZP, ZP, 4, 4),
            (17.9, rb, ZP, zs, 4, 4), (18.5, rb, ZP, zbay, 4, 4), (19.0, rb, ZP, zbay, 4, -1), (20.5, rb, ZP, ZP, 4, -1), (21.0, rb, ZP, ZP, 4, 4),
            (22.0, pick, ZP, ZP, 4, 4)]


def build_parts(mode="all"):
    if mode == "v5": return build_parts_v5()
    if mode == "v6": return build_parts_v6()
    if mode.startswith("v7"):                                               # v7:<pack>:<pick>:<ret>, e.g. v7:overseer:-150:-60
        _, pk, pi, re_ = (mode.split(":") + ["scout", "-210", "30"])[:4]
        return build_parts_v7(pk, float(pi), float(re_))
    P = {}
    hub_path = f"{OUT}/box_hub_v5.stl" if getattr(L, "HUB_DETENT", False) else f"{OUT}/box_hub_v4.stl"   # generated hub, W coords (arm_run_hub.py)
    P["hub"] = Part("hub", load_dec(hub_path) if os.path.exists(hub_path) else tm(A.hub_parts()["drawn"] - A.hub_parts()["cuts"]), "arm", "#7f8c8d")
    P["stub"] = Part("stub axle v3", tm(A.stub()), "fixed", "#566573")
    P["stub_bolts"] = Part("stub M5 button heads x4", tm(A.stub_bolts()), "fixed", "#2c3e50", printed=False)
    P["post"] = Part("sun post (fixed)", tm(A.sun_post()), "fixed", "#566573")
    # the two bars: A's along +Z in the arm frame, B's in head B's own frame (group armB: the arm frame turned -90 about Y)
    P["bar_A"] = Part("2040 bar A (112)", tm(A.bar_2040(*L.BAR_A)), "arm", "#95a5a6", printed=False)
    P["bar_B"] = Part("2040 bar B (112)", tm(A.bar_2040(*L.BAR_B)), "armB", "#95a5a6", printed=False)
    P["idlerA"] = Part("idler end A", tm(A.idler_end("A")), "arm", "#5d6d7e")
    P["idlerB"] = Part("idler end B", tm(A.idler_end("B")), "armB", "#5d6d7e")
    ybelt = {"A": L.Y_HBELT, "B": L.Y_HBELT_B, "axle": L.Y_AXBELT}
    for nm, M in L.MOTORS.items():
        P[f"motor_{nm}"] = Part(f"NEMA17 {nm}", tm(rsquare(M, L.NEMA / 2, 2.0, L.Y_PLATE0)), "arm", "#2c3e50", printed=False)
        y0, y1 = ybelt[nm][0] - 2, ybelt[nm][1] + 2
        if not (nm == "axle" and mode == "gears"):
            P[f"p20_{nm}"] = Part(f"20T {nm}", tm(cylY(M["xc"], M["zc"], y0, y1, L.P20_FL / 2)), "arm", "#b7950b", printed=False)
        P[f"shaft_{nm}"] = Part(f"shaft {nm}", tm(cylY(M["xc"], M["zc"], L.Y_PLATE0 - 0.01, L.Y_PLATE0 + 2.0, L.NEMA_BOSS / 2) + cylY(M["xc"], M["zc"], L.Y_PLATE0, y1, 2.5)), "arm", "#7f8c8d", printed=False)
    # head belts as loops in each head's own frame: motor pulley (below the axle on the belt line) to the idler at Z 82
    r = L.P20_OD / 2
    for h, xa, zm, zi, yb, grp in (("A", L.X_HBELT_C, L.Z_MOTOR_A, L.Z_IDLER, L.Y_HBELT, "arm"), ("B", L.X_HBELT_C_B, L.Z_MOTOR_B, L.Z_IDLER_B, L.Y_HBELT_B, "armB")):
        P[f"idler_{h}"] = Part(f"idler pulley {h}", tm(cylY(xa, zi, yb[0] - 1.5, yb[1] + 1.5, L.IDLER_OD / 2)), grp, "#b7950b", printed=False)
        outer = (cylY(xa, zm, yb[0], yb[1], r + 1.2) + cylY(xa, zi, yb[0], yb[1], r + 1.2)).hull()
        inner = (cylY(xa, zm, yb[0] - 1, yb[1] + 1, r) + cylY(xa, zi, yb[0] - 1, yb[1] + 1, r)).hull()
        P[f"belt_{h}"] = Part(f"head belt {h}", tm(outer - inner), grp, "#2b2b2b", printed=False)
    Mx = L.MOTORS["axle"]
    if mode == "gears":
        # option 1: 111T sun (fixed, bolted on the 80T's Ø36 circle in its place), 28T on the axle motor's 5 mm D shaft
        g = L.GEAR_REJ; y0, y1 = g["Y"]
        P["sun111"] = Part("111T sun (m1, fixed)", tm(gear_disc(0, 0, y0, y1, g["r_tip_sun"], 4.0)), "fixed", "#b7950b")
        P["pin28"] = Part("28T pinion (m1)", tm(gear_disc(Mx["xc"], Mx["zc"], y0, y1, g["r_tip_pin"], 2.5)), "arm", "#f5b041")
    else:
        # option 2 (DEFAULT): the 80T sun, FIXED (planetary), and the lap-spliced 122T loop (its shape is fixed in the arm frame)
        P["p80"] = Part("80T sun (fixed)", tm(cylY(0, 0, L.Y_P80[0], L.Y_P80[1], L.P80_FL / 2) + cylY(0, 0, L.Y_P80[1], 72.0, L.P80_BOSS / 2)), "fixed", "#b7950b", printed=False)
        y0, y1 = L.Y_AXBELT
        outer = (cylY(0, 0, y0, y1, L.P80_OD / 2 + 1.2) + cylY(Mx["xc"], Mx["zc"], y0, y1, L.P20_OD / 2 + 1.2)).hull()
        inner = (cylY(0, 0, y0 - 1, y1 + 1, L.P80_OD / 2) + cylY(Mx["xc"], Mx["zc"], y0 - 1, y1 + 1, L.P20_OD / 2)).hull()
        P["belt_axle"] = Part("axle loop 122T (spliced 6 mm)", tm(outer - inner), "arm", "#2b2b2b", printed=False)
    for k, y0 in enumerate(L.Y_BRG):                                                  # 6010-2RS x2 on the stub
        P[f"brg{k}"] = Part(f"6010 bearing {k + 1}", tm(cylY(0, 0, y0, y0 + L.BRG_W, L.BRG_OD / 2) - cylY(0, 0, y0 - 1, y0 + L.BRG_W + 1, L.BRG_ID / 2)), "fixed", "#aab4be", printed=False)
    for h in ("A", "B"):
        F = A.fingers_and_racks(h)
        hp = f"{OUT}/box_head_{h}_v4_W.stl"
        P[f"head{h}"] = Part(f"head {h}", load_dec(hp) if os.path.exists(hp) else tm(A.head_parts(h)["drawn"] - A.head_parts(h)["cuts"]), f"head{h}", "#d35400")
        P[f"fingA_{h}"] = Part(f"finger A ({h})", tm(F["A"]), f"head{h}", "#e59866")
        P[f"fingB_{h}"] = Part(f"finger B ({h})", tm(F["B"]), f"head{h}", "#e59866")
        pn = A.pinion_u().translate([-L.P_SH, -L.Y_C, 0])                      # pinion about its own axis (rotated per state)
        P[f"pinion_{h}"] = Part(f"pinion ({h})", tm(pn), f"head{h}", "#f5b041")
        P[f"servo_{h}"] = Part(f"servo ({h})", tm(A.servo_v4(h)), f"head{h}", "#1f2a33", printed=False)
    # arm auto-centering detent (next arm version, detent.py): ring fixed on the 80T, roller arm + springs + hub v5 tower turn
    import detent as DT
    c = DT.CHOSEN; Dc = DT.design(c["th"], c["s"], c["a"], c["b"], c["phi_d"], c["psi"]); Sd = DT.arm_solids(Dc, np.array(c["Q"]))
    P["detent_ring"] = Part("detent ring", tm(DT.ring(c["th"])), "fixed", "#3d5a80")
    P["detent_arm"] = Part("detent arm + spacer + nut + hook pin", tm(Sd["arm"] + Sd["spacer"] + Sd["nut"] + Sd["hook"]), "arm", "#e8914f")
    P["detent_roller"] = Part("683ZZ roller", tm(Sd["roller"]), "arm", "#aab4be", printed=False)
    P["detent_springs"] = Part("extension springs x2", tm(Sd["springs"][0] + Sd["springs"][1]), "arm", "#c0392b", printed=False)
    P["detent_tower"] = Part("detent tower (hub v5) + anchor pin", tm(Sd["tower"] + Sd["anchor"]), "arm", "#e8914f")
    if mode == "all":                                                                 # bay modules: none reach r < 99.5 (the axle drive is inside r 76)
        bp = f"{OUT}/box_baymod_v4_W.stl"
        if os.path.exists(bp): bm = A.man(load_dec(bp, 40000))
        else:
            B = A.bay_module_parts(); bm = B["drawn"] - B["cuts"]
        for k_, a in enumerate(L.SCOUT_BAYS):
            P[f"bay{k_}"] = Part(f"bay module {k_} ({a:.0f} deg)", tm(bm.rotate([0, 90.0 - a, 0])), "fixed", "#5b8ac9")
            P[f"strip{k_}"] = Part(f"TPU strip {k_}", tm(A.bay_strip().rotate([0, 90.0 - a, 0])), "fixed", "#2e4053")
        # v4c: modules bolt straight to each other (lugs + dovetails), no link plates
    # no casing (retired, Spark 16 Oct 21:09): the bay modules are the frame
    PARTS_CACHE.clear(); PARTS_CACHE.update(P)
    return P

FOCUS = {"axle": ("p80", "belt_axle", "p20_axle", "post", "detent_ring", "detent_arm", "detent_roller", "detent_springs", "detent_tower"), "gears": ("sun111", "pin28")}

# ------------------------------------------------------------------ timeline (s): phi = direction head A points (deg from +X)
ZP, ZBAY, ZDR, ZW = L.Z0_A_HOME, L.Z0_BAY, L.Z0_DRONE, L.Z0_B_WAIT
DB = L.DEMO_BAY                                # -90: the mounted bay (stand v5e cradle)
def demo_idx():
    """index of the demo bay in L.SCOUT_BAYS (the module part name is bay<idx>)."""
    return min(range(len(L.SCOUT_BAYS)), key=lambda k: abs(((L.SCOUT_BAYS[k] - DB) + 180) % 360 - 180))
KEYS = [  # t, phi, zA, zB, fA, fB   (phi = head A's direction; head B points at phi + 90)
    (0.0, DB, ZP, ZP, 4, 4), (1.0, DB, ZP, ZP, -1, -1), (2.5, DB, ZBAY, ZP, -1, -1), (3.0, DB, ZBAY, ZP, 4, -1), (4.5, DB, ZP, ZP, 4, -1),
    (5.5, DB + 90, ZP, ZP, 4, -1), (6.5, DB + 90, ZP, ZW, 4, -1), (7.0, DB + 90, ZP, ZDR, 4, -1), (7.5, DB + 90, ZP, ZDR, 4, 4),
    (9.0, DB + 90, ZP, ZP, 4, 4), (10.0, DB + 180, ZP, ZP, 4, 4), (11.5, DB + 180, ZDR, ZP, 4, 4), (12.0, DB + 180, ZDR, ZP, -1, 4),
    (13.5, DB + 180, ZP, ZP, -1, 4), (14.0, DB + 180, ZP, ZP, 4, 4), (17.0, DB - 90, ZP, ZP, 4, 4), (18.5, DB - 90, ZP, ZBAY, 4, 4),
    (19.0, DB - 90, ZP, ZBAY, 4, -1), (20.5, DB - 90, ZP, ZP, 4, -1), (21.0, DB - 90, ZP, ZP, 4, 4), (22.0, DB, ZP, ZP, 4, 4)]
T_END = 22.0
KEYS_V4 = KEYS
# v5 (24 Oct, arm-mech): XT60 + hook latch: the drone insert pushes 6 further (E) and the last 12 of every insert run at 20 mm/s (E)
ZDRI = ZDR + 6.0; ZBAY_SLOW = ZBAY - 12.0
KEYS_V5 = [
    (0.0, DB, ZP, ZP, 4, 4), (1.0, DB, ZP, ZP, -1, -1), (2.5, DB, ZBAY, ZP, -1, -1), (3.0, DB, ZBAY, ZP, 4, -1), (4.5, DB, ZP, ZP, 4, -1),
    (5.5, DB + 90, ZP, ZP, 4, -1), (6.5, DB + 90, ZP, ZW, 4, -1), (7.0, DB + 90, ZP, ZDR, 4, -1), (7.5, DB + 90, ZP, ZDR, 4, 4),
    (9.0, DB + 90, ZP, ZP, 4, 4), (10.0, DB + 180, ZP, ZP, 4, 4), (10.9, DB + 180, ZDRI - 12.0, ZP, 4, 4), (11.5, DB + 180, ZDRI, ZP, 4, 4),
    (12.0, DB + 180, ZDRI, ZP, -1, 4), (13.5, DB + 180, ZP, ZP, -1, 4), (14.0, DB + 180, ZP, ZP, 4, 4), (17.0, DB - 90, ZP, ZP, 4, 4),
    (17.9, DB - 90, ZP, ZBAY_SLOW, 4, 4), (18.5, DB - 90, ZP, ZBAY, 4, 4),
    (19.0, DB - 90, ZP, ZBAY, 4, -1), (20.5, DB - 90, ZP, ZP, 4, -1), (21.0, DB - 90, ZP, ZP, 4, 4), (22.0, DB, ZP, ZP, 4, 4)]
# v6 (swap_system): the head stops at the seat (95.6); the commanded 1.0 over-travel (arm_v6_layout.Z0_INSERT_MAX) is the arm's elastic lag
def _keys_v6():
    return list(KEYS_V5)
T_LAND = 6.5                                   # the drone lands (its dead pack appears at the top) while B waits open under the seat
PHASES = [(0, "both heads open: A at the -90 deg bay, B at the 0 deg position"), (1, "A extends into the bay"), (2.5, "A grips the charged pack"),
          (3, "A retracts with it"), (4.5, "arm turns +90: B under the drone, A (carrying) at 0 deg"), (5.5, "B rises to wait 1 mm under the seat"),
          (6.5, "drone lands; B reaches up to the dead pack"), (7, "B grips the dead pack"), (7.5, "B retracts"),
          (9, "arm turns +90: A to the top, B to 180 deg"), (10, "A inserts the charged pack"), (11.5, "A releases: latched"),
          (12, "A retracts"), (13.5, "A closes"), (14, "arm turns -270 (through the top): B to the free -90 deg bay"),
          (17, "B extends into the bay"), (18.5, "B releases into the bay"), (19, "B retracts"), (20.5, "B closes"),
          (21, "arm turns +90 home (A at the -90 bay)"), (22, "done")]
def pack_owner(t):
    ch = "bay0" if t < 3.0 else ("A" if t < 12.0 else "drone")
    de = None if t < T_LAND else ("drone" if t < 7.5 else ("B" if t < 19.0 else "bay0"))
    return ch, de

def sample_times(dt=0.2, dphi=2.0):
    """every dt, every keyframe, and every dphi degrees of every turn (a pack corner's sliver is a few degrees wide)"""
    ts = set([k[0] for k in KEYS] + [round(float(x), 3) for x in np.arange(0, T_END + 0.0001, dt)])
    for a, b in zip(KEYS[:-1], KEYS[1:]):
        if a[1] != b[1]:
            n = int(abs(b[1] - a[1]) / dphi)
            ts |= set(round(a[0] + (b[0] - a[0]) * i / n, 4) for i in range(n + 1))
    return sorted(ts)

def state(t):
    ks = KEYS
    for i in range(len(ks) - 1):
        if ks[i][0] <= t <= ks[i + 1][0]:
            a, b = ks[i], ks[i + 1]; u = 0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
            return dict(zip(("phi", "zA", "zB", "fA", "fB"), [a[k] + u * (b[k] - a[k]) for k in range(1, 6)]))
    return dict(zip(("phi", "zA", "zB", "fA", "fB"), ks[-1][1:]))

# ------------------------------------------------------------------ transforms
def rotY(deg):
    T = trimesh.transformations.rotation_matrix(math.radians(deg), [0, 1, 0]); return T
def trans(x, y, z):
    return trimesh.transformations.translation_matrix([x, y, z])
R180 = rotY(180.0)

RB = rotY(-90.0)                                # head B's frame in the arm frame: bar along the arm's -X, B's +X = arm +Z

def transforms(t):
    """dict name -> 4x4 world transform for every part (W coords) at time t, plus the cartridges (built per state)."""
    s = state(t); Tarm = rotY(90.0 - s["phi"])
    TA = Tarm @ trans(0, 0, s["zA"]); TB = Tarm @ RB @ trans(0, 0, s["zB"])
    T = {}
    for nm, p in PARTS_CACHE.items():
        if p.group == "arm": T[nm] = Tarm
        elif p.group == "armB": T[nm] = Tarm @ RB
        elif p.group == "fixed" and not nm.startswith("bay") and not nm.startswith("stubtile"): T[nm] = np.eye(4)
    for nm in PARTS_CACHE:
        if nm.startswith(("bay", "strip", "link", "springs")): T[nm] = np.eye(4)        # bay modules and strips are built in place
    for nm, p in PARTS_CACHE.items():                                      # v5: every part of a head's group rides its head
        if p.group == "headA": T[nm] = TA
        elif p.group == "headB": T[nm] = TB
    if "pin16_A" in PARTS_CACHE:                                           # v5 16T pinions roll with their heads (A in the arm frame, B mirrored in B's)
        import arm_v5_layout as V
        p_ = math.pi * V.PIN_M; gc = V.WEB_U[0] + p_ / 2
        rot = lambda z0: math.radians(-((V.PIN_Z - z0) - gc) / V.PIN_R * 180.0 / math.pi)
        T["pin16_A"] = Tarm @ trans(V.PIN_X, 0, V.PIN_Z) @ rotY(math.degrees(rot(s["zA"])))
        T["pin16_B"] = Tarm @ RB @ trans(-V.PIN_X, 0, V.PIN_Z) @ rotY(-math.degrees(rot(s["zB"])))
    for h, Th, d in (("A", TA, s["fA"]), ("B", TB, s["fB"])):
        sg = 1.0 if h == "A" else -1.0                                       # head B's gripper is mirrored in X
        T[f"head{h}"] = Th; T[f"servo_{h}"] = Th
        T[f"fingA_{h}"] = Th @ trans(sg * d, 0, 0); T[f"fingB_{h}"] = Th @ trans(-sg * d, 0, 0)
        T[f"pinion_{h}"] = Th @ trans(sg * L.P_SH, L.Y_C, 0) @ trimesh.transformations.rotation_matrix(-sg * d / 6.0, [0, 0, 1])
    # cartridges: local = top position with the bottom at Z 0
    ch, de = pack_owner(t); carts = {}
    def at_bay0(which="pick"):                                              # the pick / return bay (v5 / v6: the demo bay -90)
        a = CYCLE.get(which) if CYCLE.get(which) is not None else DB
        return rotY(90.0 - a) @ trans(0, 0, L.R)
    lift = 6.0 if (KEYS is not KEYS_V4 and t >= 11.5) else 0.0             # v5: the drone is held up on the hook-latch plungers after the insert (E)
    def at_drone(): return trans(0, 0, L.R + 5.0 + lift)  # the latched pack sits 5 above the bay seat (the head's over-travel)
    def in_head(h):
        Th = TA if h == "A" else TB; d = s["fA"] if h == "A" else s["fB"]
        return Th @ trans(0, 0, L.CART_BOT_DZ), max(d, 0.0)
    def engaged(Tpack):
        """a head whose cartridge seat coincides with this pack's place is squeezing its tabs by max(d, 0): its fingers may touch"""
        for h, Th, d in (("A", TA, s["fA"]), ("B", TB, s["fB"])):
            Tc = Th @ trans(0, 0, L.CART_BOT_DZ)
            if np.allclose(Tc[:3, 3], Tpack[:3, 3], atol=0.5) and np.allclose(Tc[:3, :3], Tpack[:3, :3], atol=1e-3):
                return max(d, 0.0), f"head{h}"
        return 0.0, None
    for nm, own in (("pack charged", ch), ("pack dead", de)):
        if own is None: continue
        if own == "bay0": Tc = at_bay0("pick" if nm == "pack charged" else "ret"); sq, eh = engaged(Tc); carts[nm] = (Tc, sq, (eh or "bay0") + "@bay0")
        elif own == "drone": Tc = at_drone(); sq, eh = engaged(Tc); carts[nm] = (Tc, sq, (eh or "drone") + "@drone")
        else:
            Tc, sq = in_head(own); carts[nm] = (Tc, sq, f"head{own}@head{own}")
    return T, carts, s

PARTS_CACHE = {}

# ------------------------------------------------------------------ sweep
# axle drive v4: the sun is fixed and must not touch the hub (it bolts to a fixed post through it, not modelled yet);
# ("hub", "p80") and ("hub", "belt_axle") are no longer allowed: any contact there is a real collision.
ALLOWED = {("stub", "post"), ("stub", "stub_bolts"), ("post", "p80"), ("hub", "bar_A"), ("hub", "bar_B"), ("bar_A", "idlerA"), ("bar_B", "idlerB"), ("stub", "casing")}
for m in L.MOTORS:
    ALLOWED |= {("hub", f"motor_{m}"), (f"motor_{m}", f"p20_{m}"), ("hub", f"p20_{m}"), (f"motor_{m}", f"shaft_{m}"), ("hub", f"shaft_{m}"),
                (f"shaft_{m}", f"p20_{m}"), (f"shaft_{m}", f"belt_{m}" if m != "axle" else "belt_axle")}
ALLOWED |= {("belt_axle", "p80"), ("belt_axle", "p20_axle"), ("sun111", "pin28"), ("shaft_axle", "pin28"), ("motor_axle", "pin28")}
for h in ("A", "B"):
    ALLOWED |= {(f"belt_{h}", f"p20_{h}"), (f"belt_{h}", f"idler_{h}"), (f"belt_{h}", f"head{h}"), (f"idler_{h}", f"idler{h}"), (f"belt_{h}", f"idler{h}"),
                (f"servo_{h}", f"head{h}"), (f"servo_{h}", f"pinion_{h}")}
for k in range(len(L.SCOUT_BAYS)):
    ALLOWED |= {(f"bay{k}", "casing"), (f"bay{k}", f"strip{k}")}
for i in range(len(L.SCOUT_BAYS)):
    for j in range(len(L.SCOUT_BAYS)):
        if i != j and abs(((L.SCOUT_BAYS[i] - L.SCOUT_BAYS[j]) + 180) % 360 - 180) < L.PITCH + 1:
            ALLOWED |= {(f"bay{i}", f"bay{j}")}                       # neighbours bolted lug to lug, dovetail in groove
for k in range(2): ALLOWED |= {(f"brg{k}", "stub"), (f"brg{k}", "hub"), ("brg0", "brg1")}
ALLOWED |= {("brg1", "post")}
ALLOWED |= {("detent_ring", "p80"), ("detent_roller", "detent_ring"), ("detent_arm", "detent_roller"), ("detent_arm", "detent_tower"),
            ("detent_arm", "detent_springs"), ("detent_springs", "detent_tower"), ("detent_tower", "hub"), ("detent_springs", "hub")}   # detent: bolted / pinned / rolling                                         # the post's flange clamps bearing 2's inner race

for h in ("A", "B"):                                                     # v5: bolted / meshing / rolling pairs
    ALLOWED |= {("hub", f"rail{h}"), (f"shaft_{h}", f"pin16_{h}"), (f"pin16_{h}", f"head{h}"), (f"rail{h}", f"wheels{h}"), (f"wheels{h}", f"axles{h}"),
                (f"axles{h}", f"head{h}"), (f"bush{h}", f"head{h}"), (f"bush{h}", f"axles{h}"), (f"servo_{h}", f"grip{h}"), (f"pinion_{h}", f"grip{h}"),
                (f"fingA_{h}", f"grip{h}"), (f"fingB_{h}", f"grip{h}"), (f"fingA_{h}", f"pinion_{h}"), (f"fingB_{h}", f"pinion_{h}")}


def allowed(a, b, owners):
    if (a, b) in ALLOWED or (b, a) in ALLOWED: return True
    if a.startswith("rec") and b.startswith("rec"): return True          # v7: bay to bay is the box structure's fit (reported apart), not the arm's
    for c, ownp in owners.items():                    # a pack may touch what holds it: "<holder>@<place>"
        if c in (a, b):
            o = b if a == c else a; own, place = ownp.split("@")
            if o in (f"bay{demo_idx()}", f"strip{demo_idx()}") and place == "bay0": return True   # packs enter and leave the demo bay
            if o.startswith("strip"): return True                              # a pack pressed into a bay's TPU strip (the install preload)
            if own in ("headA", "headB") and o in (f"fingA_{own[-1]}", f"fingB_{own[-1]}"): return True
    return False

FAR = trimesh.transformations.translation_matrix([0, 5000, 0])
SQ_STEPS = [round(x * 0.5, 1) for x in range(0, 9)]        # 0 .. 4 by 0.5

FIT_TOL = 0.5                                  # mm3: a touching pair at or under it is a contact (system_check fit_tol_mm3)
PAIR_MIN = {}                                   # (a, b) -> (min distance over the cycle, t): every checked pair, kept per pair


def sweep(dt=0.1, log=print, mode="all"):
    import trimesh.collision as tc
    import fcl
    P = build_parts(mode); names = list(P); PARTS_CACHE.clear(); PARTS_CACHE.update(P)
    focus = set(FOCUS.get(mode, ())) & set(names)
    mgr = tc.CollisionManager()
    for nm in names: mgr.add_object(nm, P[nm].mesh, np.eye(4))
    cart_meshes = {sq: tm(CART_LOCAL(sq)) for sq in SQ_STEPS}
    for cn in ("pack charged", "pack dead"):
        for sq in SQ_STEPS: mgr.add_object(f"{cn}|{sq}", cart_meshes[sq], FAR)
    moving = [n for n in names if P[n].group != "fixed"]
    ts = sample_times(dt)
    rows = []; t0 = time.time(); meshes = {}; PAIR_MIN.clear()
    for t in ts:
        T, carts, s = transforms(t)
        for nm in names: mgr.set_transform(nm, T[nm])
        owners = {}; active = {}
        for cn, (Tc, sq, own) in carts.items():
            sqr = min([v for v in SQ_STEPS if v >= sq - 1e-9] or [SQ_STEPS[-1]])   # tabs squeezed at least as far as the fingers
            for v in SQ_STEPS: mgr.set_transform(f"{cn}|{v}", Tc if v == sqr else FAR)
            owners[f"{cn}|{sqr}"] = own; active[f"{cn}|{sqr}"] = (cart_meshes[sqr], Tc)
        hit, pairs = mgr.in_collision_internal(return_names=True)
        pairs = {p for p in pairs if all((not n.startswith('pack')) or n in active for n in p)}   # parked variants ignored
        if focus: pairs = {p for p in pairs if p[0] in focus or p[1] in focus}
        bad = [p for p in pairs if not allowed(p[0].split("|")[0] if p[0].startswith("pack") else p[0],
                                              p[1].split("|")[0] if p[1].startswith("pack") else p[1], {k.split("|")[0]: v for k, v in owners.items()})]
        # clearance between non-allowed pairs: pairwise fcl distance on the manager's own BVH objects, AABB-prefiltered
        dmin, dpair = 1e9, None
        objs = {nm: (P[nm].mesh, T[nm]) for nm in names}; objs.update(active)
        own_plain = {k.split("|")[0]: v for k, v in owners.items()}
        q = sorted(focus) if focus else moving + list(active); allq = list(objs)
        def aabb(nm):
            m, Tr = objs[nm]; b = trimesh.bounds.corners(m.bounds); b = trimesh.transform_points(b, Tr); return b.min(0), b.max(0)
        boxes = {nm: aabb(nm) for nm in allq}
        req = fcl.DistanceRequest();
        for i, a in enumerate(q):
            for b in allq:
                if b == a or (b in q and q.index(b) < i): continue
                aa, bb = a.split("|")[0], b.split("|")[0]
                if allowed(aa, bb, own_plain): continue
                lo1, hi1 = boxes[a]; lo2, hi2 = boxes[b]
                if (np.maximum(lo1, lo2) - np.minimum(hi1, hi2) > 30.0).any(): continue   # boxes more than 30 apart
                res = fcl.DistanceResult()
                d = fcl.distance(mgr._objs[a]["obj"], mgr._objs[b]["obj"], req, res)
                if d < dmin: dmin, dpair = float(d), (a, b)
                kp = tuple(sorted((aa, bb)))
                if d < PAIR_MIN.get(kp, (9e9,))[0]: PAIR_MIN[kp] = (round(float(d), 3), round(t, 3))
        overlaps = []
        for a, b in bad:
            try:
                va = man(objs[a][0].copy().apply_transform(objs[a][1])); vb = man(objs[b][0].copy().apply_transform(objs[b][1]))
                overlaps.append((a, b, round((va ^ vb).volume(), 2)))
            except Exception: overlaps.append((a, b, -1.0))
        contacts = [o for o in overlaps if 0.0 <= o[2] <= FIT_TOL]            # faces touching (system_check fit_tol 0.5 mm3): contact, not overlap
        overlaps = [o for o in overlaps if not (0.0 <= o[2] <= FIT_TOL)]
        rows.append(dict(t=round(t, 3), phi=round(s["phi"], 1), zA=round(s["zA"], 1), zB=round(s["zB"], 1), fA=round(s["fA"], 2), fB=round(s["fB"], 2),
                         clearance=round(dmin, 2), pair=list(dpair) if dpair else None, overlaps=overlaps, contacts=contacts))
        log(f"t {t:5.2f} phi {s['phi']:6.1f} zA {s['zA']:5.1f} zB {s['zB']:5.1f} clearance {dmin:6.2f} {dpair} overlaps {overlaps}")
    log(f"sweep {len(rows)} states in {time.time() - t0:.0f} s")
    tag = '' if mode == 'all' else '_' + mode.replace(':', '_')
    json.dump(rows, open(f"{OUT}/arm_sweep{tag}.json", "w"))
    json.dump(sorted([[a, b, d, t] for (a, b), (d, t) in PAIR_MIN.items()], key=lambda r: r[2]), open(f"{OUT}/arm_sweep_pairs{tag}.json", "w"))
    return rows

def gear_slots(width=6.0, y_lo=50.0, y_hi=80.0, step=1.0):
    """option 1: slide the 111T (fixed, tip r 56.5) along the axle in `width` windows; for each, the arm parts it crosses
    at any arm angle (the sun is a disc on the axis, so one arm pose covers the whole cycle; heads parked and at full
    stroke). Returns [(y0, [(part, mm3), ...])]."""
    P = PARTS_CACHE or build_parts("gears")
    arm = {nm: man(p.mesh) for nm, p in P.items() if p.group in ("arm", "armB", "headA", "headB") and nm not in ("pin28",)}
    out = []
    for y0 in np.arange(y_lo, y_hi - width + 0.01, step):
        disc = gear_disc(0, 0, float(y0), float(y0) + width, L.GEAR_REJ["r_tip_sun"], 4.0)
        hits = []
        for z0 in (L.Z0_A_HOME, L.Z0_DRONE):
            for nm, m in arm.items():
                g = PARTS_CACHE[nm].group
                if g == "armB": mm = m.rotate([0, -90, 0])
                else: mm = m
                if g == "headA": mm = mm.translate([0, 0, z0])
                if g == "headB": mm = mm.translate([0, 0, z0]).rotate([0, -90, 0])
                v = (disc ^ mm).volume()
                if v > 0.05 and nm not in [h[0] for h in hits]: hits.append((nm, round(v, 1)))
        out.append((float(y0), hits))
    return out

if __name__ == "__main__":
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    MODE = sys.argv[2] if len(sys.argv) > 2 else "all"
    if MODE == "v5": KEYS = KEYS_V5
    if MODE == "v6": KEYS = _keys_v6()
    if MODE.startswith("v7"):
        import arm_v7_layout as S7; S7.use()
        _, pk, pi, re_ = (MODE.split(":") + ["scout", "-210", "30"])[:4]
        ZP = L.Z0_A_HOME; KEYS = keys_v7(float(pi), float(re_)); T_END = 22.0
    rows = sweep(dt, mode=MODE)
    worst = [r for r in rows if r["overlaps"]]
    print("states with overlaps:", len(worst), "min clearance", min(r["clearance"] for r in rows))
    tally = {}
    for r in rows:
        for a, b, v in r["overlaps"]:
            k = tuple(sorted((a, b))); tally[k] = max(tally.get(k, 0.0), v)
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]): print(f"  overlap {k[0]:12s} {k[1]:12s} max {v:9.1f} mm3")
    print("closest pairs over the cycle:")
    for (a, b), (d, t) in sorted(PAIR_MIN.items(), key=lambda kv: kv[1][0])[:25]: print(f"  {d:7.3f}  {a} / {b}  (t {t})")
    if MODE == "gears":
        for y0, hits in gear_slots(): print(f"  111T at Y {y0:5.1f}..{y0 + 6:5.1f}: {hits if hits else 'CLEAR'}")
