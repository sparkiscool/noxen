"""Battery latch v7's cycle for systems/latch7.assembly.json (build_site.py `timeline`), page frame of parts/latch7/make.py
(x across, y = z_T - pack z, the lead end toward -y; z up the install axis, the receiver on top, cut at the tab line y 0).

1 push in alone: the cartridge rises 55 into the receiver, the feet's lead-ins cam both bolts in (the cam each depth needs is
  the CAD sweep's, parts/latch7/result.json insert_rows), the bolts snap out 0.5 below the seat on their two springs.
2 pull: a load drops the cartridge 0.5 until both steel pins sit on their 20 deg seat ramps.
3 the grabber comes up open (its four heel pads land on the cartridge's rim bands, each nub rises through its tab's arch) and
  squeezes: the blade faces push the tabs in 4.1, the nubs slide onto the rib's wedge feet and draw the cartridge onto the pads.
4 pulled out, held; tipped 90 deg each way about both axes (the hold readouts follow gravity); pushed back in; the blades
  open, the springs throw the bolts out, latched; grabber away.

python3 tools/latch7_motion.py            the phase table
python3 tools/latch7_motion.py --collide  every frame of the page's motion, the real (uncut) solids at the page's poses: mm3 per pair
"""
import json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(ROOT, "parts", "latch7_overseer", "result.json")))
V = RES["v6"]
TRAVEL, R0, OVERLAP = V["TRAVEL"], V["R0"], V["OVERLAP"]
OUT, OUT_H, BLADE_PARK = 75.0, 150.0, -90.0
SP = RES["spring"]; KS = SP["each"]["rate_N_mm"]; FREE = SP["each"]["free"]; L0 = SP["rest"]["length"]
ENG = RES["engage_depth_mm"]; DROP = RES["pull_drop_to_pin_mm"] - 0.02
CAM = sorted((-d, max(x or 0.0 for x in rr)) for d, rr in RES["insert_rows"])
STROKE = V["BLADE_OPEN"] - (V["TAB_OUT"] - R0)      # blade stroke, open to tight on the wedge (5.6)
TOUCH = V["BLADE_OPEN"] - V["TAB_OUT"]              # the blade face reaches the tab after 1.0
TAN = math.tan(math.radians(V["ALPHA_deg"]))
FB = 25.0                                            # the readouts' squeeze: the heads v7 blade springs
PH = [("out", 0.8), ("push in alone", 4.0), ("latched", 0.6), ("pull load", 1.0), ("load off", 0.6), ("grabber up", 1.6),
      ("squeeze", 1.6), ("pull out, held", 3.0), ("tip nose down 90", 2.0), ("tip nose up 90", 3.0), ("level", 1.5),
      ("roll 90", 2.0), ("roll -90", 3.0), ("level", 1.5), ("push in, held", 3.0), ("release", 1.2), ("grabber away", 1.4), ("latched", 1.0)]
TT = sum(d for _, d in PH)
_H8 = next(h for h in RES["hold"]["all_gravity_directions"] if h["Fb"] == FB and h["n_g"] == 1.0 and h["pads"].startswith("TPU"))
P0 = _H8["pad_preload_N"]
W = RES["swap_g"] / 1000 * 9.81; YCG = RES["swap_cg"][1]; DZ = RES["swap_cg"][2] - RES["geom"]["Z_T"]
PX = sum(V["PAD_X"]) / 2; PZ = sum(V["PAD_Z"]) / 2


def ease(x):
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x * x * (3 - 2 * x)


def cam(d):
    """the retraction the lead-ins hold a bolt at, this deep: the larger of the two swept rows round d (never less than
    either row needs, so the page's bolts clear the feet at every frame), 0 at the seat."""
    if d <= 1e-6: return 0.0
    return max([y for x, y in CAM if abs(x - d) <= 0.5 + 1e-9] or [0.0])


def state(t):
    f, a = {}, 0.0
    for i, (nm, d) in enumerate(PH):
        f[i] = ease((t - a) / d); a += d
    d = OUT * (1 - f[1])
    rho = cam(d) if f[1] < 1 else 0.0
    drop = DROP * (f[3] - f[4])
    bz = BLADE_PARK * (1 - f[5])
    u = STROKE * (f[6] - f[15])
    if f[6] > 0:
        rho = max(0.0, min(TRAVEL, u - TOUCH))
    out = OUT_H * (f[7] - f[14])
    d += out
    bz += -out + BLADE_PARK * f[16]
    pitch = 90.0 * (f[8] - 2 * f[9] + f[10])          # about the cartridge's x: the pack's long axis to vertical, both ways
    roll = 90.0 * (f[11] - 2 * f[12] + f[13])         # about the pack's long axis
    return dict(d=d, rho=rho, drop=drop, bz=bz, u=u, pitch=pitch, roll=roll, held=f[6] >= 1 and f[15] <= 0)


def tr(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def rot(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0)}[axis]
    R[i, i] = c; R[i, j] = -s; R[j, i] = s; R[j, j] = c
    return R


def group(s):
    """the held set's pose: down by d (and the pull drop), tipped about the grip point (page origin of the set)."""
    return tr(0, 0, -s["d"] - s["drop"]) @ rot("x", s["pitch"]) @ rot("y", s["roll"])


def poses_at(t):
    s = state(t)
    C = group(s)
    sp = np.diag([(L0 - 2 * s["rho"]) / L0, 1, 1, 1.0])
    kk = (L0 - s["rho"]) / L0; spp = tr(2, 0, 0) @ np.diag([kk, 1, 1, 1.0]) @ tr(-2, 0, 0); spn = tr(-2, 0, 0) @ np.diag([kk, 1, 1, 1.0]) @ tr(2, 0, 0)
    Bm = tr(0, 0, s["bz"]) @ rot("x", s["pitch"]) @ rot("y", s["roll"]) if s["held"] else tr(0, 0, s["bz"])
    return {"cartridge": C, "stop bar": C, "cartridge strip px": C, "cartridge strip nx": C, "pack": C, "xt60 female": C,
            "slider p": C @ tr(-s["rho"], 0, 0), "slider n": C @ tr(s["rho"], 0, 0),
            "bolt pin p": C @ tr(-s["rho"], 0, 0), "bolt pin n": C @ tr(s["rho"], 0, 0), **{f"spring {k}": C @ spp for k in "abcd"}, **{f"spring {k}": C @ spn for k in "efgh"}, **{f"spring pin {k}": C for k in "abcdefgh"},
            "blade p": Bm @ tr(-s["u"], 0, 0), "blade n": Bm @ tr(s["u"], 0, 0), "heel deck": Bm, "heel pads": Bm}


def gravity_c(s):
    """gravity (unit) in cartridge coords for the set's pose: world down is page -z; c = (p.x, p.z, -p.y)."""
    R = (rot("x", s["pitch"]) @ rot("y", s["roll"]))[:3, :3]
    p = R.T @ np.array([0.0, 0.0, -1.0])
    return np.array([p[0], p[2], -p[1]])


def hold_at(s, n_g=1.0, mu_pad=None, mu_w=None):
    """the two hold margins at this pose, gravity x n_g, blades at FB (parts/latch7/make.py hold(), one direction)."""
    mu_pad = V["MU"]["pad_tpu"] if mu_pad is None else mu_pad; mu_w = V["MU"]["wedge"] if mu_w is None else mu_w
    F = n_g * W * gravity_c(s)
    Mx = YCG * F[2] - DZ * F[1]; My = DZ * F[0]; Mz = -YCG * F[0]
    N = P0 - F[1]
    if N <= 0: return 0.0, 0.0, N
    lift = 1.0 / (abs(Mx) / (N * PZ) + abs(Mz) / (N * PX) + 1e-9)
    slip = 1.0 / (math.hypot(F[0] / (mu_pad * N), F[2] / (mu_pad * N + mu_w * P0)) + abs(My) / (mu_pad * N * math.hypot(PX, PZ)) + 1e-9)
    return min(lift, 99.0), min(slip, 99.0), N


def timeline(a):
    names = {p["name"] for p in a["parts"]}
    fps, n = 24, int(TT * 24)
    ts = [i * TT / n for i in range(n)]
    st = [state(t) for t in ts]
    H = [hold_at(s) if s["held"] else (0.0, 0.0, 0.0) for s in st]
    series = [("cartridge below the seat", [s["d"] + s["drop"] for s in st], "mm"),
              ("bolt travel (lead-in cam or blades; 3.3 unlatches, 4.9 = tab on the rib)", [s["rho"] for s in st], "mm"),
              ("steel pin engagement past the receiver's locating face", [max(0.0, OVERLAP - s["rho"]) for s in st], "mm"),
              ("spring force on each bolt (its four springs)", [4 * KS * (FREE - (L0 - s["rho"])) for s in st], "N"),
              ("nub above its wedge foot (0 = tight, drawing the cartridge onto the pads)", [max(0.0, (STROKE - s["u"]) * TAN) if s["bz"] - (-s["d"]) > -1 or s["held"] else 0.0 for s in st], "mm"),
              ("heel pad load at 25 N blades (preload - what gravity takes off)", [h[2] for h in H], "N"),
              ("grip hold: pads stay down, margin at 1 g (25 N blades)", [h[0] for h in H], "x"),
              ("grip hold: no slip, margin at 1 g (25 N blades, TPU pads)", [h[1] for h in H], "x"),
              ("tip about x / roll about the pack axis", [s["pitch"] if abs(s["pitch"]) > 1e-9 else s["roll"] for s in st], "deg"),
              ("guide poles in their bores", [max(0.0, ENG["guide_poles"] - s["d"]) for s in st], "mm"),
              ("XT60 contacts engaged", [max(0.0, ENG["xt60"] - s["d"]) for s in st], "mm")]
    return (lambda t: {k: v for k, v in poses_at(t).items() if k in names}), TT, series


def collide(step=1):
    """every frame: the real solids (parts/latch7/make.py, uncut) at the page's own poses, pair by pair (mm3)."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("latch7_make", os.path.join(ROOT, "parts", "latch7", "make.py"))
    mk = importlib.util.module_from_spec(sp); sp.loader.exec_module(mk)
    B = mk.build("overseer"); g = B["g"]
    M = mk.page_matrix(g["Z_T"]); Mi = np.linalg.inv(M)
    deck, pads = mk.heel(g)
    bp, bn = mk.blades(g, mk.BLADE_OPEN, back=0.02)                       # the nub underside 0.02 off its mating face
    body = {"cartridge": mk.cart_all(B) + mk.bl.pack_body(g), "slider p": mk.moving(g, 0, 1), "slider n": mk.moving(g, 0, -1),
            "blade p": bp, "blade n": bn, "heel deck": mk.U(deck, *[p.translate([0, -0.02, 0]) for p in pads])}
    fixed = mk.receiver_all(B)
    pairs = [("cartridge", "slider p"), ("cartridge", "slider n"), ("slider p", "slider n"), ("cartridge", "blade p"), ("cartridge", "blade n"),
             ("slider p", "blade p"), ("slider n", "blade n"), ("slider p", "blade n"), ("slider n", "blade p"), ("blade p", "blade n"),
             ("cartridge", "heel deck"), ("slider p", "heel deck"), ("slider n", "heel deck")]
    worst = {}; at = {}
    n = int(TT * 24)
    for i in range(0, n, step):
        t = i * TT / n
        P = poses_at(t); s = state(t)
        # contacts that are faces meeting by design, backed off 0.02: the blade face on the tab, the tab's retraction to match
        eps = {"slider p": tr(-0.02, 0, 0), "slider n": tr(0.02, 0, 0)} if s["u"] > TOUCH else {}
        cur = {}
        for nm, m in body.items():
            T = Mi @ np.asarray(P[nm]) @ eps.get(nm, np.eye(4)) @ M
            cur[nm] = m.transform(T[:3, :4])
        for nm, m in cur.items():
            v = mk.vol(m ^ fixed); k = f"{nm} / receiver"
            if v > worst.get(k, -1): worst[k], at[k] = v, round(t, 2)
        for a, b in pairs:
            v = mk.vol(cur[a] ^ cur[b]); k = f"{a} / {b}"
            if v > worst.get(k, -1): worst[k], at[k] = v, round(t, 2)
    out = dict(frames=len(range(0, n, step)), duration_s=TT, fps=24, limit_mm3=0.05,
               worst_mm3={k: round(v, 4) for k, v in worst.items()}, at_s={k: at[k] for k, v in worst.items() if v > 0.001})
    out["max_mm3"] = max(out["worst_mm3"].values()); out["ok"] = out["max_mm3"] < out["limit_mm3"]
    return out


if __name__ == "__main__":
    if "--collide" in sys.argv:
        r = collide()
        RES["motion_collision"] = r
        json.dump(RES, open(os.path.join(ROOT, "parts", "latch7_overseer", "result.json"), "w"), indent=1)
        print(json.dumps(r, indent=1)); sys.exit(0 if r["ok"] else 1)
    a = 0.0
    for nm, d in PH:
        s = state(a + d * 0.999); h = hold_at(s) if s["held"] else (0, 0, 0)
        print(f"{nm:18s} t {a + d:5.1f}: d {s['d']:6.1f} rho {s['rho']:4.2f} drop {s['drop']:4.2f} bz {s['bz']:7.1f} u {s['u']:4.2f} "
              f"pitch {s['pitch']:5.1f} roll {s['roll']:5.1f} hold {h[0]:5.2f} {h[1]:5.2f}"); a += d
