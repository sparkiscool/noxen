"""Battery latch v5's cycle for systems/latch5.assembly.json (build_site.py `timeline`), page frame of parts/latch5/make.py
(x across, y = z_T - pack z, the lead end toward -y; z up the install axis, the receiver on top, cut at the tab line y 0).

1 push in alone: the cartridge rises 55 into the receiver, the feet's lead-ins cam both bolts in (the cam each depth needs is
  the CAD sweep's, parts/latch5/result.json insert_rows), the guide poles enter their bores 11.7 below the seat, the XT60 at 6.0,
  the bolts snap out 0.5 below the seat.
2 pull: a load drops the cartridge 0.5 until both steel pins sit on their 20 deg seat ramps (the PCTG bolts stay 0.46 clear).
3 the grabber's blades come up into the recess open and squeeze: the tabs go in 3.8 to the centre rib, the pins clear the mouth.
4 the blades pull the cartridge out squeezed, push it back in, open: the spring throws the bolts out, latched; blades away.
"""
import json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(ROOT, "parts", "latch5", "result.json")))
TRAVEL = RES["travel"]["travel"]; OVERLAP = RES["travel"]["needed"] - 0.3
OUT, BLADE_PARK = 55.0, -70.0
L0 = RES["spring_N"]["rest_len"]; K, FREE = 0.20, 22.0
ENG = RES["engage_depth_mm"]; DROP = RES["pull_drop_to_pin_mm"]
CAM = sorted((-d, max(x or 0.0 for x in rr)) for d, rr in RES["insert_rows"])   # (depth below the seat, cam retraction)
OPEN_U = 11.0 - (10.0 - TRAVEL)        # blade stroke from open (11) to squeezed (6.2)
PH = [("out", 0.8), ("push in alone", 4.0), ("latched", 0.6), ("pull load", 1.0), ("load off", 0.6), ("blades up", 1.4),
      ("squeeze", 1.2), ("pull out, held", 3.0), ("hold", 0.6), ("push in, held", 3.0), ("release", 1.0), ("blades away", 1.2),
      ("latched", 1.0)]
TT = sum(d for _, d in PH)


def ease(x):
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x * x * (3 - 2 * x)


def cam(d):
    xs, ys = zip(*CAM)
    return float(np.interp(d, xs, ys, left=0.0, right=0.0))


def state(t):
    f, a = {}, 0.0
    for i, (nm, d) in enumerate(PH):
        f[i] = ease((t - a) / d); a += d
    d = OUT * (1 - f[1])                                        # 1 push in
    rho = cam(d) if f[1] < 1 else 0.0
    drop = DROP * (f[3] - f[4])                                 # 2 pull load on, off
    bz = BLADE_PARK * (1 - f[5])                                # 3 blades up
    u = OPEN_U * (f[6] - f[10])                                 # squeeze ... release
    if f[6] > 0:
        rho = max(0.0, min(TRAVEL, u - 1.0))                     # the blades touch the tabs after 1.0 of stroke
    d += OUT * (f[7] - f[9])                                    # 4 pull out held, push in held
    bz += -OUT * (f[7] - f[9]) + BLADE_PARK * f[11]
    return dict(d=d, rho=rho, drop=drop, bz=bz, u=u)


def tr(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def poses_at(t):
    s = state(t)
    C = tr(0, 0, -s["d"] - s["drop"])
    sp = np.diag([(L0 - 2 * s["rho"]) / L0, 1, 1, 1.0])
    B = tr(0, 0, s["bz"])
    return {"cartridge": C, "cartridge strip px": C, "cartridge strip nx": C, "pack": C, "xt60 female": C,
            "slider p": C @ tr(-s["rho"], 0, 0), "slider n": C @ tr(s["rho"], 0, 0),
            "bolt pin p": C @ tr(-s["rho"], 0, 0), "bolt pin n": C @ tr(s["rho"], 0, 0), "spring": C @ sp,
            "blade p": B @ tr(-s["u"], 0, 0), "blade n": B @ tr(s["u"], 0, 0)}


def timeline(a):
    names = {p["name"] for p in a["parts"]}
    fps, n = 24, int(TT * 24)
    ts = [i * TT / n for i in range(n)]
    st = [state(t) for t in ts]
    series = [("cartridge below the seat", [s["d"] + s["drop"] for s in st], "mm"),
              ("bolt retraction (lead-in cam or blades)", [s["rho"] for s in st], "mm"),
              ("steel pin past the receiver's locating face", [OVERLAP - s["rho"] for s in st], "mm"),
              ("pen spring force", [K * (FREE - (L0 - 2 * s["rho"])) for s in st], "N"),
              ("guide poles in their bores", [max(0.0, ENG["guide_poles"] - s["d"]) for s in st], "mm"),
              ("XT60 contacts engaged", [max(0.0, ENG["xt60"] - s["d"]) for s in st], "mm"),
              ("pull load drop onto the seat ramps", [s["drop"] for s in st], "mm")]
    return (lambda t: {k: v for k, v in poses_at(t).items() if k in names}), TT, series


if __name__ == "__main__":
    a = 0.0
    for nm, d in PH:
        s = state(a + d * 0.999); print(f"{nm:16s} t {a + d:5.1f}: d {s['d']:5.1f} rho {s['rho']:4.2f} drop {s['drop']:4.2f} bz {s['bz']:6.1f} u {s['u']:4.2f}"); a += d
