#!/usr/bin/env python3
"""Compare a planar and a non-planar slice of the same part from their g-code alone (nothing sent, nothing printed).
    python tools/np_compare.py <planar.gcode|.gcode.3mf> <nonplanar.gcode|.gcode.3mf> [out.png] [--y 90] [--design-top "<expr in x>"]
Prints: printed volume (part only, supports/brim apart), outline per x (top and tunnel-roof envelope, mm difference), the share of extruding
moves whose Z changes inside the move (by height band), and draws one picture: the cut at y = --y, layer lines of both side by side (true
x/z scale, every 5th layer), and the same cut zoomed where the layers bend most."""
import sys, re, zipfile, collections
import numpy as np


def read(p):
    t = zipfile.ZipFile(p).read("Metadata/plate_1.gcode").decode("utf8", "replace") if p.endswith(".3mf") else open(p, encoding="utf8", errors="replace").read()
    x = y = z = 0.0; e = 0.0; rel = True; ft = ""; layer = -1; rows = []
    for ln in t.split("\n"):
        if ln.startswith("; FEATURE:"): ft = ln[10:].strip(); continue
        if ln.startswith("; CHANGE_LAYER") or ln.startswith(";LAYER_CHANGE"): layer += 1; continue
        if ln.startswith("M83"): rel = True; continue
        if ln.startswith("M82"): rel = False; continue
        if not ln.startswith("G1"): continue
        w = {k[0]: k[1:] for k in ln.split(";")[0].split()[1:] if len(k) > 1}
        nx = float(w.get("X", x)); ny = float(w.get("Y", y)); nz = float(w.get("Z", z))
        ne = float(w["E"]) if "E" in w else 0.0
        if "E" in w and not rel: de = ne - e; e = ne
        else: de = ne
        if de > 0 and ("X" in w or "Y" in w):
            rows.append((x, y, z, nx, ny, nz, de, layer, ft))
        x, y, z = nx, ny, nz
    return rows


def arrays(rows):
    a = np.array([r[:8] for r in rows], float)
    ft = np.array([r[8] for r in rows])
    part = ~(np.char.startswith(ft, "Support") | (ft == "Brim") | (ft == "Skirt"))
    return a, ft, part


def points(a, step=0.25):
    n = np.maximum(1, (np.hypot(a[:, 3] - a[:, 0], a[:, 4] - a[:, 1]) / step).astype(int))
    mv = np.repeat(np.arange(len(a)), n)
    t = (np.arange(len(mv)) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[mv]
    return mv, a[mv, 0] + t * (a[mv, 3] - a[mv, 0]), a[mv, 1] + t * (a[mv, 4] - a[mv, 1]), a[mv, 2] + t * (a[mv, 5] - a[mv, 2])


def main():
    args = [s for s in sys.argv[1:] if not s.startswith("--")]
    opt = {sys.argv[i][2:]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    yc = float(opt.get("y", 90.0))
    A = {}
    for k, p in (("planar", args[0]), ("nonplanar", args[1])):
        a, ft, part = arrays(read(p)); A[k] = (a, ft, part)
    res = {}
    print("%-10s %9s %9s %9s %8s" % ("", "part E mm", "part g", "support g", "layers"))
    for k, (a, ft, part) in A.items():
        g = lambda m: a[m, 6].sum() * np.pi * 0.875 ** 2 * 1.27 / 1000
        print("%-10s %9.0f %9.2f %9.2f %8d" % (k, a[part, 6].sum(), g(part), g(~part), int(a[:, 7].max()) + 1))
        res[k] = g(part)
    print("part volume difference non-planar vs planar: %+.2f%%" % (100 * (res["nonplanar"] / res["planar"] - 1)))
    env = {}
    for k, (a, ft, part) in A.items():
        mv, x, y, z = points(a[part])
        m = np.abs(y - yc) < 5
        bx = np.rint(x[m]).astype(int)
        lo, hi = collections.defaultdict(lambda: 99.0), collections.defaultdict(lambda: -1.0)
        for b, zz in zip(bx, z[m]):
            lo[b] = min(lo[b], zz); hi[b] = max(hi[b], zz)
        env[k] = (lo, hi)
    xs = sorted(set(env["planar"][1]) & set(env["nonplanar"][1]))
    dt = np.array([env["nonplanar"][1][b] - env["planar"][1][b] for b in xs])
    print("top envelope, non-planar minus planar (the planar top is stair-stepped by up to one layer): mean %+.3f mm, worst %+.3f / %+.3f mm over %d columns" % (dt.mean(), dt.min(), dt.max(), len(xs)))
    ar, _, pr = A["nonplanar"]; a = ar[pr]
    dz = np.abs(a[:, 5] - a[:, 2]); ex = dz > 0.004
    print("non-planar, part moves with Z changing inside the move, by height:")
    for z0 in range(0, int(a[:, 5].max()) + 1, 2):
        m = (a[:, 5] >= z0) & (a[:, 5] < z0 + 2)
        if m.sum() > 100: print("   z %2d-%2d mm: %5.1f%% of %d moves" % (z0, z0 + 2, 100 * ex[m].mean(), m.sum()))
    print("   all: %.1f%%" % (100 * ex.mean()))
    out = args[2] if len(args) > 2 else None
    if out:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(15, 10.5), gridspec_kw=dict(height_ratios=[1, 1, 1.15]))
        zoom = (xs[0] - 1, xs[0] + 40)
        for i, (k, ttl) in enumerate((("planar", "PLANAR (flat layers)"), ("nonplanar", "NON-PLANAR (Z moves while extruding)"))):
            a, ft, part = A[k]
            mv, x, y, z = points(a[part]); lay = a[part][mv, 7]
            m = np.abs(y - yc) < 0.6
            ax = axs[i]
            for L in np.unique(lay[m]):
                if int(L) % 4: continue
                s = m & (lay == L)
                o = np.argsort(x[s])
                ax.plot(x[s][o], z[s][o], ".", ms=0.6, color=plt.cm.viridis(L / lay.max()))
            ax.set_aspect("equal"); ax.set_title("%s: cut at y = %.0f mm, every 4th layer, true scale" % (ttl, yc), fontsize=10); ax.set_ylabel("z (mm)")
        ax = axs[2]
        for k, c, sz in (("planar", "tab:red", 2.0), ("nonplanar", "tab:blue", 1.0)):
            a, ft, part = A[k]; mv, x, y, z = points(a[part]); m = (np.abs(y - yc) < 0.6) & (x > zoom[0]) & (x < zoom[1]) & (z > 6)
            ax.plot(x[m], z[m], ".", ms=sz, color=c, label=k, alpha=0.6)
        ax.set_aspect("equal"); ax.legend(markerscale=8); ax.set_title("both over each other at the end of the arch (planar red steps, non-planar blue curved layers)", fontsize=10)
        ax.set_xlabel("x (mm)")
        fig.tight_layout(); fig.savefig(out, dpi=110); print("picture", out)


if __name__ == "__main__":
    main()
