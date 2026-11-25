#!/usr/bin/env python3
"""Non-planar slicing for Spark's A1 mini printers (Gideon, Jarvis): curved layers kept inside the printer's safe slope and nozzle
clearance. Planar slicing is untouched: a part is sliced non-planar only when its plate entry asks. Never sends or prints.

    python tools/plate.py print/plates/<plate>.json --out DIR      # a part with "nonplanar": {"mode": "wave"} is sliced non-planar
    python tools/nonplanar.py check <file.gcode.3mf>                # the collision gate alone, from the g-code in the file
    python tools/nonplanar.py preview <file.gcode.3mf> [out.png]    # the picture to look at before anything is printed

How: the layers are bent by a height field, real z = u + d(x, y, u), u being the flat slicer's height. The mesh is un-bent with the
inverse, OrcaSlicer slices it flat as always (walls, infill, supports, seams all its own), and the g-code is bent back: every move
is cut into short pieces, each gets its real Z, its extrusion scaled by the local layer thickness (the field's volume change)
and its speed held to what the Z axis follows. Then the finished g-code is replayed against the measured shape of the A1 mini's
hotend (print/profiles/a1mini_toolhead.json): a file with a collision, a move steeper than the limit or a layer outside the
nozzle's thickness range is NOT READY, and tools/bambu_send.py refuses it.

Modes ("nonplanar": {"mode": ...} on a plate part; they combine with "+"):
  wave     interlocking layers: every layer inside the part is an egg-crate wave (default 0.4 mm high, about 10 deg flanks), so no
           flat plane runs through the part for a crack or a shear load to follow. The wave fades to nothing within "skin" (1 mm) of
           every top, bottom, overhang and the bed, so outside faces, bores and first layers are exactly the planar ones.
  conform  curved layers along shallow top surfaces: tops no steeper than the design slope are un-bent flat, so their skins print
           as continuous lines along the surface (no stair steps on the face that carries the bending); flat faces stay flat, the
           bend blends out towards the bed through thicker / thinner layers.
  zaa      OrcaSlicer's own Z anti-aliasing of top surfaces (inside one layer; surface finish only), checked by the same gate.
Options: amp (mm; default: the tallest the hotend clearance allows, auto_wave), wavelength (mm), skin (mm, 0.6), slope (deg, never above the profile's limit).

State on 23 Nov 2026 (later): conform HAS now bent a part (parts/np_arch_coupon, 99% of its arched top, gate passed, nothing printed). Changes: slopes and layer thickness are
checked node to node (_fwd; the central gradient hid steps: 13.5 deg moves under a 9.7 deg field), the conform solve caps both axes together, layers may thin to x0.5 ("tau_min"),
and with conform the wave gives up amplitude (down to none) so the curved top is followed in full. Before that: the wave now takes its height from the hotend clearance (auto_wave, clearance()) and holds the slope over the supports too; np_cradle_deep is its test (1.0 mm wave every 47.5 mm, gate passed, nothing printed). Before: wave is proven on three parts (print/nonplanar/np_spool, np_turret, np_cradle: sliced, gate passed, nothing
printed). conform has NOT yet bent a part (the cradle had no shallow curved top) and zaa has NEVER been sliced (its setting names
are unconfirmed on the CLI): treat both as untested. The hotend shape is other people's measurement, not Spark's.
The gate counts what runs before a layer's first printed line (the wipe at the old height, the lift, the travel in) against the
layer below only; before that it failed all three parts on exactly such a wipe (cradle move 277001, spool 331509, turret 1909309).
"""
import argparse, hashlib, json, math, os, re, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLHEAD = os.path.join(ROOT, "print", "profiles", "a1mini_toolhead.json")
VERSION = "1"


def slice_overrides(o, th):
    """What the flat slice must be for the bend to be exact: straight moves only (no arcs), a plain lift, room to hop, no infill combining (sparse infill printed every 2nd layer is 0.4 thick: the 24 Nov snap specimen failed the layer gate on it)."""
    hop = "%g" % th["limits"]["hop_mm"]
    ov = {"p": {"enable_arc_fitting": "0", "infill_combination": "0"}, "m": {"z_hop_types": ["Normal Lift"], "z_hop": [hop]},
          "f": {"filament_z_hop_types": ["Normal Lift"], "filament_z_hop": [hop]}}
    if "zaa" in modes(o):
        ov["p"].update({"zaa_enabled": "1", "zaa_min_z": "0.05"})
    return ov


def modes(o):
    return set(str(o.get("mode", "wave")).replace(",", "+").split("+"))


def toolhead(path=None):
    return json.load(open(path or TOOLHEAD, encoding="utf8"))


def env_height(th, DX, DY):
    """How high material may stand over the nozzle tip at the horizontal offset (DX, DY) from it before the hotend hits it."""
    r = np.hypot(DX, DY)
    cone = np.array(th["nozzle"]["cone"], float)
    b, fl = th["block"], th["flank"]
    out = np.maximum.reduce([b["x"][0] - DX, DX - b["x"][1], b["y"][0] - DY, DY - b["y"][1], np.zeros_like(r)])
    flank = np.minimum(b["h"] + math.tan(math.radians(fl["deg"])) * out, min(fl["top"], th["ceiling"]["h"]))
    h = np.where(r <= cone[-1, 0], np.interp(r, cone[:, 0], cone[:, 1]), flank)
    return np.where(r > th["ceiling"]["radius"], np.inf, h)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ------------------------------------------------------------------------------------------------ the field

class Field:
    """real z = u + d(x, y, u) on a grid of nodes x0 + i c, y0 + j c, k du. d is 0 at the grid's rim and 0 outside it."""

    def __init__(self, x0, y0, c, du, d, info):
        self.x0, self.y0, self.c, self.du, self.info = x0, y0, c, du, info
        self.d = np.ascontiguousarray(d, np.float32)
        from scipy import ndimage
        n = max(1, int(round(2.0 / c)))                 # a lifted travel flies over the highest layer within 2 mm of its path
        yy, xx = np.ogrid[-n:n + 1, -n:n + 1]
        self.dT = ndimage.maximum_filter(self.d, footprint=(xx * xx + yy * yy <= n * n)[:, :, None])

    def _tri(self, a, x, y, u):
        nx, ny, nu = a.shape
        fx = (x - self.x0) / self.c; fy = (y - self.y0) / self.c
        inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
        fx = np.clip(fx, 0, nx - 1.001); fy = np.clip(fy, 0, ny - 1.001); fu = np.clip(u / self.du, 0, nu - 1.001)
        i = fx.astype(np.int64); j = fy.astype(np.int64); k = fu.astype(np.int64)
        tx = fx - i; ty = fy - j; tu = fu - k
        c00 = a[i, j, k] * (1 - tx) + a[i + 1, j, k] * tx
        c10 = a[i, j + 1, k] * (1 - tx) + a[i + 1, j + 1, k] * tx
        c01 = a[i, j, k + 1] * (1 - tx) + a[i + 1, j, k + 1] * tx
        c11 = a[i, j + 1, k + 1] * (1 - tx) + a[i + 1, j + 1, k + 1] * tx
        v = (c00 * (1 - ty) + c10 * ty) * (1 - tu) + (c01 * (1 - ty) + c11 * ty) * tu
        return np.where(inside, v, 0.0)


class Fields:
    """Every bent part on the plate. d_at(...) is the bend at points; `lifted` marks travel points flown over the layer."""

    def __init__(self, fields):
        self.fields = [f for f in fields if f is not None]
        self.piece = min([f.info["piece_mm"] for f in self.fields] or [1.0])

    def d_at(self, x, y, u, lifted=None):
        x = np.asarray(x, float); y = np.asarray(y, float); u = np.asarray(u, float)
        v = np.zeros(x.shape)
        for f in self.fields:
            v += f._tri(f.d, x, y, u)
            if lifted is not None and lifted.any():
                v[lifted] += f._tri(f.dT, x[lifted], y[lifted], u[lifted]) - f._tri(f.d, x[lifted], y[lifted], u[lifted])
        return v


def columns(mesh, x0, y0, c, nx, ny, du, nu):
    """Where the part is solid, on the grid: occ[i, j, k] at height k du, the top height of every column and how steep that top is."""
    tri = np.asarray(mesh.triangles, float); fn = np.asarray(mesh.face_normals, float)
    jx, jy = 1.2345e-4, 2.3456e-4                       # no column ever runs exactly along a CAD edge
    acc = np.zeros((nx, ny, nu + 1), np.int16)
    T = np.full((nx, ny), -np.inf); Tn = np.zeros((nx, ny))
    for f in range(len(tri)):
        nz = fn[f, 2]
        if abs(nz) < 1e-7:
            continue
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri[f]
        i0 = max(0, int(math.ceil((min(ax, bx, cx) - x0 - jx) / c))); i1 = min(nx - 1, int(math.floor((max(ax, bx, cx) - x0 - jx) / c)))
        j0 = max(0, int(math.ceil((min(ay, by, cy) - y0 - jy) / c))); j1 = min(ny - 1, int(math.floor((max(ay, by, cy) - y0 - jy) / c)))
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if i1 < i0 or j1 < j0 or abs(den) < 1e-12:
            continue
        PX, PY = np.meshgrid(x0 + jx + np.arange(i0, i1 + 1) * c, y0 + jy + np.arange(j0, j1 + 1) * c, indexing="ij")
        l1 = ((by - cy) * (PX - cx) + (cx - bx) * (PY - cy)) / den
        l2 = ((cy - ay) * (PX - cx) + (ax - cx) * (PY - cy)) / den
        l3 = 1.0 - l1 - l2
        m = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if not m.any():
            continue
        z = l1 * az + l2 * bz + l3 * cz
        ii, jj = np.nonzero(m)
        kk = np.clip(np.floor(z[m] / du).astype(np.int64) + 1, 0, nu)
        np.add.at(acc, (ii + i0, jj + j0, kk), 1 if nz < 0 else -1)
        sub = T[i0:i1 + 1, j0:j1 + 1]; upd = m & (z > sub)
        sub[upd] = z[upd]; Tn[i0:i1 + 1, j0:j1 + 1][upd] = nz
    return np.cumsum(acc[:, :, :nu], axis=2) > 0, T, Tn


def relief_ok(d2, has, c, th, margin):
    """One layer's bend d2 (2D, where the part is: has) against the hotend beyond the nozzle: no printed point may stand higher
    over another than the hotend clears at that offset. Coarse (about 1 mm) and on the safe side."""
    f = max(1, int(round(1.0 / c)))
    nx, ny = (d2.shape[0] // f) * f, (d2.shape[1] // f) * f
    if nx == 0 or ny == 0 or not has.any():
        return True
    hi = np.where(has, d2, -np.inf)[:nx, :ny].reshape(nx // f, f, ny // f, f).max(axis=(1, 3))
    lo = np.where(has, d2, np.inf)[:nx, :ny].reshape(nx // f, f, ny // f, f).min(axis=(1, 3))
    if not np.isfinite(hi).any():
        return True
    if hi.max() - lo.min() <= th["block"]["h"] - margin:
        return True
    cc = c * f
    return _far_clear(hi, lo, cc, th, margin) >= 0.0


def _far_env(cc, th, R):
    """The hotend's clearance on a coarse grid of offsets, each the lowest of everything within one cell (safe side); offsets
    closer than 4 mm are left to the fine check."""
    n = int(math.ceil(R / cc))
    fine = cc / 4.0
    m = 4 * (n + 1)
    ax = np.arange(-m, m + 1) * fine
    E = env_height(th, *np.meshgrid(ax, ax, indexing="ij"))
    from scipy import ndimage
    E = ndimage.minimum_filter(E, size=9, mode="nearest")           # +- one coarse cell
    Ec = E[4::4, 4::4][: 2 * n + 1, : 2 * n + 1]
    off = np.arange(-n, n + 1)
    OX, OY = np.meshgrid(off, off, indexing="ij")
    Ec = np.where(np.hypot(OX, OY) * cc < 4.0, np.inf, Ec)
    return off, Ec


def _far_clear(hi, lo, cc, th, margin):
    """min over all pairs of coarse cells of (tip height + clearance - margin - material height): >= 0 is clear."""
    nx, ny = hi.shape
    R = min(th["ceiling"]["radius"], math.hypot(nx, ny) * cc)
    off, Ec = _far_env(cc, th, R)
    worst = np.inf
    for a, oa in enumerate(off):
        if abs(oa) >= nx:
            continue
        for b, ob in enumerate(off):
            e = Ec[a, b]
            if not np.isfinite(e) or abs(ob) >= ny:
                continue
            # material at q = p + (oa, ob) against the tip at p
            p = lo[max(0, -oa):nx - max(0, oa), max(0, -ob):ny - max(0, ob)]
            q = hi[max(0, oa):nx - max(0, -oa), max(0, ob):ny - max(0, -ob)]
            v = (p + e - margin - q).min()
            if v < worst:
                worst = v
    return worst


def conform_relief(T, Tn, c, lim, o):
    """The bend that lays shallow top surfaces flat: r(x, y) with grad r = grad T on tops no steeper than the design slope,
    r constant on flat faces (they weigh ten times more, mating faces stay true), smooth everywhere else. None if nothing to follow."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    nx, ny = T.shape
    has = np.isfinite(T)
    tc = math.radians(float(o.get("slope", lim["slope_deg"])))
    flat = has & (Tn >= math.cos(math.radians(0.5)))
    gentle = has & (Tn >= math.cos(tc) - 1e-9)
    curved = gentle & ~flat
    if curved.sum() * c * c < float(o.get("conform_min_area", 20.0)):
        return None
    Tf = np.where(has, T, 0.0)
    idx = np.arange(nx * ny).reshape(nx, ny)
    A, B, t, w = [], [], [], []
    for sa, sb in (((slice(0, nx - 1), slice(None)), (slice(1, nx), slice(None))), ((slice(None), slice(0, ny - 1)), (slice(None), slice(1, ny)))):
        both = (gentle[sa] & gentle[sb]).ravel()
        dT = (Tf[sb] - Tf[sa]).ravel()
        same = both & (np.abs(dT) <= c * math.tan(tc) * 1.25 + 1e-6)
        A.append(idx[sa].ravel()); B.append(idx[sb].ravel())
        t.append(np.where(same, dT, 0.0))
        w.append(np.where(same, np.where((flat[sa] & flat[sb]).ravel(), 10.0, 1.0), 0.02))
    A = np.concatenate(A); B = np.concatenate(B); t = np.concatenate(t); w = np.concatenate(w)
    ne, n = len(A), nx * ny
    cap = c * math.tan(tc)
    r = None
    for _ in range(10):
        sw = np.sqrt(w)
        M = sparse.csr_matrix((np.concatenate([-sw, sw]), (np.concatenate([np.arange(ne)] * 2), np.concatenate([A, B]))), shape=(ne, n))
        r = spsolve((M.T @ M + 1e-7 * sparse.identity(n)).tocsc(), M.T @ (sw * t))
        diff = r[B] - r[A]
        over = np.abs(diff) > cap * 1.02
        sn = np.hypot(_fwd(r.reshape(nx, ny), c, 0), _fwd(r.reshape(nx, ny), c, 1)).ravel()      # both axes together (a corner node saw 12.6 deg)
        sE = np.maximum(sn[A], sn[B]); over2 = sE > math.tan(tc) * 1.02
        if not (over.any() or over2.any()):
            break
        t = np.where(over, np.clip(diff, -cap * 0.95, cap * 0.95), t)
        t = np.where(over2 & ~over, diff * 0.95 * math.tan(tc) / np.maximum(sE, 1e-9), t)
        w = np.where(over | over2, np.maximum(w, 4.0), w)
    r = r.reshape(nx, ny)
    if flat.any():                                          # the biggest flat level keeps exactly planar layers
        v = r[flat]; hist, edges = np.histogram(v, bins=max(1, int((v.max() - v.min()) / 0.02) + 1))
        k = int(hist.argmax()); r = r - v[(v >= edges[k]) & (v <= edges[k + 1])].mean()
    else:
        r = r - r[gentle].max()
    return r, curved, has


def clearance(th):
    """The A1 mini's clearance from its measured geometry: the steepest straight slope the hotend clears at each distance from the
    tip (both axes, margin kept), the height it clears at the block's edge, and the single-layer relief the ceiling allows."""
    lim = th["limits"]; mg = lim["margin_mm"]
    d = np.linspace(1.0, th["ceiling"]["radius"], 400)
    out = {}
    for name, DX, DY in (("x", d, 0 * d), ("y", 0 * d, d)):
        e = env_height(th, DX, DY) - mg
        out[name] = np.degrees(np.arctan(e / d))
    asym = float(min(out["x"][d >= 12].min(), out["y"][d >= 12].min()))             # far from the block: the flank decides
    edge = float(min(np.degrees(np.arctan((th["block"]["h"] - mg) / th["block"]["x"][1])), np.degrees(np.arctan((th["block"]["h"] - mg) / th["block"]["y"][1]))))
    return dict(tip_half_angle_deg=round(math.degrees(math.atan((th["nozzle"]["cone"][-1][0] - th["nozzle"]["cone"][0][0]) / th["nozzle"]["cone"][-1][1])), 1),
                block_clear_mm=th["block"]["h"], block_edge_mm=th["block"]["x"][1], slope_at_block_edge_deg=round(edge, 1),
                flank_slope_deg=th["flank"]["deg"], slope_far_deg=round(asym, 1), ceiling_mm=th["ceiling"]["h"],
                design_slope_deg=lim["slope_deg"], hard_slope_deg=lim["slope_hard_deg"], relief_mm=lim["relief_mm"])


def auto_wave(mesh, o, th, layer_h, tau_hi, tau_lo, tslope):
    """The tallest wave the clearance allows for this part, and its wavelength: the amplitude is raised until the layer's peak-to-peak
    relief meets the hotend's relief limit (relief_ok, the same replay maths as the gate), the thickness range, or the wavelength the
    slope needs outgrows the part. Returns (amp, wavelength)."""
    lim = th["limits"]; b = mesh.bounds
    ext = float(max(b[1][0] - b[0][0], b[1][1] - b[0][1])); zt = float(b[1][2] - b[0][2])
    c = 0.5
    best = (0.4, 2 * math.pi * 0.4 / (tslope * 0.75))
    X = np.arange(b[0][0], b[1][0] + c, c); Y = np.arange(b[0][1], b[1][1] + c, c)
    has = np.ones((len(X), len(Y)), bool)
    for amp in np.arange(0.4, 3.01, 0.1):
        lam = 2 * math.pi * amp / (tslope * 0.75)
        if lam > 1.1 * ext:                                  # at least one whole wave across the part
            break
        Dv = 1.5 * amp / min(tau_hi - 1, 1 - tau_lo) * 1.15  # the wave must grow gently enough to hold the layer thickness
        if Dv > 0.45 * zt or 2 * amp > lim["relief_mm"] - 1.0:
            break
        kx = 2 * math.pi / lam
        d2 = amp * np.cos(kx * (X - X.mean()))[:, None] * np.cos(kx * (Y - Y.mean()))[None, :]
        if not relief_ok(d2, has, c, th, lim["margin_mm"]):
            break
        best = (round(float(amp), 2), round(lam, 2))
    return best


def build_field(mesh, o, th, layer_h=0.2):
    """The height field for one part posed on the bed. o: its "nonplanar" options. Returns a Field (None if nothing bends)."""
    from scipy import ndimage
    lim = th["limits"]; md = modes(o)
    if not (md & {"wave", "conform"}):
        return None
    slope = min(float(o.get("slope", lim["slope_deg"])), lim["slope_deg"])
    tslope = math.tan(math.radians(slope))
    skin = float(o.get("skin", 0.6))
    tau_hi = min(1.4, lim["layer_max_mm"] / layer_h); tau_lo = max(float(o.get("tau_min", 0.5)), lim["layer_min_mm"] / layer_h)
    auto_lam = None
    if "wave" in md and "amp" not in o:                      # as deep as the clearance allows, unless the plate says otherwise
        amp, auto_lam = auto_wave(mesh, o, th, layer_h, tau_hi, tau_lo, tslope)
    else:
        amp = float(o.get("amp", 0.4)) if "wave" in md else 0.0
    b = mesh.bounds
    c, du = 0.5, 0.25
    near_mm = 6.0                                        # limits held over the part and this far around it: its supports and brim stand there
    fade_mm = max(5.0, 3.0 * amp / tslope)               # the wave dies away outside the part over this distance, gently enough for the slope
    margin0 = near_mm + fade_mm
    while True:
        nx = int(math.ceil((b[1][0] - b[0][0] + 2 * margin0) / c)) + 1; ny = int(math.ceil((b[1][1] - b[0][1] + 2 * margin0) / c)) + 1
        nu = int(math.ceil((b[1][2] + 2.0) / du)) + 1
        if nx * ny * nu <= 9e6:
            break
        c *= 1.25; du *= 1.25
    x0, y0 = b[0][0] - margin0, b[0][1] - margin0
    occ, T, Tn = columns(mesh, x0, y0, c, nx, ny, du, nu)
    has = np.isfinite(T)
    near = ndimage.binary_dilation(has, iterations=max(1, int(round(near_mm / c))))       # where the limits are held: the part and near_mm around
    X = x0 + np.arange(nx) * c; Y = y0 + np.arange(ny) * c; U = np.arange(nu) * du
    rim = np.minimum.reduce(np.meshgrid(np.minimum(np.arange(nx), nx - 1 - np.arange(nx)) * c, np.minimum(np.arange(ny), ny - 1 - np.arange(ny)) * c, indexing="ij"))
    win = smooth(rim / (margin0 - near_mm)).astype(np.float32)                                # 1 over the part, 0 at the rim
    info = dict(mode="+".join(sorted(md)), slope_deg=slope, amp_mm=amp, cell_mm=round(c, 3), layer_mm=layer_h)

    conf = np.zeros((nx, ny, nu), np.float32)
    if "conform" in md:
        got = conform_relief(T, Tn, c, lim, o)
        if got is None:
            info["conform"] = "no shallow curved top to follow"
        else:
            r, curved, _ = got
            u0 = skin

            def fit(k):
                rk = k * r
                U1 = float((T - rk)[curved].min()) - skin
                if U1 - u0 < 1.0:
                    return None
                dz = rk[has] / (U1 - u0)
                if dz.max() > tau_hi - 1 or dz.min() < tau_lo - 1 or np.hypot(_fwd(rk, c, 0), _fwd(rk, c, 1))[near].max() > tslope * 1.001:
                    return None
                if rk[has].max() - rk[has].min() > lim["relief_mm"] - 2 * amp or not relief_ok(rk + amp, has, c, th, lim["margin_mm"] + amp):
                    return None
                return U1
            def solve():
                k, U1 = 1.0, fit(1.0)
                if U1 is None:
                    lo_k, hi_k = 0.0, 1.0
                    for _ in range(12):
                        mid = (lo_k + hi_k) / 2
                        if fit(mid) is None: hi_k = mid
                        else: lo_k = mid
                    k = lo_k; U1 = fit(k) if k > 0.02 else None
                return k, U1
            k, U1 = solve()
            if "amp" not in o and "wave" in md and (U1 is None or k < 0.99):     # the curved top is what the part is for: the wave gives its relief up
                for amp_try in (0.8, 0.6, 0.4, 0.0):
                    if amp_try >= amp:
                        continue
                    amp = amp_try; info["amp_mm"] = amp
                    k, U1 = solve()
                    if U1 is not None and k >= 0.99:
                        break
                if amp == 0.0:
                    md = md - {"wave"}; info["mode"] = "+".join(sorted(md))
            if U1 is None:
                info["conform"] = "the part is too thin or too steep under its curved top: left planar"
            else:
                g = np.clip((U - u0) / (U1 - u0), 0.0, 1.0)
                conf = (k * r * win)[:, :, None].astype(np.float32) * g[None, None, :].astype(np.float32)
                info["conform"] = "follows %.0f mm2 of curved top at %.0f%%, relief %.2f mm, blended between %.1f and %.1f mm" % (
                    curved.sum() * c * c, 100 * k, float((k * r)[has].max() - (k * r)[has].min()), u0, U1)
                info["conform_pct"] = round(100 * k)
                # the part as the slicer will see it (its top laid flat), for the wave's skins
                kz = np.clip(np.rint((U[None, None, :] + conf) / du).astype(np.int64), 0, nu - 1)
                occ = np.take_along_axis(occ, kz, axis=2)

    wv = np.zeros((nx, ny, nu), np.float32)
    if amp > 0:
        Dv = max(2.0, 1.5 * amp / min(tau_hi - 1, 1 - tau_lo) * 1.15)       # the wave grows over this height: thickness stays in range
        Dh = max(3.0, 1.5 * amp / tslope * 1.6)
        up = np.zeros_like(occ); up[:, :, :-1] = occ[:, :, 1:]
        dn = np.zeros_like(occ); dn[:, :, 1:] = occ[:, :, :-1]
        cap = ~occ & (up | dn)                               # air right over or under the part: a top, a bottom, an overhang
        cap[:, :, 0] = True                                  # and the bed
        n = ndimage.distance_transform_edt(~cap, sampling=(c / Dh, c / Dh, du / Dv)).astype(np.float32)
        w = smooth(n - skin / Dv).astype(np.float32) * win[:, :, None]
        lam = float(o.get("wavelength", 0)) or auto_lam or 2 * math.pi * amp / (tslope * 0.75)
        lam = max(lam, 2 * math.pi * amp / (tslope * 0.97))
        kx = 2 * math.pi / lam
        xc, yc = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
        base = (np.cos(kx * (X - xc))[:, None] * np.cos(kx * (Y - yc))[None, :]).astype(np.float32)
        for _ in range(60):
            wv = amp * w * base[:, :, None]
            v = _violation(conf + wv, c, du, tslope, tau_hi, tau_lo, near)
            if v.max() <= 1.0:
                break
            att = np.minimum(1.0, 0.97 / np.maximum(v, 1e-6))
            att = ndimage.minimum_filter(att, size=3)
            att = np.minimum(att, ndimage.gaussian_filter(att, 1.0))
            w = w * att
        else:
            v = _violation(conf + wv, c, du, tslope, tau_hi, tau_lo, near)
            w = w / max(1.0, float(v.max()) * 1.02); wv = amp * w * base[:, :, None]
        info.update(wavelength_mm=round(lam, 2), skin_mm=skin)
    d = conf + wv
    if float(np.abs(d).max()) < 0.02:
        info["note"] = "nothing to bend"
        return None
    gx, gy = np.gradient(d, c, axis=(0, 1)); gu = np.gradient(d, du, axis=2)
    sl = np.hypot(_fwd(d, c, 0), _fwd(d, c, 1))[near]
    gdu = np.diff(d, axis=2) / du
    d2 = np.abs(np.gradient(gx, c, axis=0)) + np.abs(np.gradient(gy, c, axis=1)) + 2 * np.abs(np.gradient(gx, c, axis=1))
    kappa = float(d2[near].max())
    inside = occ & near[:, :, None]
    bent = float((np.abs(d)[inside] > 0.25 * max(amp, 0.1)).mean()) if inside.any() else 0.0
    info.update(max_slope_deg=round(math.degrees(math.atan(float(sl.max()))), 2), tau=[round(1 + float(gdu[near].min()), 3), round(1 + float(gdu[near].max()), 3)],
                bend_mm=[round(float(d.min()), 3), round(float(d.max()), 3)], bent_share=round(bent, 3),
                piece_mm=round(float(np.clip(math.sqrt(8 * 0.01 / max(kappa, 1e-6)), 0.4, 1.2)), 2))
    return Field(x0, y0, c, du, d, info)


def _fwd(d, c, axis):
    """The steepest difference to either neighbour along an axis, per node (a central np.gradient hides a sharp step by half: the
    sliced path walks node to node and saw 13.5 deg where the field's gradient said 9.7)."""
    a = np.abs(np.diff(d, axis=axis)) / c
    pad = [(0, 0)] * d.ndim
    lo = list(pad); lo[axis] = (1, 0); hi = list(pad); hi[axis] = (0, 1)
    return np.maximum(np.pad(a, lo), np.pad(a, hi))


def _violation(d, c, du, tslope, tau_hi, tau_lo, near):
    gxy = np.hypot(_fwd(d, c, 0), _fwd(d, c, 1))
    gu = np.diff(d, axis=2) / du
    pu = np.pad(gu, ((0, 0), (0, 0), (1, 0))); pn = np.pad(gu, ((0, 0), (0, 0), (0, 1)))
    gup = np.maximum(pu, pn); gdn = np.maximum(-pu, -pn)
    v = np.maximum(gxy / tslope, np.maximum(gup / (tau_hi - 1), gdn / (1 - tau_lo)))
    v[~near] = 0.0
    return v


def unbend_mesh(mesh, fs):
    """The mesh the flat slicer gets: every vertex moved to its slicer height u (z = u + d). The same mesh back when nothing on
    its surface moves (the wave mode: all skins are planar)."""
    import trimesh
    v, f = trimesh.remesh.subdivide_to_size(np.asarray(mesh.vertices), np.asarray(mesh.faces), max_edge=1.0, max_iter=14)
    u = v[:, 2].copy()
    for _ in range(40):
        u = v[:, 2] - fs.d_at(v[:, 0], v[:, 1], u)
    if float(np.abs(u - v[:, 2]).max()) < 0.004:
        return mesh, 0.0
    v2 = v.copy(); v2[:, 2] = np.maximum(u, 0.0)
    return trimesh.Trimesh(v2, f, process=False), float(np.abs(u - v[:, 2]).max())


# ------------------------------------------------------------------------------------------------ bending the g-code

def _words(code):
    w = {}
    for t in code.split()[1:]:
        try:
            w[t[0]] = float(t[1:])
        except ValueError:
            pass
    return w


def body_range(lines):
    a = next(i for i, l in enumerate(lines) if l.startswith("; CHANGE_LAYER"))
    ends = [i for i, l in enumerate(lines) if l.startswith("; filament end gcode") or l.startswith("; MACHINE_END_GCODE_START")]
    ends = [i for i in ends if i > a]
    return a, (ends[0] if ends else len(lines))


def _arc(x0, y0, x1, y1, i, j, cw):
    cx, cy = x0 + i, y0 + j; r = math.hypot(i, j)
    a0 = math.atan2(y0 - cy, x0 - cx); a1 = math.atan2(y1 - cy, x1 - cx)
    sw = (a0 - a1) if cw else (a1 - a0)
    if sw <= 1e-9:
        sw += 2 * math.pi
    n = max(2, int(math.ceil(r * sw / 0.5)))
    return [(cx + r * math.cos(a0 + (-1 if cw else 1) * sw * k / n), cy + r * math.sin(a0 + (-1 if cw else 1) * sw * k / n), k / n) for k in range(1, n + 1)]


def bend_gcode(g, fs, th):
    """Bend a flat OrcaSlicer g-code (A1 mini, relative E) through the fields. Returns (g-code, stats)."""
    lim = th["limits"]
    lines = g.split("\n")
    a, b = body_range(lines)
    ops = []
    m = {k: [] for k in ("x0", "y0", "u0", "x1", "y1", "u1", "e", "f", "kind", "uL", "h")}
    x = y = u = None; Fs = None; uL = 0.0; h = 0.2; skip = False; lost = False

    def add(x1, y1, u1, e, ln):
        m["x0"].append(x); m["y0"].append(y); m["u0"].append(u); m["x1"].append(x1); m["y1"].append(y1); m["u1"].append(u1)
        m["e"].append(0.0 if e is None else e); m["f"].append(Fs or 0.0); m["kind"].append(0 if e is None else (1 if e > 0 else 2))
        m["uL"].append(uL); m["h"].append(h)
        ops.append(("mv", len(m["x0"]) - 1, ln))

    for ln in lines[a:b]:
        if skip:
            if ln.startswith("; SKIPPABLE_END"):
                skip = False
            ops.append(("skipz", ln) if re.match(r"^G1 Z[\d.]+\s*(;.*)?$", ln) else ("raw", ln))
            continue
        if ln.startswith("; SKIPPABLE_START"):
            skip = True; lost = True; ops.append(("raw", ln)); continue
        if ln.startswith("; Z_HEIGHT:"):
            uL = float(ln[11:]); ops.append(("raw", ln)); continue
        if ln.startswith("; LAYER_HEIGHT:"):
            h = float(ln[15:]); ops.append(("raw", ln)); continue
        if ln[:3] not in ("G0 ", "G1 ", "G2 ", "G3 "):
            ops.append(("raw", ln)); continue
        w = _words(ln.split(";", 1)[0])
        if "F" in w:
            Fs = w["F"]
        hasxy = "X" in w or "Y" in w
        if ln[1] in "23" and x is not None and u is not None and not lost:
            x1, y1 = w.get("X", x), w.get("Y", y); u1 = w.get("Z", u); e = w.get("E")
            pts = _arc(x, y, x1, y1, w.get("I", 0.0), w.get("J", 0.0), ln[1] == "2")
            pe = 0.0
            for px, py, tt in pts:
                add(px, py, u + (u1 - u) * tt, None if e is None else e * (tt - pe), None)
                x, y, u = px, py, u + (u1 - u) * (tt - pe); pe = tt
            x, y, u = x1, y1, u1
            continue
        if not hasxy:
            if "Z" in w and x is not None and not lost:
                u = w["Z"]; ops.append(("z", x, y, u, uL, w.get("F"), Fs))
            else:
                if "Z" in w:
                    u = w["Z"]
                ops.append(("rawg", ln, w.get("F"), w.get("Z")))
            continue
        x1, y1 = w.get("X", x), w.get("Y", y); u1 = w.get("Z", u)
        if x is None or y is None or u is None or lost or x1 is None or y1 is None:
            ops.append(("jump", x1, y1, u1, uL, w.get("E"), Fs, ln, lost))
            lost = False
        else:
            add(x1, y1, u1, w.get("E"), ln if ln[1] in "01" else None)
        x, y, u = x1, y1, u1

    M = {k: np.array(v, float) for k, v in m.items()}
    kind = M["kind"].astype(int); N = len(kind)
    L = np.hypot(M["x1"] - M["x0"], M["y1"] - M["y0"])
    n = np.maximum(1, np.ceil(L / fs.piece)).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(n + 1)])[:-1]
    mv = np.repeat(np.arange(N), n + 1)
    t = (np.arange(len(mv)) - off[mv]) / n[mv]
    xs = M["x0"][mv] + t * (M["x1"] - M["x0"])[mv]; ys = M["y0"][mv] + t * (M["y1"] - M["y0"])[mv]; us = M["u0"][mv] + t * (M["u1"] - M["u0"])[mv]
    lifted = (kind[mv] == 0) & (us - M["uL"][mv] > 0.05)
    ds = fs.d_at(xs, ys, us, lifted)
    zs = us + ds
    tau = np.ones(len(mv))
    ex = kind[mv] == 1
    if ex.any():
        hh = M["h"][mv][ex]
        tau[ex] = 1.0 + (ds[ex] - fs.d_at(xs[ex], ys[ex], np.maximum(us[ex] - hh, 0.0))) / hh
    last = off + n
    dev = np.abs(zs - (zs[off][mv] + t * (zs[last] - zs[off])[mv]))
    devmax = np.maximum.reduceat(dev, off) if N else np.zeros(0)
    trange = (np.maximum.reduceat(tau, off) - np.minimum.reduceat(tau, off)) if N else np.zeros(0)
    one = (devmax <= 0.004) & (trange <= 0.02)
    flat = one & (np.maximum.reduceat(np.abs(ds), off) < 2e-4) & (np.maximum.reduceat(np.abs(tau - 1), off) < 2e-3) if N else one
    tmean = np.add.reduceat(tau, off) / (n + 1) if N else one

    out = []; st = dict(pieces=0, bent=0, slope=0.0, t_min=9.0, t_max=0.0, vz=0.0, zmax=0.0)
    Fout = None; zout = None; zmax = 0.0
    vz_ex, vz_tr, v_np = lim["z_speed_mm_s"], 25.0, lim["speed_mm_s"]

    def piece(px, py, pz, e, f, lp, dz, kd, tt=None, hh=0.2):
        nonlocal Fout, zout, zmax
        s = "G1 X%.3f Y%.3f" % (px, py)
        sl = abs(dz) / lp if lp > 1e-6 else 0.0
        if sl > 0.004:
            f = min(f, 60.0 * (vz_tr if kd == 0 else vz_ex) / sl)
            if kd == 1:
                f = min(f, 60.0 * v_np)
        f = max(300.0, 60.0 * math.floor(f / 60.0)) if f else f
        if zout is None or abs(pz - zout) > 4e-4:
            s += " Z%.3f" % pz; zout = pz
        if e is not None:
            s += " E%.5f" % e
        if f and f != Fout:
            s += " F%d" % f; Fout = f
        out.append(s)
        st["pieces"] += 1
        if kd == 1:
            if pz > zmax: zmax = pz
            if lp > 0.2 and sl > st["slope"]: st["slope"] = sl
            if sl > 0.004:
                st["bent"] += 1; st["vz"] = max(st["vz"], sl * f / 60.0)
            if tt is not None and lp > 0.05:
                st["t_min"] = min(st["t_min"], tt * hh); st["t_max"] = max(st["t_max"], tt * hh)

    for op in ops:
        k0 = op[0]
        if k0 == "raw":
            out.append(op[1])
            if op[1].startswith("; SKIPPABLE_END"):
                Fout = None; zout = None
        elif k0 == "skipz":
            v = float(re.match(r"^G1 Z([\d.]+)", op[1]).group(1))
            out.append("G1 Z%.3f" % max(v, zmax + 0.4) if zmax + 0.4 > v else op[1])
        elif k0 == "rawg":
            out.append(op[1])
            if op[2] is not None: Fout = op[2]
            if op[3] is not None: zout = op[3]
        elif k0 == "z":
            _, px, py, pu, pl, f, fsl = op
            pz = pu + float(fs.d_at(np.array([px]), np.array([py]), np.array([pu]), np.array([pu - pl > 0.05]))[0])
            s = "G1 Z%.3f" % pz; zout = pz
            if f is not None:
                s += " F%d" % f; Fout = f
            elif fsl and Fout != fsl:
                s += " F%d" % fsl; Fout = fsl
            out.append(s)
        elif k0 == "jump":
            _, px, py, pu, pl, e, fsl, ln, was_lost = op
            if pu is None:
                out.append(ln)
                if "F" in _words(ln.split(";", 1)[0]): Fout = _words(ln.split(";", 1)[0])["F"]
                continue
            pz = pu + float(fs.d_at(np.array([px]), np.array([py]), np.array([pu]), np.array([e is None and pu - pl > 0.05]))[0])
            if was_lost and e is None:
                pz = max(pz, zmax + lim["hop_mm"])              # after a park the head crosses the plate above everything printed
            s = "G1 X%.3f Y%.3f Z%.3f" % (px, py, pz); zout = pz
            if e is not None: s += " E%.5f" % e
            if fsl: s += " F%d" % fsl; Fout = fsl
            out.append(s)
        else:
            i, ln = op[1], op[2]
            kd = kind[i]; f = M["f"][i]; o0 = off[i]; ni = n[i]; hh = M["h"][i]
            e = M["e"][i] if kd else None
            if flat[i] and ln is not None and (zout is not None and abs(zout - M["u1"][i]) < 4e-4 or " Z" in ln.split(";", 1)[0]):
                w = _words(ln.split(";", 1)[0])
                if "F" not in w and f and Fout != f:
                    ln = ln.split(";", 1)[0].rstrip() + " F%d" % f
                if f: Fout = f
                if "Z" in w: zout = w["Z"]
                out.append(ln); st["pieces"] += 1
                if kd == 1:
                    zmax = max(zmax, M["u1"][i]); st["t_min"] = min(st["t_min"], hh); st["t_max"] = max(st["t_max"], hh)
                continue
            if one[i]:
                ee = None if e is None else (e * tmean[i] if kd == 1 else e)
                piece(xs[o0 + ni], ys[o0 + ni], zs[o0 + ni], ee, f, L[i], zs[o0 + ni] - zs[o0], kd, tmean[i], hh)
            else:
                lp = L[i] / ni
                for j in range(1, ni + 1):
                    q = o0 + j
                    tt = 0.5 * (tau[q] + tau[q - 1])
                    ee = None if e is None else (e / ni * tt if kd == 1 else e / ni)
                    piece(xs[q], ys[q], zs[q], ee, f, lp, zs[q] - zs[q - 1], kd, tt, hh)

    tail = lines[b:]
    for i, ln in enumerate(tail):                              # the end routine's first lift clears the real top
        mm = re.match(r"^G1 Z([\d.]+)(.*)$", ln)
        if mm:
            if float(mm.group(1)) < zmax + 0.4:
                tail[i] = "G1 Z%.3f%s" % (zmax + 0.4, mm.group(2))
            break
    head = lines[:a]
    for i, ln in enumerate(head):
        if ln.startswith("; max_z_height:"):
            head[i] = "; max_z_height: %.2f" % max(zmax, float(ln.split(":")[1]))
    st.update(zmax=round(zmax, 3), slope_deg=round(math.degrees(math.atan(st.pop("slope"))), 2), vz=round(st["vz"], 1),
              t_min=round(st["t_min"], 3), t_max=round(st["t_max"], 3))
    return "\n".join(head + out + tail), st


# ------------------------------------------------------------------------------------------------ the gate: replay against the hotend

def read_moves(g):
    """Every move of the print body as it will run: start, end, extruding or not, its layer and feature."""
    lines = g.split("\n")
    a, b = body_range(lines)
    R = {k: [] for k in ("x0", "y0", "z0", "x1", "y1", "z1", "ext", "layer", "feat", "uL")}
    feats = {}; x = y = z = None; layer = -1; feat = 0; uL = 0.0; skip = False
    for ln in lines[a:b]:
        if skip:
            if ln.startswith("; SKIPPABLE_END"):
                skip = False; x = y = None
            continue
        if ln.startswith(";"):
            if ln.startswith("; SKIPPABLE_START"): skip = True
            elif ln.startswith("; CHANGE_LAYER"): layer += 1
            elif ln.startswith("; Z_HEIGHT:"): uL = float(ln[11:])
            elif ln.startswith("; FEATURE:"): feat = feats.setdefault(ln[10:].strip(), len(feats))
            continue
        if ln[:3] not in ("G0 ", "G1 "):
            continue
        w = _words(ln.split(";", 1)[0])
        x1, y1, z1 = w.get("X", x), w.get("Y", y), w.get("Z", z)
        if "X" in w or "Y" in w or "Z" in w:
            if None not in (x1, y1, z1):
                sx, sy, sz = (x1, y1, z1) if None in (x, y, z) else (x, y, z)
                R["x0"].append(sx); R["y0"].append(sy); R["z0"].append(sz); R["x1"].append(x1); R["y1"].append(y1); R["z1"].append(z1)
                R["ext"].append(w.get("E", 0.0) > 0 and ("X" in w or "Y" in w)); R["layer"].append(layer); R["feat"].append(feat); R["uL"].append(uL)
            x, y, z = x1, y1, z1
    out = {k: np.array(v, float) for k, v in R.items()}
    out["ext"] = out["ext"].astype(bool); out["layer"] = out["layer"].astype(int); out["feat"] = out["feat"].astype(int)
    out["feats"] = [k for k, _ in sorted(feats.items(), key=lambda kv: kv[1])]
    # what runs before a layer's first printed line (the wipe at the old height, the lift, the travel in) meets the layer below
    # only: nothing of the new layer exists yet
    N = len(out["layer"])
    if N:
        first = np.full(int(out["layer"].max()) + 2, N)
        np.minimum.at(first, out["layer"][out["ext"]], np.nonzero(out["ext"])[0])
        pre = np.arange(N) < first[out["layer"]]
        out["sim_layer"] = np.maximum(out["layer"] - pre.astype(int), 0)
    else:
        out["sim_layer"] = out["layer"]
    return out


def _samples(mv, sel, step):
    """Points every `step` mm along the selected moves: x, y, z, index of the move."""
    x0, y0, z0, x1, y1, z1 = (mv[k][sel] for k in ("x0", "y0", "z0", "x1", "y1", "z1"))
    n = np.maximum(1, np.ceil(np.hypot(x1 - x0, y1 - y0) / step)).astype(np.int64)
    idx = np.repeat(np.arange(len(n)), n + 1)
    off = np.concatenate([[0], np.cumsum(n + 1)])[:-1]
    t = (np.arange(len(idx)) - off[idx]) / n[idx]
    return x0[idx] + t * (x1 - x0)[idx], y0[idx] + t * (y1 - y0)[idx], z0[idx] + t * (z1 - z0)[idx], np.nonzero(sel)[0][idx]


def simulate(g, th, cell=0.25):
    """Replay the g-code layer by layer against the hotend. Each layer's material is taken as all there already (safe side), then
    every point the nozzle visits is tested: nothing may stand higher than the hotend clears at its offset from the tip.
    ok is False on a collision, a move steeper than the hard slope, or the flat tip ploughing deeper than that slope implies."""
    from scipy import ndimage
    lim = th["limits"]; margin = lim["margin_mm"]
    mv = read_moves(g)
    if not len(mv["x0"]):
        return dict(ok=False, why="no moves found")
    ext = mv["ext"]
    X0 = min(mv["x0"].min(), mv["x1"].min()) - 1.0; Y0 = min(mv["y0"].min(), mv["y1"].min()) - 1.0
    nx = int(math.ceil((max(mv["x0"].max(), mv["x1"].max()) + 1.0 - X0) / cell)) + 1
    ny = int(math.ceil((max(mv["y0"].max(), mv["y1"].max()) + 1.0 - Y0) / cell)) + 1
    rf = th["nozzle"]["flat_d"] / 2
    tol = math.tan(math.radians(lim["slope_hard_deg"])) * (rf + 0.21 + 1.5 * cell) + 0.03       # the flat tip on the steepest allowed slope
    R1 = 5.5
    nn = int(math.ceil(R1 / cell)); ax = np.arange(-nn, nn + 1)
    OX, OY = np.meshgrid(ax, ax, indexing="ij")
    Ef = env_height(th, OX * cell, OY * cell)
    allow = np.maximum(tol, Ef - margin)
    near = [(int(i), int(j), float(allow[i + nn, j + nn])) for i in ax for j in ax if math.hypot(i, j) * cell <= R1]
    disc = (OX * OX + OY * OY) * cell * cell <= rf * rf
    fc = max(1, int(round(1.0 / cell))); cc = cell * fc
    H = np.zeros((nx, ny), np.float32)
    res = dict(layers=0, bent_layers=0, plough_mm=0.0, clear_mm=None, far_clear_mm=None, worst=None, slope_deg=0.0)
    lx = np.hypot(mv["x1"] - mv["x0"], mv["y1"] - mv["y0"])
    sl = np.where(ext & (lx > 0.2), np.abs(mv["z1"] - mv["z0"]) / np.maximum(lx, 1e-9), 0.0)
    res["slope_deg"] = round(math.degrees(math.atan(float(sl.max()))), 2)
    worst = np.inf; worst_at = None; far_worst = np.inf
    for ly in range(int(mv["layer"].max()) + 1):
        inl = mv["sim_layer"] == ly
        if not inl.any():
            continue
        res["layers"] += 1
        ex, ey, ez, _ = _samples(mv, inl & ext, cell)
        if len(ex):
            dep = np.full((nx, ny), -np.inf, np.float32)
            np.maximum.at(dep, (np.rint((ex - X0) / cell).astype(int), np.rint((ey - Y0) / cell).astype(int)), ez.astype(np.float32))
            Hk = np.maximum(H, ndimage.maximum_filter(dep, size=3))          # a bead is about three cells wide
        else:
            Hk = H
        px, py, pz, _ = _samples(mv, inl, 2 * cell)
        pi = np.rint((px - X0) / cell).astype(int); pj = np.rint((py - Y0) / cell).astype(int)
        hmax = float(Hk.max()); zlow = float(pz.min())
        if hmax - zlow > tol - 1e-6:                              # something stands over the lowest tip position: look closely
            res["bent_layers"] += 1
            D = np.full((nx, ny), -np.inf, np.float32)
            for i, j, al in near:                               # D(p) = highest (material at p + offset - what the hotend allows there)
                D[max(0, -i):nx - max(0, i), max(0, -j):ny - max(0, j)] = np.maximum(
                    D[max(0, -i):nx - max(0, i), max(0, -j):ny - max(0, j)], Hk[max(0, i):nx - max(0, -i), max(0, j):ny - max(0, -j)] - al)
            clr = pz - D[pi, pj]
            k = int(clr.argmin())
            if clr[k] < worst:
                worst = float(clr[k]); worst_at = (round(float(px[k]), 2), round(float(py[k]), 2), round(float(pz[k]), 3), ly)
            pl = ndimage.maximum_filter(Hk, footprint=disc)[pi, pj] - pz
            res["plough_mm"] = max(res["plough_mm"], float(pl.max()))
            if hmax - zlow > th["block"]["h"] - margin:           # and against the block, ducts and cover
                nxc, nyc = nx // fc, ny // fc
                hi = Hk[:nxc * fc, :nyc * fc].reshape(nxc, fc, nyc, fc).max(axis=(1, 3))
                lo = np.full((nxc, nyc), np.inf, np.float32)
                ci = np.minimum(pi // fc, nxc - 1); cj = np.minimum(pj // fc, nyc - 1)
                np.minimum.at(lo, (ci, cj), pz.astype(np.float32))
                far_worst = min(far_worst, _far_clear(hi, lo, cc, th, margin))
        H = Hk
    res["plough_mm"] = round(res["plough_mm"], 3); res["plough_allowed_mm"] = round(tol, 3)
    res["clear_mm"] = None if not np.isfinite(worst) else round(worst, 3)
    res["far_clear_mm"] = None if not np.isfinite(far_worst) else round(float(far_worst) + margin, 3)
    res["worst"] = worst_at
    bad = []
    if np.isfinite(worst) and worst < -1e-4:
        bad.append("the hotend would touch the print by %.2f mm at x %.1f y %.1f z %.2f (layer %d)" % ((-worst,) + worst_at))
    if np.isfinite(far_worst) and far_worst < -1e-4:
        bad.append("the hotend's block or ducts come within %.2f mm of higher material (needs %.1f)" % (far_worst + margin, margin))
    if res["slope_deg"] > lim["slope_hard_deg"] + 0.3:
        bad.append("a printing move climbs at %.1f deg, over the %.0f deg limit" % (res["slope_deg"], lim["slope_hard_deg"]))
    res["ok"] = not bad; res["why"] = "; ".join(bad)
    return res


# ------------------------------------------------------------------------------------------------ files and pictures

def read_3mf_gcode(path):
    z = zipfile.ZipFile(path)
    name = next(n for n in z.namelist() if n.endswith(".gcode"))
    return z.read(name).decode("utf8", "replace"), name


def write_3mf_gcode(src, dst, g, name):
    """Copy the sliced 3mf with its g-code replaced and the md5 beside it recomputed."""
    data = g.encode("utf8")
    tmp = dst + ".tmp"
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            if it.filename == name:
                zo.writestr(it.filename, data)
            elif it.filename == name + ".md5":
                zo.writestr(it.filename, hashlib.md5(data).hexdigest().upper())
            else:
                zo.writestr(it, zi.read(it.filename))
    os.replace(tmp, dst)


MARK = "; NONPLANAR"


def bend_file(path, fields, th, png=None, title=""):
    """Bend a sliced .gcode.3mf in place, run the gate, draw the preview. Returns the report: {ok, checks, bend, gate, fields}."""
    g, name = read_3mf_gcode(path)
    fs = Fields(fields)
    if fs.fields:
        g2, st = bend_gcode(g, fs, th)
    else:
        g2, st = g, dict(note="flat slicer output (Z anti-aliasing only)")
    gate = simulate(g2, th)
    lim = th["limits"]
    checks = [("non-planar: the hotend clears the print everywhere (replayed against the A1 mini hotend)", gate["ok"],
               gate["why"] or "closest %s mm, block and ducts %s mm, tip rides %.2f mm into the slope (%.2f allowed), steepest move %.1f deg" % (
                   gate["clear_mm"], gate["far_clear_mm"], gate["plough_mm"], gate["plough_allowed_mm"], gate["slope_deg"]))]
    if "t_min" in st and st["t_max"] > 0:
        okt = st["t_min"] >= lim["layer_min_mm"] - 0.005 and st["t_max"] <= lim["layer_max_mm"] + 0.005
        checks.append(("non-planar: every layer inside the nozzle's %.2f-%.2f mm" % (lim["layer_min_mm"], lim["layer_max_mm"]), okt,
                       "%.3f to %.3f mm" % (st["t_min"], st["t_max"])))
    ok = all(c[1] for c in checks)
    lines = g2.split("\n")
    lines = [l for l in lines if not l.startswith(MARK)]
    at = next((i for i, l in enumerate(lines) if l.startswith("; HEADER_BLOCK_END")), 0) + 1
    lines[at:at] = ["%s: tools/nonplanar.py v%s %s" % (MARK, VERSION, "; ".join(f.info["mode"] for f in fs.fields) or "zaa"),
                    "%s_CHECK: %s" % (MARK, "PASS" if ok else "FAIL")]
    g2 = "\n".join(lines)
    write_3mf_gcode(path, path, g2, name)
    rep = dict(ok=ok, checks=checks, bend=st, gate=gate, fields=[f.info for f in fs.fields])
    if png:
        try:
            preview(g2, png, title, rep, th)
            rep["preview"] = png
        except Exception as e:                                  # a picture never decides readiness
            rep["preview_error"] = str(e)
    return rep


def gate_file(path, th=None):
    """(ok, why) for tools/bambu_send.py: a non-planar file is sent only when this replay passes."""
    g, _ = read_3mf_gcode(path)
    if MARK not in g:
        return True, "planar"
    if MARK + "_CHECK: PASS" not in g:
        return False, "non-planar file whose own check failed"
    r = simulate(g, th or toolhead())
    return r["ok"], r["why"] or "non-planar: hotend clears the print (closest %s mm, steepest move %.1f deg)" % (r["clear_mm"], r["slope_deg"])


COL = {"Outer wall": "#c0392b", "Inner wall": "#e67e22", "Sparse infill": "#2980b9", "Internal solid infill": "#8e44ad", "Top surface": "#16a085",
       "Bottom surface": "#7f8c8d", "Support": "#27ae60", "Support interface": "#2ecc71", "Gap infill": "#bdc3c7", "Bridge": "#34495e",
       "Overhang wall": "#d35400", "Brim": "#95a5a6", "Skirt": "#95a5a6"}


def preview(g, png, title="", rep=None, th=None):
    """One picture of the bent toolpaths: two cuts through the print (true scale), one layer from above, the hotend drawn to scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = th or toolhead()
    mv = read_moves(g)
    sel = mv["ext"]
    x, y, z, idx = _samples(mv, sel, 0.15)
    bend = z - mv["uL"][idx]
    feat = mv["feat"][idx]; layer = mv["layer"][idx]
    body = np.array([mv["feats"][f] not in ("Brim", "Skirt") for f in feat])
    part = body & np.array([not mv["feats"][f].startswith("Support") for f in feat])      # the cuts are placed on the part, never on its supports
    ref = part if part.any() and np.abs(bend[part]).max() > 0.02 else body
    big = ref & (np.abs(bend) >= 0.6 * np.abs(bend[ref]).max()) if np.abs(bend[ref]).max() > 0.02 else ref
    yc = float(np.median(y[big & body])) if (big & body).any() else float(np.median(y))
    xc = float(np.median(x[big & body])) if (big & body).any() else float(np.median(x))
    yc = y[np.abs(y - yc).argmin()]; xc = x[np.abs(x - xc).argmin()]
    fig = plt.figure(figsize=(16, 11), dpi=110)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.25], hspace=0.28, wspace=0.12)

    W, Hh = 26.0, 6.5                                    # the cuts are a zoomed window: wide enough for a wave, a layer is ~12 px

    def cut(ax, along, across, c0, name, other, hotend=False):
        s = (np.abs(across - c0) <= 0.22) & body
        lo, hi = float(along[body].min()), float(along[body].max())
        zt = float(z[body].max())
        sb = s & big
        mid = float(np.median(along[sb])) if sb.any() else (lo + hi) / 2
        zm = float(np.median(z[sb])) if sb.any() else zt / 2
        a0 = max(lo - 1.0, min(mid - W / 2, hi + 1.0 - W)); a1 = a0 + W
        z0 = max(-0.2, min(zm - Hh / 2, zt + 0.6 - Hh)); z1 = z0 + Hh
        s &= (along >= a0) & (along <= a1) & (z >= z0 - 0.3) & (z <= z1 + 0.3)
        for f in np.unique(feat[s]):
            q = s & (feat == f)
            ax.scatter(along[q], z[q] - 0.1, s=2.0, c=COL.get(mv["feats"][f], "#555555"), linewidths=0, label=mv["feats"][f])
        if hotend and s.any():                           # the hotend's underside, to scale, its tip resting on the one layer it is printing:
            lsel = int(np.bincount(layer[s & part]).argmax()) if (s & part).any() else int(layer[s].max())      # that layer and all below are drawn solid, the rest faint
            on = s & (layer == lsel)
            j = int(np.abs(along[on] - (a0 + W * 0.5)).argmin())
            tx, tz = float(along[on][j]), float(z[on][j])
            hx = np.linspace(-13, 13, 521)
            hz = env_height(th, hx, np.zeros_like(hx)) if name == "x" else env_height(th, np.zeros_like(hx), hx)
            ax.fill_between(tx + hx, tz + hz, z1 + 0.3, color="#333333", alpha=0.30, linewidth=0)
            later = s & (layer > lsel)
            ax.scatter(along[later], z[later] - 0.1, s=2.6, c="white", alpha=0.75, linewidths=0)     # not printed yet when the tip is here
            ax.text(a0 + 0.4, z1 - 0.45, "grey: hotend underside to scale, tip on layer %d; lines above it are not printed yet" % (lsel + 1), fontsize=8, color="#111111")
        ax.set_xlim(a0, a1); ax.set_ylim(z0, z1); ax.set_aspect("equal")
        ax.set_title("cut at %s = %.1f mm, zoomed to %.0f x %.1f mm where the layers bend most (true scale)" % (other, c0, W, Hh), fontsize=10, loc="left")
        ax.set_xlabel(name + " (mm)"); ax.set_ylabel("z (mm)")

    ax1 = fig.add_subplot(gs[0, :]); ax2 = fig.add_subplot(gs[1, :])
    cut(ax1, x, y, yc, "x", "y", hotend=True)
    cut(ax2, y, x, xc, "y", "x")
    ax2.legend(loc="lower left", fontsize=7, ncol=4, markerscale=4, frameon=True, framealpha=0.85)

    ax3 = fig.add_subplot(gs[2, 0])
    rng = np.array([np.ptp(bend[(layer == l) & ref]) if ((layer == l) & ref).any() else 0.0 for l in range(int(layer.max()) + 1)])
    lsel = int(rng.argmax())
    q = (layer == lsel) & body
    lim_c = max(0.05, float(np.abs(bend[q]).max()))
    sc = ax3.scatter(x[q], y[q], c=bend[q], s=1.2, cmap="RdBu_r", vmin=-lim_c, vmax=lim_c, linewidths=0)
    ax3.set_aspect("equal"); ax3.set_title("layer %d of %d from above: how far each line sits over (+) or under (-) its flat height, mm" % (lsel + 1, int(layer.max()) + 1), fontsize=10, loc="left")
    fig.colorbar(sc, ax=ax3, shrink=0.8)
    ax4 = fig.add_subplot(gs[2, 1]); ax4.axis("off")
    txt = [title or "non-planar slice", ""]
    if rep:
        for f in rep.get("fields", []):
            txt.append("field: %s, steepest %.1f deg, layer thickness x%.2f to x%.2f," % (f["mode"], f["max_slope_deg"], f["tau"][0], f["tau"][1]))
            txt.append("       bend %+.2f to %+.2f mm, %d%% of the part's inside bent" % (f["bend_mm"][0], f["bend_mm"][1], round(100 * f["bent_share"])))
            if f.get("conform"): txt.append("       conform: " + f["conform"])
            if f.get("wavelength_mm"): txt.append("       wave %.2f mm high every %.1f mm, skins of %.1f mm left planar" % (f["amp_mm"], f["wavelength_mm"], f["skin_mm"]))
        b = rep.get("bend", {})
        if "pieces" in b:
            txt.append("g-code: %d moves, %d of them bent, layers %.2f to %.2f mm thick, Z at most %.1f mm/s" % (b["pieces"], b["bent"], b["t_min"], b["t_max"], b["vz"]))
        txt.append("")
        for n_, ok_, note in rep.get("checks", []):
            txt.append(("PASS  " if ok_ else "FAIL  ") + n_)
            txt.append("      " + note)
        txt.append("")
        txt.append("READY TO TEST-PRINT (nothing was sent)" if rep.get("ok") else "NOT READY: do not print")
    ax4.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=9, family="monospace", wrap=True)
    fig.suptitle(title or "non-planar slice", fontsize=13, x=0.01, ha="left")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["check", "preview"]); ap.add_argument("file"); ap.add_argument("png", nargs="?")
    a = ap.parse_args()
    th = toolhead()
    g, _ = read_3mf_gcode(a.file) if a.file.endswith(".3mf") else (open(a.file, encoding="utf8", errors="replace").read(), "")
    if a.cmd == "check":
        r = simulate(g, th)
        print(json.dumps(r, indent=1))
        print("PASS: the hotend clears the print" if r["ok"] else "FAIL: " + r["why"])
        sys.exit(0 if r["ok"] else 1)
    stem = os.path.splitext(a.file)[0].replace(".gcode", "")
    png = a.png or stem + ".preview.png"
    rep = dict(checks=[], fields=[], bend={}, ok=None)
    if os.path.exists(stem + ".report.json"):           # the plate's own report: its checks and field go on the picture, never a new verdict
        rj = json.load(open(stem + ".report.json", encoding="utf8"))
        npj = rj.get("nonplanar", {})
        rep = dict(checks=[c for c in rj.get("checks", []) if str(c[0]).startswith("non-planar")], fields=npj.get("fields", []),
                   bend=npj.get("bend", {}), ok=rj.get("ready"))
    else:
        rep["ok"] = simulate(g, th)["ok"]
    preview(g, png, os.path.basename(stem) + ": non-planar", rep, th)
    print(png)


if __name__ == "__main__":
    main()
