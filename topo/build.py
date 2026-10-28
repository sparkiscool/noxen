"""From the optimised field to a printable part, and the checks that judge it.

Surface: the filtered field x_tilde is smooth by construction (it is a cone-filtered density), so its eta_n
iso-surface IS the blueprint boundary, and near a drawn body the filter has already blended the member into the body
(a fillet whose radius is the filter radius). It is sampled with cubic interpolation at a fine pitch, held
`clip_margin` inside every keep-out (except where it meets a drawn body), meshed by marching cubes and lightly Taubin
smoothed. Then, exactly: (generated + drawn) intersect envelope minus cuts. Mounting features stay exact.

Checks: nox/solid_check.py (the hard gate, unchanged), function check (drawn complete, nothing in cuts or keep-outs),
as-built strength (voxel FE of the built part, every real load case, layer-aware failure indices), mass.
"""
import os
import json
import time
import importlib.util
import numpy as np
from scipy import ndimage as ndi

from .part import tm, man, voxelise
from .material import allowables
from .fem import Solver

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------------ surface
def surface(layout, pitch=None, clip_margin=0.5, max_points=60e6, taubin_iters=8, log=print):
    import trimesh
    from skimage import measure
    part = layout.part
    h = part.h
    shape = part.shape
    eta_n = layout.s.eta[1]
    xt = layout.result["xt"].reshape(shape)
    up = max(2, int(round(h / (pitch or 0.3))))
    while np.prod(np.array(shape) * up) > max_points and up > 2:
        up -= 1
    p = h / up
    t0 = time.time()
    xf = ndi.zoom(xt, up, order=3, mode="grid-constant", cval=0.0, grid_mode=True)
    o = np.asarray(part.origin)
    axes = [o[k] + (np.arange(xf.shape[k]) + 0.5) * p for k in range(3)]
    if layout.s.finish == "organic":
        return _organic_surface(layout, xf, axes, p, up, clip_margin, t0, log)
    f = (xf - eta_n) / max(1e-9, 1.0 - eta_n)                              # > 0 inside, ~ 1 per filter radius
    del xf
    # keep-out margin: generated material stays clip_margin inside env - cuts, except near drawn bodies
    allowed = part.env_m - part.cuts_m
    A = voxelise(tm(allowed), *axes)
    din = ndi.distance_transform_edt(A) * p
    D = voxelise(part.drawn_t, *axes)
    near_drawn = ndi.distance_transform_edt(~D) * p <= 2.0 * clip_margin + p
    g = np.where(near_drawn, din, din - clip_margin) / layout.R_mm           # same scale as f
    del din, near_drawn
    F = np.minimum(f, g)
    F[~A] = np.minimum(F[~A], -0.05)
    F = np.pad(F, 1, constant_values=-1.0)
    F[np.abs(F) < 1e-4] = -1e-4                                           # exact zeros make degenerate MC faces
    gen_vox = float((F > 0).sum()) * p ** 3
    v, fc, _, _ = measure.marching_cubes(F.astype(np.float32), 0.0, spacing=(p, p, p))
    v += o - p / 2.0 + np.array([1.1e-4, 0.7e-4, 0.9e-4])                  # pad offset (+0.5 p cell centre - 1 p pad)
    mesh = trimesh.Trimesh(v, fc, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if taubin_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=taubin_iters)
    log(f"build: surface at pitch {p:.3f} mm (x{up}), generated field {gen_vox / 1000:.2f} cm3, mesh "
        f"{len(mesh.faces):,} faces, {time.time() - t0:.0f} s")
    return mesh


def _sdf(M, p):
    """signed distance in mm of a voxel mask, positive outside."""
    return ((ndi.distance_transform_edt(~M) - ndi.distance_transform_edt(M)) * p).astype(np.float32)


def _smin(a, b, k):
    """polynomial smooth minimum (union of two signed distances with a blend of width k mm)."""
    if k <= 0:
        return np.minimum(a, b)
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - hh) + a * hh - k * hh * (1.0 - hh)


def skin_field(part, axes, p, skin_mm):
    """the functional skins at the fine pitch, from the exact bodies (same rule as Part.functional_skin): drawn voxels
    (minus cuts) within skin_mm of a cut or of the outside of the envelope as given, plus drawn voxels outside it, plus
    every drawn body that touches neither."""
    D = voxelise(part.drawn_t, *axes)
    C = voxelise(part.cuts_t, *axes)
    O = ~voxelise(part.env0_t, *axes)
    D &= ~C
    d = (ndi.distance_transform_edt(~(C | O)) - 0.5) * p
    K = D & ((d <= skin_mm) | O)
    lab, n = ndi.label(D, np.ones((3, 3, 3), bool))
    if n:
        has = np.asarray(ndi.maximum(K, lab, np.arange(1, n + 1)), bool)
        K |= np.r_[False, ~has][lab] & D
        # the bodies the main support holds stay whole (Part.held_bodies): a fine body is held where it meets one
        o = np.asarray(part.origin)
        ijk = [np.clip(np.floor((np.asarray(axes[k]) - o[k]) / part.h).astype(int), 0, part.shape[k] - 1)
               for k in range(3)]
        Hc = part.held_bodies().reshape(part.shape)[np.ix_(*ijk)]
        held = np.unique(lab[Hc & D])
        K |= np.isin(lab, held[held > 0]) & D
    return K, C, O


def _fmm(phi, p, band):
    """signed distance in mm, positive where phi > 0, from a continuous field (the zero level is found to subcell
    accuracy, so planes and curves come out without voxel steps); computed in a narrow band, +-band outside it."""
    import skfmm
    phi = np.asarray(phi, np.float64)
    if phi.min() >= 0 or phi.max() <= 0:
        return np.full(phi.shape, band if phi.max() > 0 else -band, np.float32)
    d = skfmm.distance(phi, dx=p, narrow=band)
    if np.ma.isMaskedArray(d):
        d = np.where(d.mask, np.where(phi > 0, band, -band), d.data)
    return np.clip(d, -band, band).astype(np.float32)


def _mask_dist(M, p, band):
    """signed distance in mm of a voxel mask, positive inside: the mask is blurred by one voxel and its 0.5 level
    redistanced, so a voxelised plane or cylinder comes back flat or round instead of terraced."""
    return _fmm(ndi.gaussian_filter(M.astype(np.float32), 1.0) - 0.5, p, band)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


def _organic_surface(layout, xf, axes, p, up, clip_margin, t0, log, taubin_iters=6):
    """one smooth field: the generated region (eta_n level set of the filtered field, redistanced from the continuous
    field, held clip_margin inside env - cuts with a rounded fillet) smooth-unioned with the functional skins
    (fillet_mm), the skins' free rims rounded by skin_round_mm (a morphological opening); the skins run 1 mm into the
    cuts and past the envelope so the exact booleans in assemble() cut the holes and contact faces flat and true."""
    import trimesh
    from skimage import measure
    part, s = layout.part, layout.s
    eta_n = s.eta[1]
    band = 2.0 * max(s.fillet_mm, s.skin_round_mm, clip_margin) + 4.0 * p + 2.0
    d_gen = -_fmm(xf - eta_n, p, band)                                     # > 0 outside the generated material
    del xf
    A = voxelise(tm(part.env_m - part.cuts_m), *axes)
    d_off = clip_margin - _mask_dist(A, p, band)                          # > 0 within clip_margin of a keep-out
    K, C, O = skin_field(part, axes, p, s.skin_mm)
    Kx = K | ((C | O) & (ndi.distance_transform_edt(~K) * p <= 1.0))
    r = s.skin_round_mm
    # the part of the skin the function check holds to: all of it but the free rim that the rounding may take
    Kc = K & (ndi.distance_transform_edt(K | C | O) * p > r + p) if r > 0 else K
    del C, O
    d_skin = -_mask_dist(Kx, p, band)
    del Kx
    if r > 0:                                                             # opening: round the skins' free convex rims
        d_skin = -_fmm(-(d_skin + r), p, band) - r
    # the build keeps the layout's volume: one level offset t (mm) of the generated field before it is held off the
    # keep-outs (edge rounded, not creased), so the clip, the rounding and the fillets neither thin nor fatten the
    # optimised design (organic-v2: -6 % without it)
    target = float(layout.result["rho_n"].sum()) * part.h ** 3
    field = lambda t: _smin(_smax(d_gen - t, d_off, 2.0 * clip_margin), d_skin, s.fillet_mm)
    vol = lambda t: float(((field(t) < 0) & A).sum()) * p ** 3
    lo_t, hi_t = -1.0, 1.5
    if vol(lo_t) > target:
        t = lo_t
    elif vol(hi_t) < target:
        t = hi_t
    else:
        for _ in range(18):
            t = 0.5 * (lo_t + hi_t)
            lo_t, hi_t = (t, hi_t) if vol(t) < target else (lo_t, t)
        t = 0.5 * (lo_t + hi_t)
    d = field(t)
    vt = float(((d < 0) & A).sum()) * p ** 3
    del d_gen, d_skin, d_off, A
    F = np.pad(-d, 1, constant_values=-band)
    del d
    # samples exactly on the level give degenerate marching-cubes faces; Taubin then pulls spikes off them through
    # the keep-out margin and the exact cut leaves voxel slivers (holder organic-v2: 27 shells, 5 voxel bodies)
    F[np.abs(F) < 1e-4] = -1e-4
    gen_vox = float((F > 0).sum()) * p ** 3
    v, fc, _, _ = measure.marching_cubes(F.astype(np.float32), 0.0, spacing=(p, p, p))
    del F
    o = np.asarray(part.origin)
    v += o - p / 2.0 + np.array([1.1e-4, 0.7e-4, 0.9e-4])
    mesh = trimesh.Trimesh(v, fc, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if taubin_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=taubin_iters)
    layout._skin_fine = (Kc, axes, p)
    log(f"build: organic surface at pitch {p:.3f} mm (x{up}), skins {K.sum() * p ** 3 / 1000:.2f} cm3, field "
        f"{gen_vox / 1000:.2f} cm3, in env - cuts {vt / 1000:.2f} cm3 (layout {target / 1000:.2f}, level offset {t:+.3f} mm), "
        f"fillet {s.fillet_mm} mm, rims {r} mm, mesh {len(mesh.faces):,} faces, "
        f"{time.time() - t0:.0f} s")
    return mesh


def assemble(layout, gen_mesh, speck_mm3=20.0, log=print):
    """the part as one manifold: organic finish = smooth field ^ env - cuts (exact holes and contact faces); drawn
    finish = (generated + drawn) ^ env - cuts. Loose specks under speck_mm3 dropped and reported."""
    part = layout.part
    S = man(gen_mesh)
    if layout.s.finish == "organic":
        P = (S ^ part.env_m) - part.cuts_m
    else:
        P = ((S + part.drawn_m) ^ part.env_m) - part.cuts_m
    bodies = P.decompose()
    vols = sorted([b.volume() for b in bodies], reverse=True)
    dropped, cavities = [], []
    if len(bodies) > 1:
        specks = None
        for b in bodies:
            v = b.volume()
            if v < 0:
                # an inverted shell is a sealed cavity of a body (the nominal field enclosed a void: a hollow
                # section). It stays: filling it adds mass the optimiser did not ask for (bar_half topo-v2: +4.8 g,
                # same worst index), and it prints as an internal wall.
                cavities.append(round(-v, 2))
            elif v < speck_mm3:
                dropped.append(round(v, 2))
                specks = b if specks is None else specks + b
        if specks is not None:
            P = P - specks
        log(f"build: {len(bodies)} shells, volumes {[round(v_, 1) for v_ in vols[:6]]} mm3; dropped specks {dropped}; "
            f"sealed cavities kept {cavities} mm3")
    return P, dict(bodies_raw=len(bodies), dropped_specks_mm3=dropped, cavities_mm3=cavities,
                   bodies=len([b for b in P.decompose() if b.volume() > 0]))


# -------------------------------------------------------------------------------------------------------- checks
def _solid_check_module():
    spec = importlib.util.spec_from_file_location("nox_solid_check", os.path.join(HERE, "..", "tools", "solid_check.py"))
    SC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SC)

    def _voxels(mesh, pitch, lo, shape):
        ax = [lo[k] + (np.arange(shape[k]) + 0.5) * pitch for k in range(3)]
        return voxelise(mesh, *ax)
    SC.voxels = _voxels
    return SC


def solid_check(mesh, drawn_meshes):
    return _solid_check_module().solid_check(mesh, drawn=drawn_meshes)


def function_check(P, part, layout=None):
    """drawn finish: every drawn body complete (exact). Organic finish: every functional skin present (voxels at the
    build pitch, without the free rim within skin_round_mm + 1 voxel of the design side, which the build rounds on
    purpose; 1 % allowance). Both: nothing in the cuts or outside the
    envelope (exact)."""
    in_cuts = (P ^ part.cuts_m).volume()
    in_keep = (P - (part.env_m)).volume()
    if layout is not None and getattr(layout, "_skin_fine", None) is not None:
        K, axes, p = layout._skin_fine
        Pv = voxelise(tm(P), *axes)
        skin = float(K.sum()) * p ** 3
        missing = float((K & ~Pv).sum()) * p ** 3
        ok = missing <= 0.01 * skin and max(in_cuts, in_keep) < 0.5
        return dict(skin_mm3=round(skin, 1), skin_missing_mm3=round(missing, 2), in_cuts_mm3=round(in_cuts, 2),
                    outside_env_mm3=round(in_keep, 2), ok=bool(ok))
    fin = part.drawn_m - part.cuts_m
    missing = (fin - P).volume()
    return dict(drawn_missing_mm3=round(missing, 2), in_cuts_mm3=round(in_cuts, 2), outside_env_mm3=round(in_keep, 2),
                ok=bool(max(missing, in_cuts, in_keep) < 0.5))


def strength(layout, part_mesh, log=print):
    """as-built voxel FE on the layout grid: every real load case at nominal loads, transversely isotropic material,
    failure indices in-layer (von Mises / A_xy) and across layers (sqrt((sigma_n+ / A_z)^2 + (tau_n / A_s)^2)) at
    element centres; elements touching a loaded or fixed node are left out (point-load singularities)."""
    part = layout.part
    ins = getattr(part, "insert_cells", np.zeros(part.nel, bool))
    A = voxelise(part_mesh, *part.axes).ravel() | ins                     # inserts (rods) sit in their bores
    E = np.where(ins, getattr(part, "stiff", np.ones(part.nel)), np.where(A, 1.0, 1e-9))
    real = [c for c in layout.cases if not c.tie]
    sol = Solver(part.shape, part.h, layout.C, part.supports, active=A | layout.solid, backend="auto",
                 log=lambda s_: None)
    g = sol.grid
    wg = layout.mat.density * 1e-6 * part.h ** 3 * 9.81
    F = np.zeros((part.ndof, len(real)))
    for q, c in enumerate(real):
        F[:, q] = c.F
        if c.fe is not None:
            F[:, q] += g.to_flat(g.scatter((A.astype(float)[:, None] * c.fe[None, :])[None]))[:, 0]
    U = sol.solve(E, F, [c.support for c in real], keys=[c.name for c in real])
    edof = g.edof()
    enodes = edof[:, ::3] // 3
    hot = np.zeros(part.nn, bool)
    for c in real:
        if c.nodes is not None and len(c.nodes):
            hot[c.nodes] = True
    for fx in part.supports.values():
        hot[fx.any(1)] = True
    keep = A & ~ins & ~hot[enodes].any(1)                                  # inserts carry load, are not PETG
    CB = layout.C @ sol.B0
    ax = layout.build_axis
    lay = [k for k, (i, j) in {3: (0, 1), 4: (1, 2), 5: (2, 0)}.items() if ax in (i, j)]
    M = np.array([[1, -.5, -.5, 0, 0, 0], [-.5, 1, -.5, 0, 0, 0], [-.5, -.5, 1, 0, 0, 0],
                  [0, 0, 0, 3, 0, 0], [0, 0, 0, 0, 3, 0], [0, 0, 0, 0, 0, 3.]])
    rows = []
    cen = part.cell_centres()
    idx = np.where(keep)[0]
    for q, c in enumerate(real):
        if c.cls is None:
            continue
        Axy, Az, As = allowables(layout.mat, c.cls)
        ue = U[:, q][edof[idx]]
        sgm = ue @ CB.T
        vm = np.sqrt(np.maximum(np.einsum("ei,ij,ej->e", sgm, M, sgm), 0.0))
        sn = np.maximum(sgm[:, ax], 0.0)
        tau2 = sum(sgm[:, k] ** 2 for k in lay)
        f1 = vm / Axy
        f2 = np.sqrt((sn / Az) ** 2 + tau2 / As ** 2)
        j1, j2 = int(np.argmax(f1)), int(np.argmax(f2))
        rows.append(dict(case=c.name, cls=c.cls, inlayer=round(float(f1[j1]), 3), across=round(float(f2[j2]), 3),
                         at=cen[idx[j1 if f1[j1] >= f2[j2] else j2]].round(1).tolist()))
    worst = max(rows, key=lambda r: max(r["inlayer"], r["across"])) if rows else None
    log(f"strength: worst index {max(worst['inlayer'], worst['across']):.2f} in '{worst['case']}' at {worst['at']}"
        if worst else "strength: no classed load cases")
    return dict(rows=rows, worst=None if worst is None else max(worst["inlayer"], worst["across"]),
                worst_case=None if worst is None else worst["case"], worst_at=None if worst is None else worst["at"])


# ------------------------------------------------------------------------------------------------------ outputs
def render(mesh, path, title="", info=""):
    """one PNG, four views, overhangs > 45 deg red, bed (z min) blue: the print-preview look of render_preview.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    m = mesh
    if len(m.faces) > 250_000:
        try:
            m = m.simplify_quadric_decimation(face_count=250_000)
        except Exception:
            pass
    n = m.face_normals
    ns = m.vertex_normals[m.faces].mean(1)                                # smooth shading: no marching-cubes moire
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-12)
    lo, hi = m.bounds
    fig = plt.figure(figsize=(16, 4.6), dpi=130)
    views = [(25, -60, "iso"), (25, 120, "iso back"), (90, -90, "top"), (-60, -60, "underside")]
    for i, (el, az, nm) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        a, e = np.radians(az), np.radians(el)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        shade = 0.35 + 0.65 * np.clip(ns @ (cam / np.linalg.norm(cam)), 0, 1)
        base = np.tile([0.80, 0.80, 0.78], (len(n), 1))
        down = n[:, 2] < -np.cos(np.radians(45))
        bed = down & (m.triangles_center[:, 2] < lo[2] + 0.2)
        base[down & ~bed] = [0.90, 0.35, 0.28]
        base[bed] = [0.30, 0.48, 0.90]
        col = np.clip(base * shade[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(m.triangles, facecolors=col, edgecolors=col, linewidths=0.1))
        c, r = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.view_init(elev=el, azim=az)
        ax.set_axis_off()
        ax.set_title(nm, fontsize=9)
    fig.suptitle(f"{title}\n{info}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_view_mesh(mesh, path, faces=120_000):
    """a decimated copy for the web viewer (binary STL, < ~6 MB)."""
    m = mesh
    if len(m.faces) > faces:
        try:
            m = m.simplify_quadric_decimation(face_count=faces)
        except Exception:
            pass
    m.export(path)
    return len(m.faces)


def build_and_check(layout, out_dir, name, clip_margin=0.5, pitch=None, log=print, extra=None):
    """surface -> exact part -> every check -> STL, preview PNG, viewer STL, result.json. Returns the result dict."""
    os.makedirs(out_dir, exist_ok=True)
    part = layout.part
    t0 = time.time()
    gm = surface(layout, pitch=pitch, clip_margin=clip_margin, log=log)
    P, bod = assemble(layout, gm, log=log)
    PT = tm(P)
    drawn_fin = tm(part.drawn_m - part.cuts_m)
    ok_solid, solid = solid_check(PT, [drawn_fin])
    log(f"solid check: {'PASS' if ok_solid else 'FAIL'} {json.dumps(solid)[:600]}")
    fc = function_check(P, part, layout)
    log(f"function check: {fc}")
    st = strength(layout, PT, log=log)
    mass = P.volume() / 1000.0 * layout.mat.density
    stl = os.path.join(out_dir, f"{name}.stl")
    PT.export(stl)
    view = os.path.join(out_dir, f"{name}_view.stl")
    nview = export_view_mesh(PT, view)
    info = (f"{mass:.1f} g | solid check {'PASS' if ok_solid else 'FAIL'} | worst index "
            f"{st['worst'] if st['worst'] is None else round(st['worst'], 2)} | min member {layout.s.min_member_mm} mm")
    png = os.path.join(out_dir, "preview.png")
    try:
        render(PT, png, title=f"{name} (nox/topo)", info=info)
    except Exception as e:                                                # a preview must never sink a run
        log(f"render failed: {e}")
        png = None
    res = dict(name=name, engine="nox/topo", solid_ok=bool(ok_solid), solid=solid, function_check=fc,
               strength=st, mass_g=round(mass, 2), watertight=bool(PT.is_watertight), bodies=bod,
               volume_mm3=round(P.volume(), 1), drawn_g=round((part.drawn_m - part.cuts_m).volume() / 1000.0 *
                                                               layout.mat.density, 2),
               stl=os.path.basename(stl), view_stl=os.path.basename(view), view_faces=nview,
               preview=None if png is None else os.path.basename(png), layout=layout.meta(),
               grid=part.summary(), build_s=round(time.time() - t0, 1))
    if extra:
        res.update(extra)
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=_json_default)
    log("RESULT " + json.dumps(dict(name=name, solid_ok=res["solid_ok"], mass_g=res["mass_g"], worst=st["worst"],
                                    function_ok=fc["ok"], bodies=bod["bodies"]), default=_json_default))
    return res, PT


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
