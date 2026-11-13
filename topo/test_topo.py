"""Self-test of nox/topo. Run from nox/:  python3 -m topo.test_topo [--quick]
Each test prints PASS / FAIL with its numbers; exit code 1 if any fails. ~16 min on a 4 vCPU box, ~10 of them the slender-arm run
(--quick skips the end-to-end runs: ~1.5 min)."""
import sys
import time
import json
import traceback
import numpy as np
import scipy.sparse as sp

from . import fem
from .material import MATERIALS, stiffness
from .mma import MMA
from .layout import calibration, Layout, Settings

RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)


def _C():
    return stiffness(MATERIALS["PETG"], 2)


def _assembled(shape, h, E, C):
    g = fem.Grid(shape)
    KE, _ = fem.hex_ke(h, C)
    edof = g.edof()
    iK = np.repeat(edof, 24, axis=1).ravel()
    jK = np.tile(edof, (1, 24)).ravel()
    sK = (E[:, None] * KE.ravel()[None, :]).ravel()
    return sp.coo_matrix((sK, (iK, jK)), shape=(g.ndof, g.ndof)).tocsr(), g, KE


def t_ke():
    KE, B0 = fem.hex_ke(1.0, _C())
    w = np.linalg.eigvalsh(KE)
    nz = int((np.abs(w) < 1e-8 * w.max()).sum())
    check("hex KE symmetric, PSD, 6 rigid modes", np.allclose(KE, KE.T) and w.min() > -1e-8 * w.max() and nz == 6,
          f"zero modes {nz}, min eig {w.min():.2e}")


def t_matvec():
    rng = np.random.default_rng(0)
    shape = (6, 4, 5)
    E = rng.uniform(0.1, 1.0, np.prod(shape))
    K, g, KE = _assembled(shape, 1.3, E, _C())
    u = rng.normal(size=(g.ndof, 2))
    U = g.to_field(u)
    r = (g.gather(U).reshape(-1, 24) @ KE).reshape(2, g.nel, 24) * E[None, :, None]
    Y = g.to_flat(g.scatter(r))
    err = np.abs(Y - K @ u).max() / np.abs(K @ u).max()
    check("structured gather / scatter == assembled K", err < 1e-12, f"rel err {err:.1e}")


def t_transfer():
    rng = np.random.default_rng(1)
    xc = rng.normal(size=(1, 4, 3, 5, 3))
    yf = rng.normal(size=(1, 7, 5, 9, 3))
    a = (fem.prolong(xc) * yf).sum()
    b = (xc * fem.restrict(yf)).sum()
    check("restrict = prolong^T", abs(a - b) < 1e-10 * abs(a), f"{a:.6f} vs {b:.6f}")


def t_galerkin():
    rng = np.random.default_rng(2)
    shape = (4, 4, 2)
    E = rng.uniform(0.01, 1.0, np.prod(shape))
    K, g, KE = _assembled(shape, 1.0, E, _C())
    gc = fem.Grid((2, 2, 1))
    P = np.zeros((g.ndof, gc.ndof))
    for j in range(gc.ndof):
        e = np.zeros(gc.ndof)
        e[j] = 1.0
        P[:, j] = fem.prolong(gc.to_field(e)).reshape(-1)
    Kc_true = P.T @ K.toarray() @ P
    P8 = fem.child_prolongation()
    G8 = np.einsum("cai,ab,cbj->cij", P8, KE, P8).reshape(8, 576)
    Ke1 = (fem._octants(E, shape) @ G8).reshape(-1, 24, 24)
    ed = gc.edof()
    Kc = np.zeros((gc.ndof, gc.ndof))
    for e in range(gc.nel):
        Kc[np.ix_(ed[e], ed[e])] += Ke1[e]
    err = np.abs(Kc - Kc_true).max() / np.abs(Kc_true).max()
    check("element-by-element Galerkin == P^T K P", err < 1e-10, f"rel err {err:.1e}")


def _cantilever(shape=(16, 8, 8), void_frac=0.3, seed=3):
    rng = np.random.default_rng(seed)
    nel = int(np.prod(shape))
    act = rng.uniform(size=nel) > void_frac
    E = np.where(act, rng.uniform(0.05, 1.0, nel), 0.0)
    g = fem.Grid(shape)
    fixed = np.zeros((g.nn, 3), bool)
    nodes = np.arange(g.nn).reshape(g.nshape)
    fixed[nodes[0].ravel()] = True
    F = np.zeros((g.ndof, 2))
    tip = nodes[-1].ravel()
    F[3 * tip + 2, 0] = -1.0 / len(tip)
    F[3 * tip + 1, 1] = 1.0 / len(tip)
    return g, E, act, {"default": fixed}, F


def t_mg_vs_direct():
    g, E, act, fixed, F = _cantilever()
    try:
        d = fem.Solver(g.shape, 1.0, _C(), fixed, active=act, backend="direct", log=lambda s: None)
        Ud = d.solve(E, F, ["default", "default"])
        m = fem.Solver(g.shape, 1.0, _C(), fixed, active=act, backend="mg", log=lambda s: None, mg_tol=1e-9,
                       coarse_dofs=500)
        Um = m.solve(np.where(act, E, 0.0), F, ["default", "default"], keys=["a", "b"])
        cd, cm = (F * Ud).sum(0), (F * Um).sum(0)
        err = np.abs(cm - cd).max() / np.abs(cd).max()
        its = m.stats[-1][1]
        check("multigrid CG == Pardiso (compliance, 2 rhs)", err < 1e-4, f"rel err {err:.1e}, {its} CG its, "
              f"{m.levels} levels")
        t0 = time.time()
        Um2 = m.solve(np.where(act, E * 1.01, 0.0), F, ["default", "default"], keys=["a", "b"])
        check("warm start cuts CG iterations", m.stats[-1][1] < its, f"{m.stats[-1][1]} vs {its} its")
    except Exception as e:
        traceback.print_exc()
        check("multigrid CG == Pardiso", False, str(e))


def t_mma():
    n = 50
    rng = np.random.default_rng(4)
    a = rng.uniform(0.2, 0.9, n)
    opt = MMA(n, 1, move=0.2)
    x = np.full(n, 0.5)
    for _ in range(60):
        f0 = ((x - a) ** 2).sum()
        df0 = 2 * (x - a)
        g = x.mean() / 0.4 - 1.0
        dg = np.full(n, 1.0 / (n * 0.4))
        x = opt.step(x, f0, df0, np.array([g]), dg[None])
    # KKT solution: x = clip(a - lam, 0, 1) with mean 0.4
    lo, hi = -1.0, 1.0
    for _ in range(60):
        lam = 0.5 * (lo + hi)
        if np.clip(a - lam, 0, 1).mean() > 0.4:
            lo = lam
        else:
            hi = lam
    xs = np.clip(a - lam, 0, 1)
    err = np.abs(x - xs).max()
    check("MMA solves a constrained quadratic", err < 2e-3, f"max err {err:.1e}, mean x {x.mean():.4f}")


def t_calibration():
    """independent check of layout.calibration: the exact 3D cone kernel voxelised at R = 40 cells, a plate and a rod
    of the calibrated input sizes filtered by brute force, blueprint widths read off at eta_n. (The 1D hat filter gives
    plate = 1.0 R; the 3D cone integrated over a plane has a cubic profile and gives 0.88 R, so 1.0 is not the target.)"""
    from scipy.signal import fftconvolve
    c = calibration()
    R = 40
    q = np.arange(-3 * R, 3 * R + 1)
    # plate: 1D profile of the cone = its integral over the plane, done numerically on a 3D voxel grid
    X3, Y3, Z3 = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), np.arange(-R, R + 1),
                             indexing="ij")
    w3 = np.maximum(1.0 - np.sqrt(X3 ** 2 + Y3 ** 2 + Z3 ** 2) / R, 0.0)
    prof1 = w3.sum((1, 2))
    prof1 /= prof1.sum()
    prof2 = w3.sum(2)
    prof2 /= prof2.sum()
    edge = lambda f: (np.argmax(f < 0.5) - 1 + (f[np.argmax(f < 0.5) - 1] - 0.5) /
                      (f[np.argmax(f < 0.5) - 1] - f[np.argmax(f < 0.5)]))      # 0.5 crossing, linear interpolation
    slab = np.clip(c["plate_input"] * R / 2 + 0.5 - np.abs(q), 0.0, 1.0)     # area-weighted edge cells
    f1 = np.convolve(slab, prof1, mode="same")
    plate_bp = 2 * edge(f1[3 * R:]) / R
    X, Y = np.meshgrid(q, q, indexing="ij")
    disc = np.clip(c["rod_input"] * R / 2 + 0.5 - np.sqrt(X ** 2 + Y ** 2), 0.0, 1.0)
    f2 = fftconvolve(disc, prof2, mode="same")
    rod_bp = 2 * edge(f2[3 * R:, 3 * R]) / R
    ok = (abs(f1[3 * R] - 0.75) < 0.02 and abs(f2[3 * R, 3 * R] - 0.75) < 0.02 and abs(plate_bp - c["plate"]) < 0.03
          and abs(rod_bp - c["rod"]) < 0.03 and 0 < c["erode"] < 0.5)
    check("length-scale calibration vs brute-force 3D cone filter", ok,
          f"rod {c['rod']:.3f} R (brute {rod_bp:.3f}), plate {c['plate']:.3f} R (brute {plate_bp:.3f}), centre "
          f"{f1[3 * R]:.3f} / {f2[3 * R, 3 * R]:.3f} (eta_e 0.75), erosion {c['erode']:.3f} R")


def t_voxelise():
    """a keep-out that reaches far under the grid (the stand setups' ring r 180 / BIG half spaces) must voxelise as
    inside: the ray origin sits below the mesh, not below the grid (bracket_jA topo-v1 read 43k ring cells as design)."""
    import manifold3d as mf
    from .part import tm, voxelise
    ax = np.arange(4) + 0.5
    big = tm(mf.Manifold.cube([2000.0, 2000.0, 2000.0]).translate([-1000.0, -1000.0, -1000.0]))
    ring = tm(mf.Manifold.cylinder(100.0, 180.0, 180.0, 180).rotate([90, 0, 0]).translate([0, 50, 0]))
    a = voxelise(big, ax, ax, ax)
    b = voxelise(ring, ax + 150.0, ax, ax - 60.0)           # x^2 + z^2 < 180^2 everywhere on this small grid
    check("voxelise: meshes reaching below the grid read inside", a.all() and b.all(), f"{a.mean():.2f}, {b.mean():.2f}")


def _mini_part(neck=False, h=1.0):
    """bar 24 x 10 x 10 mm: drawn plate at x 0..3 (support), drawn pad at x 20..24 loaded down; design between."""
    import manifold3d as mf
    from .part import Part
    box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    base = box(0, 3, 0, 10, 0, 10)
    pad = box(20, 24, 2, 8, 2, 8)
    p = Part("mini", h, base + pad, box(100, 101, 100, 101, 100, 101), box(0, 24, 0, 10, 0, 10), log=lambda s: None)
    p.support("default", box(-1, 1.5 * h, -1, 11, -1, 11))
    p.load("down", pad, (0, 0, -10.0))
    p.load("side", pad, (0, 5.0, 0))
    return p


def t_gradients():
    try:
        p = _mini_part()
        rng = np.random.default_rng(5)
        beta = 4.0
        info = []
        worst = 0.0
        for mode in ("masked", "anchored"):
            s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, erosion=mode, finish="drawn")
            L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
            x = rng.uniform(0.2, 0.8, L.nd)

            def J(xx):
                f = L.fields(xx, beta)
                comp, _ = L._compliance(f["rho_e"], need_grad=False)
                return L._objective(comp)[0]
            f = L.fields(x, beta)
            comp, (U, E) = L._compliance(f["rho_e"])
            Jv, dJdc, r = L._objective(comp)
            g = L._grad_rho_e(f["rho_e"], U, dJdc)
            dx = L.grad_eroded(f, g)
            # cells next to the drawn bodies (where the masked denominator differs from 1) and random ones
            near = np.where(L.den_e[L.design] < 0.95)[0]
            idx = np.r_[rng.choice(near, 3, replace=False), rng.choice(L.nd, 3, replace=False)]
            errs = []
            for i in idx:
                e = 1e-4        # 1e-5 is below the roundoff floor of J (~1e3, Emin 1e-6); truncation at 1e-4 is ~1e-8
                xp, xm = x.copy(), x.copy()
                xp[i] += e
                xm[i] -= e
                fd = (J(xp) - J(xm)) / (2 * e)
                errs.append(abs(fd - dx[i]) / max(abs(fd), 1e-3 * np.abs(dx).max()))
            worst = max(worst, max(errs))
            info.append(f"{mode} {max(errs):.1e}")
        check("objective gradient vs finite differences", worst < 1e-3, "max rel err " + ", ".join(info))
        geo = L._geometric(f, 32.0)
        errs = []
        for q in range(2):
            val, dg = geo[q]
            gx = L._back(dg)
            for i in idx[:4]:
                e = 1e-6
                xp, xm = x.copy(), x.copy()
                xp[i] += e
                xm[i] -= e
                vp = L._geometric(L.fields(xp, beta), 32.0)[q][0]
                vm = L._geometric(L.fields(xm, beta), 32.0)[q][0]
                fd = (vp - vm) / (2 * e)
                errs.append(abs(fd - gx[i]) / max(abs(fd), 1e-3 * np.abs(gx).max(), 1e-30))
        check("geometric-constraint gradients vs finite differences", max(errs) < 1e-3, f"max rel err {max(errs):.1e}")
    except Exception as e:
        traceback.print_exc()
        check("gradients", False, str(e))


def _stem_part(half):
    """mini part whose pad (x 21..24) hangs on a drawn stem x 18..21 of 2 * half mm square: a neck inside the drawn
    body, which frozen pads cannot see."""
    import manifold3d as mf
    from .part import Part
    box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    drawn = box(0, 3, 0, 10, 0, 10) + box(21, 24, 2, 8, 2, 8) + box(18, 21, 5 - half, 5 + half, 5 - half, 5 + half)
    p = Part("stem", 1.0, drawn, box(100, 101, 100, 101, 100, 101), box(0, 24, 0, 10, 0, 10), log=lambda s: None)
    p.support("default", box(-1, 1.5, -1, 11, -1, 11))
    p.load("down", box(21, 24, 2, 8, 2, 8), (0, 0, -10.0))
    return p


def t_neck():
    """the reason for anchored erosion. (a) A generated 2 mm neck at a pad must open a gap in the eroded design
    (compliance ratio >> wide root). The constant-normalised filter already erodes such a neck in the design itself, so
    frozen pads punish it too (x_tilde is the same in both modes). (b) A pad hung on a 2 mm drawn stem: only anchored
    erosion erodes the stem's skin and sees the gap; frozen pads (old behaviour) must not."""
    try:
        res = {}
        for anchored in (True, False):
            s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False,
                         erosion="anchored" if anchored else "frozen", finish="drawn")
            p = _mini_part()
            L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
            cen = p.cell_centres()[L.design]
            # member from the base to x = 18, then a neck (2 x 2) or a wide root (6 x 6) to the pad face at x 20
            core = (cen[:, 0] < 18) & (np.abs(cen[:, 1] - 5) < 3) & (np.abs(cen[:, 2] - 5) < 3)
            for kind, half in (("neck", 1.0), ("wide", 3.0)):
                lnk = (cen[:, 0] >= 18) & (np.abs(cen[:, 1] - 5) < half) & (np.abs(cen[:, 2] - 5) < half)
                x = np.where(core | lnk, 1.0, 0.0)
                f = L.fields(x, 32.0)
                comp, _ = L._compliance(f["rho_e"], need_grad=False)
                res[(anchored, kind)] = float(comp[0] / L.c_ref[0])
            for kind, half in (("stem", 1.0), ("fat stem", 3.0)):
                p = _stem_part(half)
                L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
                cen = p.cell_centres()[L.design]
                x = np.where((cen[:, 0] < 18) & (np.abs(cen[:, 1] - 5) < 3) & (np.abs(cen[:, 2] - 5) < 3), 1.0, 0.0)
                comp, _ = L._compliance(L.fields(x, 32.0)["rho_e"], need_grad=False)
                res[(anchored, kind)] = float(comp[0] / L.c_ref[0])
        neck = res[(True, "neck")] / res[(True, "wide")]
        stem_new = res[(True, "stem")] / res[(True, "fat stem")]
        stem_old = res[(False, "stem")] / res[(False, "fat stem")]
        check("anchored erosion (option) punishes a 2 mm neck at a pad", neck > 50 and stem_new > 50 and stem_new > 10 * stem_old,
              f"generated neck / wide: {neck:.3g}; drawn stem / fat stem: anchored {stem_new:.3g}, frozen pads "
              f"{stem_old:.3g}")
    except Exception as e:
        traceback.print_exc()
        check("anchored erosion neck test", False, str(e))


def _cut_part():
    """mini part with a slot cut (x 17..20, z 0..4) under the pad's root: the pad (x 20..24) can be reached over the
    cut face only, as at a bolt pad beside its access cut."""
    import manifold3d as mf
    from .part import Part
    box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    pad = box(20, 24, 2, 8, 2, 8)
    p = Part("cutmini", 1.0, box(0, 3, 0, 10, 0, 10) + pad, box(17, 20, -1, 11, -1, 4), box(0, 24, 0, 10, 0, 10),
             log=lambda s: None)
    p.support("default", box(-1, 1.5, -1, 11, -1, 11))
    p.load("down", pad, (0, 0, -10.0))
    p.load("side", pad, (0, 5.0, 0))
    return p


def _plate_part():
    """a thin held plate (2 mm, x 0..2, supported) and two pads (x 20..24): gate ties are active."""
    import manifold3d as mf
    from .part import Part
    box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    p1, p2 = box(20, 24, 1, 7, 2, 8), box(20, 24, 13, 19, 2, 8)
    p = Part("platemini", 1.0, box(0, 2, 0, 20, 0, 10) + p1 + p2, box(100, 101, 100, 101, 100, 101),
             box(0, 24, 0, 20, 0, 10), log=lambda s: None)
    p.support("default", box(-1, 1.0, -1, 21, -1, 11))
    p.load("down 1", p1, (0, 0, -10.0))
    p.load("down 2", p2, (0, 0, -10.0))
    return p


def t_gate_ties():
    """gate ties: the 2 mm held plate counts by its cover, the second pad is tied to the first. Gradient through the
    cover filter vs finite differences, and a bridge between the pads must beat two separate legs to the plate."""
    try:
        p = _plate_part()
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        ok_setup = L.cover.any() and L.gate_anchor is not None and any(c.support == "_gate" for c in L.cases)
        rng = np.random.default_rng(7)
        x = rng.uniform(0.2, 0.8, L.nd)
        beta = 4.0

        def J(xx):
            f = L.fields(xx, beta)
            comp, _ = L._compliance(f["rho_e"], need_grad=False, rho_g=f["rho_g"])
            return L._objective(comp)[0]
        f = L.fields(x, beta)
        comp, (U, E) = L._compliance(f["rho_e"], rho_g=f["rho_g"])
        Jv, dJdc, r = L._objective(comp)
        dx = L.grad_eroded(f, L._grad_rho_e(f["rho_e"], U, dJdc, rho_g=f["rho_g"]))
        ok_setup = ok_setup and bool((f["rho_e"][L.cover] == 1.0).all())   # the real cases keep the plate whole
        cen = p.cell_centres()[L.design]
        near = np.where(cen[:, 0] < 4.5)[0]                                 # cells that feed the plate's cover
        idx = np.r_[rng.choice(near, 4, replace=False), rng.choice(L.nd, 2, replace=False)]
        errs = []
        for i in idx:
            e = 1e-3        # FD at 1e-3 matches to ~1e-5; 1e-4 hits the roundoff floor on low-gradient cells
            xp, xm = x.copy(), x.copy()
            xp[i] += e
            xm[i] -= e
            fd = (J(xp) - J(xm)) / (2 * e)
            errs.append(abs(fd - dx[i]) / max(abs(fd), 1e-3 * np.abs(dx).max()))
        # two legs (pad -> plate) vs the same legs plus a bridge between the pads: the gate ties must see the bridge
        legs = (((np.abs(cen[:, 1] - 4) < 3) | (np.abs(cen[:, 1] - 16) < 3)) & (np.abs(cen[:, 2] - 5) < 3))
        bridge = legs | ((cen[:, 0] > 15) & (np.abs(cen[:, 2] - 5) < 3))
        gate = np.array([c.support == "_gate" for c in L.cases])
        cg = {}
        for kind, m in (("legs", legs), ("bridge", bridge)):
            ff = L.fields(np.where(m, 1.0, 0.0), 32.0)
            comp, _ = L._compliance(ff["rho_e"], need_grad=False, rho_g=ff["rho_g"])
            cg[kind] = (comp / L.c_ref)[gate].max()
        check("gate ties: cover gradient exact, a bridge between pads beats legs to the thin plate",
              ok_setup and max(errs) < 1e-3 and cg["legs"] > 50 * cg["bridge"],
              f"max rel err {max(errs):.1e}; worst gate tie ratio legs {cg['legs']:.3g}, bridge {cg['bridge']:.3g}")
    except Exception as e:
        traceback.print_exc()
        check("gate ties", False, str(e))


def t_masked_neck():
    """the reason for the drawn-masked eroded filter: a pad joined by a 2 mm generated neck that runs along a cut must
    open a gap in the eroded design (eroded compliance >> a full-width root), while the full-width root keeps its own
    density up to the drawn face. Frozen pads with the nominal filter are printed for comparison."""
    try:
        res = {}
        for mode in ("masked", "frozen"):
            s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, erosion=mode, finish="drawn")
            p = _cut_part()
            L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
            cen = p.cell_centres()[L.design]
            core = (cen[:, 0] < 18) & (np.abs(cen[:, 1] - 5) < 3) & (np.abs(cen[:, 2] - 5) < 3)
            for kind, (y0, y1, z0, z1) in (("neck", (4, 6, 4, 6)), ("wide", (2, 8, 4, 8))):
                lnk = (cen[:, 0] >= 18) & (cen[:, 1] > y0) & (cen[:, 1] < y1) & (cen[:, 2] > z0) & (cen[:, 2] < z1)
                x = np.where(core | lnk, 1.0, 0.0)
                comp, _ = L._compliance(L.fields(x, 32.0)["rho_e"], need_grad=False)
                res[(mode, kind)] = comp[:2]
        rm = (res[("masked", "neck")] / res[("masked", "wide")]).min()
        rf = (res[("frozen", "neck")] / res[("frozen", "wide")]).min()
        check("drawn-masked eroded filter punishes a 2 mm neck along a cut", rm > 50,
              f"eroded compliance neck / wide (min over 2 cases): masked {rm:.3g}, frozen + nominal filter {rf:.3g}")
    except Exception as e:
        traceback.print_exc()
        check("drawn-masked neck test", False, str(e))


def t_end_to_end():
    import os
    import tempfile
    from .demos.holder import setup
    from .run import optimise
    try:
        out = tempfile.mkdtemp(prefix="topo_e2e_")
        s = Settings(min_member_mm=4.0, max_stage=12, min_stage=4, backend="auto", time_limit_s=900)
        res = optimise(setup, 2.0, 6.0, out, "holder_smoke", settings=s, coarse=False, maxit=60,
                       log=lambda s_: print(s_[:300], flush=True))
        e = res.get("early") or {}
        check("end to end: holder at h 2.0 builds one watertight body", res["bodies"]["bodies"] == 1 and
              res["watertight"], f"{res['mass_g']} g, solid check {'PASS' if res['solid_ok'] else 'FAIL'}, "
                                 f"worst {res['strength']['worst']}, early check {e.get('worst')} after beta {e.get('beta')} "
                                 f"(ratio {e['worst'] / res['strength']['worst'] if e.get('worst') and res['strength']['worst'] else None}), "
                                 f"{res['total_s']} s, files in {out}")
    except Exception as e:
        traceback.print_exc()
        check("end to end", False, str(e))


def t_gate_specks():
    """solid gate (tools/solid_check.py): a detached piece under 1 mm3 is voxel dust, not a body (28 Oct: node 7 "3 bodies" were two
    single voxels, node 21's drawn core was "0.3 % off" in 41 pieces under 0.4 mm3, the head mount's "split-off" bosses the same);
    a detached piece of 3 mm3 is still a body, and a detached drawn body is still caught by the drawn-core rule."""
    import sys, os, trimesh
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    from solid_check import solid_check
    body = trimesh.creation.box((20.0, 20.0, 20.0))
    speck = trimesh.creation.box((0.45, 0.45, 0.45)).apply_translation((14.0, 0.0, 0.0))     # 0.09 mm3, detached
    blob = trimesh.creation.box((1.5, 1.5, 1.5)).apply_translation((14.0, 0.0, 0.0))        # 3.4 mm3, detached
    pad = trimesh.creation.box((4.0, 4.0, 4.0)).apply_translation((14.0, 0.0, 0.0))         # a drawn pad left floating
    ok1, i1 = solid_check(trimesh.util.concatenate([body, speck]))
    ok2, i2 = solid_check(trimesh.util.concatenate([body, blob]))
    ok3, i3 = solid_check(trimesh.util.concatenate([body, pad]), drawn=[pad])
    ok = ok1 and i1["specks_plain"] == 1 and not ok2 and i2["bodies_plain"] == 2 and not ok3 and bool(i3["drawn_not_in_main"])
    RESULTS.append(("gate_specks", ok)); print(f"gate specks: {'ok' if ok else 'FAIL'} speck {ok1}/{i1['specks_plain']} blob {ok2}/{i2['bodies_plain']} pad {ok3}/{i3['drawn_not_in_main']}")


def _mbox(sx, sy, sz, at):
    import trimesh
    from .part import man
    return man(trimesh.creation.box((sx, sy, sz)).apply_translation(at))


def t_export_cleanup():
    """export cleanup (topo/build.py export_cleanup): a 1.0 mm fin with ~6 mm3 of core and a floating 0.5 mm speck come off a 20 mm cube,
    the cube and a drawn 4 mm pad on it stay as they are and the gate then passes (29 Oct: the rear arm and the front arm root, both
    strong, failed the gate on one 1.1 / 1.4 mm sliver each); a 1.4 mm wall with ~100 mm3 of core is structure, not detail: it stays
    for the gate to judge."""
    try:
        import types
        from .part import tm
        from .build import export_cleanup, solid_check
        cube = _mbox(20, 20, 20, (0, 0, 0))
        pad = _mbox(4, 8, 8, (12, 0, 0))                                         # drawn pad on the +x face
        fin = _mbox(1.0, 4, 4, (0, 12, 0))                                       # 1.0 mm fin standing 4 mm off the +y face
        speck = _mbox(0.5, 0.5, 0.5, (0, 0, 13))                                 # floating voxel dust, 0.125 mm3
        part = types.SimpleNamespace(drawn_m=pad)
        P, rep = export_cleanup(cube + pad + fin + speck, part, log=lambda s_: None)
        fin_left = (P ^ fin).volume(); pad_lost = pad.volume() - (P ^ pad).volume(); cube_lost = (cube - P).volume()
        ok, info = solid_check(tm(P), [tm(pad)])
        wall = _mbox(1.4, 12, 12, (0, -12.7, 0))
        P3, rep3 = export_cleanup(cube + pad + wall, part, log=lambda s_: None)
        good = (len(rep["slivers"]) == 1 and rep["slivers"][0]["core_mm3"] < 10 and len(rep["specks_mm3"]) == 1 and fin_left < 1.5                 # a stub of one voxel (0.25 mm) may stay: the removal never reaches into the opened body
                and pad_lost < 0.01 and cube_lost < 1.0 and ok and not rep3["slivers"] and len(rep3["kept_walls"]) == 1
                and rep3["removed_mm3"] == 0.0)
        check("export cleanup: fin and speck removed, cube, pad and a 100 mm3 wall untouched, gate passes after", good,
              f"slivers {rep['slivers']}, specks {rep['specks_mm3']}, fin left {fin_left:.2f}, pad lost {pad_lost:.3f}, cube lost "
              f"{cube_lost:.2f} mm3, gate {'PASS' if ok else 'FAIL'} {info.get('thin_walls')}; wall kept {rep3['kept_walls']}")
    except Exception as e:
        traceback.print_exc()
        check("export cleanup", False, str(e))


def _strut_part(wall, post=False, h=1.0):
    """the Scout node pattern (parts/scout_node_lib.py), self-contained: two socket collars (a cylinder of 1.6 + wall round a 1.6 bore,
    two 0.5 resin grooves, 10 deep) whose strut rooms (2.5 round the collar) meet at 60 deg in a hub ball; support on the first socket,
    a pull-out load on the second. post: a drawn O5.6 post round a O3.2 bore (1.2 wall, the camera insert post) standing on the hub,
    declared thin_ok."""
    import trimesh
    import manifold3d as mf
    from .part import Part, man
    cyl = lambda c, u, s0, s1, r: man(trimesh.creation.cylinder(r, s1 - s0, sections=40, segment=np.vstack([c + u * s0, c + u * s1])))
    C = np.zeros(3); U = [np.array([1.0, 0.0, 0.0]), np.array([0.5, np.sqrt(0.75), 0.0])]
    s0, depth, bore = 7.0, 10.0, 1.6
    collars, cuts, struts, linings = [], [], [], []
    for u in U:
        collars.append(cyl(C, u, s0 - 2.0, s0 + depth, bore + wall))
        cuts.append(cyl(C, u, s0, s0 + depth + 1.0, bore))
        for f in (1.0 / 3.0, 2.0 / 3.0):
            cuts.append(cyl(C, u, s0 + f * depth - 0.75, s0 + f * depth + 0.75, bore + 0.5))
        cuts.append(cyl(C, u, s0 + depth, s0 + depth + 40.0, bore + 0.6))
        struts.append(cyl(C, u, 0.0, s0 + depth, bore + wall + 2.5))
        linings.append(cyl(C, u, s0 - 2.5, s0 + depth, bore + 1.0))
    struts.append(mf.Manifold.sphere(11.5, 48))
    drawn = mf.Manifold.batch_boolean(collars, mf.OpType.Add)
    thin_ok = []
    if post:
        z = np.array([0.0, 0.0, 1.0])
        tube = cyl(C, z, 6.0, 16.0, 2.8)
        cuts.append(cyl(C, z, 10.3, 17.0, 1.6))
        drawn = drawn + tube; thin_ok = [tube]
    p = Part("strut60", h, drawn, mf.Manifold.batch_boolean(cuts, mf.OpType.Add), mf.Manifold.batch_boolean(struts, mf.OpType.Add),
             build="+z", material="PETG", log=lambda s: None, thin_ok=thin_ok)
    p.support("default", linings[0])
    p.load_multi("pull-out 60 N", [(linings[1], 60.0 * U[1])], cls="crash")
    return p, U


def _strut_build(p, U):
    """the organic build of _strut_part on a synthetic layout field: a O14 hub ball, a O8 rod along each strut axis from the hub to its
    collar's floor, so both collars stand exposed beyond the generated material, and the drawn cells the organic finish opens to the
    design (the post's solid base). Returns (gate ok, gate info, cleanup report, function check, the part)."""
    from . import build as B
    from .layout import Layout, Settings
    from .part import tm
    L = Layout(p, gen_cm3=1.5, settings=Settings(), log=lambda s_: None)
    cen = p.cell_centres()
    d = np.full(p.nel, 1e9)
    for u in U:
        s = np.clip(cen @ u, -2.0, 6.0)
        d = np.minimum(d, np.linalg.norm(cen - s[:, None] * u[None, :], axis=1))
    d = np.minimum(d, np.linalg.norm(cen, axis=1) - 3.0)
    d[p.solid & ~L.solid] = 0.0
    xt = np.clip(0.5 + (4.0 - d) / 1.5, 0.0, 1.0)
    rho = np.maximum((xt > 0.5).astype(float), L.solid.astype(float))
    L.result = dict(x=None, xt=xt, rho_n=rho, rho_e=rho, beta=32.0)
    gm = B.surface(L, log=lambda s_: None)
    P, _ = B.assemble(L, gm, log=lambda s_: None)
    P2, rep = B.export_cleanup(P, p, log=lambda s_: None)
    PT, _ = B.export_mesh(P2, log=lambda s_: None)
    ok, info = B.solid_check(PT, [tm(p.drawn_m - p.cuts_m)], thin_ok=[tm(m - p.cuts_m) for m in p.thin_ok])
    fc = B.function_check(P2, p, L)
    return ok, info, rep, fc, P2


def t_organic_drawn_walls():
    """the organic finish and the export cleanup on drawn walls (7 Nov: the five Scout strut nodes passed every strength case and
    failed the gate on 2..10 'flakes' of 1.0..1.4 mm at their collars; the cleanup listed them as slivers removed and took off ~1 mm3,
    the gate then failed the same spots). The flakes are the collars' own walls at the resin grooves (2.0 wall - 0.5 groove = 1.5, under
    the 2 mm floor), drawn material the cleanup may not cut. (a) A 2.5-wall node (2.0 at the grooves) with a declared 1.2 mm post: the
    audit passes, the build passes the gate with no thin wall, the post is kept whole and reported as thin_allowed, the cleanup touches
    no drawn material. (b) The same node with a 2.0 wall: the audit refuses it (drawn_wall at the grooves) and, built anyway, the gate
    fails the groove rings, which the cleanup reports as kept drawn walls, not as removed, and leaves whole."""
    try:
        from .audit import audit
        from .part import tm
        p, U = _strut_part(2.5, post=True)
        ra = audit(p, 6.25, log=lambda s_: None, hard=False)
        ok_a, ia, ca, fa, Pa = _strut_build(p, U)
        post_fin = p.thin_ok[0] - p.cuts_m
        post_kept = (Pa ^ post_fin).volume() / post_fin.volume()
        drawn_lost = (p.drawn_m - p.cuts_m - Pa).volume()
        allowed = [t for t in ia["thin_allowed"] if t.get("kind") != "neck"]
        good_a = (not ra["fails"] and not ra["drawn_walls"] and ok_a and not ia["thin_walls"] and len(allowed) >= 1
                  and all(1.0 <= t["thick_mm"] <= 1.6 for t in allowed) and fa["ok"] and not ca["slivers"] and ca["removed_mm3"] < 1.0
                  and post_kept > 0.95 and drawn_lost < 25.0)
        p2, U2 = _strut_part(2.0)
        rb = audit(p2, 6.25, log=lambda s_: None, hard=False)
        ok_b, ib, cb, fb, Pb = _strut_build(p2, U2)
        kept_drawn = [w for w in cb["kept_walls"] if w.get("drawn")]
        drawn_lost_b = (p2.drawn_m - p2.cuts_m - Pb).volume()
        good_b = (any(f.startswith("drawn_wall") for f in rb["fails"]) and len(rb["drawn_walls"]) == 4
                  and all(1.0 <= w["thick_mm"] <= 1.8 for w in rb["drawn_walls"]) and not ok_b and len(ib["thin_walls"]) >= 1
                  and all(t["thick_mm"] <= 1.8 for t in ib["thin_walls"]) and not cb["slivers"] and len(kept_drawn) >= 1
                  and cb["removed_mm3"] < 1.0 and drawn_lost_b < 25.0)
        check("organic drawn walls: a 2.5-wall node with a 1.2 post passes whole; a 2.0-wall node is refused (1.5 at its 4 groove rings), built anyway "
              "the gate fails the rings and the cleanup keeps them as drawn walls", good_a and good_b,
              f"(a) audit {ra['fails']} walls {ra['drawn_walls']}; gate {ok_a} thin {ia['thin_walls']} allowed {allowed} necks {ia['necks']}; cleanup {ca['slivers']} "
              f"{ca['kept_walls']} removed {ca['removed_mm3']}; function {fa}; post kept {post_kept:.3f}, drawn lost {drawn_lost:.1f} mm3; "
              f"(b) audit {[f[:60] for f in rb['fails']]} walls {rb['drawn_walls']}; gate {ok_b} thin {ib['thin_walls']}; cleanup slivers {cb['slivers']} "
              f"kept {cb['kept_walls']} removed {cb['removed_mm3']}, drawn lost {drawn_lost_b:.1f} mm3")
    except Exception as e:
        traceback.print_exc()
        check("organic drawn walls", False, str(e))


def t_audit_drawn_wall_large():
    """drawn_walls on a drawn body too large for one grid at the gate's pitch (13 Nov, the landing funnel's frame halves): two 3.0
    walls leaning at 55 deg on a ledge, 120 x 110 x 36, must read no thin wall (coarsened to 0.4 they read 1.1 over 600 mm3), and
    the same body with a 1.4 mm fin on it must still be refused for the fin."""
    import manifold3d as mf, trimesh
    from .audit import drawn_walls
    def slab(sy, t=3.0):
        w = t / np.sin(np.radians(55.0)); g = 30.0 / np.tan(np.radians(55.0))
        pts = [(x, sy * y, z) for x in (0.0, 120.0) for y, z in ((34.0, 0.0), (34.0 + w, 0.0), (34.0 + g, 30.0), (34.0 + g + w, 30.0))]
        h = trimesh.convex.convex_hull(np.array(pts, float))
        return mf.Manifold(mf.Mesh(np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.uint32)))
    base = mf.Manifold.cube([120.0, 80.0, 4.0]).translate([0.0, -40.0, -3.5]) - mf.Manifold.cube([110.0, 56.0, 6.0]).translate([5.0, -28.0, -4.5])
    body = base + slab(1) + slab(-1)
    fin = mf.Manifold.cube([40.0, 1.4, 10.0]).translate([40.0, 30.0, 0.0])
    a = drawn_walls(body, mf.Manifold())
    b = drawn_walls(body + fin, mf.Manifold())
    good = not a and len(b) >= 1 and all(1.0 <= w["thick_mm"] <= 1.8 for w in b) and all(w["pitch"] <= 0.25 for w in b)
    print(f"  audit drawn wall, large body: plain {a}; with a 1.4 fin {[(w['thick_mm'], w['at'], w['pitch']) for w in b]}")
    return good


def t_gate_drawn_body():
    """gate, drawn bodies (tools/solid_check.py): (a) a 2.5 mm drawn plate on a 3.6 mm wall passes though a 4 mm generated block
    wraps its far end (the old measure took the body's thickness from the part round it, eroded 2.0 and asked for a 4 mm joint:
    node 5's collar clusters, node 21's camera cones, 29 Oct); (b) a 3.6 mm drawn tab notched to 2.2 mm stays one body (a feature is
    never split from itself: a collar's resin grooves); (c) a 4 mm pad on a 1.2 mm sliver still fails, the entry naming where and
    how much of its core is off."""
    try:
        from .part import tm
        from .build import solid_check
        cube = _mbox(20, 20, 20, (0, 0, 0))
        wall = _mbox(6, 3.6, 12, (13, 0, 0)); plate = _mbox(2.5, 12, 12, (17.25, 0, 0)); sleeve = _mbox(4, 14, 14, (19, 0, 0))
        ok_a, ia = solid_check(tm(cube + wall + plate + sleeve), [tm(plate)])
        tab = _mbox(14, 3.6, 10, (17, 0, 0)); notch = _mbox(1.5, 0.7, 10.2, (15.75, 1.45, 0)) + _mbox(1.5, 0.7, 10.2, (15.75, -1.45, 0))
        ok_b, ib = solid_check(tm(cube + (tab - notch)), [tm(tab - notch)])
        pad = _mbox(4, 8, 8, (14.6, 0, 0)); sliver = _mbox(2.6, 1.2, 8, (11.3, 0, 0))
        ok_c, ic = solid_check(tm(cube + sliver + pad), [tm(pad)])
        bad = ic["drawn_not_in_main"]
        good = (ok_a and not ia["drawn_not_in_main"] and ok_b and not ib["drawn_not_in_main"] and not ok_c and len(bad) >= 1
                and bad[-1]["at"][0] > 12.0 and bad[-1]["mm3"] > 1.0 and bad[-1]["share"] < 0.5)
        check("gate drawn bodies: own thickness, never split from itself, a pad on a sliver still fails with at / mm3", good,
              f"(a) {ok_a} {ia['drawn_not_in_main']} necks {ia['necks']} thin {ia['thin_walls']}; (b) {ok_b} {ib['drawn_not_in_main']} necks {ib['necks']} "
              f"thin {ib['thin_walls']}; (c) {ok_c} {bad}")
    except Exception as e:
        traceback.print_exc()
        check("gate drawn bodies", False, str(e))


def t_gate_thin_ok():
    """gate, declared-thin bodies (Part(..., thin_ok=[...]) -> solid_check thin_ok): a drawn O5.6 post round a O3.2 bore (1.2 wall, the
    camera M2 insert posts, Spark 29 Oct by coupon) fails the gate as a thin wall unless the setup declares it; declared, it is reported
    under thin_allowed and the part passes; an undeclared 1.0 mm fin beside it still fails."""
    try:
        import trimesh
        from .part import tm, man
        from .build import solid_check
        cube = _mbox(20, 20, 20, (0, 0, 0))
        post = man(trimesh.creation.cylinder(2.8, 10.0, sections=48).apply_translation((0, 0, 15)))
        bore = man(trimesh.creation.cylinder(1.6, 5.7, sections=32).apply_translation((0, 0, 17.15)))
        P = cube + post - bore
        ok0, i0 = solid_check(tm(P), [tm(post - bore)])
        ok1, i1 = solid_check(tm(P), [tm(post - bore)], thin_ok=[tm(post - bore)])
        fin = _mbox(1.0, 4, 4, (0, 12, 0))
        ok2, i2 = solid_check(tm(P + fin), [tm(post - bore)], thin_ok=[tm(post - bore)])
        # the post's root as a 4.5 mm blob on a 2.4 mm neck (node 21 run 74: a neck finding at every declared post): allowed when
        # the neck is at least as thick as the post's wall; the same blob without the declaration is still a neck
        neck = man(trimesh.creation.cylinder(1.2, 2.2, sections=32).apply_translation((0, 0, 11.0)))
        blob = man(trimesh.creation.icosphere(3, 2.25).apply_translation((0, 0, 13.5)))
        post2 = man(trimesh.creation.cylinder(2.8, 10.0, sections=48).apply_translation((0, 0, 18.5)))
        bore2 = man(trimesh.creation.cylinder(1.6, 5.7, sections=32).apply_translation((0, 0, 20.65)))
        Q = cube + neck + blob + post2 - bore2
        ok3, i3 = solid_check(tm(Q), [tm(post2 - bore2)], thin_ok=[tm(post2 - bore2)])
        ok4, i4 = solid_check(tm(Q), [tm(post2 - bore2)])
        neck_ok = (ok3 and not i3["necks"] and any(t.get("kind") == "neck" for t in i3["thin_allowed"]) and not ok4 and len(i4["necks"]) >= 1)   # the same blob at two erosion levels
        good = (not ok0 and len(i0["thin_walls"]) == 1 and ok1 and not i1["thin_walls"] and len(i1["thin_allowed"]) == 1
                and i1["thin_allowed"][0]["thick_mm"] < 2.0 and not ok2 and len(i2["thin_walls"]) == 1 and len(i2["thin_allowed"]) == 1 and neck_ok)
        check("gate thin_ok: a declared 1.2 mm post passes as thin_allowed, an undeclared fin still fails, its root on a 2.4 mm neck is allowed", good,
              f"undeclared {ok0} {i0['thin_walls']}; declared {ok1} allowed {i1['thin_allowed']} walls {i1['thin_walls']}; + fin {ok2} {i2['thin_walls']}; "
              f"neck: declared {ok3} {i3['necks']} allowed {[t for t in i3['thin_allowed'] if t.get('kind') == 'neck']}, undeclared {ok4} {i4['necks']}")
    except Exception as e:
        traceback.print_exc()
        check("gate thin_ok", False, str(e))


def t_export_fallback():
    """export (topo/build.py weld_clean / require_mesh): manifold3d refuses a mesh trimesh calls watertight when a face pair is
    mis-wound (NotManifold, an EMPTY solid), so the collapse after a closed first pass gave 0 faces and the gate died on None
    (9 Nov: overseer_head_mount after 7334 s). Now a refused or empty second pass falls back to the first-pass mesh (closed, same
    volume, the reason reported) and an empty export raises ExportError, never a TypeError."""
    try:
        import trimesh
        import manifold3d as mf
        from . import build as B
        from .part import man, tm
        b = trimesh.creation.box((10, 10, 10)); F = b.faces.copy(); F[0] = F[0][::-1]
        mis = trimesh.Trimesh(b.vertices, F, process=False)
        refused = mis.is_watertight and man(mis).num_tri() == 0                # the mechanism: watertight to trimesh, empty to manifold
        box = _mbox(20, 20, 20, (-240, 0, -80))
        c = trimesh.creation.cylinder(3.1, 30.0, sections=40)
        c.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [1, 0.3, 0])); cyl = man(c.apply_translation((-240, 0, -80)))
        again = man(tm(box - cyl)) - cyl                                     # coincident cut: slivers, so the simplify branch runs
        raw = again.to_mesh64(); v = np.asarray(raw.vert_properties)[:, :3]; f = np.asarray(raw.tri_verts)
        keep = B.man
        try:
            B.man = lambda t_: mf.Manifold()                                 # what a NotManifold mesh gives
            m, rep = B.weld_clean(v, f)
        finally:
            B.man = keep
        fb = rep.get("simplify_fallback") or {}
        fell = rep["watertight"] and len(m.faces) > 0 and not rep["simplified"] and fb.get("faces") == 0 and abs(m.volume - again.volume()) < 1.0
        try:
            B.require_mesh(trimesh.Trimesh(), dict(faces_in=0)); raised = False
        except B.ExportError:
            raised = True
        check("export: a manifold-refused simplify keeps the closed first pass; an empty export raises ExportError", refused and fell and raised,
              f"refused {refused}; fallback {fell} {fb} faces {len(m.faces)}; raised {raised}")
    except Exception as e:
        traceback.print_exc()
        check("export fallback", False, str(e))


def t_export_mesh():
    """export (topo/build.py weld_mesh / export_mesh): the exact part of a boolean with coincident surfaces (a mesh cut again with
    the holes it already has, as tools/recut.py does) holds sliver triangles thinner than float32; welded at STL precision, the
    collapsed faces go and the mesh is watertight, one body, same volume, and still watertight after a reload (29 Oct: node 5's
    recut had 467 open edges, the page's decimation tore it open). weld_mesh alone: a folded sliver (two coincident triangle
    fans), a 2-face open shell and a 0.1 mm3 speck beside a box all go; the box stays closed."""
    try:
        import trimesh, tempfile, os
        from .part import man, tm
        from .build import weld_mesh, export_mesh
        box = _mbox(20, 20, 20, (-240, 0, -80))                              # where node 5 sits: float32 resolves 1.5e-5 there
        c = trimesh.creation.cylinder(3.1, 30.0, sections=40)
        c.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [1, 0.3, 0])); cyl = man(c.apply_translation((-240, 0, -80)))
        once = tm(box - cyl)                                                  # float32 round trip, as an STL
        again = man(once) - cyl                                              # the same hole cut again: coincident surfaces
        raw = again.to_mesh64(); v = np.asarray(raw.vert_properties)[:, :3]; f = np.asarray(raw.tri_verts)
        from scipy.spatial import cKDTree
        uv = np.unique(v, axis=0); dnn = cKDTree(uv).query(uv, k=2)[0][:, 1]
        slivers = int((dnn < 1e-5).sum())
        m, rep = export_mesh(again, log=lambda s_: None)
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "x.stl"); m.export(fn); r = trimesh.load(fn, force="mesh")
            r.update_faces(r.nondegenerate_faces(height=1e-6))
            reload_ok = r.is_watertight and len(r.split(only_watertight=False)) == 1
        vol_ok = abs(m.volume - again.volume()) < 1.0
        # weld_mesh alone on a synthetic mesh
        b = trimesh.creation.box((10, 10, 10))
        fin = np.array([[5, -2, -2], [9, -2, -2], [9, 2, 2]], float); fin2 = fin + np.array([0, 0, 1e-9])
        n0 = len(b.vertices)
        V = np.vstack([b.vertices, fin, fin2, [[20, 0, 0], [21, 0, 0], [20, 1, 0], [21, 1, 0]], [[30, 0, 0], [30.4, 0, 0], [30, 0.4, 0], [30, 0, 0.4]]])
        F = np.vstack([b.faces, [[n0, n0 + 1, n0 + 2], [n0 + 3, n0 + 5, n0 + 4]],                     # the folded fin (two sides, 1e-9 apart)
                       [[n0 + 6, n0 + 7, n0 + 8], [n0 + 7, n0 + 9, n0 + 8]],                            # an open 2-face shell
                       [[n0 + 10, n0 + 11, n0 + 12], [n0 + 10, n0 + 13, n0 + 11], [n0 + 10, n0 + 12, n0 + 13], [n0 + 11, n0 + 13, n0 + 12]]])   # a 0.01 mm3 speck
        w, wr = weld_mesh(V, F)
        weld_ok = (wr["folded"] == 2 and wr["watertight"] and wr["bodies"] == 1 and abs(w.volume - 1000.0) < 1e-6
                   and wr["n_dropped_shells"] == 2 and any(not d_["closed"] for d_ in wr["dropped_shells"]) and any(d_["closed"] and d_["mm3"] < 1.0 for d_ in wr["dropped_shells"]))
        good = slivers > 0 and rep["watertight"] and rep["bodies"] == 1 and rep["boundary_edges"] == 0 and rep["nonmanifold_edges"] == 0 and vol_ok and reload_ok and weld_ok
        check("export: a coincident-cut part welds watertight at STL precision and survives a reload; folded slivers, open shells and specks go", good,
              f"raw sliver vertices {slivers}, export {rep}, volume {m.volume:.1f} vs {again.volume():.1f}, reload {reload_ok}; weld {wr}")
    except Exception as e:
        traceback.print_exc()
        check("export mesh", False, str(e))


def t_export_pinch():
    """export (topo/build.py weld_mesh / export_mesh / require_mesh): two surfaces of a part touching within the weld (a pinch) welded
    into a non-manifold edge, trimesh called the shell not watertight and the weld dropped it whole (10 Nov: scout_v4_frame lost its
    95 cm3 main body, the 55 mm3 of specks left passed as a watertight export, strength crashed). Now the pinch clusters stay unwelded
    and are pulled 1 um apart: both bodies kept, watertight, same volume, still closed after an STL reload. And an export that loses
    the body anyway is a failed export: lost_mm3, not watertight, require_mesh raises ExportError (build_and_check: a FAIL result)."""
    try:
        import trimesh, tempfile, os
        from . import build as B
        from .part import tm
        a = _mbox(10, 10, 10, (5, 5, 5)); b = _mbox(10, 10, 10, (15 + 5e-5, 15 + 5e-5, 5))    # edge to edge, 5e-5 mm apart
        P = a + b
        m, rep = B.export_mesh(P, log=lambda s_: None)
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "x.stl"); m.export(fn); r = trimesh.load(fn, force="mesh")
            reload_ok = r.is_watertight and abs(r.volume - 2000.0) < 0.5
        kept = rep["watertight"] and rep["bodies"] == 2 and rep.get("pinch_vertices", 0) > 0 and abs(m.volume - 2000.0) < 0.5 and not rep.get("lost_mm3")
        keep = B.weld_clean
        try:
            B.weld_clean = lambda v, f, min_mm3=1.0: B.weld_mesh(*[np.asarray(x) for x in (tm(_mbox(1, 1, 1, (0, 0, 0))).vertices, tm(_mbox(1, 1, 1, (0, 0, 0))).faces)])
            m2, rep2 = B.export_mesh(P, log=lambda s_: None)
        finally:
            B.weld_clean = keep
        try:
            B.require_mesh(m2, rep2, P); raised = False
        except B.ExportError as e:
            raised = "LOST" in str(e)
        guard = rep2.get("lost_mm3", 0) > 1900 and not rep2["watertight"] and raised
        check("export: a pinch keeps both bodies watertight through a reload; an export that loses the body fails with its reason",
              bool(kept and reload_ok and guard), f"pinch {rep.get('pinch_vertices')}, bodies {rep['bodies']}, {m.volume:.3f} mm3, "
              f"watertight {rep['watertight']}, reload {reload_ok}; guard lost {rep2.get('lost_mm3')} raised {raised}")
    except Exception as e:
        traceback.print_exc()
        check("export pinch", False, str(e))


def t_setup_reads():
    """a part's identity is every repo file its setup reads (tools/part_sha.py record): a tools/ module the part imports (already imported
    by the caller, served from __pycache__) and the systems/ data that module opens are in setup.json "reads"; a changed value there
    changes the setup sha, the result reads stale and --affected reruns the part; a file it does not read reruns nothing (10 Nov: the
    Scout v4 parts read systems/scout_v4.head_load.json, outside the old key, and CI neither reran nor superseded them)"""
    try:
        import os, sys, json, tempfile, shutil, importlib
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(root, "tools"))
        import part_sha as PS
        importlib.reload(PS)
        tag = f"_t_reads_{os.getpid()}"
        mod, dat = os.path.join(root, "tools", tag + ".py"), os.path.join(root, "systems", tag + ".json")
        d = tempfile.mkdtemp(prefix="setup_reads_", dir=os.path.join(root, "parts"))
        try:
            open(dat, "w").write('{"load": -10.0}')
            open(mod, "w").write("import json, os\nLOAD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'systems', '%s.json')))['load']\n" % tag)
            py = os.path.join(d, "part.py")
            open(py, "w").write("import os, sys\nimport manifold3d as mf\nfrom topo.part import Part\n"
                                "sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools'))\n"
                                f"from {tag} import LOAD\n"
                                "box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])\n"
                                "def setup(h):\n    p = Part('t', h, box(0, 3, 0, 12, 0, 12) + box(20, 24, 3, 9, 3, 9), box(100, 101, 100, 101, 100, 101), box(0, 24, 0, 12, 0, 12), log=lambda s_: None)\n"
                                "    p.support('default', box(-1, 1.5, -1, 13, -1, 13)); p.load('down', box(20, 24, 3, 9, 3, 9), (0, 0, LOAD)); return p\n"
                                "SPEC = dict(h=1.0, gen_cm3=1.0, max_worst=0.5)\n")
            importlib.import_module(tag)                                       # the caller has it already: a fresh import must still record it
            importlib.import_module(tag)
            a = PS.setup_sha(py, refresh=True); reads = json.load(open(os.path.join(d, "setup.json")))["reads"]
            ids = PS.files(py)
            inkey = f"tools/{tag}.py" in reads and f"systems/{tag}.json" in reads and f"systems/{tag}.json" in ids
            nojunk = not any(r.startswith(("topo/", ".git")) or "__pycache__" in r or r.endswith("setup.json") for r in reads)
            res = dict(part_sha=PS.part_sha(py), lib_sha=PS.lib_sha(py), setup_sha=a)
            json.dump(res, open(os.path.join(d, "result.json"), "w"))
            name = os.path.basename(d)
            quiet = PS.affected(["systems/runtimes.json", "tools/dim.py"], pys=[py]) == []      # not read: nothing reruns
            same = PS.affected([f"systems/{tag}.json"], pys=[py]) == []                        # read, unchanged setup: nothing reruns
            open(dat, "w").write('{"load": -14.0}')
            b = PS.setup_sha(py)                                               # the cache sees the data change without refresh
            stale = PS.stale(py, res)
            rerun = PS.affected([f"systems/{tag}.json"], pys=[py]) == [name]
            ok = inkey and nojunk and b != a and stale and quiet and same and rerun
            check("setup reads: a read systems/ or tools/ file is the part's identity, its change reruns the part, other files do not", ok,
                  f"in the key {inkey}, no engine or output files {nojunk}, sha changed {b != a}, stale {stale}, unread file quiet {quiet}, "
                  f"unchanged setup quiet {same}, changed data reruns {rerun}; reads {reads}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
            for f in (mod, dat):
                if os.path.exists(f):
                    os.remove(f)
            sys.modules.pop(tag, None)
    except Exception as e:
        traceback.print_exc()
        check("setup reads", False, str(e))


def t_setup_sha():
    """staleness by setup (tools/part_sha.py): a result carrying the setup_sha of the setup part.py builds now is current even when
    part.py's or a lib's file hash changed (30 Oct: a two-line validation in cage_node_lib flagged eight node results stale and they
    were restamped by hand); a result from another setup is stale; a result without setup_sha falls back to the file rule."""
    try:
        import os, sys, json, tempfile, shutil
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        import importlib, part_sha as PS
        importlib.reload(PS)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = tempfile.mkdtemp(prefix="setup_sha_", dir=os.path.join(root, "parts"))      # under parts/, as the cache expects
        try:
            py = os.path.join(d, "part.py")
            src = ("import manifold3d as mf\nfrom topo.part import Part\n"
                   "box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])\n"
                   "LOAD = %s\n"
                   "def setup(h):\n    p = Part('t', h, box(0, 3, 0, 12, 0, 12) + box(20, 24, 3, 9, 3, 9), box(100, 101, 100, 101, 100, 101), box(0, 24, 0, 12, 0, 12), log=lambda s_: None)\n"
                   "    p.support('default', box(-1, 1.5, -1, 13, -1, 13)); p.load('down', box(20, 24, 3, 9, 3, 9), (0, 0, LOAD)); return p\n"
                   "SPEC = dict(h=1.0, gen_cm3=1.0, max_worst=0.5)\n")
            open(py, "w").write(src % "-10.0")
            a = PS.setup_sha(py); cached = json.load(open(os.path.join(d, "setup.json")))
            open(py, "w").write(src % "-10.0" + "# a comment: the file changed, the setup did not\n")
            b = PS.setup_sha(py)
            res_same = dict(part_sha="old", lib_sha="", setup_sha=a)
            res_other = dict(part_sha=PS.part_sha(py), lib_sha="", setup_sha="0000000000000000")
            res_old = dict(part_sha="old", lib_sha="")
            not_stale = not PS.stale(py, res_same); other_stale = PS.stale(py, res_other); old_rule = PS.stale(py, res_old)
            open(py, "w").write(src % "-12.0")
            c = PS.setup_sha(py)
            # a data file the setup reads (parts/<p>/inputs/, as systems/*.json for the cage nodes) is in the cache key: a changed value
            # recomputes the sha and the result reads stale (30 Oct: hub ends moved in overseer.cage.json, nodes 3 and 4 read as current)
            os.makedirs(os.path.join(d, "inputs"))
            open(os.path.join(d, "inputs", "load.json"), "w").write('{"load": -12.0}')
            src_d = src.replace("LOAD = %s", "import json, os\nLOAD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 'load.json')))['load']")
            open(py, "w").write(src_d % ())
            d1 = PS.setup_sha(py)
            open(os.path.join(d, "inputs", "load.json"), "w").write('{"load": -14.0}')
            d2 = PS.setup_sha(py)
            data_stale = PS.stale(py, dict(part_sha=PS.part_sha(py), lib_sha="", setup_sha=d1))
            sysjson = any(f.endswith(os.path.join("inputs", "load.json")) for f in PS.data_files(py))      # the read file is in the key
            ok = a == b and cached["setup_sha"] == a and not_stale and other_stale and old_rule and c != a and d1 == c and d2 != d1 and data_stale and sysjson
            check("setup sha: a comment keeps a result current, a changed load or data file makes it stale, old results use the file rule", ok,
                  f"comment {a == b}, cache {cached['setup_sha'] == a}, current {not_stale}, other setup stale {other_stale}, old rule {old_rule}, load changed {c != a}, "
                  f"data file: same {d1 == c}, changed {d2 != d1}, stale {data_stale}; the read file in the key {sysjson}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    except Exception as e:
        traceback.print_exc()
        check("setup sha", False, str(e))


def t_early_check():
    """the early strength stop (topo/run.py early_check): on a field that joins the loads to the supports it returns the index of the
    same solver build_and_check uses and raises EarlyFail over early_k x max_worst; on a field that does not join them it is skipped
    (an index on a broken field means nothing). The holder's full run (t_end_to_end) prints early vs final so the threshold stays honest."""
    try:
        from .run import early_check, EarlyFail
        p = _mini_part()
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        cen = p.cell_centres()[L.design]
        bar = (np.abs(cen[:, 1] - 5) < 3) & (np.abs(cen[:, 2] - 5) < 3)                     # a 6 x 6 bar from the base to the pad
        f_bar = L.fields(np.where(bar, 1.0, 0.0), 32.0)
        f_none = L.fields(np.zeros(L.nd), 32.0)                                            # nothing between base and pad
        msgs = []
        st = early_check(L, f_bar, 1, 0.5, 1e6, log=msgs.append)                             # a huge k: never raises
        skipped = early_check(L, f_none, 1, 0.5, 3.0, log=msgs.append)
        try:
            early_check(L, f_bar, 1, 0.5, 1e-6, log=msgs.append); raised = False
        except EarlyFail as e:
            raised = e.info["worst"] == st["worst"] and e.info["beta"] == 2.0
        ok = st is not None and st["worst"] is not None and st["worst"] > 0 and skipped is None and raised and any("skipped" in m for m in msgs)
        check("early check: index on a joined field, skipped on a broken one, EarlyFail over k x max_worst", ok,
              f"index {None if st is None else st['worst']}, skipped {skipped is None}, raised {raised}")
    except Exception as e:
        traceback.print_exc()
        check("early check", False, str(e))


def t_strength_floating():
    """strength (topo/build.py): when no built cell is joined to a support (10 Nov: scout_v4_frame, the export had lost the 95 cm3
    main shell, the 55 mm3 left floated and np.argmax of an empty set ended the run with no result.json) it returns a FAIL with its
    reason (ok False, fail, worst None), never raises; a joined bar still gets its index."""
    try:
        from .build import strength
        p = _mini_part()
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        cen = p.cell_centres()
        speck = (np.abs(cen[:, 0] - 11) < 2) & (np.abs(cen[:, 1] - 5) < 2) & (np.abs(cen[:, 2] - 5) < 2)    # a lone blob mid-room
        msgs = []
        st = strength(L, mask=speck, log=msgs.append)
        bar = ((np.abs(cen[:, 1] - 5) < 3) & (np.abs(cen[:, 2] - 5) < 3)) | L.solid
        sb = strength(L, mask=bar, log=lambda s_: None)
        ok = (st["ok"] is False and st["worst"] is None and "support" in st["fail"] and any("FAIL" in m for m in msgs)
              and sb["ok"] and sb["worst"] is not None and sb["worst"] > 0)
        check("strength: a build joined to no support is a FAIL with its reason, not a crash; a joined bar is rated", bool(ok),
              f"floating {st}; bar worst {sb['worst']}")
    except Exception as e:
        traceback.print_exc()
        check("strength floating", False, str(e))


def t_floating_load():
    """a drawn pad no support reaches (a void slab between it and the design: the h 2.5 head mount's 1.2 mm camera posts, whose
    cell chains break) is left out of the model and its load moves to the nearest material the supports reach, with a log line,
    instead of 'zero compliance of the full solid' (head mount run 74, 30 Oct); the ordinary part is untouched."""
    try:
        import manifold3d as mf
        from .part import Part
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        pad = box(20, 24, 2, 8, 2, 8)
        p = Part("float", 1.0, box(0, 3, 0, 10, 0, 10) + pad, box(17, 20, -1, 11, -1, 11), box(0, 24, 0, 10, 0, 10), log=lambda s_: None)
        p.support("default", box(-1, 1.5, -1, 11, -1, 11))
        p.load("down", pad, (0, 0, -10.0))
        msgs = []
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(p, volfrac=0.3, settings=s, log=msgs.append)
        moved = any("moved up to" in m for m in msgs); left = any("joined to no support" in m for m in msgs)
        c = L.cases[0]
        on_act = bool(L._node_mask(~L.void).ravel()[c.nodes].all())
        p0 = _mini_part(); m0 = []
        L0 = Layout(p0, volfrac=0.3, settings=s, log=m0.append)
        ok = moved and left and on_act and L.c_ref[0] > 0 and not L.floating.sum() == 0 and L0.floating.sum() == 0 and not any("moved" in m for m in m0)
        check("floating load: a pad no support reaches is left out and its load moves to the nearest carried material", ok,
              f"floating {int(L.floating.sum())} cells, moved {moved}, c_ref {L.c_ref[0]:.3g}, nodes on material {on_act}; plain part floating {int(L0.floating.sum())}")
    except Exception as e:
        traceback.print_exc()
        check("floating load", False, str(e))


def t_audit_plate():
    """audit: an envelope longer than the 180 mm plate at every rotation is refused (the front arm, 225 x 63: 29 Oct); a 170 x 170 one passes."""
    from topo.audit import plate_fit
    import numpy as np
    def env(L, W, H, h=2.0):
        e = np.zeros((int(L / h), int(W / h), int(H / h)), bool); e[:] = True; return e
    a = plate_fit(env(224, 64, 44), 2.0); b = plate_fit(env(170, 170, 40), 2.0); c = plate_fit(env(200, 40, 40), 2.0)
    ok = (not a[3]) and b[3] and c[3] and abs(a[0] - 224) < 3 and abs(a[1] - 64) < 3      # 200 x 40 fits on the diagonal
    RESULTS.append(("audit_plate", ok)); print(f"audit plate: {'ok' if ok else 'FAIL'} 224x64 {a[3]} 170x170 {b[3]} 200x40 {c[3]}")


def t_audit_block_measure():
    """audit: the block measure (diameter_mm) reads a drawn slab the same at every pitch, to the cell: an 8 mm boss block is
    not a block (limit 8) at h 1.5, 2.0 or 1.0, a 12 mm one is at all three (29 Oct: the tube arm roots were refused at h 1.5
    for 8 mm blocks that passed at h 2.0, because the distance transform counted a centre cell twice)."""
    from topo.audit import diameter_mm
    import numpy as np
    def slab(t_mm, h):
        n = int(round(t_mm / h)); m = np.zeros((n + 4, 30, 30), bool); m[2:2 + n] = True; return m
    thin = [diameter_mm(slab(8.0, h), h) for h in (1.5, 2.0, 1.0)]
    thick = [diameter_mm(slab(12.0, h), h) for h in (1.5, 2.0, 1.0)]
    ok = all(d <= 8.0 for d in thin) and all(d > 8.0 for d in thick) and all(abs(d - 8.0) <= 2.0 for d in thin) and all(abs(d - 12.0) <= 2.0 for d in thick)
    RESULTS.append(("audit_block_measure", ok)); print(f"audit block measure: {'ok' if ok else 'FAIL'} 8 mm -> {[round(d, 1) for d in thin]}  12 mm -> {[round(d, 1) for d in thick]}")


def t_audit_parity():
    """audit: parity (11 Nov, the Scout v4 frame's flow land flush on the nose room's floor): a box stacked flush on a wider one is two
    shells in the manifold union and the +z rays meet the shared face twice: parity_columns reports those columns; the same boxes
    overlapped 0.5 report none."""
    try:
        import manifold3d as mf
        from topo.audit import parity_columns
        from .part import tm
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        xs = ys = np.arange(-59.0, 60.0, 2.0)
        flush = parity_columns(tm(box(-50, 50, -40, 40, 9.6, 24.5) + box(-52, 54, 33.8, 40, -2.3, 9.6)), xs, ys)
        lap = parity_columns(tm(box(-50, 50, -40, 40, 9.6, 24.5) + box(-52, 54, 33.8, 40, -2.3, 10.1)), xs, ys)
        check("audit parity: a box flush on another is refused, overlapped 0.5 passes", len(flush) > 0 and not lap, f"flush {len(flush)} columns, overlapped {len(lap)}")
    except Exception as e:
        traceback.print_exc()
        check("audit parity", False, str(e))


def t_audit_drawn_thin():
    """audit: drawn_thin (4 Nov: the Scout tail unit's 2 mm tape lands left blobs on necks, the arm clips' tie collar was sheared to a
    1.1 mm sliver by the plate keep-out). Exact geometry: a 2 mm land is refused, a 4 mm one passes, a collar (O8.2, 3.2 bore, 2.5 wall)
    passes, a O5.6 post round a O3.2 bore passes too (a ring's wall is the gate's thin_walls check and thin_ok), a 6 mm pad sheared
    to 1.2 mm by a cut fails, a 3 mm hook (the holder demo) passes."""
    try:
        import manifold3d as mf, trimesh
        from topo.audit import thin_drawn
        from .part import man
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        none = box(100, 101, 100, 101, 100, 101)
        collar = man(trimesh.creation.cylinder(4.1, 13.0, sections=48)) - man(trimesh.creation.cylinder(1.6, 14.0, sections=48))
        post = man(trimesh.creation.cylinder(2.8, 10.0, sections=48)) - man(trimesh.creation.cylinder(1.6, 11.0, sections=48))
        pad = box(0, 10, 0, 10, 0, 6); shear = box(-1, 11, -1, 11, 1.2, 7)
        r = [thin_drawn(box(0, 61, 0, 17, 0, 2), none), thin_drawn(box(0, 61, 0, 17, 0, 4), none), thin_drawn(collar, none), thin_drawn(post, none),
             thin_drawn(post, none, [post]), thin_drawn(pad, shear), thin_drawn(box(26, 34, 0, 3, 14, 24), none)]
        ok = len(r[0]) == 1 and not r[1] and not r[2] and not r[3] and not r[4] and len(r[5]) == 1 and not r[6]   # a ring's thin wall is the gate's thin_walls check (and thin_ok), not this one
        check("audit drawn_thin: a 2 mm land fails, 4 mm passes, a 2.5-wall collar and a O5.6 post pass (ring walls are the gate's), a pad sheared to 1.2 fails, a 3 mm hook passes", ok,
              f"land 2 {r[0]} land 4 {r[1]} collar {r[2]} post {r[3]} post thin_ok {r[4]} sheared pad {r[5]} hook {r[6]}")
    except Exception as e:
        traceback.print_exc()
        check("audit drawn_thin", False, str(e))


def t_audit_env_neck():
    """audit: env_neck (5 Nov: the Scout tail unit's node arm met its bay over a 4 mm overlap, the nose unit's camera posts had 1 mm of
    room beside their stubs). A loaded pad whose only room to the support is a 2 mm slab is refused at min member 4; the same pad on a
    10 mm slab passes."""
    try:
        import manifold3d as mf
        from topo.audit import audit, SetupError
        from .part import Part
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        def part(slab_t):
            base = box(0, 3, 0, 20, 0, 20); pad = box(30, 34, 6, 14, 6, 14)
            env = box(0, 14, 0, 20, 0, 20) + box(14, 26, 0, 20, 10 - slab_t / 2, 10 + slab_t / 2) + box(26, 34, 0, 20, 0, 20)
            p = Part("neckmini", 1.0, base + pad, box(100, 101, 100, 101, 100, 101), env, log=lambda s_: None)
            p.support("default", box(-1, 1.5, -1, 21, -1, 21)); p.load("down", pad, (0, 0, -10.0))
            return p
        def refused(t):
            try:
                audit(part(t), 4.0, log=lambda s_: None); return False
            except SetupError as e:
                return "env_neck" in str(e)
        a, b = refused(2.0), refused(10.0)
        check("audit env_neck: a loaded pad on a 2 mm slab of room is refused at member 4, on a 10 mm slab it passes", a and not b, f"2 mm refused {a}, 10 mm refused {b}")
    except Exception as e:
        traceback.print_exc()
        check("audit env_neck", False, str(e))


def t_audit_solid_ok():
    """audit: solid_ok (Spark 9 Nov: the arm motor mounts are solid under the motor so they do not crack again). A 14 mm drawn disc
    on a support pad is refused as a drawn block without solid_ok, passes with Part(..., solid_ok=[disc]) and is listed in the
    report; a second 14 mm block outside the solid_ok body is still refused; the setup hash changes with solid_ok."""
    try:
        import manifold3d as mf, trimesh
        from topo.audit import audit, SetupError
        from .part import Part, man
        from .run import design_hash
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        disc = man(trimesh.creation.cylinder(14.0, 14.0, sections=48)).translate([20, 20, 10])
        def part(ok, extra=None):
            base = box(0, 3, 0, 40, 0, 20)
            drawn = base + disc + (extra or box(100, 101, 100, 101, 100, 101))
            env = box(0, 40, 0, 40, 0, 20) + box(40, 60, 0, 40, 0, 20)
            p = Part("solidok", 1.0, drawn, box(200, 201, 200, 201, 200, 201), env, log=lambda s_: None, solid_ok=[disc] if ok else [])
            p.support("default", box(-1, 1.5, -1, 41, -1, 21)); p.load("down", disc, (0, 0, -10.0))
            return p
        def run(p):
            try:
                return audit(p, 4.0, log=lambda s_: None), ""
            except SetupError as e:
                return None, str(e)
        r0, e0 = run(part(False)); r1, e1 = run(part(True)); r2, e2 = run(part(True, box(44, 58, 13, 27, 3, 17)))
        a = r0 is None and "drawn_blocks" in e0
        b = r1 is not None and len(r1["solid_ok"]) == 1 and r1["solid_ok"][0]["cells"] > 0
        c = r2 is None and "drawn_blocks" in e2 and "[51" in e2
        d = design_hash(part(False), 1.0, 1.0, {}) != design_hash(part(True), 1.0, 1.0, {})
        check("audit solid_ok: a 14 mm drawn disc is refused as a block, passes declared solid_ok (listed), a block beside it still fails, the hash changes",
              a and b and c and d, f"refused {a} ({e0[:120]!r}), passes {b} ({e1[:200]!r}), other block refused {c}, hash {d}")
    except Exception as e:
        traceback.print_exc()
        check("audit solid_ok", False, str(e))


def t_audit_load_room():
    """audit: load_room (5 Nov: the Scout nose unit's seat regions reached above the rim into the block's room, two knobs on necks).
    A pad loaded on itself passes; the same pad loaded by a region reaching 5 mm above it into design room is refused."""
    try:
        import manifold3d as mf
        from topo.audit import audit, SetupError
        from .part import Part
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        def part(z_top):
            base = box(0, 3, 0, 20, 0, 20); pad = box(30, 34, 6, 14, 6, 14)
            env = box(0, 34, 0, 20, 0, 30)
            p = Part("roommini", 1.0, base + pad, box(100, 101, 100, 101, 100, 101), env, log=lambda s_: None)
            p.support("default", box(-1, 1.5, -1, 21, -1, 21)); p.load("down", box(30, 34, 6, 14, 6, z_top), (0, 0, -10.0))
            return p
        def refused(z_top):
            try:
                audit(part(z_top), 4.0, log=lambda s_: None); return False
            except SetupError as e:
                return "load_room" in str(e)
        a, b = refused(24.0), refused(14.0)
        check("audit load_room: a pad loaded 10 mm into the room above it is refused, loaded on itself it passes", a and not b, f"room refused {a}, on the pad refused {b}")
    except Exception as e:
        traceback.print_exc()
        check("audit load_room", False, str(e))


def t_design_hash():
    """the design hash (topo/run.py) is the same for the same setup built twice and differs when a load, a cut or the budget
    changes (29 Oct: parts.yml stops a running part whose design main has since changed)."""
    try:
        import manifold3d as mf
        from .part import Part
        from .run import design_hash
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        def mk(load=-10.0, cut=None):
            p = Part("dh", 1.0, box(0, 3, 0, 12, 0, 12) + box(20, 24, 3, 9, 3, 9), cut or box(100, 101, 100, 101, 100, 101), box(0, 24, 0, 12, 0, 12), log=lambda s_: None)
            p.support("default", box(-1, 1.5, -1, 13, -1, 13)); p.load("down", box(20, 24, 3, 9, 3, 9), (0, 0, load))
            return p
        spec = dict(max_worst=0.5)
        a, b = design_hash(mk(), 1.0, 1.0, spec), design_hash(mk(), 1.0, 1.0, spec)
        c = design_hash(mk(load=-12.0), 1.0, 1.0, spec)
        d = design_hash(mk(cut=box(10, 12, 4, 8, 4, 8)), 1.0, 1.0, spec)
        e = design_hash(mk(), 1.0, 1.5, spec)
        ok = a == b and len({a, c, d, e}) == 4
        RESULTS.append(("design_hash", ok)); print(f"design hash: {'ok' if ok else 'FAIL'} same {a == b}, load/cut/budget distinct {len({a, c, d, e})}/4")
    except Exception as ex:
        RESULTS.append(("design_hash", False)); print(f"design hash: FAIL {ex}")


def t_near_symmetry():
    """a part mirrored but for a detail (29 Oct, the head mount's L2 key slot: its camera arms came out different left and right)
    is held symmetric on the design cells whose mirror is design too; the detail's cells stay free; an exact mirror still gives
    the plain symmetry."""
    try:
        import manifold3d as mf
        from .part import Part
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        base, pad = box(0, 3, 0, 12, 0, 12), box(20, 24, 3, 9, 3, 9)
        detail = box(10, 12, 9.5, 11.5, 5, 7)                                    # a small cut on one side (+y) only, centred in z
        res = {}
        for name, cuts in (("exact", box(100, 101, 100, 101, 100, 101)), ("near", detail)):
            p = Part("sym", 1.0, base + pad, cuts, box(0, 24, 0, 12, 0, 12), log=lambda s_: None)
            p.support("default", box(-1, 1.5, -1, 13, -1, 13)); p.load("down", pad, (0, 0, -10.0))
            s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", finish="drawn", near_sym=True)
            L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
            rng = np.random.default_rng(3); x = rng.uniform(0.0, 1.0, L.nd); xs = L.symmetrise_design(x)
            full = np.zeros(L.nel); full[L.design] = xs; f3 = full.reshape(L.shape)
            d3 = L.design.reshape(L.shape); both = d3 & np.flip(d3, 1)
            held = L.sym_mask[1].reshape(L.shape) if 1 in L.sym_mask else both
            asym = float(np.abs(f3 - np.flip(f3, 1))[held].max())
            res[name] = dict(axes=list(L.sym_axes), masked=1 in L.sym_mask, asym=asym, held=int(held.sum()), free=int((d3 & ~held).sum()))
        ok = (1 in res["exact"]["axes"] and not res["exact"]["masked"] and res["exact"]["asym"] < 1e-12
              and 1 in res["near"]["axes"] and res["near"]["masked"] and res["near"]["asym"] < 1e-12 and res["near"]["free"] > 0)
        check("near symmetry: a detail on one side no longer costs the whole part its mirror", ok, str(res))
    except Exception as e:
        traceback.print_exc()
        check("near symmetry", False, str(e))


def t_skin_core():
    """masked erosion with thick drawn skins eroding (29 Oct): an 8 mm drawn pad keeps only the core the solid gate keeps and its
    skin erodes with the design, so a 5 mm member meeting the pad at its rim leaves the pad's core outside the engine's eroded body
    (the gate's verdict on the rear arm's plate bosses, which the engine passed with the pad held whole), while a member on the
    pad's centre joins it; the rim member's tie compliance is far above the centre one's."""
    try:
        import manifold3d as mf
        from scipy import ndimage as ndi
        from .part import Part
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        base, pad = box(0, 3, 0, 12, 0, 12), box(20, 28, 1, 9, 1, 9)             # an 8 mm pad: thick enough to keep a core
        p = Part("rim", 1.0, base + pad, box(100, 101, 100, 101, 100, 101), box(0, 28, 0, 12, 0, 12), log=lambda s_: None)
        p.support("default", box(-1, 1.5, -1, 13, -1, 13))
        p.load("down", pad, (0, 0, -10.0))
        p.load("side", pad, (0, 5.0, 0))
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        cen = p.cell_centres()[L.design]
        padc = (L.anchor & (np.arange(L.nel) >= 0)).reshape(L.shape)
        xs = p.cell_centres()[:, 0].reshape(L.shape)
        padc &= xs > 19                                              # the pad's anchored core
        off, ratio = {}, {}
        for kind, (y0, y1, z0, z1) in (("centre", (2.5, 7.5, 2.5, 7.5)), ("rim", (5.5, 10.5, 5.5, 10.5))):
            lnk = (cen[:, 0] < 20) & (cen[:, 1] > y0) & (cen[:, 1] < y1) & (cen[:, 2] > z0) & (cen[:, 2] < z1)
            f = L.fields(np.where(lnk, 1.0, 0.0), 32.0)
            B = ((f["rho_e"] > 0.5) | L.anchor).reshape(L.shape)
            lab, n = ndi.label(B, ndi.generate_binary_structure(3, 1))
            sizes = np.bincount(lab.ravel()); sizes[0] = 0; main = int(np.argmax(sizes))
            off[kind] = float((lab[padc] != main).mean())
            comp, _ = L._compliance(f["rho_e"], need_grad=False)
            ratio[kind] = float((comp[:2] / L.c_ref[:2]).max())
        ok = L.skin.any() and padc.sum() > 0 and off["centre"] < 0.01 and off["rim"] > 0.5 and ratio["rim"] > 20 * ratio["centre"]
        check("thick drawn skins erode: a rim-mounted member leaves the pad's core off, a centred one joins it", ok,
              f"skin cells {int(L.skin.sum())}, pad core {int(padc.sum())}; core off main: centre {off['centre']:.2f}, rim {off['rim']:.2f}; "
              f"tie ratio rim / centre {ratio['rim'] / max(ratio['centre'], 1e-12):.0f}")
    except Exception as e:
        traceback.print_exc()
        check("thick drawn skins erode", False, str(e))


def t_coarse_setup_fails():
    """a setup that cannot be built at 2h skips the coarse stage instead of killing the run (9 Nov, scout_v4_base_nose: a thin
    side land's load region was empty at h 3.0, Part.load_multi raised outside the coarse stage's try, the run died at 2 min)."""
    import os
    import tempfile
    import manifold3d as mf
    from .part import Part
    from .run import optimise
    box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

    def setup(h):   # a 0.4 mm load slab: the 3 x 3 x 3 samples catch it at h 1.0, fall either side of it at h 2.0
        p = Part("thin_load", h, box(0, 3, 0, 10, 0, 10) + box(20, 24, 2, 8, 2, 8), box(100, 101, 100, 101, 100, 101),
                 box(0, 24, 0, 10, 0, 10), log=lambda s_: None)
        p.support("default", box(-1, 1.5 * h, -1, 11, -1, 11))
        p.load("down", box(21.8, 22.2, 3, 7, 3, 7), (0, 0, -10.0))
        return p
    lines = []
    try:
        try:
            setup(2.0)
            raised = False
        except ValueError:
            raised = True
        out = tempfile.mkdtemp(prefix="topo_coarse_skip_")
        s = Settings(min_member_mm=3.0, max_stage=2, min_stage=1, audit="warn", time_limit_s=120)
        optimise(setup, 1.0, 0.6, out, "thin_load", settings=s, coarse=True, maxit=1, log=lines.append)
        skipped = any("coarse stage skipped" in l and "empty region" in l for l in lines)
        check("coarse: a setup that raises at 2h skips the coarse stage, the fine run goes on", raised and skipped,
              f"2h setup raised {raised}, skip logged {skipped}")
    except Exception as e:
        traceback.print_exc()
        check("coarse: a setup that raises at 2h skips the coarse stage", False, f"{type(e).__name__}: {e}; log tail {lines[-3:]}")


def t_audit_room_full():
    """room_full (topo/audit.py room_strength): the room filled solid must pass max_worst, else no layout can and the run is refused
    (9 Nov: the Scout v4 frame's 16 x 8 side lugs read 2.55 after 69 min). The 24 x 10 x 10 bar holds 10 N; at 20 kN it cannot."""
    try:
        from .audit import room_strength, SetupError
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        p = _mini_part()
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        ok_rep = room_strength(L, 0.5, log=lambda s_: None)
        import manifold3d as mf
        from .part import Part
        box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
        q = Part("mini_heavy", 1.0, box(0, 3, 0, 10, 0, 10) + box(20, 24, 2, 8, 2, 8), box(100, 101, 100, 101, 100, 101),
                 box(0, 24, 0, 10, 0, 10), log=lambda s_: None)
        q.support("default", box(-1, 1.5, -1, 11, -1, 11))
        q.load("down 20 kN", box(20, 24, 2, 8, 2, 8), (0, 0, -20000.0), cls="crash")
        Lq = Layout(q, volfrac=0.3, settings=s, log=lambda s_: None)
        try:
            room_strength(Lq, 0.5, log=lambda s_: None); refused = False
        except SetupError as e:
            refused = "room_full" in str(e)
        soft = room_strength(Lq, 0.5, log=lambda s_: None, hard=False)
        ok = ok_rep["worst"] is not None and ok_rep["worst"] < 0.5 and refused and soft.get("fail") and soft["thinned"] > soft["worst"]
        check("audit room_full: a room that holds its load passes, one that cannot even filled solid is refused", bool(ok),
              f"10 N {ok_rep['worst']}, 20 kN {soft['worst']} refused {refused}, thinned {soft['thinned']}")
    except Exception as e:
        traceback.print_exc()
        check("audit room_full", False, str(e))


def t_memory_gate():
    """memory (topo/audit.py memory_fit / memory_measured, fem.Solver.solve_blocks; 10 Nov: scout_v4_frame's 396 cases held
    13.6 GB and the 16 GB runner killed the run at 2 h with no log). (1) The streamed solve, 1 case a block, gives the
    compliances and the gradient of one solve of every case at once (direct and multigrid). (2) A setup over the limit is
    refused before the first solve, and optimise writes result.json with status mem_refused and the reason."""
    import os, json, tempfile
    from .audit import memory_fit, MemoryRefused
    from .run import optimise
    try:
        p = _mini_part()
        errs = []
        for be in ("direct", "mg"):
            s = Settings(min_member_mm=3.2, backend=be, reach_check="off", auto_sym=False, finish="drawn")
            L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
            x = np.random.default_rng(3).uniform(0.2, 0.8, L.nd)
            f = L.fields(x, 4.0)
            rho = f["rho_e"]
            E = (s.emin + (1.0 - s.emin) * rho ** s.penal) * L.stiff
            F = np.stack([c.Fe for c in L.cases], 1)
            L.fe_solver.warm.clear()
            U = L.fe_solver.solve(E, F, [c.support for c in L.cases], keys=[c.name for c in L.cases])
            c_all = (F * U).sum(0)
            en = L.fe_solver.element_energy(U)
            J, dJdc, r = L._objective(c_all)
            dE = s.penal * np.maximum(rho, 1e-12) ** (s.penal - 1.0) * (1.0 - s.emin) * L.stiff
            g_all = sum(-dJdc[q] * dE * en[q] for q in range(len(L.cases)))
            L.fe_solver.warm.clear()
            L.fe_solver.block_size = lambda: 1
            comp, (S, _) = L._compliance(rho)
            g = L._grad_rho_e(rho, S, dJdc)
            errs.append(max(float(np.abs(comp - c_all).max() / np.abs(c_all).max()), float(np.abs(g - g_all).max() / np.abs(g_all).max())))
        ok1 = errs[0] < 1e-9 and errs[1] < 1e-6
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        fits = "fail" not in memory_fit(L, 14.0, log=lambda s_: None)
        try:
            memory_fit(L, 1e-3, log=lambda s_: None); refused = False
        except MemoryRefused as e:
            refused = "memory" in str(e) and e.info["estimate_gb"] > 1e-3
        out = tempfile.mkdtemp(prefix="topo_mem_")
        so = Settings(min_member_mm=3.0, max_stage=2, min_stage=1, audit="warn", time_limit_s=120, mem_limit_gb=1e-3)
        lines = []
        res = optimise(lambda h: _mini_part(h=h), 1.0, 0.6, out, "mini_mem", settings=so, coarse=False, maxit=1, log=lines.append)
        rj = json.load(open(os.path.join(out, "result.json")))
        written = rj["status"] == "mem_refused" and "GB" in (rj["strength"]["fail"] or "") and res["status"] == "mem_refused"
        check("memory: streamed solve = one solve of every case; over the limit refused before the solve, reason in result.json",
              ok1 and fits and refused and written,
              f"streamed vs all (direct, mg) rel err {errs[0]:.1e} {errs[1]:.1e}, fits 14 GB {fits}, refused {refused}, result.json {written}")
    except Exception as e:
        traceback.print_exc()
        check("memory gate", False, f"{type(e).__name__}: {e}")


def t_gate_memory():
    """gate memory (tools/solid_check.py edt / edt_box, 10 Nov: scout_v4_frame, 94 M gate cells, the runner killed right after the
    export: scipy's distance transform held ~60 bytes a cell at once, ~5 GB a call). (a) edt is scipy's distance_transform_edt bit for
    bit (also on a mask at the grid's edge) and edt_box is it on the body's box; (b) the gate with them reports exactly what the
    gate with scipy's full-grid transforms reports (a pad on a sliver, a neck, thin walls, a thin_ok post); (c) its peak is under
    audit.GATE_BYTES a cell and under 70 % of the old one's (on the frame: 37.5 vs 98 B a cell)."""
    import tracemalloc
    from scipy import ndimage as ndi
    from .part import tm
    from .build import _solid_check_module
    from . import audit
    try:
        SC = _solid_check_module()
        rng = np.random.default_rng(5)
        M = ndi.binary_opening(rng.random((41, 33, 27)) > 0.35); M[0, :5, :5] = True
        ok_a = all(np.array_equal(SC.edt(m, slab=s_), ndi.distance_transform_edt(m)) for m in (M, ~M) for s_ in (1, 7, 64))
        B = np.zeros((40, 30, 20), bool); B[0:9, 3:12, 5:15] = True; B[20:30, 10:25, 2:18] = True
        own, sl = SC.edt_box(B)
        ok_a = ok_a and np.array_equal(own, ndi.distance_transform_edt(B)[sl])
        cube = _mbox(20, 20, 20, (0, 0, 0)); pad = _mbox(4, 8, 8, (14.6, 0, 0)); sliver = _mbox(2.6, 1.2, 8, (11.3, 0, 0))
        fin = _mbox(10, 1.4, 14, (-15, 0, 0)); post = _mbox(3, 3, 6, (0, 0, 13))
        mesh, drawn, thin = tm(cube + sliver + pad + fin + post), [tm(pad + post)], [tm(post)]

        def run(lean):
            S2 = _solid_check_module()
            if not lean:
                S2.edt = lambda m, slab=8: ndi.distance_transform_edt(m)
                S2.edt_box = lambda m: (ndi.distance_transform_edt(m), (slice(None),) * 3)
            tracemalloc.start()
            ok, info = S2.solid_check(mesh, drawn=drawn, thin_ok=thin)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return ok, info, peak
        ok1, i1, p1 = run(True)
        ok0, i0, p0 = run(False)
        ok_b = bool(ok1 == ok0) and json.dumps(i1, sort_keys=True, default=str) == json.dumps(i0, sort_keys=True, default=str) and not ok1 \
            and bool(i1["drawn_not_in_main"]) and bool(i1["thin_walls"])
        n = audit.gate_voxels(mesh.bounds[0], mesh.bounds[1])
        ok_c = p1 < audit.GATE_BYTES * n and p1 < 0.7 * p0
        check("gate memory: edt = scipy bit for bit, same verdict and report, peak under GATE_BYTES a cell", ok_a and ok_b and ok_c,
              f"(a) {ok_a}; (b) {ok_b} ({ok1}, {len(i1['drawn_not_in_main'])} drawn off, {len(i1['thin_walls'])} thin, {len(i1['necks'])} necks); "
              f"(c) peak {p1 / n:.0f} B a cell (old {p0 / n:.0f}, limit {audit.GATE_BYTES:.0f}) on {n:,} cells")
    except Exception as e:
        traceback.print_exc()
        check("gate memory", False, f"{type(e).__name__}: {e}")


def t_build_memory_gate():
    """memory of the build and gate (topo/audit.py build_fit, topo/build.py build_and_check; 10 Nov: scout_v4_frame killed twice right
    after the export, no result): (a) the estimate grows with the part's box (the gate grid) and its faces; (b) a setup whose gate
    grid cannot fit is refused before the first solve (memory_fit's build entry); (c) a build over the limit after the layout
    writes result.json with status mem_refused, its reason and memory_build, never a killed runner; (d) a build that fits carries
    memory_build in result.json."""
    import os, tempfile
    from . import audit
    from .run import optimise
    g0 = audit.GATE_BYTES
    try:
        t1, _, n1 = audit.build_estimate((0, 0, 0), (50, 50, 20), faces=0, held_gb=0)
        t2, _, n2 = audit.build_estimate((0, 0, 0), (180, 180, 44), faces=838_474, held_gb=0)
        ok_a = n2 > 20 * n1 and t2 > 20 * t1 and 80e6 < n2 < 110e6
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn")
        L = Layout(_mini_part(), volfrac=0.3, settings=s, log=lambda s_: None)
        fits = "fail" not in L.mem and "build" in L.mem
        audit.GATE_BYTES = 1e12
        try:
            audit.memory_fit(L, 14.0, log=lambda s_: None); pre = False
        except audit.MemoryRefused as e:
            pre = "build" in str(e) and "gate" in str(e)
        audit.GATE_BYTES = g0
        out = tempfile.mkdtemp(prefix="topo_bmem_")
        so = Settings(min_member_mm=3.0, max_stage=2, min_stage=1, audit="warn", time_limit_s=120)

        def hook(s_):
            if "layout done" in str(s_):
                audit.GATE_BYTES = 1e12                        # the build sees a part whose gate grid cannot fit
        res = optimise(lambda h: _mini_part(h=h), 1.0, 0.6, out, "mini_bmem", settings=so, coarse=False, maxit=1, log=hook)
        audit.GATE_BYTES = g0
        rj = json.load(open(os.path.join(out, "result.json")))
        ok_c = (rj.get("status") == "mem_refused" and res.get("status") == "mem_refused" and "gate" in (rj["strength"]["fail"] or "")
                and rj["memory_build"]["when"] == "before the export cleanup" and not os.path.exists(os.path.join(out, "mini_bmem.stl")))
        out2 = tempfile.mkdtemp(prefix="topo_bmem2_")
        res2 = optimise(lambda h: _mini_part(h=h), 1.0, 0.6, out2, "mini_bmem2", settings=so, coarse=False, maxit=1, log=lambda s_: None)
        mb = res2.get("memory_build") or []
        ok_d = res2.get("status") is None and len(mb) == 2 and all("fail" not in m for m in mb) and mb[0]["faces"] > 0
        check("build memory gate: estimate by the gate grid, refused before the solve or before the gate with status mem_refused",
              ok_a and fits and pre and ok_c and ok_d,
              f"(a) {n1:,} -> {n2:,} cells, {t1:.2f} -> {t2:.1f} GB; fits {fits}; (b) refused before the solve {pre}; (c) result.json "
              f"{rj.get('status')} {str(rj['strength'].get('fail'))[:120]}; (d) {[m.get('estimate_gb') for m in mb]}")
    except Exception as e:
        traceback.print_exc()
        check("build memory gate", False, f"{type(e).__name__}: {e}")
    finally:
        audit.GATE_BYTES = g0


def _arms_part(arms=((45, 1), (-45, 1), (135, 1), (-135, 1)), h=2.0, sockets=(16,)):
    """slender arms (12 Nov, scout_v4_frame): a deck held at four bolt bosses, 74 mm diagonal arms (22 x 12 mm envelope) to motor pads
    loaded by thrust 17.7 N + 11 N imbalance across / along (flight), and socket bosses on the deck with 600 N rod loads (crash) that
    compete for the material. At gen 14 cm3 (four arms) the arms come out near the 6.25 mm minimum member at h 2.0 (~3 cells)."""
    import manifold3d as mf
    from .part import Part
    box = lambda x0, x1, y0, y1, z0, z1: mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    cyl = lambda r, z0, z1, x, y: mf.Manifold.cylinder(z1 - z0, r, r, 32).translate([x, y, z0])
    env = box(-24, 24, -24, 24, 0, 12)
    drawn, cuts, pads = [], [], []
    for ang, _ in arms:
        env = env + box(0, 80, -11, 11, 0, 12).rotate([0, 0, ang])
        px, py = 74 * np.cos(np.radians(ang)), 74 * np.sin(np.radians(ang))
        pad = cyl(9, 6, 12, px, py)
        drawn.append(pad); cuts.append(cyl(2.0, -1, 13, px, py)); pads.append((ang, pad))
    regs = []
    for bx, by in ((-18, -18), (18, -18), (-18, 18), (18, 18)):
        drawn.append(cyl(5, 0, 4, bx, by)); cuts.append(cyl(1.6, -1, 13, bx, by)); regs.append(cyl(5, -1, 0.9, bx, by))
    socks = []
    for y in sockets:
        for sy in (1, -1)[:2 if len(arms) > 2 else 1]:
            sk = cyl(6, 0, 12, 0, sy * y)
            drawn.append(sk); cuts.append(cyl(3.1, 3, 13, 0, sy * y)); socks.append(sk)
    p = Part("arms", h, mf.Manifold.batch_boolean(drawn, mf.OpType.Add), mf.Manifold.batch_boolean(cuts, mf.OpType.Add), env,
             log=lambda s: None)
    p.support("default", mf.Manifold.batch_boolean(regs, mf.OpType.Add))
    for i, (ang, pad) in enumerate(pads):
        a = np.radians(ang)
        along, across = np.array([np.cos(a), np.sin(a), 0]), np.array([-np.sin(a), np.cos(a), 0])
        p.load(f"motor {i + 1} across", pad, tuple(np.array([0, 0, 17.7]) + 11 * across), cls="flight")
        p.load(f"motor {i + 1} along", pad, tuple(np.array([0, 0, 17.7]) + 11 * along), cls="flight")
    for j, sk in enumerate(socks):
        p.load(f"socket {j} tension", sk, (0, 40.0, 600.0), cls="crash")
        p.load(f"socket {j} compression", sk, (0, -40.0, -600.0), cls="crash")
    return p


def t_built_hold():
    """the built hold (layout.py Settings.nominal_hold): inactive rows before it starts; once started every real case's compliance on
    the build projection (nominal and gate-eroded) reads 1 / (1 + hold_tol) - 1 on the reference design, and its gradient (through the
    sharp projection, body forces included) matches finite differences"""
    try:
        p = _mini_part()
        p.accel_case("grav", (0, 0, -9.81 * 30), cls="crash")
        rng = np.random.default_rng(5)
        beta = 4.0
        s = Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn", hold_proj_beta=6.0)
        L = Layout(p, volfrac=0.3, settings=s, log=lambda s_: None)
        x = rng.uniform(0.3, 0.7, L.nd)
        f = L.fields(x, beta)
        comp, (U, _) = L._compliance(f["rho_e"])
        fv0, _ = L._hold_constraints(f, comp, U)
        L.start_hold(f)
        f = L.fields(x, beta)
        comp, (U, _) = L._compliance(f["rho_e"], rho_g=f["rho_g"], rho_b=f["rho_b"], rho_be=f["fb"]["rho_e"])
        fv, dg = L._hold_constraints(f, comp, U)

        def G(xx):
            ff = L.fields(xx, beta)
            c, _ = L._compliance(ff["rho_e"], need_grad=False, rho_g=ff["rho_g"], rho_b=ff["rho_b"], rho_be=ff["fb"]["rho_e"])
            return c[len(L.cases):] / ((1 + s.hold_tol) * L.c_hold) - 1
        errs = []
        for i in rng.choice(L.nd, 6, replace=False):
            e = 1e-3        # the eroded copies' roundoff (Emin cells) swamps 1e-4 steps; truncation at 1e-3 is ~1e-6 at beta 6
            xp, xm = x.copy(), x.copy()
            xp[i] += e
            xm[i] -= e
            fd = (G(xp) - G(xm)) / (2 * e)
            errs += [abs(fd[k] - dg[k][i]) / max(abs(fd[k]), 1e-2 * np.abs(dg[k]).max()) for k in range(len(fv))]
        ok = (fv0 == [-1.0] * 2 * L.n_real and len(fv) == 2 * L.n_real == 6 and np.allclose(fv, 1 / (1 + s.hold_tol) - 1, atol=1e-9)
              and max(errs) < 1e-3 and L.mma.m == 3 + 2 * L.n_real + 1)
        check("built hold: rows inactive before it starts, at the reference after, gradient vs finite differences", ok,
              f"rows {np.round(fv, 4).tolist()}, max rel err {max(errs):.1e}")
    except Exception as e:
        traceback.print_exc()
        check("built hold", False, str(e))


def t_no_cavity():
    """no sealed void (Spark 13 Nov, foot nodes v8: the build kept 2.4-8.9 cm3 sealed cavities, full of slicer supports for good).
    The virtual-temperature measure (topo/cavity.py) on a 32 mm room: an open truss, a lump with a pocket open by a 4 mm channel
    and a solid block with a blind bore read under cavity_ratio, a 3 mm sealed pocket in a lump and a 2 mm one in a block over 4 x
    it; its gradient and the layout row's (through the build projection) match finite differences; the export fills a sealed
    cavity (fill_cavities, export_mesh) and the gate fails a mesh that holds one (sealed_voids, inner_shells), passes it filled."""
    try:
        import sys, os
        import trimesh
        from .cavity import Cavity
        from .part import man
        from .build import export_mesh
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        from solid_check import solid_check
        n = 40; sh = (n, n, n); idx = np.indices(sh)
        env = np.all((idx >= 4) & (idx < n - 4), 0)
        C = Cavity(sh, 1.0, np.zeros(n ** 3, bool), (~env).ravel(), env.ravel(), ks=Settings.cavity_ks, log=lambda s_: None)
        c = (n - 1) / 2; r = np.sqrt(((idx - c) ** 2).sum(0))
        rods = env & ((((idx[0] % 10) < 4) & ((idx[1] % 10) < 4)) | (((idx[1] % 10) < 4) & ((idx[2] % 10) < 4))
                      | (((idx[0] % 10) < 4) & ((idx[2] % 10) < 4)))
        lump = rods & ~(r <= 7) | (env & (r <= 7) & (r > 3))
        bore = (np.abs(idx[0] - c) < 2) & (np.abs(idx[1] - c) < 2) & (idx[2] < c)
        m = lambda A: C.measure(np.where(A, 1.0, 0.0).ravel())[0]
        opn = [m(rods), m(lump & ~bore), m(env & ~bore)]
        sealed = [m(lump), m(env & ~(r <= 2))]
        lim = Settings.cavity_ratio
        rng = np.random.default_rng(3)
        rho = np.where(env, rng.uniform(0.2, 0.8, sh), 0.0).ravel()
        g, d = C.measure(rho, need_grad=True)
        errs = []
        for k in rng.choice(np.flatnonzero(env.ravel()), 5, replace=False):
            e = 1e-4; rp, rm = rho.copy(), rho.copy(); rp[k] += e; rm[k] -= e
            fd = (C.measure(rp)[0] - C.measure(rm)[0]) / (2 * e)
            errs.append(abs(fd - d[k]) / max(abs(fd), 1e-3 * np.abs(d).max()))
        # the layout's row, through the build projection
        p = _mini_part()
        L = Layout(p, volfrac=0.3, settings=Settings(min_member_mm=3.2, backend="direct", reach_check="off", auto_sym=False, finish="drawn",
                                                     hold_proj_beta=4.0), log=lambda s_: None)
        x = rng.uniform(0.3, 0.7, L.nd)
        fv, dg = L._cavity_constraint(L.fields(x, 2.0), 2.0)
        lerr = []
        for i in rng.choice(L.nd, 4, replace=False):
            e = 1e-4; xp, xm = x.copy(), x.copy(); xp[i] += e; xm[i] -= e
            fd = (L._cavity_constraint(L.fields(xp, 2.0), 2.0)[0] - L._cavity_constraint(L.fields(xm, 2.0), 2.0)[0]) / (2 * e)
            lerr.append(abs(fd - dg[i]) / max(abs(fd), 1e-3 * np.abs(dg).max()))
        # export and gate
        outer = trimesh.creation.box((20.0, 20.0, 20.0)); inner = trimesh.creation.box((6.0, 6.0, 6.0)); inner.invert()
        hollow = trimesh.util.concatenate([outer, inner])
        ok_h, ih = solid_check(hollow)
        P = man(trimesh.creation.box((20.0, 20.0, 20.0))) - man(trimesh.creation.box((6.0, 6.0, 6.0)))
        M, rep = export_mesh(P, log=lambda s_: None)
        ok_f, i_f = solid_check(M)
        ok = (max(opn) < lim and min(sealed) > 4 * lim and max(errs) < 1e-3 and max(lerr) < 1e-3 and L.mma.m == 3 + 2 * L.n_real + 1
              and not ok_h and ih["sealed_voids"] and ih["inner_shells"] and rep.get("cavities_filled_mm3") == [216.0]
              and abs(M.volume - 8000.0) < 1e-6 and ok_f)
        check("no sealed void: open rooms under the limit, sealed pockets over 4 x it, gradients vs FD, the export fills, the gate refuses", ok,
              f"open {np.round(opn, 2).tolist()}, sealed {np.round(sealed, 1).tolist()} (limit {lim}), grad err {max(errs):.1e}, layout row "
              f"{fv:+.3f} err {max(lerr):.1e}; hollow gate {'PASS' if ok_h else 'FAIL'} {ih['sealed_voids']} {ih['inner_shells']}, "
              f"export filled {rep.get('cavities_filled_mm3')} -> {M.volume:.0f} mm3, gate {'PASS' if ok_f else 'FAIL'}")
    except Exception as e:
        traceback.print_exc()
        check("no sealed void", False, str(e))


def t_slender_arm_gap():
    """the slender-arm gap (12 Nov, scout_v4_frame: arms built at 1.7-1.9x the run's beta-4 index): a deck, four 74 mm diagonal arms
    near the minimum member and two 600 N sockets (frame-lite), run end to end at h 2.0 (~10 min). Without the built hold it reads
    0.87 at the beta-4 check and 1.20 built (1.38x), the arms 15 -> 9 cells across. The as-built index must stay within 10 % of the run's own
    beta-4 check (early), the final field's within 10 % of the built, and the part must pass the gate."""
    import tempfile
    from .run import optimise
    try:
        out = tempfile.mkdtemp(prefix="topo_arms_")
        res = optimise(lambda h: _arms_part(h=h), 2.0, ARMS_GEN, out, "arms", settings=Settings(backend="direct", time_limit_s=1800),
                       coarse=False, log=lambda s_: None)
        e, fl, b = (res.get("early") or {}).get("worst"), (res.get("field") or {}).get("worst"), res["strength"]["worst"]
        ok = bool(e and fl and b and b <= 1.10 * e and abs(b - fl) <= 0.10 * fl and res["solid_ok"])
        check("slender arms: built index within 10 % of the beta-4 check and of the final field, gate passes", ok,
              f"early {e} (beta {(res.get('early') or {}).get('beta')}), field {fl}, built {b} (built / early "
              f"{b / e if e and b else None:.2f}), {res['mass_g']} g, solid {'PASS' if res['solid_ok'] else 'FAIL'}, "
              f"{res['total_s']} s")
    except Exception as e:
        traceback.print_exc()
        check("slender arms", False, str(e))


ARMS_GEN = 14.0


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    for t in (t_ke, t_voxelise, t_matvec, t_transfer, t_galerkin, t_mg_vs_direct, t_mma, t_calibration, t_gradients, t_masked_neck, t_gate_ties, t_neck, t_skin_core, t_near_symmetry, t_gate_specks, t_export_cleanup, t_organic_drawn_walls, t_audit_drawn_wall_large, t_gate_drawn_body, t_gate_thin_ok, t_export_mesh, t_export_fallback, t_export_pinch, t_setup_sha, t_setup_reads, t_early_check, t_strength_floating, t_floating_load, t_audit_plate, t_audit_block_measure, t_audit_drawn_thin, t_audit_parity, t_audit_env_neck, t_audit_load_room, t_audit_solid_ok, t_design_hash, t_coarse_setup_fails, t_audit_room_full, t_memory_gate, t_gate_memory, t_build_memory_gate, t_built_hold, t_no_cavity):
        t()
    if not quick:
        t_end_to_end()
        t_slender_arm_gap()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)} / {len(RESULTS)} passed in {time.time() - t0:.0f} s" +
          (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
