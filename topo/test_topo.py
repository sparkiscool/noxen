"""Self-test of nox/topo. Run from nox/:  python3 -m topo.test_topo [--quick]
Each test prints PASS / FAIL with its numbers; exit code 1 if any fails. ~2-4 min on a 4 vCPU box (--quick skips the
end-to-end run)."""
import sys
import time
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
        check("end to end: holder at h 2.0 builds one watertight body", res["bodies"]["bodies"] == 1 and
              res["watertight"], f"{res['mass_g']} g, solid check {'PASS' if res['solid_ok'] else 'FAIL'}, "
                                 f"worst {res['strength']['worst']}, {res['total_s']} s, files in {out}")
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


def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    for t in (t_ke, t_voxelise, t_matvec, t_transfer, t_galerkin, t_mg_vs_direct, t_mma, t_calibration, t_gradients, t_masked_neck, t_gate_ties, t_neck, t_skin_core, t_near_symmetry, t_gate_specks, t_audit_plate, t_audit_block_measure, t_design_hash):
        t()
    if not quick:
        t_end_to_end()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)} / {len(RESULTS)} passed in {time.time() - t0:.0f} s" +
          (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
