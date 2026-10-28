"""Structured-grid linear elasticity for voxel topology optimisation.

Grid: nx x ny x nz cubic 8-node hex elements of side h. Nodes are (nx+1, ny+1, nz+1), C order; dof = 3 * node + comp.
Element e = (i * ny + j) * nz + k, also C order, so element arrays reshape to (nx, ny, nz).
Local node a = 4 di + 2 dj + dk for the corner offset (di, dj, dk).

Displacement fields are kept as arrays of shape (k, nx+1, ny+1, nz+1, 3) for k right-hand sides, which turns the
element gather / scatter into 8 strided slices and the element products into one large GEMM (multithreaded BLAS).

Two solvers behind one facade (`Solver`):
  * DirectSolver: Pardiso (pypardiso, SPD Cholesky on the upper triangle, LU fallback) on the active dofs only
    (passive-void elements and the nodes that touch nothing else are removed). One factorisation per support group
    per call, all right-hand sides of that group in one solve. Fastest up to ~0.5 M dofs; memory grows ~ n^(4/3).
  * MGSolver: preconditioned CG with a geometric multigrid V-cycle (Amir, Aage & Lazarov 2014; Ferrari & Sigmund
    2020). Galerkin coarse operators computed element by element: for 2:1 coarsening with trilinear prolongation the
    product P^T K P equals the assembly of P_c^T K_e P_c over the 8 children, so every level is stored as one 24 x 24
    matrix per coarse element and applied matrix-free. Damped Jacobi smoothing, Pardiso / splu on the coarsest level,
    supports as stiff springs (lumped on coarse levels), warm start from the previous design iteration, all
    right-hand sides of a support group iterated together. Memory ~ linear in the grid size.
"""
import os
import time
import glob
import numpy as np
import scipy.sparse as sp

if "PYPARDISO_MKL_RT" not in os.environ:          # cloud box: pypardiso does not find pip's MKL by itself
    _mkl = glob.glob("/usr/local/lib/libmkl_rt.so*") + glob.glob(os.path.join(os.path.dirname(os.__file__), "..",
                                                                              "libmkl_rt.so*"))
    if _mkl:
        os.environ["PYPARDISO_MKL_RT"] = _mkl[0]

LOCAL =[(di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1)]


# ------------------------------------------------------------------------------------------------ element matrices
def hex_ke(h, C):
    """8-node brick of side h, constitutive matrix C (6x6 Voigt, engineering shear), 2x2x2 Gauss.
    Returns KE (24, 24) in LOCAL node order and B at the centre (6, 24)."""
    s = 2.0 * np.array(LOCAL, float) - 1.0
    J = h / 2.0
    g = 1.0 / np.sqrt(3.0)

    def bmat(xi, et, ze):
        dN = np.empty((8, 3))
        dN[:, 0] = s[:, 0] * (1 + s[:, 1] * et) * (1 + s[:, 2] * ze) / 8.0
        dN[:, 1] = s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * ze) / 8.0
        dN[:, 2] = s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * et) / 8.0
        dN /= J
        B = np.zeros((6, 24))
        for i in range(8):
            x, y, z = dN[i]
            B[0, 3 * i] = x
            B[1, 3 * i + 1] = y
            B[2, 3 * i + 2] = z
            B[3, 3 * i] = y; B[3, 3 * i + 1] = x
            B[4, 3 * i + 1] = z; B[4, 3 * i + 2] = y
            B[5, 3 * i] = z; B[5, 3 * i + 2] = x
        return B

    KE = np.zeros((24, 24))
    for a in (-g, g):
        for b in (-g, g):
            for c in (-g, g):
                B = bmat(a, b, c)
                KE += B.T @ C @ B * J ** 3
    return 0.5 * (KE + KE.T), bmat(0.0, 0.0, 0.0)


def child_prolongation():
    """P_c (8, 24, 24): trilinear interpolation from a coarse element's 8 nodes to the 8 nodes of its child c
    (c = 4 ci + 2 cj + ck, the octant), per component."""
    P = np.zeros((8, 24, 24))
    for c, cb in enumerate(LOCAL):
        for a, ab in enumerate(LOCAL):
            t = [(cb[q] + ab[q]) / 2.0 for q in range(3)]
            for A, Ab in enumerate(LOCAL):
                w = 1.0
                for q in range(3):
                    w *= t[q] if Ab[q] == 1 else 1.0 - t[q]
                if w != 0.0:
                    for comp in range(3):
                        P[c, 3 * a + comp, 3 * A + comp] = w
    return P


# ------------------------------------------------------------------------------------------------------- the grid
class Grid:
    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
        nx, ny, nz = self.shape
        self.nshape = (nx + 1, ny + 1, nz + 1)
        self.nel = nx * ny * nz
        self.nn = (nx + 1) * (ny + 1) * (nz + 1)
        self.ndof = 3 * self.nn

    def gather(self, U):
        """U (k, nx+1, ny+1, nz+1, 3) -> element vectors (k, nel, 24)."""
        nx, ny, nz = self.shape
        parts = [U[:, di:di + nx, dj:dj + ny, dk:dk + nz, :] for di, dj, dk in LOCAL]
        return np.stack(parts, axis=4).reshape(U.shape[0], self.nel, 24)

    def scatter(self, r, out=None):
        """element vectors (k, nel, 24) -> nodal sums (k, nx+1, ny+1, nz+1, 3)."""
        nx, ny, nz = self.shape
        k = r.shape[0]
        r6 = r.reshape(k, nx, ny, nz, 8, 3)
        Y = np.zeros((k,) + self.nshape + (3,)) if out is None else out
        for a, (di, dj, dk) in enumerate(LOCAL):
            Y[:, di:di + nx, dj:dj + ny, dk:dk + nz, :] += r6[:, :, :, :, a, :]
        return Y

    def enodes(self):
        nx, ny, nz = self.shape
        ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
        nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        return np.stack([nid(ex + di, ey + dj, ez + dk) for di, dj, dk in LOCAL], 1)

    def edof(self):
        en = self.enodes()
        return (en[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(self.nel, 24)

    def to_field(self, v, k=None):
        """flat dof vector(s) (ndof,) or (ndof, k) -> (k, nx+1, ny+1, nz+1, 3)."""
        v = np.asarray(v)
        if v.ndim == 1:
            return v.reshape((1,) + self.nshape + (3,))
        return np.ascontiguousarray(v.T).reshape((v.shape[1],) + self.nshape + (3,))

    def to_flat(self, U):
        """(k, nx+1, ny+1, nz+1, 3) -> (ndof, k)."""
        return U.reshape(U.shape[0], -1).T


def prolong(Uc):
    """trilinear prolongation along the 3 spatial axes of (k, X, Y, Z, 3): X -> 2X-1 nodes."""
    U = Uc
    for ax in (1, 2, 3):
        n = U.shape[ax]
        shp = list(U.shape); shp[ax] = 2 * n - 1
        F = np.empty(shp)
        ev = [slice(None)] * U.ndim; ev[ax] = slice(0, None, 2)
        od = [slice(None)] * U.ndim; od[ax] = slice(1, None, 2)
        lo = [slice(None)] * U.ndim; lo[ax] = slice(0, n - 1)
        hi = [slice(None)] * U.ndim; hi[ax] = slice(1, n)
        F[tuple(ev)] = U
        F[tuple(od)] = 0.5 * (U[tuple(lo)] + U[tuple(hi)])
        U = F
    return U


def restrict(F):
    """transpose of prolong: (k, 2X-1, ...) -> (k, X, ...)."""
    C = F
    for ax in (1, 2, 3):
        n = C.shape[ax]
        nc = (n + 1) // 2
        ev = [slice(None)] * C.ndim; ev[ax] = slice(0, None, 2)
        od = [slice(None)] * C.ndim; od[ax] = slice(1, None, 2)
        R = C[tuple(ev)].copy()
        odd = 0.5 * C[tuple(od)]
        a = [slice(None)] * C.ndim; a[ax] = slice(0, nc - 1)
        b = [slice(None)] * C.ndim; b[ax] = slice(1, nc)
        R[tuple(a)] += odd
        R[tuple(b)] += odd
        C = R
    return C


def _octants(arr, shape):
    """element-indexed array (nel, ...) on a grid `shape` (all even) -> (nel/8, 8, ...) grouped by coarse element,
    octant order = LOCAL."""
    nx, ny, nz = shape
    tail = arr.shape[1:]
    a = arr.reshape((nx // 2, 2, ny // 2, 2, nz // 2, 2) + tail)
    a = a.transpose((0, 2, 4, 1, 3, 5) + tuple(range(6, 6 + len(tail))))
    return a.reshape((nx * ny * nz // 8, 8) + tail)


def _pardiso_factor_solve(K, B, mtype=2, log=print):
    """solve K X = B with pypardiso; K full symmetric CSR. Returns X. Cholesky on the upper triangle, LU fallback."""
    import pypardiso
    from pypardiso.pardiso_wrapper import PyPardisoError
    for mt in ((2, 11) if mtype == 2 else (mtype,)):
        Kf = sp.triu(K, format="csr") if mt == 2 else K
        ps = pypardiso.PyPardisoSolver()
        ps.set_matrix_type(mt)
        try:
            os.environ.setdefault("PYPARDISO_MKL_RT", ps.libmkl._name)
        except Exception:
            pass
        try:
            X = ps.solve(Kf, np.ascontiguousarray(B))
            ps.free_memory(everything=True)
            return X.reshape(B.shape)
        except PyPardisoError as e:
            ps.free_memory(everything=True)
            if mt == 11:
                raise
            log(f"fem: Cholesky failed ({e}); LU fallback")


def _sparse_solve(K, B, log=print):
    """direct solve with the best available backend: pypardiso, else scipy splu."""
    try:
        return _pardiso_factor_solve(K, B, log=log)
    except ImportError:
        from scipy.sparse.linalg import splu
        lu = splu(K.tocsc())
        return lu.solve(np.asarray(B, float))


class _CoarseDirect:
    """factorised coarsest-level operator, reused for every V-cycle of one solve."""

    def __init__(self, K, log=print):
        self.K = K.tocsr()
        self.n = K.shape[0]
        self.kind = None
        try:
            import pypardiso
            self.ps = pypardiso.PyPardisoSolver()
            self.ps.set_matrix_type(2)
            self.Kt = sp.triu(self.K, format="csr")
            self.ps.factorize(self.Kt)
            self.kind = "pardiso"
        except Exception:
            from scipy.sparse.linalg import splu
            self.lu = splu(self.K.tocsc())
            self.kind = "splu"

    def solve(self, B):
        if self.kind == "pardiso":
            return self.ps.solve(self.Kt, np.ascontiguousarray(B)).reshape(B.shape)
        return self.lu.solve(np.asarray(B))

    def free(self):
        if self.kind == "pardiso":
            self.ps.free_memory(everything=True)


# ------------------------------------------------------------------------------------------------------ the solver
class Solver:
    """K(E) u = f for every load case, grouped by support. E: element moduli (nel,) relative to the material
    (1 = solid). fixed: {support name: bool (nn, 3) mask of fixed dofs}. Returns U as (ndof, ncase) flat array.

    backend: "direct", "mg" or "auto" (direct when the active system is small enough for the RAM budget)."""

    EMIN_MG = 1e-6

    def __init__(self, shape, h, C, fixed, active=None, backend="auto", mem_gb=None, log=print, mg_tol=1e-5,
                 mg_maxit=300, nu_smooth=2, omega=0.6, coarse_dofs=40000):
        self.grid = Grid(shape)
        self.h = h
        self.KE, self.B0 = hex_ke(h, C)
        self.C = C
        self.fixed = fixed
        self.log = log
        self.active = np.ones(self.grid.nel, bool) if active is None else np.asarray(active, bool)
        g = self.grid
        # nodes that belong to at least one active element
        used = np.zeros(g.nshape, bool)
        A = self.active.reshape(g.shape)
        nx, ny, nz = g.shape
        for di, dj, dk in LOCAL:
            used[di:di + nx, dj:dj + ny, dk:dk + nz] |= A
        self.used_nodes = used
        self.n_active_dofs = int(used.sum()) * 3
        if mem_gb is None:
            mem_gb = _machine_mem_gb() * 0.75
        self.mem_gb = mem_gb
        if backend == "auto":
            backend = "direct" if self.direct_mem_estimate() < 0.6 * mem_gb else "mg"
        self.backend = backend
        self.mg_tol, self.mg_maxit, self.nu, self.omega, self.coarse_dofs = mg_tol, mg_maxit, nu_smooth, omega, coarse_dofs
        self.warm = {}
        self.stats = []
        log(f"fem: grid {g.shape} = {g.nel:,} elements, {self.n_active_dofs:,} active dofs; backend {backend} "
            f"(direct estimate {self.direct_mem_estimate():.1f} GB, budget {mem_gb:.1f} GB)")
        if backend == "direct":
            self._setup_direct()
        else:
            self._setup_mg()

    # ---- memory model (E): Pardiso 3D hex elasticity, fitted to the old engine's cloud OOM points (6 GB at ~300k
    # dofs) with nested-dissection n^(4/3) scaling
    def direct_mem_estimate(self):
        n = max(self.n_active_dofs, 1)
        return 6.0 * (n / 4.5e5) ** (4.0 / 3.0)

    # ------------------------------------------------------------------------------------------------ direct
    def _setup_direct(self):
        g = self.grid
        act = np.where(self.active)[0]
        edof = g.edof()[act]
        self.d_act = act
        self.d_edof = edof
        used = np.zeros(g.ndof, bool)
        used[edof.ravel()] = True
        self.d_map = -np.ones(g.ndof, np.int64)
        idx = np.where(used)[0]
        self.d_map[idx] = np.arange(len(idx))
        self.d_used = idx
        le = self.d_map[edof].astype(np.int32)                      # local dof numbers (nact, 24)
        iK = np.repeat(le, 24, axis=1).ravel()
        jK = np.tile(le, (1, 24)).ravel()
        n = len(idx)
        # CSR pattern once; data via bincount every call
        pat = sp.coo_matrix((np.ones(len(iK), np.float32), (iK, jK)), shape=(n, n)).tocsr()
        pat.sum_duplicates()
        pat.sort_indices()
        self.d_indptr, self.d_indices = pat.indptr, pat.indices
        # position of every (i, j) entry in the CSR data array: CSR keys row * n + col are sorted, so a search finds it
        pos = np.empty(len(iK), np.int64)
        keys = self._csr_keys(n)
        CH = 20_000_000
        for a in range(0, len(iK), CH):
            pos[a:a + CH] = np.searchsorted(keys, iK[a:a + CH].astype(np.int64) * n + jK[a:a + CH])
        self.d_pos = pos
        self.d_n = n
        self._keys = None
        del iK, jK, pat, keys
        self.d_free = {}
        for name, fx in self.fixed.items():
            fl = fx.reshape(-1)
            keep = ~fl[idx]
            self.d_free[name] = np.where(keep)[0]

    def _csr_keys(self, n):
        if getattr(self, "_keys", None) is None:
            rows = np.repeat(np.arange(len(self.d_indptr) - 1, dtype=np.int64), np.diff(self.d_indptr))
            self._keys = rows * n + self.d_indices.astype(np.int64)
        return self._keys

    def _assemble_direct(self, E):
        vals = (E[self.d_act][:, None] * self.KE.ravel()[None, :]).ravel()
        data = np.bincount(self.d_pos, weights=vals, minlength=len(self.d_indices))
        return sp.csr_matrix((data, self.d_indices, self.d_indptr), shape=(self.d_n, self.d_n))

    def _solve_direct(self, E, F, groups):
        U = np.zeros_like(F)
        K = self._assemble_direct(E)
        for sup, cols in groups.items():
            fr = self.d_free[sup]
            Kf = K[fr][:, fr].tocsr()
            gdof = self.d_used[fr]
            X = _sparse_solve(Kf, F[gdof][:, cols], log=self.log)
            U[np.ix_(gdof, cols)] = X.reshape(len(fr), len(cols))
            del Kf
        return U

    # ---------------------------------------------------------------------------------------------------- MG
    def _setup_mg(self):
        g = self.grid
        shp = np.array(g.shape)
        L = 1
        while True:
            if np.any(shp % 2) or np.any(shp // 2 < 2):
                break
            ncd = 3 * np.prod(shp + 1)
            if ncd <= self.coarse_dofs:
                break
            shp = shp // 2
            L += 1
        self.levels = L
        self.mg_shapes = [tuple(np.array(g.shape) // (2 ** l)) for l in range(L)]
        self.P8 = child_prolongation()
        self.G8 = np.einsum("cai,ab,cbj->cij", self.P8, self.KE, self.P8).reshape(8, 576)
        # dead nodes: touch no active element -> fixed (removes floppy void modes)
        self.dead = ~self.used_nodes
        # fine level over the active elements only when most of the box is passive void
        self.act_idx = np.where(self.active)[0]
        self.sparse_fine = len(self.act_idx) < 0.6 * g.nel
        if self.sparse_fine:
            self.edof_act = g.edof()[self.act_idx]
        self.kpen = 1e4 * float(np.abs(np.diag(self.KE)).max())
        cd = 3 * np.prod(np.array(self.mg_shapes[-1]) + 1)
        self.log(f"fem: multigrid {L} levels {self.mg_shapes}, coarsest {cd:,} dofs (direct)")
        if L == 1:
            self.log("fem: WARNING the grid does not coarsen (odd sizes); pad it to a multiple of 2^k for multigrid")

    def _mg_hierarchy(self, E, fixed_mask):
        """per-level operators for one support group: level 0 matrix-free with E * KE, coarse levels element matrices."""
        g = self.grid
        pen0 = np.zeros(g.nshape + (3,))
        pen0[fixed_mask.reshape(g.nshape + (3,))] = self.kpen
        pen0[self.dead] = self.kpen
        levels = [dict(grid=g, E=E, Ke=None, pen=pen0)]
        Kprev = None
        for l in range(1, self.levels):
            shp_f = self.mg_shapes[l - 1]
            gl = Grid(self.mg_shapes[l])
            if l == 1:
                E8 = _octants(E, shp_f)                                    # (nel1, 8)
                Ke = (E8 @ self.G8).reshape(-1, 24, 24)
            else:
                K8 = _octants(Kprev, shp_f)                                # (nel_l, 8, 24, 24)
                Ke = np.zeros((gl.nel, 24, 24))
                for c in range(8):
                    Ke += np.matmul(np.matmul(self.P8[c].T, K8[:, c]), self.P8[c])
                del K8
            pen = restrict(levels[-1]["pen"][None])[0]
            levels.append(dict(grid=gl, E=None, Ke=Ke, pen=pen))
            Kprev = Ke
        for l, lv in enumerate(levels):
            lv["diag"] = self._diag(lv)
            if l < len(levels) - 1 or self.levels == 1:
                lv["omega"] = min(self.omega, 4.0 / (3.0 * self._lam_max(lv, l)))
        # coarsest direct
        last = levels[-1]
        if self.levels > 1:
            last["coarse"] = _CoarseDirect(self._assemble_level(last), log=self.log)
        return levels

    def _lam_max(self, lv, l, its=12):
        """largest eigenvalue of D^-1 K on one level (power iteration, warm-started from the last call, +10 %).
        Damped Jacobi only smooths when omega * lam_max < 2; for 3D hex elasticity lam_max is 4 to 6, so a fixed
        omega 0.6 diverges on the high modes and the V-cycle stops being a valid CG preconditioner."""
        g = lv["grid"]
        key = ("lam", l)
        x = self.warm.get(key)
        if x is None or x.shape != (1,) + g.nshape + (3,):
            x = np.random.default_rng(l).normal(size=(1,) + g.nshape + (3,))
        lam = 1.0
        for _ in range(its):
            y = self._apply(lv, x) / lv["diag"][None]
            lam = np.sqrt((y * y).sum() / max((x * x).sum(), 1e-300))
            x = y / max(np.abs(y).max(), 1e-300)
        self.warm[key] = x
        return 1.1 * lam

    def _diag(self, lv):
        g = lv["grid"]
        if lv["Ke"] is None:
            if self.sparse_fine:
                w = (lv["E"][self.act_idx][:, None] * np.diag(self.KE)[None, :]).ravel()
                D = np.bincount(self.edof_act.ravel(), weights=w, minlength=g.ndof).reshape(g.nshape + (3,))
                return D + lv["pen"]
            d = lv["E"][:, None] * np.diag(self.KE)[None, :]
        else:
            d = np.einsum("eii->ei", lv["Ke"])
        D = g.scatter(d[None])[0] + lv["pen"]
        return D

    def _assemble_level(self, lv):
        g = lv["grid"]
        edof = g.edof()
        iK = np.repeat(edof, 24, axis=1).ravel()
        jK = np.tile(edof, (1, 24)).ravel()
        K = sp.coo_matrix((lv["Ke"].reshape(-1), (iK, jK)), shape=(g.ndof, g.ndof)).tocsr()
        K = K + sp.diags(lv["pen"].reshape(-1))
        return K.tocsr()

    def _apply(self, lv, U):
        g = lv["grid"]
        k = U.shape[0]
        if lv["Ke"] is None and self.sparse_fine:
            Uf = U.reshape(k, -1)
            ue = Uf[:, self.edof_act]                                      # (k, nact, 24)
            r = (ue.reshape(-1, 24) @ self.KE).reshape(k, len(self.act_idx), 24)
            r *= lv["E"][self.act_idx][None, :, None]
            Y = np.empty_like(Uf)
            ed = self.edof_act.ravel()
            for q in range(k):
                Y[q] = np.bincount(ed, weights=r[q].ravel(), minlength=g.ndof)
            Y = Y.reshape(U.shape)
            Y += lv["pen"][None] * U
            return Y
        ue = g.gather(U)                                                   # (k, nel, 24)
        if lv["Ke"] is None:
            r = (ue.reshape(-1, 24) @ self.KE).reshape(k, g.nel, 24)
            r *= lv["E"][None, :, None]
        else:
            r = np.matmul(lv["Ke"][None], ue[..., None])[..., 0] if k == 1 else \
                np.einsum("eij,kej->kei", lv["Ke"], ue, optimize=True)
        Y = g.scatter(r)
        Y += lv["pen"][None] * U
        return Y

    def _vcycle(self, levels, l, B):
        lv = levels[l]
        if l == len(levels) - 1 and "coarse" in lv:
            g = lv["grid"]
            X = lv["coarse"].solve(g.to_flat(B))
            return g.to_field(X)
        Dinv = 1.0 / lv["diag"][None]
        w = lv["omega"]
        X = w * Dinv * B
        for _ in range(self.nu - 1):
            X += w * Dinv * (B - self._apply(lv, X))
        if l == len(levels) - 1:
            return X
        R = B - self._apply(lv, X)
        Ec = self._vcycle(levels, l + 1, restrict(R))
        X += prolong(Ec)
        for _ in range(self.nu):
            X += w * Dinv * (B - self._apply(lv, X))
        return X

    def _solve_mg(self, E, F, groups, keys):
        g = self.grid
        E = np.where(self.active, np.maximum(E, self.EMIN_MG), 0.0)
        U = np.zeros_like(F)
        for sup, cols in groups.items():
            fixed = self.fixed[sup]
            levels = self._mg_hierarchy(E, fixed)
            fmask = (fixed.reshape(g.nshape + (3,)) | self.dead[..., None])[None]
            B = g.to_field(F[:, cols]) * ~fmask
            X = np.zeros_like(B)
            for q, c in enumerate(cols):
                w = self.warm.get(keys[c])
                if w is not None:
                    X[q] = w
            X *= ~fmask
            t0 = time.time()
            its = self._pcg(levels, B, X, ~fmask)
            self.stats.append((sup, its, time.time() - t0))
            for q, c in enumerate(cols):
                self.warm[keys[c]] = X[q].copy()
            U[:, cols] = g.to_flat(X)
            if "coarse" in levels[-1]:
                levels[-1]["coarse"].free()
            del levels
        return U

    def _pcg(self, levels, B, X, free):
        lv = levels[0]
        k = B.shape[0]
        dot = lambda a, b: (a * b).reshape(a.shape[0], -1).sum(1)
        bn = np.sqrt(dot(B, B))
        bn[bn == 0] = 1.0
        R = (B - self._apply(lv, X)) * free                               # supports eliminated exactly
        Z = self._vcycle(levels, 0, R) * free
        P = Z.copy()
        rz = dot(R, Z)
        active = np.ones(k, bool)
        its = 0
        for it in range(self.mg_maxit):
            rel = np.sqrt(dot(R, R)) / bn
            active = rel > self.mg_tol
            if not active.any():
                break
            its = it + 1
            AP = self._apply(lv, P) * free
            pap = dot(P, AP)
            alpha = np.where(active & (pap > 0), rz / np.where(pap > 0, pap, 1.0), 0.0)
            sh = (k,) + (1,) * (B.ndim - 1)
            X += alpha.reshape(sh) * P
            R -= alpha.reshape(sh) * AP
            Z = self._vcycle(levels, 0, R) * free
            rz_new = dot(R, Z)
            beta = np.where(active, rz_new / np.where(rz != 0, rz, 1.0), 0.0)
            P = Z + beta.reshape(sh) * P
            rz = rz_new
        else:
            self.log(f"fem: WARNING multigrid CG hit {self.mg_maxit} iterations (rel residual {rel.max():.2e})")
        return its

    # ---------------------------------------------------------------------------------------------- facade
    def solve(self, E, F, supports, keys=None):
        """E (nel,) element moduli (relative); F (ndof, k) loads; supports: list of k support names;
        keys: k hashable names for warm starts (multigrid). Returns U (ndof, k)."""
        F = np.asarray(F, float)
        if F.ndim == 1:
            F = F[:, None]
        groups = {}
        for c, s in enumerate(supports):
            groups.setdefault(s, []).append(c)
        if self.backend == "direct":
            return self._solve_direct(np.asarray(E, float), F, groups)
        keys = list(range(F.shape[1])) if keys is None else keys
        return self._solve_mg(np.asarray(E, float), F, groups, keys)

    def element_energy(self, U, fe=None):
        """u_e^T KE u_e for every element and case: U (ndof, k) -> (k, nel). With fe (k, 24) or None rows, also
        returns u_e . fe (k, nel) for design-dependent body forces. Inactive elements read 0."""
        g = self.grid
        U = np.asarray(U)
        if U.ndim == 1:
            U = U[:, None]
        k = U.shape[1]
        if not hasattr(self, "_edof_all_act"):
            self._act_idx = np.where(self.active)[0]
            self._edof_all_act = g.edof()[self._act_idx]
        out = np.zeros((k, g.nel))
        fo = np.zeros((k, g.nel)) if fe is not None else None
        for q in range(k):
            ue = U[:, q][self._edof_all_act]                               # (nact, 24)
            out[q, self._act_idx] = np.einsum("ei,ei->e", ue @ self.KE, ue)
            if fe is not None and fe[q] is not None:
                fo[q, self._act_idx] = ue @ fe[q]
        return (out, fo) if fe is not None else out


def _machine_mem_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 8.0
