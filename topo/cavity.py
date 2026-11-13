"""No sealed void, as part of the optimisation (Spark 13 Nov, the printed foot nodes v8: "the bottom nodes have an empty hole in the
middle ... support there just rattle around"; node 3 2.7, node 4 2.4, node 5 8.9 cm3 of sealed cavity the build kept).

Virtual temperature (Liu et al. 2015, Li & Chen 2016): every design cell's void share is a heat source, void conducts (k 1), solid
nearly insulates (k cavity_ks), and the temperature is 0 on the grid's boundary and on every fixed void (outside the envelope,
the cuts) the boundary reaches. A void the outside reaches drains its heat through void; a sealed one can only lose it through
solid and heats up by about 1 / cavity_ks more. The measure is the p-mean (cavity_p) of the cell temperatures over the design
cells, on the nominal field (the design as the build thresholds it), divided by its value with the whole design void (T0, set once:
the shape of the room, so the limit is scale free). One MMA row holds it at cavity_ratio or under (Settings.no_cavity)."""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi

from .fem import Grid, LOCAL


def heat_ke():
    """8-node trilinear hexahedron conductivity matrix of the unit cube, unit conductivity (scales with h for a cube of side h)."""
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    sgn = np.array(LOCAL, float) * 2.0 - 1.0                   # node signs (-1 / +1) in LOCAL order
    KE = np.zeros((8, 8))
    for xi in g:
        for et in g:
            for ze in g:
                dN = np.stack([sgn[:, 0] * (1 + sgn[:, 1] * et) * (1 + sgn[:, 2] * ze),
                               sgn[:, 1] * (1 + sgn[:, 0] * xi) * (1 + sgn[:, 2] * ze),
                               sgn[:, 2] * (1 + sgn[:, 0] * xi) * (1 + sgn[:, 1] * et)], 0) / 8.0 * 2.0   # d/dx on [0, 1]
                KE += dN.T @ dN / 8.0
    return KE


class Cavity:
    def __init__(self, shape, h, solid, void, design, ks=1e-3, pk=3.0, p=8.0, log=print):
        self.shape, self.h, self.ks, self.pk, self.p = tuple(shape), float(h), float(ks), float(pk), float(p)
        g = Grid(shape)
        nx, ny, nz = shape
        self.design, self.solid, self.void = design, solid, void
        # fixed void the grid boundary reaches through fixed void: its nodes, and the boundary's, are held at 0
        lab, _ = ndi.label(void.reshape(shape))
        edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                         lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        outside = np.isin(lab, edge[edge > 0]).ravel()
        en = g.enodes()
        fixed = np.zeros(g.nn, bool)
        fixed[en[outside].ravel()] = True
        fn = np.zeros(g.nshape, bool)
        fn[0], fn[-1], fn[:, 0], fn[:, -1], fn[:, :, 0], fn[:, :, -1] = True, True, True, True, True, True
        fixed |= fn.ravel()
        cells = ~outside                                            # every other cell takes part
        self.cells = np.flatnonzero(cells)
        self.sealed_fixed_cells = int((void & ~outside).sum())     # fixed void no outside reaches (a drawn chamber): conducts, no source
        free = ~fixed
        self.nfree = int(free.sum())
        num = -np.ones(g.nn, np.int64)
        num[free] = np.arange(self.nfree)
        e = num[en[self.cells]]                                     # (nc, 8), -1 = held node
        self.e = e
        KE = heat_ke() * self.h
        self.KE = KE
        I = np.repeat(e, 8, axis=1).ravel().astype(np.int32); J = np.tile(e, (1, 8)).ravel().astype(np.int32)
        keep = (I >= 0) & (J >= 0)
        self._I, self._J, self._keep = I[keep], J[keep], keep
        self._kev = np.tile(KE.ravel(), len(self.cells))[keep]
        self.dcell = design[self.cells]                             # design cells among the active ones
        self.qn = self.h ** 3 / 8.0                                 # nodal share of a unit source in a cell
        self.T0 = None
        z = np.where(solid, 1.0, 0.0)
        self.T0 = self.measure(z)[0]                               # the whole design void
        log(f"cavity: no-sealed-void constraint on {len(self.cells):,} cells ({self.nfree:,} free nodes), T0 {self.T0:.4g}"
            + (f"; {self.sealed_fixed_cells} fixed-void cells no outside reaches (drawn chamber)" if self.sealed_fixed_cells else ""))

    def _solve(self, rho_c):
        from .fem import _Factor
        k = self.ks + (1.0 - self.ks) * (1.0 - rho_c) ** self.pk
        vals = (np.repeat(k, 64)[self._keep]) * self._kev
        K = sp.csr_matrix((vals, (self._I, self._J)), shape=(self.nfree, self.nfree))
        K.sum_duplicates()
        src = np.where(self.dcell, 1.0 - rho_c, 0.0) * self.qn
        F = np.zeros(self.nfree)
        m = self.e >= 0
        np.add.at(F, self.e[m], np.repeat(src[:, None], 8, 1)[m])
        fac = _Factor(K, log=lambda *_: None)
        T = fac.solve(F[:, None])[:, 0]
        return k, T, fac

    def _cellT(self, T):
        Tn = np.where(self.e >= 0, T[np.maximum(self.e, 0)], 0.0)
        return Tn, Tn.mean(1)

    def measure(self, rho, need_grad=False):
        """(ratio, dratio/drho (nel,) or None): the p-mean of the cell temperatures over the design cells on the density rho (nel,),
        over T0 (T0 itself while it is being set)"""
        rho_c = np.clip(rho[self.cells], 0.0, 1.0)
        k, T, fac = self._solve(rho_c)
        Tn, Te = self._cellT(T)
        Td = np.maximum(Te[self.dcell], 0.0)
        p, N = self.p, max(int(self.dcell.sum()), 1)
        s = float((Td ** p).sum() / N)
        g = s ** (1.0 / p)
        ref = self.T0 if self.T0 else 1.0
        if not need_grad:
            fac.free()
            return g / ref, None
        dTe = np.zeros(len(self.cells))
        dTe[self.dcell] = g ** (1.0 - p) * Td ** (p - 1.0) / N if g > 0 else 0.0
        R = np.zeros(self.nfree)
        m = self.e >= 0
        np.add.at(R, self.e[m], np.repeat(dTe[:, None] / 8.0, 8, 1)[m])
        lam = fac.solve(R[:, None])[:, 0]
        fac.free()
        Ln, _ = self._cellT(lam)
        dk = -(1.0 - self.ks) * self.pk * (1.0 - rho_c) ** (self.pk - 1.0)
        dQ = np.where(self.dcell, -self.qn, 0.0) * Ln.sum(1)
        dK = dk * np.einsum("ea,ab,eb->e", Ln, self.KE, Tn)
        d = np.zeros(len(rho))
        d[self.cells] = (dQ - dK) / ref
        return g / ref, d


def sealed_voids(V, min_cells=1):
    """components of the enclosed void of the voxel mask V (a void no face path reaches the grid's outside from): (labels, sizes)"""
    E = ndi.binary_fill_holes(V) & ~V
    lab, n = ndi.label(E)
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    return lab, sizes
