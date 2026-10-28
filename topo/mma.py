"""Method of Moving Asymptotes (Svanberg 1987, asymptote update and regularisation of Svanberg 2007), dual solve.

min f0(x)  s.t.  f_i(x) <= 0 (i = 1..m),  xmin <= x <= xmax.  m is small (1 to a few), n up to millions.
The separable convex subproblem is solved through its concave dual in the m multipliers (L-BFGS-B on m variables),
with elastic variables y_i (c = 1000, d = 1) so an infeasible start stays solvable. State (x history, asymptotes)
is exposed for checkpoints.
"""
import numpy as np
from scipy.optimize import minimize


class MMA:
    def __init__(self, n, m, xmin=0.0, xmax=1.0, move=0.1, asyinit=0.5, asyincr=1.2, asydecr=0.7, c=1000.0):
        self.n, self.m = n, m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.move, self.asyinit, self.asyincr, self.asydecr = move, asyinit, asyincr, asydecr
        self.c = np.full(m, c)
        self.d = np.ones(m)
        self.reset()

    def reset(self):
        """forget the history (asymptotes restart wide): call after a continuation step."""
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None
        self.k = 0

    def state(self):
        z = np.zeros(0)
        return dict(xold1=z if self.xold1 is None else self.xold1, xold2=z if self.xold2 is None else self.xold2,
                    low=z if self.low is None else self.low, upp=z if self.upp is None else self.upp, k=self.k)

    def load_state(self, st):
        g = lambda a: None if a.size == 0 else a.copy()
        self.xold1, self.xold2, self.low, self.upp = g(st["xold1"]), g(st["xold2"]), g(st["low"]), g(st["upp"])
        self.k = int(st["k"])

    def step(self, x, f0, df0, fval, dfdx, active=None):
        """one MMA update. x (n,), f0 scalar, df0 (n,), fval (m,), dfdx (m, n). active: bool (n,) of free variables
        (others stay put). Returns the new x."""
        x = np.asarray(x, float)
        fval = np.atleast_1d(np.asarray(fval, float))
        dfdx = np.atleast_2d(np.asarray(dfdx, float))
        n = self.n
        act = np.ones(n, bool) if active is None else active
        dx = self.xmax - self.xmin
        self.k += 1
        if self.k <= 2 or self.xold2 is None or self.low is None:
            low = x - self.asyinit * dx
            upp = x + self.asyinit * dx
        else:
            zzz = (x - self.xold1) * (self.xold1 - self.xold2)
            gam = np.ones(n)
            gam[zzz > 0] = self.asyincr
            gam[zzz < 0] = self.asydecr
            low = x - gam * (self.xold1 - self.low)
            upp = x + gam * (self.upp - self.xold1)
            low = np.clip(low, x - 10.0 * dx, x - 0.01 * dx)
            upp = np.clip(upp, x + 0.01 * dx, x + 10.0 * dx)
        alpha = np.maximum.reduce([self.xmin, low + 0.1 * (x - low), x - self.move * dx])
        beta = np.minimum.reduce([self.xmax, upp - 0.1 * (upp - x), x + self.move * dx])
        ux = upp - x
        xl = x - low
        eps = 1e-5 / np.maximum(dx, 1e-12)
        dp, dm = np.maximum(df0, 0.0), np.maximum(-df0, 0.0)
        p0 = ux ** 2 * (1.001 * dp + 0.001 * dm + eps)
        q0 = xl ** 2 * (0.001 * dp + 1.001 * dm + eps)
        Dp, Dm = np.maximum(dfdx, 0.0), np.maximum(-dfdx, 0.0)
        P = ux[None] ** 2 * (1.001 * Dp + 0.001 * Dm + eps[None])
        Q = xl[None] ** 2 * (0.001 * Dp + 1.001 * Dm + eps[None])
        # only free variables enter the subproblem (fixed ones contribute nothing to the approximation)
        ia = np.where(act)[0]
        b = (P[:, ia] / ux[None, ia] + Q[:, ia] / xl[None, ia]).sum(1) - fval
        p0a, q0a, Pa, Qa = p0[ia], q0[ia], P[:, ia], Q[:, ia]
        la, ua, al, be = low[ia], upp[ia], alpha[ia], beta[ia]

        def xsol(lam):
            pj = p0a + lam @ Pa
            qj = q0a + lam @ Qa
            sp_, sq = np.sqrt(np.maximum(pj, 0.0)), np.sqrt(np.maximum(qj, 0.0))
            xs = (sp_ * la + sq * ua) / np.maximum(sp_ + sq, 1e-300)
            return np.clip(xs, al, be), pj, qj

        def negdual(lam):
            xs, pj, qj = xsol(lam)
            y = np.maximum(0.0, (lam - self.c) / self.d)
            W = (pj / (ua - xs) + qj / (xs - la)).sum() - lam @ b + (self.c * y + 0.5 * self.d * y * y - lam * y).sum()
            g = (Pa / (ua - xs)[None] + Qa / (xs - la)[None]).sum(1) - b - y
            return -W, -g

        lam0 = np.ones(self.m)
        res = minimize(negdual, lam0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * self.m,
                       options=dict(maxiter=500, ftol=1e-14, gtol=1e-10))
        xs, _, _ = xsol(res.x)
        xnew = x.copy()
        xnew[ia] = xs
        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = x.copy()
        self.low, self.upp = low, upp
        self.lam = res.x
        return xnew
