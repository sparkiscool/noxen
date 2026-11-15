"""Force vs stroke of the Overseer boot candidates (10 Nov design session; make.py imports column_curve and drop for its preview and numbers) (E, simple models) and the 9.7 J drop balance against the cage (87 N/mm in series).
Column: discrete elastica (30 segments, rotational + axial springs, clamped ends with an outward bow), displacement controlled,
energy minimised (L-BFGS); F = dU/d(delta). TPU 95A HF: E 8 MPa (Bambu TDS: 9.8 X-Y, 7.4 Z; the ribs bend about a horizontal axis
with the stress along z). 68D TPU for AMS: 600 (Z) .. 1190 (X-Y) MPa."""
import numpy as np, sys
from scipy.optimize import minimize

def column_curve(E, t, w, L, bow, n=30, dmax_frac=0.7, steps=35):
    """F(delta) of one rib: thickness t (radial), width w (tangential), length L, initial outward bow (mid-span, mm)"""
    EI, EA, l0 = E * w * t ** 3 / 12.0, E * w * t, L / n
    s = np.linspace(0, L, n + 1)
    x0 = bow * np.sin(np.pi * s / L)                                  # initial shape (half sine bow), z along the column
    P0 = np.c_[x0, s]
    th0 = np.arctan2(np.diff(P0[:, 0]), np.diff(P0[:, 1]))          # segment angles from vertical
    kb = EI / l0; ka = EA / l0; kc = 50 * kb                           # clamp penalty springs
    def energy(q, dz):
        P = np.vstack([[0.0, 0.0], q.reshape(-1, 2), [x0[-1] + 0.0, L - dz]])
        d = np.diff(P, axis=0); l = np.hypot(d[:, 0], d[:, 1]); th = np.arctan2(d[:, 0], d[:, 1])
        U = 0.5 * ka * np.sum((l - l0) ** 2) + 0.5 * kb * np.sum((np.diff(th) - np.diff(th0)) ** 2)
        U += 0.5 * kc * ((th[0] - th0[0]) ** 2 + (th[-1] - th0[-1]) ** 2)
        return U
    q = P0[1:-1].ravel().copy()
    out = [(0.0, 0.0, 0.0, 0.0)]
    Us = [0.0]; ds = np.linspace(0, dmax_frac * L, steps)[1:]
    for dz in ds:
        r = minimize(energy, q, args=(dz,), method="L-BFGS-B", options=dict(maxiter=3000, ftol=1e-14, gtol=1e-9))
        q = r.x; Us.append(r.fun)
        P = np.vstack([[0.0, 0.0], q.reshape(-1, 2), [x0[-1], L - dz]])
        d = np.diff(P, axis=0); th = np.arctan2(d[:, 0], d[:, 1]); kap = np.abs(np.diff(th)) / l0
        out.append((dz, 0.0, float(P[:, 0].max()), float(kap.max() * t / 2)))       # stroke, F (filled below), max outward excursion, max fibre strain
    Us = np.array(Us); dd = np.r_[0.0, ds]
    F = np.gradient(Us, dd)
    return np.array([(dd[i], F[i], out[i][2], out[i][3]) for i in range(len(dd))])

def honeycomb_curve(E_s, sig_y, t, l, W, D, H, steps=200, hw=None):
    """in-plane regular hex honeycomb (side l, wall t) crushed along the cell vertices (Gibson-Ashby X2): elastic E* = 2.3 E_s (t/l)^3
    (regular hex, ~ the same both ways), plastic plateau sigma_pl = sig_y (t/l)^2 / (2 cos^2 30) with 10 % hardening over the
    stroke, densification from eps_D = 1 - 1.4 rho (rho = 2/sqrt3 t/l) as a steep ramp. Loaded face W x D, height H (the stroke axis).
    hw: the face outline's half-width over the height, hw(y) for y in [-H/2, H/2] (v3's rounded rectangle; None: the W box). The
    narrow slices (the rounded ends) fold first, so the force at a crushed fraction f = e / eps_D carries the f-quantile of the
    slice widths, ascending: a slightly rising plateau that starts under the box's and ends on it (E)."""
    rho = 2 / np.sqrt(3) * t / l; A = W * D
    Es = 2.3 * E_s * (t / l) ** 3; s_pl = sig_y * (t / l) ** 2 / 1.5; eD = 1 - 1.4 * rho
    x = np.linspace(0, H * 0.9, steps); e = x / H
    s_el = Es * e; s_pt = s_pl * (1 + 0.1 * e / eD)
    s = np.minimum(s_el, s_pt)
    dens = np.where(e > eD, s_pl * ((e - eD) / (1 - eD)) ** 2 * 12, 0.0)          # the densification ramp (E)
    if hw is None:
        w = np.full_like(x, W)
    else:
        ws = np.sort(2 * hw(np.linspace(-H / 2, H / 2, 801)))
        w = np.interp(np.clip(e / eD, 0, 1), np.linspace(0, 1, len(ws)), ws)
    return np.c_[x, D * w * (s + dens), np.full_like(x, rho), np.full_like(x, eD * H)]


def drop(F_of_x, x_max, E_drop=9700.0, k_cage=87.0, k_end=600.0):
    """energy balance: boot F(x) in series with the cage spring; past x_max the boot is a k_end N/mm stop. returns peak N, boot stroke, cage mm"""
    x = np.linspace(0, x_max, 400); F = F_of_x(x); Ub = np.concatenate([[0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(x))])
    for i in range(len(x)):
        if Ub[i] + 0.5 * F[i] ** 2 / k_cage >= E_drop:
            return F[i], x[i], F[i] / k_cage
    Fe = F[-1]; Ur = E_drop - Ub[-1]
    # past the stroke: F = Fe + k_end y, energy Ub + Fe y + 0.5 k y^2 + F^2/(2 k_cage) = E
    for y in np.linspace(0, 40, 4000):
        Fy = Fe + k_end * y
        if Ub[-1] + Fe * y + 0.5 * k_end * y ** 2 + 0.5 * Fy ** 2 / k_cage >= E_drop:
            return Fy, x_max + y, Fy / k_cage
    return np.nan, np.nan, np.nan

def candidates(E95=8.0):
    """the table in the make.py docstring: every candidate through the 9.7 J drop balance"""
    E95 = float(E95)
    main_table(E95)


def main_table(E95):
    print("== A. plain O48 x 15 sole, 20 % gyroid (lattice 12.6 under 3+3 solid layers): sigma 0.2 MPa at 0.25 strain rising to 0.5 at 0.5, densifies past 0.55 (E, TPU gyroid tests)")
    A = 1810.0; hl = 12.6
    def FA(x):
        e = x / hl; s = np.where(e < 0.25, 0.8 * e, 0.2 + 1.2 * (e - 0.25)); return A * s
    print("   drop:", drop(FA, 0.55 * hl, k_end=800.0))
    print("== B. buckling columns, TPU 95A E %.0f: n ribs t x w x L, bow" % E95)
    for (n, t, w, L, bow) in [(12, 5.0, 8.0, 24.0, 0.8), (12, 5.0, 8.0, 30.0, 0.8), (12, 5.0, 8.0, 32.0, 0.8), (12, 5.5, 8.0, 32.0, 0.8), (12, 6.0, 8.0, 32.0, 0.8), (10, 6.0, 9.0, 32.0, 1.0), (12, 5.0, 8.0, 32.0, 2.0)]:
        c = column_curve(E95, t, w, L, bow)
        F = n * c[:, 1]; x = c[:, 0]
        i_pl = (x > 0.1 * L) & (x < 0.6 * L)
        xmax = 0.6 * L
        pk, st, cg = drop(lambda q: np.interp(q, x, F), xmax)
        print(f"   n{n} t{t} w{w} L{L} bow{bow}: F at 5/10/20/40/60 % of L = {np.interp([0.05,0.1,0.2,0.4,0.6], x/L, F).round(0)}  plateau mean {F[i_pl].mean():.0f} N, "
              f"max outward {c[-1,2]:.1f} mm, fibre strain at 60 % {np.interp(0.6, x/L, c[:,3]):.2f}; drop: peak {pk:.0f} N, boot {st:.1f} mm, cage {cg:.1f} mm")
    print("== C. bellows / thin shells, E 8: a Belleville disc De 44 Di 20 t 1.6 h0 2 -> F_flat ~ 4E/(1-nu2) t^4/(K1 De^2) sqrt2")
    print("   ", 4 * 8 / 0.77 * 1.6 ** 4 / (0.7 * 44 ** 2) * 1.414, "N per disc: a feather at E 8; at t 5 it is a solid cone, not a bellows")
    print("== C2. in-plane honeycomb, E 8, plateau 0.22 E (t/l)^3 over 1500 mm2; need 0.33 MPa -> t/l", (0.33 / (0.22 * 8)) ** (1 / 3), "-> rel density", 1.15 * (0.33 / (0.22 * 8)) ** (1 / 3))
    print("== D. steel spring, linear k: peak = 2 E / s. s 25 ->", 2 * 9700 / 25, "N (before the cage share); with the cage:")
    for k in (20.0, 30.0, 45.0):
        print("    k", k, drop(lambda q: k * q, 30.0))
    print("== E. 68D in-plane honeycomb crush puck (the design for the TPU for AMS spool): 50 x 20 face, 40 tall, wall 2.0, side 9.0 (3 x 3 cells); sig_y 15 MPa, E 1190")
    hw_round = lambda y: 25.0 - 8.0 + np.sqrt(np.maximum(0.0, 64.0 - np.maximum(0.0, np.abs(y) - 12.0) ** 2))   # v3: the face a rounded rectangle, R 8
    for name, hw in (("v2 box", None), ("v3 rounded R 8", hw_round)):
        for sy in (9.0, 15.0, 21.0):
            c = honeycomb_curve(1190.0, sy, 2.0, 9.0, 50.0, 20.0, 40.0, hw=hw)
            x, F = c[:, 0], c[:, 1]; pk, st, cg = drop(lambda q: np.interp(q, x, F), x[-1])
            print(f"   {name} sig_y {sy}: rho {c[0,2]:.2f}, plateau {F[(x > 2) & (x < c[0,3])].mean():.0f} N, densifies at {c[0,3]:.1f} mm; drop: peak {pk:.0f} N, puck {st:.1f} mm, cage {cg:.1f} mm; "
                  f"normal landing 0.16 J: {np.interp(160.0, np.concatenate([[0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(x))]), x):.2f} mm")
    print("== E2. 68D crumple skirt: Abramowicz-Jones axisymmetric mean crush, sigma0 15 MPa, t 0.86 (2 walls), D 42")
    s0, t, D = 15.0, 0.86, 42.0; M0 = s0 * t * t / 4; Pm = M0 * (20.79 * (D / t) ** 0.5 + 11.9)
    print(f"    Pm {Pm:.0f} N; first-fold peak ~1.4 Pm with a taper trigger {1.4*Pm:.0f}; stroke 70 % of 28 = 19.6 mm ->", drop(lambda q: np.where(q < 2, Pm * 1.4 * q / 2, Pm * (1 + 0.4 * np.exp(-(q - 2) / 2))), 19.6))


if __name__ == "__main__":
    candidates(sys.argv[1] if len(sys.argv) > 1 else 8.0)
