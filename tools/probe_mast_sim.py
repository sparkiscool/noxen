"""Scout v4 probe: the mast's mechanics, simulated (9 Nov, Spark: "I'm skeptical about how the probe is going to work with two bends
fighting when retracting ... run simulations on the design of this probe, as much as possible to see if the one spool would work or if we
need two spools"). Writes systems/probe_sim/*.png and systems/probe_sim/sim.json; the shell FE (CalculiX) is tools/probe_mast_fe.py.
Run: python tools/probe_mast_sim.py [coil|layout|stroke|blossom|cable|all]

Model (plate theory of a tape spring, Rimrott / Seffen & Pellegrino): a blade of flat width b, thickness t, transverse cup radius R,
flattened and coiled to radius r stores, per length,
    U'(r) = D b / 2 [1/r^2 + 1/R^2 - 2 s nu / (r R)],   D = E t^3 / 12 (1 - nu^2),
s = +1 coiled equal-sense (cup toward the hub, a tape measure), -1 opposite-sense. Its surface stresses
    sx = E t / 2(1 - nu^2) (s/r - nu/R),  sy = E t / 2(1 - nu^2) (-1/R + nu s/r)   (von Mises from these).
Spool torque (energy, quasi-static): Q = sum_i r_i (U'_i - P_i) - with P_i the push the blade's free span delivers out (head, friction).
Blossoming: the free span's axial force reaches the coil; in compression the outer turn has nothing holding it on the coil (pressure
P/r < 0), so it lifts; a hub-braked coil also pulls tape back with Db/2 (1/r^2 - 1/R^2) when r < R. Sim 'blossom' is a discrete
elastica (EI = D b per blade, frictionless contact with the coil, the housing, the throat's slot and a sprung pressure roller,
energy minimisation with an analytic gradient)."""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "systems", "probe_sim")
os.makedirs(OUT, exist_ok=True)

# ---- the blade (Topzone 16 ft tape, B00ZMHXV4I; t and R E until Spark measures them) ----------------------------------------
E, NU, B = 200e3, 0.3, 19.0            # MPa, -, mm (flat width, 3/4 in)
T0, R0 = 0.13, 15.0                     # thickness, cup radius (E; patents give t 0.11..0.19, R 12.7..25.4)
SY = 1600.0                             # hardened carbon tape steel yield, MPa (E: SAE 1074/1095 at HRC 48-52, 1500-1800)
ALLOW = 0.9 * SY                        # the coil is a static, cycled state: 90 % of yield
RHO_STEEL = 7.85e-3                     # g/mm^3
HEAD_G = 60.5                           # head frame 2.5 + SG90 10 + cradle 2.5 + camera 45 + LEDs 0.5 (systems/scout_v4.assembly.json)
REACH = 150.0
CABLE_T = 0.6                           # the head's 6 wires as a flat ribbon of 28 / 30 AWG PTFE hook-up wire (the 26 AWG pair made it 1.2)
CABLE_T_OLD = 1.2
G = 9.81

# palette (dataviz: one hue family per meaning; colour-blind-safe pair blue / orange, neutral grey)
C_EQ, C_OP, C_STACK, C_GREY, C_LIM, C_OK = "#2a6fb0", "#d9822b", "#6a4fa3", "#8a8f96", "#b3261e", "#2e7d4f"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e5e8", "grid.linewidth": 0.8, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "legend.frameon": False, "axes.edgecolor": "#555"})
RES = {}


def D_(t):
    return E * t ** 3 / (12 * (1 - NU ** 2))


def U(r, t=T0, R=R0, s=1, b=B):
    """stored energy per length (N) of a blade flattened and coiled at r"""
    return D_(t) * b / 2 * (1 / r ** 2 + 1 / R ** 2 - 2 * s * NU / (r * R))


def vm(r, t=T0, R=R0, s=1):
    k = E * t / (2 * (1 - NU ** 2)); dx, dy = s / r, -1 / R
    sx, sy = k * (dx + NU * dy), k * (dy + NU * dx)
    return np.sqrt(sx ** 2 + sy ** 2 - sx * sy)


def r_min(t, R, s):
    rs = np.linspace(4, 60, 2000)
    ok = rs[vm(rs, t, R, s) <= ALLOW]
    return float(ok[0]) if len(ok) else float("nan")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white"); plt.close(fig)


# ================================================================================================= 1. coil mechanics
def coil():
    r = np.linspace(8, 26, 200)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for s, c, lab in ((1, C_EQ, "equal sense (cup to the hub)"), (-1, C_OP, "opposite sense")):
        ax[0].plot(r, U(r, s=s), c=c, lw=2, label=lab)
        ax[1].plot(r, r * U(r, s=s), c=c, lw=2, label=lab)
        ax[2].plot(r, vm(r, s=s), c=c, lw=2, label=lab)
    ax[0].plot(r, D_(T0) * B / 2 * np.maximum(1 / r ** 2 - 1 / R0 ** 2, 0), c=C_GREY, lw=1.5, ls="--", label="balloon pull-in, hub held (either)")
    ax[0].set(xlabel="coil radius r (mm)", ylabel="N", title="Stored energy per mm = self-deploy push")
    ax[1].set(xlabel="coil radius r (mm)", ylabel="N mm", title="Unwind torque on the spool, per blade")
    ax[2].axhline(SY, c=C_LIM, lw=1, ls=":"); ax[2].axhline(ALLOW, c=C_LIM, lw=1.2)
    ax[2].text(25.8, ALLOW + 20, "90 % of yield (1600 E)", ha="right", va="bottom", color=C_LIM, fontsize=8.5)
    ax[2].set(xlabel="coil radius r (mm)", ylabel="MPa", title="Von Mises at the surface", ylim=(800, 2200))
    for a in ax:
        a.axvline(R0, c=C_GREY, lw=0.8, ls=":")
    ax[0].text(R0 + 0.2, ax[0].get_ylim()[1] * 0.92, "r = R (cup)", fontsize=8, color=C_GREY)
    ax[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"One 19 mm blade, t {T0} mm, cup R {R0:.0f} mm (E)", x=0.01, ha="left", fontsize=11)
    save(fig, "coil.png")
    # minimum hub radius over the patents' range of t and R
    ts = np.array([0.11, 0.13, 0.15, 0.17, 0.19]); Rs = np.array([12.7, 15.0, 19.0, 25.4])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    tab = {}
    for k, (s, lab) in enumerate(((1, "equal sense"), (-1, "opposite sense"))):
        for Rv, ls in zip(Rs, ("-", "--", "-.", ":")):
            rm = [r_min(tv, Rv, s) for tv in ts]
            ax[k].plot(ts, rm, ls=ls, c=C_EQ if s > 0 else C_OP, lw=2, marker="o", ms=4, label=f"R {Rv}")
            tab[f"{lab} R{Rv}"] = dict(zip([f"{v}" for v in ts], [round(v, 1) for v in rm]))
        ax[k].set(title=f"Smallest hub radius, {lab}", xlabel="blade thickness t (mm)")
        ax[k].legend(fontsize=8.5)
    ax[0].set_ylabel("r at 90 % of yield (mm)")
    save(fig, "hub_min.png")
    RES["coil"] = dict(t=T0, R=R0, D=round(D_(T0), 2),
                       at_r={f"{rv}": dict(U_eq=round(float(U(rv, s=1)), 2), U_op=round(float(U(rv, s=-1)), 2), Q_eq=round(float(rv * U(rv, s=1)), 1),
                                            Q_op=round(float(rv * U(rv, s=-1)), 1), vm_eq=round(float(vm(rv, s=1))), vm_op=round(float(vm(rv, s=-1))))
                             for rv in (10.0, 12.0, 14.0, 15.0)},
                       hub_min=tab)


# ================================================================================================= layouts: one vs two spools
def wound(r0, h, s):
    """a spool of hub r0 and layer h paying out s: (turns, outer radius at the departure)"""
    a = h / (4 * math.pi)                                    # s = r0 th + a th^2  (th = angle wound)
    th = (-r0 + math.sqrt(r0 ** 2 + 4 * a * s)) / (2 * a)
    return th / (2 * math.pi), r0 + h * th / (2 * math.pi)


def mismatch(r0, d, h):
    """one spool, two blades stacked d apart (neutral to neutral), h per turn: the outer blade pays out d per radian more"""
    n, _ = wound(r0, h, REACH)
    return d * 2 * math.pi * n, n


def layout():
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    th = np.linspace(0, 2 * math.pi, 200)

    def circle(ax, cx, cz, r, **kw):
        ax.plot(cx + r * np.cos(th), cz + r * np.sin(th), **kw)

    def blade(ax, xs, zs, cupx, col, lab=None):
        ax.plot(xs, zs, c=col, lw=2.4, label=lab)
        ax.annotate("", xy=(xs[-1] + cupx * 2.6, zs[-1] - 4), xytext=(xs[-1], zs[-1] - 4), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2))
    # (a) one spool: the stack leaves the +x tangent
    ax = fig.add_subplot(gs[0]); r = 12.0
    circle(ax, 0, 0, 10, c=C_GREY, lw=1); circle(ax, 0, 0, r, c=C_STACK, lw=2.2)
    blade(ax, [r + 1.3, r + 1.3], [0, 30], -1, C_EQ, "A (cup faces -x)"); blade(ax, [r, r], [0, 30], 1, C_OP, "B (cup faces +x)")
    ax.plot([r + 0.65] * 2, [0, 30], c="#222", lw=1, ls=":", label="cable")
    ax.set_title("1 spool: A equal, B opposite,\nthe cable between"); ax.set_aspect("equal"); ax.set_xlim(-16, 22); ax.set_ylim(-16, 34); ax.legend(fontsize=7.5, loc="lower left")
    # (b) two spools, inner tangents
    ax = fig.add_subplot(gs[1]); c = 16.0
    for sx, col in ((1, C_EQ), (-1, C_OP)):
        circle(ax, sx * c, 0, 12.5, c=col, lw=2.2); circle(ax, sx * c, 0, 10, c=C_GREY, lw=1)
    blade(ax, [3.5, 3.5, 0.6], [0, 14, 30], -1, C_EQ); blade(ax, [-3.5, -3.5, -0.6], [0, 14, 30], 1, C_OP)
    ax.plot([0, 0], [-14, 30], c="#222", lw=1, ls=":")
    ax.set_title("2 spools, inner tangents:\nboth opposite"); ax.set_aspect("equal"); ax.set_xlim(-31, 31); ax.set_ylim(-16, 34)
    # (c) two spools, outer tangents: an S-bend over rollers
    ax = fig.add_subplot(gs[2]); c = 22.0
    for sx, col in ((1, C_OP), (-1, C_EQ)):
        circle(ax, sx * c, -2, 9, c=col, lw=2.2)
    zz = np.linspace(0, 1, 50)
    for sx, col in ((1, C_EQ), (-1, C_OP)):
        x0 = sx * (c - 9 - 0.0) * -1
    ax.plot(-31 + 0 * zz, -2 + 14 * zz, c=C_EQ, lw=2.4); ax.plot(-31 + 30.4 * (1 - np.cos(zz * math.pi)) / 2, 12 + 18 * zz, c=C_EQ, lw=2.4)
    ax.plot(31 + 0 * zz, -2 + 14 * zz, c=C_OP, lw=2.4); ax.plot(31 - 30.4 * (1 - np.cos(zz * math.pi)) / 2, 12 + 18 * zz, c=C_OP, lw=2.4)
    ax.plot([-0.6, 0.6], [30, 30], "k.", ms=2)
    ax.text(0, 4, "the S-bend's\nreturn half is\nopposite sense", ha="center", fontsize=8, color=C_LIM)
    ax.set_title("2 spools, outer tangents:\nequal, but an opposite S-bend"); ax.set_aspect("equal"); ax.set_xlim(-36, 36); ax.set_ylim(-16, 34)
    for a in fig.axes:
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    # (d) the one-spool length mismatch vs hub radius, with the cable between the blades
    ax = fig.add_subplot(gs[3])
    r0s = np.linspace(8, 20, 60)
    for d, lab, col in ((T0, "no cable between (2 pi t a turn)", C_GREY), (T0 + CABLE_T, f"{CABLE_T} ribbon between", C_EQ), (T0 + CABLE_T_OLD, f"{CABLE_T_OLD} cable between (9 Nov design)", C_OP)):
        ax.plot(r0s, [mismatch(r0, d, 2 * T0 + (d - T0))[0] for r0 in r0s], c=col, lw=2, label=lab)
    ax.axhspan(-3, 3, color=C_OK, alpha=0.10); ax.text(19.8, 2.5, "the clip's +-3 slot", ha="right", va="top", fontsize=8, color=C_OK)
    ax.plot([10], [mismatch(10, T0 + CABLE_T_OLD, 2 * T0 + CABLE_T_OLD)[0]], "o", c=C_OP)
    ax.set(xlabel="hub radius (mm)", ylabel="A pays out more than B (mm, 150 out)", title="One spool: the two blades' length fight")
    ax.legend(fontsize=7.5, loc="upper right")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "layouts.png")
    # what fits: the props' keep-out (disc r 64.7 + 4 + 12) over z 27.7..59.7, the turret's band, in plan
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    th = np.linspace(0, 2 * math.pi, 400); rk = 64.7 + 4 + 12
    for mx, my in ((74.3, 74.3), (74.3, -74.3), (-74.3, 74.3), (-74.3, -74.3)):
        ax.fill(mx + rk * np.cos(th), my + rk * np.sin(th), color=C_LIM, alpha=0.12, lw=0); ax.plot(mx + rk * np.cos(th), my + rk * np.sin(th), c=C_LIM, lw=1)
    ax.add_patch(plt.Rectangle((-15.9, -11.7), 31.8, 23.4, fc=C_STACK, alpha=0.35, ec=C_STACK, lw=1.5, label="1 spool O31.8 + 40 T gear"))
    ax.add_patch(plt.Rectangle((-16.8, 11.7), 33.6, 3.0, fc=C_STACK, alpha=0.6, ec=C_STACK))
    for sx in (-1, 1):
        ax.add_patch(plt.Rectangle((15.1 + sx * 18.4 - 15.9, -11.7), 31.8, 23.4, fc="none", ec=C_OP, lw=1.6, ls="--", label="2 spools at +-18.4" if sx > 0 else None))
        ax.add_patch(plt.Rectangle((15.1 + sx * 18.4 - 19.2, 11.7), 38.4, 3.0, fc="none", ec=C_OP, lw=1.2, ls="--"))
    ax.set(xlim=(-45, 60), ylim=(-45, 45), xlabel="x (mm, nose +)", ylabel="y (mm)", title="What fits under the props' keep-out (z 28..60)")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
    save(fig, "fit.png")
    m_old, n_old = mismatch(10.0, T0 + CABLE_T_OLD, 2 * T0 + CABLE_T_OLD)
    RES["layout"] = dict(one_spool_mismatch_9nov=round(m_old, 1), turns_9nov=round(n_old, 2),
                         one_spool_mismatch_hub14_ribbon=round(mismatch(14.0, T0 + CABLE_T, 2 * T0 + CABLE_T)[0], 1),
                         one_spool_mismatch_no_cable_hub10=round(mismatch(10.0, T0, 2 * T0)[0], 1),
                         two_spool_equal_sense_both="impossible in a plane without crossing or an opposite-sense S-bend")


# ================================================================================================= 2. force and torque over the stroke
MOTORS = {  # output stall torque at 6 V (N mm) and no-load rpm; all E: vendors' tables disagree by 2x; measure
    "N20 spur 300 rpm (now, B0BZDC951L)": dict(stall=0.5 * G * 10, rpm=300, lock=False),
    "N20 worm 1:236 (GA12T, B0D9CMZPT4)": dict(stall=0.6 * G * 10, rpm=80, lock=True),
    "N20 spur 100 rpm (chosen, B07MZH851L)": dict(stall=1.5 * G * 10, rpm=100, lock=False),
}
CF_F = 1.96 * 4.448                      # JCF160-250 constant-force spring (B07N1SNSHX): 1.96 lb, 12.7 wide, 0.15 thick, ID 11.2


def stroke_case(kind, hub, mu, cf_r=None, att=0.0, G_=40 / 12, eta=0.85):
    """spool torque (wind-in +, N mm) along the stroke, deploy and retract. kind: 'one' (A eq + B op stacked, cable between),
    'two' (A op with the ribbon on spool 1, B op on spool 2, geared 1:1). mu: the throat's forming friction coefficient."""
    s = np.linspace(0, REACH, 151)
    W = HEAD_G / 1000 * G * math.cos(math.radians(att)) + 2 * 0.15 * T0 * B * RHO_STEEL * G / 1000 * 0  # head (blade self-weight < 0.05 N)
    Lt = 30.0; Nform = 4 * (D_(T0) / R0) / B * Lt / 2          # wall force forming one blade's cup over the transition (N)
    Ff = mu * Nform                                             # per blade
    out = {}
    for d, sg in (("deploy", -1), ("retract", 1)):
        q = []
        for si in s:
            left = REACH - si                                   # still wound (the spool holds 150 when parked)
            if kind == "one":
                h = 2 * T0 + CABLE_T; _, rB = wound(hub, h, max(left, 1e-6)); rA = rB + T0 + CABLE_T
                tq = rA * (U(rA, s=1) - W + sg * Ff) + rB * (U(rB, s=-1) + sg * Ff)
            else:
                _, rA = wound(hub, T0 + CABLE_T, max(left, 1e-6)); _, rB = wound(hub + 0.57, T0, max(left, 1e-6))
                tq = rA * (U(rA, s=-1) - W + sg * Ff) + rB * (U(rB, s=-1) + sg * Ff)
            if cf_r:
                tq -= CF_F * cf_r                               # the constant-force spring winds in: it takes over the coil's push
            q.append(tq)
        out[d] = np.array(q)
    return s, out, dict(Ff=Ff, W=W, Nform=Nform)


def stroke():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
    rows = {}
    for k, (kind, hub, lab) in enumerate((("one", 10.0, "1 spool, hub r 10 (9 Nov)"), ("one", 14.0, "1 spool, hub r 14"), ("two", 13.0, "2 spools, hub r 13"))):
        a = ax[k]
        for mu, ls in ((0.05, "-"), (0.3, "--")):
            s, q, info = stroke_case(kind, hub, mu)
            a.plot(s, q["deploy"], c=C_EQ, ls=ls, lw=2, label=f"pay out, mu {mu}")
            a.plot(s, q["retract"], c=C_OP, ls=ls, lw=2, label=f"wind in, mu {mu}")
            rows[f"{lab} mu{mu}"] = dict(deploy=[round(float(q['deploy'].min()), 1), round(float(q['deploy'].max()), 1)],
                                         retract=[round(float(q['retract'].min()), 1), round(float(q['retract'].max()), 1)])
        for name, m in MOTORS.items():
            st = m["stall"] * (40 / 12) * 0.85
            a.axhline(st, c=C_LIM if "spur" in name else C_GREY, lw=1, ls=":")
            a.text(1, st + 3, f"{name.split(' (')[0]} stall x 40/12", fontsize=7.5, color=C_LIM if "spur" in name else C_GREY)
        a.set(title=lab, xlabel="mast out (mm)", ylim=(0, 260))
        if k == 0:
            a.set_ylabel("spool torque the motor holds (N mm, wind-in +)")
            a.legend(fontsize=7.5, loc="lower right")
    save(fig, "stroke_torque.png")
    # the chosen drive: 1 spool, hub r 14, the 100 rpm N20 through 40 / 12, any attitude
    fig, a = plt.subplots(figsize=(7.2, 4.1))
    best = {}
    for att, col in ((0, C_EQ), (90, C_GREY), (180, C_OP)):
        for mu, ls in ((0.05, "-"), (0.3, "--")):
            s_, q, info = stroke_case("one", 14.0, mu, att=att)
            a.fill_between(s_, q["deploy"], q["retract"], color=col, alpha=0.10 if ls == "--" else 0.22, lw=0)
            a.plot(s_, q["deploy"], c=col, lw=1.3, ls=ls); a.plot(s_, q["retract"], c=col, lw=1.3, ls=ls)
            best[f"att{att} mu{mu}"] = [round(float(min(q['deploy'].min(), q['retract'].min())), 1), round(float(max(q['deploy'].max(), q['retract'].max())), 1)]
    st = MOTORS["N20 spur 100 rpm (chosen, B07MZH851L)"]["stall"] * (40 / 12) * 0.85
    a.axhline(st, c=C_LIM, lw=1, ls=":"); a.text(2, st - 14, f"100 rpm N20 stall x 40/12: {st:.0f} (E)", fontsize=8, color=C_LIM)
    a.axhline(st * 0.5, c=C_LIM, lw=0.8, ls=":", alpha=0.6); a.text(2, st * 0.5 + 4, "half of stall", fontsize=8, color=C_LIM)
    a.text(149, 30, "upright", color=C_EQ, ha="right", fontsize=8); a.text(149, 16, "on its side", color=C_GREY, ha="right", fontsize=8); a.text(149, 2, "inverted", color=C_OP, ha="right", fontsize=8)
    a.set(xlabel="mast out (mm)", ylabel="spool torque (N mm, wind-in +)", title="Chosen: 1 spool hub r 14; band = pay out .. wind in; dashed mu 0.3", ylim=(0, 450))
    save(fig, "drive.png")
    _, _, info = stroke_case("one", 14.0, 0.3)
    RES["stroke"] = dict(rows=rows, spring=best, Ff_per_blade_mu03=round(info["Ff"], 2), Nform=round(info["Nform"], 1), head_N=round(info["W"], 2),
                         motors={k: dict(v, stall_at_spool=round(v["stall"] * 40 / 12 * 0.85)) for k, v in MOTORS.items()}, cf_spring_N=round(CF_F, 2))


# ================================================================================================= 3. blossoming: the discrete elastica
class Coil:
    """the outer turn of a coil and its free span up the throat's slot, as a chain of N links (angles phi). Spool at (0, 0); the strip
    runs counter-clockwise on the coil (radius rc) from an anchor 350 deg back and leaves at (rc, 0) straight up (+y) into the slot
    (x in rc +- c, y 2..34). End force Pn down (Pn > 0: the free span in compression). Optional roller (radius a) at angle th_r on a
    constant-force spring F_r pushing it toward the axis. Contact: penalty k. EI per the stack, k0 its natural curvature (CCW +)."""

    def __init__(self, rc=13.0, EI=764.0, k0=0.0, rh=None, c=0.6, ds=0.5, span=34.0, kpen=4e3, anchor_deg=350.0):
        self.rc, self.EI, self.k0, self.rh, self.c, self.ds, self.kp = rc, EI, k0, rh, c, ds, kpen
        self.tha = -math.radians(anchor_deg)
        self.p0 = rc * np.array([math.cos(self.tha), math.sin(self.tha)])
        La = rc * math.radians(anchor_deg); self.Na = int(round(La / ds)); self.N = self.Na + int(round(span / ds))
        self.phi_a = self.tha + math.pi / 2                       # the anchor's tangent
        self.y0, self.y1 = 2.0, span

    def tight(self):
        phi = np.empty(self.N)
        k = np.arange(self.Na); phi[:self.Na] = self.phi_a + (k + 0.5) * self.ds / self.rc
        phi[self.Na:] = math.pi / 2
        return phi

    def nodes(self, phi):
        u = np.stack([np.cos(phi), np.sin(phi)], 1) * self.ds
        return np.vstack([self.p0, self.p0 + np.cumsum(u, 0)])

    def energy(self, x, Pn, roller):
        phi = x[:self.N]; p = self.nodes(phi)
        dphi = np.diff(np.concatenate([[self.phi_a], phi])) - self.k0 * self.ds
        Eb = self.EI / (2 * self.ds) * np.sum(dphi ** 2)
        g_phi = np.zeros(self.N)
        w = self.EI / self.ds * dphi; g_phi += w; g_phi[:-1] -= w[1:]
        f = np.zeros_like(p); Ec = 0.0; kp = self.kp
        rr = np.linalg.norm(p, axis=1) + 1e-12
        pen = np.minimum(rr - self.rc, 0.0); Ec += kp / 2 * np.sum(pen ** 2); f += (kp * pen / rr)[:, None] * p
        if self.rh:
            win = (p[:, 1] > -0.5) & (np.abs(p[:, 0] - self.rc) < 2.5)   # the housing's exit window into the throat
            pen = np.where(win, 0.0, np.maximum(rr - self.rh, 0.0)); Ec += kp / 2 * np.sum(pen ** 2); f += (kp * pen / rr)[:, None] * p
        inslot = (p[:, 1] > self.y0) & (np.arange(len(p)) >= self.Na - 2)
        for sg in (1, -1):                                        # the slot's two walls
            pen = np.where(inslot, np.maximum(sg * (p[:, 0] - self.rc) - self.c, 0.0), 0.0)
            Ec += kp / 2 * np.sum(pen ** 2); f[:, 0] += kp * pen * sg
        Ec += Pn * p[-1, 1]; f[-1, 1] += Pn
        g_rho = 0.0
        if roller:
            th, a, Fr = roller; rho = x[self.N]
            C = rho * np.array([math.cos(th), math.sin(th)])
            d = p - C; dn = np.linalg.norm(d, axis=1) + 1e-12
            pen = np.minimum(dn - a, 0.0); Ec += kp / 2 * np.sum(pen ** 2)
            fd = (kp * pen / dn)[:, None] * d; f += fd
            g_rho = -np.sum(fd @ np.array([math.cos(th), math.sin(th)])) + Fr
            pr = min(rho - (self.rc + a), 0.0); Ec += kp / 2 * pr ** 2 * 10; g_rho += kp * 10 * pr
            Ec += Fr * rho
        f[0] = 0.0
        S = np.cumsum(f[::-1], 0)[::-1]                           # S[i] = sum_{k>=i} f_k
        Sx, Sy = S[1:, 0], S[1:, 1]
        g_phi += self.ds * (-np.sin(phi) * Sx + np.cos(phi) * Sy)
        g = np.concatenate([g_phi, [g_rho]]) if roller else g_phi
        return Eb + Ec, g

    def solve(self, Pn, roller=None, x0=None):
        if x0 is None:
            x0 = self.tight()
            if roller:
                x0 = np.concatenate([x0, [self.rc + roller[1] + 0.05]])
        r = minimize(self.energy, x0, args=(Pn, roller), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, maxcor=30, ftol=1e-13, gtol=1e-7))
        p = self.nodes(r.x[:self.N]); rr = np.linalg.norm(p[:self.Na + 1], axis=1)
        lost = self.nodes(self.tight())[-1, 1] - p[-1, 1]        # tape drawn back from the mast (mm)
        F = self.forces(r.x, Pn, roller); rad = self.p0 / np.linalg.norm(self.p0)
        return dict(x=r.x, p=p, gap=float((rr - self.rc).max()), lost=float(lost), ok=bool(r.success), rho=float(r.x[self.N]) if roller else None,
                    pull_out=float(-F @ rad), pull_tan=float(F @ np.array([-rad[1], rad[0]])))

    def forces(self, x, Pn, roller):
        """the force the strip puts on its anchor (N): the sum of the loads and contact forces on the chain"""
        eps = 1e-6; p = self.nodes(x[:self.N])
        # numerically: the derivative of the energy with respect to a rigid shift of the anchor (all nodes move together)
        out = []
        for dvec in (np.array([eps, 0.0]), np.array([0.0, eps])):
            self.p0 = self.p0 + dvec; e1 = self.energy(x, Pn, roller)[0]; self.p0 = self.p0 - 2 * dvec; e0 = self.energy(x, Pn, roller)[0]; self.p0 = self.p0 + dvec
            out.append((e1 - e0) / (2 * eps))
        return np.array(out)


def roller_force_needed(cl, Pn, th, a=3.5, gap_tol=0.15):
    lo, hi = 0.0, 80.0
    if cl.solve(Pn, (th, a, hi))["gap"] > gap_tol:
        return float("nan")
    for _ in range(14):
        m = 0.5 * (lo + hi)
        (hi, lo) = (m, lo) if cl.solve(Pn, (th, a, m))["gap"] <= gap_tol else (hi, m)
    return hi


def blossom():
    """(a, b) frictionless with only a housing 1.5 out: when does the outer turn hold? (c) a pressure roller presses the outer turn on the
    turn below; friction there locks the coil up to the roller (the chain anchored at the roller): the arc from the roller to the exit
    stays down at any push; the roller must give the friction mu F_r >= the span's push + the coil's own pull EI k^2 / 2"""
    EI1 = D_(T0) * B; rc = 15.6; e_flat = EI1 / (2 * R0 ** 2)     # the chosen coil (hub r 15 + the layers); the throat's re-cupping push per blade
    Ps = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.7), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    out = {}
    free = {"1 spool: the stack A + ribbon + B": Coil(rc=rc, EI=2 * EI1, k0=0.0, rh=rc + 1.5),
            "a single blade (2 spools), opposite": Coil(rc=rc, EI=EI1, k0=-NU / R0, rh=rc + 1.5)}

    def lift(cl, r):                                              # the turn's lift-off, the last 25 deg before the exit left out
        nn = cl.Na - int(math.radians(25) * cl.rc / cl.ds)
        return float((np.linalg.norm(r["p"][:max(nn, 1)], axis=1) - cl.rc).max())
    th = np.linspace(0, 2 * math.pi, 300)
    cl = free["1 spool: the stack A + ribbon + B"]
    ax[0].plot(rc * np.cos(th), rc * np.sin(th), c=C_GREY, lw=1); ax[0].plot(cl.rh * np.cos(th), cl.rh * np.sin(th), c=C_GREY, lw=1, ls="--")
    ax[0].fill_between([rc + cl.c, rc + 3], cl.y0, cl.y1, color=C_GREY, alpha=0.25, lw=0); ax[0].fill_between([rc - 3, rc - cl.c], cl.y0, cl.y1, color=C_GREY, alpha=0.25, lw=0)
    for Pn, col in ((-4.0, C_OK), (0.0, C_LIM)):
        r = cl.solve(Pn); ax[0].plot(r["p"][:, 0], r["p"][:, 1], c=col, lw=1.6, label=f"no roller, span {'pulled ' + str(int(-Pn)) + ' N' if Pn < 0 else 'unloaded'}: lifts {lift(cl, r):.2f}")
    lk = Coil(rc=rc, EI=2 * EI1, k0=0.0, rh=rc + 1.5, anchor_deg=40)
    r = lk.solve(3.0); ax[0].plot(r["p"][:, 0], r["p"][:, 1], c=C_EQ, lw=2.4, label="roller 40 deg before the exit, pushed 3 N: holds")
    C = (rc + 3.6) * np.array([math.cos(-math.radians(40)), math.sin(-math.radians(40))])
    ax[0].plot(C[0] + 3.5 * np.cos(th), C[1] + 3.5 * np.sin(th), c=C_EQ, lw=1.2)
    ax[0].set_aspect("equal"); ax[0].set(xlim=(-20, 25), ylim=(-20, 37), title=f"The coil's outer turn and free span\n(1 spool stack, coil r {rc}, housing +1.5)")
    ax[0].legend(fontsize=7.2, loc="upper left"); ax[0].set_xticks([]); ax[0].set_yticks([])
    for (name, cl), col in zip(free.items(), (C_STACK, C_OP)):
        g, l = [], []
        for Pn in Ps:
            r = cl.solve(Pn); g.append(lift(cl, r)); l.append(r["lost"])
        ax[1].plot(Ps, g, c=col, lw=2, marker="o", ms=4, label=name)
        ax[1].axvline(-cl.EI / (2 * rc ** 2), c=col, lw=0.8, ls=":")
        out[name] = dict(P=Ps, lift=[round(v, 2) for v in g], drawn_back=[round(v, 1) for v in l], holds_below=round(-cl.EI / (2 * rc ** 2), 2))
    ax[1].set(xlabel="flat free span's force (N, + push, - pull)", ylabel="outer turn's lift-off (mm)", title="Nothing on the coil but a housing")
    ax[1].text(-3.9, 1.05, "dotted: it holds only while the span\npulls more than EI k^2 / 2", fontsize=8, color=C_GREY)
    ax[1].legend(fontsize=7.5, loc="center right")
    # (c) the roller force: the mast's push at the throat (head + guide friction) against the throat's re-cupping push e_flat
    Pm = np.linspace(0, 6, 61)
    req = {}
    for (name, EI, nb), col in ((("1 spool stack", 2 * EI1, 2), C_STACK), (("2 spools, each blade", EI1, 1), C_OP)):
        for mu, ls in ((0.3, "-"), (0.2, "--")):
            Fr = np.maximum(Pm - nb * e_flat + EI / (2 * rc ** 2), 0) / mu
            ax[2].plot(Pm, Fr, c=col, ls=ls, lw=2, label=f"{name}, mu {mu}")
            req[f"{name} mu{mu}"] = dict(P=[0, 1, 2, 3, 4], F=[round(float(np.interp(v, Pm, Fr)), 1) for v in (0, 1, 2, 3, 4)])
    lk_g = []
    for Pn in (0.5, 1.0, 2.0, 3.0):
        r = lk.solve(Pn); lk_g.append(round(lift(lk, r) if lk.Na > 30 else r["gap"], 2))
    ax[2].set(xlabel="push the mast needs at the throat (N): head + guide friction", ylabel="pressure roller force (N)", title="Roller that locks the coil by friction", ylim=(0, None))
    ax[2].legend(fontsize=7.5)
    save(fig, "blossom.png")
    out["roller"] = req; out["locked_arc_lift"] = lk_g; out["rc"] = rc; out["e_flat"] = round(e_flat, 2)
    RES["blossom"] = out


# ================================================================================================= 5. the cable
def cable():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    hw = R0 * math.sin(B / (2 * R0)); sag = R0 * (1 - math.cos(B / (2 * R0)))
    y = np.linspace(-hw, hw, 100); bow = np.sqrt(R0 ** 2 - y ** 2) - math.sqrt(R0 ** 2 - hw ** 2)
    ax[0].plot(bow, y, c=C_EQ, lw=2.5, label="blade A"); ax[0].plot(-bow, y, c=C_OP, lw=2.5, label="blade B")
    wy = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]; dia = [0.6, 0.6, 0.5, 0.5, 0.5, 0.5]
    for yy, dd in zip(wy, dia):
        ax[0].add_patch(plt.Circle((sag - T0 - dd / 2 - 0.05, yy * 1.0), dd / 2, color="#333"))
    ax[0].set_aspect("equal"); ax[0].set(title=f"The lens: {2 * hw:.1f} across, {2 * sag:.1f} deep; the ribbon on A", xlabel="x (mm)", ylabel="y (mm)")
    ax[0].legend(fontsize=8, loc="upper left")
    # the ribbon on spool A: payout vs blade A's (the ribbon is outside A on the coil)
    s = np.linspace(1, REACH, 100); hub = 13.0
    dA = []
    for si in s:
        th_n, _ = wound(hub, T0 + CABLE_T, si); dA.append((T0 / 2 + CABLE_T / 2) * 2 * math.pi * th_n)
    ax[1].plot(s, dA, c=C_EQ, lw=2, label="ribbon slack along the lens")
    th_n, _ = wound(hub, T0 + CABLE_T, REACH)
    ax[1].set(title="The ribbon outside A on spool A", xlabel="mast out (mm)", ylabel="mm")
    ax[1].legend(fontsize=8)
    save(fig, "cable.png")
    RES["cable"] = dict(lens_across=round(2 * hw, 2), lens_depth=round(2 * sag, 2), ribbon_slack_full=round(dA[-1], 1), ribbon=f"6 x PTFE, 2 x 28 + 4 x 30 AWG, {CABLE_T} thick, 3.6 wide")


def embed():
    """writes every systems/probe_sim PNG into the appendix's <img data-sim="name.png" ...> tags as data URIs (the page is one file)"""
    import base64, re
    F = os.path.join(ROOT, "systems", "scout_v4.appendix.html"); h = open(F).read()

    def sub(m):
        f = os.path.join(OUT, m.group(1))
        if not os.path.exists(f):
            return m.group(0)
        return f'<img data-sim="{m.group(1)}" src="data:image/png;base64,{base64.b64encode(open(f, "rb").read()).decode()}"'
    h = re.sub(r'<img data-sim="([^"]+)" src="[^"]*"', sub, h)
    open(F, "w").write(h)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "embed":
        embed(); sys.exit(0)
    for name, fn in (("coil", coil), ("layout", layout), ("stroke", stroke), ("blossom", blossom), ("cable", cable)):
        if what in (name, "all"):
            fn(); print(name, "done")
    J = os.path.join(OUT, "sim.json"); old = json.load(open(J)) if os.path.exists(J) else {}
    old.update(RES); json.dump(old, open(J, "w"), indent=1)
    print(json.dumps(RES, indent=1)[:4000])
