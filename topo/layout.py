"""Layout optimisation: robust three-field density topology optimisation with connected roots by construction.

Formulation (references in nox/topo/README.md):
  x (design cells) -> x_hat (x on design, 1 on drawn, 0 on void) -> cone filter (constant normalisation: outside the
  grid and every void counts as empty, so the length scale holds against keep-outs too) -> x_tilde
  -> Heaviside projections at eta_e > eta_n > eta_d -> eroded / nominal (blueprint) / dilated designs.

  min  p-mean over load cases of c_i(eroded) / c_i(full solid)        (real cases + tie cases on every drawn body)
  s.t. volume(dilated) <= V*  (V* adapted so the nominal volume meets the mass target)
       Zhou et al. 2015 geometric length-scale constraints on the blueprint (solid and void), from beta >= geo_beta.

Drawn-masked eroded filter (default, Settings.erosion = "masked"): the eroded realization filters the design density
over the NON-drawn cells only, x_tilde_e = conv(x on design, 0 elsewhere) / (1 - conv(drawn)) (voids and the grid
outside count as empty in both, drawn cells are left out of both; the denominator is clamped at 0.05). Drawn cells stay
1 in the eroded design and the loads stay where they are. A wide member keeps its own density up to the drawn face and
survives erosion; a thin neck is averaged with the empty space round it and erodes, so every load through it blows up
and the optimiser widens the root. A thin drawn plate next to a cut no longer erodes, so it cannot isolate anything.
Nominal and dilated keep the constant-normalised filter of x_hat (drawn = 1).

Anchored-core erosion (option, erosion = "anchored", round 1 default): only the drawn bodies' cores stay 1 in the eroded
design; their skins are projected like design material and loads on them move to the nearest core node. It isolated
thin drawn plates next to cuts (bracket_jA topo-v1: 3 bodies off from beta 8, tie compliance ratio ~1e5).
erosion = "frozen": drawn cells 1, eroded field = the nominal filter (the classic robust formulation).
In "masked" (29 Oct) a thick drawn body is held only at the core the solid gate keeps (the part eroded by
clip(0.8 x its half thickness, 0.9, 2.0) mm): its skin takes the nominal filter's density and erodes with the design, so a
member that meets the body at its rim is seen to come off, as the gate sees it; loads on a skin move to the nearest core
node. A body too thin for a core stays whole, as before (the gate judges it by its own size).

The minimum member (rod diameter of the blueprint) is set in mm; the filter radius follows from a calibration of the
robust thresholds for a cone filter in 3D (rods and plates), so the engine, not the part script, owns the length scale.
"""
import os
import time
import json
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

from .material import MATERIALS, AXES, stiffness
from .fem import Solver
from .mma import MMA

S26 = np.ones((3, 3, 3), bool)
S6 = ndi.generate_binary_structure(3, 1)


@dataclass
class Settings:
    min_member_mm: float = 3.2          # blueprint minimum rod diameter; plates come out ~0.75 of it (calibration)
    eta: tuple = (0.75, 0.5, 0.25)      # eroded / nominal / dilated thresholds (Wang, Lazarov, Sigmund 2011)
    betas: tuple = (1, 2, 4, 8, 16, 32)
    penal: float = 3.0
    emin: float = 1e-6
    pmean: float = 6.0                  # p-mean exponent over the load cases (worst cases dominate)
    tie_weight: float = 0.5
    tie_moments: bool = True
    tie_min_cells: int = 4
    move: float = 0.1
    min_stage: int = 10
    max_stage: int = 35
    grey_stop: float = 0.02
    vol_gain: float = 0.3               # damped volume-limit correction (1.0 limit-cycles at low volfrac)
    geometric: bool = False             # Zhou 2015 constraints (off: README, Geometric constraints)
    geo_margin: float = 0.05            # reachability margin of the geometric-constraint masks
    geo_beta: float = 16.0
    geo_eps: float = 0.0                # 0 = automatic from the filter radius
    erosion: str = "masked"            # "masked" (drawn-masked eroded filter) | "anchored" | "frozen"
    gate_ties: bool = True              # thin held drawn bodies count by their cover + ties to one body (masked only)
    gate_thin_mm: float = 2.0           # the solid gate's erosion cap: held bodies with half thickness under it are thin
    # case normalisation: "solid" divides every case by the full solid's compliance; "frame" (option) divides by its
    # own compliance on the first frame the run reaches (the warm start of a warm-started run, else the design at the
    # first beta step), so each case counts by how much it worsens from a frame. Against the solid, concentrated loads
    # and ties swamp the distributed cases (holder organic-v7: each misuse case 1e-4 of the objective), but "frame"
    # did not lower the worst index (organic-v8 0.596 vs v7 0.569) and spiked once at beta 32, so it stays off.
    case_norm: str = "solid"
    held_ties: bool = True              # held bodies (each on its own support patch) tied to one hub body (see _cases)
    gate_member: bool = True            # optimise() raises min_member_mm to what the solid gate needs (gate_member_mm)
    collar_mm: float = 0.0              # optional passive collar round every drawn body (off: anchored erosion does it)
    # finish (23 Oct, Spark: the whole part smooth; only flat contact faces and mounting holes geometric).
    # "organic": a drawn body stays passive only as a functional skin: its cells within skin_mm of a cut (holes, mating
    # and contact faces, keep-outs) or of the envelope's outside; the rest of every drawn body is design space, and the
    # build makes the part one smooth field (generated material blended into the skins with fillet_mm) cut exactly by
    # the cuts and the envelope, so holes and contact faces come out exact and everything else smooth.
    # "drawn": the drawn bodies stay whole and are added exactly at the build (the v1 behaviour).
    finish: str = "organic"
    skin_mm: float = 2.5                # functional skin thickness (>= the 2.0 wall rule round holes and contact faces)
    fillet_mm: float = 2.5              # blend radius between generated material and the skins (and off keep-outs)
    skin_round_mm: float = 1.0          # the skins' free rims rounded with this radius (they never meet a cut there)
    reach_check: str = "raise"          # "raise" | "warn" | "off"
    audit: str = "raise"                # setup audit (topo/audit.py): "raise" | "warn" | "off"
    auto_sym: bool = True
    backend: str = "auto"               # "auto" | "direct" | "mg"
    mg_tol: float = 1e-5
    time_limit_s: float = 5.5 * 3600
    checkpoint_every: int = 10


# --------------------------------------------------------------------------------------- length-scale calibration
_CAL = {}


def _w1(s):
    """cone kernel (R = 1) integrated over a plane, normalised to unit 3D integral: the 1D (plate) profile."""
    a = np.abs(s)
    return np.where(a < 1.0, (np.pi / 3.0) * (1.0 - a) ** 2 * (1.0 + 2.0 * a), 0.0) / (np.pi / 3.0)


def _w2(s):
    """cone kernel (R = 1) integrated along a line: the 2D (rod cross-section) profile, unit 3D integral."""
    a = np.abs(s)
    t = np.sqrt(np.maximum(1.0 - a * a, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = t - a * a * np.log((1.0 + t) / np.maximum(a, 1e-12))
    return np.where(a < 1.0, v, 0.0) / (np.pi / 3.0)


def _bisect(f, lo, hi, target, it=60, increasing=True):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibration(eta=(0.75, 0.5, 0.25)):
    """sizes for a cone filter of radius 1 at beta -> infinity: rod = blueprint diameter of the thinnest rod that
    survives erosion, plate = same for a plate, erode = erosion depth of a flat face (cells inside), dilate = growth."""
    key = tuple(round(e, 6) for e in eta)
    if key in _CAL:
        return _CAL[key]
    ee, en, ed = eta
    ds = 1e-3
    s = np.arange(-3.0, 3.0 + ds, ds)
    C = np.cumsum(_w1(s)) * ds
    Cf = lambda x: np.interp(x, s, C)
    plate = lambda T, x=0.0: Cf(x + T / 2.0) - Cf(x - T / 2.0)
    Te = _bisect(lambda T: plate(T), 0.0, 3.0, ee)
    xn = _bisect(lambda x: plate(Te, x), 0.0, 3.0, en, increasing=False)
    erode = _bisect(Cf, 0.0, 1.0, ee)
    dilate = -_bisect(Cf, -1.0, 0.0, ed)
    # rods: 2D integral of the projected kernel over a disc
    dq = 0.01

    def rod(a, rho):
        n = int(np.ceil(a / dq))
        q = (np.arange(-n, n) + 0.5) * dq
        QX, QY = np.meshgrid(q, q, indexing="ij")
        m = QX ** 2 + QY ** 2 <= a * a
        return float(_w2(np.sqrt((QX[m] - rho) ** 2 + QY[m] ** 2)).sum() * dq * dq)

    ae = _bisect(lambda a: rod(a, 0.0), 0.05, 2.0, ee, it=30)
    rn = _bisect(lambda r: rod(ae, r), 0.0, 2.5, en, it=30, increasing=False)
    out = dict(rod=2.0 * rn, plate=2.0 * xn, erode=erode, dilate=dilate, rod_input=2.0 * ae, plate_input=Te)
    _CAL[key] = out
    return out


# ---------------------------------------------------------------------------------------------------- helpers
def project(xt, beta, eta):
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1.0 - eta))
    th = np.tanh(beta * (xt - eta))
    return (tb + th) / den, beta * (1.0 - th * th) / den


def auto_symmetry(solid3, void3, near_tol=0.08):
    axes, near = [], []
    n_pass = max(int(solid3.sum() + void3.sum()), 1)
    for a in range(3):
        miss = (np.logical_xor(solid3, np.flip(solid3, a)).sum()
                + np.logical_xor(void3, np.flip(void3, a)).sum()) / (2.0 * n_pass)
        if miss == 0:
            axes.append(a)
        elif miss < near_tol:
            near.append((a, float(miss)))
    return axes, near


class ReachabilityError(ValueError):
    pass


def members_reach(part, m_mm):
    """True when every drawn body (>= 4 cells) can be joined to one common region of the design space that holds a
    rod of m_mm (the test Layout._reach makes before a run)."""
    shape, h = part.shape, part.h
    A = (part.design | part.solid).reshape(shape)
    r = 0.45 * m_mm / h
    E = ndi.distance_transform_edt(A) > r
    labE, nE = ndi.label(E, S6)
    if nE == 0:
        return False
    dist, ind = ndi.distance_transform_edt(~E, return_indices=True)
    near_lab = labE[tuple(ind)]
    lab, nb = ndi.label(part.solid.reshape(shape), S26)
    common = None
    for b in range(1, nb + 1):
        cells = lab == b
        if cells.sum() < 4:
            continue
        ok = cells & (dist <= r + 1.5)
        labs = set(np.unique(near_lab[ok]).tolist()) if ok.any() else set()
        common = labs if common is None else common & labs
        if not common:
            return False
    return True


def gate_member_mm(part, floor_mm, gate_erode_cap=2.0, gate_erode_min=0.9, margin_mm=2.25, log=print):
    """minimum member that lets the solid gate pass. nox/solid_check.py erodes the part round every drawn body by
    clip(0.8 x its depth, 0.9, 2.0) mm (depth measured in the finished part, where generated material has fused onto
    the body) and requires the body's core to stay in the main eroded body. Members joining such a body must then be
    thicker than twice that depth; a 3.2 mm frame splits into dozens of pieces at 2.0 mm (holder topo-v1: 72).
    Returns the member size (blueprint rod diameter, mm) = 2 x depth + margin, lowered to the largest size the design
    space can still route to every drawn body (a setup that cannot host it is reported, not refused).
    margin 2.25 mm (was 1.0): holder organic-v2 with 5.0 mm members built struts of 5.2 to 6.4 mm whose thinnest
    joints broke at the gate's 2.0 mm erosion; organic-v3 with 6.25 mm members kept a core in every strut."""
    lab, nb = ndi.label(part.solid.reshape(part.shape), S26)
    if nb == 0:
        return floor_mm
    d_own = float(ndi.distance_transform_edt(part.solid.reshape(part.shape)).max()) * part.h
    ef = float(np.clip(0.8 * (d_own + 0.5 * floor_mm), gate_erode_min, gate_erode_cap))
    want = max(floor_mm, 2.0 * ef + margin_mm)
    if want <= floor_mm or members_reach(part, want):
        if want > floor_mm:
            log(f"layout: min member {floor_mm:.2f} -> {want:.2f} mm so members survive the solid gate's "
                f"{ef:.1f} mm erosion round the drawn bodies")
        return want
    lo, hi = floor_mm, want
    if not members_reach(part, lo):
        return floor_mm
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if members_reach(part, mid) else (lo, mid)
    log(f"layout: WARNING the solid gate wants {want:.2f} mm members round the drawn bodies ({ef:.1f} mm erosion), "
        f"the design space routes at most {lo:.2f} mm to every body: using {lo:.2f} mm; the gate may fail on the setup")
    return lo


# ------------------------------------------------------------------------------------------------------ layout
class Layout:
    def __init__(self, part, gen_cm3=None, volfrac=None, settings=None, log=print):
        self.s = s = settings or Settings()
        self.part = part
        self.log = log
        self.h = h = part.h
        self.shape = shape = part.shape
        self.nel = part.nel
        solid = part.solid.copy()
        void = part.void.copy()
        design = part.design.copy()
        if s.finish not in ("organic", "drawn"):
            raise ValueError(f"layout: finish must be organic or drawn, not {s.finish!r}")
        free_cm3 = 0.0
        if s.finish == "organic":
            skin = part.functional_skin(s.skin_mm)
            free = solid & ~skin
            solid, design = skin, design | free
            free_cm3 = free.sum() * h ** 3 / 1000.0
            log(f"layout: organic finish: drawn bodies kept as {s.skin_mm} mm functional skins "
                f"({skin.sum() * h ** 3 / 1000:.2f} cm3); {free.sum() * h ** 3 / 1000:.2f} cm3 of drawn volume opened to "
                f"the design")
        if s.collar_mm > 0:
            d = ndi.distance_transform_edt(~solid.reshape(shape)).ravel() * h
            add = design & (d <= s.collar_mm)
            solid |= add
            design &= ~add
            log(f"layout: collar {s.collar_mm} mm round the drawn bodies: +{add.sum() * h ** 3 / 1000:.2f} cm3 passive")
        self.solid, self.void, self.design = solid, void, design
        self.stiff = getattr(part, "stiff", np.ones(self.nel))   # inserts (Part.insert): solid at their own modulus
        ins = getattr(part, "insert_cells", np.zeros(self.nel, bool))
        self.has_inserts = bool(ins.any())
        self.stiff_tie = np.where(ins, s.emin, self.stiff)        # tie cases: inserts absent (bonded in after printing)
        self.nd = int(design.sum())
        if self.nd == 0:
            raise ValueError("layout: no design cells")
        if gen_cm3 is not None:
            # gen_cm3 is the volume added to the drawn bodies as given, in both finishes, so a part keeps one mass
            # target: the drawn volume the organic finish opened to the design is put back on top of it
            gen_cm3 = gen_cm3 + free_cm3
            volfrac = gen_cm3 * 1000.0 / (self.nd * h ** 3)
        self.volfrac = float(volfrac)
        if not 0.0 < self.volfrac < 1.0:
            raise ValueError(f"layout: mass target gives volfrac {self.volfrac:.3f}; must be in (0, 1)")
        # ---- length scale
        cal = calibration(s.eta)
        self.cal = cal
        self.R_mm = s.min_member_mm / cal["rod"]
        self.R = self.R_mm / h
        self.erode_cells = cal["erode"] * self.R
        log(f"layout: min member {s.min_member_mm:.2f} mm (rods), plates {cal['plate'] * self.R_mm:.2f} mm, "
            f"filter radius {self.R_mm:.2f} mm = {self.R:.2f} cells, erosion depth {cal['erode'] * self.R_mm:.2f} mm; "
            f"volfrac {self.volfrac:.3f} of {self.nd:,} design cells")
        if self.R < 2.0:
            log(f"layout: WARNING filter radius {self.R:.2f} cells < 2: the grid is too coarse for "
                f"{s.min_member_mm} mm members; use h <= {self.R_mm / 2.5:.2f}")
        k = int(np.ceil(self.R))
        ii = np.arange(-k, k + 1)
        I, J, K = np.meshgrid(ii, ii, ii, indexing="ij")
        ker = np.maximum(0.0, self.R - np.sqrt(I * I + J * J + K * K))
        self.ker = ker / ker.sum()
        # ---- symmetry
        self.sym_axes = []
        if s.auto_sym:
            axes, near = auto_symmetry(solid.reshape(shape), void.reshape(shape))
            self.sym_axes = axes
            if axes:
                log(f"layout: drawn bodies and keep-outs mirror about {', '.join('xyz'[a] for a in axes)}: the design is "
                    f"held symmetric there")
            for a, miss in near:
                log(f"layout: WARNING nearly mirrored in {'xyz'[a]} ({miss:.1%} of passive cells differ): no symmetry")
        # ---- material, anchors, loads
        mat = MATERIALS[part.material]
        self.mat = mat
        self.build_axis = AXES[part.build][0]
        self.C = stiffness(mat, self.build_axis)
        self.supports = dict(part.supports)
        self._anchors()
        self._cases()
        self._reach()
        # ---- solver
        active = ~void
        self.fe_solver = Solver(shape, h, self.C, self.supports, active=active,
                                backend=os.environ.get("NOX_TOPO_BACKEND", s.backend), log=log, mg_tol=s.mg_tol)
        self.KE = self.fe_solver.KE
        # ---- reference compliances (full solid)
        full = np.where(void, 0.0, 1.0)
        c_ref, _ = self._compliance(full, need_grad=False)
        bad = [c.name for c, v in zip(self.cases, c_ref) if not v > 0]
        if bad:
            raise ValueError(f"layout: zero compliance of the full solid for {bad}: the load sits on fixed nodes")
        self.c_ref = c_ref
        self.normed = False
        self.warm = False
        self.hist = []
        self.scale = None
        self.vlim = self.volfrac
        self.x = np.full(self.nd, self.volfrac)
        self.mma = MMA(self.nd, 3, move=s.move)

    # ------------------------------------------------------------------------------------------ anchors
    def _anchors(self):
        s, shape, part = self.s, self.shape, self.part
        sol3 = self.solid.reshape(shape)
        if s.erosion not in ("masked", "anchored", "frozen"):
            raise ValueError(f"layout: erosion must be masked, anchored or frozen, not {s.erosion!r}")
        if s.erosion == "frozen":
            self.anchor = self.solid.copy()                     # the classic formulation: drawn cells held whole
        elif s.erosion == "masked":
            # a thick drawn body keeps only the core the solid gate keeps (tools/solid_check.py: the part eroded by
            # clip(0.8 x the body's half thickness, 0.9, 2.0) mm must still hold the body's core); its skin erodes with the
            # design, so a member meeting the body at its rim is seen to come off, as the gate sees it (29 Oct: the rear
            # arm's plate bosses and a tip spoke collar passed the engine's eroded check held whole and failed the gate).
            # A body too thin for a core stays whole (the gate skips it: judged by its own size check).
            din = ndi.distance_transform_edt(sol3) * self.h
            lab, nb = ndi.label(sol3, S26)
            anchor = sol3.copy()
            for b in range(1, nb + 1):
                m = lab == b
                ef = float(np.clip(0.8 * din[m].max(), 0.9, s.gate_thin_mm))
                core = m & (din > ef)
                if core.any():
                    anchor[m] = core[m]
            self.anchor = anchor.ravel()
        else:
            din = ndi.distance_transform_edt(sol3)
            core = sol3 & (din > self.erode_cells + 0.5)
            lab, nb = ndi.label(sol3, S26)
            if nb:
                idx = np.arange(1, nb + 1)
                has = ndi.maximum(core, lab, idx).astype(bool)
                mx = ndi.maximum(din, lab, idx)
                ridge_thr = np.zeros(nb + 1)
                ridge_thr[1:] = np.where(has, np.inf, mx - 0.5)
                core |= sol3 & (din >= ridge_thr[lab])
            sup = np.zeros(self.nel, bool)
            for el in part.support_cells.values():
                sup[el] = True
            sup3 = sup.reshape(shape)
            if sup3.any():
                near = ndi.distance_transform_edt(~sup3) <= self.erode_cells + 1.5
                core |= sol3 & near
            self.anchor = core.ravel()
        self.skin = self.solid & ~self.anchor
        # drawn bodies for the ties: inserts (Part.insert) are bonded in after printing, so they never join two drawn
        # bodies into one (a rod would chain a motor pad to a socket the PETG reaches, and the pad's ties would pass)
        ins3 = getattr(part, "insert_cells", np.zeros(self.nel, bool)).reshape(shape)
        lab, nb = ndi.label(sol3 & ~ins3, S26)
        self.body_lab, self.nbody = lab.ravel(), nb
        if s.erosion == "anchored" or self.skin.any():
            self.log(f"layout: {nb} drawn bodies; anchored cores {int(self.anchor.sum()):,} cells, skins "
                     f"{int(self.skin.sum()):,} cells (eroded with the design); eroded filter: {s.erosion}")
        else:
            self.log(f"layout: {nb} drawn bodies ({int(self.solid.sum()):,} cells), held solid in every realization; "
                     f"eroded filter: {s.erosion}")

    def _node_mask(self, cells):
        nx, ny, nz = self.shape
        m = np.zeros(self.part.nshape, bool)
        c3 = cells.reshape(self.shape)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    m[di:di + nx, dj:dj + ny, dk:dk + nz] |= c3
        return m

    # -------------------------------------------------------------------------------------------- cases
    def _cases(self):
        s, part, h = self.s, self.part, self.h
        nshape = part.nshape
        # nearest anchored node for every node (loads on drawn skins move there in the eroded analysis)
        anc_n = self._node_mask(self.anchor)
        sol_n = self._node_mask(self.solid)
        remap = None
        if anc_n.any() and self.skin.any():
            _, ind = ndi.distance_transform_edt(~anc_n, return_indices=True)
            nid = np.ravel_multi_index(tuple(ind), nshape).ravel()
            move = (sol_n & ~anc_n).ravel()
            remap = np.arange(part.nn)
            remap[move] = nid[move]
        self.remap = remap

        def eroded(F):
            if remap is None:
                return F.copy()
            Fn = F.reshape(-1, 3)
            out = np.zeros_like(Fn)
            np.add.at(out, remap, Fn)
            return out.ravel()

        wg = self.mat.density * 1e-6 * h ** 3 * 9.81                         # N per g of a full element
        self.cases = []
        for lc in part.cases:
            fe = None if lc.accel is None else np.tile(np.asarray(lc.accel, float) * wg / 8.0, 8)
            self.cases.append(_Case(lc.name, lc.F, eroded(lc.F), lc.support, fe, 1.0, False, lc.cls, lc.nodes))
        n_real = len(self.cases)
        # tie cases: every drawn body the main support does not hold
        count = {}
        for lc in part.cases:
            count[lc.support] = count.get(lc.support, 0) + 1
        if not count:
            raise ValueError("layout: no load cases")
        main = max(count, key=count.get)
        fixed_n = part.supports[main].any(1).reshape(nshape)
        nx, ny, nz = self.shape
        origin = np.asarray(part.origin)
        held_b, free_b = [], []
        for b in range(1, self.nbody + 1):
            cells = self.body_lab == b
            if cells.sum() < s.tie_min_cells:
                continue
            (held_b if (self._node_mask(cells) & fixed_n).any() else free_b).append(b)

        def ties(b, support, tag):
            cells = self.body_lab == b
            an = self._node_mask(cells & self.anchor)
            nodes = np.where((an if an.any() else self._node_mask(cells)).ravel())[0]
            P = np.stack(np.unravel_index(nodes, nshape), 1) * h + origin
            c = P.mean(0)
            for ax in range(3):
                F = np.zeros(part.ndof)
                F[3 * nodes + ax] = 1.0 / len(nodes)
                self.cases.append(_Case(f"{tag} b{b} f{'xyz'[ax]}", F, F, support, None, s.tie_weight, True, None,
                                        nodes))
            if s.tie_moments:
                r = P - c
                rr = (r ** 2).sum()
                if rr > 0:
                    for ax in range(3):
                        e = np.zeros(3)
                        e[ax] = 1.0
                        f = np.cross(e, r) / rr
                        F = np.zeros(part.ndof)
                        for q in range(3):
                            F[3 * nodes + q] = f[:, q]
                        self.cases.append(_Case(f"{tag} b{b} m{'xyz'[ax]}", F, F, support, None, s.tie_weight, True,
                                                None, nodes))
        for b in free_b:
            ties(b, main, "tie")
        n_tie = len(self.cases) - n_real
        # gate ties (masked erosion): a held drawn body thinner than the solid gate's erosion keeps no core of its own
        # there, so the gate counts it only where generated material covers it. In the eroded realization such a body
        # is credited by its cover (x_tilde_c = conv(x on design) / conv(design): the density of the design material
        # next to it), and a second tie set holds the largest unheld body instead of the support: every other unheld
        # body must then join it through generated material, not through the thin plate or the ground.
        self.cover = np.zeros(self.nel, bool)
        self.gate_anchor = None
        n_gate = 0
        if s.erosion == "masked" and s.gate_ties and len(free_b) >= 2:
            for b in held_b:
                cells3 = (self.body_lab == b).reshape(self.shape)
                own = float(ndi.distance_transform_edt(np.pad(cells3, 1))[1:-1, 1:-1, 1:-1].max()) * h
                if own <= s.gate_thin_mm + 0.5 * h:
                    self.cover |= cells3.ravel()
            if self.cover.any():
                a = max(free_b, key=lambda b: int((self.body_lab == b).sum()))
                fx = np.zeros((part.nn, 3), bool)
                fx[self._node_mask(self.body_lab == a).ravel()] = True
                self.supports["_gate"] = fx
                self.gate_anchor = a
                for b in free_b:
                    if b != a:
                        ties(b, "_gate", "gate")
                n_gate = len(self.cases) - n_real - n_tie
                self.log(f"layout: gate ties: held bodies thinner than {s.gate_thin_mm} mm ({int(self.cover.sum()):,} "
                         f"cells) count by their cover; {len(free_b) - 1} bodies tied to body {a}")
        # held ties: a body the main support holds on its own patch (a bolt boss on its seat) needs no material to
        # stay put, so a boss the load cases do not lean on is left loose (holder smoke run: one corner boss never
        # joined). With two or more held bodies, a third tie set holds one hub body (the largest unheld body, else the
        # largest held one) and loads every other held body with 3 forces + 3 moments: each must join the hub.
        n_held = 0
        if s.held_ties and len(held_b) >= 2:
            size = lambda b: int((self.body_lab == b).sum())
            hub = max(free_b, key=size) if free_b else max(held_b, key=size)
            fx = np.zeros((part.nn, 3), bool)
            fx[self._node_mask(self.body_lab == hub).ravel()] = True
            self.supports["_hub"] = fx
            for b in held_b:
                if b != hub:
                    ties(b, "_hub", "held")
            n_held = len(self.cases) - n_real - n_tie - n_gate
        self.n_real = n_real
        self.log(f"layout: {n_real} load cases + {n_tie} tie cases ({len(free_b)} bodies tied, {len(held_b)} held by "
                 f"support '{main}') + {n_gate} gate ties + {n_held} held ties")

    # -------------------------------------------------------------------------------------- reachability
    def _reach(self):
        s, shape, h = self.s, self.shape, self.h
        if s.reach_check == "off":
            return
        A = (self.design | self.solid).reshape(shape)
        r = 0.45 * s.min_member_mm / h
        E = ndi.distance_transform_edt(A) > r
        labE, nE = ndi.label(E, S6)
        if nE == 0:
            msg = (f"layout: the design space holds no member of {s.min_member_mm} mm anywhere: the envelope minus the "
                   f"cuts is thinner than the minimum member")
            if s.reach_check == "raise":
                raise ReachabilityError(msg)
            self.log("WARNING " + msg)
            return
        dist, ind = ndi.distance_transform_edt(~E, return_indices=True)
        near_lab = labE[tuple(ind)]
        reach = []
        origin = np.asarray(self.part.origin)
        for b in range(1, self.nbody + 1):
            cells = (self.body_lab == b).reshape(shape)
            if cells.sum() < s.tie_min_cells:
                continue
            ok = cells & (dist <= r + 1.5)
            labs = set(np.unique(near_lab[ok]).tolist()) if ok.any() else set()
            cen = (np.argwhere(cells).mean(0) + 0.5) * h + origin
            reach.append((b, labs, cen.round(1).tolist(), int(cells.sum())))
        if not reach:
            return
        common = set.intersection(*[r_[1] for r_ in reach]) if all(r_[1] for r_ in reach) else set()
        if common:
            self.log(f"layout: reachability OK: every drawn body can be joined by members of {s.min_member_mm} mm")
            return
        sizes = np.bincount(labE.ravel())
        sizes[0] = 0
        main = int(np.argmax(sizes))
        bad = [r_ for r_ in reach if main not in r_[1]]
        msg = ("layout: these drawn bodies cannot be joined to the rest by a member of "
               f"{s.min_member_mm} mm (the design space next to them is narrower, or a cut separates them): "
               + "; ".join(f"body {b} at {c} ({n} cells, reaches {sorted(l) or 'nothing'})" for b, l, c, n in bad)
               + ". Fix the setup: open the design space there, shrink the cut, or draw the link.")
        if s.reach_check == "raise":
            raise ReachabilityError(msg)
        self.log("WARNING " + msg)

    # -------------------------------------------------------------------------------------------- fields
    def filt(self, v):
        return fftconvolve(v.reshape(self.shape), self.ker, mode="same").ravel()

    @property
    def den_e(self):
        """denominator of the drawn-masked eroded filter: kernel weight of the non-drawn cells (voids and the grid
        outside included), 1 - conv(drawn), clamped at 0.05 of the kernel sum (1)."""
        if getattr(self, "_den_e", None) is None:
            self._den_e = np.maximum(1.0 - self.filt(self.solid.astype(float)), 0.05)
        return self._den_e

    def symmetrise(self, v):
        if not self.sym_axes:
            return v
        a = v.reshape(self.shape)
        for ax in self.sym_axes:
            a = 0.5 * (a + np.flip(a, ax))
        return a.ravel()

    def fields(self, x, beta):
        xh = self.solid.astype(float)
        xh[self.design] = x
        xt = self.filt(xh)
        ee, en, ed = self.s.eta
        xte = xt
        if self.s.erosion == "masked":
            xd = np.zeros(self.nel)
            xd[self.design] = x
            xte = self.filt(xd) / self.den_e
        He, dHe = project(xte, beta, ee)
        dHc = None
        if self.cover.any():
            xd = np.zeros(self.nel)
            xd[self.design] = x
            xc = self.filt(xd) / self.den_c
            Hc, dHc = project(xc, beta, ee)
        Hn, dHn = project(xt, beta, en)
        Hd, dHd = project(xt, beta, ed)
        rho_e = np.where(self.anchor, 1.0, np.where(self.void, 0.0, He))
        dHs = None
        if self.s.erosion == "masked" and self.skin.any():
            Hs, dHs = project(xt, beta, ee)
            rho_e = np.where(self.skin, Hs, rho_e)
        # gate realization (gate ties only): the thin held bodies count by their cover; everywhere else = rho_e
        rho_g = None if dHc is None else np.where(self.cover, Hc, rho_e)
        rho_n = np.where(self.solid, 1.0, np.where(self.void, 0.0, Hn))
        rho_d = np.where(self.solid, 1.0, np.where(self.void, 0.0, Hd))
        return dict(xt=xt, xte=xte, dHc=dHc, dHs=dHs, rho_e=rho_e, rho_g=rho_g, rho_n=rho_n, rho_d=rho_d, dHe=dHe, dHn=dHn, dHd=dHd, Hn=Hn)

    def _back(self, g_xt):
        """d/dx_tilde -> d/dx on the design cells (filter transpose; the kernel is symmetric)."""
        return self.filt(g_xt)[self.design]

    def _back_e(self, g_xte):
        """d/dx_tilde_e -> d/dx: x_tilde_e = conv(x_d) / den with den independent of x, so the transpose carries the
        1 / den weight before the (symmetric) convolution."""
        if self.s.erosion == "masked":
            return self.filt(g_xte / self.den_e)[self.design]
        return self._back(g_xte)

    def grad_eroded(self, f, g_rho):
        """dJ/d rho (nel,) -> dJ/dx on the design cells through the eroded projection and filter. g_rho is one array
        (one realization) or (g on rho_e, g on rho_g) from _grad_rho_e when gate ties run on the gate realization."""
        g_g = None
        if isinstance(g_rho, tuple):
            g_rho, g_g = g_rho
        m_e = self.design | (self.skin if self.s.erosion == "anchored" else np.zeros(self.nel, bool))
        g_all = g_rho if g_g is None else g_rho + g_g                          # rho_g = rho_e outside the cover
        d = self._back_e(np.where(m_e, g_all * f["dHe"], 0.0))
        if g_g is not None:
            d = d + self.filt(np.where(self.cover, g_g * f["dHc"], 0.0) / self.den_c)[self.design]
        if f.get("dHs") is not None:
            d = d + self.filt(np.where(self.skin, g_all * f["dHs"], 0.0))[self.design]     # skins: nominal filter, symmetric
        return d

    @property
    def den_c(self):
        """denominator of the cover filter: kernel weight of the design cells, clamped at 0.05."""
        if getattr(self, "_den_c", None) is None:
            self._den_c = np.maximum(self.filt(self.design.astype(float)), 0.05)
        return self._den_c

    # ---------------------------------------------------------------------------------------- compliance
    def _compliance(self, rho_e, need_grad=True, rho_g=None):
        """compliance of every case. Gate ties run on rho_g (the gate realization) when it is given, the rest on
        rho_e (a held thin body stays whole there, so the real cases and the ordinary ties keep it attached)."""
        s = self.s
        E = (s.emin + (1.0 - s.emin) * rho_e ** s.penal) * self.stiff
        g = self.fe_solver.grid
        cases = self.cases
        F = np.zeros((self.part.ndof, len(cases)))
        for q, c in enumerate(cases):
            F[:, q] = c.Fe
            if c.fe is not None:
                r = (rho_e[:, None] * c.fe[None, :])[None]
                F[:, q] += g.to_flat(g.scatter(r))[:, 0]
        gate = np.array([c.support == "_gate" for c in cases])
        tie = np.array([c.tie for c in cases]) & self.has_inserts
        if (rho_g is None or not gate.any()) and not tie.any():
            U = self.fe_solver.solve(E, F, [c.support for c in cases], keys=[c.name for c in cases])
        else:
            # gate ties on the gate realization; with inserts, every tie case without them: the printed part must hold
            # its drawn bodies together by itself (an insert is bonded in after printing)
            U = np.zeros_like(F)
            rg = rho_e if (rho_g is None or not gate.any()) else rho_g
            Eg = (s.emin + (1.0 - s.emin) * rg ** s.penal) * self.stiff_tie
            Et = (s.emin + (1.0 - s.emin) * rho_e ** s.penal) * self.stiff_tie
            for m, EE in ((~gate & ~tie, E), (~gate & tie, Et), (gate, Eg)):
                idx = np.where(m)[0]
                if len(idx) == 0:
                    continue
                U[:, idx] = self.fe_solver.solve(EE, F[:, idx], [cases[q].support for q in idx],
                                                 keys=[cases[q].name for q in idx])
        comp = (F * U).sum(0)
        if not need_grad:
            return comp, None
        return comp, (U, E)

    def _objective(self, comp):
        s = self.s
        r = comp / self.c_ref
        w = np.array([c.w for c in self.cases])
        P = s.pmean
        sw = w.sum()
        J = (np.sum(w * r ** P) / sw) ** (1.0 / P)
        dJdc = J ** (1.0 - P) * w * r ** (P - 1.0) / sw / self.c_ref
        return J, dJdc, r

    def _grad_rho_e(self, rho_e, U, dJdc, rho_g=None):
        """dJ / d rho_e (nel,), accumulated case by case; with rho_g, (dJ / d rho_e, dJ / d rho_g) where the gate
        ties contribute to the second."""
        s = self.s
        g = np.zeros(self.nel)
        gg = np.zeros(self.nel) if rho_g is not None else None
        dE = s.penal * np.maximum(rho_e, 1e-12) ** (s.penal - 1.0) * (1.0 - s.emin) * self.stiff
        dEt = dE if not self.has_inserts else dE / self.stiff * self.stiff_tie
        dEg = None if rho_g is None else s.penal * np.maximum(rho_g, 1e-12) ** (s.penal - 1.0) * (1.0 - s.emin) * self.stiff_tie
        CH = 4
        for a in range(0, len(self.cases), CH):
            sl = list(range(a, min(a + CH, len(self.cases))))
            fe = [self.cases[q].fe for q in sl]
            en, fo = self.fe_solver.element_energy(U[:, sl], fe=fe)
            for j, q in enumerate(sl):
                if dJdc[q] == 0.0:
                    continue
                on_g = gg is not None and self.cases[q].support == "_gate"
                d = -(dEg if on_g else (dEt if self.cases[q].tie else dE)) * en[j]
                if fe[j] is not None:
                    d = d + 2.0 * fo[j]
                if on_g:
                    gg += dJdc[q] * d
                else:
                    g += dJdc[q] * d
        return g if gg is None else (g, gg)

    # --------------------------------------------------------------------------------- geometric constraints
    def _geo_masks(self):
        """design cells where each indicator can be met at all. Solid: x_tilde with every design cell at 1 reaches
        eta_e + geo_margin (round 1 counted design cells beside cuts whose x was already 1: bar_half topo-v1 read +294
        at beta 32 from 977 such cells, all at x = 1, 1 to 2 cells off a cut). Void: x_tilde with every design cell at 0
        (the drawn bodies' spill alone) stays under eta_d - geo_margin."""
        if getattr(self, "_gmask", None) is None:
            ee, en, ed = self.s.eta
            mg = self.s.geo_margin
            d3 = self.design.reshape(self.shape)
            hi = self.filt((self.design | self.solid).astype(float)).reshape(self.shape)
            lo = self.filt(self.solid.astype(float)).reshape(self.shape)
            self._gmask = (d3 & (hi >= ee + mg)).astype(float), (d3 & (lo <= ed - mg)).astype(float)
        return self._gmask

    def _geometric(self, f, beta):
        """Zhou, Lazarov, Wang, Sigmund (2015): (g_s, g_v) and d/dx_tilde of each, over the design cells where each
        can be met (_geo_masks), normalised by the number of design cells."""
        xt = f["xt"].reshape(self.shape)
        rb = f["Hn"].reshape(self.shape)
        drb = f["dHn"].reshape(self.shape)
        ee, en, ed = self.s.eta
        c = self.R ** 4
        pad = np.pad(xt, 1)
        D = [(np.take(pad, range(2, pad.shape[a]), axis=a) - np.take(pad, range(0, pad.shape[a] - 2), axis=a)) / 2.0
             for a in range(3)]
        D = [d[tuple(slice(1, -1) if b != a else slice(None) for b in range(3))] for a, d in enumerate(D)]
        G = D[0] ** 2 + D[1] ** 2 + D[2] ** 2
        ex = np.exp(-c * G)
        ms, mv = self._geo_masks()
        n = float(self.nd)
        out = []
        for kind in ("s", "v"):
            msk = ms if kind == "s" else mv
            if kind == "s":
                ind = rb * ex
                m = np.minimum(xt - ee, 0.0)
                dind_drb = ex
                dm = (xt < ee).astype(float)
            else:
                ind = (1.0 - rb) * ex
                m = np.minimum(ed - xt, 0.0)
                dind_drb = -ex
                dm = -(xt > ed).astype(float)
            ind = ind * msk
            val = float((ind * m * m).sum() / n)
            # d/dxt: through rb (projection), through m, and through ex (gradient stencil)
            dg = (msk * dind_drb * drb * m * m + ind * 2.0 * m * dm) / n
            wgt = msk * (rb if kind == "s" else 1.0 - rb) * (-c) * ex * m * m / n
            for a in range(3):
                wd = np.pad(wgt * D[a], [(1, 1) if b == a else (0, 0) for b in range(3)])
                lo = np.take(wd, range(0, wd.shape[a] - 2), axis=a)
                hi = np.take(wd, range(2, wd.shape[a]), axis=a)
                dg += lo - hi
            out.append((val, dg.ravel()))
        return out

    # ---------------------------------------------------------------------------------------------- run
    def _save(self, path, it, bi, since, stopped=""):
        tmp = path + ".tmp.npz"
        st = self.mma.state()
        np.savez_compressed(tmp, x=self.x, it=it, bi=bi, since=since, vlim=self.vlim,
                            scale=self.scale if self.scale else 0.0, hist=np.array(self.hist, float),
                            xold1=st["xold1"], xold2=st["xold2"], low=st["low"], upp=st["upp"], k=st["k"],
                            stopped=stopped, c_ref=self.c_ref, normed=self.normed)
        os.replace(tmp, path)

    def _load(self, path):
        ck = np.load(path, allow_pickle=False)
        self.x = ck["x"].copy()
        self.vlim = float(ck["vlim"])
        self.scale = float(ck["scale"]) or None
        self.hist = [tuple(r) for r in ck["hist"]]
        if "c_ref" in ck.files and len(ck["c_ref"]) == len(self.c_ref):
            self.c_ref = ck["c_ref"].copy()
            self.normed = bool(ck["normed"])
        self.mma.load_state(dict(xold1=ck["xold1"], xold2=ck["xold2"], low=ck["low"], upp=ck["upp"], k=ck["k"]))
        return int(ck["it"]), int(ck["bi"]), int(ck["since"]), str(ck["stopped"])

    def init_from(self, other_x_full, other_part):
        """warm start from another layout (e.g. the same part on a coarser grid): sample its design field at this
        grid's cell centres (nearest cell)."""
        P = self.part.cell_centres()[self.design]
        o = np.asarray(other_part.origin)
        ijk = np.floor((P - o) / other_part.h).astype(int)
        for a in range(3):
            ijk[:, a] = np.clip(ijk[:, a], 0, other_part.shape[a] - 1)
        flat = np.ravel_multi_index(ijk.T, other_part.shape)
        self.x = np.clip(np.asarray(other_x_full)[flat], 0.0, 1.0)
        self.warm = True
        self.log(f"layout: warm start from a {other_part.shape} grid (h {other_part.h}): mean x {self.x.mean():.3f}")

    def x_hat(self):
        """the design field on the whole grid with drawn cells at 1 (for warm-starting another grid)."""
        xh = self.solid.astype(float)
        xh[self.design] = self.symmetrise_design(self.x)
        return xh

    def run(self, maxit=400, checkpoint=None, resume=None, beta_start=0, beta_stop=None, log_every=1,
            on_checkpoint=None):
        s, log = self.s, self.log
        t0 = time.time()
        it, bi, since = 0, int(beta_start), 0
        if resume and os.path.exists(resume):
            it, bi, since, stopped = self._load(resume)
            log(f"layout: resumed from {resume} at it {it}, beta {s.betas[bi]}{' (already stopped: ' + stopped + ')' if stopped else ''}")
            if stopped:
                self.stopped = stopped
                return self.final(bi)
        last_bi = len(s.betas) - 1 if beta_stop is None else int(beta_stop)
        if not (resume and os.path.exists(resume)):
            self._reset_vlim(s.betas[bi])
        geo_eps = s.geo_eps or (1e-6 if self.R < 5 else 1e-7)
        self.stopped = "maxit"
        while it < maxit:
            it += 1
            since += 1
            beta = float(s.betas[bi])
            ti = time.time()
            x = self.symmetrise_design(self.x)
            f = self.fields(x, beta)
            comp, (U, E) = self._compliance(f["rho_e"], rho_g=f["rho_g"])
            if s.case_norm == "frame" and not self.normed and (self.warm or bi > int(beta_start)):
                self._renorm(comp)
            J, dJdc, r = self._objective(comp)
            if self.scale is None:
                self.scale = 10.0 / J
            g_rho = self._grad_rho_e(f["rho_e"], U, dJdc, rho_g=f["rho_g"])
            del U
            df0 = self.scale * self.grad_eroded(f, g_rho)
            vd = float(f["rho_d"][self.design].mean())
            vn = float(f["rho_n"][self.design].mean())
            gvol = vd / self.vlim - 1.0
            dvol = self._back(np.where(self.design, f["dHd"], 0.0) / (self.nd * self.vlim))
            fval = [gvol, -1.0, -1.0]
            dfdx = [dvol, np.zeros(self.nd), np.zeros(self.nd)]
            geo = None
            if s.geometric and beta >= s.geo_beta:
                geo = self._geometric(f, beta)
                for q, (val, dg) in enumerate(geo):
                    fval[1 + q] = val / geo_eps - 1.0
                    dfdx[1 + q] = self._back(dg) / geo_eps
            df0 = self.symmetrise_design(df0)
            dfdx = [self.symmetrise_design(d) for d in dfdx]
            xnew = self.mma.step(x, self.scale * J, df0, np.array(fval), np.array(dfdx))
            ch = float(np.abs(xnew - x).max())
            self.x = xnew
            grey = float(((f["rho_n"][self.design] > 0.1) & (f["rho_n"][self.design] < 0.9)).mean())
            worst = int(np.argmax(r))
            dt = time.time() - ti
            self.hist.append((it, beta, self.scale * J, gvol, vn, grey, ch, dt,
                              fval[1] if geo else 0.0, fval[2] if geo else 0.0))
            if it % log_every == 0:
                st = self.fe_solver.stats[-1] if self.fe_solver.stats else None
                log(f"it {it:3d} beta {beta:4.0f} J {self.scale * J:8.4f} vol {vn:.4f} (dil {vd:.4f} g {gvol:+.3f}) "
                    f"grey {grey:5.1%} ch {ch:.3f} worst {self.cases[worst].name[:28]} {r[worst]:.3g}"
                    + (f" geo {fval[1]:+.2f}/{fval[2]:+.2f}" if geo else "")
                    + (f" cg {st[1]}" if st else "") + f" {dt:.1f}s")
            if it % 5 == 0 and vn > 0:
                new = self.volfrac * vd / vn
                new = self.vlim * (new / self.vlim) ** s.vol_gain
                self.vlim = float(np.clip(new, self.volfrac, 3.0 * self.volfrac))
            objs = [hh[2] for hh in self.hist[-4:]]
            settled = since >= s.min_stage and len(objs) == 4 and abs(objs[-1] - objs[0]) <= 0.01 * abs(objs[-1]) \
                and abs(gvol) < 0.02
            elapsed = time.time() - t0
            hurry = s.time_limit_s and elapsed > 0.75 * s.time_limit_s
            if bi < last_bi and (settled or since >= s.max_stage or (hurry and since >= 3)):
                self._connect_report(f, beta)
                bi += 1
                since = 0
                self.mma.reset()
                self._reset_vlim(s.betas[bi])
            elif bi >= last_bi and beta_stop is not None and (settled or since >= s.max_stage):
                self.stopped = "beta_stop"
                break
            elif bi >= last_bi and ((grey <= s.grey_stop and settled) or since >= s.max_stage):
                self.stopped = "grey" if grey <= s.grey_stop else "max_stage"
                break
            if checkpoint and it % s.checkpoint_every == 0:
                self._save(checkpoint, it, bi, since)
                if on_checkpoint:
                    on_checkpoint(it)
            if s.time_limit_s and elapsed > s.time_limit_s:
                self.stopped = "time"
                break
        if checkpoint:
            self._save(checkpoint, it, bi, since, self.stopped)
        log(f"layout: stopped ({self.stopped}) at it {it}, beta {s.betas[bi]}, {time.time() - t0:.0f} s")
        return self.final(bi)

    def _renorm(self, comp):
        """case_norm "frame": every case from now on counts by its compliance over its value on this design."""
        r = comp / self.c_ref
        self.c_ref = comp.copy()
        self.normed = True
        self.scale = None
        self.mma.reset()
        o = np.argsort(-r)[:3]
        self.log("layout: cases renormalised on this frame (case_norm frame); largest solid ratios were "
                 + ", ".join(f"{self.cases[q].name} {r[q]:.3g}" for q in o))

    def _reset_vlim(self, beta):
        """V* at the fixed point of the volume-limit update for the current design at this beta. A beta step (or a warm
        start from the coarse grid) changes the dilated / nominal ratio at once; left to the damped update, V* lags
        for ~15 iterations, the dilated constraint reads up to +70 % violated and MMA strips the design."""
        f = self.fields(self.symmetrise_design(self.x), float(beta))
        vd, vn = float(f["rho_d"][self.design].mean()), float(f["rho_n"][self.design].mean())
        if vn > 0:
            self.vlim = float(np.clip(self.volfrac * vd / vn, self.volfrac, 3.0 * self.volfrac))

    def symmetrise_design(self, v):
        if not self.sym_axes:
            return v
        full = np.zeros(self.nel)
        full[self.design] = v
        return self.symmetrise(full)[self.design]

    def _connect_report(self, f, beta):
        """eroded design (projected > 0.5, anchors) as one face-connected body holding every anchored core?"""
        B = ((f["rho_e"] > 0.5) | self.anchor).reshape(self.shape)
        lab, n = ndi.label(B, S6)
        if n == 0:
            return
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        main = int(np.argmax(sizes))
        anc = self.anchor.reshape(self.shape)
        off = np.unique(lab[anc & (lab != main)])
        off = off[off > 0]
        self.log(f"layout: beta {beta:.0f} done: eroded design {n} bodies, main {sizes[main] / sizes.sum():.1%}; "
                 f"drawn {'cores' if self.s.erosion == 'anchored' else 'bodies'} outside the main body: {len(off)}")

    def final(self, bi):
        beta = float(self.s.betas[bi])
        x = self.symmetrise_design(self.x)
        f = self.fields(x, beta)
        self._connect_report(f, beta)
        self.result = dict(x=x, xt=f["xt"], rho_n=f["rho_n"], rho_e=f["rho_e"], beta=beta)
        return self.result

    def save(self, path):
        r = self.result
        xf = np.zeros(self.nel)
        xf[self.design] = r["x"]
        np.savez_compressed(path, x_full=xf, xt=r["xt"], rho_n=r["rho_n"], rho_e=r["rho_e"], solid=self.solid,
                            void=self.void, design=self.design, anchor=self.anchor, shape=np.array(self.shape),
                            h=self.h, origin=np.array(self.part.origin), R=self.R, eta=np.array(self.s.eta),
                            beta=r["beta"], hist=np.array(self.hist, float))

    def meta(self):
        return dict(settings=asdict(self.s), R_mm=self.R_mm, R_cells=self.R, calibration=self.cal,
                    volfrac=self.volfrac, nd=self.nd, sym_axes=self.sym_axes, n_real=self.n_real,
                    n_ties=len(self.cases) - self.n_real, backend=self.fe_solver.backend,
                    iterations=len(self.hist), stopped=getattr(self, "stopped", None))


@dataclass
class _Case:
    name: str
    F: np.ndarray        # nominal nodal loads
    Fe: np.ndarray       # loads for the eroded analysis (moved to anchored nodes)
    support: str
    fe: np.ndarray       # (24,) body force per full element, or None
    w: float
    tie: bool
    cls: str
    nodes: np.ndarray
