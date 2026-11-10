#!/usr/bin/env python3
"""Nonlinear shell FE study of the ceiling probe's double tape-spring mast (CalculiX 2.21).

Run:  python tools/probe_mast_fe.py [--work DIR] [--jobs N] [--only transition,mast,buckle,mesh,flex] [--force]
Out:  systems/probe_sim/fe.json, systems/probe_sim/fe_transition.png, fe_fold.png, fe_modes.png
      (decks, ccx logs, .dat/.frd stay in the scratch dir, default $TMPDIR/probe_mast_fe; finished runs are
      reused unless --force).

Model
-----
* Two identical steel tape-measure blades, E 200 GPa, nu 0.3, rho 7850 kg/m3, linear elastic (no plasticity: a
  fold that would yield the steel is still reported as the elastic peak).  Flat width w = 19.0 mm, thickness t
  (nominal 0.13 mm), natural transverse cup radius R (nominal 15 mm).  Units: mm, N, s, MPa.
* Natural cross-section of a blade: circular arc of radius R, arc length w (chord 2R sin(w/2R), sag
  R(1 - cos(w/2R))).  Mast axis z, lens thickness direction x, width direction y.  Blade A bows to +x, blade B
  to -x, concave sides facing, their mid-surface edges coincident on the plane x = 0 at y = +-R sin(w/2R)
  (the "()" lens of a bi-STEM / collapsible tube mast).  Mid-surface edges coincide; the real edges touch at
  their outer faces, an offset of t/2 that is ignored (0.065 mm against a 2.9 mm sag).
* Elements: one layer of C3D20R (20-node quadratic, reduced integration) bricks through the thickness, i.e.
  the solid-shell that CalculiX itself expands S8R shells into.  Written out explicitly because CalculiX's
  shell expansion ties shell nodes to a *RIGID BODY / rotational SPC through "knots" whose residual on the
  rotation node never converged under NLGEOM (tried first, with S8R; the explicit bricks converge).  12
  elements across the width, 2 mm along (transition model: 1 mm along); mesh check 18 across x 1.33 mm.
* Geometric nonlinearity (NLGEOM) in every nonlinear step.  Static, quasi-static displacement control (no
  arc-length: CalculiX 2.21 has no *STATIC,RIKS), so a fold that snaps back is caught at its peak and the
  run stops (or creeps on through the snap) after it.

Boundary conditions
-------------------
* Base (z = 0, top of the 32 mm throat): every node of both blades' base face (both brick faces) fixed:
  clamped, in the natural lens shape.  Flexible-base variant: only the base mid-surface line held (corner
  mid-nodes fixed, the mean of each pair of face mid-side nodes fixed): a hinge along the base edge of each
  blade, i.e. the rotation about the local edge tangent (y at the crown) free, the section's shape and
  position held.  It frees the rotation about y at the crown, and with it the edge-line rotation elsewhere
  across the arc (an upper bound on the throat's give).
* Tip (z = 150): a rigid printed head with reference node HEAD at (0, 0, 150).  Blade A's tip row is a
  *RIGID BODY on HEAD (all DOF).  Blade B's tip row follows the head in x and y and is free in z (the +-3 mm
  slot; the slot stop is not modelled, its slide is reported): linear *EQUATIONs
      u_x(B) = u_x(HEAD) - theta_z * y_B ,   u_y(B) = u_y(HEAD) + theta_z * x_B ,
  (theta from the rigid body's rotation node).  The tilt terms of the head rotation are dropped: a tip node
  at x_B <= 2.9 mm off the head axis moves x_B(1 - cos theta) <= 0.13 mm at theta = 0.3 rad (assumption).
* Head mass is not in the fold / stiffness runs (no gravity); it enters as the comparison loads: weight
  0.59 N (60.5 g), and a 1 g lateral load at the head CG 40 mm above the tip = 0.59 x 0.190 = 0.112 N m at
  the base.

Edge contact between the blades (the three mast configurations)
---------------------------------------------------------------
* unbonded: frictionless node-to-node unilateral contact with CalculiX GAPUNI gap elements between each pair
  of coincident edge nodes of A and B (both edges, every node row except the base row (both clamped) ),
  clearance 0, direction fixed +x (lens thickness direction), penalty stiffness 1e3 N/mm.  The gap resists
  only A's edge moving to -x relative to B's (interpenetration); the edges slide freely in y and z and can
  separate.  The direction is global and does not rotate with the mast (assumption: the fold localises at
  the clamped base, where the mast is still near vertical).  A pair whose edges slide apart in y keeps
  acting as if the edges were still over each other (node-to-node, not node-to-surface).
* tied: the same edge pairs tied in all three translations (*EQUATION), the stiff bound.
* single: blade A alone, the soft reference.
* The gap elements are nonlinear; in the linear *BUCKLE step the unbonded mast is the edges tied in x only
  (the closed-gap tangent), and the nonlinear compression run uses the real gaps.

Studies
-------
(a) Flat-to-lens transition: one blade, length 120, natural cup R.  At z = 0 the first 1 mm (node rows at
    z = 0, 0.5, 1.0) is forced flat by prescribing every node's x displacement onto the plane x = 0 (z held
    at z = 0, y held at the centre node, everything else free; NLGEOM, ramped).  Sag of the deformed
    section (largest distance of the section from its edge-to-edge chord) against z; transition lengths
    where the sag first reaches 90 % / 95 % of the natural sag.  t in {0.11, 0.13, 0.19} x R in {12.7, 15,
    20, 25.4}.  Then the pair (lens, GAPUNI edges) flattened together at t 0.13, R 15.
(b) The extended mast, 150 mm, unbonded / tied / single:
    * lateral stiffness at the tip in +x and +y: the first increment of the fold run (0.25 mm in x, 0.02 mm
      in y prescribed head displacement), k = F/u;
    * fold: head displacement prescribed up to 25 mm in +-x, 6 mm in +-y (other head DOF free, a 0.005 N
      transverse force as imperfection); base moment M = F (150 + u_z,head) from global equilibrium (the only
      external load is at the head); fold moment = peak M.  Without arc-length the static path ends where
      the fold snaps back (increment cut to the 1e-7 minimum): the peak is then the last converged
      equilibrium, reported with snapped = False and complete = False (an equilibrium limit, not a bound).
    * axial buckling: linear *BUCKLE (head free, unit compression on HEAD), then a nonlinear compression in
      displacement control with the first mode added to the geometry at 0.15 mm (L/1000) amplitude;
      nonlinear load = peak head reaction.
    * mesh check: single and unbonded +-x folds with 18 across x 1.33 mm along over the first 48 mm (and
      single +x at 2x: 24 across x 1 mm);
      flexible base: unbonded, tied and single in +-x (the base hinge frees only the x-bending rotation).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "systems", "probe_sim")

E, NU, RHO = 200000.0, 0.3, 7.85e-9          # MPa, -, t/mm^3
W = 19.0                                      # blade flat width, mm
L_MAST = 150.0                                # free length out of the throat, mm
T_NOM, R_NOM = 0.13, 15.0
HEAD_W = 0.0605 * 9.81                        # head weight, N
HEAD_CG = 40.0                                # head CG above the tip, mm
THROAT = 32.0                                 # guide length that must re-form the lens, mm
GAP_K = 1.0e3                                 # GAPUNI penalty stiffness, N/mm
HEAD, HEADROT = 900001, 900002                # head reference node and its rotation node
B_OFF = 200000                                # node / element id offset of blade B
# prescribed head travel in the fold runs (dof: total, first increment, largest increment), mm: x is the soft
# lens-thickness direction, y the stiff width direction
U_FOLD = {1: (25.0, 0.25, 0.5), 2: (6.0, 0.02, 0.1)}
F_IMP = 0.005                                 # transverse imperfection force in the fold runs, N


# ----------------------------------------------------------------------------------------------- geometry
def sag_nat(R):
    return R * (1.0 - math.cos(W / (2 * R)))


class Blade:
    """One blade as a single layer of C3D20R bricks through the thickness (a solid shell).

    The mid-surface grid is (2n+1) x (2m+1) nodes, i across the width, j along z, without the (odd, odd) ones.
    Layers: 0 at -t/2, 2 at +t/2 along q = (d/dphi) x z (unit), 1 the mid-surface (only at even i and even j,
    the bricks' through-thickness mid-side nodes).  sign +1: blade A (bows to +x), -1: blade B.
    Node id = off + 1 + i + (2n+1) j + LAY * layer, element id = off + 1 + k.
    """
    LAY = 20000

    def __init__(self, R, zc, n, t, sign=1, off=0):
        self.n, self.m, self.off, self.sign = n, len(zc) - 1, off, sign
        a = W / (2 * R)
        zc = np.asarray(zc, float)
        zr = np.empty(2 * self.m + 1)
        zr[0::2] = zc
        zr[1::2] = 0.5 * (zc[:-1] + zc[1:])
        self.z = zr
        phi = np.linspace(-a, a, 2 * n + 1)
        self.xs = sign * R * (np.cos(phi) - math.cos(a))
        self.ys = R * np.sin(phi)
        tx, ty = -sign * np.sin(phi), np.cos(phi)          # d/dphi of (x, y), unit
        self.qx, self.qy = ty, -tx                         # q = tangent x z
        self.nodes = {}
        for j in range(2 * self.m + 1):
            for i in range(2 * n + 1):
                if i % 2 and j % 2:
                    continue
                for lay in (0, 1, 2):
                    if lay == 1 and (i % 2 or j % 2):
                        continue
                    h = (lay - 1) * t / 2
                    self.nodes[self.id(i, j, lay)] = (self.xs[i] + h * self.qx[i], self.ys[i] + h * self.qy[i], zr[j])
        self.elems = []
        for j in range(self.m):
            for i in range(n):
                I, J = 2 * i, 2 * j
                c = [(I, J), (I + 2, J), (I + 2, J + 2), (I, J + 2)]
                md = [(I + 1, J), (I + 2, J + 1), (I + 1, J + 2), (I, J + 1)]
                nn = ([self.id(*p, 0) for p in c] + [self.id(*p, 2) for p in c] + [self.id(*p, 0) for p in md]
                      + [self.id(*p, 2) for p in md] + [self.id(*p, 1) for p in c])
                self.elems.append((off + 1 + len(self.elems), *nn))

    def id(self, i, j, lay=1):
        return self.off + 1 + i + (2 * self.n + 1) * j + self.LAY * lay

    def row(self, j):
        """Every node (all layers) of node row j."""
        return [k for k in (self.id(i, j, lay) for i in range(2 * self.n + 1) for lay in (0, 1, 2))
                if k in self.nodes]

    def mid(self, i, j):
        """Mid-surface position (mean of the faces) of grid point (i, j)."""
        return (np.add(self.nodes[self.id(i, j, 0)], self.nodes[self.id(i, j, 2)])) / 2

    def edge_rows(self):
        """(i, j) of the mid-surface edge nodes (i = 0 and 2n, even j): the contact / tie points."""
        return [(i, j) for j in range(0, 2 * self.m + 1, 2) for i in (0, 2 * self.n)]


def zgrid(L, dz):
    m = max(1, int(round(L / dz)))
    return np.linspace(0.0, L, m + 1)


# ------------------------------------------------------------------------------------------- deck writer
def nset(name, ids):
    out = [f"*NSET,NSET={name}"]
    ids = list(ids)
    for k in range(0, len(ids), 12):
        out.append(",".join(str(x) for x in ids[k:k + 12]))
    return out


def mesh_cards(blades, extra_nodes=()):
    L = ["*HEADING", "probe mast tape-spring FE (tools/probe_mast_fe.py)", "*NODE"]
    for b in blades:
        L += [f"{k},{x:.9g},{y:.9g},{z:.9g}" for k, (x, y, z) in b.nodes.items()]
    L += [f"{k},{x:.9g},{y:.9g},{z:.9g}" for k, (x, y, z) in extra_nodes]
    for b in blades:
        es = "EA" if b.sign > 0 else "EB"
        L.append(f"*ELEMENT,TYPE=C3D20R,ELSET={es}")
        for e in b.elems:                                  # 21 entries: 16 on the first line, 5 continued
            L += [",".join(str(v) for v in e[:16]) + ",", ",".join(str(v) for v in e[16:])]
        L += [f"*SOLID SECTION,ELSET={es},MATERIAL=STEEL"]
    L += ["*MATERIAL,NAME=STEEL", "*ELASTIC", f"{E},{NU}", "*DENSITY", f"{RHO}"]
    return L


def gap_cards(A, B, rows):
    """GAPUNI between coincident edge nodes: node 1 = B, node 2 = A, direction +x (opens when A moves +x)."""
    L = ["*ELEMENT,TYPE=GAPUNI,ELSET=GAPS"]
    eid = 900100
    for (i, j) in rows:
        L.append(f"{eid},{B.id(i, j)},{A.id(i, j)}")
        eid += 1
    L += ["*GAP,ELSET=GAPS", f"0.,1.,0.,0.,,{GAP_K}"]
    return L


def tie_cards(A, B, rows, dofs=(1, 2, 3)):
    L = ["*EQUATION"]
    for (i, j) in rows:
        for d in dofs:
            L += ["2", f"{A.id(i, j)},{d},1.,{B.id(i, j)},{d},-1."]
    return L


def run_ccx(job, deck, work, force=False, timeout=1500):
    """Write deck to work/job/job.inp and run ccx there (1 thread).  Returns (dir, wall seconds, ok)."""
    d = os.path.join(work, job)
    os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, job + ".inp")
    stamp = os.path.join(d, "done.json")
    text = "\n".join(deck) + "\n"
    if not force and os.path.exists(stamp) and open(inp).read() == text:
        s = json.load(open(stamp))
        return d, s["sec"], s["ok"]
    open(inp, "w").write(text)
    if os.path.exists(stamp):
        os.remove(stamp)
    env = dict(os.environ, OMP_NUM_THREADS="1", CCX_NPROC_EQUATION_SOLVER="1")
    t0 = time.time()
    try:
        with open(os.path.join(d, "ccx.log"), "w") as log:
            p = subprocess.run(["ccx", "-i", job], cwd=d, stdout=log, stderr=subprocess.STDOUT,
                               timeout=timeout, env=env)
        tail = open(os.path.join(d, "ccx.log")).read()[-3000:]
        ok = p.returncode == 0 and "Job finished" in tail and "*ERROR" not in tail
    except subprocess.TimeoutExpired:
        ok = False
    sec = time.time() - t0
    json.dump({"sec": sec, "ok": ok}, open(stamp, "w"))
    print(f"  {job:34s} {'ok ' if ok else 'END'} {sec:7.1f} s", flush=True)
    return d, sec, ok


# ------------------------------------------------------------------------------------------------ parsing
HDR = re.compile(r"^\s*(displacements|forces) \(.*\) for set (\S+) and time\s+(\S+)")


def parse_dat(path):
    """{(kind, set): [(time, {node: (v1, v2, v3)}), ...]} from *NODE PRINT blocks of a .dat file."""
    out = {}
    if not os.path.exists(path):
        return out
    lines = open(path).read().splitlines()
    k = 0
    while k < len(lines):
        mm = HDR.match(lines[k])
        if not mm:
            k += 1
            continue
        key = (mm.group(1), mm.group(2).upper())
        tm = float(mm.group(3))
        k += 2
        blk = {}
        while k < len(lines) and lines[k].strip():
            p = lines[k].split()
            blk[int(p[0])] = tuple(float(v) for v in p[1:4])
            k += 1
        out.setdefault(key, []).append((tm, blk))
    return out


def parse_buckle(path):
    txt = open(path).read() if os.path.exists(path) else ""
    i = txt.find("B U C K L I N G")
    if i < 0:
        return []
    f = []
    for ln in txt[i:].splitlines()[1:]:
        p = ln.split()
        if len(p) == 2 and p[0].isdigit():
            f.append(float(p[1]))
        elif f and not p:
            break
    return f


def parse_frd_modes(path):
    """List of {node: (ux, uy, uz)} for every DISP block of an .frd file (buckling mode shapes)."""
    modes, cur, inblk = [], None, False
    if not os.path.exists(path):
        return modes
    for ln in open(path):
        if ln.startswith(" -4  DISP"):
            cur, inblk = {}, True
            continue
        if inblk:
            if ln.startswith(" -1"):
                nid = int(ln[3:13])
                cur[nid] = (float(ln[13:25]), float(ln[25:37]), float(ln[37:49]))
            elif ln.startswith(" -3"):
                modes.append(cur)
                inblk = False
    return modes


# ------------------------------------------------------------------------------------ (a) transition
def transition_deck(t, R, pair=False, L=120.0, dz=1.0, n=12):
    zc = zgrid(L, dz)
    A = Blade(R, zc, n, t, +1, 0)
    blades = [A]
    if pair:
        B = Blade(R, zc, n, t, -1, B_OFF)
        blades.append(B)
    D = mesh_cards(blades)
    flat_rows = [j for j in range(2 * A.m + 1) if A.z[j] <= 1.0 + 1e-9]
    bc = ["*BOUNDARY"]
    pres = []
    for b in blades:
        for j in flat_rows:
            for nid in b.row(j):                           # faces go to x = +-t/2, mid-surface to x = 0
                lay = (nid - b.off - 1) // Blade.LAY
                pres.append(f"{nid},1,1,{(lay - 1) * t / 2 - b.nodes[nid][0]:.9g}")
        bc += [f"{nid},3,3" for nid in b.row(0)]
        bc += [f"{b.id(b.n, 0, lay)},2,2" for lay in (0, 1, 2)]
    if pair:
        D += gap_cards(A, B, [(i, j) for (i, j) in A.edge_rows() if A.z[j] > 1.0 + 1e-9])
    D += bc
    rows_out = [A.id(i, j, lay) for j in range(0, 2 * A.m + 1, 2) for i in range(2 * n + 1) for lay in (0, 2)]
    D += nset("SECT", rows_out)
    D += ["*STEP,NLGEOM,INC=500", "*STATIC", "0.05,1.,1e-5,0.1", "*BOUNDARY"] + pres
    D += ["*NODE PRINT,NSET=SECT", "U", "*END STEP"]
    return D, A


def transition_result(d, job, A):
    res = parse_dat(os.path.join(d, job + ".dat")).get(("displacements", "SECT"), [])
    if not res:
        return None
    tm, U = res[-1]
    s0 = sag_nat_from(A)
    zs, sags = [], []
    for j in range(0, 2 * A.m + 1, 2):
        P = np.array([(np.add(A.nodes[A.id(i, j, 0)], U[A.id(i, j, 0)]) +
                       np.add(A.nodes[A.id(i, j, 2)], U[A.id(i, j, 2)])) / 2 for i in range(2 * A.n + 1)])
        p0, p1 = P[0, :2], P[-1, :2]
        c = p1 - p0
        c /= np.linalg.norm(c)
        dev = (P[:, 0] - p0[0]) * c[1] - (P[:, 1] - p0[1]) * c[0]
        sags.append(float(np.max(np.abs(dev))))
        zs.append(float(P[:, 2].mean()))
    zs, sags = np.array(zs), np.array(sags)

    def reach(fr):
        k = np.nonzero(sags >= fr * s0)[0]
        if not len(k):
            return None
        k = k[0]
        if k == 0:
            return float(zs[0])
        # linear interpolation between the bracketing rows
        z0, z1, s_a, s_b = zs[k - 1], zs[k], sags[k - 1], sags[k]
        return float(z0 + (fr * s0 - s_a) / (s_b - s_a) * (z1 - z0))

    return dict(time=tm, sag_nat=s0, z=zs.round(3).tolist(), sag=np.round(sags, 4).tolist(),
                L90=reach(0.90), L95=reach(0.95), max_sag_ratio=float(sags.max() / s0))


def sag_nat_from(blade):
    P = np.array([(blade.xs[i], blade.ys[i]) for i in range(2 * blade.n + 1)])
    return float(np.max(np.abs(P[:, 0] - P[0, 0])))


# ---------------------------------------------------------------------------------- (b) the mast
def mast_model(config, t=T_NOM, R=R_NOM, n=12, dz=2.0, base="clamped", buckle=False, coords_shift=None,
               fine_base=False):
    """Cards up to (not including) the step, plus the blades.  config: unbonded | tied | single.
    fine_base: the mesh check, 18 across and 1.33 mm along over the first 48 mm (where the fold forms), 2 mm on."""
    zc = zgrid(L_MAST, dz)
    if fine_base:                                     # True = 1.5x; a number = that refinement factor
        f = 1.5 if fine_base is True else float(fine_base)
        n = int(round(12 * f))
        zc = np.concatenate([np.linspace(0.0, 48.0, int(round(24 * f)) + 1), np.linspace(48.0, L_MAST, 52)[1:]])
    A = Blade(R, zc, n, t, +1, 0)
    blades = [A]
    if config != "single":
        B = Blade(R, zc, n, t, -1, B_OFF)
        blades.append(B)
    if coords_shift:                                  # geometric imperfection (buckling mode)
        for b in blades:
            for k in b.nodes:
                if k in coords_shift:
                    b.nodes[k] = tuple(np.add(b.nodes[k], coords_shift[k]))
    D = mesh_cards(blades, extra_nodes=[(HEAD, (0.0, 0.0, L_MAST)), (HEADROT, (0.0, 0.0, L_MAST))])
    jt = 2 * A.m
    D += nset("TIPA", A.row(jt)) + nset("HEAD", [HEAD]) + nset("BASE", sum((b.row(0) for b in blades), []))
    D += [f"*RIGID BODY,NSET=TIPA,REF NODE={HEAD},ROT NODE={HEADROT}"]
    if config != "single":
        eq = ["*EQUATION"]
        for nid in B.row(jt):
            x, y, _ = B.nodes[nid]
            eq += ["3", f"{nid},1,1.,{HEAD},1,-1.,{HEADROT},3,{y:.9g}",
                   "3", f"{nid},2,1.,{HEAD},2,-1.,{HEADROT},3,{-x:.9g}"]
        D += eq
        inner = [(i, j) for (i, j) in A.edge_rows() if 0 < j < jt]
        if config == "tied":
            D += tie_cards(A, B, inner)
        elif buckle:                                  # closed-gap tangent for the linear buckle
            D += tie_cards(A, B, inner, dofs=(1,))
        else:
            D += gap_cards(A, B, [(i, j) for (i, j) in A.edge_rows() if j > 0])
        D += nset("TIPB", B.row(jt))
    if base == "clamped":                             # every node of the base face held: clamped
        D += ["*BOUNDARY", "BASE,1,3"] + nset("REACT", sum((b.row(0) for b in blades), []))
    else:                                             # hinge: only the base mid-surface line held
        # (the brick mid-nodes at the corners; the face mid-side nodes between them are left free: an
        #  equation tying them would carry reaction that CalculiX's RF does not report)
        mids = [b.id(i, 0, 1) for b in blades for i in range(0, 2 * b.n + 1, 2)]
        D += nset("BASEMID", mids) + nset("REACT", mids) + ["*BOUNDARY", "BASEMID,1,3"]
    return D, A, blades


# more iterations per increment (IC 30) and more cutbacks (IA 12) than the CalculiX defaults (16, 5): the gap
# elements open and close inside an increment
CONTROLS = "*CONTROLS,PARAMETERS=TIME INCREMENTATION\n4,8,9,30,10,4,0,12"
DIRS = {"+x": (1, +1), "-x": (1, -1), "+y": (2, +1), "-y": (2, -1)}


def fold_deck(config, direction, **kw):
    D, A, blades = mast_model(config, **kw)
    dof, sg = DIRS[direction]
    other = 2 if dof == 1 else 1
    U, d0, dmax = U_FOLD[dof]
    D += ["*STEP,NLGEOM,INC=4000", CONTROLS, "*STATIC", f"{d0 / U},1.,1e-7,{dmax / U}",
          "*BOUNDARY", f"{HEAD},{dof},{dof},{sg * U}",
          "*CLOAD", f"{HEAD},{other},{F_IMP}",
          "*NODE PRINT,NSET=HEAD", "U", "*NODE PRINT,NSET=REACT", "RF"]
    if config != "single":
        D += ["*NODE PRINT,NSET=TIPB", "U"]
    D += ["*END STEP"]
    return D, all_nodes(blades)


def all_nodes(blades):
    out = {}
    for b in blades:
        out.update({k: np.array(v) for k, v in b.nodes.items()})
    return out


def base_load(B, coords):
    """Force and moment the mast puts on its base: minus the sum of the base reactions and their moments about
    the origin (the base nodes do not move).  The head reaction alone would miss blade B's share, which reaches
    the head through *EQUATIONs (CalculiX's RF does not carry MPC forces)."""
    F = np.zeros(3)
    M = np.zeros(3)
    for k, rf in B.items():
        F -= rf
        M -= np.cross(coords[k], rf)
    return F, M


def fold_result(d, job, direction, coords):
    P = parse_dat(os.path.join(d, job + ".dat"))
    U = P.get(("displacements", "HEAD"), [])
    F = P.get(("forces", "REACT"), [])
    dof, sg = DIRS[direction]
    e = np.zeros(3)
    e[dof - 1] = sg
    axis = np.cross([0.0, 0.0, 1.0], e)              # bending axis of a tip force along e
    u, f, M = [0.0], [0.0], [0.0]
    for (tu, bu), (tf, bf) in zip(U, F):
        Fb, Mb = base_load({k: np.array(v) for k, v in bf.items()}, coords)
        u.append(float(np.dot(bu[HEAD], e)))
        f.append(float(np.dot(Fb, e)))
        M.append(float(np.dot(Mb, axis)) / 1000.0)    # N m
    u, f, M = map(np.array, (u, f, M))
    if len(u) < 2:
        return None
    k = int(np.argmax(M))
    # a fold = the moment drops >= 10 % after its peak (else the run ended / stayed stiff: lower bound)
    snapped = bool(M[k:].min() < 0.9 * M[k]) if k < len(M) - 1 else False
    slide = None
    tb = P.get(("displacements", "TIPB"), [])
    if tb:
        slide = float(max(abs(v[2] - U[-1][1][HEAD][2]) for v in tb[-1][1].values()))
    return dict(u=np.round(u, 4).tolist(), F=np.round(f, 6).tolist(), M=np.round(M, 6).tolist(),
                k0=float(f[1] / u[1]), M_peak=float(M[k]), u_at_peak=float(u[k]), snapped=snapped,
                u_end=float(u[-1]), complete=bool(u[-1] >= U_FOLD[dof][0] * 0.999), tipB_slide_end=slide)


def buckle_deck(config, nmodes=4):
    D, A, blades = mast_model(config, buckle=True)
    D += ["*STEP", f"*BUCKLE", f"{nmodes}", "*CLOAD", f"{HEAD},3,-1.", "*NODE FILE", "U", "*END STEP"]
    return D, blades


def compress_deck(config, shift, u_max):
    D, A, blades = mast_model(config, coords_shift=shift)
    D += ["*STEP,NLGEOM,INC=2000", CONTROLS, "*STATIC", "0.01,1.,1e-6,0.02",
          "*BOUNDARY", f"{HEAD},3,3,{-u_max}", "*NODE PRINT,NSET=HEAD", "U", "*NODE PRINT,NSET=REACT", "RF",
          "*END STEP"]
    return D


def compress_result(d, job):
    P = parse_dat(os.path.join(d, job + ".dat"))
    U = P.get(("displacements", "HEAD"), [])
    F = P.get(("forces", "REACT"), [])
    if not F:
        return None
    w = [0.0] + [-b[HEAD][2] for _, b in U]
    p = [0.0] + [sum(v[2] for v in b.values()) for _, b in F]       # base reaction up = head compression
    lat = [0.0] + [math.hypot(b[HEAD][0], b[HEAD][1]) for _, b in U]
    k = int(np.argmax(p))
    return dict(w=np.round(w, 5).tolist(), P=np.round(p, 4).tolist(), lateral=np.round(lat, 4).tolist(),
                P_peak=float(p[k]), w_at_peak=float(w[k]), peak_is_last=bool(k == len(p) - 1))


# ------------------------------------------------------------------------------------------------- plots
def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})

    # transition: sag / natural vs distance, t family at R 15 and R family at t 0.13
    tr = res.get("transition", {})
    if tr:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, dpi=100)
        cmap = plt.get_cmap("viridis")
        for k, tt in enumerate((0.11, 0.13, 0.19)):
            r = tr.get(f"t{tt}_R15.0")
            if r:
                ax[0].plot(r["z"], np.array(r["sag"]) / r["sag_nat"], color=cmap(k / 2.5), lw=2,
                           label=f"t {tt} mm  (L95 {fmt(r['L95'])} mm)")
        for k, RR in enumerate((12.7, 15.0, 20.0, 25.4)):
            r = tr.get(f"t0.13_R{RR}")
            if r:
                ax[1].plot(r["z"], np.array(r["sag"]) / r["sag_nat"], color=cmap(k / 3.5), lw=2,
                           label=f"R {RR} mm  (L95 {fmt(r['L95'])} mm)")
        rp = tr.get("pair_t0.13_R15.0")
        if rp:
            ax[0].plot(rp["z"], np.array(rp["sag"]) / rp["sag_nat"], "k--", lw=1.5,
                       label=f"pair, t 0.13  (L95 {fmt(rp['L95'])} mm)")
        for a, ttl in zip(ax, ("R 15 mm, thickness family", "t 0.13 mm, cup radius family")):
            a.axhline(0.9, color="0.5", lw=0.8, ls=":")
            a.axhline(0.95, color="0.5", lw=0.8, ls=":")
            a.axvline(THROAT, color="C3", lw=1.2)
            a.text(THROAT + 1, 0.5, "32 mm throat", color="C3", fontsize=9)
            a.set_xlim(0, 80)
            a.set_ylim(0, 1.1)
            a.set_xlabel("distance from flattened end, z (mm)")
            a.set_title(ttl, fontsize=11)
            a.legend(fontsize=9, loc="lower right")
        ax[0].set_ylabel("cup sag / natural sag")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fe_transition.png"))
        plt.close(fig)

    # fold: base moment vs head displacement, 4 directions, three configurations
    fo = res.get("fold", {})
    if fo:
        fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), dpi=100)
        sty = {"unbonded": ("C0", "-"), "tied": ("C1", "-"), "single": ("C2", "-")}
        for a, dname in zip(ax.flat, ("+x", "-x", "+y", "-y")):
            for cfg in ("unbonded", "tied", "single"):
                r = fo.get(f"{cfg}_{dname}")
                if r:
                    c, ls = sty[cfg]
                    a.plot(r["u"], r["M"], color=c, ls=ls, lw=2)
                    a.plot([r["u_at_peak"]], [r["M_peak"]], "o", color=c, ms=5)
                    a.annotate(f"{r['M_peak']:.2f}", (r["u_at_peak"], r["M_peak"]), textcoords="offset points",
                               xytext=(4, 4), fontsize=9, color=c, fontweight="bold")
                rf = fo.get(f"flex_{cfg}_{dname}")
                if rf:
                    c, _ = sty[cfg]
                    a.plot(rf["u"], rf["M"], color=c, ls="--", lw=1.2)
            a.axhline(0.2, color="k", lw=1, ls=":")
            a.axhline(0.3, color="k", lw=1, ls="--")
            a.axhline(HEAD_W * (L_MAST + HEAD_CG) / 1000, color="C3", lw=1, ls=":")
            a.set_yscale("log")
            a.set_ylim(0.01, 40)
            a.set_xlim(left=0)
            x0 = a.get_xlim()[1] * 0.98
            a.text(x0, 0.2 * 0.88, "bench target 0.2", fontsize=8, ha="right", va="top")
            a.text(x0, 0.3 * 1.08, "turret design 0.3", fontsize=8, ha="right")
            a.text(x0, 0.112 * 0.88, "1 g head lateral 0.11", fontsize=8, color="C3", ha="right", va="top")
            a.set_title(f"tip force in {dname}", fontsize=11)
        for a in ax.flat:
            a.set_xlabel("head displacement along the load (mm)")
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], color=sty[c][0], lw=2, label=c) for c in ("unbonded", "tied", "single")]
        hd.append(Line2D([], [], color="0.3", lw=1.2, ls="--", label="flexible base (hinge)"))
        hd.append(Line2D([], [], color="0.3", marker="o", ls="", label="peak (N m)"))
        fig.legend(handles=hd, loc="upper center", ncol=5, fontsize=10, frameon=False)
        for a in ax[:, 0]:
            a.set_ylabel("base moment (N m)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(OUT, "fe_fold.png"))
        plt.close(fig)

    md = res.get("_modes")
    if md:
        fig, ax = plt.subplots(1, len(md), figsize=(3.3 * len(md), 4.5), dpi=100)
        ax = np.atleast_1d(ax)
        for a, (lab, z, ux, uy) in zip(ax, md):
            a.plot(ux, z, lw=2, label="x (lens thickness)")
            a.plot(uy, z, lw=2, label="y (width)")
            a.set_title(lab, fontsize=10)
            a.set_xlabel("mode displacement (normalised)")
            a.set_xlim(-1.05, 1.05)
        ax[0].set_ylabel("z (mm)")
        ax[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fe_modes.png"))
        plt.close(fig)


def fmt(v):
    return "n/a" if v is None else f"{v:.0f}"


# -------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(tempfile.gettempdir(), "probe_mast_fe"))
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--only", default="transition,mast,buckle,mesh,flex")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    groups = set(a.only.split(","))
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    res_path = os.path.join(OUT, "fe.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res.pop("_modes", None)
    pool = ThreadPoolExecutor(a.jobs)
    runs = {}

    def submit(job, deck, post):
        coords = None
        if isinstance(deck, tuple):
            deck, coords = deck
        runs[job] = (pool.submit(run_ccx, job, deck, a.work, a.force), post + (coords,))

    # linear buckling (fast) first, its modes feed the nonlinear compression
    bk = {}
    if "buckle" in groups:
        for cfg in ("unbonded", "tied", "single"):
            D, blades = buckle_deck(cfg)
            bk[cfg] = (pool.submit(run_ccx, f"buckle_{cfg}", D, a.work, a.force), blades)

    # (b) fold + stiffness
    if "mast" in groups:
        for cfg in ("unbonded", "tied", "single"):
            for dn in DIRS:
                submit(f"fold_{cfg}_{dn[0]}{dn[1]}".replace("+", "p").replace("-", "m"),
                       fold_deck(cfg, dn), ("fold", f"{cfg}_{dn}", dn))
    if "flex" in groups:                              # the base hinge only matters for bending in x
        for cfg in ("unbonded", "tied", "single"):
            for dn in ("+x", "-x"):
                submit(f"flex_{cfg}_{dn}".replace("+", "p").replace("-", "m"),
                       fold_deck(cfg, dn, base="flexible"), ("fold", f"flex_{cfg}_{dn}", dn))
    if "mesh" in groups:
        for cfg, dn, f in (("single", "+x", True), ("single", "-x", True), ("unbonded", "+x", True),
                           ("unbonded", "-x", True), ("single", "+x", 2.0)):
            tag = "fine" if f is True else "fine2"
            submit(f"{tag}_{cfg}_{dn}".replace("+", "p").replace("-", "m"), fold_deck(cfg, dn, fine_base=f),
                   ("fold", f"{tag}_{cfg}_{dn}", dn))
    # (a) transition
    if "transition" in groups:
        for tt in (0.11, 0.13, 0.19):
            for RR in (12.7, 15.0, 20.0, 25.4):
                D, A = transition_deck(tt, RR)
                submit(f"tr_t{tt}_R{RR}", D, ("transition", f"t{tt}_R{RR}", A))
        D, A = transition_deck(T_NOM, R_NOM, pair=True)
        submit("tr_pair", D, ("transition", f"pair_t{T_NOM}_R{R_NOM}", A))
    if bk:
        modes_plot = []
        comp = {}
        for cfg, (fut, blades) in bk.items():
            d, sec, ok = fut.result()
            f = parse_buckle(os.path.join(d, f"buckle_{cfg}.dat"))
            modes = parse_frd_modes(os.path.join(d, f"buckle_{cfg}.frd"))
            res.setdefault("buckle", {})[cfg] = dict(linear_factors_N=f, sec=round(sec, 1), ccx_ok=ok)
            if not (f and modes):
                continue
            m0 = modes[-len(f)]                     # the .frd holds the static pre-step first, then the modes
            A = blades[0]
            amp = max(math.sqrt(sum(c * c for c in v)) for k, v in m0.items() if k < HEAD)
            shift = {k: tuple(0.15 * c / amp for c in v) for k, v in m0.items()}
            k_ax = E * W * T_NOM * len(blades) / L_MAST
            u_max = max(3.0 * f[0] / k_ax, 1.5)
            comp[cfg] = pool.submit(run_ccx, f"compress_{cfg}", compress_deck(cfg, shift, u_max), a.work, a.force)
            # mode shape along the mast: blade A's centre line, normalised
            jc = [A.id(A.n, j) for j in range(0, 2 * A.m + 1, 2)]
            z = [A.nodes[k][2] for k in jc]
            ux = np.array([m0.get(k, (0, 0, 0))[0] for k in jc])
            uy = np.array([m0.get(k, (0, 0, 0))[1] for k in jc])
            s = max(np.abs(ux).max(), np.abs(uy).max()) or 1.0
            modes_plot.append((f"{cfg}: mode 1, {f[0]:.1f} N", z, (ux / s).tolist(), (uy / s).tolist()))
        for cfg, fut in comp.items():
            d, sec, ok = fut.result()
            r = compress_result(d, f"compress_{cfg}") or {"failed": True}
            r.update(sec=round(sec, 1), ccx_ok=ok)
            res["buckle"][cfg]["nonlinear"] = r
        res["_modes"] = modes_plot

    for job, (fut, post) in runs.items():
        d, sec, ok = fut.result()
        kind, key = post[0], post[1]
        r = transition_result(d, job, post[2]) if kind == "transition" else fold_result(d, job, post[2], post[3])
        if r is None:
            r = {"failed": True}
        r.update(sec=round(sec, 1), ccx_ok=ok)
        res.setdefault(kind, {})[key] = r

    # mesh check and summary
    fo = res.get("fold", {})
    res["mesh_check"] = {}
    for key in fo:
        if key.startswith("fine"):
            base_key = key.split("_", 1)[1]
            if base_key in fo and "M_peak" in fo[key] and "M_peak" in fo[base_key]:
                res["mesh_check"][key] = dict(M_coarse=fo[base_key]["M_peak"], M_fine=fo[key]["M_peak"],
                                              change_pct=round(100 * (fo[key]["M_peak"] / fo[base_key]["M_peak"] - 1), 1))
    res["summary"] = {
        "transition_L95_mm": {k: v.get("L95") for k, v in res.get("transition", {}).items()},
        "stiffness_N_per_mm": {c: {d: fo.get(f"{c}_{d}", {}).get("k0") for d in ("+x", "+y")}
                               for c in ("unbonded", "tied", "single")},
        "fold_moment_Nm": {c: {d: fo.get(f"{c}_{d}", {}).get("M_peak") for d in DIRS}
                           for c in ("unbonded", "tied", "single")},
        "buckle_N": {c: dict(linear=(v.get("linear_factors_N") or [None])[0],
                             nonlinear=v.get("nonlinear", {}).get("P_peak")) for c, v in res.get("buckle", {}).items()},
    }
    # comparisons
    res["reference"] = dict(
        bench_fold_target_Nm=0.2, turret_design_Nm=0.3, head_weight_N=round(HEAD_W, 3),
        head_1g_lateral_base_moment_Nm=round(HEAD_W * (L_MAST + HEAD_CG) / 1000, 4),
        throat_mm=THROAT, natural_sag_mm={str(R): round(sag_nat(R), 3) for R in (12.7, 15.0, 20.0, 25.4)},
        bending_D_Nmm=round(E * T_NOM ** 3 / (12 * (1 - NU ** 2)), 2))
    res["model"] = dict(
        elements="C3D20R solid shell, 1 through t", across=12, along_mm=2.0, transition_along_mm=1.0,
        contact=f"GAPUNI node-to-node at the mid-surface edge nodes every 2 mm, frictionless, fixed +x, k {GAP_K:g} N/mm",
        base="clamped: every base-face node fixed; flexible: base mid-surface line pinned (edge hinge)",
        tip="A: *RIGID BODY on HEAD; B: x, y by linear *EQUATION to HEAD (incl. twist), z free",
        nlgeom=True, fold_force="from the base reactions", imperfection_force_N=F_IMP)
    res["wall_s_this_run"] = round(time.time() - t_start, 1)
    plots(res)
    res.pop("_modes", None)
    json.dump(res, open(res_path, "w"), indent=1)
    print(f"wrote {res_path}  ({res['wall_s_this_run']:.0f} s)")


if __name__ == "__main__":
    main()
