"""Standard handling and assembly load cases (15 Nov, Spark's Scout v4 bench print: an SG90 tab post snapped as its screw went in; every
part had passed its flight and crash cases, none had a case for being put together or held). A setup that draws one of these features
through a shared helper (or declares it on the Part) gets its cases without writing them:

  screw   a screw boss or post (a self-tapper, or a bolt through it with a nut or into the part behind): "tighten" = the hand driver's
          torque about the screw axis on the boss + the clamp that torque gives, along the axis (one case: they come together)
  insert  a heat-set insert boss: "install" = the iron's push into the hole; "tighten" = the screw's torque + its clamp pulling the
          insert out of the boss (the pull-out a tightened screw puts on it)
  clip    a snap clip: "deflect" = the hand's snap force at its tip, along the way it flexes
  hand    a hand-held part (SPEC["hand_held"] = True, or Part.hand_held()): "push" and "lever" = a hand's push at each end of its longest
          extent, along that extent (inward) and across it (its weakest bending direction)

Every case is class "crash" (topo/material.py SAFETY 1.1): the numbers below are already the top of what a hand does, a one-off at its
limit, as a crash's. The support is the part's main support (the one most of its cases use). Numbers (E, with their sources):
"""
import numpy as np

# hand-tight driving torque into a printed part or a heat-set insert, N m: about half the steel 8.8 snug torque (M3 1.1-1.3 N m dry,
# VDI 2230 tables), the insert makers' "do not exceed" for plastics (M3 0.4-0.5 N m), a precision driver in the fingers (0.2-0.6 N m) E
TORQUE_NM = {"M2": 0.15, "M2.5": 0.25, "M3": 0.4, "M4": 0.9, "M5": 1.6}
THREAD_MM = {"M2": 2.0, "M2.5": 2.5, "M3": 3.0, "M4": 4.0, "M5": 5.0}
NUT_K = 0.3                  # E: clamp F = T / (K d); K 0.2 is dry steel on steel (Shigley), a thread in a printed seat or an insert under a PCTG head runs 0.3-0.4
INSTALL_N = 30.0             # E: pressing a heat-set insert home with the iron's tip, a 2-4 kg hand push (the insert makers' install guides: let the heat do it)
CLIP_N = 20.0                # E: a hand's snap force on a small cantilever clip, 10-30 N (Bayer / Covestro snap-fit design guide's typical assembly forces)
HAND_N = 50.0                # E: a thumb or finger push on a part held in the other hand, 50-100 N (MIL-STD-1472 / NASA-STD-3000 one-finger and thumb push)
CLS = "crash"
SRC = dict(torque="E: hand-tight, half the steel 8.8 snug torque (VDI 2230), the insert makers' plastics limit", clamp=f"E: T / (K d), K {NUT_K}",
           install=f"E: {INSTALL_N:.0f} N iron push", clip=f"E: {CLIP_N:.0f} N hand snap", hand=f"E: {HAND_N:.0f} N thumb push (MIL-STD-1472)")


def clamp_n(thread):
    return TORQUE_NM[thread] * 1000.0 / (NUT_K * THREAD_MM[thread])


# ---- the features a setup draws (recorded while setup(h) runs: topo/fit.py shapes; the shared helpers declare them) --------------
def declare(kind, region, at, axis=None, thread="M3", name=None):
    """a shared helper drawing a feature calls this: kind screw / insert / clip; region the manifold the load goes on (the boss or pad,
    the clip's tip); at a point on the axis (the hole's mouth); axis the screw's way in (into the part), or the clip's way of flexing"""
    from . import fit
    rec = fit._SHAPES
    if rec is None:
        return
    rec.features.append(dict(kind=kind, region=region, at=np.asarray(at, float), axis=None if axis is None else np.asarray(axis, float) /
                             np.linalg.norm(axis), thread=thread, name=name or f"{kind} at {np.round(np.asarray(at, float), 1).tolist()}"))


def _torque_F(part, nd, at, axis, T_nmm):
    """nodal forces of a torque T (N mm) about the axis: tangential, in proportion to the radius from the loaded nodes' own centre (a rigid
    twist with no net force, wherever the grid puts the nodes round the hole)"""
    P = _node_xyz(part, nd)
    r = P - P.mean(0); r -= np.outer(r @ axis, axis)
    rr = (r * r).sum(1); s = rr.sum()
    if s <= 1e-9:
        return np.zeros((len(nd), 3))
    return np.cross(axis, r) * (T_nmm / s)


def _node_xyz(part, nd):
    nx, ny, nz = part.shape
    i, rem = np.divmod(nd, (ny + 1) * (nz + 1)); j, k = np.divmod(rem, nz + 1)
    o = np.asarray(part.origin, float)
    return np.stack([o[0] + i * part.h, o[1] + j * part.h, o[2] + k * part.h], 1)


def _case(part, name, nd, F, support):
    from .part import LoadCase
    G = np.zeros(part.ndof)
    for a in range(3):
        G[3 * nd + a] = F[:, a] if np.ndim(F) == 2 else F[a] / len(nd)
    part.cases.append(LoadCase(name, G, cls=CLS, support=support, nodes=nd))


def apply(part, features, hand_held=False, log=print):
    """the standard cases for the recorded features (and a hand-held part's), on the part's main support. Returns the rows added."""
    sup = part.main_support() or (next(iter(part.supports)) if part.supports else None)
    rows = []
    if sup is None:
        return rows
    for f in features:
        el = part.elems(f["region"], part.solid | part.design, what=f"handling {f['name']}")
        if len(el) == 0:
            log(f"handling: {f['name']}: its region holds no part cells, no case"); continue
        nd = part.nodes_of(el); ax = f["axis"]
        if f["kind"] in ("screw", "insert"):
            T = TORQUE_NM[f["thread"]] * 1000.0; C = clamp_n(f["thread"])
            if f["kind"] == "insert":
                _case(part, f"handling: {f['name']} install ({INSTALL_N:.0f} N)", nd, INSTALL_N * ax, sup)
                rows.append(dict(case=f"{f['name']} install", N=INSTALL_N, src=SRC["install"]))
            F = _torque_F(part, nd, f["at"], ax, T) + (-C if f["kind"] == "insert" else C) * ax / len(nd)   # an insert is pulled out, a boss pressed
            _case(part, f"handling: {f['name']} tighten {f['thread']} ({T / 1000:.2f} N m, {C:.0f} N)", nd, F, sup)
            rows.append(dict(case=f"{f['name']} tighten", Nm=T / 1000, N=round(C), src=SRC["torque"] + "; " + SRC["clamp"]))
        elif f["kind"] == "clip":
            _case(part, f"handling: {f['name']} deflect ({CLIP_N:.0f} N)", nd, CLIP_N * ax, sup)
            rows.append(dict(case=f"{f['name']} deflect", N=CLIP_N, src=SRC["clip"]))
    if hand_held:
        rows += hand_cases(part, sup, log)
    if rows:
        log(f"handling: {len(rows)} standard cases ({', '.join(r['case'] for r in rows[:6])}{' ...' if len(rows) > 6 else ''})")
    part.handling = rows
    return rows


def hand_cases(part, sup, log=print):
    """a hand-held part: a push at each end of its longest extent along it (inward), and a lever across it (its thinnest extent's axis)"""
    mat = part.solid | part.design
    X = part.cell_centres()[mat]
    if not len(X):
        return []
    ext = X.max(0) - X.min(0); a = int(np.argmax(ext)); b = int(np.argmin(ext))
    rows = []
    idx = np.where(mat)[0]
    for end, sg in (("max", 1.0), ("min", -1.0)):
        edge = X[:, a].max() if sg > 0 else X[:, a].min()
        el = idx[np.abs(X[:, a] - edge) <= 1.5 * part.h]
        nd = part.nodes_of(el)
        e_a = np.zeros(3); e_a[a] = -sg; e_b = np.zeros(3); e_b[b] = 1.0
        nm = "xyz"[a]
        _case(part, f"handling: hand push at the {nm} {end} end ({HAND_N:.0f} N)", nd, HAND_N * e_a, sup)
        _case(part, f"handling: hand lever at the {nm} {end} end ({HAND_N:.0f} N across)", nd, HAND_N * e_b, sup)
        rows += [dict(case=f"hand push {nm} {end}", N=HAND_N, src=SRC["hand"]), dict(case=f"hand lever {nm} {end}", N=HAND_N, src=SRC["hand"])]
    return rows
