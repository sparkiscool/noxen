"""Strict connectivity check (Spark 20 Oct 01:08: no disconnected structures, ever). A part passes only if it is ONE body
after a 0.9 mm erosion (just under half the 2.0 wall floor, so a legal 2.0 wall survives): a feature that touches the rest along an edge, a corner, a sliver or a
neck thinner than 2.0 splits off when eroded and fails here, even when a plain body count says 1. Every drawn feature
(pads, cheeks, plates, slot walls) must also sit in the main eroded body (drawn_not_in_main: body, share, erode_mm, and at / mm3 of the detached core).
Neck test: also eroded at 1.2 / 1.5 (necks under 3 mm); a blob that splits off, is >= 1.6 mm thicker than its neck and joins the rest over less than
4 x its thickness squared fails (pads on
~2 mm necks). Each drawn body is also eroded to 0.4 x its own thickness (its own wall, 90th percentile, not what wraps it) with its voxels
kept connected (a body is never split from itself by its own grooves or bores) and its core must stay in the main body.
python3 solid_check.py part.stl [--erode 1.0] [--pitch 0.25]   -> exit 1 on fail
from solid_check import solid_check; ok, info = solid_check(trimesh_part, drawn=[trimesh, ...], thin_ok=[trimesh, ...])
thin_ok: drawn bodies the setup declares deliberately thin (Part(..., thin_ok=[...])): their thin walls, and a neck under a blob holding one
when the neck is at least as thick as the body's wall, are reported as thin_allowed, not failed."""
import sys, numpy as np, trimesh
from scipy import ndimage as ndi


def _ray_parity(mesh, xs, ys, zs, chunk=20000):
    """inside test by ray parity along Z, rays in chunks: noxopt._voxelise verbatim, so the check needs no noxopt / pymoto
    (CI). The old fallback, mesh.contains on every cell centre at once, was killed for memory on a 17 MB part (P)."""
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    org = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, zs[0] - 5.0)], 1)
    A = np.zeros((len(xs) * len(ys), len(zs)), bool)
    for c0 in range(0, len(org), chunk):
        o = org[c0:c0 + chunk]
        loc, ray, _ = mesh.ray.intersects_location(o, np.tile([0, 0, 1.0], (len(o), 1)), multiple_hits=True)
        if not len(ray): continue
        order = np.argsort(ray, kind="stable"); ray = ray[order] + c0; z = loc[order][:, 2]
        st = np.r_[0, np.where(np.diff(ray))[0] + 1, len(ray)]
        for a, b in zip(st[:-1], st[1:]):
            zz = np.sort(z[a:b]); zz = zz[np.r_[True, np.diff(zz) > 1e-6]]
            for p, q in zip(zz[0::2], zz[1::2]):
                A[ray[a], (zs > p) & (zs < q)] = True
    return A.reshape(len(xs), len(ys), len(zs))


def voxels(mesh, pitch, lo, shape):
    """inside test on cell centres by ray parity along Z (one ray per (x, y) column)."""
    xs = lo[0] + (np.arange(shape[0]) + 0.5) * pitch; ys = lo[1] + (np.arange(shape[1]) + 0.5) * pitch
    zs = lo[2] + (np.arange(shape[2]) + 0.5) * pitch
    return _ray_parity(mesh, xs, ys, zs)


NECK_LEVELS = (0.9, 1.2, 1.5)        # erosion depths for the neck test (necks under 1.8 / 2.4 / 3.0 mm); 2.0 flagged 4.0 bolt-down plates joining two thick lumps (plates, not necks)
NECK_AREA_K = 4.0                    # ... and its joint to the rest is smaller than 4 x (blob thickness)^2
NECK_MARGIN = 0.8                    # a split-off blob fails if it is >= 1.6 mm thicker than the neck it hangs on


def _interface_mm2(V, Le, k, pitch, pad):
    """area of the joint between eroded component k and the rest: every solid voxel goes to its nearest eroded core
    (inside a box round k), and the faces between k's share and any other share are counted."""
    idx = np.argwhere(Le == k); a = np.maximum(idx.min(0) - pad, 0); b = np.minimum(idx.max(0) + pad + 1, Le.shape)
    sl = tuple(slice(a[i], b[i]) for i in range(3))
    Ls, Vs = Le[sl], V[sl]
    _, ind = ndi.distance_transform_edt(Ls == 0, return_indices=True)
    W = Ls[tuple(ind)]; W[~Vs] = -1
    A = 0
    for ax in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3; s0[ax] = slice(0, -1); s1[ax] = slice(1, None)
        p, q = W[tuple(s0)], W[tuple(s1)]
        A += int((((p == k) & (q >= 1) & (q != k)) | ((q == k) & (p >= 1) & (p != k))).sum())
    return A * pitch ** 2


def thin_walls(V, pitch, erode_mm=0.9, lo=(0.0, 0.0, 0.0), core_r_mm=0.35, min_core_mm3=0.3):
    """the material an opening with a ball of radius erode_mm removes (R), its components (labr) and, for every residue
    component whose core (inscribed radius >= core_r_mm, a real sliver / wall under 2 x erode_mm) is at least min_core_mm3,
    a dict: label (component of labr), core_label (component of labc, the cores), kind ("wall": the core spreads in 2 directions; "edge": the residue a convex edge
    loses to the opening spreads in 1), thick_mm, core_mm3, at, span, mm3 (the whole residue component). Shared by the gate
    (the thin_walls report) and the engine's export cleanup (topo/build.py drops the wall-kind ones under its core limit)."""
    lo = np.asarray(lo, float)
    E = ndi.distance_transform_edt(V) * pitch > erode_mm
    O = ndi.distance_transform_edt(~E) * pitch <= erode_mm
    R = V & ~O
    dr = ndi.distance_transform_edt(R) * pitch
    st = ndi.generate_binary_structure(3, 1)
    labr, nr = ndi.label(R, st)
    thin = []
    if nr:
        vol = np.bincount(labr.ravel()) * pitch ** 3
        core = (dr >= core_r_mm) & R
        labc, nc = ndi.label(core, ndi.generate_binary_structure(3, 3))
        for k in range(1, nc + 1):
            idx = np.argwhere(labc == k)
            if len(idx) * pitch ** 3 < min_core_mm3: continue
            P = lo + (idx + 0.5) * pitch
            ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(P.T)) if len(P) > 3 else np.zeros(3), 0))[::-1]
            kind = "wall" if ev[1] > 0.8 else "edge"                  # a wall's core spreads in 2 directions, an edge residue in 1
            th = float(2 * dr[tuple(idx.T)].max())
            lab = int(labr[tuple(idx[0])])
            thin.append(dict(kind=kind, thick_mm=round(th, 2), core_mm3=round(len(idx) * pitch ** 3, 2), at=np.round(P.mean(0), 1).tolist(),
                             span=np.round(ev, 1).tolist(), label=lab, core_label=k, mm3=round(float(vol[lab]), 1)))
    return R, labr, thin, labc


def _where(core, off, lo, pitch):
    """'at' and 'mm3' of a detached drawn core: core the body's core voxels (bool grid), off which of them (in core's order) sit
    off the main body"""
    idx = np.argwhere(core)[off]
    return dict(at=np.round(lo + (idx.mean(0) + 0.5) * pitch, 1).tolist(), mm3=round(float(len(idx)) * pitch ** 3, 2))


def solid_check(mesh, drawn=(), pitch=0.25, erode_mm=0.9, min_cc_mm3=1.0, thin_ok=()):
    lo = mesh.bounds[0] - 2 * pitch; hi = mesh.bounds[1] + 2 * pitch
    shape = tuple(int(np.ceil((hi[k] - lo[k]) / pitch)) for k in range(3))
    V = voxels(mesh, pitch, lo, shape)
    st = ndi.generate_binary_structure(3, 1)                         # face connectivity only: no corner or edge links
    lab0, n0 = ndi.label(V, st)
    d = ndi.distance_transform_edt(V) * pitch
    E = d > erode_mm
    lab, n = ndi.label(E, st)
    vols = np.bincount(lab.ravel())[1:] * pitch ** 3
    keep = np.where(vols >= min_cc_mm3)[0] + 1
    main = keep[np.argmax(vols[keep - 1])] if len(keep) else 0
    # a body is a piece of at least min_cc_mm3: a speck of a few voxels (a sliver at an acute edge, the 0.5 mm core thread of a
    # O5 arm at the 2 mm erosion) is voxel dust, not a body (28 Oct: node 7 "3 bodies" were two single voxels, node 21's drawn
    # core was "0.3 % off" in 41 pieces under 0.4 mm3)
    n_plain = int((np.bincount(lab0.ravel())[1:] * pitch ** 3 >= min_cc_mm3).sum())
    info = dict(bodies_plain=n_plain, specks_plain=int(n0 - n_plain), bodies_eroded=int(len(keep)),
                eroded_volumes_mm3=sorted([round(float(v), 1) for v in vols[keep - 1]], reverse=True)[:6])
    bad = []
    big = np.r_[False, vols >= min_cc_mm3]                           # by label: is this eroded component a body
    for i, t in enumerate(drawn):
        Dv = voxels(t, pitch, lo, shape) & E
        if Dv.sum() == 0: continue                                   # a drawn feature thinner than 2 x erode: judged by its own size check
        off = (lab[Dv] != main) & big[lab[Dv]]
        share = 1.0 - float(off.mean())
        if share < 0.999: bad.append(dict(body=str(i), share=round(share, 3), erode_mm=erode_mm, **_where(Dv, off, lo, pitch)))
    # neck test (Spark 20 Oct 14:19, bracket_e v8 pads on ~2 mm necks passed the 0.9 test): erode deeper; a blob that splits
    # off and is clearly thicker than the neck it hung on (max inscribed radius >= e + NECK_MARGIN) is a sliver attachment.
    # A wall that is simply thin vanishes at that depth instead of splitting, so legal 2.0 walls are not flagged.
    # thin walls (the opening's residue, thin_walls) and the declared-thin bodies (thin_ok: the camera M2 insert posts, 1.2 wall, Spark
    # 29 Oct by coupon): a thin wall whose core lies in one of them is reported under thin_allowed and does not fail the part; each
    # such body's wall is the thickness of its own thin finding (a post is solid below its bore: its thickest part is not its wall)
    R, labr, thin, labc = thin_walls(V, pitch, erode_mm, lo)
    T = np.zeros(shape, bool); bodies_T = []
    for t in thin_ok:
        Tb = voxels(t, pitch, lo, shape)
        if Tb.any():
            T |= Tb; bodies_T.append(Tb)
    allowed = [t for t in thin if t["kind"] == "wall" and T.any() and T[labc == t["core_label"]].mean() >= 0.9]
    walls = []
    for Tb in bodies_T:
        own = [t["thick_mm"] for t in allowed if Tb[labc == t["core_label"]].mean() >= 0.5]
        walls.append((Tb, max(own) if own else round(2 * float(ndi.distance_transform_edt(Tb).max()) * pitch, 2)))
    necks, allowed_necks = [], []
    prev = {e: (NECK_LEVELS[i - 1] if i else 0.0) for i, e in enumerate(NECK_LEVELS)}
    for e in NECK_LEVELS:
        Le, ne = ndi.label(d > e, st)
        if ne < 2: continue
        ve = np.bincount(Le.ravel())[1:] * pitch ** 3; mx = ndi.maximum(d, Le, index=np.arange(1, ne + 1))
        mn = int(np.argmax(ve)) + 1
        for k in range(1, ne + 1):
            if k == mn or ve[k - 1] < 2.0 or mx[k - 1] < e + NECK_MARGIN: continue
            A = _interface_mm2(V, Le, k, pitch, pad=int(np.ceil((e + 8.0) / pitch)))
            tb = 2 * float(mx[k - 1])
            if A >= NECK_AREA_K * tb ** 2: continue               # joined over a wide band (a thin-walled box on a channel): not a neck
            blob = Le == k
            c = lo + (np.array(ndi.center_of_mass(blob)) + 0.5) * pitch
            row = dict(neck_under_mm=round(2 * e, 1), blob_thick_mm=round(tb, 1), joint_mm2=round(A, 1), blob_mm3=round(float(ve[k - 1]), 1), at=np.round(c, 1).tolist())
            # a blob that is, or holds, a declared-thin body (the root of a 1.2 mm post) on a neck at least as thick as that body's
            # wall: the neck is not the weak point (30 Oct, node 21: 4.5 mm post roots on 2.4 mm necks under 1.2 mm posts carrying 2 N)
            own = [w for Tb, w in walls if (Tb & blob).any()]
            if own and 2 * prev[e] >= max(own):
                row.update(kind="neck", thin_wall_mm=max(own), neck_over_mm=round(2 * prev[e], 1)); allowed_necks.append(row)
            else:
                necks.append(row)
    info["necks"] = necks
    # every drawn body on its own, eroded to 0.4 x its own thickness (0.9..2.0): its core must stay in the main body. Its
    # thickness is its own (the body alone, 90th percentile of its wall: a 2.5 collar reads 2.5 where a boss overlaps it or a
    # fillet wraps it, 29 Oct: node 5's collar clusters and node 21's camera cones were eroded 2.0 for the material round them
    # and had to hang on 4 mm) and its voxels are kept connected (a collar's grooves or a cone's bore never split a body from
    # itself): the body is joined when any of it meets the main eroded body.
    for i, t in enumerate(drawn):
        for j, b in enumerate(t.split(only_watertight=False) if len(t.faces) else []):
            Dv = voxels(b, pitch, lo, shape) & V
            if Dv.sum() == 0: continue
            own = ndi.distance_transform_edt(Dv) * pitch
            ef = float(np.clip(0.8 * np.percentile(own[Dv], 90), erode_mm, 2.0))
            Ee = d > ef
            core = Dv & Ee
            if core.sum() == 0: continue
            Le, ne = ndi.label(Ee, st)
            ve = np.bincount(Le.ravel())[1:]; mn = int(np.argmax(ve)) + 1
            Lc, _ = ndi.label(Ee | Dv, st)
            mc = int(Lc[Le == mn][0])
            off = Lc[core] != mc
            share = 1.0 - float(off.mean())
            if share < 0.999: bad.append(dict(body=f"{i}.{j}", share=round(share, 3), erode_mm=round(ef, 2), **_where(core, off, lo, pitch)))
    info["drawn_not_in_main"] = bad
    strip = lambda t: {k: v for k, v in t.items() if k not in ("label", "core_label")}
    info["thin_walls"] = sorted([strip(t) for t in thin if t["kind"] == "wall" and t not in allowed], key=lambda t: -t["core_mm3"])[:12]
    info["thin_allowed"] = sorted([strip(t) for t in allowed], key=lambda t: -t["core_mm3"])[:12] + allowed_necks
    info["acute_edges"] = len([t for t in thin if t["kind"] == "edge"])
    ok = n_plain == 1 and len(keep) == 1 and not bad and not info["thin_walls"] and not necks
    info["ok"] = bool(ok)
    return ok, info


if __name__ == "__main__":
    p = sys.argv[1]; er = float(sys.argv[sys.argv.index("--erode") + 1]) if "--erode" in sys.argv else 0.9
    pt = float(sys.argv[sys.argv.index("--pitch") + 1]) if "--pitch" in sys.argv else 0.25
    ok, info = solid_check(trimesh.load(p, force="mesh"), pitch=pt, erode_mm=er)
    print(p, "PASS" if ok else "FAIL", info); sys.exit(0 if ok else 1)
