"""Setup audit: runs before the first solve and refuses a setup that would waste a run or hide a patch.

The hub deck v6..v9 story (Oct 2026): a strength fail was answered by drawing solid blocks round the sockets (v7), then
bars from the sockets back to the feet (v9), and the room behind the sockets stayed a 38 mm box. The gate passed, the
strength index passed, and the part was blocky at the sockets with no rear web. None of that was visible in the numbers
the run reported. These checks make it visible and, for the clear cases, stop the run before it starts.

    audit(part, settings, log) -> dict (also written to result.json as "audit"); raises SetupError on a hard fail.

Checks (hard = raises, soft = logged + in the report):
  drawn_share    hard  drawn solid > 30 % of (drawn + design) volume: the part is mostly hand-drawn, not generated.
                       Attachments are small: pads, collars, seats. Over 30 % means a patch.
  drawn_blocks   hard  a drawn body whose inscribed diameter > 3 x min member and which touches no cut and no support:
                       a block drawn into the open, not an attachment (a collar touches a bore, a pad touches a bolt
                       hole or the support face).
  room           hard  for each loaded drawn body: the design cells within 2 x min member of it, as a share of the
                       body's exposed surface cells. Under 35 %: the body is boxed in, nothing can grow onto it.
  room_depth     soft  for each loaded drawn body: the longest straight run of design cells away from it (any axis).
                       Under 3 x min member: a web cannot form behind it; expect a thin bar.
  envelope_waste soft  design cells that are > 8 x min member from every drawn body and every support: unreachable
                       by any sensible load path, they only cost solve time. Reported as a share.
  plate          hard  the envelope (plan hull's smallest rectangle x height) does not fit the printer's 180 mm cube at any
                       rotation (Spark 29 Oct): the part must be split into bolted pieces that each fit.
  drawn_thin     hard  a drawn body (exact geometry after its cuts, thin_ok bodies left out) under 2.5 mm across its smallest
                       extent: a plate the gate's 2.0 mm erosion leaves no core, so the engine's material round it ends as blobs on necks.
                       4 Nov: the Scout tail unit's 2 mm tape lands under the UBEC and XR2 (two necks), the arm clips' tie collar
                       sheared by the plate keep-out to a 1.1 mm sliver. Thicken it (lands >= 4 mm and 2 cells), move it out of
                       the cut, or declare it thin_ok (the camera posts).
  env_neck       hard  the room (design + drawn cells) eroded by half a member no longer joins every loaded drawn body to the
                       support: the only way from that body to the support is a slab or a stub thinner than a member, so the
                       engine builds a blob there on a neck the gate refuses. 5 Nov: the Scout tail unit's tail-node arm met its
                       bay over a 4 mm overlap (three necks), the nose unit's lower camera posts had 1.2 mm of room beside their
                       2 mm stubs behind the plug keep-out (two loose blobs). Widen the overlap or the room round the body.
  cases          soft  no load cases, or no support: the run would be meaningless (raised by Layout anyway).
  min_member     soft  the gate-adjusted min member > 2 x the floor: the drawn bodies are so thick that every strut
                       must be very fat; usually a drawn block again.
"""
import numpy as np
from scipy import ndimage as ndi


S6 = ndi.generate_binary_structure(3, 1)
S26 = np.ones((3, 3, 3), bool)


class SetupError(ValueError):
    pass


PLATE_MM = 180.0      # Bambu A1 mini: 180 x 180 plate, 180 tall (CLAUDE.md design rules)


def plate_fit(env, h):
    """(L, W, H, fits): the smallest rectangle round the envelope's plan hull, its height, and whether some rotation on the plate
    fits (the rotated rectangle's bounding square inside the plate, a sufficient test)."""
    from scipy.spatial import ConvexHull
    ij = np.argwhere(env.any(axis=2)).astype(float)
    kz = np.where(env.any(axis=(0, 1)))[0]
    H = float(kz.max() - kz.min() + 1) * h if len(kz) else 0.0
    if len(ij) < 3:
        return 0.0, 0.0, H, H <= PLATE_MM
    pts = ij[ConvexHull(ij).vertices] * h
    best = None
    for ang in np.radians(np.arange(0.0, 90.0, 1.0)):
        c, s_ = np.cos(ang), np.sin(ang)
        r = pts @ np.array([[c, -s_], [s_, c]]); ext = r.max(0) - r.min(0) + h
        if best is None or ext.prod() < best.prod():
            best = ext
    L, W = float(max(best)), float(min(best))
    fits = H <= PLATE_MM and any(L * abs(np.cos(t)) + W * abs(np.sin(t)) <= PLATE_MM and L * abs(np.sin(t)) + W * abs(np.cos(t)) <= PLATE_MM
                                 for t in np.radians(np.arange(0.0, 90.5, 0.5)))
    return L, W, H, bool(fits)




def diameter_mm(mask, h):
    """inscribed diameter of a voxel body, to the cell: the distance transform counts cell centres, so a slab n cells thick
    reads n + 1 cells across when n is odd (29 Oct: the tube arm roots' 8 mm boss blocks measured 9 mm at h 1.5 and were
    refused as blocks, while the same blocks measured 8 at h 2.0). 2 * (EDT - 0.5) * h: n * h for odd n, (n - 1) * h for even,
    never over, so the limit means the same at every pitch."""
    if not mask.any():
        return 0.0
    return max(0.0, (2.0 * float(ndi.distance_transform_edt(mask).max()) - 1.0) * h)


THIN_MM = 2.5             # a drawn body thinner than this across its smallest extent: the gate's 2.0 mm erosion leaves it no core (PETG: nothing under 2)


def thin_drawn(drawn_m, cuts_m, thin_ok=(), thin_mm=THIN_MM, min_mm3=5.0):
    """exact geometry, independent of the pitch: the drawn bodies less the cuts and the declared-thin bodies, split into pieces; a
    piece whose smallest bounding-box extent is under thin_mm is a plate or a sliver. Returns [(extent_mm, volume_mm3, centre)]"""
    body = drawn_m - cuts_m
    for t in thin_ok:
        body = body - t
    out = []
    for piece in body.decompose():
        v = float(piece.volume())
        if v < min_mm3:
            continue
        bb = piece.bounding_box(); ext = [bb[k + 3] - bb[k] for k in range(3)]
        if min(ext) < thin_mm:
            out.append((round(min(ext), 2), round(v, 1), [round((bb[k] + bb[k + 3]) / 2, 1) for k in range(3)]))
    return out


def audit(part, min_member_mm, log=print, hard=True, floor_mm=3.2):
    sh, h = part.shape, part.h
    sol = (part.solid & ~part.insert_cells).reshape(sh)
    des = part.design.reshape(sh)
    cut = part.cut.reshape(sh)
    in_env = part.in_env0.reshape(sh) | sol
    rep, fails = {}, []
    mm = float(min_member_mm)
    r_mm = max(1, int(round(mm / h)))

    # drawn share
    n_sol, n_des = int(sol.sum()), int(des.sum())
    share = n_sol / max(n_sol + n_des, 1)
    rep["drawn_share"] = round(share, 3)
    if share > 0.30:
        fails.append(f"drawn_share: drawn solid is {share:.0%} of the part volume (limit 30 %). Attachments only: "
                     f"pads, collars, seats. Remove the drawn blocks and let the engine grow the body.")

    # supports and loaded cells
    sup = np.zeros(part.nel, bool)
    for el in part.support_cells.values():
        sup[el] = True
    sup = sup.reshape(sh)
    loaded = np.zeros(part.nel, bool)
    for c in part.cases:
        if c.nodes is not None and len(c.nodes):
            # nodes -> the solid cells touching them: mark cells whose any corner node is loaded
            nx, ny, nz = sh
            nid = np.asarray(c.nodes)
            k = nid % (nz + 1); j = (nid // (nz + 1)) % (ny + 1); i = nid // ((nz + 1) * (ny + 1))
            for a in (0, -1):
                for b in (0, -1):
                    for cc in (0, -1):
                        ii, jj, kk = i + a, j + b, k + cc
                        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                        loaded[np.ravel_multi_index((ii[ok], jj[ok], kk[ok]), sh)] = True
    loaded = loaded.reshape(sh) & sol

    # drawn bodies
    lab, nb = ndi.label(sol, S26)
    bodies = []
    for b in range(1, nb + 1):
        cells = lab == b
        n = int(cells.sum())
        if n < 8:
            continue
        inscribed = diameter_mm(cells, h)
        # core thickness: distance to the nearest non-body cell OR cut, so a bore inside a collar counts as surface.
        # A block has cells far from every cut and every edge; an attachment (collar, pad, seat) does not.
        core_mm = diameter_mm(cells & ~ndi.binary_dilation(cut, S6), h)
        touch_cut = bool((ndi.binary_dilation(cells, S6) & cut).any())
        touch_sup = bool((cells & sup).any())
        is_loaded = bool((cells & loaded).any())
        surf = cells & ~ndi.binary_erosion(cells, S6)
        near = ndi.binary_dilation(cells, S6, iterations=2 * r_mm) & ~cells & in_env
        room = float((near & des).sum()) / max(float(near.sum()), 1.0)
        # room depth: longest straight run of design cells starting next to the body, along +-x, +-y, +-z
        depth = 0
        if is_loaded or touch_sup:
            # deepest straight run of design cells leaving the body along any axis, over the body's whole face
            # (not one centre line: a seat with a shaft bore has nothing on its centre line and plenty beside it)
            for ax in range(3):
                for sgn in (1, -1):
                    run = np.zeros(sh, int)
                    alive = ndi.binary_dilation(cells, S6) & ~cells & des
                    front = alive.copy()
                    n_run = 0
                    while front.any() and n_run < 200:
                        n_run += 1
                        front = np.roll(front, sgn, axis=ax) & des
                        if sgn > 0:
                            front[(slice(None),) * ax + (0,)] = False
                        else:
                            front[(slice(None),) * ax + (-1,)] = False
                    depth = max(depth, n_run - 1)
        bodies.append(dict(id=b, cells=n, inscribed_mm=round(inscribed, 1), touches_cut=touch_cut,
                           touches_support=touch_sup, loaded=is_loaded, room=round(room, 2),
                           room_depth_mm=round(depth * h, 1),
                           at=[round(float(v), 1) for v in (np.argwhere(cells).mean(0) + 0.5) * h + np.asarray(part.origin)]))
        bodies[-1]["core_mm"] = round(core_mm, 1)
        if core_mm > 2.5 * floor_mm:
            fails.append(f"drawn_blocks: drawn body {b} at {bodies[-1]['at']} has a solid core {core_mm:.0f} mm thick "
                         f"away from every hole and face (limit {2.5 * floor_mm:.0f} mm): a drawn block, not an "
                         f"attachment. Delete it and open design room there instead.")
        if (is_loaded or touch_sup) and room < 0.35:
            fails.append(f"room: loaded drawn body {b} at {bodies[-1]['at']} has design room on only {room:.0%} of its "
                         f"surface (need 35 %): boxed in by cuts or the envelope. Open the envelope round it.")
    if bodies:
        big = max(bodies, key=lambda b: b["cells"])
        dom = big["cells"] / max(n_sol, 1)
        big["share_of_drawn"] = round(dom, 2)
        if dom > 0.6 and big["core_mm"] > 2.0 * floor_mm and len(bodies) > 1:
            fails.append(f"drawn_blocks: drawn body {big['id']} at {big['at']} holds {dom:.0%} of all drawn material with a "
                         f"{big['core_mm']:.0f} mm core: that is a drawn structure, not attachments. Keep only the collars, "
                         f"pads and seats; delete the blocks and bars; open design room where they were.")
    rep["bodies"] = bodies
    shallow = [b for b in bodies if b["loaded"] and b["room_depth_mm"] < 1.0 * mm]
    for b in shallow:
        fails.append(f"room_depth: loaded drawn body {b['id']} at {b['at']} has {b['room_depth_mm']} mm of design room "
                     f"behind it (need {mm:.0f} mm, one member): nothing can grow onto it. Extend the envelope behind it.")
    warn = [b for b in bodies if b["loaded"] and not b["touches_support"] and mm <= b["room_depth_mm"] < 3.0 * mm]
    if warn:
        rep["room_depth_warn"] = [b["id"] for b in warn]
        log(f"audit: WARNING room behind loaded bodies {[b['id'] for b in warn]} is under {3 * mm:.0f} mm: expect "
            f"thin bars there, not a web. Extend the envelope behind them.")

    # envelope waste
    if n_des:
        anchors = sol | sup
        if anchors.any():
            far = ndi.distance_transform_edt(~anchors) * h > 8.0 * mm
            waste = float((far & des).sum()) / n_des
            rep["envelope_waste"] = round(waste, 3)
            if waste > 0.25:
                log(f"audit: WARNING {waste:.0%} of the design room is over {8 * mm:.0f} mm from every drawn body and "
                    f"support: it will stay empty and only costs solve time. Trim the envelope.")

    L, W, H, fits = plate_fit(in_env, h)
    rep["plate"] = dict(L_mm=round(L, 1), W_mm=round(W, 1), H_mm=round(H, 1), fits=fits)
    if not fits:
        fails.append(f"plate: the envelope is {L:.0f} x {W:.0f} x {H:.0f} mm and does not fit the {PLATE_MM:.0f} mm plate at any "
                     f"rotation: split the part into bolted pieces that each fit, joined so the positions it sets are kept "
                     f"(Spark 29 Oct).")

    # env_neck: every loaded drawn body must reach the support through room at least one member wide
    room = (des | sol)
    r_half = max(1, int(np.ceil(mm / 2.0 / h)))
    core = ndi.binary_erosion(room, S6, iterations=r_half)
    core = ndi.binary_dilation(core, S6, iterations=r_half) & room          # the room without its slabs and stubs thinner than a member
    lab_c, nc = ndi.label(core, S26)
    sup_ids = set(np.unique(lab_c[sup & core])) - {0}
    for b in bodies:
        if not b["loaded"] or b["touches_support"]:
            continue
        cells = lab == b["id"]
        ids = set(np.unique(lab_c[ndi.binary_dilation(cells, S6) & core])) - {0}
        if sup_ids and not (ids & sup_ids):
            fails.append(f"env_neck: loaded drawn body {b['id']} at {b['at']} reaches the support only through room thinner than one member "
                         f"({mm:.1f} mm): a slab or a stub the engine can only bridge with a blob on a neck. Widen the overlap or the room round it.")
    thin = thin_drawn(part.drawn_m, part.cuts_m, getattr(part, "thin_ok", ()) or ())
    rep["drawn_thin"] = [dict(extent_mm=t[0], mm3=t[1], at=t[2]) for t in thin]
    for t in thin:
        fails.append(f"drawn_thin: a drawn body at {t[2]} ({t[1]:.0f} mm3) is {t[0]:.1f} mm across its smallest extent (limit {THIN_MM}): a plate or "
                     f"sliver the gate's 2 mm erosion leaves no core (a 2 mm tape land, a collar sheared by a keep-out). Thicken it, move it "
                     f"out of the cut, or declare it thin_ok.")

    rep["cases"] = len(part.cases)
    rep["supports"] = len(part.supports)
    if not part.cases or not part.supports:
        fails.append("cases: no load cases or no support: nothing to optimise for.")

    rep["fails"] = fails
    for f in fails:
        log("audit: FAIL " + f)
    if fails and hard:
        raise SetupError("setup audit failed:\n  " + "\n  ".join(fails))
    if not fails:
        log(f"audit: OK  drawn {share:.0%}, {len(bodies)} drawn bodies, room ok")
    return rep
