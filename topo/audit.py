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
                       hole or the support face). Also the drawn-structure variant: one body with over 60 % of the drawn
                       material and a core over 2 x the floor. Both skip drawn material inside the solid_ok bodies
                       (Part(..., solid_ok=[...]): solid on purpose, Spark 9 Nov, the arm motor mounts); listed as "solid_ok".
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
  load_room      hard  a load case with over a third (and over 2) of its loaded cells in design room, not drawn and not beside
                       a cut (a bolt hole's lining is the wall the engine builds round the hole): the region reaches past the
                       drawn face it loads, so the engine grows a knob or a lip out to those cells that carries nothing. 5 Nov: the Scout nose unit's seat side-face regions ran 2.7 mm above the 3 mm rim up the
                       chamfer (two 150 mm3 knobs on necks). Clip the region to the drawn face.
  env_neck       hard  the room (design + drawn cells) eroded by half a member no longer joins every loaded drawn body to the
                       support: the only way from that body to the support is a slab or a stub thinner than a member, so the
                       engine builds a blob there on a neck the gate refuses. 5 Nov: the Scout tail unit's tail-node arm met its
                       bay over a 4 mm overlap (three necks), the nose unit's lower camera posts had 1.2 mm of room beside their
                       2 mm stubs behind the plug keep-out (two loose blobs). Widen the overlap or the room round the body.
  drawn_wall     hard  a drawn body (after its cuts, thin_ok bodies left out) with a wall under the gate's thin rule (the 0.9 mm
                       opening's residue: under the 2 mm floor) anywhere: the gate fails it wherever the engine leaves it exposed
                       and the export cleanup may not cut drawn material (7 Nov: the Scout node collars at their resin grooves).
  room_full      hard  (room_strength, on the Layout before the first solve) the as-built strength index with every design cell
                       filled solid: the strongest part the room can hold. Over max_worst, no layout in it can pass whatever
                       the budget, so the run is refused. Also reported: the budget's share of the room and full / share, the
                       index a uniformly thinned room would read (a rough guide: a good layout beats it). 9 Nov: the Scout v4
                       frame's side lugs (16 x 8 room carrying 134 N 29 mm under the deck) read 2.55 after a 69 min run.
                       Open the room along the load path (deeper behind the socket, wider at the root), move the support
                       nearer the load, or shorten the lever. When the peak sits in a fin (room_fins: design room one cell
                       thin), the refusal says so and gives its box: close the fin first (13 Nov, the funnel's nose half).
  parity         hard  a column of the grid where the room, the drawn or the cuts mesh has an odd number of distinct crossings: the
                       ray-parity voxeliser (part.voxelise) then flips that column from the odd crossing up. Two bodies that only
                       touch on a face (a box stacked flush on another) stay two shells in a manifold union, the ray meets the
                       shared face twice at one height and counts it once. 11 Nov, the Scout v4 frame: the flow land's top flush
                       on the nose room's floor (z -10) and the side rooms' tops flush under the deck (z 9.6) emptied 52 columns
                       up to the deck: a 1 mm U slot round the flow pod, 1.4 mm walls, two blobs on necks. Overlap the bodies.
  memory         hard  (memory_fit, on the Layout before the first solve; memory_measured after the first iteration) the run's
                       peak memory over Settings.mem_limit_gb (14 GB of the GitHub runner's 16): the runner kills such a run
                       with no log and no result (10 Nov: scout_v4_frame, 396 cases, 14.3 GB, died at 2 h). The estimate is
                       what the process holds + the solver (direct factor and assembly, or the multigrid hierarchy and one
                       warm start per case) + the per-case element energies + one block of right-hand sides; after the first
                       iteration the measured peak (ru_maxrss) is checked too. The reason goes to result.json (status
                       "mem_refused"). Coarsen h, trim the envelope (fewer active dofs), or merge drawn bodies (fewer tie cases).
                       The build and gate after the layout too (build_fit: held + GATE_BYTES a cell of the 0.25 mm gate grid over
                       the part's box + the export's mesh), from the envelope's box before the first solve and from the part's
                       before the export cleanup and before the solid check (10 Nov: the frame was killed twice right after the
                       export): a part over it ends with status mem_refused and the reason (split it: the grid is its box).
  bed            hard where the setup asks (part.bed_min_mm2), reported for every part: the flat the DRAWN solid lays on the bed, i.e. its
                       section 0.1 mm inside the room's extreme plane along the build direction. The engine knows the build direction
                       only as the layers' weak axis: nothing makes a result touch the bed. 14 Nov, the funnel's halves (build -z, 176 wide):
                       both passed strength and the gate and stood on 40 to 66 mm2, the side walls' top edges being square to their
                       55 deg faces (a line) and the rib's room ending 1.0 short of that plane. Draw a flat rim on the bed plane and end
                       the room on it. Under bed_min_mm2 the run is refused.
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


class MemoryRefused(SetupError):
    """the run would not fit the runner's memory (memory_fit / memory_measured); .info is the report for result.json"""

    def __init__(self, msg, info):
        super().__init__(msg)
        self.info = info


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


def _peak_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    except Exception:
        return 0.0


def memory_estimate(L):
    """(total GB, parts dict) for a Layout's optimisation: what the process holds now + its solver + the per-case element
    energies + one block of right-hand sides in flight (fem.Solver.solve_blocks). Measured 10 Nov on scout_v4_frame (see
    memory_fit's log line for estimate vs peak)."""
    sol = L.fe_solver
    g = sol.grid
    real = [c for c in L.cases if not c.tie] if getattr(L.s, "nominal_hold", False) else []   # the nominal-hold copies (layout.py)
    nc = len(L.cases) + len(real)
    n_fe = sum(c.fe is not None for c in L.cases) + sum(c.fe is not None for c in real)
    GB = 1024.0 ** 3
    blk = sol.block_size()
    parts = dict(held=_rss_gb())
    if sol.backend == "direct":
        nnz = len(sol.d_indices)
        parts["factor"] = sol.direct_mem_estimate()
        parts["assembly"] = nnz * 40.0 / GB                  # K, K[free][:, free], its upper triangle, int32 index copies
        parts["block"] = blk * (2.0 * g.ndof + 4.0 * sol.d_n) * 8.0 / GB   # F and U (full dofs), F[free], Fortran copy, x, C copy
    else:
        lv = sum(g.nel / 8 ** l * 576 * 8.0 for l in range(1, sol.levels))
        parts["factor"] = lv / GB + 0.5                       # coarse element matrices + the coarsest direct factor
        parts["warm"] = nc * g.ndof * 8.0 / GB                 # one warm start per case
        parts["block"] = blk * g.ndof * 8.0 * 10 / GB          # CG vectors (X, R, Z, P, AP, B) + V-cycle temporaries
    parts["energies"] = (nc + n_fe) * g.nel * 8.0 / GB
    return sum(parts.values()), {k: round(v, 2) for k, v in parts.items()}


def memory_fit(L, limit_gb, log=print, hard=True):
    """memory (see the module doc), before the first solve. Returns the report; raises MemoryRefused over the limit when hard."""
    tot, parts = memory_estimate(L)
    sol = L.fe_solver
    rep = dict(estimate_gb=round(tot, 2), parts=parts, limit_gb=limit_gb, cases=len(L.cases), active_dofs=sol.n_active_dofs,
               backend=sol.backend, block=sol.block_size())
    msg = (f"memory: estimated peak {tot:.1f} GB ({', '.join(f'{k} {v}' for k, v in parts.items())}) for {len(L.cases)} cases on "
           f"{sol.n_active_dofs:,} active dofs ({sol.backend})")
    if tot > limit_gb:
        why = (f"{msg}: over the {limit_gb:.0f} GB limit (the runner has 16 GB and kills a run over it with no log). Coarsen h, trim "
               f"the envelope (fewer active dofs) or join drawn bodies (fewer tie cases: 6 per unheld body).")
        log("audit: FAIL " + why)
        rep["fail"] = why
        if hard:
            raise MemoryRefused("setup audit failed:\n  " + why, rep)
    else:
        log(f"audit: {msg}, limit {limit_gb:.0f} GB")
    # the build and gate steps after the layout, on the envelope's box (the part's bounds are inside it): a part whose gate grid
    # cannot fit is refused before the run, not killed after it (10 Nov, scout_v4_frame, twice right after the export)
    part = L.part
    g0 = np.asarray(part.origin, float); g1 = g0 + np.asarray(part.shape) * part.h
    try:
        b = np.asarray(part.env_m.bounding_box(), float)
        lo, hi = np.maximum(b[:3], g0), np.minimum(b[3:], g1)
    except Exception:
        lo, hi = g0, g1
    rep["build"] = build_fit(lo, hi, limit_gb, held_gb=parts["held"], log=log, hard=hard, when="from the envelope")
    if "fail" in rep["build"]:
        rep["fail"] = rep["build"]["fail"]
    return rep


def memory_measured(L, limit_gb, log=print, hard=True):
    """memory after the first iteration: the measured peak (ru_maxrss) against the limit and the estimate."""
    peak = _peak_gb()
    rep = getattr(L, "mem", None) or {}
    rep["peak_it1_gb"] = round(peak, 2)
    msg = f"memory: peak {peak:.1f} GB after the first iteration (estimate {rep.get('estimate_gb')} GB, limit {limit_gb:.0f} GB)"
    if peak > limit_gb:
        why = f"{msg}: over the limit, the runner (16 GB) would kill this run. Coarsen h, trim the envelope or join drawn bodies."
        log("audit: FAIL " + why)
        rep["fail"] = why
        if hard:
            raise MemoryRefused("memory gate:\n  " + why, rep)
    else:
        log(msg)
    return rep


GATE_PITCH = 0.25        # the solid gate's voxel pitch (tools/solid_check.py solid_check, topo/build.py export_cleanup)
GATE_MARGIN = 2          # voxels of air round the part's bounds on the gate grid
GATE_BYTES = 45.0        # bytes per gate cell at the gate's peak, its voxelisation included (bool grids, labels, one float64
                         # distance field held and one being built by solid_check.edt). Measured 10 Nov on scout_v4_frame's mesh
                         # (82 M cells, 524 k faces): +3.09 GB = 37.5 B a cell (the old gate, scipy's transform: +7.57 GB, 98 B)
MESH_BYTES = 6000.0      # bytes per face of the surface still to be exported (the boolean's mesh, the weld, their garbage until
                         # the collector runs): the frame's export took the process from 1.05 to 5.28 GB on an 838 k-face surface


def gate_voxels(lo, hi, pitch=GATE_PITCH):
    """cells of the gate grid over the box lo..hi (the part's bounds)"""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return int(np.prod([np.ceil((hi[k] - lo[k] + 2 * GATE_MARGIN * pitch) / pitch) for k in range(3)]))


def build_estimate(lo, hi, faces=0, held_gb=None):
    """(total GB, parts, gate cells) of the build: what the process holds + the gate grid (export cleanup and solid check,
    GATE_BYTES a cell at GATE_PITCH over the part's bounds) + the export of a surface of `faces` faces still to come. 10 Nov: scout_v4_frame (180 x 180 x 44 mm, 94 M
    gate cells, 838 k faces) was killed on the 16 GB runner right after the export, the gate's distance transforms at ~5 GB each."""
    n = gate_voxels(lo, hi)
    GB = 1024.0 ** 3
    parts = dict(held=_rss_gb() if held_gb is None else float(held_gb), gate=GATE_BYTES * n / GB, mesh=MESH_BYTES * faces / GB)
    return sum(parts.values()), {k: round(v, 2) for k, v in parts.items()}, n


def build_fit(lo, hi, limit_gb, faces=0, held_gb=None, log=print, hard=True, when="before the gate"):
    """memory of the build and gate steps (see the module doc): over the limit raises MemoryRefused (hard), so the run ends with
    result.json status mem_refused and the reason, not with the runner killed after the export."""
    tot, parts, n = build_estimate(lo, hi, faces, held_gb)
    size = np.round(np.asarray(hi, float) - np.asarray(lo, float), 1).tolist()
    rep = dict(estimate_gb=round(tot, 2), parts=parts, limit_gb=limit_gb, gate_cells=n, bounds_mm=size, faces=int(faces), when=when)
    msg = (f"memory: build {when}: estimated peak {tot:.1f} GB ({', '.join(f'{k} {v}' for k, v in parts.items())}) for a "
           f"{' x '.join(str(x) for x in size)} mm part ({n / 1e6:.0f} M gate cells at {GATE_PITCH} mm{f', {faces:,} faces' if faces else ''})")
    if tot > limit_gb:
        why = (f"{msg}: over the {limit_gb:.0f} GB limit (the runner has 16 GB and kills a run over it with no result). Split the part "
               f"(the gate grid grows with its bounding box), or trim the envelope.")
        log("audit: FAIL " + why)
        rep["fail"] = why
        if hard:
            raise MemoryRefused("memory gate:\n  " + why, rep)
    else:
        log(f"audit: {msg}, limit {limit_gb:.0f} GB")
    return rep


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




def bed_contact(drawn_m, cuts_m, env_m, build, inside=0.1):
    """(mm2 of the drawn solid, mm2 of the whole room) in the plane `inside` mm above the bed: the room's extreme along the build
    direction ("-z": the room's top is on the bed). Exact sections of the manifolds, not cells."""
    T = {"+z": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "-z": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]],      # proper rotations taking the build direction to +z
         "+x": [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], "-x": [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]],
         "+y": [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0]], "-y": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]]}[build]
    T = np.array(T, float)
    room = ((env_m + drawn_m) - cuts_m).transform(T.tolist()); d = (drawn_m - cuts_m).transform(T.tolist())
    z = room.bounding_box()[2] + inside
    return float(d.slice(z).area()), float(room.slice(z).area())


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


WALL_CORE_MM3 = 1.0       # a drawn wall under the gate's thin rule counts from this much core (voxel noise at an acute edge is less)
WALL_PITCHES = (0.15, 0.2, 0.25, 0.4, 0.5, 0.8)   # drawn_walls samples at one of these: a 2.0 wall holds a whole number of cells, so its
#   middle cell is 1.0 from both faces and the 0.9 opening keeps it (at 0.3 a 2.0 wall can cover 6 cells, middle 0.88: a false thin wall)


def drawn_walls(drawn_m, cuts_m, thin_ok=(), pitch=0.15, erode_mm=0.9, min_core_mm3=WALL_CORE_MM3, max_voxels=20e6, min_mm3=5.0):
    """the gate's own thin rule (tools/solid_check.py thin_walls: the residue of a 2 x erode_mm opening, walls under the 2 mm floor)
    on the drawn geometry alone, body by body: the drawn bodies less the cuts and the declared-thin bodies, each piece voxelised at
    pitch (finer than the gate's 0.25: a 1.5 mm ring on a grid axis is 6 voxels there and its residue breaks into edge pieces; coarsened
    so a piece stays under max_voxels). A wall found here is in the part whatever the engine does: where the generated
    material leaves it exposed the gate fails it, and the cleanup may not touch drawn material. Returns [dict(thick_mm, core_mm3, mm3,
    at, pitch)] (7 Nov: the Scout node collars, 2.0 wall round a 1.6 bore with 0.5 resin grooves, are 1.5 at every groove ring)."""
    from .build import _solid_check_module
    from .part import tm
    SC = _solid_check_module()
    body = drawn_m - cuts_m
    for t in thin_ok:
        body = body - t
    out = []
    for whole in body.decompose():
        if float(whole.volume()) < min_mm3:
            continue
        for piece, keep in _wall_tiles(whole, max_voxels):
            T = tm(piece)
            if not len(T.faces):
                continue
            ext = T.bounds[1] - T.bounds[0]
            p = float(pitch)
            for p in [q for q in WALL_PITCHES if q >= pitch]:             # pitches a 2.0 wall sits on with a centre 1.0 from its faces
                if np.prod((ext + 4 * p) / p) <= max_voxels:
                    break
            lo = T.bounds[0] - 2 * p
            shape = tuple(int(np.ceil((ext[k] + 4 * p) / p)) for k in range(3))
            V = SC.voxels(T, p, lo, shape)
            _, _, thin, _ = SC.thin_walls(V, p, erode_mm, lo)
            for t in thin:
                if keep is not None and not (keep[1] <= t["at"][keep[0]] < keep[2]):
                    continue                                              # in a slab's margin: the next slab holds it whole
                if t["kind"] == "wall" and t["core_mm3"] >= min_core_mm3:
                    out.append(dict(thick_mm=t["thick_mm"], core_mm3=t["core_mm3"], mm3=t["mm3"], at=t["at"], pitch=round(p, 3)))
    return sorted(out, key=lambda t: -t["core_mm3"])


def _wall_tiles(piece, max_voxels, pitch=0.25, margin=4.0):
    """a drawn body too large for one grid at the gate's 0.25 pitch, as slabs along its longest extent, each with a margin either
    side: [(slab, (axis, from, to))]; a body that fits: [(body, None)]. 13 Nov, the landing funnel's frame halves: one drawn shell
    115 x 121 x 40 (two 3.0 walls at 55 deg and their ledges) was coarsened to 0.4, where a sloped 3.0 wall reads 1.1 thick over
    600 mm3: a refusal for a wall the 0.15 grid finds nowhere. A finding counts in the slab whose own range holds it; a sliver a
    slab's cut makes lies in a margin and is dropped there."""
    import manifold3d as mf
    bb = np.asarray(piece.bounding_box(), float); lo, hi = bb[:3], bb[3:]; ext = hi - lo
    if np.prod((ext + 4 * pitch) / pitch) <= max_voxels:
        return [(piece, None)]
    ax = int(np.argmax(ext)); other = float(np.prod((np.delete(ext, ax) + 4 * pitch) / pitch))
    length = max_voxels / other * pitch - 2 * margin - 4 * pitch
    if length < 2 * margin:                                               # too large across as well: the coarser grid, as before
        return [(piece, None)]
    edges = np.linspace(lo[ax], hi[ax], int(np.ceil(ext[ax] / length)) + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        blo, bhi = lo - 1.0, hi + 1.0
        blo[ax], bhi[ax] = a - margin, b + margin
        out.append((piece ^ mf.Manifold.cube([float(v) for v in bhi - blo]).translate([float(v) for v in blo]), (ax, float(a), float(b) + (1e-6 if b == edges[-1] else 0.0))))
    return out


def parity_columns(mesh, xs, ys, chunk=20000):
    """the (x, y) columns of a grid where a closed mesh has an odd number of distinct crossings along +z (part.voxelise's rays,
    the same offsets and the same 1e-6 merge): the voxeliser inverts each such column above its odd crossing. Returns
    [(x, y, [z of the crossings])]."""
    if len(mesh.faces) == 0:
        return []
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    org = np.stack([XX.ravel() + 1.234e-5, YY.ravel() + 2.345e-5, np.full(XX.size, float(mesh.bounds[0][2]) - 5.0)], 1)
    bad = []
    for c0 in range(0, len(org), chunk):
        o = org[c0:c0 + chunk]
        loc, ray, _ = mesh.ray.intersects_location(o, np.tile([0, 0, 1.0], (len(o), 1)), multiple_hits=True)
        if not len(ray):
            continue
        order = np.argsort(ray, kind="stable"); ray = np.asarray(ray)[order]; z = loc[order][:, 2]
        st = np.r_[0, np.where(np.diff(ray))[0] + 1, len(ray)]
        for a, b in zip(st[:-1], st[1:]):
            zz = np.sort(z[a:b]); zz = zz[np.r_[True, np.diff(zz) > 1e-6]]
            if len(zz) % 2:
                bad.append((round(float(o[ray[a], 0]), 1), round(float(o[ray[a], 1]), 1), [round(float(v), 2) for v in zz]))
    return bad


def audit(part, min_member_mm, log=print, hard=True, floor_mm=3.2):
    sh, h = part.shape, part.h
    sol = (part.solid & ~part.insert_cells).reshape(sh)
    des = part.design.reshape(sh)
    cut = part.cut.reshape(sh)
    in_env = part.in_env0.reshape(sh) | sol
    rep, fails = {}, []
    mm = float(min_member_mm)
    # declared-solid bodies (Part(..., solid_ok=[...])): drawn_blocks and the drawn-structure check skip the drawn cells inside them
    from .part import tm
    ok_solid = np.zeros(sh, bool)
    rep["solid_ok"] = []
    for m in getattr(part, "solid_ok", ()) or ():
        cells_ok = part.vox(tm(m)).reshape(sh) & sol
        ok_solid |= cells_ok
        bb = np.asarray(m.bounding_box(), float)
        rep["solid_ok"].append(dict(mm3=round(float(m.volume()), 1), cells=int(cells_ok.sum()),
                                    at=[round(float((bb[k] + bb[k + 3]) / 2), 1) for k in range(3)]))
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

    # load_room: a load region must lie on drawn material. Its cells are the part cells whose centre it holds (Part.elems); those that
    # are design room, not drawn, are nodes the engine must grow material out to so the load is carried: a knob or a lip that carries
    # nothing (5 Nov, Scout nose run 6: the seat's side-face regions reached 2.7 mm above the 3 mm rim up the chamfer, 2 of their 3
    # cell layers were room, two 150 mm3 knobs on necks at the rim). A region that wraps a body by a margin has a few room cells on
    # its surface (the camera posts' root regions, 0.6 over the boss): under a third is fine. Room cells beside a cut are a bearing
    # lining (a bolt's shear on its hole's wall, the Overseer arms): the wall the engine builds round the hole is the attachment
    # itself, so they do not count.
    rep["load_room"] = []
    nx, ny, nz = sh
    lining = ndi.binary_dilation(cut, S6)
    for c in part.cases:
        if c.nodes is None or not len(c.nodes):
            continue
        ln = np.zeros((nx + 1) * (ny + 1) * (nz + 1), bool); ln[np.asarray(c.nodes)] = True
        ln = ln.reshape(nx + 1, ny + 1, nz + 1)
        cells = np.ones(sh, bool)                                           # a cell is loaded when all 8 of its corner nodes are
        for a in (0, 1):
            for b in (0, 1):
                for cc in (0, 1):
                    cells &= ln[a:a + nx, b:b + ny, cc:cc + nz]
        cells &= (sol | des)
        room_c = cells & des & ~sol & ~lining
        n_all, n_room = int(cells.sum()), int(room_c.sum())
        if n_all and n_room > 2 and n_room > n_all / 3.0:
            ii, jj, kk = np.nonzero(room_c)
            at = [round(float(part.origin[d] + (v.mean() + 0.5) * h), 1) for d, v in enumerate((ii, jj, kk))]
            rep["load_room"].append(dict(case=c.name, cells=n_all, in_room=n_room, at=at))
            fails.append(f"load_room: case '{c.name}' loads {n_room} of {n_all} cells in design room (not drawn, not a cut's lining) near {at}: the engine grows "
                         f"material out to them to carry the load, a knob or a lip that carries nothing. Clip the region to the drawn face it loads.")

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
        core_mm = diameter_mm(cells & ~ndi.binary_dilation(cut, S6) & ~ok_solid, h)   # solid_ok cells count as surface
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
        bodies[-1]["free_cells"] = int((cells & ~ok_solid).sum())           # drawn cells outside the solid_ok bodies
        if bodies[-1]["free_cells"] < n:
            bodies[-1]["solid_ok_cells"] = n - bodies[-1]["free_cells"]
        if core_mm > 2.5 * floor_mm:
            fails.append(f"drawn_blocks: drawn body {b} at {bodies[-1]['at']} has a solid core {core_mm:.0f} mm thick "
                         f"away from every hole and face (limit {2.5 * floor_mm:.0f} mm): a drawn block, not an "
                         f"attachment. Delete it and open design room there instead.")
        if (is_loaded or touch_sup) and room < 0.35:
            fails.append(f"room: loaded drawn body {b} at {bodies[-1]['at']} has design room on only {room:.0%} of its "
                         f"surface (need 35 %): boxed in by cuts or the envelope. Open the envelope round it.")
    if bodies:
        big = max(bodies, key=lambda b: b["free_cells"])
        dom = big["free_cells"] / max(n_sol - int(ok_solid.sum()), 1)
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
    r_half = max(1, int(mm / 2.0 / h))                                      # half a member in whole cells, rounded down: at h 2.0 and member 4.4 that is 1 cell (ceil gave 2: 8 mm of erosion, which cut a 9 mm yoke)
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
    walls = drawn_walls(part.drawn_m, part.cuts_m, getattr(part, "thin_ok", ()) or ())
    rep["drawn_walls"] = walls
    for i, w in enumerate(walls[:6]):
        more = f" (and {len(walls) - 6} more)" if i == 5 and len(walls) > 6 else ""
        fails.append(f"drawn_wall: a drawn body has a wall under 2 mm at {w['at']} after its cuts ({w['thick_mm']:.1f} mm by the gate's rule, "
                     f"{w['core_mm3']:.1f} mm3){more}: the gate fails it wherever the engine leaves it exposed, and the cleanup may not cut drawn "
                     f"material (a 2.0 collar wall with 0.5 resin grooves is 1.5 at the grooves). Thicken the wall, shallow the cut, or declare it thin_ok.")

    bd, br = bed_contact(part.drawn_m, part.cuts_m, part.env_m, part.build)
    need = getattr(part, "bed_min_mm2", None)
    rep["bed"] = dict(build=part.build, drawn_mm2=round(bd, 1), room_mm2=round(br, 1), min_mm2=need)
    if need is not None and bd < need:
        fails.append(f"bed: the drawn solid lays {bd:.0f} mm2 on the bed (build {part.build}, the setup asks {need:.0f}): nothing makes the engine's "
                     f"material touch the bed, so the part would stand on edges. Draw a flat rim or pad on the room's extreme plane along the "
                     f"build direction and end the room on that plane.")

    rep["parity"] = []
    for nm, t in (("room", part.env_t), ("drawn", part.drawn_t), ("cuts", part.cuts_t)):
        cols = parity_columns(t, part.axes[0], part.axes[1])
        if cols:
            zs_ = sorted({c[2][len(c[2]) // 2] for c in cols})[:4]
            rep["parity"].append(dict(mesh=nm, columns=len(cols), at=[list(c[:2]) for c in cols[:4]], z=zs_))
            fails.append(f"parity: the {nm} mesh has {len(cols)} grid columns with an odd number of crossings (near {[list(c[:2]) for c in cols[:3]]}, "
                         f"crossings {cols[0][2]}): two bodies that only touch on a face (one flush on another) stay two shells, and the "
                         f"voxeliser empties or fills those columns from the shared face up. Overlap them (0.5 mm) instead of stacking them flush.")

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


def room_fins(part):
    """design cells in no 2 x 2 x 2 block of design cells: room one cell thin, between a drawn face, a cut and the envelope's edge
    in any mix. Common and harmless where nothing peaks (a skin on a collar); room_strength names it when the peak sits in it."""
    des = np.asarray(part.design, bool).reshape(part.shape)
    return des & ~ndi.binary_opening(np.pad(des, 1), np.ones((2, 2, 2), bool))[1:-1, 1:-1, 1:-1]


def room_strength(L, max_worst, log=print, hard=True):
    """room_full (see the module doc): the build's own as-built strength (topo/build.py strength) on the room filled solid.
    Returns the report dict; raises SetupError over max_worst when hard."""
    from .build import strength
    part = L.part
    full = np.asarray(L.design, bool).ravel() | np.asarray(L.solid, bool).ravel()
    st = strength(L, mask=full, log=lambda s_: None)
    share = float(L.volfrac)
    rep = dict(worst=st["worst"], worst_case=st["worst_case"], worst_at=st["worst_at"], budget_share=round(share, 3),
               thinned=None if st["worst"] is None else round(st["worst"] / max(share, 1e-6), 2), max_worst=max_worst)
    if st["worst"] is None:
        log("audit: room_full: no classed load cases")
        return rep
    # is the peak in a fin (13 Nov, the funnel's nose half: a cut stopped 0.9 mm short of the room's edge, and a wedge under a member
    # wide between a sloped drawn wall and a cut, each one cell at h 2.2 ending on the motor's drawn front plate, read 0.60 and 0.67;
    # three changes to the room along the load path did not move it, and the mirrored half passed only because its grid put that
    # cell among the ones left out beside a loaded node). No layout builds a member there: the fix is to close the fin, and only the
    # reading after that says anything about the room.
    fins = room_fins(part); rep["fin_cells"] = int(fins.sum())
    ijk = np.floor((np.asarray(st["worst_at"], float) - np.asarray(part.origin, float)) / part.h).astype(int)
    lo, hi = np.maximum(ijk - 1, 0), np.minimum(ijk + 2, part.shape)
    near = fins[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    rep["worst_in_fin"] = bool(near.any())
    if rep["worst_in_fin"]:
        lab, _ = ndi.label(fins, S26); k = lab[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]][near][0]
        ii = np.argwhere(lab == k); c0, c1 = (np.asarray(part.origin) + ii.min(0) * part.h).round(1).tolist(), (np.asarray(part.origin) + (ii.max(0) + 1) * part.h).round(1).tolist()
        rep["fin"] = dict(cells=int(len(ii)), lo=c0, hi=c1)
        msg = (f"room_full: the room filled solid reads {st['worst']:.2f} in '{st['worst_case']}' at {st['worst_at']} (max_worst {max_worst}), and that "
               f"peak sits in a fin of room one cell thin ({len(ii)} cells, {c0} to {c1}): room under a member wide between a drawn face, a cut "
               f"or the envelope's edge, which no layout can build in. Close the fin first (run the cut out past the room's edge or up to the "
               f"drawn face), then read the room again: the reading says nothing about the load path until then.")
    else:
        msg = (f"room_full: the room filled solid reads {st['worst']:.2f} in '{st['worst_case']}' at {st['worst_at']} (max_worst {max_worst}): "
               f"no layout in this room can pass. Open the room along that load's path (deeper behind the socket or pad, wider at the root), "
               f"move a support nearer the load, or shorten the lever.")
    if st["worst"] > max_worst:
        log("audit: FAIL " + msg)
        rep["fail"] = msg
        if hard:
            raise SetupError("setup audit failed:\n  " + msg)
    else:
        log(f"audit: room_full {st['worst']:.3f} filled solid ('{st['worst_case']}'), budget {share:.0%} of the room, "
            f"uniformly thinned ~{rep['thinned']}")
    return rep
