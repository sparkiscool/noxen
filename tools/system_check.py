"""Rule checks on the assembled model (Spark 24 Oct: system first, cage first, compact, reachable). One manifest lists
every part in its assembled pose; the tool reports:
  cage first  flat walls: support functions over N Fibonacci directions: margin(d) = h_cage(d) - h_protected(d),
              h(d) = max v.d. PASS if margin >= cage_margin_mm (3.0 DEFAULT) in every direction for every protected part;
              inside an opening cone (axis + half-angle, the cartridge path, the head) that part only needs margin >= 0.
  probes      crash probes, edges and posts that poke between cage members (default on with a cage and a protected
              part): an edge probe is an infinite line (table edge, door frame), a post probe an infinite cylinder of
              radius post_r_mm (10 DEFAULT; pole, branch). 500 Fibonacci probe axes (hemisphere) x 24 approach angles
              round each (15 deg) = 12000 orientations, every lateral offset. Seen along the axis the line is a point and
              the post a disc, so the first contact is solved in that plane: each role's surface points (vertices plus
              edge samples every res_mm, 1.0 DEFAULT: a line first touches a mesh on an edge) are rasterised at res_mm,
              the outline kept, the furthest point along the approach per 1.5 res lateral bin is the line's contact, the
              post's is that profile dilated by a semicircle of radius r. margin = cage contact - part contact. PASS if
              no protected part is met before the cage or with the cage ahead by < probe_margin_mm (0.0 DEFAULT; flat
              walls keep their 3.0); "unshielded" = no cage member in that probe's path at all, always a FAIL (margin
              then = minus the probe's reach past the cage envelope). An opening with "skip_probes": true exempts its
              cone for its part. Protected parts shadow each other: a probe stops at the first solid, so a part
              behind another protected part (an exempt one included) is not reached there. About 6 s on a
              73k-triangle assembly, no decimation needed.
  access      every path that must be reached (rule 2, without taking another part off): the tool swept from `from`
              along `dir` for length_mm, intersected with every other part (manifold3d booleans, trimesh / voxel
              fallback). PASS if each path's overlap <= access_tol_mm3 (0.5). Report: blocked by which part, mm^3.
              tool {"type": "cylinder", "d"} | {"type": "box", "w", "h"(, "w_axis")} (a w x h section swept) | "part"
              (the part's own mesh moved along the path: >= 20 steps, step <= half its depth along dir, unioned; only
              what the move adds counts, the part's overlap at rest is the fit check's). The part itself is always
              ignored; ignore lists the parts the tool may touch. from / dir are in the assembled frame ("frame": "part"
              = the part's own frame). Examples:
                {"name": "pack swap", "part": "pack", "dir": [1,0,0], "length_mm": 120, "tool": "part", "ignore": ["tray"]}
                {"name": "XT60 unplug", "part": "xt60_f", "dir": [0,-1,0], "length_mm": 30, "tool": "part"}
                {"name": "M3 key, arm bolt", "part": "hub", "from": [32,0,6], "dir": [0,0,1], "length_mm": 70,
                 "tool": {"type": "cylinder", "d": 4}}            (d 8 for a driver shaft)
                {"name": "FC USB-C", "part": "fc", "from": [15,0,4], "dir": [1,0,0], "length_mm": 35,
                 "tool": {"type": "box", "w": 12, "h": 7}}
  compact     assembled AABB volume (L), extents (mm), largest extent, convex hull volume (L), guard offsets (reported).
  fit         pairwise interference volume (mm^3, manifold3d booleans, trimesh / voxel fallback). PASS if every pair
              <= fit_tol_mm3 (0.5), a contact_pairs entry [a, b, allowed_mm3] lifts that pair's limit. Minimum clearance
              on clearance_pairs [a, b(, min_mm)] (reported).
  mass / CG   mass_g if given, else mesh volume x density (PETG 1.27, TPU 1.21, carbon 1.6 g/cm^3); bought parts must
              give mass_g. Whole mass and CG. Balance (Spark 24 Oct, CG in the middle): with cg_ref (the thrust centre)
              the CG offset normal to thrust_axis (+z) must be <= cg_tol_mm (3.0 DEFAULT), with and without the parts
              marked "swap": true (pack, cartridge), so a swap does not move the CG off the axis.
  budget      mass_budget_g (whole system) and per-part budget_g (dimensions.md L "Mass budget"). A SOFT limit (Spark
              24 Oct: as low as we can, never sacrifice design for weight): never FAILs, never changes the exit code;
              the System line reads "mass <g> of <budget> g (<pct> %)", plus "over budget (...)" and a WARN line.
  shell       geodesic cage (CLAUDE.md "Cages are geodesic shells", Spark 26 Oct): a drone manifest ("machine" scout / overseer)
              with any role "cage" part must carry "cage_shell": {"nodes": [[x,y,z]], "rods": [[i,j]], "open": [{"axis",
              "half_angle_deg"} or {"ring": [node ids round the hole]}], "ratio_min" (optional), "rod_r_mm", "node_r_mm"}. PASS if the rod triangles close every
              direction from the node centroid except the declared open cones (the bottom opening for the pack and the
              swap), and, when ratio_min is given, the node set's height / width >= ratio_min. Reports h/w, nodes,
              rods, triangles, longest rod, largest cell (inscribed circle between rod surfaces). No cage_shell on a
              non-drone manifest: n/a.
Cage first, shell, probes, access, fit and balance gate (exit 1); everything else is reported. Tags: C manifest
geometry / mass, M mesh-derived, E when any input part is tagged E, P measured.
Limits: flat walls see each role's convex hull (a protected part in a concave pocket of the cage is seen only by the
probes). Probes are infinite lines and cylinders: a short stub or a corner poking through a cage window is not modelled,
a closed cage window stops every probe; resolution res_mm (contact +-0.7 res, a cage gap narrower than 1.5 res is not
seen by the edge probe); 24 approach angles per axis. Only role "cage" shields: a frame or plate that is not cage does
not count. Access paths are straight lines (no tilt or two-step moves); a "part" sweep is sampled. Clearance is sampled.

python3 system_check.py manifest.json [--out runs/<sys>/<ver>/system_check.json] [--report] [--dirs 2000] [--no-probes]
from system_check import run; res = run(manifest_dict_or_path[, probes=False | {settings}]); res["summary"], res["pass"]

Manifest: {"name", "parts": [{"name", "mesh": path (STL/OBJ/PLY, relative to the manifest, cwd or repo root)
  | "box": [x0,x1,y0,y1,z0,z1] | "cylinder": {"r", "p0", "p1"} or {"r", "h", "center", "axis"} | "sphere": {"r", "center"},
  "pose": 4x4 | {"t": [x,y,z], "rpy_deg": [r,p,y]} (R = Rz(y) Ry(p) Rx(r)), "role": cage|protected|structure|bought,
  "protect": true (any role), "material", "mass_g", "budget_g", "tag": M|C|E|P, "cg": [x,y,z] in part frame (optional),
  "swap": true (optional)}],
  "openings": [{"part" (omit = every protected part), "axis", "half_angle_deg", "min_mm": 0.0, "skip_probes": false} or
  {"part", "axis", "half_angle_deg", "field": true} (a sensor's field outside the cage, Spark 26 Oct: no cage-first
  requirement inside the cone, skip_probes there; every other direction gated as usual)],
  "contact_pairs": [[a, b, allowed_mm3]], "clearance_pairs": [[a, b, min_mm]], "guard_offsets": {name: mm},
  "cg_ref": [x,y,z], "thrust_axis": [0,0,1], "cg_tol_mm": 3.0, "balance_no_swap_gate": true, "cage_margin_mm": 3.0, "fit_tol_mm3": 0.5, "n_dirs": 2000,
  "probes": true | false | {"edge": true, "post": true, "post_r_mm": 10 or [radii], "n_dirs": 500, "n_roll": 24, "res_mm": 1.0},
  "probe_margin_mm": 0.0, "access": [paths as above], "access_tol_mm3": 0.5, "mass_budget_g": g, "machine", "cage_shell": {...}}.
From Python a part may carry "trimesh": <Trimesh>."""
import sys, os, json, argparse, numpy as np, trimesh

DENSITY = {"petg": 1.27, "tpu": 1.21, "carbon": 1.6, "carbon rod": 1.6, "cf": 1.6}   # g/cm^3
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG_RANK = {"P": 0, "C": 1, "M": 2, "E": 3}                                          # combined tag = worst input


def tag_of(tags):
    tags = [t for t in tags if t]
    return max(tags, key=lambda t: TAG_RANK.get(t, 3)) if tags else "C"


def fib_dirs(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n; r = np.sqrt(1 - z * z); a = np.pi * (3 - np.sqrt(5)) * i
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


def pose_matrix(p):
    if p is None: return np.eye(4)
    if isinstance(p, dict):
        r, pi, y = np.radians(p.get("rpy_deg", [0, 0, 0]))
        cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(pi), np.sin(pi), np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        T = np.eye(4); T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = p.get("t", [0, 0, 0]); return T
    T = np.asarray(p, float); assert T.shape == (4, 4), "pose must be 4x4 or {t, rpy_deg}"; return T


def _resolve(path, base):
    for d in (base, os.getcwd(), REPO):
        q = os.path.join(d, path)
        if os.path.exists(q): return q
    raise FileNotFoundError(path)


def load_part(p, base="."):
    """part dict -> (trimesh in assembled pose, geometry tag)."""
    if "trimesh" in p:                                                # Python API: a trimesh already built
        m = p["trimesh"].copy(); src = p.get("geom_tag", "M")
    elif "mesh" in p:
        m = trimesh.load(_resolve(p["mesh"], base), force="mesh"); src = "M"
    elif "box" in p:
        x0, x1, y0, y1, z0, z1 = p["box"]
        m = trimesh.creation.box([x1 - x0, y1 - y0, z1 - z0]); m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]); src = "C"
    elif "cylinder" in p:
        c = p["cylinder"]
        if "p0" in c:
            m = trimesh.creation.cylinder(radius=c["r"], segment=[c["p0"], c["p1"]], sections=64)
        else:
            ax = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}.get(c.get("axis", "z"), c.get("axis"))
            ax = np.asarray(ax, float); ax /= np.linalg.norm(ax); ctr = np.asarray(c.get("center", [0, 0, 0]), float)
            m = trimesh.creation.cylinder(radius=c["r"], segment=[ctr - ax * c["h"] / 2, ctr + ax * c["h"] / 2], sections=64)
        src = "C"
    elif "sphere" in p:
        s = p["sphere"]; m = trimesh.creation.icosphere(subdivisions=4, radius=s["r"]); m.apply_translation(s.get("center", [0, 0, 0])); src = "C"
    else:
        raise ValueError(f"part {p.get('name')}: needs mesh, box, cylinder or sphere")
    m.apply_transform(pose_matrix(p.get("pose")))
    return m, ("E" if p.get("tag") == "E" else src)


def hull_pts(m):
    try:
        from scipy.spatial import ConvexHull
        V = m.vertices; return V[ConvexHull(V).vertices]
    except Exception:
        return m.vertices


def dir_label(d):
    k = int(np.argmax(np.abs(d)))
    if abs(d[k]) > np.cos(np.radians(10)): return ("+" if d[k] > 0 else "-") + "XYZ"[k]
    return "(" + ", ".join(f"{x:+.2f}" for x in d) + ")"


# ---------- cage first
def cage_first(parts, meshes, openings, margin_mm, n_dirs):
    D = fib_dirs(n_dirs)
    D = np.vstack([D, np.eye(3), -np.eye(3)])                         # the six axes always sampled
    cage = [i for i, p in enumerate(parts) if p.get("role") == "cage"]
    prot = [i for i, p in enumerate(parts) if p.get("role") == "protected" or p.get("protect")]
    if not prot: return dict(applies=False, pass_=True)
    if not cage: return dict(applies=True, pass_=False, error="protected parts but no cage part")
    hc = (np.vstack([hull_pts(meshes[i]) for i in cage]) @ D.T).max(0)
    worst = None; per = {}
    for i in prot:
        nm = parts[i]["name"]; hp = (hull_pts(meshes[i]) @ D.T).max(0); mg = hc - hp
        req = np.full(len(D), float(margin_mm))
        for o in openings:
            if o.get("part") not in (None, nm): continue
            ax = np.asarray(o["axis"], float); ax /= np.linalg.norm(ax)
            inc = D @ ax >= np.cos(np.radians(o["half_angle_deg"]))     # overlapping openings: the loosest requirement holds
            req[inc] = np.minimum(req[inc], -np.inf if o.get("field") else min(float(o.get("min_mm", 0.0)), margin_mm))
        sl = mg - req; k = int(np.argmin(sl)); free = req == margin_mm
        per[nm] = dict(min_margin_mm=round(float(mg.min()), 2), worst_dir=[round(float(x), 3) for x in D[k]],
                       worst_slack_mm=round(float(sl[k]), 2), min_margin_outside_openings_mm=round(float(mg[free].min()), 2) if free.any() else None,
                       dirs_in_openings=int((~free).sum()), pass_=bool(sl.min() >= -1e-9))
        if worst is None or sl[k] < worst[0]: worst = (float(sl[k]), nm, D[k], float(mg[k]), float(req[k]))
    s, nm, d, mg, rq = worst
    return dict(applies=True, pass_=bool(s >= -1e-9), worst_part=nm, worst_dir=[round(float(x), 3) for x in d], worst_dir_label=dir_label(d),
                margin_mm=round(mg, 2), required_mm=rq, n_dirs=len(D), cage_margin_mm=margin_mm, per_part=per)


# ---------- compact
def compact(meshes, guard):
    V = np.vstack([m.vertices for m in meshes]); lo, hi = V.min(0), V.max(0); ext = hi - lo
    try: hv = trimesh.PointCloud(V).convex_hull.volume
    except Exception: hv = float("nan")
    return dict(bbox_min=lo.round(2).tolist(), bbox_max=hi.round(2).tolist(), extents_mm=ext.round(1).tolist(),
                largest_extent_mm=round(float(ext.max()), 1), bbox_L=round(float(np.prod(ext)) / 1e6, 3),
                hull_L=round(float(hv) / 1e6, 3), guard_offsets_mm=guard or {})


# ---------- fit
def _manifold(m):
    import manifold3d as M3
    mm = M3.Manifold(M3.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    if mm.status() != M3.Error.NoError: raise ValueError(str(mm.status()))
    return mm


def _voxel_overlap(a, b, lo, hi, pitch=0.5):
    """fallback: inside-both count on a grid over the AABB overlap (needs closed meshes)."""
    n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    if np.prod(n) > 4e6: pitch = float((np.prod(hi - lo) / 4e6) ** (1 / 3)); n = np.maximum(np.ceil((hi - lo) / pitch).astype(int), 1)
    g = [lo[k] + (np.arange(n[k]) + 0.5) * (hi[k] - lo[k]) / n[k] for k in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    ins = a.contains(P); ins[ins] = b.contains(P[ins])
    return float(ins.sum() * np.prod((hi - lo) / n))


def interference(a, b, cache, ia, ib):
    lo = np.maximum(a.bounds[0], b.bounds[0]); hi = np.minimum(a.bounds[1], b.bounds[1])
    if np.any(hi <= lo): return 0.0, "aabb"
    try:
        for i, m in ((ia, a), (ib, b)):
            if i not in cache: cache[i] = _manifold(m)
        return float((cache[ia] ^ cache[ib]).volume()), "manifold3d"
    except Exception:
        pass
    try:
        return float(trimesh.boolean.intersection([a, b]).volume), "trimesh"
    except Exception:
        return _voxel_overlap(a, b, lo, hi), "voxel"


def clearance(a, b, n=3000):
    sub = lambda m: m.vertices[np.random.default_rng(0).choice(len(m.vertices), min(len(m.vertices), n), replace=False)]
    pa = np.vstack([sub(a), a.sample(n)]); pb = np.vstack([sub(b), b.sample(n)])
    da = trimesh.proximity.ProximityQuery(b).signed_distance(pa)        # + inside b
    db = trimesh.proximity.ProximityQuery(a).signed_distance(pb)
    if (da > 0).any() or (db > 0).any(): return 0.0
    return float(min((-da).min(), (-db).min()))


def fit(parts, meshes, contact, tol, clear_pairs):
    idx = {p["name"]: i for i, p in enumerate(parts)}
    allow = {frozenset(c[:2]): float(c[2]) if len(c) > 2 else float("inf") for c in contact}
    pairs = []; cache = {}
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            v, how = interference(meshes[i], meshes[j], cache, i, j)
            key = frozenset((parts[i]["name"], parts[j]["name"])); lim = allow.get(key, tol)
            if v > 1e-6 or key in allow:
                pairs.append(dict(a=parts[i]["name"], b=parts[j]["name"], overlap_mm3=round(v, 2), limit_mm3=lim,
                                  contact=key in allow, method=how, pass_=bool(v <= lim)))
    bad = [p for p in pairs if not p["pass_"]]
    nonex = [p for p in pairs if not p["contact"]]
    w = max(bad or nonex or [None], key=lambda p: -1 if p is None else p["overlap_mm3"])
    cl = []
    for c in clear_pairs:
        d = clearance(meshes[idx[c[0]]], meshes[idx[c[1]]])
        cl.append(dict(a=c[0], b=c[1], clearance_mm=round(d, 2), min_mm=c[2] if len(c) > 2 else None,
                       ok=None if len(c) < 3 else bool(d >= c[2])))
    return dict(pass_=not bad, tol_mm3=tol, worst=w, pairs=pairs, clearance=cl)


# ---------- mass / CG
def mass_cg(parts, meshes):
    rows = []; M = 0.0; S = np.zeros(3)
    for p, m in zip(parts, meshes):
        if "mass_g" in p:
            g = float(p["mass_g"]); t = p.get("tag", "C")
        else:
            if p.get("role") == "bought": raise ValueError(f"bought part {p['name']}: give mass_g")
            mat = str(p.get("material", "PETG")).lower()
            if mat not in DENSITY: raise ValueError(f"part {p['name']}: unknown material {mat} (PETG, TPU, carbon)")
            g = abs(m.volume) / 1000 * DENSITY[mat]; t = "E" if p.get("tag") == "E" else ("M" if "mesh" in p or "trimesh" in p else "C")
        if "cg" in p:
            c = (pose_matrix(p.get("pose")) @ np.r_[np.asarray(p["cg"], float), 1])[:3]
        else:
            c = m.center_mass if m.is_watertight and abs(m.volume) > 1e-9 else m.bounds.mean(0)
        rows.append(dict(name=p["name"], mass_g=round(g, 2), cg=np.round(c, 2).tolist(), tag=t, swap=bool(p.get("swap")), _g=g, _c=c))
        M += g; S += g * c
    cg = S / M if M > 0 else np.zeros(3)
    return dict(mass_g=round(M, 2), cg=np.round(cg, 2).tolist(), parts=rows, tag=tag_of([r["tag"] for r in rows]))


def balance(rows, ref, axis, tol, gate_no_swap=True):
    """CG offset from ref, split into the part along the thrust axis and the horizontal part normal to it; with and
    without the swap parts. PASS if every horizontal offset <= tol (the no-swap case only reported when gate_no_swap is
    false: a manifest's "balance_no_swap_gate": false, set only with a dimensions.md L Balance exception)."""
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax); ref = np.asarray(ref, float); out = {}
    cases = [("all", rows)] + ([("no_swap", [r for r in rows if not r["swap"]])] if any(r["swap"] for r in rows) else [])
    for k, rs in cases:
        g = sum(r["_g"] for r in rs)
        if g <= 0: continue
        off = sum(r["_g"] * r["_c"] for r in rs) / g - ref; h = off - (off @ ax) * ax
        out[k] = dict(mass_g=round(g, 2), cg_offset=np.round(off, 2).tolist(), horizontal_mm=round(float(np.linalg.norm(h)), 2),
                      axial_mm=round(float(off @ ax), 2), pass_=bool(np.linalg.norm(h) <= tol + 1e-9 or (k == "no_swap" and not gate_no_swap)),
                      gated=bool(k != "no_swap" or gate_no_swap))
    w = max(out, key=lambda k: out[k]["horizontal_mm"])
    return dict(applies=True, ref=ref.tolist(), thrust_axis=ax.round(4).tolist(), tol_mm=tol, cases=out, worst_case=w,
                worst_horizontal_mm=out[w]["horizontal_mm"], pass_=all(c["pass_"] for c in out.values()))


# ---------- access (rule 2: reachable without taking another part off)
def _unit(v):
    v = np.asarray(v, float); n = np.linalg.norm(v); assert n > 0, "zero direction"; return v / n


def _perp(d, hint=None):
    """unit vector normal to d: the hint made normal to d, else world z made normal to d (x if d is along z)."""
    for h in ([hint] if hint is not None else []) + [[0, 0, 1], [1, 0, 0]]:
        h = np.asarray(h, float); u = h - (h @ d) * d
        if np.linalg.norm(u) > 1e-6: return u / np.linalg.norm(u)


def _vol_and(a, b, ma=None, mb=None):
    """intersection volume of two meshes (manifold3d, trimesh, voxel fallback) -> (mm^3, method)."""
    lo = np.maximum(a.bounds[0], b.bounds[0]); hi = np.minimum(a.bounds[1], b.bounds[1])
    if np.any(hi <= lo): return 0.0, "aabb"
    try:
        return float(((ma or _manifold(a)) ^ (mb or _manifold(b))).volume()), "manifold3d"
    except Exception:
        pass
    try:
        return float(trimesh.boolean.intersection([a, b]).volume), "trimesh"
    except Exception:
        return _voxel_overlap(a, b, lo, hi), "voxel"


def sweep_tool(path, part_mesh, part_pose):
    """swept solid of an access path -> (trimesh, manifold or None, n_steps, dir).
    cylinder {d}: a cylinder of diameter d from `from` along dir for length_mm; box {w, h(, w_axis)}: a w x h section
    (w along w_axis made normal to dir, default world z made normal to dir) extruded the same way; "part": the part's
    own mesh moved 0..length_mm along dir, sampled in n steps (>= 20, step <= half the part's depth along dir, <= 200)
    and unioned (manifold3d)."""
    d = _unit(path["dir"]); L = float(path["length_mm"]); tool = path.get("tool", "part")
    f = path.get("from")
    if path.get("frame") == "part":                                   # from / dir given in the part frame
        d = _unit(part_pose[:3, :3] @ d)
        if f is not None: f = (part_pose @ np.r_[np.asarray(f, float), 1])[:3]
    f = None if f is None else np.asarray(f, float)
    if tool == "part" or (isinstance(tool, dict) and tool.get("type") == "part"):
        dep = float(np.ptp(part_mesh.vertices @ d)) or 1.0
        n = int(min(200, max(20, np.ceil(L / (0.5 * dep)))))
        ts = np.linspace(0, L, n + 1)
        try:
            import manifold3d as M3
            base = _manifold(part_mesh)
            u = M3.Manifold.batch_boolean([base.translate(tuple(float(x) for x in t * d)) for t in ts], M3.OpType.Add)
            o = u.to_mesh(); m = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=False)
            return m, u, n, d
        except Exception:
            return trimesh.util.concatenate([part_mesh.copy().apply_translation(t * d) for t in ts]), None, n, d
    assert f is not None, f"access {path.get('name')}: a cylinder or box tool needs `from`"
    if tool.get("type") == "cylinder":
        m = trimesh.creation.cylinder(radius=float(tool["d"]) / 2, segment=[f, f + d * L], sections=48)
    elif tool.get("type") == "box":
        u = _perp(d, tool.get("w_axis")); v = np.cross(d, u)
        m = trimesh.creation.box([float(tool["w"]), float(tool["h"]), L])
        T = np.eye(4); T[:3, :3] = np.stack([u, v, d], 1); T[:3, 3] = f + d * L / 2; m.apply_transform(T)
    else:
        raise ValueError(f"access {path.get('name')}: tool must be cylinder, box or part")
    try: mm = _manifold(m)
    except Exception: mm = None
    return m, mm, 1, d


def access(parts, meshes, paths, tol):
    """every path's swept tool against every other part. A "part" sweep counts only what its moves add (the sweep's
    overlap minus the part's own overlap at rest, which is the fit check's)."""
    if not paths: return dict(applies=False, pass_=True, paths=[])
    idx = {p["name"]: i for i, p in enumerate(parts)}; cache = {}; out = []
    def man(i):
        if i not in cache:
            try: cache[i] = _manifold(meshes[i])
            except Exception: cache[i] = None
        return cache[i]
    for k, pth in enumerate(paths):
        nm = pth.get("name", f"path{k}"); pi = idx[pth["part"]]; tool = pth.get("tool", "part")
        sw, swm, n, d = sweep_tool(pth, meshes[pi], pose_matrix(parts[pi].get("pose")))
        is_part = not isinstance(tool, dict) or tool.get("type") == "part"
        ign = set(pth.get("ignore", [])) | {pth["part"]}
        bl = []
        for j, p in enumerate(parts):
            if p["name"] in ign: continue
            v, how = _vol_and(sw, meshes[j], swm, man(j))
            if is_part and v > 1e-6:
                v0, _ = _vol_and(meshes[pi], meshes[j], man(pi), man(j)); v = max(0.0, v - v0)
            if v > 1e-6: bl.append(dict(part=p["name"], overlap_mm3=round(v, 2), method=how))
        bl.sort(key=lambda b: -b["overlap_mm3"])
        worst = bl[0]["overlap_mm3"] if bl else 0.0
        out.append(dict(name=nm, part=pth["part"], tool=tool if isinstance(tool, str) else dict(tool), dir=np.round(d, 4).tolist(),
                        length_mm=float(pth["length_mm"]), steps=n, blocked_by=[b for b in bl if b["overlap_mm3"] > tol],
                        touches=bl, overlap_mm3=worst, pass_=bool(worst <= tol)))
    bad = [p for p in out if not p["pass_"]]
    return dict(applies=True, pass_=not bad, tol_mm3=tol, paths=out, n_clear=len(out) - len(bad),
                worst=max(out, key=lambda p: p["overlap_mm3"]))


# ---------- crash probes (rule 3 against edges and posts, not only flat walls)
def surface_points(m, spacing):
    """vertices plus points every <= spacing along every edge (a line's first contact with a mesh lies on an edge),
    thinned to one per spacing/2 voxel."""
    V = np.asarray(m.vertices, float); E = m.edges_unique; a, b = V[E[:, 0]], V[E[:, 1]]
    n = np.ceil(np.linalg.norm(b - a, axis=1) / spacing).astype(int); lg = np.nonzero(n > 1)[0]
    reps = n[lg] - 1; ix = np.repeat(lg, reps)
    t = (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps) + 1) / np.repeat(n[lg], reps)
    P = np.vstack([V, a[ix] + (b[ix] - a[ix]) * t[:, None]])
    _, u = np.unique(np.floor(P / (0.5 * spacing)).astype(np.int64), axis=0, return_index=True)
    return P[u]


def _hemi(n):
    D = fib_dirs(2 * n); return D[D[:, 2] >= 0][:n]


def crash_probes(parts, meshes, cfg, req_mm, openings):
    """Edge probe: an infinite line (table edge, door frame); post probe: an infinite cylinder of radius r (pole,
    branch). Each probe axis e (n_dirs Fibonacci axes over the hemisphere) meets the drone from n_roll approach
    directions d normal to e, at every lateral offset. Seen along e the line is a point and the post a disc, so the
    contact is solved in the plane normal to e: the cage's and each protected part's surface points (edge samples)
    are rasterised at res_mm, the outline pixels kept, and for each d the furthest point along d per lateral bin
    (1.5 res wide) is the line's first contact; the post's is that profile dilated by a semicircle of radius r.
    margin = cage contact - part contact (mm along d). FAIL if a protected part is met before the cage, ahead by less
    than probe_margin_mm, or with no cage member in that probe's path at all ("unshielded": the margin is then minus
    the probe's reach past the cage envelope). An opening with "skip_probes": true exempts its cone for its part.
    A probe stops at the first solid: protected parts shadow each other, so in each (axis, approach, bin) only the
    protected part(s) met first count (ties count for all); a part behind another (exempt ones included) is +inf there
    (no row if it is shadowed or exempt in every probe)."""
    cage = [i for i, p in enumerate(parts) if p.get("role") == "cage"]
    prot = [i for i, p in enumerate(parts) if p.get("role") == "protected" or p.get("protect")]
    if not prot or not cage: return dict(applies=False, pass_=True)
    cfg = {} if cfg is True else dict(cfg)
    res = float(cfg.get("res_mm", 1.0)); n_ax = int(cfg.get("n_dirs", 500)); n_th = int(cfg.get("n_roll", 24))
    radii = ([0.0] if cfg.get("edge", True) else []) + ([float(r) for r in np.atleast_1d(cfg.get("post_r_mm", 10.0))] if cfg.get("post", True) else [])
    pts = [np.vstack([surface_points(meshes[i], res) for i in cage])] + [surface_points(meshes[i], res) for i in prot]
    allp = np.vstack(pts); c0 = (allp.min(0) + allp.max(0)) / 2; pts = [p - c0 for p in pts]
    R = float(np.linalg.norm(allp - c0, axis=1).max()) + 2 * res
    bw = 1.5 * res; nb = int(np.ceil(2 * R / bw)) + 2; nro = len(pts)
    th = np.arange(n_th) * 2 * np.pi / n_th; Dm = np.stack([np.cos(th), np.sin(th)]); Lm = np.stack([-np.sin(th), np.cos(th)])
    cut = np.cumsum([0] + [len(p) for p in pts]); P = np.vstack(pts)
    axes = _hemi(n_ax); names = [parts[i]["name"] for i in prot]
    skip = [[o for o in openings if o.get("skip_probes") and o.get("part") in (None, nm)] for nm in names]
    K = {r: np.arange(-int(np.ceil(r / bw)), int(np.ceil(r / bw)) + 1) for r in radii}
    worst = {(r, j): (np.inf, None) for r in radii for j in range(len(prot))}
    for e in axes:
        a1 = _perp(e); a2 = np.cross(e, a1)
        ij = np.floor((P @ np.stack([a1, a2], 1) + R) / res).astype(np.int64)
        B = []; Br = []
        for k in range(nro):                                                         # outline pixels of each role
            s = ij[cut[k]:cut[k + 1]]; lo = s.min(0) - 1; s = s - lo; n0, n1 = s.max(0) + 2
            g = np.zeros((n0, n1), bool); g[s[:, 0], s[:, 1]] = True
            inn = np.zeros_like(g); inn[1:-1, 1:-1] = g[1:-1, 1:-1] & g[:-2, 1:-1] & g[2:, 1:-1] & g[1:-1, :-2] & g[1:-1, 2:]
            q = np.argwhere(g & ~inn) + lo; B.append((q + 0.5) * res - R); Br.append(np.full(len(q), k))
        nc = len(B[0]); B = np.vstack(B); Br = np.concatenate(Br)
        S = B @ Dm; W = np.floor((B @ Lm + R) / bw).astype(np.int64)
        key = (Br[:, None] * n_th + np.arange(n_th)[None, :]) * nb + W
        prof = np.full(nro * n_th * nb, -np.inf); np.maximum.at(prof, key.ravel(), S.ravel()); prof = prof.reshape(nro, n_th, nb)
        dv = np.cos(th)[:, None] * a1 + np.sin(th)[:, None] * a2                    # approach dirs in 3D
        hull = S[:nc].max(0)                                                         # cage support along each d
        for r in radii:
            if r > 0:
                pr = np.full_like(prof, -np.inf)
                for kk in K[r]:
                    add = np.sqrt(max(r * r - (kk * bw) ** 2, 0.0))
                    dst = pr[:, :, max(0, -kk):nb - max(0, kk)]; np.maximum(dst, prof[:, :, max(0, kk):nb + min(0, kk)] + add, out=dst)
            else:
                pr = prof
            hc = pr[0]; shield = np.isfinite(hc); front = pr[1:].max(0)              # first protected solid per bin
            for j in range(len(prot)):
                hp = pr[j + 1]; has = np.isfinite(hp) & (hp >= front)              # met first (ties: both); else shadowed
                if not has.any(): continue
                with np.errstate(invalid="ignore"):
                    mg = np.where(shield, hc - hp, -np.abs(hull[:, None] + r - hp))
                mg = np.where(has, mg, np.inf)
                for o in skip[j]:
                    mg[dv @ _unit(o["axis"]) >= np.cos(np.radians(o["half_angle_deg"]))] = np.inf
                sc = np.where(np.isfinite(mg) & ~shield, -1e6 - mg, mg)             # unshielded first, least reach (most exposed) first
                t = np.unravel_index(int(np.argmin(sc)), sc.shape)
                if sc[t] < worst[(r, j)][0]:
                    worst[(r, j)] = (float(sc[t]), dict(axis=e, dir=dv[t[0]], roll=float(np.degrees(th[t[0]])), w=(t[1] + 0.5) * bw - R,
                                                         margin=float(mg[t]), unshielded=bool(not shield[t])))
    rows = []
    for (r, j), (sc, w) in worst.items():
        if w is None: continue
        d = w["dir"]; Pp = pts[j + 1]; u = np.cross(w["axis"], d); u /= np.linalg.norm(u)
        lw = Pp @ u; sel = np.abs(lw - w["w"]) <= r + bw                            # contact point on the part
        reach = Pp @ d + (np.sqrt(np.maximum(r * r - (lw - w["w"]) ** 2, 0)) if r > 0 else 0)
        cp = Pp[np.argmax(np.where(sel, reach, -np.inf))] + c0 if sel.any() else None
        ok = (not w["unshielded"]) and w["margin"] >= req_mm - 1e-9
        rows.append(dict(probe="edge" if r == 0 else f"post r{r:g}", r_mm=r, part=names[j], margin_mm=round(w["margin"], 2),
                         unshielded=w["unshielded"], dir=np.round(d, 3).tolist(), dir_label=dir_label(d), axis=np.round(w["axis"], 3).tolist(),
                         roll_deg=round(w["roll"], 1), lateral_mm=round(float(w["w"]), 1),
                         contact=None if cp is None else np.round(cp, 1).tolist(), pass_=bool(ok)))
    rows.sort(key=lambda x: (x["pass_"], not x["unshielded"], -x["margin_mm"] if x["unshielded"] else x["margin_mm"]))
    return dict(applies=True, pass_=all(x["pass_"] for x in rows), probe_margin_mm=req_mm, worst=rows[0] if rows else None,
                per_probe_part=rows, n_axes=len(axes), n_roll=n_th, orientations=len(axes) * n_th, res_mm=res, bin_mm=bw,
                radii_mm=radii, n_points=int(len(P)))


# ---------- mass budget
def mass_budget(parts, rows, total_budget):
    per = {p["name"]: float(p["budget_g"]) for p in parts if p.get("budget_g") is not None}
    if total_budget is None and not per: return dict(applies=False, pass_=True)
    over = [dict(name=r["name"], mass_g=r["mass_g"], budget_g=per[r["name"]]) for r in rows
            if r["name"] in per and r["mass_g"] > per[r["name"]] + 1e-9]
    tot = sum(r["mass_g"] for r in rows); out = dict(applies=True, mass_g=round(tot, 2), part_budgets_g=per, over_parts=over)
    if total_budget is not None:
        out.update(budget_g=float(total_budget), pct=round(100 * tot / float(total_budget), 1), total_ok=bool(tot <= float(total_budget) + 1e-9))
    out["over"] = bool(not out.get("total_ok", True) or over)
    out["pass_"] = True                                                   # soft limit: reported, never gates
    return out


# ---------- geodesic shell
def cage_shell(manifest, parts, n=2000):
    spec = manifest.get("cage_shell")
    drone = manifest.get("machine") in ("scout", "overseer") and any(p.get("role") == "cage" for p in parts)
    if spec is None:
        if drone:
            return dict(applies=True, pass_=False, error="no cage_shell: a drone cage is a geodesic shell (CLAUDE.md, Spark 26 Oct)")
        return dict(applies=False, pass_=True)
    N = np.asarray(spec["nodes"], float); E = [tuple(sorted(map(int, r))) for r in spec["rods"]]
    adj = {i: set() for i in range(len(N))}
    for a, b in E:
        adj[a].add(b); adj[b].add(a)
    tris = sorted({tuple(sorted((a, b, c))) for a, b in E for c in adj[a] & adj[b]})
    c0 = N.mean(0); D = fib_dirs(n)
    skip = np.zeros(n, bool)
    for o in spec.get("open", []):
        if "ring" in o:                                                 # a polygon of nodes: rays through it are the opening
            P = N[list(o["ring"])]; pc = P.mean(0); nrm = np.linalg.svd(P - pc)[2][2]
            den = D @ nrm; t = ((pc - c0) @ nrm) / np.where(np.abs(den) > 1e-12, den, np.nan)
            X = c0 + t[:, None] * D; e1 = P[0] - pc; e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
            ang = np.arctan2((P - pc) @ e2, (P - pc) @ e1); Q = 1.02 * (P - pc)[np.argsort(ang)] @ np.stack([e1, e2], 1)   # 2 % for edge rays
            xy = (X - pc) @ np.stack([e1, e2], 1); inside = np.zeros(n, bool)
            for i in range(len(Q)):                                     # even-odd point in polygon
                a_, b_ = Q[i], Q[(i + 1) % len(Q)]
                cr = ((a_[1] > xy[:, 1]) != (b_[1] > xy[:, 1])) & (xy[:, 0] < (b_[0] - a_[0]) * (xy[:, 1] - a_[1]) / (b_[1] - a_[1] + 1e-12) + a_[0])
                inside ^= cr
            skip |= inside & (t > 0)
            continue
        ax = np.asarray(o["axis"], float); ax /= np.linalg.norm(ax)
        skip |= D @ ax >= np.cos(np.radians(o["half_angle_deg"]))
    hit = np.zeros(n, bool)
    for a, b, c in tris:                                            # Moller-Trumbore, rays from the centroid
        e1, e2 = N[b] - N[a], N[c] - N[a]; pv = np.cross(D, e2); det = pv @ e1
        ok = np.abs(det) > 1e-12; inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        tv = c0 - N[a]; u = (pv @ tv) * inv; qv = np.cross(tv, e1); v = (D @ qv) * inv; t = (qv @ e2) * inv
        hit |= ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
    gaps = ~hit & ~skip
    ex = N.max(0) - N.min(0); ratio = float(ex[2] / max(ex[0], ex[1]))
    rmin = spec.get("ratio_min"); rr = float(spec.get("rod_r_mm", 1.5))
    L = [float(np.linalg.norm(N[a] - N[b])) for a, b in E]
    cells = []
    for a, b, c in tris:
        la, lb, lc = np.linalg.norm(N[b] - N[c]), np.linalg.norm(N[a] - N[c]), np.linalg.norm(N[a] - N[b]); sp = (la + lb + lc) / 2
        cells.append(2 * np.sqrt(max(sp * (sp - la) * (sp - lb) * (sp - lc), 0)) / sp - 2 * rr)
    closed = not gaps.any()
    mirror = None; mok = True
    pl = spec.get("mirror", "y" if drone else None)                 # every node and rod has its mirror (Spark 27 Oct: perfect left-right symmetry)
    if pl:
        k = "xyz".index(pl); M = N.copy(); M[:, k] *= -1
        dd = np.linalg.norm(M[:, None] - N[None], axis=2); j = dd.argmin(1)
        Es = set(E); bad = sum(tuple(sorted((int(j[a]), int(j[b])))) not in Es for a, b in E)
        mirror = dict(plane=pl, node_err_mm=round(float(dd.min(1).max()), 3), rods_unmatched=int(bad))
        mok = mirror["node_err_mm"] <= 0.5 and bad == 0
    return dict(applies=True, pass_=bool(closed and mok and (rmin is None or ratio >= float(rmin))), closed=bool(closed), mirror=mirror, gap_share=round(float(gaps.mean()), 4),
                open_share=round(float(skip.mean()), 4), ratio=round(ratio, 3), ratio_min=rmin, nodes=len(N), rods=len(E), triangles=len(tris),
                longest_rod_mm=round(max(L), 1), largest_cell_mm=round(max(cells), 1) if cells else None)


# ---------- run
def run(manifest, n_dirs=None, probes=None):
    """probes: None = the manifest's `probes` (default on with a cage), False = skip, dict = override settings."""
    base = "."
    if isinstance(manifest, str):
        base = os.path.dirname(os.path.abspath(manifest)); manifest = json.load(open(manifest))
    parts = manifest["parts"]
    names = [p["name"] for p in parts]; assert len(set(names)) == len(names), "part names must be unique"
    loaded = [load_part(p, base) for p in parts]; meshes = [m for m, _ in loaded]; gtags = [t for _, t in loaded]
    anyE = any(p.get("tag") == "E" for p in parts)
    gtag = "E" if anyE else tag_of(gtags)
    n = n_dirs or manifest.get("n_dirs", 2000); cm = float(manifest.get("cage_margin_mm", 3.0))
    # an opening with "field": true is a sensor's field cone outside the cage (Spark 26 Oct: the L2 and the front camera with
    # nothing in their views, guarded only outside them): no cage-first requirement and no probes inside it, gated elsewhere
    assert not any(o.get("exempt") for o in manifest.get("openings", [])), "exempt openings are retired: declare the sensor's field cone (field: true)"
    # a probe can always reach a sensor's face that stands outside the cage, so the probes skip a field part (every direction);
    # its protrusion past the cage envelope is reported instead
    fo = [o for o in manifest.get("openings", []) if o.get("field")]
    manifest = dict(manifest, openings=[o for o in manifest.get("openings", [])]
                    + [dict(part=o["part"], axis=[1, 0, 0], half_angle_deg=180.0, min_mm=cm, skip_probes=True) for o in fo])
    cf = cage_first(parts, meshes, manifest.get("openings", []), cm, n)
    cp = compact(meshes, manifest.get("guard_offsets"))
    ft = fit(parts, meshes, manifest.get("contact_pairs", []), float(manifest.get("fit_tol_mm3", 0.5)), manifest.get("clearance_pairs", []))
    pc = manifest.get("probes", True) if probes is None else probes
    pr = (crash_probes(parts, meshes, pc, float(manifest.get("probe_margin_mm", 0.0)), manifest.get("openings", []))
          if pc else dict(applies=False, pass_=True, skipped=True))
    ac = access(parts, meshes, manifest.get("access", []), float(manifest.get("access_tol_mm3", 0.5)))
    sh = cage_shell(manifest, parts)
    mc = mass_cg(parts, meshes)
    if anyE: mc["tag"] = "E"
    mb = mass_budget(parts, mc["parts"], manifest.get("mass_budget_g"))
    ref = manifest.get("cg_ref")
    if ref is not None:
        off = np.asarray(mc["cg"]) - np.asarray(ref, float)
        mc["cg_ref"] = list(ref); mc["cg_offset"] = np.round(off, 2).tolist(); mc["cg_offset_mm"] = round(float(np.linalg.norm(off)), 2)
        bal = balance(mc["parts"], ref, manifest.get("thrust_axis", [0, 0, 1]), float(manifest.get("cg_tol_mm", 3.0)),
                      bool(manifest.get("balance_no_swap_gate", True)))
    else:
        bal = dict(applies=False, pass_=True)
    for r in mc["parts"]: r.pop("_g"); r.pop("_c")
    name = manifest.get("name", "system")
    if not cf["applies"]: cfs = "cage first n/a (no protected part)"
    elif "error" in cf: cfs = f"cage first FAIL ({cf['error']})"
    else: cfs = (f"cage first {cf['margin_mm']:.1f} mm ({gtag}) at {cf['worst_dir_label']} ({cf['worst_part']}){'' if cf['pass_'] else ' FAIL'}"
                 + (" (field cones open: " + ", ".join(f"{o['part']} {o['half_angle_deg']:g} deg, out {max(0.0, -cf['per_part'][o['part']]['min_margin_mm']):g} mm"
                                                      for o in fo if o["part"] in cf.get("per_part", {})) + ")" if fo else ""))
    w = ft["worst"]
    fts = f"fit {w['overlap_mm3']:.1f} mm³ ({gtag}) ({w['a']}/{w['b']}){'' if ft['pass_'] else ' FAIL'}" if w else f"fit 0.0 mm³ ({gtag})"
    ex = cp["extents_mm"]
    if bal["applies"]:
        a = bal["cases"]["all"]; ns = bal["cases"].get("no_swap")
        cgs = (f"CG {a['horizontal_mm']:.1f} mm off thrust axis ({mc['tag']}) [{', '.join(f'{x:+.1f}' for x in a['cg_offset'])}]"
               + (f", {ns['horizontal_mm']:.1f} mm without swap parts" + ("" if ns.get("gated", True) else " (reported, L exception)") if ns else "") + ("" if bal["pass_"] else f" FAIL (tol {bal['tol_mm']:g})"))
    else:
        cgs = f"CG [{', '.join(f'{x:.1f}' for x in mc['cg'])}] ({mc['tag']}, no cg_ref)"
    fmt = lambda g: f"{g:.{0 if g >= 100 else 1}f}"
    if mb["applies"] and "budget_g" in mb:
        ms = f"mass {fmt(mc['mass_g'])} of {mb['budget_g']:g} g ({mb['pct']:.0f} %) ({mc['tag']})"
    else:
        ms = f"mass {fmt(mc['mass_g'])} g ({mc['tag']})"
    if mb["applies"] and mb["over"]:
        ov = ", ".join(["total"] * (not mb.get("total_ok", True)) + ["%s %s of %g g" % (o["name"], fmt(o["mass_g"]), o["budget_g"]) for o in mb["over_parts"]])
        ms += f" over budget ({ov})"
    pw = pr.get("worst")
    if not pr["applies"]: prs = ""
    elif pw is None: prs = ", probes n/a"
    else:
        prs = (f", probes {pw['margin_mm']:.1f} mm ({gtag}) {pw['probe']} at {pw['dir_label']} ({pw['part']}"
               + (", unshielded" if pw["unshielded"] else "") + ")" + ("" if pr["pass_"] else " FAIL"))
    if not ac["applies"]: acs = ""
    else:
        bad = [p for p in ac["paths"] if not p["pass_"]]
        acs = f", access {ac['n_clear']}/{len(ac['paths'])} clear" + (
            " (" + "; ".join(f"{p['name']} blocked by {', '.join(b['part'] for b in p['blocked_by'])} {p['overlap_mm3']:.0f} mm³" for p in bad) + ") FAIL" if bad else "")
    if not sh["applies"]: shs = ""
    elif "error" in sh: shs = f", shell FAIL ({sh['error']})"
    else:
        shs = (f", shell {sh['rods']} rods {'closed' if sh['closed'] else 'OPEN ' + format(100 * sh['gap_share'], '.1f') + ' %'}, h/w {sh['ratio']:.2f}"
               + ("" if sh["pass_"] else " FAIL"))
    summary = (f"System: {name}, {cfs}{shs}, {fts}, bbox {cp['bbox_L']:.3g} L ({gtag}), extents {ex[0]:.0f} x {ex[1]:.0f} x {ex[2]:.0f} mm ({gtag}), "
               f"{ms}, {cgs}{prs}{acs}")
    ok = bool(cf["pass_"] and ft["pass_"] and bal["pass_"] and pr["pass_"] and ac["pass_"] and sh["pass_"])      # mass budget is soft
    return dict(name=name, pass_=ok, cage_first=cf, shell=sh, compact=cp, fit=ft, mass_cg=mc, balance=bal, probes=pr, access=ac,
                mass_budget=mb, geometry_tag=gtag, summary=summary, **{"pass": ok})


def report(res):
    cf, cp, ft, mc, t = res["cage_first"], res["compact"], res["fit"], res["mass_cg"], res["geometry_tag"]
    L = [f"system_check {res['name']}: {'PASS' if res['pass'] else 'FAIL'}"]
    if cf["applies"] and "error" not in cf:
        L.append(f"  cage first {'PASS' if cf['pass_'] else 'FAIL'}: worst {cf['margin_mm']} mm ({t}) at {cf['worst_dir_label']} on {cf['worst_part']} "
                 f"(needs {cf['required_mm']}), {cf['n_dirs']} dirs")
        for nm, r in cf["per_part"].items():
            L.append(f"    {nm}: min margin {r['min_margin_mm']} mm, outside openings {r['min_margin_outside_openings_mm']}, slack {r['worst_slack_mm']} "
                     f"at {dir_label(np.array(r['worst_dir']))}{'' if r['pass_'] else ' FAIL'}")
    else:
        L.append("  cage first: " + cf.get("error", "n/a (no protected part)"))
    sh = res.get("shell", {"applies": False})
    if sh["applies"]:
        L.append(f"  shell {'PASS' if sh['pass_'] else 'FAIL'}: " + (sh["error"] if "error" in sh else
                 f"{sh['nodes']} nodes, {sh['rods']} rods, {sh['triangles']} triangles, {'closed' if sh['closed'] else 'open in ' + str(100 * sh['gap_share']) + ' % of directions'}"
                 f" outside the declared openings ({100 * sh['open_share']:.1f} %), h/w {sh['ratio']}" + (f" (min {sh['ratio_min']})" if sh['ratio_min'] is not None else "") + f", longest rod {sh['longest_rod_mm']} mm, "
                 f"largest cell Ø{sh['largest_cell_mm']} mm"
                 + (f", mirror {sh['mirror']['plane']}: nodes within {sh['mirror']['node_err_mm']} mm, {sh['mirror']['rods_unmatched']} rods without a mirror"
                    if sh.get("mirror") else "")))
    L.append(f"  compact: bbox {cp['bbox_L']} L ({t}), hull {cp['hull_L']} L ({t}), extents {cp['extents_mm']} mm ({t}), largest {cp['largest_extent_mm']} mm"
             + (f", guards {cp['guard_offsets_mm']}" if cp["guard_offsets_mm"] else ""))
    L.append(f"  fit {'PASS' if ft['pass_'] else 'FAIL'} (tol {ft['tol_mm3']} mm³):")
    for p in ft["pairs"]:
        L.append(f"    {p['a']} / {p['b']}: {p['overlap_mm3']} mm³ ({t}) limit {p['limit_mm3']}{' contact' if p['contact'] else ''} [{p['method']}]{'' if p['pass_'] else ' FAIL'}")
    for c in ft["clearance"]:
        L.append(f"    clearance {c['a']} / {c['b']}: {c['clearance_mm']} mm ({t})" + ("" if c["ok"] is None else f" (min {c['min_mm']}{'' if c['ok'] else ', LOW'})"))
    L.append(f"  mass {mc['mass_g']} g ({mc['tag']}), CG {mc['cg']}" + (f", offset {mc['cg_offset']} = {mc['cg_offset_mm']} mm from {mc['cg_ref']}" if "cg_ref" in mc else ""))
    for r in mc["parts"]:
        L.append(f"    {r['name']}: {r['mass_g']} g ({r['tag']}) at {r['cg']}{' swap' if r['swap'] else ''}")
    b = res["balance"]
    if b["applies"]:
        L.append(f"  balance {'PASS' if b['pass_'] else 'FAIL'} (horizontal CG offset <= {b['tol_mm']} mm, thrust axis {b['thrust_axis']}):")
        for k, c in b["cases"].items():
            L.append(f"    {k}: {c['horizontal_mm']} mm horizontal, {c['axial_mm']} mm axial, {c['mass_g']} g{'' if c['pass_'] else ' FAIL'}"
                     + ("" if c.get("gated", True) else " (reported, not gated)"))
    else:
        L.append("  balance: n/a (no cg_ref)")
    mb = res["mass_budget"]
    if mb["applies"]:
        L.append(f"  mass budget (soft): {mb['mass_g']} g" + (f" of {mb['budget_g']} g ({mb['pct']} %)" if "budget_g" in mb else ""))
        if mb["over"]:
            L.append("WARN  over budget: " + "; ".join(([f"total {mb['mass_g']} of {mb['budget_g']} g ({mb['pct']} %)"] if not mb.get("total_ok", True) else [])
                                                        + [f"{o['name']} {o['mass_g']} of {o['budget_g']} g" for o in mb["over_parts"]]))
    pr = res["probes"]
    if pr["applies"]:
        L.append(f"  crash probes {'PASS' if pr['pass_'] else 'FAIL'} (cage ahead >= {pr['probe_margin_mm']} mm; {pr['n_axes']} axes x {pr['n_roll']} approach "
                 f"angles, res {pr['res_mm']} mm, bins {pr['bin_mm']} mm, {pr['n_points']} surface points):")
        for r in pr["per_probe_part"]:
            L.append(f"    {r['probe']} / {r['part']}: {r['margin_mm']} mm ({t}) from {r['dir_label']}, axis {r['axis']}, roll {r['roll_deg']}, "
                     f"lateral {r['lateral_mm']} mm, contact {r['contact']}{' unshielded' if r['unshielded'] else ''}{'' if r['pass_'] else ' FAIL'}")
    else:
        L.append("  crash probes: " + ("skipped" if pr.get("skipped") else "n/a (no cage or no protected part)"))
    ac = res["access"]
    if ac["applies"]:
        L.append(f"  access {'PASS' if ac['pass_'] else 'FAIL'} (swept tool overlap <= {ac['tol_mm3']} mm³):")
        for p in ac["paths"]:
            tl = p["tool"] if isinstance(p["tool"], str) else ", ".join(f"{k} {v}" for k, v in p["tool"].items())
            L.append(f"    {p['name']} ({p['part']}, {tl}, {p['length_mm']:g} mm along {p['dir']}): {p['overlap_mm3']} mm³ ({t})"
                     + ("".join(f"; {b['part']} {b['overlap_mm3']} mm³" for b in p["touches"])) + ("" if p["pass_"] else " BLOCKED"))
    L.append(res["summary"])
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="cage first, compact, fit, mass/CG on an assembled manifest")
    ap.add_argument("manifest"); ap.add_argument("--out"); ap.add_argument("--report", action="store_true", help="report only, exit 0")
    ap.add_argument("--dirs", type=int, default=None)
    ap.add_argument("--no-probes", action="store_true", help="skip the edge / post crash probes")
    a = ap.parse_args(argv)
    res = run(a.manifest, n_dirs=a.dirs, probes=False if a.no_probes else None)
    print(report(res))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
    return 0 if (res["pass"] or a.report) else 1


if __name__ == "__main__":
    sys.exit(main())
