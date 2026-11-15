"""Assembly paths (15 Nov, Spark's Scout v4 bench print: the camera had no way into its sleeve; nothing kept its insertion path open).

A path is how a piece goes in: its body (a bought part's mesh, a drawn part's mesh, or a declared envelope) moved from where it sits
out the way it came, by a straight pull (`out` + `length_mm`) or a turn (`turn`: axis, a point on it, the angle range in deg).
The same sweep serves two checks:
  * the part a path runs through declares it (SPEC["assembly_paths"] = ["<drone>:<path name>", ...] in its part.py, or
    Part.assembly_path in the setup): the engine cuts the sweep, grown by the path's fit gap, through that part's envelope like a
    tool sweep, so nothing generated can fill it, and the audit refuses the run when drawn material stands in it (audit `path`);
  * the drone's assembly check (tools/assembly_check.py) sweeps the piece through every part already placed and fails a blocked path.

A path in systems/<drone>.assembly.json "assembly" -> "paths":
    {"name": "camera into cradle", "part": "scout_v4_probe_tilt", "order": 10, "out": [-0.97, 0, -0.26], "length_mm": 30,
     "fit": "slide", "into": ["scout_v4_cradle"], "why": "..."}
    "body": {"box": [x0, x1, y0, y1, z0, z1]} | {"cylinder": {"p0", "p1", "r"}} | {"mesh": "<path>"} replaces the part's mesh as
    the swept body (a generated part's mesh is its result, so a cut in another part cannot be made from it: give a body).
    "turn": {"axis": [0, 1, 0], "at": [16.5, 0, 110], "deg": [-20, 90]} instead of out / length_mm: a nod through its range.
    "moves": [{"out": [0, 1, 0], "length_mm": 4}, {"out": [0, 0, 1], "length_mm": 40}] instead: a pull in steps (a step may be
    {"turn": {"axis", "at", "deg": d}}: turned by d first, the check only; the engine's cut takes pulls).
    "min_body_mm3": n  the check leaves out the moving parts' loose bodies under n mm3 (a screw drawn with the camera that goes in later).
    "against": [ids] a step done in hand, off the drone (the camera into the cradle before the cradle goes in): only these are in its way.
    "with": [ids] the parts that ride on the piece (the LEDs on the camera); "ignore": [ids] what it may pass; "order": when it goes in.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _man(t):
    from .part import man
    return man(t)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def body(path, part_mesh=None, root=ROOT):
    """the swept piece as a manifold: the path's body, else the part's mesh (a trimesh, world frame)"""
    import manifold3d as mf
    import trimesh
    b = path.get("body")
    if b is None:
        if part_mesh is None:
            raise ValueError(f"path {path.get('name')}: no body and no mesh for {path.get('part')}")
        return _man(part_mesh)
    if "box" in b:
        x0, x1, y0, y1, z0, z1 = b["box"]
        return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    if "cylinder" in b:
        c = b["cylinder"]
        return _man(trimesh.creation.cylinder(radius=float(c["r"]), segment=[c["p0"], c["p1"]], sections=48))
    if "mesh" in b or "meshes" in b:
        ms = [_man(trimesh.load(os.path.join(root, f), force="mesh")) for f in ([b["mesh"]] if "mesh" in b else b["meshes"])]
        return mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else ms[0]
    raise ValueError(f"path {path.get('name')}: body must be box, cylinder or mesh")


def _rot(axis, at, deg):
    a = np.radians(deg); u = _unit(axis); at = np.asarray(at, float)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = at - R @ at
    return T


def sweep(path, piece, grow=0.0):
    """the volume the piece passes through on its way in, grown by `grow` per side (a manifold). A pull: the piece moved 0..length_mm
    along `out`, exactly (a Minkowski sum with the segment); grown, the seated end is not grown into its stop (the growth starts
    `grow` out along the pull, so a face resting on a stop stays on it). A turn: the piece at every 2 deg of the range, unioned."""
    import manifold3d as mf
    if "turn" in path:
        t = path["turn"]; d0, d1 = (float(x) for x in t["deg"])
        n = max(2, int(np.ceil(abs(d1 - d0) / 2.0)) + 1)
        pcs = []
        for d in np.linspace(d0, d1, n):
            T = _rot(t["axis"], t["at"], d)
            pcs.append(piece.transform(T[:3, :].tolist()))
        s = mf.Manifold.batch_boolean(pcs, mf.OpType.Add)
        if grow > 0:
            s = s.minkowski_sum(mf.Manifold.sphere(grow, 12))
        return s
    if "moves" in path:                                           # a pull in steps (out of a spline, then up): each from where the last ended
        segs, off = [], np.zeros(3)
        for k, mv in enumerate(path["moves"]):
            q = dict(out=mv["out"], length_mm=mv["length_mm"])
            segs.append(sweep(q, piece.translate(tuple(float(x) for x in off)), grow))
            off = off + _unit(mv["out"]) * float(mv["length_mm"])
        return mf.Manifold.batch_boolean(segs, mf.OpType.Add)
    u = _unit(path["out"]); L = float(path["length_mm"])
    if grow > 0:
        r = float(grow)
        seg = mf.Manifold.batch_hull([mf.Manifold.sphere(r, 12).translate(tuple(u * r)),
                                      mf.Manifold.sphere(r, 12).translate(tuple(u * (L + r)))])
    else:
        seg = mf.Manifold.batch_hull([mf.Manifold.sphere(1e-3, 6), mf.Manifold.sphere(1e-3, 6).translate(tuple(u * L))])
    try:
        return piece.minkowski_sum(seg)
    except Exception:                                             # a piece minkowski cannot take: sampled translates (step <= 0.5 mm)
        n = max(2, int(np.ceil(L / 0.5)) + 1)
        return mf.Manifold.batch_boolean([piece.translate(tuple(u * t)) for t in np.linspace(0, L, n)], mf.OpType.Add)


def poses(path, step_mm=0.5, step_deg=0.5):
    """the piece's poses along the path, seated first (4 x 4 world transforms): a pull every step_mm, a turn every step_deg both ways from 0"""
    if "turn" in path:
        t = path["turn"]; d0, d1 = (float(x) for x in t["deg"])
        out = [np.eye(4)]
        for lo_hi in ((0.0, d1), (0.0, d0)):
            n = int(np.ceil(abs(lo_hi[1]) / step_deg))
            out += [_rot(t["axis"], t["at"], d) for d in np.linspace(0, lo_hi[1], n + 1)[1:]]
        return out
    moves = path.get("moves") or [dict(out=path["out"], length_mm=path["length_mm"])]
    out, cur = [np.eye(4)], np.eye(4)
    for mv in moves:
        if "turn" in mv:                                          # a turn inside a sequence (the cradle nodded to 0 before it lifts out)
            t = mv["turn"]; d = float(t["deg"]); n = max(1, int(np.ceil(abs(d) / step_deg)))
            for a in np.linspace(0, d, n + 1)[1:]:
                out.append(_rot(t["axis"], t["at"], a) @ cur)
            cur = _rot(t["axis"], t["at"], d) @ cur
            continue
        u = _unit(mv["out"]); L = float(mv["length_mm"]); n = max(1, int(np.ceil(L / step_mm)))
        for tt in np.linspace(0, L, n + 1)[1:]:
            T = np.eye(4); T[:3, 3] = u * tt; out.append(T @ cur)
        T = np.eye(4); T[:3, 3] = u * L; cur = T @ cur
    return out


def step_label(path, k):
    """where step k of poses(path) is, in words"""
    if k is None:
        return ""
    T = poses(path)[k]
    if "turn" in path:
        R = T[:3, :3]; ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        u = _unit(path["turn"]["axis"]); s = np.sign(R[2, 1] * u[0] + R[0, 2] * u[1] + R[1, 0] * u[2] or 1.0)
        return f"{s * ang:+.1f} deg into the turn"
    return f"{np.linalg.norm(T[:3, 3]):.1f} mm out"


def drone_file(drone, root=ROOT):
    return os.path.join(root, "systems", f"{drone}.assembly.json")


def declared(ref, root=ROOT):
    """'<drone>:<path name>' -> (path dict, the drone's assembly dict)"""
    drone, name = ref.split(":", 1)
    a = json.load(open(drone_file(drone, root)))
    for p in (a.get("assembly") or {}).get("paths", []):
        if p["name"] == name:
            return p, a
    raise KeyError(f"assembly path {ref!r}: no path of that name in systems/{drone}.assembly.json assembly.paths")


def piece_mesh(path, a, root=ROOT):
    """the mover's mesh for a path (the part's line in the drone file), or None when the path gives its own body"""
    import trimesh
    if path.get("body") is not None:
        return None
    line = next((q for q in a["parts"] if (q.get("id") or q.get("name")) == path["part"]), None)
    if line is None:
        raise KeyError(f"path {path['name']}: part {path['part']} is not in the drone file")
    if os.path.exists(os.path.join(root, "parts", path["part"], "part.py")):
        raise ValueError(f"path {path['name']}: {path['part']} is generated (its mesh is its result): a cut made from it would change with "
                         f"every run of it. Give the path a body (its drawn envelope: box, cylinder or mesh).")
    return trimesh.load(os.path.join(root, line["mesh"]), force="mesh")


def cut_of(path, a, root=ROOT):
    """the manifold a part a path runs through is cut by: the sweep grown by the path's fit gap"""
    from . import fit
    g = max(0.0, fit.gap(path.get("fit", "slide")))
    return sweep(path, body(path, piece_mesh(path, a, root), root), grow=g)


def apply_declared(part, refs, root=ROOT):
    """SPEC["assembly_paths"]: cut each declared path through the part (Part.assembly_path)"""
    for ref in refs or ():
        p, a = declared(ref, root)
        part.assembly_path(ref, cut_of(p, a, root), why=p.get("why", ""))
    return part
