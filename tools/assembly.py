"""One assembly file per system (24 Oct 2026). Every tool reads the same part poses from
`nox/<machine>/<system>.assembly.json`; the system's owner session is the only writer of its file (`nox/ASSEMBLY.md`).

The format IS the `system_check.py` manifest (read its docstring for the part and top-level keys); this module adds:
  part keys   "owner" (session / system that owns the part), "version", "source" (script that made it),
              "moves" ("module:function" or "path/file.py:function"; function(s, part) -> 4x4 world pose of the part at
              state s, s = 0 is the file's pose), "note" (free text). Unknown part keys are kept on save.
  top keys    "machine" (box / scout / overseer), "note", and the reserved "access" and "probes" (system_check
              checks; passed through untouched). Unknown top-level keys are kept on save.

Python:
    from assembly import load
    a = load("nox/overseer/overseer.assembly.json")
    a.parts, a.names, a.part("pack"), a.pose("pack")
    a.meshes()                      # posed trimesh list, same order as a.parts
    res = a.check()                 # system_check.run on the file; res["summary"] is the System line for summary.md
    a.set_pose("pack", {"t": [0, 0, -40]}); a.save()        # owner session only
    a.at(**{"plunger +y_x+": 1.0}).check()                   # kinematic state from the part's "moves" function
    a.scene().write("runs/<sys>/<ver>/assembly.html")       # motion.py Scene, bought parts printed=False
    a.export_glb("x.glb"); a.export_stl("x.stl")

CLI:
    python3 nox/assembly.py <file> [--list] [--check] [--report] [--out check.json] [--dirs N] [--glb out.glb]
                                   [--stl out.stl] [--html out.html]
--check prints the system_check report and exits 1 on a gate FAIL (cage first, probes, access, fit, balance; mass budget only warns) unless --report.
"""
import copy, importlib, importlib.util, json, os, sys
import numpy as np, trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (HERE, REPO):
    if _p not in sys.path: sys.path.insert(0, _p)
import system_check as sc

GEOM = ("mesh", "box", "cylinder", "sphere", "trimesh")
ROLES = ("cage", "protected", "structure", "bought")
TAGS = ("P", "M", "C", "E", "F")
TOP_ORDER = ("name", "machine", "note", "cg_ref", "thrust_axis", "cg_tol_mm", "cage_margin_mm", "fit_tol_mm3", "n_dirs",
             "openings", "contact_pairs", "clearance_pairs", "guard_offsets", "access", "probes", "parts")
PART_ORDER = ("name", "role", "protect", "swap", "mesh", "box", "cylinder", "sphere", "pose", "material", "mass_g", "cg",
              "tag", "owner", "version", "source", "moves", "note")
ROLE_COLOR = {"cage": "#333333", "protected": "#2f4fbf", "structure": "#8899aa", "bought": "#b0b0b0"}


def _resolve_fn(spec):
    """'module:function' (importable with nox/ and the repo root on the path) or 'path/file.py:function'."""
    mod, fn = spec.rsplit(":", 1)
    if mod.endswith(".py"):
        path = mod if os.path.isabs(mod) else next((p for p in (os.path.join(REPO, mod), os.path.join(HERE, mod)) if os.path.exists(p)), mod)
        name = "_moves_" + os.path.splitext(os.path.basename(path))[0]
        m = sys.modules.get(name)
        if m is None:
            sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
            sys.modules[name] = m; sp.loader.exec_module(m)
        return getattr(m, fn)
    return getattr(importlib.import_module(mod), fn)


def _pose_out(p):
    """pose for the file: {t, rpy_deg} dicts kept, matrices as nested lists rounded to 1e-6."""
    if isinstance(p, dict): return {k: [round(float(x), 6) for x in v] for k, v in p.items()}
    return [[round(float(x), 6) for x in row] for row in np.asarray(p, float).reshape(4, 4)]


class Assembly:
    def __init__(self, data, path=None):
        self.data = data
        self.path = os.path.abspath(path) if path else None
        self.base = os.path.dirname(self.path) if self.path else os.getcwd()
        self._cache = {}

    # ---- parts
    @property
    def parts(self):
        return self.data["parts"]

    @property
    def names(self):
        return [p["name"] for p in self.parts]

    @property
    def name(self):
        return self.data.get("name", "system")

    def part(self, name):
        for p in self.parts:
            if p["name"] == name: return p
        raise KeyError(f"{self.name}: no part {name!r}")

    def pose(self, name):
        return sc.pose_matrix(self.part(name).get("pose"))

    def set_pose(self, name, pose):
        """pose: 4x4 (array or nested list), flat 16, or {"t": [x, y, z], "rpy_deg": [r, p, y]}."""
        if not isinstance(pose, dict):
            a = np.asarray(pose, float)
            if a.size == 16: pose = a.reshape(4, 4)
            sc.pose_matrix(pose)                                   # validates the shape
        else:
            sc.pose_matrix(pose)
        self.part(name)["pose"] = _pose_out(pose)
        self._cache.pop(name, None)
        return self

    def move(self, name, s):
        """world pose of a part at state s from its "moves" function."""
        p = self.part(name)
        if not p.get("moves"): raise ValueError(f"part {name!r} has no moves function")
        return np.asarray(_resolve_fn(p["moves"])(float(s), p), float)

    def at(self, **states):
        """copy of the assembly with the named parts moved to state s (their "moves" functions)."""
        a = Assembly(copy.deepcopy(self.data), self.path)
        for nm, s in states.items():
            a.set_pose(nm, self.move(nm, s))
        return a

    def validate(self):
        """format errors (empty list = valid)."""
        err = []
        names = self.names
        if len(set(names)) != len(names): err.append("part names must be unique")
        for p in self.parts:
            nm = p.get("name", "?")
            g = [k for k in GEOM if k in p]
            if len(g) != 1: err.append(f"{nm}: needs exactly one of mesh / box / cylinder / sphere (has {g})")
            if p.get("role", "structure") not in ROLES: err.append(f"{nm}: role {p.get('role')!r} not in {ROLES}")
            if p.get("role") == "bought" and "mass_g" not in p: err.append(f"{nm}: bought part needs mass_g")
            if p.get("tag") and p["tag"] not in TAGS: err.append(f"{nm}: tag {p['tag']!r} not in {TAGS}")
            try: sc.pose_matrix(p.get("pose"))
            except Exception as e: err.append(f"{nm}: pose {e}")
            if "mesh" in p:
                try: sc._resolve(p["mesh"], self.base)
                except FileNotFoundError: err.append(f"{nm}: mesh {p['mesh']} not found")
        for c in self.data.get("contact_pairs", []) + self.data.get("clearance_pairs", []):
            for n in c[:2]:
                if n not in names: err.append(f"pair {c}: no part {n!r}")
        return err

    # ---- geometry
    def mesh(self, name):
        p = self.part(name)
        key = json.dumps({k: p.get(k) for k in ("mesh", "box", "cylinder", "sphere", "pose")}, sort_keys=True, default=str)
        hit = self._cache.get(name)
        if hit is None or hit[0] != key:
            hit = (key, sc.load_part(p, self.base)[0]); self._cache[name] = hit
        return hit[1].copy()

    def meshes(self):
        return [self.mesh(n) for n in self.names]

    def manifest(self):
        """the system_check manifest: this file with mesh paths made absolute."""
        m = copy.deepcopy(self.data)
        for p in m["parts"]:
            if "mesh" in p: p["mesh"] = sc._resolve(p["mesh"], self.base)
        return m

    def check(self, **kw):
        """system_check.run on the assembly (kw passed through, e.g. n_dirs)."""
        bad = self.validate()
        if bad: raise ValueError(f"{self.name}: " + "; ".join(bad))
        return sc.run(self.manifest(), **kw)

    def scene(self, title=None, subtitle="", faces=30000, move_s=None, duration_s=6.0):
        """motion.py Scene: every part posed, grouped by role, bought parts printed=False. Parts with a "moves" function
        run 0 -> 1 -> 0 over duration_s (move_s=False keeps everything at the file's pose)."""
        from motion import Scene
        s = Scene(title or f"{self.name} assembly", subtitle or self.data.get("note", "")[:160])
        movers = [p["name"] for p in self.parts if p.get("moves")] if move_s is not False else []
        for p in self.parts:
            role = p.get("role", "structure")
            s.part(p["name"], self.mesh(p["name"]), group=role, color=p.get("color", ROLE_COLOR.get(role)),
                   printed=p.get("printed", role != "bought"), faces=faces,
                   note=" ".join(str(x) for x in (p.get("version", ""), p.get("tag", ""), p.get("owner", "")) if x))
        if movers:
            P0 = {n: self.pose(n) for n in movers}
            def poses(t):
                u = t / duration_s; st = 1 - abs(2 * u - 1)
                return {n: self.move(n, st) @ np.linalg.inv(P0[n]) for n in movers}
            s.timeline(poses, duration_s=duration_s, fps=15)
        return s

    def export_glb(self, path):
        scn = trimesh.Scene()
        for p in self.parts:
            m = self.mesh(p["name"])
            m.visual.face_colors = trimesh.visual.color.hex_to_rgba(p.get("color", ROLE_COLOR.get(p.get("role", "structure"))))
            scn.add_geometry(m, node_name=p["name"], geom_name=p["name"])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        scn.export(path); return path

    def export_stl(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        trimesh.util.concatenate(self.meshes()).export(path); return path

    # ---- file
    def dumps(self):
        """JSON with one part per line (diffs read part by part); known keys first, unknown keys kept after them."""
        d = self.data
        def ordered(x, order):
            return {**{k: x[k] for k in order if k in x}, **{k: v for k, v in x.items() if k not in order}}
        top = ordered(d, TOP_ORDER)
        lines = []
        for k, v in top.items():
            if k == "parts":
                body = ",\n".join("  " + json.dumps(ordered(p, PART_ORDER), ensure_ascii=False) for p in v)
                lines.append(f' "parts": [\n{body}\n ]')
            else:
                lines.append(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}")
        return "{\n" + ",\n".join(lines) + "\n}\n"

    def save(self, path=None):
        if not (path or self.path): raise ValueError("no path")
        path = os.path.abspath(path or self.path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(self.dumps())
        if self.path is None or os.path.dirname(path) != self.base:
            self.path, self.base = path, os.path.dirname(path)
        return path

    def table(self):
        rows = [f"{self.name} ({self.data.get('machine', '?')}): {len(self.parts)} parts"]
        rows.append(f"  {'name':34s} {'role':9s} {'geometry':9s} {'tag':3s} {'mass_g':>8s}  {'extents mm':22s} owner / version")
        for p, m in zip(self.parts, self.meshes()):
            g = next(k for k in GEOM if k in p)
            ex = "x".join(f"{e:.0f}" for e in m.extents)
            mg = f"{p['mass_g']:.1f}" if "mass_g" in p else "(vol)"
            flags = ("P" if p.get("protect") else "") + ("S" if p.get("swap") else "") + ("K" if p.get("moves") else "")
            rows.append(f"  {p['name'][:34]:34s} {p.get('role', 'structure'):9s} {g:9s} {p.get('tag', ''):3s} {mg:>8s}  {ex:22s} "
                        f"{p.get('owner', '')} {p.get('version', '')} {flags}".rstrip())
        rows.append("  flags: P protect, S swap, K moves")
        return "\n".join(rows)


def load(path):
    return Assembly(json.load(open(path)), path)


def new(name, machine, parts=None, **top):
    return Assembly(dict(name=name, machine=machine, **top, parts=list(parts or [])))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="load a system assembly file, list, check, export")
    ap.add_argument("file"); ap.add_argument("--list", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true", help="with --check: exit 0 on a FAIL")
    ap.add_argument("--out"); ap.add_argument("--dirs", type=int); ap.add_argument("--glb"); ap.add_argument("--stl"); ap.add_argument("--html")
    a = ap.parse_args(argv)
    asm = load(a.file)
    bad = asm.validate()
    if bad:
        print("INVALID:\n  " + "\n  ".join(bad)); return 2
    if a.list or not (a.check or a.glb or a.stl or a.html):
        print(asm.table())
    rc = 0
    if a.check:
        res = asm.check(n_dirs=a.dirs)
        print(sc.report(res))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1, default=float)
        rc = 0 if (res["pass"] or a.report) else 1
    if a.glb: print("glb:", asm.export_glb(a.glb))
    if a.stl: print("stl:", asm.export_stl(a.stl))
    if a.html: asm.scene().write(a.html)
    return rc


if __name__ == "__main__":
    sys.exit(main())
