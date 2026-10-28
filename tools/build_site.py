"""Build the viewer page from systems/*.assembly.json and parts/*/.

    python tools/build_site.py            -> site/assembly.json + site/meshes/<id>.glb
    python tools/build_site.py --inline   -> also site/<drone>.html, one self-contained file per system with every current
                                             part embedded: the drone's artifact. A session publishes it at the system's
                                             `artifact` URL (systems/<drone>.assembly.json), never a hand-built page.
    python tools/build_site.py --offline  -> skip the origin/main check (no network)

Outside CI the build refuses a checkout behind origin/main: a page built from an old checkout shows old parts.

Status is computed, never stored:
  done     parts/<id>/result.json exists, solid_ok true, and its part_sha matches the current part.py
  failed   result.json exists, solid_ok false, sha matches
  stale    result.json exists but part.py changed since (a run is due)
  pending  part.py exists, no result.json yet
  placed   no part.py: a bought part or a hand-drawn mesh, shown as is
Nothing here is ever read by Claude; CI runs it.
"""
import base64, hashlib, json, os, sys, glob, subprocess
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
MAX_FACES = 40000


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def git(*args, **kw):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, **kw)


def head(offline=False):
    """The commit this build shows. Outside CI the checkout must be at origin/main (the gate behind every page and artifact)."""
    commit = git("rev-parse", "--short", "HEAD").stdout.strip()
    if os.environ.get("GITHUB_ACTIONS") or offline:
        return commit
    f = git("fetch", "-q", "origin", "main", timeout=120)
    if f.returncode:
        sys.exit(f"build_site: git fetch failed ({f.stderr.strip()}); --offline builds this checkout as is")
    behind = git("rev-list", "--count", "HEAD..origin/main").stdout.strip()
    if behind != "0":
        sys.exit(f"build_site: this checkout is {behind} commit(s) behind origin/main; git pull first, the page must show the current parts")
    return commit


def packed(m):
    """A mesh as base64 float32 positions + uint32 faces: the viewer builds the geometry itself (no loader, no blob URLs)."""
    return {"pos": base64.b64encode(np.asarray(m.vertices, np.float32).tobytes()).decode(),
            "idx": base64.b64encode(np.asarray(m.faces, np.uint32).tobytes()).decode()}


def status(pid):
    d = os.path.join(ROOT, "parts", pid)
    py, rj = os.path.join(d, "part.py"), os.path.join(d, "result.json")
    if not os.path.exists(py):
        return "placed", None
    if not os.path.exists(rj):
        return "pending", None
    r = json.load(open(rj))
    if r.get("part_sha") != sha(py):
        return "stale", r
    ok = r.get("solid_ok") and r.get("strength_ok", True)
    return ("done" if ok else "failed"), r


def pose_matrix(pose):
    if pose is None:
        return np.eye(4)
    if isinstance(pose, list):
        return np.array(pose, float)
    T = np.eye(4)
    rpy = np.radians(pose.get("rpy_deg", [0, 0, 0]))
    cr, sr, cp, sp, cy, sy = np.cos(rpy[0]), np.sin(rpy[0]), np.cos(rpy[1]), np.sin(rpy[1]), np.cos(rpy[2]), np.sin(rpy[2])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = pose.get("t", [0, 0, 0])
    return T


def placeholder(p):
    if "box" in p:
        x0, x1, y0, y1, z0, z1 = p["box"]
        m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0]); m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]); return m
    if "cylinder" in p:
        c = p["cylinder"]
        if "p0" in c:
            return trimesh.creation.cylinder(radius=c["r"], segment=[c["p0"], c["p1"]], sections=48)
        ax = np.array(c.get("axis", [0, 0, 1]), float); ax /= np.linalg.norm(ax); ctr = np.array(c["center"], float)
        return trimesh.creation.cylinder(radius=c["r"], segment=[ctr - ax * c["h"] / 2, ctr + ax * c["h"] / 2], sections=48)
    if "sphere" in p:
        m = trimesh.creation.icosphere(subdivisions=2, radius=p["sphere"]["r"]); m.apply_translation(p["sphere"]["center"]); return m
    return None


def build(inline=False, offline=False):
    os.makedirs(os.path.join(SITE, "meshes"), exist_ok=True)
    out = {"systems": [], "repo": os.environ.get("GITHUB_REPOSITORY", "sparkiscool/noxen"), "commit": head(offline)}
    meshes = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "systems", "*.assembly.json"))):
        a = json.load(open(f))
        sysname = os.path.basename(f).split(".")[0]
        S = {"name": a.get("name", sysname), "file": sysname, "title": a.get("title") or a.get("name", sysname),
             "artifact": a.get("artifact"), "note": a.get("note", ""), "mass_budget_g": a.get("mass_budget_g"), "parts": []}
        for p in a["parts"]:
            pid = p.get("id") or p["name"]
            st, r = status(pid)
            mesh = None
            cand = os.path.join(ROOT, "parts", pid, "part.stl")
            if st in ("done", "stale", "pending", "failed") and os.path.exists(cand):
                mesh = cand
            elif st == "placed" and "mesh" in p and os.path.exists(os.path.join(ROOT, p["mesh"])):
                mesh = os.path.join(ROOT, p["mesh"])
            if mesh:
                m = trimesh.load(mesh, force="mesh")
                if len(m.faces) > MAX_FACES:
                    m = m.simplify_quadric_decimation(face_count=MAX_FACES)
                kind = "mesh"
            else:
                m = placeholder(p)
                kind = "placeholder"
                if m is None:                       # a part that will be generated but has no placeholder shape yet
                    m = trimesh.creation.box(extents=[20, 20, 20]);
            m.apply_transform(pose_matrix(p.get("pose")))
            glb = os.path.join(SITE, "meshes", f"{sysname}__{pid}.glb")
            m.export(glb)
            if inline:
                meshes[f"{sysname}__{pid}"] = packed(m)
            prev = os.path.join(ROOT, "parts", pid, "preview.png")
            if os.path.exists(prev):
                os.makedirs(os.path.join(SITE, "previews"), exist_ok=True)
                import shutil; shutil.copy(prev, os.path.join(SITE, "previews", f"{pid}.png"))
            fb = os.path.join(ROOT, "parts", pid, "feedback.md")
            stg = (r or {}).get("strength") or {}
            S["parts"].append(dict(id=pid, name=p["name"], role=p.get("role", ""), status=st, kind=kind,
                                   glb=os.path.relpath(glb, SITE), mass_g=(r or {}).get("mass_g", p.get("mass_g")),
                                   worst=stg.get("worst"), worst_case=stg.get("worst_case"),
                                   max_worst=(r or {}).get("max_worst"), strength_ok=(r or {}).get("strength_ok"),
                                   solid_ok=(r or {}).get("solid_ok"), total_s=(r or {}).get("total_s"),
                                   preview=f"previews/{pid}.png" if os.path.exists(prev) else None,
                                   feedback=open(fb).read() if os.path.exists(fb) else "",
                                   has_script=os.path.exists(os.path.join(ROOT, "parts", pid, "part.py"))))
        out["systems"].append(S)
    json.dump(out, open(os.path.join(SITE, "assembly.json"), "w"), indent=1)
    n = sum(len(s["parts"]) for s in out["systems"])
    print(f"site: {len(out['systems'])} systems, {n} parts, meshes {sum(os.path.getsize(os.path.join(SITE,'meshes',x)) for x in os.listdir(os.path.join(SITE,'meshes')))/1e6:.1f} MB")
    if inline:
        html = open(os.path.join(SITE, "index.html")).read()

        def page(systems, title, path):
            data = json.loads(json.dumps({"systems": systems, "repo": out["repo"], "commit": out["commit"], "title": title}))
            data["meshes"] = {k: v for k, v in meshes.items() if k.split("__")[0] in {s["file"] for s in systems}}
            for s in data["systems"]:
                for p in s["parts"]:
                    if p["preview"]:                    # the page stands alone: previews travel with it
                        b = open(os.path.join(SITE, p["preview"]), "rb").read()
                        p["preview"] = "data:image/png;base64," + base64.b64encode(b).decode()
            h = html.replace("<title>Nox assembly</title>", f"<title>{title}</title>", 1)
            h = h.replace("/*INLINE*/", "window.INLINE=" + json.dumps(data) + ";", 1)
            open(path, "w").write(h)
            print(os.path.relpath(path, ROOT), round(os.path.getsize(path) / 1e6, 1), "MB")

        page(out["systems"], "Nox assembly", os.path.join(SITE, "standalone.html"))
        for s in out["systems"]:
            page([s], s["title"], os.path.join(SITE, f"{s['file']}.html"))


if __name__ == "__main__":
    build(inline="--inline" in sys.argv, offline="--offline" in sys.argv)
