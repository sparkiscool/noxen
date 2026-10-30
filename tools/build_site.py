"""Build the viewer page from systems/*.assembly.json and parts/*/.

    python tools/build_site.py            -> site/assembly.json + site/meshes/<id>.glb
    python tools/build_site.py --inline   -> also site/<drone>.html, one self-contained page per system (tools/motion.py Scene:
                                             parts by group and colour, the assembly's overlays, a turntable) with every current
                                             part embedded: the drone's artifact. A session publishes it at the system's
                                             `artifact` URL (systems/<drone>.assembly.json), never a hand-built page.
    python tools/build_site.py --offline  -> skip the origin/main check (no network)

Outside CI the build refuses a checkout behind origin/main: a page built from an old checkout shows old parts.

Status is computed, never stored:
  done     parts/<id>/result.json exists, solid_ok true, and its setup_sha is the setup part.py builds now (tools/part_sha.py)
  failed   result.json exists, solid_ok false, sha matches
  stale    result.json exists but the setup changed since (a run is due)
  pending  part.py exists, no result.json yet
  placed   no part.py: a bought part or a hand-drawn mesh, shown as is
Nothing here is ever read by Claude; CI runs it.
"""
import hashlib, json, os, sys, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_sha import stale as _stale
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
def view_mesh(m, pid=""):
    """the page's copy of a part: stray faces and shells dropped, then decimated. An exported STL carries a few loose
    triangles and 1-3 face shells beside the part (the manifold -> STL export, 29 Oct: node 21 had 126 of them); the
    quadric decimation of a mesh holding them tears the part open (node 5 on the page lost a quarter of its volume and
    its faces, Spark 29 Oct: "looks weird and unrendered"), so only the closed shells of at least 1 mm3 are kept. A
    part whose main shell itself is open (a recut export) still decimates, and the volume lost is printed."""
    if len(m.faces) <= 200:
        return m
    m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.update_faces(m.unique_faces()); m.remove_unreferenced_vertices()
    bodies = m.split(only_watertight=False)
    keep = [b for b in bodies if b.is_watertight and abs(b.volume) >= 1.0]
    if not keep or len(keep) < len(bodies) // 2:                 # a drawn multi-body mesh whose shells are not closed: drop only the loose triangles
        keep = [b for b in bodies if len(b.faces) > 3] or bodies
    m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    if len(m.faces) > MAX_FACES:
        v0 = abs(m.volume)
        m = m.simplify_quadric_decimation(face_count=MAX_FACES)
        if v0 > 0 and abs(abs(m.volume) - v0) > 0.02 * v0:
            print(f"site: {pid}: the decimated mesh lost {100 * (v0 - abs(m.volume)) / v0:.0f} % of its volume (the STL is not watertight)")
    return m


MAX_FACES = 20000                        # the inline page must stay under the 16 MB artifact limit (29 Oct: 16.8 MB at 40k, 15.2 MB at 24k with 58 parts)


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


def overlays(o):
    """Thin-line context for the artifact page (systems/<drone>.assembly.json `overlays`): swept props, the L2 field, the camera views."""
    import math
    line = lambda a, b, rr=0.7: trimesh.creation.cylinder(rr, np.linalg.norm(np.subtract(b, a)), sections=6, segment=np.vstack([a, b]))
    out = []
    if "motors_xy" in o:
        r0, r1 = o.get("prop_r", [124.5, 127.0]); rings = []
        for mx, my in o["motors_xy"]:
            a = trimesh.creation.annulus(r0, r1, 1.0, sections=96); a.apply_translation([mx, my, o.get("prop_z", 82.0)]); rings.append(a)
        out.append(("props (swept)", trimesh.util.concatenate(rings), "#d9a400"))
    if "l2_axis" in o:
        Ax, B = np.array(o["l2_axis"]), np.array(o["l2_base"]); up = np.array([-Ax[2], 0, Ax[0]]); ey = np.array([0, 1.0, 0]); Rf = 220.0
        lines = []
        for t in np.linspace(0, math.pi, 7):
            d = math.cos(t) * up + math.sin(t) * ey
            pts = [B + Rf * (math.cos(f) * d + math.sin(f) * Ax) for f in np.linspace(0, math.pi, 25)]
            lines += [line(pts[i], pts[i + 1]) for i in range(24)]
        rim = [B + Rf * (math.cos(f) * up + math.sin(f) * ey) for f in np.linspace(0, 2 * math.pi, 49)]
        lines += [line(rim[i], rim[i + 1]) for i in range(48)]
        out.append(("L2 field (90 deg mode)", trimesh.util.concatenate(lines), "#4a90d9"))
    if o.get("cams"):
        fr = []; Rf = 220.0
        for c in o["cams"]:
            L, f, u, r = (np.array(c[k]) for k in ("lens", "f", "u", "r")); ha, va = math.radians(c["ha"]), math.radians(c["hv"])
            cor = []
            for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                d = f + math.tan(ha) * sy * r + math.tan(va) * sz * u; cor.append(L + Rf * d / np.linalg.norm(d))
            fr += [line(L, c_) for c_ in cor] + [line(cor[i], cor[(i + 1) % 4]) for i in range(4)]
        out.append(("camera views", trimesh.util.concatenate(fr), "#3fa34d"))
    return out


def note(p, r):
    """What the part's line on the artifact page says: the run's numbers for a generated part, the layout model's for the rest."""
    if r:
        stg = r.get("strength") or {}
        s = f"{r.get('mass_g', 0):.1f} g (M, run {'PASS' if r.get('solid_ok') and r.get('strength_ok', True) else 'FAIL'}"
        if stg.get("worst") is not None:
            s += f", worst {stg['worst']} {stg.get('worst_case', '')}".rstrip()
        return s + ")"
    return f"{p.get('mass_g') or 0:.0f} g ({p.get('tag', 'E')}, layout model)"


def artifact_page(a, S, meshes, commit, path):
    """One self-contained page per drone, the look of the system artifact: parts by group and colour, the overlays, a turntable."""
    import math
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from motion import Scene
    listed = sum(p.get("mass_g") or 0 for p in S["parts"])
    sub = f"{a.get('note', '').split('. ')[0]}. {len(S['parts'])} parts, {listed:.0f} g listed"
    if a.get("mass_budget_g"):
        sub += f" of {a['mass_budget_g']} g budget"
    sub += f"; built from {os.environ.get('GITHUB_REPOSITORY', 'sparkiscool/noxen')} @ {commit}"
    s = Scene(S["title"], sub)
    by_id = {p["id"]: p for p in S["parts"]}
    for p in a["parts"]:
        pid = p.get("id") or p["name"]; q = by_id[pid]
        s.part(p["name"], meshes[pid], group=p.get("group", "part"), color=p.get("color"), opacity=p.get("opacity"),
               printed=p.get("material") in ("PETG", "TPU") and p.get("role") != "bought", note=note(p, q.get("_result")))
    for nm, m, col in overlays(a.get("overlays", {})):
        s.part(nm, m, group="views" if "view" in nm or "field" in nm else "bought", color=col, printed=False)
    names = [p["name"] for p in s.parts]
    spin = lambda t: trimesh.transformations.rotation_matrix(math.radians(360 * t / 24), [0, 0, 1])
    s.timeline(lambda t: {n: spin(t) for n in names}, duration_s=24, fps=24)
    s.view("iso", (700, -800, 500), (60, 0, 20)); s.view("side", (60, -1300, 40), (60, 0, 40)); s.view("front", (1400, 0, 60), (0, 0, 40))
    s.view("top", (60, 0, 1300), (60, 0, 0)); s.view("under", (300, -300, -900), (0, 0, -50))
    s.write(path)


def status(pid):
    d = os.path.join(ROOT, "parts", pid)
    py, rj = os.path.join(d, "part.py"), os.path.join(d, "result.json")
    if not os.path.exists(py):
        return "placed", None
    if not os.path.exists(rj):
        return "pending", None
    r = json.load(open(rj))
    if _stale(py, r):                                 # tools/part_sha.py: the setup changed since this result
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
            src = p.get("part", pid)             # a mirror or a second placing of another part's folder: its mesh and status, this pose
            st, r = status(src)
            mesh = None
            cand = os.path.join(ROOT, "parts", src, "part.stl")
            if st in ("done", "stale", "pending", "failed") and os.path.exists(cand):
                mesh = cand
            elif st == "placed" and "mesh" in p and os.path.exists(os.path.join(ROOT, p["mesh"])):
                mesh = os.path.join(ROOT, p["mesh"])
            if mesh:
                m = view_mesh(trimesh.load(mesh, force="mesh"), pid)
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
                meshes[pid] = m
            prev = os.path.join(ROOT, "parts", src, "preview.png")
            if os.path.exists(prev):
                os.makedirs(os.path.join(SITE, "previews"), exist_ok=True)
                import shutil; shutil.copy(prev, os.path.join(SITE, "previews", f"{src}.png"))
            fb = os.path.join(ROOT, "parts", src, "feedback.md")
            stg = (r or {}).get("strength") or {}
            S["parts"].append(dict(id=pid, name=p["name"], role=p.get("role", ""), status=st, kind=kind,
                                   glb=os.path.relpath(glb, SITE), mass_g=(r or {}).get("mass_g", p.get("mass_g")),
                                   worst=stg.get("worst"), worst_case=stg.get("worst_case"),
                                   max_worst=(r or {}).get("max_worst"), strength_ok=(r or {}).get("strength_ok"),
                                   solid_ok=(r or {}).get("solid_ok"), total_s=(r or {}).get("total_s"),
                                   preview=f"previews/{src}.png" if os.path.exists(prev) else None,
                                   feedback=open(fb).read() if os.path.exists(fb) else "",
                                   has_script=os.path.exists(os.path.join(ROOT, "parts", src, "part.py"))))
            S["parts"][-1]["_result"] = r
        out["systems"].append(S)
        if inline:
            artifact_page(a, S, meshes, out["commit"], os.path.join(SITE, f"{sysname}.html"))
            meshes.clear()
    for s in out["systems"]:
        for p in s["parts"]:
            p.pop("_result", None)
    json.dump(out, open(os.path.join(SITE, "assembly.json"), "w"), indent=1)
    n = sum(len(s["parts"]) for s in out["systems"])
    print(f"site: {len(out['systems'])} systems, {n} parts, meshes {sum(os.path.getsize(os.path.join(SITE,'meshes',x)) for x in os.listdir(os.path.join(SITE,'meshes')))/1e6:.1f} MB")


if __name__ == "__main__":
    build(inline="--inline" in sys.argv, offline="--offline" in sys.argv)
