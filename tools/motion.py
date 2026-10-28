"""Nox motion artifact (23 Oct 2026): any set of part meshes plus a timeline -> one HTML file (meshes embedded; three.js from the CDN) with the parts
in motion (scrub, speed, views, groups, printed-only, exploded, optional readout series), and headless an mp4 / gif / png
of the same page. Generic replacement for the hand-rolled arm / detent pages: a part script builds its meshes and poses
and hands them here; nothing visual is written per part.

Python:
    from motion import Scene
    s = Scene("Swap arm v4", "cycle at the -90 bay")
    s.part("hub", hub_mesh, group="arm", color="#8899aa", printed=True)          # trimesh, in world coords at pose identity
    s.part("motor_A", motor_mesh, group="arm", printed=False)                    # bought context part, no mass
    s.timeline(lambda t: {"hub": T_hub(t), "motor_A": T_hub(t)}, duration_s=18, fps=30)   # t in s -> {part: 4x4 or (xyz, rpy_deg)}
    s.series("clearance", values, unit="mm", bad_below=0.5)                       # optional readout strip, one value per frame
    s.write("runs/arm/v4/motion.html")                                           # the artifact
    s.video("runs/arm/v4/motion.mp4")                                            # + .gif + .png beside it (headless Chromium + ffmpeg)

CLI (JSON spec, poses as [x, y, z, rx, ry, rz] mm / deg or a flat 4x4 row-major):
    python3 nox/motion.py spec.json [--video]
    {"title": "...", "subtitle": "...", "out": "runs/<part>/<version>/motion", "duration_s": 6,
     "parts": [{"name": "hub", "file": "print/box_hub_v5.stl", "group": "arm", "color": "#8899aa", "printed": true, "faces": 30000}],
     "frames": [{"hub": [0, 0, 0, 0, 0, 30]}, ...]}

Masses on the page: mesh volume (M) x 1.27 g/cm3 PETG (E, solid) for printed parts. Meshes are decimated (fast_simplification, default
30k faces) and quantised to 0.02 mm int16 (coarser, extent / 32000, on a part over 640 mm), normals int8, so a 10 MB STL costs ~0.5 MB in the page; keep the page under 15 MB.
Needs: trimesh, numpy; fast_simplification (decimation); video only: playwright, the box's Chromium (/opt/pw-browsers/chromium or
CHROMIUM_PATH), ffmpeg on PATH, three.js from npm (fetched once into ~/.cache/nox-motion).
"""
import base64, json, math, os, shutil, subprocess, sys, tempfile
import numpy as np, trimesh

RHO = {"PETG": 1.27e-3, "TPU": 1.21e-3}     # g / mm3 (E)
THREE_VER = "0.128.0"
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_template.html")
PALETTE = ["#8899aa", "#cc6644", "#4477cc", "#44aa66", "#aa8833", "#9955bb", "#dd9977", "#556677", "#33aaaa", "#cc4477"]


# ------------------------------------------------------------------ meshes
def decimate(m, target):
    m = m.copy(); m.merge_vertices()
    if target and len(m.faces) > target:
        try:
            import fast_simplification as fsimp
            v, f = fsimp.simplify(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32), target_count=target)
            m = trimesh.Trimesh(v, f, process=True)
        except ImportError:
            m = m.simplify_quadric_decimation(face_count=target)
    return m


def pack(m, crease_deg=35.0):
    """crease-split smooth normals; positions int16 at 0.02 mm + offset, normals int8, indices uint16 / uint32; base64."""
    s = trimesh.graph.smooth_shade(m, angle=math.radians(crease_deg))
    v = np.asarray(s.vertices, np.float64); off = v.min(0)
    st = max(0.02, float((v - off).max()) / 32000)              # 0.02 mm up to 640 mm of extent, coarser beyond (int16)
    q = np.round((v - off) / st).astype(np.int16)
    n = np.round(np.asarray(s.vertex_normals) * 127).astype(np.int8)
    big = len(v) > 65000
    idx = np.asarray(s.faces).astype(np.uint32 if big else np.uint16).ravel()
    b = lambda a: base64.b64encode(a.tobytes()).decode()
    return dict(off=[round(float(x), 3) for x in off], st=st, n=len(v), big=big, nf=int(len(s.faces)), p=b(q), nrm=b(n), i=b(idx))


# ------------------------------------------------------------------ poses
def as_matrix(pose):
    """4x4 from: 4x4 array / nested list, flat 16, or [x, y, z, rx, ry, rz] (mm, deg, R = Rz Ry Rx applied to the mesh)."""
    a = np.asarray(pose, np.float64)
    if a.shape == (4, 4):
        return a
    if a.size == 16:
        return a.reshape(4, 4)
    if a.size == 6:
        x, y, z, rx, ry, rz = a
        T = trimesh.transformations.euler_matrix(math.radians(rx), math.radians(ry), math.radians(rz), "sxyz")
        T[:3, 3] = (x, y, z)
        return T
    raise ValueError(f"pose must be 4x4, 16 or 6 numbers, got shape {a.shape}")


# ------------------------------------------------------------------ scene
class Scene:
    def __init__(self, title, subtitle="", material="PETG"):
        self.title, self.subtitle, self.material = title, subtitle, material
        self.parts, self.frames, self.duration, self.series_ = [], None, 0.0, []
        self.views = {}

    def part(self, name, mesh, group="part", color=None, printed=True, faces=30000, note=""):
        """mesh: trimesh in world coords at the identity pose. Same mesh object under several names is embedded once."""
        self.parts.append(dict(name=name, mesh=mesh, group=group, color=color, printed=printed, faces=faces, note=note))
        return self

    def timeline(self, poses, duration_s, fps=30):
        """poses: callable t_seconds -> {part: pose} (parts not named keep identity), or a list of such dicts (one per frame)."""
        self.duration = float(duration_s)
        if callable(poses):
            n = max(2, int(round(duration_s * fps)))
            self.frames = [poses(i * duration_s / n) for i in range(n)]
        else:
            self.frames = list(poses)
        return self

    def series(self, name, values, unit="", bad_below=None, bad_above=None):
        """one number per frame, drawn as a strip under the scrub bar and read out at the current frame."""
        self.series_.append(dict(name=name, unit=unit, values=[None if v is None else round(float(v), 3) for v in values],
                                 bad_below=bad_below, bad_above=bad_above))
        return self

    def view(self, name, eye, target=(0, 0, 0)):
        """extra camera preset (mm): eye position and look-at point."""
        self.views[name] = dict(eye=list(map(float, eye)), target=list(map(float, target)))
        return self

    # ---- build
    def data(self):
        if self.frames is None:
            self.frames, self.duration = [{}], 1.0
        meshes, parts, seen = {}, [], {}
        for k, p in enumerate(self.parts):
            key = seen.get(id(p["mesh"]))
            if key is None:
                key = p["name"]; seen[id(p["mesh"])] = key
                m = decimate(p["mesh"], p["faces"])
                meshes[key] = pack(m)
                vol = abs(float(p["mesh"].volume)) if p["mesh"].is_volume else None
                meshes[key]["mass"] = round(vol * RHO.get(self.material, 1.27e-3), 1) if (vol and p["printed"]) else None
            parts.append(dict(name=p["name"], mesh=key, group=p["group"], color=p["color"] or PALETTE[k % len(PALETTE)],
                              printed=bool(p["printed"]), note=p["note"]))
        names = [p["name"] for p in self.parts]
        F = np.zeros((len(self.frames), len(names), 16), np.float32)
        for i, fr in enumerate(self.frames):
            for j, nm in enumerate(names):
                F[i, j] = (as_matrix(fr[nm]) if nm in fr else np.eye(4)).ravel()
        total = sum(v["mass"] for v in meshes.values() if v["mass"]) if meshes else 0
        return dict(title=self.title, subtitle=self.subtitle, material=self.material, duration=self.duration,
                    n_frames=len(self.frames), parts=parts, meshes=meshes,
                    frames=base64.b64encode(F.tobytes()).decode(), series=self.series_, views=self.views,
                    mass_total=round(total, 1), three=THREE_VER)

    def html(self, three_src=None, orbit_src=None):
        d = self.data()
        page = open(TEMPLATE).read()
        page = page.replace("__TITLE__", self.title).replace("__DATA__", json.dumps(d).replace("</", "<\\/"))
        page = page.replace("__THREE_SRC__", three_src or f"https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js")
        page = page.replace("__ORBIT_SRC__", orbit_src or f"https://cdn.jsdelivr.net/npm/three@{THREE_VER}/examples/js/controls/OrbitControls.js")
        return page

    def write(self, path):
        page = self.html()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        open(path, "w").write(page)
        mb = len(page.encode()) / 1e6
        print(f"motion: {path} {mb:.2f} MB, {len(self.parts)} parts, {len(self.frames)} frames" + ("  TOO BIG for an artifact (15 MB): lower faces=" if mb > 15 else ""))
        return path

    def video(self, mp4_path, width=1280, height=720, fps=30, view="iso", gif=True, png=True):
        """mp4 + gif + png beside it, rendered headless from the same page."""
        return render_video(self, mp4_path, width, height, fps, view, gif, png)


# ------------------------------------------------------------------ headless render
def _three_files():
    """three.min.js + OrbitControls.js from npm (the CDNs are blocked in some cloud boxes); cached under ~/.cache/nox-motion."""
    cands = [os.environ.get("NOX_THREE_DIR"), os.path.join(os.path.dirname(TEMPLATE), "..", "node_modules", "three"),
             os.path.expanduser("~/.cache/nox-motion/node_modules/three")]
    for d in cands:
        if d and os.path.exists(os.path.join(d, "build", "three.min.js")):
            return os.path.join(d, "build", "three.min.js"), os.path.join(d, "examples", "js", "controls", "OrbitControls.js")
    cache = os.path.expanduser("~/.cache/nox-motion"); os.makedirs(cache, exist_ok=True)
    subprocess.run(["npm", "install", "--silent", "--no-audit", "--no-fund", "--prefix", cache, f"three@{THREE_VER}"], check=True)
    d = os.path.join(cache, "node_modules", "three")
    return os.path.join(d, "build", "three.min.js"), os.path.join(d, "examples", "js", "controls", "OrbitControls.js")


def render_video(scene, mp4_path, width=1280, height=720, fps=30, view="iso", gif=True, png=True):
    from playwright.sync_api import sync_playwright
    three, orbit = _three_files()
    page_html = scene.html(three_src="three.min.js", orbit_src="OrbitControls.js")
    stem = os.path.splitext(mp4_path)[0]
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(three, os.path.join(tmp, "three.min.js")); shutil.copy(orbit, os.path.join(tmp, "OrbitControls.js"))
        open(os.path.join(tmp, "page.html"), "w").write(page_html)
        fdir = os.path.join(tmp, "f"); os.makedirs(fdir)
        with sync_playwright() as pw:
            args = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox", "--allow-file-access-from-files"]
            try:
                br = pw.chromium.launch(headless=True, args=args)
            except Exception:
                br = pw.chromium.launch(headless=True, args=args, executable_path=os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium"))
            pg = br.new_page(viewport={"width": width, "height": height})
            errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto("file://" + os.path.join(tmp, "page.html"))
            pg.wait_for_function("window.__motion && window.__motion.ready", timeout=120000)
            pg.evaluate(f"window.__motion.record({width}, {height}, {json.dumps(view)})")
            n = pg.evaluate("window.__motion.n")
            step = max(1, round((n / scene.duration) / fps)) if scene.duration else 1
            k = 0
            for i in range(0, n, step):
                pg.evaluate(f"window.__motion.setFrame({i})")
                pg.screenshot(path=os.path.join(fdir, f"{k:05d}.png")); k += 1
            br.close()
            if errs:
                raise SystemExit("motion page errors while rendering:\n" + "\n".join(errs))
        out_fps = max(1, round(k / scene.duration)) if scene.duration else fps
        os.makedirs(os.path.dirname(os.path.abspath(mp4_path)), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(out_fps), "-i", os.path.join(fdir, "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-vf", f"scale={width}:-2", mp4_path], check=True)
        if gif:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(out_fps), "-i", os.path.join(fdir, "%05d.png"),
                            "-vf", "fps=15,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", stem + ".gif"], check=True)
        if png:
            shutil.copy(os.path.join(fdir, "00000.png"), stem + ".png")
    for ext in (".mp4", ".gif", ".png"):
        f = stem + ext
        if os.path.exists(f):
            print(f"motion: {f} {os.path.getsize(f) / 1e6:.2f} MB")
    return mp4_path


# ------------------------------------------------------------------ CLI
def from_spec(spec):
    s = Scene(spec.get("title", "Nox motion"), spec.get("subtitle", ""), spec.get("material", "PETG"))
    base = os.path.dirname(os.path.abspath(spec.get("_path", ".")))
    for p in spec["parts"]:
        m = trimesh.load(os.path.join(base, p["file"]) if not os.path.isabs(p["file"]) else p["file"], force="mesh")
        s.part(p["name"], m, group=p.get("group", "part"), color=p.get("color"), printed=p.get("printed", True),
               faces=p.get("faces", 30000), note=p.get("note", ""))
    s.timeline(spec.get("frames", [{}]), spec.get("duration_s", max(1.0, len(spec.get("frames", [1])) / 30)))
    for se in spec.get("series", []):
        s.series(se["name"], se["values"], se.get("unit", ""), se.get("bad_below"), se.get("bad_above"))
    for nm, v in spec.get("views", {}).items():
        s.view(nm, v["eye"], v.get("target", (0, 0, 0)))
    return s


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    spec = json.load(open(sys.argv[1])); spec["_path"] = sys.argv[1]
    s = from_spec(spec)
    out = spec.get("out", os.path.splitext(sys.argv[1])[0])
    s.write(out + ".html")
    if "--video" in sys.argv:
        s.video(out + ".mp4")
