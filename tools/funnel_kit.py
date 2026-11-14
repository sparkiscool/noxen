"""The landing funnel's print kit as exact files (14 Nov; the measure of a print pose is the file on its bed, not a sentence).
Every printed piece is written to print/<name>.stl IN ITS PRINT POSE (bed at z 0, centred on the 180 plate) and measured there:
bed size, the flat it lays on the bed, its overhang past 45 deg, the solid check on that exact file, a picture of it on its bed
(parts/box_funnel/kit/<name>.png: bed contact blue, overhang red). Engine pieces come from their run's print mesh
(tools/print_mesh.py local), drawn pieces from parts/box_funnel/make.py. Writes parts/box_funnel/kit.json; the box page's print
table (tools/import_box.py) is that file. A piece under its BED_MIN or off the plate fails the kit (exit 1).

    python3 tools/funnel_kit.py [--no-gate] [piece ...]     --no-gate: skip the solid checks (minutes and GBs on a half)
"""
import hashlib, importlib.util, json, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
PLATE = 180.0
BRIM = 5.0
PCTG = "PCTG (PETG settings + 10 C)"
UP = {"+z": np.eye(3), "-z": np.diag([1.0, -1.0, -1.0]),                      # proper rotations taking the build direction to bed +z
      "+x": np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]), "-x": np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]]),
      "+y": np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]), "-y": np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])}
POSES = dict(UP)                                                              # the end walls' candidates: the six, and lying on the 55 deg face


def fm():
    sp = importlib.util.spec_from_file_location("box_funnel_make", os.path.join(ROOT, "parts", "box_funnel", "make.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def tri(man):
    mm = man.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def posed(m, R, mirror_y=False):
    """the mesh turned so R's build direction is up, set on the bed (z 0) at the plate's centre"""
    t = m.copy()
    if mirror_y:
        t.vertices = t.vertices * [1.0, -1.0, 1.0]; t.faces = t.faces[:, ::-1]
    T = np.eye(4); T[:3, :3] = R; t.apply_transform(T)
    lo, hi = t.bounds; t.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return t


def measure(t):
    """on the bed: size, the flat on the bed, the overhang past 45 deg (not on the bed), the bed contact's own extent (a brim goes round it)"""
    n, c, A = t.face_normals, t.triangles_center, t.area_faces
    bed = (n[:, 2] < -0.99) & (c[:, 2] < 0.05)
    over = (n[:, 2] < -np.cos(np.radians(45.0))) & ~bed
    ext = t.extents
    out = dict(bed_mm=[round(float(v), 1) for v in ext], bed_contact_mm2=round(float(A[bed].sum())), overhang_mm2=round(float(A[over].sum())),
               volume_cm3=round(float(t.volume) / 1000, 1))
    if bed.any():
        v = t.triangles[bed].reshape(-1, 3); w = v[:, :2].max(0) - v[:, :2].min(0)
        out["contact_extent_mm"] = [round(float(w[0]), 1), round(float(w[1]), 1)]
        out["brim_mm"] = BRIM if (w + 2 * BRIM <= PLATE).all() else 0.0
        out["overhang_over_bed_mm"] = [round(float(c[over, 2].min()), 1), round(float(c[over, 2].max()), 1)] if over.any() else None
    out["fits_180"] = bool((ext[:2] <= PLATE).all() and ext[2] <= PLATE)
    return out


def on_face(F):
    """the rotation that lays an end wall on its 55 deg sliding face (the nose wall's face normal is (-sin 55, 0, cos 55) in the seat frame)"""
    a = np.radians(F.FACE_DEG); n = np.array([-np.sin(a), 0.0, np.cos(a)])       # the nose wall's inner face normal, up and inboard
    z = -n; y = np.array([0.0, 1.0, 0.0]); x = np.cross(y, z)                 # bed z is -n: the face down
    return np.array([x, y, z])


def best_wall_pose(m, F, sx):
    """every candidate pose of an end wall, measured; the one with the least overhang that lays over 200 mm2 on the bed. (14 Nov: the
    page said "on its back, the sliding face up". Measured, neither the back nor the face lies flat: the corner hips fold out of
    the face's plane and the key bosses stand on the back, 0 mm2 either way. It stands on its foot.)"""
    R55 = on_face(F) if sx > 0 else on_face(F) @ np.diag([-1.0, -1.0, 1.0])     # the tail wall is the nose wall turned half round z
    names = {"+z": "standing on its foot, as it sits in the funnel", "-z": "upside down on its top edge", "+x": "on its x- side", "-x": "on its x+ side", "+y": "standing on its -y end", "-y": "standing on its +y end"}
    cand = dict({names[k]: R for k, R in POSES.items()}, **{"on its 55 deg sliding face": R55, "on its back (the sliding face up)": np.diag([1.0, -1.0, -1.0]) @ R55})
    rows = {k: measure(posed(m, R)) for k, R in cand.items()}
    ok = [k for k in rows if rows[k]["bed_contact_mm2"] >= 200 and rows[k]["fits_180"]] or list(rows)
    k = min(ok, key=lambda q: rows[q]["overhang_mm2"])
    return k, cand[k], {q: dict(bed_contact_mm2=r["bed_contact_mm2"], overhang_mm2=r["overhang_mm2"]) for q, r in rows.items()}


def gate(path, loose=False):
    """tools/solid_check.py on the file as it lies; a plate of loose pieces (the strips) body by body"""
    from solid_check import solid_check
    m = trimesh.load(path, force="mesh"); why = {}
    for k, b in enumerate(m.split(only_watertight=False) if loose else [m]):
        ok, info = solid_check(b)
        if not ok:
            why[f"body {k}"] = {q: info[q] for q in ("necks", "thin_walls", "bodies_eroded", "sealed_voids") if info.get(q)}
    return ("FAIL" if why else "PASS"), why


def picture(t, path, title, info):
    from topo.build import render
    os.makedirs(os.path.dirname(path), exist_ok=True); render(t, path, title=title, info=info)


def pieces(only=()):
    """name -> (mesh in the seat frame or as built, rotation, mirror, count, material, pose words, supports words, BED_MIN mm2, source)"""
    import print_mesh as pm
    F = fm(); out = {}; want = lambda n: not only or n in only                # an engine piece's mesh is fetched only when asked for
    def engine(part):
        r = json.load(open(os.path.join(ROOT, "parts", part, "result.json")))
        return trimesh.load(pm.local(part), force="mesh"), dict(part=part, part_sha=r.get("part_sha"), print_stl_sha256=(r.get("print_stl") or {}).get("sha256"),
                                                                 mass_g=r.get("mass_g"), worst=max(max(x["inlayer"], x["across"]) for x in r["strength"]["rows"]), solid_ok=r.get("solid_ok"))
    for nm in ("nose", "tail"):
        m, src = engine(f"box_funnel_half_{nm}") if want(f"box_funnel_half_{nm}") else (None, None)
        out[f"box_funnel_half_{nm}"] = (m, UP["-z"], False, 1, PCTG, "upside down: the two rims behind the side walls' top edges and the rib's top on the bed",
                                        f"yes, everywhere over 45 deg: under the beam (about 42 over the bed, to the plate's edge), the motor seats and the ledges. The ledges' plastic "
                                        f"face stands {F.LIP} under the seat plane, so a scar there never touches a drone; scrape the strips' pocket floors flat before gluing the strips",
                                        1200.0, src)
    m, src = engine("box_funnel_arm") if want("box_funnel_arm") or want("box_funnel_arm_mirrored") else (None, {})
    for nm, mir in (("box_funnel_arm", False), ("box_funnel_arm_mirrored", True)):
        out[nm] = (m, UP["-x"], mir, 2, PCTG, "on its outer face (the side away from the half's beam), the lap's four bolt heads' face down",
                   None,
                   300.0, dict(src, corners="nose +y and tail -y" if not mir else "nose -y and tail +y (the mirror image)"))
    for nm, sx in (("nose", 1), ("tail", -1)):
        m = tri(F.carriage(sx, with_pins=False)); k, R, rows = best_wall_pose(m, F, sx)
        out[f"box_funnel_wall_{nm}"] = (m, R, False, 1, PCTG, k, None, 200.0, dict(drawn="parts/box_funnel/make.py carriage(%+d)" % sx, poses_measured=rows))
    bodies = sorted(F.strips().decompose(), key=lambda b: (b.bounding_box()[1], b.bounding_box()[0])); parts = []
    for i, b in enumerate(bodies):                                               # the six strips side by side, 4 apart, flat as they lie in the ledges
        t = tri(b); lo, hi = t.bounds; t.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1] + i * (hi[1] - lo[1] + 4.0), -lo[2]]); parts.append(t)
    out["box_funnel_strips"] = (trimesh.util.concatenate(parts), UP["+z"], False, 1, "TPU 95A", "flat, the six side by side (one plate)", "none", 300.0,
                                dict(drawn="parts/box_funnel/make.py strips()", lengths_mm=[round(float(p.extents[0]), 1) for p in parts], section_mm=[round(float(parts[0].extents[1]), 1), round(float(parts[0].extents[2]), 1)]))
    return out


def main(only=(), do_gate=True):
    kit_path = os.path.join(ROOT, "parts", "box_funnel", "kit.json")
    old = {p["name"]: p for p in json.load(open(kit_path))["pieces"]} if os.path.exists(kit_path) else {}
    rows, bad = [], []
    P = pieces(only)
    for name, (m, R, mir, count, mat, pose, sup, bed_min, src) in P.items():
        if only and name not in only:
            if name in old:
                rows.append(old[name])
            continue
        t = posed(m, R, mir); f = os.path.join(ROOT, "print", name + ".stl"); os.makedirs(os.path.dirname(f), exist_ok=True); t.export(f)
        ms = measure(t)
        if sup is None:                                                           # from the measured overhang
            sup = "none" if ms["overhang_mm2"] < 30 else f"yes: {ms['overhang_mm2']} mm2 over 45 deg, {ms['overhang_over_bed_mm'][0]} to {ms['overhang_over_bed_mm'][1]} over the bed"
        row = dict(name=name, file=f"print/{name}.stl", sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(), count=count, material=mat, pose=pose, supports=sup,
                   bed_min_mm2=bed_min, **ms, source=src)
        row["solid_check"], why = gate(f, loose=name.endswith("strips")) if do_gate else ((old.get(name) or {}).get("solid_check") if (old.get(name) or {}).get("sha256") == row["sha256"] else "not run", {})
        if why:
            row["solid_fail"] = why
        pic = f"parts/box_funnel/kit/{name}.png"
        picture(t, os.path.join(ROOT, pic), f"{name} on its bed, x {count} ({mat})", f"{ms['bed_mm'][0]} x {ms['bed_mm'][1]} x {ms['bed_mm'][2]} | on the bed {ms['bed_contact_mm2']} mm2 (blue) | overhang {ms['overhang_mm2']} mm2 (red) | solid check {row['solid_check']}")
        row["picture"] = pic
        row["ok"] = bool(ms["fits_180"] and ms["bed_contact_mm2"] >= bed_min and row["solid_check"] == "PASS")
        if not row["ok"]:
            bad.append(name)
        rows.append(row)
        print(f"{name:28} x{count} {ms['bed_mm']} bed {ms['bed_contact_mm2']} mm2 (min {bed_min:.0f}) overhang {ms['overhang_mm2']} solid {row['solid_check']} {'ok' if row['ok'] else 'NOT READY'}", flush=True)
    json.dump(dict(what="the landing funnel's print kit: each file is in its print pose and was measured and gated as it lies (tools/funnel_kit.py)",
                   printer="Bambu A1 mini, 180 x 180 x 180", bought="tools/import_box.py BOUGHT_LIST and FASTENERS (the page's Bought table): gearmotors, switches, probes and sleeves, the four hold-down keys",
                   not_in_the_kit="the stand's mast (15 longer): not drawn", ok=not bad and len(rows) == len(P), not_ready=bad, pieces=rows), open(kit_path, "w"), indent=1)
    return not bad


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    sys.exit(0 if main(a, do_gate="--no-gate" not in sys.argv) else 1)
