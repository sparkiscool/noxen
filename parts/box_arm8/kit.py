"""Swap arm v8 print kit (14 Nov 2026, Spark: "lets keep working on everything i want to start printing soon").
Every printed piece of the arm as the exact file to put on the bed, in its print pose, measured on that file:

    python3 parts/box_arm8/kit.py [--only name,name]   -> print/box_arm8_*.stl (meshes.yml publishes them as print-<name>.stl),
                                                          parts/box_arm8/kit/*.png, parts/box_arm8/kit.json

Each file: solid check (tools/solid_check.py, the gate) on the file as exported; the first layer's area; the area that overhangs past
45 deg and the part of it wider than BRIDGE_MM (that part needs supports: the decision comes from this number, not from words);
its size on the A1 mini's 180 bed and the brim that still fits. A pose is fixed here only where the layers' direction carries the
load (said beside it); otherwise it is the pose with the least to support, then the most on the bed.
The three engine parts (hub, carriage, lift) join from their runs' print meshes (tools/print_mesh.py), turned so their setup's build
direction is up; one whose result is not of the setup on main is listed as waiting. The install sweeps for one grab unit (every piece
along its way in, against the body and what is already in) run here too: a piece that cannot be put in fails the kit.
"""
import json, math, os, sys
import numpy as np, trimesh
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "parts")); sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
import importlib.util
_sp = importlib.util.spec_from_file_location("box_arm8_make", os.path.join(HERE, "make.py")); M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(M)
L = M.L
from box_arm8_lib import U, vol, box, cyl

BED = 180.0
BRIDGE_MM = 6.0          # an overhang no wider than this bridges or is a short lip: no supports for it
PITCH = 0.25
LOW_MM = 0.75            # an engine part's generated surface stops 0.25 to 0.5 short of the room's end plane (the bed): measured apart, as low_gap_mm2
REL = "https://github.com/sparkiscool/noxen/releases/download/meshes/print-"
DOWN = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def on_bed(t, down):
    """the mesh turned so its `down` direction points into the bed, standing on z 0, turned about z to the smallest square it fits, centred"""
    d = np.array(DOWN[down], float); a = np.cross(d, [0, 0, -1.0]); c = float(d @ [0, 0, -1.0])
    if np.linalg.norm(a) < 1e-9: R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]); R = np.eye(3) + K + K @ K / (1 + c)
    T = np.eye(4); T[:3, :3] = R; t = t.copy(); t.apply_transform(T)
    t.update_faces(t.nondegenerate_faces(height=1e-6)); t.remove_unreferenced_vertices()        # a zero-area sliver where two faces meet 0.1 apart is no surface (the unit body had one)
    xy = t.convex_hull.vertices[:, :2]; best = None
    for th in np.arange(0.0, 180.0, 0.5):
        cs, sn = math.cos(math.radians(th)), math.sin(math.radians(th)); q = xy @ np.array([[cs, -sn], [sn, cs]]).T
        e = q.max(0) - q.min(0); key = (round(max(e), 2), 0 if th in (0.0, 90.0) else 1, th)
        if e[0] >= e[1] - 1e-9 and (best is None or key < best[0]): best = (key, th)
    th = best[1]
    if max(t.extents[:2]) <= BED - 10.0: th = 0.0 if t.extents[0] >= t.extents[1] else 90.0          # it fits square with a brim: keep it square
    Rz = trimesh.transformations.rotation_matrix(math.radians(th), [0, 0, 1]); t.apply_transform(Rz)
    lo, hi = t.bounds; t.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return t, th


def measure(t):
    """on the file as it lies on the bed (z up): first layer mm2, overhang mm2 past 45 deg, the part of it wider than BRIDGE_MM, its widest span, and where"""
    import solid_check as sc
    lo = t.bounds[0] - 2 * PITCH; hi = t.bounds[1] + 2 * PITCH; shape = tuple(int(np.ceil((hi[k] - lo[k]) / PITCH)) for k in range(3))
    V = sc.voxels(t, PITCH, lo, shape)
    k0 = int(np.argmax(V.any(axis=(0, 1))))
    first = float(V[:, :, k0].sum()) * PITCH ** 2
    st = np.ones((3, 3), bool); over = 0.0; need = 0.0; span = 0.0; where = None; low = 0.0
    for k in range(k0 + 1, V.shape[2]):
        cur = V[:, :, k]
        if not cur.any(): continue
        u = cur & ~ndi.binary_dilation(V[:, :, k - 1], st)             # nothing under it within 45 deg
        n = int(u.sum())
        if not n: continue
        if (k - k0) * PITCH <= LOW_MM:                                     # a face up to 0.75 above the bed: no support fits under it, it prints as a rough underside
            low += n * PITCH ** 2; continue
        over += n * PITCH ** 2
        d = ndi.distance_transform_edt(np.pad(u, 1))[1:-1, 1:-1] * PITCH * 2
        lab, nl = ndi.label(u, st)
        if nl:
            w = ndi.maximum(d, lab, range(1, nl + 1)); a = ndi.sum(u, lab, range(1, nl + 1)) * PITCH ** 2
            for i in range(nl):
                if w[i] > BRIDGE_MM:
                    need += a[i]
                    if w[i] > span:
                        span = float(w[i]); c = ndi.center_of_mass(lab == i + 1); where = [round(float(lo[0] + c[0] * PITCH), 1), round(float(lo[1] + c[1] * PITCH), 1), round(float(lo[2] + k * PITCH), 1)]
    return dict(first_layer_mm2=round(first), low_gap_mm2=round(low), overhang_mm2=round(over), support_mm2=round(need), widest_overhang_mm=round(span, 1), widest_at=where)


def all_poses(t):
    out = {}
    for dn in DOWN:
        q, th = on_bed(t, dn); r = measure(q); e = q.extents
        r.update(fits=bool(max(e[:2]) <= BED and e[2] <= BED), size=[round(float(v), 1) for v in e], turned=th); out[dn] = r
    return out


def pieces():
    """name -> (manifold, material, count for the whole arm, pose or None, why that pose, set)"""
    S = M.sets(); FX = M.fixed_parts(); f = L.faces()["grip"]; body, pads = L.unit_body()
    bb = [q.bounding_box() for q in pads]; pa = next(b for b in bb if b[3] - b[0] > 4.0); pb = next(b for b in bb if b[3] - b[0] <= 4.0)
    P = {
        "unit_body": (body, "PCTG", 2, "-y", "base plate on the bed: the deck's seats and the rod bores stand as drawn", "grab unit"),
        "blade": (L.blade(1, f), "PCTG", 4, "-z", "on its end: the hook's pull bends the blade along its length, in the layers (flat on its face it stands on 119 mm2 and needs supports under the hook)", "grab unit"),
        "pilot": (L.pilot(1, 1), "PCTG", 4, None, "", "grab unit"),
        "paddle": (L.paddle("grip"), "PCTG", 2, "-z", "cam end on the bed, the spline bore up: round and true", "grab unit"),
        "rail": (M.head_parts("grip")["rail"][0], "PCTG", 2, "+y", "the ridge side on the bed, keel up: both flanks of each ridge print alike", "hub side"),
        "wheel": (L.wheel(0, -L.Z0), "PCTG", 6, None, "", "carriage"),
        "tie": (L.tie(L.FLOAT_Z[0]), "PCTG", 4, "-z", "bar on the bed, the rod bores lying: the bar bends in the layers", "lift"),
        "stub": (FX["stub"][0], "PCTG", 1, None, "", "stub side"), "post": (FX["post"][0], "PCTG", 1, None, "", "stub side"),
        "detent_arm": (FX["detent_arm"][0], "PCTG", 1, None, "", "stub side"),
        "saddle_overseer": (M.saddle(S, "overseer"), "PCTG", 4, None, "", "ring"), "saddle_scout": (M.saddle(S, "scout"), "PCTG", 7, None, "", "ring"),
        "link": (M.link_bar(S), "PCTG", 20, None, "", "ring")}
    for nm, b in (("pad_a", pa), ("pad_b", pb)):
        for th in L.PAD_STEPS:
            P[f"{nm}_{int(round(th * 10))}"] = (box(b[0], b[3], -th, 0.0, b[2], b[5]), "TPU 95A", 8 if th == L.PAD_T else 4, "+y",
                                                 "top face on the bed (smooth to the pack)" + ("" if th == L.PAD_T else "; a set-up spare: print once"), "grab unit")
    return P


def install():
    """one grab unit, every piece along its way in (swept out, 1 mm steps) against the body and what is in before it: mm3, 0 = it goes in"""
    body, pads = L.unit_body(); f = L.faces(); ub = L.unit_bought("grip")
    bl = L.blade(1, f["grip"]) + L.blade(-1, f["grip"])
    def path(m, fixed, moves):
        worst = 0.0; cur = m
        for d, n in moves:
            for i in range(int(n) + 1): worst = max(worst, vol(cur.translate([d[0] * i, d[1] * i, d[2] * i]) ^ fixed))
            cur = cur.translate([d[0] * n, d[1] * n, d[2] * n])
        return round(worst, 3)
    pins = [L.pilot_pin(sx, sz) for sx, sz in L.PILOTS]; pil = [L.pilot(sx, sz) for sx, sz in L.PILOTS]
    steps = [
        dict(step="1 servo in from the open side (+x), its spline through the slot in the wall; two M3 x 8 through the inner ear from the far end (driver through the end bar's two holes)",
             mm3=max(path(ub["servo"], body, [((1, 0, 0), 90)]), path(U(*[cyl("z", L.SERVO_EAR["x"][0], L.CAM_Y + sy * 5.0, 2.4, L.SERVO_EAR["z"] + 2.5, 160) for sy in (-1, 1)]), body + ub["servo"], [((0, 0, 1), 1)]))),
        dict(step="2 paddle down between the walls 4.5 short of the spline, slid onto it, one M3 x 35 through it from the hub end (driver through the o 6.4 hole)",
             mm3=max(path(ub["paddle"], body + ub["servo"], [((0, 0, -1), 4.5), ((0, 1, 0), 60)]), path(cyl("z", 0, L.CAM_Y, 2.8, -120, -6.0), body, [((0, 0, -1), 40)]))),
        dict(step="3 each pilot: two o 3 springs into its pocket, the pilot down on them, pressed 1 mm, its o 3 x 9 pin pushed in from the blade bay",
             mm3=max([path(p, body, [((0, 1, 0), 50)]) for p in pil] + [path(q, body + ub["paddle"], [((0, 0, -sz), 11)]) for q, (sx, sz) in zip(pins, L.PILOTS)])),
        dict(step="4 each blade down into its bay at the grip position (paddle turned along the bay), its two springs between it and the wall",
             mm3=max(path(L.blade(s_, f["grip"]), body + ub["paddle"] + ub["servo"] + U(*pil) + U(*pins), [((0, 1, 0), 70)]) for s_ in (1, -1))),
        dict(step="5 the two o 3 x 66 rods in from one side through wall, springs and both blades; a drop of CA at each end", mm3=path(ub["blade_rods"], body, [((1, 0, 0), 80)])),
        dict(step="6 the eight pads into their seats (2.4 first), CA", mm3=path(U(*pads), body, [((0, 1, 0), 5)])),
    ]
    q = L.paddle("grip").translate([0, -L.CAM_Y, 0])
    turn = round(max(vol(q.rotate([0, 0, a]).translate([0, L.CAM_Y, 0]) ^ (body + ub["servo"])) for a in range(0, 91, 5)), 3)
    work = dict(paddle_turning_mm3=turn,
                blades_grip_open_limit_mm3=[round(vol((L.blade(1, f[s_]) + L.blade(-1, f[s_])) ^ (body + L.unit_bought("grip" if s_ == "limit" else s_)["paddle"] + ub["servo"] + U(*pil))), 3) for s_ in ("grip", "open", "limit")],
                pilots_down_mm3=round(max(vol(p.translate([0, -L.PILOT_TRAVEL, 0]) ^ (body + bl + U(*pins))) for p in pil), 3),
                pilots_held_by_pin_mm3=round(min(vol(p.translate([0, 0.5, 0]) ^ U(*pins)) for p in pil), 3))
    ok = all(s_["mm3"] < 0.05 for s_ in steps) and turn < 0.05 and max(work["blades_grip_open_limit_mm3"]) < 0.05 and work["pilots_down_mm3"] < 0.05 and work["pilots_held_by_pin_mm3"] > 0.5
    return dict(steps=steps, working=work, ok=bool(ok))


def engine_parts():
    """hub, carriage, lift: (mesh or None, down, count, why waiting)"""
    import part_sha, print_mesh
    out = {}
    for k, n in (("hub", 1), ("carriage", 2), ("lift", 2)):
        p = "box_arm8_" + k; d = os.path.join(ROOT, "parts", p); r = json.load(open(os.path.join(d, "result.json")))
        setup = getattr(__import__("box_arm8_setups"), k)
        import re
        build = re.search(r'build="([+-][xyz])"', open(os.path.join(ROOT, "parts", "box_arm8_setups.py")).read().split(f"def {k}(h)")[1]).group(1)
        down = ("-" if build[0] == "+" else "+") + build[1]
        now = part_sha.setup_sha(os.path.join(d, "part.py"))
        cur = M.current(k)
        why = None if cur else ("its run is not of the setup on main (%s, the result %s): rerunning" % (now, r.get("setup_sha")) if r.get("setup_sha") != now else "its run did not pass")
        out[k] = dict(part=p, n=n, down=down, build=build, current=cur, why=why, worst=r["strength"]["worst"], mass_g=r["mass_g"],
                      mesh=(trimesh.load(print_mesh.local(p), force="mesh") if cur else None))
    return out


def one_file(fn, t0, mat, n, pose, why, group, pd, kd, poses=None, extra=None):
    import solid_check as sc
    from topo.build import render
    if pose is None:
        ps = poses or all_poses(t0); ok = [k for k in ps if ps[k]["fits"]] or list(ps)
        lo = min(ps[k]["support_mm2"] for k in ok)
        pose = max((k for k in ok if ps[k]["support_mm2"] <= lo + 5.0), key=lambda k: ps[k]["first_layer_mm2"]); why = why or "least to support, then most on the bed, of the six flat poses"
    t, th = on_bed(t0, pose); f = os.path.join(pd, fn + ".stl"); t.export(f)
    g = trimesh.load(f, force="mesh"); ok, info = sc.solid_check(g)
    ok = bool(ok and g.is_watertight and len(g.split(only_watertight=False)) == 1)                # the file itself: closed, one shell
    ms = measure(g); e = [round(float(v), 1) for v in g.extents]; spare = round((BED - max(e[:2])) / 2, 1)
    fits = bool(max(e[:2]) <= BED and e[2] <= BED)
    sup = ms["support_mm2"] >= 5.0
    png = os.path.join(kd, fn + ".png")
    render(g, png, title=f"{fn}.stl on its bed", info=f"{mat} | {e[0]} x {e[1]} x {e[2]} mm | first layer {ms['first_layer_mm2']} mm2 | supports {'YES, ' + str(ms['support_mm2']) + ' mm2' if sup else 'none'} | solid check {'PASS' if ok else 'FAIL'}")
    d = dict(file=f"print/{fn}.stl", release=REL + fn + ".stl", picture=os.path.relpath(png, ROOT), set=group, material=mat, count=n, solid_check="PASS" if ok else "FAIL", watertight=bool(g.is_watertight),
             thin_walls=len(info.get("thin_walls", [])), pose=f"the part's {pose} side on the bed" + (f", turned {th:g} deg on the plate" if th not in (0.0, 90.0) else ""), pose_why=why,
             size_on_bed_mm=e, fits_bed=fits, brim=("a 5 mm brim fits" if spare >= 5.0 else f"NO room for a 5 mm brim: {spare} mm to the bed's edge each side, print without brim or skirt") if fits else "DOES NOT FIT",
             first_layer_mm2=ms["first_layer_mm2"], low_gap_mm2=ms["low_gap_mm2"], overhang_mm2=ms["overhang_mm2"], support_mm2=ms["support_mm2"], widest_overhang_mm=ms["widest_overhang_mm"], widest_at=ms["widest_at"],
             supports=("yes: %d mm2 of overhang wider than %g mm (the widest %.1f); tree supports from the bed" % (ms["support_mm2"], BRIDGE_MM, ms["widest_overhang_mm"])) if sup else
                      ("none: %d mm2 past 45 deg, none of it wider than %g mm" % (ms["overhang_mm2"], BRIDGE_MM)),
             bed_note=("%d mm2 of the underside lies 0.25 to 0.5 above the bed beside the first layer (the engine's surface stops short of the bed plane): it prints as a rough, slightly sagged face; the drawn seats are on the bed" % ms["low_gap_mm2"]) if ms["low_gap_mm2"] >= 20 else "",
             grams_each=round(abs(g.volume) / 1000 * (1.21 if mat.startswith("TPU") else 1.23), 2))
    if extra: d.update(extra)
    return d


TEST_SET = ["unit_body", "blade", "pilot", "paddle", "pad_a_24", "pad_b_24"]
ORDER = [("1 grab-unit test set, one head (try it on the Scout cartridge)", TEST_SET, {"unit_body": 1, "blade": 2, "pilot": 2, "paddle": 1, "pad_a_24": 4, "pad_b_24": 4}),
         ("2 pad set-up spares, once (pick per pad at set-up)", [f"pad_{a}_{t}" for a in "ab" for t in (20, 22, 26, 28)], None),
         ("3 one head's slide: carriage, lift, rail, three wheels, two ties (proves the reach and the push on the bench before the hub is printed)", ["carriage", "lift", "rail", "wheel", "tie"], {"carriage": 1, "lift": 1, "rail": 1, "wheel": 3, "tie": 2}),
         ("4 stub side and hub (the longest print)", ["stub", "post", "detent_arm", "hub"], None),
         ("5 the second head: everything of 1 and 3 again", ["unit_body", "blade", "pilot", "paddle", "pad_a_24", "pad_b_24", "carriage", "lift", "rail", "wheel", "tie"], None),
         ("6 the ring: saddles and links, with the bays", ["saddle_scout", "saddle_overseer", "link"], None)]

A = "https://www.amazon.com/"
TEST_BOUGHT = [
    dict(part="grab servo", n=1, spec="DS3218, the 180 deg version (on hand)", url=A + "DS3218-Digital-Servo-Torque-Waterproof/dp/B07TKTQ2NZ",
         check="E: the spline's tip 6.0 out of the case top and the ears 2.5 thick (not in the sheet). Measure both: the paddle's nose and the ear tab follow"),
    dict(part="blade rods and pilot pins", n=1, spec="o 3 carbon rod (on hand): two 66 long, two 9 long", url=A + "Carbon-Fiber-Length-Diameter-Suitable/dp/B09N91J1FC"),
    dict(part="blade spring", n=4, spec="o 6 x 0.7 wire x 25, two per blade", url=A + "uxcell-0-7mmx6mmx25mm-Stainless-Compression-Springs/dp/B076LRJH99", check="on a scale: 12.5 N at 17.1 long"),
    dict(part="pilot spring", n=4, spec="the latch's spring (o 3.0, 0.3 wire, 30 free) cut to 15 long, two side by side under each pilot (they were one o 6 x 20, which has no way into the pocket)",
         url=A + "PATIKIL-Compression-Stainless-Mechanical-Assortment/dp/B0DKCRCBV2", check="E 0.25 N/mm each cut to 15: about 2 N a pilot at rest, 5 N flush. The pilot must come back up by itself"),
    dict(part="paddle screw", n=1, spec="M3 x 35 socket head", url=A + "Socket-Screws-Available-Stainless-Machine/dp/B08P6DYH7V", check="the M3 x 12 listing's other lengths (M3 x 35 not confirmed on it: any M3 x 35 does). E: it reaches 7 into the servo's shaft; if it bottoms before the paddle is tight, washers under its head"),
    dict(part="servo ear screws", n=2, spec="M3 x 8 socket head, into the tab's o 2.5 holes (they cut their thread)", url=A + "Socket-Screws-Available-Stainless-Machine/dp/B08P6DYH7V"),
    dict(part="CA glue", n=1, spec="thin cyanoacrylate: rod ends, pilot pins, pads", url=A + "s?k=thin+CA+glue", check="any; on hand"),
]
TEST_STEPS = ["Trial fit dry first: the servo must slide in from the open side with its spline in the slot, and the paddle's o 5.8 bore must press onto the spline by the screw (it cuts its own teeth; E).",
              "1 Servo in from the open side; two M3 x 8 through its inner ear into the tab, driver through the two holes in the end bar. The outer ear stays free.",
              "2 Paddle: lower it between the walls, slide it onto the spline, M3 x 35 through it from the hub end (driver through the o 6.4 hole). Servo at grip: paddle along the bay.",
              "3 Each pilot: two cut springs into the pocket, pilot on them, press 1 mm, push the o 3 x 9 pin in from the blade bay. It must sink 5 and come back.",
              "4 Each blade down into its bay with its two springs against the wall; then both o 3 x 66 rods through wall, springs and blades. CA at the rod ends when it works.",
              "5 Pads: the 2.4 ones into their seats, dry for now.",
              "Try: turn the paddle a quarter turn by hand or with the servo: the blades open to 22.6 between faces (grip 12.4). Set the unit under the Scout cartridge (pilots into the recess corners), let the blades close on the tabs: both bolts must draw in and the cartridge lift out of its receiver with the unit; open: it must stay behind latched.",
              "Report: does the paddle hold on the spline, do the blades reach the tabs' stop (bolts fully in), does the cartridge sit on all four Scout pads with the blades closed (if not, which pad and by how much: that picks the pad thickness)."]
OPEN = [dict(item="MGN12H rail hole positions", state="E until the rail is here",
             what="the carriage's four M3 inserts are 25 apart from 12.5 in from the rail's end (a 100 rail with four holes can only be 12.5 / 25 / 25 / 25 / 12.5 if it is cut evenly). Measure the first hole from each end and the pitch before printing the carriage",
             if_off="parts/box_arm8_lib.py MGN first / pitch, and the carriage reruns (its rail seat and loads follow)"),
        dict(item="float spring's rate", state="E 5.8 N/mm from wire and coil count",
             what="on a kitchen scale: 18 N at 11.9 long (the preload), about 32 N at 9.4", if_off="the push limit (118 N) and the float's preload follow the rate: parts/box_arm8_lib.py K_FLOAT, P_FLOAT; no printed part changes unless the limit leaves 97..125 N"),
        dict(item="M4 inserts", state="E: the ties' holes are o 5.6", what="measure the insert's body before printing the ties (o 5.6 suits a 6.0 knurl, 5.0 long)", if_off="parts/box_arm8_lib.py tie(): the hole; the ties are drawn pieces, no run")]


def main():
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    pd, kd = os.path.join(ROOT, "print"), os.path.join(HERE, "kit"); os.makedirs(pd, exist_ok=True); os.makedirs(kd, exist_ok=True)
    kj = os.path.join(HERE, "kit.json"); old = {q["file"].split("box_arm8_")[1][:-4]: q for q in json.load(open(kj)).get("files", [])} if os.path.exists(kj) else {}
    ins = install(); print("install", ins["ok"], [s_["mm3"] for s_ in ins["steps"]], ins["working"])
    files = {}; waiting = []
    for k, (m, mat, n, pose, why, group) in pieces().items():
        if only and k not in only:
            if k in old: files[k] = old[k]
            continue
        files[k] = one_file("box_arm8_" + k, M.tm(m), mat, n, pose, why, group, pd, kd)
    for k, e in engine_parts().items():
        if not e["current"]:
            waiting.append(dict(part=e["part"], count=e["n"], why=e["why"])); continue
        if only and k not in only and k in old:
            files[k] = old[k]; continue
        files[k] = one_file("box_arm8_" + k, e["mesh"], "PCTG", e["n"], e["down"], f"the setup's build direction ({e['build']}): the layers its strength was checked with", "engine part", pd, kd,
                            extra=dict(worst=e["worst"], run_mass_g=e["mass_g"]))
    for k, q in files.items(): print(k, q["solid_check"], q["pose"], q["size_on_bed_mm"], "first", q["first_layer_mm2"], "low", q.get("low_gap_mm2"), "over", q["overhang_mm2"], "sup", q["support_mm2"], q["widest_overhang_mm"], q["widest_at"], q["brim"][:12], q["grams_each"], "thin", q["thin_walls"])
    order = [dict(step=nm, files=[f"print/box_arm8_{k}.stl" for k in ks if k in files], waiting=[k for k in ks if k not in files], counts=cnt) for nm, ks, cnt in ORDER]
    out = dict(note="Bambu A1 mini, PCTG (PETG settings + 10 C) unless TPU. Counts are for the whole arm (two heads); the test set's own counts are in test_set. Nothing here is handed: "
                    "both blades are one file, both pilots are one file (the second is the first turned half round).",
               ok=bool(ins["ok"] and all(q["solid_check"] == "PASS" and q["fits_bed"] for q in files.values())),
               test_set=dict(what="one grab unit, to try on the bench against the Scout latch cartridge (parts/latch7/kit.json)", files={f"print/box_arm8_{k}.stl": ORDER[0][2][k] for k in TEST_SET},
                             bought=TEST_BOUGHT, steps=TEST_STEPS, install=ins),
               order=order, files=list(files.values()), waiting=waiting, open=OPEN, assembly=M.ORDER)
    json.dump(out, open(kj, "w"), indent=1)
    print("kit ok", out["ok"], "waiting", [w["part"] for w in waiting])


if __name__ == "__main__":
    main()
