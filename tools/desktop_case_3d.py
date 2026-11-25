"""Desktop dual-board case, the 3D interactive artifact (Spark 25 Nov, from bed on the projector: rotate, zoom, explode, every part labelled, true size).
    python tools/desktop_case_3d.py        -> artifacts/desktop_dual_board_case_3d.html (one self-contained page: meshes, three.js, OrbitControls inlined)
It runs parts/desktop_dual_board_case/make.py (the design itself: tiles, handle, projector cradle, rails, fans, boards, rods) and groups its geometry into
named parts, so the page always shows the latest make.py. Added here, not in make.py: the six USB sockets, the cable runs, the power strip (E, its place is
still open) and the airflow. Sizes on the page are mesh bounding boxes in the case frame (x along the width, y depth, z up, the way it stands), mm."""
import base64, json, os, runpy, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ns = runpy.run_path(os.path.join(ROOT, "parts", "desktop_dual_board_case", "make.py"), run_name="case_make")
tm = trimesh.transformations
g = lambda k: ns[k]
X, Y, T, HB, HL = g("X"), g("Y"), g("T"), g("HB"), g("HL")
Z_D0, Z_BW, Z_D1, Z_B, Z_LID, Z_TOP = g("Z_D0"), g("Z_BW"), g("Z_D1"), g("Z_B"), g("Z_LID"), g("Z_TOP")
Tr, Rx, Rz, tiles_of = g("Tr"), g("Rx"), g("Rz"), g("tiles_of")
FEET, FEET_RAIL, FEET_TPU = g("FEET"), g("FEET_RAIL"), g("FEET_TPU")
BW, BH, BD, BY, STAND = g("BW"), g("BH"), g("BD"), g("BY"), g("STAND")
PCTG, TPU = 1.23e-3, 1.21e-3
cat = lambda ms: trimesh.util.concatenate(ms)


def box(ext, c):
    b = trimesh.creation.box(extents=ext); b.apply_translation(c); return b


def seg(p0, p1, r, n=12):
    return trimesh.creation.cylinder(radius=r, segment=[list(p0), list(p1)], sections=n)


def poly(pts, r):
    return cat([seg(a, b, r) for a, b in zip(pts[:-1], pts[1:])])


def arrow(p0, p1, r=2.2, head=9.0):
    d = np.array(p1, float) - np.array(p0, float); L = np.linalg.norm(d); u = d / L
    shaft = seg(p0, np.array(p1) - u * head, r, 10)
    cone = trimesh.creation.cone(radius=r * 2.6, height=head, sections=14)
    z = np.array([0, 0, 1.0]); ax = np.cross(z, u)
    if np.linalg.norm(ax) < 1e-9: ax = np.array([1.0, 0, 0])
    cone.apply_transform(tm.rotation_matrix(np.arccos(np.clip(z @ u, -1, 1)), ax)); cone.apply_translation(np.array(p1) - u * head)
    return cat([shaft, cone])


def layer(key, z0, kind):
    out = []
    if kind == "deck": return cat(tiles_of(key, lambda u, v: Tr(u, 0, z0)))
    if kind == "front": return cat(tiles_of(key, lambda u, v: Tr(u, T, z0 + v) @ Rx(np.pi / 2)))
    if kind == "rear": return cat(tiles_of(key, lambda u, v: Tr(u, Y, z0 + v) @ Rx(np.pi / 2)))
    x0 = kind
    return cat(tiles_of(key, lambda u, v: Tr(x0, T + u, z0 + v) @ Rz(np.pi / 2) @ Rx(np.pi / 2)))


XS = [270.0, 302.0, 334.0]
parts = []


def add(id_, name, mesh, kind, ex, qty, src, dims, spin=None, pivot=None, sizing=None):
    parts.append(dict(id=id_, name=name, mesh=mesh, kind=kind, ex=ex, qty=qty, src=src, dims=dims, pivot=pivot, spin=spin, sizing=sizing))


MK = "parts/desktop_dual_board_case/make.py"
# ---- printed shell, bottom up
rails, pads = [], []
for x0, y0, w, rw, holes in ns["rails"]:
    rails.append(box([w, rw, FEET_RAIL], [x0 + w / 2, y0 + rw / 2, FEET_TPU + FEET_RAIL / 2]))
    pads.append(box([w, rw, FEET_TPU], [x0 + w / 2, y0 + rw / 2, FEET_TPU / 2]))
add("pad", "TPU 95A foot pads", cat(pads), "tpu", (0, 0, -95), len(pads), MK, "3 mm TPU strips under the rails (40 wide, 130 / 122 / 120 long), cut flat; the case stands on these, 11 mm off the table with the rails")
add("rail", "Foot rails (PCTG)", cat(rails), "printed", (0, 0, -70), len(rails), MK, "8 mm x 40 wide rails along x at y 5..45 and 81..121, three lengths each; the four rods and two M3 x 12 into heat-set inserts clamp them; they lift the intake 11 mm")
add("d0", "D0 floor deck (intake bores)", layer("d0", Z_D0, "deck"), "printed", (0, 0, -40), 3, MK, "3 mm, 3 tiles of 124 x 126; a 100 bore under the 120 fan and 60 bores under each 70 fan: cold air comes in from below; four rod holes")
add("base_front", "Base front wall (USB panel)", layer("base_front", Z_BW, "front"), "printed", (0, -75, -15), 3, MK, "3 mm, 372 x 34; two rows of three 14 x 7 USB panel sockets with M3 holes at x 270 / 302 / 334 (z 38 and 24); solid otherwise")
add("base_rear", "Base rear wall (intake slots)", layer("base_rear", Z_BW, "rear"), "printed", (0, 75, -15), 3, MK, "3 mm, 372 x 34; 8 mm slots at 16 mm pitch: intake, to be backed by a filter sheet")
add("base_end", "Base -x end wall (intake slots)", layer("base_end", Z_BW, 0.0), "printed", (-90, 0, -15), 1, MK, "3 mm, 120 x 34; slotted intake; this is the end the case stands on")
add("base_endp", "Base +x end wall (webcam holes)", layer("base_endp", Z_BW, X - T), "printed", (90, 0, -15), 1, MK, "3 mm, 120 x 34; solid, two M3 holes for the webcam shelf brackets")
add("d1", "D1 deck (fan bores)", layer("d1", Z_D1, "deck"), "printed", (0, 0, 35), 3, MK, "3 mm, 3 tiles; a 113 bore on the 105 hole square for the 120 fan and 66 bores for each 70 fan: the fans blow up through it")
add("bay_front", "Bay front wall (solid)", layer("bay_front", Z_B, "front"), "printed", (0, -75, 60), 6, MK, "3 mm, 372 x 250 in 6 tiles (3 x 2); solid so the air goes along the boards")
add("bay_rear", "Bay rear wall (solid)", layer("bay_rear", Z_B, "rear"), "printed", (0, 75, 60), 6, MK, "3 mm, 372 x 250 in 6 tiles; solid")
add("bay_end", "Bay -x end wall (upper exhaust slots)", layer("bay_end", Z_B, 0.0), "printed", (-90, 0, 60), 2, MK, "3 mm, 120 x 250 in 2 tiles; slots only above z 221 (over the handle line): the hot air leaves here too")
add("bay_endh", "Bay +x end wall (handle, projector plate)", layer("bay_endh", Z_B, X - T), "printed", (90, 0, 60), 2, MK, "3 mm, 120 x 250 in 2 tiles; 8 M3 holes for the handle feet through the doublers and 4 for the projector plate; no slots")
add("lid", "Lid (grille + extractor)", layer("lid", Z_LID, "deck"), "printed", (0, 0, 135), 3, MK, "3 mm, 3 tiles; 6 mm slots over the whole width; a 66 bore for the fifth 70 fan (extractor) over the chimney; four rod holes")
dbl = [ns["place"](d, T, Tr(X, ns["DOUBL_Y0"], lz - 13.5) @ Rz(np.pi / 2) @ Rx(np.pi / 2)) for d, lz in zip(ns["doublers"], ns["LEGZ"])]
add("doubler", "Handle doublers (3 mm)", cat(dbl), "printed", (45, 0, 60), 2, MK, "66 x 27 x 3 inside-to-outside reinforcement behind each handle foot, 4 M3 holes; with the 3 mm wall the end is 6 mm there (margin 6)")
h = ns["handle"].copy(); h.apply_transform(np.array([[1, 0, 0, X + T], [0, 0, -1, ns["YH"]], [0, 1, 0, ns["ZH0"]], [0, 0, 0, 1]], float))
add("handle", "Carry handle", h, "printed", (110, 0, 60), 1, MK, "bar 100 along z (70 clear) 24 x 14, two 15 wide legs, 55 out of the +x end wall; printed flat; each foot on two M3 x 12 into heat-set inserts; hand-calc margin 9.6 on the bar")
cs = ns["cam_shelf"].copy(); cs.apply_transform(Tr(X, Y / 2 - 45.0, Z_D1))
add("camshelf", "Webcam shelf", cs, "printed", (70, 0, 20), 1, MK, "3 mm x 70 x 90, flush with the D1 deck (z 48..51), hung on two M3 L-brackets; a 6.8 hole for the camera's 1/4-20 bolt")
cam = box([ns["CAM_D"], ns["CAM_W"], ns["CAM_H"]], [X + ns["CAM_X0"] + ns["CAM_D"] / 2, Y / 2, Z_D1 + T + ns["CAM_H"] / 2])
add("cam", "NexiGo N60 webcam", cam, "bought", (70, 0, 55), 1, "listing (owned)", "about 82 x 53 x 50 with its clip, 150 g (listing); own microphone; looks out of the +x end below the handle; drawn as a block")
mt = ns["mount"].copy(); mt.apply_transform(Tr(X, Y / 2, ns["ZP"]))
add("pmount", "Projector hinge plate and ears", mt, "printed", (80, 0, 90), 1, MK, "plate 4 x 150 x 64 screwed to the +x end wall top, two 4 mm ears carrying the hinge and the two M3 lock-bolt holes (stowed and 20 deg)")
# the cradle and projector hinge about (X + XP, Y/2, ZP): meshes in the hinge frame, pose set on the page
HX, HY, HZ = X + ns["XP"], Y / 2, ns["ZP"]
M0 = Tr(-ns["S_P"], 0, -ns["N_P"])
cr = ns["cradle"].copy(); cr.apply_transform(M0)
pj = trimesh.creation.box(extents=[ns["PJ_D"], ns["PJ_W"], ns["PJ_H"]]); pj.apply_translation([ns["S_P"] + ns["PJ_D"] / 2, 0, 0.5 + ns["PJ_H"] / 2]); pj.apply_transform(M0)
hinge = dict(at=[HX, HY, HZ], axis="y", pose=20.0)
add("cradle", "Projector cradle (flips)", cr, "printed", (115, 0, 125), 1, MK, "tray 132 x 134 x 4 with a 4 mm stop lip and two 4 mm cheeks 59 high on a hinge at 4 above the tray; swings 70 deg between upright (stowed) and 20 deg above horizontal; M3 lock bolt and wing nut at each end", pivot=hinge)
add("proj", "S2-TEK / HOMPOW projector", pj, "bought", (115, 0, 125), 1, "listing", "124.5 x 124.5 x 53.3 mm, 640 g (listing); vents and ports on its side faces and the speaker grille on top stay open; lens position is an estimate (E)", pivot=hinge)
# ---- fans (Kootek pad, owned), boards, rods
f120 = box([120, 120, 25], [g("FAN120")[0], g("FAN120")[1], Z_BW + 12.5])
f70 = cat([box([70, 70, 15], [fx, fy, Z_BW + 7.5]) for fx, fy in g("FAN70")])
fx_ = box([70, 70, 15], [g("FAN70_TOP")[0], Y / 2, Z_LID - 7.5])
add("fan120", "120 mm fan (Kootek pad)", f120, "bought", (0, 0, 15), 1, "Kootek B01469DJLM (owned)", "120 x 120 x 25 (thickness estimated); stands flat on D0, blows up through D1; 113 bore, 105 hole square")
add("fan70", "70 mm fans (Kootek pad)", f70, "bought", (0, 0, 15), 3, "Kootek B01469DJLM (owned)", "70 x 70 x 15 (thickness estimated) x 3 in the row beside the 120; blow up through D1")
add("fanx", "70 mm extractor fan (top)", fx_, "bought", (0, 0, 160), 1, "Kootek B01469DJLM (owned)", "the fifth fan, under the lid over the 60 mm chimney, pulls the hot air out")
bA = box([BW, BD, BH], [X / 2, BY + BD / 2, Z_B + STAND + BH / 2]); bB = box([BW, BD, BH], [X / 2, Y - BY - BD / 2, Z_B + STAND + BH / 2])
add("boardA", "Board A (TIMI A7S)", bA, "bought", (0, -30, 95), 1, "estimate", "340 x 20 x 230 (E: outline unpublished, +-10): board plus cooler, standing on its 340 edge on 8 mm standoffs, hot side toward the 60 mm gap; mounts wait for measuring")
add("boardB", "Board B (second board)", bB, "bought", (0, 30, 95), 1, "estimate", "340 x 20 x 230 (E), the same size, mirrored so its hot side also faces the gap")
rods = cat([seg((x, y, Z_D0), (x, y, Z_TOP + 8), 1.5) for x, y in g("RODS")])
add("rods", "M3 x 300 threaded rods", rods, "hardware", (0, 0, 190), 4, "Amazon search (bought)", "M3 x 300 at (12,12) and the other three corners; nuts both sides of D0, D1 and the lid clamp and hang the stack")
# ---- the USB panel sockets and the cable runs (added here)
sock = []
for zc in (Z_BW + 24.0, Z_BW + 10.0):
    for u in XS:
        sock.append(box([14, 18, 7], [u, T + 9, zc]))
add("usb", "USB panel sockets (6)", cat(sock), "bought", (0, -105, -15), 6, "Amazon search: USB 3.0 panel mount extension", "E sizes: opening 14 x 7, body about 18 deep, flange two M3 at +-12; top row to board A, bottom row to board B")
cab = []
zp_A, zp_B = Z_B + STAND + 150.0, Z_B + STAND + 100.0
for row, (zc, zp, yb) in enumerate(((Z_BW + 24.0, zp_A, BY + 6), (Z_BW + 10.0, zp_B, Y - BY - 6))):
    for k, u in enumerate(XS):
        y0 = 5.0 + 1.6 * k if row == 0 else 5.0 + 1.6 * k
        pts = [(u, T + 14, zc), (u, y0, zc + 4), (u, y0, zp + 10 * k - 10), (X - 14 - 2 * k, y0, zp + 10 * k - 10), (X - 14 - 2 * k, yb, zp + 10 * k - 10), ((X + BW) / 2, yb, zp + 10 * k - 10)]
        cab.append(poly(pts, 1.6))
add("usbcab", "USB cables to the boards", cat(cab), "cable", (0, -55, 95), 6, "estimate", "each rises in the gap behind the front wall (board A is at y 10), crosses over to the 13 mm end gap and plugs into the board's side ports (port places are E)")
strip = box([200, 45, 28], [X - 130, Y + 30, 24])
add("strip", "Power strip (place open)", strip, "bought", (0, 105, -50), 1, "estimate", "slim 4-outlet strip, about 200 x 45 x 28 (E); there is no room for it inside the box (the base is full of fans), so it is drawn outside against the rear wall; its place is still to be decided")
pw = []
for k, yb in enumerate((BY + 14, Y - BY - 14)):
    pw.append(poly([(X - 100 - 20 * k, Y + 20, 28), (X - 100 - 20 * k, Y - 3, 36), (X - 100 - 20 * k, Y - 8, Z_D1 + 12), (X - 100 - 20 * k, Y - 8, Z_B + 80 + 12 * k), (X - 9 - 3 * k, Y - 8, Z_B + 80 + 12 * k), (X - 9 - 3 * k, yb, Z_B + 80 + 12 * k), ((X + BW) / 2, yb, Z_B + 80 + 12 * k)], 2.0))
add("pwrcab", "Power leads to the boards", cat(pw), "cable", (0, 55, 95), 2, "estimate", "from the strip through a base rear-wall slot, up the rear gap (the D1 deck needs one pass-through hole, not yet drawn), into the end gap and to each board's power jack (E)")
# ---- air: chimney volume and arrows
ch = box([BW, 60 - 2 * 0, BH], [X / 2, Y / 2, Z_B + STAND + BH / 2])
ar = [arrow((x, Y / 2, Z_B + STAND + 40), (x, Y / 2, Z_B + STAND + BH + 30), 2.2, 10) for x in np.linspace(70, X - 70, 6)]
ar += [arrow((x, Y / 2, Z_D0 - 4), (x, Y / 2, Z_D0 + 26), 2.0, 9) for x in (63, 166, 244, 322)]
ar += [arrow((X / 2, y, Z_BW + 20), (X / 2, y + (1 if y > Y / 2 else -1) * -25, Z_BW + 20), 2.0, 9) for y in (-28, Y + 28)]
ar += [arrow((X / 2, Y / 2, Z_TOP + 4), (X / 2, Y / 2, Z_TOP + 55), 3.0, 14)]
add("air", "Airflow and 60 mm chimney", cat([ch] + ar), "air", (0, 0, 0), 1, "design", "blue arrows: in under the base (D0 bores) and through the rear and -x slots, up the 60 mm chimney between the boards (translucent block), out through the lid grille, the extractor and the upper end slots")

# ---- pack
def quant(m):
    m = m.copy(); m.merge_vertices()
    lo, hi = m.bounds; span = np.maximum(hi - lo, 1e-6)
    q = np.round((m.vertices - lo) / span * 65535).astype(np.uint16)
    idx = m.faces.astype(np.uint16 if len(m.vertices) < 65536 else np.uint32)
    return dict(lo=lo.round(4).tolist(), span=span.round(4).tolist(), v=base64.b64encode(q.tobytes()).decode(),
                f=base64.b64encode(idx.tobytes()).decode(), w=("H" if idx.dtype == np.uint16 else "I"))


out, mass = [], 0.0
for p in parts:
    m = p["mesh"]
    e = m.extents
    mg = None
    if p["kind"] in ("printed", "tpu"):
        mg = round(abs(float(m.volume)) * (TPU if p["kind"] == "tpu" else PCTG), 0); mass += mg
    bb = m.bounds if not p["pivot"] else m.bounds
    out.append(dict(id=p["id"], name=p["name"], kind=p["kind"], ex=p["ex"], qty=p["qty"], src=p["src"], dims=p["dims"], pivot=p["pivot"],
                    bbox=[round(float(v), 1) for v in e], at=[round(float(v), 1) for v in bb[0]], mass=mg, mesh=quant(m)))
allb = trimesh.util.concatenate([p["mesh"] for p in parts if p["id"] not in ("air", "strip", "pwrcab", "cradle", "proj")]).bounds
meta = dict(X=X, Y=Y, Z=Z_TOP, feet=FEET, tiles=48, mass=mass, ext=[round(float(v), 1) for v in (allb[1] - allb[0])])
tmpl = open(os.path.join(ROOT, "tools", "desktop_case_3d_template.html"), encoding="utf-8").read()
three = open(os.path.join(ROOT, "tools", "vendor", "three.min.js"), encoding="utf-8").read()
orbit = open(os.path.join(ROOT, "tools", "vendor", "OrbitControls.js"), encoding="utf-8").read()
page = (tmpl.replace("__DATA__", json.dumps(dict(parts=out, meta=meta), separators=(",", ":")))
        .replace("/*__THREE__*/", three.replace("</script", "<\\/script")).replace("/*__ORBIT__*/", orbit.replace("</script", "<\\/script")))
dst = os.path.join(ROOT, "artifacts", "desktop_dual_board_case_3d.html")
open(dst, "w", encoding="utf-8").write(page)
print(dst, f"{len(page) / 1e6:.2f} MB", len(out), "parts, printed mass", mass, "g, case", meta["ext"])
