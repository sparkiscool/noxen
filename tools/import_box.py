"""The box (Nox Ground Unit) as a system of this repo: one page of everything round the landing funnel, and the one check between its
streams (tools/box_check.py). Writes systems/box.assembly.json (box W frame), systems/box.check.json and systems/box.appendix.html.
  swap arm         systems/swap_arm.assembly.json once the arm stream has pushed it; until then the arm v7 meshes imported from the
                   nox repo on 3 Nov (hub, stub, rails, heads, grabbers, detent, drives, 11 bay receivers, two packs), marked superseded
  stand, enclosure the built demo's mast, cradle keel, kb, toe and cross bars; the v2 sheets and PSU as boxes (from the nox repo, 3 Nov)
  funnel           parts/box_funnel/make.py v5: the two frame halves and the cross arm (engine parts: their result once run, a
                   placeholder and their drawn faces and room until then), the two end walls, strips, drive
  dome             parts/box_dome (its entries are kept as they stand in the file)
  latch            latch v7, both packs, seated: receiver and cartridge with its pack (parts/latch7/make.py build, in the seat frame)
  drones           the Scout v4 and the Overseer as their own systems stand today, seated (their meshes in place, no copies)
Run: python tools/import_box.py            everything but the 3 Nov import is rebuilt; the arm, stand and enclosure lines are kept
     python tools/import_box.py <nox>      also re-imports the arm v7, stand and enclosure from a checkout of sparkiscool/nox
"""
import json, os, re, shutil, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import seat_check as _sc
SEAT_TO_W = _sc.SEAT_TO_W.tolist()   # the seat's place in the box: tools/seat_check.py
EXTRA_BAYS = {60: "rec60", 120: "rec120"}       # 8 Nov (Spark: fill the wheel without getting in the way of the landing): the two slots beside the drone, never in arm v7, as Overseer bays (rec0 turned); 11.4 clear of the Overseer, 13.2 of the Scout, 17.2 of the funnel
DROPPED_BAYS = set()                                # 3 Nov rec30 and rec-210 were dropped (through the deck at W Z 66); 8 Nov back: no deck, seat at W Z 220, 49.8 clear of the Overseer's loop
BOUGHT = {"motor", "shaft", "p20", "p80", "belt", "brg", "axles", "servo", "detent_roller", "detent_springs", "detent_tower", "ties", "stub_bolts", "pack"}
GROUP = {"rec": ("bays", "#8a94a6"), "pack": ("packs", "#2f4fbf"), "motor": ("arm drives", "#555555"), "servo": ("arm drives", "#555555"),
         "brg": ("arm drives", "#9a9a9a"), "shaft": ("arm drives", "#9a9a9a"), "p20": ("arm drives", "#777777"), "p80": ("arm drives", "#777777"),
         "belt": ("arm drives", "#333333"), "axles": ("arm drives", "#9a9a9a"), "ties": ("arm drives", "#333333"), "stub_bolts": ("arm drives", "#9a9a9a"),
         "detent": ("swap arm", "#99a6b3")}


def slug(s):
    return "box_" + re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def copy_mesh(src, pid):
    d = os.path.join(ROOT, "parts", pid); os.makedirs(d, exist_ok=True)
    m = trimesh.load(src, force="mesh")
    if len(m.faces) > 60000:                                   # the topology bars carry 270k faces (13 MB each); the page decimates to 20k anyway
        m = m.simplify_quadric_decimation(face_count=40000)
    m.export(os.path.join(d, "part.stl"))
    return f"parts/{pid}/part.stl", len(m.faces)


def imported(nox):
    """the arm v7, the stand and the enclosure from a checkout of sparkiscool/nox (3 Nov)"""
    parts = []
    arm = json.load(open(os.path.join(nox, "nox", "box", "box_swap_arm.assembly.json")))
    for p in arm["parts"]:
        nm = p["name"]; key = next((k for k in GROUP if nm.startswith(k)), None)
        if nm in DROPPED_BAYS:
            continue
        grp, col = GROUP.get(key, ("swap arm", "#99a6b3"))
        bought = any(nm.startswith(b) for b in BOUGHT)
        pid = slug("arm " + nm)
        mesh, nf = copy_mesh(os.path.normpath(os.path.join(nox, "nox", "box", p["mesh"])), pid)
        parts.append(dict(id=pid, name=f"arm: {nm}", role="bought" if bought else "structure", group=grp, color=col, mesh=mesh,
                          pose=p.get("pose"), material=None if bought else "PETG", mass_g=p.get("mass_g"), tag=p.get("tag", "C"),
                          owner="box swap arm v7", version="v7", source="nox/box/box_swap_arm.assembly.json (runs/arm/v7/asm)",
                          note=p.get("note", "")))
    for deg, nm in EXTRA_BAYS.items():                         # every slot of the 30 deg ring but the drone's: rec0's receiver and cartridge turned about the axle (W Y)
        c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        m = trimesh.load(os.path.join(ROOT, "parts", "box_arm_rec0", "part.stl"), force="mesh")
        m.apply_transform(np.array([[c, 0, -s_, 0], [0, 1, 0, 0], [s_, 0, c, 0], [0, 0, 0, 1.0]]))
        pid = slug("arm " + nm); d = os.path.join(ROOT, "parts", pid); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
        parts.append(dict(id=pid, name=f"arm: {nm}", role="structure", group="bays", color=GROUP["rec"][1], mesh=f"parts/{pid}/part.stl",
                          pose=np.eye(4).tolist(), material="PETG", mass_g=None, tag="C", owner="box swap arm v7", version="v7 + 8 Nov",
                          source="tools/import_box.py (rec0 turned about the axle)",
                          note=f"overseer bay at {deg} deg, beside the drone slot (8 Nov): latch receiver + a seated cartridge, the 0 deg bay turned"))
    st = json.load(open(os.path.join(nox, "nox", "box", "box_structure.assembly.json")))
    for p in st["parts"]:
        if p["name"] not in ("mast", "cradle_keel", "kb", "toe", "barA", "barB"):
            continue
        pid = slug("stand " + p["name"])
        mesh, nf = copy_mesh(os.path.join(nox, p["mesh"]), pid)
        parts.append(dict(id=pid, name=f"stand: {p['name']}", role="structure", group="stand", color="#b7bfc9", mesh=mesh, pose=p.get("pose"),
                          material="PETG", mass_g=p.get("mass_g"), tag=p.get("tag", "M"), owner="box structure", version=p.get("version"),
                          source="nox/box/box_structure.assembly.json (runs/box_system/v1/W)", note=p.get("note", "")))
    v2 = json.load(open(os.path.join(nox, "nox", "box", "box_structure_v2.assembly.json")))
    for p in v2["parts"]:
        if "box" not in p or not (p["name"].startswith("sheet") or p["name"].startswith("PSU")):
            continue
        pid = slug("enclosure " + p["name"])
        bx = list(p["box"])
        if p["name"].startswith("sheet"):
            bx[5] = 145.0                                                  # the sheets end under the rail tips (r 151) and the Overseer's loop feet (W Z 155): no deck (3 Nov)
        parts.append(dict(id=pid, name=f"enclosure: {p['name']}", role="bought", group="enclosure", color="#c9d6e2" if p["name"].startswith("sheet") else "#777777",
                          box=bx, opacity=0.25 if p["name"].startswith("sheet") else None, mass_g=p.get("mass_g"), tag="C",
                          owner="box structure v2", version="v2 layout", source="nox/box/box_structure_v2.assembly.json", note=p.get("note", "")))
    parts.append(dict(id="box_arm_head_at_drone", name="arm: head A at the drone, extended +60 (placeholder)", role="structure", group="swap arm", color="#d08a3a",
                      box=[-20.0, 20.0, 59.0, 74.0, 100.0, 177.0], material="PETG", tag="E", owner="box swap arm", version="concept 3 Nov",
                      source="tools/import_box.py", note="a second stage on the head: 65 more radial reach at the drone slot only, so the cartridge seats at W Z 176 (the grip is the box below)"))
    parts.append(dict(id="box_arm_grip_at_drone", name="arm: grip at the drone cartridge (placeholder)", role="structure", group="swap arm", color="#d08a3a",
                      box=[-15.5, 15.5, 149.0, 214.0, 167.0, 186.0], material="PETG", tag="E", owner="box swap arm", version="concept 3 Nov",
                      source="tools/import_box.py", note="the grabber at the cartridge tabs, W Z 176..183, inside the funnel's hole"))
    return parts


def view(m, pid, name):
    """a mesh this tool makes (seat frame) into parts/<pid>/part.stl"""
    d = os.path.join(ROOT, "parts", pid); os.makedirs(d, exist_ok=True)
    mm = m.to_mesh(); t = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
    if len(t.faces) > 60000:
        t = t.simplify_quadric_decimation(face_count=40000)
    t.export(os.path.join(d, "part.stl")); return f"parts/{pid}/part.stl"


def funnel_parts():
    """the funnel v5 (parts/box_funnel/make.py): engine pieces by their folder (the page shows their result once it exists), plus,
    until then, their drawn faces and their room"""
    import box_check as bc
    fm = bc.load("parts/box_funnel/make.py", "box_funnel_make"); T = np.array(SEAT_TO_W); out = []
    base = dict(role="structure", group="funnel", material="PCTG", tag="C", owner="box funnel", source="parts/box_funnel/make.py")
    def placing(sx, sy):                                         # the one arm (drawn at nose +y) at each corner, seat frame
        P = np.eye(4); P[0, 0], P[1, 1] = sx, sy
        if sx < 0:
            P[0, 3] = fm.BEAM_X[-1][0] + fm.BEAM_X[1][1]         # x -> (tail beam's outer face + nose beam's outer face) - x
        return P
    eng = [("box_funnel_half_nose", "funnel frame, nose half (generated)", fm.half_bodies(1), np.eye(4), None),
           ("box_funnel_half_tail", "funnel frame, tail half (generated)", fm.half_bodies(-1), np.eye(4), None)]
    eng += [(f"box_funnel_arm{'' if (sx, sy) == (1, 1) else '_' + 'nt'[sx < 0] + 'pm'[sy < 0]}", f"funnel cross arm, {'nose' if sx > 0 else 'tail'} {'+y' if sy > 0 else '-y'}"
             f"{'' if sx * sy > 0 else ' (mirrored print)'} (generated)", fm.arm_bodies(1, 1), placing(sx, sy), "box_funnel_arm") for sx in (1, -1) for sy in (1, -1)]
    for pid, name, (drawn, cuts, env, R), P, src in eng:
        folder = src or pid; bb = np.asarray(drawn.bounding_box(), float)
        e = dict(base, id=pid, name=name, color="#e07020", pose=(T @ P).tolist(), version="v5 setup 13 Nov",
                 box=[float(bb[0]), float(bb[3]), float(bb[1]), float(bb[4]), float(bb[2]), float(bb[5])], opacity=0.15,
                 note="engine part (parts/%s/part.py): the box is its drawn faces' extent until its run commits a mesh" % folder)
        if src:
            e["part"] = src
        rj = os.path.join(ROOT, "parts", folder, "result.json")
        done = os.path.exists(rj) and bool(json.load(open(rj)).get("print_stl"))      # a CI run's result (its print mesh published), not a local smoke's
        if done:
            e.pop("opacity"); e["mesh"] = f"parts/{folder}/part.stl"
        out.append(e)
        if not done and (pid == folder):                         # one drawing of the drawn faces and the room per folder
            for suf, m, col, op, what in (("drawn", drawn - cuts, "#e07020", None, "drawn faces (what a drone, a wall, a bolt or a motor touches)"),
                                          ("room", fm.room(drawn, cuts, env), "#f2c9a0", 0.25, "room the engine may fill (what every clearance check uses)")):
                for k, (qid, qname, _, QP, qsrc) in enumerate([q for q in eng if (q[4] or q[0]) == folder]):
                    vid = f"{qid}_{suf}"
                    mesh = view(m, f"{folder}_{suf}", vid) if k == 0 else f"parts/{folder}_{suf}/part.stl"
                    out.append(dict(base, id=vid, name=f"{qname.replace(' (generated)', '')}: {what}", color=col, opacity=op, mesh=mesh, pose=(T @ QP).tolist(),
                                    part=f"{folder}_{suf}", version="v5 13 Nov", note="shown until the run's mesh is in"))
    for pid, name, col, mat, role, note in (
            ("box_funnel_wall_nose", f"funnel nose wall (slides {fm.TRAVEL} in for the Scout)", "#3070e0", "PCTG", "structure", "two tabs out through the side walls to an M4 nut block each; the +y block carries the switch dog"),
            ("box_funnel_wall_tail", f"funnel tail wall (slides {fm.TRAVEL} in for the Scout)", "#3070e0", "PCTG", "structure", "as the nose wall"),
            ("box_funnel_strips", "funnel ledge strips (TPU 95A, landing impact)", "#3a3a3a", "TPU", "structure", "flush in the side ledges; holes for the four probes"),
            ("box_funnel_drive", "funnel wall drive and fasteners (bought)", "#555555", None, "bought",
             "4 N20 gearmotors with an M4 x 100 screw shaft cut to 62, 4 KW12-3 roller switches (+y), 4 P156 spring probes in R156 sleeves, 16 M4 x 60 lap bolts, 8 M4 x 16 sheet bolts, 4 M4 x 16 and 2 M3 x 16 joint bolts")):
        out.append(dict(base, id=pid, name=name, role=role, color=col, material=mat, mesh=f"parts/{pid}/part.stl", pose=SEAT_TO_W, version="v5 drawn 13 Nov", note=note))
    return out


def latch_parts():
    """latch v7, both packs, at the seat (as tools/box_check.py poses them): receiver, cartridge with its pack"""
    import box_check as bc
    out = []
    for drone, (R, C) in bc.latch_bodies().items():
        pack = bc.PACK[drone]; grp = "landing: Scout" if pack == "scout" else "landing: Overseer"
        for body, kind in ((R, "receiver"), (C, "cartridge")):
            pid = f"box_latch7_{pack}_{kind}"
            out.append(dict(id=pid, name=f"{'Scout' if pack == 'scout' else 'Overseer'}: latch v7 {kind}" + (" and pack" if kind == "cartridge" else ""), role="protected", group=grp,
                            color="#8899aa" if kind == "receiver" else "#2f4fbf", mesh=view(body.man, pid, pid), pose=SEAT_TO_W, material="PCTG", tag="C", owner="battery latch",
                            version="v7", source=f"parts/latch7/make.py build({pack!r}), seated", note="the latch set the funnel seats"))
    return out


def drone_parts():
    """both drones as their systems stand, seated (tools/seat_envelope.py's seating), their own meshes in place"""
    import seat_envelope as se
    from build_site import pose_matrix
    out = []
    for sysname, label, TD in (("scout_v4", "Scout", np.eye(4)), ("overseer", "Overseer", se.T_OV)):
        a = json.load(open(os.path.join(ROOT, "systems", f"{sysname}.assembly.json")))
        for p in a["parts"]:
            if any(k in p["name"].lower() for k in se.SKIP):
                continue
            src = p.get("part", p.get("id")); cand = f"parts/{src}/part.stl"
            e = dict(id=f"box_{sysname}_{p['id']}", name=f"{label}: {p['name']}", role=p.get("role", "bought"), group=f"landing: {label}", color=p.get("color"),
                     pose=(np.array(SEAT_TO_W) @ TD @ pose_matrix(p.get("pose"))).tolist(), material=p.get("material"), tag=p.get("tag", "E"), owner=sysname,
                     version=p.get("version"), source=f"systems/{sysname}.assembly.json, seated")
            if os.path.exists(os.path.join(ROOT, cand)):
                e["mesh"] = cand; e["part"] = f"box_{sysname}_{p['id']}"          # no folder of that name: the page takes the mesh as placed
            elif p.get("mesh") and os.path.exists(os.path.join(ROOT, p["mesh"])):
                e["mesh"] = p["mesh"]; e["part"] = f"box_{sysname}_{p['id']}"
            else:
                e.update({k: p[k] for k in ("box", "cylinder", "sphere") if k in p})
                if not any(k in e for k in ("box", "cylinder", "sphere")):
                    continue
            if p.get("opacity") is not None:
                e["opacity"] = p["opacity"]
            e["max_faces"] = 8000 if any(k in p["name"].lower() for k in ("rods", "tubes")) else 1200      # context on this page: 113 drone parts must fit the 16 MB artifact
            out.append(e)
    return out


BOUGHT_LIST = [
    ("N20 gearmotor, M4 x 100 screw shaft, 6 V 100 rpm (12 x 10 x 25 body)", 4, "https://www.amazon.com/Thread-Shaft-100mm-Motor-terminals/dp/B0GDTKHL8T",
     "one per wall tab; pick the 6 V 100 rpm option (alt B07JD2SY8W); each shaft cut to 62: Amazon lists this motor with a 34, 55 or 100 shaft only (searched 13 Nov), and the wall's travel needs 59.3 (front plate 3.5 + half the nut block 6.0 + 48.2 of travel + half the nut 1.6), so the 55 is 4.3 short; 37.2 mm at 0.7 mm a turn = 32 s"),
    ("KW12-3 roller lever micro switch, 10 pack", 2, "https://www.amazon.com/SDTC-Tech-Momentary-Microswitch-Normally/dp/B0CPYNDYPP", "one home switch a wall (outmost), on the +y side; every other position is timed from it"),
    ("P156-E spring probe, 100 pack", 4, "https://www.amazon.com/Spring-Probes-Accuracy-Testing-Contact/dp/B0CC2GQKL1", "seat sense and keep-alive, in the ledges"),
    ("3 x 3 x 16 steel parallel key, round ended, 10 pack", 4, "https://www.amazon.com/Feather-Stainless-Parallel-Fastening-Connection/dp/B0GC54N5FH",
     "the hold-down keys, two pressed into each end wall's square sockets, flats up and down; to length, nothing to cut. Round ended: 2.0 in the receiver's notch it bears on 5.0 mm2 (a square end 6.0), 5.5 MPa at 27.7 N a key against PCTG's 46 (listing found by search 14 Nov, not opened: check 3 x 3 x 16 on the page)"),
    ("R156-4S probe sleeve, 100 pack", 4, "https://www.amazon.com/100Pcs-Spring-Sleeve-R156-4S-Welding/dp/B0CPWSZN8L", "glued in the ledge bores"),
]
FASTENERS = "16 M4 x 60 with nuts and washers (the four laps), 8 M4 x 16 with nuts and washers (arms to the side sheets: 8 holes O4.5 in the acrylic, 99 under the seat plane's W Z 220.3 = W Z 121.3), 4 M4 x 16 and 2 M3 x 16 with nuts and washers (the halves' joint: an M4 behind each side wall, an M4 outside each rib at |y| 83, an M3 under each ledge), 4 M4 nuts (the walls' blocks), 8 M2 x 10 self-tapping (switches), 4 cable ties (motors)"
def print_rows():
    """the funnel's printed pieces: parts/box_funnel/kit.json (tools/funnel_kit.py: each file in its print pose, measured and gated as it lies)"""
    k = json.load(open(os.path.join(ROOT, "parts", "box_funnel", "kit.json")))
    return k, [(p["name"], f"x {p['count']}", " x ".join(f"{v:g}" for v in p["bed_mm"]), p["pose"], f"{p['bed_contact_mm2']} mm2" + (f", brim {p['brim_mm']:g}" if p.get("brim_mm") else ""),
                p["supports"], p["material"], p["solid_check"], p["file"]) for p in k["pieces"]]


def appendix(chk):
    """systems/box.appendix.html: the check's result, the funnel's print list and bought parts"""
    import html
    e = html.escape
    rows = "".join(f"<tr><td>{e(c['what'])}</td><td>{'clear' if c['ok'] else 'OVERLAP'}</td></tr>" for c in chk["checked"])
    pairs = "".join(f"<tr><td>{e(p['a'])}</td><td>{e(p['b'])}</td><td>{p['mm3'] if p['mm3'] is not None else str(p['samples_inside']) + ' samples'}</td><td>{p['states']}: {e(p['state'])}</td></tr>" for p in chk["overlapping_pairs"])
    clr = "".join(f"<tr><td>{c['mm']:.1f}</td><td>{e(c['between'])}</td><td>{e(c['and_'])}</td><td>{e(str(c['at']))}</td></tr>" for c in chk["clearances_mm"])
    kit, pr = print_rows()
    prt = "".join("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in pr)
    bought = "".join(f"<tr><td><a href=\"{u}\">{e(n)}</a></td><td>{q}</td><td>{e(note)}</td></tr>" for n, q, u, note in BOUGHT_LIST)
    return f"""<h2>Compatibility check (tools/box_check.py)</h2>
<p><b>{'PASS: 0 overlaps' if chk['ok'] else 'FAIL: ' + str(len(chk['overlapping_pairs'])) + ' overlapping pairs'}</b>. Every pair of pieces from different streams (funnel, latch v7, drones, arm, box), seated, landing and on the swap path. Arm: {e(chk['arm'])}.{' Not checked, no mesh in the tree: ' + e(', '.join(chk['no_mesh_in_the_tree'])) + '.' if chk['no_mesh_in_the_tree'] else ''}</p>
<table><tr><th>checked</th><th>result</th></tr>{rows}</table>
{'<table><tr><th>piece</th><th>piece</th><th>mm3</th><th>states, worst</th></tr>' + pairs + '</table>' if pairs else ''}
<h3>Clearances (mm, sampled to about 1)</h3><table><tr><th>mm</th><th>between</th><th>and</th><th>where</th></tr>{clr}</table>
<h2>Funnel v5: print kit (A1 mini, 180 bed; parts/box_funnel/kit.json)</h2>
<p><b>{'Every file is ready: in its print pose, on the plate, solid check PASS on that file.' if kit['ok'] else 'NOT ready to print: ' + e(', '.join(kit['not_ready']) or 'pieces missing') + ' (the halves and arms are written from their runs; a piece is ready when its file lies over its bed minimum, fits the plate and passes the solid check).'}</b> The mast is not in the kit: the stand is not redrawn (the note below).</p>
<table><tr><th>file</th><th>count</th><th>on the bed (x, y, z)</th><th>pose</th><th>flat on the bed</th><th>supports</th><th>material</th><th>solid check</th><th>print file</th></tr>{prt}</table>
<h3>Bought</h3><table><tr><th>part</th><th>qty</th><th>note</th></tr>{bought}</table><p>Fasteners: {e(FASTENERS)}.</p>
<p>Hold-down: each end wall carries two 3 x 3 x 16 steel keys that enter the notches in the receiver's end faces; the walls land a drone at park, back off to open and drive in to seated (the arm v8 head pushes up to 118 N; the Overseer weighs 21.6 N, the Scout 4.9 N).</p>
<p><b>Stand: not ready to print.</b> {e(STAND_NOTE)}</p>"""


UNCUT_CM3 = {"box_stand_cradle_keel": 55.2, "box_stand_toe": 41.5}     # the built demo's pieces before any cut (git 33cd639)
STAND_NOTE = ("The arm v8 wheel (seat faces at R 127, was 113) swings its bays to within 8.1 mm of the floor (the -90 bay's lowest point W Z -192.9, "
              "its corners r 195.4 over W Y 108..222; the floor is W Z -201), through the built demo's cradle keel and toe. Cut back 2.5 clear of that bay "
              "(what the page shows) the keel keeps 30.0 of 55.2 cm3: its cradle (W Z -181..-153 over W Y 143..197) is gone and its 20 mm bar is 9.2 high "
              "over W Y 107..197, section modulus 89 mm3 where it was 5637 (W Y 143..197) and about 105 where it was 580 (W Y 107..137); the toe keeps 37.7 "
              "of 41.5 cm3, 95 to 241 mm3 at its root (W Y 200..212) where it was 354 to 2657. In PETG at index 0.5 (flight class, 10.8 MPa in the layers) "
              "the cut keel carries 3 N at the toe's front edge (uncut 193 N); a 50 N hand there reads 5.1 against the misuse limit of 1.0. FAIL: the cut "
              "pieces must not be printed or the demo's keel cut to match. Nothing thicker than 3 mm fits under the wheel on the centre line; the room "
              "is at the sides (the swept circle is 15 above the floor at |X| 70, 28 at |X| 100), so the spine from the cross bars to the front sheet has "
              "to become two rails outside |X| 70, or the hub has to stand 15 higher over the floor (mast, sheets and funnel with it). That is a change "
              "to the box structure with the arm, not a recut.")


def stand_versions(parts):
    """the cut stand pieces say what is left of them and that they fail (recut_stand only knows its own last cut)"""
    for p in parts:
        if p["id"] in UNCUT_CM3 and p.get("mesh"):
            v = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh").volume / 1000
            if v < UNCUT_CM3[p["id"]] - 0.2:
                p["version"] = f"{str(p.get('version') or '').split('; recut')[0]}; recut round the arm v8 bay -90: {v:.1f} of {UNCUT_CM3[p['id']]} cm3 left, 2.5 clear. FAILS strength as cut (the page's stand note): not to print"


def recut_stand(parts, arm_parts, clear=2.5):
    """the stand's pieces cut back from the arm's bays where they cross (13 Nov: the arm v8 ring is R 127, was 113, and its -90 deg
    bay sits in the built demo's cradle keel, 8.5 cm3): the bays' hull, grown by `clear`, out of the stand piece's mesh. Geometry
    only: the keel's strength with that pocket is not checked here, and the printed demo keel needs the same pocket or a new keel."""
    import manifold3d as mf
    def load_w(p):
        m = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh"); m.apply_transform(np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4)); return m
    bays = {}
    for p in arm_parts:
        if p.get("group") == "bays" or (p.get("kin") or {}).get("bay") is not None:
            if p.get("mesh") and os.path.exists(os.path.join(ROOT, p["mesh"])):
                bays.setdefault(p["name"].split(":")[0].strip(), []).append(load_w(p))          # "bay -90": its receiver, cartridge, pack and saddle together
    done = []
    for p in parts:
        if not p["id"].startswith("box_stand_") or not p.get("mesh"):
            continue
        sm = load_w(p); cut = []
        for key, ms in bays.items():
            V = np.vstack([m.vertices for m in ms])
            if (V.min(0) > sm.bounds[1]).any() or (V.max(0) < sm.bounds[0]).any():
                continue
            h = trimesh.convex.convex_hull(V); h = trimesh.convex.convex_hull(h.vertices + clear * h.vertex_normals)
            cut.append((key, mf.Manifold(mf.Mesh(np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.uint32)))))
        if not cut:
            continue
        S = mf.Manifold(mf.Mesh(np.asarray(sm.vertices, np.float32), np.asarray(sm.faces, np.uint32)))
        if S.is_empty():
            print(f"recut_stand: {p['id']} is not a solid mesh, left as it is"); continue
        v0 = S.volume(); keys = []
        for key, c in cut:
            if (S ^ c).volume() > 0.5:
                S = S - c; keys.append(key)
        if not keys:
            continue
        mm = S.to_mesh(); t = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
        t.apply_transform(np.linalg.inv(np.array(p["pose"], float)) if isinstance(p.get("pose"), list) else np.eye(4)); t.export(os.path.join(ROOT, p["mesh"]))
        p["version"] = f"{str(p.get('version') or '').split('; recut')[0]}; recut 13 Nov round the arm v8 {', '.join(keys)} ({(v0 - S.volume()) / 1000:.1f} cm3 off, {clear} clear; strength not rechecked)"
        done.append((p["id"], keys, round((v0 - S.volume()) / 1000, 1)))
    print(f"recut_stand: {done or 'nothing crosses a bay'}")


def main(nox=None):
    out = os.path.join(ROOT, "systems", "box.assembly.json"); old = json.load(open(out))
    kept = lambda pre: [p for p in old["parts"] if p["id"].startswith(pre)]
    parts = imported(nox) if nox else kept(("box_arm_", "box_stand_", "box_enclosure_"))
    new_arm = os.path.join(ROOT, "systems", "swap_arm.assembly.json")
    if os.path.exists(new_arm):                                  # the arm stream's system replaces the 3 Nov import (bays and packs with it)
        parts = [p for p in parts if not p["id"].startswith("box_arm_")]
        for p in json.load(open(new_arm))["parts"]:
            if p.get("group") in ("drone seat", "box (context)") or (p.get("kin") or {}).get("pack") == "dead":      # its own copies of the seat's latch set and the funnel
                continue
            parts.append(dict(p, id="box_swaparm_" + p["id"], name="arm: " + p["name"], part=p.get("part", p["id"]), owner="swap arm", source="systems/swap_arm.assembly.json", max_faces=4000))
    else:
        for p in parts:
            if p["id"].startswith("box_arm_"):
                p["version"] = "v7, superseded (the arm stream's systems/swap_arm.assembly.json replaces it when pushed)"
    if os.path.exists(new_arm):
        recut_stand(parts, json.load(open(new_arm))["parts"])
    stand_versions(parts)
    parts += funnel_parts() + kept("box_dome_") + latch_parts() + drone_parts()
    a = dict(old, parts=parts, appendix="systems/box.appendix.html", fps=8,
             note=("Box W frame (X across, Y along the axle toward the front, Z up; the swap arm hub at the origin, the drone seat at (0, 167.1, 220.3)). "
                   "In: the landing funnel v5 (13 Nov: the frame in two halves on four cross arms under the rain lid, each end wall on two lead-screw gearmotors; the halves and the arm are "
                   "engine parts, shown as their drawn faces and room until their runs), latch v7 for both packs at the seat, the Scout v4 and the Overseer seated as they stand today, "
                   + ("the swap arm from systems/swap_arm.assembly.json, " if os.path.exists(new_arm) else "the swap arm v7 at park with 11 bay receivers and two packs (3 Nov import, SUPERSEDED: the arm stream's new arm replaces it when pushed), ")
                   + "the stand of the built demo with its keel and toe cut back from the arm v8 bays (they FAIL strength as cut: the note under the viewer), the v2 enclosure sheets to W Z 145 and the PSU as boxes, the Scout's rain dome and the rain lid. Out: the electronics. "
                   "Compatibility between the streams: tools/box_check.py, its result under the viewer."))
    import box_check
    json.dump(a, open(out, "w"), indent=0, ensure_ascii=False)
    chk = box_check.main(quick="--quick" in sys.argv)
    open(os.path.join(ROOT, "systems", "box.appendix.html"), "w").write(appendix(chk))
    print(f"box: {len(parts)} parts -> {out}")
    return chk["ok"]


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    sys.exit(0 if main(args[0] if args else None) else 1)
