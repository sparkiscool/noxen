"""The box (Nox Ground Unit) as a system of this repo: one page of everything round the landing funnel, and the one check between its
streams (tools/box_check.py). Writes systems/box.assembly.json (box W frame), systems/box.check.json and systems/box.appendix.html.
  swap arm         systems/swap_arm.assembly.json once the arm stream has pushed it; until then the arm v7 meshes imported from the
                   nox repo on 3 Nov (hub, stub, rails, heads, grabbers, detent, drives, 11 bay receivers, two packs), marked superseded
  stand            the stand v2 (parts/box_stand2_lib.py, 14 Nov): mast, two base halves, two rails (engine parts: their result once
                   run, their drawn faces and room until then) and four TPU feet
  enclosure        the v2 sheets and PSU as boxes (from the nox repo, 3 Nov)
  funnel           parts/box_funnel/make.py v5: the two frame halves and the cross arm (engine parts: their result once run, a
                   placeholder and their drawn faces and room until then), the two end walls, strips, drive
  dome             parts/box_dome (its entries are kept as they stand in the file)
  latch            latch v7, both packs, seated: receiver and cartridge with its pack (parts/latch7/make.py build, in the seat frame)
  drones           the Scout v4 and the Overseer as their own systems stand today, seated (their meshes in place, no copies)
Run: python tools/import_box.py            everything but the 3 Nov import is rebuilt; the arm and enclosure lines are kept
     python tools/import_box.py <nox>      also re-imports the arm v7 and enclosure from a checkout of sparkiscool/nox
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
    """the arm v7 and the enclosure from a checkout of sparkiscool/nox (3 Nov)"""
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
<p><b>{'Every file is ready: in its print pose, on the plate, solid check PASS on that file.' if kit['ok'] else 'NOT ready to print: ' + e(', '.join(kit['not_ready']) or 'pieces missing') + ' (the halves and arms are written from their runs; a piece is ready when its file lies over its bed minimum, fits the plate and passes the solid check).'}</b> The stand is not in this kit: its own section below.</p>
<table><tr><th>file</th><th>count</th><th>on the bed (x, y, z)</th><th>pose</th><th>flat on the bed</th><th>supports</th><th>material</th><th>solid check</th><th>print file</th></tr>{prt}</table>
<h3>Bought</h3><table><tr><th>part</th><th>qty</th><th>note</th></tr>{bought}</table><p>Fasteners: {e(FASTENERS)}.</p>
<p>Hold-down: each end wall carries two 3 x 3 x 16 steel keys that enter the notches in the receiver's end faces; the walls land a drone at park, back off to open and drive in to seated (the arm v8 head pushes up to 118 N; the Overseer weighs 21.6 N, the Scout 4.9 N).</p>
<h2>Stand v2</h2><p>{e(STAND_NOTE)}</p>{stand_rows()}"""


STAND_PIECES = [("box_stand2_mast", "stand v2: mast (the stub's face down to the foot)"),
                ("box_stand2_base_r", "stand v2: base half, right (+X, under the PSU)"), ("box_stand2_base_l", "stand v2: base half, left"),
                ("box_stand2_rail_r", "stand v2: rail, right"), ("box_stand2_rail_l", "stand v2: rail, left")]
STAND_BOUGHT = [("M5 x 16 socket cap, 12.9 (3 stub to mast from behind, 8 base halves up into the mast foot)", 11, "https://www.amazon.com/dp/B0BLNLH7YY"),
                ("M5 x 15 x 1.2 washer, 304 (under the three stub bolts' heads)", 3, "https://www.amazon.com/dp/B0DX6TD1Q7"),
                ("M5 x 60 socket cap, 12.9 (rail laps)", 8, "https://www.amazon.com/dp/B06XCMHM41"),
                ("M5 nyloc, 304 (rail laps)", 8, "https://www.amazon.com/dp/B078NNFT93"),
                ("M5 x D7 x L6 heat-set insert (8 in the mast foot, 3 in the stub flange)", 11, "https://www.amazon.com/dp/B0H142Y3BM")]
STAND_NOTE = ("Stand v2 (14 Nov, parts/box_stand2_lib.py): five PCTG engine pieces and four TPU 95A feet, every piece in the 180 bed. The mast takes the "
              "arm v8 stub's flange on its face at W Y 0 (bearing ring r 33..45, three M5 x 16 on Ø15 washers from behind into heat-set inserts in the flange) and "
              "runs down behind the hub (X -42..40, Y -64..0, clear of the axle stepper and the PSU) to its foot at W Z -135, where two base halves "
              "bolt up into it (8 M5 into inserts). Each half reaches out under the ring's sweep to its rail at |X| 120..177.5 (the sweep leaves 42 mm "
              "over the floor at |X| 120), with the rear foot at (+-149, -44); the rail laps its cheek over Y 72..110 (4 M5 x 60 across) and runs "
              "forward to the front foot at (+-149, 232). Four feet, so every floor reaction is known: flat floor and each foot lifted in turn "
              "(uneven floor). Loads: the arm v8 with its 11 bays and packs (9.7 kg at W Y 129) and every case of parts/box_stand_mast/check.py, "
              "the stand's own weight. Not for the arm stream to miss: the stub's 135 deg M5 hole sits 0.5 mm into the axle stepper behind "
              "the flange (the mast uses the other three), and nothing fits on the flange's front face (2.0 mm to the bearings), so the stub's "
              "holes need M5 heat-set inserts (5.5 to 6.4). The built demo stand (mast, kb, bars, keel, toe) is out: it fails with the arm v8.")


def stand_rows():
    """the stand v2's pieces (status of each run) and its bought parts, for the appendix"""
    from html import escape as e
    rows = ""
    for pid, name in STAND_PIECES:
        rj = os.path.join(ROOT, "parts", pid, "result.json"); r = json.load(open(rj)) if os.path.exists(rj) else {}
        st = r.get("strength") or {}; w = st.get("worst") if isinstance(st, dict) else None
        sc = r.get("solid_ok")
        rel = (r.get("print_stl") or {}).get("url", "") if isinstance(r.get("print_stl"), dict) else ""
        rows += (f"<tr><td>{e(name)}</td><td>{e(str(r.get('status', 'not run yet')))}</td><td>{e(str(r.get('mass_g', '')))}</td><td>{e(str(w if w is not None else ''))}</td>"
                 f"<td>{e(str(sc if sc is not None else ''))}</td><td>{('<a href=' + chr(34) + rel + chr(34) + '>' + e(pid) + '.stl</a>') if rel else ''}</td></tr>")
    bought = "".join(f"<tr><td><a href=\"{u}\">{e(n)}</a></td><td>{q}</td></tr>" for n, q, u in STAND_BOUGHT)
    return ("<table><tr><th>piece</th><th>run</th><th>g</th><th>worst</th><th>solid check</th><th>print file</th></tr>" + rows + "</table>"
            "<p>Also: four TPU 95A feet (parts/box_stand2_foot, Ø36 x 5 with a Ø8 x 4 stud).</p>"
            "<h3>Stand bought</h3><table><tr><th>part</th><th>qty</th></tr>" + bought + "</table>")


def stand_parts():
    """the stand v2: the five engine pieces (their result once CI has run them, else their drawn faces and room) and the four TPU feet"""
    import importlib
    import box_check as bc
    sys.path.insert(0, ROOT)
    L = importlib.import_module("parts.box_stand2_lib")
    base = dict(role="structure", group="stand", material="PCTG", tag="C", owner="box stand v2", source="parts/box_stand2_lib.py")
    out = []
    for pid, name in STAND_PIECES:
        p = importlib.import_module(f"parts.{pid}.part").setup(importlib.import_module(f"parts.{pid}.part").SPEC["h"])
        drawn, cuts, env = p.drawn_m, p.cuts_m, p.env0_m
        rj = os.path.join(ROOT, "parts", pid, "result.json")
        done = os.path.exists(rj) and bool(json.load(open(rj)).get("print_stl"))
        e = dict(base, id=pid, name=name, color="#b7bfc9", pose=np.eye(4).tolist(), version="v2 setup 14 Nov",
                 note=f"engine part (parts/{pid}/part.py)")
        if done:
            e["mesh"] = f"parts/{pid}/part.stl"; out.append(e); continue
        for suf, m, col, op, what in (("drawn", drawn - cuts, "#7f8a96", None, "drawn faces (what the stub, a bolt or the floor touches)"),
                                      ("room", (env + drawn) - cuts, "#d5dbe2", 0.25, "room the engine may fill (what every clearance check uses)")):
            out.append(dict(base, id=f"{pid}_{suf}", name=f"{name}: {what}", color=col, opacity=op, mesh=view(m, f"{pid}_{suf}", f"{pid}_{suf}"),
                            pose=np.eye(4).tolist(), part=f"{pid}_{suf}", version="v2 14 Nov", note="shown until the run's mesh is in"))
    fm = L.foot_mesh()
    fv = view(fm, "box_stand2_foot", "box_stand2_foot")
    for s in (1, -1):
        for k, y in L.FEET.items():
            P = np.eye(4); P[:3, 3] = [s * L.FOOT_X, y, L.FLOOR]
            out.append(dict(base, id=f"box_stand2_foot_{'r' if s > 0 else 'l'}_{k}", name=f"stand v2: TPU 95A foot, {'right' if s > 0 else 'left'} {k}", color="#3a3a3a",
                            material="TPU", mesh=fv, part="box_stand2_foot", pose=P.tolist(), version="v2 14 Nov",
                            note="Ø36 x 5 pad, its Ø8 x 4 stud pressed into the piece's foot pad"))
    return out


def main(nox=None):
    out = os.path.join(ROOT, "systems", "box.assembly.json"); old = json.load(open(out))
    kept = lambda pre: [p for p in old["parts"] if p["id"].startswith(pre)]
    parts = [p for p in (imported(nox) if nox else kept(("box_arm_", "box_enclosure_"))) if not p["id"].startswith("box_stand")]
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
    parts += stand_parts() + funnel_parts() + kept("box_dome_") + latch_parts() + drone_parts()
    a = dict(old, parts=parts, appendix="systems/box.appendix.html", fps=8,
             note=("Box W frame (X across, Y along the axle toward the front, Z up; the swap arm hub at the origin, the drone seat at (0, 167.1, 220.3)). "
                   "In: the landing funnel v5 (13 Nov: the frame in two halves on four cross arms under the rain lid, each end wall on two lead-screw gearmotors; the halves and the arm are "
                   "engine parts, shown as their drawn faces and room until their runs), latch v7 for both packs at the seat, the Scout v4 and the Overseer seated as they stand today, "
                   + ("the swap arm from systems/swap_arm.assembly.json, " if os.path.exists(new_arm) else "the swap arm v7 at park with 11 bay receivers and two packs (3 Nov import, SUPERSEDED: the arm stream's new arm replaces it when pushed), ")
                   + "the stand v2 (mast, two base halves, two rails: engine parts, shown as their drawn faces and room until their runs; four TPU feet), the v2 enclosure sheets to W Z 145 and the PSU as boxes, the Scout's rain dome and the rain lid. Out: the electronics. "
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
