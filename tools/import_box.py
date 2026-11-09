"""The box (Nox Ground Unit) as a system of this repo, from the nox repo's assembly files (3 Nov, Spark: the funnel with the most
up-to-date box mechanism CAD, as real as it gets, one artifact other sessions can reach). Copies the real meshes into
parts/box_<slug>/part.stl (a folder belongs to one system) and writes systems/box.assembly.json in the box W frame:
  swap arm v7      nox/box/box_swap_arm.assembly.json (hub, stub, rails, heads, grabbers, detent, drives, the bay receivers and two
                   packs; meshes runs/arm/v7/asm), poses as in the file (cycle t 0, park); all 9 bays (the two nearest the top, 30 and -210 deg,
                   dropped 3 Nov when they rose through the deck, are back 8 Nov: the seat is 65 higher and there is no deck), plus
                   two Overseer bays at 60 and 120 deg beside the drone (8 Nov): 11 bays, every slot of the 30 deg ring but the drone's
  stand            nox/box/box_structure.assembly.json: mast, cradle keel, kb, toe, the two cross bars (the built demo v1); the v1
                   ring frames, posts, brackets and v4i bay modules are left out: the v2 layout superseded them with the 9 bays
                   the arm file carries, and nothing of v2's bodies is drawn yet
  enclosure        box_structure_v2.assembly.json: the four sheets and the PSU, as boxes
  funnel           parts/box_funnel (drawn here, 3 Nov), posed with the catch's SEAT_TO_W (nox/box/catch_system.py)
  seated Scout     the receiver v3 and cartridge v6 (copies of this repo's latch meshes) at the seat, as context
Run: python tools/import_box.py <path to a nox checkout>
"""
import json, os, re, shutil, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEAT_TO_W = [[0, -1.0, 0, 0], [1.0, 0, 0, 167.1], [0, 0, 1.0, 220.3], [0, 0, 0, 1.0]]   # catch_system.py's Y_FUN 167.1; Z_SEAT 220.3 = the catch's 155.3 + 65 (3 Nov: the head extends 65 more at the drone slot)
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


def box_check(parts):
    """every box part with a mesh against the deck and the funnel (W frame): the Overseer's descent check covers the drones, this covers the box"""
    sys.path.insert(0, os.path.join(ROOT, "tools")); import seat_check as sc
    S = [(f"{nm} ({st} walls)", mm, tt) for st in ("overseer", "scout") for nm, mm, tt in sc.solids(st)]; T = np.array(SEAT_TO_W)   # the box parts against the funnel at both wall settings
    sol = [(nm, m.transform(T[:3, :])) for nm, m, _ in S]                                   # seat frame -> W
    sol_t = [(nm, sc.to_tri(mm)) for nm, mm in sol]
    hits = []
    for p in parts:
        if not p.get("mesh") or p.get("group", "").startswith("landing") or p["id"].startswith(("box_funnel", "box_deck", "box_dome")):
            continue
        m = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh"); m.apply_transform(np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4))
        for (nm, mm), (_, st) in zip(sol, sol_t):
            if not ((m.bounds[0] <= st.bounds[1]).all() and (m.bounds[1] >= st.bounds[0]).all()):
                continue
            mp = sc.man(m)
            if mp is None:
                continue
            v = sc.volume(mp ^ mm)
            if v > 0.5:
                hits.append((p["name"], nm, v, sc.to_tri(mp ^ mm).bounds.round(0).tolist()))
    print(f"box parts vs deck / funnel: {len(hits)} crossings" + "".join(f"\n  {a} x {b}: {v:.0f} mm3 at W {bb}" for a, b, v, bb in hits))
    land = landing_check(parts)
    if hits or land:
        sys.exit("import_box: a box part crosses the funnel or a landing path (above); move or drop it")


def landing_check(parts):
    """every box part with a mesh against both drones seated and at each descent state (seat_check.descent_states, the capture
    limits): a bay or arm part may not cross a landing path (8 Nov: the bays nearest the top came back after the seat rose 65)"""
    import seat_check as sc, seat_envelope as se
    W2S = np.linalg.inv(np.array(SEAT_TO_W))
    box = []
    for p in parts:
        if not p.get("mesh") or p.get("group", "").startswith("landing") or p["id"].startswith(("box_funnel", "box_deck", "box_dome")):   # the funnel's own parts: seat_check, with each drone's wall setting
            continue
        m = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh")
        m.apply_transform(W2S @ (np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4)))
        box.append((p["name"], m, sc.man(m)))
    hits = []
    for drone in [d for d in ("scout", "scout_v4", "overseer") if os.path.exists(os.path.join(ROOT, "systems", f"{d}.assembly.json"))]:   # every drone that seats in the funnel
        dparts = [(nm, m, sc.man(m) or sc.man(m.convex_hull)) for nm, m in se.parts_of(drone)]
        for lab, ox, oy, h in sc.descent_states(drone):
            off = np.array([ox, oy, h])
            for bn, bm, bmm in box:
                for dn, dm, dmm in dparts:
                    if bmm is None or dmm is None or not ((bm.bounds[0] <= dm.bounds[1] + off).all() and (bm.bounds[1] >= dm.bounds[0] + off).all()):
                        continue
                    v = sc.volume(bmm ^ dmm.translate(off.tolist()))
                    if v > 0.5:
                        hits.append((bn, f"{drone} {dn}", lab, v))
    print(f"box parts vs the drones landing: {len(hits)} crossings" + "".join(f"\n  {a} x {b} at {l}: {v:.0f} mm3" for a, b, l, v in hits))
    return hits


def main(nox):
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
    parts.append(dict(id="box_funnel", name="funnel v4 frame (side walls, ledges, beams; the end walls slide)", role="structure", group="funnel", color="#e07020", mesh="parts/box_funnel/part.stl",
                      pose=SEAT_TO_W, material="PCTG", tag="C", owner="box catch", version="v4 drawn 9 Nov", source="parts/box_funnel/make.py",
                      note="v4 9 Nov: the fixed frame (side walls, ledges, skirt, beams, buttress, four brackets under the seat); the nose and tail walls slide (box_funnel_wall_*): 184 x 66.8 seat for the Overseer, the Scout receiver's own 105.6 x 65.5 with the walls 39.2 in (Spark 9 Nov: the Scout seats on its receiver alone)"))
    for q in [{"id": "box_funnel_wall_nose", "name": "funnel nose wall (slides 39.2 in for the Scout)", "role": "structure", "group": "funnel", "color": "#e07020", "mesh": "parts/box_funnel_wall_nose/part.stl", "material": "PCTG", "tag": "C", "owner": "box catch", "version": "v1 drawn 9 Nov", "source": "parts/box_funnel/make.py", "note": "the 55 deg nose face with two tabs out through slots in the side walls to nuts on the lead screws; at the Overseer setting as drawn"}, {"id": "box_funnel_wall_tail", "name": "funnel tail wall (slides 39.2 in for the Scout)", "role": "structure", "group": "funnel", "color": "#e07020", "mesh": "parts/box_funnel_wall_tail/part.stl", "material": "PCTG", "tag": "C", "owner": "box catch", "version": "v1 drawn 9 Nov", "source": "parts/box_funnel/make.py", "note": "the 55 deg tail face with two tabs out through slots in the side walls to nuts on the lead screws; at the Overseer setting as drawn"}, {"id": "box_funnel_strips", "name": "funnel ledge strips (TPU, landing impact)", "role": "structure", "group": "funnel", "color": "#3a3a3a", "mesh": "parts/box_funnel_strips/part.stl", "material": "TPU", "tag": "C", "owner": "box catch", "version": "v1 drawn 9 Nov", "source": "parts/box_funnel/make.py", "note": "flush in the side ledges, |y| 25.5..33; both drones land on them; holes for the four probes"}, {"id": "box_funnel_drive", "name": "funnel wall drive (lead screws, motors, switches, probes)", "role": "bought", "group": "funnel", "color": "#555555", "mesh": "parts/box_funnel_drive/part.stl", "tag": "E", "owner": "box catch", "version": "placeholder 9 Nov", "source": "parts/box_funnel/make.py", "note": "two Tr8 lead screws with opposite-hand halves under the side walls' overhang, a NEMA 8 per side geared to each, a switch at each wall position; four spring probes in the ledges at (1.5 +- 30, +-29.5) for seat sense and keep-alive; sizes E, Amazon listings to pick"}]:
        parts.append(dict(q, pose=SEAT_TO_W))                    # the sliding end walls, the ledge strips and the wall drive (parts/box_funnel/make.py writes them)
    for q in [{"id": "box_dome_hoop_1", "name": "dome hoop 1 (3 mm carbon, r 272) and its hub plates", "role": "structure", "group": "dome", "color": "#3a3f46", "mesh": "parts/box_dome/gore_meshes/hoop_1.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "material": "PCTG", "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "shut at 45 deg; its plates turn on the hub bolts, picked up / pulled out by hoop 2 through the pin in a 45 deg slot"}, {"id": "box_dome_hoop_2", "name": "dome hoop 2 (3 mm carbon, r 281) and its hub plates", "role": "structure", "group": "dome", "color": "#3a3f46", "mesh": "parts/box_dome/gore_meshes/hoop_2.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "material": "PCTG", "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "shut at 90 deg; its plates turn on the hub bolts, picked up / pulled out by hoop 3 through the pin in a 45 deg slot"}, {"id": "box_dome_hoop_3", "name": "dome hoop 3 (3 mm carbon, r 290) and its hub plates", "role": "structure", "group": "dome", "color": "#3a3f46", "mesh": "parts/box_dome/gore_meshes/hoop_3.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "material": "PCTG", "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "shut at 135 deg; its plates turn on the hub bolts, picked up / pulled out by hoop 4 through the pin in a 45 deg slot"}, {"id": "box_dome_hoop_4", "name": "dome hoop 4 (3 mm carbon, r 299) and its hub plates", "role": "structure", "group": "dome", "color": "#3a3f46", "mesh": "parts/box_dome/gore_meshes/hoop_4.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "material": "PCTG", "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "shut at 180 deg; its plates turn on the hub bolts, driven by the motors (the free edge)"}, {"id": "box_dome_fabric_00", "name": "dome fabric, panel 0 strip 0", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_00.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 0 and 1, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_01", "name": "dome fabric, panel 0 strip 1", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_01.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 0 and 1, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_02", "name": "dome fabric, panel 0 strip 2", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_02.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 0 and 1, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_03", "name": "dome fabric, panel 0 strip 3", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_03.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 0 and 1, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_10", "name": "dome fabric, panel 1 strip 0", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_10.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 1 and 2, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_11", "name": "dome fabric, panel 1 strip 1", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_11.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 1 and 2, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_12", "name": "dome fabric, panel 1 strip 2", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_12.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 1 and 2, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_13", "name": "dome fabric, panel 1 strip 3", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_13.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 1 and 2, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_20", "name": "dome fabric, panel 2 strip 0", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_20.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 2 and 3, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_21", "name": "dome fabric, panel 2 strip 1", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_21.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 2 and 3, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_22", "name": "dome fabric, panel 2 strip 2", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_22.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 2 and 3, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_23", "name": "dome fabric, panel 2 strip 3", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_23.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 2 and 3, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_30", "name": "dome fabric, panel 3 strip 0", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_30.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 3 and 4, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_31", "name": "dome fabric, panel 3 strip 1", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_31.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 3 and 4, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_32", "name": "dome fabric, panel 3 strip 2", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_32.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 3 and 4, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_fabric_33", "name": "dome fabric, panel 3 strip 3", "role": "bought", "group": "dome", "color": "#dfe7ee", "mesh": "parts/box_dome/gore_meshes/fabric_33.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "opacity": 0.35, "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "silnylon between hoops 3 and 4, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)"}, {"id": "box_dome_hoop_0", "name": "dome hoop 0 (3 mm carbon, r 263, fixed at the base, rear)", "role": "structure", "group": "dome", "color": "#3a3f46", "mesh": "parts/box_dome/gore_meshes/hoop_0.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "fixed: its plates are the hub brackets"}, {"id": "box_dome_hubs", "name": "dome hub brackets and bolts (printed)", "role": "structure", "group": "dome", "color": "#c9d2d9", "mesh": "parts/box_dome/gore_meshes/hubs.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "left and right on the hinge axis (W X): a block through the lid's edge onto the ring, plate 0 with its slot, the cheeks, a 6 mm shoulder bolt", "material": "PCTG"}, {"id": "box_dome_drive", "name": "dome drive: 2 worm motors, cams, 4 limit switches", "role": "bought", "group": "dome", "color": "#555555", "mesh": "parts/box_dome/gore_meshes/drive.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "JGY-370 12 V 6 rpm double shaft, self-locking, one per hub on plate 4; a cam on the outer shaft between two KW11 switches (open, shut)"}, {"id": "box_dome_ring", "name": "dome base ring (printed segments)", "role": "structure", "group": "dome", "color": "#c9d2d9", "mesh": "parts/box_dome/gore_meshes/ring.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "r 255..297 at W Z 132..142, the lid's edge on its inner lip, segments pinned with 3 mm carbon rods", "material": "PCTG"}, {"id": "box_dome_spokes", "name": "dome spokes (6 mm carbon, under the lid)", "role": "bought", "group": "dome", "color": "#2b2b2b", "mesh": "parts/box_dome/gore_meshes/spokes.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "eight, from the sheets' tops out to the ring"}, {"id": "box_dome_lid", "name": "rain lid (3 mm UV polycarbonate)", "role": "bought", "group": "dome", "color": "#c9d6e2", "mesh": "parts/box_dome/gore_meshes/lid.stl", "pose": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], "tag": "C", "owner": "box dome", "version": "v3 mechanism 9 Nov", "source": "parts/box_dome/make.py", "note": "at the sheets' top (W Z 142..145), flush between them and out to the ring; open within r 160 of the seat (the funnel, the 60 / 120 deg bays, the swap head)", "opacity": 0.5}]:
        parts.append(q)                                              # the Scout's rain and dust dome, its mechanism, and the box's rain lid (parts/box_dome/make.py writes the meshes in W and checks them against both drones and itself)
    # the two drones for the landing timeline (tools/landing_box.py): rigid copies of their current pages, posed seated
    scout = json.load(open(os.path.join(ROOT, "systems", "scout.assembly.json")))
    for p in scout["parts"]:
        if p["id"] not in ("frame_plate_bee35_carbon", "fc", "pack_4s_1500", "camera", "scout_nose_unit", "scout_tail_unit", "scout_nose_unit_room", "scout_tail_unit_room", "scout_cage_v19") and not p["id"].startswith(("scout_motor", "scout_prop")):
            continue
        pid = slug("scout " + p["id"]); mesh, nf = copy_mesh(os.path.join(ROOT, p["mesh"]), pid)
        parts.append(dict(id=pid, name=f"Scout: {p['name']}", role=p.get("role", "bought"), group="landing: Scout", color=p.get("color"), mesh=mesh,
                          pose=(np.array(SEAT_TO_W) @ np.array(p["pose"], float)).tolist() if p.get("pose") else SEAT_TO_W,
                          material=p.get("material"), tag=p.get("tag", "E"), owner="scout v3", version=p.get("version"), source="systems/scout.assembly.json, seated"))
    for src, nm in (("receiver_v3_latch", "Scout: receiver v3"), ("cartridge_v6_latch", "Scout: cartridge v6")):
        pid = slug("ctx " + nm.replace(":", "")); mesh, nf = copy_mesh(os.path.join(ROOT, "parts", src, "part.stl"), pid)
        parts.append(dict(id=pid, name=nm, role="protected", group="landing: Scout", color="#8899aa", mesh=mesh, pose=SEAT_TO_W,
                          material="PETG", tag="M", owner="battery latch", version="v3 / v6", source=f"parts/{src} (this repo), at the seat",
                          note="the latch set every drone carries: what the funnel seats"))
    T_OV = np.array([[1, 0, 0, 31.5], [0, 1, 0, 0], [0, 0, 1, 32.4], [0, 0, 0, 1.0]])     # Overseer drone frame -> seat frame: its catch centre (x -30)
    ov = json.load(open(os.path.join(ROOT, "systems", "overseer.assembly.json")))         # and cartridge bottom (z -76.9: cage v5, the loop 3 under it; an earlier -54 was v3's) onto the Scout's (1.5, -44.5)
    keep = {"cage_shell_tubes", "cage_shell_nodes", "spokes_ties_head_rods", "overseer_base", "top_plate", "bottom_plate", "overseer_head_mount",
            "unitree_l2", "camera_left", "camera_right", "motor_1", "motor_2", "motor_3", "motor_4", "prop_1", "prop_2", "prop_3", "prop_4", "overseer_battery_holder"}
    for p in ov["parts"]:
        if p["id"] not in keep:
            continue
        pid = slug("overseer " + p["id"]); P = np.array(SEAT_TO_W) @ T_OV @ (np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4))
        e = dict(id=pid, name=f"Overseer: {p['name']}", role=p.get("role", "bought"), group="landing: Overseer", color=p.get("color"), pose=P.tolist(),
                 material=p.get("material"), tag=p.get("tag", "E"), owner="overseer", version=p.get("version"), source="systems/overseer.assembly.json, seated")
        srcd = p.get("part", p["id"]); cand = os.path.join(ROOT, "parts", srcd, "part.stl")
        if os.path.exists(cand):
            e["mesh"], nf = copy_mesh(cand, pid)
        elif "box" in p:
            e["box"] = p["box"]
        else:
            continue
        parts.append(e)
    for pid, nm, bx in (("box_ctx_overseer_receiver", "Overseer: receiver v2 (E box)", [1.5 - 92, 1.5 + 92, -33.4, 33.4, -44.3, 27.7]),
                        ("box_ctx_overseer_cartridge", "Overseer: cartridge (E box)", [1.5 - 80, 1.5 + 80, -25, 25, -44.5, 0.5])):
        parts.append(dict(id=pid, name=nm, role="protected", group="landing: Overseer", color="#8899aa", box=bx, pose=SEAT_TO_W,
                          material="PETG", tag="E", owner="battery latch", version="v2 (bay envelope)", source="the swap arm's bay receiver envelope, at the seat",
                          note="the Overseer latch set back in its base slot (out since 29 Oct): 184 x 67 x 72 from the arm's bay envelope; the cartridge width is a CHECK"))
    a = dict(name="Nox Ground Unit", title="Nox Ground Unit", readme_view="iso", artifact=None, machine="box",
             note=("Box W frame (X across, Y along the axle toward the front, Z up; the swap arm hub at the origin, the drone seat at (0, 167.1, 220.3): "
                   "the catch's SEAT_TO_W raised 65, the arm's head extending 65 more at the drone slot, Spark 3 Nov). In: the swap arm v7 at park with 11 bay receivers, every slot of the ring but the drone's (8 Nov) and two packs, "
                   "a placeholder extended head at the drone, the stand of the built demo, the v2 enclosure sheets to W Z 145 and the PSU as boxes, the funnel on its nose and tail beams across the side sheets, both drones seated in it. "
                   "Out: the v1 ring frames, posts, brackets and bay modules (superseded by the v2 layout, whose bodies are not drawn), the hold-down, "
                   "the electronics. The nox repo's own master page of the box: https://claude.ai/artifact/RX2CexkfhB5UoQHnGTVgCC (27 Oct). "
                   "Imported by tools/import_box.py from sparkiscool/nox."),
             cg_ref=None, timeline="tools/landing_box.py:timeline", parts=parts)
    box_check(parts)
    out = os.path.join(ROOT, "systems", "box.assembly.json")
    if os.path.exists(out):
        old = json.load(open(out)); a["artifact"] = old.get("artifact")
    json.dump(a, open(out, "w"), indent=0, ensure_ascii=False)
    print(f"box: {len(parts)} parts -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
