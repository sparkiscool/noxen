"""The box (Nox Ground Unit) as a system of this repo, from the nox repo's assembly files (3 Nov, Spark: the funnel with the most
up-to-date box mechanism CAD, as real as it gets, one artifact other sessions can reach). Copies the real meshes into
parts/box_<slug>/part.stl (a folder belongs to one system) and writes systems/box.assembly.json in the box W frame:
  swap arm v7      nox/box/box_swap_arm.assembly.json (55 parts: hub, stub, rails, heads, grabbers, detent, drives, the 9 bay
                   receivers and two packs; meshes runs/arm/v7/asm), poses as in the file (cycle t 0, park)
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
SEAT_TO_W = [[0, -1.0, 0, 0], [1.0, 0, 0, 167.1], [0, 0, 1.0, 155.3], [0, 0, 0, 1.0]]   # catch_system.py: Y_FUN 167.1, Z_SEAT 155.3 (C)
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


def main(nox):
    parts = []
    arm = json.load(open(os.path.join(nox, "nox", "box", "box_swap_arm.assembly.json")))
    for p in arm["parts"]:
        nm = p["name"]; key = next((k for k in GROUP if nm.startswith(k)), None)
        grp, col = GROUP.get(key, ("swap arm", "#99a6b3"))
        bought = any(nm.startswith(b) for b in BOUGHT)
        pid = slug("arm " + nm)
        mesh, nf = copy_mesh(os.path.normpath(os.path.join(nox, "nox", "box", p["mesh"])), pid)
        parts.append(dict(id=pid, name=f"arm: {nm}", role="bought" if bought else "structure", group=grp, color=col, mesh=mesh,
                          pose=p.get("pose"), material=None if bought else "PETG", mass_g=p.get("mass_g"), tag=p.get("tag", "C"),
                          owner="box swap arm v7", version="v7", source="nox/box/box_swap_arm.assembly.json (runs/arm/v7/asm)",
                          note=p.get("note", "")))
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
        parts.append(dict(id=pid, name=f"enclosure: {p['name']}", role="bought", group="enclosure", color="#c9d6e2" if p["name"].startswith("sheet") else "#777777",
                          box=p["box"], opacity=0.25 if p["name"].startswith("sheet") else None, mass_g=p.get("mass_g"), tag="C",
                          owner="box structure v2", version="v2 layout", source="nox/box/box_structure_v2.assembly.json", note=p.get("note", "")))
    parts.append(dict(id="box_funnel", name="funnel (latch-footprint seat, 55 deg)", role="structure", group="funnel", color="#e07020", mesh="parts/box_funnel/part.stl",
                      pose=SEAT_TO_W, material="PETG", tag="E", owner="box catch", version="v1 drawn 3 Nov", source="parts/box_funnel/make.py",
                      note="centres on the receiver v3 footprint 106 x 66 so both drones seat at one height; mouth 204 x 164, faces 55 deg, 70 tall; hold-down not drawn"))
    for src, nm in (("receiver_v3_latch", "receiver v3 (seated Scout)"), ("cartridge_v6_latch", "cartridge v6 (seated Scout)")):
        pid = slug("ctx " + nm.split(" (")[0])
        mesh, nf = copy_mesh(os.path.join(ROOT, "parts", src, "part.stl"), pid)
        parts.append(dict(id=pid, name=nm, role="protected", group="seated drone (context)", color="#8899aa", mesh=mesh, pose=SEAT_TO_W,
                          material="PETG", tag="M", owner="battery latch", version="v3 / v6", source=f"parts/{src} (this repo), at the seat",
                          note="the latch set every drone carries: what the funnel seats"))
    a = dict(name="Nox Ground Unit", title="Nox Ground Unit", readme_view="iso", artifact=None, machine="box",
             note=("Box W frame (X across, Y along the axle toward the front, Z up; the swap arm hub at the origin, the drone seat at (0, 167.1, 155.3), "
                   "nox/box/catch_system.py SEAT_TO_W). In: the swap arm v7 at park with its 9 bay receivers and two packs, the stand of the built demo, "
                   "the v2 enclosure sheets and PSU as boxes, the funnel drawn 3 Nov on the latch footprint, the Scout's receiver and cartridge seated in it. "
                   "Out: the v1 ring frames, posts, brackets and bay modules (superseded by the v2 layout, whose bodies are not drawn), the hold-down, "
                   "the electronics. The nox repo's own master page of the box: https://claude.ai/artifact/RX2CexkfhB5UoQHnGTVgCC (27 Oct). "
                   "Imported by tools/import_box.py from sparkiscool/nox."),
             cg_ref=None, parts=parts)
    out = os.path.join(ROOT, "systems", "box.assembly.json")
    if os.path.exists(out):
        old = json.load(open(out)); a["artifact"] = old.get("artifact")
    json.dump(a, open(out, "w"), indent=0, ensure_ascii=False)
    print(f"box: {len(parts)} parts -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
