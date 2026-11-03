"""The xswap short lock v5 as a system of this repo (3 Nov, Spark: our own copy of the lock page, in the drones' style). The lock's
CAD lives in the xswap session; its page (https://claude.ai/artifact/H3KQecHZXuoXDumyBZo4TU) carries every mesh as JSON. This copies them
into parts/xswap_<slug>/part.stl in the Overseer drone frame (x nose, y left, z up, z 0 bottom plate top) and writes
systems/xswap.assembly.json, whose timeline (tools/xswap_motion.py) is the page's swap cycle. The Overseer round it is three merged
context meshes from systems/overseer.assembly.json (cage, frame and electronics, base), copied so the folders belong to this system.

Page frame -> drone frame: the page's y is the drone's x less 20 (the pack centre at x -20), its x the drone's -y, its z the drone's z
plus 14.1 (the holder's mount face on the base inserts, z -8.1). Pincer parts are stored at the cycle's rest pose (55 below the hold).

Run: python tools/import_xswap.py <dir holding the page's meshes/*.json>
"""
import base64, json, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
A = np.array([[0, 1.0, 0, -20.0], [-1.0, 0, 0, 0], [0, 0, 1.0, -14.1], [0, 0, 0, 1.0]])      # page -> drone
REST_DZ = -55.0                                                                                # the pincer at t 0 (page: dz)
FEET_Z = -93.2                                                                                 # the cage's lowest point, drone frame (cage nodes 0..5)
# page mesh file, part id, name, group, colour, role, material, bought listing / note
LOCK = [
    ("holder", "xswap_holder", "holder", "holder (on the drone)", "#8fa9c4", "structure", "PETG", "PETG, on the base's 4 M3 inserts (80 x 50 on z -8.1); two end plates and a ribbed deck carrying the XT60 male and the guide poles"),
    ("guide_poles", "xswap_guide_poles", "guide poles", "holder (on the drone)", "#9aa4ad", "bought", None, "4 steel poles, 20 long, locate the shell on the way up"),
    ("XT60_male", "xswap_xt60_male", "XT60 male (panel mount)", "holder (on the drone)", "#e0a53a", "bought", None, "XT60E1-M on the deck at the nose end; body rises 5.2 above the inserts, 0.8 under the bottom plate"),
    ("shell", "xswap_shell", "shell", "shell (carried with the pack)", "#4fb3a8", "structure", "PETG", "PETG, holds the pack, its XT60 and the two gate bars; the flared skirt under its bottom plate is the pincer's lead-in"),
    ("pack", "xswap_pack", "pack (Zeee 4S 5200)", "shell (carried with the pack)", "#d9c27a", "bought", None, "138 x 47 x 48, ~475 g (E until measured)"),
    ("XT60_female", "xswap_xt60_female", "XT60 female (pack plug)", "shell (carried with the pack)", "#e0a53a", "bought", None, "the pack's own plug, pocketed in the shell beside the pack's nose end"),
    ("bar_A", "xswap_gate_bar_a", "gate bar A", "shell (carried with the pack)", "#2b3540", "structure", "PETG", "lever under the pack floor, pivot at drone x 20; its far end seats in the holder's nose end plate, the finger swings it 17 deg"),
    ("bar_B", "xswap_gate_bar_b", "gate bar B", "shell (carried with the pack)", "#2b3540", "structure", "PETG", "mirror of A, pivot at drone x -60, seats in the tail end plate"),
    ("finger_A", "xswap_finger_a", "finger A", "pincer (box side)", "#c4561c", "structure", "PETG", "rack finger, 14.6 stroke toward B on the pinion"),
    ("finger_B", "xswap_finger_b", "finger B", "pincer (box side)", "#c4561c", "structure", "PETG", "rack finger, 14.6 stroke toward A"),
    ("pinion", "xswap_pinion", "pinion", "pincer (box side)", "#c4561c", "structure", "PETG", "r 13.5, drives both fingers"),
    ("lid", "xswap_lid", "lid", "pincer (box side)", "#6d8f6a", "structure", "PETG", "jaw lid: the fingers pass through it, its top meets the shell's bottom plate at the 2.3 lift"),
    ("jaw_base", "xswap_jaw_base", "jaw base", "pincer (box side)", "#6d8f6a", "structure", "PETG", "jaw base on the rail; carries the servo and the pinion"),
    ("servo", "xswap_servo", "servo", "pincer (box side)", "#3f4a55", "bought", None, "standard servo on the pinion"),
]
CTX = {  # id -> (name, the Overseer assembly ids merged into it); the battery holder v1 (a placeholder box) is left out: this lock replaces it
    "xswap_ctx_overseer_cage": ("Overseer: cage", lambda p: p.get("group") == "cage"),
    "xswap_ctx_overseer_base": ("Overseer: base", lambda p: p["id"] in ("overseer_base", "flow_clip")),
    "xswap_ctx_overseer_frame": ("Overseer: frame and electronics", lambda p: p.get("group") != "cage" and p["id"] not in ("overseer_base", "flow_clip", "overseer_battery_holder")),
}


def page_mesh(path):
    j = json.load(open(path))
    v = np.frombuffer(base64.b64decode(j["v"]), dtype=np.float32).reshape(-1, 3).astype(np.float64)
    f = np.frombuffer(base64.b64decode(j["f"]), dtype=np.uint32).reshape(-1, 3)
    return trimesh.Trimesh(v, f, process=False)


def save(m, pid):
    d = os.path.join(ROOT, "parts", pid); os.makedirs(d, exist_ok=True)
    m.export(os.path.join(d, "part.stl"))
    return f"parts/{pid}/part.stl"


def main(src):
    from build_site import pose_matrix, view_mesh
    parts = []
    for fn, pid, name, group, color, role, mat, note in LOCK:
        m = page_mesh(os.path.join(src, "meshes", fn + ".json"))
        if group.startswith("pincer"):
            m.apply_translation([0, 0, REST_DZ])
        m.apply_transform(A)
        mesh = save(m, pid)
        parts.append(dict(id=pid, name=name, role=role, group=group, color=color, mesh=mesh, pose=None, material=mat,
                          tag="C", owner="xswap lock session", version="short lock v5 (3 Nov)",
                          source="xswap page H3KQecHZXuoXDumyBZo4TU, tools/import_xswap.py", note=note))
    ov = json.load(open(os.path.join(ROOT, "systems", "overseer.assembly.json")))
    for pid, (name, pick) in CTX.items():
        ms = []
        for p in ov["parts"]:
            if not pick(p) or not p.get("mesh"):
                continue
            path = os.path.join(ROOT, p["mesh"])
            if not os.path.exists(path):
                continue
            m = trimesh.load(path, force="mesh")
            if len(m.faces) > 8000:
                m = m.simplify_quadric_decimation(face_count=8000)
            m.apply_transform(pose_matrix(p.get("pose"))); ms.append(m)
        m = trimesh.util.concatenate(ms)
        if len(m.faces) > 40000:
            m = m.simplify_quadric_decimation(face_count=40000)
        mesh = save(m, pid)
        parts.append(dict(id=pid, name=name, role="bought", group="Overseer", color="#aab4be", mesh=mesh, pose=None, material=None,
                          opacity=0.3, tag="C", owner="overseer", version="systems/overseer.assembly.json, current parts merged",
                          source="tools/import_xswap.py", note=f"{len(ms)} parts of the Overseer page, as context"))
    a = dict(name="xswap short lock", title="xswap Short Lock in the Overseer", readme_view="iso", artifact=None, machine="overseer",
             note=("The xswap short lock v5 on the Overseer's 4 base inserts, one swap cycle from the lock's CAD: the pincer rises, "
                   "the fingers close and swing both gate bars open, the tabs drop into their seats, the rail pulls the shell off the "
                   "XT60 and the guide poles, the carriage flips, then the reverse puts a pack back. Drone frame (x nose, y left, z up, z 0 "
                   "bottom plate top); the Overseer drawn see-through round it. Readouts: finger stroke, bar swing, rail, flip, and the "
                   "shell's lowest point over the cage feet (z -93.2): -9.5 at rest, the skirt under the shell's bottom plate. "
                   "Copied from the lock's own page by tools/import_xswap.py; the lock's CAD stays in the xswap session."),
             cg_ref=None, timeline="tools/xswap_motion.py:timeline", parts=parts)
    out = os.path.join(ROOT, "systems", "xswap.assembly.json")
    if os.path.exists(out):
        a["artifact"] = json.load(open(out)).get("artifact")
    json.dump(a, open(out, "w"), indent=0, ensure_ascii=False)
    print(f"xswap: {len(parts)} parts -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
