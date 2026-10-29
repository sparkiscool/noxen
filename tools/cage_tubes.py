"""The cage's carbon as the page shows it and as it is cut: parts/cage_shell_tubes/part.stl (every tube and every 3 mm rod between
two cage nodes, from socket floor to socket floor) and the cut list systems/overseer.cage.json "rod_cuts_mm" {member: length}.
Each rod runs from its socket's start at one node to its socket's start at the other (parts/cage_node_lib.py SOCKET_START: 10.15 mm
from the node centre, or further out where two members meet at a small angle so that no bores overlap), so the cut length is the
node-to-node distance minus the two starts.   python tools/cage_tubes.py"""
import json, os, sys
import numpy as np, trimesh
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from parts.cage_node_lib import SOCKET_START, NODES, CAGE, KIND

out, cuts = [], {}
for a, b in CAGE["rods"]:
    nm = f"tube {a}-{b}"; p, q = NODES[a], NODES[b]; u = (q - p) / np.linalg.norm(q - p)
    s0, s1 = SOCKET_START[(a, nm)], SOCKET_START[(b, nm)]
    L = float(np.linalg.norm(q - p)) - s0 - s1; cuts[nm] = round(L, 1)
    out.append(trimesh.creation.cylinder(KIND["tube"]["bore_r"] - 0.1, L, sections=24, segment=np.vstack([p + u * s0, q - u * s1])))
for nm, (p, q) in CAGE["links"].items():
    p, q = np.array(p, float), np.array(q, float)
    da, db = np.linalg.norm(NODES - p, axis=1), np.linalg.norm(NODES - q, axis=1)
    if da.min() > 0.6 or db.min() > 0.6:
        continue                                                     # one end is not a cage node (spokes, core ties, head rods): not this mesh
    a, b = int(np.argmin(da)), int(np.argmin(db)); u = (q - p) / np.linalg.norm(q - p)
    s0, s1 = SOCKET_START[(a, nm)], SOCKET_START[(b, nm)]
    L = float(np.linalg.norm(q - p)) - s0 - s1; cuts[nm] = round(L, 1)
    out.append(trimesh.creation.cylinder(KIND["link"]["bore_r"] - 0.1, L, sections=16, segment=np.vstack([p + u * s0, q - u * s1])))
m = trimesh.util.concatenate(out); m.export(os.path.join(ROOT, "parts", "cage_shell_tubes", "part.stl"))
# the 3 mm rods with one end off the cage (spokes to the arm nodes, core and bay ties to the hub, head rods): parts/spokes_ties_head_rods
# (29 Oct: the page still drew the v5 layout model's rods, through tubes that moved in v6). Far end: the rod ends NODE_R into its
# socket there (the arm lib's rule; the hub and head sockets likewise, E).
rods = []
for nm, (p, q) in CAGE["links"].items():
    p, q = np.array(p, float), np.array(q, float)
    da, db = np.linalg.norm(NODES - p, axis=1), np.linalg.norm(NODES - q, axis=1)
    if da.min() < 0.6 and db.min() < 0.6:
        continue                                                     # both ends on cage nodes: in the tube mesh above
    if db.min() < 0.6:
        p, q, da, db = q, p, db, da                                  # p is the cage-node end (if any)
    s0 = SOCKET_START[(int(np.argmin(da)), nm)] if da.min() < 0.6 else float(CAGE["node_r_mm"])
    s1 = float(CAGE["node_r_mm"]); u = (q - p) / np.linalg.norm(q - p); kind = "link"
    if nm.startswith("head rod"):
        # 6/4 tube into the head mount: the mount's socket mouth is at the link point and its bore runs 15 deep into the
        # mount, away from the cage (parts/overseer_head_mount/part.py Sock, mouth_at_end), so the tube starts 15 inside
        # (29 Oct, Spark: the rods floated 10 mm short of the mount)
        kind, s1 = "tube", -KIND["tube"]["depth"]
    L = float(np.linalg.norm(q - p)) - s0 - s1; cuts[nm] = round(L, 1)
    rods.append(trimesh.creation.cylinder(KIND[kind]["bore_r"] - 0.1, L, sections=16, segment=np.vstack([p + u * s0, q - u * s1])))
# the arm tubes (29 Oct, tube arms): two 6/4 tubes per arm from the root block's socket floors to the pad's, arms 3 / 4 mirrored in y
from parts.overseer_tube_arm_lib import TubeArm, TUBE_BORE_R
for nm, arm in (("arm 1", TubeArm("front", 45.0, 0, ["spoke 1.1", "spoke 1.3"], [178.94, 146.7, 8.0], "root")),
                ("arm 2", TubeArm("rear", 135.0, 1, ["spoke 2.1", "spoke 2.3"], [-53.26, 119.7, 25.0], "root"))):
    for (p, q), tag in zip(zip(arm.tube_points("root"), arm.tube_points("pad")), ("lo", "hi")):
        for sgn, k in ((1, nm), (-1, {"arm 1": "arm 4", "arm 2": "arm 3"}[nm])):
            pp, qq = p * np.array([1, sgn, 1]), q * np.array([1, sgn, 1]); L = float(np.linalg.norm(qq - pp))
            cuts[f"{k} tube {tag}"] = round(L, 1)
            rods.append(trimesh.creation.cylinder(TUBE_BORE_R - 0.1, L, sections=24, segment=np.vstack([pp, qq])))
mr = trimesh.util.concatenate(rods); mr.export(os.path.join(ROOT, "parts", "spokes_ties_head_rods", "part.stl"))
print(f"{len(rods)} rods off the cage -> parts/spokes_ties_head_rods/part.stl ({len(mr.faces)} faces)")
path = os.path.join(ROOT, "systems", "overseer.cage.json"); s = open(path).read(); c = json.loads(s)
block = json.dumps(cuts, indent=1).replace("\n", "\n ")
if '"rod_cuts_mm"' in s:
    i = s.index('"rod_cuts_mm"'); j = s.index("}", i) + 1
    s = s[:i] + '"rod_cuts_mm": ' + block + s[j:]
else:
    i = s.rindex("}"); s = s[:i].rstrip().rstrip(",") + ',\n "rod_cuts_mm": ' + block + "\n}" + s[i + 1:]
json.loads(s); open(path, "w").write(s)
print(f"{len(out)} members, {len(m.faces)} faces; cuts {min(cuts.values()):.0f}..{max(cuts.values()):.0f} mm, total {sum(cuts.values()) / 1000:.2f} m")
