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
path = os.path.join(ROOT, "systems", "overseer.cage.json"); s = open(path).read(); c = json.loads(s)
block = json.dumps(cuts, indent=1).replace("\n", "\n ")
if '"rod_cuts_mm"' in s:
    i = s.index('"rod_cuts_mm"'); j = s.index("}", i) + 1
    s = s[:i] + '"rod_cuts_mm": ' + block + s[j:]
else:
    i = s.rindex("}"); s = s[:i].rstrip().rstrip(",") + ',\n "rod_cuts_mm": ' + block + "\n}" + s[i + 1:]
json.loads(s); open(path, "w").write(s)
print(f"{len(out)} members, {len(m.faces)} faces; cuts {min(cuts.values()):.0f}..{max(cuts.values()):.0f} mm, total {sum(cuts.values()) / 1000:.2f} m")
