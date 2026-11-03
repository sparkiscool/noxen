"""The Scout cage as a clearance pattern (3 Nov, Spark: build the Scout's cage against the funnel in a simple clearance pattern).
Not a design: a rectangular 3 mm rod pattern whose every member clears the funnel of parts/box_funnel/make.py when the Scout is
seated, with the camera inside its nose. Rules it is built from (seat frame = drone frame):
  - nothing below the funnel mouth (z 7.7) outside the faces: the bottom ring sits at z 12, outside the rim (|x| <= 138.4, |y| <= 79.3)
  - the top ring clears the props (plane z 41.2, sweep +-4): z 60
  - the legs stand outside the funnel's legs (at x -139 / +142, y +-73) and end at z -36, 2 under the belly skirt, 53 over the deck
  - the nose holds the camera (lens (149, 0, 47)) 20 clear all round
Writes parts/scout_cage_pattern/part.stl (tubes r 1.5 + node spheres r 6, the look) and the nodes / rods into
systems/scout.assembly.json cage_shell. Run: python tools/scout_cage_pattern.py
"""
import json, os, re
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XB, YB, ZB, ZT = 150.0, 92.0, 12.0, 60.0            # bottom ring half-sizes and heights
XN, YN, ZN = 178.0, 42.0, 36.0                       # nose node pair (round the camera)
ZL = -36.0                                           # leg feet
NODES = {0: (XB, YB, ZB), 1: (XB, -YB, ZB), 2: (-XB, -YB, ZB), 3: (-XB, YB, ZB),
         4: (XB, YB, ZT), 5: (XB, -YB, ZT), 6: (-XB, -YB, ZT), 7: (-XB, YB, ZT),
         8: (XN, YN, ZN), 9: (XN, -YN, ZN),
         10: (XB, YB, ZL), 11: (XB, -YB, ZL), 12: (-XB, -YB, ZL), 13: (-XB, YB, ZL)}
RODS = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7),   # rings and posts
        (0, 5), (1, 6), (2, 7), (3, 4),                                                             # one diagonal per side face
        (8, 9), (0, 8), (4, 8), (1, 9), (5, 9),                                                     # the nose round the camera
        (0, 10), (1, 11), (2, 12), (3, 13)]                                                         # legs


def mesh():
    parts = []
    for i, j in RODS:
        a, b = np.array(NODES[i], float), np.array(NODES[j], float)
        parts.append(trimesh.creation.cylinder(radius=1.5, segment=[a, b], sections=12))
    for p in NODES.values():
        s = trimesh.creation.icosphere(subdivisions=1, radius=6.0); s.apply_translation(p); parts.append(s)
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    m = mesh(); d = os.path.join(ROOT, "parts", "scout_cage_pattern"); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
    L = sum(float(np.linalg.norm(np.array(NODES[i]) - np.array(NODES[j]))) for i, j in RODS)
    F = os.path.join(ROOT, "systems", "scout.assembly.json"); raw = open(F).read(); a = json.loads(raw)
    a["cage_shell"] = dict(nodes=[list(NODES[k]) for k in sorted(NODES)], rods=[list(r) for r in RODS], node_r_mm=6.0, rod_r_mm=1.5,
                           total_rods=len(RODS), open=[], note="clearance pattern of 3 Nov round the shared funnel (tools/scout_cage_pattern.py), not a design")
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    print(f"cage pattern: {len(NODES)} nodes, {len(RODS)} rods, {L / 1000:.2f} m of 3 mm rod ({L * 0.011:.0f} g) + nodes, extents {m.extents.round(0).tolist()}")
