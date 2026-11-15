"""Scout v4's assembly-check geometry (tools/assembly_check.py, systems/scout_v4.assembly.json "assembly"):
  rods(line)   the cage rods as they are cut: each from its bore floor to the far bore floor (parts/scout_v4_node_lib SOCKET_START, "rod
               end on the bore floor, s0 to s0"); the page's rods mesh runs 3 mm short of the node centres and stood 8.5 mm into every node.
  generate(a)  one assembly path per rod end: the rod's end (O3, from its bore floor to the mouth) pulled out of its socket along the rod,
               the socket's depth + 1: every bore open from its floor out past its mouth (the node's own generated material, a
               neighbour's collar, another rod, the frame). The rods go on after the turret and before the head (order 50).
"""
import os, sys
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _cage():
    import parts.scout_v4_node_lib as N
    return N


def _ends():
    """[(rod name, node, u away from the node, floor s0, depth)] for both ends of every rod"""
    N = _cage(); S = N.S; L = N.L
    P = {k: np.asarray(v, float) for k, v in S.NODES.items()}
    depth = {}
    for k in S.NODES:
        try:
            for m in N.members(k):
                depth[(k, m["name"])] = float(m["depth"])
        except Exception:
            pass
    out = []
    for i, j in list(S.RODS) + list(getattr(S, "TIES", [])):
        nm = f"{i}-{j}"; u = P[j] - P[i]; u = u / np.linalg.norm(u)
        for k, uk in ((i, u), (j, -u)):
            s0 = L.SOCKET_START.get((k, nm), L.NODE_R)
            if k == "RM" and hasattr(S, "RM_MOUTH"):
                s0 = float(S.RM_MOUTH.get(j if k == i else i, s0))
            out.append((nm, k, uk, float(s0), depth.get((k, nm), float(getattr(L, "DEPTH", 7.0)))))
    return P, out


def rods(line=None):
    P, ends = _ends()
    by = {}
    for nm, k, u, s0, d in ends:
        by.setdefault(nm, []).append(P[k] + u * s0)
    ms = [trimesh.creation.cylinder(radius=1.5, segment=[a, b], sections=24) for a, b in by.values()]
    return trimesh.util.concatenate(ms)


def generate(a):
    P, ends = _ends()
    paths = []
    node_of = {k: f"scout_v4_node_{k.lower()}" for k in P}
    for nm, k, u, s0, d in ends:
        f0, f1 = P[k] + u * s0, P[k] + u * (s0 + d)
        into = node_of[k]
        if not os.path.exists(os.path.join(ROOT, "parts", into)):
            into = "scout_v4_probe_turret" if k == "RM" else "scout_v4_frame"
        paths.append(dict(name=f"rod {nm} out of {k}", part="scout_v4_cage_rods", order=50, fit="bond",
                          body={"cylinder": {"p0": f0.round(3).tolist(), "p1": f1.round(3).tolist(), "r": 1.5}},
                          out=u.round(5).tolist(), length_mm=round(d + 1.0, 2), into=[into],
                          why="the rod's end slides out of its socket along the rod: the bore open from its floor past the mouth"))
    return dict(paths=paths)
