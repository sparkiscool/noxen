"""Print the names of parts that need a run: part.py exists but result.json is missing or was made from an older part.py or lib (tools/part_sha.py)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_sha import stale
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def socket_mismatch(name, stl=None):
    """a cage node whose mesh does not hold the sockets its setup cuts (3 Nov, Spark: node 16's head rod socket was 3.2 while the
    head mount's is 6.2; nodes 13 and 15 had been stamped current by the file rule after a lib edit, from older setups): the bore
    radius mid-socket, 3 directions round each member, against cage_node_lib.members. Returns the mismatched members."""
    if not name.startswith("overseer_cage_node_"):
        return []
    import numpy as np, trimesh
    sys.path.insert(0, ROOT)
    import parts.cage_node_lib as L
    n = int(name.rsplit("_", 1)[1]); stl = stl or os.path.join(ROOT, "parts", name, "part.stl")
    if not os.path.exists(stl):
        return []
    try:                                                         # the bores to 0.35 mm: the full print mesh, not the viewing one (9 Nov)
        import print_mesh; stl = print_mesh.local(name) if stl.endswith(os.path.join(name, "part.stl")) else stl
    except Exception as e:
        print(f"stale_parts: {name}: print mesh not fetched ({e}), the viewing mesh measured", file=sys.stderr)
    m, bad = trimesh.load(stl), []
    for mm in L.members(n):
        u = mm["u"]; e1 = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.99 else np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        P = L.NODES[n] + u * (mm["s0"] + 0.5 * mm["depth"])
        rs = [next((t for t in np.arange(0.1, 7, 0.1) if m.contains([P + (np.cos(a) * e1 + np.sin(a) * e2) * t])[0]), None) for a in (0, 2.1, 4.2)]
        r = min([x for x in rs if x is not None], default=None)
        if r is None or abs(r - mm["bore_r"]) > 0.35:
            bad.append(mm["name"])
    return bad


if __name__ == "__main__":
  out = []
  for py in sorted(glob.glob(os.path.join(ROOT, "parts", "*", "part.py"))):
    d = os.path.dirname(py); rj = os.path.join(d, "result.json")
    if not os.path.exists(rj) or stale(py, json.load(open(rj))) or socket_mismatch(os.path.basename(d)):
        out.append(os.path.basename(d))
  print(" ".join(out))
