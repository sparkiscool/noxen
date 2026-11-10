"""Print the names of parts that need a run: part.py exists but result.json is missing or was made from an older part.py or lib (tools/part_sha.py)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_sha import stale
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def socket_mismatch(name, stl=None):
    """a cage node whose mesh does not hold the sockets its setup cuts (3 Nov, Spark: node 16's head rod socket was 3.2 while the
    head mount's is 6.2; nodes 13 and 15 had been stamped current by the file rule after a lib edit, from older setups): the bore
    radius of each member against cage_node_lib.members' bore_r. Every cut round a bore (the injection annulus between its 3 lands,
    feed ring, grooves, port, vent, crown, lead-in) only widens it, and the lip at the mouth and the lands are at the bore radius,
    so the radius is the nearest wall over rays from the axis, 24 round at 12 depths along the socket (9 Nov: one depth mid-socket
    landed in the annulus, bore_r + 0.5, and flagged all 21 nodes). Works on the viewing mesh. Returns the mismatched members."""
    if not name.startswith("overseer_cage_node_"):
        return []
    import numpy as np, trimesh
    sys.path.insert(0, ROOT)
    import parts.cage_node_lib as L
    n = int(name.rsplit("_", 1)[1]); stl = stl or os.path.join(ROOT, "parts", name, "part.stl")
    if not os.path.exists(stl) or not os.path.exists(os.path.join(ROOT, "parts", name, "part.py")):   # no part.py: a mirror instance of another node
        return []
    m, bad = trimesh.load(stl), []
    A = np.linspace(0, 2 * np.pi, 24, endpoint=False); F = np.linspace(0.1, 0.95, 12)
    for mm in L.members(n):
        u = np.asarray(mm["u"], float); e1 = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.99 else np.cross(u, [1.0, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
        O = np.repeat([L.NODES[n] + u * (mm["s0"] + f * mm["depth"]) for f in F], len(A), axis=0)
        D = np.tile([np.cos(a) * e1 + np.sin(a) * e2 for a in A], (len(F), 1))
        hit, ray, _ = m.ray.intersects_location(O, D, multiple_hits=False)
        r = np.linalg.norm(hit - O[ray], axis=1).min() if len(ray) else None
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
