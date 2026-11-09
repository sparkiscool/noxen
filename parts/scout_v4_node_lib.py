"""Scout v4 cage node setups (9 Nov, Spark: "generate everything with topology"): the Scout's node generator (parts/scout_node_lib.py,
after the Overseer's parts/cage_node_lib.py) on the v4 cage of tools/scout_v4_shape.py.

Drawn: only the socket collars and floors (3.0 wall, 3.0 floor round each 3.2 x 10 bore, the injection pattern of parts/socket_inject.py);
everything joining them is generated inside a strut per member and a hub ball. Loads: the v4 frame model's per-member axial and side
forces (systems/scout_v4.cage_loads.json, written by tools/scout_v4_shape.py: per member the worse of the standard 35 g crash set and
the head-first crash onto the parked probe head). Supports: the two stiffest members' sockets. The v4 nodes all sit above the funnel's
top (z -4.3), so the v21 funnel sweep is not cut. The sockets printed into the frame and the base (PRINTED) and RM (printed with the
probe's turret, TIED) are not node parts.
Run: parts/scout_v4_node_<key>/part.py -> setup(h) = make("<KEY>")."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import scout_v4_shape as S                                # noqa: E402
import scout_cage as sc                                   # noqa: E402

sc.NODES, sc.RODS, sc.TIES, sc.TIE_ROOTS = dict(S.NODES), list(S.RODS), [], {}
sc.PRINTED = {k: "frame" for k in tuple(S.PRINTED) + tuple(S.TIED)}
sc.BRIDGES = []; sc.GROUPS = sc._groups(); sc.GROUP_OF = {k: g for g in sc.GROUPS for k in g}; sc.RODS_CARBON = list(S.RODS)
sc.group_id = lambda g: "scout_v4_node_" + "_".join(k.lower() for k in ((g,) if isinstance(g, str) else g))

import parts.scout_node_lib as L                          # noqa: E402  (reads the patched cage at import: members, socket starts)

L.LOADS = json.load(open(os.path.join(ROOT, "systems", "scout_v4.cage_loads.json")))["node_loads"]
L.P = {**sc.NODES}
L.KEYS = [k for k in sc.NODES if k not in sc.PRINTED]
L.SOCKET_START = L._socket_starts()


def _no_funnel():
    import manifold3d as mf
    return mf.Manifold()


L.funnel_sweep = _no_funnel
KEYS = L.KEYS
make, spec_for, build, members, check_members = L.make, L.spec_for, L.build, L.members, L.check_members
