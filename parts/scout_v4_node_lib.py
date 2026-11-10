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

# The drawn collar (9 Nov, Spark: "the nodes seem very bulky ... why does the node opening have to be the same thickness"): only the
# bore's minimum wall is drawn, 2.6 round the 3.2 bore (2.1 at the 0.5 resin annulus, over the audit's 2.0), floor to mouth; the room
# round it stays the 3.0-wall strut + STRUT_M, so the engine grows the root where the collar's bending needs it. Numbers (worst socket,
# TB's BT-TB / DT-TB: side 42.5 N, axial 454 N): mouth bearing, the side force on the outer quarter of the bore (2.5 x 3.0), 5.7 MPa;
# root bending of the bare 2.1 wall alone (O8.2 / O4.2 tube, Z 50 mm3, 13 mm lever) 11 MPa + 454 N on its 39 mm2 12 MPa: under PCTG's
# ~25 MPa across layers before any generated root, which the engine adds and its strength index checks. The 3.0 wall drew ~43 % of
# every node's mass for a wall the mouth does not need. v4-only: the v3 Scout and Overseer nodes keep WALL (their setups unchanged).
FORK_S0 = 20.0                                           # a bore pushed out past this by a near-parallel neighbour is one of a Y's two fork sockets
L.DRAWN_WALL = lambda m: L.WALL if m["s0"] > FORK_S0 else 2.6
# 9 Nov, run 4: at NB / TB / S0 / S1 (the Ys) the two fork sockets are the node's supports and its whole root: every other rod's load
# reaches them as bending down a 29-34 mm fork onto two side-by-side collars, and the worst (0.35-0.43) sat in the 2.6 drawn wall at
# their floors. Those two collars keep the 3.0 wall (section modulus x ~1.3); every other socket, a collar off a hub, stays 2.6.
# The support / load regions stay the lib's (from 0.5 behind the floor): bore-only regions read 0.43 at TB against 0.32 (run 3).

# Budgets (9 Nov): the 1.54 x skeleton of spec_for was sized for strength at limit 0.3, but the v4 nodes came in at 0.005-0.18. The
# light nodes (worst <= 0.042) go back to 1.1 x, the budget that joins every collar without flakes (5 Nov); TB / NB / S0 / S1 (0.16-
# 0.18) to 1.25 x: an index ~ volume^-1.25 puts 1.54 -> 1.25 at ~0.23, short of the 0.25 target to leave the thinner collar its share.
# The frame's paired sockets (9 Nov, the frame and base as one body inside the A1 mini's 180: DN/BN and DT/BT 17.2 apart at x +-75.5,
# DS/BS 28.7 apart) are 12.3 deg apart from NB / TB and 14.2 from S0 / S1 (17 before), so their bores start 34 / 29 mm out. The bed
# and the funnel fix those sockets; a short Y is the honest node here, its two bores side by side and the engine growing the fork
# (the 25 mm limit was set on 5 Nov for ties 39-85 mm out, 290k-586k cells; these stay under 36 mm). v4-only.
L.MAX_START = 36.0
# 10 Nov, run 5: TB 0.320 (96 N pull-out) just outside its fork collar's floor, NB 0.278 at 77 N: the moment down the fork. The two fork
# collars are one drawn body, so their bores need only the audit's wall between the grooves (2.1, as the hub collars), not a whole
# WALL: the fork 33.5 -> 29.4 at NB / TB, 29.2 -> 25.6 at S0 / S1, the lever 12 % shorter.
L.BETWEEN_WALL = 2.1
L.SOCKET_START = L._socket_starts()                       # recomputed with it (the import above computed them with WALL)

# 9 Nov, run 2 (the frame's sockets moved: NB / TB / S0 / S1 are Ys held at the fork's two sockets 29-34 mm out, the load path from the
# cage rods' sockets runs through the hub and the fork): at 1.25 x NB 0.357 / TB 0.378 (fail), S0 0.266 / S1 0.283. Index ~ V^-1.25:
# NB / TB 1.25 -> 1.7, S0 / S1 1.25 -> 1.35, each toward 0.25.
BUDGET_X = {"TB": 1.9, "NB": 1.7, "S0": 1.35, "S1": 1.35}


def spec_for(group):
    s = L.spec_for(group)
    key = group if isinstance(group, str) else group[0]
    s["gen_cm3"] = round(BUDGET_X.get(key, 1.1) * L.skeleton_cm3(group), 1)
    return s


KEYS = L.KEYS
make, build, members, check_members = L.make, L.build, L.members, L.check_members
