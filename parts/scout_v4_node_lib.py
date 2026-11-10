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
import numpy as np

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

# 10 Nov, round 2 (Spark 9 Nov: "the nodes on this seem very bulky, we are using pctg ... why does the node opening have to be the same
# thickness ... this isnt optimal"): the light nodes' mass was mostly their drawn collars (2.2 g drawn of 4.3), each 10 deep whatever its
# load. Now each socket is as deep as its own load needs, its collar at the audit's wall, the room clear of the mouth plane.

# Socket depth per socket, from its frame-model loads (systems/scout_v4.cage_loads.json, the worse of the 35 g crash set and the
# head-first crash; the node's own longest-rod case uses the same two numbers). The resin is West System G/flex 650 (dimensions.md).
#   Pull-out: the rod's bond to the resin, over the bore less its 2.0 lip (the lead-in and the plain bore at the mouth). G/flex 650 on
#   sanded G-10 (glass/epoxy laminate, the nearest substrate to the carbon/epoxy rod) reads 3,459 psi = 23.8 MPa in West System's
#   tensile adhesion (PATTI) table (westsystem.com G/flex TDS, E for this joint): half of it, 12 MPa, for a rod in shear with its
#   peel at the mouth, and x2 on the crash loads (already the worst case): 12 * pi * 3.0 / 2 = 56.5 N per mm of bond. The resin
#   sleeve itself cannot leave the print: it is cast in the 0.5 annulus and the grooves under the lip (a key, not a bond: no PCTG
#   adhesion figure is needed). The 33-43 N pull-outs of the P / R nodes need 0.8 mm; 96 N 1.7 mm; DN-NB's 645 N 11.4 mm.
#   Side: the side force V at the mouth is a couple on the bore, peak bearing 4 V / (D d) (a rigid pin in an elastic socket, no
#   lever); allowed 0.3 x the across-layer Zt 23 x 0.85 / 1.1 (the engine's own crash strength at the node limit) = 5.3 MPa:
#   d >= V / 4.3, 5.3 mm at the worst hub socket (22.8 N), 9.1 at BT-TB (38.9 N).
#   Floor: 7.0, the injection pattern (parts/socket_inject.py) at its least: the O2.5 fill port in the 2.5 feed ring, 1.0 of wall,
#   the O1.5 vent, the 2.0 lip; its two grooves, the annulus and its three lands scale with the depth.
# Every cage-rod socket comes to 7.0 (was 10); the Ys' frame sockets DN-NB / DT-TB 13.5 and BN-NB / BT-TB 11.5 (were 10: at 10 the
# 645 N bond had x1.4, not x2). A rod's cut length does not change (rod end on the bore floor, s0 to s0).
BOND_N_PER_MM = 12.0 * np.pi * 3.0 / 2.0
LIP_MM, BEAR_MPA, DEPTH_MIN = 2.0, 0.3 * 23.0 * 0.85 / 1.1, 7.0


def socket_depth(axial, side):
    d = max(DEPTH_MIN, LIP_MM + abs(axial) / BOND_N_PER_MM, 4.0 * abs(side) / (2 * L.BORE_R * BEAR_MPA))
    return float(np.ceil(d * 2.0) / 2.0)


_raw0 = L._raw_members


def _raw_members(k, bridges=False):
    out = _raw0(k, bridges)
    for m in out:
        f = L.LOADS.get(k, {}).get(m["name"]) or {}
        m["depth"] = socket_depth(f.get("axial", 0.0), f.get("side", 0.0)) if f else L.DEPTH
    return out


L._raw_members = _raw_members
L.DEPTH = DEPTH_MIN                                       # a socket with no frame-model load (none here: check_members refuses it)

# The drawn collar: the audit's wall. 2.55 round the 3.2 bore, 2.05 at the 0.5 grooves and annulus (the audit's drawn_wall floor 2.0;
# the 0.05 is the 40-gon's flats, 1.99 between a flat and a vertex). The Ys' two fork collars keep 3.0 (9 Nov run 4: the whole root
# of the fork, worst at their floors). The room round every collar stays the 3.0-wall strut + STRUT_M, so the engine grows the root
# wherever bending needs it. v4-only: the v3 Scout and Overseer nodes keep WALL.
FORK_S0 = 20.0                                           # a bore pushed out past this by a near-parallel neighbour is one of a Y's two fork sockets
HUB_WALL = 2.55
L.DRAWN_WALL = lambda m: L.WALL if m["s0"] > FORK_S0 else HUB_WALL

# The collar's outside is round with only the wall over the bore's crown (the crown's apex r sqrt 2 + HUB_WALL), not the bore's
# teardrop scaled to the outside (apex (r + wall) sqrt 2): that put a sharp 1.7 mm ridge along the top of every collar, ending in
# a tick at each mouth (Spark's fins, 9 Nov). The top of a collar is an up-facing surface, it needs no teardrop to print.
_collar0 = L.collar_shape


def collar_shape(c, u, s0, s1, r, wall, build=(0.0, 0.0, 1.0)):
    from parts.cage_node_lib import crown_dir, _prism
    d = crown_dir(u, build)
    if d is None:
        return _collar0(c, u, s0, s1, r, wall, build)
    u = np.asarray(u, float) / np.linalg.norm(u); e1 = d; e2 = np.cross(u, e1); ro, apex = r + wall, r * np.sqrt(2.0) + wall
    t0 = np.arccos(ro / apex)                             # where the apex's tangents touch the circle
    a = np.linspace(t0, 2 * np.pi - t0, 48)
    return _prism([(ro * np.cos(t), ro * np.sin(t)) for t in a] + [(apex, 0.0)], c, u, s0, s1, e1, e2)


L.collar_shape = collar_shape

# The mouth plane: the collar's end face, the strut's room and the keep-out round the rod all ended on it, and the export left
# sliver sheets where the generated surface met those coplanar faces (the fins in the mouths of P5, P1, R0: P1's S0-P1 lead-in held
# an open sheet in the print mesh). The room now ends 1.0 short of the mouth (the last mm is the drawn collar alone) and the keep-out
# starts 0.5 past it (the lead-in cone, cut to 1.0 past the mouth, covers the gap).
L.STRUT_BACK, L.KEEP_OFF = 1.0, 0.5
# Round 3 (10 Nov, the design session on the page: R0, P1, P3, P5): on every node with a pair of side-by-side sockets each mouth had
# ragged stepped notches, and on every node a drawn ring stood proud of the barrel at the mouth. Cause: each socket's strut room
# (r 8.6) reached past its neighbour's mouth plane, over that mouth, where the neighbour's keep-out and lead-in carved what grew
# there; and the room ended 1.0 short of the mouth, so the collar's last mm stood bare. Now every socket has a mouth cut, the room's
# radius + 0.5, from its mouth plane out along the rod (other collars spared by 0.5): no room survives beyond any mouth plane, and the
# room runs 2.0 past the mouth into the cut, so the mouth is one exact cut face, the barrel meeting the collar on it (no room face on
# the plane: round 2's sliver sheets). The collar ends on the plane (drawn 0.5 into the cut, the Ys' shorter fork collar kept a stub
# beside its neighbour, inside the 0.5 its cut spares).
L.STRUT_BACK, L.COLLAR_PAST = -2.0, 0.0
L.MOUTH_R = L.BORE_R + L.WALL + L.STRUT_M + 0.5

# The Ys' forks (9 Nov): their bores side by side need only the audit's wall between the grooves (2.1). A hub's pair of near-parallel
# bores (P1's RS0-P1 / S0-P1 13 mm out) at 2.1 put each groove 0.05 off its neighbour's 2.55 collar face: near-coincident faces,
# the ticks in P1's bores. A hub pair gets 2.4 (0.35 off); a fork pair keeps 2.1 (its 3.0 collars 0.4 into the groove).
L.MAX_START = 36.0
L.BETWEEN_WALL = 2.1
BETWEEN_HUB = 2.4


def _socket_starts():
    st = {}
    for k in L.sc.NODES:
        ms = L._raw_members(k, bridges=True)
        for m in ms:
            def start(bw):
                s0 = L.NODE_R
                for o in ms:
                    if o is m:
                        continue
                    t = float(np.arccos(np.clip(m["u"] @ o["u"], -1.0, 1.0)))
                    if t > 1e-6:
                        s0 = max(s0, (m["bore_r"] + o["bore_r"] + 2 * L.GROOVE_D + bw) / (2.0 * np.sin(t / 2.0)))
                return s0
            s0 = start(L.BETWEEN_WALL)
            if L.NODE_R + 1e-6 < s0 <= FORK_S0:
                s0 = start(BETWEEN_HUB)
            st[(k, m["name"])] = float(np.ceil(s0 * 10.0) / 10.0)
    return st


L.SOCKET_START = _socket_starts()

# Budgets (10 Nov): the light nodes read 6-40 x under the 0.3 limit at 1.1 x skeleton (P / R 0.017-0.054 at 1.7 cm3, RF / RB / RS
# 0.005-0.014 at 1.2): their collars now overlap at the hub and carry most of the load drawn. BUDGET_CM3 sets their generated volume
# outright toward ~0.25 (index ~ volume^-1.25), above what joins the collars' roots. The Ys (S0 / S1 0.20 / 0.23, NB 0.20, TB 0.23)
# keep the volume they passed at.
# Run 200 (round 2, 10 Nov): all 16 pass. P / R at 0.9: 0.018-0.027, P3 / P4 at 1.0: 0.091, RF / RB / RS at 0.4: 0.030-0.043, S0 / S1
# at 3.6: 0.209 / 0.210, NB at 4.9: 0.220, TB at 5.5: 0.210. Round 3: the Ys toward 0.25 (index ~ V^-1.25: x0.87), P3 / P4 0.7
# (~0.14), P / R 0.6 (two 6.25 members from the hub collar to the paired collars' floors are ~0.6 cm3), RF / RB / RS 0.3.
BUDGET_CM3 = {"P3": 0.7, "P4": 0.7, "P1": 0.6, "P2": 0.6, "P5": 0.6, "P6": 0.6, "R0": 0.6, "R7": 0.6,
              "RF": 0.3, "RB": 0.3, "RS0": 0.3, "RS1": 0.3, "S0": 3.1, "S1": 3.1, "NB": 4.3, "TB": 4.8}


def spec_for(group):
    s = L.spec_for(group)
    key = group if isinstance(group, str) else group[0]
    s["gen_cm3"] = BUDGET_CM3[key]
    return s


KEYS = L.KEYS
make, build, members, check_members = L.make, L.build, L.members, L.check_members
