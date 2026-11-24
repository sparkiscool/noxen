"""Scout v4 cage node setups (9 Nov, Spark: "generate everything with topology"): the Scout's node generator (parts/scout_node_lib.py,
after the Overseer's parts/cage_node_lib.py) on the v4 cage of tools/scout_v4_shape.py.

Drawn: only the socket collars and floors (3.0 wall, 3.0 floor round each 3.2 x 10 bore, the injection pattern of parts/socket_inject.py),
two sockets under TWIN_DEG (45 deg) apart as one twin-bore socket, shared wall and outer walls from the floor to the mouths (24 Nov);
everything joining them is generated inside a strut per member and a hub ball.
The shared wall by hand (shared_wall; the whole account at TWIN_DEG below): a rod's end moment and side force at mid-bore,
Me = M + V d / 2, are the couple Fc = Me / (2 L3) on its lining's thirds (L3 = (d + 3.5) / 3); with the neighbour's rod held the mouth
third's Fc crosses the wall between the bores in tension across the layers, over A = w Lb less the core pocket (w the twin's height
between the bores, Lb the mouth third the neighbour stands beside, flanges 3.0-3.5 over and under the pocket that keeps the drawn
body under the layout's 6 mm): Fc / A <= 0.27 x 17.8 = 4.8 MPa. The worst end moment, BT1-TB 1434 N mm + 53.4 N on 13.0 deep:
Me 1781 N mm, Fc 162 N, needs 33.7 mm2, has 34.0 (w 9.2, Lb 5.0, flanges 3.4): index 0.268. The worst wall, P3's TB-P3 858 N mm:
Fc 127 N, 26.4 of 26.4 mm2 with RB-P3 15.5 deep: 0.269. Loads: the v4 frame model's per-member axial and side
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
from parts.cage_node_lib import GROOVE_D                  # noqa: E402

L.LOADS = json.load(open(os.path.join(ROOT, "systems", "scout_v4.cage_loads.json")))["node_loads"]
L.P = {**sc.NODES}
L.KEYS = [k for k in sc.NODES if k not in sc.PRINTED]
L.SOCKET_START = L._socket_starts()



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


#   End moment (14 Nov: the frame model's rod end moments, checked nowhere before): the moment M with the side force V bears on the resin
#   sleeve at peak 6 M / (D d^2) + 4 V / (D d) (the same rigid pin; the old 2000 N mm cap was this at 40 MPa over 10 deep), held to the
#   G/flex 650 compression yield 5,270 psi = 36.3 MPa (ASTM D-695, West System G/flex TDS) on the rod's D 3.0: parts/scout_v4_units_lib.
#   moment_depth. The printed collar behind it is the engine's: every socket's side case carries the moment as a couple across the bore.
RESIN_BEAR_MPA = 36.3


def socket_depth(axial, side, moment=0.0):
    sD = RESIN_BEAR_MPA * 3.0; dm = (4 * abs(side) + np.sqrt(16 * side * side + 24 * sD * abs(moment))) / (2 * sD)
    d = max(DEPTH_MIN, LIP_MM + abs(axial) / BOND_N_PER_MM, 4.0 * abs(side) / (2 * L.BORE_R * BEAR_MPA), dm)
    return float(np.ceil(d * 2.0) / 2.0)


# The backed socket of a twin (24 Nov): a twin's neighbour is as deep as the shared wall's hand calculation asks (the twin-bore socket
# below: shared_wall, _back_depths).
BACK_DEPTH = {}
_raw0 = L._raw_members


def _raw_members(k, bridges=False):
    out = _raw0(k, bridges)
    for m in out:
        f = L.LOADS.get(k, {}).get(m["name"]) or {}
        m["depth"] = max(socket_depth(f.get("axial", 0.0), f.get("side", 0.0), f.get("moment", 0.0)) if f else L.DEPTH, BACK_DEPTH.get((k, m["name"]), 0.0))
        m["root_max"] = ROOT_FLAT if all(abs(o["u"][2]) < 0.1 for o in out) else ROOT_MAX
    return out


L._raw_members = _raw_members
L.DEPTH = DEPTH_MIN                                       # a socket with no frame-model load (none here: check_members refuses it)

# The drawn collar (24 Nov, Spark: "the nodes are way too thick"): each socket's wall from its own rod's end moment by a hand calculation,
# never over the 3.0 it was (14 Nov: 3.0 where two side-by-side collars meet at their roots) and never under the audit's wall.
# Where the mass was (14 Nov runs, 70.8 g): 37.7 g collars, every one 3.0 thick mouth to floor whatever its moment, and 33.1 g generated,
# 12.3 g of it at P2 / P5 round one socket each (the staggered starts below).
#   The collar carries its rod's end moment M (frame model, N mm) with the side force V at mid-bore, Me = M + V d / 2, the way the setup
#   applies it: a couple Fc = Me / (2 L3) across the lining's thirds (L3 = (d + LINING_BACK) / 3). Two modes, both read in the 14 Nov runs:
#   Shear, mouth third to floor third: the tube round the 4.2 annulus carries Fc across the layers, peak 2 Fc / A (a thin tube). The
#   pushed-out sockets' worst sat there (mid-bore, on the collar's skin, across layers: P1's RS0-P1 0.214 where this reads 0.194 on the
#   3.0 tube, so x SHEAR_K 1.1): A >= 1.1 x 2 Fc / (COLLAR_INDEX x the across-layer crash strength 23 x 0.85 / 1.1).
#   Bending, at the root: Me on the tube, pi (D^4 - 4.2^4) / (32 D) >= Me / (COLLAR_INDEX x the crash strength along the socket, the
#   in-layer 46 for a socket lying in the layers, toward the across-layer 23 with the square of its rise). The hub sockets' worst sat
#   there (RF's RF-RM, a bare 3.0 collar: 0.25 by this, 0.233 in its run).
#   COLLAR_INDEX 0.27 under the 0.3 limit. A 3 mm rod's 1.0-1.5 N m needs more than the 3.0 wall in both modes (D 11-12), so those
#   collars stay 3.0 mouth to floor with the engine's material round them; only the sockets under ~650 N mm thin: the wall the shear
#   needs from the mouth to d / 6 above the floor (the floor third's centre, where the couple closes), straight from there to the root
#   wall the bending needs, the root wall back through the floor.
# The room round every collar stays the 3.0-wall strut + STRUT_M, so the engine still grows a root wherever more than the tube bears.
# The sockets held as the support (24 Nov, was the missing case: a force on a fixed region is nothing, so no case sized the generated
# root behind them for their own rod; by the bending line above the bare 3.0 roots read 0.40-0.69, BT0 / BT1-TB 1.4 N m the worst): each
# gets its own cases (build, below), the node held at its other sockets, with and without a near-parallel neighbour. The audit's
# interface_case refuses a setup where a socket with a frame-model load is loaded in no case.
# 24 Nov, the flat hubs' roots (run bd7af5d5: RB 0.394 at 1.2 cm3 where it read 0.40 at 0.8, RS1 0.338 / RS0 0.287 at 1.45): RF / RB / RS
# lie flat, every socket in the layer plane and every end moment about z, so nothing braces a socket's root but its own tube; the worst
# sat in-layer on the collar's side at the floor plane, in the corner to the next collar (the bending line above reads 0.53 on RM-RB's
# bare 3.0 tube), and no budget moved it: a 1 mm thickening round a collar is under the engine's 6.25 member. A flat hub's sockets take
# the root wall the bending line asks, to ROOT_FLAT (RM-RB 4.15, RS-P2 / RS-P5 4.4, RB-P 3.85; RF's stay 3.0), tapered to the mouth wall.
ROOT_MAX, ROOT_FLAT = 3.0, 4.5
MIN_WALL, COLLAR_INDEX, TAPER_FROM, SHEAR_K = 2.55, 0.27, 1.0 / 6.0, 1.1   # MIN_WALL: 2.05 at the 0.5 annulus (the audit's drawn_wall floor 2.0, the 0.05 the 40-gon's flats)
XT_MPA, ZT_MPA = 46.0 * 0.85 / 1.1, 23.0 * 0.85 / 1.1    # topo/material.py PCTG Xt / Zt, KNOCKDOWN, SAFETY["crash"]


def collar_walls(moment, side, depth, bore_r, rise=0.0, root_max=None):
    """(root wall, mouth wall) of a socket's collar for its rod's end moment (the hand calculation above), each MIN_WALL to ROOT_MAX,
    up to the next 0.05; rise = the socket axis's component along the build direction"""
    di = 2.0 * (bore_r + GROOVE_D); me = abs(moment) + abs(side) * depth / 2.0
    up = lambda D: min(ROOT_MAX, max(MIN_WALL, float(np.ceil((D / 2.0 - bore_r) * 20.0 - 1e-9) / 20.0)))
    fc = me / (2.0 * (depth + L.LINING_BACK) / 3.0)
    mouth = up(np.sqrt(4.0 * SHEAR_K * 2.0 * fc / (COLLAR_INDEX * ZT_MPA) / np.pi + di ** 2))
    z = me / (COLLAR_INDEX * (XT_MPA + (ZT_MPA - XT_MPA) * rise ** 2)); lo, hi = di, 40.0
    for _ in range(60):
        D = 0.5 * (lo + hi)
        lo, hi = (D, hi) if np.pi * (D ** 4 - di ** 4) / (32.0 * D) < z else (lo, D)
    root = float(np.ceil((hi / 2.0 - bore_r) * 20.0 - 1e-9) / 20.0)
    return max(min(max(root_max or ROOT_MAX, ROOT_MAX), max(root, MIN_WALL)), mouth), mouth


class Wall(float):
    """a tapered collar's wall: its value the root wall, .mouth the wall at the mouth, the taper from .ta to .tb along the socket axis.
    + keeps the taper (the mouth cuts spare every other collar grown by 0.5)"""
    def __new__(cls, root, mouth, ta, tb):
        w = float.__new__(cls, root); w.mouth, w.ta, w.tb = float(mouth), float(ta), float(tb)
        return w

    def __add__(self, g):
        return Wall(float(self) + g, self.mouth + g, self.ta, self.tb)


def drawn_wall(m):
    root, mouth = collar_walls(m.get("moment") or 0.0, m.get("side") or 0.0, m["depth"], m["bore_r"], float(m["u"][2]), m.get("root_max"))
    return Wall(root, mouth, m["s0"] + TAPER_FROM * m["depth"], m["s0"] + m["depth"])


L.DRAWN_WALL = drawn_wall

# The collar's outside is round with only the wall over the bore's crown (the crown's apex r sqrt 2 + wall), not the bore's
# teardrop scaled to the outside (apex (r + wall) sqrt 2): that put a sharp 1.7 mm ridge along the top of every collar, ending in
# a tick at each mouth (Spark's fins, 9 Nov). The top of a collar is an up-facing surface, it needs no teardrop to print.
_collar0 = L.collar_shape


def _section(r, wall, crowned):
    ro, apex = r + wall, r * np.sqrt(2.0) + wall
    if not crowned:
        return [(ro * np.cos(t), ro * np.sin(t)) for t in np.linspace(0.0, 2 * np.pi, 48, endpoint=False)]
    t0 = np.arccos(ro / apex)                             # where the apex's tangents touch the circle
    return [(ro * np.cos(t), ro * np.sin(t)) for t in np.linspace(t0, 2 * np.pi - t0, 48)] + [(apex, 0.0)]


def collar_shape(c, u, s0, s1, r, wall, build=(0.0, 0.0, 1.0)):
    import manifold3d as mf
    from parts.cage_node_lib import crown_dir, _prism
    u = np.asarray(u, float) / np.linalg.norm(u); d = crown_dir(u, build)
    e1 = d if d is not None else np.cross(u, [0, 0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0, 0]); e1 = e1 / np.linalg.norm(e1); e2 = np.cross(u, e1)
    sec = lambda w, a, b: _prism(_section(r, w, d is not None), c, u, a, b, e1, e2)
    if not isinstance(wall, Wall) or wall.mouth >= float(wall) - 1e-9:
        return sec(float(wall), s0, s1)
    ta, tb = max(wall.ta, s0), min(wall.tb, s1)
    # one convex body (both sections convex, the root's the larger): the root wall from the floor to ta, straight to the mouth wall at
    # tb, the mouth wall on past it (the mouth cut's spare). Not prisms stacked on a loft: their shared faces left the drawn mesh with
    # odd-parity columns (the audit's parity check, 24 Nov)
    return mf.Manifold.batch_hull([sec(float(wall), s0, ta), sec(wall.mouth, tb - 0.02, max(s1, tb))])


L.collar_shape = collar_shape

_port_dir0 = L.port_dir


def port_dir(u, prefer, others=(), blockers=(), origin=None, clear_r=0.0, **kw):
    """a port's channel stays clear of every other collar (the lib's blockers, at the 3.0-wall radius). A collar this socket rides on
    (staggered starts: the port's origin already inside that collar's radius) cannot be cleared: there the channel keeps PORT_WALL of
    print between itself and that socket's bore and feed ring instead (the collar's floor is solid)"""
    out = []
    for p0, p1, r in blockers:
        ax = (p1 - p0) / np.linalg.norm(p1 - p0)
        if origin is not None and _sd(np.asarray(origin, float), np.asarray(origin, float), p0, p1) < r + clear_r:
            p0, r = p0 + ax * (L.FLOOR - 0.5), L.BORE_R + GROOVE_D + PORT_WALL - 0.5
        out.append((p0, p1, r))
    return _port_dir0(u, prefer, others, out, origin=origin, clear_r=clear_r, **kw)


PORT_WALL = 2.0
L.port_dir = port_dir

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
# 24 Nov, the rider's root: where a collar rides on a near-parallel neighbour's (_beside) and runs on past that neighbour's mouth plane, the
# neighbour's mouth cut left it bare from that plane out (its own wall + 0.5), and the rider's end moment met the neighbour's rod there:
# P3's TB-P3 beside RB-P3 read 0.258 across the layers with the room filled solid, on the rider's skin at RB-P3's mouth plane (the bare
# 3.0 tube is 0.30 there by the collar's shear line below: no layout could pass). A neighbour's mouth cut now spares the rider's whole
# strut (RIDER_M of room round its collar) outside a tunnel round the neighbour's own rod (its collar's radius + 0.5: the mouth stays
# one exact face, its lead-in and keep-out carve nothing grown). The cuts are made here (the lib's own MOUTH_R is off; the v4 nodes sit
# above the funnel, so the lib's funnel hook carries them).
L.STRUT_BACK, L.COLLAR_PAST = -2.0, 0.0
L.MOUTH_R = None
MOUTH_R = L.BORE_R + L.WALL + L.STRUT_M + 0.5
RIDER_M = L.STRUT_M
_GROUP = []


# 24 Nov, the twin-bore socket (the design chat on the run of 0275b432, 5 fails; it replaces that push's notch web, the hull of the two
# collars only as far as their skins were 0.8 apart): two sockets under TWIN_DEG apart are 0-3.9 mm apart skin to skin all the way to
# their mouths, under the engine's 6.25 member, so the compliance objective leaves the notch between them open at any budget, and the
# audit's solid room cannot tell (run bd7af5d5: 9 of 11 fails in the notch at mid-bore; run 0275b432 with the web: P3 / P4 / TB
# 0.315-0.317 in generated material 1 mm past the web's end, P2 / P5 0.371 / 0.387 on the rider's skin at RS's mouth plane, where the
# web ended). Such a pair is one drawn socket with two bores: the hull of the two collars, each at its own hand-calc walls
# (collar_walls: the outer walls), from the outer socket's floor plane to both mouth planes (each mouth one exact face), every bore
# with its port and pockets as before. Every pair under TWIN_DEG at every node (12 of the 16; none at RF / RB / RS).
#   The shared wall, by hand (shared_wall): a rod's end moment with its side force at mid-bore, Me = M + V d / 2, is the couple
#   Fc = Me / (2 L3) across its lining's thirds (L3 = (d + LINING_BACK) / 3, the way the setup applies it). With the neighbour's rod
#   held (the held sockets' cases) the mouth third's Fc crosses the shared wall straight into that rod, in tension for one sign,
#   across the layers at worst. The section it crosses, midway between the bores: the twin's height there, w = the two collars'
#   mouth radii together, over the length Lb of the mouth third the neighbour's collar stands beside (L3 less what the rod's mouth
#   runs on past the neighbour's), less the core pocket below: Fc / A <= COLLAR_INDEX x the across-layer crash strength
#   (0.27 x 17.8 = 4.8 MPa). _back_depths deepens the neighbour 0.5 at a time until every twin passes (BACK_DEPTH; it was P3 / P4
#   by a table, 0.8 short of the rider's mouth); check_pairs refuses a twin that does not.
#   The core pocket (_pockets, a cut): a drawn body 6 mm thick anywhere is no longer held whole by the layout, only at its gate core
#   (topo.layout CORE_MIN_HALF_MM), and every socket's load moves to that core: the solid hull (6.4 between the bores at a P node's
#   mouths, 9.4 at TB's) made P3 / P4's P-P cases zero compliance and took the third socket's end moment out of every other
#   node's layout (P2's P2-P3: 831 N mm read as 117). So where the bores are more than two TWIN_WALL walls (2.05, the collar's
#   least) + POCKET_MIN apart the shared wall is those two walls and a flange over and under (FLANGE 3.0, to FLANGE_MAX where the
#   section above needs it: twin_flange), a pocket between them open through the face between the mouths; build refuses a node
#   whose drawn sockets reach the 6 mm at its h.
#   The worst end moment, BT1-TB 1434 N mm + 53.4 N on 13.0: Me 1781, L3 5.5, Fc 162 N: 33.7 mm2; w 9.2, Lb 5.0 (BT0's mouth 0.5
#   short), all of it over the pocket (from 17.4), flanges 3.4: 34.0 mm2, index 0.268. The worst wall, P3's TB-P3 858 N mm + 18.6 N
#   on 7.5: Fc 127 N, 26.4 mm2; RB-P3 15.5 deep (was 14.5: 0.8 short of TB-P3's mouth, 0.317 in its run) stands beside the whole
#   3.67, w 8.8, pocket from 18.0, flanges 3.15: 26.4 mm2, 0.269. P2 / P5's S0-P2 / S1-P5 693 N mm on 7.0 (Fc 105 N, 21.8 mm2):
#   RS-P ended 4.3 short of its mouth, no wall beside the mouth third at all (0.371 / 0.387); RS-P 14.0 deep from the hub (was 9.0
#   from 6.5): 21.9 mm2, 0.269.
TWIN_DEG = 45.0
TWIN_WALL, POCKET_MIN, FLANGE, FLANGE_MAX = MIN_WALL - 0.5, 1.0, 3.0, 3.5


def _twin_pairs(k):
    ms = L.members(k)
    return [(a, b) for i, a in enumerate(ms) for b in ms[i + 1:] if 1e-6 < np.degrees(np.arccos(np.clip(a["u"] @ b["u"], -1.0, 1.0))) <= TWIN_DEG]


def _twin_frame(a, b):
    """a twin's half angle, bisector, in-plane normal to it (a toward b), the pair plane's normal, its height w between the bores
    and the reach along the bisector where its core pocket starts (POCKET_MIN wide between the two TWIN_WALL walls)"""
    t2 = 0.5 * float(np.arccos(np.clip(a["u"] @ b["u"], -1.0, 1.0)))
    bis = a["u"] + b["u"]; bis = bis / np.linalg.norm(bis); n = np.cross(a["u"], b["u"]); n = n / np.linalg.norm(n)
    w = sum(m["bore_r"] + drawn_wall(m).mouth for m in (a, b))    # midway between the bores, the hull of the two collars
    lo = (POCKET_MIN + sum(m["bore_r"] + GROOVE_D + TWIN_WALL for m in (a, b)) / np.cos(t2)) / (2.0 * np.tan(t2))
    return t2, bis, np.cross(n, bis), n, w, lo


def _wall_area(x, y, tf):
    """(the shared wall's section x's end moment needs beside its mouth third, the section the twin has there, mm2) with flanges tf"""
    t2, _, _, _, w, lo = _twin_frame(x, y)
    me = abs(x.get("moment") or 0.0) + abs(x.get("side") or 0.0) * x["depth"] / 2.0; l3 = (x["depth"] + L.LINING_BACK) / 3.0
    end = min(x["s0"] + x["depth"], y["s0"] + y["depth"]); top = x["s0"] + x["depth"] - l3
    lb, lp = max(0.0, end - top), max(0.0, end - max(top, lo / np.cos(t2)))
    return me / (2.0 * l3) / (COLLAR_INDEX * ZT_MPA), w * lb - max(0.0, w - 2.0 * tf) * lp


def twin_flange(a, b):
    """the flange over and under a twin's core pocket: FLANGE, or what the shared wall's hand calculation asks, to FLANGE_MAX"""
    for tf in np.arange(FLANGE, FLANGE_MAX + 1e-9, 0.05):
        if all(have >= need - 1e-6 for need, have in (_wall_area(a, b, tf), _wall_area(b, a, tf))):
            break
    return float(tf)


def shared_wall(x, y):
    """(needed, there): the shared wall's section beside x's mouth third, mm2 (the hand calculation above)"""
    return _wall_area(x, y, twin_flange(x, y))


def _back_depths():
    """BACK_DEPTH: a twin's shorter-reaching socket 0.5 deeper wherever its neighbour's shared wall is short (to the
    neighbour's mouth, the next 0.5). True if a depth changed (the socket starts are then worked out again: clear reads the depths)"""
    moved = False
    for k in L.KEYS:
        for a, b in _twin_pairs(k):
            for x, y in ((a, b), (b, a)):
                need, have = shared_wall(x, y)
                if have < need - 1e-6 and y["s0"] + y["depth"] < x["s0"] + x["depth"] - 1e-6:
                    BACK_DEPTH[(k, y["name"])] = y["depth"] + 0.5; moved = True
    return moved


def _webs(keys):
    """the twin-bore sockets' bodies between their collars (the collars themselves are the lib's): the hull of the two collars from
    the outer socket's floor plane to both mouth planes"""
    import manifold3d as mf
    out = []
    for k in keys:
        c = np.array(L.P[k], float)
        for a, b in _twin_pairs(k):
            body = mf.Manifold.batch_hull([L.collar_shape(c, x["u"], x["s0"] - L.FLOOR, x["s0"] + x["depth"], x["bore_r"], drawn_wall(x)) for x in (a, b)])
            for x in (a, b):
                body = body.trim_by_plane((-x["u"]).tolist(), -float(x["u"] @ c + x["s0"] + x["depth"]))
            r = max((a, b), key=lambda x: x["s0"])
            out.append(body.trim_by_plane(r["u"].tolist(), float(r["u"] @ c + r["s0"] - L.FLOOR)))
    return out


def _pockets(keys):
    """the twins' core pockets (cuts): between the two bores' TWIN_WALL walls and the two flanges, from where it is POCKET_MIN
    wide out through the face between the mouths"""
    from parts.cage_node_lib import _prism
    out = []
    for k in keys:
        c = np.array(L.P[k], float)
        for a, b in _twin_pairs(k):
            t2, bis, e2, n, w, lo = _twin_frame(a, b); hh = w / 2.0 - twin_flange(a, b)
            hi = max(x["s0"] + x["depth"] for x in (a, b)) + 10.0
            if hh < 1.0 or lo > min(x["s0"] + x["depth"] for x in (a, b)) * np.cos(t2) - 0.5:
                continue
            tn = np.tan(t2)
            wedge = _prism([(lo, -lo * tn), (hi, -hi * tn), (hi, hi * tn), (lo, lo * tn)], c, n, -hh, hh, bis, e2)
            out.append(wedge - L.U([L.along(c, x["u"], 0.0, hi + 5.0, x["bore_r"] + GROOVE_D + TWIN_WALL, 64) for x in (a, b)]))
    return out


_Part0 = L.Part


def _part(name, h, drawn, cuts, env, **kw):
    return _Part0(name, h, L.U([drawn] + _webs(_GROUP)), cuts, env, **kw)


L.Part = _part


def _mouth_cuts():
    import manifold3d as mf
    allm = []; webs = _webs(_GROUP)
    for k in _GROUP:
        c = np.array(L.P[k], float)
        for m in L.members(k):
            m["c"], m["s_end"], m["ro"], m["wall_d"] = c, m["s0"] + m["depth"], m["bore_r"] + L.WALL, drawn_wall(m)
            allm.append(m)
    cuts = []
    for m in allm:
        spare = [L.collar_shape(o["c"], o["u"], o["s0"] - L.FLOOR - 0.5, o["s_end"] + 0.5, o["bore_r"], o["wall_d"] + 0.5) for o in allm if o is not m]
        riders = [L.along(o["c"], o["u"], 0.0, o["s_end"] + 0.5, o["ro"] + RIDER_M) for o in allm
                  if o is not m and np.allclose(o["c"], m["c"]) and _beside(m, o) and o["s_end"] > m["s_end"]]
        if riders:
            spare.append(L.U(riders) - L.along(m["c"], m["u"], m["s_end"] - 1.0, m["s_end"] + 61.0, m["ro"] + 0.5, 64))
        slab = L.along(m["c"], m["u"], m["s_end"], m["s_end"] + 60.0, MOUTH_R, 64)
        spare += webs
        cuts.append(slab - L.U(spare) if spare else slab)
    cuts += _pockets(_GROUP)
    return L.U(cuts) if cuts else mf.Manifold()


L.funnel_sweep = _mouth_cuts

# The Ys' forks (9 Nov): their bores side by side need only the audit's wall between the grooves (2.1). A hub's pair of near-parallel
# bores (P1's RS0-P1 / S0-P1 13 mm out) at 2.1 put each groove 0.05 off its neighbour's 2.55 collar face: near-coincident faces,
# the ticks in P1's bores. A hub pair gets 2.4 (0.35 off); a fork pair keeps 2.1 (its 3.0 collars 0.4 into the groove).
FORK_S0 = 20.0                                           # a bore pushed out past this by a near-parallel neighbour would be a fork socket (none in v4: the paired hub bores start at 12.4)
L.MAX_START = 36.0
L.BETWEEN_WALL = 2.1
BETWEEN_HUB = 2.4


# Staggered starts (24 Nov): a hub's near-parallel pair both started 12.4 out (two 7.4 necks, generated), and the pair's larger end moment
# crossed its neck to reach the hub: P2 / P5 took 5.0 cm3 (6.1 g each) to bring RS0-P2 / RS1-P5's 1.2 N m in at 0.225, the worst across
# the layers at mid-bore of the pushed-out collar, where RS0 holds the same rod's 1.43 N m at 0.279 on 1.1 cm3 from a hub start.
# Now only one of the pair moves out: the socket with the larger end moment starts as near the hub as its mouth allows (its lead-in
# cone clear of the other's collar, STAGGER_GAP), the other where its bore clears that one's by the same BETWEEN_HUB wall; its collar
# rides on the first's. A pair within STAGGER_TIE of each other's moment (the Ys' mirrored forks BN / BT, NB-R, TB-P) stays symmetric.
# The exact geometry (clear below), not the symmetric formula; check_pairs refuses any pair of a node part that fails it.
STAGGER_TIE, STAGGER_GAP, STAGGER_OUT = 0.9, 0.3, 1.5     # STAGGER_OUT: the moved socket at most this far past its symmetric start


def _sd(p1, q1, p2, q2):
    """the least distance between two segments (closed form; a point is a segment of no length)"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2; a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s_, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e < 1e-12:
            s_, t = np.clip(-c / a, 0.0, 1.0), 0.0
        else:
            b = d1 @ d2; den = a * e - b * b
            s_ = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            t = (b * s_ + f) / e
            if t < 0.0:
                s_, t = np.clip(-c / a, 0.0, 1.0), 0.0
            elif t > 1.0:
                s_, t = np.clip((b - c) / a, 0.0, 1.0), 1.0
    return float(np.linalg.norm(p1 + d1 * s_ - p2 - d2 * t))


def clear(a, sa, b, sb, why=None):
    """two sockets of one node, their bore floors sa / sb from its centre: the bores BETWEEN_HUB apart at their grooves, each collar
    (the 3.0 wall, floor to mouth) clear of the other's lead-in cone by STAGGER_GAP and of its rod by ROD_GAP"""
    seg = lambda m, s, t0, t1: (m["u"] * (s + t0), m["u"] * (s + t1))
    bad = []
    if _sd(*seg(a, sa, 0.0, a["depth"]), *seg(b, sb, 0.0, b["depth"])) < a["bore_r"] + b["bore_r"] + 2 * L.GROOVE_D + BETWEEN_HUB - 1e-6:
        bad.append("bores")
    for x, sx, y, sy in ((a, sa, b, sb), (b, sb, a, sa)):
        col = seg(y, sy, -L.FLOOR, y["depth"]); ro = y["bore_r"] + L.WALL
        for t in np.linspace(0.0, 2.0, 9):                # x's lead-in: bore_r at 1.0 inside the mouth to bore_r + 2.0 at 1.0 past it
            q = x["u"] * (sx + x["depth"] - 1.0 + t)
            if _sd(q, q, *col) < ro + x["bore_r"] + t + STAGGER_GAP - 1e-6:
                bad.append(f"{x['name']}'s lead-in in {y['name']}'s collar"); break
        if _sd(*seg(x, sx, x["depth"] + 1.0, x["depth"] + 60.0), *col) < ro + L.ROD_R + L.ROD_GAP - 1e-6:
            bad.append(f"{x['name']}'s rod at {y['name']}'s collar")
    if why is not None:
        why.extend(bad)
    return not bad


def _socket_starts():
    st = {}
    for k in L.sc.NODES:
        ms = L._raw_members(k, bridges=True)
        pair = lambda m, o, bw: (m["bore_r"] + o["bore_r"] + 2 * L.GROOVE_D + bw) / (2.0 * np.sin(np.arccos(np.clip(m["u"] @ o["u"], -1.0, 1.0)) / 2.0))

        def start(m, skip=None):
            def one(bw):
                return max([L.NODE_R] + [pair(m, o, bw) for o in ms if o is not m and o is not skip and m["u"] @ o["u"] < 1.0 - 1e-12])
            s0 = one(L.BETWEEN_WALL)
            return float(np.ceil((one(BETWEEN_HUB) if L.NODE_R + 1e-6 < s0 <= FORK_S0 else s0) * 10.0 - 1e-9) / 10.0)
        s = {m["name"]: start(m) for m in ms}
        if k in L.KEYS:                                   # the node parts; the frame's printed sockets keep the symmetric rule
            mom = {m["name"]: abs((L.LOADS.get(k, {}).get(m["name"]) or {}).get("moment", 0.0)) for m in ms}
            for i, a in enumerate(ms):
                for b in ms[i + 1:]:
                    hi, lo = (a, b) if mom[a["name"]] >= mom[b["name"]] else (b, a)
                    both = pair(a, b, BETWEEN_HUB)
                    if both <= L.NODE_R + 1e-6 or min(s[a["name"]], s[b["name"]]) < both - 0.1 or mom[lo["name"]] > STAGGER_TIE * mom[hi["name"]]:
                        continue                          # not a pair that pushed each other out, or a mirrored fork
                    others = [o for o in ms if o is not hi and o is not lo]
                    ok = lambda sh, sl: clear(hi, sh, lo, sl) and all(clear(x, sx, o, s[o["name"]]) for x, sx in ((hi, sh), (lo, sl)) for o in others)
                    grid = lambda m, skip, top: np.round(np.arange(start(m, skip=skip), top + 1e-9, 0.1), 1)

                    def first(xs, good):                  # the first of xs that is good (good from there on: the rays part)
                        i, j = 0, len(xs) - 1
                        if len(xs) == 0 or not good(xs[j]):
                            return None
                        while i < j:
                            h = (i + j) // 2
                            i, j = (i, h) if good(xs[h]) else (h + 1, j)
                        return float(xs[i])
                    lows = grid(lo, hi, s[lo["name"]] + STAGGER_OUT)
                    sh = first(grid(hi, lo, s[hi["name"]] - 1.0), lambda x: first(lows, lambda y: ok(x, y)) is not None)
                    found = None if sh is None else (sh, first(lows, lambda y: ok(sh, y)))
                    if found:
                        s[hi["name"]], s[lo["name"]] = found
        for m in ms:
            st[(k, m["name"])] = s[m["name"]]
    return st


def check_pairs(k):
    """every pair of sockets of node part k by the exact geometry (clear); the symmetric formula in check_members no longer says it"""
    ms = L.members(k); bad = []
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            why = []
            if not clear(a, a["s0"], b, b["s0"], why):
                bad.append(f"{a['name']} ({a['s0']:.1f}) / {b['name']} ({b['s0']:.1f}): {', '.join(why)}")
    if bad:
        raise ValueError(f"node {k}: sockets too close: " + "; ".join(bad))
    for a, b in _twin_pairs(k):
        for x, y in ((a, b), (b, a)):
            need, have = shared_wall(x, y)
            if have < need - 1e-6:
                raise ValueError(f"node {k}: the twin {x['name']} / {y['name']}: {x['name']}'s end moment needs {need:.1f} mm2 of shared wall beside its "
                                 f"mouth third, the twin has {have:.1f}: BACK_DEPTH (_back_depths), FLANGE_MAX")


_check0 = L.check_members


def _check_members(k, log=print):
    ms = _check0(k, log); check_pairs(k)
    return ms


L.check_members = _check_members
L.SOCKET_START = _socket_starts()
for _ in range(40):                                       # the twins' backed depths and the starts that read them, to a fixed point
    if not _back_depths():
        break
    L.SOCKET_START = _socket_starts()

# Budgets (10 Nov): the light nodes read 6-40 x under the 0.3 limit at 1.1 x skeleton (P / R 0.017-0.054 at 1.7 cm3, RF / RB / RS
# 0.005-0.014 at 1.2): their collars now overlap at the hub and carry most of the load drawn. BUDGET_CM3 sets their generated volume
# outright toward ~0.25 (index ~ volume^-1.25), above what joins the collars' roots. The Ys (S0 / S1 0.20 / 0.23, NB 0.20, TB 0.23)
# keep the volume they passed at.
# Run 200 (round 2, 10 Nov): all 16 pass. P / R at 0.9: 0.018-0.027, P3 / P4 at 1.0: 0.091, RF / RB / RS at 0.4: 0.030-0.043, S0 / S1
# at 3.6: 0.209 / 0.210, NB at 4.9: 0.220, TB at 5.5: 0.210. Round 3: the Ys toward 0.25 (index ~ V^-1.25: x0.87), P3 / P4 0.7
# (~0.14), P / R 0.6 (two 6.25 members from the hub collar to the paired collars' floors are ~0.6 cm3), RF / RB / RS 0.3.
# 14 Nov, the end moments (each socket's side case carries its rod's end moment as a couple; RS0-P2 / RS1-P5 1.2-1.4 N m): the audit
# refused P2 / P5 at 0.6 (room thinned 1.92) and RS0 / RS1 at 0.3 (2.16), RB read 1.66 at 0.3: P2 / P5 1.0, RS0 / RS1 0.6, RB 0.5 (~0.9-1.0 by V^-1.25).
# Run of 14 Nov (dispatch 13:18 UTC) on those: worst in every case the end-moment case: P2 / P5 0.67, P3 / P4 0.50, RS0 / RS1 0.44, RB 0.39,
# S1 0.37, P6 0.36, R0 / R7 0.33, S0 0.31; P1 0.29 with a 1.1 web (solid fail); NB 0.10, TB 0.18, RF 0.22 pass. Each toward ~0.22 by V^-1.25.
# 3.0 collars (push 7db172f3): 14 pass, P2 / P5 0.39 across layers in RS0-P2 / RS1-P5's moment case, now just outside the collar
# (4.9 off the axis, the generated root): 2.4 -> 3.6 (~0.24 by V^-1.25).
# Then 0.34 at 3.6 (6.5 g). A 13.0 bore on their pushed-out RS0-P2 / RS1-P5 collars (s0 12.4, 1.19 N m) read 0.41 at 3.0: the longer
# collar's lever grew faster than the couple across it fell; dropped. Room filled solid 0.115: index ~ V^-1.14 from 3.6 to the room, 5.0 (~0.24).
# 24 Nov (Spark: "way too thick"), each toward ~0.25 built by index ~ V^-1.25 from its 14 Nov run (built worst at the old budget):
# P2 / P5 5.0 -> 1.2 (the staggered starts: RS0-P2 / RS1-P5 now start 6.5 from the hub, as RS0 holds the same rod's 1.43 N m at 0.28 on
# 1.1; the pushed-out S0-P2 0.69 N m is P1's RS0-P1, 0.21 on 0.9), NB 1.8 -> 1.0 (0.093), TB 2.4 -> 2.2 (0.216), S0 1.6 -> 1.3 (0.184),
# S1 1.8 -> 1.55 (0.203), R0 / R7 0.85 -> 0.7 (0.175). The rest read 0.21-0.28 and keep theirs. 27.2 -> 17.75 cm3 (33.5 -> 21.8 g).
# Run of 24 Nov on those, with the held sockets' own cases (push 427c3eaa; 57.07 g): 8 pass, 8 fail, every worst an end-moment case in the
# generated root: P3 / P4 0.41 (TB-P3 / TB-P4), RB 0.40 (RM-RB), RF 0.385, RS0 / RS1 0.36 / 0.35 (RM-RS), TB 0.35 (BT0 / BT1-TB), P1 0.305.
# The held sockets' cases now run on both supports (build: the two stiffest other sockets, and those with the near-parallel neighbour
# left out), the worse governs. Each toward 0.25 by index ~ V^-1.25 from its built worst, a case on the new support read from its
# socket's built case x the two supports' readings with the room filled solid (the audit's room_full, per case):
# P3 / P4 1.4 -> 1.7 (0.32: the backed rider, solid 0.115 where the old support read 0.148 and built 0.411), P2 / P5 1.2 -> 1.5 (0.33: S0-P2 /
# S1-P5 with RS held, solid 0.096 on 0.298 built), R0 / R7 0.7 -> 0.8 (0.29, NB-R with RF-R held), P1 0.9 -> 1.0, RB 0.8 -> 1.2,
# RF 0.3 -> 0.45, RS0 / RS1 1.1 -> 1.45, TB 2.2 -> 2.9. P6, S0, S1, NB read under 0.25 on both supports and keep theirs. 17.75 -> 21.2 cm3.
# Run of 24 Nov on those (push bd7af5d5; 62.3 g): 5 pass (NB 0.205, RF 0.218, S0 0.215, S1 0.219, RS0 0.287), 11 fail, none of them a budget's
# to fix: 9 in the notch between a near-parallel pair (the notch web above, drawn), RB / RS1 at the flat hubs' roots (ROOT_FLAT, drawn).
# RB 1.2 -> 0.8 and RS0 / RS1 1.45 -> 1.1: the volume added for those roots moved nothing (RB) or little (RS 0.355 -> 0.31) and the
# drawn roots are 0.3-0.5 cm3; every other budget stays.
# Run of 24 Nov on those (push 0275b432; 64.31 g): 11 pass, P2 / P5 / P3 / P4 / TB fail at the notch web's end: the twin-bore socket above
# (drawn 32.3 -> 35.0 cm3). Every budget stays: each room still reads the same 0.04-0.14 filled solid.
BUDGET_CM3 = {"P3": 1.7, "P4": 1.7, "P1": 1.0, "P2": 1.5, "P5": 1.5, "P6": 0.9, "R0": 0.8, "R7": 0.8,
              "RF": 0.45, "RB": 0.8, "RS0": 1.1, "RS1": 1.1, "S0": 1.3, "S1": 1.55, "NB": 1.0, "TB": 2.9}   # 14 Nov, the bolt forks: their runs read NB / TB 0.051 / 0.052 at 4.0, S0 / S1 0.032 at 3.1; on the fold's loads (core_of) TB-P3 154 N (was 99), S0-P2 109 (was 78): toward ~0.15 by index ~ V^-1.25.   # 13 Nov: NB 5.7 -> 6.9, 0.314 at 5.7 (index ~ V^-1.25 toward 0.25). 12 Nov: NB 4.3 -> 5.7, 0.354 at 4.3 once BS moved to the nose block (NB-R7 pull-out 126 N; index ~ V^-1.25 toward 0.25)   # 12 Nov: TB 4.8 -> 5.6, 0.301 at 4.8 under the heavier frame's TB-P3 108 N (index ~ V^-1.25 toward 0.25)


_build0 = L.build


def _beside(a, b):
    """two sockets of one node near-parallel enough to push each other out from the hub (the pair rule of _socket_starts)"""
    t = np.arccos(np.clip(a["u"] @ b["u"], -1.0, 1.0))
    return t < 1e-6 or (a["bore_r"] + b["bore_r"] + 2 * L.GROOVE_D + BETWEEN_HUB) / (2.0 * np.sin(t / 2.0)) > L.NODE_R + 1e-6


def build(group, h, log=print):
    """L.build + the cases it leaves out: the two sockets it holds as the support (the stiffest by axial load) each get their own rod's
    pull-out and side force + end moment (the same couple across the lining's thirds, both signs), the node held for them at its other
    sockets, on two supports and the worse governs (24 Nov: never the favourable one): the two stiffest of the rest, and the two
    stiffest with a near-parallel neighbour left out (_beside). With the neighbour held its rod carries the couple straight across the
    wall between the two bores (P3's TB-P3 on RB-P3: the backed rider, BACK_DEPTH); without it the root to the hub carries it all.
    Every socket with a frame-model load is declared to the audit (Part.interface_loads -> interface_case)."""
    _GROUP[:] = (group,) if isinstance(group, str) else tuple(group)
    p = _build0(group, h, log)
    from scipy import ndimage as ndi
    from topo.layout import CORE_MIN_HALF_MM
    sol3 = p.solid.reshape(p.shape); din = ndi.distance_transform_edt(sol3) * p.h
    if h <= L.spec_for(group)["h"] + 1e-9 and din.max() >= CORE_MIN_HALF_MM:   # the layout's own measure (Layout._anchors), at the SPEC h (the coarse stage only warms the start)
        at = (np.array(np.unravel_index(int(din.argmax()), p.shape)) + 0.5) * p.h + np.asarray(p.origin)
        raise ValueError(f"node {_GROUP}: the drawn sockets are {2 * din.max():.1f} mm thick at {np.round(at, 1).tolist()} at h {h}: from "
                         f"{2 * CORE_MIN_HALF_MM:.0f} mm the layout holds a drawn body at its core only and moves every socket's load there "
                         f"(a twin's shared wall: its core pocket, TWIN_WALL / FLANGE)")
    along, U = L.along, L.U
    lining = lambda m, grow=1.0: along(m["c"], m["u"], m["s0"] - L.LINING_BACK, m["s_end"], m["bore_r"] + grow)
    nodes = lambda reg, what: p.nodes_of(p.elems(reg, p.solid | p.design, what=what))
    p.interface_loads = []
    for k in ((group,) if isinstance(group, str) else tuple(group)):
        ms = L.members(k); c = np.array(L.P[k], float)
        for m in ms:
            m["c"], m["s_end"] = c, m["s0"] + m["depth"]
        by = sorted(ms, key=lambda m: -m["axial"])
        for m in by[:2]:
            rest = [o for o in by if o is not m]; sets = []
            for cand in (rest[:2], ([o for o in rest if not _beside(m, o)] or rest)[:2]):
                if not any(cand == s_ for s_ in sets):
                    sets.append(cand)
            side = np.cross(m["u"], [0, 0, 1.0] if abs(m["u"][2]) < 0.9 else [1.0, 0, 0]); side /= np.linalg.norm(side)
            a0, a1 = m["s0"] - L.LINING_BACK, m["s_end"]; L3 = (a1 - a0) / 3.0
            th = [along(c, m["u"], a0 + i * L3, a0 + (i + 1) * L3, m["bore_r"] + 1.0) for i in range(3)]
            for held in sets:
                at = " + ".join(o["name"] for o in held); sup = f"hold {m['name']} at {at}"
                p.support(sup, U([lining(o, 1.4) for o in held]))
                log(f"  {m['name']} (held in the other cases): its own cases held at {at}")
                p.load_multi(f"frame {m['name']} axial {m['axial']:.0f} N (pull-out), held at {at}", [(lining(m), m["axial"] * m["u"])], cls="crash", support=sup)
                if not m.get("moment"):
                    p.load_multi(f"frame {m['name']} side {m['side']:.1f} N, held at {at}", [(lining(m), m["side"] * side)], cls="crash", support=sup)
                    continue
                Fc = np.cross(m["moment"] * np.asarray(m["moment_axis"], float), m["u"]) / (2.0 * L3)
                w = Fc / np.linalg.norm(Fc) if np.linalg.norm(Fc) > 1e-6 else side
                for sg, lab in ((1, "+"), (-1, "-")):
                    F = sg * m["side"] * w; c_ = sg * Fc
                    p.load_multi(f"frame {m['name']} side {m['side']:.1f} N + end moment {m['moment']:.0f} N mm ({lab}), held at {at}",
                                 [(th[0], F / 3 - c_), (th[1], F / 3), (th[2], F / 3 + c_)], cls="crash", support=sup)
        p.interface_loads += [dict(name=m["name"], nodes=nodes(lining(m), f"socket {m['name']}")) for m in ms if m["axial"] is not None]
    return p


L.build = build


def spec_for(group):
    s = L.spec_for(group)
    key = group if isinstance(group, str) else group[0]
    s["gen_cm3"] = BUDGET_CM3[key]
    return s


KEYS = L.KEYS
make, members, check_members = L.make, L.members, L.check_members
