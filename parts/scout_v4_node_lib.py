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


# A backed rider (24 Nov, the held sockets' cases with the near-parallel neighbour held too): P3's TB-P3 rides on RB-P3's collar and ran
# 4.8 on past RB-P3's mouth; with RB-P3's rod held its end moment peeled it off that collar across the layers at the mouth plane, 0.224
# with the room filled solid (0.258 before the rider's root room below; no budget passes on that). The neighbour's socket is as deep as
# brings its mouth to within BACK_REACH of the rider's (0.168 at 2.3 short, 0.115 at 0.8), so its collar backs the rider's whole bore;
# check_pairs refuses a BACK_DEPTH that no longer reaches. The mirrored pair at P4 the same.
BACK_DEPTH = {("P3", "RB-P3"): 14.5, ("P4", "RB-P4"): 14.5}
BACK_RIDER = {("P3", "RB-P3"): "TB-P3", ("P4", "RB-P4"): "TB-P4"}
BACK_REACH = 1.0
_raw0 = L._raw_members


def _raw_members(k, bridges=False):
    out = _raw0(k, bridges)
    for m in out:
        f = L.LOADS.get(k, {}).get(m["name"]) or {}
        m["depth"] = max(socket_depth(f.get("axial", 0.0), f.get("side", 0.0), f.get("moment", 0.0)) if f else L.DEPTH, BACK_DEPTH.get((k, m["name"]), 0.0))
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
ROOT_MAX = 3.0
MIN_WALL, COLLAR_INDEX, TAPER_FROM, SHEAR_K = 2.55, 0.27, 1.0 / 6.0, 1.1   # MIN_WALL: 2.05 at the 0.5 annulus (the audit's drawn_wall floor 2.0, the 0.05 the 40-gon's flats)
XT_MPA, ZT_MPA = 46.0 * 0.85 / 1.1, 23.0 * 0.85 / 1.1    # topo/material.py PCTG Xt / Zt, KNOCKDOWN, SAFETY["crash"]


def collar_walls(moment, side, depth, bore_r, rise=0.0):
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
    return max(up(hi), mouth), mouth


class Wall(float):
    """a tapered collar's wall: its value the root wall, .mouth the wall at the mouth, the taper from .ta to .tb along the socket axis.
    + keeps the taper (the mouth cuts spare every other collar grown by 0.5)"""
    def __new__(cls, root, mouth, ta, tb):
        w = float.__new__(cls, root); w.mouth, w.ta, w.tb = float(mouth), float(ta), float(tb)
        return w

    def __add__(self, g):
        return Wall(float(self) + g, self.mouth + g, self.ta, self.tb)


def drawn_wall(m):
    root, mouth = collar_walls(m.get("moment") or 0.0, m.get("side") or 0.0, m["depth"], m["bore_r"], float(m["u"][2]))
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


def _mouth_cuts():
    import manifold3d as mf
    allm = []
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
        cuts.append(slab - L.U(spare) if spare else slab)
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
    by = {m["name"]: m for m in ms}
    for (kk, hub), rider in BACK_RIDER.items():
        if kk == k and by[hub]["s0"] + by[hub]["depth"] < by[rider]["s0"] + by[rider]["depth"] - BACK_REACH - 1e-6:
            raise ValueError(f"node {k}: {hub}'s mouth ({by[hub]['s0'] + by[hub]['depth']:.1f}) no longer backs its rider {rider} "
                             f"({by[rider]['s0'] + by[rider]['depth']:.1f}): BACK_DEPTH")


_check0 = L.check_members


def _check_members(k, log=print):
    ms = _check0(k, log); check_pairs(k)
    return ms


L.check_members = _check_members
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
BUDGET_CM3 = {"P3": 1.7, "P4": 1.7, "P1": 1.0, "P2": 1.5, "P5": 1.5, "P6": 0.9, "R0": 0.8, "R7": 0.8,
              "RF": 0.45, "RB": 1.2, "RS0": 1.45, "RS1": 1.45, "S0": 1.3, "S1": 1.55, "NB": 1.0, "TB": 2.9}   # 14 Nov, the bolt forks: their runs read NB / TB 0.051 / 0.052 at 4.0, S0 / S1 0.032 at 3.1; on the fold's loads (core_of) TB-P3 154 N (was 99), S0-P2 109 (was 78): toward ~0.15 by index ~ V^-1.25.   # 13 Nov: NB 5.7 -> 6.9, 0.314 at 5.7 (index ~ V^-1.25 toward 0.25). 12 Nov: NB 4.3 -> 5.7, 0.354 at 4.3 once BS moved to the nose block (NB-R7 pull-out 126 N; index ~ V^-1.25 toward 0.25)   # 12 Nov: TB 4.8 -> 5.6, 0.301 at 4.8 under the heavier frame's TB-P3 108 N (index ~ V^-1.25 toward 0.25)


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
