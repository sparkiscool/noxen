"""Scout v4 generated parts (9 Nov, Spark: "generate everything with topology"): the shared pieces of the frame, the two base units and
the probe's turret, head frame and camera cradle. Geometry comes from tools/scout_v4_shape.py (the design every check runs on); the
rod sockets' loads from its frame model (systems/scout_v4.cage_loads.json); masses from systems/scout_v4.assembly.json and the battery
set (BATTERY_SET, not drawn: it mounts on the 4 M3 bolts).

Rules (CLAUDE.md): draw only attachments (pads, bosses, socket collars, seats, lands); everything between them is generated; bolts,
inserts, rods and tool paths cut through the whole envelope; supports are the faces a part really bolts to; loads are the carried
masses at their CG (35 g crash in 14 directions, flight, landing) and the joint reactions of the frame model, each a case of its own."""
import json, os, sys
import numpy as np, trimesh, manifold3d as mf
from topo.part import man
from parts.cage_node_lib import U
from parts.scout_units_lib import box, cyl, along, hull, unit, rigid_split, Sock, DIRS14, FLIGHT, check_collars, G, FLOOR, DEPTH, BORE_R, WALL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import scout_v4_shape as S                                   # noqa: E402

NL4 = json.load(open(os.path.join(ROOT, "systems", "scout_v4.cage_loads.json")))["node_loads"]
ASM = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "systems", "scout_v4.assembly.json")))["parts"]}
DRONE_G = sum(p["mass_g"] for p in ASM.values()) + S.BATTERY_SET[0]
HIT_N = round(2 * DRONE_G * G * 600.0 / 15.0, -1)            # ground hit (the Overseer's rule: 2 m g h / 15 mm from 0.6 m): the battery mechanism's feet take it into its 4 bolts
M3_INS, M3_CLR, M2_INS, M25_INS = (2.0, 6.0), 1.7, (1.6, 4.5), (1.75, 4.5)   # heat-set insert holes (r, depth): M3 x 5.7 O4 x 6; M3 clearance r; M2 O3.2; M2.5 O3.5
BOSS_R = 5.0                                                  # O10 boss round an M3 insert (wall 3)


def m(t):
    """a trimesh from the shape tool as a manifold"""
    return man(t)


def ins_cut(x, y, z, r_d, up=True, extra=1.0):
    """an insert hole r_d = (r, depth) from z going up (into material above) or down"""
    r, d = r_d
    return cyl(x, y, z - (0 if up else d), z + (d if up else 0), r) if True else None


def socks_of(node, ring_r=None):
    """the 3 mm rod sockets of a printed node: one per rod ending there, toward its far node. ring_r: the node is a ring (RM): its
    sockets sit with their mouths at ring_r from the centre, along each rod"""
    out = []
    for a, b in S.RODS:
        if node in (a, b):
            o = b if a == node else a
            c, far = np.array(S.P(node), float), np.array(S.P(o), float)
            if ring_r is not None:
                u = unit(far - c); c = c + u * (ring_r - FLOOR - DEPTH)
            out.append(Sock(f"{node} {a}-{b}", c, far))
    return out


def socket_cases(p, socks, support="default"):
    """the v4 frame model's member forces at each socket: tension and compression along it, plus the side force across it"""
    for s in socks:
        node, mem = s.name.split()
        ld = NL4[node][mem]
        for sgn, lab in ((1, "tension"), (-1, "compression")):
            F = sgn * ld["axial"] * s.d + ld["side"] * s.side()
            p.load_multi(f"{mem} {lab} {ld['axial']:.0f} N + side {ld['side']:.0f} N", [(s.region(), F)], cls="crash", support=support)


def socket_bodies(socks, prefer_out):
    """drawn collars; cuts: bores (with the injection pattern), the rods beyond the mouths, the ports out of the part"""
    drawn, cuts = [], []
    for s in socks:
        drawn.append(s.collar())
        cuts.append(s.port(unit(prefer_out(s)), socks)); cuts += s.bores(); cuts.append(s.clear())
    return drawn, cuts


def payload_cases(p, name, pts, regs, mass_g, cg, support="default", crash=True, flight=True):
    """a carried mass on its attachment points: 35 g crash in 14 directions, flight 4 g, 2 g x / y, landing 6 g; forces by rigid_split"""
    if crash:
        for k, d in enumerate(DIRS14):
            a = 35.0 * d; f = rigid_split(pts, -mass_g * G * a, cg)
            p.load_multi(f"{name} crash {k} (35 g {np.round(d, 2).tolist()})", [(regs[i], f[i]) for i in range(len(pts))], cls="crash", accel=-a, support=support)
    if flight:
        for nm, a in FLIGHT.items():
            f = rigid_split(pts, -mass_g * G * a, cg)
            p.load_multi(f"{name} {nm}", [(regs[i], f[i]) for i in range(len(pts))], cls="flight", accel=-a, support=support)


def point_case(p, name, pts, regs, F, at, cls="crash", support="default"):
    """one force F applied at `at`, carried by the attachment points (rigid_split)"""
    f = rigid_split(pts, np.asarray(F, float), at)
    p.load_multi(name, [(regs[i], f[i]) for i in range(len(pts))], cls=cls, support=support)


def disc_cuts():
    """the swept prop discs grown by the cage rule (4) and the printed-part keep-out (12)"""
    r = S.PROP_R + 4.0 + S.KEEP; z0, z1 = S.Z_PROP - S.Z_SWEEP - S.KEEP, S.Z_PROP + S.Z_SWEEP + S.KEEP
    return [cyl(mx, my, z0, z1, r, 96) for mx, my in S.MOTORS]


def bolt_pad(x, y, z0, z1, r=BOSS_R):
    return cyl(x, y, z0, z1, r)


def region_top(x, y, z, r=BOSS_R, t=1.5):
    return cyl(x, y, z - t, z + 0.3, r)


def region_bot(x, y, z, r=BOSS_R, t=1.5):
    return cyl(x, y, z - 0.3, z + t, r)


# ---- the interfaces the parts share (all drone frame, with BX where the part is under the deck) ---------------------------------
DZ0, DZ1 = S.DECK_Z0, S.DECK_TOP                               # 18.5, 24.5: the deck's underside and top
BASE_BOLTS = [(12.0, 25.0), (12.0, -25.0), (40.0, 25.0), (40.0, -25.0), (-12.0, 25.0), (-12.0, -25.0), (-40.0, 18.0), (-40.0, -18.0)]   # (-40, +-18): ahead of the Radxa, clear of the UBEC and the battery bolts   # M3 from the deck's top down into inserts in the base units (12+ from the battery bolts)
NOSE_BOLTS, TAIL_BOLTS = BASE_BOLTS[:4], BASE_BOLTS[4:]
SPLIT_X = 0.0 + S.BX                                           # the base splits into a nose unit and a tail unit here (183 mm long: the A1 mini prints 180)
TURRET_BOLTS = [(S.SPOOL["c"][0] + dx, dy) for dx in (-13.0, 13.0) for dy in (-21.0, 19.0)]   # M3 down through the turret's floor into the deck's inserts, beside the gear and the drum (driver from above)
REC_BOLTS = [(x + S.BX, y) for x, y in S.REC_BOLTS]           # the battery mechanism's 4 M3, up into inserts in the base units' yokes
