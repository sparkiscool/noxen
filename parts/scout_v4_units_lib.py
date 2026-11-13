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


def socks_of(node, ring_r=None, cls=None):
    """the 3 mm rod sockets of a printed node: one per rod ending there, toward its far node. ring_r: the node is a ring (RM): its
    sockets sit with their mouths at ring_r from the centre, along each rod"""
    out = []
    for a, b in S.RODS:
        if node in (a, b):
            o = b if a == node else a
            c, far = np.array(S.P(node), float), np.array(S.P(o), float)
            if ring_r is not None:
                u = unit(far - c); c = c + u * (ring_r - FLOOR - DEPTH)
            out.append((cls or Sock)(f"{node} {a}-{b}", c, far))
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
TURRET_BOLTS = [(S.SPOOL["c"][0] + dx, dy) for dx in (-13.0, 13.0) for dy in (-21.0, 19.0)]   # M3 down through the turret's floor into the deck's inserts, beside the gear and the drum (driver from above)
REC_BOLTS = [(x + S.BX, y) for x, y in S.REC_BOLTS]           # the battery mechanism's 4 M3, up into inserts in the frame's yokes (9 Nov: one body)


def socket_depth(axial, side):
    """the bore depth a socket's frame-model load needs (parts/scout_v4_node_lib.socket_depth, the NODES stream's rule of 10 Nov:
    G/flex 650 bond 56.5 N per mm past the 2.0 lip, mouth bearing 4 V / (D d) at 5.3 MPa, 7.0 the injection pattern's least)"""
    d = max(7.0, 2.0 + abs(axial) / (12.0 * np.pi * 3.0 / 2.0), 4.0 * abs(side) / (2 * BORE_R * (0.3 * 23.0 * 0.85 / 1.1)))
    return float(np.ceil(d * 2.0) / 2.0)


class SockDown(Sock):
    """a socket printed with the build axis down (-z: the one-body frame prints deck-top down, 9 Nov): its collar's and bore's teardrop
    crown point to -z, the print's up. Its depth from its own load (socket_depth, 10 Nov); the mouth never moves out past the standard
    socket's (the frame's end sockets are at the A1 mini's edge): a deeper bore takes its extra behind the standard floor (the rod
    that much longer), a shallower one keeps the floor and its mouth comes in"""
    BUILD = (0.0, 0.0, -1.0)

    def __init__(self, name, c, far):
        super().__init__(name, c, far)
        node, mem = name.split()
        ld = NL4[node][mem]
        self.depth = socket_depth(ld["axial"], ld["side"])
        self.t_floor = FLOOR - max(0.0, self.depth - DEPTH)
        self.t_mouth = self.t_floor + self.depth

    def collar(self):
        from parts.cage_node_lib import collar_shape
        return collar_shape(self.c, self.d, min(0.0, self.t_floor), self.t_mouth, BORE_R, WALL, build=self.BUILD)   # a deep socket's floor behind its bore is generated (11 Nov: DT's collar met the Radxa)

    def bores(self):
        from parts.socket_inject import injectable_socket_cuts
        return injectable_socket_cuts(self.pt(self.t_floor), self.d, 0.0, self.depth, BORE_R, build=self.BUILD, port=getattr(self, "port_d", None))

    def region(self, extra=0.4):
        return along(self.pt(self.t_floor), self.pt(self.t_mouth - 1.0), BORE_R + WALL + extra)

    def port(self, prefer, others=()):
        from parts.cage_node_lib import port_dir
        from parts.socket_inject import FEED_H, PORT_R, VENT_R
        self.port_d = port_dir(self.d, prefer, [o.d for o in others if o is not self])
        f, v = self.pt(self.t_floor + FEED_H / 2), self.pt(self.t_mouth - 2.0 - VENT_R)
        return U([along(f, f + self.port_d * 150.0, PORT_R), along(v, v + self.port_d * 150.0, VENT_R)])


TUBE_BORE_R, TUBE_DEPTH_MIN = 3.1, 15.0                       # a 6 mm (6/4) carbon tube: 6.2 bore, 15 deep at least (13 Nov, the design chat)


def socket_depth6(axial, side):
    """socket_depth for a 6.2 bore: G/flex 650 at the nodes' 12 MPa over pi x 6 / 2 = 113 N per mm past the 2.0 lip; mouth bearing
    4 V / (D d) at 5.3 MPa; 15.0 at least"""
    d = max(TUBE_DEPTH_MIN, 2.0 + abs(axial) / (12.0 * np.pi * 6.0 / 2.0), 4.0 * abs(side) / (2 * TUBE_BORE_R * (0.3 * 23.0 * 0.85 / 1.1)))
    return float(np.ceil(d * 2.0) / 2.0)


class SockDown6(SockDown):
    """SockDown for a 6 mm tube (6.2 bore, the same 3.0 wall and the same injection pattern scaled to the bore)"""
    R = TUBE_BORE_R

    def __init__(self, name, c, far):
        Sock.__init__(self, name, c, far)
        node, mem = name.split()
        ld = NL4[node][mem]
        self.depth = socket_depth6(ld["axial"], ld["side"])
        self.t_floor = FLOOR - max(0.0, self.depth - DEPTH)
        self.t_mouth = self.t_floor + self.depth

    def collar(self):
        from parts.cage_node_lib import collar_shape
        return collar_shape(self.c, self.d, min(0.0, self.t_floor), self.t_mouth, self.R, WALL, build=self.BUILD)

    def bores(self):
        from parts.socket_inject import injectable_socket_cuts
        return injectable_socket_cuts(self.pt(self.t_floor), self.d, 0.0, self.depth, self.R, build=self.BUILD, port=getattr(self, "port_d", None))

    def region(self, extra=0.4):
        return along(self.pt(self.t_floor), self.pt(self.t_mouth - 1.0), self.R + WALL + extra)

    def clear(self, length=300.0):
        return along(self.pt(self.t_mouth + 0.5), self.pt(self.t_mouth + length), self.R + 0.9)
