"""Overseer front arm, motor PAD, v3 setup (7 Nov, Spark: the printed pads cracked at the motor mount; topology arms with a thick motor
mount, the same on all four). Arm 1 (+y); arm 4 is the y-mirror. Geometry and cases: parts/overseer_tube_arm_lib.py on
parts/overseer_arm_lib.py (flight 4 g + 21 N thrust +-0.5 N m on "tubes+spokes", landing 6 g, HANDLING, the cage crash set).
* Motor mount (arm lib, 8 Nov): pad 14 thick (Spark: thicker; 10 on 7 Nov), no head counterbore (the full 14 under each M3 x 18 head), plus a drawn ring boss
  O32 / O20 x 8 under the pad rim here (a solid boss reads as a drawn block at the audit).
* The old pads stay glued and their rods are cut flush behind them, one socket shorter (Spark 5 Nov): the sockets sit one socket
  length further along the rods (tubes 17 toward the root, spokes 12 further from the arm node), each a channel open outboard
  across the arm, a half-pipe as wide as the grooved bore (tubes 7.6, spokes 4.4: no snap lip, a lip under 2 mm is drawn material
  the audit refuses), so the pad slides onto the 4 fixed rod ends and the glue holds it (tape it while curing). Each channelled
  socket gets a thicker collar ring (tubes 3.5, spokes 3.0 wall) so the walls beside the channel stay over 2 mm, and the channel
  widens to full clearance outboard of where they would thin. The grooves and injection pattern stay.
* 25 Nov (Spark: more generated, only the motor bed thick, the rod joints thinner): the collar rings down to the least the 2 mm floor
  allows beside a channel (tubes 3.5 -> 2.9 wall, spokes 3.0 -> 2.7), the channel lips cut down to 1.0 / 0.8 past the rod axis for it;
  bores, grooves, channel widths and socket positions unchanged. The motor bed (pad, cone, ring boss) as on 9 Nov. The budget stays 40:
  the room filled solid reads 0.121 and thinned to 40 cm3 ~0.5, the 9 Nov layout 0.176 of the 0.2 limit.
* Printed on its side (build +y): layers along the arm and through the pad, thrust in-layer (the flat prints cracked across layers).
* Rod windows unchanged from the solid pad: tubes 4 1/8..4 9/16 in out of the root collar, spokes .1 5 1/16..5 3/8 in and .3
  5 1/4..5 9/16 in out of the cage node collar."""
import numpy as np, manifold3d as mf
from parts.overseer_tube_arm_lib import TubeArm, build_piece, TUBE_Z, TUBE_DEPTH
from parts.overseer_arm_lib import Z_PAD, PAD_T, cyl, NODE_R, SOCK_DEPTH

EXT_TUBE, EXT_SPOKE = 34.0, 24.0                                         # 9 Nov (Spark: replaced once already, cut again: two socket lengths back, was 17 / 12)
TUBE_GAP, SPOKE_GAP, DROP = 7.6, 4.4, 60.0                             # channel widths: wider than the grooved bore (7.2 / 4.2): a half-pipe, no lip (a lip under 2 mm is drawn material the audit refuses)
RING_T, RING_S = 2.9, 2.7                                                # collar wall beside a channelled tube / spoke socket: 25 Nov (Spark: the rod joints thinner) 3.5 / 3.0 -> the least
TUBE_LIP, SPOKE_LIP = 1.0, 0.8                                           # that leaves the lips beside the channel 2.0 (the floor) with the lips cut down from 2.6 / 1.9 past the axis: R^2 = (gap/2 + 2.0)^2 + lip^2
BOSS_R, BOSS_RI, BOSS_D = 16.0, 10.0, 8.0


def trough(P, u, slide, s0, s1, w, drop, lip_d, wide):
    """the socket's channel, stepped: the narrow slot (w) from the axis out to lip_d, where the collar lips beside it are still
    2 mm thick (a straight slot through a round collar feathers to nothing where its flat sides meet the outside: audit drawn_wall
    1.1 mm), then full clearance (wide) from lip_d outward"""
    u = np.asarray(u, float); d = slide - np.dot(slide, u) * u; d /= np.linalg.norm(d); n = np.cross(u, d)
    T = np.c_[u, n, d, np.asarray(P, float)]
    return mf.Manifold.cube([s1 - s0, w, lip_d + 0.01]).translate([s0, -w / 2, 0.0]).transform(T) + \
        mf.Manifold.cube([s1 - s0, wide, drop]).translate([s0, -wide / 2, lip_d]).transform(T)


class FrontPad(TubeArm):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pad_mouth -= EXT_TUBE                                         # tube sockets 34 closer to the root (two socket lengths)
        self.node_r = NODE_R + EXT_SPOKE                                   # spoke sockets 24 further out from the node (arm lib hook)

    def drawn_cuts_env(self):
        drawn, cuts, env = super().drawn_cuts_env()
        M = self.M; n3 = np.array([-self.a3[1], self.a3[0], 0.0])          # +n: outboard, the slide direction
        boss = cyl(M[0], M[1], Z_PAD - PAD_T - BOSS_D, Z_PAD - PAD_T + 0.01, BOSS_R, 64) - \
            cyl(M[0], M[1], Z_PAD - PAD_T - BOSS_D - 1.0, Z_PAD - PAD_T + 1.0, BOSS_RI, 64)
        from parts.overseer_arm_lib import along
        slots, rings = [], []
        for z in TUBE_Z:
            B = self.p(self.pad_mouth + TUBE_DEPTH, 0.0, z)
            slots.append(trough(B, -self.a3, n3, 0.0, TUBE_DEPTH + 7.0, TUBE_GAP, DROP, TUBE_LIP, 18.0))    # from the floor plane (not into the 3 mm floor: a 1 mm sliver of it failed the audit) to past the mouth
            rings.append(along(B, -self.a3, 0.0, TUBE_DEPTH + 1.0, 3.1 + RING_T))         # from the floor plane: a ring past it left a 1 mm slab beside the channel's clearance
        Bm = self.p(self.pad_mouth + TUBE_DEPTH, 0.0, 0.5 * sum(TUBE_Z))     # 25 Nov: the 4.0 fin of room between the two channels' clearances (z 23..27, thinner than a
        slots.append(trough(Bm, -self.a3, n3, 0.0, TUBE_DEPTH + 7.0, 5.0, DROP, TUBE_LIP, 5.0))   # member; cell_thick: 3 cells for 4.0) closed: one clearance for both tubes
        node = np.asarray(self.node, float)
        for sp in self.spokes:
            slots.append(trough(node, sp["u"], n3, self.node_r, self.node_r + SOCK_DEPTH + 16.0, SPOKE_GAP, DROP, SPOKE_LIP, 14.0))   # from the floor plane to past the mouth; walls >= 2.0 to d 1.9
            rings.append(along(node, sp["u"], self.node_r - 3.0, self.node_r + SOCK_DEPTH + 1.0, 1.6 + RING_S))   # 9 Nov: from behind the floor (two back, the 1.1 collar stands alone and its lib wall was 2.0 at the grooves)
        for sp in self.spokes:                                             # room round the pushed-out spoke collars (they leave z 0..45): a cone from the
            col = along(node, sp["u"], self.node_r - 6.0, self.node_r + SOCK_DEPTH + 4.0, 10.5)          # arm node (r 18) out to the collar (r 10.5)
            root = along(node, sp["u"], -2.0, 2.0, 18.0)                   # 10 Nov: the r 9 tube alone read 1.40 filled solid (room_full, cage crash, spoke 1.1 -238 N)
            env = env + mf.Manifold.batch_hull([col, root])
        return drawn + boss + mf.Manifold.batch_boolean(rings, mf.OpType.Add), cuts + mf.Manifold.batch_boolean(slots, mf.OpType.Add), env


ARM = FrontPad("front", 45.0, 0, ["spoke 1.1", "spoke 1.3"], [178.94, 146.7, 8.0], "pad")
SPEC = dict(h=1.5, gen_cm3=40.0, max_worst=0.2)     # 10 Nov: the spoke room a cone (room_full 1.40 -> 0.10, 22 % of it at 36): 36 -> 40   # 9 Nov (Spark: the motor must not tear off again): limit 0.3 -> 0.2, budget 28 -> 36 with the solid pad + cone   # 9 Nov: two sockets back the spoke collars reach 24 further out: 18 stopped at 1.03 (cage crash, spoke 1.1 -238 N) -> 28   # v3 (7 Nov): 18 (v2 passed at 18 with a skirt, 36 g, 0.27), limit 0.3 (printed PETG broke at 0.30 of the engine's strength)


def setup(h):
    p = build_piece(ARM, h)
    p.build = "+y"                                                         # printed on its side
    return p
