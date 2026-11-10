"""Overseer head mount, v10 setup (8 Nov, Spark: "a new headmount that attaches to old rods one socket back since i will be cutting the
rods off at the socket ... the lidar pushed as far back as allowed to reduce the chances it gets hit in a collision and the camera
should also be back where it wont make contact in a collision"; "i dont care about the camera being mounted in the same way i only care
about how much vision it has"; option A of the 8 Nov study, Spark: "lets go with A"; "i can also cut the rods more if thats better").
ONE printed PCTG piece carrying the Unitree L2 and both cameras on the cage's front node (node 17) and the top crossbar 15-16.
* The old mount is cut off: every rod into it is cut flush at its old socket mouth. The new sockets start where the old mouths were
  and run one socket length further along each rod: node 17's 4 tubes 6/4 (bore 18..33 from the node centre), the front core tie
  3.0 (13..23). The head rods +1 / -1 are cut off flush at nodes 15 / 16 and dropped: their sockets one back would make the mount
  ~192 wide, over the 180 plate; the top crossbar 15-16 (uncut) runs 14 behind the L2's base plane and takes their place in a
  50 long saddle (y -25..25).
* v11 (9 Nov, Spark: go): the top crossbar 15-16 is cut in the middle (each half 5 3/4 in out of its node collar, ends at |y| 31.1)
  and the saddle is gone: the mount has one channelled socket on each cut end, bridging the gap, so nodes 15-16 stay tied through it
  (the bar's frame force, 239 N, loads each socket against the others). That lets the L2 sit 40 back and 40 up from v9: base
  (344.0, 0, 91.0), axis unchanged; the crossbar line passes on its base plane 39 above the axis.
* Slide-on from the front: the tubes and the crossbar sit in half-pipe channels open toward -x (the front pad's pattern: 7.6 wide,
  3.5 collar walls, full clearance from 2.6 out), so the mount is pushed straight back onto the 4 cut tube ends and the bar and
  glued; the core tie (along -x) goes axially into a closed injectable socket. A frontal hit pushes the mount onto the rods (the
  closed side of every channel).
* (v10, superseded by v11 above) L2: 20 back and 60 up from v9 (base (364.0, 0, 111.0), axis 15 deg below +x as before): the furthest back it can sit outside
  the cage, i.e. in front of node 17's 4 tubes (30..80 back it hits them; 105 back inside the cage costs 16 % of its field to the
  rods). Nothing of the drone in its field (8 Nov study: props 0 %, rods 0 %), 60 clear of the props, 11 of the rods.
* Cameras: under node 17, behind the cage's front face (x 354), lenses (340, +-30, 10), yawed 20 out, level: 0.0 / 0.1 % of their
  view blocked, nothing of the drone and not in the L2 field, ~17 above the landing nodes 18 / 19.
Drawn = attachments only: 4 L2 bolt pads, 2 x 4 camera standoffs, the socket collars. Everything between them is
generated. The L2 field (half ball R 220 ahead of its base plane) and both camera frustums are cuts. Frame: x fwd, y left, z up,
mm, z 0 = bottom plate top; geometry from systems/overseer.cage.json (C: l2_base and cams moved there 8 Nov), forces from its
node_loads (F)."""
import json, os
import numpy as np, trimesh, manifold3d as mf
from topo import Part
from topo.part import man
from parts.cage_node_lib import socket_cuts, collar_shape, cone, U as UNION
from parts.socket_inject import injectable_socket_cuts

HERE = os.path.dirname(os.path.abspath(__file__))
CAGE = json.load(open(os.path.join(HERE, "..", "..", "systems", "overseer.cage.json")))
SPEC = dict(h=2.0, gen_cm3=100.0, max_worst=0.3, settings=dict(near_sym=True))     # v10 (8 Nov): the node rules (limit 0.3, budget x 1.4: 60 -> 84); 9 Nov: 84 gave 0.305 (core tie tension 425 N) -> 100      # v8 (30 Oct 11 pm): L2 keys removed (Spark); v9 (1 Nov): L2 bolts a diamond, fully mirrored   # h 2.5: 184k cells (h 1.5 was 917k and killed the 7 GB
#   runner); refine once the shape is accepted. near_sym (29 Oct, Spark: the camera arms came out different left and right): the part
#   mirrors about y but for the L2's key slot, so the design is held symmetric away from it

G = 9.81e-3                                                    # N per g per g of mass
NODES = np.array(CAGE["nodes"], float)
N17 = NODES[17]
A = np.array(CAGE["l2_axis"], float); A /= np.linalg.norm(A)    # L2 axis, 15 deg below +x
B = np.array(CAGE["l2_base"], float)                            # L2 base-plane centre
L2_CG = B + np.array([27.3, 0.0, -5.0])                         # CG 27.3 ahead / 5 below the base centre (v9: (411.3, 0, 46) on base (384.0, 0, 51), C)
Y = np.array([0.0, 1.0, 0.0])
P = np.cross(A, Y); P /= np.linalg.norm(P)                     # up in the base plane (a x y): (0.259, 0, 0.966)
CABLE = -P                                                      # cable notch down (L2 rolled 180, dimensions.md "Sensor head")

# ---- numbers and tags -------------------------------------------------------------------------------------------------
L2_R_BOLT, L2_ROT = 25.5, 0.0       # Unitree L2: 4 x M3, 6 deep, on a O51 circle, 22.5 deg off its cable axis (dimensions.md, spec P).
                                    # v9 (1 Nov, Spark: the mount must be fully symmetric): the L2 is rolled 22.5 deg so the 4 M3 sit as a
                                    # diamond (down, +y, up, -y) and the mount mirrors about y; its cable then leaves 22.5 deg off down
CABLE_ROLL = 22.5                   # cable slot on one side only (Spark 1 Nov); near_sym holds the rest of the design mirrored
L2_PAD_R, L2_PAD_T = 8.0, 8.0       # O16 pads (v5 row), 8 deep behind the base plane; M3 hole O3.4, head/driver tunnel O6.5 from 5 behind
KEY_R, KEY_W, KEY_L, KEY_H, KEY_BACK = 30.0, 3.3, 3.3, 2.0, 6.0   # L2 manual drawing p8 (28 Oct): 4 positioning slots 3.5 wide on a O60 circle at the
                                    # body axes (0/90/180/270 deg from the cable axis): keys 3.5 - 0.2 fit = 3.3 wide, 3.3 long (radial, E), 2.0 tall
                                    # ahead of the base plane. SLOT DEPTH IS NOT IN THE MANUAL: the 2.0 key height is E (check on the unit).
CAM_SQ = 14.0                       # DECXIN 2633V2: M2 on 28 x 28 (Spark 29 Oct, coupon; read as M2.5 before); heat-set insert hole O3.2 x 5.7 (E, dimensions.md); boss O12 (v5 row)
CAM_BOARD_BACK = 12.0               # board back face 12 behind the lens plane (E placeholder: lens and thickness open, Q3); body 38 x 38 (P)
CAM_BOSS_L = 10.0                   # standoff length (29 Oct, Spark: the board's plugs sit on its back near the holes, so the board stands
CAM_TIP_R, CAM_BASE_R = 2.8, 2.8     #   10 off the body on 4 posts O5.6 straight (1.2 wall round the O3.2 x 5.7 M2 insert; Spark 29 Oct, coupon A fits the
                                    #   board and the insert, the O7.6 -> O11 cones did not clear it; 10 long so the plug and its wire fit behind the board)
CAM_PLUG_BACK = 8.0                 # keep-out behind the whole board between the standoffs: the plugs (nothing grows there)
TUBE_BORE_R, TUBE_DEPTH = 3.1, 15.0 # 6/4 carbon: bore 6.2, 15 deep (Spark 27 Oct, coupons v2)
ROD_BORE_R, ROD_DEPTH = 1.6, 10.0   # 3.0 rod: bore 3.2, 10 deep
WALL, FLOOR = 3.0, 3.0              # 3.0 (8 Nov node rules: the injection annulus), floor 3.0
RING_T, GAP_W, LIP_D, WIDE, DROP = 3.5, 7.6, 2.6, 18.0, 70.0   # channelled sockets (the front pad v3 pattern): collar wall, half-pipe width, narrow depth, clearance, sweep
SLIDE = np.array([-1.0, 0.0, 0.0])  # the mount goes on moving straight back (-x)
BAR_COLLAR = 30.8 + TUBE_DEPTH      # crossbar 15-16: the node 15 / 16 socket mouths, 45.8 from the node centres (cage_node_lib SOCKET_START 30.8 + 15)
BAR_CUT_IN = 5.75                   # 9 Nov (Spark): the crossbar is cut in the middle; each half cut 5 3/4 in out of its node collar, so the
                                    #   rod ends sit at |y| 31.1, the collars 1.6 clear of the L2 body (r 39) on the base plane
BAR_END = float(np.linalg.norm(np.array(CAGE["nodes"][15], float)[1])) - BAR_COLLAR - 25.4 * BAR_CUT_IN   # |y| of the cut ends
TUBE_MOUTH, ROD_MOUTH = FLOOR + TUBE_DEPTH, FLOOR + ROD_DEPTH   # mouth distance from the node centre: bore bottom 3.0 from the centre (E)
L2_MASS = 230.0                                                  # P
CAM_MASS = 25.0                                                  # E: board 15 + a heatsink on its back 10 (Spark 29 Oct)
FIELD_R = 220.0                                                  # L2 field: R 220 from its window
FIELD_EYE, FIELD_HALF = 40.0, 98.0                               # 9 Nov: the field is the L2's 96 deg cone (+2) from its window 40 ahead of the base, not a half ball
                                                                 #   ahead of the base plane: v10's L2 sits 60 up, node 17 is 9.7 ahead of that plane but 116 deg off the axis

DIRS14 = [np.array(v, float) / np.linalg.norm(v) for v in
          [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
FLIGHT = {"flight 4 g": np.array([0, 0, 4.0]), "flight 2 g x": np.array([2.0, 0, 1]), "flight 2 g y": np.array([0, 2.0, 1]),
          "landing 6 g": np.array([0, 0, 6.0])}


# ---- helpers ----------------------------------------------------------------------------------------------------------
def U(ms):
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)


def along(q0, q1, r, n=32):
    """a cylinder of radius r from point q0 to point q1"""
    return man(trimesh.creation.cylinder(r, float(np.linalg.norm(np.subtract(q1, q0))), sections=n, segment=np.vstack([q0, q1])))


def hull(pts):
    return man(trimesh.convex.convex_hull(np.asarray(pts, float)))


def fbox(o, e1, e2, e3, r1, r2, r3):
    """convex box at origin o with orthonormal axes e1..e3 and ranges r1..r3 (min, max)"""
    return hull([o + a * e1 + b * e2 + c * e3 for a in r1 for b in r2 for c in r3])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def local(s, y, p):
    return B + s * A + y * Y + p * P


# ---- members that end at the mount -------------------------------------------------------------------------------------
def trough(P, u, s0, s1, collar_r, wide=WIDE):
    """a channelled socket's opening (the front pad v3 pattern): the slot GAP_W wide from the axis out along SLIDE projected off the
    rod axis, the whole DROP (the rod's path as the mount slides on); full clearance WIDE only from LIP_D to just past the collar,
    where the collar beside the slot would otherwise feather under 2 mm (9 Nov: run 2's WIDE band ran the whole 70 out of the part
    and left two members stopped short at its wall, stubs ending in air)"""
    u = np.asarray(u, float); d = SLIDE - np.dot(SLIDE, u) * u; d /= np.linalg.norm(d); n = np.cross(u, d)
    T = np.c_[u, n, d, np.asarray(P, float)]
    return mf.Manifold.cube([s1 - s0, GAP_W, DROP]).translate([s0, -GAP_W / 2, 0.0]).transform(T) + \
        mf.Manifold.cube([s1 - s0, wide, collar_r + 1.5 - LIP_D]).translate([s0, -wide / 2, LIP_D]).transform(T)


class Sock:
    """a socket one socket length further along a cut rod (8 Nov): the rod ends at the old mouth (t_cut from `end`, the member's
    node end), so the new bore runs from there to t_cut + depth toward `other`; channel=True: a half-pipe open toward -SLIDE"""
    def __init__(self, name, end, other, kind, t_cut, channel):
        self.name, self.end = name, np.asarray(end, float)
        self.d = unit(np.asarray(other, float) - self.end)
        self.kind, self.channel = kind, channel
        self.bore_r, self.depth = (TUBE_BORE_R, TUBE_DEPTH) if kind == "tube" else (ROD_BORE_R, ROD_DEPTH)
        self.wall = RING_T if channel else WALL
        self.collar_r = self.bore_r + self.wall
        self.t_bot = t_cut; self.t_mouth = t_cut + self.depth
        self.t_floor = self.t_bot - FLOOR                           # collar starts one floor behind the bore bottom
        self.mouth = self.pt(self.t_mouth)
        self.wide = WIDE                                            # the clearance band's width beside the channel

    def pt(self, t):
        return self.end + self.d * t

    def collar(self):
        if self.channel:
            return along(self.pt(self.t_floor), self.pt(self.t_mouth), self.collar_r)
        return collar_shape(self.end, self.d, self.t_floor, self.t_mouth, self.bore_r, self.wall)      # follows the bore's crown

    def bores(self):
        if self.channel:                                            # the grooved bore + its half-pipe opening, from the floor plane past the mouth
            return socket_cuts(self.pt(self.t_bot), self.d, 0.0, self.depth, self.bore_r) + [trough(self.pt(self.t_bot), self.d, 0.0, self.depth + 7.0, self.collar_r, self.wide)]
        port = np.array([0.0, 0.0, -1.0]) - self.d[2] * self.d; port /= np.linalg.norm(port)   # 9 Nov: down, between the cameras (up, into the hub, left a 1.1 mm wall)
        return injectable_socket_cuts(self.pt(self.t_bot), self.d, 0.0, self.depth, self.bore_r, port=port)

    def clear(self, length=300.0):
        """the member outside its mouth: nothing of the mount may grow into the carbon (r + 1.0)"""
        return along(self.pt(self.t_mouth + 0.5), self.pt(self.t_mouth + length), self.bore_r - (0.1 if self.kind == "tube" else 0.0) + 0.9)

    def region(self, extra=0.4):
        """the collar's grip zone (support / member load): bore wall from the bore bottom to the mouth"""
        return along(self.pt(self.t_bot), self.pt(self.t_mouth - 1.0), self.collar_r + extra)


def sockets():
    S = {}
    for (a, b) in CAGE["rods"]:
        if 17 in (a, b):
            o = b if a == 17 else a
            nm = f"tube {min(a, b)}-{max(a, b)}"
            S[nm] = Sock(nm, N17, NODES[o], "tube", FLOOR + TUBE_DEPTH, channel=True)       # v9 mouth 18 from the node
    S["core tie front.2"] = Sock("core tie front.2", N17, CAGE["links"]["core tie front.2"][0], "rod", FLOOR + ROD_DEPTH, channel=False)   # v9 mouth 13
    mid = 0.5 * (NODES[15] + NODES[16])                                                  # 9 Nov: the cut crossbar, one channelled socket per end
    for nm, o in (("crossbar 15-16 +y", NODES[15]), ("crossbar 15-16 -y", NODES[16])):
        S[nm] = Sock(nm, mid, o, "tube", BAR_END - 0.5, channel=True)                  # bore bottom 0.5 inside the rod end (saw kerf)
        S[nm].wide = 2 * S[nm].collar_r                             # 10 Nov: the 18 band carved +-9 round the bar behind its r 6.6 collar and left two
                                                                    #   12 mm members ending in air under it (349, +-35, 115); the collar's own width
    return S


# ---- L2 and cameras ------------------------------------------------------------------------------------------------------
def l2_bolts():
    out = []
    for k in range(4):
        phi = np.radians(L2_ROT + 90.0 * k)
        out.append(B + L2_R_BOLT * (np.cos(phi) * CABLE + np.sin(phi) * Y))      # in the base plane
    return out


def cams():
    out = []
    for c in CAGE["cams"]:
        out.append(dict(lens=np.array(c["lens"], float), f=unit(c["f"]), u=unit(c["u"]), r=unit(c["r"]), ha=c["ha"], hv=c["hv"]))
    return out


def cam_boss_pts(c):
    """centres of the 4 insert bosses' front faces (board back plane)"""
    base = c["lens"] - c["f"] * CAM_BOARD_BACK
    return [base + sr * CAM_SQ * c["r"] + su * CAM_SQ * c["u"] for sr in (1, -1) for su in (1, -1)], base


def frustum(c, margin_deg=2.0, length=400.0):
    """the camera view: apex at the lens, half angles + margin, 400 long"""
    ha, hv = np.radians(c["ha"] + margin_deg), np.radians(c["hv"] + margin_deg)
    far = c["lens"] + c["f"] * length
    w, hh = length * np.tan(ha), length * np.tan(hv)
    pts = [c["lens"] - c["f"] * 0.5] + [far + sr * w * c["r"] + su * hh * c["u"] for sr in (1, -1) for su in (1, -1)]
    return hull(pts)


def bodies():
    S = sockets()
    cs = cams()
    bolts = l2_bolts()
    drawn, cuts = [], []
    # --- L2 bolt pads: O16 x 8 behind the base plane, M3 hole through the whole envelope, driver tunnel O6.5 from 5 behind
    for q in bolts:
        drawn.append(along(q - A * L2_PAD_T, q, L2_PAD_R))
        cuts.append(along(q - A * 80.0, q + A * 1.0, 1.7))
        cuts.append(along(q - A * 80.0, q - A * 5.0, 3.25))
    # (30 Oct 11 pm, Spark: the L2 notches were in the wrong place, right outside the screw holes on the unit; the 4 M3 locate the L2:
    #  the positioning keys, their stems and bars are gone)
    # --- camera standoffs: 4 cones 10 long behind the board back plane, insert hole O3.2 x 5.7 from the front; the board, its body and
    # the plugs behind it are cuts (29 Oct)
    envs_cam, posts = [], []                                  # posts: the O5.6 / 1.2 wall standoffs, declared thin_ok to the gate
    for c in cs:
        pts, base = cam_boss_pts(c)
        cones = []
        for q in pts:
            cones.append(cone(q, -c["f"], 0.0, CAM_BOSS_L, CAM_TIP_R, CAM_BASE_R))                              # standoff cone
            cuts.append(along(q + c["f"] * 1.0, q - c["f"] * 5.7, 1.6))
        drawn += cones; posts += cones
        cuts.append(fbox(base, c["f"], c["r"], c["u"], (0.0, 25.0), (-22.0, 22.0), (-22.0, 22.0)))          # board + body (38 x 38 + clearance)
        cuts.append(fbox(base, c["f"], c["r"], c["u"], (-CAM_PLUG_BACK, 0.0), (-19.0, 19.0), (-19.0, 19.0)) - UNION(cones))   # plugs behind the board
        # (29 Oct: the drawn 38 x 38 frame joining the 4 camera bosses is gone, as on node 21; the topology joins them)
        cuts.append(frustum(c))
        envs_cam.append(fbox(base, c["f"], c["r"], c["u"], (-34.0, 0.0), (-22.0, 22.0), (-22.0, 22.0)))
    # --- sockets
    for s in S.values():
        drawn.append(s.collar())
        cuts += s.bores()
        cuts.append(s.clear())
    # --- L2 field: half ball R 220 ahead of the base plane; cable slot below the L2 flange (the cable leaves the base downward)
    eye = B + A * FIELD_EYE
    ball = mf.Manifold.sphere(FIELD_R, 96).translate(list(eye))
    back = cone(eye, -A, 0.0, FIELD_R + 5.0, 0.0, (FIELD_R + 5.0) * np.tan(np.radians(180.0 - FIELD_HALF)), 96)   # the unseen cone behind the window
    cuts.append(ball - back)
    cuts.append(along(B, B + A * 70.0, 39.0, 96))                                   # the L2 body (O75 x 65, + 1)
    dn = np.cos(np.radians(-CABLE_ROLL)) * CABLE + np.sin(np.radians(-CABLE_ROLL)) * Y   # v9: the cable leaves 22.5 deg off down (-y side)
    cuts.append(fbox(B, A, np.cross(A, dn), -dn, (-60.0, 1.0), (-7.0, 7.0), (-55.0, -32.5)))   # 9 Nov: a notch past the flange (was to -120: with the L2 60 up it cut 120 down through the mount's right side and left members ending at its wall)
    # --- envelope: slab behind the base (O80 x 46), camera blocks behind the boards, a seat round every collar (+4), one hull over all
    slab = along(B - A * 46.0, B, 40.0, 48)
    seats = [along(s.pt(s.t_floor - 4.0), s.pt(s.t_mouth), s.collar_r + 4.0) for s in S.values()]
    pts = []
    for m in [slab] + envs_cam + seats:
        pts.append(np.asarray(m.to_mesh().vert_properties)[:, :3])
    bridge = hull(np.vstack(pts))
    env = U([slab, bridge] + envs_cam + seats)
    # the 180 mm plate: the envelope is clipped to |y| <= 89.9 so the generated part cannot outgrow it (the audit's plate check)
    env = env ^ mf.Manifold.cube([600.0, 2 * 89.9, 400.0]).translate([100.0, -89.9, -100.0])
    return U(drawn), U(cuts), env, dict(S=S, cs=cs, bolts=bolts, posts=posts)


def setup(h):
    drawn, cuts, env, X = bodies()
    S, cs, bolts = X["S"], X["cs"], X["bolts"]
    p = Part("overseer_head_mount", h, drawn, cuts, env, build="+z", material="PCTG", thin_ok=X["posts"])   # the engine's PETG model; printed in PCTG (8 Nov), as the nodes   # the 1.2 mm post walls pass the gate
    # supports: every socket (the cage holds the head there); the 3 rods of node 17's neighbours are the cage, node 17 itself is this part
    for nm, s in S.items():
        p.support(nm, s.region())
    p.support("all sockets", U([s.region() for s in S.values()]))

    # payload regions: the L2 pads' tops (force + moment through the 4 M3), the camera bosses' front faces
    l2_reg = [along(q - A * 4.0, q + A * 0.3, L2_PAD_R) for q in bolts]
    cam_reg = []
    for c in cs:
        pts, _ = cam_boss_pts(c)
        cam_reg.append((pts, [along(q - c["f"] * 3.0, q + c["f"] * 0.3, 5.0) for q in pts]))

    def rigid_split(pts, F, Pt):
        pts = np.asarray(pts, float); c0 = pts.mean(0); n = len(pts); Am = np.zeros((6, 3 * n))
        for i, q in enumerate(pts - c0):
            Am[:3, 3 * i:3 * i + 3] = np.eye(3)
            Am[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
        Mc = np.cross(np.asarray(Pt, float) - c0, F)
        return (np.linalg.pinv(Am) @ np.r_[F, Mc]).reshape(n, 3)

    cam_cg = [c["lens"] - c["f"] * 8.9 for c in cs]               # assembly CG: 8.9 behind the lens (E)

    def payload(name, a, cls):
        f = rigid_split(bolts, -L2_MASS * G * a, L2_CG)
        reg = [(l2_reg[i], f[i]) for i in range(4)]
        for (pts, regs), cg in zip(cam_reg, cam_cg):
            fc = rigid_split(pts, -CAM_MASS * G * a, cg)
            reg += [(regs[i], fc[i]) for i in range(4)]
        p.load_multi(name, reg, cls=cls, accel=-a, support="all sockets")

    # cases: crash 35 g in the 14 directions (the nose hit is +x deceleration through the L2 base: a = -35 x), flight, landing
    for k, d in enumerate(DIRS14):
        payload(f"crash {k} (35 g)" + (" nose hit" if k == 1 else ""), 35.0 * d, "crash")
    for nm, a in FLIGHT.items():
        payload(nm, a, "flight")

    # node 17's member forces at its sockets (frame model, F): each member loaded alone, the others held (tension = pulls the mount
    # toward the member's far end, compression pushes in), plus the side force across the member in the horizontal plane
    nl = dict(CAGE["node_loads"]["17"])
    bar = CAGE["node_loads"]["15"]["tube 15-16"]                  # 9 Nov: the cut crossbar's tie now runs through the mount, end to end
    nl.update({"crossbar 15-16 +y": bar, "crossbar 15-16 -y": bar})
    others_of = lambda nm: [n for n in S if n != nm]
    for nm, ld in nl.items():
        if nm not in S:
            continue
        s = S[nm]
        side = unit(np.cross(s.d, [0, 0, 1.0]))
        sup = "held but " + nm
        p.support(sup, U([S[o].region() for o in others_of(nm)]))
        for sgn, lab in ((1, "tension"), (-1, "compression")):
            F = sgn * ld["axial"] * s.d + ld["side"] * side
            p.load_multi(f"{nm} {lab} {ld['axial']:.0f} N + side {ld['side']:.0f} N", [(s.region(), F)], cls="crash", support=sup)
    return p
