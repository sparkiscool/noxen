"""Overseer head mount (v6 setup, 28 Oct): ONE printed PETG piece that carries the Unitree L2 and both cameras and absorbs cage
node 17 (the cage node nearest the L2), so node 17 is no longer a separate part.

Drawn = attachments only: 4 L2 bolt pads, 2 x 4 camera insert bosses, the socket collars of node 17 (4 tubes 6/4 + the front core
tie 3.0) and of the head rods +1 / -1 (6/4). Everything between them is generated. The L2 field (half ball R 220 ahead of its
base plane) and both camera frustums are CUTS (void), so nothing can grow into them. Frame: x fwd, y left, z up, mm, z 0 = bottom
plate top; geometry from systems/overseer.cage.json (C), forces from its node_loads (F)."""
import json, os
import numpy as np, trimesh, manifold3d as mf
from topo import Part
from topo.part import man

HERE = os.path.dirname(os.path.abspath(__file__))
CAGE = json.load(open(os.path.join(HERE, "..", "..", "systems", "overseer.cage.json")))
SPEC = dict(h=1.5, gen_cm3=60.0, max_worst=0.5)

G = 9.81e-3                                                    # N per g per g of mass
NODES = np.array(CAGE["nodes"], float)
N17 = NODES[17]
A = np.array(CAGE["l2_axis"], float); A /= np.linalg.norm(A)    # L2 axis, 15 deg below +x
B = np.array(CAGE["l2_base"], float)                            # L2 base-plane centre
Y = np.array([0.0, 1.0, 0.0])
P = np.cross(A, Y); P /= np.linalg.norm(P)                     # up in the base plane (a x y): (0.259, 0, 0.966)
CABLE = -P                                                      # cable notch down (L2 rolled 180, dimensions.md "Sensor head")

# ---- numbers and tags -------------------------------------------------------------------------------------------------
L2_R_BOLT, L2_ROT = 25.5, 22.5      # Unitree L2: 4 x M3, 6 deep, on a O51 circle, 22.5 deg off the cable axis (dimensions.md, spec P);
                                    # which side of the cable axis the 22.5 deg goes is E (positive about +y here)
L2_PAD_R, L2_PAD_T = 8.0, 8.0       # O16 pads (v5 row), 8 deep behind the base plane; M3 hole O3.4, head/driver tunnel O6.5 from 5 behind
KEY_R, KEY_W, KEY_L, KEY_H, KEY_BACK = 30.0, 3.3, 3.3, 2.0, 6.0   # L2 manual drawing p8 (28 Oct): 4 positioning slots 3.5 wide on a O60 circle at the
                                    # body axes (0/90/180/270 deg from the cable axis): keys 3.5 - 0.2 fit = 3.3 wide, 3.3 long (radial, E), 2.0 tall
                                    # ahead of the base plane. SLOT DEPTH IS NOT IN THE MANUAL: the 2.0 key height is E (check on the unit).
CAM_SQ = 14.0                       # DECXIN 2633V2: M2.5 on 28 x 28 (P); heat-set insert hole O3.2 x 5.7 (E, dimensions.md); boss O12 (v5 row)
CAM_BOARD_BACK = 12.0               # board back face 12 behind the lens plane (E placeholder: lens and thickness open, Q3); body 38 x 38 (P)
CAM_BOSS_L = 10.0
TUBE_BORE_R, TUBE_DEPTH = 3.1, 15.0 # 6/4 carbon: bore 6.2, 15 deep (Spark 27 Oct, coupons v2)
ROD_BORE_R, ROD_DEPTH = 1.6, 10.0   # 3.0 rod: bore 3.2, 10 deep
WALL, FLOOR = 2.0, 3.0              # wall 2.0 (28 Oct), floor 3.0
TUBE_MOUTH, ROD_MOUTH = FLOOR + TUBE_DEPTH, FLOOR + ROD_DEPTH   # mouth distance from the node centre: bore bottom 3.0 from the centre (E)
L2_MASS, L2_CG = 230.0, np.array([411.3, 0.0, 46.0])            # P; CG from dimensions.md "Sensor head requirements" (C)
CAM_MASS = 15.0                                                  # E
FIELD_R = 220.0                                                  # L2 field half ball

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
class Sock:
    """a socket collar: carbon member line from `end` (the member's force point) toward `other`; bore bottom FLOOR behind the mouth"""
    def __init__(self, name, end, other, kind, mouth_at_end=False):
        self.name, self.end = name, np.asarray(end, float)
        self.d = unit(np.asarray(other, float) - self.end)          # toward the rest of the cage
        self.kind = kind
        self.bore_r, self.depth = (TUBE_BORE_R, TUBE_DEPTH) if kind == "tube" else (ROD_BORE_R, ROD_DEPTH)
        self.collar_r = self.bore_r + WALL
        if mouth_at_end:                                            # head rods: the mouth is at the force point, the bore runs into the mount
            self.t_mouth = 0.0
        else:                                                       # node 17: bore bottom FLOOR from the node centre
            self.t_mouth = FLOOR + self.depth
        self.t_floor = self.t_mouth - self.depth - FLOOR            # collar starts one floor behind the bore bottom
        self.mouth = self.end + self.d * self.t_mouth

    def pt(self, t):
        return self.end + self.d * t

    def collar(self):
        return along(self.pt(self.t_floor), self.pt(self.t_mouth), self.collar_r)

    def bore(self):
        return along(self.pt(self.t_mouth - self.depth), self.pt(self.t_mouth + 1.0), self.bore_r)

    def clear(self, length=300.0):
        """the member outside its mouth: nothing of the mount may grow into the carbon (r + 1.0)"""
        return along(self.pt(self.t_mouth + 0.5), self.pt(self.t_mouth + length), self.bore_r - (0.1 if self.kind == "tube" else 0.0) + 0.9)

    def region(self, extra=0.4):
        """the collar's grip zone (support / member load): bore wall from the bore bottom to the mouth"""
        return along(self.pt(self.t_mouth - self.depth), self.pt(self.t_mouth - 1.0), self.collar_r + extra)


def sockets():
    S = {}
    for (a, b) in CAGE["rods"]:
        if 17 in (a, b):
            o = b if a == 17 else a
            S[f"tube {min(a, b)}-{max(a, b)}"] = Sock(f"tube {min(a, b)}-{max(a, b)}", N17, NODES[o], "tube")
    S["core tie front.2"] = Sock("core tie front.2", N17, CAGE["links"]["core tie front.2"][0], "rod")
    for nm in ("head rod +1", "head rod -1"):
        p0, node = CAGE["links"][nm]
        S[nm] = Sock(nm, p0, node, "tube", mouth_at_end=True)
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
    # --- L2 positioning keys: small drawn bodies in the slots (they stand 2.0 ahead of the base plane, so the field cut leaves them out)
    keys = []
    for k in range(4):
        phi = np.radians(90.0 * k)
        rad, tan = np.cos(phi) * CABLE + np.sin(phi) * Y, -np.sin(phi) * CABLE + np.cos(phi) * Y
        keys.append(fbox(B + KEY_R * rad, A, rad, tan, (-KEY_BACK, KEY_H), (-KEY_L / 2, KEY_L / 2), (-KEY_W / 2, KEY_W / 2)))
    drawn += keys
    # --- camera insert bosses: O12 x 10 behind the board back plane, insert hole O3.2 x 5.7 from the front; the camera body is a cut
    envs_cam = []
    for c in cs:
        pts, base = cam_boss_pts(c)
        for q in pts:
            drawn.append(along(q, q - c["f"] * CAM_BOSS_L, 6.0))
            cuts.append(along(q + c["f"] * 1.0, q - c["f"] * 5.7, 1.6))
        cuts.append(fbox(base, c["f"], c["r"], c["u"], (0.0, 25.0), (-22.0, 22.0), (-22.0, 22.0)))          # board + body (38 x 38 + clearance)
        cuts.append(frustum(c))
        envs_cam.append(fbox(base, c["f"], c["r"], c["u"], (-34.0, 0.0), (-22.0, 22.0), (-22.0, 22.0)))
    # --- sockets
    for s in S.values():
        drawn.append(s.collar())
        cuts.append(s.bore())
        cuts.append(s.clear())
    # --- L2 field: half ball R 220 ahead of the base plane; cable slot below the L2 flange (the cable leaves the base downward)
    half = fbox(B, A, Y, P, (0.0, FIELD_R + 5), (-FIELD_R - 5, FIELD_R + 5), (-FIELD_R - 5, FIELD_R + 5))
    ball = mf.Manifold.sphere(FIELD_R, 96).translate(list(B))
    cuts.append((ball ^ half) - U(keys))
    cuts.append(fbox(B, A, Y, P, (-60.0, 1.0), (-7.0, 7.0), (-120.0, -32.5)))
    # --- envelope: slab behind the base (O80 x 46), camera blocks behind the boards, the head rod seat cylinders, one hull over all
    slab = along(B - A * 46.0, B, 40.0, 48)
    heads = [S[n].pt(S[n].t_floor - 6.0) for n in ("head rod +1", "head rod -1")]
    head_c = [along(S[n].pt(S[n].t_floor - 6.0), S[n].pt(S[n].t_mouth), S[n].collar_r + 4.0) for n in ("head rod +1", "head rod -1")]
    pts = []
    for m in [slab] + envs_cam + head_c:
        pts.append(np.asarray(m.to_mesh().vert_properties)[:, :3])
    bridge = hull(np.vstack(pts))
    env = U([slab, bridge] + envs_cam + head_c)
    return U(drawn), U(cuts), env, dict(S=S, cs=cs, bolts=bolts)


def setup(h):
    drawn, cuts, env, X = bodies()
    S, cs, bolts = X["S"], X["cs"], X["bolts"]
    p = Part("overseer_head_mount", h, drawn, cuts, env, build="+z", material="PETG")
    # supports: every socket (the cage holds the head there); the 3 rods of node 17's neighbours are the cage, node 17 itself is this part
    for nm, s in S.items():
        p.support(nm, s.region())
    p.support("all sockets", U([s.region() for s in S.values()]))

    # payload regions: the L2 pads' tops (force + moment through the 4 M3), the camera bosses' front faces
    l2_reg = [along(q - A * 4.0, q + A * 0.3, L2_PAD_R) for q in bolts]
    cam_reg = []
    for c in cs:
        pts, _ = cam_boss_pts(c)
        cam_reg.append((pts, [along(q - c["f"] * 3.0, q + c["f"] * 0.3, 6.0) for q in pts]))

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
    nl = CAGE["node_loads"]["17"]
    others_of = lambda nm: [n for n in S if n != nm]
    for nm, ld in nl.items():
        if nm not in S:
            continue                                     # head rod c: now inside this part (its ends are both on the mount)
        s = S[nm]
        side = unit(np.cross(s.d, [0, 0, 1.0]))
        sup = "held but " + nm
        p.support(sup, U([S[o].region() for o in others_of(nm)]))
        for sgn, lab in ((1, "tension"), (-1, "compression")):
            F = sgn * ld["axial"] * s.d + ld["side"] * side
            p.load_multi(f"{nm} {lab} {ld['axial']:.0f} N + side {ld['side']:.0f} N", [(s.region(), F)], cls="crash", support=sup)
    return p
