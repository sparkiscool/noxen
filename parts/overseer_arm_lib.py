"""Overseer printed arm, one generator for the front pair and the rear pair (28 Oct, v6 setup).

ONE printed PETG piece from the plate bolts to the motor pad: the jig that puts the motor in place and square to the FC.
Drawn (attachments only, nothing between them):
  * 4 plate bosses: per side a top-face block and a bottom-face block (u 57.5..72, |v| 3..22, 8 tall; nut pocket or relief 2.6 in
    the face) round the two bolts (top bolt u 63.17 v +-15, bottom bolt u 63.74 v +-10, O section M row), 2 locating pegs (u 57.66,
    v +-11); 29 Oct: was a stadium r 5.5, 5.6 thick, whose core at the gate's 2 mm erosion was slivers between the shank holes;
  * the motor pad (disc 36 x 6, level with the arm top z 45) with 4 bolt slots r 8..9.5 on the 45 deg lines, shaft bore 8.6;
  * 3 socket collars for the 3 mm carbon spokes (bore 3.2, 10 deep, 3.0 floor, wall 2.0) aimed at their cage node;
Cut through the whole envelope: bolt shanks, nut pockets, motor-bolt tool sweeps, the shaft bore, the own spokes beyond their
collars, every other cage tube / node / link near the envelope, the prop disc, the pack / cartridge / receiver zone.
Generated: everything else, in a hull of the root block, the motor disc and the spoke node (z 0..45, between the plates' inner faces).

Supports: "plates" = the plate contact lips of the 4 stadiums (stiffness, bolt and spoke-check cases, the arm alone);
"plates+spokes" = the same + the 3 spoke sockets, fully fixed (main cases). The engine has fixed supports only (no spring, no
axis-only dofs along an oblique rod), so the three 3 mm rods that meet at the arm node are taken as a rigid pin there (EA/L of each
rod is 100x the arm's own tip stiffness, printed in the log); the stiffness case, the spoke check cases and the bolt cases do not lean on it.
Cases (each separate): flight 4 g with thrust and +-0.5 N m motor torque, landing 6 g (plates + spokes); STIFFNESS (arm alone, no
spokes: thrust 21 N + 4 g of the 96 g motor, all up); the cage frame model's crash set (tools/cage_loads.py, 29 Oct: the drone
decelerating 35 g onto each cage face it can land on, and landing 6 g: the three spokes loaded together as the frame finds them
with the arm at its own stiffness, the motor's and the arm's inertia with them, plates only; replaces the 14 crash directions at
the motor and each spoke's pull / push alone at a rigid-core maximum, which put 1465 N through one socket); each bolt's pull and
shear (joint reactions of the cases above by least norm over the 4 bolts, top bolts at z 45, bottom at z 0).
"""
import json, os
import numpy as np, trimesh, manifold3d as mf
from topo import Part
from topo.part import man
from parts.cage_node_lib import socket_cuts, collar_shape

HERE = os.path.dirname(os.path.abspath(__file__))
CAGE = json.load(open(os.path.join(HERE, "..", "systems", "overseer.cage.json")))
FRAME = json.load(open(os.path.join(HERE, "..", "systems", "overseer.cage_loads.json")))   # tools/cage_loads.py: the crash set
G = 9.81e-3                                      # N per g per g

# ---- numbers (dimensions.md; tags in the setup docstring of each arm)
Z_BOT, Z_PAD = 0.0, 45.0                         # bottom plate top, top plate underside = arm top = motor pad top (arms section, C)
Z_THRUST = 82.0                                  # prop plane = pad + 37 (O layout rows, C)
ROOT_R = 66.5                                    # root centre on the 45 deg diagonal (arms section, C)
BOLT_TOP = (63.17, 15.0)                         # u, |v|: top plate screws (O Arms and hub row, M)
BOLT_BOT = (63.74, 10.0)                         # bottom plate screws (M)
PEG = (57.66, 11.0, 5.8, 1.5)                    # u, |v|, dia, height (M)
HOLE_D = 3.2                                     # M3 shank (E: close fit like the rod bore; plates Ø3.1 / 3.0, M)
NUT_AF, NUT_T = 5.7, 2.6                         # M3 nut AF 5.5 x 2.4 + clearance (E: not recorded in dimensions.md)
BOSS_R, FLOOR = 5.5, 3.0                         # boss radius (2.2 wall round the relief), 3.0 floor behind the pocket
BOSS_U, BOSS_V, BOSS_T = (57.5, 72.0), (3.0, 22.0), 8.0   # plate boss block per side and face (29 Oct): u 57.5..72 (the lips reach 57.7; the Pi deck keeps u >= 58),
#   |v| 3..22, 8 tall. Was a stadium r 5.5 round the two bolts, 5.6 tall: its core at the gate's 2 mm erosion was slivers between the
#   shank holes, so no joint could keep it in the main body (root and rear arm: plate bosses off at the gate, the engine stuck on
#   their ties). This block keeps a 10 x 15 x 4 core round the holes that a member lands on full-face.
RELIEF_R = 3.0                                   # round the neighbour's shank in the nut layer, merges with the pocket (no sliver)
PAD_D, PAD_T = 36.0, 6.0                         # motor pad (arms section, C): Ø36 reach, 6 thick
LUG_R = 5.5                                      # lug radius round each slot (slot 1.6 + 2.0 wall + 1.9)
MOTOR_SLOT = (8.0, 9.5)                          # X2216 base pattern radii 16 / 19, 45 deg to the leads (O motor row, spec)
SLOT_D = 3.2                                     # M3 into the motor's tapped base (2.8 deep: no nut exists on a motor)
HEAD_D, HEAD_T = 6.2, 2.6                        # M3 pan head counterbore under the pad, leaves 3.4 of pad
SWEEP_D = 7.0                                    # driver sweep under each motor bolt, through the whole envelope
BORE_D = 8.6                                     # shaft / clip bore (arms section, CHECK P)
ESC_SEAT = (30.0, 16.0, 3.0)                     # pad on the arm top (E)
M_MOTOR, M_ESC = 96.0, 0.0                       # g: motor 81 + prop 10 + screws 5 (arms section, M); no ESC on the arms (arms section)
Z_MOTOR_CG = 60.0                                # motor CG (arms section)
THRUST_N, TORQUE_NM = 21.0, 0.5                  # arms section (E)
E_CARBON, E_PETG = 120e3, 2780.0                 # MPa (arms section)
NODE_R, SOCK_BORE_R, SOCK_DEPTH, SOCK_WALL = 10.15, 1.6, 10.0, 2.5     # cage node rule (cage_node_lib); wall 2.5 (29 Oct): 2.0 at the resin grooves
ROD_A = 7.07                                     # 3.0 solid carbon area, mm2 (dimensions.md C)



def cyl(x, y, z0, z1, r, n=40):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def U(ms):
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)


def along(c, u, s0, s1, r, n=32):
    return man(trimesh.creation.cylinder(r, s1 - s0, sections=n, segment=np.vstack([c + u * s0, c + u * s1])))


def hull_xy(circles, z0, z1):
    """convex hull (plan) of discs/points [(x, y, r)] extruded z0..z1 (r 0 = a point)"""
    pts = []
    for x, y, r in circles:
        a = np.linspace(0, 2 * np.pi, 24 if r > 0 else 1, endpoint=False)
        pts += [(x + r * np.cos(t), y + r * np.sin(t)) for t in a]
    pts = np.array(pts)
    pts3 = np.vstack([np.c_[pts, np.full(len(pts), z0)], np.c_[pts, np.full(len(pts), z1)]])
    return man(trimesh.convex.convex_hull(pts3))


def hex_nut(x, y, z0, z1, toward_deg):
    """hexagonal nut pocket, AF NUT_AF, one flat facing `toward_deg`"""
    r = NUT_AF / np.sqrt(3.0)
    return mf.Manifold.cylinder(z1 - z0, r, r, 6).rotate([0, 0, toward_deg - 30.0]).translate([x, y, z0])


def stadium(a, b, z0, z1, r):
    return hull_xy([(a[0], a[1], r), (b[0], b[1], r)], z0, z1)


def mvec(M):
    """a moment given as a z scalar or a vector"""
    return np.array([0, 0, M], float) if np.isscalar(M) else np.asarray(M, float)


def frame_cases(arm):
    """the cage frame model's cases on this arm (tools/cage_loads.py, systems/overseer.cage_loads.json): [(name, cls, inertial
    body force direction x g, {spoke name: force on the arm node, N})]: in each case the three spokes carry what the cage
    frame puts through them while the whole drone decelerates, the arm holding at its root"""
    out = []
    for c in FRAME["cases"]:
        a = float(c["g"]) * np.array(c["d"], float); sp = {}
        for s_ in arm.spokes:
            r = c["arms"][str(arm.motor_idx + 1)]["spokes"][s_["name"]]
            side = np.cross(s_["u"], [0, 0, 1.0]); side /= np.linalg.norm(side)
            sp[s_["name"]] = r["N"] * s_["u"] + r["side"] * side                 # N > 0: the rod pulls the node toward its far end
        tag = ", ".join(f"{k.split()[1]} {v['N']:+.0f}" for k, v in c["arms"][str(arm.motor_idx + 1)]["spokes"].items())
        out.append((f"cage {c['name']} (spokes {tag} N)", "crash" if c["name"].startswith("crash") else "flight", a, sp))
    return out


def rigid_split(pts, F, M, c):
    """least-norm forces at pts (n,3) that carry force F and moment M about c (a payload on a rigid link)"""
    pts = np.asarray(pts, float); n = len(pts); Am = np.zeros((6, 3 * n))
    for i, q in enumerate(pts - c):
        Am[:3, 3 * i:3 * i + 3] = np.eye(3); Am[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
    return (np.linalg.pinv(Am) @ np.r_[F, M]).reshape(n, 3)


class Arm:
    """geometry of one arm in the drone frame (x fwd, y left, z up, mm), left side (+y)"""

    def __init__(self, name, theta_deg, motor_idx, spokes, node):
        self.name, self.theta, self.motor_idx = name, np.radians(theta_deg), motor_idx
        self.ud = np.array([np.cos(self.theta), np.sin(self.theta)])            # plate-frame u (outward along the diagonal)
        self.vd = np.array([-np.sin(self.theta), np.cos(self.theta)])
        self.M = np.array(CAGE["motors_xy"][motor_idx], float)
        self.R = ROOT_R * self.ud
        # ---- asserts before anything runs (Spark's brief)
        assert abs(Z_PAD + 37.0 - Z_THRUST) < 1e-9, "thrust plane must be the pad + 37 (z 82)"
        assert abs(np.hypot(*self.M) - np.hypot(*CAGE["motors_xy"][motor_idx])) < 1e-9
        assert np.allclose(self.M, CAGE["motors_xy"][motor_idx]), "motor pad centre must be motors_xy"
        self.node = np.array(node, float)
        self.spokes = []
        for nm in spokes:
            p, q = (np.array(e, float) for e in CAGE["links"][nm])
            if np.linalg.norm(q - self.node) < 0.05:
                p, q = q, p
            assert np.linalg.norm(p - self.node) < 0.05, f"{nm} does not start at the arm node {node}"
            d = q - p; L = float(np.linalg.norm(d)); u = d / L
            far = int(np.argmin([np.linalg.norm(np.array(n_) - q) for n_ in CAGE["nodes"]]))
            assert np.linalg.norm(np.array(CAGE["nodes"][far]) - q) < 0.6, f"{nm} far end is not a cage node"
            ld = CAGE["node_loads"][str(far)][nm]
            self.spokes.append(dict(name=nm, u=u, len=L, far=far, axial=float(ld["axial"]), side=float(ld["side"]),
                                    k=E_CARBON * ROD_A / (L - 2 * NODE_R + 2 * SOCK_DEPTH)))
        a = self.M - self.R; self.len = float(np.linalg.norm(a)); self.a = a / self.len
        self.n = np.array([-self.a[1], self.a[0]])

    # ---- plate frame -> drone
    def xy(self, u, v):
        return self.ud * u + self.vd * v

    def bolts(self):
        """[(name, xy, z_head_side, nut_z0, nut_z1, face, other_xy)] for A+- (top bolts, nut at the bottom) and B+- (bottom bolts, nut at the top)"""
        out = []
        for s in (1, -1):
            A, B = self.xy(BOLT_TOP[0], s * BOLT_TOP[1]), self.xy(BOLT_BOT[0], s * BOLT_BOT[1])
            out.append(dict(name=f"top bolt v{s:+d}", xy=A, other=B, nut="bot", pair=s, z=Z_PAD))
            out.append(dict(name=f"bottom bolt v{s:+d}", xy=B, other=A, nut="top", pair=s, z=Z_BOT))
        return out

    def motor_bolts(self):
        """4 slots at 45, 135, 225, 315 deg to the arm axis; returns centre points at r 8.75 (middle of the slot)"""
        r = 0.5 * (MOTOR_SLOT[0] + MOTOR_SLOT[1])
        return [self.M + r * (np.cos(np.radians(t)) * self.a + np.sin(np.radians(t)) * self.n) for t in (45, 135, 225, 315)]

    def esc_c(self):
        s = 0.5 * (24.0 + self.len - PAD_D / 2 - 2.0)            # between the root block end (s 24) and the pad
        return self.R + self.a * s

    # ------------------------------------------------------------------------------------------------- geometry
    def drawn_cuts_env(self):
        # --- plate bosses (stadiums at the faces) and their cuts
        drawn, cuts = [], []
        for s in (1, -1):
            A, B = self.xy(BOLT_TOP[0], s * BOLT_TOP[1]), self.xy(BOLT_BOT[0], s * BOLT_BOT[1])
            dAB = np.degrees(np.arctan2(*(B - A)[::-1])); dBA = dAB + 180.0
            blk = [(q[0], q[1], 0.0) for q in (self.xy(u, s * v) for u in BOSS_U for v in BOSS_V)]
            # top face (z 37..45): nut of the BOTTOM bolt B, relief round the top bolt A (its head-side contact is the plate lip)
            drawn.append(hull_xy(blk, Z_PAD - BOSS_T, Z_PAD))
            cuts += [hex_nut(B[0], B[1], Z_PAD - NUT_T, Z_PAD + 0.5, dBA), cyl(A[0], A[1], Z_PAD - NUT_T, Z_PAD + 0.5, RELIEF_R)]
            # bottom face (z 0..8): nut of the TOP bolt A, relief round the bottom bolt B
            drawn.append(hull_xy(blk, Z_BOT, Z_BOT + BOSS_T))
            cuts += [hex_nut(A[0], A[1], Z_BOT - 0.5, Z_BOT + NUT_T, dAB), cyl(B[0], B[1], Z_BOT - 0.5, Z_BOT + NUT_T, RELIEF_R)]
            for P in (A, B):                                       # shanks through the whole envelope (and out of the plates' way)
                cuts.append(cyl(P[0], P[1], Z_BOT - 2.0, Z_PAD + 2.0, HOLE_D / 2, 24))
            pg = self.xy(PEG[0], s * PEG[1])
            drawn.append(cyl(pg[0], pg[1], Z_PAD - 0.5, Z_PAD + PEG[3], PEG[2] / 2, 24))
        # --- motor pad
        M = self.M
        for t in (45, 135, 225, 315):                                   # the pad is 4 lugs (a star), not a disc: a disc is a block to the audit
            e = np.cos(np.radians(t)) * self.a + np.sin(np.radians(t)) * self.n
            drawn.append(stadium(M + 4.0 * e, M + (PAD_D / 2 - LUG_R) * e, Z_PAD - PAD_T, Z_PAD, LUG_R))
        cuts.append(cyl(M[0], M[1], Z_BOT - 1.0, Z_PAD + 1.0, BORE_D / 2, 32))
        for t in (45, 135, 225, 315):
            e = np.cos(np.radians(t)) * self.a + np.sin(np.radians(t)) * self.n
            p0, p1 = M + MOTOR_SLOT[0] * e, M + MOTOR_SLOT[1] * e
            cuts.append(stadium(p0, p1, Z_PAD - PAD_T - 0.5, Z_PAD + 1.0, SLOT_D / 2))                  # shank slot through the pad
            cuts.append(stadium(p0, p1, Z_PAD - PAD_T - 0.5, Z_PAD - PAD_T + HEAD_T, HEAD_D / 2))       # head seat under it
            cuts.append(stadium(p0, p1, Z_BOT - 1.0, Z_PAD - PAD_T + HEAD_T, SWEEP_D / 2))              # driver sweep, whole envelope
        # (no ESC seat: the 4-in-1 ESC sits under the FC, none on the arms; dimensions.md System Overseer: arms, Spark 27 Oct)
        # --- spoke sockets (cage_node_lib rule: bore from NODE_R out, 10 deep, 3.0 floor, wall 2.0), aimed at their node
        for sp in self.spokes:
            u, P0 = sp["u"], self.node
            ro, s_end = SOCK_BORE_R + SOCK_WALL, NODE_R + SOCK_DEPTH
            drawn.append(collar_shape(P0, u, NODE_R - FLOOR, s_end, SOCK_BORE_R, SOCK_WALL))
            cuts += socket_cuts(P0, u, NODE_R, SOCK_DEPTH, SOCK_BORE_R)                 # coupon socket: grooves, lead-in, crown
            cuts.append(along(P0, u, s_end, s_end + 70.0, SOCK_BORE_R + 0.6))          # the rod beyond its collar: nothing grows on it
            sp["lining"] = along(P0, u, NODE_R - FLOOR - 0.5, s_end, SOCK_BORE_R + 1.4)
        # --- neighbours: prop disc, pack / cartridge / receiver zone, every other cage member near the envelope
        cuts.append(cyl(M[0], M[1], Z_THRUST - 14.0, Z_THRUST + 12.0, 127.0, 96))
        cuts.append(box(-125.0, 65.0, -40.0, 40.0, -110.0, 0.0))
        # --- envelope: hull of the root block, the motor disc and the node disc, z 0..45
        r0 = [self.xy(u, v) for u in (58.0, 90.0) for v in (-22.0, 22.0)]
        env = hull_xy([(p[0], p[1], 0.0) for p in r0] + [(M[0], M[1], 27.0), (self.node[0], self.node[1], 17.0)], Z_BOT, Z_PAD)
        self.env_pts = r0
        lo, hi = np.array(env.bounding_box())[:3], np.array(env.bounding_box())[3:]
        sp_names = {s_["name"] for s_ in self.spokes}
        for a_, b_ in CAGE["rods"]:
            p, q = np.array(CAGE["nodes"][a_]), np.array(CAGE["nodes"][b_])
            if self._near(p, q, lo, hi, 12.0):
                cuts.append(along(p, (q - p) / np.linalg.norm(q - p), 0.0, float(np.linalg.norm(q - p)), 4.5))
        for k, (p, q) in CAGE["links"].items():
            if k in sp_names:
                continue
            p, q = np.array(p, float), np.array(q, float)
            if self._near(p, q, lo, hi, 12.0):
                cuts.append(along(p, (q - p) / np.linalg.norm(q - p), 0.0, float(np.linalg.norm(q - p)), 3.5))
        for n_ in CAGE["nodes"]:
            n_ = np.array(n_, float)
            if np.all(n_ > lo - 14) and np.all(n_ < hi + 14):
                cuts.append(mf.Manifold.sphere(NODE_R + 2.0, 32).translate(n_.tolist()))
        return U(drawn), U(cuts), env

    @staticmethod
    def _near(p, q, lo, hi, m):
        t = np.linspace(0, 1, 60)[:, None]
        pts = p + (q - p) * t
        return bool(np.any(np.all((pts > lo - m) & (pts < hi + m), axis=1)))

    def footprint(self):
        """minimum-area bounding rectangle of the plan hull of everything drawn + the envelope (mm): (long, short, angle deg)"""
        from scipy.spatial import ConvexHull
        pts = [p for p in self.env_pts] + [self.M + 27.0 * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 6.28, 24)] \
            + [self.node[:2] + 17.0 * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 6.28, 24)]
        for sp in self.spokes:
            pts.append((self.node + sp["u"] * (NODE_R + SOCK_DEPTH))[:2])
        pts = np.array(pts); h = pts[ConvexHull(pts).vertices]
        best = None
        for ang in np.radians(np.arange(0, 90, 0.5)):
            c, s = np.cos(ang), np.sin(ang)
            r = h @ np.array([[c, -s], [s, c]]); ext = r.max(0) - r.min(0)
            if best is None or ext.prod() < best[0]:
                best = (ext.prod(), max(ext), min(ext), np.degrees(ang))
        return best[1], best[2], best[3]


def build(arm, h, log=print):
    drawn, cuts, env = arm.drawn_cuts_env()
    p = Part(f"overseer_arm_{arm.name}", h, drawn, cuts, env, build="+z", material="PETG", log=log)

    # ---- supports: plate contact lips (top and bottom faces of the 4 stadiums), and + the 3 spoke sockets
    lips = []
    for s in (1, -1):
        A, B = arm.xy(BOLT_TOP[0], s * BOLT_TOP[1]), arm.xy(BOLT_BOT[0], s * BOLT_BOT[1])
        lips += [stadium(A, B, Z_PAD - 2.5, Z_PAD, BOSS_R), stadium(A, B, Z_BOT, Z_BOT + 2.5, BOSS_R)]     # 2.5 (29 Oct): a 1.0 lip held no cell at the coarse pitch 4.0 and the run crashed
    plates = U(lips)
    p.support("plates", plates)
    p.support("plates+spokes", U([plates] + [sp["lining"] for sp in arm.spokes]))

    # ---- load points: motor bolt slots on the pad top, ESC seat corners
    mb = arm.motor_bolts()
    mreg = [cyl(q[0], q[1], Z_PAD - 2.0, Z_PAD + 0.2, 4.0) for q in mb]
    mpts = np.array([[q[0], q[1], Z_PAD] for q in mb])
    c_m = np.array([arm.M[0], arm.M[1], Z_MOTOR_CG])
    ereg, epts, c_e = mreg, mpts, c_m                                     # no ESC on the arm (M_ESC 0): its load terms vanish

    def payload(F_m, F_e, M_m=None):
        """parts list: motor loads (force at its CG, optional pure moment about it) and ESC force at its CG"""
        fm = rigid_split(mpts, F_m, np.cross(c_m - mpts.mean(0), F_m) + (np.zeros(3) if M_m is None else M_m), mpts.mean(0))
        fe = rigid_split(epts, F_e, np.cross(c_e - epts.mean(0), F_e), epts.mean(0))
        return [(mreg[i], fm[i]) for i in range(4)] + [(ereg[i], fe[i]) for i in range(4)]

    reactions = []                                                           # (F, point) pairs for the joint reactions

    def note(F_m, F_e, M_z=0.0):
        reactions.append((np.r_[arm.M, Z_MOTOR_CG], F_m, M_z))                # no ESC on the arm: F_e is zero

    thrust = np.array([0, 0, THRUST_N])
    for tag, sgn in (("+", 1), ("-", -1)):
        a = np.array([0, 0, 4.0])
        Fm, Fe = -M_MOTOR * G * a + thrust, -M_ESC * G * a
        Mz = np.array([0, 0, sgn * TORQUE_NM * 1000.0])
        p.load_multi(f"flight 4 g, thrust 21 N, motor torque {tag}0.5 N m", payload(Fm, Fe, Mz), cls="flight", support="plates+spokes", accel=-a)
        note(Fm, Fe, sgn * TORQUE_NM * 1000.0)
    a = np.array([0, 0, 6.0])
    p.load_multi("landing 6 g", payload(-M_MOTOR * G * a, -M_ESC * G * a), cls="flight", support="plates+spokes", accel=-a)
    note(-M_MOTOR * G * a, -M_ESC * G * a)
    # ---- the cage frame's crash set: the drone decelerating on a cage face, the three spokes loaded as the frame model finds them
    # (tools/cage_loads.py, the arm at its own stiffness there), the motor's inertia with them, the arm alone holding at the plates
    for name, cls, a, sp in frame_cases(arm):
        Fm, Fe = M_MOTOR * G * a, M_ESC * G * a
        p.load_multi(name, payload(Fm, Fe) + [(s_["lining"], sp[s_["name"]]) for s_ in arm.spokes], cls=cls, support="plates", accel=a)
        Fs = sum(sp.values()); note(Fm + Fs, Fe, np.cross(arm.node - np.r_[arm.M, Z_MOTOR_CG], Fs))

    # ---- STIFFNESS: the arm alone (no spokes), thrust 21 N + 4 g of motor and ESC, all up
    a = np.array([0, 0, 4.0])
    Fm, Fe = thrust + M_MOTOR * G * a, M_ESC * G * a
    p.load_multi("STIFFNESS arm alone: motor 96 g + ESC 25 g, 4 g and 21 N thrust up", payload(Fm, Fe), cls="flight", support="plates", accel=a)
    F_stiff = float(Fm[2] + Fe[2])

    # ---- bolts: pull and shear at each boss from the joint reactions (least norm over the 4 bolts)
    B = arm.bolts()
    bpts = np.array([[b["xy"][0], b["xy"][1], b["z"]] for b in B]); cB = bpts.mean(0)
    worst = [dict(pull=0.0, shear=np.zeros(3)) for _ in B]
    for P_, F, Mz in reactions:
        Mx = np.cross(P_ - cB, F) + mvec(Mz)
        f = rigid_split(bpts, F, Mx, cB)
        for i, b in enumerate(B):
            tens = f[i][2] * (1.0 if b["nut"] == "bot" else -1.0)          # + : the nut presses its seat toward the plate (bolt in tension)
            worst[i]["pull"] = max(worst[i]["pull"], tens)
            sh = np.array([f[i][0], f[i][1], 0.0])
            if np.linalg.norm(sh) > np.linalg.norm(worst[i]["shear"]):
                worst[i]["shear"] = sh
    for i, b in enumerate(B):
        x, y = b["xy"]
        if b["nut"] == "bot":
            seat = cyl(x, y, Z_BOT + NUT_T, Z_BOT + NUT_T + 2.5, 3.3); dz = +1.0      # 2.5 thick: a 1.2 seat held no cell at the coarse pitch 4.0
        else:
            seat = cyl(x, y, Z_PAD - NUT_T - 2.5, Z_PAD - NUT_T, 3.3); dz = -1.0
        wall = cyl(x, y, Z_BOT + NUT_T + FLOOR, Z_PAD - NUT_T - FLOOR, 2.7)
        pl, sh = max(worst[i]["pull"], 1.0), worst[i]["shear"]
        p.load_multi(f"{b['name']} pull {pl:.0f} N", [(seat, np.array([0, 0, dz * pl]))], cls="crash", support="plates")
        p.load_multi(f"{b['name']} shear {np.linalg.norm(sh):.0f} N", [(wall, sh)], cls="crash", support="plates")
        log(f"  {b['name']}: pull {pl:.0f} N, shear {np.linalg.norm(sh):.0f} N (joint reactions, least norm)")
    arm.report = dict(footprint=arm.footprint(), stiff_force_N=F_stiff,
                      spoke_k=[(s_["name"], round(s_["k"] / 1000.0, 2)) for s_ in arm.spokes])
    log(f"  envelope footprint {arm.report['footprint'][0]:.0f} x {arm.report['footprint'][1]:.0f} mm at {arm.report['footprint'][2]:.0f} deg, 45 tall; "
        f"stiffness force {F_stiff:.1f} N; spoke rod EA/L (kN/mm) {arm.report['spoke_k']}")
    return p
