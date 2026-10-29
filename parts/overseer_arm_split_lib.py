"""Overseer front arm in two printed pieces (Spark 29 Oct: every print fits the A1 mini's 180 plate; the arm's one job is to put the
motor where the layout says, relative to the plates, so any construction that keeps that precision is fine).

The one-piece front arm is 226 x 65 mm, over the plate at every rotation (the audit's plate check). It is cut at a plane across the
arm axis, JOINT_S (130 mm) out from the root centre: the ROOT piece (plate bosses to the joint, ~144 mm) and the TIP piece (joint to
the motor pad with the 3 spoke sockets, ~74 mm). Everything about the arm itself (plate bosses, motor pad, sockets, neighbours,
loads, numbers) is parts/overseer_arm_lib.py; this module only adds the joint and splits the loads.

The joint, in the arm frame (s along the axis a, v across it, z up; parts/overseer_arm_lib.py Arm):
  * 2 pins, 6.0 carbon x 40 (3 mm and 6 mm carbon only), at v +-PIN_V, z PIN_Z, 20 deep each side in 6.2 bores with 2.0 walls and a
    3.0 floor (the cage socket rule): they set the tip's position and angles (6.0 in 6.2 over 26 mm between the pins: ~0.1 mm at the motor);
  * 2 x M3 along the axis at v +-JB_V, z JB_Z: head in the root piece (counterbore behind its pad, driver sweep straight back through
    the root envelope), nut in a hex pocket open at the tip piece's joint face. They clamp the two contact pads together;
  * contact faces: on the tip the ends of the 2 bolt bosses (O11, 3.0 floor + 2.6 nut pocket, the plate boss rule) and of the 2 pin
    collars at the joint plane, 4 spots; on the root one joint block 8 thick round all four (29 Oct: four spots never stayed joined).
Loads: the TIP piece carries every case of the one-piece arm (flight, landing, stiffness, the cage frame's crash set with the three
spokes loaded together, tools/cage_loads.py) on the joint as its support (pins + contact face; "+spokes" for the flight cases), plus
the joint bolts' pull / pin shear from the joint reactions (least norm over the 4 joint points). The ROOT piece carries those same
cases as the joint reactions (force and moment of the motor and spoke loads about the joint, applied over the 4 joint points) on the
plate lips, and the plate bolts' pull / shear from them as in the one-piece arm.
"""
import numpy as np, manifold3d as mf
from topo import Part
from parts.overseer_arm_lib import (Arm, cyl, box, U, along, hull_xy, stadium, rigid_split, frame_cases, mvec, G, Z_BOT, Z_PAD, Z_THRUST, BOLT_TOP, BOLT_BOT,
                                    HOLE_D, NUT_AF, NUT_T, FLOOR, BOSS_R, HEAD_D, HEAD_T, SWEEP_D, M_MOTOR, M_ESC, Z_MOTOR_CG, THRUST_N,
                                    TORQUE_NM, NODE_R, SOCK_BORE_R, SOCK_DEPTH, SOCK_WALL, CAGE)

JOINT_S = 130.0                                  # the joint plane, mm out along the arm axis from the root centre (C)
PIN_R, PIN_BORE_R, PIN_HALF = 3.0, 3.1, 20.0     # 6.0 carbon pin, 6.2 bore, 20 into each piece
PIN_V, PIN_Z = 13.0, 31.0                        # the two pins, across the axis and up
JB_V, JB_Z = 13.0, 13.0                          # the two joint bolts (M3 along the axis)
PAD_V, PAD_Z0, PAD_Z1 = 20.0, 3.0, 42.0          # the joint face region (support selector on the tip piece)
PAD_T = NUT_T + FLOOR                            # 5.6: floor + nut pocket (tip) or head seat (root)
JB_T, JB_W = 7.0, PIN_V + 6.0                    # the root's joint block: 7 thick at the face, |v| <= 19, z 7..37, a O10 window in its
#   middle (29 Oct; 8 thick without the window read as a 10 mm drawn block to the audit at the coarse pitch)
SWEEP_BACK = 160.0                               # driver sweep from the root pad back through the whole root envelope


class SplitArm(Arm):
    """one piece of the split arm: piece 'root' or 'tip'"""

    def __init__(self, name, theta_deg, motor_idx, spokes, node, piece, s_joint=JOINT_S):
        super().__init__(name, theta_deg, motor_idx, spokes, node)
        assert piece in ("root", "tip")
        self.piece, self.sJ = piece, float(s_joint)
        self.R3, self.A3, self.N3 = np.r_[self.R, 0.0], np.r_[self.a, 0.0], np.r_[self.n, 0.0]
        self.deg = float(np.degrees(np.arctan2(self.a[1], self.a[0])))

    # ---- arm frame (s, v, z) -> drone
    def p(self, s, v, z):
        return self.R3 + self.A3 * s + self.N3 * v + np.array([0.0, 0.0, z])

    def slab(self, s0, s1, v0, v1, z0, z1):
        return mf.Manifold.cube([s1 - s0, v1 - v0, z1 - z0]).rotate([0, 0, self.deg]).translate(self.p(s0, v0, z0).tolist())

    def hexa(self, s0, s1, v, z):
        """hex nut pocket along the axis, s0..s1, AF NUT_AF"""
        r = NUT_AF / np.sqrt(3.0)
        return mf.Manifold.cylinder(s1 - s0, r, r, 6).rotate([0, 90, 0]).rotate([0, 0, self.deg]).translate(self.p(s0, v, z).tolist())

    def axial(self, s0, s1, v, z, r):
        return along(self.p(0.0, v, z), self.A3, s0, s1, r)

    def joint_points(self):
        """the 4 joint regions of this piece as (point, region manifold): pins (shear) and bolts (pull)"""
        sJ, sg = self.sJ, (1.0 if self.piece == "tip" else -1.0)
        out = []
        for v in (PIN_V, -PIN_V):
            out.append((self.p(sJ + sg * PIN_HALF / 2, v, PIN_Z), self.axial(min(sJ + sg * 0.5, sJ + sg * PIN_HALF), max(sJ + sg * 0.5, sJ + sg * PIN_HALF), v, PIN_Z, PIN_BORE_R + 1.4)))
        for v in (JB_V, -JB_V):
            if self.piece == "tip":
                seat = self.axial(sJ + NUT_T, sJ + NUT_T + 2.5, v, JB_Z, 3.3)       # nut seat behind the pocket (2.5 thick for the coarse pitch)
            else:
                seat = self.axial(sJ - PAD_T + HEAD_T, sJ - PAD_T + HEAD_T + 2.5, v, JB_Z, 3.3)   # head seat at the counterbore floor
            out.append((self.p(sJ + sg * (NUT_T + 0.6), v, JB_Z), seat))
        return out

    # ------------------------------------------------------------------------------------------------- geometry
    def drawn_cuts_env(self):
        full_drawn, full_cuts, _ = super().drawn_cuts_env()      # the one-piece arm's features (sets self.env_pts, spoke linings)
        sJ = self.sJ
        drawn, cuts = [], []
        if self.piece == "tip":
            # keep: motor pad and spoke sockets; the plate bosses lie beyond the joint plane and are clipped off by the envelope
            for v in (JB_V, -JB_V):
                drawn.append(self.axial(sJ, sJ + PAD_T, v, JB_Z, BOSS_R))                                # bolt boss, nut side (a pad is a block)
            for v in (PIN_V, -PIN_V):
                drawn.append(self.axial(sJ, sJ + PIN_HALF + FLOOR, v, PIN_Z, PIN_BORE_R + SOCK_WALL))    # pin collar + floor
                cuts.append(self.axial(sJ - 1.0, sJ + PIN_HALF, v, PIN_Z, PIN_BORE_R))                  # pin bore
            for v in (JB_V, -JB_V):
                cuts.append(self.hexa(sJ - 0.5, sJ + NUT_T, v, JB_Z))                                    # nut pocket, open at the face
                cuts.append(self.axial(sJ - 2.0, sJ + 14.0, v, JB_Z, HOLE_D / 2))                        # shank + bolt end clearance
            corners = [self.p(sJ, v, 0.0)[:2] for v in (-PAD_V - 4.0, PAD_V + 4.0)]
            env = hull_xy([(c[0], c[1], 0.0) for c in corners] + [(self.M[0], self.M[1], 27.0), (self.node[0], self.node[1], 17.0)], Z_BOT, Z_PAD)
            self.env_pts = [np.array(c) for c in corners]
        else:
            # one joint block at the face (29 Oct): the two bolt bosses and two pin collars were four small bodies the engine never
            # kept joined (three root runs: bosses off, 56 eroded bodies, the ties stuck at 6e4); as the plate boss blocks, a
            # block 8 thick round all four fasteners keeps a core the gate holds and one body to join. The pin collars run on
            # behind it to the pins' full depth.
            drawn.append(self.slab(sJ - JB_T, sJ, -JB_W, JB_W, JB_Z - 6.0, PIN_Z + 6.0))
            cuts.append(self.axial(sJ - JB_T - 0.5, sJ + 0.5, 0.0, 0.5 * (JB_Z + PIN_Z), 5.0))            # window: no solid core far from a face
            for v in (PIN_V, -PIN_V):
                drawn.append(self.axial(sJ - PIN_HALF - FLOOR, sJ - JB_T + 0.5, v, PIN_Z, PIN_BORE_R + SOCK_WALL))
                cuts.append(self.axial(sJ - PIN_HALF, sJ + 1.0, v, PIN_Z, PIN_BORE_R))
            for v in (JB_V, -JB_V):
                cuts.append(self.axial(sJ - 14.0, sJ + 2.0, v, JB_Z, HOLE_D / 2))                        # shank
                cuts.append(self.axial(sJ - PAD_T - 0.5, sJ - PAD_T + HEAD_T, v, JB_Z, HEAD_D / 2))      # head counterbore
                cuts.append(self.axial(sJ - SWEEP_BACK, sJ - PAD_T + HEAD_T, v, JB_Z, SWEEP_D / 2))      # driver sweep, whole envelope
            r0 = self.env_pts
            corners = [self.p(sJ, v, 0.0)[:2] for v in (-PAD_V - 4.0, PAD_V + 4.0)]
            env = hull_xy([(p[0], p[1], 0.0) for p in r0] + [(c[0], c[1], 0.0) for c in corners], Z_BOT, Z_PAD)
            self.env_pts = list(r0) + [np.array(c) for c in corners]
        # the one-piece features on this side of the joint plane stay (the pegs stand proud of the envelope: a half-space, not the envelope)
        side = self.slab(sJ, sJ + 400.0, -200.0, 200.0, -60.0, 120.0) if self.piece == "tip" else self.slab(sJ - 400.0, sJ, -200.0, 200.0, -60.0, 120.0)
        return U([full_drawn ^ side] + drawn), U([full_cuts] + cuts), env

    def footprint(self):
        """minimum-area rectangle of this piece's plan hull (mm): (long, short, angle deg)"""
        from scipy.spatial import ConvexHull
        pts = [np.asarray(p, float) for p in self.env_pts]
        if self.piece == "tip":
            pts += [self.M + 27.0 * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 6.28, 24)]
            pts += [self.node[:2] + 17.0 * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 6.28, 24)]
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


def motor_cases(arm):
    """the one-piece arm's motor cases: [(name, cls, F at the motor CG, Mz, accel)]"""
    thrust = np.array([0, 0, THRUST_N]); out = []
    for tag, sgn in (("+", 1), ("-", -1)):
        a = np.array([0, 0, 4.0])
        out.append((f"flight 4 g, thrust 21 N, motor torque {tag}0.5 N m", "flight", -M_MOTOR * G * a + thrust, sgn * TORQUE_NM * 1000.0, -a))
    a = np.array([0, 0, 6.0])
    out.append(("landing 6 g", "flight", -M_MOTOR * G * a, 0.0, -a))
    return out


def build_piece(arm, h, log=print):
    drawn, cuts, env = arm.drawn_cuts_env()
    p = Part(f"overseer_arm_{arm.name}_{arm.piece}", h, drawn, cuts, env, build="+z", material="PETG", log=log)
    jp = arm.joint_points()
    jpts = np.array([q for q, _ in jp]); cJ = jpts.mean(0)
    c_m = np.array([arm.M[0], arm.M[1], Z_MOTOR_CG])
    stiff_a = np.array([0, 0, 4.0]); F_stiff = np.array([0, 0, THRUST_N]) + M_MOTOR * G * stiff_a

    if arm.piece == "tip":
        # ---- supports: the joint (pins + contact face), and + the 3 spoke sockets
        face = arm.slab(arm.sJ, arm.sJ + 2.5, -PAD_V, PAD_V, PAD_Z0, PAD_Z1)
        joint = U([face] + [reg for (_, reg) in jp[:2]])
        p.support("joint", joint)
        p.support("joint+spokes", U([joint] + [sp["lining"] for sp in arm.spokes]))
        mb = arm.motor_bolts()
        mreg = [cyl(q[0], q[1], Z_PAD - 2.0, Z_PAD + 0.2, 4.0) for q in mb]
        mpts = np.array([[q[0], q[1], Z_PAD] for q in mb])

        def payload(F_m, M_z=0.0):
            fm = rigid_split(mpts, F_m, np.cross(c_m - mpts.mean(0), F_m) + mvec(M_z), mpts.mean(0))
            return [(mreg[i], fm[i]) for i in range(4)]
        reactions = []
        for name, cls, F, Mz, acc in motor_cases(arm):
            p.load_multi(name, payload(F, Mz), cls=cls, support="joint+spokes", accel=acc)
            reactions.append((c_m, F, Mz))
        p.load_multi("STIFFNESS tip alone on the joint: motor 96 g, 4 g and 21 N thrust up", payload(F_stiff), cls="flight", support="joint", accel=stiff_a)
        for name, cls, a, sp in frame_cases(arm):                    # the cage frame's crash set: spokes + motor inertia, the joint holds
            Fm = M_MOTOR * G * a; Fs = sum(sp.values())
            p.load_multi(name, payload(Fm) + [(s_["lining"], sp[s_["name"]]) for s_ in arm.spokes], cls=cls, support="joint", accel=a)
            reactions.append((c_m, Fm + Fs, np.cross(arm.node - c_m, Fs)))
    else:
        # ---- supports: the plate lips, as the one-piece arm
        lips = []
        for s in (1, -1):
            A, B = arm.xy(BOLT_TOP[0], s * BOLT_TOP[1]), arm.xy(BOLT_BOT[0], s * BOLT_BOT[1])
            lips += [stadium(A, B, Z_PAD - 2.5, Z_PAD, BOSS_R), stadium(A, B, Z_BOT, Z_BOT + 2.5, BOSS_R)]   # 2.5: a 1.0 lip holds no cell at the coarse pitch 4.0
        p.support("plates", U(lips))

        def via_joint(P_, F, Mz=0.0):
            """the tip's load (F at P_, moment Mz) carried into the root piece over its 4 joint regions"""
            f = rigid_split(jpts, F, np.cross(P_ - cJ, F) + mvec(Mz), cJ)
            return [(jp[i][1], f[i]) for i in range(4)]
        reactions = []
        for name, cls, F, Mz, acc in motor_cases(arm):
            p.load_multi(name + " (via the joint)", via_joint(c_m, F, Mz), cls=cls, support="plates", accel=acc)
            reactions.append((c_m, F, Mz))
        p.load_multi("STIFFNESS arm alone: motor 96 g, 4 g and 21 N thrust up (via the joint)", via_joint(c_m, F_stiff), cls="flight", support="plates", accel=stiff_a)
        for name, cls, a, sp in frame_cases(arm):                    # the cage frame's crash set, arriving over the joint
            Fm = M_MOTOR * G * a; Fs = sum(sp.values()); Ms = np.cross(arm.node - c_m, Fs)
            p.load_multi(name + " (via the joint)", via_joint(c_m, Fm + Fs, Ms), cls=cls, support="plates", accel=a)
            reactions.append((c_m, Fm + Fs, Ms))
        # ---- plate bolts: pull and shear from the reactions (the one-piece arm's rule)
        B = arm.bolts()
        bpts = np.array([[b["xy"][0], b["xy"][1], b["z"]] for b in B]); cB = bpts.mean(0)
        worst = [dict(pull=0.0, shear=np.zeros(3)) for _ in B]
        for P_, F, Mz in reactions:
            f = rigid_split(bpts, F, np.cross(P_ - cB, F) + mvec(Mz), cB)
            for i, b in enumerate(B):
                tens = f[i][2] * (1.0 if b["nut"] == "bot" else -1.0)
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

    # ---- the joint's own fasteners: bolt pull along the axis and pin shear across it, from the reactions (least norm over the 4 points)
    sg = 1.0 if arm.piece == "tip" else -1.0
    worst_pull, worst_shear = [0.0, 0.0], [np.zeros(3), np.zeros(3)]
    for P_, F, Mz in reactions:
        f = rigid_split(jpts, F, np.cross(P_ - cJ, F) + mvec(Mz), cJ)
        for i in range(2):                                           # pins: the part across the axis
            sh = f[i] - (f[i] @ arm.A3) * arm.A3
            if np.linalg.norm(sh) > np.linalg.norm(worst_shear[i]):
                worst_shear[i] = sh
        for i in range(2, 4):                                        # bolts: pulling this piece off the other = tension
            worst_pull[i - 2] = max(worst_pull[i - 2], sg * float(f[i] @ arm.A3))
    for i, v in enumerate((PIN_V, -PIN_V)):
        sh = worst_shear[i]
        if arm.piece == "root":                                      # on the tip piece the pins are the support: no case to run there
            p.load_multi(f"joint pin v{v:+.0f} shear {np.linalg.norm(sh):.0f} N", [(jp[i][1], sh)], cls="crash", support="plates")
    for i, v in enumerate((JB_V, -JB_V)):
        pl = max(worst_pull[i], 1.0)
        p.load_multi(f"joint bolt v{v:+.0f} pull {pl:.0f} N", [(jp[2 + i][1], sg * pl * arm.A3)], cls="crash", support=("joint" if arm.piece == "tip" else "plates"))
        log(f"  joint bolt v{v:+.0f}: pull {pl:.0f} N; pin shear {np.linalg.norm(worst_shear[i]):.0f} N")
    fp = arm.footprint()
    log(f"  {arm.piece} piece footprint {fp[0]:.0f} x {fp[1]:.0f} mm at {fp[2]:.0f} deg, 45 tall")
    return p
