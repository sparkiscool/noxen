"""Overseer tube arm (29 Oct, Spark: "as light as possible", no line-up precision needed, everything glued, no clamps).

Each arm is two 6/4 carbon tubes, stacked (z 14 and 36, on the arm axis), running from a small printed ROOT block between the
plates to a printed motor PAD; both ends glued in coupon sockets 15 deep (parts/cage_node_lib.py socket_cuts: resin grooves,
lead-in, teardrop crown, 2.5 collar). The two spokes per arm (the .1 and .3; the .2 spokes were dropped the same day) end on the PAD.
  ROOT (between the plates, bolted and pegged to them as the one-piece arm was): the two plate boss blocks, the two pegs and two
       tube collars pointing out along the arm; the topology joins them. Nothing else.
  PAD: the motor pad lugs (slots r 8..9.5, shaft bore), the two 3 mm spoke collars and two tube collars facing the root.
Assembly: glue the tubes into the ROOT first (the ROOT bolted to the plates, the tubes pushed home against the socket floors),
then the PAD onto the tubes with the spokes dry-fitted so the pad sits where the cage puts it, then the spokes.
Loads (each its own case): PAD: flight 4 g + 21 N thrust with +-0.5 N m torque, landing 6 g, on "tubes+spokes"; HANDLING (the 96 g
motor at 2 g, no spokes) on "tubes"; the cage frame's crash set (tools/cage_loads.py: the three spokes together + the motor's
inertia) on "tubes". ROOT: every PAD case arrives as the tubes' forces (the pad load's force and moment about the pad's tube
points, least norm over the two tubes, applied on the ROOT's two tube linings) on the plate lips, and the plate bolts' pull and
shear from them (the one-piece arm's rule). The tubes themselves: a 6/4 pair 22 apart, EI 3.9e8 N mm2 in the thrust plane (5x the
printed arm), 1.2e7 across; the frame model's arm beam is this pair.
"""
import numpy as np, manifold3d as mf
from topo import Part
from parts.cage_node_lib import socket_cuts, collar_shape, FLOOR as C_FLOOR
from parts.overseer_arm_lib import (Arm, cyl, U, along, hull_xy, stadium, rigid_split, frame_cases, mvec, G, Z_BOT, Z_PAD,
                                    BOLT_TOP, BOLT_BOT, BOSS_R, NUT_T, FLOOR, M_MOTOR, Z_MOTOR_CG, THRUST_N, TORQUE_NM, PAD_D,
                                    NODE_R, SOCK_BORE_R, SOCK_DEPTH, SOCK_WALL)

TUBE_BORE_R, TUBE_DEPTH, TUBE_WALL = 3.1, 15.0, 2.5     # 6/4 tube: bore 6.2, 15 deep (coupon), 2.5 collar (2.0 at the grooves)
TUBE_Z = (14.0, 36.0)                                   # the two tubes, stacked on the arm axis
ROOT_S0 = 12.0                                          # root socket floor, mm out along the arm axis from the root centre R
PAD_BACK = 42.0                                         # pad socket mouth, mm short of the motor centre along the axis
SPLIT_S = 45.0                                          # the one-piece arm's features on this side of s = SPLIT_S belong to the root


class TubeArm(Arm):
    def __init__(self, name, theta_deg, motor_idx, spokes, node, piece):
        super().__init__(name, theta_deg, motor_idx, spokes, node)
        assert piece in ("root", "pad")
        self.piece = piece
        self.a3 = np.r_[self.a, 0.0]
        self.pad_mouth = self.len - PAD_BACK                           # s of the pad sockets' mouths (facing the root)
        assert self.pad_mouth > ROOT_S0 + TUBE_DEPTH + 10.0, f"{name}: the arm is too short for two sockets and a tube between"

    def p(self, s, v, z):
        xy = self.R + self.a * s + self.n * v
        return np.array([xy[0], xy[1], z])

    def slab(self, s0, s1, v0, v1, z0, z1):
        corners = [self.p(s, v, 0.0)[:2] for s in (s0, s1) for v in (v0, v1)]
        return hull_xy([(c[0], c[1], 0.0) for c in corners], z0, z1)

    def tube_points(self, piece=None):
        """the two tube socket floor points of this piece (where the tube ends sit)"""
        piece = piece or self.piece
        s = ROOT_S0 if piece == "root" else self.pad_mouth + TUBE_DEPTH
        return [self.p(s, 0.0, z) for z in TUBE_Z]

    def tube_length(self):
        return self.pad_mouth + TUBE_DEPTH - ROOT_S0

    # ------------------------------------------------------------------------------------------------- geometry
    def drawn_cuts_env(self):
        full_drawn, full_cuts, _ = super().drawn_cuts_env()      # sets self.env_pts and the spoke linings
        drawn, cuts, lin = [], [], []
        if self.piece == "root":
            side = self.slab(-400.0, SPLIT_S, -200.0, 200.0, -60.0, 120.0)
            for z in TUBE_Z:
                P = self.p(0.0, 0.0, z)
                drawn.append(collar_shape(P, self.a3, ROOT_S0 - C_FLOOR, ROOT_S0 + TUBE_DEPTH, TUBE_BORE_R, TUBE_WALL))
                cuts += socket_cuts(P, self.a3, ROOT_S0, TUBE_DEPTH, TUBE_BORE_R)
                cuts.append(along(P, self.a3, ROOT_S0 + TUBE_DEPTH, ROOT_S0 + TUBE_DEPTH + 300.0, TUBE_BORE_R + 1.0))   # the tube: nothing grows on it
                lin.append(along(P, self.a3, ROOT_S0 - C_FLOOR - 0.5, ROOT_S0 + TUBE_DEPTH, TUBE_BORE_R + 1.4))
            blk = [self.xy(u, v) for u in (57.5, 72.0) for v in (-22.0, 22.0)]
            mouth = [self.p(ROOT_S0 + TUBE_DEPTH + 2.0, v, 0.0)[:2] for v in (-TUBE_BORE_R - TUBE_WALL - 3.0, TUBE_BORE_R + TUBE_WALL + 3.0)]
            env = hull_xy([(c[0], c[1], 0.0) for c in list(blk) + mouth], Z_BOT, Z_PAD)
        else:
            side = self.slab(SPLIT_S, 400.0, -200.0, 200.0, -60.0, 120.0)
            for z in TUBE_Z:
                B = self.p(self.pad_mouth + TUBE_DEPTH, 0.0, z)      # socket floor (toward the motor); the mouth faces the root
                drawn.append(collar_shape(B, -self.a3, -C_FLOOR, TUBE_DEPTH, TUBE_BORE_R, TUBE_WALL))
                cuts += socket_cuts(B, -self.a3, 0.0, TUBE_DEPTH, TUBE_BORE_R)
                cuts.append(along(B, -self.a3, TUBE_DEPTH, TUBE_DEPTH + 300.0, TUBE_BORE_R + 1.0))
                lin.append(along(B, -self.a3, -C_FLOOR - 0.5, TUBE_DEPTH, TUBE_BORE_R + 1.4))
            mouth = [self.p(self.pad_mouth - 4.0, v, 0.0)[:2] for v in (-TUBE_BORE_R - TUBE_WALL - 3.0, TUBE_BORE_R + TUBE_WALL + 3.0)]
            env = hull_xy([(c[0], c[1], 0.0) for c in mouth] + [(self.M[0], self.M[1], 27.0), (self.node[0], self.node[1], 17.0)], Z_BOT, Z_PAD)
        self.tube_linings = lin
        return U([full_drawn ^ side] + drawn), U([full_cuts] + cuts), env

    def footprint(self):
        """plan extents (long, short, angle) of this piece's envelope: must fit the 180 plate"""
        import trimesh
        _, _, env = self.drawn_cuts_env()
        pts = np.asarray(env.to_mesh().vert_properties)[:, :2]
        h = trimesh.convex.hull_points(pts)
        best = None
        for ang in np.linspace(0, np.pi / 2, 91):
            c, s = np.cos(ang), np.sin(ang)
            r = h @ np.array([[c, -s], [s, c]]); ext = r.max(0) - r.min(0)
            if best is None or ext.prod() < best[0]:
                best = (ext.prod(), max(ext), min(ext), np.degrees(ang))
        return best[1], best[2], best[3]


def motor_cases(arm):
    """flight with thrust and torque, landing: [(name, cls, F at the motor CG, Mz, accel)]"""
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
    c_m = np.array([arm.M[0], arm.M[1], Z_MOTOR_CG])
    hand_a = np.array([0, 0, 2.0]); F_hand = M_MOTOR * G * hand_a
    tubes = U(arm.tube_linings)
    pad_pts = np.array(arm.tube_points("pad")); c_t = pad_pts.mean(0)

    def tube_split(P_, F, Mz):
        """a load F at P_ (moment Mz about z) carried by the two tubes: least-norm forces at the pad's tube points"""
        return rigid_split(pad_pts, F, np.cross(P_ - c_t, F) + mvec(Mz), c_t)

    if arm.piece == "pad":
        p.support("tubes", tubes)
        p.support("tubes+spokes", U([tubes] + [sp["lining"] for sp in arm.spokes]))
        mb = arm.motor_bolts()
        mreg = [cyl(q[0], q[1], Z_PAD - 2.0, Z_PAD + 0.2, 4.0) for q in mb]
        mpts = np.array([[q[0], q[1], Z_PAD] for q in mb])

        def payload(F_m, M_z=0.0):
            fm = rigid_split(mpts, F_m, np.cross(c_m - mpts.mean(0), F_m) + mvec(M_z), mpts.mean(0))
            return [(mreg[i], fm[i]) for i in range(4)]
        for name, cls, F, Mz, acc in motor_cases(arm):
            p.load_multi(name, payload(F, Mz), cls=cls, support="tubes+spokes", accel=acc)
        p.load_multi("HANDLING pad on the tubes: motor 96 g at 2 g, no spokes", payload(F_hand), cls="flight", support="tubes", accel=hand_a)
        for name, cls, a, sp in frame_cases(arm):
            Fm = M_MOTOR * G * a
            p.load_multi(name, payload(Fm) + [(s_["lining"], sp[s_["name"]]) for s_ in arm.spokes], cls=cls, support="tubes", accel=a)
    else:
        lips = []
        for s in (1, -1):
            A, B = arm.xy(BOLT_TOP[0], s * BOLT_TOP[1]), arm.xy(BOLT_BOT[0], s * BOLT_BOT[1])
            lips += [stadium(A, B, Z_PAD - 2.5, Z_PAD, BOSS_R), stadium(A, B, Z_BOT, Z_BOT + 2.5, BOSS_R)]
        p.support("plates", U(lips))
        root_lin = arm.tube_linings
        reactions = []

        def via_tubes(P_, F, Mz=0.0):
            f = tube_split(P_, F, Mz)
            return [(root_lin[i], f[i]) for i in range(2)]
        for name, cls, F, Mz, acc in motor_cases(arm):
            p.load_multi(name + " (via the tubes)", via_tubes(c_m, F, Mz), cls=cls, support="plates", accel=acc)
            reactions.append((c_m, F, Mz))
        p.load_multi("HANDLING arm on the tubes: motor 96 g at 2 g, no spokes (via the tubes)", via_tubes(c_m, F_hand), cls="flight", support="plates", accel=hand_a)
        reactions.append((c_m, F_hand, 0.0))
        for name, cls, a, sp in frame_cases(arm):
            Fm = M_MOTOR * G * a; Fs = sum(sp.values()); Ms = np.cross(arm.node - c_m, Fs)
            p.load_multi(name + " (via the tubes)", via_tubes(c_m, Fm + Fs, Ms), cls=cls, support="plates", accel=a)
            reactions.append((c_m, Fm + Fs, Ms))
        # ---- plate bolts: pull and shear from the reactions (the one-piece arm's rule)
        Bq = arm.bolts()
        bpts = np.array([[b["xy"][0], b["xy"][1], b["z"]] for b in Bq]); cB = bpts.mean(0)
        worst = [dict(pull=0.0, shear=np.zeros(3)) for _ in Bq]
        for P_, F, Mz in reactions:
            f = rigid_split(bpts, F, np.cross(P_ - cB, F) + mvec(Mz), cB)
            for i, b in enumerate(Bq):
                tens = f[i][2] * (1.0 if b["nut"] == "bot" else -1.0)
                worst[i]["pull"] = max(worst[i]["pull"], tens)
                sh = np.array([f[i][0], f[i][1], 0.0])
                if np.linalg.norm(sh) > np.linalg.norm(worst[i]["shear"]):
                    worst[i]["shear"] = sh
        for i, b in enumerate(Bq):
            x, y = b["xy"]
            if b["nut"] == "bot":
                seat = cyl(x, y, Z_BOT + NUT_T, Z_BOT + NUT_T + 2.5, 3.3); dz = +1.0
            else:
                seat = cyl(x, y, Z_PAD - NUT_T - 2.5, Z_PAD - NUT_T, 3.3); dz = -1.0
            wall = cyl(x, y, Z_BOT + NUT_T + FLOOR, Z_PAD - NUT_T - FLOOR, 2.7)
            pl, sh = max(worst[i]["pull"], 1.0), worst[i]["shear"]
            p.load_multi(f"{b['name']} pull {pl:.0f} N", [(seat, np.array([0, 0, dz * pl]))], cls="crash", support="plates")
            p.load_multi(f"{b['name']} shear {np.linalg.norm(sh):.0f} N", [(wall, sh)], cls="crash", support="plates")
            log(f"  {b['name']}: pull {pl:.0f} N, shear {np.linalg.norm(sh):.0f} N (tube reactions, least norm)")
    fp = arm.footprint()
    log(f"  {arm.piece} footprint {fp[0]:.0f} x {fp[1]:.0f} mm at {fp[2]:.0f} deg; tubes {arm.tube_length():.0f} long, z {TUBE_Z}")
    return p
