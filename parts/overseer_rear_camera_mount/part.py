"""Overseer rear camera mount (v1 setup, 28 Oct): a third DECXIN 2633V2 at the tail, looking back and down at the floor behind the
drone. One printed PETG piece: a split clamp round the 6 mm tube 13-14 (the top edge of the rear downward facet 13-14-5) with the
camera hanging below it inside the facet plane, 4 camera insert bosses behind the board, and the web between them generated.

Drawn = attachments only: the clamp sleeve and its two bolt ears, the 4 camera insert bosses. Cuts = tube bore, the 1.2 slit that
makes the clamp split, 2 x (M3 hole + head seat sweep + nut pocket sweep), camera body, camera frustum (nothing may grow into the
view). Frame: x fwd, y left, z up, mm, z 0 = bottom plate top; geometry from systems/overseer.cage.json (C)."""
import json, os
import numpy as np, trimesh, manifold3d as mf
from topo import Part
from topo.part import man

HERE = os.path.dirname(os.path.abspath(__file__))
CAGE = json.load(open(os.path.join(HERE, "..", "..", "systems", "overseer.cage.json")))
SPEC = dict(h=1.0, gen_cm3=9.0, max_worst=0.5)

G = 9.81e-3                                                    # N per g per g of mass
NODES = np.array(CAGE["nodes"], float)
N5, N13, N14 = NODES[5], NODES[13], NODES[14]
T = 0.5 * (N13 + N14); T[1] = 0.0                              # tube 13-14 at y 0 (the top edge of the rear facet)

# ---- camera pose (C): faces back (-x), pitched PITCH down; lens RECESS mm inside the facet plane (13, 14, 5) -----------------
PITCH, LENS_Z, RECESS = 40.0, 54.3, 11.0
p_ = np.radians(PITCH)
F = np.array([-np.cos(p_), 0.0, -np.sin(p_)])                  # forward
UP = np.array([-np.sin(p_), 0.0, np.cos(p_)])                  # up in the image
RT = np.cross(UP, F)                                           # same convention as the two head cameras (u x f)
NRM = np.cross(N14 - N13, N5 - N13); NRM /= np.linalg.norm(NRM)
if NRM[0] > 0: NRM = -NRM                                      # outward = back and down
_s = (N13[2] - LENS_Z) / (N13[2] - N5[2])
LENS = np.array([N13[0] + _s * (N5[0] - N13[0]), 0.0, LENS_Z]) - RECESS * NRM
HA, HV = 44.5, 36.5                                            # landscape 89 x 73 (P)

# ---- numbers and tags -------------------------------------------------------------------------------------------------
CAM_SQ = 14.0                       # DECXIN 2633V2: M2.5 on 28 x 28 (P); heat-set insert hole O3.2 x 5.7 (E); boss O12 (head mount v5 row)
CAM_BOARD_BACK = 12.0               # board back face 12 behind the lens plane (E placeholder, as the head mount); body 38 x 38 (P)
CAM_BOSS_L, BOSS_R = 8.0, 6.0
CAM_MASS = 15.0                     # E
TUBE_R = 3.0                        # 6 mm carbon tube 13-14 (cage rod_r_mm 3.0); bore O6.2
BORE_R = 3.1
WALL = 2.0
SLEEVE_R = BORE_R + WALL + 0.4
CLAMP_HL = 13.5                     # clamp half length along the tube
BOLT_Y, BOLT_Q = 7.2, 9.5           # M3 through-bolts across the slit: 2 at y +-7.2, 9.5 from the tube axis away from the camera
EAR_Q, EAR_W = 14.0, 6.0
SLIT = 1.2                          # the real slit is ~1.0 (E); 1.2 so the h 1.0 grid resolves it
PRELOAD = 60.0                      # N per bolt, hand-tight M3 on PETG (E)
SHEAR = 35.0 * CAM_MASS * G         # a bolt shear case: the whole camera crash load (35 g) on one bolt (conservative)

DIRS14 = [np.array(v, float) / np.linalg.norm(v) for v in
          [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
FLIGHT = {"flight 4 g": np.array([0, 0, 4.0]), "flight 2 g x": np.array([2.0, 0, 1]), "flight 2 g y": np.array([0, 2.0, 1]),
          "landing 6 g": np.array([0, 0, 6.0])}
EY = np.array([0.0, 1.0, 0.0])      # clamp frame: y along the tube, q = UP (ears, away from the camera), w = F (bolt axes)


# ---- helpers (copied from the head mount) ------------------------------------------------------------------------------
def U(ms):
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)


def along(q0, q1, r, n=32):
    return man(trimesh.creation.cylinder(r, float(np.linalg.norm(np.subtract(q1, q0))), sections=n, segment=np.vstack([q0, q1])))


def hull(pts):
    return man(trimesh.convex.convex_hull(np.asarray(pts, float)))


def fbox(o, e1, e2, e3, r1, r2, r3):
    return hull([o + a * e1 + b * e2 + c * e3 for a in r1 for b in r2 for c in r3])


def cp(y, q, w):
    return T + y * EY + q * UP + w * F


def cam_boss_pts():
    base = LENS - F * CAM_BOARD_BACK
    return [base + sr * CAM_SQ * RT + su * CAM_SQ * UP for sr in (1, -1) for su in (1, -1)], base


def frustum(margin_deg=2.0, length=400.0):
    ha, hv = np.radians(HA + margin_deg), np.radians(HV + margin_deg)
    far = LENS + F * length
    w, hh = length * np.tan(ha), length * np.tan(hv)
    return hull([LENS - F * 0.5] + [far + sr * w * RT + su * hh * UP for sr in (1, -1) for su in (1, -1)])


def bodies():
    drawn, cuts = [], []
    # --- camera insert bosses (O12 x 8 behind the board back plane), insert hole O3.2 x 5.7 from the front; body and frustum cut
    pts, base = cam_boss_pts()
    for q in pts:
        drawn.append(along(q, q - F * CAM_BOSS_L, BOSS_R))
        cuts.append(along(q + F * 1.0, q - F * 5.7, 1.6))
    cuts.append(fbox(base, F, RT, UP, (0.0, 25.0), (-22.0, 22.0), (-22.0, 22.0)))
    cuts.append(frustum())
    env_cam = fbox(base, F, RT, UP, (-32.0, 0.0), (-22.0, 22.0), (-22.0, 22.0))
    # --- clamp: sleeve round the tube, ear block on the side away from the camera, slit through the ear and the ring wall on that side
    drawn.append(along(cp(-CLAMP_HL, 0, 0), cp(CLAMP_HL, 0, 0), SLEEVE_R))
    drawn.append(fbox(T, EY, UP, F, (-CLAMP_HL, CLAMP_HL), (0.0, EAR_Q), (-EAR_W, EAR_W)))
    cuts.append(along(cp(-40, 0, 0), cp(40, 0, 0), BORE_R))
    cuts.append(fbox(T, EY, UP, F, (-40.0, 40.0), (0.0, 30.0), (-SLIT / 2, SLIT / 2)))
    for sy in (BOLT_Y, -BOLT_Y):
        cuts.append(along(cp(sy, BOLT_Q, -15), cp(sy, BOLT_Q, 15), 1.7))               # M3 clearance through the whole ear
        cuts.append(along(cp(sy, BOLT_Q, 3.0), cp(sy, BOLT_Q, 60), 3.2))                # head seat + driver sweep, out through the envelope
        cuts.append(along(cp(sy, BOLT_Q, -3.4), cp(sy, BOLT_Q, -60), 3.4, 6))           # nut pocket (hex, 5.5 AF + 0.3) + slide-in sweep
    env_cl = fbox(T, EY, UP, F, (-CLAMP_HL, CLAMP_HL), (-9.0, EAR_Q), (-EAR_W - 1, EAR_W + 1))
    env_cyl = along(cp(-CLAMP_HL, 0, 0), cp(CLAMP_HL, 0, 0), 9.0)
    allp = np.vstack([np.asarray(m.to_mesh().vert_properties)[:, :3] for m in (env_cam, env_cl, env_cyl)])
    env = U([env_cam, env_cl, env_cyl, hull(allp)])
    return U(drawn), U(cuts), env, dict(pts=pts)


def setup(h):
    drawn, cuts, env, X = bodies()
    p = Part("overseer_rear_camera_mount", h, drawn, cuts, env, build="+z", material="PETG")
    grip = along(cp(-CLAMP_HL, 0, 0), cp(CLAMP_HL, 0, 0), BORE_R + 1.0)               # the clamp's bore face on the tube (the support)
    p.support("clamp bore", grip)

    pts = X["pts"]
    cam_reg = [along(q - F * 3.0, q + F * 0.3, BOSS_R) for q in pts]
    cam_cg = LENS - F * 8.9                                                            # assembly CG 8.9 behind the lens (E)

    def rigid_split(P4, Fv, Pt):
        P4 = np.asarray(P4, float); c0 = P4.mean(0); n = len(P4); Am = np.zeros((6, 3 * n))
        for i, q in enumerate(P4 - c0):
            Am[:3, 3 * i:3 * i + 3] = np.eye(3)
            Am[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
        Mc = np.cross(np.asarray(Pt, float) - c0, Fv)
        return (np.linalg.pinv(Am) @ np.r_[Fv, Mc]).reshape(n, 3)

    def payload(name, a, cls):
        f = rigid_split(pts, -CAM_MASS * G * a, cam_cg)
        p.load_multi(name, [(cam_reg[i], f[i]) for i in range(4)], cls=cls, accel=-a, support="clamp bore")

    # cases: camera 15 g (E) at its CG: crash 35 g in 14 directions, flight 4 g / 2 g, landing 6 g
    for k, d in enumerate(DIRS14):
        payload(f"crash {k} (35 g)", 35.0 * d, "crash")
    for nm, a in FLIGHT.items():
        payload(nm, a, "flight")

    # clamp bolts, each: a pull case (hand-tight preload as two opposite pushes on the head seat and the nut pocket floor)
    # and shear cases (camera crash load on that bolt's hole wall, along the tube and across it)
    for i, sy in enumerate((BOLT_Y, -BOLT_Y), 1):
        head = along(cp(sy, BOLT_Q, 2.7), cp(sy, BOLT_Q, 3.3), 4.6)
        nut = along(cp(sy, BOLT_Q, -3.7), cp(sy, BOLT_Q, -3.1), 4.6)
        p.load_multi(f"bolt {i} pull {PRELOAD:.0f} N", [(head, -PRELOAD * F), (nut, PRELOAD * F)], cls="misuse", support="clamp bore")
        wall = along(cp(sy, BOLT_Q, -2.5), cp(sy, BOLT_Q, 2.5), 2.6)
        for lab, v in (("+y", EY), ("-y", -EY), ("+q", UP), ("-q", -UP)):
            p.load_multi(f"bolt {i} shear {lab} {SHEAR:.1f} N", [(wall, SHEAR * v)], cls="crash", support="clamp bore")
    return p
