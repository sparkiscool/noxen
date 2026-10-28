"""Overseer cage node 21: the rear camera node (Spark 28 Oct). A cage node like the others (parts/cage_node_lib.py: three socket collars,
drawn arms to the centre, the rest generated) sunk 45 mm inside the rear facet 13-14-5 on three 3 mm rods to nodes 13, 14 and 5, so a
tail-first hit lands on the facet tubes and the lens, 26 mm inside the plane, never touches. It carries the DECXIN 2633V2 looking
back and 40 deg down: 4 insert bosses behind the board (M2.5 on 28 x 28, insert O3.2 x 5.7, E) on a 38 x 38 frame (6 wide, 2 thick) as the seat the
node body grows on, as light as possible (Spark), the camera body and its view are cuts. Camera load: 15 g (E) at its CG split equally over the 4 bosses, flight 4 g, landing
6 g, crash 35 g on 6 axes (the moment from the 3 mm CG offset is neglected). The rods' own loads are the cage file's 100 N estimates
(E, no frame model for them yet)."""
import numpy as np
import trimesh
from topo.part import man
from parts.cage_node_lib import make, along, GN

PITCH = 40.0                                                     # camera looks back (-x) and 40 deg down
F = np.array([-np.cos(np.radians(PITCH)), 0.0, -np.sin(np.radians(PITCH))])
UP = np.array([-np.sin(np.radians(PITCH)), 0.0, np.cos(np.radians(PITCH))])
RT = np.cross(F, UP)
BASE_AHEAD = 8.0                                                 # board back plane 8 ahead of the node centre (clear of the collars)
CAM_BOARD_BACK, CAM_SQ = 12.0, 14.0                              # lens 12 ahead of the board back (E); M2.5 on 28 x 28 (P)
BOSS_L, BOSS_R = 6.0, 5.0                                        # boss O10 x 6 round the O3.2 x 5.7 insert: the lightest that holds it
HA, HV = 44.5, 36.5                                              # view half angles (dimensions.md Camera)
CAM_MASS = 15.0                                                  # g (E)


def hull(pts):
    return man(trimesh.convex.convex_hull(np.asarray(pts, float)))


def fbox(o, e1, e2, e3, r1, r2, r3):
    return hull([o + a * e1 + b * e2 + c * e3 for a in r1 for b in r2 for c in r3])


def camera(c, ms):
    base = c + F * BASE_AHEAD
    lens = base + F * CAM_BOARD_BACK
    pts = [base + sr * CAM_SQ * RT + su * CAM_SQ * UP for sr in (1, -1) for su in (1, -1)]
    drawn, cuts = [], []
    for q in pts:
        drawn.append(along(q, -F, 0.0, BOSS_L, BOSS_R))                          # boss O12 x 8 behind the board
        cuts.append(along(q, F, -5.7, 1.0, 1.6))                                 # insert hole O3.2 x 5.7 from the board face
    cuts.append(fbox(base, F, RT, UP, (0.0, 25.0), (-22.0, 22.0), (-22.0, 22.0)))  # board + body (38 x 38 + clearance), nothing grows there
    ring = fbox(base, F, RT, UP, (-BOSS_L - 2.0, -BOSS_L + 0.5), (-19.0, 19.0), (-19.0, 19.0)) - fbox(base, F, RT, UP, (-BOSS_L - 3.0, -BOSS_L + 1.5), (-13.0, 13.0), (-13.0, 13.0))
    drawn.append(ring)                                                            # 38 x 38 frame, 6 wide x 2 thick, joining the 4 bosses: the seat the node body grows on (2 g)
    far = lens + F * 400.0
    w, hh = 400.0 * np.tan(np.radians(HA + 2.0)), 400.0 * np.tan(np.radians(HV + 2.0))
    cuts.append(hull([lens - F * 0.5] + [far + sr * w * RT + su * hh * UP for sr in (1, -1) for su in (1, -1)]))   # the view
    env = [fbox(base, F, RT, UP, (-BOSS_L - 8.0, 1.0), (-24.0, 24.0), (-24.0, 24.0))]                 # room behind the board
    faces = [along(q, F, -3.0, 0.3, BOSS_R) for q in pts]

    def loads(p):
        cases = [("camera flight 4 g", np.array([0, 0, 4.0]), "flight"), ("camera landing 6 g", np.array([0, 0, 6.0]), "flight")]
        cases += [(f"camera crash 35 g {d}", 35.0 * np.array(v, float), "crash") for d, v in
                  (("+x", (1, 0, 0)), ("-x", (-1, 0, 0)), ("+y", (0, 1, 0)), ("-y", (0, -1, 0)), ("+z", (0, 0, 1)), ("-z", (0, 0, -1)))]
        for name, a, cls in cases:
            Fv = -CAM_MASS * GN * a
            p.load_multi(name, [(fc, Fv / 4.0) for fc in faces], cls=cls)
    return dict(drawn=drawn, cuts=cuts, env=env, loads=loads)


setup = make(21, extra=camera)
SPEC = dict(h=1.0, gen_cm3=2.5, max_worst=0.5)      # as light as it goes (Spark 28 Oct): three 100 N rods and a 15 g camera
