"""Overseer cage node 21: the rear camera node (Spark 28 Oct). A cage node like the others (parts/cage_node_lib.py: three socket collars,
drawn arms to the centre, the rest generated) sunk 45 mm inside the rear facet 13-14-5 on three 3 mm rods to nodes 13, 14 and 20 (29 Oct, Spark: a rod must not be in the way of
mounting the camera: the rod to node 5 ran straight down 1 mm behind the board's lower edge, through the plug zone; the rod to node 20
leaves forward and up, 99 long, clear of the board, its plugs and the view), so a
tail-first hit lands on the facet tubes and the lens, 26 mm inside the plane, never touches. It carries the DECXIN 2633V2 looking
back and 40 deg down: 4 standoff cones 10 long behind the board (M2.5 on 28 x 28, insert O3.2 x 5.7, E; O7.6 at the board to O11; 29 Oct: the plugs on the board's back
need the standoff, the plug zone behind the board is a cut; was O7.6 x 6, before that O10 on a drawn frame),
as light as possible (Spark), the camera body and its view are cuts. Camera load: 15 g (E) at its CG split equally over the 4 bosses, flight 4 g, landing
6 g, crash 35 g on 6 axes (the moment from the 3 mm CG offset is neglected). The rods' own loads come from the cage file (tools/cage_loads.py, 29 Oct: 13 N; they were 100 N estimates)."""
import numpy as np
import trimesh
from topo.part import man
from parts.cage_node_lib import make, along, GN, cone, U

PITCH = 40.0                                                     # camera looks back (-x) and 40 deg down
F = np.array([-np.cos(np.radians(PITCH)), 0.0, -np.sin(np.radians(PITCH))])
UP = np.array([-np.sin(np.radians(PITCH)), 0.0, np.cos(np.radians(PITCH))])
RT = np.cross(F, UP)
BASE_AHEAD = 14.0                                                # board back plane 14 ahead of the node centre (29 Oct, was 8: the 10 mm standoffs
#   and the plug zone behind the board must clear the node's core and its socket floors; the lens moves 6 further out, 20 inside the facet)
CAM_BOARD_BACK, CAM_SQ = 12.0, 14.0                              # lens 12 ahead of the board back (E); M2 on 28 x 28 (Spark 29 Oct, coupon)
BOSS_L, BOSS_R, BOSS_BASE_R = 10.0, 2.8, 2.8                     # standoff post 10 long, O5.6 straight: 1.2 wall round the O3.2 x 5.7 M2 insert (Spark 29 Oct: coupon A fits,
                                                                 # the O7.6 -> O11 cones did not clear the board; 10 long for the plug and its wire behind the board)
PLUG_BACK = 8.0                                                  #   (29 Oct, Spark: plugs on the board's back near the holes); keep-out behind the board
HA, HV = 44.5, 36.5                                              # view half angles (dimensions.md Camera)
CAM_MASS = 25.0                                                  # g (E): board 15 + a heatsink on its back 10 (Spark 29 Oct)


def hull(pts):
    return man(trimesh.convex.convex_hull(np.asarray(pts, float)))


def fbox(o, e1, e2, e3, r1, r2, r3):
    return hull([o + a * e1 + b * e2 + c * e3 for a in r1 for b in r2 for c in r3])


POSTS = []                                                       # the 4 standoff posts: the gate allows their 1.2 mm walls (thin_ok)


def camera(c, ms):
    base = c + F * BASE_AHEAD
    lens = base + F * CAM_BOARD_BACK
    pts = [base + sr * CAM_SQ * RT + su * CAM_SQ * UP for sr in (1, -1) for su in (1, -1)]
    drawn, cuts = [], []
    cones = []
    for q in pts:
        cones.append(cone(q, -F, 0.0, BOSS_L, BOSS_R, BOSS_BASE_R))                # standoff cone behind the board
        cuts.append(along(q, F, -5.7, 1.0, 1.6))                                 # insert hole O3.2 x 5.7 from the board face
    drawn += cones; POSTS[:] = cones                                                 # declared thin_ok to the gate (1.2 wall)
    cuts.append(fbox(base, F, RT, UP, (0.0, 25.0), (-22.0, 22.0), (-22.0, 22.0)))  # board + body (38 x 38 + clearance), nothing grows there
    cuts.append(fbox(base, F, RT, UP, (-PLUG_BACK, 0.0), (-19.0, 19.0), (-19.0, 19.0)) - U(cones))   # the plugs behind the board
    # (29 Oct: the drawn 38 x 38 frame joining the bosses is gone, 2 g; the topology joins them, as everywhere else)
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


_setup = make(21, extra=camera)


def setup(h):
    p = _setup(h)
    p.thin_ok = list(POSTS)                                      # the O5.6 / 1.2 wall posts pass the gate as thin_allowed (Spark 29 Oct by coupon)
    return p


SPEC = dict(h=1.0, gen_cm3=2.0, max_worst=0.5, settings=dict(near_sym=True))   # near_sym: mirrored but for 0.4 % (29 Oct)      # as light as it goes (Spark 28 Oct); 2.5 gave 8.7 g at index 0.09 on 100 N rod estimates, the frame
#   model (29 Oct) puts 13 N on the rods: O7.6 bosses and no drawn frame (Spark 29 Oct: "not very optimised"); 1.2 gave 4.1 g with two bosses
#   standing proud of the body (their cores off at the gate) and a 3 mm neck: 2.0 so the body envelops the bosses
