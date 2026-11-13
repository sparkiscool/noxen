"""Swap arm v8: the four engine setups (hub, carriage, lift, boom). Geometry from parts/box_arm8_lib.py and parts/box_arm8/make.py,
loads from box_arm8_lib.cases (each case its own, the plug's force at the plug). Arm frame: x across, y along the axle, z radial.
Classes: a swap's own loads are "flight" (they repeat every swap), the float limit, an off-square insert and 3 g handling "misuse"."""
import importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import box_arm8_lib as L
from box_arm8_lib import box, cyl, U
from topo.part import Part
_sp = importlib.util.spec_from_file_location("box_arm8_make", os.path.join(HERE, "box_arm8", "make.py")); M = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(M)

PACKS = ("overseer", "scout")


def _cases():
    out = []
    for pk in PACKS:
        for c in L.cases(pk):
            if pk == "scout" and ("handling" in c[0] or "carrying" in c[0]): continue        # the Overseer's envelope them
            out.append(c)
    return out


def _pad(p, r=3.5, t=3.0, ax="z"):
    """a small box round a point: a load or support patch"""
    return box(p[0] - r, p[0] + r, p[1] - r, p[1] + r, p[2] - t, p[2] + t)


# ------------------------------------------------------------------ lift: bearing housings and nut seat to the float bolts' bosses
def lift(h):
    env = L.lift_env()
    pad, nut = L.mount_pad(), L.nut_seat()
    y0, y1 = L.SOCKET[0] - 2.0, L.SOCKET[1]; ym = (y0 + y1) / 2
    halves = [(cyl("y", bx, bu, L.COLLAR_R, ya, yb), np.array([bx, (ya + yb) / 2, bu])) for bx, bu in L.boom_axes() for ya, yb in ((y0, ym), (ym, y1))]
    drawn = U(pad, nut, *L.sockets())
    p = Part("box_arm8_lift", h, drawn, L.lift_cuts(), env, build="+z", material="PCTG")                      # upright, the plate's lower end down: the yoke's twist into the plate is in-layer (filled solid +z 0.28, +x 0.31, +y 0.69)
    p.support("default", U(pad, nut))                                                                         # bolted on the MGN block's face; the nut holds the bar direction
    # the grab unit's load reaches the lift through the three boom rods: each rod bears on its socket near the floor and near the mouth (a force each: the rod's shear and its bending)
    for nm, cls, F, P, s in _cases():
        f = L.distribute(F, P, [q for _, q in halves])
        p.load_multi(nm, [(c, fi) for (c, _), fi in zip(halves, f)], cls=cls)
    return p


# ------------------------------------------------------------------ carriage: wheels to the MGN rail's seat
def carriage(h):
    env = L.carriage_env(); bodies = L.carriage_bodies(); drawn = U(*bodies)
    p = Part("box_arm8_carriage", h, drawn, L.carriage_cuts(), env, build="-y", material="PCTG")
    p.support("default", U(*bodies[:3]))
    ys = L.SPINE["y"][1]; bolts = L.mgn_bolts()
    for nm, cls, F, P, s in _cases():
        R, M, C, _ = L.mgn_wrench(F, P)
        uc = C[2] + min(s, L.T2)                                               # where the block is on the rail (carriage coordinates = the arm's at park)
        near = bolts                                                           # the steel rail (12 x 8) is stiff beside the print: it passes the block's force and moment to the seat at all four bolts
        pts = [np.array([L.RAIL_X + sx * 4.0, ys - 1.5, u]) for u in near for sx in (-1, 1)]
        A = np.zeros((6, 3 * len(pts))); rhs = np.concatenate([R, np.cross([C[0], C[1], uc], R) + M])
        for i, q in enumerate(pts):
            for j in range(3):
                e = np.zeros(3); e[j] = 1.0; A[:3, 3 * i + j] = e; A[3:, 3 * i + j] = np.cross(q, e)
        xs = np.linalg.lstsq(A, rhs, rcond=None)[0]
        seat = lambda q: box(min(q[0], L.RAIL_X), max(q[0], L.RAIL_X) + (2.0 if q[0] > L.RAIL_X else 0.0) - (2.0 if q[0] < L.RAIL_X else 0.0), ys - 3.0, ys, q[2] - 12.0, q[2] + 12.0)
        p.load_multi(nm, [(seat(q), xs[3 * i:3 * i + 3]) for i, q in enumerate(pts)], cls=cls)
    return p


# ------------------------------------------------------------------ hub: bearing seats to the rail pads and the stepper walls
def hub(h):
    y0, y1, R = L.PLATE
    bodies = M.hub_drawn(); drawn = U(*bodies)
    w0, t0 = L.MOTOR_WALL, L.THRUST_WALL
    room = [box(w0["x"][0], w0["x"][1], w0["y"][0], 78.0, w0["u"][0] - 0.01, w0["u"][1] + 12.0),             # over each stepper wall, either side of the plate
            box(67.8, 79.0, y1 - 0.1, t0["y"][1], w0["u"][1], 4.0)]                                           # and a rib from it up past the thrust wall's outer end (the screw's lane is inside it)
    keel = [box(-20.0, 20.0, 27.0, y1, 40.0, R - 0.5)]                                                       # under each rail: the full depth behind the plate (the rail's couple goes down it to the sleeve)
    import manifold3d as mf
    flare = mf.Manifold.cylinder(8.0, 50.2, 58.0, 96).rotate([-90, 0, 0]).translate([0, 26.2, 0])           # the sleeve's wall grows from the belt's lane to its full 58 over 8 mm: no step at the lane's edge
    env = U(cyl("y", 0, 0, 50.2, 8.0, 26.3, 96), flare, cyl("y", 0, 0, 58.0, 34.1, y1, 96), cyl("y", 0, 0, R, y1 - 8.0, y1, 96) - cyl("y", 0, 0, 58.0, 0, 100, 96), *keel, *[L.rot180(g_) for g_ in keel],
            *room, *[L.rot180(g_) for g_ in room])                                                             # an 8 mm plate, the sleeve's wall, the two keels: the room is the load's way, not the whole disc
    p = Part("box_arm8_hub", h, drawn, U(*M.hub_cuts()), env, build="+y", material="PCTG")
    p.support("default", cyl("y", 0, 0, 42.5, 8.0, 40.0, 96) - cyl("y", 0, 0, 39.5, 7, 41, 96))
    x, y = L.SCREW; w = L.THRUST_WALL                         # the screw's force enters the hub at its thrust wall
    for hd, sg in (("A", 1.0), ("B", -1.0)):
        wall = box(w["x"][0], w["x"][1], w["y"][0], w["y"][1], w["u"][0], w["u"][1]); wall = wall if sg > 0 else L.rot180(wall)
        pads = [np.array([sg * px, y1 - 2.0, sg * u]) for u in L.RAIL_BOLTS for px in (-5.0, 5.0)]
        for nm, cls, F, P, s in _cases():
            if hd == "B" and not ("insert at the drone, overseer" in nm or "float limit, overseer" in nm): continue
            F = np.asarray(F, float); P = np.asarray(P, float) + [0, 0, s]
            Fu = np.array([0, 0, F[2]]); rest = np.array([F[0], F[1], 0.0])
            # the screw takes the force along the bar at its axis; the rail's two bolt pads take the rest of the wrench
            mom = np.cross(P, F) - np.cross([x, y, w["u"][0]], Fu)
            A = np.zeros((6, 12)); rhs = np.concatenate([rest, mom]); loc = [q * [sg, 1, sg] for q in pads]
            for i, q in enumerate(loc):
                for j in range(3):
                    e = np.zeros(3); e[j] = 1.0; A[:3, 3 * i + j] = e; A[3:, 3 * i + j] = np.cross(q, e)
            xs = np.linalg.lstsq(A, rhs, rcond=None)[0]
            # each pad is the quarter of the rail's seat round its bolt (9.5 x 18): the rail bears on the seat and pulls on the bolt's insert under it
            seat = lambda q: box(min(0.0, q[0] * 1.9), max(0.0, q[0] * 1.9), y1 - 3.0, y1, q[2] - 9.0 * sg, q[2] + 9.0 * sg) if sg > 0 else box(min(0.0, q[0] * 1.9), max(0.0, q[0] * 1.9), y1 - 3.0, y1, q[2] - 9.0, q[2] + 9.0)
            parts = [(wall, Fu * [sg, 1, sg])] + [(seat(pads[i]), xs[3 * i:3 * i + 3] * [sg, 1, sg]) for i in range(4)]
            p.load_multi(f"head {hd}: {nm}", parts, cls=cls)
    ring = cyl("y", 0, 0, M.RING_R + 1.5, 9.5, 19.5, 96) - cyl("y", 0, 0, 45.0, 9, 20, 96)
    p.load("axle belt at the stepper's stall: 2.95 N m at the ring, 62 N across the sleeve", ring, [62.0, 0, 0], cls="misuse")
    for nm, g in (("handling 3 g along the axle", [0, -3, 0]), ("handling 3 g across", [3, 0, 0])):
        w = L.MOTOR_WALL; mw = [box(w["x"][0], w["x"][1], w["y"][0], w["y"][1], w["u"][0], w["u"][1])]; mw.append(L.rot180(mw[0]))
        p.load_multi(nm, [(m_, np.array(g, float) * 9.81 * 0.40) for m_ in mw], cls="misuse", accel=np.array(g, float))
    return p
