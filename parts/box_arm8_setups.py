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
    env = L.lift_env(); b = L.BLOCK; x, y = L.SCREW
    housings = [box(xx - 8.0, xx + 8.0, b["y"][0], b["y"][1], a, e) for xx, a, e in L.lm_seats()]
    nut = L._b(L.LUG)
    bosses = [box(px - 9.0, px + 9.0, py - 7.0, py + 7.0, L.TIE_U[0], L.TIE_U[1]) for px, py in L.float_pts()]
    drawn = U(*housings, nut, *bosses)
    p = Part("box_arm8_lift", h, drawn, L.lift_cuts(), env, build="+z", material="PCTG")
    p.support("default", U(*housings, nut))
    pts = [np.array([px, py, sum(L.TIE_U) / 2]) for px, py in L.float_pts()]
    for nm, cls, F, P, s in _cases():
        f = L.distribute(F, P, pts)
        p.load_multi(nm, [(c, fi) for c, fi in zip(bosses, f)], cls=cls)
    return p


# ------------------------------------------------------------------ carriage: wheels to the rod clamps
def carriage(h):
    env = L.carriage_env(); bodies = L.carriage_bodies(); drawn = U(*bodies)
    p = Part("box_arm8_carriage", h, drawn, L.carriage_cuts(), env, build="-y", material="PCTG")
    p.support("default", U(*bodies[:3]))
    top, bot = bodies[3], bodies[4]
    for nm, cls, F, P, s in _cases():
        if s < L.S_BAY: continue                                              # carried loads at park go through the same clamps, smaller
        seats, _ = L.lift_reactions(F, P, s)                                   # the block at the top of the rods: its four bearings load the rods beside the top clamp
        d = L.T2                                                               # lift's height on the carriage in these cases
        tot = sum(f for _, f in seats); mom = sum(np.cross(q + [0, 0, d], f) for q, f in seats)
        pts = [np.array([x, L.ROD_Y, L.Z0 + L.CAR_L - 3.0]) for x in L.RODS_X] + [np.array([x, L.ROD_Y, L.Z0 + L.C_BOT + 2.0]) for x in L.RODS_X]
        n = len(pts); A = np.zeros((6, 2 * n)); rhs = np.concatenate([tot, mom])
        for i, q in enumerate(pts):
            for j, e in enumerate((np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))):
                A[:3, 2 * i + j] = e; A[3:, 2 * i + j] = np.cross(q, e)
        xs = np.linalg.lstsq(A, rhs, rcond=None)[0]
        p.load_multi(nm, [(bodies[3 + i], [xs[2 * i], xs[2 * i + 1], 0.0]) for i in range(n)], cls=cls)        # on the whole collar, as the rod bears in it
    return p


# ------------------------------------------------------------------ hub: bearing seats to the rail pads and the stepper walls
def hub(h):
    y0, y1, R = L.PLATE
    bodies = M.hub_drawn(); drawn = U(*bodies)
    w0, t0 = L.MOTOR_WALL, L.THRUST_WALL
    room = [box(w0["x"][0], w0["x"][1], 26.5, 69.0, w0["u"][0] - 0.01, w0["u"][1] + 12.0),                    # round each stepper wall, down to the body under the plate
            box(t0["x"][0] - 8.0, t0["x"][1] + 8.0, 28.0, 55.6, t0["u"][0] - 8.0, t0["u"][1] + 6.0)]           # and round each thrust wall, under the lift's block
    env = U(cyl("y", 0, 0, R, 8.0, y1, 96), *room, *[L.rot180(g_) for g_ in room])
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
            parts = [(wall, Fu * [sg, 1, sg])] + [(_pad(pads[i], 3.0, 3.0), xs[3 * i:3 * i + 3] * [sg, 1, sg]) for i in range(4)]
            p.load_multi(f"head {hd}: {nm}", parts, cls=cls)
    ring = cyl("y", 0, 0, M.RING_R + 1.5, 9.5, 19.5, 96) - cyl("y", 0, 0, 45.0, 9, 20, 96)
    p.load("axle belt at the stepper's stall: 2.95 N m at the ring, 62 N across the sleeve", ring, [62.0, 0, 0], cls="misuse")
    for nm, g in (("handling 3 g along the axle", [0, -3, 0]), ("handling 3 g across", [3, 0, 0])):
        w = L.MOTOR_WALL; mw = [box(w["x"][0], w["x"][1], w["y"][0], w["y"][1], w["u"][0], w["u"][1])]; mw.append(L.rot180(mw[0]))
        p.load_multi(nm, [(m_, np.array(g, float) * 9.81 * 0.40) for m_ in mw], cls="misuse", accel=np.array(g, float))
    return p
