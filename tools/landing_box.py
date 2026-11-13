"""The box page's timeline: both drones landing into the funnel (Spark 3 Nov: animate how both drones will land on this).
The funnel's end walls slide in 39.2 for the Scout and out for the Overseer, and the Scout's rain dome folds shut over it while it sits and open again (9 Nov). Kinematic, not simulated: each drone comes down off-centre and yawed, meets a 55 deg face with its receiver, and the offset then
follows the face to the seat (offset = height above the seat / tan 55, the geometry of a block sliding down the face); yaw
straightens in the last 40 mm. Scout first (0..10 s), then the Overseer (10..24 s). Groups "landing: Scout" and "landing: Overseer"
in systems/box.assembly.json move as rigid bodies; everything else stays. Transforms are in the box W frame: the drones are posed
seated (SEAT_TO_W from the catch), so the motion is a seat-frame offset conjugated by SEAT_TO_W.
"""
import math
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seat_check as _sc
SEAT_TO_W = _sc.SEAT_TO_W   # the seat's place in the box: tools/seat_check.py
W_TO_SEAT = np.linalg.inv(SEAT_TO_W)
TAN = math.tan(math.radians(55.0))
PARK_Z, PARK_X = 560.0, -420.0          # where a drone waits when it is not the one landing (above and behind the box, out of the front view)

# per drone: (t_start, t_seat, t_leave, t_end, start height, offset x, offset y, yaw deg)
PLAN = {"landing: Scout": (0.0, 5.0, 11.0, 13.0, 180.0, 30.0, 22.0, 6.0),             # 9 Nov: the Scout sits 6 s, for its dome to shut and open
        "landing: Overseer": (13.0, 20.0, 22.5, 24.0, 260.0, -25.0, 30.0, -5.0)}


def seat_pose(t, plan, t_total):
    """the drone's offset from its seated pose in the seat frame at time t"""
    t0, ts, tl, te, h0, ox, oy, yaw = plan
    if t < t0 or t >= te:                                       # parked: above and behind, out of the way
        return np.array([PARK_X, 0.0, PARK_Z]), 0.0
    if t < ts:                                                  # descent: height eases from h0 to 0
        u = (t - t0) / (ts - t0); h = h0 * (1 - u) ** 1.6
    elif t < tl:                                                # seated
        h = 0.0
    else:                                                       # climb out, straight up
        u = (t - tl) / (te - tl); h = PARK_Z * u ** 1.8
    if t >= tl:                                                 # leaving: straight up out of the mouth
        return np.array([0.0, 0.0, h]), 0.0
    free = h / TAN                                              # how far off-centre the receiver can be at this height above the seat
    k = min(1.0, free / max(1e-6, math.hypot(ox, oy)))         # in contact with a face once the offset exceeds that: it follows the face down
    return np.array([ox * k, oy * k, h]), yaw * min(1.0, h / 40.0)


WALL_IN = 37.2                           # parts/box_funnel/make.py SETTINGS["scout"]: each end wall moves this far in for the Scout (Spark 9 Nov)
WALL_MOVES = ((11.8, 13.0, 1.0, 0.0), (22.6, 24.0, 0.0, 1.0))   # (t0, t1, from, to) in the "in" fraction: in for the Scout, out for the Overseer once the Scout is up, in again once the Overseer is up
WALLS = {"box_funnel_wall_nose": 1, "box_funnel_wall_tail": -1}
DOME = ((5.3, 7.8, True), (8.4, 10.9, False))   # (t0, t1, shutting): hoop 4 sweeps 0 -> 180 (shut) over the seated Scout, then back (2.5 s each here; 5 s at the motors' 6 rpm)


def dome_a4(t):
    """hoop 4's angle (deg) and whether it is shutting, at time t: open (0) except while the Scout sits"""
    for t0, t1, shutting in DOME:
        if t0 <= t < t1:
            u = (t - t0) / (t1 - t0); return (180.0 * u if shutting else 180.0 * (1 - u)), shutting
        if shutting and t1 <= t < DOME[1][0]:
            return 180.0, True
    return 0.0, False


def _dome():
    import importlib.util, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = importlib.util.spec_from_file_location("box_dome_make", os.path.join(root, "parts", "box_dome", "make.py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def walls_in(t):
    """how far in the end walls are (0 = the Overseer's 184 seat, 1 = the Scout's receiver) at time t"""
    u = 1.0
    for t0, t1, a0, a1 in WALL_MOVES:
        if t >= t1: u = a1
        elif t > t0: u = a0 + (a1 - a0) * (t - t0) / (t1 - t0)
    return u


def timeline(a):
    names = {g: [p["name"] for p in a["parts"] if p.get("group") == g] for g in PLAN}
    walls = {p["name"]: WALLS[p["id"]] for p in a["parts"] if p["id"] in WALLS}
    dm = _dome() if any(p["id"].startswith("box_dome_") for p in a["parts"]) else None
    moving = {p["name"]: p["id"] for p in a["parts"] if dm and p["id"] in dm.poses(0.0, False)}
    t_total = 24.0

    def poses(t):
        out = {}
        for g, plan in PLAN.items():
            d, yaw = seat_pose(t, plan, t_total)
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            D = np.array([[c, -s, 0, d[0]], [s, c, 0, d[1]], [0, 0, 1, d[2]], [0, 0, 0, 1]])
            M = SEAT_TO_W @ D @ W_TO_SEAT
            for n in names[g]:
                out[n] = M
        for n, sx in walls.items():
            D = np.eye(4); D[0, 3] = -sx * WALL_IN * walls_in(t)
            out[n] = SEAT_TO_W @ D @ W_TO_SEAT
        if moving:                                                   # the dome's hoops and fabric (drawn shut, in W) at hoop 4's angle: the folding fan
            P = dm.poses(*dome_a4(t))
            for n, pid in moving.items():
                out[n] = P[pid]
        return out

    n = int(t_total * a.get("fps", 24))                # the page's frame rate (systems/box.assembly.json "fps": 251 parts x 576 frames was 12 MB of poses)
    series = []
    for g, plan in PLAN.items():
        series.append((g.split(": ")[1] + " height over the seat", [seat_pose(i * t_total / n, plan, t_total)[0][2] if plan[0] <= i * t_total / n < plan[3] else None
                                                                      for i in range(n)], "mm"))
    series.append(("end walls in", [WALL_IN * walls_in(i * t_total / n) for i in range(n)], "mm"))
    if moving:
        series.append(("dome hoop 4", [dome_a4(i * t_total / n)[0] for i in range(n)], "deg"))
    return poses, t_total, series
