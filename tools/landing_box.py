"""The box page's timeline: both drones landing into the funnel (Spark 3 Nov: animate how both drones will land on this).
Kinematic, not simulated: each drone comes down off-centre and yawed, meets a 55 deg face with its receiver, and the offset then
follows the face to the seat (offset = height above the seat / tan 55, the geometry of a block sliding down the face); yaw
straightens in the last 40 mm. Scout first (0..10 s), then the Overseer (10..24 s). Groups "landing: Scout" and "landing: Overseer"
in systems/box.assembly.json move as rigid bodies; everything else stays. Transforms are in the box W frame: the drones are posed
seated (SEAT_TO_W from the catch), so the motion is a seat-frame offset conjugated by SEAT_TO_W.
"""
import math
import numpy as np

SEAT_TO_W = np.array([[0, -1.0, 0, 0], [1.0, 0, 0, 167.1], [0, 0, 1.0, 220.3], [0, 0, 0, 1.0]])   # seat z 0 at W Z 220.3 (3 Nov: +65, the arm's head extends 60 more at the drone slot)
W_TO_SEAT = np.linalg.inv(SEAT_TO_W)
TAN = math.tan(math.radians(55.0))
PARK_Z, PARK_X = 560.0, -420.0          # where a drone waits when it is not the one landing (above and behind the box, out of the front view)

# per drone: (t_start, t_seat, t_leave, t_end, start height, offset x, offset y, yaw deg)
PLAN = {"landing: Scout": (0.0, 6.0, 8.0, 10.0, 180.0, 30.0, 22.0, 6.0),
        "landing: Overseer": (10.0, 18.0, 20.5, 24.0, 260.0, -25.0, 30.0, -5.0)}


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


def timeline(a):
    names = {g: [p["name"] for p in a["parts"] if p.get("group") == g] for g in PLAN}
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
        return out

    n = int(t_total * 24)
    series = []
    for g, plan in PLAN.items():
        series.append((g.split(": ")[1] + " height over the seat", [seat_pose(i * t_total / n, plan, t_total)[0][2] if plan[0] <= i * t_total / n < plan[3] else None
                                                                      for i in range(n)], "mm"))
    return poses, t_total, series
