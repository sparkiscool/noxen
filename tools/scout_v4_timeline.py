"""The Scout v4 page's timeline (systems/scout_v4.assembly.json "timeline", run by tools/build_site.py --inline): the drone turns on
the turntable while the ceiling probe rolls its two-blade mast out, nods the camera through its whole range, and rolls it back; then,
parked, the front camera nods.

    poses(t) -> {part name: 4x4}. Every part spins about z (360 deg over the loop). The spool turns as the blades pay out (and the
    pinion with it, 40 / 15 faster, the other way); the head rises by the length out; the camera cradle and its LED bars nod about
    the SG90's axis; each blade's 1 mm mesh above the throat (tools/scout_v4_shape.py mast()) stretches to the length out.
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scout_v4_shape as S                 # the geometry's own numbers, so the page never drifts from the design

DUR = 28.0
Z0, STUB = S.THROAT["z1"], S.MAST_STUB     # throat top, the blades' drawn length
OUT = S.REACH
AXIS = np.array(S.AXIS, float)             # the tilt axis parked
TILT0 = math.degrees(S.TILT)               # parked tilt, deg up
SPOOL = S.SPOOL["c"]
PINION = (SPOOL[0], SPOOL[1] - (S.GEAR["spool_t"] + S.GEAR["pinion_t"]) * S.GEAR["m"] / 2)
R_WOUND, RATIO = S.SPOOL["hub"] + 1.6, S.GEAR["spool_t"] / S.GEAR["pinion_t"]   # mean wound radius; the pinion turns RATIO times the spool

ease = lambda u: 0.0 if u <= 0 else 1.0 if u >= 1 else u * u * (3 - 2 * u)


def keys(t, pts):
    """piecewise eased between (time, value) keys"""
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0)) if t1 > t0 else v1
    return pts[-1][1]


def out_mm(t):
    return keys(t, [(0, 0), (2, 0), (5, OUT), (21, OUT), (24, 0), (DUR, 0)])


def tilt_deg(t):
    return keys(t, [(0, TILT0), (2, TILT0), (5, 5), (8, -20), (13, 90), (16, 90), (19, 50), (21, TILT0), (24, TILT0),
                    (25.3, -10), (26.7, 40), (DUR, TILT0)])


def rot_y(a, c):
    """rotation by a (rad) about the y axis through c; a > 0 turns +x toward -z"""
    M = np.eye(4); ca, sa = math.cos(a), math.sin(a)
    M[:3, :3] = [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]
    M[:3, 3] = np.asarray(c, float) - M[:3, :3] @ np.asarray(c, float)
    return M


def timeline(a):
    by_id = {p.get("id"): p["name"] for p in a["parts"]}
    names = [p["name"] for p in a["parts"]]
    get = lambda k: by_id.get("scout_v4_" + k)

    def poses(t):
        c, s = math.cos(2 * math.pi * t / DUR), math.sin(2 * math.pi * t / DUR)
        R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        e = out_mm(t)
        lift = np.eye(4); lift[2, 3] = e
        nod = rot_y(-math.radians(tilt_deg(t) - TILT0), AXIS + [0, 0, e])        # look up = +x toward +z
        k = max(e, 0.05) / STUB
        S = np.eye(4); S[2, 2] = k; S[2, 3] = Z0 * (1 - k)                       # the blades' stub stretched about the throat top
        turn = e / R_WOUND                                                       # the spool's turn for the length paid out
        P = {n: R for n in names}
        for nm, M in (("probe_head", lift), ("probe_tilt", nod @ lift), ("probe_leds", nod @ lift), ("probe_mast_a", S), ("probe_mast_b", S),
                      ("probe_spool", rot_y(-turn, (SPOOL[0], 0, SPOOL[1]))), ("probe_pinion", rot_y(turn * RATIO, (PINION[0], 0, PINION[1])))):
            if get(nm):
                P[get(nm)] = R @ M
        return P

    n = int(round(DUR * 24))
    ts = [i * DUR / n for i in range(n)]
    return poses, DUR, [("probe mast out", [out_mm(t) for t in ts], "mm"), ("camera tilt", [tilt_deg(t) for t in ts], "deg")]
