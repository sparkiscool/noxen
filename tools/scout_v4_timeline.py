"""The Scout v4 page's timeline (systems/scout_v4.assembly.json "timeline", run by tools/build_site.py --inline): the drone turns
on the turntable while the ceiling probe rolls its two-blade mast out 300 mm, holds it up for the scan, and rolls it back.

    poses(t) -> {part name: 4x4}: every part spins about z (360 deg in 24 s); the head rises by the length out; each blade's 1 mm
    mesh above the throat (tools/scout_v4_shape.py mast()) stretches to that length.
"""
import math
import numpy as np

DUR = 24.0
Z0, STUB, OUT = 73.0, 1.0, 300.0           # throat top (scout_v4_shape.THROAT z1), the blades' drawn length, the full roll-out
PHASES = ((3.0, "parked"), (7.0, "rolling out"), (15.0, "scan"), (19.0, "rolling back"), (DUR, "parked"))


def out_mm(t):
    ease = lambda u: 0.0 if u <= 0 else 1.0 if u >= 1 else u * u * (3 - 2 * u)
    if t < 3.0:
        return 0.0
    if t < 7.0:
        return OUT * ease((t - 3.0) / 4.0)                   # ~1.6 s at the N20's speed; slowed here to be seen
    if t < 15.0:
        return OUT
    if t < 19.0:
        return OUT * (1 - ease((t - 15.0) / 4.0))
    return 0.0


def timeline(a):
    by_id = {p.get("id"): p["name"] for p in a["parts"]}
    names = [p["name"] for p in a["parts"]]
    head = by_id["scout_v4_probe_head"]
    blades = [by_id["scout_v4_probe_mast_a"], by_id["scout_v4_probe_mast_b"]]

    def poses(t):
        c, s = math.cos(2 * math.pi * t / DUR), math.sin(2 * math.pi * t / DUR)
        R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        e = out_mm(t)
        lift = np.eye(4); lift[2, 3] = e
        k = max(e, 0.05) / STUB                              # the stub's length scaled to the length out (hidden in the clip when parked)
        S = np.eye(4); S[2, 2] = k; S[2, 3] = Z0 * (1 - k)   # scale about the throat top
        P = {n: R for n in names}
        P[head] = R @ lift
        for b in blades:
            P[b] = R @ S
        return P

    n = int(round(DUR * 24))
    vals = [out_mm(i * DUR / n) for i in range(n)]
    return poses, DUR, [("probe mast out", vals, "mm")]
