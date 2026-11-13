"""The swap arm page's timeline (systems/swap_arm.assembly.json): one Overseer swap, kinematic. Head B fetches the charged pack from the
bay at 0 deg, the arm turns, head A pulls the dead pack at the drone, the arm turns 180, head B inserts. Parts carry `kin` (stage, head,
blade) from parts/box_arm8/make.py; a head's lift moves s, its carriage max(0, s - T2), the blades 5.1 apart when open."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "parts"))
import box_arm8_lib as L

# t, phi (head A's bar), sA, sB, A open, B open
KEYS = [(0, 180, 0, 0, 1, 1), (3, 180, 0, L.S_BAY, 1, 1), (4, 180, 0, L.S_BAY, 1, 0), (7, 180, 0, 0, 1, 0), (9, 90, 0, 0, 1, 0), (13, 90, L.S_DRONE, 0, 1, 0),
        (14, 90, L.S_DRONE, 0, 0, 0), (18, 90, 0, 0, 0, 0), (21, 270, 0, 0, 0, 0), (25, 270, 0, L.S_DRONE, 0, 0), (26, 270, 0, L.S_DRONE, 0, 1), (30, 270, 0, 0, 0, 1), (32, 270, 0, 0, 0, 1)]
T_PICK_B, T_PICK_A, T_DROP_B = 4.0, 14.0, 26.0


def state(t):
    t = max(0.0, min(t, KEYS[-1][0]))
    for a, b in zip(KEYS[:-1], KEYS[1:]):
        if a[0] <= t <= b[0]:
            u = (t - a[0]) / (b[0] - a[0]); u = u * u * (3 - 2 * u)
            return [a[i] + (b[i] - a[i]) * u for i in range(1, 6)]
    return list(KEYS[-1][1:])


def stage_T(phi, s, head, stage, blade=0, opened=0.0):
    R = L.Ry(phi); sg = 1.0 if head == "A" else -1.0
    d = s if stage == "lift" else L.carriage_s(s) if stage == "carriage" else 0.0
    T = R.copy(); T[:3, 3] = (R @ np.array([-blade * 5.1 * opened, 0, sg * d, 1.0]))[:3]
    return T


def timeline(a):
    parts = [(p["name"], p.get("kin") or {}) for p in a["parts"]]
    def lift(t, head):
        phi, sA, sB, oA, oB = state(t); return stage_T(phi, sA if head == "A" else sB, head, "lift")
    def poses(t):
        phi, sA, sB, oA, oB = state(t); out = {}
        for nm, k in parts:
            if k.get("stage") == "hub": out[nm] = L.Ry(phi)
            elif k.get("stage"):
                hd = k["head"]; out[nm] = stage_T(phi, sA if hd == "A" else sB, hd, k["stage"], k.get("blade", 0), oA if hd == "A" else oB)
            elif k.get("bay") == 0:                                  # the charged pack: with head B from its pick to its drop, then in the drone
                tt = min(max(t, T_PICK_B), T_DROP_B); out[nm] = lift(tt, "B") @ np.linalg.inv(lift(T_PICK_B, "B"))
            elif k.get("pack") == "dead":                            # the dead pack: with head A from its pick on
                out[nm] = lift(max(t, T_PICK_A), "A") @ np.linalg.inv(lift(T_PICK_A, "A"))
        return out
    n = int(KEYS[-1][0] * 24); ts = [i * KEYS[-1][0] / n for i in range(n)]
    series = [("head A travel", [state(t)[1] for t in ts], "mm"), ("head B travel", [state(t)[2] for t in ts], "mm"), ("arm angle", [state(t)[0] for t in ts], "deg")]
    return poses, float(KEYS[-1][0]), series
