"""The xswap short lock's swap cycle for systems/xswap.assembly.json (build_site.py `timeline`): the pincer rises 55 (+2.3 lift), the
fingers close 14.6 on the pinion and swing the gate bars 17 deg open, the lift comes off (tabs into their seats), the rail pulls the
carrier 45 down, it flips 180 deg about the carrier axis, holds, then every phase in reverse. Constants and phase lengths are the lock
page's (xswap short lock v5, 3 Nov), in the page frame; poses come out in the drone frame through import_xswap.A. The meshes sit at
their page identity pose except the pincer, stored at the rest pose (55 down), so every matrix here is relative to the file's pose.
"""
import math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from import_xswap import A, REST_DZ, FEET_Z

PIV_Y, Y_FLIP, Z_AX, R_P, FIN_A = 40.0, 63.6, -130.3, 13.5, 12.0
CLR, STROKE, LIFT, RAIL = 10.453588980387476, 14.6, 2.3, 45.0
PH = [("pincer rises", 2.2), ("fingers close, bars swing open", 3.0), ("tabs drop into their seats", 1.0),
      ("rail pulls the pack down", 3.0), ("flip 180", 3.0), ("hold", 1.2)]
T1 = sum(d for _, d in PH); TT = 2 * T1
A_INV = np.linalg.inv(A)


def ease(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x * x * (3 - 2 * x)


def theta(u):
    if u <= CLR:
        return 0.0
    r, phi = math.hypot(u, FIN_A), math.atan2(FIN_A, u)
    return max(0.0, math.acos(min(1.0, CLR / r)) - phi)


def state(t):
    if t > T1:
        t = TT - t
    f, a = [], 0.0
    for _, d in PH:
        f.append(ease((t - a) / d)); a += d
    return dict(dz=-55 + (55 + LIFT) * f[0] - LIFT * f[2], u=STROKE * f[1], rail=RAIL * f[3], flip=math.pi * f[4])


def tr(x, y, z):
    T = np.eye(4); T[:3, 3] = (x, y, z); return T


def rx(a):
    c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def page_poses(s):
    """part id -> 4x4 in the page frame, relative to the file's pose."""
    inner = tr(0, Y_FLIP, Z_AX - s["rail"]) @ rx(s["flip"]) @ tr(0, -Y_FLIP, -Z_AX)
    th = theta(s["u"])
    pincer = inner @ tr(0, 0, s["dz"] - REST_DZ)
    return {"xswap_shell": inner, "xswap_pack": inner, "xswap_xt60_female": inner,
            "xswap_gate_bar_a": inner @ tr(0, PIV_Y, 0) @ rz(th) @ tr(0, -PIV_Y, 0),
            "xswap_gate_bar_b": inner @ tr(0, -PIV_Y, 0) @ rz(-th) @ tr(0, PIV_Y, 0),
            "xswap_lid": pincer, "xswap_jaw_base": pincer, "xswap_servo": pincer,
            "xswap_finger_a": pincer @ tr(0, -s["u"], 0), "xswap_finger_b": pincer @ tr(0, s["u"], 0),
            "xswap_pinion": pincer @ rz(s["u"] / R_P)}


def drone_poses(t):
    return {k: A @ M @ A_INV for k, M in page_poses(state(t)).items()}


def timeline(a):
    name = {p["id"]: p["name"] for p in a["parts"]}
    shell = trimesh.load(os.path.join(ROOT, "parts", "xswap_shell", "part.stl"), force="mesh")
    V = np.c_[shell.vertices, np.ones(len(shell.vertices))]

    def poses(t):
        return {name[k]: M for k, M in drone_poses(t).items() if k in name}

    fps, n = 24, int(TT * 24)
    ts = [i * TT / n for i in range(n)]
    st = [state(t) for t in ts]
    low = [float((V @ drone_poses(t)["xswap_shell"].T)[:, 2].min()) - FEET_Z for t in ts]
    series = [("shell lowest point over the cage feet", low, "mm", dict(bad_below=0.0)),
              ("finger stroke", [s["u"] for s in st], "mm"), ("gate bar swing", [math.degrees(theta(s["u"])) for s in st], "deg"),
              ("rail", [s["rail"] for s in st], "mm"), ("flip", [math.degrees(s["flip"]) for s in st], "deg")]
    return poses, TT, series


if __name__ == "__main__":
    for t in (0, 2.2, 5.2, 6.2, 9.2, 12.2, 13.4, 26.7):
        s = state(t); print(f"t {t:5.1f}: dz {s['dz']:6.1f} u {s['u']:5.1f} th {math.degrees(theta(s['u'])):5.1f} rail {s['rail']:5.1f} flip {math.degrees(s['flip']):6.1f}")
