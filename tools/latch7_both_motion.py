"""Both latch v7 sets on one page (systems/latch7_both.assembly.json): the Scout's cycle (tools/latch7_motion.py) and the Overseer's
(tools/latch7_overseer_motion.py) side by side, each moved across by its part's own offset (parts/latch7/make.py BOTH_X)."""
import importlib.util, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFF = {"scout": -45.0, "overseer": 45.0}
KEEP = ("cartridge below the seat", "bolt travel", "spring force", "heel pad load", "grip hold: pads stay down")


def _load(f):
    sp = importlib.util.spec_from_file_location(f, os.path.join(ROOT, "tools", f + ".py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def timeline(a):
    mods = {"scout": _load("latch7_motion"), "overseer": _load("latch7_overseer_motion")}
    sub, series, dur = {}, [], 0.0
    for pk, m in mods.items():
        names = [p["name"][len(pk) + 2:] for p in a["parts"] if p["name"].startswith(pk + ": ")]
        poses, TT, ser = m.timeline({"parts": [{"name": n} for n in names]})
        T = np.eye(4); T[0, 3] = OFF[pk]; Ti = np.eye(4); Ti[0, 3] = -OFF[pk]
        sub[pk] = (poses, T, Ti); dur = max(dur, TT)
        series += [(f"{pk.capitalize()}: {nm}", vals, unit) for nm, vals, unit, *_ in ser if nm.startswith(KEEP)]

    def poses_at(t):
        out = {}
        for pk, (poses, T, Ti) in sub.items():
            for k, M in poses(t).items():
                out[f"{pk}: {k}"] = T @ np.asarray(M) @ Ti
        return out
    return poses_at, dur, series
