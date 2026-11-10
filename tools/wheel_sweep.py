"""The swap wheel's rotational sweep (3 Nov): every turning part of the arm (hub, rails, pins, heads, grabbers, fingers, pinions, servos,
wheels, bushes, ties, the carried packs, the hub-side pulley and belt, the detent ring) rotated about the axle (W X 0, Z 0, along Y)
in 5 deg steps; its surface samples tested inside the funnel, the enclosure's top band and every watertight part of each seated
drone (the cage rods, not watertight, by distance). The placeholder head at the drone is static (it is the head at the top).
Prints every colliding pair with its angles. Slow (exact containment): run it in the background.
    python tools/wheel_sweep.py
"""
import json, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import seat_envelope as se, seat_check as sc

TURN = ("hub", "rail", "pin16", "head", "wheels", "axles", "bush", "grip", "fing", "pinion", "servo", "motor_A", "motor_B", "shaft_A", "shaft_B", "ties", "pack", "p80", "belt", "detent_ring")


def main():
    a = json.load(open(os.path.join(ROOT, "systems", "box.assembly.json")))
    turning = []
    for p in a["parts"]:
        nm = p["name"].replace("arm: ", "")
        if p["name"].startswith("arm: ") and nm.startswith(TURN) and p.get("mesh"):
            m = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh"); m.apply_transform(np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4))
            pts, _ = trimesh.sample.sample_surface(m, 3000); turning.append((nm, np.vstack([pts, m.vertices])))
    statics = []
    for nm, mm, tri in sc.solids():
        t = tri.copy(); t.apply_transform(sc.SEAT_TO_W); statics.append((nm, t, None))
    for drone in ("overseer", "scout_v4"):                         # the live Scout (v3 archived 9 Nov)
        for nm, m in se.parts_of(drone):
            m = m.copy(); m.apply_transform(sc.SEAT_TO_W)
            if m.bounds[0][2] > 260 or m.bounds[1][1] < -10 or m.bounds[0][1] > 300:
                continue
            statics.append((f"{drone}: {nm}", m, "dist" if not m.is_watertight else None))
    print(f"turning parts {len(turning)}, static parts {len(statics)}")
    hits = {}
    for ang in range(0, 360, 5):
        c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang)); R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        for tn, pts in turning:
            q = pts @ R.T
            for sn, m, mode in statics:
                if not ((q.min(0) <= m.bounds[1]).all() and (q.max(0) >= m.bounds[0]).all()):
                    continue
                n = int((trimesh.proximity.closest_point(m, q)[1] < 1.5).sum()) if mode == "dist" else int(m.contains(q).sum())
                if n:
                    hits.setdefault((sn, tn), []).append(ang)
    print(f"{len(hits)} colliding pairs")
    for (sn, tn), angs in sorted(hits.items()):
        print(f"  {sn:44} <- {tn:14} at {len(angs)} of 72 angles: {angs[0]}..{angs[-1]} deg")


if __name__ == "__main__":
    main()
