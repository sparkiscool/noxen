"""The seated-drone envelope (3 Nov): what a landing funnel must clear. Both drones are posed in the seat frame (drone coordinates of
the seated Scout: x nose, y left, z up; the receiver bottom at z -44.3) with their latch footprint at the same place, and every
part but the latch set is sliced in 2 mm bands of height above the seat. Per band: the half-widths in x (nose / tail, about the
receiver centre x 1.5) and y, and the part that sets each. Writes runs/seat_envelope/<drone>.json and prints the table.
    python tools/seat_envelope.py scout_v4|overseer|both   (both = scout_v4 + overseer; Scout v3 archived 9 Nov)
Overseer -> seat: T_OV (tools/import_box.py): its catch centre x -30 and cartridge bottom z -76.9 (cage v5: the loop at -79.9 is 3 under it) onto the Scout's (1.5, -44.5).
"""
import json, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from build_site import pose_matrix, placeholder
SEAT_Z, RX = -44.3, 1.5
T_OV = np.array([[1, 0, 0, 29.52], [0, 1, 0, 0], [0, 0, 1, 32.4], [0, 0, 0, 1.0]])   # its cartridge bottom z -76.9 (cage v5) onto the Scout's -44.5
SKIP = ("receiver", "cartridge", "slider", "latch", "pack", "battery holder", "wires", "cables", "props (swept)", "ghost", "view")


def parts_of(sysname):
    a = json.load(open(os.path.join(ROOT, "systems", f"{sysname}.assembly.json")))
    out = []
    for p in a["parts"]:
        nm = p["name"]
        if any(k in nm.lower() for k in SKIP):
            continue
        src = p.get("part", p.get("id"))
        cand = os.path.join(ROOT, "parts", src, "part.stl")
        if os.path.exists(cand):
            m = trimesh.load(cand, force="mesh")
        elif p.get("mesh") and os.path.exists(os.path.join(ROOT, p["mesh"])):
            m = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh")
        else:
            m = placeholder(p)
            if m is None:
                continue
        m.apply_transform(pose_matrix(p.get("pose")))
        if sysname == "overseer":
            m.apply_transform(T_OV)
        out.append((nm, m))
    return out


def envelope(parts, hmax=260.0, step=2.0):
    rows = []
    for h0 in np.arange(0.0, hmax, step):
        z0, z1 = SEAT_Z + h0, SEAT_Z + h0 + step
        best = dict(h=float(h0), xp=None, xn=None, yp=None, yn=None, xp_by="", xn_by="", yp_by="", yn_by="")
        for nm, m in parts:
            v = m.vertices; sel = (v[:, 2] >= z0) & (v[:, 2] < z1)
            if not sel.any():
                continue
            x, y = v[sel, 0] - RX, v[sel, 1]
            for k, val in (("xp", x.max()), ("xn", -x.min()), ("yp", y.max()), ("yn", -y.min())):
                if best[k] is None or val > best[k]:
                    best[k] = float(val); best[k + "_by"] = nm
        rows.append(best)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    os.makedirs(os.path.join(ROOT, "runs", "seat_envelope"), exist_ok=True)
    for d in (["scout_v4", "overseer"] if which == "both" else [which]):   # the live Scout (v3 archived 9 Nov)
        ps = parts_of(d)
        lo = min(float(m.bounds[0][2]) for _, m in ps)
        rows = envelope(ps)
        json.dump(dict(drone=d, seat_z=SEAT_Z, lowest_z=lo, rows=rows), open(os.path.join(ROOT, "runs", "seat_envelope", f"{d}.json"), "w"), indent=1)
        print(f"== {d}: {len(ps)} parts, lowest point z {lo:.1f} ({lo - SEAT_Z:+.1f} from the seat)")
        print("  h   | x nose  by                      | x tail  by                      | y       by")
        for r in rows[::5]:
            if r["xp"] is None:
                continue
            print(f"  {r['h']:4.0f} | {r['xp']:6.1f} {r['xp_by'][:24]:24} | {r['xn']:6.1f} {r['xn_by'][:24]:24} | {max(r['yp'], r['yn']):6.1f} {(r['yp_by'] if r['yp'] >= r['yn'] else r['yn_by'])[:24]}")
