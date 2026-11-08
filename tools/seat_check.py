"""Seated-drone clearance check (3 Nov): does anything of a seated drone cross the funnel, the deck or the funnel's legs? Each
drone's parts (systems/<drone>.assembly.json, posed seated as tools/seat_envelope.py does) are intersected with the funnel and deck
solids as exact booleans (manifold3d; the funnel and deck come from their make.py, in the seat frame). A part that is not a
manifold (the cage pattern's concatenated tubes) falls back to dense surface sampling against the funnel. Prints every crossing
with the overlap volume, its bounding box in the seat frame and an approximate depth. python tools/seat_check.py scout|overseer|both
"""
import importlib.util, os, sys
import numpy as np, trimesh, manifold3d as mf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import seat_envelope as se

SEAT_TO_W = np.array([[0, -1.0, 0, 0], [1.0, 0, 0, 167.1], [0, 0, 1.0, 220.3], [0, 0, 0, 1.0]])   # seat z 0 at W Z 220.3 (3 Nov: +65)


def load_make(rel):
    sp = importlib.util.spec_from_file_location("mk_" + rel.replace("/", "_"), os.path.join(ROOT, rel)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def man(tri):
    try:
        mm = mf.Manifold(mf.Mesh(np.asarray(tri.vertices, np.float32), np.asarray(tri.faces, np.uint32)))
        return None if mm.is_empty() else mm
    except Exception:
        return None


def volume(mm):
    try:
        return float(mm.volume())
    except Exception:
        return float(mm.get_properties()[1])


def to_tri(mm):
    m = mm.to_mesh(); return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)


def solids():
    """the funnel (seat frame) and the enclosure top (W Z 145, the sheets' top edge as a 3 mm band, seat frame): there is no deck (3 Nov)"""
    fun = load_make("parts/box_funnel/make.py").funnel()
    top = mf.Manifold.cube([418.0, 409.0, 3.0]).translate([-209.0, -69.0, 142.0]) - mf.Manifold.cube([412.0, 403.0, 5.0]).translate([-206.0, -66.0, 141.0])
    top = top.transform(np.linalg.inv(SEAT_TO_W)[:3, :])
    return [("funnel", fun, to_tri(fun)), ("enclosure top", top, to_tri(top))]


def descent_states(drone):
    """the landing path of tools/landing_box.py at its capture limits: (label, offset x, offset y, height) with the offset following the
    55 deg faces down (offset = height / tan 55 once in contact); the states where a part can meet a wall or a leg"""
    import math
    tan = math.tan(math.radians(55.0)); out = [("seated", 0.0, 0.0, 0.0)]
    X = 28.0 if drone == "scout" else 18.0                 # the Overseer's x capture is +-18: its loop diagonals meet the end walls' top corners beyond that
    for lab, ox, oy in ((f"+x {X:.0f}", X, 0.0), (f"-x {X:.0f}", -X, 0.0), ("+y 20", 0.0, 20.0), ("-y 20", 0.0, -20.0), ("+x+y", X * 0.7, 14.0)):
        o0 = math.hypot(ox, oy)
        for h in (60.0, 45.0, 40.0, 35.0, 30.0, 27.0, 24.0, 20.0, 16.0, 12.0, 9.0, 6.0, 3.0):   # 4 Nov: every 3..5 (at 6 steps the skirt's graze on the buttress between 26 and 30 was missed)
            k = min(1.0, (h / tan) / o0)
            out.append((f"{lab} at h {h:.0f}", ox * k, oy * k, h))
    return out


def check(drone, verbose=True, offset=(0.0, 0.0, 0.0)):
    S = solids()
    hits = []
    T = np.eye(4); T[:3, 3] = offset
    for nm, m in se.parts_of(drone):
        if any(offset):
            m = m.copy(); m.apply_transform(T)
        for sname, sm, st in S:
            if not ((m.bounds[0] <= st.bounds[1]).all() and (m.bounds[1] >= st.bounds[0]).all()):
                continue
            mp = man(m)
            if mp is not None:
                inter = mp ^ sm
                v = volume(inter)
                if v > 0.5:
                    t = to_tri(inter); e = t.extents
                    hits.append((nm, sname, v, float(e.min()), t.bounds.round(1).tolist()))
            else:                                                           # not a manifold: sample the part's surface against the solid
                p, _ = trimesh.sample.sample_surface(m, 20000); p = np.vstack([p, m.vertices])
                inside = st.contains(p)
                if inside.any():
                    q = p[inside]; hits.append((nm, sname + " (sampled)", float("nan"), float("nan"), np.array([q.min(0), q.max(0)]).round(1).tolist()))
    if verbose:
        print(f"== {drone}: {len(hits)} crossings")
        for nm, s, v, d, bb in sorted(hits, key=lambda h: -(h[2] if h[2] == h[2] else 1e9)):
            print(f"  {nm:40} x {s:18} overlap {v:8.0f} mm3, about {d:4.1f} deep, seat box {bb}")
    return hits


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    for d in (["scout", "overseer"] if which == "both" else [which]):
        check(d)
        if "--descent" in sys.argv:
            bad = 0
            for lab, ox, oy, h in descent_states(d)[1:]:
                hs = check(d, verbose=False, offset=(ox, oy, h))
                if hs:
                    bad += 1; print(f"  descent {lab:14}: " + "; ".join(f"{nm[:28]} x {s} {v:.0f} mm3 at {bb}" for nm, s, v, dd, bb in hs))
            print(f"  descent: {bad} of {len(descent_states(d)) - 1} states with a crossing")
