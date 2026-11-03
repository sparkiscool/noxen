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

SEAT_TO_W = np.array([[0, -1.0, 0, 0], [1.0, 0, 0, 167.1], [0, 0, 1.0, 155.3], [0, 0, 0, 1.0]])


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
    fun = load_make("parts/box_funnel/make.py").funnel()
    dk = load_make("parts/box_deck/make.py")
    deck = dk.box(dk.X0, dk.X1, dk.Y0, dk.Y1, dk.Z_TOP - dk.T, dk.Z_TOP) - dk.box(-dk.HOLE_X / 2, dk.HOLE_X / 2, dk.YC - dk.HOLE_Y / 2, dk.YC + dk.HOLE_Y / 2, dk.Z_TOP - dk.T - 1, dk.Z_TOP + 1)
    deck = deck.transform(np.linalg.inv(SEAT_TO_W)[:3, :])
    return [("funnel", fun, to_tri(fun)), ("deck", deck, to_tri(deck))]


def check(drone, verbose=True):
    S = solids()
    hits = []
    for nm, m in se.parts_of(drone):
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
