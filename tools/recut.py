"""Recut a finished part's holes without rerunning it (Spark 29 Oct: "just changing the hole sizes"): the part's CURRENT setup
is built, its cuts (bores, resin grooves, lead-ins, teardrop crowns, bolt holes, keep-outs) are subtracted from the existing
parts/<part>/part.stl, the export cleanup and the solid gate (with the setup's thin_ok bodies) run on the result, and result.json is updated (solid, mass, hashes, a recut note).
Strength is not re-solved: a cut only removes material round a hole, so the recorded index is a lower bound; the gate's
thin-wall check is what guards the walls round the new grooves and crowns.
Run: python tools/recut.py <part> [<part> ...]      (from the repo root; writes part.stl, preview.png, result.json)
"""
import importlib, json, os, sys, time
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from topo.part import man, tm                                                   # noqa: E402
from topo import build as B                                                     # noqa: E402
from part_sha import part_sha, lib_sha                                          # noqa: E402
from topo.material import MATERIALS                                             # noqa: E402


def recut(name, log=print):
    d = os.path.join(ROOT, "parts", name); stl = os.path.join(d, "part.stl"); rj = os.path.join(d, "result.json")
    res = json.load(open(rj))
    mod = importlib.import_module(f"parts.{name}.part")
    part = mod.setup(float(res["h"]))
    m0 = trimesh.load(stl, force="mesh")
    # a big exported STL (the head mount, 500k faces) carries a few duplicate vertices and degenerate faces that make manifold3d
    # read it as NotManifold (empty, volume 0: 29 Oct, the recut then crashed in export_cleanup); merge and drop them first
    m0.merge_vertices(); m0.update_faces(m0.nondegenerate_faces()); m0.update_faces(m0.unique_faces()); m0.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m0)
    M0 = man(m0)
    if M0.volume() < 1.0:
        raise RuntimeError(f"{name}: part.stl does not read as a solid (manifold status {M0.status()}); nothing recut")
    P = M0 - part.cuts_m
    P, cleanup = B.export_cleanup(P, part, log=log)                              # the export cleanup, as a fresh run would do it
    PT = B.clean_mesh(tm(P), log=log)
    drawn_fin = tm(part.drawn_m - part.cuts_m)
    ok, solid = B.solid_check(PT, [drawn_fin], thin_ok=[tm(m - part.cuts_m) for m in getattr(part, "thin_ok", [])])
    mass = PT.volume / 1000.0 * MATERIALS[part.material].density
    removed = (m0.volume - PT.volume) / 1000.0
    PT.export(stl)
    try:
        B.render(PT, os.path.join(d, "preview.png"), title=f"{name} (nox/topo, recut)",
                 info=f"{mass:.1f} g | solid check {'PASS' if ok else 'FAIL'} | recut holes")
    except Exception as e:
        log(f"  render failed: {e}")
    py = os.path.join(d, "part.py")
    res.update(solid_ok=bool(ok), solid=solid, export_cleanup=cleanup, mass_g=round(mass, 2), volume_mm3=round(PT.volume, 1), watertight=bool(PT.is_watertight),
               part_sha=part_sha(py), lib_sha=lib_sha(py),
               recut=dict(at=time.strftime("%Y-%m-%d %H:%M"), removed_cm3=round(removed, 3), note="holes recut from the current setup; strength not re-solved"))
    json.dump(res, open(rj, "w"), indent=1)
    log(f"{name}: recut, {removed:.2f} cm3 removed, {mass:.1f} g, solid check {'PASS' if ok else 'FAIL'}"
        + ("" if ok else f" {json.dumps({k: v for k, v in solid.items() if k in ('necks', 'drawn_not_in_main', 'thin_walls', 'bodies_plain')})[:400]}"))
    return ok


if __name__ == "__main__":
    bad = [p for p in sys.argv[1:] if not recut(p)]
    sys.exit(1 if bad else 0)
