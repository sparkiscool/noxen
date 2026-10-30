"""Re-export an existing parts/<part>/part.stl through the engine's export (topo/build.py weld_clean), no re-solve: the STL's own
triangles welded at STL precision, collapsed faces and open or sub-1 mm3 shells dropped, slivers collapsed, checked watertight.
The geometry is the stored result itself; nothing else of the result changes. A part whose mesh does not come out watertight is
left as it is and reported. result.json gets "export" (the report) and "watertight".

    python tools/weld_stl.py <part> [<part> ...]          (from the repo root)
    python tools/weld_stl.py --all                         every part with a result.json whose STL is not watertight
"""
import glob, json, os, sys, time
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from topo import build as B                                                     # noqa: E402


def weld(name, log=print):
    d = os.path.join(ROOT, "parts", name); stl = os.path.join(d, "part.stl"); rj = os.path.join(d, "result.json")
    if not (os.path.exists(stl) and os.path.exists(rj)):
        log(f"{name}: no part.stl + result.json"); return None
    t0 = time.time()
    m0 = trimesh.load(stl, force="mesh", process=False)
    shells0 = len(trimesh.Trimesh(m0.vertices, m0.faces, process=True).split(only_watertight=False))
    m, rep = B.weld_clean(np.asarray(m0.vertices), np.asarray(m0.faces))
    rep["volume_mm3"] = round(float(m.volume), 1); rep["source"] = dict(faces=int(len(m0.faces)), shells=shells0, mm3=round(float(m0.volume), 1))
    log(B.export_report(rep, tag=f"{name}: weld"))
    if not rep["watertight"]:
        log(f"{name}: left as it is ({time.time() - t0:.0f} s)"); return False
    m.export(stl)
    res = json.load(open(rj))
    res.update(export=rep, watertight=True, volume_mm3=rep["volume_mm3"])
    res["export"]["welded_at"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(res, open(rj, "w"), indent=1)
    log(f"{name}: written, {len(m.faces):,} faces, {time.time() - t0:.0f} s")
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--all"]:
        args = []
        for rj in sorted(glob.glob(os.path.join(ROOT, "parts", "*", "result.json"))):
            p = os.path.basename(os.path.dirname(rj)); stl = os.path.join(os.path.dirname(rj), "part.stl")
            if os.path.exists(stl) and not trimesh.load(stl, force="mesh").is_watertight:
                args.append(p)
        print("not watertight:", " ".join(args))
    bad = [p for p in args if weld(p) is False]
    sys.exit(1 if bad else 0)
