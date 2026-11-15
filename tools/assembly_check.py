"""Does the drone go together? (15 Nov, Spark's Scout v4 bench print: an SG90 tab post snapped, the camera could not slide into its
sleeve, the head frame and the cradle would not assemble; earlier the frame and the turret clashed, latch v7 clashed with the frame,
caught only by hand. Every part had passed its own gates.) The gate on the parts of a drone together, from systems/<drone>.assembly.json:

  interference  every pair of placed parts (printed with printed, printed with bought; two bought parts only when declared in
                `mates`): the volume they share, from the meshes as placed. Over fit_tol_mm3 (0.5) FAILS, unless the pair is a declared
                `contacts` entry (an insert pressed in, a gear mesh, a cable coiled in its drum: allow_mm3 and why). Each failing pair
                is reported piece by piece: mm3, where, extent, depth (its smallest extent: how far one stands in the other).
  mates         declared mating pairs {"a", "b", "fit"}: the least gap between them (manifold3d min_gap) must be at least the fit
                table's gap for the kind (topo/fit.py: the printer's own numbers from the Scout v4 coupon), and they must not interfere.
  paths         declared assembly paths (topo/paths.py), in `order`: the piece (with the parts that ride on it, `with`) stepped out the
                way it goes in (0.5 mm / 0.5 deg a step), through every part already placed (every part with no path, and the paths of
                lower order; `ignore` lists what it may pass). Blocked: at some step it stands in a part over the tolerance more than it
                does seated: FAILS (how much, where, how far along). Tight: at some step past the seat its gap to a part is under the
                path's fit gap (topo/fit.py): FAILS. A pair touching seated (a horn on its spline) is held to the volume rule only.
  print-ready   a printed part named in any failing pair, mate or path is not print-ready (the page says so on its line and in the
                drone's assembly report; tools/build_site.py), whatever its own gates say.

Format (systems/<drone>.assembly.json, top level):
  "assembly": {"fit_tol_mm3": 0.5,
               "contacts": [{"a": id, "b": id, "allow_mm3": 5, "why": "..."}],
               "mates":    [{"a": id, "b": id, "fit": "slide", "why": "..."}],
               "paths":    [ see topo/paths.py ],
               "order":    {"<id or fnmatch pattern>": n}       # when a part with no path of its own goes in (default 0: placed first)
               "generate": ["tools/<file>.py:<function>"]     # function(assembly dict) -> {"paths": [...], "contacts": [...], "mates": [...]}
              }
  a part line may carry "fit_mesh": "tools/<file>.py:<function>" (function(line) -> trimesh, world frame): the geometry the check
  uses instead of the page's mesh (the Scout v4 rods: cut to their socket floors, the page draws them node to node).
  role "placeholder" parts are left out, and "exclude": [ids] (a second drawing of a part already placed, e.g. a page's cut-away copy).
  a generated part's line may carry "prototype": "<folder of a make.py>" (tools/make_parts.py): the drawn prototype it replaces. The
  generated part is placed (here and on the page) only when its own gates pass on the current setup and, placed, it is named in no
  failing pair, mate or path; otherwise the prototype is, and the report's "placed" says which ({id: {src, stage, why}}).

    python tools/assembly_check.py <drone> [<drone> ...] | --all   [--write]  [--quiet]
      --all: every drone file not archived; each gates, with or without an "assembly" block (without one: interference alone, every
      overlap a clash until its owner declares the contacts, mates and paths).
      --write: systems/<drone>.assembly_check.json, and each printed part's result.json "assembly" (its verdict, the lines that name it)
      exit 1 when a gated drone fails.
"""
import importlib.util, json, os, sys, time
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from topo import fit as FIT                                     # noqa: E402
from topo import paths as PATHS                                 # noqa: E402
sys.path.insert(0, HERE)
import make_parts as MAKE                                       # noqa: E402
import print_mesh                                               # noqa: E402

PRINTED_MAT = ("PCTG", "PETG", "TPU", "PLA")


def _fn(spec):
    mod, fn = spec.rsplit(":", 1)
    path = os.path.join(ROOT, mod)
    sp = importlib.util.spec_from_file_location("ac_" + os.path.basename(mod)[:-3], path)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m, fn)


def pose_matrix(pose):
    if pose is None:
        return np.eye(4)
    if isinstance(pose, list):
        return np.array(pose, float)
    T = np.eye(4)
    rpy = np.radians(pose.get("rpy_deg", [0, 0, 0]))
    cr, sr, cp, sp, cy, sy = np.cos(rpy[0]), np.sin(rpy[0]), np.cos(rpy[1]), np.sin(rpy[1]), np.cos(rpy[2]), np.sin(rpy[2])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    T[:3, :3] = Rz @ Ry @ Rx; T[:3, 3] = pose.get("t", [0, 0, 0])
    return T


def is_printed(line):
    src = line.get("part", line.get("id"))
    return (line.get("role") not in ("bought", "cage", "placeholder") and
            (line.get("material") in PRINTED_MAT or os.path.exists(os.path.join(ROOT, "parts", src, "part.py"))
             or os.path.exists(os.path.join(ROOT, "parts", src, "make.py")) or line.get("role") == "structure"))


def load_mesh(line):
    if line.get("fit_mesh"):
        return _fn(line["fit_mesh"])(line)
    src = line.get("part", line.get("id"))
    cands = [line.get("mesh"), f"parts/{src}/part.stl"]
    for c in cands:
        if c and os.path.exists(os.path.join(ROOT, c)):
            m = trimesh.load(os.path.join(ROOT, c), force="mesh")
            m.apply_transform(pose_matrix(line.get("pose")))
            return m
    return None


def to_man(m):
    import manifold3d as mf
    from topo.part import man
    mm = man(m)
    if mm.status() != mf.Error.NoError or mm.is_empty():
        m = m.copy(); m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); trimesh.repair.fill_holes(m)
        mm = man(m)
        if mm.status() != mf.Error.NoError:
            raise ValueError(f"not a solid ({mm.status()})")
    return mm


def pieces(I, n=4):
    out = []
    for p in sorted(I.decompose(), key=lambda q: -q.volume())[:n]:
        bb = np.asarray(p.bounding_box(), float); ext = bb[3:] - bb[:3]
        out.append(dict(mm3=round(float(p.volume()), 2), at=[round(float(v), 1) for v in (bb[:3] + bb[3:]) / 2],
                        extent_mm=[round(float(v), 1) for v in ext], depth_mm=round(float(ext.min()), 2)))
    return out


def run(drone, log=print, _proto=None):
    f = os.path.join(ROOT, "systems", f"{drone}.assembly.json")
    a = json.load(open(f))
    A = dict(a.get("assembly") or {})
    for g in A.get("generate", []):
        extra = _fn(g)(a)
        for k in ("paths", "contacts", "mates"):
            A[k] = list(A.get(k, [])) + list(extra.get(k, []))
    tol = float(A.get("fit_tol_mm3", 0.5))
    t0 = time.time()
    lines = [p for p in a["parts"] if p.get("role") != "placeholder" and (p.get("id") or p["name"]) not in A.get("exclude", [])]
    stage_of = {}
    for k, p in enumerate(lines):                                  # a generated part on its prototype's line: placed only when its gates pass
        if not p.get("prototype"):
            continue
        i = p.get("id") or p["name"]; gen = p.get("part", i); st = MAKE.part_state(gen)
        if not st["own_ok"] or i in (_proto or ()):
            why = f"{gen}: {st['status']}" if not st["own_ok"] else f"{gen}: passes its own gates, fails the assembly check: {'; '.join(_proto[i][:2])}"
            lines[k] = dict(p, part=p["prototype"], mesh=f"parts/{p['prototype']}/part.stl")
            stage_of[i] = dict(src=p["prototype"], stage="prototype", why=why)
        else:
            stage_of[i] = dict(src=gen, stage="optimised", why=None)
    ids = [p.get("id") or p["name"] for p in lines]
    by = dict(zip(ids, lines))
    M, MM, skipped = {}, {}, []
    for i, p in zip(ids, lines):
        try:
            m = load_mesh(p)
            if m is None:
                skipped.append(dict(part=i, why="no mesh")); continue
            M[i] = m; MM[i] = to_man(m)
        except Exception as e:
            skipped.append(dict(part=i, why=str(e)[:160])); M.pop(i, None)
    printed = {i for i in M if is_printed(by[i])}
    key = lambda x, y: tuple(sorted((x, y)))
    contacts = {key(c["a"], c["b"]): c for c in A.get("contacts", [])}
    mates = {key(c["a"], c["b"]): c for c in A.get("mates", [])}
    blame = {}

    def bad(part, what):
        if part in printed:
            blame.setdefault(part, []).append(what)

    # interference
    pairs = []
    have = [i for i in ids if i in M]
    for x in range(len(have)):
        for y in range(x + 1, len(have)):
            i, j = have[x], have[y]
            k = key(i, j)
            if i not in printed and j not in printed and k not in mates and k not in contacts:
                continue
            bi, bj = M[i].bounds, M[j].bounds
            if np.any(np.minimum(bi[1], bj[1]) <= np.maximum(bi[0], bj[0])):
                continue
            I = MM[i] ^ MM[j]; v = float(I.volume())
            if v <= 1e-3 and k not in contacts:
                continue
            c = contacts.get(k)
            lim = float(c["allow_mm3"]) if c else tol
            row = dict(a=i, b=j, mm3=round(v, 2), limit_mm3=lim, contact=c.get("why") if c else None, ok=bool(v <= lim))
            if not row["ok"] or v > tol:
                row["pieces"] = pieces(I)
            pairs.append(row)
            if not row["ok"]:
                bad(i, f"clashes with {j}: {v:.1f} mm3"); bad(j, f"clashes with {i}: {v:.1f} mm3")
    for k, c in contacts.items():
        if not any(key(r["a"], r["b"]) == k for r in pairs) and all(x in M for x in k):
            pairs.append(dict(a=k[0], b=k[1], mm3=0.0, limit_mm3=float(c["allow_mm3"]), contact=c.get("why"), ok=True))

    # mates
    mrows = []
    for k, c in mates.items():
        if not all(x in M for x in k):
            mrows.append(dict(a=c["a"], b=c["b"], fit=c.get("fit"), ok=False, why="a part has no mesh")); continue
        need = FIT.gap(c.get("fit", "slide"))
        I = MM[c["a"]] ^ MM[c["b"]]; v = float(I.volume())
        g = 0.0 if v > 1e-3 else float(MM[c["a"]].min_gap(MM[c["b"]], max(2.0, need + 1.0)))
        ok = (v <= tol and g >= need) if need >= 0 else True
        r = dict(a=c["a"], b=c["b"], fit=c.get("fit", "slide"), need_mm=need, gap_mm=round(g, 3), overlap_mm3=round(v, 2), ok=bool(ok), why=c.get("why"))
        if v > 1e-3:
            r["pieces"] = pieces(I)
        mrows.append(r)
        if not ok:
            msg = f"mate with {{}} ({r['fit']}): gap {g:.2f} mm, the fit table asks {need:.2f}" + (f", overlapping {v:.1f} mm3" if v > 1e-3 else "")
            bad(c["a"], msg.format(c["b"])); bad(c["b"], msg.format(c["a"]))

    # paths
    prow = []
    plist = sorted(A.get("paths", []), key=lambda p: p.get("order", 0))
    movers = {}                                                    # when each part goes in: its path's order, else the "order" map's, else 0 (first)
    import fnmatch
    for i in M:
        for pat, o in (A.get("order") or {}).items():
            if fnmatch.fnmatch(i, pat):
                movers[i] = o
    for p in plist:
        for r in [p["part"]] + list(p.get("with", [])):
            movers[r] = p.get("order", 0)
    for p in plist:
        nm = p["name"]; mv = p["part"]
        ride = [mv] + list(p.get("with", []))
        try:
            import manifold3d as mf
            if p.get("body") is not None:
                piece = PATHS.body(p)
            else:
                ms = [MM[r] for r in ride if r in MM]
                if p.get("min_body_mm3"):                       # leave out a part's small loose bodies (a screw drawn with it that goes in later)
                    ms = [b_ for m_ in ms for b_ in m_.decompose() if b_.volume() >= float(p["min_body_mm3"])]
                piece = mf.Manifold.batch_boolean(ms, mf.OpType.Add)
            Ts = PATHS.poses(p)
        except Exception as e:
            prow.append(dict(name=nm, part=mv, ok=False, why=f"sweep failed: {str(e)[:160]}")); bad(mv, f"path '{nm}': sweep failed"); continue
        g = FIT.gap(p.get("fit", "slide"))
        order = p.get("order", 0)
        placed = [i for i in M if i not in ride and i not in p.get("ignore", []) and movers.get(i, -1e9) < order]
        if p.get("against") is not None:                       # a step done in hand off the drone: only these are in its way
            placed = [i for i in p["against"] if i in M]
        into = set(p.get("into", []))
        Ps = [piece.transform(T[:3, :].tolist()) for T in Ts]
        BB = [np.asarray(q.bounding_box(), float) for q in Ps]
        lo = np.min([b_[:3] for b_ in BB], 0) - max(g, 0); hi = np.max([b_[3:] for b_ in BB], 0) + max(g, 0)
        blocked, tight = [], []
        for i in placed:
            bi = M[i].bounds
            if np.any(np.minimum(bi[1], hi) <= np.maximum(bi[0], lo)):
                continue
            rest = float((piece ^ MM[i]).volume())                 # what it shares with it seated is the interference check's
            worst, wk = 0.0, None
            near = [k for k, b_ in enumerate(BB) if not np.any(np.minimum(bi[1], b_[3:] + max(g, 0)) <= np.maximum(bi[0], b_[:3] - max(g, 0)))]
            for k in near:
                v = float((Ps[k] ^ MM[i]).volume()) - rest
                if v > worst:
                    worst, wk = v, k
            if worst > tol:
                blocked.append(dict(part=i, mm3=round(worst, 2), at_step=wk, of_steps=len(Ts), along=PATHS.step_label(p, wk),
                                    pieces=pieces(Ps[wk] ^ MM[i])))
                continue
            if g > 0 and rest <= 1e-3:                               # a joint touching at rest (a horn on its spline) has no gap to keep
                gmin, gk = 1e9, None
                for k in near:
                    if k == 0:
                        continue                                    # seated: resting on its stop is the seat, not the way in
                    d = float(Ps[k].min_gap(MM[i], g + 0.5))
                    if d < gmin:
                        gmin, gk = d, k
                if gk is not None and gmin < g:
                    tight.append(dict(part=i, gap_mm=round(gmin, 3), need_mm=g, at_step=gk, of_steps=len(Ts), along=PATHS.step_label(p, gk), into=i in into))
        ok = not blocked and not tight
        r = dict(name=nm, part=mv, order=order, fit=p.get("fit", "slide"), gap_mm=g, steps=len(Ts),
                 move={k: p[k] for k in ("turn", "moves", "out", "length_mm") if k in p}, into=sorted(into), blocked_by=blocked, tight=tight,
                 ok=bool(ok), why=p.get("why"))
        prow.append(r)
        for b_ in blocked:
            bad(b_["part"], f"blocks the path '{nm}' ({b_['mm3']:.1f} mm3 {b_['along']})"); bad(mv, f"path '{nm}' blocked by {b_['part']} ({b_['mm3']:.1f} mm3 {b_['along']})")
        for b_ in tight:
            bad(b_["part"], f"tight on the path '{nm}' (gap {b_['gap_mm']:.2f} of {g:.2f} {b_['along']})"); bad(mv, f"path '{nm}' tight on {b_['part']} (gap {b_['gap_mm']:.2f} of {g:.2f} {b_['along']})")

    ok = all(r["ok"] for r in pairs) and all(r["ok"] for r in mrows) and all(r["ok"] for r in prow)
    gate = not a.get("archived")                              # every live drone gates (design chat 15 Nov: project-wide); archived ones only report
    rep = dict(drone=drone, ok=bool(ok), gate=gate, fit_tol_mm3=tol, fit_table=FIT.table(), grow_mm=FIT.GROW, parts_checked=len(M), printed=sorted(printed),
               skipped=skipped, pairs=sorted(pairs, key=lambda r: (r["ok"], -r["mm3"])), mates=mrows, paths=prow,
               not_print_ready={k: v for k, v in sorted(blame.items())} if gate else {},
               would_block={} if gate else {k: v for k, v in sorted(blame.items())}, seconds=round(time.time() - t0, 1), placed=stage_of)
    back = {i: blame[i] for i, v in stage_of.items() if v["stage"] == "optimised" and i in blame}
    if back and _proto is None:                                    # the generated part does not go together: its prototype stays on the drone
        log(f"assembly {drone}: {', '.join(back)} fail the check placed; their prototypes stay")
        return run(drone, log, _proto=back)
    return rep


def summary(rep):
    L = [f"assembly {rep['drone']}: {'PASS' if rep['ok'] else 'FAIL'}{'' if rep.get('gate', True) else ' (report only: archived)'} "
         f"({rep['parts_checked']} parts, {rep['seconds']} s)"]
    for r in rep["pairs"]:
        if not r["ok"]:
            pc = r.get("pieces") or [{}]
            L.append(f"  CLASH {r['a']} x {r['b']}: {r['mm3']} mm3 (limit {r['limit_mm3']}), worst piece {pc[0].get('mm3')} mm3 at {pc[0].get('at')} "
                     f"depth {pc[0].get('depth_mm')}")
    for r in rep["mates"]:
        if not r["ok"]:
            L.append(f"  MATE  {r['a']} x {r['b']} ({r.get('fit')}): gap {r.get('gap_mm')} < {r.get('need_mm')}, overlap {r.get('overlap_mm3')} mm3")
    for r in rep["paths"]:
        if not r["ok"]:
            for b in r.get("blocked_by", []):
                pc = b["pieces"][0] if b["pieces"] else {}
                L.append(f"  PATH  '{r['name']}' ({r['part']}) blocked by {b['part']}: {b['mm3']} mm3 {b['along']}, at {pc.get('at')} depth {pc.get('depth_mm')}")
            for b in r.get("tight", []):
                L.append(f"  PATH  '{r['name']}' ({r['part']}) tight on {b['part']}: gap {b['gap_mm']} of {r['gap_mm']} {b['along']}")
            if r.get("why", "").startswith("sweep failed"):
                L.append(f"  PATH  '{r['name']}': {r['why']}")
    n = sum(1 for r in rep["paths"] if r["ok"])
    L.append(f"  paths {n} / {len(rep['paths'])} clear, mates {sum(1 for r in rep['mates'] if r['ok'])} / {len(rep['mates'])}, "
             f"not print-ready: {', '.join(rep['not_print_ready']) or 'none'}")
    for s in rep["skipped"]:
        L.append(f"  skipped {s['part']}: {s['why']}")
    return "\n".join(L)


def write(rep):
    out = os.path.join(ROOT, "systems", f"{rep['drone']}.assembly_check.json")
    old = json.load(open(out)) if os.path.exists(out) else None
    strip = lambda r: {k: v for k, v in (r or {}).items() if k != "seconds"}
    if strip(old) != strip(rep):
        json.dump(rep, open(out, "w"), indent=1)
    for pid in rep["printed"]:
        src = (rep.get("placed") or {}).get(pid, {}).get("src", pid)
        rj = print_mesh.record(src)                            # result.json (generated), print.json (drawn: tools/make_parts.py)
        if not rj or not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        v = dict(drone=rep["drone"], ok=pid not in rep["not_print_ready"], why=rep["not_print_ready"].get(pid, []))
        if r.get("assembly") != v:
            r["assembly"] = v
            json.dump(r, open(rj, "w"), indent=1)


def drones():
    out = []
    for f in sorted(os.listdir(os.path.join(ROOT, "systems"))):
        if f.endswith(".assembly.json"):
            a = json.load(open(os.path.join(ROOT, "systems", f)))
            if not a.get("archived"):   # check_only files (a bench kit) too
                out.append(f[:-len(".assembly.json")])
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    w = "--write" in argv; q = "--quiet" in argv
    names = [x for x in argv if not x.startswith("--")]
    if "--all" in argv:
        names = drones()
    fail = False
    for d in names:
        rep = run(d)
        print(summary(rep))
        if w:
            write(rep)
        fail |= rep.get("gate", True) and not rep["ok"]
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
