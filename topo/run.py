"""Run a part end to end: coarse warm start -> fine layout -> build -> checks -> files, with checkpoints.

    python3 -m topo.run <part> [--h 1.0] [--gen 12] [--min-member 3.2] [--out runs/<part>/topo-v1] [--no-coarse]
                               [--git] [--backend auto|direct|mg] [--time-limit 19800] [--set KEY=VALUE ...]
(run from nox/, i.e. `cd nox && python3 -m topo.run holder`). <part> is a name from topo.demos.PARTS.

Speed plan for a 4 vCPU / 16 GB cloud box (measured numbers go in the run log):
  * stage 1 on a 2h grid (8x fewer cells) for beta 1..4, where most iterations are spent;
  * stage 2 on the h grid from beta 4, warm-started from stage 1;
  * Pardiso for small active systems, multigrid CG beyond (topo.fem.Solver picks by RAM);
  * time-aware continuation: past 75 % of the time limit the beta steps are taken as soon as allowed.
"""
import os
import sys
import json
import time
import argparse
import subprocess
from dataclasses import replace

from .layout import Layout, Settings, gate_member_mm
from .build import build_and_check
from .audit import SetupError

MAX_CELLS = 300_000      # the grid the GitHub runner solves direct (16 GB, direct estimate 6 (n/4.5e5)^(4/3) GB): over it the multigrid
                         # fallback stalls (the arms at 1.0 M cells, 28-29 Oct: 300 CG iterations without converging, no attempt did one iteration)


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, s):
        s = str(s)
        print(s[:2000], flush=True)
        self.f.write(s + "\n")


def git_checkpoint(out_dir, log):
    def hook(it):
        try:
            root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                                  cwd=out_dir).stdout.strip()
            if not root:
                return
            subprocess.run(["git", "add", os.path.relpath(out_dir, root)], cwd=root, capture_output=True)
            subprocess.run(["git", "commit", "-q", "-m", f"ckpt {it}"], cwd=root, capture_output=True)
            r = subprocess.run(["git", "push", "-q"], cwd=root, capture_output=True, text=True)
            if r.returncode:
                log(f"git push failed: {r.stderr.strip()[:200]}")
        except Exception as e:                                   # a checkpoint push must never stop the run
            log(f"git checkpoint failed: {e}")
    return hook


def peak_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return None


def optimise(setup, h, gen_cm3, out_dir, name, settings=None, coarse=True, git=False, log=None, maxit=400,
             extra=None):
    os.makedirs(out_dir, exist_ok=True)
    log = log or Tee(os.path.join(out_dir, "log.txt"))
    s = settings or Settings()
    t0 = time.time()
    hook = git_checkpoint(out_dir, log) if git else None
    ck = os.path.join(out_dir, "ck.npz")
    log(f"=== {name}: h {h}, generated {gen_cm3} cm3, min member {s.min_member_mm} mm, cores {os.cpu_count()}")
    part0 = setup(h)
    if part0.nel > MAX_CELLS:
        raise SetupError(f"{name}: {part0.nel:,} cells at h {h} (limit {MAX_CELLS:,}): coarsen h (cells scale with 1/h^3) or trim the "
                         f"envelope (the audit's envelope_waste says where); the multigrid fallback does not converge on grids this size.")
    if s.gate_member:
        s = replace(s, min_member_mm=round(gate_member_mm(part0, s.min_member_mm, log=log), 2))
    from .audit import audit
    rep_audit = audit(part0, s.min_member_mm, log=log, hard=s.audit == "raise", floor_mm=settings.min_member_mm if settings else Settings.min_member_mm)
    del part0
    xh_c = part_c = None
    if coarse and not os.path.exists(ck):
        part_c = setup(2.0 * h)
        sc = replace(s, reach_check="off", geometric=False, max_stage=25)
        Lc = Layout(part_c, gen_cm3=gen_cm3, settings=sc, log=log)
        ckc = os.path.join(out_dir, "coarse_ck.npz")
        Lc.run(maxit=maxit, checkpoint=ckc, resume=ckc, beta_stop=2, on_checkpoint=hook)
        xh_c = Lc.x_hat()
        log(f"coarse stage done: {len(Lc.hist)} iterations, {time.time() - t0:.0f} s, peak {peak_gb()} GB")
        del Lc
    part = setup(h)
    L = Layout(part, gen_cm3=gen_cm3, settings=s, log=log)
    if xh_c is not None:
        L.init_from(xh_c, part_c)
    L.run(maxit=maxit, checkpoint=ck, resume=ck, beta_start=2 if coarse else 0, on_checkpoint=hook)
    L.save(os.path.join(out_dir, "density.npz"))
    t_layout = time.time() - t0
    log(f"layout done: {t_layout:.0f} s, peak {peak_gb()} GB")
    ex = dict(layout_s=round(t_layout), peak_gb=peak_gb(), gen_cm3=gen_cm3, h=h, audit=rep_audit)
    if extra:
        ex.update(extra)
    res, P = build_and_check(L, out_dir, name, log=log, extra=ex)
    res["total_s"] = round(time.time() - t0)
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    if hook:
        hook("final")
    return res


def rebuild(setup, h, gen_cm3, src_dir, out_dir, name, settings=None, log=None, extra=None):
    """build and check again from a finished layout (src_dir/density.npz), e.g. after a build-stage change. The
    layout is not rerun; result.json names the source run."""
    import shutil
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    log = log or Tee(os.path.join(out_dir, "log.txt"))
    s = settings or Settings()
    t0 = time.time()
    log(f"=== {name}: rebuild from {src_dir}/density.npz (layout not rerun)")
    if s.gate_member:
        s = replace(s, min_member_mm=round(gate_member_mm(setup(h), s.min_member_mm, log=log), 2))
    part = setup(h)
    L = Layout(part, gen_cm3=gen_cm3, settings=s, log=log)
    d = np.load(os.path.join(src_dir, "density.npz"))
    if tuple(d["shape"]) != tuple(part.shape) or not np.array_equal(d["design"], L.design):
        raise ValueError("rebuild: the source layout's grid or design mask differs from this setup")
    L.x = d["x_full"][L.design]
    L.hist = [tuple(r) for r in d["hist"]]
    L.result = dict(x=L.x, xt=d["xt"], rho_n=d["rho_n"], rho_e=d["rho_e"], beta=float(d["beta"]))
    shutil.copy(os.path.join(src_dir, "density.npz"), os.path.join(out_dir, "density.npz"))
    src = {}
    try:
        with open(os.path.join(src_dir, "result.json")) as fh:
            r0 = json.load(fh)
        src = dict(layout_s=r0.get("layout_s"), peak_gb=r0.get("peak_gb"))
    except OSError:
        pass
    ex = dict(gen_cm3=gen_cm3, h=h, rebuilt_from=os.path.basename(os.path.normpath(src_dir)), **src)
    if extra:
        ex.update(extra)
    res, P = build_and_check(L, out_dir, name, log=log, extra=ex)
    res["total_s"] = round(time.time() - t0)
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    return res


def parse_sets(items):
    """['pmean=8', 'eta=0.7,0.5,0.3', 'geometric=true'] -> typed Settings overrides (types from the defaults)."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(Settings)}
    out = {}
    for kv in items:
        if "=" not in kv:
            raise SystemExit(f"--set needs KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip().replace("-", "_")
        if k not in names:
            raise SystemExit(f"unknown setting {k!r}; settings: {', '.join(sorted(names))}")
        d = getattr(Settings, k)
        if isinstance(d, bool):
            out[k] = v.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(d, tuple):
            out[k] = tuple(type(d[0])(float(t)) if d else float(t) for t in v.split(","))
        elif isinstance(d, int):
            out[k] = int(float(v))
        elif isinstance(d, float):
            out[k] = float(v)
        else:
            out[k] = v.strip()
    return out


def load_part_dir(path):
    """parts/<name>/part.py -> spec dict. The file defines setup(h) -> Part and SPEC = dict(h=..., gen_cm3=...)."""
    import importlib.util
    d = os.path.abspath(path)
    f = d if d.endswith(".py") else os.path.join(d, "part.py")
    name = os.path.basename(os.path.dirname(f))
    m = importlib.util.spec_from_file_location(f"part_{name}", f)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    spec = dict(mod.SPEC)
    spec.setdefault("setup", mod.setup)
    spec.setdefault("name", name)
    spec.setdefault("out", os.path.dirname(f))
    return spec


def main(argv=None):
    from .demos import PARTS
    ap = argparse.ArgumentParser()
    ap.add_argument("part", help="a demo name (%s) or a path to parts/<name>/ or its part.py" % ", ".join(sorted(PARTS)))
    ap.add_argument("--h", type=float)
    ap.add_argument("--gen", type=float)
    ap.add_argument("--min-member", type=float, default=Settings.min_member_mm)
    ap.add_argument("--out")
    ap.add_argument("--no-coarse", action="store_true")
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--time-limit", type=float, default=5.5 * 3600)
    ap.add_argument("--maxit", type=int, default=400)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any Settings field, e.g. --set pmean=8 --set eta=0.7,0.5,0.3 (repeatable)")
    ap.add_argument("--rebuild-from", metavar="RUN_DIR", help="build and check again from RUN_DIR/density.npz")
    a = ap.parse_args(argv)
    if a.part in PARTS:
        spec = PARTS[a.part]()
        name = a.part
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", a.part, "topo-v1")
    else:
        spec = load_part_dir(a.part)
        name = spec["name"]
        out = a.out or spec["out"]                      # results land beside part.py
    h = a.h or spec["h"]
    gen = a.gen or spec["gen_cm3"]
    sets = dict(spec.get("settings", {}))
    sets.update(parse_sets(a.set))
    s = Settings(min_member_mm=a.min_member, backend=a.backend, time_limit_s=a.time_limit, **sets)
    if a.rebuild_from:
        res = rebuild(spec["setup"], h, gen, os.path.abspath(a.rebuild_from), os.path.abspath(out), name,
                      settings=s, extra=dict(baseline=spec.get("baseline")))
    else:
        res = optimise(spec["setup"], h, gen, os.path.abspath(out), name, settings=s, coarse=not a.no_coarse,
                       git=a.git, maxit=a.maxit, extra=dict(baseline=spec.get("baseline")))
    if a.part not in PARTS:                             # parts/<name>/: the result is always part.stl beside part.py
        import shutil
        src = os.path.join(out, f"{name}.stl")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, "part.stl"))
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        from part_sha import part_sha, lib_sha          # the site builder compares these with part.py and its libs to flag stale results
        rj = os.path.join(out, "result.json")
        with open(rj) as fh:
            r = json.load(fh)
        r["part_sha"] = part_sha(os.path.join(out, "part.py"))
        r["lib_sha"] = lib_sha(os.path.join(out, "part.py"))
        # strength gate: SPEC["max_worst"] (default 0.5). Above it the part is failed like a solid-check fail.
        mw = float(spec.get("max_worst", 0.5))
        r["max_worst"] = mw
        r["strength_ok"] = bool(res["strength"]["worst"] is not None and res["strength"]["worst"] <= mw)
        with open(rj, "w") as fh:
            json.dump(r, fh, indent=1, default=str)
        if not r["strength_ok"]:
            print(f"STRENGTH FAIL: worst index {res['strength']['worst']} > max_worst {mw}")
            return 1
    print(json.dumps(dict(name=res["name"], solid_ok=res["solid_ok"], mass_g=res["mass_g"],
                          worst=res["strength"]["worst"], total_s=res["total_s"]), indent=1))
    return 0 if res["solid_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
