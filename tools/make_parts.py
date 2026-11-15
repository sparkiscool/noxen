"""Drawn parts (parts/<p>/make.py, no topology run) as first-class parts (15 Nov): CI runs every changed make.py, solid-checks its print
mesh, puts it on the `meshes` release (tools/print_mesh.py) and the assembly check (tools/assembly_check.py) runs on the result, as
for a generated part. A drawn part is the PROTOTYPE stage; a generated part (part.py) the OPTIMISED stage. The page (tools/build_site.py)
shows each part's stage and whether it is print-ready (its own gates and the assembly check pass); a generated part replaces the
prototype it was drawn from (its assembly line's "prototype": "<folder of the make.py>") only when all its gates pass.

What a make.py run is made of goes to the part's print record (print.json; result.json where the make writes one) as "make":
  sha    {file: sha12} make.py and every repo file the run read (opened under the repo, tools/part_sha.py's audit hook), less what
         it wrote; a change to any of them makes the drawn part stale (CI runs it again)
  solid  the solid check (tools/solid_check.py) on its print mesh: print/<p>.stl when the make writes one, else part.stl; a generator
         with neither (latch7) is held to every print file its run wrote
  at, commit

    python tools/make_parts.py --affected < changed-files     the drawn parts a push reruns (git diff --name-only on stdin), and every
                                                              drawn part whose recorded reads changed since its run
    python tools/make_parts.py run <p> ... | --all            run each make.py, solid-check, record (exit 1 if any fails)
    python tools/make_parts.py check <p> ... | --all          solid-check the committed print mesh only (no run), record
    python tools/make_parts.py status [<p> ...]               stage, status and solid verdict per drawn part
"""
import glob, hashlib, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import print_mesh                                                                   # noqa: E402

OUTPUTS = {"part.stl", "print.stl", "print.json", "preview.png", "feedback.md", "setup.json", "log.txt", "result.json"}
TIMEOUT_S = 1800                                                                    # one make.py in CI


def names():
    return sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(ROOT, "parts", "*", "make.py")))


def _h(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12] if os.path.isfile(path) else "gone"


def rec_path(p):
    return print_mesh.record(p) or os.path.join(ROOT, "parts", p, "print.json")


def record(p):
    f = rec_path(p)
    try:
        return json.load(open(f)) if os.path.exists(f) else {}
    except ValueError:
        return {}


def _write(p, key, val):
    f = rec_path(p); r = record(p); r[key] = val
    json.dump(r, open(f, "w"), indent=1)


_IMP = re.compile(r"^\s*(?:from\s+(?:parts\.)?(\w+)\s+import|import\s+(?:parts\.)?(\w+))", re.M)
_MAKE = re.compile(r"[\"']parts[\"']\s*,\s*[\"'](\w+)[\"']\s*,\s*[\"']make\.py[\"']")


def fallback_files(p):
    """the files a drawn part with no recorded run is made of: make.py, its folder's own inputs, the parts/*.py libs and other parts'
    make.py it imports (transitively)"""
    d = os.path.join(ROOT, "parts", p)
    out = {os.path.join(d, "make.py")}
    for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
        if os.path.isfile(f) and os.path.basename(f) not in OUTPUTS and not f.endswith(".stl") and "__pycache__" not in f:
            out.add(f)
    todo = [f for f in out if f.endswith(".py")]
    while todo:
        src = open(todo.pop()).read()
        for a, b in _IMP.findall(src):
            lib = os.path.join(ROOT, "parts", f"{a or b}.py")
            if os.path.exists(lib) and lib not in out:
                out.add(lib); todo.append(lib)
        for m in _MAKE.findall(src):
            mk = os.path.join(ROOT, "parts", m, "make.py")
            if os.path.exists(mk) and mk not in out:
                out.add(mk); todo.append(mk)
    return sorted(os.path.relpath(f, ROOT).replace(os.sep, "/") for f in out)


def files(p):
    m = record(p).get("make") or {}
    return sorted(m["sha"]) if m.get("sha") else fallback_files(p)


def stale(p):
    """True: a file the drawn part is made of changed since its recorded run (or it has none); False: current"""
    m = record(p).get("make") or {}
    if not m.get("sha"):
        return True
    return any(_h(os.path.join(ROOT, f)) != s for f, s in m["sha"].items() if f not in ("tools/make_parts.py", "tools/print_mesh.py"))


def affected(changed):
    ch = {c.strip().replace(os.sep, "/") for c in changed if c.strip()}
    return [p for p in names() if os.path.exists(os.path.join(ROOT, "parts", p, "make.py")) and ch & set(files(p))]


def print_file(p):
    """the part's print mesh: print/<p>.stl as the make wrote it (bed pose), else part.stl"""
    for f in (os.path.join(ROOT, "print", f"{p}.stl"), os.path.join(ROOT, "parts", p, "part.stl")):
        if os.path.exists(f):
            return f
    return None


def check(p, path=None, log=print):
    """the solid check on the part's print mesh -> the record's make.solid; True when it passes"""
    import trimesh
    from solid_check import solid_check
    path = path or print_file(p)
    if path is None:                                              # a generator: its last run's print files, or nothing to check until it runs
        fs = [r["file"] for r in ((record(p).get("make") or {}).get("solid") or {}).get("files", []) if os.path.exists(os.path.join(ROOT, r["file"]))]
        if fs:
            return check_set(p, fs, log=log)
        log(f"make_parts: {p}: no print mesh of its own (a generator): made, its print files are checked; nothing to check now")
        return True
    try:
        m = trimesh.load(path, force="mesh")
        ok, info = solid_check(m)
        keep = {k: v for k, v in (info or {}).items() if isinstance(v, (int, float, str, bool, type(None))) and k not in ("ok", "file", "sha256", "cm3")}
        sol = dict(ok=bool(ok), file=os.path.relpath(path, ROOT), sha256=print_mesh.sha256(path), cm3=round(abs(m.volume) / 1000, 3), **keep)
    except Exception as e:
        sol = dict(ok=False, file=os.path.relpath(path, ROOT), why=f"the check did not run: {str(e)[:200]}")
    m = record(p).get("make") or {}
    m["solid"] = sol
    _write(p, "make", m)
    log(f"make_parts: {p}: solid check {'PASS' if sol['ok'] else 'FAIL'} on {sol.get('file')}" + (f" ({sol['why']})" if sol.get("why") else ""))
    return sol["ok"]


def _changed_now():
    r = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=ROOT, capture_output=True, text=True)
    return {ln[3:].strip().strip('"') for ln in r.stdout.splitlines() if len(ln) > 3}


def run(p, log=print):
    """python parts/<p>/make.py with its reads recorded, then the solid check; the record's make block"""
    d = os.path.join(ROOT, "parts", p)
    before = _changed_now(); t0 = time.time()
    out = os.path.join(d, ".make_reads.json")
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_exec", p, out], cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT_S)
        rc, tail = r.returncode, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        rc, tail = -1, f"timed out after {TIMEOUT_S} s"
    reads = json.load(open(out)) if os.path.exists(out) else []
    if os.path.exists(out):
        os.remove(out)
    wrote = _changed_now() - before
    own = f"parts/{p}/make.py"
    reads = sorted(({own} | set(reads)) - (wrote - {own}))
    reads = [f for f in reads if not f.startswith("print/") and os.path.basename(f) not in OUTPUTS and not f.endswith(".stl")
             and f not in ("tools/make_parts.py", "tools/print_mesh.py")]                 # this runner's own modules, not the part's
    m = dict(sha={f: _h(os.path.join(ROOT, f)) for f in reads}, rc=rc, s=round(time.time() - t0, 1), at=time.strftime("%Y-%m-%d %H:%M"),
             commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    if rc != 0:
        m["solid"] = dict(ok=False, why=f"make.py failed (exit {rc}): {tail.strip().splitlines()[-1][:200] if tail.strip() else ''}")
        _write(p, "make", m)
        log(f"make_parts: {p}: make.py FAILED (exit {rc}) in {m['s']} s\n{tail}")
        return False
    _write(p, "make", m)
    log(f"make_parts: {p}: made in {m['s']} s ({len(reads)} files read, {len(wrote)} written)")
    if print_file(p) is None:                                    # a generator (latch7: its parts' folders and print files, none of its own)
        made = sorted(f for f in wrote if f.startswith("print/") and f.endswith(".stl"))
        return check_set(p, made, log=log)
    return check(p, log=log)


def check_set(p, files_, log=print):
    """a generator's verdict: the solid check on every print file its run wrote (make.solid.files); it passes when all do"""
    import trimesh
    from solid_check import solid_check
    rows = []
    for f in files_:
        try:
            ok, _ = solid_check(trimesh.load(os.path.join(ROOT, f), force="mesh"))
        except Exception as e:
            ok = False
        rows.append(dict(file=f, ok=bool(ok)))
    bad = [r["file"] for r in rows if not r["ok"]]
    sol = dict(ok=not bad, files=rows, why=None if rows and not bad else                  # nothing printed of its own this run: nothing to fail
               ("a generator: no print file of its own in this run (its parts carry their own records)" if not rows else f"solid check FAIL on {', '.join(bad[:6])}"))
    m = record(p).get("make") or {}; m["solid"] = sol; _write(p, "make", m)
    log(f"make_parts: {p}: {len(rows)} print files, solid check {'PASS' if sol['ok'] else 'FAIL'}" + (f" ({sol['why']})" if sol["why"] else ""))
    return sol["ok"]


def _exec(p, out):
    """inside the subprocess: run the make.py as __main__ with the repo reads recorded (tools/part_sha.py record)"""
    import runpy
    import part_sha
    sys.argv = [os.path.join(ROOT, "parts", p, "make.py")]
    sys.path.insert(0, os.path.join(ROOT, "parts", p))
    reads = []

    def go():
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as e:                                  # a make that ends with sys.exit(0) has run (its reads are kept)
            if e.code not in (0, None):
                raise
    try:
        _, reads = part_sha.record(go)
    finally:
        json.dump(sorted(set(reads)), open(out, "w"))


def state(p):
    """what the page and the assembly check say of a drawn part: stage 'prototype', status drawn / failed / stale / unchecked, solid_ok"""
    m = record(p).get("make") or {}
    sol = m.get("solid")
    if sol is None:
        st = "unchecked"
    elif m.get("sha") and stale(p):
        st = "stale"
    else:
        st = "drawn" if sol.get("ok") else "failed"
    return dict(stage="prototype", status=st, solid_ok=None if sol is None else bool(sol.get("ok")), why=(sol or {}).get("why"))



def part_state(src):
    """one rule for the page and the assembly check: stage (optimised: generated by part.py; prototype: drawn by make.py; placed: neither,
    a bought or hand-made mesh), status (generated: done / failed / stale / pending; drawn: drawn / failed / stale / unchecked; placed)
    and own_ok: its own gates pass on the current setup (generated: solid and strength; drawn: the solid check on its print mesh)"""
    d = os.path.join(ROOT, "parts", src)
    py = os.path.join(d, "part.py")
    if os.path.exists(py):
        rj = os.path.join(d, "result.json")
        if not os.path.exists(rj):
            return dict(stage="optimised", status="pending", own_ok=False, result=None)
        from part_sha import stale as _stale
        r = json.load(open(rj))
        if _stale(py, r):
            return dict(stage="optimised", status="stale", own_ok=False, result=r)
        ok = bool(r.get("solid_ok") and r.get("strength_ok", True))
        return dict(stage="optimised", status="done" if ok else "failed", own_ok=ok, result=r)
    if os.path.exists(os.path.join(d, "make.py")):
        s = state(src)
        return dict(stage="prototype", status=s["status"], own_ok=s["status"] == "drawn", why=s.get("why"), result=None)
    return dict(stage="placed", status="placed", own_ok=None, result=None)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "_exec":
        _exec(a[1], a[2]); sys.exit(0)
    if a[0] == "--affected":                                     # and every drawn part whose recorded reads changed since its run (a push
        ch = affected(sys.stdin.read().split())                  # whose meshes run was cancelled in the queue still gets its parts made)
        print(" ".join(sorted(set(ch) | {p for p in names() if (record(p).get("make") or {}).get("sha") and stale(p)})))
        sys.exit(0)
    cmd, ps = a[0], a[1:]
    if ps == ["--all"] or (cmd == "status" and not ps):
        ps = names()
    if cmd == "status":
        for p in ps:
            print(p, json.dumps(state(p)))
        sys.exit(0)
    bad = [p for p in ps if not (run(p) if cmd == "run" else check(p))]
    print(f"make_parts: {len(ps) - len(bad)} / {len(ps)} pass" + (f"; FAIL: {' '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)
