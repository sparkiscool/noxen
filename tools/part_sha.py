"""What a result was made from. setup_sha(py): the sha of the setup part.py builds (topo/run.py design_hash: voxel masks,
supports, loads, SPEC, thin_ok, solid_ok), cached in parts/<name>/setup.json against the code (part_sha, lib_sha) and the data it reads
(data_sha: every repo file the setup opened while it was built, recorded in setup.json "reads"). A result is stale when its setup_sha differs from that
(30 Oct, Spark: a two-line validation in cage_node_lib changed the lib's file hash and eight node results read as stale with
nothing solved differently; staleness is by setup, not by file). part_sha / lib_sha (the file hashes) key the cache and judge
results from before setup_sha was recorded.

    python tools/part_sha.py parts/<name>/part.py      -> "<part_sha> <lib_sha> <setup_sha>"
    python tools/part_sha.py --refresh [<part> ...]    -> refresh the setup.json caches (CI does it with every results commit)
    python tools/part_sha.py --files <part> ...        -> the repo files the part's identity is made of
    python tools/part_sha.py --affected <file> ...     -> the parts a change to these files reruns (parts.yml on a push)
"""
import glob, hashlib, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMPORT = re.compile(r"^\s*(?:from\s+parts\.(\w+)\s+import|import\s+parts\.(\w+))", re.M)


def _h(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def libs(py):
    """the parts/<lib>.py files py imports, transitively, sorted"""
    seen, todo = [], [py]
    while todo:
        f = todo.pop()
        for a, b in IMPORT.findall(open(f).read()):
            lib = os.path.join(ROOT, "parts", f"{a or b}.py")
            if os.path.exists(lib) and lib not in seen:
                seen.append(lib); todo.append(lib)
    return sorted(seen)


def part_sha(py):
    return _h(py)


def engine_sha():
    """the engine itself (topo/*.py): a checkpoint from another engine is not resumed either"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fs = sorted(f for f in os.listdir(os.path.join(root, "topo")) if f.endswith(".py"))
    return hashlib.sha256("".join(_h(os.path.join(root, "topo", f)) for f in fs).encode()).hexdigest()[:12]


def lib_sha(py):
    ls = libs(py)
    return hashlib.sha256("".join(_h(l) for l in ls).encode()).hexdigest()[:12] if ls else ""


SKIP_DIRS = ("topo", ".git", "site", "__pycache__")                 # the engine is engine_sha; site/ is built from results
OUTPUTS = ("result.json", "setup.json", "part.stl", "preview.png", "log.txt", "SUPERSEDED")
_REC = None                                                          # the reads being recorded (a set), or None


def _audit(ev, args):
    """sys.addaudithook: every file opened (open, io.open_code for imports, os.open) under ROOT while a setup is built; no effect when
    nothing is recording. Folder listings are left out: the import system lists every sys.path folder, so a new file anywhere in
    tools/ would be a new identity"""
    if _REC is None or ev != "open" or not args:
        return
    f = args[0]
    if not isinstance(f, (str, bytes, os.PathLike)):
        return
    if ev == "open" and len(args) > 1 and isinstance(args[1], str) and any(c in args[1] for c in "wax+"):
        return                                                       # a file the setup writes is its output, not its input
    try:
        f = os.path.abspath(os.fsdecode(f))
    except Exception:
        return
    rel = os.path.relpath(f, ROOT)
    if rel.startswith("..") or rel.split(os.sep)[0] in SKIP_DIRS or "__pycache__" in rel.split(os.sep):
        return
    if os.path.basename(f) in OUTPUTS or re.match(r"ck.*\.npz$", os.path.basename(f)) or rel == os.path.join("tools", "part_sha.py") \
            or rel == os.path.join("systems", "runtimes.json"):
        return
    if os.path.isfile(f):
        _REC.add(rel.replace(os.sep, "/"))


sys.addaudithook(_audit)


def record(fn):
    """fn() -> (its value, the sorted repo files it read). Modules from the repo are imported afresh inside
    (and the caller's put back after), so a tools/ or parts/ module the caller had already imported is read, and recorded, again"""
    global _REC
    mine = lambda m: (getattr(m, "__file__", None) or "").startswith(ROOT + os.sep) and \
        os.path.relpath(m.__file__, ROOT).split(os.sep)[0] not in ("topo",) and not m.__file__.endswith("part_sha.py")
    saved = {k: m for k, m in list(sys.modules.items()) if mine(m)}
    for k in saved:
        del sys.modules[k]
    _REC = set()
    try:
        v = fn()
        for m in list(sys.modules.values()):                 # an import served from __pycache__ opens no source: the modules it loaded
            if mine(m):
                _audit("open", (m.__file__, "r"))
        return v, sorted(_REC)
    finally:
        _REC = None
        for k, m in list(sys.modules.items()):
            if mine(m):
                del sys.modules[k]
        sys.modules.update(saved)


def _cached(py):
    import json
    try:
        return json.load(open(os.path.join(os.path.dirname(os.path.abspath(py)), "setup.json")))
    except (OSError, ValueError):
        return {}


def data_files(py, reads=None):
    """the data a setup reads besides part.py: the files it opened while it was built (setup.json "reads", recorded by
    setup_sha: systems/*.json, tools/*.py and their data, parts/ libs, inputs/; 10 Nov: the Scout v4 head-first load in
    systems/scout_v4.head_load.json and tools/scout_v4_shape.py were outside the old key), plus the part's inputs/. A part with
    no recorded reads yet falls back to every systems/*.json but runtimes.json, every tools/*.py and the inputs/ files"""
    d = os.path.dirname(os.path.abspath(py))
    reads = _cached(py).get("reads") if reads is None else reads
    if reads is None:
        fs = [f for f in glob.glob(os.path.join(ROOT, "systems", "*.json")) if os.path.basename(f) != "runtimes.json"]
        fs += [f for f in glob.glob(os.path.join(ROOT, "tools", "*.py")) if os.path.basename(f) != "part_sha.py"]
    else:
        fs = [os.path.join(ROOT, r) for r in reads]
    fs += [f for f in glob.glob(os.path.join(d, "inputs", "**", "*"), recursive=True) if os.path.isfile(f)]
    py = os.path.abspath(py)
    return sorted(set(fs) - {py})


def _hd(f):
    return _h(f) if os.path.isfile(f) else "gone"             # a read file deleted is a change too


def data_sha(py, reads=None):
    fs = data_files(py, reads)
    return hashlib.sha256("".join(os.path.relpath(f, ROOT) + _hd(f) for f in fs).encode()).hexdigest()[:12] if fs else ""


def files(py):
    """the repo files a part's identity is made of (relative): part.py, its libs and data_files; a change to any of them can change the setup"""
    fs = [os.path.abspath(py)] + libs(py) + data_files(py)
    return sorted({os.path.relpath(f, ROOT).replace(os.sep, "/") for f in fs})


def affected(changed, pys=None):
    """the parts a change to the repo files `changed` (relative paths, as git diff --name-only) reruns: part.py or inputs/ changed
    (always, as before), or a file of its identity (libs, recorded reads) changed and the setup it builds now differs from its
    result's (stale): a data or lib edit that moves no geometry runs nothing"""
    import json
    ch = {c.strip().replace(os.sep, "/") for c in changed if c.strip()}
    out = []
    for py in pys or sorted(glob.glob(os.path.join(ROOT, "parts", "*", "part.py"))):
        name = os.path.basename(os.path.dirname(py)); rel = f"parts/{name}/"
        if rel + "part.py" in ch or any(c.startswith(rel + "inputs/") for c in ch):
            out.append(name); continue
        ids = files(py)
        if not ch & set(ids):
            continue
        rj = os.path.join(os.path.dirname(py), "result.json")
        try:
            if not os.path.exists(rj) or stale(py, json.load(open(rj))):
                out.append(name)
        except Exception as e:                               # a setup that no longer builds: the run says why
            print(f"{name}: setup does not build ({str(e)[:120]}), run it", file=sys.stderr); out.append(name)
    return out


def setup_sha(py, log=None, refresh=False):
    """the sha of the setup part.py builds now (topo/run.py design_hash: masks, supports, loads, SPEC, thin_ok, solid_ok), cached in
    parts/<name>/setup.json against everything the setup reads: part_sha + lib_sha (the code) + data_sha (the files it read,
    recorded as setup.json "reads", and inputs/: 30 Oct, Spark lowered four hub ends in overseer.cage.json and nodes 3 and 4 read as current from a cache keyed on
    the code alone), so setup(h) (seconds to a minute) runs only when one of those changed. CI commits the cache with the results
    (parts.yml); a session or the page builder recomputes a miss; refresh=True recomputes regardless."""
    import json
    d = os.path.dirname(os.path.abspath(py)); cache = os.path.join(d, "setup.json")
    ps, ls = part_sha(py), lib_sha(py)
    if not refresh:
        c = _cached(py)
        if c.get("part_sha") == ps and c.get("lib_sha") == ls and c.get("data_sha") == data_sha(py) and c.get("setup_sha") and "reads" in c:
            return c["setup_sha"]
    sys.path.insert(0, ROOT)
    from topo.run import load_part_dir, design_hash

    def build():
        spec = load_part_dir(d)
        h, gen = spec["h"], spec["gen_cm3"]
        return design_hash(spec["setup"](h), h, gen, spec)
    sh, reads = record(build)
    json.dump(dict(part_sha=ps, lib_sha=ls, data_sha=data_sha(py, reads), setup_sha=sh, reads=reads), open(cache, "w"), indent=1)
    if log:
        log(f"{os.path.basename(d)}: setup {sh}")
    return sh


def stale(py, r):
    """r = result.json dict: made from another setup than the one part.py builds now (setup_sha); a result from before setup_sha
    was recorded falls back to the file rule (older part.py, or older lib when it recorded one)"""
    if r.get("setup_sha"):
        return r["setup_sha"] != setup_sha(py)
    ls = lib_sha(py)                                     # a result with no lib_sha from a part that imports a lib is of unknown make: stale
    return r.get("part_sha") != part_sha(py) or (bool(ls) and r.get("lib_sha") != ls)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--files"]:                     # the identity files of the named parts (the watchdog's and the dispatch's diff)
        for p in sys.argv[2:]:
            print("\n".join(files(os.path.join(ROOT, "parts", p, "part.py"))))
    elif sys.argv[1:2] == ["--affected"]:                # the parts a push of these changed files reruns (parts.yml)
        print(" ".join(affected(sys.argv[2:] or sys.stdin.read().split())))
    elif sys.argv[1:2] == ["--refresh"]:                   # recompute the setup.json caches: the named parts, or every part with a part.py
        pys = [os.path.join(ROOT, "parts", p, "part.py") for p in sys.argv[2:]] or sorted(glob.glob(os.path.join(ROOT, "parts", "*", "part.py")))
        for py in pys:
            try:
                setup_sha(py, log=print, refresh=True)
            except Exception as e:                       # a setup that does not build has no sha; the run will say why
                print(f"{os.path.basename(os.path.dirname(py))}: no setup sha ({str(e)[:120]})")
    else:
        print(part_sha(sys.argv[1]), lib_sha(sys.argv[1]), setup_sha(sys.argv[1]))


def recut_of(old, new):
    """True when the tree's result `old` is a recut (tools/recut.py) of the run result `new` itself: same design and the same
    engine run (part_sha and total_s, which a recut keeps). A run's batch commit must not put `new` back over its own recut
    (10 Nov: the 00:41 and 04:36 batch commits undid the rear arm's and the front pad's solid-pad fills); a rerun of the
    same setup has its own total_s and still lands."""
    return bool(old.get("recut")) and old.get("part_sha") == new.get("part_sha") and old.get("total_s") == new.get("total_s")
