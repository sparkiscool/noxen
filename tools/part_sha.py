"""What a result was made from. setup_sha(py): the sha of the setup part.py builds (topo/run.py design_hash: voxel masks,
supports, loads, SPEC, thin_ok), cached in parts/<name>/setup.json against the code (part_sha, lib_sha) and the data it reads
(data_sha: systems/*.json, inputs/). A result is stale when its setup_sha differs from that
(30 Oct, Spark: a two-line validation in cage_node_lib changed the lib's file hash and eight node results read as stale with
nothing solved differently; staleness is by setup, not by file). part_sha / lib_sha (the file hashes) key the cache and judge
results from before setup_sha was recorded.

    python tools/part_sha.py parts/<name>/part.py      -> "<part_sha> <lib_sha> <setup_sha>"
    python tools/part_sha.py --refresh [<part> ...]    -> refresh the setup.json caches (CI does it with every results commit)
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


def data_files(py):
    """the data a setup may read besides its code: every systems/*.json but runtimes.json (the cage model, loads, assembly: cage_node_lib
    reads overseer.cage.json for every node's socket directions and loads) and the part's inputs/ files"""
    d = os.path.dirname(os.path.abspath(py))
    fs = [f for f in glob.glob(os.path.join(ROOT, "systems", "*.json")) if os.path.basename(f) != "runtimes.json"]
    fs += [f for f in glob.glob(os.path.join(d, "inputs", "**", "*"), recursive=True) if os.path.isfile(f)]
    return sorted(fs)


def data_sha(py):
    fs = data_files(py)
    return hashlib.sha256("".join(_h(f) for f in fs).encode()).hexdigest()[:12] if fs else ""


def setup_sha(py, log=None, refresh=False):
    """the sha of the setup part.py builds now (topo/run.py design_hash: masks, supports, loads, SPEC, thin_ok), cached in
    parts/<name>/setup.json against everything the setup reads: part_sha + lib_sha (the code) + data_sha (systems/*.json and
    inputs/: 30 Oct, Spark lowered four hub ends in overseer.cage.json and nodes 3 and 4 read as current from a cache keyed on
    the code alone), so setup(h) (seconds to a minute) runs only when one of those changed. CI commits the cache with the results
    (parts.yml); a session or the page builder recomputes a miss; refresh=True recomputes regardless."""
    import json
    d = os.path.dirname(os.path.abspath(py)); cache = os.path.join(d, "setup.json")
    ps, ls, ds = part_sha(py), lib_sha(py), data_sha(py)
    if not refresh:
        try:
            c = json.load(open(cache))
            if c.get("part_sha") == ps and c.get("lib_sha") == ls and c.get("data_sha") == ds and c.get("setup_sha"):
                return c["setup_sha"]
        except (OSError, ValueError):
            pass
    sys.path.insert(0, ROOT)
    from topo.run import load_part_dir, design_hash
    spec = load_part_dir(d)
    h, gen = spec["h"], spec["gen_cm3"]
    sh = design_hash(spec["setup"](h), h, gen, spec)
    json.dump(dict(part_sha=ps, lib_sha=ls, data_sha=ds, setup_sha=sh), open(cache, "w"), indent=1)
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
    if sys.argv[1:2] == ["--refresh"]:                   # recompute the setup.json caches: the named parts, or every part with a part.py
        pys = [os.path.join(ROOT, "parts", p, "part.py") for p in sys.argv[2:]] or sorted(glob.glob(os.path.join(ROOT, "parts", "*", "part.py")))
        for py in pys:
            try:
                setup_sha(py, log=print, refresh=True)
            except Exception as e:                       # a setup that does not build has no sha; the run will say why
                print(f"{os.path.basename(os.path.dirname(py))}: no setup sha ({str(e)[:120]})")
    else:
        print(part_sha(sys.argv[1]), lib_sha(sys.argv[1]), setup_sha(sys.argv[1]))
