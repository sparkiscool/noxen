"""What a result was made from: part_sha(py) hashes part.py, lib_sha(py) hashes the parts/<lib>.py modules it imports
(transitively). A result is stale when part.py changed, or when a lib changed since a result that recorded lib_sha (older
results carry no lib_sha and count as current; 28 Oct: the node lib lost its drawn arms and twelve node results read as
current because only part.py was hashed).

    python tools/part_sha.py parts/<name>/part.py      -> "<part_sha> <lib_sha>"
"""
import hashlib, os, re, sys

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


def lib_sha(py):
    ls = libs(py)
    return hashlib.sha256("".join(_h(l) for l in ls).encode()).hexdigest()[:12] if ls else ""


def stale(py, r):
    """r = result.json dict: made from an older part.py, or from an older lib when it recorded one"""
    ls = lib_sha(py)                                     # a result with no lib_sha from a part that imports a lib is of unknown make: stale
    return r.get("part_sha") != part_sha(py) or (bool(ls) and r.get("lib_sha") != ls)


if __name__ == "__main__":
    print(part_sha(sys.argv[1]), lib_sha(sys.argv[1]))
