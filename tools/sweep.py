#!/usr/bin/env python3
"""The Monday housekeeping report (CLAUDE.md Sessions): what the repo carries that nothing uses, in a few lines.

    python3 tools/sweep.py            -> the report
    python3 tools/sweep.py --apply    -> also git-mv every archive candidate to archive/parts/ (commit is the caller's)

Archive candidate: a part folder holding only part.stl that nothing draws, names or regenerates. A folder is kept when
  * a live systems/*.assembly.json draws it: a line whose `mesh` is inside it, or a line with no mesh whose `part` (else `id`) is
    it. An id alone is not a use: the box page's seated drones are `box_overseer_<id>` lines drawn from the drone's own folder,
    and a mirrored node is drawn from the folder its `part` names;
  * live code or data names it: any .py/.yml/.md/.html, or a JSON file in a part folder (a generator's result.json or kit.json
    lists what it writes: parts/latch7/make.py writes latch7_* and latch7_ov_* by prefix and names each in its result.json).
    Text in archive/, site/, systems/runtimes.json and in a folder only an archived system draws (its generator's leftovers)
    does not count;
  * a glob in code matches it (.github/workflows/box.yml commits parts/box_latch7_*, parts/box_funnel_*_drawn);
  * it is `<X>_drawn` or `<X>_room` and parts/<X> is an engine part: tools/import_box.py draws an engine part's drawn faces and
    its room into those two folders until its run's mesh is in.
A folder with a part.py or make.py is never a candidate (a jig or a coupon is printed without an assembly). No engine, no
network: seconds.
"""
import fnmatch, glob, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
VIEWS = ("_drawn", "_room")                      # tools/import_box.py view(): an engine part's drawn faces and room until its run lands
SHAPES = ("box", "cylinder", "sphere")           # tools/build_site.py placeholder()


def systems():
    out = {}
    for f in sorted(glob.glob("systems/*.assembly.json")):
        try:
            out[f] = json.load(open(f))
        except ValueError:
            print(f"sweep: {f} is not JSON")
    return out


def drawn(a):
    """the part folders an assembly draws"""
    ds = set()
    for p in a.get("parts", []):
        m = p.get("mesh")
        if m:
            g = re.match(r"parts/([^/]+)/", m)
            if g:
                ds.add(g.group(1))
        else:
            ds.add(p.get("part") or p.get("id") or p.get("name"))
    return ds


def code_text(dead_owners):
    fs = [f for f in glob.glob("**/*", recursive=True) if os.path.isfile(f) and not f.startswith(("archive/", "site/"))
          and (f.endswith((".py", ".yml", ".md", ".html")) or (f.startswith("parts/") and f.endswith(".json")))]
    fs += glob.glob(".github/workflows/*.yml")
    fs += [f for f in glob.glob("systems/*.json") if not f.endswith((".assembly.json", "runtimes.json"))]
    fs = [f for f in fs if not (f.startswith("parts/") and f.split("/")[1] in dead_owners)]
    return {f: open(f, errors="ignore").read() for f in fs}


def main(apply):
    S = systems()
    live = {f: a for f, a in S.items() if not a.get("archived")}
    use = set().union(*(drawn(a) for a in live.values()))
    dead_owners = set().union(*(drawn(a) for a in S.values() if a.get("archived"))) - use   # folders only an archived system draws
    txt = code_text(dead_owners)
    fam = set()                                   # folder families code writes or reads by name pattern
    for f, t in txt.items():
        if f.endswith(".yml"):                    # a workflow's globs (box.yml git add parts/box_latch7_*), not its push triggers
            fam |= set(re.findall(r"parts/([A-Za-z0-9_]{4,}\*[A-Za-z0-9_*]*)", re.sub(r"(?m)^\s*paths:.*$", "", t)))
        elif f.endswith(".py"):                   # a built name: "scout_motor_" + str(k), f"box_latch7_{pack}_{kind}", parts/latch7_*
            fam |= set(re.findall(r"parts/([A-Za-z0-9_]{4,}\*[A-Za-z0-9_*]*)", t))
            fam |= {a + b + "*" for a, b in re.findall(r'"([a-z][a-z0-9_]{5,}_)"\s*\+|f"([a-z][a-z0-9_]{5,}_)\{', t)}
    redirected = {p["id"] for a in live.values() for p in a.get("parts", []) if p.get("id") and p["id"] not in drawn(dict(parts=[p]))}
    dirs = sorted(d for d in os.listdir("parts") if os.path.isdir(f"parts/{d}"))
    engine = {d for d in dirs if os.path.exists(f"parts/{d}/part.py")}
    cand = []
    for d in dirs:
        if d in use or set(os.listdir(f"parts/{d}")) != {"part.stl"}:
            continue
        if any(d.endswith(v) and d[: -len(v)] in engine for v in VIEWS):
            continue
        if d not in redirected and any(fnmatch.fnmatch(d, g) for g in fam):   # a line drawing id d from another folder: only its name keeps it
            continue
        pat = re.compile(r"\b%s\b" % re.escape(d))
        if not any(pat.search(t) for f, t in txt.items() if not f.startswith(f"parts/{d}/")):
            cand.append(d)
    print(f"archive candidates ({len(cand)}): {' '.join(cand) or '-'}")
    if apply and cand:
        os.makedirs("archive/parts", exist_ok=True)
        for d in cand:
            subprocess.run(["git", "mv", f"parts/{d}", f"archive/parts/{d}"], check=True)
        print(f"  moved {len(cand)} to archive/parts/ (not committed); add a line to archive/README.md saying why")

    scripts = [d for d in sorted(engine) if d not in use and not any(d + v in use for v in VIEWS)]
    print(f"part.py in no assembly ({len(scripts)}): {' '.join(scripts) or '-'}")

    dangling, ph = [], 0
    for f, a in live.items():
        for p in a.get("parts", []):
            src = p.get("part") or p.get("id") or p.get("name")
            if p.get("mesh"):
                ok = os.path.exists(p["mesh"])
            elif any(k in p for k in SHAPES):
                ok = True; ph += 1
            else:
                ok = any(os.path.exists(f"parts/{src}/{x}") for x in ("part.stl", "part.py"))
            if not ok:
                dangling.append(f"{os.path.basename(f).split('.')[0]}:{p.get('id')}" + (f" ({p['mesh']})" if p.get("mesh") else ""))
    print(f"live assembly lines that draw nothing ({len(dangling)}; {ph} placeholder shapes are fine): {' '.join(dangling) or '-'}")
    gone = [p["mesh"] for a in S.values() if a.get("archived") for p in a.get("parts", []) if p.get("mesh") and not os.path.exists(p["mesh"])]
    lost = [m for m in gone if not os.path.exists("archive/" + m)]
    print(f"archived systems (off the page): {' '.join(os.path.basename(f).split('.')[0] for f, a in S.items() if a.get('archived')) or '-'};"
          f" {len(gone) - len(lost)} of their meshes are in archive/, {len(lost)} nowhere{': ' + ' '.join(lost) if lost else ''}")

    refs = re.findall(r"`((?:nox|docs|runs|tools|parts|systems)/[^`\s]+)`", open("systems/dimensions.md").read())
    dead = sorted({p for p in refs if not os.path.exists(p.rstrip("/").split(":")[0])})
    moved = [p for p in dead if os.path.exists("archive/" + p.rstrip("/").split(":")[0])]
    print(f"dimensions.md paths that do not exist: {len(dead)} (nox/, docs/, runs/ and the rest are the old repo, the README says so); "
          f"moved to archive/ ({len(moved)}, point them there): {' '.join(moved) or '-'}")

    n = sum(1 for _ in open("CLAUDE.md"))
    print(f"CLAUDE.md {n} lines" + ("  OVER the 60-line limit: move something into code" if n >= 60 else ""))
    fb = glob.glob("parts/*/feedback.md")
    print(f"feedback.md files: {len(fb)}")

    stl = sorted(((os.path.getsize(f), f) for f in glob.glob("parts/*/part.stl")), reverse=True)
    tot = sum(s for s, _ in stl) / 1e6
    print(f"committed meshes: {len(stl)} part.stl, {tot:.0f} MB; largest: " + ", ".join(f"{f.split('/')[1]} {s / 1e6:.0f} MB" for s, f in stl[:5]))
    rt = json.load(open("systems/runtimes.json")) if os.path.exists("systems/runtimes.json") else {"runs": []}
    print(f"runtimes.json: {len(rt['runs'])} runs, engine {rt.get('engine')}")


if __name__ == "__main__":
    main("--apply" in sys.argv[1:])
