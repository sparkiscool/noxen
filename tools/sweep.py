#!/usr/bin/env python3
"""The Monday housekeeping report (CLAUDE.md Sessions): what the repo carries that nothing uses, in a few lines.

    python3 tools/sweep.py            -> the report
    python3 tools/sweep.py --apply    -> also git-mv every archive candidate to archive/parts/ (commit is the caller's)

Archive candidate: a part folder holding only part.stl, placed by no systems/*.assembly.json (id, part, src or mesh), named in
no part's inputs/*.json, and named by no code or doc. A folder with a part.py or make.py is never a candidate (a jig or a coupon
is printed without an assembly). No engine, no network: seconds.
"""
import glob, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def placed():
    ids = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("id", "part", "src", "mesh") and isinstance(v, str):
                    ids.add(v)
                    m = re.match(r"parts/([^/]+)/", v)
                    if m:
                        ids.add(m.group(1))
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    for f in glob.glob("systems/*.json") + glob.glob("runs/*/*.json") + glob.glob("parts/*/inputs/*.json"):
        try:
            walk(json.load(open(f)))
        except ValueError:
            print(f"sweep: {f} is not JSON")
    return ids


def code_text():
    fs = [f for f in glob.glob("**/*", recursive=True) if os.path.isfile(f) and not f.startswith(("archive/", "site/"))
          and f.endswith((".py", ".yml", ".md", ".html")) and "/inputs/" not in f]
    fs += glob.glob(".github/workflows/*.yml") + glob.glob("parts/*/inputs/*.json")   # an inputs file naming a folder under any key keeps it
    return {f: open(f, errors="ignore").read() for f in fs}


def main(apply):
    own, txt = placed(), code_text()
    dirs = sorted(d for d in os.listdir("parts") if os.path.isdir(f"parts/{d}"))
    cand = []
    for d in dirs:
        if d in own or set(os.listdir(f"parts/{d}")) != {"part.stl"}:
            continue
        pat = re.compile(r"\b%s\b" % re.escape(d))
        if not any(pat.search(t) for f, t in txt.items() if not f.startswith(f"parts/{d}/")):
            cand.append(d)
    print(f"archive candidates ({len(cand)}): {' '.join(cand) or '-'}")
    if apply and cand:
        os.makedirs("archive/parts", exist_ok=True)
        for d in cand:
            subprocess.run(["git", "mv", f"parts/{d}", f"archive/parts/{d}"], check=True)
        print(f"  moved {len(cand)} to archive/parts/ (not committed)")

    scripts = [d for d in dirs if os.path.exists(f"parts/{d}/part.py") and d not in own]
    print(f"part.py in no assembly ({len(scripts)}): {' '.join(scripts) or '-'}")
    missing = sorted({i for f in glob.glob("systems/*.assembly.json") for i in (p.get("part", p["id"]) for p in json.load(open(f))["parts"])
                      if not os.path.isdir(f"parts/{i}") and not (json.load(open(f)).get("archived"))})
    print(f"assembly ids with no folder ({len(missing)}): {' '.join(missing) or '-'}  (fine for placeholders; a typo shows here)")

    dead = sorted({p for p in re.findall(r"`((?:nox|docs|runs|tools|parts|systems)/[^`\s]+)`", open("systems/dimensions.md").read())
                   if not os.path.exists(p.rstrip("/").split(":")[0])})
    print(f"dimensions.md paths that do not exist ({len(dead)}; nox/ ones are the old repo, header says so): "
          f"{' '.join(p for p in dead if not p.startswith('nox/'))[:300] or '-'}")

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
