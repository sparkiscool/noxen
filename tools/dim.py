#!/usr/bin/env python3
"""Print one section of systems/dimensions.md, so a session reads only the numbers it uses (CLAUDE.md Every session, 1).

python3 tools/dim.py               -> every section heading with its size
python3 tools/dim.py L             -> section "## L. ..." (the letter code)
python3 tools/dim.py "box catch"   -> every section whose heading contains the text (case-insensitive)
"""
import pathlib, re, sys

text = (pathlib.Path(__file__).resolve().parents[1] / "systems/dimensions.md").read_text()
secs = re.split(r"(?m)^(?=## )", text)
heads = [(s.splitlines()[0][3:], s) for s in secs if s.startswith("## ")]
if len(sys.argv) < 2:
    for h, s in heads:
        print(f"{len(s) / 1024:6.1f} KB  {h[:110]}")
    sys.exit(0)
q = " ".join(sys.argv[1:])
hit = [s for h, s in heads if h.split(".")[0] == q] or [s for h, s in heads if q.lower() in h.lower()]
if not hit:
    sys.exit(f"dim: no section matches {q!r}; python3 tools/dim.py lists them")
print("\n".join(s.rstrip() for s in hit))
