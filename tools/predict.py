"""Predict run time for parts, from the runs that already finished. No model, no tokens: a session reads one line.

    python tools/predict.py hub_deck holder          # "hub_deck ~34 min  holder ~58 min  latest 23:40 PDT  check-in 65 min"
    python tools/predict.py --update                 # CI: append every finished run to systems/runtimes.json
    python tools/predict.py --engine v2              # engine changed: older entries are ignored from now on

systems/runtimes.json: {"engine": "<tag>", "runs": [{part, engine, cells, design, cases, minutes, peak_gb, at}]}
Fit: minutes ~ a * design_cells * cases ^ b over the current engine's runs (log-log least squares), with a safety
factor. Fewer than 3 runs: use the per-(design*cases) median. A part that is still running past its prediction gets
its remaining time re-estimated from the fit, plus 15 min.
"""
import glob, json, os, sys, time, math
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "systems", "runtimes.json")
SAFETY = 1.25
MIN_CHECK = 15


def load():
    if os.path.exists(RT):
        return json.load(open(RT))
    return {"engine": "v1", "runs": []}


def save(d):
    json.dump(d, open(RT, "w"), indent=1)


def size(r):
    g = r.get("grid") or {}
    return g.get("design") or g.get("cells") or 0, len(g.get("cases") or []) or 1


def update():
    d = load()
    seen = {(x["part"], x["at"]) for x in d["runs"]}
    for rj in sorted(glob.glob(os.path.join(ROOT, "parts", "*", "result.json"))):
        r = json.load(open(rj))
        part = os.path.basename(os.path.dirname(rj))
        at = r.get("part_sha", "") + ":" + str(r.get("total_s"))
        if (part, at) in seen or not r.get("total_s"):
            continue
        des, cases = size(r)
        d["runs"].append(dict(part=part, engine=d["engine"], design=des, cases=cases,
                              minutes=round(r["total_s"] / 60, 1), peak_gb=r.get("peak_gb"), at=at))
    save(d)
    print(f"runtimes: {len(d['runs'])} runs, engine {d['engine']}")


def fit(d):
    runs = [x for x in d["runs"] if x["engine"] == d["engine"] and x["design"] and x["minutes"]]
    if not runs:
        return lambda des, cases: 30.0
    if len(runs) < 3:
        k = sorted(x["minutes"] / (x["design"] * x["cases"]) for x in runs)[len(runs) // 2]
        return lambda des, cases: k * des * cases
    # log(min) = la + b * log(design * cases)
    xs = [math.log(x["design"] * x["cases"]) for x in runs]
    ys = [math.log(x["minutes"]) for x in runs]
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9)
    la = my - b * mx
    return lambda des, cases: math.exp(la + b * math.log(max(des * cases, 1)))


def part_size(part):
    """design cells and cases for a part: from its last result.json, else a 2-iteration smoke is too slow here, so
    fall back to the median of known runs."""
    rj = os.path.join(ROOT, "parts", part, "result.json")
    if os.path.exists(rj):
        return size(json.load(open(rj)))
    d = load()
    runs = [x for x in d["runs"] if x["engine"] == d["engine"]] or [dict(design=60000, cases=20)]
    runs.sort(key=lambda x: x["design"] * x["cases"])
    m = runs[len(runs) // 2]
    return m["design"], m["cases"]


def predict(parts, started=None):
    d = load(); f = fit(d)
    started = started or datetime.now(timezone.utc)
    out, latest = [], started
    for p in parts:
        des, cases = size_or_running(p)
        m = f(des, cases) * SAFETY
        eta = started + timedelta(minutes=m)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 60
        if elapsed > m:                                  # overdue: push out by the shortfall plus a margin
            eta = datetime.now(timezone.utc) + timedelta(minutes=max(MIN_CHECK, 0.5 * m))
        out.append((p, round(m), eta)); latest = max(latest, eta)
    return out, latest


def size_or_running(p):
    return part_size(p)


def main(argv):
    if "--update" in argv:
        return update()
    if "--engine" in argv:
        d = load(); d["engine"] = argv[argv.index("--engine") + 1]; save(d); print("engine", d["engine"]); return
    started = None
    if "--started" in argv:                                 # ISO time the run began (default now)
        i = argv.index("--started"); started = datetime.fromisoformat(argv[i + 1]).astimezone(timezone.utc)
        argv = argv[:i] + argv[i + 2:]
    parts = [a for a in argv if not a.startswith("--")]
    out, latest = predict(parts, started)
    try:
        from zoneinfo import ZoneInfo
        local = latest.astimezone(ZoneInfo("America/Los_Angeles")).strftime("%H:%M %Z")
    except Exception:
        local = latest.strftime("%H:%M UTC")
    wait = max(MIN_CHECK, int((latest - datetime.now(timezone.utc)).total_seconds() / 60) + 2)
    print("  ".join(f"{p} ~{m} min" for p, m, _ in out) + f"  latest {local}  check-in {wait} min  ({latest.isoformat(timespec='minutes')})")


if __name__ == "__main__":
    main(sys.argv[1:])
