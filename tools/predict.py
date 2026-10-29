"""Predict run time for parts, from the runs that already finished. No model, no tokens: a session reads one line.

    python tools/predict.py hub_deck holder          # "hub_deck ~34 min  holder ~58 min  latest 23:40 PDT  check-in 65 min"
    python tools/predict.py --update                 # CI: append every finished run to systems/runtimes.json
    python tools/predict.py --engine v2              # engine changed: older entries are ignored from now on

systems/runtimes.json: {"engine": "<tag>", "runs": [{part, engine, cells, design, cases, minutes, peak_gb, at}]}
Fit (29 Oct, after the nodes ran 2 h on a 20 min prediction): minutes ~ a_engine * (design_cells * cases) ^ b, log-log
least squares over EVERY engine's runs of 3 min or more (shorter ones are audit refusals and cut-offs): one shared slope
b, one intercept per engine. The current engine's intercept comes from its own runs once it has two, until then from the
latest engine that has any (an engine tag reset used to throw the whole history away and fit a line through 1..3 points).
Safety: the 80th percentile of actual / fit over the latest engines (back until 10 runs), at least 1.25. A part still running past its
prediction is given max(0.5 x prediction, MIN_CHECK) more. MIN_CHECK 60: check-ins are never under an hour apart (CLAUDE.md).
"""
import glob, json, os, sys, time, math
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "systems", "runtimes.json")
SAFETY_MIN = 1.25
MIN_MINUTES = 3.0                                     # shorter runs are audit refusals or cut-offs, not timings
MIN_CHECK = 60


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


def _usable(d):
    return [x for x in d["runs"] if x["design"] and x["minutes"] and x["minutes"] >= MIN_MINUTES]


def fit(d):
    """minutes(design, cases) for the current engine, and the safety factor"""
    runs = _usable(d)
    if not runs:
        return (lambda des, cases: 60.0), SAFETY_MIN
    engines = []
    for x in runs:
        if x["engine"] not in engines:
            engines.append(x["engine"])
    xs = [math.log(x["design"] * x["cases"]) for x in runs]
    ys = [math.log(x["minutes"]) for x in runs]
    # shared slope on engine-centred data
    mx, my = {}, {}
    for e in engines:
        idx = [k for k, x in enumerate(runs) if x["engine"] == e]
        mx[e] = sum(xs[k] for k in idx) / len(idx); my[e] = sum(ys[k] for k in idx) / len(idx)
    num = sum((xs[k] - mx[x["engine"]]) * (ys[k] - my[x["engine"]]) for k, x in enumerate(runs))
    den = sum((xs[k] - mx[x["engine"]]) ** 2 for k, x in enumerate(runs))
    b = num / den if den > 1e-9 else 1.0
    b = min(max(b, 0.5), 1.5)                                   # a few points can give any slope: keep it near linear
    la = {e: my[e] - b * mx[e] for e in engines}                # intercept per engine
    cur = d["engine"]
    n_cur = sum(1 for x in runs if x["engine"] == cur)
    use = cur if n_cur >= 2 else [e for e in engines if e in la][-1]
    f = lambda des, cases: math.exp(la[use] + b * math.log(max(des * cases, 1)))
    recent, k = [], len(engines)                                 # the latest engines, back until 10 runs are in hand
    while len(recent) < 10 and k > 0:
        k -= 1; recent += [x for x in runs if x["engine"] == engines[k]]
    ratios = sorted(x["minutes"] / math.exp(la[x["engine"]] + b * math.log(x["design"] * x["cases"])) for x in recent)
    safety = max(SAFETY_MIN, ratios[int(0.8 * (len(ratios) - 1))]) if ratios else SAFETY_MIN
    return f, safety


def part_size(part):
    """design cells and cases for a part: from its last result.json, else a 2-iteration smoke is too slow here, so
    fall back to the median of known runs."""
    rj = os.path.join(ROOT, "parts", part, "result.json")
    if os.path.exists(rj):
        return size(json.load(open(rj)))
    d = load()
    runs = _usable(d) or [dict(design=60000, cases=20)]
    runs.sort(key=lambda x: x["design"] * x["cases"])
    m = runs[len(runs) // 2]
    return m["design"], m["cases"]


def predict(parts, started=None):
    d = load(); f, safety = fit(d)
    started = started or datetime.now(timezone.utc)
    out, latest = [], started
    for p in parts:
        des, cases = size_or_running(p)
        m = f(des, cases) * safety
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
