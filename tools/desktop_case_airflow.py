"""Desktop dual-board case: airflow network model for the cooling study (job 1116, 25 Nov).
A lumped model, the usual first pass for an enclosure: every opening is an orifice (dp = rho/2 (Q / (Cd A))^2), every fan a straight
line from its free flow Q0 to its stall pressure P0 (past stall it lets air back through its own area), every space one pressure; the
flows are balanced at each space. Heat is an energy balance on the air (dT = P / (rho cp Q)). Good to about +-30 %: it ranks layouts
and sizes openings, it does not predict a chip temperature.
Geometry: parts/desktop_dual_board_case/make.py at commit ddc31d48 (412 x 126 x 304, fans 1 x 120 + 4 x 70 from the Kootek LCP04 pad).
E = estimate: the Kootek listing publishes no fan speed, flow or pressure, and the boards' heat is not published; each has a low /
nominal / high guess here and the answer is given as a range.
Run: python tools/desktop_case_airflow.py            -> the table
     python tools/desktop_case_airflow.py --json     -> the same as JSON (the study page reads it)"""
import json, math, sys
from scipy.optimize import least_squares

RHO, CP = 1.2, 1005.0
CFM = 2.11888                      # 1 L/s in CFM
EPS = 1e-3                         # Pa, smooths the square root at zero flow
CS = 0.35                          # discharge coefficient of a stalled or over-driven rotor (E)

# ---- as drawn (mm2): counted from make.py's own slot rules
A_BASE_SLOTS = 24 * 8 * 18 + 5 * 8 * 18            # rear wall 24 slots + -x end wall 5 slots = 4176
A_UNDERFLOOR = 2 * 36 * 11                          # the only way under the floor: the 36 x 11 channel between the two foot rails, open at both ends = 792


def strip(d, half=18.0):                            # part of a floor bore the rails leave open (the strip y 45..81 across a circle)
    r = d / 2
    return 2 * (half * math.sqrt(r * r - half * half) + r * r * math.asin(half / r))


A_BORE_STRIPS = strip(100) + 3 * strip(60)          # 9588: what the rails leave of the D0 bores
A_FLOOR_PATH = 1 / math.sqrt(1 / A_UNDERFLOOR ** 2 + 1 / A_BORE_STRIPS ** 2)     # the two in series = 789
A_D1_BORES = math.pi / 4 * (113 ** 2 + 3 * 66 ** 2)                               # 20292
A_LID_SLOTS = 18 * 6 * 94                           # 10152
A_END_SLOTS = 3 * 8 * 68                            # 1632 (-x end wall, top)
A_PORT_WIN = 4 * 21.6 * 100                         # 8640 (-x end wall, open port windows)
A_FAN = {120: math.pi / 4 * 113 ** 2, 70: math.pi / 4 * 66 ** 2}

# ---- fans (E): free flow L/s, stall pressure Pa. 5 V USB pad fans; low / nominal / high
FANS = {"low": {120: (14.0, 5.0), 70: (4.7, 8.0)},
        "nominal": {120: (19.0, 8.0), 70: (6.6, 12.0)},
        "high": {120: (24.0, 12.0), 70: (8.5, 18.0)}}
HEAT = {"low": 180.0, "nominal": 270.0, "high": 360.0}       # W, both boards together (E: 90 / 135 / 180 W each)


def orifice(area_mm2, cd):
    c = cd * area_mm2 * 1e-6 * math.sqrt(2 / RHO)
    return lambda dp: c * dp / (dp * dp + EPS * EPS) ** 0.25


def fan_rise(q, q0, p0, a):                          # pressure rise the fan gives at flow q (m3/s), monotone falling
    if q < 0:
        return p0 + RHO / 2 * (q / (CS * a)) ** 2
    if q > q0:
        return -RHO / 2 * ((q - q0) / (CS * a)) ** 2
    return p0 * (1 - q / q0)


def fan(size, level, n=1, inlet_mm2=None):
    """n equal fans side by side; inlet_mm2 = a choked inlet in series with each (one velocity head lost on that area)."""
    q0, p0 = FANS[level][size]
    q0, a = q0 * 1e-3, A_FAN[size] * 1e-6
    k = RHO / 2 / (inlet_mm2 * 1e-6) ** 2 if inlet_mm2 else 0.0

    def q_of(rise):
        lo, hi = -0.2, 0.2
        for _ in range(60):
            mid = (lo + hi) / 2
            if fan_rise(mid, q0, p0, a) - k * mid * abs(mid) > rise:
                lo = mid
            else:
                hi = mid
        return n * (lo + hi) / 2
    return q_of


def solve(nodes, elems):
    """elems: (name, a, b, kind, f); flow a -> b. kind 'o': f(p_a - p_b); kind 'f': f(p_b - p_a). 'amb' is 0 Pa."""
    def flows(p):
        pr = dict(zip(nodes, p)); pr["amb"] = 0.0
        return pr, [(nm, a, b, f(pr[a] - pr[b]) if kind == "o" else f(pr[b] - pr[a])) for nm, a, b, kind, f in elems]

    def res(p):
        _, fl = flows(p)
        return [1e3 * (sum(q for _, a, b, q in fl if b == n) - sum(q for _, a, b, q in fl if a == n)) for n in nodes]
    s = least_squares(res, [0.0] * len(nodes), xtol=1e-12, ftol=1e-12)
    assert max(abs(r) for r in s.fun) < 1e-3, (nodes, s.fun)
    pr, fl = flows(s.x)
    return pr, {nm: q * 1e3 for nm, a, b, q in fl}


def exits(level, lid_mm2, leak_mm2, lid_fan):
    e = [("lid", "B", "amb", "o", orifice(lid_mm2, 0.62)), ("leaks", "B", "amb", "o", orifice(leak_mm2, 0.62))]
    if lid_fan:
        e.append(("lid fan", "B", "amb", "f", fan(70, level)))
    return e


def case(name, level):
    """-> (pressures, flows L/s, air that passes the boards L/s, note)"""
    if name == "as drawn, fans on the floor deck":          # fans sealed over the D0 bores, as the picture shows them
        nodes = ["U", "S", "B"]
        el = [("floor path", "amb", "U", "o", orifice(A_FLOOR_PATH, 0.62)),
              ("fan 120", "U", "S", "f", fan(120, level)), ("fans 70", "U", "S", "f", fan(70, level, 3)),
              ("deck bores", "S", "B", "o", orifice(A_D1_BORES, 0.8)),
              ("base slots", "amb", "S", "o", orifice(A_BASE_SLOTS, 0.62))]
        el += exits(level, A_LID_SLOTS, A_END_SLOTS + A_PORT_WIN, True)
    elif name == "as drawn, fans under the upper deck":     # fans screwed under D1 (where their holes are), the base a suction box
        nodes = ["S", "B"]
        el = [("intake", "amb", "S", "o", orifice(A_BASE_SLOTS + A_FLOOR_PATH, 0.62)),
              ("fan 120", "S", "B", "f", fan(120, level, 1, math.pi * 113 * 9)),      # its inlet is 9 mm off the floor
              ("fans 70", "S", "B", "f", fan(70, level, 3, math.pi * 66 * 19))]       # theirs 19 mm
        el += exits(level, A_LID_SLOTS, A_END_SLOTS + A_PORT_WIN, True)
    elif name == "open floor only":                          # the new intake, today's lid, port windows and lid fan
        return open_floor(level, lid=A_LID_SLOTS, leak=A_END_SLOTS + A_PORT_WIN, lid_fan=True)
    elif name == "recommended":
        return open_floor(level)
    elif name == "recommended + lid fan":
        return open_floor(level, lid=LID - A_FAN[70], lid_fan=True)
    elif name == "recommended, all five fans in the base":
        return open_floor(level, n70=4)
    else:
        raise KeyError(name)
    pr, fl = solve(nodes, el)
    return pr, fl, fl["lid"] + fl["leaks"] + fl.get("lid fan", 0.0)


FEET, BOX, KF, LID, LEAK = 30.0, 15.0, 4.0, 25000.0, A_END_SLOTS + 4000.0     # the recommended sizes (mm, mm2); KF = filter loss in velocity heads on its face (E)


def open_floor(level, feet=FEET, box=BOX, kf=KF, lid=LID, leak=LEAK, n70=3, lid_fan=False):
    """Fans on the floor deck over full bores, drawing through a filter that is the floor of a shallow box under them (box mm deep), the
    case on feet (feet mm clear) open all round; the upper deck one long window; the lid an open grille (lid mm2 free)."""
    gap = (2 * (412 + 126) - 4 * 40) * feet                                         # way in under the floor, less four 40 mm feet
    face = min(406 * 120, sum(A_FAN[s] + math.pi * d * box for s, d in ((120, 113),) + ((70, 66),) * n70))   # filter area the fans really draw through
    terms = 1 / (0.82 * gap) ** 2 + (kf / face ** 2 if kf else 0.0)
    el = [("intake", "amb", "F", "o", orifice(1 / math.sqrt(terms), 1.0)),
          ("fan 120", "F", "B", "f", fan(120, level)), ("fans 70", "F", "B", "f", fan(70, level, n70))]
    el += exits(level, lid, leak, lid_fan)
    pr, fl = solve(["F", "B"], el)
    return pr, fl, fl["lid"] + fl["leaks"] + fl.get("lid fan", 0.0)


CASES = ["as drawn, fans on the floor deck", "as drawn, fans under the upper deck", "open floor only", "recommended",
         "recommended + lid fan", "recommended, all five fans in the base"]
SWEEPS = {"feet": ("feet", [11, 15, 20, 25, 30, 40, 50]), "filter": ("kf", [0, 2, 4, 8, 12]),
          "lid": ("lid", [5000, 10000, 15000, 20000, 25000, 30000, 35000]), "box": ("box", [0, 5, 10, 15, 20])}


# ---- the layout asked for on 25 Nov: the whole Kootek pad lying under the case, the two halves of the case sliding apart in y
PAD_X, PAD_Y = 380.0, 300.0                                   # exact (listing)
PAD_FANS = [(120, 0.0, 0.0)] + [(70, sx * 108.0, sy * 55.0) for sx in (-1, 1) for sy in (-1, 1)]   # (size, x, y) from the big fan's centre: E, read off the listing photo, +-15 mm
BOARD_T, BOARD_L, OUTER = 20.0, 340.0, 13.0                   # board thickness (E), length (E), the channel between a board and its outer wall
K_MESH = 1.5                                                  # the pad's perforated top, velocity heads on the fan bore (E)
GAPS = [54, 80, 100, 120, 140, 160, 180, 220]
OPEN_GAP = 120                                                # the open spacing to set first (see the study)


def disc_part(r, yc, a, b):
    """Share of a disc (radius r, centre yc) with a <= |y| < b."""
    def below(t):
        t = max(-r, min(r, t - yc))
        return r * r * (math.pi / 2 + math.asin(t / r)) + t * math.sqrt(r * r - t * t)
    return (below(b) - below(a) + below(-a) - below(-b)) / (math.pi * r * r)


def pad_flows(level, clearance, fans=None, kf=0.0):
    """What each pad fan delivers through the mesh with `clearance` mm open under the pad all round -> {(size, x, y): L/s}.
    kf = a filter sleeve round that gap (velocity heads on its face)."""
    fans = fans or PAD_FANS
    a = 2 * (PAD_X + PAD_Y) * clearance
    el = [("intake", "amb", "P", "o", orifice(1 / math.sqrt(1 / (0.82 * a) ** 2 + kf / a ** 2), 1.0))]
    for i, (size, x, y) in enumerate(fans):
        el.append((i, "P", "amb", "f", fan(size, level, 1, A_FAN[size] / math.sqrt(K_MESH))))
    pr, fl = solve(["P"], el)
    return {fans[i]: fl[i] for i in range(len(fans))}


def spread(gap, flows, half_depth=None):
    """Where the pad's air goes with the cooler faces `gap` apart: between the boards, onto the outer faces, or past the case.
    A fan under a board's bottom edge is split half and half. half_depth = inner half depth of a closed box (None = the halves' own walls)."""
    wall = half_depth if half_depth else gap / 2 + BOARD_T + OUTER
    q = {"between": 0.0, "outer": 0.0, "missed": 0.0}
    for (size, x, y), f in flows.items():
        r = (113.0 if size == 120 else 66.0) / 2
        inside, edge = disc_part(r, y, 0, gap / 2), disc_part(r, y, gap / 2, gap / 2 + BOARD_T)
        outer = disc_part(r, y, gap / 2 + BOARD_T, wall)
        q["between"] += f * (inside + edge / 2); q["outer"] += f * (outer + edge / 2); q["missed"] += f * (1 - inside - edge - outer)
    q["speed"] = q["between"] * 1e-3 / (gap * BOARD_L * 1e-6)
    return q


def pad_layout():
    out = {"fans": [{"size": s_, "x": x, "y": y} for s_, x, y in PAD_FANS], "gaps": [], "clearance": [], "shift": []}
    for g in GAPS:
        row = {"gap": g}
        for lv in ("low", "nominal", "high"):
            q = spread(g, pad_flows(lv, 30.0), 60.0 if g == 54 else None)
            row[lv] = {k: round(v, 2 if k == "speed" else 1) for k, v in q.items()}
        out["gaps"].append(row)
    for c in (5, 10, 20, 30, 40):                               # how much the pad needs under it
        out["clearance"].append({"x": c, **{lv: round(sum(pad_flows(lv, float(c)).values()), 1) for lv in ("low", "nominal", "high")}})
    out["open_gap"] = OPEN_GAP
    q0, q1 = spread(54, pad_flows("nominal", 30.0), 60.0), spread(OPEN_GAP, pad_flows("nominal", 30.0))
    out["closed_w_10K"] = round(q0["between"] * 1e-3 * RHO * CP * 10)
    out["open_w_10K"] = round(q1["between"] * 1e-3 * RHO * CP * 10)
    out["filter_loss_pct"] = round(100 * (1 - sum(pad_flows("nominal", 30.0, kf=KF).values()) / sum(pad_flows("nominal", 30.0).values())))
    out["flat_between"] = {lv: round(spread(54, pad_flows(lv, 5.0), 60.0)["between"], 1) for lv in ("low", "nominal", "high")}
    for yy in (45.0, 55.0, 70.0):                               # if the small fans are not where the photo suggests
        fans = [(120, 0.0, 0.0)] + [(70, sx * 108.0, sy * yy) for sx in (-1, 1) for sy in (-1, 1)]
        fl = pad_flows("nominal", 30.0, fans)
        out["shift"].append({"y70": yy, **{str(g): round(spread(g, fl)["between"], 1) for g in (100, 120, 140, 160, 180)}})
    return out


def run():
    out = {"areas": {"base slots": A_BASE_SLOTS, "under-floor channel": A_UNDERFLOOR, "bore strips": round(A_BORE_STRIPS), "floor path": round(A_FLOOR_PATH),
                     "deck bores": round(A_D1_BORES), "lid slots": A_LID_SLOTS, "end slots": A_END_SLOTS, "port windows": round(A_PORT_WIN),
                     "fan 120": round(A_FAN[120]), "fan 70": round(A_FAN[70])}, "cases": []}
    for nm in CASES:
        row = {"name": nm}
        for lv in ("low", "nominal", "high"):
            pr, fl, q = case(nm, lv)
            row[lv] = {"flow_ls": round(q, 1), "flow_cfm": round(q * CFM), "bay_pa": round(pr["B"], 2),
                       "pressures": {k: round(v, 2) for k, v in pr.items() if k != "amb"},
                       "flows": {k: round(v, 1) for k, v in fl.items()},
                       "rise_K": {h: round(HEAT[h] / (RHO * CP * q * 1e-3), 1) if q > 0.05 else None for h in HEAT}}
        out["cases"].append(row)
    out["sweeps"] = {k: [{"x": v, **{lv: round(open_floor(lv, **{arg: v})[2], 1) for lv in ("low", "nominal", "high")}} for v in vals]
                     for k, (arg, vals) in SWEEPS.items()}
    # fans off: the chimney alone. Draft = rho g H dT / T over the 250 mm bay at a 15 K rise, against the recommended openings in series
    dp = RHO * 9.81 * 0.25 * 15 / 300
    cda = 1 / math.sqrt(1 / (0.62 * (LID + LEAK)) ** 2 + 1 / (0.82 * (2 * (412 + 126) - 160) * FEET) ** 2 + KF / (406 * 120.0) ** 2)
    q = cda * 1e-6 * math.sqrt(2 * dp / RHO)
    out["fans_off"] = {"draft_pa": round(dp, 2), "flow_ls": round(q * 1e3, 1), "heat_w_at_15K": round(q * RHO * CP * 15)}
    out["pad"] = pad_layout()
    return out


if __name__ == "__main__":
    r = run()
    if "--json" in sys.argv:
        print(json.dumps(r, indent=1))
    else:
        print("areas mm2:", r["areas"])
        for row in r["cases"]:
            print(f"\n{row['name']}")
            for lv in ("low", "nominal", "high"):
                d = row[lv]
                print(f"  fans {lv:8s} {d['flow_ls']:5.1f} L/s {d['flow_cfm']:3d} CFM  bay {d['bay_pa']:+.2f} Pa  rise at {HEAT['low']:.0f}/{HEAT['nominal']:.0f}/{HEAT['high']:.0f} W: "
                      f"{d['rise_K']['low']} / {d['rise_K']['nominal']} / {d['rise_K']['high']} K   {d['flows']}")
        for k, rows in r["sweeps"].items():
            print(f"\nrecommended layout, {k} varied (L/s at low / nominal / high fans):")
            for x in rows:
                print(f"  {x['x']:>6}: {x['low']:5.1f} {x['nominal']:5.1f} {x['high']:5.1f}")
        print("\nfans off:", r["fans_off"])
        print("\nwhole pad under the case, 30 mm clear under the pad: L/s between the boards / onto the outer faces / past the case, and m/s between the boards (nominal fans)")
        for row in r["pad"]["gaps"]:
            n = row["nominal"]
            print(f"  gap {row['gap']:3d}: {n['between']:5.1f} / {n['outer']:4.1f} / {n['missed']:4.1f}   {n['speed']:.2f} m/s   (low {row['low']['between']:.1f}, high {row['high']['between']:.1f} between)")
        print("clearance under the pad mm -> total pad flow L/s (low / nominal / high):", [(c["x"], c["low"], c["nominal"], c["high"]) for c in r["pad"]["clearance"]])
        print("small fans at y +-45 / 55 / 70 -> L/s between the boards at gap 100..180:", r["pad"]["shift"])
