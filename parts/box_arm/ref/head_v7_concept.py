"""Heads A / B v7 grabber concept (24 Oct 2026, swap arm system worker; Spark: plug / unplug loads, every piece optimised for the parts we
use, the good ideas from servo gripper designs: `claude/research/2026-10-24-servo-gripper-head-v7.md`). Numbers C geometry from the latch's
grabber interface (dimensions "System box swap: latch") and the arm (dimensions "System box swap: arm"), E forces and stiffnesses.

Chosen (A): two blades, each sprung closed by a bought steel compression spring onto the cartridge's tabs (squeezed = unlatched from the
receiver and held), opened by one servo cam between them (the two blade forces cancel on the spline), each blade on a guide boss (16 long
in x, in line with the tab) sliding on two 3.0 carbon rods along x fixed in the head body's walls (concept review r3: a short bush racks), so
the 50..74 N pull and push go cartridge -> grip pocket -> blade -> boss -> rods -> head body, never through the servo; the bought metal horn
on the servo spline, no outboard pin (the spline sees only the friction imbalance). The grip holds with the servo unpowered.
Alternatives: (B) servo-driven toggle fingers (the servo closes them past dead centre onto a stop), (C) the v4 gripper (servo-driven
fingers, the servo holds the grip), (D) A + a push pad under the XT60 end.
python3 head_v7_concept.py -> runs/head/v7-concept/concept.html (the visual: states and alternatives, to scale) + concept.json"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "runs", "head", "v7-concept"); os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- the latch's grabber interface (C, latch 258becf) and loads (E)
RECESS_X, RECESS_Y = 15.5, 12.0          # recess x +-15.5, y 0..12 (cartridge coords, y up from the flat bottom)
TAB_LATCHED, TAB_SQUEEZED = (4.0, 10.0), (0.0, 6.0)   # tab inner / outer faces, x
POCKET = dict(y=4.0, h=4.0, depth=3.0, draft_deg=5.0)
RELEASE_N = dict(scout=3.0, overseer=4.5)               # per finger to squeeze
HOLD_N = dict(scout=52.1, overseer=57.3)                # while the XT60 unmates (both tabs)
TAB_SPRING_SQUEEZED = 2.4                               # the latch's own spring at squeezed, N
F_STALL, F_INSERT = 74.0, 60.0
BLADE = dict(t=3.0, w=13.0)                             # as the latch's interface (3 thick, 13 wide)
# ---------------------------------------------------------------- chosen A (DEFAULT)
OPEN_X = TAB_LATCHED[1] + 0.5                           # blade inner face clear of the tab's outer face at rest: 10.5
STROKE = OPEN_X - TAB_SQUEEZED[1]                       # 4.5 per blade
SPRING = dict(k=1.0, closed=15.0)                       # N/mm, N at squeezed (bought spring, e.g. 0.8 wire OD 6, E)
CAM = dict(base=4.5, rise=6.0, turn_deg=90.0, mu=0.3)  # base 4.5 + rise 6.0 = 10.5 = OPEN_X (concept review r3: 5.0 stopped 1 short); a dwell at open
SERVO = dict(DS3218=dict(kgcm=20.0, body=(40.5, 20.0, 40.7)), MG995=dict(kgcm=9.0, body=(40.7, 19.7, 42.9)))   # 4.8..5 V, seller E
ROD = dict(d=3.0, span=52.0, sigma=1000.0, E=120000.0, per_blade=2, boss=16.0, offset=25.0, mu=0.3)   # carbon rods along x between the head
                                        # body's walls (x +-26), the blade boss 16 long on both; the tab reaction ~25 off the rods (C / E)
BOSS_IN = 5.0                            # each guide boss starts 5.0 inboard of its blade's inner face and runs outboard (concept review r3d: centred
                                         # bosses met at x 0 and the cam): squeezed x 1..17 / -17..-1 (2.0 apart), open 5.5..21.5, inside the walls +-26;
BOSS_Y = (-11.0, -5.0)                   # boss and rods (y -8) above the cam's top (y -14.5): 3.5 clear
Y_C, PLATE_TOP = 165.0, 72.75
XT60_Y = dict(scout=121.75, overseer=86.7)


def numbers():
    f_open = SPRING["closed"] + SPRING["k"] * STROKE
    lever = CAM["rise"] / math.radians(CAM["turn_deg"])            # effective cam lever, mm
    torque = 2 * f_open * lever * (1 + CAM["mu"]) / 1e3            # N.m: both blades on one cam, friction
    r = {}
    r["cam"] = dict(base=CAM["base"], rise=CAM["rise"], open_reach=CAM["base"] + CAM["rise"], open_x=OPEN_X, ok=bool(CAM["base"] + CAM["rise"] >= OPEN_X))
    r["blade"] = dict(open_x=OPEN_X, squeezed_x=TAB_SQUEEZED[1], stroke=STROKE, recess_gap=round(RECESS_X - TAB_LATCHED[1], 1),
                      fits_open=bool(OPEN_X + BLADE["t"] <= RECESS_X - 1.0))
    r["spring"] = dict(closed_N=SPRING["closed"], open_N=round(f_open, 1),
                       release_margin={p: round(SPRING["closed"] / RELEASE_N[p], 1) for p in RELEASE_N})
    hold = {}
    for p, H in HOLD_N.items():
        per = H / 2; opening = per * math.tan(math.radians(POCKET["draft_deg"])) + TAB_SPRING_SQUEEZED
        hold[p] = dict(per_tab_N=round(per, 1), opening_N=round(opening, 2), margin=round(SPRING["closed"] / opening, 1))
    r["hold_while_unmating"] = hold
    r["servo"] = dict(cam_lever_mm=round(lever, 2), torque_Nm=round(torque, 3), torque_kgcm=round(torque * 10.2, 2),
                      margin={s: round(v["kgcm"] / (torque * 10.2), 1) for s, v in SERVO.items()},
                      spline_side_N=round(CAM["mu"] * f_open, 1), note="the two blades load the cam equal and opposite: the spline sees only "
                      "the friction imbalance (no outboard pin); at squeezed the cam is off the blades (the springs hold, servo unloaded)")
    per_rod = max(HOLD_N.values()) / 2 / ROD["per_blade"]
    M = per_rod * ROD["span"] / 4; W = math.pi * ROD["d"] ** 3 / 32; I = math.pi * ROD["d"] ** 4 / 64
    r["bosses"] = dict(squeezed_x=(TAB_SQUEEZED[1] - BOSS_IN, TAB_SQUEEZED[1] - BOSS_IN + ROD["boss"]), open_x=(OPEN_X - BOSS_IN, OPEN_X - BOSS_IN + ROD["boss"]),
                       gap_at_centre=round(2 * (TAB_SQUEEZED[1] - BOSS_IN), 1), y=BOSS_Y, cam_top_y=-19.0 + CAM["base"],
                       cam_clear=round(BOSS_Y[0] - (-19.0 + CAM["base"]), 1), inside_walls=bool(OPEN_X - BOSS_IN + ROD["boss"] <= 26.0))
    r["rods"] = dict(per_rod_N=round(per_rod, 1), sigma_MPa=round(M / W, 1), index=round(M / W / ROD["sigma"], 3),
                     deflection_mm=round(per_rod * ROD["span"] ** 3 / (48 * ROD["E"] * I), 3), boss_MPa=round(per_rod / (ROD["d"] * ROD["boss"]), 2),
                     anti_rack_boss_min=round(2 * ROD["mu"] * ROD["offset"], 1), boss=ROD["boss"], ok=bool(ROD["boss"] >= 2 * ROD["mu"] * ROD["offset"]))
    r["head_body"] = dict(grip_lever_mm=round(Y_C - PLATE_TOP, 1), stall_Nm=round(F_STALL * (Y_C - PLATE_TOP) / 1e3, 2),
                          pull_Nm=round(max(HOLD_N.values()) * (Y_C - PLATE_TOP) / 1e3, 2),
                          pad_insert_Nm={p: round(F_STALL * (y - PLATE_TOP) / 1e3, 2) for p, y in XT60_Y.items()},
                          note="alt D: a push pad under the XT60 end cuts the insert moment, the pull still loads the grip (5.3 N.m); "
                               "frame model: the mate tilt barely moves (Scout 0.822 -> 0.817 deg), so D is not taken")
    r["pinion"] = dict(face_now=10.5, face_v3=13.5, tail_face_v3=10.25, swap_drone_insert=dict(scout=0.447, overseer=0.486),
                       stall_misuse=dict(scout=0.572, overseer=0.611), note_src="swap_system frame (round 3d), the drone insert on the rack tail",
                       note="research: module 1 is the printable limit; pinion v3 teeth Y 59.75..73.25 on rack teeth Y 59.25..73.25 (grown "
                            "down: the top stays under the carriage-lane keep-out 73.75, concept review r3); on the rack tail (drone insert) "
                            "the engaged face is 10.25 from the tail's step at Y 63 (swap_system)")
    r["alternatives"] = {
        "A sprung blades + servo cam (chosen)": dict(holds_unpowered=True, servo_sees_pull=False, servo_torque_kgcm=round(torque * 10.2, 2),
                                                     parts="2 springs, 2 carbon rods along x, 2 blade guide bosses 16 long, metal horn"),
        "B servo toggle fingers": dict(holds_unpowered=True, servo_sees_pull=False, servo_torque_kgcm="~0.5 at the squeeze, rises near dead centre",
                                       parts="4 pins + links per finger", note="needs a tuned stop per pack (the Scout / Overseer tabs differ "
                                       "in release force); a stop error loses the lock or overloads the servo; more parts"),
        "C v4 servo fingers": dict(holds_unpowered=False, servo_sees_pull=True, note="the servo holds the grip; a power cut drops the pack; "
                                   "the spline takes the pull side load (no published limit)"),
        "D A + push pad": dict(holds_unpowered=True, servo_sees_pull=False, note="a pad under the XT60 end: the insert moment on the head body "
                               "drops (Scout 6.8 -> 3.6 N.m) but the pull still loads the grip (5.3) and the mate tilt barely moves (0.822 -> 0.817 deg): "
                               "an extra contact for little; not taken"),
    }
    return r


def svg_state(x0, title, blade_x, cam_deg, latched, sub):
    """section through the tab line (x across, y up = radial), to scale (4 px / mm)"""
    s = 4.0; ox, oy = x0 + 120, 250
    X = lambda x: ox + x * s; Yp = lambda y: oy - y * s
    g = [f'<g><text x="{ox}" y="22" class="h" text-anchor="middle">{title}</text>']
    for k, ln in enumerate(sub):
        g.append(f'<text x="{ox}" y="{482 + 15 * k}" class="s" text-anchor="middle">{ln}</text>')
    g.append(f'<rect x="{X(-25)}" y="{Yp(40)}" width="{50 * s}" height="{40 * s}" class="cart"/>')               # cartridge (part)
    g.append(f'<rect x="{X(-RECESS_X)}" y="{Yp(RECESS_Y)}" width="{2 * RECESS_X * s}" height="{RECESS_Y * s}" class="recess"/>')
    ti, to = TAB_LATCHED if latched else TAB_SQUEEZED
    for sg in (-1, 1):
        a, b = sorted((sg * ti, sg * to))
        g.append(f'<rect x="{X(a)}" y="{Yp(11.5)}" width="{(b - a) * s}" height="{11.5 * s}" class="tab"/>')
        if latched:
            g.append(f'<rect x="{X(sg * 10.0 if sg > 0 else -12.3)}" y="{Yp(9)}" width="{2.3 * s}" height="{3 * s}" class="bolt"/>')
        bi = sg * blade_x; bo = sg * (blade_x + BLADE["t"])
        a, b = sorted((bi, bo))
        g.append(f'<rect x="{X(a)}" y="{Yp(10)}" width="{(b - a) * s}" height="{(10 + 22) * s}" class="blade"/>')    # blade into the recess
        a2, b2 = sorted((sg * (blade_x - BOSS_IN), sg * (blade_x - BOSS_IN + ROD["boss"])))       # boss outboard: x 1..17 squeezed, 5.5..21.5 open
        g.append(f'<rect x="{X(a2)}" y="{Yp(-5)}" width="{ROD["boss"] * s}" height="{6 * s}" class="boss"/>')          # guide boss 16 long, y -11..-5
        sx = sg * (blade_x + BLADE["t"]); se = sg * 24.0
        a, b = sorted((sx, se))
        g.append(f'<rect x="{X(a)}" y="{Yp(-20)}" width="{(b - a) * s}" height="{4 * s}" class="spring"/>')
    g.append(f'<rect x="{X(-27)}" y="{Yp(-12)}" width="{54 * s}" height="{24 * s}" class="body" fill-opacity="0.18"/>')    # head body
    g.append(f'<rect x="{X(-26)}" y="{Yp(-7.25)}" width="{52 * s}" height="{1.5 * s}" class="rod"/>')                  # the rods along x at y -8
    g.append(f'<text x="{X(0)}" y="{Yp(-33)}" class="s" text-anchor="middle">head body</text>')
    r0 = CAM["base"] + CAM["rise"] * min(cam_deg, 90) / 90
    g.append(f'<ellipse cx="{X(0)}" cy="{Yp(-19)}" rx="{(r0 if cam_deg else CAM["base"]) * s}" ry="{CAM["base"] * s}" class="cam"/>')
    g.append(f'<rect x="{X(-22)}" y="{Yp(-40)}" width="{44 * s}" height="{14 * s}" class="servo"/>')
    g.append(f'<text x="{X(0)}" y="{Yp(-49)}" class="s servo-t" text-anchor="middle">servo (DS3218 A / MG995 B)</text>')
    g.append(f'<text x="{X(0)}" y="{Yp(30)}" class="s" text-anchor="middle">cartridge</text>')
    g.append("</g>")
    return "\n".join(g)


def page(r):
    states = [svg_state(0, "1. approach: open", OPEN_X, 90, True, ["cam on its dwell: blades out", "tabs latched in the receiver"]),
              svg_state(250, "2. hold: squeezed", TAB_SQUEEZED[1], 0, False, ["springs squeeze the tabs: unlatched", "and held; the servo is unloaded"]),
              svg_state(500, "3. release at the seat", OPEN_X, 90, True, ["cam opens: the tabs spring into", "the receiver; the head retracts"])]
    alt = r["alternatives"]
    rows = "".join(f"<tr><td>{k}</td><td>{'yes' if v.get('holds_unpowered') else ('no' if 'holds_unpowered' in v else '')}</td>"
                   f"<td>{'no' if v.get('servo_sees_pull') is False else ('yes' if v.get('servo_sees_pull') else '')}</td>"
                   f"<td>{v.get('servo_torque_kgcm', '')}</td><td>{v.get('parts', v.get('note', ''))}</td></tr>" for k, v in alt.items())
    sv = r["servo"]; sp = r["spring"]; h = r["hold_while_unmating"]; rd = r["rods"]; hb = r["head_body"]; pn = r["pinion"]
    nums = [("Spring per blade", f"{sp['closed_N']} N squeezed, {sp['open_N']} N open (bought steel, k 1.0 N/mm, E)"),
            ("Release margin", f"Scout x{sp['release_margin']['scout']}, Overseer x{sp['release_margin']['overseer']} over the tabs' 3.0 / 4.5 N"),
            ("Hold while the XT60 unmates", f"Scout {h['scout']['per_tab_N']} N per tab, opening {h['scout']['opening_N']} N, x{h['scout']['margin']}; "
                                           f"Overseer {h['overseer']['per_tab_N']} N, {h['overseer']['opening_N']} N, x{h['overseer']['margin']}"),
            ("Servo", f"cam base 4.5 + rise 6.0 over 90 deg = 10.5, the open blade position (lever {sv['cam_lever_mm']} mm): {sv['torque_kgcm']} kg·cm to open; "
                      f"margin DS3218 x{sv['margin']['DS3218']}, MG995 x{sv['margin']['MG995']}; spline side load ≤ {sv['spline_side_N']} N (friction imbalance)"),
            ("Pull path", f"pocket → blade → guide boss 16 long (≥ {rd['anti_rack_boss_min']} to slide without racking) → 2 x 3.0 carbon rods along x "
                          f"({rd['per_rod_N']} N each, {rd['sigma_MPa']} MPa, index {rd['index']}, {rd['deflection_mm']} mm) → head body walls"),
            ("Head body (to its topology run)", f"{hb['pull_Nm']} N·m pull / {hb['stall_Nm']} N·m stall at the grip, lever {hb['grip_lever_mm']} from the plate"),
            ("Pinion v3", f"face {pn['face_now']} → {pn['face_v3']} (rack grown down; {pn['tail_face_v3']} on the rack tail at the drone insert): Lewis at the 60 N "
                          f"drone insert Scout {pn['swap_drone_insert']['scout']}, Overseer {pn['swap_drone_insert']['overseer']} (≤ 0.5); at the 74 N stall "
                          f"{pn['stall_misuse']['scout']} / {pn['stall_misuse']['overseer']} (misuse ≤ 1.0)")]
    nrows = "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in nums)
    return f"""<title>Heads v7 grabber concept</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root {{ --bg:#f6f5f1; --fg:#1d232a; --muted:#5d6670; --line:#c9ccd0; --cart:#dfe7d8; --recess:#f6f5f1; --tab:#6c8f5a; --blade:#2f6db3;
  --rod:#222; --spring:#c47a1d; --body:#7a8794; --cam:#b33a3a; --servo:#3d4a57; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181c; --fg:#e7eaee; --muted:#9aa4ad; --line:#323a42;
  --cart:#2b3a2b; --recess:#14181c; --tab:#8fbf74; --blade:#6aa5ec; --rod:#ddd; --spring:#e3a24a; --body:#93a1ae; --cam:#e06a6a;
  --servo:#4d5d6d; --card:#1b2026; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg:#14181c; --fg:#e7eaee; --muted:#9aa4ad; --line:#323a42; --cart:#2b3a2b; --recess:#14181c; --tab:#8fbf74;
  --blade:#6aa5ec; --rod:#ddd; --spring:#e3a24a; --body:#93a1ae; --cam:#e06a6a; --servo:#4d5d6d; --card:#1b2026; color-scheme: dark }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 "IBM Plex Sans", system-ui, sans-serif; margin:0;
  padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:20px; }}
main {{ max-width:1100px; margin:0 auto; display:grid; gap:20px; }}
h1 {{ font-size:1.5rem; margin:0; text-wrap:balance }} h2 {{ font-size:1.1rem; margin:0 0 8px }}
p {{ margin:0; color:var(--muted); max-width:70ch }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; min-width:0 }}
.scroll {{ overflow-x:auto }}
svg .h {{ font-weight:600; font-size:14px; fill:var(--fg) }} svg .s {{ font-size:11px; fill:var(--muted) }}
.cart {{ fill:var(--cart) }} .recess {{ fill:var(--recess) }} .tab {{ fill:var(--tab) }} .bolt {{ fill:var(--tab); opacity:.6 }}
.blade {{ fill:var(--blade) }} .rod {{ fill:var(--rod) }} .spring {{ fill:var(--spring) }} .body {{ fill:var(--body) }}
.cam {{ fill:var(--cam) }} .servo {{ fill:var(--servo) }} .boss {{ fill:var(--blade); opacity:.55 }} svg .servo-t {{ fill:#fff }}
table {{ border-collapse:collapse; width:100%; font-size:14px }} th, td {{ text-align:left; padding:6px 8px; border-top:1px solid var(--line); vertical-align:top }}
th {{ font-weight:600; white-space:nowrap }} .legend span {{ display:inline-block; margin-right:14px }} .sw {{ display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:4px }}
</style>
<main>
<header><h1>Heads v7 grabber concept</h1>
<p>Nox swap arm, both heads. The plug and unplug loads (60 to 74 N push, 50 to 57 N pull) go through the grip pocket, the blades and carbon rods into the head body. Springs hold the grip; the servo only opens it. Section through the tab line, to scale (4 px per mm), latch 258becf grabber interface.</p></header>
<section class="card"><h2>One swap, three states</h2><div class="scroll"><svg viewBox="0 0 750 520" width="750" height="520" role="img" aria-label="grabber states">
{''.join(states)}
</svg></div>
<div class="legend"><span><i class="sw" style="background:var(--tab)"></i>cartridge tabs</span><span><i class="sw" style="background:var(--blade)"></i>blades (PETG 3 x 13) and guide bosses</span><span><i class="sw" style="background:var(--rod)"></i>3.0 carbon rods along x</span><span><i class="sw" style="background:var(--spring)"></i>steel springs</span><span><i class="sw" style="background:var(--cam)"></i>servo cam</span><span><i class="sw" style="background:var(--body)"></i>head body</span></div></section>
<section class="card"><h2>Numbers</h2><table>{nrows}</table></section>
<section class="card"><h2>Alternatives</h2><div class="scroll"><table><tr><th>Option</th><th>Holds unpowered</th><th>Servo sees the pull</th><th>Servo kg·cm</th><th>Parts / why not</th></tr>{rows}</table></div></section>
<section class="card"><p>Source: <code>nox/box/head_v7_concept.py</code>; research <code>claude/research/2026-10-24-servo-gripper-head-v7.md</code>; requirements <code>nox/dimensions.md</code> "System box swap: arm". Concept only: the head body comes from its topology run on the frame's reactions after the concept check.</p></section>
</main>"""


if __name__ == "__main__":
    r = numbers()
    json.dump(r, open(os.path.join(OUT, "concept.json"), "w"), indent=1)
    open(os.path.join(OUT, "concept.html"), "w").write(page(r))
    print(json.dumps({k: v for k, v in r.items() if k != "alternatives"}, indent=1))
    print("->", os.path.join(OUT, "concept.html"))
