"""Scout cage v14, the funnel cage (3 Nov, Spark: a cage that fits the funnel perfectly, fully enclosed, designed like the
Overseer's, all 3 mm rods). A closed triangulated shell of 3.0 carbon rods in printed PETG nodes, formed to the Scout as it sits
in the shared funnel (parts/box_funnel/make.py). Drone frame = seat frame (x nose, y left, z up).
Rules (checked below, every rod):
  prop     every rod >= 12 from the four swept discs (r 44.45 + 4 sweep, z 41.2 +- 4): the Overseer's keep-out
  body     every rod >= 3 from every part of the Scout (cage_margin_mm)
  camera   no rod in the camera's view (89 x 73 deg from the lens (149, 0, 47), looking +x): the nose quad is an opening
  funnel   the bottom loop stays outside the funnel's rim and legs through the descent: at the seat it is at z -46, under the
           mouth (z -4.3) and outside the legs (x -128 / +131, y +-40) and the rim (x +-131, y +-65); 28 off in x it is 8 from
           the rim, 20 off in y 7 (tools/seat_check.py confirms the seated case against the real funnel)
  shell    closed but for two declared openings: the bottom (inside the base loop: the belly skirt, the funnel) and the nose window
Three rings: the base loop (z -46, the feet, TPU caps O16 x 3 under its 4 nodes), the prop ring (z 41.2, outside the discs by 12)
and the flat roof (z 70, 12 over the sweep), triangulated between them; the flat roof, raked nose and camera window keep the
race-car look of layout v13. Writes parts/scout_cage_v14/part.stl (tubes + node spheres, the look), the shell into
systems/scout.assembly.json (cage_shell, the part entry) and prints the checks. Run: python tools/scout_cage.py
"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ZB, ZM, ZT = -46.0, 41.2, 76.0
NODES = {
    "B0": (155, 92, ZB), "B1": (-155, 92, ZB), "B2": (-155, -92, ZB), "B3": (155, -92, ZB),                # base loop, the feet
    "NB": (178, 0, -12), "TB": (-192, 0, -12),                                                               # nose and tail bottom nodes (under the camera's view)
    "M0": (186, 76, ZM), "M1": (0, 142, ZM), "M2": (-202, 76, ZM), "M3": (-202, -76, ZM), "M4": (0, -142, ZM), "M5": (186, -76, ZM),   # prop ring
    "TT": (-206, 0, 60),                                                                                     # tail face centre
    "T0": (160, 56, ZT), "T1": (0, 102, ZT), "T2": (-186, 56, ZT), "T3": (-186, -56, ZT), "T4": (0, -102, ZT), "T5": (160, -56, ZT),   # roof ring
    "RC": (82, 0, ZT), "RR": (-96, 0, ZT),                                                                   # roof centre nodes (no crossing chords)
}
RODS = [("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B0"),                                              # base loop
        ("M0", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "M4"), ("M4", "M5"),                                  # prop ring (open at the nose: the window)
        ("T0", "T1"), ("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5"), ("T5", "T0"),                    # roof ring
        ("B0", "M0"), ("B0", "M1"), ("B1", "M1"), ("B1", "M2"), ("B2", "M3"), ("B2", "M4"), ("B3", "M4"), ("B3", "M5"),   # lower band
        ("NB", "B0"), ("NB", "B3"), ("NB", "M0"), ("NB", "M5"), ("TB", "B1"), ("TB", "B2"), ("TB", "M2"), ("TB", "M3"),   # nose and tail, lower band
        ("M0", "T0"), ("M1", "T1"), ("M2", "T2"), ("M3", "T3"), ("M4", "T4"), ("M5", "T5"),                    # posts
        ("M0", "T1"), ("M5", "T4"), ("M1", "T2"), ("M4", "T3"),                                                # upper band diagonals, mirrored
        ("TT", "M2"), ("TT", "M3"), ("TT", "T2"), ("TT", "T3"),                                                # tail face
        ("T1", "T4"), ("RC", "T0"), ("RC", "T5"), ("RC", "T1"), ("RC", "T4"), ("RR", "T1"), ("RR", "T2"), ("RR", "T3"), ("RR", "T4")]   # roof
OPEN = [dict(ring=["B0", "B1", "B2", "B3"], what="bottom: the belly skirt into the funnel"),
        dict(ring=["NB", "M0", "T0", "T5", "M5"], what="nose window: the camera's view")]
MOTORS = [(54.1, 54.1), (-54.1, 54.1), (-54.1, -54.1), (54.1, -54.1)]
R_DISC, Z_DISC = 44.45 + 4.0, (37.2, 45.2)
LENS, HFOV, VFOV = np.array([149.0, 0.0, 47.0]), math.radians(44.5), math.radians(36.5)


def seg_pts(a, b, n=60):
    a, b = np.array(a, float), np.array(b, float)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def disc_dist(p):
    """distance from a point to the nearest swept disc solid (cylinder r R_DISC between Z_DISC)"""
    best = 1e9
    for mx, my in MOTORS:
        dr = max(0.0, math.hypot(p[0] - mx, p[1] - my) - R_DISC)
        dz = max(0.0, Z_DISC[0] - p[2], p[2] - Z_DISC[1])
        best = min(best, math.hypot(dr, dz))
    return best


def in_view(p):
    d = p - LENS
    if d[0] <= 0:
        return False
    return abs(math.atan2(d[1], d[0])) < HFOV and abs(math.atan2(d[2], d[0])) < VFOV


def checks():
    import seat_envelope as se
    body = [(nm, m) for nm, m in se.parts_of("scout") if "cage" not in nm.lower() and "legs" not in nm.lower()]
    worst = dict(prop=(1e9, ""), body=(1e9, ""), camera=[])
    for i, j in RODS:
        pts = seg_pts(NODES[i], NODES[j])
        d = min(disc_dist(p) for p in pts)
        if d < worst["prop"][0]:
            worst["prop"] = (d, f"{i}-{j}")
        for nm, m in body:
            if not ((pts.min(0) - 5 <= m.bounds[1]).all() and (pts.max(0) + 5 >= m.bounds[0]).all()):
                continue
            dd = float(trimesh.proximity.closest_point(m, pts)[1].min())
            if dd < worst["body"][0]:
                worst["body"] = (dd, f"{i}-{j} vs {nm}")
        if any(in_view(p) for p in pts):
            worst["camera"].append(f"{i}-{j}")
    return worst


def shell_report(spec):
    import system_check as sc
    return sc.cage_shell({"machine": "scout", "cage_shell": spec}, [{"role": "cage"}])


def mesh():
    parts = []
    for i, j in RODS:
        parts.append(trimesh.creation.cylinder(radius=1.5, segment=[NODES[i], NODES[j]], sections=12))
    for p in NODES.values():
        s = trimesh.creation.icosphere(subdivisions=1, radius=8.0); s.apply_translation(p); parts.append(s)
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    keys = list(NODES); idx = {k: n for n, k in enumerate(keys)}
    L = [float(np.linalg.norm(np.array(NODES[i]) - np.array(NODES[j]))) for i, j in RODS]
    print(f"cage v14: {len(NODES)} nodes, {len(RODS)} rods, {sum(L) / 1000:.2f} m of 3.0 rod ({sum(L) * 0.011:.0f} g), shortest {min(L):.0f}, longest {max(L):.0f}; "
          f"extents {max(n[0] for n in NODES.values()) - min(n[0] for n in NODES.values()):.0f} x {2 * max(n[1] for n in NODES.values()):.0f} x {ZT - ZB:.0f}")
    w = checks()
    print(f"  prop keep-out: nearest rod {w['prop'][0]:.1f} mm ({w['prop'][1]}; need 12)")
    print(f"  body margin:   nearest rod {w['body'][0]:.1f} mm ({w['body'][1]}; need 3)")
    print(f"  camera view:   {'clear' if not w['camera'] else 'BLOCKED by ' + ', '.join(w['camera'])}")
    spec = dict(nodes=[list(map(float, NODES[k])) for k in keys], rods=[[idx[i], idx[j]] for i, j in RODS], node_r_mm=8.0, rod_r_mm=1.5,
                total_rods=len(RODS), open=[dict(ring=[idx[k] for k in o["ring"]], what=o["what"]) for o in OPEN],
                note="cage v14, the funnel cage (tools/scout_cage.py, 3 Nov): a closed 3 mm rod shell formed to the Scout in the shared funnel")
    try:
        r = shell_report(spec)
        print("  shell:         " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("tris",)))
    except Exception as e:
        print("  shell:         (system_check unavailable:", e, ")")
    m = mesh(); d = os.path.join(ROOT, "parts", "scout_cage_v14"); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
    F = os.path.join(ROOT, "systems", "scout.assembly.json"); raw = open(F).read(); a = json.loads(raw)
    a["cage_shell"] = spec
    a["parts"] = [p for p in a["parts"] if p["id"] not in ("scout_cage_pattern", "scout_cage_v14")]
    a["parts"].append(dict(id="scout_cage_v14", name="cage v14 (funnel cage, 3 mm rods, layout model)", role="cage", group="cage", color="#2b2b2b",
                           mesh="parts/scout_cage_v14/part.stl", mass_g=round(sum(L) * 0.011 + len(NODES) * 5.0, 1), tag="E", owner="scout v3",
                           version="v14 (3 Nov)", source="tools/scout_cage.py",
                           note=f"{len(NODES)} nodes, {len(RODS)} rods 3.0 ({sum(L) / 1000:.2f} m), base loop z -46 with TPU feet, prop ring z 41 outside the discs by 12, flat roof z 70, nose window for the camera; nodes E 5 g each until designed"))
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    print("  written: parts/scout_cage_v14/part.stl and systems/scout.assembly.json cage_shell")
