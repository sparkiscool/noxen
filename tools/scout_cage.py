"""Scout cage v17, the compact funnel cage on skid trusses (3 Nov, Spark: v14 was far too large; meet every requirement and make the Scout
as compact as possible; 4 Nov: no standalone legs, the cage itself lands, like the Overseer's loop; then "the bottom is still not
triangulated enough, we are printing all nodes in PETG, lets do a stiffness check": tools/scout_cage_loads.py). A closed triangulated shell of 3.0 carbon rods in printed PETG nodes, formed to the Scout as it sits in the
shared funnel (parts/box_funnel/make.py). Drone frame = seat frame (x nose, y left, z up).
What sets the size, and only this:
  prop     every rod >= 12 from the four swept discs (r 44.45 + 4, z 41.2 +- 4): a rounded square of half-width 114.6 with corner
           arcs; the ring is the octagon tangent to it, flats at |y| 115, corner chamfers on x + y = 193.7
  nose     the camera just ahead of the front discs (lens (122, 0, 40), board x 105..120): the nose flat at x 136, the window
           between the dipped front nodes (z 20) and the roof, nothing in the 89 x 73 deg view
  tail     the tail electronics now sit under the plate's tail (x -118..-53, z 7.5..20, over the funnel mouth): the tail flat at x -130
  roof     one ring at z 62 is both the prop keep-out ring and the roof's edge (17 over the sweep, 3 over the camera board):
           no posts; the roof is a spine of two nodes fanned to the ring
  bottom   the whole shell stays above z 0, over the funnel's wall tops (z -4.3) in every state of the descent (the end walls
           sweep |x| <= 153, |y| <= 85 at the capture limits): base rails at |y| 115, z 0, nose and tail bottom nodes at z 8, the
           window sill at z 17
  skids    the cage lands on two skids at |y| 84, z -48 (3.5 under the belly skirt and the cartridge; a TPU strip under each), each
           hung from its base rail as a Warren truss (v17: mid nodes on the rail and the skid, end and mid posts, diagonals from the
           rail's mid node to the skid's ends; the frame model on v16's single-diagonal panel had the 200 rails buckling at 1.10 and
           the panels hanging on socket bending at 93 % of the cap), braced out of plane by rods from the skid ends to the window's
           top corners (nose) and the ring's tail nodes (the only cage nodes a rod from a skid end reaches outside the prop keep-out
           and clear of the funnel's walls: via the dipped front nodes it grazed the nose wall's top corner). Outboard of the funnel in every
           state of the descent: a point at seated height h over the seat clears the y walls (to h 28, 55 deg) through the +-20
           descent only beyond |y| 77 - 0.7 h, so the skids need |y| >= 80 and the panels lean outward and up to the rails at 115;
           nothing can cross under the belly (the funnel's seat ring and skirt, the arm's head) or at the ends below the wall tops
           (the end walls to h 40, the tail beam's buttress), so the belly, nose V and tail V are one declared opening
  funnel   nothing of the cage inside the funnel's walls when seated or during the +-28 / +-20 descent (tools/seat_check.py)
  shell    closed, mirror-exact, declared openings: the belly between the skids, its nose and tail ends, the nose and tail Vs, the nose window
Writes parts/scout_cage_v17/part.stl (tubes + node spheres, the look), the shell into systems/scout.assembly.json (cage_shell, the
part entry) and prints the checks. Run: python tools/scout_cage.py
"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ZR, ZB, ZF, ZK, YK = 62.0, 0.0, 17.0, -48.0, 84.0     # the ring (prop keep-out and roof edge), the base rails, the dipped front nodes, the skids (z, |y|)
XN, XT, YS, C = 136.0, -130.0, 115.0, 193.7          # nose flat, tail flat, side flats, the corner chamfer line |x| + |y| = C
NODES = {
    "P0": (XN, C - XN, ZF), "P7": (XN, -(C - XN), ZF),                                                         # front nodes, dipped: the window's bottom corners
    "R0": (XN, C - XN, ZR), "R7": (XN, -(C - XN), ZR),                                                         # front nodes, roof: the window's top corners
    "P1": (C - YS, YS, ZR), "P2": (-(C - YS), YS, ZR), "P3": (XT, C + XT, ZR),                                  # the ring, left half (nose -> tail)
    "P4": (XT, -(C + XT), ZR), "P5": (-(C - YS), -YS, ZR), "P6": (C - YS, -YS, ZR),                             # right half
    "RF": (70, 0, ZR), "RB": (-70, 0, ZR),                                                                     # roof spine
    "B0": (100, YS, ZB), "B1": (-100, YS, ZB), "B2": (-100, -YS, ZB), "B3": (100, -YS, ZB),                     # base rails at z 0: above the funnel's walls in every state
    "BM0": (0, YS, ZB), "BM1": (0, -YS, ZB),                                                                   # the rails' mid nodes (v17)
    "K0": (100, YK, ZK), "K1": (-100, YK, ZK), "K2": (-100, -YK, ZK), "K3": (100, -YK, ZK),                     # the skids' ends, under the rail ends
    "KM0": (0, YK, ZK), "KM1": (0, -YK, ZK),                                                                   # the skids' mid nodes (v17)
    "NB": (118, 0, 8), "TB": (-130, 0, 14),                                                                    # nose and tail bottom nodes, 12 / 18 over the wall tops (TB up 6 so the tail ties pass 3 over the Radxa)
}
RODS = [("R0", "P1"), ("P1", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5"), ("P5", "P6"), ("P6", "R7"), ("R7", "R0"),   # the ring at z 62 (the roof's edge)
        ("P0", "R0"), ("P7", "R7"), ("P0", "P1"), ("P7", "P6"), ("P0", "P7"),                                  # the window's posts, the front slants, the window sill (z 17, under the view)
        ("RF", "R0"), ("RF", "R7"), ("RF", "P1"), ("RF", "P6"), ("RF", "RB"),                                  # roof, front
        ("RB", "P1"), ("RB", "P6"), ("RB", "P2"), ("RB", "P5"), ("RB", "P3"), ("RB", "P4"),                    # roof, rear
        ("B0", "BM0"), ("BM0", "B1"), ("B3", "BM1"), ("BM1", "B2"),                                            # base rails, two halves
        ("K0", "KM0"), ("KM0", "K1"), ("K3", "KM1"), ("KM1", "K2"),                                            # the skids, two halves
        ("B0", "K0"), ("B1", "K1"), ("B3", "K3"), ("B2", "K2"), ("BM0", "KM0"), ("BM1", "KM1"),                 # the skid trusses' posts
        ("BM0", "K0"), ("BM0", "K1"), ("BM1", "K3"), ("BM1", "K2"),                                            # the skid trusses' diagonals
        ("K0", "R0"), ("K3", "R7"), ("K1", "P3"), ("K2", "P4"),                                                # the skid ends braced out of plane
        ("B0", "R0"), ("B3", "R7"),                                                                            # close the nose brace's skew quad K0-B0-P0-R0 into triangles
        ("B0", "P0"), ("B0", "P1"), ("B1", "P2"), ("B1", "P3"), ("B2", "P4"), ("B2", "P5"), ("B3", "P6"), ("B3", "P7"),   # lower band
        ("BM0", "P1"), ("BM0", "P2"), ("BM1", "P6"), ("BM1", "P5"),                                            # the side faces fanned from the rails' mid nodes
        ("NB", "B0"), ("NB", "B3"), ("NB", "P0"), ("NB", "P7"), ("TB", "B1"), ("TB", "B2"), ("TB", "P3"), ("TB", "P4")]   # nose and tail, lower band
TIE_ROOTS = {"A0": (62, 62, 17), "A1": (-62, 62, 17), "A2": (-62, -62, 17), "A3": (62, -62, 17),              # clips under the four arm tips (the motor's M2 screws), z 17 under the plate
             "C0": (60, 30, 22), "C1": (-60, 30, 22), "C2": (-60, -30, 22), "C3": (60, -30, 22),               # saddles on the plate's four body corners, on its top face
             "T0": (-64, 12, 22), "T1": (-64, -12, 22)}                                                       # a saddle on the plate's tail end: the tail ties run level over the electronics bay to TB
TIES = [("A0", "B0"), ("A1", "B1"), ("A2", "B2"), ("A3", "B3"), ("C0", "B0"), ("C1", "B1"), ("C2", "B2"), ("C3", "B3"),   # 3.0 rods: the cage hangs on the plate by its four rail corners
        ("T0", "TB"), ("T1", "TB")]   # and the tail node: without these a 35 g tail-first wall hit racked the cage 17 mm with 1157 N in a corner tie (frame model, 4 Nov); with them 0.2 mm, 144 N
OPEN = [dict(ring=["K0", "KM0", "K1", "K2", "KM1", "K3"], what="bottom: the belly skirt into the funnel"), dict(ring=["B0", "B3", "NB"], what="bottom, nose V"), dict(ring=["B1", "TB", "B2"], what="bottom, tail V"),
        dict(ring=["K0", "B0", "B3", "K3"], what="bottom, nose end between the skids"), dict(ring=["K1", "B1", "B2", "K2"], what="bottom, tail end between the skids"),
        dict(ring=["P0", "R0", "R7", "P7"], what="nose window: the camera's view")]
MOTORS = [(54.1, 54.1), (-54.1, 54.1), (-54.1, -54.1), (54.1, -54.1)]
R_DISC, Z_DISC = 44.45 + 4.0, (37.2, 45.2)
LENS, HFOV, VFOV = np.array([122.0, 0.0, 40.0]), math.radians(44.5), math.radians(36.5)


def seg_pts(a, b, n=60):
    a, b = np.array(a, float), np.array(b, float)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def disc_dist(p):
    best = 1e9
    for mx, my in MOTORS:
        dr = max(0.0, math.hypot(p[0] - mx, p[1] - my) - R_DISC)
        dz = max(0.0, Z_DISC[0] - p[2], p[2] - Z_DISC[1])
        best = min(best, math.hypot(dr, dz))
    return best


def in_view(p):
    d = p - LENS
    return d[0] > 0 and abs(math.atan2(d[1], d[0])) < HFOV and abs(math.atan2(d[2], d[0])) < VFOV


def all_nodes():
    return {**NODES, **TIE_ROOTS}


def funnel_check(margin=2.0):
    """every rod and tie, on the whole descent of tools/seat_check.py (the capture limits, heights every 3..5), against the funnel
    mesh: the least distance from a rod axis to the funnel surface, less the rod radius; need >= margin. Fast (330 faces)."""
    import seat_check as skc
    fun = trimesh.load(os.path.join(ROOT, "parts", "box_funnel", "part.stl"))
    pq = trimesh.proximity.ProximityQuery(fun)
    P = all_nodes(); pts = np.vstack([seg_pts(P[i], P[j], 30) for i, j in RODS + TIES])
    tag = np.repeat([f"{i}-{j}" for i, j in RODS + TIES], 30)
    worst = (1e9, "", "")
    for lab, ox, oy, h in skc.descent_states("scout"):
        d = -pq.signed_distance(pts + np.array([ox, oy, h])) - 1.5          # trimesh: positive inside
        k = int(d.argmin())
        if d[k] < worst[0]:
            worst = (float(d[k]), tag[k], lab)
    return worst


def checks():
    import seat_envelope as se
    body = [(nm, m) for nm, m in se.parts_of("scout") if "cage" not in nm.lower() and "legs" not in nm.lower()]
    worst = dict(prop=(1e9, ""), body=(1e9, ""), camera=[])
    P = all_nodes()
    for i, j in RODS + TIES:
        pts = seg_pts(P[i], P[j])
        if i in TIE_ROOTS:
            pts = pts[np.linalg.norm(pts - np.array(P[i], float), axis=1) > 12.0]      # a tie leaves its clip on the plate: the first 12 are the clip
        d = min(disc_dist(p) for p in pts)
        if d < worst["prop"][0]:
            worst["prop"] = (d, f"{i}-{j}")
        for nm, m in body:
            if not ((pts.min(0) - 5 <= m.bounds[1]).all() and (pts.max(0) + 5 >= m.bounds[0]).all()):
                continue
            dd = float(trimesh.proximity.closest_point(m, pts)[1].min()) - 1.5
            if dd < worst["body"][0]:
                worst["body"] = (dd, f"{i}-{j} vs {nm}")
        if any(in_view(p) for p in pts):
            worst["camera"].append(f"{i}-{j}")
    return worst


def shell_report(spec):
    import system_check as sc
    return sc.cage_shell({"machine": "scout", "cage_shell": spec}, [{"role": "cage"}])


def mesh():
    P = all_nodes()
    parts = [trimesh.creation.cylinder(radius=1.5, segment=[P[i], P[j]], sections=12) for i, j in RODS + TIES]
    for p in NODES.values():
        s = trimesh.creation.icosphere(subdivisions=1, radius=7.0); s.apply_translation(p); parts.append(s)
    for p in TIE_ROOTS.values():
        s = trimesh.creation.icosphere(subdivisions=1, radius=4.0); s.apply_translation(p); parts.append(s)
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    keys = list(NODES); idx = {k: n for n, k in enumerate(keys)}; P = all_nodes()
    L = [float(np.linalg.norm(np.array(NODES[i]) - np.array(NODES[j]))) for i, j in RODS]
    xs = [n[0] for n in NODES.values()]; ys = [n[1] for n in NODES.values()]; zs = [n[2] for n in NODES.values()]
    print(f"cage v17: {len(NODES)} nodes, {len(RODS)} rods, {sum(L) / 1000:.2f} m of 3.0 rod ({sum(L) * 0.011:.0f} g), shortest {min(L):.0f}, longest {max(L):.0f}; "
          f"node extents {max(xs) - min(xs):.0f} x {max(ys) - min(ys):.0f} x {max(zs) - min(zs):.0f}")
    w = checks()
    print("  shortest rods: " + ", ".join(f"{i}-{j} {l:.0f}" for l, (i, j) in sorted(zip(L, RODS))[:4]))
    print(f"  prop keep-out: nearest rod {w['prop'][0]:.1f} mm ({w['prop'][1]}; need 12)")
    print(f"  body margin:   nearest rod surface {w['body'][0]:.1f} mm ({w['body'][1]}; need 3)")
    print(f"  camera view:   {'clear' if not w['camera'] else 'BLOCKED by ' + ', '.join(w['camera'])}")
    fw = funnel_check()
    print(f"  funnel:        least clearance {fw[0]:.1f} mm ({fw[1]} at {fw[2]}; need 2, exact gate tools/seat_check.py scout --descent)")
    LT = [float(np.linalg.norm(np.array(P[i]) - np.array(P[j]))) for i, j in TIES]
    print(f"  ties:          {len(TIES)} rods 3.0, {sum(LT) / 1000:.2f} m, {min(LT):.0f}..{max(LT):.0f}")
    spec = dict(nodes=[list(map(float, NODES[k])) for k in keys], rods=[[idx[i], idx[j]] for i, j in RODS], node_r_mm=7.0, rod_r_mm=1.5,
                total_rods=len(RODS) + len(TIES), open=[dict(ring=[idx[k] for k in o["ring"]], what=o["what"]) for o in OPEN],
                ties=dict(roots={k: list(map(float, v)) for k, v in TIE_ROOTS.items()}, rods=[[i, keys.index(j)] for i, j in TIES],
                          what="3.0 rods from clips under the four arm tips (A) and saddles on the plate's body corners (C) to the base rail corners B0..B3, and from a saddle on the plate's tail end (T) to the tail node TB"),
                note="cage v17, the compact funnel cage on skid trusses (tools/scout_cage.py, 4 Nov): a closed 3 mm rod octagonal prism formed to the Scout in the shared funnel, landing on two skids hung from its base rails as Warren trusses, their ends braced to the ring")
    try:
        r = shell_report(spec)
        print("  shell:         " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("tris",)))
    except Exception as e:
        print("  shell:         (system_check unavailable:", e, ")")
    m = mesh(); d = os.path.join(ROOT, "parts", "scout_cage_v17"); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
    F = os.path.join(ROOT, "systems", "scout.assembly.json"); raw = open(F).read(); a = json.loads(raw)
    a["cage_shell"] = spec
    a["parts"] = [p for p in a["parts"] if p["id"] not in ("scout_cage_pattern", "scout_cage_v14", "scout_cage_v15", "scout_cage_v16", "scout_cage_v17")]
    a["parts"].append(dict(id="scout_cage_v17", name="cage v17 (compact funnel cage on skid trusses, 3 mm rods, frame-checked)", role="cage", group="cage", color="#2b2b2b",
                           mesh="parts/scout_cage_v17/part.stl", mass_g=round(sum(L) * 0.011 + len(NODES) * 5.0, 1), tag="E", owner="scout v3",
                           version="v17 (4 Nov)", source="tools/scout_cage.py",
                           note=f"{len(NODES)} nodes, {len(RODS)} rods 3.0 ({sum(L) / 1000:.2f} m), an octagonal prism tangent to the prop keep-out, base rails z 0, two skids at |y| 84, z -48 (TPU strips) as Warren trusses braced to the ring, 8 ties to the plate, roof z 62, nose window for the camera; nodes E 5 g each until designed; frame model tools/scout_cage_loads.py"))
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    print("  written: parts/scout_cage_v17/part.stl and systems/scout.assembly.json cage_shell")
