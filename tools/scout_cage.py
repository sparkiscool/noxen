"""Scout cage v19, the compact funnel cage on skid trusses, lowered, domed and shortened (3 Nov, Spark: v14 was far too large; meet every requirement and make the Scout
as compact as possible; 4 Nov: no standalone legs, the cage itself lands, like the Overseer's loop; then "the bottom is still not
triangulated enough, we are printing all nodes in PETG, lets do a stiffness check": tools/scout_cage_loads.py). A closed triangulated shell of 3.0 carbon rods in printed PETG nodes, formed to the Scout as it sits in the
shared funnel (parts/box_funnel/make.py). Drone frame = seat frame (x nose, y left, z up).
What sets the size, and only this:
  prop     every rod >= 12 from the four swept discs (r 44.45 + 4, z 41.2 +- 4): a rounded square of half-width 114.6 with corner
           arcs; the ring is the octagon tangent to it, flats at |y| 115, corner chamfers on x + y = 193.7
  nose     the camera just ahead of the front discs, 30 deg up (lens (128, 0, 32); 4 Nov, Spark: close-ups and ceilings): the nose flat
           at x 136, the window between the dipped front nodes (z 17) and the ring (z 58), nothing in the 89 x 73 deg view
  tail     the tail electronics sit on edge in the tail unit's bay over the tail runner (4 Nov, Spark: into the printed base): nothing
           behind the plate's tail but the tail node, so the tail flat comes in to x -116, the least the rear discs' 12 mm keep-out allows
           (54.1 + 48.45 + 12 = 114.6; v18 had -130 for the tail bay)
  roof     one ring at z 58 is both the prop keep-out ring and the roof's edge (v18, 4 Nov, Spark: the look; was 62; at 54 the roof rods
           coming down to it passed 10 over the discs' far edges), the roof a ridge of three nodes at z 70 fanned to the ring (a 12 dome;
           the mid node halves the ridge rods: an upside-down landing put 407 N in a 140 ridge rod, 1.69 of its Euler load), no posts
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
Writes parts/scout_cage_v19/part.stl (tubes + node spheres, the look), the shell into systems/scout.assembly.json (cage_shell, the
part entry) and prints the checks. Run: python tools/scout_cage.py
"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ZR, ZB, ZF, ZK, YK, ZS = 58.0, 0.0, 17.0, -48.0, 84.0, 70.0   # the ring (prop keep-out and roof edge), the base rails, the dipped front nodes, the skids (z, |y|), the roof ridge
XN, XT, YS, C = 136.0, -116.0, 115.0, 193.7          # nose flat, tail flat (v19: -116, was -130), side flats, the corner chamfer line |x| + |y| = C
NODES = {
    "P0": (XN, C - XN, ZF), "P7": (XN, -(C - XN), ZF),                                                         # front nodes, dipped: the window's bottom corners
    "R0": (XN, C - XN, ZR), "R7": (XN, -(C - XN), ZR),                                                         # front nodes, roof: the window's top corners
    "P1": (C - YS, YS, ZR), "P2": (-(C - YS), YS, ZR), "P3": (XT, C + XT, ZR),                                  # the ring, left half (nose -> tail)
    "P4": (XT, -(C + XT), ZR), "P5": (-(C - YS), -YS, ZR), "P6": (C - YS, -YS, ZR),                             # right half
    "RF": (70, 0, ZS), "RM": (0, 0, ZS), "RB": (-70, 0, ZS),                                                   # roof ridge
    "B0": (100, YS, ZB), "B1": (-100, YS, ZB), "B2": (-100, -YS, ZB), "B3": (100, -YS, ZB),                     # base rails at z 0: above the funnel's walls in every state
    "BM0": (0, YS, ZB), "BM1": (0, -YS, ZB),                                                                   # the rails' mid nodes (v17)
    "K0": (100, YK, ZK), "K1": (-100, YK, ZK), "K2": (-100, -YK, ZK), "K3": (100, -YK, ZK),                     # the skids' ends, under the rail ends
    "KM0": (0, YK, ZK), "KM1": (0, -YK, ZK),                                                                   # the skids' mid nodes (v17)
    "NB": (114, 0, 8), "TB": (-112, 0, 10),                                                                    # nose and tail bottom nodes over the wall tops, printed into the nose and tail units (PRINTED); NB 4 back from 118 (4 Nov: the tilted camera's plug keep-out nicked its collar toward P0); TB in to -112 with the tail flat (its rods to P3 / P4 pass the rear discs by 12.3)
}
RODS = [("R0", "P1"), ("P1", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5"), ("P5", "P6"), ("P6", "R7"), ("R7", "R0"),   # the ring at z 62 (the roof's edge)
        ("P0", "R0"), ("P7", "R7"), ("P0", "P1"), ("P7", "P6"), ("P0", "P7"),                                  # the window's posts, the front slants, the window sill (z 17, under the view)
        ("RF", "R0"), ("RF", "R7"), ("RF", "P1"), ("RF", "P6"), ("RF", "RM"),                                  # roof, front
        ("RM", "P1"), ("RM", "P6"), ("RM", "P2"), ("RM", "P5"), ("RM", "RB"),                                  # roof, middle
        ("RB", "P2"), ("RB", "P5"), ("RB", "P3"), ("RB", "P4"),                                                # roof, rear
        ("B0", "BM0"), ("BM0", "B1"), ("B3", "BM1"), ("BM1", "B2"),                                            # base rails, two halves
        ("K0", "KM0"), ("KM0", "K1"), ("K3", "KM1"), ("KM1", "K2"),                                            # the skids, two halves
        ("B0", "K0"), ("B1", "K1"), ("B3", "K3"), ("B2", "K2"), ("BM0", "KM0"), ("BM1", "KM1"),                 # the skid trusses' posts
        ("BM0", "K0"), ("BM0", "K1"), ("BM1", "K3"), ("BM1", "K2"),                                            # the skid trusses' diagonals
        ("K0", "P0"), ("K3", "P7"), ("K1", "P3"), ("K2", "P4"),                                                # the skid ends braced out of plane (5 Nov: K0-R0 / K3-R7 passed 16 mm from P0 / P7, through their P0-P1 / P7-P6 collars; to P0 / P7, one printed structure with R0 / R7 now)
        ("B0", "R0"), ("B3", "R7"),                                                                            # close the nose brace's skew quad K0-B0-P0-R0 into triangles
        ("B0", "P0"), ("B0", "P1"), ("B1", "P2"), ("B1", "P3"), ("B2", "P4"), ("B2", "P5"), ("B3", "P6"), ("B3", "P7"),   # lower band
        ("BM0", "P1"), ("BM0", "P2"), ("BM1", "P6"), ("BM1", "P5"),                                            # the side faces fanned from the rails' mid nodes
        ("NB", "B0"), ("NB", "B3"), ("NB", "P0"), ("NB", "P7"), ("TB", "B1"), ("TB", "B2"), ("TB", "P3"), ("TB", "P4")]   # nose and tail, lower band
TIE_ROOTS = {"A0": (62, 62, 13.5), "A1": (-62, 62, 13.5), "A2": (-62, -62, 13.5), "A3": (62, -62, 13.5),      # clips under the four arm tips (the motor's M2 screws), z 13.5 under the plate (4 Nov run 1: at 17 the collar's top was sheared by the plate)
             "C0": (58, 30, 14), "C1": (-50, 40, 12), "C2": (-50, -40, 12), "C3": (58, -30, 14)}               # sockets on the nose and tail units' yoke corners, under the plate (v18; v17 had saddles on the plate top at z 22); the tail pair out at |y| 40 (4 Nov run 1: at (-58, +-30, 14) their collars were sheared by the UBEC lead and XR2 clearances)
PRINTED = {"NB": "scout_nose_unit", "TB": "scout_tail_unit"}   # cage nodes printed into a unit bolted to the plate (v18; v17's tail ties T0/T1 to TB are the tail unit itself:
TIES = [("A0", "B0"), ("A1", "B1"), ("A2", "B2"), ("A3", "B3"), ("C0", "BM0"), ("C1", "BM0"), ("C2", "BM1"), ("C3", "BM1")]   # 3.0 rods: the arm-tip ties to the rail corners, the yoke-corner ties to the rail mid nodes (5 Nov: at a corner the two ties arrived 2..13 deg apart, their bores had to start 39 mm out and the node grew past the cell limit)
#   without a stiff tail link a 35 g tail-first wall hit racked the cage 17 mm with 1157 N in a corner tie, frame model 4 Nov; with one 0.2 mm, 144 N)
NODE_CLEAR = 20.0                                              # a rod passing a node it does not end at keeps its axis this far from the centre (a screen: a socket reaches ~27 but points at its own rod; the node library checks every collar exactly, K0-R0 passed P0 at 15.9)
MERGE_MM = 45.0                                                # two nodes closer than this are one printed structure (5 Nov, Spark: "if any two nodes are too
BRIDGES = [(i, j) for i, j in RODS if np.linalg.norm(np.array(NODES[i], float) - np.array(NODES[j], float)) <= MERGE_MM]   # close together ... make that into
#   one structure, the same all across the scout"): the rod between them is printed PETG (a bridge), its two sockets are gone. At 45 (8 Nov,
#   Spark: "the nodes are super long": a 57 mm pair was a 112 mm part, a single node 43..58): only the window posts P0-R0 / P7-R7 (41)
#   merge; the ring's tail corners P2-P3 (53) and the skid-truss posts B-K / BM-KM (57) are single nodes again. Next up: NB-P0 62, B0-P0 70. Frame model with the bridges as PETG O8: 241 Hz (246), crash
#   deflection 1.44 mm (1.41), Euler and rod forces unchanged, a bridge under 8 MPa; mass a wash (a bridge O8 weighs what the rod,
#   its two collars and the engine's material behind them did, +-1 g a pair).


def _groups():
    """the printed structures: nodes joined by bridges (union-find), the rest single; PRINTED nodes left out"""
    par = {k: k for k in NODES}
    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, j in BRIDGES:
        par[f(i)] = f(j)
    out = {}
    for k in NODES:
        if k not in PRINTED:
            out.setdefault(f(k), []).append(k)
    return [tuple(sorted(g, key=list(NODES).index)) for g in out.values()]


GROUPS = _groups()
GROUP_OF = {k: g for g in GROUPS for k in g}
RODS_CARBON = [r for r in RODS if r not in BRIDGES]


def group_id(g):
    return "scout_cage_node_" + "_".join(k.lower() for k in g)


OPEN = [dict(ring=["K0", "KM0", "K1", "K2", "KM1", "K3"], what="bottom: the belly skirt into the funnel"), dict(ring=["B0", "B3", "NB"], what="bottom, nose V"), dict(ring=["B1", "TB", "B2"], what="bottom, tail V"),
        dict(ring=["K0", "B0", "B3", "K3"], what="bottom, nose end between the skids"), dict(ring=["K1", "B1", "B2", "K2"], what="bottom, tail end between the skids"),
        dict(ring=["P0", "R0", "R7", "P7"], what="nose window: the camera's view")]
MOTORS = [(54.1, 54.1), (-54.1, 54.1), (-54.1, -54.1), (54.1, -54.1)]
R_DISC, Z_DISC = 44.45 + 4.0, (37.2, 45.2)
LENS, TILT, HFOV, VFOV = np.array([130.0, 0.0, 38.0]), math.radians(30.0), math.radians(44.5), math.radians(36.5)   # lens and tilt: tools/scout_meshes.CAMERA_LENS / CAMERA_TILT (30 deg up, 4 Nov)
CAM_F, CAM_U, CAM_R = np.array([math.cos(TILT), 0, math.sin(TILT)]), np.array([-math.sin(TILT), 0, math.cos(TILT)]), np.array([0, 1.0, 0])


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
    d = p - LENS; f, r, u = d @ CAM_F, d @ CAM_R, d @ CAM_U
    return f > 0 and abs(math.atan2(r, f)) < HFOV and abs(math.atan2(u, f)) < VFOV


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
    body = [(nm, m) for nm, m in se.parts_of("scout") if not any(k in nm.lower() for k in ("cage", "legs", "unit", "clip"))]   # the units and clips are the cage's own sockets
    worst = dict(prop=(1e9, ""), body=(1e9, ""), camera=[], node=(1e9, ""))
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
        for k in NODES:                                                          # a rod passing another node: its axis must clear the node's socket reach
            if k in (i, j) or k in GROUP_OF.get(i, ()) or k in GROUP_OF.get(j, ()):
                continue
            d = float(np.linalg.norm(pts - np.array(P[k], float), axis=1).min())
            if d < worst["node"][0]:
                worst["node"] = (d, f"{i}-{j} past {k}")
    return worst


def shell_report(spec):
    import system_check as sc
    return sc.cage_shell({"machine": "scout", "cage_shell": spec}, [{"role": "cage"}])


def rod_ends():
    """each carbon rod and tie as it is cut: from the floor of its socket at one end to the floor at the other (the node library's
    socket starts; 3.0 at a clip or a unit), {name: (p0, p1, cut mm)}. Node to node the rods met at the node centres, and the page's
    mesh cleanup merged two in-line rods through a mid node into one body and dropped it: RF-RM-RB, the rails and the skids showed
    no rod on the page (7 Nov, Spark: "nodes with place for rods that dont have rods")."""
    import importlib
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    nl = importlib.import_module("parts.scout_node_lib")
    P = all_nodes(); out = {}
    for i, j in RODS_CARBON + TIES:
        a, b = np.array(P[i], float), np.array(P[j], float); u = (b - a) / np.linalg.norm(b - a)
        sa = nl.SOCKET_START.get((i, f"{i}-{j}"), 3.0) if i in NODES and i not in PRINTED else 3.0
        sb = nl.SOCKET_START.get((j, f"{i}-{j}"), 3.0) if j in NODES and j not in PRINTED else 3.0
        p0, p1 = a + u * sa, b - u * sb
        out[f"{i}-{j}"] = (p0, p1, float(np.linalg.norm(p1 - p0)))
    return out


def mesh():
    parts = [trimesh.creation.cylinder(radius=1.5, segment=[p0, p1], sections=12) for p0, p1, L in rod_ends().values()]   # the rods as cut, the bridges printed
    # no node spheres since 4 Nov: the 25 printed nodes are parts of their own (parts/scout_node_lib.py), NB / TB are in the units
    return trimesh.util.concatenate(parts)


if __name__ == "__main__":
    keys = list(NODES); idx = {k: n for n, k in enumerate(keys)}; P = all_nodes()
    L = [float(np.linalg.norm(np.array(NODES[i]) - np.array(NODES[j]))) for i, j in RODS]
    xs = [n[0] for n in NODES.values()]; ys = [n[1] for n in NODES.values()]; zs = [n[2] for n in NODES.values()]
    print(f"cage v19: {len(NODES)} nodes, {len(RODS)} rods of which {len(BRIDGES)} printed bridges, {sum(l for l, r in zip(L, RODS) if r not in BRIDGES) / 1000:.2f} m of 3.0 rod, {len(GROUPS)} printed structures, shortest rod {min(L):.0f}, longest {max(L):.0f}; "
          f"node extents {max(xs) - min(xs):.0f} x {max(ys) - min(ys):.0f} x {max(zs) - min(zs):.0f}")
    w = checks()
    print("  shortest rods: " + ", ".join(f"{i}-{j} {l:.0f}" for l, (i, j) in sorted(zip(L, RODS))[:4]))
    print(f"  prop keep-out: nearest rod {w['prop'][0]:.1f} mm ({w['prop'][1]}; need 12)")
    print(f"  body margin:   nearest rod surface {w['body'][0]:.1f} mm ({w['body'][1]}; need 3)")
    print(f"  camera view:   {'clear' if not w['camera'] else 'BLOCKED by ' + ', '.join(w['camera'])}")
    print(f"  passing rods:  nearest rod axis to a node it does not end at {w['node'][0]:.1f} mm ({w['node'][1]}; need {NODE_CLEAR}: the node library checks every collar exactly)")
    fw = funnel_check()
    print(f"  funnel:        least clearance {fw[0]:.1f} mm ({fw[1]} at {fw[2]}; need 2, exact gate tools/seat_check.py scout --descent)")
    LT = [float(np.linalg.norm(np.array(P[i]) - np.array(P[j]))) for i, j in TIES]
    print(f"  ties:          {len(TIES)} rods 3.0, {sum(LT) / 1000:.2f} m, {min(LT):.0f}..{max(LT):.0f}")
    RC = rod_ends(); print(f"  cut list:      {len(RC)} rods as cut (socket floor to socket floor) {sum(v[2] for v in RC.values()) / 1000:.2f} m, {min(v[2] for v in RC.values()):.0f}..{max(v[2] for v in RC.values()):.0f} mm (systems/scout.assembly.json cage_shell.rod_cuts_mm)")
    spec = dict(nodes=[list(map(float, NODES[k])) for k in keys], rods=[[idx[i], idx[j]] for i, j in RODS], node_r_mm=7.0, rod_r_mm=1.5,
                total_rods=len(RODS) + len(TIES), open=[dict(ring=[idx[k] for k in o["ring"]], what=o["what"]) for o in OPEN],
                rod_cuts_mm={k: round(v[2], 1) for k, v in rod_ends().items()},
                ties=dict(roots={k: list(map(float, v)) for k, v in TIE_ROOTS.items()}, rods=[[i, keys.index(j)] for i, j in TIES],
                          what="3.0 rods from clips under the four arm tips (A, parts/scout_arm_clip_*) and the nose / tail units' yoke corners (C) to the base rail corners B0..B3; NB and TB are printed into the units"),
                note="cage v19, the compact funnel cage on skid trusses, lowered, domed and shortened (tools/scout_cage.py, 4 Nov): a closed 3 mm rod octagonal prism formed to the Scout in the shared funnel, landing on two skids hung from its base rails as Warren trusses, their ends braced to the ring; NB and TB printed into the nose and tail units")
    try:
        r = shell_report(spec)
        print("  shell:         " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("tris",)))
    except Exception as e:
        print("  shell:         (system_check unavailable:", e, ")")
    m = mesh(); d = os.path.join(ROOT, "parts", "scout_cage_v19"); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
    F = os.path.join(ROOT, "systems", "scout.assembly.json"); raw = open(F).read(); a = json.loads(raw)
    a["cage_shell"] = spec
    a["parts"] = [p for p in a["parts"] if p["id"] not in ("scout_cage_pattern", "scout_cage_v14", "scout_cage_v15", "scout_cage_v16", "scout_cage_v17", "scout_cage_v18", "scout_cage_v19")]
    a["parts"] = [p for p in a["parts"] if not p["id"].startswith("scout_cage_node_")]
    for g in GROUPS:
        cs = np.array([NODES[k] for k in g], float); lo, hi = cs.min(0) - 5, cs.max(0) + 5
        mesh_path = f"parts/{group_id(g)}/part.stl"; n_m = sum(any(k in r for k in g) for r in RODS_CARBON + TIES)
        e = dict(id=group_id(g), name="cage node " + "+".join(g), role="cage", group="cage", color="#eef2f5", opacity=0.8, material="PETG",
                 box=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]], mass_g=5.0 * len(g), tag="E",   # a box over the centres until the run lands (+-9 crossed the funnel's y walls in the descent gate; the node setups cut the funnel's sweep)
                 owner="scout v3", version="v19 nodes (5 Nov: struts, merged pairs)", source="parts/scout_node_lib.py",
                 note=f"the socket collars of the {n_m} rods and ties that end here, the rest generated (parts/{group_id(g)}/part.py)"
                      + (f"; nodes {' and '.join(g)} are one printed structure, the rod between them a printed bridge (MERGE_MM {MERGE_MM:.0f})" if len(g) > 1 else "") + "; a box until the run lands")
        if os.path.exists(os.path.join(ROOT, mesh_path)):
            e["mesh"] = mesh_path                                                 # the run's mesh once it has landed (system_check resolves mesh paths strictly)
        a["parts"].append(e)
    a["parts"].append(dict(id="scout_cage_v19", name="cage v19 rods and ties (low domed funnel cage on skid trusses, tail in to -116, 3 mm carbon, frame-checked)", role="cage", group="cage", color="#2b2b2b",
                           mesh="parts/scout_cage_v19/part.stl", mass_g=round(sum(l for l, r in zip(L, RODS) if r not in BRIDGES) * 0.011 + sum(LT) * 0.011, 1), tag="E", owner="scout v3",
                           version="v19 (4 Nov)", source="tools/scout_cage.py",
                           note=f"{len(NODES)} nodes, {len(RODS_CARBON)} rods 3.0 ({sum(l for l, r in zip(L, RODS) if r not in BRIDGES) / 1000:.2f} m) and {len(BRIDGES)} printed bridges ({len(GROUPS)} printed structures), an octagonal prism tangent to the prop keep-out, base rails z 0, two skids at |y| 84, z -48 (TPU strips) as Warren trusses braced to the ring, 8 ties to the plate, ring z 54 and a 12 dome, nose window for the camera, NB / TB printed into the nose / tail units, the other 25 nodes their own printed parts (parts/scout_node_lib.py); frame model tools/scout_cage_loads.py"))
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    print("  written: parts/scout_cage_v19/part.stl and systems/scout.assembly.json cage_shell")
