"""Scout cage v21, the half cage (8 Nov, Spark: "why are we still keeping the lower half of the cage, it provides little value as the
bottom shape for the funnel already had plenty of material ... get rid of any ongoing full cage and focus on the new half cage design";
the Overseer's brief the same day: a half cage is fine, it protects the top: the electronics, the camera, the props; the bottom has the
base units). v20 (the light funnel cage, 39 rods and 4 ties in 17 printed structures) kept v19's skids; v21 drops them with their
A-frames, verticals, braces, skid nodes and TPU feet: the Scout lands on the nose and tail units' runners, the funnel covers the belly.
What is left is the top: the ring, the ridge with its waist hoop, the nose window, the nose and tail rods into the units, and at each
side one knee node under the ring's side mid (S0 / S1, at x 0 between the discs, under the plate's height) that holds the ring's side
to two arm-tip clips: a leg up to the side mid, a fan to the two ring corners of that side, and the two ties in from the clips.
31 rods (2 printed bridges) in 15 printed structures, 4 ties, 4 clips. A closed shell over the top of 3.0 carbon rods in printed PCTG
nodes, its bottom declared open. Drone frame = seat frame (x nose, y left, z up).
What sets the size, and only this:
  prop     every rod >= 12 from the four swept discs (r 44.45 + 4, z 41.2 +- 4): a rounded square of half-width 114.6 with corner
           arcs; the ring is the octagon tangent to it, flats at |y| 115, corner chamfers on x + y = 193.7. The rods stay out of the
           discs' plan too (disc_shadow; a 3.0 rod across a disc's inflow or wash costs thrust, E: a 90 mm rod is 4 % of a disc's area):
           only the last 51 mm of the two rear ridge rods and the four clip ties (under the arm tips, 25 under the disc) cross a plan
  nose     the camera just ahead of the front discs, 30 deg up (lens (130, 0, 38); 4 Nov, Spark: close-ups and ceilings): the nose flat
           at x 136, the window between the dipped front nodes (z 17) and the ring (z 58), nothing in the 89 x 73 deg view
  tail     the tail electronics sit on edge in the tail unit's bay over the tail runner (4 Nov, Spark: into the printed base): nothing
           behind the plate's tail but the tail node, so the tail flat comes in to x -116, the least the rear discs' 12 mm keep-out allows
  roof     one ring at z 58 is both the prop keep-out ring and the roof's edge, the roof a ridge of three nodes at z 70 (a 12 dome) with
           a triangle to the window's top and to the tail flat, and the waist hoop to the ring's side mids (the one transverse line that
           passes between the four discs). The roof's side panels are declared openings (ring to ridge, 70 x 110): a flat ceiling
           touches the ring and the ridge first at any roll under 46 deg and any pitch (the ring stands 12 outside and 13 above every
           disc edge), so only a poker can reach a disc through them
  sides    the side flats are planes at |y| 115, 12.4 outside every disc in plan, so the knee and its leg and fan stand in them at any
           height: the knee at z 12 (6.5 under the plate's underside, level with the clip roots, so the ties run level under the arm tips; at z 18 the leg is
           40 long and merges into a printed bridge, MERGE_MM). Without the knees the ring hangs on the four unit rods alone: 26 mm and
           2.6 N m over the socket's 2.0 in a side crash (8 Nov frame model); the leg alone 13.7 mm and 2.02 N m; with the fan 13.0 mm,
           1.86 N m. The yoke instead of the clips (a rod in at x 0 to the nose unit's side) put 2.9 N m on that rod's sockets
  bottom   open (the half cage): the Scout lands on the base units, the nose and tail runners' seat rims at z -44.3 (184 x 66.8,
           BASE: the frame model's contact points there, stiff to the core); the whole shell stays above z 0 over the funnel's wall
           tops (z -4.3) in every state of the descent: nose and tail bottom nodes at z 8 / 10 in the units, the window sill at z 17,
           the knees at z 12
  funnel   nothing of the cage inside the funnel's walls when seated or during the +-28 / +-20 descent (tools/seat_check.py)
  shell    closed over the top, mirror-exact, declared openings: the bottom (in planar pieces), the nose window, the roof panels
Writes parts/scout_cage_v21/part.stl (the rods as cut), the shell into systems/scout.assembly.json (cage_shell, the part entries) and
prints the checks. Run: python tools/scout_cage.py
"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

ZR, ZF, ZS, ZKN = 58.0, 17.0, 70.0, 12.0          # the ring (prop keep-out and roof edge), the dipped front nodes, the roof ridge, the side knees
XN, XT, YS, C = 136.0, -116.0, 115.0, 193.7          # nose flat, tail flat, side flats, the corner chamfer line |x| + |y| = C
NODES = {
    "P0": (XN, C - XN, ZF), "P7": (XN, -(C - XN), ZF),                                                         # front nodes, dipped: the window's bottom corners
    "R0": (XN, C - XN, ZR), "R7": (XN, -(C - XN), ZR),                                                         # front nodes, roof: the window's top corners
    "P1": (C - YS, YS, ZR), "PM0": (0.0, YS, ZR), "P2": (-(C - YS), YS, ZR), "P3": (XT, C + XT, ZR),             # the ring, left half (nose -> tail); PM0 the side flat's mid (the hoop's foot, the leg's top)
    "P4": (XT, -(C + XT), ZR), "P5": (-(C - YS), -YS, ZR), "PM1": (0.0, -YS, ZR), "P6": (C - YS, -YS, ZR),       # right half
    "RF": (70, 0, ZS), "RM": (0, 0, ZS), "RB": (-70, 0, ZS),                                                   # roof ridge; RM the hoop's apex
    "S0": (0.0, YS, ZKN), "S1": (0.0, -YS, ZKN),                                                               # the side knees (v21): under the side mids, in the side flats' planes, between the discs
    "NB": (114, 0, 8), "TB": (-112, 0, 10),                                                                    # nose and tail bottom nodes over the wall tops, printed into the nose and tail units (PRINTED)
}
BASE = {"G0": (93.5, 33.4, -44.3), "G1": (-90.5, 33.4, -44.3), "G2": (-90.5, -33.4, -44.3), "G3": (93.5, -33.4, -44.3)}   # what the Scout lands on: the runners' seat rim corners (parts/scout_units_lib: CX 1.5, SEAT_L 184, SEAT_W 66.8, SEAT_Z -44.3; the frame model's contact points, stiff to the core)
RODS = [("R0", "P1"), ("P1", "PM0"), ("PM0", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5"), ("P5", "PM1"), ("PM1", "P6"), ("P6", "R7"), ("R7", "R0"),   # the ring at z 58 (the roof's edge), side flats in two
        ("P0", "R0"), ("P7", "R7"), ("P0", "P7"),                                                              # the window's posts (printed bridges) and sill (z 17, under the view)
        ("RF", "R0"), ("RF", "R7"), ("RF", "RM"), ("RM", "RB"), ("RB", "P3"), ("RB", "P4"),                     # the ridge: a triangle to the window's top, the spine in two, a triangle to the tail flat
        ("RM", "PM0"), ("RM", "PM1"),                                                                          # the waist hoop: ridge mid to the ring's side mids, between the discs (5.65 from their edges in plan, 19 above)
        ("NB", "P0"), ("NB", "P7"), ("TB", "P3"), ("TB", "P4"),                                                # nose and tail bottoms to the units' printed nodes
        ("PM0", "S0"), ("PM1", "S1"),                                                                          # the side legs (v21): side mid down to the knee, 46
        ("S0", "P1"), ("S0", "P2"), ("S1", "P6"), ("S1", "P5")]                                                # the side fans: knee to the ring's corners on that side (in the side flat's plane, 12.4 from the discs)
TIE_ROOTS = {"A0": (62, 62, 12.0), "A1": (-62, 62, 12.0), "A2": (-62, -62, 12.0), "A3": (62, -62, 12.0)}      # clips under the four arm tips (the motor's M2 screws), z 12 under the plate (4 Nov run 1: at 17 the collar's top was sheared by the plate; 8 Nov v21: at 13.5 the level tie's collar ran along under the plate and lost 6 % to its keep-out, clear from 12.5)
PRINTED = {"NB": "scout_nose_unit", "TB": "scout_tail_unit"}   # cage nodes printed into a unit bolted to the plate
TIES = [("A0", "S0"), ("A1", "S0"), ("A2", "S1"), ("A3", "S1")]   # 3.0 rods: from each arm-tip clip in under the disc to that side's knee (v21; v20 down to the skid ends), level at z 12, 25 under the disc
NODE_CLEAR = 20.0                                              # a rod passing a node it does not end at keeps its axis this far from the centre (a screen: the node library checks every collar exactly)
MERGE_MM = 45.0                                                # two nodes closer than this are one printed structure (5 Nov, Spark: "if any two nodes are too
BRIDGES = [(i, j) for i, j in RODS if np.linalg.norm(np.array(NODES[i], float) - np.array(NODES[j], float)) <= MERGE_MM]   # close together ... make that into
#   one structure, the same all across the scout"): the rod between them is printed PETG (a bridge), its two sockets are gone. At 45 (8 Nov,
#   Spark: "the nodes are super long") only the window posts P0-R0 / P7-R7 (41) merge. Frame model with the bridges as PETG O8: mass a wash
#   (a bridge O8 weighs what the rod, its two collars and the engine's material behind them did, +-1 g a pair).


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


OPEN = [dict(ring=["NB", "P0", "S0"], what="bottom (the half cage: the base units and the funnel below), nose end"),   # every ring planar (system_check projects a ring onto one plane: a bent pentagon left sliver gaps along its folds, 8 Nov)
        dict(ring=["NB", "S0", "S1"], what="bottom, the belly between the knees, nose half"), dict(ring=["NB", "S1", "P7"], what="bottom, nose end"),
        dict(ring=["TB", "S0", "S1"], what="bottom, the belly between the knees, tail half"), dict(ring=["S0", "P3", "TB"], what="bottom, tail end"), dict(ring=["S1", "TB", "P4"], what="bottom, tail end"),
        dict(ring=["P0", "R0", "P1"], what="side, front left: under the ring between the window post and the fan"), dict(ring=["P0", "P1", "S0"], what="side, front left"),
        dict(ring=["S0", "P2", "P3"], what="side, rear left: under the ring behind the fan"),
        dict(ring=["P7", "R7", "P6"], what="side, front right"), dict(ring=["P7", "P6", "S1"], what="side, front right"), dict(ring=["S1", "P5", "P4"], what="side, rear right"),
        dict(ring=["P0", "R0", "R7", "P7"], what="nose window: the camera's view"),
        dict(ring=["P1", "PM0", "RM", "RF"], what="roof, front left panel: ring to ridge; a flat ceiling touches the ring and the ridge first"), dict(ring=["R0", "P1", "RF"], what="roof, front left corner"),
        dict(ring=["RM", "PM0", "P2", "RB"], what="roof, rear left panel"), dict(ring=["P2", "P3", "RB"], what="roof, rear left corner"),
        dict(ring=["P6", "PM1", "RM", "RF"], what="roof, front right panel"), dict(ring=["R7", "P6", "RF"], what="roof, front right corner"),
        dict(ring=["RM", "PM1", "P5", "RB"], what="roof, rear right panel"), dict(ring=["P5", "P4", "RB"], what="roof, rear right corner")]
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


def disc_shadow(rods=None, ties=None):
    """rod length whose plan projection lies inside a disc (over it in the inflow or under it in the wash): what a rod costs in thrust
    (8 Nov, Spark: material near the props takes thrust). (total mm, [(rod, mm in shadow)]). The ties under the arm tips are under the
    plate's arms anyway."""
    P = all_nodes(); tot = 0.0; which = []
    for i, j in (RODS if rods is None else rods) + (TIES if ties is None else ties):
        pts = seg_pts(P[i], P[j], 200); L = float(np.linalg.norm(np.array(P[j], float) - np.array(P[i], float)))
        inside = np.zeros(len(pts), bool)
        for mx, my in MOTORS:
            inside |= np.hypot(pts[:, 0] - mx, pts[:, 1] - my) < R_DISC
        if inside.any():
            tot += L * inside.mean(); which.append((f"{i}-{j}", L * inside.mean()))
    return tot, which


def funnel_check(margin=2.0):
    """every rod and tie, on the whole descent of tools/seat_check.py (the capture limits, heights every 3..5), against the funnel
    mesh: the least distance from a rod axis to the funnel surface, less the rod radius; need >= margin. Fast (330 faces)."""
    import seat_check as skc
    fun = skc.to_tri(skc.load_make("parts/box_funnel/make.py").funnel(skc.walls_for("scout")))     # the frame and both end walls (part.stl is the frame alone since 9 Nov)
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
    worst["shadow"] = disc_shadow()
    return worst


def shell_report(spec):
    import system_check as sc
    return sc.cage_shell({"machine": "scout", "cage_shell": spec}, [{"role": "cage"}])


def rod_ends():
    """each carbon rod and tie as it is cut: from the floor of its socket at one end to the floor at the other (the node library's
    socket starts; 3.0 at a clip or a unit), {name: (p0, p1, cut mm)}. Node to node the rods met at the node centres, and the page's
    mesh cleanup merged two in-line rods through a mid node into one body and dropped it (7 Nov, Spark: "nodes with place for rods
    that dont have rods")."""
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
    return trimesh.util.concatenate(parts)                   # the 19 printed nodes are parts of their own (parts/scout_node_lib.py), NB / TB are in the units


def _current_mesh(gid):
    """the run's mesh for a node structure, only when its result was made from the setup part.py builds now (tools/part_sha.stale):
    a stale mesh is an older cage's node shape on this cage's rods (8 Nov)"""
    d = os.path.join(ROOT, "parts", gid); stl, res = os.path.join(d, "part.stl"), os.path.join(d, "result.json")
    if not (os.path.exists(stl) and os.path.exists(res)):
        return None
    try:
        import part_sha
        return None if part_sha.stale(os.path.join(d, "part.py"), json.load(open(res))) else f"parts/{gid}/part.stl"
    except Exception as e:                                  # a setup that no longer builds: no mesh
        print(f"  ({gid}: no current mesh: {str(e)[:80]})"); return None


if __name__ == "__main__":
    keys = list(NODES); idx = {k: n for n, k in enumerate(keys)}; P = all_nodes()
    L = [float(np.linalg.norm(np.array(NODES[i]) - np.array(NODES[j]))) for i, j in RODS]
    xs = [n[0] for n in NODES.values()]; ys = [n[1] for n in NODES.values()]; zs = [n[2] for n in NODES.values()]
    m_rod = sum(l for l, r in zip(L, RODS) if r not in BRIDGES)
    print(f"cage v21 (half cage): {len(NODES)} nodes, {len(RODS)} rods of which {len(BRIDGES)} printed bridges, {m_rod / 1000:.2f} m of 3.0 rod, {len(GROUPS)} printed structures, shortest rod {min(L):.0f}, longest {max(L):.0f}; "
          f"node extents {max(xs) - min(xs):.0f} x {max(ys) - min(ys):.0f} x {max(zs) - min(zs):.0f}")
    w = checks()
    print("  shortest rods: " + ", ".join(f"{i}-{j} {l:.0f}" for l, (i, j) in sorted(zip(L, RODS))[:4]))
    print(f"  prop keep-out: nearest rod {w['prop'][0]:.1f} mm ({w['prop'][1]}; need 12)")
    print(f"  disc shadow:   {w['shadow'][0]:.0f} mm of rod over or under a disc" + (": " + ", ".join(f"{n} {v:.0f}" for n, v in w["shadow"][1]) if w["shadow"][1] else "") + " (v19: 1744 mm over 26 rods)")
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
                rod_cuts_mm={k: round(v[2], 1) for k, v in RC.items()},
                ties=dict(roots={k: list(map(float, v)) for k, v in TIE_ROOTS.items()}, rods=[[i, keys.index(j)] for i, j in TIES],
                          what="3.0 rods from clips under the four arm tips (A, parts/scout_arm_clip_*) in under the discs to the side knees S0 / S1; NB and TB are printed into the units"),
                note="cage v21, the half cage (tools/scout_cage.py, 8 Nov): a 3 mm rod octagonal ring tangent to the prop keep-out with a three-node ridge, a waist hoop between the discs, "
                     "a knee under each side mid (leg, fan to the ring corners, ties in from the arm clips), the window sill and the nose / tail bottoms into the units; the bottom open, "
                     "the Scout lands on the base units' runners; 31 rods, 4 ties")
    try:
        r = shell_report(spec)
        print("  shell:         " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("tris",)))
    except Exception as e:
        print("  shell:         (system_check unavailable:", e, ")")
    m = mesh(); d = os.path.join(ROOT, "parts", "scout_cage_v21"); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
    F = os.path.join(ROOT, "systems", "scout.assembly.json"); raw = open(F).read(); a = json.loads(raw)
    a["cage_shell"] = spec
    a["parts"] = [p for p in a["parts"] if p["id"] not in ("scout_cage_pattern", "scout_cage_v14", "scout_cage_v15", "scout_cage_v16", "scout_cage_v17", "scout_cage_v18", "scout_cage_v19", "scout_cage_v20", "scout_cage_v21")]
    a["parts"] = [p for p in a["parts"] if not p["id"].startswith("scout_cage_node_") and p["id"] != "scout_tpu_feet"]
    for g in GROUPS:
        cs = np.array([NODES[k] for k in g], float); lo, hi = cs.min(0) - 5, cs.max(0) + 5
        n_m = sum(any(k in r for k in g) for r in RODS_CARBON + TIES)
        e = dict(id=group_id(g), name="cage node " + "+".join(g), role="cage", group="cage", color="#eef2f5", opacity=0.8, material="PCTG",
                 box=[lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]], mass_g=5.0 * len(g), tag="E",   # a box over the centres until the run lands
                 owner="scout v3", version="v21 nodes (8 Nov: the half cage, 15 structures)", source="parts/scout_node_lib.py",
                 note=f"the socket collars of the {n_m} rods and ties that end here, the rest generated (parts/{group_id(g)}/part.py)"
                      + (f"; nodes {' and '.join(g)} are one printed structure, the rod between them a printed bridge (MERGE_MM {MERGE_MM:.0f})" if len(g) > 1 else "") + "; a box until the run lands")
        mp = _current_mesh(group_id(g))
        if mp:
            e["mesh"] = mp                                                         # the run's mesh once it has landed on this setup
        a["parts"].append(e)
    a["parts"].append(dict(id="scout_cage_v21", name="cage v21 rods and ties (the half cage: ring, ridge, waist hoop, side knees, 3 mm carbon, frame-checked)", role="cage", group="cage", color="#2b2b2b",
                           mesh="parts/scout_cage_v21/part.stl", mass_g=round(m_rod * 0.011 + sum(LT) * 0.011, 1), tag="E", owner="scout v3",
                           version="v21 (8 Nov)", source="tools/scout_cage.py",
                           note=f"{len(NODES)} nodes, {len(RODS_CARBON)} rods 3.0 ({m_rod / 1000:.2f} m) and {len(BRIDGES)} printed bridges ({len(GROUPS)} printed structures), 4 ties to the arm clips; an octagonal ring tangent to the prop keep-out at z 58, a three-node ridge at z 70 with a waist hoop to the ring's side mids, a knee at z 12 under each side mid (a leg, a fan to the ring's two corners on that side, two ties in from the arm clips), nose window for the camera, NB / TB printed into the nose / tail units, the other nodes their own printed parts (parts/scout_node_lib.py); the bottom open: the Scout lands on the base units' runners; frame model tools/scout_cage_loads.py"))
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + ("\n" if raw.endswith("\n") else ""))
    print("  written: parts/scout_cage_v21/part.stl and systems/scout.assembly.json cage_shell")
