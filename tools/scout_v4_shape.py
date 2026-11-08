"""Scout v4, the shape design (8 Nov, Spark: "we are going to start on scout v4, should have new props meaning we are no longer using
the bee35 plate and will be printing everything ourselves with pctg, we also need to make sure it still fits inside the landing
funnel for the battery swapper, lets first create a shape design for that"). A layout of envelopes and bought parts, not engine
setups: what sits where, how big the printed frame is, and that it seats in the shared funnel. Drone frame = the seat frame of
tools/seat_envelope.py (x nose, y left, z up; the receiver's bottom at z -44.3 on the funnel's ledge).

What sets the shape, and only this:
  props    Gemfan Hurricane SL 5125, 5.1 in (O129.4, r 64.7; systems/dimensions.md A, Spark 8 Nov) on the same EMAX ECO II 2004
           motors, hubs at (+-74.3, +-74.3): the 3520's 19.2 tip gap kept between neighbours (2 x 74.3 - 129.4), wheelbase 210
  frame    printed PCTG, no bought plate: a deck where the Bee35 was (underside z 18.5, the units bolt up into it as they did into
           the plate), 6 thick (E, until the engine sizes it), four arms out to motor pods at the hubs; the motors on the pods, the
           prop plane DECK_TOP + 19.2 (the 2004's listing height, as on v3)
  battery  unchanged, so the swapper takes the v3 pack: the Overseer receiver v3 under the deck, the cartridge v6 and the 4S 1500
           in it, at the v3 place in the seat frame (x -51.3..54.3, |y| 32.8, z -44.3..9.6)
  funnel   the shared funnel (parts/box_funnel/make.py) is a belly catch: its faces stop at z -4.3 (x walls) and -16.3 (y walls),
           so only what hangs below z -4.3 meets it: the v3 base, kept, the nose and tail runners on the receiver's outline
           (184 x 66.8 at the ledge, 55 deg lead-ins), the keel |y| <= 33.4 up to the deck. The props, arms and cage are all
           above the funnel when seated and on the whole descent (tools/seat_check.py scout_v4 --descent is the gate)
  camera   the v3 camera (DECXIN 2633V2, 30 deg up, 89 x 73 view) moved forward to lens x 149, its board 21 outside the front
           discs' 12 mm keep-out in plan (at the v3 lens x 130 it was 7), 6 inside the cage's nose flat as on v3
  flow     MTF-02P in the nose runner's window looking down, as v3 (70, 0, -24.5)
  tail     Radxa Zero 3W, UBEC, XR2 in the tail unit's bay over the tail runner, as v3; the FC on top of the deck at the centre
  cage     a half cage of 3.0 carbon rods only (Spark 8 Nov: keep the 3 mm, the cage's structure first, nodes later). Every rod
           12 from the swept discs (r 64.7 + 4, z prop plane +- 4): the octagon ring tangent to that keep-out (flats at |x|, |y|
           155, chamfers on |x| + |y| = 262.7) at z 61 (13 over the sweep); a FLAT roof in the ring's plane: a spine on y 0 and a
           hoop on x 0 (never over a disc) with the nose and tail triangles from RF / RB at x +-130 (at x 90 they crossed the front
           discs); the cage hangs on four nodes printed into the frame's outriggers, NB / TB at the ends of the base's booms and
           S0 / S1 at the ends of two side spars on x 0 between the discs, all at z 12 in the flats' planes (every point there 12
           from the discs), with a V up to the ring at each end and a fan at each side. Why (frame model, frame_check): rods
           rising from below the props to the ring can only stand in the flats' planes, and the core's crash load must reach the
           cage axially: on ties under the arm tips (v21's way) the 35 g load went through level rods in bending (31 mm, 2 rods
           over Euler); a ridge 12 over the ring took the upside-down hit alone and pushed its triangles into the ring at 1.3 x
           Euler; flat, the hit spreads over the whole roof. Bottom open, the Scout lands on the runners
Writes parts/scout_v4_<part>/part.stl (one folder per v4 part, build_site: a folder belongs to one drone), systems/scout_v4.assembly.json
and prints the checks. Run: python tools/scout_v4_shape.py
"""
import json, math, os, sys
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# ---- props, motors, frame ---------------------------------------------------------------------------------------------------
PROP_D, HUB = 129.4, 74.3                       # Gemfan 5125 (dimensions.md A), hubs at (+-HUB, +-HUB)
PROP_R = PROP_D / 2
MOTORS = [(HUB, -HUB), (-HUB, HUB), (HUB, HUB), (-HUB, -HUB)]   # PX4 quad X: motor 1 front right, 2 rear left, 3 front left, 4 rear right
DECK_Z0, DECK_T = 18.5, 6.0                     # the deck's underside where the Bee35's was; 6 thick (E)
DECK_TOP = DECK_Z0 + DECK_T
MOTOR_H = 19.2                                   # ECO II 2004: arm top to the prop hub face (listing, E; dimensions.md C)
Z_PROP = DECK_TOP + MOTOR_H                     # 43.7
R_KEEP, Z_SWEEP = PROP_R + 4.0, 4.0             # the disc grown 4 and swept +-4 (the cage rule's disc, as v3)
KEEP = 12.0                                      # every rod 12 from the swept disc
DECK = (-60.0, 60.0, 40.0)                       # the deck's body: x0, x1, |y| (E), arms from its corners to the pods
ARM_W, POD_R = 18.0, 15.0                        # arm width, motor pod radius (the 2004's base O27 + 1.5)
# ---- the v3 base and the battery, unchanged ---------------------------------------------------------------------------------
SEAT_Z, CX, SEAT_L, SEAT_W, RUN_H, CH = -44.3, 1.5, 184.0, 66.8, 20.0, 4.0
# ---- camera, cage --------------------------------------------------------------------------------------------------------------
LENS_V3, LENS = np.array([130.0, 0.0, 38.0]), np.array([149.0, 0.0, 38.0])
TILT, HFOV, VFOV = math.radians(30.0), math.radians(44.5), math.radians(36.5)
CAM_F, CAM_U, CAM_R = np.array([math.cos(TILT), 0, math.sin(TILT)]), np.array([-math.sin(TILT), 0, math.cos(TILT)]), np.array([0, 1.0, 0])
FLAT = HUB + R_KEEP + KEEP                       # 155.0: the ring's flats
CH_LINE = 2 * HUB + (R_KEEP + KEEP) * math.sqrt(2)   # 262.7: the corner chamfers |x| + |y| = CH_LINE
ZR, ZB, XR = 61.0, 12.0, 130.0                    # ring and flat roof, the outrigger nodes, the roof's nose / tail apex
ZR = max(ZR, Z_PROP + Z_SWEEP + 13.0)
YC = CH_LINE - FLAT                              # 107.7: where a flat meets a chamfer
NODES = {
    "R0": (FLAT, YC, ZR), "P1": (YC, FLAT, ZR), "PM0": (0.0, FLAT, ZR), "P2": (-YC, FLAT, ZR), "P3": (-FLAT, YC, ZR),       # the ring (left half, nose -> tail)
    "P4": (-FLAT, -YC, ZR), "P5": (-YC, -FLAT, ZR), "PM1": (0.0, -FLAT, ZR), "P6": (YC, -FLAT, ZR), "R7": (FLAT, -YC, ZR),   # right half
    "RF": (XR, 0, ZR), "RM": (0.0, 0, ZR), "RB": (-XR, 0, ZR),                                                              # the flat roof's spine
    "NB": (FLAT, 0, ZB), "TB": (-FLAT, 0, ZB), "S0": (0.0, FLAT, ZB), "S1": (0.0, -FLAT, ZB),                               # printed into the frame's outriggers (PRINTED)
}
PRINTED = ("NB", "TB", "S0", "S1")
ROOTS = {}
RODS = [("R0", "P1"), ("P1", "PM0"), ("PM0", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5"), ("P5", "PM1"), ("PM1", "P6"), ("P6", "R7"), ("R7", "R0"),   # the ring
        ("RF", "R0"), ("RF", "R7"), ("RF", "RM"), ("RM", "RB"), ("RB", "P3"), ("RB", "P4"), ("RM", "PM0"), ("RM", "PM1"),                                 # the flat roof: nose and tail triangles, spine, hoop
        ("NB", "R0"), ("NB", "R7"), ("TB", "P3"), ("TB", "P4"),                                                                                            # the end Vs (the camera's window stays clear between them)
        ("S0", "P1"), ("S0", "PM0"), ("S0", "P2"), ("S1", "P6"), ("S1", "PM1"), ("S1", "P5")]                                                              # the side fans
TIES = []
SPAR_W = 12.0                                     # the side spars on x 0, between the discs (their keep-out edges at |x| 5.6 at y 74.3: 21 under the sweep)
ROD_G_MM, NODE_G = 0.011, 6.0                    # 3.0 carbon rod; a printed PCTG node (v21 run: 5..13 g, E)
PCTG = 1.23e-3                                   # g / mm3


def P(k):
    return np.array({**NODES, **ROOTS}[k], float)


def disc_dist(p):
    best = 1e9
    for mx, my in MOTORS:
        dr = max(0.0, math.hypot(p[0] - mx, p[1] - my) - R_KEEP)
        dz = max(0.0, (Z_PROP - Z_SWEEP) - p[2], p[2] - (Z_PROP + Z_SWEEP))
        best = min(best, math.hypot(dr, dz))
    return best


def in_view(p, lens=LENS):
    d = np.asarray(p, float) - lens; f, r, u = d @ CAM_F, d @ CAM_R, d @ CAM_U
    return f > 0 and abs(math.atan2(r, f)) < HFOV - math.radians(2) * 0 and abs(math.atan2(u, f)) < VFOV


def seg(a, b, n=80):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def box(x0, x1, y0, y1, z0, z1):
    m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0]); m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]); return m


def cyl(x, y, z0, z1, r, n=48):
    m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=n); m.apply_translation([x, y, (z0 + z1) / 2]); return m


def rod(a, b, r=1.5):
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(a, float), np.asarray(b, float)], sections=12)


def union(ms):
    import manifold3d as mf
    out = None
    for m in ms:
        mm = mf.Manifold(mf.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))
        out = mm if out is None else out + mm
    t = out.to_mesh(); return trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=False)


def frame():
    """the printed deck: the body plate, four arms from its corners to the pods, the pods, two side spars out to the cage's side nodes
    S0 / S1 (an envelope: the engine sizes it)"""
    x0, x1, yb = DECK
    parts = [box(x0, x1, -yb, yb, DECK_Z0, DECK_TOP)]
    for mx, my in MOTORS:
        a = np.array([np.sign(mx) * 30.0, np.sign(my) * 30.0]); b = np.array([mx, my]); d = b - a; L = float(np.linalg.norm(d))
        arm = box(-L / 2, L / 2, -ARM_W / 2, ARM_W / 2, DECK_Z0, DECK_TOP)
        T = np.eye(4); c, s = d / L
        T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = (a + b) / 2
        arm.apply_transform(T); parts.append(arm)
        parts.append(cyl(mx, my, DECK_Z0, DECK_TOP, POD_R))
    for s in (1, -1):                                                # the side spars to S0 / S1: under the deck, dropping to the node at z 12
        parts.append(box(-SPAR_W / 2, SPAR_W / 2, *sorted((s * (yb - 2.0), s * (FLAT + 6.0))), ZB - 6.0, DECK_Z0 + 0.5))
    return union(parts)


def base():
    """the v3 base kept: the nose and tail runners on the receiver's outline (184 x 66.8 at the ledge, 4 mm 55 deg lead-ins) and the
    keel |y| <= 33.4 from the runners up to the deck, the receiver's room cut out; the nose boom forward to the camera and NB, the
    tail boom back to TB, both under the props' sweep less 12 (z <= 27.7)"""
    import manifold3d as mf
    xa, xb, W2 = CX - SEAT_L / 2, CX + SEAT_L / 2, SEAT_W / 2
    t = math.tan(math.radians(55.0))
    pts = []
    for x in (xa, xb):
        for s in (-1, 1):
            pts += [(x - (CH if x == xa else -CH) * 0, s * (W2 - CH), SEAT_Z), (x, s * W2, SEAT_Z + CH * t)]
    lower = trimesh.convex.convex_hull(np.array(pts + [(xa, s * W2, SEAT_Z + RUN_H) for s in (-1, 1)] + [(xb, s * W2, SEAT_Z + RUN_H) for s in (-1, 1)]
                                                 + [(xa + CH, s * (W2 - CH), SEAT_Z) for s in (-1, 1)] + [(xb - CH, s * (W2 - CH), SEAT_Z) for s in (-1, 1)], float))
    keel = box(xa, xb, -W2, W2, SEAT_Z + RUN_H - 0.5, DECK_Z0)
    nose_boom = box(xb - 2, FLAT + 4.0, -12.0, 12.0, 0.0, Z_PROP - Z_SWEEP - KEEP)
    tail_boom = box(-FLAT - 4.0, xa + 2, -12.0, 12.0, 0.0, 16.0)
    m = union([lower, keel, nose_boom, tail_boom])
    rec = trimesh.load(os.path.join(ROOT, "parts", "box_ctx_scout_receiver_v3", "part.stl"), force="mesh")
    lo, hi = rec.bounds
    cut = box(lo[0] - 0.5, hi[0] + 0.5, lo[1] - 0.5, hi[1] + 0.5, SEAT_Z - 1, hi[2] + 0.5)
    A = mf.Manifold(mf.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))
    B = mf.Manifold(mf.Mesh(np.asarray(cut.vertices, np.float32), np.asarray(cut.faces, np.uint32)))
    r = (A - B).to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3], np.asarray(r.tri_verts), process=False)


def prop(mx, my):
    """the swept disc (0.8 thick) and a 3 blade sketch, the hub"""
    disc = cyl(mx, my, Z_PROP - 0.4, Z_PROP + 0.4, PROP_R, 96)
    blades = []
    for k in range(3):
        a = math.radians(120 * k + (30 if mx * my > 0 else -30))
        b = box(0, PROP_R - 1, -6.0, 6.0, -1.2, 1.2); b.apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1])); b.apply_translation([mx, my, Z_PROP])
        blades.append(b)
    return disc, union(blades + [cyl(mx, my, Z_PROP - 3, Z_PROP + 3, 7.0)])


def moved(src, dx=0.0, dy=0.0, dz=0.0):
    m = trimesh.load(os.path.join(ROOT, "parts", src, "part.stl"), force="mesh"); m.apply_translation([dx, dy, dz]); return m


def cage():
    rods = [rod(P(i), P(j)) for i, j in RODS + TIES]
    nodes = [trimesh.creation.icosphere(subdivisions=1, radius=5.0).apply_translation(P(k)) for k in NODES if k not in PRINTED]
    nodes += [trimesh.creation.icosphere(subdivisions=1, radius=7.0).apply_translation(P(k)) for k in PRINTED]   # the outriggers' printed nodes
    return trimesh.util.concatenate(rods), trimesh.util.concatenate(nodes)


def frame_check(core_g):
    """the Scout's frame model (tools/scout_cage_loads.py, the standard cases: 35 g crash on the hull face the CG ray leaves through, 14
    directions, landing and rough landing 6 g on the runners) on these tables: the core one rigid body at the origin with the
    outriggers' printed nodes and the runner corners stiff to it"""
    import io, contextlib, scout_cage as sc, scout_cage_loads as sl
    sc.NODES, sc.RODS, sc.TIES, sc.TIE_ROOTS = dict(NODES), list(RODS), list(TIES), dict(ROOTS)
    sc.PRINTED = {k: "frame" for k in PRINTED}
    sc.BASE = {"G0": (CX + SEAT_L / 2, SEAT_W / 2, SEAT_Z), "G1": (CX - SEAT_L / 2, SEAT_W / 2, SEAT_Z), "G2": (CX - SEAT_L / 2, -SEAT_W / 2, SEAT_Z), "G3": (CX + SEAT_L / 2, -SEAT_W / 2, SEAT_Z)}
    sc.BRIDGES = []; sc.GROUPS = sc._groups(); sc.GROUP_OF = {k: g for g in sc.GROUPS for k in g}; sc.RODS_CARBON = list(RODS)
    sl.core_mass = lambda: (core_g, np.zeros(3))
    with contextlib.redirect_stdout(io.StringIO()):
        s_, rec = sl.run(nodes=sc.NODES, rods=sc.RODS, ties=sc.TIES, dry=True, log=lambda *a: print(*a))
    w = max(rec["cases"], key=lambda r: r["deflection_mm"])
    eu = max(rec["rods"].items(), key=lambda kv: kv[1]["euler"]); mo = max(rec["rods"].items(), key=lambda kv: kv[1]["moment"])
    return dict(mode_hz=rec["modes_hz"][0], deflection=(w["deflection_mm"], w["name"]), euler=(eu[1]["euler"], eu[0]), moment=(mo[1]["moment"], mo[0]),
                pull=max(v["tension"] for v in rec["rods"].values()))


def checks(meshes):
    out = {}
    gaps = [math.dist(MOTORS[a], MOTORS[b]) - PROP_D for a in range(4) for b in range(a + 1, 4)]
    out["tip gap"] = (round(min(gaps), 1), "neighbour props, tip to tip (3520 on v3: 19.1)")
    worst = (1e9, "")
    for i, j in RODS + TIES:
        pts = seg(P(i), P(j))
        d = min(disc_dist(p) for p in pts)
        if d < worst[0]:
            worst = (round(d, 1), f"{i}-{j}")
    out["rods to discs"] = (worst[0], f"{worst[1]} (need {KEEP:.0f})")
    blk = [f"{i}-{j}" for i, j in RODS + TIES if any(in_view(p) for p in seg(P(i), P(j)))]
    out["camera view"] = ("clear" if not blk else "BLOCKED by " + ", ".join(blk), "89 x 73 deg from lens (149, 0, 38), 30 deg up")
    disc_pts = [np.array([mx + PROP_R * math.cos(a), my + PROP_R * math.sin(a), Z_PROP]) for mx, my in MOTORS for a in np.linspace(0, 2 * math.pi, 90)]
    out["props in view"] = ("none" if not any(in_view(p) for p in disc_pts) else "IN VIEW", "the disc rims")
    cam = meshes["camera"]; cv = np.asarray(cam.vertices)
    out["camera to discs"] = (round(min(disc_dist(p) for p in cv[::4]), 1), "board and lens to the swept disc (need 12; v3 lens x 130: " + f"{min(disc_dist(p - (LENS - LENS_V3)) for p in cv[::4]):.1f})")
    for nm in ("frame", "base"):
        v = np.asarray(meshes[nm].vertices); top = v[:, 2].max()
        out[f"{nm} to discs"] = (round(min(disc_dist(p) for p in v if p[2] > DECK_Z0 - 7), 1), f"top z {top:.1f}; the props sweep down to {Z_PROP - Z_SWEEP:.1f}")
    return out


def write():
    meshes = {}
    meshes["frame"] = frame()
    meshes["base"] = base()
    for k, (mx, my) in enumerate(MOTORS, 1):
        sx, sy = np.sign(mx), np.sign(my)
        meshes[f"motor_{k}"] = moved("scout_motor_" + str(k), sx * (HUB - 54.1), sy * (HUB - 54.1), DECK_TOP - 22.0)
        meshes[f"prop_{k}"], meshes[f"blades_{k}"] = prop(mx, my)
    meshes["camera"] = moved("camera", *(LENS - LENS_V3))
    meshes["fc"] = moved("fc", 0, 0, DECK_TOP - 22.0)
    for src in ("ubec", "xr2", "radxa_zero_3w", "scout_flow_mtf_02p", "pack_4s_1500", "pack_xt60_plug", "box_ctx_scout_receiver_v3", "box_ctx_scout_cartridge_v6"):
        meshes[src] = moved(src)
    meshes["cage_rods"], meshes["cage_nodes"] = cage()
    for nm, m in meshes.items():
        d = os.path.join(ROOT, "parts", f"scout_v4_{nm.replace('box_ctx_scout_', '').replace('scout_', '')}"); os.makedirs(d, exist_ok=True)
        m.export(os.path.join(d, "part.stl"))
    return meshes


def entries(meshes):
    fid = lambda nm: f"scout_v4_{nm.replace('box_ctx_scout_', '').replace('scout_', '')}"
    rows = []
    def add(nm, name, role, group, color, mass, tag, note, material=None, opacity=None):
        e = dict(id=fid(nm), name=name, role=role, group=group, color=color, mesh=f"parts/{fid(nm)}/part.stl", mass_g=round(mass, 1), tag=tag, owner="scout v4", version="shape 8 Nov", source="tools/scout_v4_shape.py", note=note)
        if material:
            e["material"] = material
        if opacity is not None:
            e["opacity"] = opacity
        rows.append(e)
    fv = abs(meshes["frame"].volume); bv = abs(meshes["base"].volume)
    add("frame", "frame v4 room (printed PCTG deck, arms and motor pods: the design envelope)", "placeholder", "frame", "#d9d2c3", fv * PCTG * 0.45, "E",
        f"replaces the Bee35 plate: deck x {DECK[0]:.0f}..{DECK[1]:.0f}, |y| {DECK[2]:.0f}, z {DECK_Z0}..{DECK_TOP} (the plate's underside kept, the units bolt up into it), arms {ARM_W:.0f} wide to O{2 * POD_R:.0f} pods at (+-{HUB}, +-{HUB}); mass at 45 % of the envelope (E, the engine sizes it); printed in PCTG", None, 0.85)
    add("base", "base v4 room (the v3 nose and tail units' envelope: runners, keel, camera and tail booms)", "placeholder", "base", "#c9c2b5", bv * PCTG * 0.30, "E",
        "the v3 base kept so the funnel seat and the swapper do not change: runners on the 184 x 66.8 outline at z -44.3 with 55 deg lead-ins, keel |y| 33.4 to the deck, the receiver's room cut; nose boom to the camera and NB, tail boom to TB; mass at 30 % of the envelope (E); printed in PCTG", None, 0.55)
    for k in range(1, 5):
        add(f"motor_{k}", f"motor {k} (EMAX ECO II 2004)", "bought", "motors", "#8b8f96", 18.0, "listing", f"on its pod at (+-{HUB}, +-{HUB}), base z {DECK_TOP}")
        add(f"prop_{k}", f"prop {k} disc (Gemfan 5125, O{PROP_D}, swept)", "bought", "props", "#9fd3ff", 0.0, "listing", f"prop plane z {Z_PROP:.1f}", opacity=0.18)
        add(f"blades_{k}", f"prop {k} (Gemfan Hurricane SL 5125, 3 blade)", "bought", "props", "#3b6ea8", 2.8, "listing", "a sketch of the three blades")
    add("camera", "camera (DECXIN 2633V2, 30 deg up)", "bought", "electronics", "#222222", 10.0, "P", f"lens ({LENS[0]:.0f}, 0, {LENS[2]:.0f}), 19 forward of v3: its board clears the front discs' keep-out")
    add("fc", "FC MicoAir743 AIO (on top of the deck)", "bought", "electronics", "#1f6f3f", 10.0, "P", f"36 x 36 at the centre on the deck top z {DECK_TOP}")
    for src, nm, g, note in (("ubec", "UBEC", 15.0, "tail bay, as v3"), ("xr2", "XR2", 1.0, "tail bay, as v3"), ("radxa_zero_3w", "Radxa Zero 3W", 10.0, "tail bay on edge, as v3"),
                             ("scout_flow_mtf_02p", "flow MTF-02P", 1.5, "the nose runner's window, as v3")):
        add(src, nm, "bought", "electronics", "#2b5d8a", g, "P", note)
    add("pack_4s_1500", "pack 4S 1500 (in the cartridge)", "protected", "battery", "#c0392b", 165.0, "P", "the v3 pack, unchanged: the swapper's")
    add("pack_xt60_plug", "pack XT60 plug", "bought", "battery", "#e2b007", 10.0, "P", "as v3")
    add("box_ctx_scout_receiver_v3", "receiver v3 (the Overseer's latch receiver)", "structure", "battery", "#7f8c8d", 60.0, "E", "unchanged, at the v3 place: the funnel seat and the swap arm see the same belly")
    add("box_ctx_scout_cartridge_v6", "cartridge v6", "structure", "battery", "#95a5a6", 30.0, "E", "unchanged")
    L = sum(float(np.linalg.norm(P(i) - P(j))) for i, j in RODS + TIES)
    add("cage_rods", "half cage v4, rods (3 mm carbon: the ring, the flat roof, end Vs and side fans)", "cage", "cage", "#2b2b2b", L * ROD_G_MM, "E",
        f"{len(RODS)} rods, {L / 1000:.2f} m, every one 12 from the swept discs and none over or under a disc: ring at z {ZR:.0f} with flats at {FLAT:.0f} (310 across), a flat roof (spine on y 0, hoop on x 0, nose / tail triangles), a V at each end and a fan at each side down to the four outrigger nodes")
    add("cage_nodes", "half cage v4, nodes (spheres for now: the node design comes once the geometry is settled)", "cage", "cage", "#eef2f5", (len(NODES) - len(PRINTED)) * NODE_G, "E",
        f"{len(NODES) - len(PRINTED)} cage nodes; NB, TB, S0, S1 (the larger spheres) printed into the frame's outriggers", opacity=0.9)
    return rows


if __name__ == "__main__":
    M = write()
    L = sum(float(np.linalg.norm(P(i) - P(j))) for i, j in RODS)
    print(f"Scout v4 shape: props {PROP_D} on hubs +-{HUB} (wheelbase {2 * HUB * math.sqrt(2):.0f}), prop plane z {Z_PROP:.1f}, deck z {DECK_Z0}..{DECK_TOP}, ring flats {FLAT:.1f}, chamfer {CH_LINE:.1f}, ring and roof z {ZR:.1f}; "
          f"cage {len(NODES) - len(PRINTED)} nodes + {len(PRINTED)} printed, {len(RODS)} rods {L / 1000:.2f} m")
    for k, (v, why) in checks(M).items():
        print(f"  {k:16} {v}  ({why})")
    core = sum(r["mass_g"] for r in entries(M) if "cage" not in r["id"])
    fc = frame_check(core)
    print(f"  frame model     first mode {fc['mode_hz']:.0f} Hz, worst {fc['deflection'][0]:.1f} mm ({fc['deflection'][1][:60]}), Euler <= {fc['euler'][0]:.2f} ({fc['euler'][1]}), "
          f"end moment <= {fc['moment'][0]:.0f} N mm ({fc['moment'][1]}; socket cap 2000), pull-out <= {fc['pull']:.0f} N (cap 940)")
    F = os.path.join(ROOT, "systems", "scout_v4.assembly.json")
    old = json.load(open(F)) if os.path.exists(F) else {}
    rows = entries(M)
    lo = np.min([m.bounds[0] for m in M.values()], 0); hi = np.max([m.bounds[1] for m in M.values()], 0)
    mass = sum(r["mass_g"] for r in rows)
    a = dict(name="Scout v4", title="Scout v4", artifact=old.get("artifact"), machine="scout_v4", order=0, readme_view=old.get("readme_view", "iso"),
             note=f"Scout v4 shape design (8 Nov, Spark: new props, no Bee35, everything printed in PCTG, still seats in the landing funnel for the battery swapper): "
                  f"Gemfan 5125 props on a printed frame, the v3 base and battery unchanged, a half cage on the new discs. {lo[0]:.0f}..{hi[0]:.0f} x {lo[1]:.0f}..{hi[1]:.0f} x {lo[2]:.0f}..{hi[2]:.0f} mm, "
                  f"~{mass:.0f} g (E). tools/scout_v4_shape.py",
             mass_budget_g=None, parts=rows)
    open(F, "w").write(json.dumps(a, indent=0, ensure_ascii=False) + "\n")
    print(f"  extents         {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} x {hi[2] - lo[2]:.0f} mm; all-up ~{mass:.0f} g (E)")
    for r in sorted(rows, key=lambda r: -r["mass_g"])[:8]:
        print(f"    {r['mass_g']:6.1f} g  {r['name'][:70]}")
    print("  written: systems/scout_v4.assembly.json, parts/scout_v4_*/part.stl")
