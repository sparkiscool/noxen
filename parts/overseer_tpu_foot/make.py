"""TPU boots for the Overseer's PRINTED base nodes, v2 (10 Nov design session; Spark 10 Nov: every bottom node cracked in landings,
resined back, "need much better shock absorption", "an actual design session for this, for best padding"; "its tpu for ams": the
spool is Bambu TPU for AMS, 68D). v1 (4fcba76, the same day) was the cup below over a plain O48 x 15 sole at 20 % gyroid; its cup
and zip tie groove are kept, its fit is adapted to 68D, its sole is replaced by a replaceable crush puck.

Printed nodes (the meshes are the full part.stl of these commits, read with git show; mirrors share a design, overseer.cage.json):
  node 5  36fce92 (the 2239525 result of 30 Oct 03:02 UTC, "FINAL, printed 29 Oct", stray shells dropped)  1 boot
  node 2  3432732 (30 Oct 09:25 UTC)                                  1 boot
  node 3  188a69f (31 Oct 01:02 UTC), fits nodes 1 and 3              2 boots
  node 4  6457305 (30 Oct 23:53 UTC), fits nodes 0 and 4              2 boots
Every printed base node ends in the same kind of lobe: a bulb from the O16 landing face (z -13.2 from the node centre) widening to
its equator near z -11 (r 12..14) and necking in again to z -6, under the collars (horizontal tube collars from z -5.6 up).

THE LOAD CASE. Worst case 1.65 kg from 0.6 m onto one node: 9.7 J at 3.4 m/s. The cage behind the node is a spring of about
87 N/mm (the v7 frame model, 1300 N over 15 mm), in series with the boot: the peak at the node is where boot energy + cage energy
reach 9.7 J. A flat force F over a stroke s stores F s; a linear spring stores only F s / 2, so for one stroke a plateau halves the
peak. The lever is stroke: the node's peak is about 9.7 J / stroke once the force is flat. Normal landing: 1.65 kg at ~1 m/s shared
by 4..6 feet, 0.13..0.2 J per foot.

THE MATERIAL (Bambu TDS, V1.0; measured, not hunches). "TPU for AMS" (68D, the spool on hand): Young's modulus 1190 MPa in X-Y and
600 in Z, tensile strength 22 (X-Y) / 11 (Z), elongation 650 % in X-Y but 31 % in Z. TPU 95A HF: 9.8 / 7.4 MPa, > 650 %. The two
differ by a factor of 100, not 3 (v1 said 3x), and 68D is a tough semi-rigid plastic, not a rubber: it yields near 2..3 % strain,
so its ELASTIC energy capacity is about 0.07 J/cm3 of beam at yield: 9.7 J would need 140 cm3 of 68D flexing at its limit. No
elastic 68D pad of this size exists. What 68D does superbly is yield and fold without breaking, IN PLANE (650 %): across layers
(Z) it cracks at 31 %. So the 68D absorber is a crumple zone, sacrificial in a full-energy crash, with its fold lines along the
layers: it prints on its side. It does not engage in a normal landing (the cage takes those, 1.9 mm at 170 N, as today); it caps
the node's load in a hard one and is replaced after (19 g, 30 min).

THE CANDIDATES (absorber_model.py: Gibson-Ashby honeycomb, a discrete elastica for columns, a foam curve for the lattice,
closed forms for the rest; every number E, the drop test below is the calibration):
  A. v1 plain O48 x 15 sole, 20 % gyroid, in 95A: TPU gyroid plateaus near 0.2 MPa at 25 % strain and hardens to 0.5 by 50 %,
     densifies past 55 %: 3.8 J in the 12.6 mm of lattice, the other 6 J goes into the cage and the densified sole: peak ~1060 N,
     bottomed out. In 68D the same sole is 100x stiffer: a rigid block, the node takes the 1300 N it takes today.
  E. IN-PLANE HONEYCOMB CRUSH PUCK, 68D (chosen for the spool on hand): a block 50 wide x 40 tall x 20 deep of regular hexagonal
     cells, 3 x 3 (side 9.0, wall 2.0: the 2 mm floor, which sets the cell size, 26 % dense in the cells, 2.0 skins top and
     bottom) printed on its side so the cells' axis is horizontal in use and the load crushes the cells in plane: the cell walls
     fold at plastic hinges whose stress runs along the layers. Plateau sigma_y (t/l)^2 / 1.5 over 50 x 20 = 520 N at sigma_y
     15 MPa, flat to 25.6 mm (64 % of the height), then densifies: the drop peaks at 525 N with 16.3 mm of puck and 6 mm of cage.
     Flow stress 40 % low (9 MPa): 312 N plateau, it runs into densification and peaks at 416 N at 28 mm; 40 % high (21, or
     dynamic hardening): 718 N at 10 mm. A normal landing (0.16 J) is 0.7 mm, elastic. Peak at first collapse ~1.2x the plateau; with only 3 x 3
     cells the collapse goes row by row (a chevron stack), the continuum formula is a guide, the test below the measure. 19 g;
     replace after a crash (the folds are permanent).
     The sole-of-the-boot alternative, a thin 68D skirt folding across layers (Abramowicz-Jones ~440 N), cracks at 31 % Z
     elongation and is under the 2 mm floor: not drawn.
  B. BUCKLING COLUMNS, 95A (the elastic design, if Spark buys a 95A HF spool: Bambu TPU 95A HF on Amazon, ~$30/kg; the _95a
     files): 12 ribs 5.5 x 8 x 32 on a ring r 16.5..22, bowed 0.8 outward so they fold outward smoothly. At E 8 the ring takes
     200 N at 1.6 mm, buckles, then holds 370 -> 610 N from 10 to 60 % of its length (mean 513): the drop peaks at 589 N with
     17.3 mm of boot stroke; E 40 % low 709 N at 20.3 mm, 40 % high 737 N at 11.9 mm; a normal landing 1.5 mm at 189 N. Fibre
     strain at the folds ~0.9, fine for 95A; a solid cone under the cup floor meets the sole past 19.5 mm as a 2nd stage. 41 g,
     reusable. Thick stubby ribs because at E 8 a 2 mm rib buckles at 3 N; fewer thicker ribs are lighter for one plateau.
  Rejected for both: a honeycomb or lattice carrying 500 N over O48 in 95A is over 60 % dense (no stroke); a bellows of 1.6 mm
  cones carries 0.3 N per cone in 95A and is a rigid can in 68D; a steel coil spring is linear (peak 2 E / s: 660 N for 30 N/mm
  over 22 mm, a 20 g bought part, no better than B); a tilted-beam snap-through lattice needs 1 mm beams at 20 deg the A1 mini
  cannot print in TPU; in 68D every elastic form fails on energy capacity (above). Graded stiffness: E is two-stage by nature
  (the cage elastic to 500 N, then the puck), B by its bow and cone.

Boot, per design (no topology run, drawn here): the cavity is the node's own lobe, slice by slice every 0.25 mm, taken INTF small
for a light interference grip; a mirror-y design's cavity is the union with its mirror, so one boot fits both nodes of a pair.
Above the equator the cavity closes in by up to LIP at 45 deg (prints without support) to the cup top at z -6.5, so the boot
snaps over the bulb and holds under its shoulder. 68D: INTF 0.08, LIP 0.5 (a 4 % hoop stretch going on; 1.2 would be 9 %, past
yield) and the 2.5 mm zip tie in the 3.0 x 1.0 groove in the cup's top 3 mm is the retention, not an option (it keeps the
lip from spreading over the equator; the tie does not reach the neck itself). 95A: INTF 0.15, LIP 1.2 (v1).
Wall 3.0 at the equator (>= 2.0 at the groove). 68D under the landing face: a PLATE 3.0 (54.5 x 24.5, the puck's 50 x 20 + 2.0
walls + 0.25 clearance) with a 6 mm skirt: the puck slides in from below and is CA-glued; the puck is the sole (its 2.0 bottom
skin on the ground). 95A: the plate 3.0 O46, the ribs, the cone, a SOLE 2.5 O46. Rods, bores and socket mouths: the boot ends 0.9
under the lowest collar and every member's rod (r 3.1 + 0.5) is cut out of it anyway. v1's sole offsets for the funnel (node 2
+2 x, node 3 -5 x) are dropped: Spark 10 Nov, "dont worry about funnel for now" (v1's O48 x 15 sole crossed at the +x 18 capture
limit on nodes 1 and 3; the boot is now 43 under the landing face instead of 15: the funnel check, tools/seat_check.py
overseer --descent with the boots posed, is the next step).

Print (Bambu A1 mini, TPU for AMS 68D, through the AMS lite is fine; dry 4 h at 70 C first):
  Cups (6: node 5, node 2, node 3 x 2, node 4 x 2), sole down as exported, no supports, no brim, textured PEI + glue stick,
    Bambu "TPU for AMS" profile, 0.2 mm layers, 0.42 line, 3 walls, 100 % infill, 230 C nozzle / 35 C bed, 8 mm3/s (~50 mm/s),
    retraction 0.8 mm at 30 mm/s, fan 100 % after layer 3; ~11 g, 45 min each; all six on one plate.
  Pucks (print 12: 6 + 6 spares, they are the crumple zone), as exported (the 50 x 40 honeycomb face on the bed, cells
    vertical, no bridging), same profile, 0.4 line, 3 walls, 100 % infill (the 2.0 walls come out solid: 3 loops each side
    overlap), 2 top / 2 bottom layers, ~19 g, 30 min each. Do not scale them: the wall thickness is the plateau.
  Fit: warm the cup in 60 C water, push it straight up onto the node, zip tie in the groove, pull it tight; puck into the skirt
    with a drop of CA on each 2.0 skin edge. After a hard landing: look at the pucks; a folded one is cut out and replaced.
  If a 95A HF spool is bought (the _95a files, one piece each, ~41 g): spool on the external holder straight into the extruder,
    never through the AMS lite; Generic TPU / Bambu TPU 95A HF profile, 0.2 mm layers, 3 walls, 100 % infill, 230 / 35 C, max
    volumetric 3.5 mm3/s (~25 mm/s), retraction 0.4 mm, fan 60 % after layer 3; snap lip 1.2, the zip tie optional.

DROP TEST (5 minutes, calibrates the E numbers):
  1. Plateau: cup + puck on a spare or cracked (resined) node, puck down on a bathroom scale, lean on the node with a hand: the
     puck's cells should start folding between 40 and 55 kg (400..550 N). Under 30 kg the stroke runs out in a full drop (print
     the next pucks with HEX_L 5.0); over 70 kg the node sees over 700 N (HEX_L 6.4). The 95A ribs: bow out and fold in the same
     window, and spring back straight within seconds (under 30 kg: RIB_T 6.0; over 70: 5.0).
  2. Drop: strap 1.65 kg to that node (a 1.5 l bottle filled to 1.65 kg on the scale), drop it puck down from 0.6 m onto
     concrete, once. Pass: the node unmarked, the cup still on, the puck folded over part of its height with every fold a
     crease, not a tear (a tear = the material is cracking across layers: the puck was printed in the wrong orientation, or
     wet). Measure the folded height: 40 minus it is the stroke used; 12..26 mm is on model; under 8 means the plateau is
     high (HEX_L up); over 27 means it bottomed (HEX_L down). Replace the puck. A 95A boot: three drops, the ribs recover, no
     cracks at the folds or the plate, no chalk on the cone tip (chalk the sole's centre first: cone on sole = bottomed).

    python parts/overseer_tpu_foot/make.py      writes print/overseer_tpu_foot_node{2,3,4,5}.stl (68D cups),
                                                print/overseer_tpu_foot_crush.stl (the 68D puck), the _95a column boots,
                                                part.stl (the four cups + pucks on a plate, the page's view) and preview.png
Needs the commits above in the clone (a shallow clone: git fetch --shallow-since=2026-10-29 origin main).
"""
import json, os, subprocess, io, sys
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
CAGE = json.load(open(os.path.join(ROOT, "systems", "overseer.cage.json")))
NODES = np.array(CAGE["nodes"], float)
PRINTED = {5: "36fce92", 2: "3432732", 3: "188a69f", 4: "6457305"}     # design node -> commit of the printed result
MIRRORS = {5: (5,), 2: (2,), 3: (1, 3), 4: (0, 4)}                      # the nodes each boot is printed for
DZ = 0.25                                # slice step
Z_TOP = -6.5                             # cup top, node frame (lowest collar -5.6)
WALL, GROOVE_H, GROOVE_D = 3.0, 3.0, 1.0
PLATE, SOLE_T, SOLE_R, SOLE_ROUND = 3.0, 2.5, 23.0, 2.0   # plate 3.0: at 2.5 it was a neck under 3 mm between the cup and the ribs (solid check, node 2)
RIB_R_IN, RIB_L = 16.5, 32.0             # the ring's inner radius at the rib ends; rib length (the stroke is 60 % of it)
STOP_STROKE, STOP_R_BOT = 19.5, 9.0      # the cone under the cup floor meets the sole after this stroke; its tip radius
GRADES = {"68d": dict(intf=0.08, lip=0.5, rho=1.26, suffix=""),                                      # the design: the spool on hand
          "95a": dict(intf=0.15, lip=1.2, rho=1.20, suffix="_95a", n=12, t=5.5, w=8.0, bow=0.8, E=8.0)}   # the elastic column boot
PUCK_W, PUCK_H, PUCK_D = 50.0, 40.0, 20.0  # the crush puck: width (drone x), height (the stroke axis: 3 rows end at their side vertices, 13.5 x 2 + 9 + 2 skins), depth (drone y, the print height)
HEX_L, HEX_T, SKIN = 9.0, 2.0, 2.0         # cell side, wall (the 2.0 floor: 5 lines of 0.4), the top and bottom skins; 3 x 3 cells
POCKET_CLR, POCKET_WALL, POCKET_DEEP = 0.25, 2.0, 6.0
ROD_R = 3.1 + 0.5
N_SEG = 96


def node_mesh(n):
    blob = subprocess.check_output(["git", "-C", ROOT, "show", f"{PRINTED[n]}:parts/overseer_cage_node_{n}/part.stl"])
    t = trimesh.load(io.BytesIO(blob), file_type="stl", force="mesh", process=True)
    t.merge_vertices(); t.apply_translation(-NODES[n])                                   # node frame: centre at 0, drone axes
    return t


def man(t):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))


def mirror_y(cs):
    return cs.transform(np.array([[1.0, 0, 0], [0, -1.0, 0]]))


def section(m, z, sym):
    cs = m.slice(z)
    return cs + mirror_y(cs) if sym else cs


def slab(cs, z0, z1):
    return mf.Manifold.extrude(cs, z1 - z0).translate([0, 0, z0])


def rods(n):
    """every member of node n as its rod: r ROD_R from the node centre out along the member (node frame)"""
    out, c = [], NODES[n]
    fars = [NODES[b if a == n else a] for a, b in CAGE["rods"] if n in (a, b)]
    for nm, (p, q) in CAGE["links"].items():
        for e, f in ((p, q), (q, p)):
            if np.linalg.norm(np.array(e) - c) < 0.5:
                fars.append(np.array(f, float))
    for f in fars:
        d = f - c; L = np.linalg.norm(d); u = d / L
        cyl = mf.Manifold.cylinder(L, ROD_R, ROD_R, 32)
        z = np.array([0, 0, 1.0]); v = np.cross(z, u); s = np.linalg.norm(v); cth = u @ z
        if s < 1e-9:
            R = np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); R = np.eye(3) + K + K @ K * ((1 - cth) / s ** 2)
        out.append(cyl.transform(np.hstack([R, np.zeros((3, 1))])))
    return out


def cup_of(n, intf, lip):
    """v1's cup: the cavity (the lobe 0.15 under, the 45 deg lip above the equator) and the 3.0 wall round it with the zip tie
    groove, node frame. Returns (cup, cavity, z_face, z_eq, S_eq)"""
    t = node_mesh(n); m = man(t)
    sym = len(MIRRORS[n]) > 1
    z0 = float(t.bounds[0][2])                                        # the landing face
    zs = np.arange(z0 + 0.02, Z_TOP + 1e-6, DZ)
    areas = [m.slice(z).area() for z in zs]
    z_eq = float(zs[int(np.argmax(areas))])                           # the bulb's equator
    S_eq = section(m, z_eq, sym)
    cav, cup = [], []
    groove_z0 = Z_TOP - GROOVE_H                                     # the groove is the top 3 mm of the cup: no flange above it (a 1 mm one failed the solid check on node 2)
    for z in zs:
        S = section(m, z, sym)
        if z <= z_eq:
            cs = S.offset(-intf, mf.JoinType.Round)
        else:
            cs = S_eq.offset(-min(lip, z - z_eq) - intf, mf.JoinType.Round) + S.offset(-intf, mf.JoinType.Round)
        lo, hi = (z - 0.5 * DZ if z > zs[0] else z0 - 0.01), z + 0.5 * DZ + 0.01
        cav.append(slab(cs, lo, hi))
        out = (S_eq + cs).offset(WALL, mf.JoinType.Round).simplify(0.02)         # the wall follows a lobe that widens above its equator
        if groove_z0 <= z <= groove_z0 + GROOVE_H:
            out = out.offset(-GROOVE_D, mf.JoinType.Round)                       # the zip tie groove: wall 2.0 left
        cup.append(slab(out, z0 - 0.5 if z == zs[0] else lo, min(hi, Z_TOP)))
    cavity = mf.Manifold.batch_boolean(cav, mf.OpType.Add)
    cup = mf.Manifold.batch_boolean(cup, mf.OpType.Add)
    return t, m, cup, cavity, z0, z_eq, S_eq


def rib(t, w, L, bow, z_bot, ang):
    """one column: thickness t (radial), width w (tangential), length L from z_bot up, bowed outward by bow at mid-height,
    at angle ang (deg) about the node axis"""
    s = np.linspace(0, 1, 17)
    inner = np.c_[RIB_R_IN + bow * np.sin(np.pi * s), s * (L + 0.5)]       # 0.5 into the plate for a clean union
    outer = inner[::-1] + np.array([t, 0.0])
    poly = np.vstack([inner, outer])[::-1]                              # counter-clockwise
    cs = mf.CrossSection([poly.tolist()])
    assert cs.area() > 0
    r = mf.Manifold.extrude(cs, w).translate([0, 0, -w / 2]).rotate([90, 0, 0])   # polygon (radial, axial) -> (x, z); the extrusion along -y
    return r.rotate([0, 0, ang]).translate([0, 0, z_bot])


def puck():
    """the 68D crush puck in its PRINT frame: x = PUCK_W, y = PUCK_H (the stroke axis in use), z = PUCK_D (the cells' axis, the print
    height). Regular pointy-top hexagons (side HEX_L, wall HEX_T) between two SKIN skins, HEX_T side walls."""
    W, H, D = PUCK_W, PUCK_H, PUCK_D
    outer = mf.CrossSection.square([W, H], center=True)
    inner = mf.CrossSection.square([W - 2 * HEX_T, H - 2 * SKIN], center=True)
    r_hole = HEX_L - HEX_T / np.sqrt(3)                               # a hole's circumradius for a wall HEX_T between cells
    px, py = np.sqrt(3) * HEX_L, 1.5 * HEX_L
    holes = []
    for j in range(-6, 7):
        for i in range(-6, 7):
            cx, cy = i * px + (px / 2 if j % 2 else 0.0), j * py
            if abs(cx) - r_hole < W / 2 - HEX_T - 3.0 and abs(cy) - r_hole < H / 2 - SKIN - 3.0:   # a cell reaching at least 3 mm into the block
                holes.append(mf.CrossSection.circle(r_hole, 6).rotate(30).translate([cx, cy]))
    cs = outer - (mf.CrossSection.batch_boolean(holes, mf.OpType.Add) ^ inner)
    return mf.Manifold.extrude(cs, D), cs


def puck_in_use(zp):
    """the puck under a plate whose underside is at zp (node frame): x width, z the stroke axis, y depth, centred on the node"""
    m, cs = puck()
    rho = cs.area() / (PUCK_W * PUCK_H)
    return m.rotate([90, 0, 0]).translate([0, PUCK_D / 2, -PUCK_H / 2]).translate([0, 0, zp]), rho


def boot(n, grade="68d"):
    g = GRADES[grade]
    t, m, cup, cavity, z0, z_eq, S_eq = cup_of(n, g["intf"], g["lip"])
    zp = z0 - PLATE                                                   # the plate's underside
    if grade == "68d":
        W, D = PUCK_W + 2 * POCKET_CLR, PUCK_D + 2 * POCKET_CLR
        plate = mf.Manifold.cube([W + 2 * POCKET_WALL, D + 2 * POCKET_WALL, PLATE + 0.5 + POCKET_DEEP], center=True).translate([0, 0, zp - POCKET_DEEP + (PLATE + 0.5 + POCKET_DEEP) / 2])
        pocket = mf.Manifold.cube([W, D, POCKET_DEEP + 1.0], center=True).translate([0, 0, zp - POCKET_DEEP + (POCKET_DEEP + 1.0) / 2 - 1.0])
        under = plate - pocket
        zg = zp - PUCK_H                                              # the ground: the puck's bottom skin
    else:
        zs_bot = zp - RIB_L                                           # the ribs' feet = the sole's top
        zg = zs_bot - SOLE_T                                          # the ground
        plate = mf.Manifold.cylinder(PLATE + 0.5, SOLE_R, SOLE_R, N_SEG).translate([0, 0, zp])
        ribs = [rib(g["t"], g["w"], RIB_L, g["bow"], zs_bot, 360.0 * i / g["n"]) for i in range(g["n"])]
        stop_len = RIB_L - STOP_STROKE
        stop = mf.Manifold.cylinder(stop_len + 0.5, STOP_R_BOT, RIB_R_IN, N_SEG).translate([0, 0, zp - stop_len])
        sole = mf.Manifold.cylinder(SOLE_T - SOLE_ROUND + 0.5, SOLE_R, SOLE_R, N_SEG).translate([0, 0, zg + SOLE_ROUND])
        sole += mf.Manifold.cylinder(SOLE_ROUND, SOLE_R - SOLE_ROUND, SOLE_R - SOLE_ROUND, N_SEG).translate([0, 0, zg])
        sole += mf.Manifold.revolve(mf.CrossSection.circle(SOLE_ROUND, 32).translate([SOLE_R - SOLE_ROUND, SOLE_ROUND]), N_SEG).translate([0, 0, zg])
        under = mf.Manifold.batch_boolean([plate, stop, sole] + ribs, mf.OpType.Add)
    body = (under + cup) - cavity                                     # overlaps the node by intf and, above the equator, by the lip
    sym = len(MIRRORS[n]) > 1
    rod_cut = mf.Manifold.batch_boolean(rods(n) + ([r.mirror([0, 1, 0]) for r in rods(n)] if sym else []), mf.OpType.Add)
    clash = (under + cup) ^ rod_cut
    body -= rod_cut
    parts = sorted(body.decompose(), key=lambda q: -q.volume())
    body = parts[0]                                                   # slivers between the lobe and its mirror dropped
    over = (body ^ m).volume()                                        # the node pushed in: the lip + the interference
    thin = []
    for z in np.arange(z0 + 0.1, Z_TOP, 0.5):                         # cup wall under 2.0: what a 1.0 erosion and dilation of the slice loses
        B = body.slice(z); thin.append(B.area() - B.offset(-0.99, mf.JoinType.Round).offset(0.99, mf.JoinType.Round).area())
    info = dict(node=n, grade=grade, commit=PRINTED[n], fits=MIRRORS[n], z_face=round(z0, 2), z_eq=round(z_eq, 2), r_eq=round(float(np.sqrt(S_eq.area() / np.pi)), 1),
                height_under_face=round(z0 - zg, 1), rod_clash_mm3=round(clash.volume(), 1), node_overlap_mm3=round(over, 1), cup_wall_under_2_mm2=round(max(thin), 2),
                dropped_mm3=round(sum(q.volume() for q in parts[1:]), 1), mass_g=round(body.volume() * 1e-3 * g["rho"], 1))
    return body.translate([0, 0, -zg]), info, zp - zg                 # ground at z 0, flat on the bed; the plate underside over the ground


def to_tri(mm):
    g = mm.to_mesh(); return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts), process=False)


def preview(cup68, puck68, node_tri, boot95, curves, path):
    """preview.png: the section through the axis (node + 68D cup + puck), the 68D boot and the 95A boot in 3D, the force-stroke curves"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 4, 1)
    for mesh, col, lab in ((cup68, "#2a7fff", "cup (68D)"), (puck68, "#ff8c2a", "crush puck (68D, in-plane honeycomb)"), (node_tri, "#444444", "printed node")):
        segs = trimesh.intersections.mesh_plane(mesh, [0, 1.0, 0], [0, 0, 0])
        for i, sgm in enumerate(segs):
            ax.plot(sgm[:, 0], sgm[:, 2], color=col, lw=1.0, label=lab if i == 0 else None)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("z from the ground (mm)"); ax.legend(loc="upper left", fontsize=7)
    ax.set_title("section y = 0, node 5: 68D cup, pocket, puck 50 x 40 x 20 (cells 9.0 / 2.0)", fontsize=8)
    for k, (mesh, col, ttl) in enumerate((([cup68, puck68], "#2a7fff", "68D: cup + puck, ~29 g, 43 under the landing face"), (boot95, "#55aa55", "95A option: 12 bowed ribs 5.5 x 8 x 32, ~41 g"))):
        ax = fig.add_subplot(1, 4, 2 + k, projection="3d")
        for b in (mesh if isinstance(mesh, list) else [mesh]):
            b = b.copy()
            if len(b.faces) > 15000:
                b = b.simplify_quadric_decimation(face_count=12000)
            ax.plot_trisurf(b.vertices[:, 0], b.vertices[:, 1], b.vertices[:, 2], triangles=b.faces, color=col, edgecolor="none", alpha=0.95, shade=True)
        lim = 30; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 2 * lim); ax.view_init(elev=18, azim=35); ax.set_axis_off(); ax.set_title(ttl, fontsize=8)
    ax = fig.add_subplot(1, 4, 4)
    for lab, x, F, col in curves:
        ax.plot(x, F, label=lab, color=col, lw=1.6)
    ax.axhline(1300, color="#aa0000", ls=":", lw=1); ax.text(0.3, 1320, "cage model's 1300 N (what cracked the nodes, E)", fontsize=7, color="#aa0000")
    ax.set_xlabel("boot stroke (mm)"); ax.set_ylabel("force at the node (N)"); ax.set_ylim(0, 1500); ax.set_xlim(0, 34); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("force vs stroke (E, absorber_model.py); 9.7 J drop balanced against the cage", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    import absorber_model as am
    os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
    pk, cs = puck(); tp = to_tri(pk); tp.export(os.path.join(ROOT, "print", "overseer_tpu_foot_crush.stl"))
    rho = cs.area() / (PUCK_W * PUCK_H)
    print(json.dumps(dict(puck="overseer_tpu_foot_crush", size=[PUCK_W, PUCK_H, PUCK_D], cell_side=HEX_L, wall=HEX_T, rel_density=round(rho, 3), mass_g=round(pk.volume() * 1e-3 * 1.26, 1),
                          watertight=bool(tp.is_watertight), genus=pk.genus(), cells=round((PUCK_W - 2 * HEX_T) * (PUCK_H - 2 * SKIN) / (2.598 * HEX_L ** 2), 1))))
    plate, keep = [], {}
    for grade in ("68d", "95a"):
        for i, n in enumerate((5, 2, 3, 4)):
            b, info, zp = boot(n, grade)
            tb = to_tri(b)
            tp = tb.copy(); tp.apply_translation([0, 0, -float(tb.bounds[0][2])])                 # the print file on the bed (the 68D cup's skirt bottom; the 95A sole)
            tp.export(os.path.join(ROOT, "print", f"overseer_tpu_foot_node{n}{GRADES[grade]['suffix']}.stl"))
            if grade == "68d":
                pu, _ = puck_in_use(zp)
                plate.append((b + pu).translate([(i % 2) * 55.0 - 27.5, (i // 2) * 55.0 - 27.5, 0]))
            if n == 5:
                keep[grade] = (tb, info, zp)
            info.update(vol_mm3=round(b.volume()), height=round(tb.extents[2], 1), watertight=bool(tb.is_watertight), genus=b.genus(), min_z=round(float(tb.bounds[0][2]), 3))
            print(json.dumps(info))
    p = to_tri(mf.Manifold.batch_boolean(plate, mf.OpType.Add))      # the page's view: the four 68D cups with their pucks on one plate, decimated (print/ has the meshes)
    p.simplify_quadric_decimation(face_count=60000).export(os.path.join(HERE, "part.stl"))
    # the curves: v1 sole, the 68D puck at sigma_y 15 and +-40 %, the 95A ribs at E 8 and +-40 %, a 30 N/mm steel spring
    curves = []
    hl = 12.6; xa = np.linspace(0, 0.55 * hl, 50); ea = xa / hl
    curves.append(("v1 (95A): O48 x 15 sole, 20 % gyroid: bottoms at 6.9 mm, peak ~1060 N", xa, 1810 * np.where(ea < 0.25, 0.8 * ea, 0.2 + 1.2 * (ea - 0.25)), "#999999"))
    for sy, lab, col in ((15.0, "v2 68D crush puck, flow stress 15 MPa: peak 525 N at 16.3 mm", "#ff8c2a"), (9.0, "   same at 9 MPa (-40 %): 416 N at 28 mm", "#ffc48c"), (21.0, "   same at 21 MPa (+40 %): 718 N at 10 mm", "#b35a00")):
        c = am.honeycomb_curve(1190.0, sy, HEX_T, HEX_L, PUCK_W, PUCK_D, PUCK_H); curves.append((lab, c[:, 0], c[:, 1], col))
    g = GRADES["95a"]
    for E, lab, col in ((8.0, "95A option: 12 ribs 5.5 x 8 x 32 at E 8: peak 589 N at 17.3 mm", "#2a7fff"), (5.6, "   same at E 5.6 (-40 %): 709 N at 20.3 mm", "#9ec2ff"), (11.2, "   same at E 11.2 (+40 %): 737 N at 11.9 mm", "#0a3f9f")):
        c = am.column_curve(E, g["t"], g["w"], RIB_L, g["bow"]); curves.append((lab, c[:, 0], g["n"] * c[:, 1], col))
    xs = np.linspace(0, 22, 20); curves.append(("steel coil spring 30 N/mm (linear): peak 660 N at 22 mm", xs, 30 * xs, "#55aa55"))
    tb, info, zp = keep["68d"]
    pu, _ = puck_in_use(zp); tpu = to_tri(pu)
    node_t = node_mesh(5); node_t.apply_translation([0, 0, info["height_under_face"] + info["z_face"]])      # the node in the boot's ground frame
    preview(tb, tpu, node_t, keep["95a"][0], curves, os.path.join(HERE, "preview.png"))
    print("preview.png written")
