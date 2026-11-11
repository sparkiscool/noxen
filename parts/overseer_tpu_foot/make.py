"""TPU boot for the Overseer's base nodes, v4 (11 Nov, Spark: "the foot nodes broke again ... it pretty much just snaps everywhere",
"a full in depth design"): ONE boot for all six nodes, on the stud the v7 nodes draw for it (parts/cage_node_lib.py: a disc O30 x 3.5,
0.6 chamfer on its bed edge, on a O24 neck 3.5 tall, under a O37 shoulder: the cup's rim bears on the shoulder, 0.4 of air under the disc). v3 (10 Nov) was four boots cast slice by slice from the lobes of the 29-31 Oct
PRINTED nodes; those nodes are being cut off the rods. The crush zone, the look and the print are v3's, the cup is now a plain socket.

WHICH MATERIAL TAKES THE HIT (the 11 Nov study; absorber_model.py, every number E):
  the node itself in TPU: 95A (E 8 MPa) cannot hold a glued rod or the base ring's shape; 68D is rigid (E 600..1190) but 11 MPa across
    layers: weaker than PCTG at the sockets and no stroke. A TPU skin on a PCTG core has no stroke either. The energy needs ~15 mm
    of travel under the node, so the absorber is its own part under a rigid node.
  95A columns (reusable; candidate B below) at this boot's 31 mm: 8 ribs 5.4 x 8 x 25.5 give a rising 260 -> 600 N, not a plateau:
    the drop peaks at 713 N and uses the whole stroke; 945 N on a spool 40 % soft, 770 on one 40 % hard. At 37.5 tall (v2's 32 mm
    ribs) it is 589 N, but Spark called 43 too tall. External spool only, ~35 g, 3.5 mm3/s.
  68D honeycomb (candidate E, built): a flat plateau, so 561 -> ~585 N at the same 31 mm (the block is 36 deep now), 632 / 763 on a
    soft / hard spool at v3's size; through the AMS, 19 g, 1 h. It folds for good in a full-energy hit and is replaced (a zip tie);
    in a normal landing (0.16 J) it moves 0.7 mm, elastic. The node passes the full 1300 N without it (its gate), so a spent boot
    is a lost margin, not a broken node.

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

THE LOOK (v3, Spark 10 Nov: "make the boot more aesthetic", then on the renders "too tall" and "it needs to be one body"). v2 was a
lobed cup standing on a 54.5 x 24.5 box with a separate box puck under it (and the cup was up to 34 deep in y on a 24.5 plate: it
overhung the plate and would have printed over air). v3 is ONE 68D body per node, 34 x 34.5 in plan, 31 under the landing face:
  Crush zone: the lower 28 of the body is the in-plane honeycomb, the same mechanism as v2's puck, cut into the body: two rows of
    regular cells (side 9.5, wall 2.0; a 3-2 pattern) between a 2.0 skin on the ground and the 3.0 plate under the landing face,
    inside a 2.0 rim, open through both y faces (the cells' axis is y: the boot prints on one of those faces, so the fold lines run
    along the layers, as v2's puck did). The block's x-z face is a rounded rectangle, R 8 at the ground and R 8 into the deck; the
    end rows run straight on to the skins from their side vertices (squared-off cells) so every wall meets a skin at 2.0 (a wedge
    between hex tips on the skin fails the solid check). Why 34 x 34.5: printed on its y face nothing may hang below that face, so
    the block is as deep as the widest cup (node 4, 34.2 in y); the plateau is sigma (t/l)^2 W D, so the face 50 x 20 -> 34 x 34.5
    (1173 mm2 for 1000) with the cell side 9.0 -> 9.5 keeps it: 537 N (v2 520). Height (Spark: "too tall" at 43): 2 + 26 + 3 = 31
    under the landing face; the plateau does not depend on the height, the stroke does: 19.8 mm to densification (was 25.6).
    Nominal spool (15 MPa): peak 561 N at 15.6 mm, 4.2 mm of margin (v2 525 at 16.3 with 9.3). A 40 % soft spool (9 MPa) now
    bottoms: 632 N (v2 438), still half of the 1300 N that cracked the nodes; a 40 % hard one 763 N at 9.7 (v2 718). absorber_model.py
    takes the face outline: the rounded ends fold first, so the plateau starts under the mean and rises onto it (E).
  Body: one piece from the ground to the lip, built slice by slice as the convex hull of two outlines, lofted smooth: the BLOCK
    (the face outline above in x, the flat y faces, rounding over the deck in y with R 3) and the CUP, the lobe's equator outline
    offset -2.5 at the ground growing (smoothstep) to the 3.0 wall at the equator, so the cup's wall flows out of the deck with a
    continuous tangent into the vertical wall, lip and groove of v1. The deck is at the landing-face level; the cup meets it flush
    in x, in y it stands 0..3 inside the block's faces.
  One body: nothing to assemble but the zip tie; after a hard landing the whole boot is replaced (18.5 g, ~1 h; v2 replaced a 19 g
    puck and kept an 11 g cup). Mass 18.2..18.8 g (v2 cup + puck 30).

Boot, per design (no topology run, drawn here): the cavity is the node's own lobe, slice by slice every 0.25 mm, taken INTF small
for a light interference grip; a mirror-y design's cavity is the union with its mirror, so one boot fits both nodes of a pair.
Above the equator the cavity closes in by up to LIP at 45 deg (prints without support) to the cup top at z -6.5, so the boot
snaps over the bulb and holds under its shoulder. 68D: INTF 0.08, LIP 0.5 (a 4 % hoop stretch going on; 1.2 would be 9 %, past
yield) and the 2.5 mm zip tie in the 3.0 x 1.0 groove in the cup's top 3 mm is the retention, not an option (it keeps the
lip from spreading over the equator; the tie does not reach the neck itself). 95A: INTF 0.15, LIP 1.2 (v1).
Wall 3.0 at the equator (>= 2.0 at the groove). 68D under the landing face: the PLATE 3.0 over the crush zone, the crush zone's 2.0 skin on the ground (THE
LOOK). 95A: the plate 3.0 O46, the ribs, the cone, a SOLE 2.5 O46. Rods, bores and socket mouths: the boot ends 0.9
under the lowest collar and every member's rod (r 3.1 + 0.5) is cut out of it anyway. v1's sole offsets for the funnel (node 2
+2 x, node 3 -5 x) are dropped: Spark 10 Nov, "dont worry about funnel for now" (v1's O48 x 15 sole crossed at the +x 18 capture
limit on nodes 1 and 3; the boot is now 31 under the landing face instead of 15: the funnel check, tools/seat_check.py
overseer --descent with the boots posed, is the next step).

Print (Bambu A1 mini, TPU for AMS 68D, through the AMS lite is fine; dry 4 h at 70 C first):
  Boots (6: node 5, node 2, node 3 x 2, node 4 x 2; print 2..4 spares, they are the crumple zone), as exported: on the -y face (a
    34 x 34 flat with the open cells, 34.5 tall), the cells vertical, the cup a horizontal socket (its crown spans 6.5 in z, so
    every perimeter there is a short bridge), the deck round-over and the lip under 45 deg; no supports, no brim, textured PEI +
    glue stick, Bambu "TPU for AMS" profile, 0.2 mm layers, 0.4 line, 3 walls, 100 % infill (the 2.0 cell walls come out solid: 3
    loops each side overlap), 230 C nozzle / 35 C bed, 8 mm3/s (~50 mm/s), retraction 0.8 mm at 30 mm/s, fan 100 % after layer 3;
    ~18.5 g, ~1 h each; three on a plate. Do not scale them: the wall thickness is the plateau.
  Fit: warm the cup in 60 C water, push it straight up onto the node, zip tie in the groove, pull it tight. If it is tight at the
    +y side of the lobe (the socket's printed crown), warm it longer. After a hard landing: look at the cells through the y faces;
    a folded boot is cut off (the zip tie) and replaced.
  If a 95A HF spool is bought (the _95a files, one piece each, ~41 g): spool on the external holder straight into the extruder,
    never through the AMS lite; Generic TPU / Bambu TPU 95A HF profile, 0.2 mm layers, 3 walls, 100 % infill, 230 / 35 C, max
    volumetric 3.5 mm3/s (~25 mm/s), retraction 0.4 mm, fan 60 % after layer 3; snap lip 1.2, the zip tie optional.

DROP TEST (5 minutes, calibrates the E numbers):
  1. Plateau: a boot on a spare or cracked (resined) node, on a bathroom scale, lean on the node with a hand: the cells should
     start folding between 40 and 55 kg (400..550 N). Under 30 kg the stroke runs out in a full drop (print the next boots with
     HEX_L 8.5); over 70 kg the node sees over 700 N (HEX_L 10.5). The 95A ribs: bow out and fold in the same
     window, and spring back straight within seconds (under 30 kg: RIB_T 6.0; over 70: 5.0).
  2. Drop: strap 1.65 kg to that node (a 1.5 l bottle filled to 1.65 kg on the scale), drop it boot down from 0.6 m onto
     concrete, once. Pass: the node unmarked, the boot still on, the cells folded over part of their height with every fold a
     crease, not a tear (a tear = the material is cracking across layers: the boot was printed in the wrong orientation, or
     wet). Measure the folded height: 31 minus it is the stroke used; 11..19 mm is on model; under 8 means the plateau is
     high (HEX_L up); over 19 means it bottomed (HEX_L down). Replace the boot. A 95A boot: three drops, the ribs recover, no
     cracks at the folds or the plate, no chalk on the cone tip (chalk the sole's centre first: cone on sole = bottomed).

    python parts/overseer_tpu_foot/make.py      writes print/overseer_tpu_foot_node{2,3,4,5}.stl (the 68D boots, on their print
                                                face), the _95a column boots, part.stl (the six boots on their nodes in the drone frame, the page's view) and preview.png
                                                (the boot on node 5 rendered, the section through it, force vs stroke v2 -> v3)
Needs the commits above in the clone (a shallow clone: git fetch --shallow-since=2026-10-29 origin main).
"""
import json, os, subprocess, io, sys
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
CAGE = json.load(open(os.path.join(ROOT, "systems", "overseer.cage.json")))
NODES = np.array(CAGE["nodes"], float)
from parts.cage_node_lib import FOOT_R, FOOT_T, NECK_R, NECK_T, FOOT_CH, NODE_R as _NODE_R   # the stud the v7 nodes draw
AIR = 0.4                                                              # under the disc: the boot bears on the node's shoulder by its rim, never on the disc (cage_node_lib docstring)
DESIGN = 2                                                             # the node the one boot is built and previewed on
MIRRORS = {DESIGN: (0, 1, 2, 3, 4, 5)}                                 # v4: one boot fits every base node (v3: four, cast from the printed lobes)
DZ = 0.25                                # slice step
Z_TOP = -(_NODE_R + 3.0) + FOOT_T + NECK_T   # cup top = the node's shoulder (-6.15, node frame): the rim bears on it
WALL, GROOVE_H, GROOVE_D = 3.0, 3.0, 1.0
PLATE, SOLE_T, SOLE_R, SOLE_ROUND = 3.0, 2.5, 23.0, 2.0   # plate 3.0: at 2.5 it was a neck under 3 mm between the cup and the ribs (solid check, node 2)
DECK_R = PLATE                           # the plate's top edge rounds over with this radius into the deck (THE LOOK)
CUP_IN, CUP_K = 2.5, 5.5                 # the cup outline at the skirt bottom is the equator offset -CUP_IN; it grows by CUP_K (smoothstep) to WALL at the equator
RIB_R_IN, RIB_L = 16.5, 32.0             # the ring's inner radius at the rib ends; rib length (the stroke is 60 % of it)
STOP_STROKE, STOP_R_BOT = 19.5, 9.0      # the cone under the cup floor meets the sole after this stroke; its tip radius
GRADES = {"68d": dict(intf=0.08, lip=0.5, rho=1.26, suffix=""),                                      # the design: the spool on hand
          "95a": dict(intf=0.15, lip=1.2, rho=1.20, suffix="_95a", n=12, t=5.5, w=8.0, bow=0.8, E=8.0)}   # the elastic column boot
CRUSH_W, CRUSH_H, CRUSH_D = 34.0, 30.0, 36.0  # the crush block (v2's puck, now part of the body, THE LOOK): width (drone x), height (the stroke axis: 2 skin + 26 of
                                           # cells + the 3 plate overlapping the top skin; Spark 10 Nov "too tall" at 40), depth (drone y: the cell axis, the print height)
HEX_ROWS = 2                               # rows of cells over the height, centred (3 at H 40)
CRUSH_R = 8.0                              # the block's face outline: corner radius at the ground and into the deck
HEX_L, HEX_T, SKIN = 9.5, 2.0, 2.0         # cell side (9.0 at the 50 x 20 face: the plateau is sigma (t/l)^2 W D), wall (the 2.0 floor: 5 lines of 0.4), the skins
ROD_R = 3.1 + 0.5
N_SEG = 96


def node_mesh(n):
    """the boot stud of a v7 base node, node frame (the same on every node): the disc with its bed chamfer and the neck, run on up
    past the cup top so every cup slice has a section"""
    z0 = -(_NODE_R + 3.0)
    m = mf.Manifold.cylinder(FOOT_CH, FOOT_R - FOOT_CH, FOOT_R, N_SEG).translate([0, 0, z0 - AIR])
    m += mf.Manifold.cylinder(FOOT_T - FOOT_CH + AIR, FOOT_R, FOOT_R, N_SEG).translate([0, 0, z0 - AIR + FOOT_CH])
    m += mf.Manifold.cylinder(NECK_T + 4.0, NECK_R, NECK_R, N_SEG).translate([0, 0, z0 + FOOT_T - 0.01])
    return to_tri(m)


def man(t):
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))


def mirror_y(cs):
    return cs.transform(np.array([[1.0, 0, 0], [0, -1.0, 0]]))


def section(m, z, sym):
    cs = m.slice(z)
    return cs + mirror_y(cs) if sym else cs


def slab(cs, z0, z1):
    return mf.Manifold.extrude(cs, z1 - z0).translate([0, 0, z0])


def rounded_rect(w, h, r):
    return mf.CrossSection.square([w - 2 * r, h - 2 * r], center=True).offset(r, mf.JoinType.Round, circular_segments=48)


def smoothstep(s):
    s = min(1.0, max(0.0, s)); return s * s * (3 - 2 * s)


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


def lobe(n, intf, lip):
    """the printed node's lobe: its mesh, manifold, landing face z0, equator z_eq and section S_eq, and the CAVITY (the lobe intf
    under, the 45 deg lip above the equator, v1) as slabs; node frame. Returns (t, m, z0, z_eq, S_eq, cavity_slabs, cav_cs: {z: cs})"""
    t = node_mesh(n); m = man(t)
    sym = len(MIRRORS[n]) > 1
    z0 = float(t.bounds[0][2])                                        # the landing face
    zs = np.arange(z0 + 0.02, Z_TOP + 1e-6, DZ)
    areas = [m.slice(z).area() for z in zs]
    z_eq = float(zs[int(np.argmax(areas))])                           # the bulb's equator (v4: the disc's edge above its chamfer)
    S_eq = section(m, z_eq, sym)
    cav, cav_cs = [], {}
    for z in zs:
        S = section(m, z, sym)
        if z <= z_eq:
            cs = S.offset(-intf, mf.JoinType.Round)
        else:
            cs = S_eq.offset(-min(lip, z - z_eq) - intf, mf.JoinType.Round) + S.offset(-intf, mf.JoinType.Round)
        lo, hi = (z - 0.5 * DZ if z > zs[0] else z0 - 0.01), (z + 0.5 * DZ + 0.01 if z < zs[-1] else Z_TOP + 1.0)   # the last slab opens the top (v2 left a 0.08 skin on three nodes)
        cav.append(slab(cs, lo, hi)); cav_cs[round(float(z), 4)] = cs
    return t, m, z0, z_eq, S_eq, mf.Manifold.batch_boolean(cav, mf.OpType.Add), cav_cs, zs


def star_radii(cs, th):
    """the outline's radius from the origin along each angle th (its outer ring; the max hit, so a slightly non-star outline is
    shrink-wrapped along the rays)"""
    polys = cs.to_polygons()
    def area(q):
        q = np.asarray(q, float); r = np.roll(q, -1, 0); return abs((q[:, 0] * r[:, 1] - q[:, 1] * r[:, 0]).sum())
    ring = max(polys, key=area)
    P = np.asarray(ring, float); E = np.roll(P, -1, 0) - P
    d = np.c_[np.cos(th), np.sin(th)]
    den = d[:, 0, None] * E[None, :, 1] - d[:, 1, None] * E[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (P[None, :, 0] * E[None, :, 1] - P[None, :, 1] * E[None, :, 0]) / den
        u = (P[None, :, 0] * d[:, 1, None] - P[None, :, 1] * d[:, 0, None]) / den
    ok = (u >= -1e-9) & (u <= 1 + 1e-9) & (t > 0) & np.isfinite(t)
    return np.where(ok, t, 0.0).max(1)


def loft(rings, n_th=240):
    """one smooth solid through outlines star-shaped about the z axis: rings [(z, CrossSection)] bottom to top, sampled at n_th
    angles and joined ring to ring (the skirt-to-lip body: v2 stacked slabs and showed every 0.25 mm step)"""
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    V, F = [], []
    for z, cs in rings:
        r = star_radii(cs, th); V.append(np.c_[r * np.cos(th), r * np.sin(th), np.full(n_th, z)])
    V = np.vstack(V); K = len(rings)
    for k in range(K - 1):
        a = k * n_th; b = a + n_th
        for i in range(n_th):
            j = (i + 1) % n_th
            F.append([a + i, a + j, b + i]); F.append([a + j, b + j, b + i])
    cb, ct = len(V), len(V) + 1
    V = np.vstack([V, [[0, 0, rings[0][0]], [0, 0, rings[-1][0]]]])
    for i in range(n_th):
        j = (i + 1) % n_th
        F.append([cb, j, i]); F.append([ct, (K - 1) * n_th + i, (K - 1) * n_th + j])
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(F, np.uint32)))
    assert m.status() == mf.Error.NoError and m.volume() > 0, m.status()
    return m


def body_outer(z0, z_eq, S_eq, cav_cs, zs, z_bot, P):
    """THE LOOK: the one-piece body from z_bot to Z_TOP, slice by slice the hull of P(z) (the skirt / plate outline, None above the
    deck) and the cup outline (the equator offset -CUP_IN at z_bot growing to WALL at the equator; v1's wall, lip and groove above),
    lofted into one smooth surface"""
    groove_z0 = Z_TOP - GROOVE_H                                      # the groove is the top 3 mm of the cup: no flange above it (a 1 mm one failed the solid check on node 2)
    def outline(z, groove):
        if z <= z_eq:
            cup = S_eq.offset(WALL - CUP_K * (1 - smoothstep((z - z_bot) / (z_eq - z_bot))), mf.JoinType.Round)
        else:
            zc = zs[np.argmin(np.abs(zs - z))]; cs = cav_cs[round(float(zc), 4)]
            cup = (S_eq + cs).offset(WALL, mf.JoinType.Round)         # the wall follows a lobe that widens above its equator
        p = P(z)
        o = (cup + p).hull() if p is not None else cup
        return o.offset(-GROOVE_D, mf.JoinType.Round) if groove else o   # the zip tie groove: wall 2.0 left
    rings = [(z_bot, outline(z_bot, False))]
    for z in np.arange(z_bot + DZ, groove_z0 - 1e-6, DZ):
        rings.append((z, outline(z, False)))
    rings.append((groove_z0, outline(groove_z0, False))); rings.append((groove_z0 + 1e-3, outline(groove_z0, True)))   # the groove's step
    for z in np.arange(groove_z0 + DZ, Z_TOP - 1e-6, DZ):
        rings.append((z, outline(z, True)))
    rings.append((Z_TOP, outline(Z_TOP, True)))
    return loft(rings)


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


def block_hx(zz):
    """the crush block's half-width (x) at height zz over the ground: a rounded rectangle CRUSH_W x (CRUSH_H + PLATE - SKIN), corner
    CRUSH_R at the ground and at the deck"""
    top = CRUSH_H + PLATE - SKIN
    if zz < CRUSH_R:
        return CRUSH_W / 2 - CRUSH_R + np.sqrt(max(0.0, CRUSH_R ** 2 - (CRUSH_R - zz) ** 2))
    if zz > top - CRUSH_R:
        return CRUSH_W / 2 - CRUSH_R + np.sqrt(max(0.0, CRUSH_R ** 2 - (zz - (top - CRUSH_R)) ** 2))
    return CRUSH_W / 2


def block_hw(y):
    """the block's half-width over its height from the centre of the crush zone (absorber_model's hw; y in [-CRUSH_H/2, CRUSH_H/2])"""
    return np.array([block_hx(float(v) + CRUSH_H / 2) for v in np.atleast_1d(y)])


def cells(zg, z0):
    """the honeycomb cut: regular pointy-top hexagons (side HEX_L, wall HEX_T) in HEX_ROWS rows between the SKIN over the ground and
    the PLATE under the landing face, inside the block's face outline less its 2.0 rim, open through both y faces (prisms along the
    cell axis y: printed on that face, the fold lines run along the layers). The end rows run straight on to the skins from their
    side vertices (squared-off cells) so every wall meets a skin at 2.0 (a wedge between hex tips failed the solid check)."""
    zs = np.linspace(0.0, z0 - zg, 121)
    face = mf.CrossSection([[[float(block_hx(z)), float(z)] for z in zs] + [[-float(block_hx(z)), float(z)] for z in zs[::-1]]])
    zc0, zc1 = SKIN, z0 - zg - PLATE                                  # the cell zone over the ground
    inner = face.offset(-HEX_T, mf.JoinType.Round) ^ mf.CrossSection.square([2 * CRUSH_W, zc1 - zc0]).translate([-CRUSH_W, zc0])
    zc = (zc0 + zc1) / 2
    r_hole = HEX_L - HEX_T / np.sqrt(3)                               # a hole's circumradius for a wall HEX_T between cells
    px, py = np.sqrt(3) * HEX_L, 1.5 * HEX_L
    holes = []
    for j in range(HEX_ROWS):
        for i in range(-4, 5):
            cx, cy = i * px + (px / 2 if j % 2 else 0.0), (j - (HEX_ROWS - 1) / 2) * py
            if abs(cx) - r_hole < CRUSH_W / 2 - HEX_T - 3.0:          # a cell reaching at least 3 mm into the block
                h = mf.CrossSection.circle(r_hole, 6).rotate(30).translate([cx, cy])
                if abs(cy) + r_hole > (zc1 - zc0) / 2:                 # an end row: straight on to the skin from the side vertices
                    ext = mf.CrossSection.square([np.sqrt(3) * r_hole, r_hole / 2 + 1.0]).translate([cx - np.sqrt(3) / 2 * r_hole, cy + r_hole / 2 if cy > 0 else cy - r_hole - 1.0])
                    ext = ext ^ mf.CrossSection.square([CRUSH_W - 4 * HEX_T, 4 * CRUSH_H], center=True)      # 2.0 kept to the side rim
                    h = h + ext
                holes.append(h.translate([0, zc]))
    cs = mf.CrossSection.batch_boolean(holes, mf.OpType.Add) ^ inner
    cut = mf.Manifold.extrude(cs, CRUSH_D + 2.0).rotate([90, 0, 0]).translate([0, CRUSH_D / 2 + 1.0, zg])   # (x, z) -> extruded along -y
    return cut, cs, face


def boot(n, grade="68d", page=False):
    g = GRADES[grade]
    t, m, z0, z_eq, S_eq, cavity, cav_cs, zs = lobe(n, g["intf"], g["lip"])
    zp = z0 - PLATE                                                   # the plate's underside
    if grade == "68d":
        zg = z0 - (CRUSH_H + PLATE - SKIN)                            # the ground: 2 skin + 26 cells + the 3 plate = 31 under the landing face
        def P(z):
            if z >= z0:
                return None
            hx = block_hx(z - zg)
            hy = CRUSH_D / 2 - (0.0 if z <= z0 - DECK_R else DECK_R - np.sqrt(max(0.0, DECK_R ** 2 - (z - z0 + DECK_R) ** 2)))   # the y faces flat (one is the bed), rounding over the deck
            return mf.CrossSection.square([2 * hx, 2 * hy], center=True)
        body = body_outer(z0, z_eq, S_eq, cav_cs, zs, zg, P)
        cut, cell_cs, face = cells(zg, z0)
        body = body - cut
        under = body
    else:
        zs_bot = zp - RIB_L                                           # the ribs' feet = the sole's top
        zg = zs_bot - SOLE_T                                          # the ground
        disc = mf.CrossSection.circle(SOLE_R, N_SEG)
        def P(z):
            if z <= zp:
                return disc
            if z < z0:
                return disc.offset(-(DECK_R - np.sqrt(max(0.0, DECK_R ** 2 - (z - zp) ** 2))), mf.JoinType.Round)
            return None
        cup = body_outer(z0, z_eq, S_eq, cav_cs, zs, zp, P)
        ribs = [rib(g["t"], g["w"], RIB_L, g["bow"], zs_bot, 360.0 * i / g["n"]) for i in range(g["n"])]
        stop_len = RIB_L - STOP_STROKE
        stop = mf.Manifold.cylinder(stop_len + 0.5, STOP_R_BOT, RIB_R_IN, N_SEG).translate([0, 0, zp - stop_len])
        sole = mf.Manifold.cylinder(SOLE_T - SOLE_ROUND + 0.5, SOLE_R, SOLE_R, N_SEG).translate([0, 0, zg + SOLE_ROUND])
        sole += mf.Manifold.cylinder(SOLE_ROUND, SOLE_R - SOLE_ROUND, SOLE_R - SOLE_ROUND, N_SEG).translate([0, 0, zg])
        sole += mf.Manifold.revolve(mf.CrossSection.circle(SOLE_ROUND, 32).translate([SOLE_R - SOLE_ROUND, SOLE_ROUND]), N_SEG).translate([0, 0, zg])
        under = mf.Manifold.batch_boolean([cup, stop, sole] + ribs, mf.OpType.Add)
        body = under
    if not page:                                                      # the page's view keeps the cavity full: the node on the page fills it, and its 70k stair faces decimated tore the cells (10 Nov)
        body = body - cavity                                          # overlaps the node by intf and, above the equator, by the lip
    sym = len(MIRRORS[n]) > 1
    rod_cut = mf.Manifold.batch_boolean(rods(n) + ([r.mirror([0, 1, 0]) for r in rods(n)] if sym else []), mf.OpType.Add)
    clash = under ^ rod_cut
    body -= rod_cut
    parts = sorted(body.decompose(), key=lambda q: -q.volume())
    body = parts[0]                                                   # slivers between the lobe and its mirror dropped
    over = (body ^ m).volume()                                        # the node pushed in: the lip + the interference
    thin = []
    for z in np.arange(z0 + 0.1, Z_TOP, 0.5):                         # cup wall under 2.0: what a 1.0 erosion and dilation of the slice loses
        B = body.slice(z); thin.append(B.area() - B.offset(-0.99, mf.JoinType.Round).offset(0.99, mf.JoinType.Round).area())
    bb = body.bounding_box()
    info = dict(node=n, grade=grade, fits=MIRRORS[n], z_face=round(z0, 2), z_eq=round(z_eq, 2), r_eq=round(float(np.sqrt(S_eq.area() / np.pi)), 1),
                height_under_face=round(z0 - zg, 1), rod_clash_mm3=round(clash.volume(), 1), node_overlap_mm3=round(over, 1), cup_wall_under_2_mm2=round(max(thin), 2),
                beyond_bed_face_mm=round(max(0.0, -bb[1] - CRUSH_D / 2, bb[4] - CRUSH_D / 2), 2) if grade == "68d" else None,   # the cup past the block's y faces: printed on one, nothing may hang below it
                dropped_mm3=round(sum(q.volume() for q in parts[1:]), 1), mass_g=round(body.volume() * 1e-3 * g["rho"], 1))
    return body.translate([0, 0, -zg]), info, -zg                     # ground at z 0; the node frame's shift


def to_tri(mm):
    g = mm.to_mesh(); return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts), process=False)


def render(meshes, elev=22.0, azim=-55.0, size=900, pad=1.08, light=(-0.4, -0.6, 0.7)):
    """a clean orthographic render: z-buffer rasteriser (numpy), Lambert + ambient per face, no library beyond numpy.
    meshes: [(trimesh, (r, g, b))]; elev/azim in deg (azim 0 looks along +x... matplotlib's convention: azim -90 looks along +y)"""
    el, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])        # camera direction (towards the camera)
    up = np.array([0, 0, 1.0]); right = np.cross(up, view); right /= np.linalg.norm(right); upv = np.cross(view, right)
    allv = np.vstack([m.vertices for m, _ in meshes]); c = (allv.max(0) + allv.min(0)) / 2
    pr = (allv - c) @ np.c_[right, upv]; half = np.abs(pr).max() * pad
    scale = (size - 1) / (2 * half)
    img = np.ones((size, size, 3)); zbuf = np.full((size, size), -np.inf)
    L = np.array(light, float); L /= np.linalg.norm(L)
    for m, col in meshes:
        V = m.vertices - c; F = m.faces
        px = (V @ right) * scale + size / 2; py = size / 2 - (V @ upv) * scale; pz = V @ view
        nrm = m.face_normals; shade = 0.35 + 0.65 * np.clip(nrm @ L, 0, 1) + 0.15 * np.clip(nrm @ view, 0, 1)
        col = np.asarray(col, float)
        for k in np.argsort(pz[F].mean(1)):                            # back to front, the z-buffer decides per pixel
            f = F[k]; x = px[f]; y = py[f]; z = pz[f]
            x0, x1 = int(max(0, np.floor(x.min()))), int(min(size - 1, np.ceil(x.max())))
            y0, y1 = int(max(0, np.floor(y.min()))), int(min(size - 1, np.ceil(y.max())))
            if x1 < x0 or y1 < y0:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            d = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            if abs(d) < 1e-9:
                continue
            l1 = ((X - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (Y - y[0])) / d
            l2 = ((x[1] - x[0]) * (Y - y[0]) - (X - x[0]) * (y[1] - y[0])) / d
            l0 = 1 - l1 - l2
            inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
            if not inside.any():
                continue
            Z = l0 * z[0] + l1 * z[1] + l2 * z[2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]; win = inside & (Z > sub)
            sub[win] = Z[win]
            img[y0:y1 + 1, x0:x1 + 1][win] = np.clip(col * shade[k], 0, 1)
    return img


def preview(boot68, node_tri, curves, path, info):
    """preview.png: the boot on node 5 rendered (node grey, boot blue, its crush zone orange), the section through the axis, force vs stroke v2 -> v3"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(19, 6.4))
    ax = fig.add_subplot(1, 3, 1)
    cols = dict(node=(0.62, 0.62, 0.64), cup=(0.16, 0.50, 1.0), crush=(1.0, 0.55, 0.16))
    dec = lambda m, n: m.simplify_quadric_decimation(face_count=n) if len(m.faces) > n else m
    z_deck = info["height_under_face"]
    hi, lo = boot68.split_by_plane([0, 0, 1.0], z_deck - PLATE); hi, lo = to_tri(hi), to_tri(lo)   # the crush zone and the cup, coloured apart
    img = render([(dec(node_tri, 30000), cols["node"]), (dec(hi, 30000), cols["cup"]), (dec(lo, 40000), cols["crush"])], elev=20, azim=-62)
    ax.imshow(img); ax.set_axis_off()
    ax.set_title(f"the one boot on a v7 node's stud: one 68D body, {info['mass_g']} g, {z_deck:.0f} under the landing face (cup blue, crush zone orange)", fontsize=9)
    ax = fig.add_subplot(1, 3, 2)
    for mesh, col, lab in ((to_tri(boot68), "#2a7fff", "boot (68D, one body; cells 9.5 / 2.0 along y)"), (node_tri, "#444444", "the node's stud (O30 disc, O24 neck)")):
        segs = trimesh.intersections.mesh_plane(mesh, [0, 1.0, 0], [0, 0, 0])
        for i, sgm in enumerate(segs):
            ax.plot(sgm[:, 0], sgm[:, 2], color=col, lw=1.0, label=lab if i == 0 else None)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("z from the ground (mm)"); ax.legend(loc="upper left", fontsize=7)
    ax.set_title(f"section y = 0 (stud overlap {info['node_overlap_mm3']} mm3 = the 0.08 interference + the 0.5 lip; cells clear)", fontsize=8)
    ax = fig.add_subplot(1, 3, 3)
    for lab, x, F, col, ls in curves:
        ax.plot(x, F, label=lab, color=col, lw=1.6, ls=ls)
    ax.axhline(1300, color="#aa0000", ls=":", lw=1); ax.text(19, 1230, "cage model's 1300 N (what cracked the nodes, E)", fontsize=7, color="#aa0000")
    ax.set_xlabel("crush stroke (mm)"); ax.set_ylabel("force at the node (N)"); ax.set_ylim(0, 1500); ax.set_xlim(0, 34); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("force vs stroke (E, absorber_model.py), 9.7 J drop balanced against the cage: v2 puck -> v3 one body", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    import absorber_model as am
    os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
    old = os.path.join(ROOT, "print", "overseer_tpu_foot_crush.stl")
    if os.path.exists(old):
        os.remove(old)                                                # v2's separate puck: the boot is one body now
    plate, keep = [], {}
    for stale in [f for f in os.listdir(os.path.join(ROOT, "print")) if f.startswith("overseer_tpu_foot_node")]:
        os.remove(os.path.join(ROOT, "print", stale))                 # v3's four per-node boots
    for grade in ("68d",):                                            # the 95A column boot (GRADES, rib) is not exported: the study above
        for i, n in enumerate((DESIGN,)):
            b, info, shift = boot(n, grade)
            tb = to_tri(b)
            if grade == "68d":
                tp = to_tri(b.rotate([90, 0, 0]))                     # the print file: on its -y face (the honeycomb's open face), the cell axis up
                nm = man(node_mesh(n)).translate([0, 0, shift])
                cut, cell_cs, face = cells(shift + info["z_face"] - (CRUSH_H + PLATE - SKIN), shift + info["z_face"])
                info["cells_node_overlap_mm3"] = round((cut ^ nm).volume(), 2)                    # the node must not reach the cells
                info["cell_zone_rel_density"] = round(1 - cell_cs.area() / (CRUSH_W * (CRUSH_H - 2 * SKIN)), 3)
                bp = boot(n, grade, page=True)[0]
                for k in MIRRORS[n]:                                  # the page's view (10 Nov, Spark: put the boots on): on the drone, at each node
                    bk = bp.translate([0, 0, -shift])                  # node frame
                    if k != n:
                        bk = bk.mirror([0, 1, 0])                      # the mirrored node of the pair
                    plate.append(bk.translate([float(v) for v in NODES[k]]))
                    os.makedirs(os.path.join(HERE, "page"), exist_ok=True)   # one mesh per boot: six in one entry got ~3k faces each on the page (boxy, Spark 10 Nov "looks weird"); 10k each keeps the page under 16 MB
                    pb = to_tri(plate[-1]); print("page boot", k, len(pb.faces))
                    pb.simplify_quadric_decimation(face_count=10000).export(os.path.join(HERE, "page", f"node{k}.stl"))
            else:
                tp = tb.copy()
            tp.apply_translation([0, 0, -float(tp.bounds[0][2])])
            tp.export(os.path.join(ROOT, "print", "overseer_tpu_foot.stl"))
            if n == DESIGN:
                keep[grade] = (b, info, shift)
            info.update(vol_mm3=round(b.volume()), height=round(tb.extents[2], 1), print_height=round(tp.extents[2], 1), watertight=bool(tb.is_watertight), genus=b.genus())
            print(json.dumps(info))
    p = to_tri(mf.Manifold.batch_boolean(plate, mf.OpType.Add))      # the page's view: the six 68D boots on their nodes, drone frame, decimated (print/ has the meshes)
    p.simplify_quadric_decimation(face_count=180000).export(os.path.join(HERE, "part.stl"))   # 10 Nov: 60k for six boots wiped the cells out on the page
    # the curves: v2's 50 x 40 x 20 box puck and v3's 34 x 34 x 30 block at sigma_y 15 and +-40 %
    curves = []
    c = am.honeycomb_curve(1190.0, 15.0, HEX_T, 9.0, 50.0, 20.0, 40.0)                            # v2: the 40 tall box, side 9.0
    pk2 = am.drop(lambda q: np.interp(q, c[:, 0], c[:, 1]), c[-1, 0])
    curves.append((f"v2 puck 50 x 40 x 20, cells 9.0 / 2.0, flow stress 15 MPa: peak {pk2[0]:.0f} N at {pk2[1]:.1f} mm", c[:, 0], c[:, 1], "#777777", "--"))
    for sy, lab, col in ((15.0, "v3 block 34 x 34 x 30, cells 9.5 / 2.0, 15 MPa: peak {:.0f} N at {:.1f} mm", "#ff8c2a"), (9.0, "   same at 9 MPa (-40 %): {:.0f} N at {:.0f} mm", "#ffc48c"), (21.0, "   same at 21 MPa (+40 %): {:.0f} N at {:.0f} mm", "#b35a00")):
        c = am.honeycomb_curve(1190.0, sy, HEX_T, HEX_L, CRUSH_W, CRUSH_D, CRUSH_H, hw=block_hw)
        pk3 = am.drop(lambda q: np.interp(q, c[:, 0], c[:, 1]), c[-1, 0])
        x = c[:, 0]; plateau = c[(x > 2) & (x < c[0, 3]), 1].mean()
        curves.append((lab.format(pk3[0], pk3[1]), x, c[:, 1], col, "-"))
        print(json.dumps(dict(curve="v3", sigma_y=sy, plateau_N=round(plateau), densify_mm=round(c[0, 3], 1), peak_N=round(pk3[0]), stroke_mm=round(pk3[1], 1), cage_mm=round(pk3[2], 1))))
    print(json.dumps(dict(curve="v2", sigma_y=15.0, peak_N=round(pk2[0]), stroke_mm=round(pk2[1], 1), cage_mm=round(pk2[2], 1))))
    tb, info, shift = keep["68d"]
    node_t = node_mesh(DESIGN); node_t.apply_translation([0, 0, shift])      # the node in the boot's ground frame (v2 drew it 26 low: z_face with the wrong sign)
    preview(tb, node_t, curves, os.path.join(HERE, "preview.png"), info)
    print("preview.png written")
