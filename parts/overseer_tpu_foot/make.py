"""TPU boots for the Overseer's PRINTED base nodes (10 Nov, Spark: "all bottom nodes have cracked, i resined them back but i need
better shock absorption ... the last version of the nodes we actually printed"; replaces the 8 Nov O30 x 6 glued cap, which fit the
O24 foot of the 8 Nov node designs, never printed).

Printed nodes (the meshes are the full part.stl of these commits, read with git show; mirrors share a design, overseer.cage.json):
  node 5  36fce92 (the 2239525 result of 30 Oct 03:02 UTC, "FINAL, printed 29 Oct", stray shells dropped)  1 boot
  node 2  3432732 (30 Oct 09:25 UTC)                                  1 boot
  node 3  188a69f (31 Oct 01:02 UTC), fits nodes 1 and 3              2 boots
  node 4  6457305 (30 Oct 23:53 UTC), fits nodes 0 and 4              2 boots
Every printed base node ends in the same kind of lobe: a bulb from the O16 landing face (z -13.2 from the node centre) widening to
its equator near z -11 (r 12..14) and necking in again to z -6, under the collars (horizontal tube collars from z -5.6 up).

Boot, per design (no topology run, drawn here): the cavity is the node's own lobe, slice by slice every 0.25 mm, taken 0.15 mm
small for a light interference grip; a mirror-y design's cavity is the union with its mirror, so one boot fits both nodes of a
pair. Above the equator the cavity closes in by up to LIP 1.2 mm at 45 deg (prints without support) to the cup top at z -6.5, so
the boot snaps over the bulb and holds under its shoulder; the TPU stretches 1.2 mm over the equator going on. A 3.0 x 1.0 groove
round the outside at the lip takes a 2.5 mm zip tie if the snap alone does not hold. Wall 3.0 at the equator (>= 2.0 at the
groove). Under the lobe: a SOLE 15 thick, O48 with a 5 mm round on the ground edge, flat on the bed (node 2's sole 2 mm to +x so
the seated drone's boot clears the enclosure top edge; node 3's 5 mm to -x, off the funnel end wall). Rods, bores and socket
mouths: the boot ends 0.9 under the lowest collar and every member's rod (r 3.1 + 0.5) is cut out of it anyway.
Funnel (tools/seat_check.py with the six boots posed on the nodes, 10 Nov): seated clear; descent 3 of 65 states cross, all the
node 3 cup on the funnel top rim at the +x 18 capture limit (h 9..16, up to 4 mm deep, 125 mm3): +x capture is about 14, not 18.

Print (Bambu A1 mini, TPU 95A): the spool on the external holder straight into the extruder, NOT through the AMS lite (95A
buckles in the AMS path; only Bambu "TPU for AMS", 68D, is made for it: if that is the spool, use 10 % gyroid, it is ~3x stiffer).
Generic TPU / Bambu TPU 95A HF profile: 0.2 mm layers, 2 walls, 3 top / 3 bottom layers, gyroid 20 %, no supports, no brim, flat
as exported (ground face down), textured PEI with glue stick, 230 C nozzle / 35 C bed, max volumetric 3.5 mm3/s (~25 mm/s walls),
retraction 0.4 mm, fan 60 %. All six on one plate in 3 x 2 at 55 mm pitch; ~10 g each. Fit: warm the boot in hot water (60 C),
push it straight up onto the node, check the lip has snapped over the bulb all round; a 2.5 mm zip tie in the groove if it walks.

Shock (E, no coupon): 1.65 kg from 0.6 m on one node = 9.7 J. The sole at 20 % gyroid, 2 walls crushes at about 0.5..0.6 MPa
(TPU 95A lattice, E) over ~1400 mm2 of flat contact: a plateau near 800 N. In series with the cage (1300 N over 15 mm, 87 N/mm,
3.7 J at 800 N) the sole takes 6.0 J over about 7.5 mm, half its 15 mm, before densifying: peak about 800..900 N at the node
instead of 1300 N.

    python parts/overseer_tpu_foot/make.py      writes print/overseer_tpu_foot_node{2,3,4,5}.stl and part.stl (the four, a plate)
Needs the commits above in the clone (a shallow clone: git fetch --shallow-since=2026-10-29 origin main).
"""
import json, os, subprocess, io
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CAGE = json.load(open(os.path.join(ROOT, "systems", "overseer.cage.json")))
NODES = np.array(CAGE["nodes"], float)
PRINTED = {5: "36fce92", 2: "3432732", 3: "188a69f", 4: "6457305"}     # design node -> commit of the printed result
MIRRORS = {5: (5,), 2: (2,), 3: (1, 3), 4: (0, 4)}                      # the nodes each boot is printed for
INTF, LIP, DZ = 0.15, 1.2, 0.25          # interference, snap lip, slice step
Z_TOP = -6.5                             # cup top, node frame (lowest collar -5.6)
WALL, GROOVE_H, GROOVE_D = 3.0, 3.0, 1.0
SOLE_T, SOLE_R, SOLE_ROUND = 15.0, 24.0, 5.0
SOLE_DX = {2: 2.0, 3: -5.0}             # sole centre off the node in x (drone frame): node 2 off the enclosure top edge seated, node 3 off the funnel end wall
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


def boot(n):
    t = node_mesh(n); m = man(t)
    sym = len(MIRRORS[n]) > 1
    z0 = float(t.bounds[0][2])                                        # the landing face
    zs = np.arange(z0 + 0.02, Z_TOP + 1e-6, DZ)
    areas = [m.slice(z).area() for z in zs]
    z_eq = float(zs[int(np.argmax(areas))])                           # the bulb's equator
    S_eq = section(m, z_eq, sym)
    cav, cup = [], []
    groove_z0 = min(z_eq + LIP, Z_TOP - GROOVE_H - 0.3)
    for z in zs:
        S = section(m, z, sym)
        if z <= z_eq:
            cs = S.offset(-INTF, mf.JoinType.Round)
        else:
            cs = S_eq.offset(-min(LIP, z - z_eq) - INTF, mf.JoinType.Round) + S.offset(-INTF, mf.JoinType.Round)
        lo, hi = (z - 0.5 * DZ if z > zs[0] else z0 - 0.01), z + 0.5 * DZ + 0.01
        cav.append(slab(cs, lo, hi))
        out = (S_eq + cs).offset(WALL, mf.JoinType.Round).simplify(0.02)         # the wall follows a lobe that widens above its equator
        if groove_z0 <= z <= groove_z0 + GROOVE_H:
            out = out.offset(-GROOVE_D, mf.JoinType.Round)                       # the zip tie groove: wall 2.0 left
        cup.append(slab(out, z0 - 0.5 if z == zs[0] else lo, min(hi, Z_TOP)))
    cavity = mf.Manifold.batch_boolean(cav, mf.OpType.Add)
    cup = mf.Manifold.batch_boolean(cup, mf.OpType.Add)
    outer_cs = S_eq.offset(WALL, mf.JoinType.Round)
    # sole: a disc, ground at z0 - SOLE_T, its ground edge a quarter round
    zg = z0 - SOLE_T
    dx = SOLE_DX.get(n, 0.0)
    disc = mf.Manifold.cylinder(SOLE_T - SOLE_ROUND, SOLE_R, SOLE_R, N_SEG).translate([0, 0, zg + SOLE_ROUND])
    disc += mf.Manifold.cylinder(SOLE_ROUND, SOLE_R - SOLE_ROUND, SOLE_R - SOLE_ROUND, N_SEG).translate([0, 0, zg])
    disc += mf.Manifold.revolve(mf.CrossSection.circle(SOLE_ROUND, 32).translate([SOLE_R - SOLE_ROUND, SOLE_ROUND]), N_SEG).translate([0, 0, zg])
    disc = disc.translate([dx, 0, 0])
    body = (disc + cup) - cavity                                      # overlaps the node by INTF and, above the equator, by the lip
    rod_cut = mf.Manifold.batch_boolean(rods(n) + ([r.mirror([0, 1, 0]) for r in rods(n)] if sym else []), mf.OpType.Add)
    clash = (disc + cup) ^ rod_cut
    body -= rod_cut
    parts = sorted(body.decompose(), key=lambda q: -q.volume())
    body = parts[0]                                                  # slivers between the lobe and its mirror dropped
    # push-on: the lobe going in from above meets the boot only at the lip (TPU stretches over the equator); its overlap volume
    push = [round((body ^ m.translate([0, 0, d])).volume(), 1) for d in (1.0, 2.0, 3.0, 5.0)]
    # push-on check: the node moved down onto the boot; the boot material it meets beyond the lip is the stretch
    over = (body ^ m).volume()
    thin = []
    for z in np.arange(z0 + 0.1, Z_TOP, 0.5):                        # wall under 2.0: what a 1.0 erosion and dilation of the slice loses
        B = body.slice(z); thin.append(B.area() - B.offset(-0.99, mf.JoinType.Round).offset(0.99, mf.JoinType.Round).area())
    info = dict(node=n, commit=PRINTED[n], fits=MIRRORS[n], z_face=round(z0, 2), z_eq=round(z_eq, 2), r_eq=round(float(np.sqrt(S_eq.area() / np.pi)), 1),
                outer=[round(v, 1) for v in outer_cs.bounds()], rod_clash_mm3=round(clash.volume(), 1), node_overlap_mm3=round(over, 1), wall_under_2_mm2=round(max(thin), 2), dropped_mm3=round(sum(q.volume() for q in parts[1:]), 1),
                push_on_overlap_mm3_at_1_2_3_5mm=push)
    return body.translate([0, 0, -zg]), info                         # ground at z 0, flat on the bed


def to_tri(mm):
    g = mm.to_mesh(); return trimesh.Trimesh(np.asarray(g.vert_properties)[:, :3], np.asarray(g.tri_verts), process=False)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
    plate = []
    for i, n in enumerate((5, 2, 3, 4)):
        b, info = boot(n)
        tb = to_tri(b)
        tb.export(os.path.join(ROOT, "print", f"overseer_tpu_foot_node{n}.stl"))
        plate.append(b.translate([(i % 2) * 55.0 - 27.5, (i // 2) * 55.0 - 27.5, 0]))
        info.update(vol_mm3=round(b.volume()), height=round(tb.extents[2], 1), watertight=bool(tb.is_watertight), genus=b.genus(),
                    min_z=round(float(tb.bounds[0][2]), 3))
        print(json.dumps(info))
    p = to_tri(mf.Manifold.batch_boolean(plate, mf.OpType.Add))      # the page's view: the four on one plate, decimated (print/ has the meshes)
    p.simplify_quadric_decimation(face_count=60000).export(os.path.join(HERE, "part.stl"))
