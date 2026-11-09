"""The box's rain and dust dome over the Scout (9 Nov, Spark: "the same dome shaped rain and dust cover" as noxensys.com's dock;
then "i only need it as big as the scout, the overseer will not need to be in that dome"). As on the website
(assets/dock/dock.js, illustrative there): a hemisphere of 16 fabric gores on one hinge axis along the box's length (W Y), a rib
on every other gore edge, a ring at the base; opening, every gore swings down to one side about the hinge, so the open dome is a
thin bundle at its base on that side. Here: centre the seat, W (0, 168), base plane W Z 155 (the ring sits on the box's rain lid
at the sheets' top, 145); radius 263 = the smallest hemisphere holding the seated Scout v4 above the base (233, its camera as of 9 Nov) + 30. It folds to
-W X: within 280 of the seat on that side the Overseer has nothing at all, and it lands with the dome open (its low cage feet,
W Z ~160, are all within 240 of the seat, inside the ring), so the Overseer never goes under the dome.
Gore i spans hinge angles phi i*s..(i+1)*s shut (s = pi/16, phi 0 = the base on -W X, pi/2 the crown, pi the base on +W X) at
radius 263 + 2 i, so opening (every gore turning about the hinge to phi 0..s) never passes one through another. Ribs: 3 mm carbon
rods bent to the hoop (0.6 % strain at r 263) on gores 0, 2 .. 14 at their phi i*s edge and on gore 15's free edge (the hoop the
drive turns). The fabric is drawn as 1.5 mm lunes, one per gore; the real skin is one coated ripstop sheet sewn to every hoop.
Base: a printed PCTG ring (segments pinned with 3 mm rods) on the lid's rim, the two hinge hubs on it at W Y 168 -+ 263, eight 6 mm
carbon spokes under the lid from the sheets' tops to the ring, a geared motor on the rear hub (placeholder), and the rain lid: a
3 mm polycarbonate sheet at the sheets' top (W Z 142..145), flush between them and as a flange outside them out to the ring, open within r 160 of
the seat (the funnel, the 60 / 120 deg bays and the swap head rise there; the dome covers it), the funnel's beams standing on it.
Not drawn: the seal flap where the last gore meets the ring on +W X, the hoop stops, the fabric's gathers.
W frame (the box's): X across, Y along the axle toward the front, Z up, mm.
Run: python parts/box_dome/make.py   (writes gore_meshes/*.stl, then checks every dome part against both drones, shut and open)
"""
import math, os, sys
import numpy as np, trimesh

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
YC, ZB, R = 168.0, 155.0, 263.0                  # the dome's centre (W X 0, W Y YC) on its base plane W Z ZB, inner radius
N = 16; S = math.pi / N; DR, SKIN, ROD = 2.0, 1.5, 1.5   # gores, their angle, the nesting step, fabric thickness drawn, rib radius (3 mm rod)
BOX = dict(x=209.0, y0=-69.0, y1=340.0, top=145.0)   # the enclosure sheets' outline and top
RING = (R - 8.0, R + N * DR + 7.0, ZB - 10.0, ZB)    # the base ring: r in, r out, z (on the lid, 145)
LID = (142.0, 145.0, 160.0)                          # the rain lid: z, and the radius it stays open within
SPOKES = 8


def r_of(i):
    return R + DR * i


def sph(phi, th, r):
    """W point on the dome sphere: hinge angle phi (0 at -X on the base, pi/2 the crown), polar th from the hinge axis (0 at the front hub)"""
    return np.array([-r * math.sin(th) * math.cos(phi), YC + r * math.cos(th), ZB + r * math.sin(th) * math.sin(phi)])


def lune(p0, p1, r, t, nphi=3, nth=48):
    """a closed shell: the sphere between hinge angles p0..p1, radius r..r+t"""
    V, F = [], []
    grid = [(p0 + (p1 - p0) * a / nphi, math.pi * b / nth) for a in range(nphi + 1) for b in range(nth + 1)]
    for rr in (r, r + t):
        V += [sph(ph, th, rr) for ph, th in grid]
    n = len(grid); idx = lambda a, b, layer: layer * n + a * (nth + 1) + b
    for a in range(nphi):
        for b in range(nth):
            q = [idx(a, b, 1), idx(a + 1, b, 1), idx(a + 1, b + 1, 1), idx(a, b + 1, 1)]; F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            q = [idx(a, b, 0), idx(a, b + 1, 0), idx(a + 1, b + 1, 0), idx(a + 1, b, 0)]; F += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    for a in (0, nphi):                                                   # the two meridian edges
        for b in range(nth):
            o, i_ = (idx(a, b, 1), idx(a, b + 1, 1)), (idx(a, b, 0), idx(a, b + 1, 0))
            F += ([[i_[0], o[1], o[0]], [i_[0], i_[1], o[1]]] if a == 0 else [[i_[0], o[0], o[1]], [i_[0], o[1], i_[1]]])
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True); m.merge_vertices(); trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()                                                        # outward normals (the faces above are wound inward)
    return m


def hoop(phi, r):
    """a rib: the 3 mm rod bent to the half hoop at hinge angle phi, radius r (outside the fabric)"""
    pts = np.array([sph(phi, th, r) for th in np.linspace(0.02, math.pi - 0.02, 60)])
    return trimesh.util.concatenate([trimesh.creation.cylinder(radius=ROD, segment=[a, b], sections=8) for a, b in zip(pts[:-1], pts[1:])])


def gore(i):
    """gore i closed (its own part: it turns about the hinge axis), with its rib"""
    r = r_of(i); parts = [lune(i * S, (i + 1) * S, r, SKIN)]
    if i % 2 == 0:
        parts.append(hoop(i * S, r + SKIN + ROD))
    if i == N - 1:
        parts.append(hoop((i + 1) * S, r + SKIN + ROD))                    # the free edge's hoop, the one the drive turns
    return trimesh.util.concatenate(parts)


def turn(phi_add):
    """4x4 (W) turning a gore about the hinge axis by phi_add (its hinge angles + phi_add)"""
    c, s = math.cos(phi_add), math.sin(phi_add)
    T = np.eye(4); T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    A = np.eye(4); A[:3, 3] = (0.0, 0.0, -ZB); B = np.eye(4); B[:3, 3] = (0.0, 0.0, ZB)
    return B @ T @ A


def open_turn(i, u):
    """gore i's turn at open fraction u (0 shut, 1 open): every gore down to phi 0..s on -W X"""
    return turn(-i * S * u)


def ring_pts(r, z, n=128):
    return np.array([[-r * math.cos(a), YC + r * math.sin(a), z] for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])


def box_edge(a):
    """where the ray from the dome centre at angle a meets the box's top outline"""
    d = np.array([-math.cos(a), math.sin(a)]); ts = []
    for k, lim in ((0, BOX["x"]), (0, -BOX["x"]), (1, BOX["y0"] - YC), (1, BOX["y1"] - YC)):
        if abs(d[k]) > 1e-9:
            t = lim / d[k]
            if t > 0: ts.append(t)
    t = min(t for t in ts if abs((d * t)[0]) <= BOX["x"] + 1e-6 and BOX["y0"] - YC - 1e-6 <= (d * t)[1] <= BOX["y1"] - YC + 1e-6)
    return np.array([(d * t)[0], YC + (d * t)[1]])


def lid(n=128):
    """the rain lid: the box's top outline and the ring's disk, less the opening within LID[2] of the seat, a 3 mm sheet"""
    import manifold3d as mf
    def prism(poly, z0, z1):
        return mf.Manifold.extrude(mf.CrossSection([poly]), z1 - z0).translate([0, 0, z0])
    disk = [(RING[1] * math.cos(t), YC + RING[1] * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]   # counter-clockwise (a clockwise loop is a hole)
    hole = [(LID[2] * math.cos(t), YC + LID[2] * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    outline = [(-BOX["x"], BOX["y0"]), (BOX["x"], BOX["y0"]), (BOX["x"], BOX["y1"]), (-BOX["x"], BOX["y1"])]
    sheets = [mf.Manifold.cube([x1 - x0, y1 - y0, 12.0]).translate([x0, y0, LID[0] - 6.0]) for x0, x1, y0, y1 in
              ((206.0, 209.0, -63.0, 334.0), (-209.0, -206.0, -63.0, 334.0), (-203.0, 203.0, 337.0, 340.0), (-203.0, 203.0, -69.0, -66.0))]   # the enclosure sheets pass through: the lid sits flush between them, its flange outside them
    m = (prism(outline, LID[0], LID[1]) + prism(disk, LID[0], LID[1])) - prism(hole, LID[0] - 1, LID[1] + 1) - mf.Manifold.batch_boolean(sheets, mf.OpType.Add)
    t = m.to_mesh(); return trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=True)


def base():
    """the printed ring, the two hinge hubs, the eight 6 mm carbon spokes (each as its own mesh: ring, hubs, spokes, drive)"""
    ring = trimesh.creation.annulus(r_min=RING[0], r_max=RING[1], height=RING[3] - RING[2], sections=128)
    ring.apply_translation([0, YC, (RING[2] + RING[3]) / 2])
    hubs = []
    for sy in (-1, 1):
        y0, y1 = YC + sy * (R - 6.0), YC + sy * (R + N * DR + 12.0)
        h = trimesh.creation.cylinder(radius=22.0, segment=[[0, y0, ZB], [0, y1, ZB]], sections=32); hubs.append(h)
    spokes = []
    for k in range(SPOKES):
        a = 2 * math.pi * k / SPOKES + math.pi / SPOKES
        e = box_edge(a); d = np.array([-math.cos(a), math.sin(a)]); p0 = [*(np.array(e) - d * 30.0), LID[0] - 4.0]
        p1 = [*(np.array([0, YC]) + d * (RING[1] - 6.0)), LID[0] - 4.0]                                  # under the lid, from inside the sheets out to the ring
        spokes.append(trimesh.creation.cylinder(radius=3.0, segment=[p0, p1], sections=12))
    drive = trimesh.creation.box(extents=[42.0, 60.0, 42.0]); drive.apply_translation([0, YC - R - N * DR - 45.0, ZB + 4.0])   # geared motor on the rear hub's axis (placeholder)
    return dict(ring=ring, hubs=trimesh.util.concatenate(hubs), spokes=trimesh.util.concatenate(spokes), drive=drive)


def check():
    """the dome against the drones (tools/seat_check.descent_states, every state): shut, its inside (R) against the seated Scout v4;
    open, the gores' bundle (phi 0..s, radius R..R+(N-1)DR+skin+rib) against the Overseer and the Scout v4 at every state; the fixed
    parts (ring, hubs, drive, lid, spokes) against both at every state, as solids. Returns the least clearances / crossings."""
    sys.path.insert(0, os.path.join(ROOT, "tools")); import seat_envelope as se, seat_check as sc
    S2W = np.array(sc.SEAT_TO_W); W2S = np.linalg.inv(S2W); out = {}
    fixed = dict(base(), lid=lid())
    fx = {k: sc.man(m.copy().apply_transform(W2S)) for k, m in fixed.items()}
    assert all(v is not None for v in fx.values()), f"not a solid: {[k for k, v in fx.items() if v is None]}"
    for d in ("overseer", "scout_v4"):
        parts = se.parts_of(d)
        P = np.vstack([np.asarray(m.vertices) for _, m in parts])
        man = [(nm, sc.man(m) or sc.man(m.convex_hull), m.bounds) for nm, m in parts]
        shut, opened, hits = None, 1e9, []
        for lab, ox, oy, h in sc.descent_states(d):
            Q = (S2W[:3, :3] @ (P + [ox, oy, h]).T).T + S2W[:3, 3]
            rel = Q - [0.0, YC, ZB]; dist = np.linalg.norm(rel, axis=1); up = rel[:, 2] > -1.0
            phi = np.arctan2(rel[:, 2], -rel[:, 0])
            if lab == "seated" and d == "scout_v4":
                shut = float((R - dist[up]).min())
            r0, r1 = R, R + (N - 1) * DR + SKIN + 2 * ROD
            radial = np.maximum(0.0, np.maximum(r0 - dist, dist - r1)); ang = dist * np.sin(np.clip(phi - S, 0.0, math.pi / 2))
            opened = min(opened, float(np.where(up, np.hypot(radial, ang), 1e9).min()))
            for k, f in fx.items():
                for nm, dm, bb in man:
                    if dm is None or f is None: continue
                    v = sc.volume(f ^ dm.translate([ox, oy, h]))
                    if v > 0.5: hits.append(f"{k} x {nm[:30]} at {lab} {v:.0f} mm3")
        out[d] = (shut, opened, hits)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "gore_meshes"), exist_ok=True)
    for f in os.listdir(os.path.join(HERE, "gore_meshes")):
        os.remove(os.path.join(HERE, "gore_meshes", f))
    for i in range(N):
        gore(i).export(os.path.join(HERE, "gore_meshes", f"g{i:02d}.stl"))
    for k, m in base().items():
        m.export(os.path.join(HERE, "gore_meshes", f"{k}.stl"))
    lid().export(os.path.join(HERE, "gore_meshes", "lid.stl"))
    for f in sorted(os.listdir(os.path.join(HERE, "gore_meshes"))):                      # no empty or open part (9 Nov: the lid came out empty and the check skipped it)
        t = trimesh.load(os.path.join(HERE, "gore_meshes", f), force="mesh")
        assert len(t.faces) > 0 and t.volume > 1.0, f"{f}: empty ({len(t.faces)} faces, {t.volume:.0f} mm3)"
    res = check()
    for d, (shut, opened, hits) in res.items():
        print(f"{d}: " + (f"shut dome {shut:.1f} clear of it seated; " if shut is not None else "") + f"open bundle {opened:.1f} clear of it at every state; fixed parts: {len(hits)} crossings" + "".join("\n  " + h for h in hits[:8]))
        assert (shut is None or shut >= 25.0) and opened >= 10.0 and not hits, f"the dome is in {d}'s way"
    print(f"dome r {R:.0f}..{r_of(N - 1) + SKIN:.0f} on W Z {ZB:.0f}, centre W Y {YC:.0f}: {N} gores, ring r {RING[0]:.0f}..{RING[1]:.0f}, lid open within r {LID[2]:.0f}")
