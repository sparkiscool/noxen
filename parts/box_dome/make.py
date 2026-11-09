"""The box's rain and dust dome (9 Nov, Spark: "the same dome shaped rain and dust cover" as noxensys.com's dock), drawn.
As on the website (assets/dock/dock.js, illustrative there): a hemisphere of 16 fabric gores on one hinge axis along the box's
length (W Y), a rib on every other gore edge, a ring at the base; opening, every gore swings down to one side about the hinge,
so the open dome is a thin bundle at its base on that side, clear of the landing. Here it is sized on the real drones: one
dome over the funnel (both drones land in it, one at a time), centre W (0, 255) on a base plane at W Z 120 (25 under the box
top: the Overseer's cage feet hang 14 over the top outside the box), radius 490 = the smallest hemisphere holding the seated
Overseer (its head reaches W Y 627), the seated Scouts, the box's top corners and the funnel, + 30 (tools/seat_envelope's
parts, 9 Nov: 459 + 30).
Gore i spans hinge angles phi i*s..(i+1)*s closed (s = pi/16, phi 0 = the base on -W X, pi/2 = the crown, pi = the base on
+W X) at radius 490 + 3 i, so opening (every gore turning about the hinge to phi 0..s, -W X side) never passes one through
another. Ribs: 3 mm carbon rods bent to the hoop (0.3 % strain at r 490, two 1 m lengths joined at the crown by a printed
ferrule) on gores 0, 2 .. 14 at their phi i*s edge and on gore 15's free edge (the hoop the drive turns). The fabric is drawn
as 1.5 mm lunes, one per gore; the real skin is one coated ripstop sheet sewn to every hoop (it gathers between them open).
Base: a printed PCTG ring (segments pinned with 3 mm rods) r 482..540 at Z 110..120 on eight 6 mm carbon spokes from the
box's top edge, the two hinge hubs on it at W Y 255 -+ 490, the drive (a geared motor) at the rear hub, and a coated-fabric
apron sloping from the box top (Z 145) to the ring (Z 120) so rain runs off outward. Not drawn: the seal flap where the last
gore meets the ring on +W X, the hoop stops, the fabric's gathers.
W frame (the box's): X across, Y along the axle toward the front, Z up, mm.
Run: python parts/box_dome/make.py   (writes gore_meshes/gNN.stl and the fixed parts, then checks the clearances)
"""
import math, os, sys
import numpy as np, trimesh

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
YC, ZB, R = 255.0, 120.0, 490.0                  # the dome's centre (W X 0, W Y YC) on its base plane W Z ZB, inner radius
N = 16; S = math.pi / N; DR, SKIN, ROD = 3.0, 1.5, 1.5   # gores, their angle, the nesting step, fabric thickness drawn, rib radius (3 mm rod)
BOX = dict(x=209.0, y0=-69.0, y1=340.0, top=145.0)   # the enclosure sheets' outline and top
RING = (R - 8.0, R + N * DR + 7.0, ZB - 10.0, ZB)    # the base ring: r in, r out, z
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


def apron(t=1.0, n=128):
    """the fabric apron: from the box's top outline (Z 145) down to the ring (Z 120), a closed thin skin"""
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    inner = np.array([[*box_edge(a), BOX["top"]] for a in ang]); outer = ring_pts(RING[0] + 2.0, ZB, n)
    V = np.vstack([inner, outer, inner - [0, 0, t], outer - [0, 0, t]]); F = []
    for k in range(n):
        j = (k + 1) % n; a, b, c, d = k, j, n + j, n + k
        F += [[a, b, c], [a, c, d]]; F += [[2 * n + a, 2 * n + c, 2 * n + b], [2 * n + a, 2 * n + d, 2 * n + c]]
        F += [[a, 2 * n + a, 2 * n + b], [a, 2 * n + b, b]]; F += [[n + k, n + j, 3 * n + j], [n + k, 3 * n + j, 3 * n + k]]
    m = trimesh.Trimesh(V, np.array(F), process=True); trimesh.repair.fix_normals(m); return m


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
        e = box_edge(a); p0 = [e[0], e[1], BOX["top"] - 6.0]; d = np.array([-math.cos(a), math.sin(a)])
        p1 = [*(np.array([0, YC]) + d * (RING[0] + 4.0)), RING[2] + 2.0]
        spokes.append(trimesh.creation.cylinder(radius=3.0, segment=[p0, p1], sections=12))
    drive = trimesh.creation.box(extents=[42.0, 60.0, 42.0]); drive.apply_translation([0, YC - R - N * DR - 45.0, ZB + 2.0])   # geared motor on the rear hub's axis (placeholder)
    return dict(ring=ring, hubs=trimesh.util.concatenate(hubs), spokes=trimesh.util.concatenate(spokes), drive=drive)


def check():
    """the dome against the drones: shut over each seated drone (31 clear), and open (the bundle on -W X) against every descent
    state of every drone (tools/seat_check.descent_states). Returns the least clearances."""
    sys.path.insert(0, os.path.join(ROOT, "tools")); import seat_envelope as se, seat_check as sc
    S2W = np.array(sc.SEAT_TO_W); out = {}
    for d in ("overseer", "scout_v4"):
        P = np.vstack([np.asarray(m.vertices) for _, m in se.parts_of(d)])
        shut, opened = 1e9, 1e9
        for lab, ox, oy, h in sc.descent_states(d):
            Q = (S2W[:3, :3] @ (P + [ox, oy, h]).T).T + S2W[:3, 3]
            rel = Q - [0.0, YC, ZB]; dist = np.linalg.norm(rel, axis=1)
            phi = np.arctan2(rel[:, 2], -rel[:, 0])                         # hinge angle of each drone point
            up = rel[:, 2] > -1.0
            if lab == "seated":
                shut = float((R - dist[up]).min())                           # the shut dome's inside is at R
            r0, r1 = R, R + (N - 1) * DR + SKIN + 2 * ROD                     # the open bundle: phi 0..s, radius r0..r1 (on -W X, at the base)
            radial = np.maximum(0.0, np.maximum(r0 - dist, dist - r1))
            ang = dist * np.sin(np.clip(phi - S, 0.0, math.pi / 2))
            gap = np.where(up, np.hypot(radial, ang), 1e9)
            opened = min(opened, float(gap.min()))
        out[d] = (shut, opened)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "gore_meshes"), exist_ok=True)
    for i in range(N):
        gore(i).export(os.path.join(HERE, "gore_meshes", f"g{i:02d}.stl"))
    for k, m in base().items():
        m.export(os.path.join(HERE, "gore_meshes", f"{k}.stl"))
    apron().export(os.path.join(HERE, "gore_meshes", "apron.stl"))
    res = check()
    for d, (shut, opened) in res.items():
        print(f"{d}: shut dome {shut:.1f} clear of the seated drone; open bundle {opened:.1f} clear of it at every descent state")
        assert shut >= 25.0 and opened >= 10.0, f"the dome is in {d}'s way"
    print(f"dome r {R:.0f}..{r_of(N - 1) + SKIN:.0f} on W Z {ZB:.0f}, centre W Y {YC:.0f}: {N} gores, ring r {RING[0]:.0f}..{RING[1]:.0f}")
