"""The Scout's rain and dust dome and its folding mechanism (9 Nov, Spark: the dome of noxensys.com's dock, "only as big as the
scout, the overseer will not need to be in that dome", then "lets actually design the mechanism for the cover").

What it is
  A hemisphere of coated ripstop on five 3 mm carbon hoops, all pivoting on one hinge axis along the box's length (W Y, at X 0,
  Z 170), centred on the seat (W Y 168). Shut, the hoops stand at hinge angles 0, 45, 90, 135, 180 deg (0 = the base on -W X,
  90 the crown, 180 the base on +W X). Open, all five lie at 0, a bundle on the -W X side of the base: within 280 of the seat
  on that side the Overseer has nothing (tools/seat_check.descent_states), so it lands with the dome open; the dome is the
  Scout's only (radius: the seated Scout v4 + 28).
  Hoop k (k 0..4) is one 1 m rod bent to radius 263 + 9 k (0.6 % strain), its ends in hub plate k at each end of the axis. The
  hoops nest 9 apart, so they fold past each other and their plates stack on the axis 9.1 apart.

How it folds: a folding fan
  Hoop 0 is fixed (its plate is the hub bracket). At each hub, plates 1..4 turn on one 6 mm shoulder bolt; plate k+1 carries a
  3 mm carbon pin at r 24 that rides in a 45 deg slot in plate k, so neighbouring hoops stay 0..45 deg apart. A self-locking
  worm-gear motor at each hub turns plate 4 only (the free edge's hoop): opening, hoop 4 sweeps from 180 down to 0 and picks up
  hoops 3, 2, 1 in turn as it reaches them; shutting, it rises and pulls each one out after it once their slot is at its end,
  so they end 45 apart. Shut, hoop 4 is held by the worm and hoop 0 is fixed; the slots fix every hoop between (gravity pulls
  the low hoops down onto their slot ends), so nothing sags. A cam on each motor's outer shaft trips one of two roller limit switches at each end of the
  travel and stops the motor; the box lands a drone only with the dome open and confirmed.
  The fabric is one silnylon sheet sewn in sleeves to the five hoops (drawn as four sub-strips per panel, which turn to follow
  the hoops; open, they gather in the bundle), with a valance hanging over the ring's outer edge from hoops 0 and 4 against
  wind-driven rain, and a silicone seam seal.

Base
  A printed PCTG ring (r 255..297, W Z 145..155, segments pinned with 3 mm carbon rods) on the rain lid's rim; the two hub
  brackets on it (a block under the plate stack, plate 0, and an outer cheek that carries the bolt's end, the motor and the
  switches); eight 6 mm carbon spokes under the lid from the sheets' tops out to the ring; the rain lid itself: a 3 mm UV
  polycarbonate sheet at the sheets' top (W Z 142..145), flush between them and as a flange outside them out to the ring, open
  within r 160 of the seat (the funnel, the 60 / 120 deg bays and the swap head rise there, under the dome).

Bought (Amazon, Spark 29 Oct)
  hoops     KARBXON 3 x 1000 mm pultruded carbon rod            https://www.amazon.com/KARBXON-Carbon-Pultruded-Planes-Drones-Projects/dp/B09Q8CKC2L
  motors    JGY-370 12 V 6 rpm self-locking worm gear motor, double output shaft, 2   https://www.amazon.com/JGY370-Double-Electric-Self-Locking-6RPM-150RPM/dp/B0DL2S9GPZ
  switches  KW11-3Z roller lever micro switch, 4 of a 10-pack     https://www.amazon.com/clp/B0B342RXJM
  pivots    socket shoulder screw, 6 mm shoulder x 50, 2          https://www.amazon.com/Socket-Shoulder-Screws-Bolts-50MM/dp/B009TE1YYE
  fabric    emma kites 20D silnylon, white, 60 x 36 in, 2         https://www.amazon.com/emma-kites-Waterproof-Ultralight-Resistant/dp/B09VWZQT22
  lid       24 x 24 in 3 mm UV-stabilised polycarbonate sheet    https://www.amazon.com/clp/B01KZG8NBQ
  spokes    6 mm carbon tube (the project's stock)
W frame (the box's): X across, Y along the axle toward the front, Z up, mm.
Run: python parts/box_dome/make.py   (writes gore_meshes/*.stl, then checks every part against both drones, shut, moving, open)
"""
import math, os, sys
import numpy as np, trimesh

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
YC, ZB = 168.0, 157.0                           # the dome's centre: W (0, YC), its hinge axis at W Z ZB
PLAN = math.radians(90.0)                       # the dome is drawn with its axis along W Y folding to -W X, then turned 90 deg about the centre: axis along W X, hubs left and right, folding to the rear (-W Y), where the Overseer leaves the most room
R0, DH, NH = 263.0, 9.0, 5                      # hoop 0's radius, the step between hoops, the hoops
SHUT = [0.0, 45.0, 90.0, 135.0, 180.0]          # hoop angles shut (deg)
GAP = 45.0                                      # the slot: neighbouring hoops 0..45 apart
ROD = 1.5                                       # 3 mm carbon rod
RHO_END, RHO_SOCK, RHO_PIN = 30.0, 50.0, 24.0   # the rod's end and the socket's mouth from the axis, the pin's radius
PLATE_T, PLATE_R = 7.4, 32.0                    # hub plates: thickness, sector radius
M = 4                                           # fabric sub-strips per panel
FAB = 0.8                                       # fabric thickness drawn
RING = (255.0, 297.0, 132.0, 142.0)             # the base ring: r in, r out, z (beside the lid, its inner lip under the lid's edge)
LID = (142.0, 145.0, 160.0, 270.0)              # the rain lid: z, the radius it stays open within, its edge (on the ring)
BOX = dict(x=209.0, y0=-69.0, y1=340.0, top=145.0)
SPOKES = 8


def r_of(k):
    return R0 + DH * k


def y_plate(k):
    """hoop k's plate plane, from the dome centre along the axis (the rod's end region, rho 30..50)"""
    return r_of(k) * math.cos(math.asin(40.0 / r_of(k)))


def sph(phi, th, r):
    """W point: hinge angle phi (rad), polar th from the axis (0 at the front hub), radius r"""
    return np.array([-r * math.sin(th) * math.cos(phi), YC + r * math.cos(th), ZB + r * math.sin(th) * math.sin(phi)])


def G():
    """the plan turn about the dome's centre: drawn (axis W Y, folding to -W X) -> placed (axis W X, folding to -W Y)"""
    c, s_ = math.cos(PLAN), math.sin(PLAN)
    T = np.eye(4); T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s_, s_, c
    A = np.eye(4); A[1, 3] = -YC; B = np.eye(4); B[1, 3] = YC
    return B @ T @ A


def placed(m):
    t = m.copy(); t.apply_transform(G()); return t


def turn(dphi):
    """4x4 (W): turn about the hinge axis by dphi (rad), hinge angles + dphi"""
    c, s = math.cos(dphi), math.sin(dphi)
    T = np.eye(4); T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    A = np.eye(4); A[2, 3] = -ZB; B = np.eye(4); B[2, 3] = ZB
    return B @ T @ A


def tube(pts, r=ROD, sections=8):
    return trimesh.util.concatenate([trimesh.creation.cylinder(radius=r, segment=[a, b], sections=sections) for a, b in zip(pts[:-1], pts[1:])])


def hoop_rod(k, phi):
    """hoop k's rod at hinge angle phi (rad): the bent arc from the sockets' mouths, and the straight ends into the sockets"""
    r = r_of(k); t0 = math.asin(RHO_SOCK / r)
    arc = [sph(phi, th, r) for th in np.linspace(t0, math.pi - t0, 56)]
    ends = []
    for sgn, th in ((1, t0), (-1, math.pi - t0)):
        y = YC + sgn * y_plate(k); d = np.array([-math.cos(phi), 0.0, math.sin(phi)])
        p_mouth = np.array([0.0, y, ZB]) + d * RHO_SOCK; p_end = np.array([0.0, y, ZB]) + d * RHO_END
        ends.append(tube([sph(phi, th, r), p_mouth, p_end]))
    return trimesh.util.concatenate([tube(arc)] + ends)


def sector(phi0, phi1, rad, y0, y1, n=16):
    """a flat sector of the axis's disc (hinge angles phi0..phi1 rad, radius rad), between y0..y1 (W), with the bolt's bore"""
    pts = [(0.0, 0.0)] + [(-rad * math.cos(a), rad * math.sin(a)) for a in np.linspace(phi0, phi1, n)]
    poly = trimesh.path.polygons.Polygon([(x, z) for x, z in pts]).buffer(0)
    m = trimesh.creation.extrude_polygon(poly, y1 - y0)                       # in (x, z) extruded along local z
    m.apply_transform(np.array([[1.0, 0, 0, 0], [0, 0, 1.0, y0], [0, 1.0, 0, ZB], [0, 0, 0, 1]]))   # local (x, y=z_poly, z=extrusion) -> W (x, z, y)
    return m


def plate(k, phi, sgn):
    """hub plate k at hinge angle phi (rad), at the front (sgn +1) or rear (-1) hub: a sector round the bore (the slot's range
    for k < 4), an arm out to the rod's socket, the hub ring round the bolt; the bore, the socket and the slot cut"""
    import manifold3d as mf
    yc = YC + sgn * y_plate(k); y0, y1 = yc - PLATE_T / 2, yc + PLATE_T / 2
    sec = sector(phi - math.radians(8), phi + math.radians(GAP + 8 if k < NH - 1 else 8), PLATE_R, y0, y1)
    d = np.array([-math.cos(phi), 0.0, math.sin(phi)]); n = np.array([math.sin(phi), 0.0, math.cos(phi)])
    arm_pts = []
    for a in (RHO_END - 6, RHO_SOCK + 2):
        for b in (-7.0, 7.0):
            arm_pts.append(np.array([0, 0, ZB]) + d * a + n * b)
    arm = trimesh.convex.convex_hull(np.vstack([[p + [0, y0, 0] for p in arm_pts], [p + [0, y1, 0] for p in arm_pts]]))
    hubring = trimesh.creation.cylinder(radius=8.0, segment=[[0, y0, ZB], [0, y1, ZB]], sections=24)
    man = lambda t: mf.Manifold(mf.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
    body = man(sec) + man(arm) + man(hubring)
    cuts = [man(trimesh.creation.cylinder(radius=3.1, segment=[[0, y0 - 1, ZB], [0, y1 + 1, ZB]], sections=24)),                           # the bolt's bore
            man(trimesh.creation.cylinder(radius=1.6, segment=list(np.array([0, yc, ZB]) + np.outer([RHO_END, RHO_SOCK + 3], d)), sections=12))]  # the rod's socket
    if k < NH - 1:                                                                                                                  # the 45 deg slot for plate k+1's pin
        arc = [np.array([0, 0, ZB]) + RHO_PIN * np.array([-math.cos(a), 0, math.sin(a)]) for a in np.linspace(phi, phi + math.radians(GAP), 12)]
        cuts += [man(trimesh.creation.cylinder(radius=1.7, segment=[p + [0, y0 - 1, 0], p + [0, y1 + 1, 0]], sections=12)) for p in arc]
        segs = []
        for i in range(len(arc) - 1):
            a_, b_ = arc[i], arc[i + 1]; t_ = (b_ - a_) / np.linalg.norm(b_ - a_); n2 = np.cross(t_, [0, 1.0, 0])
            corners = [p + n2 * s + [0, yy, 0] for p in (a_, b_) for s in (-1.7, 1.7) for yy in (y0 - 1, y1 + 1)]
            segs.append(man(trimesh.convex.convex_hull(np.array(corners))))
        cuts += segs
    m = body - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)
    if k > 0:                                                                                                                       # the pin into plate k-1's slot
        p = np.array([0, 0, ZB]) + RHO_PIN * d
        ys, ye = (y0, y0 - (y_plate(k) - y_plate(k - 1) - PLATE_T) - 5.0) if sgn > 0 else (y1, y1 + (y_plate(k) - y_plate(k - 1) - PLATE_T) + 5.0)
        m = m + man(trimesh.creation.cylinder(radius=1.5, segment=[p + [0, ys + sgn * 1.0, 0], p + [0, ye, 0]], sections=12))
    t = m.to_mesh(); return trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=True)


def hoop(k, phi=None):
    """hoop k (rod and both plates, pins) at its hinge angle (rad); default shut"""
    phi = math.radians(SHUT[k]) if phi is None else phi
    return trimesh.util.concatenate([hoop_rod(k, phi), plate(k, phi, 1), plate(k, phi, -1)])


def strip(k, j):
    """fabric sub-strip j of panel k (between hoops k and k+1), shut: a lune at hinge angles 45 k + 45 j / M .. + 45 / M, outside
    hoop k+1's rod, from just past the hub plates' arms (rho 55) round to the other hub"""
    r = r_of(k + 1) + ROD + 1.5 + 0.3 * j; p0 = math.radians(SHUT[k] + GAP * j / M); p1 = p0 + math.radians(GAP / M)
    tf = math.asin(55.0 / r); nphi, nth = 2, 40
    grid = [(p0 + (p1 - p0) * a / nphi, tf + (math.pi - 2 * tf) * b / nth) for a in range(nphi + 1) for b in range(nth + 1)]
    V = [sph(ph, th, rr) for rr in (r, r + FAB) for ph, th in grid]; n = len(grid)
    idx = lambda a, b, l: l * n + a * (nth + 1) + b; F = []
    for a in range(nphi):
        for b in range(nth):
            F += [[idx(a, b, 1), idx(a + 1, b, 1), idx(a + 1, b + 1, 1)], [idx(a, b, 1), idx(a + 1, b + 1, 1), idx(a, b + 1, 1)]]
            F += [[idx(a, b, 0), idx(a + 1, b + 1, 0), idx(a + 1, b, 0)], [idx(a, b, 0), idx(a, b + 1, 0), idx(a + 1, b + 1, 0)]]
    for a, w in ((0, -1), (nphi, 1)):
        for b in range(nth):
            q = [idx(a, b, 0), idx(a, b + 1, 0), idx(a, b + 1, 1), idx(a, b, 1)]
            F += ([[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if w < 0 else [[q[0], q[2], q[1]], [q[0], q[3], q[2]]])
    for b in (0, nth):
        for a in range(nphi):
            q = [idx(a, b, 0), idx(a + 1, b, 0), idx(a + 1, b, 1), idx(a, b, 1)]
            F += ([[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if b == 0 else [[q[0], q[2], q[1]], [q[0], q[3], q[2]]])
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True); m.merge_vertices(); trimesh.repair.fix_normals(m)
    if m.volume < 0: m.invert()
    return m


def angles(a4, shutting):
    """the five hoop angles (deg) with hoop 4 at a4: opening (hoop 4 coming down) it has picked up every hoop it reached; shutting
    (going up) each hoop follows once its neighbour above is 45 ahead"""
    if shutting:
        return [0.0] + [max(0.0, a4 - GAP * (NH - 1 - k)) for k in range(1, NH - 1)] + [a4]
    return [0.0] + [min(SHUT[k], a4) for k in range(1, NH - 1)] + [a4]


def poses(a4, shutting):
    """part id -> 4x4 (W) for every moving part (drawn shut) with hoop 4 at a4 deg"""
    ph = angles(a4, shutting); out = {}
    for k in range(1, NH):
        out[f"box_dome_hoop_{k}"] = G() @ turn(math.radians(ph[k] - SHUT[k])) @ np.linalg.inv(G())
    for k in range(NH - 1):
        for j in range(M):
            start = ph[k] + (ph[k + 1] - ph[k]) * j / M
            out[f"box_dome_fabric_{k}{j}"] = G() @ turn(math.radians(start - (SHUT[k] + GAP * j / M))) @ np.linalg.inv(G())
    return out


def ring_pts(r, z, n=128):
    return np.array([[r * math.cos(a), YC + r * math.sin(a), z] for a in np.linspace(0, 2 * math.pi, n, endpoint=False)])


def box_edge(a):
    """where the ray from the dome centre at angle a meets the box's top outline"""
    d = np.array([math.cos(a), math.sin(a)]); best = None
    for k, lim in ((0, BOX["x"]), (0, -BOX["x"]), (1, BOX["y0"] - YC), (1, BOX["y1"] - YC)):
        if abs(d[k]) > 1e-9:
            t = lim / d[k]; p = d * t
            if t > 0 and abs(p[0]) <= BOX["x"] + 1e-6 and BOX["y0"] - YC - 1e-6 <= p[1] <= BOX["y1"] - YC + 1e-6:
                best = t if best is None else min(best, t)
    return np.array([0.0, YC]) + d * best


def lid(n=128):
    """the rain lid: the box's top outline and the ring's disk, less the opening within LID[2] of the seat and the sheets, 3 mm"""
    import manifold3d as mf
    prism = lambda poly, z0, z1: mf.Manifold.extrude(mf.CrossSection([poly]), z1 - z0).translate([0, 0, z0])
    disk = [(LID[3] * math.cos(t), YC + LID[3] * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]   # counter-clockwise, to its edge on the ring
    hole = [(LID[2] * math.cos(t), YC + LID[2] * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    outline = [(-BOX["x"], BOX["y0"]), (BOX["x"], BOX["y0"]), (BOX["x"], BOX["y1"]), (-BOX["x"], BOX["y1"])]
    sheets = [mf.Manifold.cube([x1 - x0, y1 - y0, 12.0]).translate([x0, y0, LID[0] - 6.0]) for x0, x1, y0, y1 in
              ((206.0, 209.0, -63.0, 334.0), (-209.0, -206.0, -63.0, 334.0), (-203.0, 203.0, 337.0, 340.0), (-203.0, 203.0, -69.0, -66.0))]
    m = (prism(outline, LID[0], LID[1]) + prism(disk, LID[0], LID[1])) - prism(hole, LID[0] - 1, LID[1] + 1) - mf.Manifold.batch_boolean(sheets, mf.OpType.Add)
    t = m.to_mesh(); return trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=True)


def hubs():
    """both hub brackets (fixed): a block on the ring under the plate stack, plate 0 (hoop 0's, with the slot), the outer cheek
    beyond plate 4 that carries the bolt's end, the motor and the switches; and the bolts"""
    out = []
    for sgn in (1, -1):
        ya, yb = YC + sgn * (y_plate(0) - PLATE_T / 2 - 6.0), YC + sgn * (y_plate(NH - 1) + PLATE_T / 2 + 8.0)
        y0, y1 = min(ya, yb), max(ya, yb)
        block = trimesh.creation.box(extents=[56.0, y1 - y0, ZB - 9.0 - RING[2]]); block.apply_translation([0, (y0 + y1) / 2, (RING[2] + ZB - 9.0) / 2])   # from the ring up through the lid's edge to under the plates
        cheeks = []
        for yy in ((ya, ya + sgn * 6.0), (yb - sgn * 6.0, yb)):                                       # the inner cheek (behind plate 0) and the outer one
            c0, c1 = min(yy), max(yy); c = trimesh.creation.box(extents=[24.0, c1 - c0, ZB + 12.0 - RING[2]]); c.apply_translation([0, (c0 + c1) / 2, (RING[2] + ZB + 12.0) / 2]); cheeks.append(c)
        bolt = trimesh.creation.cylinder(radius=3.0, segment=[[0, ya, ZB], [0, yb, ZB]], sections=16)
        out += [block, *cheeks, bolt, plate(0, 0.0, sgn)]
    return trimesh.util.concatenate(out)


def drive():
    """both motors (JGY-370 with a double output shaft: the worm gearbox on the outer cheek, its inner 6 mm output on the axis keyed
    to plate 4, the motor hanging below), and on each outer shaft a cam between two limit switches (KW11): open at 0, shut at 180"""
    out = []
    for sgn in (1, -1):
        yo = YC + sgn * (y_plate(NH - 1) + PLATE_T / 2 + 8.0)
        g = trimesh.creation.box(extents=[32.0, 24.0, 46.0]); g.apply_translation([0, yo + sgn * 12.0, ZB - 13.0]); out.append(g)
        mot = trimesh.creation.cylinder(radius=12.2, height=31.0, sections=24); mot.apply_translation([0, yo + sgn * 12.0, ZB - 36.0 - 15.5]); out.append(mot)
        cam = trimesh.creation.cylinder(radius=10.0, segment=[[0, yo + sgn * 24.0, ZB], [0, yo + sgn * 28.0, ZB]], sections=24); out.append(cam)   # on the motor's outer shaft
        for sx in (-1, 1):                                                                            # the two switches either side of the cam: open (0 deg) and shut (180 deg)
            s_ = trimesh.creation.box(extents=[6.4, 10.0, 20.0]); s_.apply_translation([sx * 16.0, yo + sgn * 30.0, ZB]); out.append(s_)
    return trimesh.util.concatenate(out)


def base():
    ring = trimesh.creation.annulus(r_min=RING[0], r_max=RING[1], height=RING[3] - RING[2], sections=128)
    ring.apply_translation([0, YC, (RING[2] + RING[3]) / 2])
    spokes = []
    for k in range(SPOKES):
        a = 2 * math.pi * k / SPOKES + math.pi / SPOKES; d = np.array([math.cos(a), math.sin(a)])
        e = box_edge(a); p0 = [*(e - d * 30.0), LID[0] - 4.0]; p1 = [*(np.array([0, YC]) + d * (RING[1] - 6.0)), LID[0] - 4.0]
        spokes.append(trimesh.creation.cylinder(radius=3.0, segment=[p0, p1], sections=12))
    import manifold3d as mf
    L = lid(); man = lambda t: mf.Manifold(mf.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
    h = (man(placed(hubs())) - man(L)).to_mesh()                                    # the hub brackets stand through the lid's edge, notched round it
    hb = trimesh.Trimesh(np.asarray(h.vert_properties)[:, :3], np.asarray(h.tri_verts), process=True)
    return dict(ring=ring, spokes=trimesh.util.concatenate(spokes), hubs=hb, drive=placed(drive()), hoop_0=placed(hoop_rod(0, 0.0)), lid=L)


def moving():
    """every moving part, shut (its rest pose): id -> mesh"""
    out = {f"box_dome_hoop_{k}": placed(hoop(k)) for k in range(1, NH)}
    out.update({f"box_dome_fabric_{k}{j}": placed(strip(k, j)) for k in range(NH - 1) for j in range(M)})
    return out


def check(mov, fixed):
    """every dome part against both drones (tools/seat_check.descent_states): the fixed parts at every state of each; the moving
    parts open (hoop 4 at 0) at every state of each, and at every 15 deg of hoop 4 both ways against the seated Scout v4 (the dome
    moves only with it seated); the least gap to the seated Scout with the dome shut. Crossings as solids (seat frame), or as sampled points for a part whose mesh is not a solid."""
    sys.path.insert(0, os.path.join(ROOT, "tools")); import seat_envelope as se, seat_check as sc
    W2S = np.linalg.inv(np.array(sc.SEAT_TO_W))
    def to_s(m, T=np.eye(4)):
        t = m.copy(); t.apply_transform(W2S @ T); return (sc.man(t), t)
    fx = {k: to_s(m) for k, m in fixed.items()}
    assert all(v[0] is not None for v in fx.values()), f"not a solid: {[k for k, v in fx.items() if v[0] is None]}"
    def at(a4, shutting):
        P = poses(a4, shutting); return {k: to_s(m, P[k]) for k, m in mov.items()}
    def drone(d):
        """each part as a solid, or, where its mesh is not one (the Overseer's cage tubes), as sampled surface points (seat_check's way)"""
        out = []
        for nm, m in se.parts_of(d):
            mm = sc.man(m)
            out.append((nm, mm, None if mm is not None else np.vstack([trimesh.sample.sample_surface(m, 30000)[0], m.vertices])))
        return out
    def cross(f, part, off=(0.0, 0.0, 0.0)):
        (fm, ft), (nm, dm, pts) = f, part
        if dm is not None:
            return sc.volume(fm ^ dm.translate(list(off)))
        q = pts + off
        sel = np.all((q >= ft.bounds[0] - 1) & (q <= ft.bounds[1] + 1), axis=1)
        return float(ft.contains(q[sel]).sum()) if sel.any() else 0.0     # points inside: a crossing
    opened = at(0.0, False)
    assert all(v[0] is not None for v in opened.values()), f"not a solid: {[k for k, v in opened.items() if v[0] is None]}"
    res = {}
    own = []
    for a4 in range(0, 181, 15):                                                   # the mechanism against itself: moving parts vs the fixed ones, every 15 deg both ways
        for shutting in (True, False):
            for k, f in at(float(a4), shutting).items():
                for kk, g in fx.items():
                    v = sc.volume(f[0] ^ g[0])
                    if v > 0.5: own.append(f"{k} x {kk} at hoop 4 {a4} deg {'shutting' if shutting else 'opening'}: {v:.0f} mm3")
    res["mechanism"] = (None, own)
    for d in ("overseer", "scout_v4"):
        parts = drone(d); hits = []
        for lab, ox, oy, h in sc.descent_states(d):
            for k, f in list(fx.items()) + list(opened.items()):
                for part in parts:
                    v = cross(f, part, (ox, oy, h))
                    if v > 0.5: hits.append(f"{k} (open) x {part[0][:28]} at {lab}: {v:.0f} {'mm3' if part[1] is not None else 'points'}")
        gap = None
        if d == "scout_v4":
            for a4 in range(0, 181, 15):
                for shutting in (True, False):
                    for k, f in at(float(a4), shutting).items():
                        for part in parts:
                            v = cross(f, part)
                            if v > 0.5: hits.append(f"{k} at hoop 4 {a4} deg x {part[0][:28]} seated: {v:.0f}")
            shut = at(180.0, True)
            gap = min(f[0].min_gap(dm, 80.0) for f in shut.values() for _, dm, _p in parts if dm is not None)
        res[d] = (gap, hits)
    return res


if __name__ == "__main__":
    out_dir = os.path.join(HERE, "gore_meshes"); os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    mov, fixed = moving(), base()
    for k, m in list(mov.items()) + [(f"box_dome_{k}", m) for k, m in fixed.items()]:
        m.export(os.path.join(out_dir, f"{k.replace('box_dome_', '')}.stl"))
    for f in sorted(os.listdir(out_dir)):                                     # no empty or inside-out part
        t = trimesh.load(os.path.join(out_dir, f), force="mesh")
        assert len(t.faces) > 0 and t.volume > 1.0, f"{f}: empty or inside out ({len(t.faces)} faces, {t.volume:.0f} mm3)"
    for k in range(1, NH - 1):                                                # the linkage's own clearances: plate k's pin rides in plate k-1's slot both ways
        assert y_plate(k) - y_plate(k - 1) - PLATE_T > 1.0, "the plates touch"
    res = check(mov, fixed)
    for d, (gap, hits) in res.items():
        print(f"{d}: " + (f"the shut dome {gap:.1f} clear of it seated; " if gap is not None else "") + f"{len(hits)} crossings (fixed parts at every state, the open dome at every state" + (", the dome moving over it seated)" if d == "scout_v4" else ")") + "".join("\n  " + h for h in hits[:10]))
        assert not hits and (gap is None or gap >= 20.0), f"the dome is in {d}'s way"
    print(f"dome: hoops r {r_of(0):.0f}..{r_of(NH - 1):.0f} on the axis W Z {ZB:.0f} at W Y {YC:.0f}, plates at +-{y_plate(0):.1f}..{y_plate(NH - 1):.1f}, ring r {RING[0]:.0f}..{RING[1]:.0f}")
