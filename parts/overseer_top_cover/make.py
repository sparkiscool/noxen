"""Overseer top cover v2 (Spark 3 Nov): a canopy over the FC for looks, not airtight. Spark's asks:
* lots of room for the cables and around the FC;
* a super smooth, aerodynamic dome, kept low under the rear props; it need not be symmetric;
* the Noxen N mark raised on top (thicker, not holes; noxensys.com/assets/icon.svg);
* thin, for translucent PETG;
* held by the 8 top plate bolts;
* clamps seamlessly to the top plate, the plate sandwiched;
* fully overlaps the rectangular arm roots.
Drawn only, no topology run (a cosmetic shell, no load path): python parts/overseer_top_cover/make.py writes part.stl.

Shape: one continuous height field H(x, y), soft-blended, the 3D field blurred 0.27 mm and meshed by marching cubes (no facets).
* Footprint = the top plate's own outline (sectioned from parts/top_plate/part.stl) + 1.35, plus each arm root's rectangular
  block where it leaves the plate: hull of the plate boss block (u 57.5..72, |v| <= 22) and the tube mouths (29 out along
  the arm axis, |v| 8.6), parts/overseer_tube_arm_lib.py. The rear arms are one-piece and generated; their root is taken as
  the same block on their own axis (CHECK on the printed arm). The lip runs down past the plate edge to z 45 and sits on
  the root tops beyond it, flush with their sides: the plate is sandwiched, no seam.
* From the edge a steep skirt (3:1) blends into a teardrop dome. Its peak is z 86.6 (40 over the plate top) at x +5. It runs
  longer to the rear (95) than the front (75), y 65.
* The dome softens down under the rear prop disks (10 in, r 127). There it stays 6 under the blade band (z 82) at the disk
  edge and slopes 1:2 lower inside it, so the wash runs off it. The floor is 12 over the plate everywhere: the cable shelf
  over the lobes and roots.
* Shell 1.26 normal (3 lines at 0.42). The N mark is raised 2.0 on the dome top, 50 wide, upright seen from above with the
  nose forward.
* The 8 top plate bolts A (arm lib BOLT_TOP): a 1.6 seat ring on the plate top and a tube up to the shell round a O7.2
  driver well. M3 x 50 as now (drilled bottom plate).
* Cable exits:
  - one 15 x 11 at the end of each arm root, along the tubes (the motor leads ride the root tops under the cover);
  - one 48 x 18 on each side face.
* Print (Bambu A1 mini, translucent PETG, dried 65 C 8 h):
  - upright: the lip on the smooth PEI plate with glue stick;
  - variable layer height (adaptive, 0.08..0.2, smooth 5) for a stair-free dome; line width 0.42, 3 wall loops;
  - top / bottom 5 layers, monotonic; infill 100 %;
  - supports: tree (organic), on build plate only, inside the dome only, top Z distance 0.2, interface 2 layers;
  - nozzle 255, bed 70; outer wall 40, inner 60, first layer 20 mm/s; max volumetric 8; fan 20 %; flow 1.03;
  - seam: back, scarf seam on; ironing on the top surfaces only (glossy logo).
Grid: 0.45 mm; decimated to 150k faces."""
import os, sys, json, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage
from skimage import measure
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from parts.overseer_tube_arm_lib import ROOT_S0, TUBE_DEPTH, TUBE_BORE_R, TUBE_WALL

PB, PT, STACK = 45.0, 46.6, 26.0                            # plate underside / top (mesh), stack height (Spark)
T, LIP, GAP = 1.26, 1.2, 0.15
HD, FLOOR, XC, AXF, AXR, AY = 40.0, 12.0, 5.0, 75.0, 95.0, 65.0
PROP_Z, PROP_R, PROP_CLR = 82.0, 127.0, 6.0
LOGO_W, LOGO_C, LOGO_T = 50.0, (12.0, 0.0), 2.0
BOLT_U, BOLT_V, HOLE_R, SEAT_R, DRIVE_R = 63.17, 15.0, 1.7, 5.0, 3.6
ARM_EXIT, SIDE_EXIT = (15.0, 11.0, 8.0), (48.0, 18.0, 38.0)  # width, height over the plate, start (s along the arm / r)
RES, XY = 0.45, 92.0
SKIRT = 3.0                                                 # skirt rise per mm in from the edge (72 deg; prints upright)
LOGO = [[(0, 0), (78, 0), (286, 257), (210, 257), (56, 66.7), (56, 257), (0, 257)],
        [(135, 0), (215, 0), (472, 257), (392, 257)],
        [(392, 0), (472, 0), (365, 107), (325, 67)]]          # noxensys.com/assets/icon.svg (viewBox units, y down)
J = mf.JoinType.Round
cage = json.load(open(os.path.join(ROOT, "systems", "overseer.cage.json")))
MOTORS = np.array(cage["motors_xy"], float)


def ccw(p):
    p = np.asarray(p, float)
    return p if np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) > 0 else p[::-1]


def plate_outline():
    m = trimesh.load(os.path.join(ROOT, "parts", "top_plate", "part.stl"))
    loops = [d[:, :2] for d in m.section(plane_origin=[0, 0, 0.5 * (PB + PT)], plane_normal=[0, 0, 1]).discrete]
    area = lambda q: abs(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    return mf.CrossSection([ccw(max(loops, key=area))]).simplify(0.05)


def arms():
    """per arm: root centre R, plate frame (ud, vd) on the diagonal, arm axis a (root to motor)"""
    out = []
    for th in (45.0, 135.0, 225.0, 315.0):
        t = np.radians(th); ud = np.array([np.cos(t), np.sin(t)]); vd = np.array([-ud[1], ud[0]]); R = 66.5 * ud
        M = [m for m in MOTORS if np.sign(m[0]) == np.sign(ud[0]) and np.sign(m[1]) == np.sign(ud[1])][0]   # this arm's motor
        out.append((R, ud, vd, (M - R) / np.linalg.norm(M - R)))
    return out


def roots():
    cs = []
    sm = ROOT_S0 + TUBE_DEPTH + 2.0; vm = TUBE_BORE_R + TUBE_WALL + 3.0
    for R, ud, vd, a in arms():
        n = np.array([-a[1], a[0]])
        pts = [ud * u + vd * v for u in (57.5, 72.0) for v in (-22.0, 22.0)] + [R + a * sm + n * s * vm for s in (-1, 1)]
        cs.append(mf.CrossSection.hull_points([tuple(p) for p in pts]))
    return mf.CrossSection.batch_boolean(cs, mf.OpType.Add)


def bolts():
    return [ud * BOLT_U + vd * s * BOLT_V for R, ud, vd, a in arms() for s in (1, -1)]


# ---------------- 2D fields on the grid
O = plate_outline(); RT = roots()
Fo = O.offset(GAP + LIP, J) + RT                            # outer footprint (lip outside the plate, flush with the roots)
Fi = O.offset(GAP, J) + RT.offset(-LIP, J)                  # below the plate top: open inside the plate and over the roots
xs = np.arange(-XY, XY + RES / 2, RES); X, Y = np.meshgrid(xs, xs, indexing="ij"); P2 = np.c_[X.ravel(), Y.ravel()]


def mask(cs):
    m = np.zeros(X.shape, bool)
    for poly in cs.to_polygons():
        q = np.asarray(poly)
        inside = Path(q).contains_points(P2).reshape(X.shape)
        sgn = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) > 0
        m = (m | inside) if sgn else (m & ~inside)
    return m


def sdf(cs):
    m = mask(cs)
    return (ndimage.distance_transform_edt(m) - ndimage.distance_transform_edt(~m)) * RES


def softmin(a, b, k):
    return -k * np.logaddexp(-a / k, -b / k)


def softmax(a, b, k):
    return k * np.logaddexp(a / k, b / k)


sd_o, sd_i = sdf(Fo), sdf(Fi)
ax = np.where(X > XC, AXF, AXR)
rho4 = (((X - XC) / ax) ** 2 + (Y / AY) ** 2) ** 2
H_dome = PT + FLOOR + (HD - FLOOR) * np.clip(1.0 - rho4, 0.0, None) ** 1.5
d_disk = np.min([np.hypot(X - mx, Y - my) for mx, my in MOTORS], axis=0) - PROP_R        # + outside every disk
H_prop = PROP_Z - PROP_CLR + np.where(d_disk > 0, d_disk, 0.5 * d_disk)
Hc = softmax(softmin(H_dome, H_prop, 5.0), np.full_like(X, PT + FLOOR), 2.5)
H = softmin(Hc, PT + SKIRT * sd_o, 3.0)
H = ndimage.gaussian_filter(H, 1.0 / RES)                    # 1 mm blur: no creases
H = np.minimum(H, PT + SKIRT * np.maximum(sd_o, 0.0))               # keep the edge true after the blur
gx, gy = np.gradient(H, RES)
Tn = T * np.minimum(np.sqrt(1.0 + gx ** 2 + gy ** 2), 2.5)
s = LOGO_W / 472.0
logo_cs = mf.CrossSection([ccw([(LOGO_C[0] - (v - 128.5) * s, LOGO_C[1] - (u - 236.0) * s) for u, v in p]) for p in LOGO])
sd_logo = sdf(logo_cs)
rb = np.min([np.hypot(X - bx, Y - by) for bx, by in bolts()], axis=0)


def exit_field(Z):
    """> 0 inside a cable exit"""
    E = np.full(X.shape + (Z.shape[-1],), -1e3, np.float32)
    w, h, s0 = ARM_EXIT
    for R, ud, vd, a in arms():
        n = np.array([-a[1], a[0]]); dx, dy = X - R[0], Y - R[1]
        sa, ta = dx * a[0] + dy * a[1], dx * n[0] + dy * n[1]
        E = np.maximum(E, np.minimum(np.minimum(w / 2 - np.abs(ta), sa - s0)[..., None], PT + h - Z))
    w, h, s0 = SIDE_EXIT
    for th in (0.0, 90.0, 180.0, 270.0):
        c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
        sr, tr = X * c + Y * sn, -X * sn + Y * c
        E = np.maximum(E, np.minimum(np.minimum(w / 2 - np.abs(tr), sr - s0)[..., None], PT + h - Z))
    return E


# ---------------- 3D field (> 0 solid), meshed
zs = np.arange(PB - 1.5, H.max() + LOGO_T + 2.0, RES); Z = zs[None, None, :]
Hz, Tz = H[..., None], Tn[..., None]
shell = np.minimum(np.minimum(sd_o[..., None], Hz - Z), Z - PB)
hollow = np.minimum(Hz - Tz - Z, np.maximum(sd_i[..., None], Z - PT))
F = np.minimum(shell, -hollow).astype(np.float32)
F = np.maximum(F, np.minimum(np.minimum(sd_logo[..., None], Hz + LOGO_T - Z), Z - (Hz - Tz)))                # raised logo
F = np.maximum(F, np.minimum(np.minimum(SEAT_R - rb[..., None], Z - PT), PT + 1.6 - Z))                    # bolt seat rings
F = np.maximum(F, np.minimum(np.minimum(np.minimum(DRIVE_R + T - rb[..., None], Hz - Z), Z - PT), sd_o[..., None]))  # tubes
F = np.minimum(F, -np.minimum(DRIVE_R - rb[..., None], Z - (PT + 1.6)))                                      # driver wells
F = np.minimum(F, -(HOLE_R - rb[..., None]) + 0 * Z)                                                          # M3 holes
F = np.minimum(F, -exit_field(Z))
F = np.pad(ndimage.gaussian_filter(F, 0.6), 1, constant_values=-1.0)    # smooth field: smooth surface
v, f, _, _ = measure.marching_cubes(F, 0.0, spacing=(RES, RES, RES))
v += np.array([xs[0], xs[0], zs[0]]) - RES
t = trimesh.Trimesh(v, f, process=True)
if t.volume < 0:
    t.invert()
for n_f in (150000, 250000):                                         # decimate (0.1 mm off the smooth surface) if it stays closed
    d = t.simplify_quadric_decimation(face_count=n_f); d.merge_vertices(); trimesh.repair.fill_holes(d)
    if d.is_watertight and abs(d.volume - t.volume) < 0.01 * abs(t.volume):
        t = d; break
t.export(os.path.join(HERE, "part.stl"))

# ---------------- checks
b = t.bounds
stack = (np.abs(X) + np.abs(Y) <= 25.5)                    # the FC (36 square turned 45) under its 26 top
ceil = (H - Tn)[stack].min() - PT
top_in_disk = (H + LOGO_T * (sd_logo > 0))[(d_disk < 0) & (sd_o > 0)].max()
hits = []
P = [np.array(n, float) for n in cage["nodes"]]
segs = [(f"rod {a}-{b_}", P[a], P[b_]) for a, b_ in cage["rods"]] + [(k, np.array(q[0], float), np.array(q[1], float)) for k, q in cage["links"].items()]
for nm, A, B in segs:
    for u in np.linspace(0, 1, 400):
        q = A + (B - A) * u; i, j = int(round((q[0] + XY) / RES)), int(round((q[1] + XY) / RES))
        if 0 <= i < len(xs) and 0 <= j < len(xs) and sd_o[i, j] > -4.0 and PB - 4.0 < q[2] < H[i, j] + 4.0:
            hits.append(nm); break
print(f"{abs(t.volume) / 1000:.1f} cm3 = {abs(t.volume) * 1.27e-3:.1f} g PETG, {len(t.faces)} faces, watertight {t.is_watertight}")
print(f"box x {b[0][0]:.1f}..{b[1][0]:.1f} y {b[0][1]:.1f}..{b[1][1]:.1f} z {b[0][2]:.1f}..{b[1][2]:.1f}; peak {H.max():.1f}")
print(f"ceiling over the stack >= {ceil:.1f} over the plate (stack {STACK}); top under the prop disks <= {top_in_disk:.1f} (blades {PROP_Z}); cage members near: {hits or 'none'}")
