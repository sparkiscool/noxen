"""Smooth drawn solids for the demo arms (5 Nov, Spark: no topology, as fast as possible, smooth not geometric, lots of support round
the motor plate). A signed-distance field on a 0.6 grid (inside > 0): rounded boxes, cylinders, capsules and a bell (a cone frustum)
under the motor pad, soft-unioned and blurred 1 mm, meshed by marching cubes and turned into a Manifold so the arm lib's exact drawn
pieces (plate bosses, pegs, lugs) can be added and its cuts (bolts, nut pockets, slots, driver sweeps, shaft bore, neighbour rods,
sockets) subtracted. The flat plate faces come from the lib's drawn bosses, so the blur never rounds a contact face."""
import numpy as np, trimesh, manifold3d as mf
from scipy import ndimage
from skimage import measure

PITCH, BLUR, K = 0.6, 1.0, 3.0


def grid(lo, hi, pad=6.0):
    lo, hi = np.asarray(lo, float) - pad, np.asarray(hi, float) + pad
    ax = [np.arange(lo[i], hi[i] + PITCH / 2, PITCH) for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return ax, np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)


def rbox(P, origin, A, N, half, r=4.0):
    """rounded box: centre origin, axes A (along), N (across), z up; half sizes (a, n, z); corner radius r. inside > 0"""
    d = P - np.asarray(origin, np.float32); A = np.asarray(A, np.float32); N = np.asarray(N, np.float32)
    q = np.abs(np.stack([d @ A, d @ N, d[:, 2]], 1)) - (np.asarray(half, np.float32) - r)
    out = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return r - out


def cylinder(P, c, z0, z1, r):
    rho = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    return np.minimum(np.minimum(r - rho, P[:, 2] - z0), z1 - P[:, 2])


def bell(P, c, z0, r0, z1, r1):
    """a cone frustum on axis z at c: radius r0 at z0 (bottom) to r1 at z1 (top)"""
    rho = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]); t = np.clip((P[:, 2] - z0) / (z1 - z0), 0, 1)
    return np.minimum(np.minimum((r0 + (r1 - r0) * t) - rho, P[:, 2] - z0), z1 - P[:, 2])


def capsule(P, a, b, r):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return r - np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def softmax(fields, k=K):
    F = np.stack(fields, 0) / k
    m = F.max(0)
    return (m + np.log(np.exp(F - m).sum(0))) * k


def mesh(ax, F, P=None, z0=None, z1=None):
    """z0 / z1: hard planes the body never crosses (plate and pad faces): applied after the blur so they stay exactly flat"""
    F = F.reshape(len(ax[0]), len(ax[1]), len(ax[2]))
    F = ndimage.gaussian_filter(F, BLUR / PITCH)
    Z = ax[2][None, None, :]
    if z1 is not None:
        F = np.minimum(F, z1 - Z)
    if z0 is not None:
        F = np.minimum(F, Z - z0)
    F = np.pad(F, 1, constant_values=-1.0)
    v, f, _, _ = measure.marching_cubes(F, 0.0, spacing=(PITCH, PITCH, PITCH))
    v += np.array([ax[0][0], ax[1][0], ax[2][0]]) - PITCH
    t = trimesh.Trimesh(v, f, process=True)
    if t.volume < 0:
        t.invert()
    t.merge_vertices(); trimesh.repair.fill_holes(t)
    m = mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    assert m.status() == mf.Error.NoError, m.status()
    return m


def export(m, path):
    g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts); t.merge_vertices()
    parts = t.split(only_watertight=False)
    if len(parts) > 1:
        t = max(parts, key=lambda q: abs(q.volume))
    t.export(path); b = t.bounds
    print(f"{abs(t.volume) / 1000:.1f} cm3 = {abs(t.volume) * 1.27e-3:.0f} g, bodies {len(parts)}, watertight {t.is_watertight}, box {np.round(b[1] - b[0], 1)}")
    return t
