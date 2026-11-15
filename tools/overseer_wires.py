"""The Overseer's four motor-lead runs, redrawn out through the top cover's side windows (15 Nov).

The wire runs (parts/wires, parts/wires_red) are a schematic from the core v2 layout (26-30 Oct, its script gone): straight
O4 bundles. Each motor run went straight along its arm's diagonal from the stack (r 45, z 49.6) to the arm (r 110, z 50.6),
through the top cover's root end wall, where the real leads cannot go: the cover has no exits at the root ends (Spark 3 Nov),
its cable exits are the 48 x 18 windows on its four side faces (parts/overseer_top_cover/make.py SIDE_EXIT). This script
redraws those four runs only, along the route the leads take: from the stack's joint up to z 53, out through the +-x side
window (y +-12 in the 48 window), round the outside of the arm root's side face (CLEAR off it) and onto the arm past the root
end, to the run's old outer joint. Every other piece of the two meshes is kept as it is.

    python tools/overseer_wires.py          rewrites parts/wires/part.stl and parts/wires_red/part.stl, prints each run's
                                            clearance to the cover (rerunnable: a run already redrawn is found and redrawn)
"""
import os, sys, numpy as np, trimesh, manifold3d as mf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = 2.0                                     # the runs' bundle radius (the old mesh's O4)
Z_IN, Z_OUT, Z_RUN = 49.6, 50.6, 53.0       # the stack joint, the arm joint (old mesh, E), the route through the window
JOINT_IN, JOINT_OUT = 31.82, 77.78          # |x| = |y| of the old joints on each diagonal
WIN_X0, WIN_Y = 40.0, 12.0                  # inside the window's start (r 38) and the lead's offset in the 48 window
CLEAR = 4.5                                 # centreline off the cover's outside (the radius 2.0 + 2.5)
# the arm roots' flat tops (u .. u1, v0 .. v1 in each diagonal's frame), as parts/overseer_top_cover/make.py ROOT_RECT
ROOT_RECT = {45.0: (72.8, -23.6, 22.7), 135.0: (74.1, -23.6, 21.9), 225.0: (74.1, -21.9, 23.6), 315.0: (72.8, -22.7, 23.6)}
RED = {135.0}                               # the -x +y run is the red (+) lead's, the rest black, as the old meshes drew them


def route(th):
    """the run's polyline for the arm on diagonal th (deg)"""
    t = np.radians(th); ud = np.array([np.cos(t), np.sin(t)]); vd = np.array([-ud[1], ud[0]])
    sx, sy = np.sign(ud)
    u1, v0, v1 = ROOT_RECT[th]
    side = v0 if vd @ np.array([sx, 0.0]) < 0 else v1          # the root's side face that looks to the +-x window
    corner = ud * (u1 + CLEAR) + vd * (side + np.sign(side) * CLEAR)
    return [np.array([sx * JOINT_IN, sy * JOINT_IN, Z_IN]), np.array([sx * WIN_X0, sy * WIN_Y, Z_RUN]),
            None,                                                # out through the window: set from the cover below
            np.array([corner[0], corner[1], Z_RUN]), np.array([sx * JOINT_OUT, sy * JOINT_OUT, Z_OUT])]


def tube(pts):
    s = [mf.Manifold.sphere(R, 24).translate(tuple(p)) for p in pts]
    return mf.Manifold.batch_boolean([mf.Manifold.batch_hull([a, b]) for a, b in zip(s[:-1], s[1:])], mf.OpType.Add)


def touches(v, c):
    """the piece reaches the joint at c (a run ends in its joints' spheres)"""
    return bool(np.min(np.linalg.norm(v - c, axis=1)) <= R + 0.3)


def main():
    cover = trimesh.load(os.path.join(ROOT, "parts", "overseer_top_cover", "part.stl"))
    pq = trimesh.proximity.ProximityQuery(cover)
    runs = {}
    for th in ROOT_RECT:
        pts = route(th)
        sx = np.sign(np.cos(np.radians(th))); sy = np.sign(np.sin(np.radians(th)))
        v = cover.vertices                                       # the window cuts the wall to the bottom: go past the lip's outer edge
        x = np.abs(v[(np.abs(v[:, 1] - sy * WIN_Y) < R + CLEAR) & (np.sign(v[:, 0]) == sx), 0]).max() + CLEAR
        pts[2] = np.array([sx * x, sy * WIN_Y, Z_RUN])
        runs[th] = pts
    for name in ("wires", "wires_red"):
        f = os.path.join(ROOT, "parts", name, "part.stl")
        m = trimesh.load(f)
        keep = []
        for p in m.split(only_watertight=False):
            old = any(touches(p.vertices, runs[th][0]) and touches(p.vertices, runs[th][-1]) for th in ROOT_RECT)   # a run, old or redrawn
            joint = p.extents.max() <= 2 * R + 0.3                  # the joints' spheres stay
            if joint or not old:
                keep.append(p)
        new = [tube(runs[th]).to_mesh() for th in ROOT_RECT if (th in RED) == (name == "wires_red")]
        new = [trimesh.Trimesh(np.asarray(n.vert_properties)[:, :3], np.asarray(n.tri_verts)) for n in new]
        out = trimesh.util.concatenate(keep + new)
        out.export(f)
        print(f"{name}: {len(m.split(only_watertight=False))} pieces -> {len(keep)} kept + {len(new)} runs redrawn, {len(out.faces)} faces")
    for th, pts in runs.items():
        P = np.concatenate([a + np.outer(np.linspace(0, 1, 60), b - a) for a, b in zip(pts[:-1], pts[1:])])
        sd = pq.signed_distance(P)                                  # + inside the cover
        print(f"run {th:.0f}: {' -> '.join('(%.1f, %.1f, %.1f)' % tuple(p) for p in pts)}; centreline off the cover >= {-sd.max():.1f} (bundle r {R})")


if __name__ == "__main__":
    main()
