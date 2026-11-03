"""The Scout's own bought parts for its page, drawn as they really are (the Overseer's are tools/bought_meshes.py). Written in the
Scout drone frame (x nose, y left, z up; plate underside 18.5, top 22.0; systems/dimensions.md A and C) to parts/scout_<id>/part.stl,
pose identity. Until 3 Nov the Scout assembly placed the Overseer's folders (motor_1..4 = SunnySky X2216 at the Overseer's motors,
prop_1..4 = 10 x 4.5 props, flow_mtf_02p on the Overseer's flow clip), so the Scout page showed the wrong drone; a folder now
belongs to one system (tools/build_site.py refuses a shared one).
  scout_motor_1..4         EMAX ECO II 2004: bell O24.7, 19.2 from the arm top to the prop hub face (listing, E; dimensions.md C
                           prop plane z 41.2), base 4 x M2 on 12 x 12, O3 shaft, T-mount O1.5 locating shaft; on the plate top at
                           (+-54.1, +-54.1), PX4 quad X order (1 front right, 2 rear left, 3 front left, 4 rear right)
  scout_prop_1..4          Gemfan Hurricane 3520: 3 blades r 44.45, hub O11 x 5.5 on the T-mount, chord 10 -> 13 -> 5, pitch 2.0 in;
                           blades along the arm diagonal so a nose-on view sees them edge-on (E look; the swept disc is the keep-out)
  scout_flow_mtf_02p       MTF-02P from the MicoAir drawing (as bought_meshes.flow), face down at the belly mount concept's
                           (0, -40, -30) (dimensions.md System Scout v3, Belly mount row; E until the mount is designed)
Run: python tools/scout_meshes.py
"""
import os
import numpy as np, trimesh, manifold3d as mf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATE_TOP = 22.0
MOTORS_XY = [(54.1, -54.1), (-54.1, 54.1), (54.1, 54.1), (-54.1, -54.1)]      # PX4 quad X: 1 FR, 2 RL, 3 FL, 4 RR (y left)
MOTOR_D, MOTOR_H = 24.7, 19.2                                                 # bell, arm top to prop hub face (listing, E)
PROP_R, PROP_HUB_D, PROP_HUB_H, PITCH_MM = 44.45, 11.0, 5.5, 50.8
FLOW_AT = (0.0, -40.0, -30.0)                                                 # lens, belly mount concept v1 (C on E)


def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cbox(cx, cy, cz, lx, ly, lz):
    return box(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, cz - lz / 2, cz + lz / 2)
def cyl(x, y, z0, z1, r, n=48):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])
def union(ms):
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add)
def rotz(m, deg):
    return m.rotate([0, 0, deg])
def to_trimesh(man):
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
def write(name, t):
    d = os.path.join(ROOT, "parts", name); os.makedirs(d, exist_ok=True); t.export(os.path.join(d, "part.stl"))
    print(f"  {name:24s} centre {np.round(t.bounds.mean(0), 1).tolist()} extents {np.round(t.extents, 1).tolist()} faces {len(t.faces)} wt {t.is_watertight}")


# ---- motor: EMAX ECO II 2004 on the arm top, the prop hub face MOTOR_H up
def motor(mx, my):
    z0 = PLATE_TOP
    base = cyl(mx, my, z0, z0 + 2.0, 8.0) + union([rotz(cbox(0, 0, 0, 19.0, 5.0, 2.0), a) for a in (45, 135)]).translate([mx, my, z0 + 1.0])
    holes = union([cyl(mx + sx * 6.0, my + sy * 6.0, z0 - 1, z0 + 3, 1.0, 16) for sx in (-1, 1) for sy in (-1, 1)])   # 4 x M2 on 12 x 12
    stator = cyl(mx, my, z0 + 2.0, z0 + 5.0, 6.0)
    bell = cyl(mx, my, z0 + 4.0, z0 + MOTOR_H - 1.5, MOTOR_D / 2, 72)
    windows = union([rotz(box(8.0, 14.0, -1.6, 1.6, 0, 6.0), a) for a in range(0, 360, 40)]).translate([mx, my, z0 + 6.0])   # side windows
    slots = union([rotz(cbox(8.5, 0, 0, 6.0, 2.4, 4.0), a) for a in range(20, 360, 60)]).translate([mx, my, z0 + MOTOR_H - 1.5])   # cap slots
    seat = cyl(mx, my, z0 + MOTOR_H - 1.5, z0 + MOTOR_H, 4.5)                                # the T-mount seat up to the prop hub face
    pin = cyl(mx, my, z0 + MOTOR_H, z0 + MOTOR_H + PROP_HUB_H + 1.0, 0.75, 16)              # O1.5 locating shaft through the prop hub
    return union([base, stator, bell, seat, pin]) - union([holes, windows, slots])


# ---- prop: 3 lofted blades on a hub, blade 1 along +x of the hub
def blade_section(r, m=14):
    t = (r - PROP_HUB_D / 2) / (PROP_R - PROP_HUB_D / 2)
    c = 10.0 + 3.0 * np.sin(np.pi * min(t / 0.4, 1.0) / 2) - 8.0 * max(0.0, (t - 0.4) / 0.6) ** 1.6     # chord 10 -> 13 -> 5
    th = 2.0 - 1.2 * t                                                                                  # thickness 2.0 -> 0.8
    beta = np.arctan(PITCH_MM / (2 * np.pi * r))
    a = np.linspace(0, 2 * np.pi, m, endpoint=False)
    y = 0.5 * c * np.cos(a); z = np.where(np.sin(a) > 0, 0.5 * th * np.sin(a), 0.15 * th * np.sin(a))  # cambered top, flat bottom
    cb, sb = np.cos(beta), np.sin(beta)
    return np.stack([np.full(m, r), y * cb - z * sb, y * sb + z * cb], 1)


def loft(rings):
    """closed solid through a list of rings (each m points, same m): side quads, fan caps"""
    n, m = len(rings), len(rings[0])
    V = np.vstack(rings); F = []
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m; c, d = a + m, b + m
            F += [[a, b, d], [a, d, c]]
    c0, c1 = len(V), len(V) + 1
    V = np.vstack([V, rings[0].mean(0), rings[-1].mean(0)])
    F += [[c0, (j + 1) % m, j] for j in range(m)] + [[c1, (n - 1) * m + j, (n - 1) * m + (j + 1) % m] for j in range(m)]
    t = trimesh.Trimesh(V, np.array(F), process=False); t.fix_normals()
    return t


def prop(mx, my, k):
    rs = np.linspace(PROP_HUB_D / 2 - 0.5, PROP_R, 22)
    one = loft([blade_section(r) for r in rs])
    zc = PLATE_TOP + MOTOR_H + PROP_HUB_H / 2
    parts = []
    for b in range(3):
        bl = one.copy(); bl.apply_transform(trimesh.transformations.rotation_matrix(np.radians(120 * b), [0, 0, 1])); parts.append(bl)
    hub = trimesh.creation.cylinder(radius=PROP_HUB_D / 2, height=PROP_HUB_H, sections=48)
    p = trimesh.util.concatenate(parts + [hub])
    p.apply_transform(trimesh.transformations.rotation_matrix(np.arctan2(my, mx), [0, 0, 1]))     # blade 1 along the arm diagonal
    p.apply_translation([mx, my, zc])
    return p


# ---- flow sensor: board back up, ToF and lens down (the drawing of tools/bought_meshes.flow, moved to the Scout's belly)
def flow(cx, cy, lens_z):
    zt = lens_z + 1.0 + 4.6                   # board back: the lens barrel bottom sits 5.6 under it
    board = cbox(cx, cy, zt - 0.5, 21.6, 16, 1.0)
    tof = cbox(cx + 4.0, cy + 4.3, zt - 1.0 - 2.75, 6.8, 6.0, 5.5)
    tofw = [cyl(x, cy + 4.3, zt - 1.0 - 5.5 - 0.3, zt - 1.0 - 5.5 + 1.5, 1.3, 24) for x in (cx + 2.3, cx + 5.7)]
    lensm = cbox(cx + 4.0, cy - 2.5, zt - 1.0 - 1.75, 7.0, 7.0, 3.5); barrel = cyl(cx + 4.0, cy - 2.5, zt - 1.0 - 4.5, zt - 1.0 - 3.5, 2.5, 32)
    sock = cbox(cx - 10.8 + 2.0, cy, zt - 1.0 - 1.6, 6.0, 6.0, 3.2)
    chips = [cbox(cx - 3, cy, zt - 1.0 - 0.6, 4, 4, 1.2), cbox(cx + 6, cy - 6, zt - 1.0 - 0.4, 2, 3, 0.8)]
    return union([board, tof, lensm, barrel, sock] + chips) - union(tofw + [cyl(cx + 4.0, cy - 2.5, zt - 1.0 - 4.6, zt - 1.0 - 4.0, 1.5, 24)])


if __name__ == "__main__":
    for k, (mx, my) in enumerate(MOTORS_XY, 1):
        write(f"scout_motor_{k}", to_trimesh(motor(mx, my)))
        write(f"scout_prop_{k}", prop(mx, my, k))
    write("scout_flow_mtf_02p", to_trimesh(flow(*FLOW_AT)))
