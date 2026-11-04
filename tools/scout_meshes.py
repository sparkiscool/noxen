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
FLOW_AT = (70.0, 0.0, -24.5)                                                  # lens: in the nose unit's runner, looking down through a window in it (4 Nov; beside the receiver there is no room between it and the funnel's y face, and under the plate the receiver clips its view)


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
    holes = union([cyl(mx + 8.49 * c, my + 8.49 * s, z0 - 1, z0 + 3, 1.0, 16) for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))])   # 4 x M2 on 12 x 12, along the arm
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


def frame(m, o, ex, ey, ez):
    """m drawn in a local right-handed frame (x, y, z) -> placed with axes ex, ey, ez at origin o"""
    T = np.array([[ex[0], ey[0], ez[0], o[0]], [ex[1], ey[1], ez[1], o[1]], [ex[2], ey[2], ez[2], o[2]]], float)
    return m.transform(T)
def cylx(x0, x1, y, z, r, n=24):
    return mf.Manifold.cylinder(x1 - x0, r, r, n).rotate([0, 90, 0]).translate([x0, y, z])
PCB = 1.6


# ---- the bought parts of the layout (dimensions.md A, H, M), each at the centre of the box that stood on the page (layout v3 / v12);
#      the Bee35 plate itself is traced from SpeedyBee's render by tools/bee35_plate.py
def fc_aio():
    """MicoAir743 AIO, 36 x 36 x 1.6 as a diamond at (1.9, 0), ON TOP of the plate (Spark 4 Nov: "we can keep the fc on top of the base plate";
    was under it, lowest point z 9.5) on 4 standoffs through the 25.5 holes, component side up, board bottom z 26 (4 over the plate top 22),
    tallest parts (the caps) to z 32.5; USB-C at the tail corner, slightly right; the 4-in-1 FETs and caps on the same board.
    The same M3 bolts go on through the plate into the nose unit's inserts (parts/scout_units_lib.BOLT_NOSE)."""
    cx, zb = 1.9, 26.0                                                  # board bottom
    zt = zb + PCB                                                       # board top: components above it
    board = cbox(0, 0, zb + PCB / 2, 36, 36, PCB)
    posts = [cyl(sx * 12.75, sy * 12.75, PLATE_TOP, zb, 2.5, 6) for sx in (-1, 1) for sy in (-1, 1)]
    holes = [cyl(sx * 12.75, sy * 12.75, PLATE_TOP - 1, zb + PCB + 1, 1.5) for sx in (-1, 1) for sy in (-1, 1)]
    fets = [cbox(sx * 11, sy * 11, zt + 0.9, 6.0, 5.0, 1.8) for sx in (-1, 1) for sy in (-1, 1)]
    caps = [cyl(0, sy * 13, zt, zt + 4.9, 3.2) for sy in (-1, 1)]
    mcu = cbox(0, 0, zt + 0.6, 8, 8, 1.2); gyro = cbox(-8, 0, zt + 0.5, 3, 3, 1.0); baro = cbox(8, 4, zt + 0.6, 3, 3, 1.2)
    usb = cbox(-14, -5, zt + 1.6, 7.5, 9.0, 3.2)                        # tail corner (-x), slightly right (-y)
    pads = [cbox(sx * 15, 0, zt + 0.3, 4, 10, 0.6) for sx in (-1, 1)]
    m = union([board, mcu, gyro, baro, usb] + posts + fets + caps + pads) - union(holes)
    return m


def radxa():
    """Radxa Zero 3W from the STEP numbers (dimensions.md H): 65 x 30 x 1.6, holes O2.8 at 57.85 x 22.85, two USB-C and a micro HDMI on the
    Y 0 long edge, SD slot under the X 0 end. On edge in the tail bay (layout v12 box x -144.4..-79.3, y +-3.2, z 8..38): long side along x,
    SD end to the tail, ports edge down, components to the right (-y, E)."""
    b = cbox(32.5, 15.0, -PCB / 2, 65, 30, PCB)
    holes = [cyl(x, y, -PCB - 1, 1, 1.4, 16) for x, y in ((3.55, 3.60), (3.60, 26.45), (61.40, 3.60), (61.40, 26.50))]
    usb = [box(37.1, 46.0, -1.25, 6.0, 0, 3.16), box(49.6, 58.5, -1.25, 6.0, 0, 3.16)]
    hdmi = box(9.2, 15.7, -1.5, 5.0, 0, 3.03)
    soc = cbox(30, 14, 0.6, 12, 12, 1.2); ram = cbox(44, 20, 0.5, 8, 8, 1.0); wifi = cbox(62, 15, 0.6, 3, 10, 1.2)
    sd = box(0, 4.9, 7.8, 22.3, -PCB - 2.86, -PCB)
    m = union([b, hdmi, soc, ram, wifi, sd] + usb) - union(holes)
    return frame(m, (-118.0, -15.0, 7.5), (1, 0, 0), (0, 1, 0), (0, 0, 1))   # 3 Nov compact layout: flat under the plate's tail, x -118..-53, z 7.5..15, over the funnel mouth


def ubec():
    """5 V UBEC, a heatshrunk brick (makers 43..50 x 17..25 x 7..10, Q4: 48 x 17 x 8 E) with leads out both short ends, in the layout's
    box x -178.6..-117.6, y 18.2..33.6, z 16..24 (tail bay, behind the rear disks)."""
    cx, cy, cz = -90.5, 26.0, 11.5                                                   # 3 Nov compact layout: beside the Radxa under the plate's tail; 6.5 aft 4 Nov (its lead clearance cut the tail unit's yoke)
    brick = union([cbox(cx, cy, cz, 48, 15, 8), cbox(cx, cy, cz, 50, 17, 6)])
    leads = [cylx(cx - 30.5, cx - 24, cy + dy, cz, 0.9) for dy in (-3, 3)] + [cylx(cx + 24, cx + 30.5, cy + dy, cz, 0.9) for dy in (-3, 3)]
    return union([brick] + leads)


def xr2():
    """RadioMaster XR2 nano: 16 x 12 x 6 board with the antenna up (dimensions.md M), in the layout's box x -110..-94, y -27..-15, z 23..45."""
    cx, cy, z0 = -70.0, -24.0, 8.0                                                    # 3 Nov compact layout: under the plate's tail; antenna laid flat along -x
    board = cbox(cx, cy, z0 + 0.5, 16, 12, 1.0)
    chip = cbox(cx, cy, z0 + 1.0 + 1.0, 5, 5, 2.0); rf = cbox(cx + 5, cy - 3, z0 + 1.0 + 0.8, 3, 3, 1.6); conn = cbox(cx - 6, cy, z0 + 1.0 + 1.5, 3, 8, 3.0)
    ant = union([cylx(cx - 26, cx - 6, cy + 4, z0 + 2.0, 0.5, 12), cbox(cx - 26, cy + 4, z0 + 2.0, 1.0, 6.0, 1.0)])      # T antenna, flat along -x (3 Nov)
    return union([board, chip, rf, conn, ant])


def pack():
    """4S 1500 pack 70 x 33 x 32 (dimensions.md A: 70 x 32 x 33, 165 g), heatshrunk edges, in the layout's box x -42.5..27.5, z -32.5..-0.5;
    leads and the balance lead out the nose end toward the XT60 plug at x 36.7 (parts/pack_xt60_plug)."""
    cx, cy, cz = -7.5, 0.0, -16.5
    body = union([cbox(cx, cy, cz, 70, 30, 32), cbox(cx, cy, cz, 70, 33, 29), cbox(cx, cy, cz, 68, 33, 32)])
    leads = [cylx(27.5, 36.7, sy * 3.6, -13.0, 1.75) for sy in (-1, 1)]
    bal = cbox(31.0, -10.0, -8.0, 7.0, 6.0, 1.2)
    return union([body, bal] + leads)


def camera():
    """DECXIN 2633V2 (as tools/bought_meshes.camera): board 38 x 38 x 1.6, holes on 28 x 28, lens holder 16 x 16 x 7, M12 barrel with the
    focus ring, glass O8; plug block and chips on the back. Local: x right, y up, z forward, glass front at z 0, board back at z -12."""
    bb = -12.0
    board = cbox(0, 0, bb + PCB / 2, 38, 38, PCB)
    holes = [cyl(sx * 14, sy * 14, bb - 1, bb + PCB + 1, 1.1) for sx in (-1, 1) for sy in (-1, 1)]        # M2 (coupon A, Spark 29 Oct)
    holder = cbox(0, 0, bb + PCB + 3.5, 16, 16, 7.0)
    barrel = cyl(0, 0, bb + PCB + 7.0, -1.2, 6.0)
    ring = cyl(0, 0, -4.0, -1.2, 7.0, 24)
    glass_cut = cyl(0, 0, -0.5, 0.5, 4.0)
    front = cyl(0, 0, -1.2, 0.0, 6.0)
    plug = cbox(-12, 12, bb - 1.6, 9, 4, 3.2)
    chips = [cbox(0, -11, bb - 0.8, 7, 7, 1.6), cbox(9, 5, bb - 0.6, 5, 4, 1.2), cbox(-9, 6, bb - 0.6, 4, 4, 1.2)]
    return union([board, holder, barrel, ring, front, plug] + chips) - union(holes + [glass_cut])


CAMERA_LENS = (126.0, 0.0, 40.0)                                                # 4 Nov: 4 forward so the nose unit's M2 posts (10 behind the board back, |y| 14) clear the front discs' 12 mm keep-out
# was (122.0, 0.0, 40.0)                                                # 3 Nov compact layout: just ahead of the front discs (board x 105..120), lens low                                     # one camera at the nose, horizon, looking +x (dimensions.md System Scout v3)


if __name__ == "__main__":
    for k, (mx, my) in enumerate(MOTORS_XY, 1):
        write(f"scout_motor_{k}", to_trimesh(motor(mx, my)))
        write(f"scout_prop_{k}", prop(mx, my, k))
    write("scout_flow_mtf_02p", to_trimesh(flow(*FLOW_AT)))
    write("fc", to_trimesh(fc_aio()))                   # the plate is traced from SpeedyBee's render: tools/bee35_plate.py
    write("radxa_zero_3w", to_trimesh(radxa()))
    write("ubec", to_trimesh(ubec()))
    write("xr2", to_trimesh(xr2()))
    write("pack_4s_1500", to_trimesh(pack()))
    write("camera", to_trimesh(frame(camera(), CAMERA_LENS, (0, 1, 0), (0, 0, 1), (1, 0, 0))))
