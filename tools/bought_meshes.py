"""Bought parts as they really look, for the page (Spark 30 Oct 1:40 am: "the rear camera doesn't match the angle, make the
electronics accurate too, everything as real as possible"). Drawn with manifold3d from the dimensions in systems/dimensions.md and
the makers' drawings, written in drone coordinates (x fwd, y left, z up, z 0 = bottom plate top) to parts/<id>/part.stl, pose
identity. Placement = the boxes that stood on the page before (their centres), poses of the cameras from systems/overseer.cage.json
cams and parts/overseer_cage_node_21/part.py. Run: python tools/bought_meshes.py
  camera_left / camera_right / camera_rear   DECXIN 2633V2: board 38 x 38 x 1.6, holes on 28 x 28, lens holder 16 x 16 x 7, M12
                                              barrel O12 with the O14 focus ring, glass O8; back: USB-C / FPC plug block and chips
  fc_stack_micoair743_lite_on_55a_4_in_1      MicoAir 55A 4-in-1 44 x 43.5 (FETs, caps, pads) under the MicoAir743-Lite 36 x 36 on
                                              four M3 posts, both at 45 deg, 30.5 holes; USB-C, barometer, gyro, plug rows
  esc_cap                                     the ESC's electrolytic, O10 x 20, leads to the board
  pi_5_active_cooler + pi_5_cooler             Pi 5 board 85 x 56 x 1.6 (holes 58 x 49, USB x2 + Ethernet aft, micro HDMI x2 + USB-C
                                              on the side, GPIO 2 x 20, SoC, RAM, PMIC) and the Active Cooler 63.5 x 42.5 x 13.7 (fins + fan)
  ubec_5_v / buck_12_v / rx_xr2                UBEC in heatshrink with wire tails; buck board with inductor, cap, IC; XR2 board + antenna
  flow_mtf_02p                                MTF-02P from the MicoAir drawing: board, ToF block, lens module, SH1.0 socket (face down)
  motor_1..4                                  SunnySky X2216: base + bell O27.7 with cooling slots, shaft O3, prop adapter
"""
import json, os, numpy as np, trimesh, manifold3d as mf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAGE = json.load(open(os.path.join(ROOT, "systems", "overseer.cage.json")))
PCB = 1.6

def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cbox(cx, cy, cz, lx, ly, lz):
    return box(cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, cz - lz / 2, cz + lz / 2)
def cyl(x, y, z0, z1, r, n=48):
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])
def cylx(x0, x1, y, z, r, n=48):
    return mf.Manifold.cylinder(x1 - x0, r, r, n).rotate([0, 90, 0]).translate([x0, y, z])
def union(ms):
    return mf.Manifold.batch_boolean(ms, mf.OpType.Add)
def rotz(m, deg):
    return m.rotate([0, 0, deg])
def frame(m, o, ex, ey, ez):
    """m drawn in a local frame (x, y, z) -> placed with axes ex, ey, ez at origin o"""
    T = np.array([[ex[0], ey[0], ez[0], o[0]], [ex[1], ey[1], ez[1], o[1]], [ex[2], ey[2], ez[2], o[2]]], float)
    return m.transform(T)
def write(name, man):
    m = man.to_mesh(); t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)   # manifold's own topology
    d = os.path.join(ROOT, "parts", name); os.makedirs(d, exist_ok=True); t.export(os.path.join(d, "part.stl"))
    print(f"  {name:40s} {np.round(t.bounds.mean(0), 1).tolist()} extents {np.round(t.extents, 1).tolist()} faces {len(t.faces)} wt {t.is_watertight}")

# ---- camera: local x right, y up, z forward; lens glass front at z 0, board back at z -12
def camera():
    bb = -12.0
    board = cbox(0, 0, bb + PCB / 2, 38, 38, PCB)
    holes = [cyl(sx * 14, sy * 14, bb - 1, bb + PCB + 1, 1.3) for sx in (-1, 1) for sy in (-1, 1)]
    holder = cbox(0, 0, bb + PCB + 3.5, 16, 16, 7.0)                           # lens holder on the board face (z -10.4 .. -3.4)
    barrel = cyl(0, 0, bb + PCB + 7.0, -1.2, 6.0)                                 # M12 barrel
    ring = cyl(0, 0, -4.0, -1.2, 7.0, 24)                                         # focus ring (knurled: 24 sides)
    glass_cut = cyl(0, 0, -0.5, 0.5, 4.0)
    front = cyl(0, 0, -1.2, 0.0, 6.0)
    plug = cbox(-12, 12, bb - 1.6, 9, 4, 3.2)                                      # USB-C / FPC plug block on the back near a corner
    chips = [cbox(0, -11, bb - 0.8, 7, 7, 1.6), cbox(9, 5, bb - 0.6, 5, 4, 1.2), cbox(-9, 6, bb - 0.6, 4, 4, 1.2)]
    cam = union([board, holder, barrel, ring, front, plug] + chips) - union(holes + [glass_cut])
    return cam
def place_cams():
    cam = camera()
    for nm, c in zip(("camera_left", "camera_right"), CAGE["cams"]):
        write(nm, frame(cam, np.array(c["lens"]), np.array(c["r"]), np.array(c["u"]), np.array(c["f"])))
    # rear camera on node 21 (parts/overseer_cage_node_21/part.py): back, 40 deg down; board back 14 ahead of the node centre
    pitch = np.radians(40.0); F = np.array([-np.cos(pitch), 0.0, -np.sin(pitch)]); UP = np.array([-np.sin(pitch), 0.0, np.cos(pitch)]); RT = np.cross(F, UP)
    c21 = np.array(CAGE["nodes"][21]); lens = c21 + F * 14.0 + F * 12.0
    write("camera_rear", frame(cam, lens, RT, UP, F))

# ---- FC stack at (0, 0), 45 deg: ESC board z 49.55, FC board above on posts
def fc_stack():
    z0 = 49.55
    esc = cbox(0, 0, z0 + PCB / 2, 44, 43.5, PCB)
    fets = [cbox(sx * 14, sy * 14, z0 + PCB + 0.9, 6.5, 5.0, 1.8) for sx in (-1, 1) for sy in (-1, 1)]
    caps = [cyl(sx * 17, 0, z0 + PCB, z0 + PCB + 3.4, 2.5) for sx in (-1, 1)]
    pads = [cbox(0, sy * 19, z0 + PCB + 0.3, 12, 4, 0.6) for sy in (-1, 1)]
    mcu = cbox(0, 6, z0 + PCB + 0.6, 7, 7, 1.2)
    posts = [cyl(sx * 15.25, sy * 15.25, z0 + PCB, z0 + 9.5, 2.75, 6) for sx in (-1, 1) for sy in (-1, 1)]   # M3 standoffs
    z1 = z0 + 9.5
    fc = cbox(0, 0, z1 + PCB / 2, 36, 36, PCB)
    usb = cbox(14, -12, z1 + PCB + 1.6, 9, 7.5, 3.2)
    gyro = cbox(0, 0, z1 + PCB + 0.5, 3, 3, 1.0); baro = cbox(-8, 8, z1 + PCB + 0.6, 3, 3, 1.2); mcu2 = cbox(6, 6, z1 + PCB + 0.6, 8, 8, 1.2)
    plugs = [cbox(-14, 0, z1 + PCB + 2.0, 4, 18, 4.0), cbox(0, 14, z1 + PCB + 2.0, 18, 4, 4.0), cbox(0, -14, z1 + PCB + 2.0, 14, 4, 4.0)]
    holes = [cyl(sx * 15.25, sy * 15.25, z0 - 1, z1 + PCB + 1, 2.0) for sx in (-1, 1) for sy in (-1, 1)]
    stack = union([esc, fc, mcu, usb, gyro, baro, mcu2] + fets + caps + pads + posts + plugs) - union(holes)
    write("fc_stack_micoair743_lite_on_55a_4_in_1", rotz(stack, 45.0))
    write("esc_cap", union([cylx(-62, -42, 0, 51.6, 5.0), cylx(-42, -38, 1.5, 48.5, 0.5, 12), cylx(-42, -38, -1.5, 48.5, 0.5, 12)]))

# ---- Pi 5 and its cooler
def pi5():
    x0, x1, z0 = -53.5, 31.5, 8.0
    board = box(x0, x1, -28, 28, z0, z0 + PCB)
    holes = [cyl(x, y, z0 - 1, z0 + PCB + 1, 1.35) for x in (-30, 28) for y in (-24.5, 24.5)]
    ports = [box(x0 - 2.5, x0 + 14.6, 16.0, 29.0, z0 + PCB, z0 + PCB + 15.6), box(x0 - 2.5, x0 + 14.6, 1.0, 14.0, z0 + PCB, z0 + PCB + 15.6),   # USB stacks
             box(x0 - 2.5, x0 + 19.0, -25.5, -4.0, z0 + PCB, z0 + PCB + 13.5)]                                                         # Ethernet
    side = [cbox(x0 + 12, -28 + 1.5, z0 + PCB + 1.6, 9, 7, 3.2), cbox(x0 + 27, -28 + 1.5, z0 + PCB + 1.6, 7, 7, 3.2), cbox(x0 + 41, -28 + 1.5, z0 + PCB + 1.6, 7, 7, 3.2)]  # USB-C, 2 x micro HDMI
    gpio = box(x0 + 27.5, x0 + 78.5, 28 - 6.5, 28 - 1.5, z0 + PCB, z0 + PCB + 8.5)
    soc = cbox(x0 + 47, 3, z0 + PCB + 0.7, 15, 15, 1.4); ram = cbox(x0 + 47, -14, z0 + PCB + 0.6, 12, 12, 1.2); pmic = cbox(x0 + 70, 10, z0 + PCB + 0.5, 6, 6, 1.0)
    body = union([board, gpio, soc, ram, pmic] + ports + side) - union(holes)
    write("pi_5_active_cooler", body)
    cx, cy, zc = x0 + 47.0, 0.0, z0 + PCB + 0.01
    base = cbox(cx, cy, zc + 1.5, 63.5, 42.5, 3.0)
    fins = [cbox(cx - 31.75 + 2.0 + i * 3.5, cy, zc + 3.0 + 5.35, 1.5, 42.5, 10.7) for i in range(18)]
    fan_house = cbox(cx + 14, cy, zc + 3.0 + 5.35, 32, 32, 10.7); fan_cut = cyl(cx + 14, cy, zc + 3.0, zc + 13.7 + 1, 14.5)
    hub = cyl(cx + 14, cy, zc + 4.0, zc + 12.5, 5.0)
    blades = union([cbox(cx + 14, cy, zc + 8.5, 26, 3, 1.2).rotate([0, 0, a]).translate([0, 0, 0]) for a in range(0, 180, 36)])
    blades = union([rotz(cbox(0, 0, 0, 26, 3, 1.2), a) for a in range(0, 180, 36)]).translate([cx + 14, cy, zc + 8.5])
    cooler = union([base] + [f for f in fins]) - fan_cut
    cooler = union([cooler, fan_house - fan_cut, hub, blades])
    write("pi_5_cooler", cooler)

# ---- small boards
def small():
    ubec = union([cbox(58, -8, 11.0, 44, 18, 8), cbox(58, -8, 11.0, 48, 20, 6)])                                   # heatshrunk brick, rounded by the two boxes
    ubec = union([ubec, cylx(34, 36, -11, 11, 0.9, 12), cylx(34, 36, -5, 11, 0.9, 12), cylx(80, 84, -11, 11, 0.9, 12), cylx(80, 84, -5, 11, 0.9, 12)])
    write("ubec_5_v", ubec)
    z = 6.5
    buck = union([cbox(45, 13, z + PCB / 2, 20, 16, PCB), cyl(49, 13, z + PCB, z + PCB + 5.0, 4.0), cyl(39, 9, z + PCB, z + PCB + 5.5, 2.0), cbox(39, 17, z + PCB + 0.6, 5, 4, 1.2),
                  cbox(36.5, 13, z + PCB + 1.0, 3, 10, 2.0), cbox(53.5, 13, z + PCB + 1.0, 3, 10, 2.0)])
    write("buck_12_v", buck)
    z = 46.6
    rx = union([cbox(-24, 32, z + 0.5, 12, 16, 1.0), cbox(-24, 32, z + 1.0 + 1.0, 5, 5, 2.0), cbox(-24, 26, z + 1.0 + 0.6, 8, 3, 1.2),
                cylx(-25, -23, 40, z + 1.0, 0.5, 12), cyl(-24, 40, z + 1.0, z + 6.0, 0.5, 12)])                   # ceramic antenna stub up
    write("rx_xr2", rx)

# ---- flow sensor, component face down under the plate at (94, 0): nose end at x 104.8, socket edge at the tail
def flow():
    zt = -2.1
    board = cbox(94, 0, zt - 0.5, 21.6, 16, 1.0)
    tof = cbox(98.0, 4.3, zt - 1.0 - 2.75, 6.8, 6.0, 5.5)
    tofw = [cyl(x, 4.3, zt - 1.0 - 5.5 - 0.3, zt - 1.0 - 5.5 + 1.5, 1.3, 24) for x in (96.3, 99.7)]
    lensm = cbox(98.0, -2.5, zt - 1.0 - 1.75, 7.0, 7.0, 3.5); barrel = cyl(98.0, -2.5, zt - 1.0 - 4.5, zt - 1.0 - 3.5, 2.5, 32)
    sock = cbox(94 - 10.8 + 2.0, 0, zt - 1.0 - 1.6, 6.0, 6.0, 3.2)
    chips = [cbox(91, 0, zt - 1.0 - 0.6, 4, 4, 1.2), cbox(100, -6, zt - 1.0 - 0.4, 2, 3, 0.8)]
    write("flow_mtf_02p", union([board, tof, lensm, barrel, sock] + chips) - union(tofw + [cyl(98.0, -2.5, zt - 1.0 - 4.6, zt - 1.0 - 4.0, 1.5, 24)]))

# ---- motors: SunnySky X2216, bell O27.7, 26 tall over all, at motors_xy, base at z 45
def motors():
    for k, (mx, my) in enumerate(CAGE["motors_xy"], 1):
        z0 = 45.0
        base = cyl(mx, my, z0, z0 + 3.0, 9.5) + union([rotz(cbox(0, 0, 0, 26, 6, 3.0), a) for a in (0, 90)]).translate([mx, my, z0 + 1.5])
        stator = cyl(mx, my, z0 + 3.0, z0 + 7.0, 7.0)
        bell = cyl(mx, my, z0 + 6.0, z0 + 23.0, 13.85, 72)
        slots = union([rotz(cbox(10.5, 0, 0, 7, 3.0, 8), a) for a in range(0, 360, 45)]).translate([mx, my, z0 + 23.0])   # cooling slots in the cap
        windows = union([rotz(box(9, 16, -2.0, 2.0, 0, 7.0), a) for a in range(22, 360, 45)]).translate([mx, my, z0 + 9.0])   # side windows
        shaft = cyl(mx, my, z0 + 23.0, z0 + 26.0, 1.5); adapter = cyl(mx, my, z0 + 23.0, z0 + 25.0, 4.0)
        holes = union([cyl(mx + 9.5 * np.cos(a), my + 9.5 * np.sin(a), z0 - 1, z0 + 4, 1.5) for a in np.radians((0, 90, 180, 270))])
        m = union([base, stator, bell, shaft, adapter]) - union([slots, windows, holes, cyl(mx, my, z0 - 1, z0 + 26.5, 0.0001)])
        write(f"motor_{k}", m)

if __name__ == "__main__":
    place_cams(); fc_stack(); pi5(); small(); flow(); motors()
