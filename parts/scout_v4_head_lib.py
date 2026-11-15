"""Scout v4 probe head: the interfaces the drawn head frame and cradle share (15 Nov PROTOTYPE, Spark: a tab post snapped off the bench
print, the camera could not slide into the cradle, the two pieces would not go together). Both make.py files read this; the assembly
checks below run in parts/scout_v4_head_frame/make.py on the real meshes.

Why they did not go together (14 Nov setups): the frame's idler boss (y -24..-20 off the axis) and the cradle's idler boss sat in the
same space; the SG90's face was 1.5 off the cradle's wall where its horn needs ~4.2 (hub and arm, E); and nothing left the camera a way
in (the generated room filled the space behind the case).

Frame of reference: the cradle's own (x forward along the lens, y along the tilt axis, z up; the axis at the origin), posed at nod d by
turning it -d about y and moving it to tools/scout_v4_shape.AXIS (d = TILT, 15 deg, is parked; the range is -20..+90).
  y +: the cradle's +y wall (outer face WY) with the horn's arm in a channel CH deep, open at the wall's bottom edge; the horn's arm
       face at WY - CH, the SG90's output face HORN further (YS from the axis). y -: the cradle's idler boss to -Y_BOSS; the frame's
       bracket from -Y_BRK, an M3 insert in it, the M3 x 10 screw's shank turning in the cradle's A1 3.3 bore (coupon A: slides free).
Assembly (each step's path is checked on the meshes):
  1. the horn on the SG90's spline with its own centre screw (on the bench), its arm along the servo's long side, then turned to point up;
  2. the SG90 slid +y into the frame's closed ring from the cradle's room (the ring's hole takes the body, the tabs stop on its face),
     its two mounting self-tappers driven +y through the tabs into the ring's arms (a PH1 driver, shaft >= 75 mm, past the bracket);
  3. the camera slid into the cradle from behind until its front meets the lip, its own two side screws through the slots; the
     two LED bars stuck on the case's top and bottom fronts through their windows;
  4. the cradle lowered straight down between the bracket and the horn (nod 0), the horn's hub, centre screw and arm entering the
     channel from its open bottom edge; two of the horn's small screws through the arm into the slot pilot above the axis;
  5. the M3 x 10 from -y through the bracket's insert into the cradle's bore (the idler)."""
import math, os, sys
import numpy as np, trimesh
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import scout_v4_shape as S                                                                       # noqa: E402
from parts.scout_v4_units_lib import box, U                                                     # noqa: E402

PCTG = 1.23                                       # g/cm3 (topo/material.py)
# the camera and the cradle's sleeve (tools/scout_v4_shape.CAM, the coupon's C: the case 36 square, 12..18 deep)
H = S.CAM["side"] / 2                             # 18
SI = H + S.CAM["fit"]                             # 18.45: inside the sleeve (0.45 a side: printed holes come out small)
SO = SI + S.CAM["wall"]                           # 20.65: the sleeve's top and bottom walls
F0 = S.CAM["case"][1]                             # 19.5: the case's front on the lip
D0, D1 = S.CAM_DEPTH                              # 12, 18
SLV, LIP = S.CAM["sleeve"], S.CAM["lip"]          # 12 long, the lip 3 in
XB = F0 - SLV                                     # 7.5: the sleeve's back
# the side walls and the axis
WY = 21.8                                         # the side walls' outer faces (3.35 thick off SI)
CH, CHW = 1.0, 7.6                                # the horn's channel: 1.0 deep, 7.6 wide (the hub O7 and the arm's root ~6.6, E)
HORN = 4.2                                        # SG90 horn: its arm's outer face to the servo's output face (hub 4.2 over the spline, E)
YS = WY - CH + HORN                               # 25.0: the SG90's output face off the axis (was 23.0 in tools/scout_v4_shape)
Y_BOSS = WY + 1.2                                 # 23.0: the cradle's idler boss face (O10); the camera's side screw heads reach 22.9
Y_BRK = Y_BOSS + 0.5                              # 23.5: the frame's bracket's inner face
INSERT = (4.0, 5.7)                               # M3 x 5.7 heat-set (Ruthex-type, B08BCRZZS3): its hole O4.0, the bracket that thick
BORE = S.pick("A")                                # 3.3: the coupon's A1, slides free on an M3
# the SG90 (typical drawing and the DORHEA listing, E; tools/scout_v4_shape.head_parts), in frame coordinates off the axis
SG = dict(body_x=(-16.4, 6.0), body_y=22.8, half_z=6.25, tab_x=(-21.2, 11.6), tab_y=(6.0, 8.5), holes=27.7, xc=-5.2)
CLR = 0.3                                         # its cut's clearance


def ycyl(x, z, r, y0, y1, n=48):
    import manifold3d as mf
    return mf.Manifold.cylinder(y1 - y0, r, r, n).rotate([-90, 0, 0]).translate([x, y0, z])


def to_tm(m):
    o = m.to_mesh()
    return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)


def man(t):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))


def pose(d_deg):
    """the cradle's frame at nod d (deg) to the drone's"""
    T = trimesh.transformations.rotation_matrix(-math.radians(d_deg), [0, 1, 0]); T[:3, 3] = S.AXIS
    return T


def place(m, d_deg, shift=(0.0, 0.0, 0.0)):
    T = pose(d_deg); T[:3, 3] += np.asarray(shift, float)
    return m.transform(T[:3, :4])


def camera_local(leds=True):
    """the camera in its metal case (the deepest, 18), its lens and the two LED bars on the case's top and bottom fronts"""
    lens = S.CAM["lens"]; leds_on = leds
    lens_m = man(trimesh.creation.cylinder(radius=lens[2], segment=[(lens[0], 0, 0), (lens[1], 0, 0)], sections=32))
    leds = box(F0 - 8.0, F0, -12.5, 12.5, H, H + 1.2) + box(F0 - 8.0, F0, -12.5, 12.5, -H - 1.2, -H)
    body = box(F0 - D1, F0, -H, H, -H, H) + lens_m
    return body + leds if leds_on else body


def horn_local():
    """the SG90's horn and its centre screw, in the cradle's frame with the arm along +z (E: hub O7, arm 6.6 wide at the root, 1.5 thick, to r 16)"""
    a = WY - CH
    return (ycyl(0, 0, 3.5, a, YS - 0.25) + box(-3.3, 3.3, a, a + 1.5, 0, 16.0)                  # the hub on the spline, riding its shoulder 0.25 off the SG90's case (E)
            + ycyl(0, 0, 2.0, a - 1.4, a))                                                    # the centre screw's head proud of the hub


def servo_frame():
    """the SG90's body and tabs in the drone's frame (seated), the horn excluded"""
    ax, ay, az = S.AXIS
    bx, (tx0, tx1), (ty0, ty1), hz = SG["body_x"], SG["tab_x"], SG["tab_y"], SG["half_z"]
    return (box(ax + bx[0], ax + bx[1], ay + YS, ay + YS + SG["body_y"], az - hz, az + hz)
            + box(ax + tx0, ax + tx1, ay + YS + ty0, ay + YS + ty1, az - hz, az + hz))


def hole_x():
    ax = S.AXIS[0]; xc = ax + SG["xc"]
    return [xc - SG["holes"] / 2, xc + SG["holes"] / 2]


def gap(a, b):
    """the overlap volume (mm3) of two manifolds: 0 when they clear"""
    return float((a ^ b).volume())


def sweep_hull(local_tm, d0, d1, step=2.5, grow=0.5):
    """the union of the convex hulls of a local mesh posed over the nod d0..d1, each grown by `grow` along the axes (a check volume)"""
    v = local_tm.vertices
    off = np.array([[grow, 0, 0], [-grow, 0, 0], [0, grow, 0], [0, -grow, 0], [0, 0, grow], [0, 0, -grow]])
    vv = (v[:, None, :] + off[None]).reshape(-1, 3)
    out = None
    for d in np.arange(d0, d1 + 1e-6, step):
        q = trimesh.transform_points(vv, pose(d))
        h = man(trimesh.convex.convex_hull(q))
        out = h if out is None else out + h
    return out
