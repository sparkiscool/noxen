"""Scout v4 probe camera cradle (9 Nov, Spark: "generate everything with topology"): the printed cradle that holds the Arducam 12MP
autofocus camera in its metal case (36 x 36, bracket off) and nods on the SG90's horn (+y) and the idler screw (-y).
Laid out in the camera's frame (x forward along the lens, y across, z up; the tilt axis at the origin) and posed parked: turned TILT up
and moved to tools/scout_v4_shape.AXIS (the head's CG over the mast).
14 Nov (Spark's coupon C: the case fits the 18.6 gap, so it is under ~18 deep; no listing gives the depth and Spark is not to measure it):
nothing here fits the depth. The case is held by its back plate: four M2 x 6 from behind into the back plate's holes, through slots
on the 28 pitch of the 32 x 32 board (E) that run +-1.5 along each diagonal (any pitch 25.9..30.1); side pads on its back 10.5 only;
nothing drawn, generated or cut to fit ahead of 12 (the envelope stops there), so a case 12..18 deep fits and its lens is clear.
Drawn (attachments only): the four M2 bosses on the back, two side pads, the horn's pad (O14, its two screws 5 out, the spline's
hub hole), the idler's boss with an M3 x 5.7 insert.
Cuts: the case (drawn at 18), the M2 screws, their heads and the driver behind, the lens's view forward, the plug and its cable out the
back, the LED bars on the case's top and bottom (for any depth) and their light forward, the horn and the spline, the screws.
Support: the horn's pad and the idler's boss (the axis). Loads: the camera (45 g) and LED bars on the bosses and pads, CG at the
deepest case's middle (the longer lever), in 35 g crashes and flight; the head-first crash (systems/scout_v4.head_load.json, ~250 N
down onto the case's top)."""
import math
import numpy as np, trimesh
from topo import Part
from parts.scout_v4_units_lib import S, ASM, box, cyl, U, payload_cases, point_case

import json as _json, os as _os
HEAD_N = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "systems", "scout_v4.head_load.json")))["head_first_N"]   # the frame model's head-first contact force (tools/scout_v4_shape.py)
SPEC = dict(h=0.8, gen_cm3=2.0, max_worst=0.3)   # E: cradle, arms and coupler 1.5 g, at the nodes' PCTG limit
X0 = S.CAM["case"][0]                            # the case's back, ahead of the axis
D0, D1 = S.CAM_DEPTH                             # 12..18: the coupon's C
H = S.CAM["side"] / 2                             # 18
M2 = [(sy * S.CAM["m2"] / 2, sz * S.CAM["m2"] / 2) for sy in (-1, 1) for sz in (-1, 1)]   # the back plate's holes (y, z), 28 pitch E


def ycyl(x, z, r, y0, y1):
    import manifold3d as mf
    return mf.Manifold.cylinder(y1 - y0, r, r, 40).rotate([-90, 0, 0]).translate([x, y0, z])


def pose():
    T = trimesh.transformations.rotation_matrix(-S.TILT, [0, 1, 0]); T[:3, 3] = S.AXIS
    return T


def place(m):
    return m.transform(pose()[:3, :4])


def wpt(p):
    return tuple((pose() @ np.r_[np.asarray(p, float), 1.0])[:3])


def cylx(y, z, r, x0, x1):
    import manifold3d as mf
    return mf.Manifold.cylinder(x1 - x0, r, r, 32).rotate([0, 90, 0]).translate([x0, y, z])


def hull_x(y, z, d, r, ds, x0, x1):
    """a round slot along x at (y, z), running +-ds along the direction d (the M2 pitch 28 -/+ 2 ds / sqrt 2)"""
    return U([cylx(y + d[0] * t, z + d[1] * t, r, x0, x1) for t in np.linspace(-ds, ds, 7)])


def bodies():
    w = H + 2.0; ds = S.CAM["m2_slot"]
    drawn, cuts, env = [], [], []
    case = box(X0 - 0.2, X0 + D1 + 0.2, -H - 0.2, H + 0.2, -H - 0.2, H + 0.2); cuts.append(case)                   # the deepest case
    bosses = [hull_x(y, z, np.array([y, z]) / math.hypot(y, z), 3.4, ds, X0 - 2.8, X0) for y, z in M2]          # the M2 bosses on the case's back, 2.8 thick
    pads = [box(X0 + 0.5, X0 + 10.5, s * H, s * (H + 2.8), -10.0, 10.0) for s in (-1, 1)]                      # the side pads, the case's back 10.5 only
    horn = ycyl(0.0, 0.0, 7.0, w - 1.1, w + 1.5); idler = ycyl(0.0, 0.0, 4.5, -w - 4.0, -w)
    drawn += bosses + pads + [horn, idler]
    for y, z in M2:
        d = np.array([y, z]) / math.hypot(y, z)
        cuts.append(hull_x(y, z, d, 1.2, ds, X0 - 3.0, X0 + 0.1))                                                # M2 x 6 through the boss (O2.4 slot)
        cuts.append(hull_x(y, z, d, 2.3, ds, -40.0, X0 - 2.8))                                                  # its head (O4) and the driver, out the back
    cuts.append(ycyl(0.0, 0.0, 2.6, w - 6.0, w + 3.0))                                     # the spline's hub
    cuts += [ycyl(s * 5.0, 0.0, 0.7, w - 4.0, w + 3.0) for s in (-1, 1)]                  # the horn's two screws
    cuts.append(ycyl(0.0, 0.0, 2.0, -w - 5.0, -w - 4.0 + 5.7))                            # the idler's M3 x 5.7 insert
    cuts.append(box(X0 + D0, 80.0, -H + 3.0, H - 3.0, -H + 3.0, H - 3.0))                 # the lens's view forward, from the shallowest case's front
    cuts.append(box(-40.0, X0, -6.0, 6.0, -6.0, 6.0))                                      # the plug and its cable out the back
    for sz in (-1, 1):                                                                      # the LED bars on the case's top and bottom fronts (any depth 12..18), their light forward
        cuts.append(box(X0 + D0 - 8.5, 80.0, -13.0, 13.0, sz * H, sz * (H + 1.8)))
    cuts.append(ycyl(0.0, 0.0, 8.0, w + 1.5, w + 30.0))                                    # the SG90 beyond the horn
    env.append(box(X0 - 6.0, X0 + D0, -w - 4.0, w + 1.5, -H - 2.5, H + 2.5))             # nothing ahead of the shallowest case's front
    regs = [hull_x(y, z, np.array([y, z]) / math.hypot(y, z), 3.4, ds, X0 - 0.6, X0) - hull_x(y, z, np.array([y, z]) / math.hypot(y, z), 1.2, ds, X0 - 1, X0 + 1) for y, z in M2]
    regs += [box(X0 + 0.5, X0 + 10.5, s * H, s * (H + 0.6), -10.0, 10.0) for s in (-1, 1)]
    pts = [(X0, y, z) for y, z in M2] + [(X0 + 5.5, s * H, 0.0) for s in (-1, 1)]
    sup = U([ycyl(0.0, 0.0, 7.0, w + 1.2, w + 1.5), ycyl(0.0, 0.0, 4.5, -w - 4.0, -w - 3.7)])
    return dict(drawn=place(U(drawn)), cuts=place(U(cuts)), env=place(U(env)), sup=place(sup),
                regs=[place(r) for r in regs], pts=[wpt(q) for q in pts])


def setup(h):
    B = bodies()
    p = Part("scout_v4_cradle", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PCTG")
    p.support("default", B["sup"])
    g = ASM["scout_v4_probe_tilt"]["mass_g"] + ASM["scout_v4_probe_leds"]["mass_g"]                # the camera (45) and the LED bars
    cg = wpt((X0 + D1 / 2, 0.0, 0.0))                                                     # the deepest case's middle: the longer lever
    payload_cases(p, "camera and LED bars", B["pts"], B["regs"], g, cg)
    point_case(p, f"head-first crash: {HEAD_N} N down onto the case's top", B["pts"], B["regs"], (0, 0, -float(HEAD_N)), wpt((X0 + D1 / 2, 0.0, H)))
    return p
