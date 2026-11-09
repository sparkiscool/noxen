"""Scout v4 probe camera cradle (9 Nov, Spark: "generate everything with topology"): the printed cradle that holds the Arducam 12MP
autofocus camera in its metal case (36 x 36, depth 20 E, bracket off) and nods on the SG90's horn (+y) and the idler screw (-y).
Laid out in the camera's frame (x forward along the lens, y across, z up; the tilt axis at the origin) and posed parked: turned TILT up
and moved to tools/scout_v4_shape.AXIS (the head's CG over the mast).
Drawn (attachments only): two side pads against the case's sides, four lips round the case's front corners (it cannot slide forward),
two bars of a back stop (open in the middle for the 4-pin plug), the horn's pad (O14, its two screws 5 out, the spline's hub hole),
the idler's boss with an M3 x 5.7 insert.
Cuts: the case, the lens's view forward, the plug and its cable out the back, the LED bars on the case's top and bottom and their
light forward, the horn and the spline, the screws.
Support: the horn's pad and the idler's boss (the axis). Loads: the camera (45 g) and LED bars on the pads, lips and back stop in 35 g
crashes and flight; the head-first crash (230 N down onto the case's top)."""
import math
import numpy as np, trimesh
from topo import Part
from parts.scout_v4_units_lib import S, ASM, box, cyl, U, payload_cases, point_case

SPEC = dict(h=0.8, gen_cm3=2.0, max_worst=0.3)   # E: cradle, arms and coupler 1.5 g, at the nodes' PCTG limit
X0, X1 = S.CAM["case"]                            # the case's back and front, ahead of the axis (depth 20 E)
H = S.CAM["side"] / 2                             # 18


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


def bodies():
    w = H + 2.0
    drawn, cuts, env = [], [], []
    case = box(X0 - 0.2, X1 + 0.2, -H - 0.2, H + 0.2, -H - 0.2, H + 0.2); cuts.append(case)
    pads = [box(X0 + 1.0, X1 - 1.0, s * H, s * (H + 2.8), -10.0, 10.0) for s in (-1, 1)]          # the side pads against the case's sides
    lips = [box(X1, X1 + 2.5, sy * (H - 3.0), sy * (H + 0.5), sz * (H - 3.0), sz * (H + 0.5)) for sy in (-1, 1) for sz in (-1, 1)]   # the front corner lips
    back = [box(X0 - 2.8, X0, -H, H, sz * 6.0, sz * (H - 1.0)) for sz in (-1, 1)]        # the back stop, open in the middle for the plug
    horn = ycyl(0.0, 0.0, 7.0, w - 1.1, w + 1.5); idler = ycyl(0.0, 0.0, 4.5, -w - 4.0, -w)
    drawn += pads + lips + back + [horn, idler]
    cuts.append(ycyl(0.0, 0.0, 2.6, w - 6.0, w + 3.0))                                     # the spline's hub
    cuts += [ycyl(s * 5.0, 0.0, 0.7, w - 4.0, w + 3.0) for s in (-1, 1)]                  # the horn's two screws
    cuts.append(ycyl(0.0, 0.0, 2.0, -w - 5.0, -w - 4.0 + 5.7))                            # the idler's M3 x 5.7 insert
    cuts.append(box(X1, 80.0, -H + 3.0, H - 3.0, -H + 3.0, H - 3.0))                      # the lens's view forward
    cuts.append(box(-40.0, X0, -6.0, 6.0, -6.0, 6.0))                                      # the plug and its cable out the back
    for sz in (-1, 1):                                                                      # the LED bars on the case's top and bottom fronts, their light forward
        cuts.append(box(X1 - 8.5, 80.0, -13.0, 13.0, sz * H, sz * (H + 1.8)))
    cuts.append(ycyl(0.0, 0.0, 8.0, w + 1.5, w + 30.0))                                    # the SG90 beyond the horn
    env.append(box(X0 - 6.0, X1 + 3.0, -w - 4.0, w + 1.5, -H - 2.5, H + 2.5))
    pad_regs = [box(X0 + 1.0, X1 - 1.0, s * H, s * (H + 0.6), -10.0, 10.0) for s in (-1, 1)]
    lip_reg = U([box(X1, X1 + 0.6, sy * (H - 3.0), sy * H, sz * (H - 3.0), sz * H) for sy in (-1, 1) for sz in (-1, 1)])
    back_reg = U([box(X0 - 0.6, X0, -H, H, sz * 6.0, sz * (H - 1.0)) for sz in (-1, 1)])
    sup = U([ycyl(0.0, 0.0, 7.0, w + 1.2, w + 1.5), ycyl(0.0, 0.0, 4.5, -w - 4.0, -w - 3.7)])
    return dict(drawn=place(U(drawn)), cuts=place(U(cuts)), env=place(U(env)), sup=place(sup),
                regs=[place(pad_regs[0]), place(pad_regs[1]), place(lip_reg), place(back_reg)],
                pts=[wpt(((X0 + X1) / 2, -H, 0)), wpt(((X0 + X1) / 2, H, 0)), wpt((X1, 0, 0)), wpt((X0, 0, 0))])


def setup(h):
    B = bodies()
    p = Part("scout_v4_cradle", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PCTG")
    p.support("default", B["sup"])
    g = ASM["scout_v4_probe_tilt"]["mass_g"] + ASM["scout_v4_probe_leds"]["mass_g"]                # the camera (45) and the LED bars
    cg = wpt(((X0 + X1) / 2, 0.0, 0.0))
    payload_cases(p, "camera and LED bars", B["pts"], B["regs"], g, cg)
    point_case(p, "head-first crash: 230 N down onto the case's top", B["pts"], B["regs"], (0, 0, -230.0), wpt(((X0 + X1) / 2, 0.0, H)))
    return p
