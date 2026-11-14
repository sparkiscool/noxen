"""Scout v4 probe camera cradle (9 Nov, Spark: "generate everything with topology"): the printed cradle that holds the Arducam 12MP
autofocus camera in its metal case (36 x 36, bracket off) and nods on the SG90's horn (+y) and the idler screw (-y).
Laid out in the camera's frame (x forward along the lens, y across, z up; the tilt axis at the origin) and posed parked: turned TILT up
and moved to tools/scout_v4_shape.AXIS (the head's CG over the mast).
14 Nov (Spark's coupon C: the case fits the 18.6 gap, under ~18 deep; no drawing gives the depth; the case has no holes on its back, one
threaded hole a side for the bracket it came with; design chat): nothing here fits the depth. A square sleeve round the case's four
sides (36 + 0.45 a side inside: printed holes come out small; wall 2.2), 12 long (the shallowest case), open front and back, a 3.0 lip
on its front rim's top and upper sides, outside the lens (cut at the LED bars' window; none along the bottom: the nod's clearance): the case slides in from behind until its front meets the lip, so the
lens sits at the same place for any depth, and a deeper case stands out behind. A slot along each side wall at mid-height (3.4 wide,
up to M3) through which the camera's own side screws clamp it to the sleeve wherever along the depth its holes are, the heads on
the slot's edges.
Drawn (attachments only): the sleeve and its lip, its side walls on back into the horn's pad and the idler's boss (the room beside the
case is under a member), the horn's pad (O14, its two screws 5 out, the spline's hub hole), the idler's boss
with an M3 x 5.7 insert.
Cuts: the deepest case, the side screws' slots and their heads, the lens's view forward, the plug and its cable out the back, the LED
bars' windows in the sleeve's top and bottom fronts and their light forward, the horn and the spline, the screws.
Support: the horn's pad and the idler's boss (the axis); in the head-first crash also the sleeve's bottom front edge on the head
frame's crash stop ("landed"). Loads: the camera (45 g) and LED bars on the lip, the two side screws and
the sleeve's top and bottom, CG at the shallowest case's middle (the longer lever), in 35 g crashes and flight; the head-first crash (systems/scout_v4.head_load.json, ~250 N
down onto the case's top)."""
import math
import numpy as np, trimesh
from topo import Part
from parts.scout_v4_units_lib import S, ASM, box, cyl, U, payload_cases, point_case

import json as _json, os as _os
HEAD_N = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "systems", "scout_v4.head_load.json")))["head_first_N"]   # the frame model's head-first contact force (tools/scout_v4_shape.py)
SPEC = dict(h=0.8, gen_cm3=6.0, max_worst=0.3)   # 14 Nov: the sleeve run at 2.0 failed early (1.85, head-first 280 N): 9 % of a 21 cm3 room, thinned 2.41 (audit budget check now); 6.0 thins it to ~0.8
F0 = S.CAM["case"][1]                            # the case's front on the lip, ahead of the axis (fixed: the lens never moves)
D0, D1 = S.CAM_DEPTH                             # 12..18: the coupon's C
H = S.CAM["side"] / 2                             # 18
SI = H + S.CAM["fit"]; SO = SI + S.CAM["wall"]    # the sleeve inside 18.45, outside 20.65
L, LIP = S.CAM["sleeve"], S.CAM["lip"]            # 12 long; the lip 3 in from the inside


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
    xb = F0 - L                                                                             # the sleeve's back
    cuts.append(box(F0 - D1 - 0.2, F0, -SI, SI, -SI, SI))                                   # the deepest case (the shallowest inside the sleeve)
    sleeve = (box(xb, F0 + 2.0, -SO, SO, -SO, SO) - box(xb - 1, F0 + 0.01, -SI, SI, -SI, SI) - box(F0 - 1, F0 + 3, -SI + LIP, SI - LIP, -SI + LIP, SI - LIP)
              - box(F0, F0 + 3, -SO - 1, SO + 1, -SO - 1, -SI + LIP))                     # no lip along the bottom: its corner swung within 0.1 of the base plate at -20 deg (tilt_sweep)
    horn = ycyl(0.0, 0.0, 7.0, w - 1.1, w + 1.5); idler = ycyl(0.0, 0.0, 4.5, -w - 4.0, -w)
    arms = [box(-2.0, xb + 0.5, *((SI, SO) if sy > 0 else (-SO, -SI)), -10.0, 10.0) for sy in (-1, 1)]   # 14 Nov run of 8c421a5 (early fail 1.55 at the sleeve's back): the side walls drawn
    # back into the horn's pad and the idler's boss: beside the case the room is 3 wide (y 18.45..21.5, the SG90's face at 23), under the 6.25 member, so nothing was generated there
    drawn += [sleeve, horn, idler] + arms
    for sy in (-1, 1):                                                                      # the camera's own side screws (up to M3) in a slot along each side at mid-height
        cuts.append(box(xb + 1.5, F0 - 1.5, sy * SI - 0.5 if sy > 0 else -SO - 1.0, SO + 1.0 if sy > 0 else -SI + 0.5, -1.7, 1.7))
        cuts.append(box(xb - 1.0, F0 + 0.5, *((SO, SO + 2.25) if sy > 0 else (-SO - 2.25, -SO)), -3.5, 3.5))   # the heads (O7, 2.2 tall) and a driver along the slot: room only to y 22.9 (the SG90's face at 23)
    cuts.append(ycyl(0.0, 0.0, 2.6, w - 6.0, w + 3.0))                                     # the spline's hub
    cuts += [ycyl(s * 5.0, 0.0, 0.7, w - 4.0, w + 3.0) for s in (-1, 1)]                  # the horn's two screws
    cuts.append(ycyl(0.0, 0.0, 2.0, -w - 5.0, -w - 4.0 + 5.7))                            # the idler's M3 x 5.7 insert
    cuts.append(box(F0, 80.0, -SI + LIP, SI - LIP, -SI + LIP, SI - LIP))                  # the lens's view forward, inside the lip
    cuts.append(box(-40.0, F0 - D0, -6.0, 6.0, -6.0, 6.0))                                 # the plug and its cable out the back (any depth)
    for sz in (-1, 1):                                                                      # the LED bars on the case's top and bottom fronts: windows through the sleeve and the lip, their light forward
        cuts.append(box(F0 - 8.6, 80.0, -13.0, 13.0, sz * (H - 0.5) if sz > 0 else -SO - 1, SO + 1 if sz > 0 else -(H - 0.5)))
    cuts.append(ycyl(0.0, 0.0, 8.0, w + 1.5, w + 30.0))                                    # the SG90 beyond the horn
    env.append(box(-6.0, F0, -w - 4.0, w + 1.5, -SO, SO + 3.0)); env.append(box(F0, F0 + 4.0, -w - 4.0, w + 1.5, -SI + LIP, SO + 3.0))                # room over and ahead of the sleeve (its inside is the case), none under it or ahead of its bottom (the nod's clearance over the base plate: tools/scout_v4_shape.head_parts draws this reach); +y stops short of the SG90
    lipreg = box(F0, F0 + 0.6, -SI, SI, -SI + LIP, SI) - box(F0 - 1, F0 + 1, -SI + LIP, SI - LIP, -SI + LIP, SI - LIP)
    regs = [lipreg - box(F0 - 8.6, 80.0, -13.0, 13.0, -40, 40),                             # the case's front on the lip
            box(xb + 1.5, F0 - 1.5, SO - 0.6, SO, -3.5, 3.5) - box(-50, 50, -50, 50, -1.7, 1.7),   # the +y screw's head on its slot's edges
            box(xb + 1.5, F0 - 1.5, -SO, -SO + 0.6, -3.5, 3.5) - box(-50, 50, -50, 50, -1.7, 1.7),
            box(xb, F0, -SI, SI, SI, SI + 0.6) - box(F0 - 8.6, 80.0, -13.0, 13.0, 0.0, 40.0),
            box(xb, F0, -SI, SI, -SI - 0.6, -SI) - box(F0 - 8.6, 80.0, -13.0, 13.0, -40.0, 0.0)]   # the case's top and bottom bearing on the whole sleeve (less the LED windows)
    pts = [(F0, 0.0, 0.0), (F0 - L / 2, SO, 0.0), (F0 - L / 2, -SO, 0.0), (F0 - L / 2, 0.0, SI), (F0 - L / 2, 0.0, -SI)]
    sup = U([ycyl(0.0, 0.0, 7.0, w + 1.2, w + 1.5), ycyl(0.0, 0.0, 4.5, -w - 4.0, -w - 3.7)])
    land = box(F0 - 1.5, F0, -SO, SO, -SO, -SO + 0.6) - box(F0 - 8.6, 80.0, -13.0, 13.0, -40.0, 0.0)   # the sleeve's bottom front edge on the head frame's crash stop (at -20 deg), beside the LED window
    squeeze = box(F0 - 1.5, F0, -SI, SI, -SI - 0.6, -SI) - box(F0 - 8.6, 80.0, -13.0, 13.0, -40.0, 0.0)   # the metal case's bottom front corners on the sleeve, right over it
    return dict(drawn=place(U(drawn)), cuts=place(U(cuts)), env=place(U(env)), sup=place(sup), land=place(U([sup, land])), squeeze=place(squeeze),
                regs=[place(r) for r in regs], pts=[wpt(q) for q in pts])


def setup(h):
    B = bodies()
    p = Part("scout_v4_cradle", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PCTG")
    p.support("default", B["sup"])
    g = ASM["scout_v4_probe_tilt"]["mass_g"] + ASM["scout_v4_probe_leds"]["mass_g"]                # the camera (45) and the LED bars
    cg = wpt((F0 - D0 / 2, 0.0, 0.0))                                                     # the shallowest case's middle: the farthest from the axis
    payload_cases(p, "camera and LED bars", B["pts"], B["regs"], g, cg)
    # 14 Nov (run of 8c421a5: 1.55 with the 280 N through the pivots): as parts/scout_v4_head_frame models it, the SG90 slips at 1.8 kg cm
    # and the cradle nods down until the sleeve's bottom front edge lands on the head frame's crash stop (-20 deg). The metal case is rigid:
    # its bottom front corners squeeze the sleeve's bottom wall onto the stop (world-down in that pose); the pivots carry what the
    # slipping SG90 holds (18 N at 10 mm) until then. (Split rigidly over the lip, screws and faces, the 280 N read 2.7 filled solid.)
    p.support("landed", B["land"])
    R = trimesh.transformations.rotation_matrix(-S.TILT, [0, 1, 0])[:3, :3] @ trimesh.transformations.rotation_matrix(math.radians(-20.0), [0, 1, 0])[:3, :3]
    p.load(f"head-first crash: {HEAD_N} N, the case landed on the crash stop through the sleeve's bottom", B["squeeze"], tuple(R @ np.array([0, 0, -float(HEAD_N)])), cls="crash", support="landed")
    point_case(p, "head-first crash: the pivots until the SG90 slips (18 N)", B["pts"], B["regs"], (0, 0, -18.0), wpt((F0 - D0 / 2, 0.0, H)))
    return p
