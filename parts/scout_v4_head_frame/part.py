"""Scout v4 probe head frame (9 Nov, Spark: "generate everything with topology"): the printed frame on the two tape blades' tip that
carries the SG90 and the tilt cradle's idler (tools/scout_v4_shape.head_parts: the head's CG over the mast, the tilt axis 37 over the throat).
Drawn (attachments only): the clip, a 14 mm sleeve round the lens at the blades' tips (blade A held by an M2 through the +x wall at
y +5 at its top, blade B's M2 and nut ride an 11 mm slot in the -x wall at y -5: on one spool B pays out 7.5 mm less than A over the
150, the ribbon being between them, tools/probe_mast_sim.py; the ribbon's 4 mm loop under the sleeve's top); the SG90's two tab pads (M2 into its 32.8 tabs, holes 27.7 apart: the typical
drawing, E); the idler's boss (an M3 screw along the tilt axis into the cradle's insert).
Cuts: the SG90 and its spline; the cradle, camera and LED bars swept over the nod (-20..+90 deg) with 1 mm to spare; the blades'
lens and the mast below; the screws and their drivers.
Support: the clip (the blades hold the head). Loads: the cradle, camera and LEDs (48.5 g) through the idler and the SG90 in 35 g crashes
and flight; the SG90 (9 g); the head-first crash (systems/scout_v4.head_load.json, ~250 N down onto the camera's top); the SG90's stall torque (1.8 kg cm) on its tabs."""
import math
import numpy as np, trimesh
from topo import Part
from topo.part import man
from parts.scout_v4_units_lib import S, ASM, box, cyl, U, payload_cases, point_case

import json as _json, os as _os
HEAD_N = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "systems", "scout_v4.head_load.json")))["head_first_N"]   # the frame model's head-first contact force (tools/scout_v4_shape.py)
SPEC = dict(h=1.0, gen_cm3=3.0, max_worst=0.3)   # E: clip, plate and brackets 2.5 g as estimated, at the nodes' PCTG limit
HOLES = 27.7                                      # SG90 tab holes apart (typical drawing, E; the DORHEA listing gives the tabs' 32.8 span)


def ycyl(x, z, r, y0, y1):
    import manifold3d as mf
    return mf.Manifold.cylinder(y1 - y0, r, r, 40).rotate([-90, 0, 0]).translate([x, y0, z])


def xcyl(y, z, r, x0, x1):
    import manifold3d as mf
    return mf.Manifold.cylinder(x1 - x0, r, r, 32).rotate([0, 90, 0]).translate([x0, y, z])


def sweep():
    """the cradle, camera and LED bars over the nod, a hull per 5 deg step, unioned (the horn and the idler pin, on the axis, left out)"""
    _, tilt, led = S.head_parts()
    v = np.vstack([tilt.vertices, led.vertices]); w = S.CAM["side"] / 2 + 2.0
    v = v[np.abs(v[:, 1] - S.HEAD["ay"]) <= w + 0.01] - S.AXIS
    hulls = []
    for d in np.arange(S.HEAD["range"][0], S.HEAD["range"][1] + 0.1, 5.0):
        R = trimesh.transformations.rotation_matrix(-(math.radians(d) - S.TILT), [0, 1, 0])[:3, :3]
        q = v @ R.T + S.AXIS
        hulls.append(man(trimesh.convex.convex_hull(q)))
    import manifold3d as mf
    return mf.Manifold.batch_boolean(hulls, mf.OpType.Add).offset(1.0) if hasattr(mf.Manifold, "offset") else mf.Manifold.batch_boolean(hulls, mf.OpType.Add)


def bodies():
    hz, ax, az = S.THROAT["z1"], S.AXIS[0], S.AXIS[2]; ay = S.HEAD["ay"]; mx = S.MAST_X
    h = S.CAM["side"] / 2; w = h + 2.0; sv = w + 3.0; hw = S.BLADE_W / 2
    drawn, cuts = [], []
    # the clip: round the blades' tip, the lens slot through it
    top = hz + S.HEAD["clip"]
    clip = box(mx - 6.5, mx + 6.5, -hw - 2.5, hw + 2.5, hz, top + 2.0); drawn.append(clip)
    lens = box(mx - 4.6, mx + 4.6, -hw - 0.3, hw + 0.3, hz - 1, top - 0.5); cuts.append(lens)
    cuts.append(xcyl(5.0, top - 3.0, 0.8, mx, mx + 20))                                  # blade A's M2 (self-tapping) through the +x wall at y +5
    cuts.append(xcyl(5.0, top - 3.0, 2.0, mx + 6.5, mx + 30))                            #   its head and driver
    cuts.append(box(mx - 30, mx - 1.0, -5.0 - 1.15, -5.0 + 1.15, hz + 1.0, hz + 12.0))   # blade B's M2 in an 11 mm slot of the -x wall (8.8 of travel)
    cuts.append(box(mx - 1.0, mx + 1.0, -2.0, 2.0, top - 1.0, top + 3.0))                # the ribbon up out of the sleeve
    clip_reg = box(mx - 5.2, mx + 5.2, -hw - 0.9, hw + 0.9, hz, top - 0.5) - lens
    # the SG90: its tab pads (+y) and its body / spline as cuts
    xc = ax + (-16.4 + 6.0) / 2; pads, pad_pts, pad_regs = [], [], []
    for s in (-1, 1):
        x = xc + s * HOLES / 2
        pads.append(box(x - 3.5, x + 3.5, ay + sv + 8.6, ay + sv + 12.0, az - 5.5, az + 5.5))
        cuts.append(ycyl(x, az, 0.8, ay + sv + 5.0, ay + sv + 13.0))                    # M2 self-tapping into the pad
        pad_pts.append((x, ay + sv + 10.3, az)); pad_regs.append(box(x - 3.5, x + 3.5, ay + sv + 8.3, ay + sv + 9.2, az - 5.5, az + 5.5))
    drawn += pads
    cuts.append(box(ax - 16.7, ax + 6.3, ay + sv - 0.3, ay + sv + 23.1, az - 6.55, az + 6.55))   # the SG90's body
    cuts.append(box(ax - 21.5, ax + 11.9, ay + sv + 5.7, ay + sv + 8.6, az - 6.55, az + 6.55))    # its tabs
    cuts.append(ycyl(ax, az, 7.6, ay + w, ay + sv + 0.1))                                     # the horn and the spline
    # the idler's boss (-y): an M3 screw along the axis into the cradle
    drawn.append(ycyl(ax, az, 4.5, ay - w - 4.0, ay - w - 1.5))
    cuts.append(ycyl(ax, az, 1.7, ay - w - 10.0, ay - w + 0.5)); cuts.append(ycyl(ax, az, 3.2, ay - w - 30.0, ay - w - 4.0))   # the screw; its head and driver from -y
    idler_reg = ycyl(ax, az, 4.5, ay - w - 1.8, ay - w - 1.5)
    cuts.append(sweep())
    cuts.append(box(mx - 4.6, mx + 4.6, -hw - 0.3, hw + 0.3, -50, hz))                     # the mast below
    env = box(min(ax - 22.0, mx - 7.0), max(ax + 12.5, mx + 7.0), ay - w - 10.0, ay + sv + 12.5, hz, az + 7.0)   # -y: 10 of room for the idler's bracket (a 4.4 slab was a neck)
    return dict(drawn=U(drawn), cuts=U(cuts), env=env, clip_reg=clip_reg, pad_pts=pad_pts, pad_regs=pad_regs, idler=((ax, ay - w - 2.75, az), idler_reg))


def setup(h):
    B = bodies()
    p = Part("scout_v4_head_frame", h, B["drawn"], B["cuts"], B["env"], build="+z", material="PCTG")
    p.support("default", B["clip_reg"])
    _, tilt, led = S.head_parts()
    g = ASM["scout_v4_probe_tilt"]["mass_g"] + ASM["scout_v4_probe_leds"]["mass_g"] + ASM["scout_v4_cradle"]["mass_g"]
    cg = (tilt.center_mass * tilt.volume + led.center_mass * led.volume) / (tilt.volume + led.volume) if tilt.is_watertight else tilt.bounds.mean(0)
    pts = [B["idler"][0]] + B["pad_pts"]; regs = [B["idler"][1]] + B["pad_regs"]
    payload_cases(p, "cradle, camera and LEDs", pts, regs, g, cg)
    payload_cases(p, "SG90", B["pad_pts"], B["pad_regs"], 9.0, (S.AXIS[0] - 5.2, S.AXIS[1] + S.CAM["side"] / 2 + 2.0 + 3.0 + 11.4, S.AXIS[2]), flight=False)
    point_case(p, f"head-first crash: {HEAD_N} N down onto the camera's top", pts, regs, (0, 0, -float(HEAD_N)), (S.AXIS[0] + 10.0, S.AXIS[1], float(tilt.bounds[1][2])))
    F = 0.18 / (HOLES / 1000.0)
    for sgn in (1, -1):
        p.load_multi(f"SG90 stall torque {'+' if sgn > 0 else '-'}1.8 kg cm on its tabs", [(B["pad_regs"][0], (0, 0, sgn * F)), (B["pad_regs"][1], (0, 0, -sgn * F))], cls="flight")
    return p
