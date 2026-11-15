"""Scout v4 probe head frame, PROTOTYPE (15 Nov, drawn; was generated: on Spark's bench print an SG90 tab post, a lone 7 x 3.4 pad on
one neck, snapped off, and the frame would not go together with the cradle). The printed frame on the two tape blades' tip: it carries
the SG90 and the cradle's idler. Interfaces, assembly order and the checks: parts/scout_v4_head_lib.py.
Drawn: python parts/scout_v4_head_frame/make.py writes part.stl (the drone's frame), print/scout_v4_head_frame.stl, preview.png and the
SG90's page mesh (parts/scout_v4_probe_servo/part.stl, at its new seat), and runs the assembly checks.
  Clip (unchanged, 12 Nov): the 14 mm sleeve round the lens at the blades' tips, blade A's M2 through the +x wall, blade B's M2 in its
    12 mm slot of the -x wall, the ribbon up through the plate.
  Plate 4 thick (z 86..90, its top where the 14 Nov setup had it: the camera's lowest corner clears it by 2.6 at -20 deg and lands on it
    in the head-first crash), from the idler's gusset to the SG90's ring.
  SG90 ring: a closed frame round the servo's body at its tabs, 4 deep at the tab face with its sides (the tab screws' columns) 11.5
    deep along the body, from the plate to 3.5 over the body. The SG90 (horn on) slides +y into it from the cradle's room; its two
    mounting self-tappers (from its bag, M2 x 8, E) go +y through the tabs into round O1.7 pilots (print small: ~1.5) 11.5 deep.
  Idler bracket: a 12 x 5.7 plate from the plate to 7 over the axis, an M3 x 5.7 heat-set insert in an O4.0 hole on the axis, a gusset
    behind it to the plate; its inner face 0.5 off the cradle's boss.
Hand calc (PCTG E: 45 MPa in-layer, ~20 across the layers, topo/material.py), each tab column as a lone cantilever (the ring ties them):
  - a hand pushing the SG90's end 20 N (E) along y at the tab, 28 over the plate: M = 560 N mm on a 6.35 x 11.5 column across the
    layers: 6 M / (b d^2) = 4.0 MPa. Margin 5.
  - driving a tab screw home: 0.3 N m (an M2 strips PCTG near there, E) about y: 6 T / (d b^2) = 6 x 300 / (11.5 x 6.35^2) = 3.9 MPa.
    Margin 5. The 20 N driver push as above. The ring's top bar ties the columns: both halve in fact.
  - stall 1.8 kg cm (TowerPro, E) across the tabs: 6.4 N each way in z, in the columns' layers: < 0.2 MPa.
  - 35 g crash of the head (SG90 9 g, cradle + camera + LEDs 58.5 g): the ring's half 11 N at 30 over the clip's edge in the 4 mm
    plate, 36 wide: 6 x 330 / (36 x 16) = 3.4 MPa in-layer, margin 13; the bracket's half 8.3 N at 28: 232 N mm on 12 x 5.7 across
    the layers: 3.6 MPa, margin 5 before its gusset.
  - head-first crash 260 N (systems/scout_v4.head_load.json): the camera's lower front edge lands on the plate over the clip, straight
    down into it: 260 / (36 x 2) = 3.6 MPa compression. The pivots carry the 18 N the slipping SG90 holds.
Print: clip down (as the bench print), tree supports under the plate where it overhangs the clip; 0.2 layers, 4 walls, 40 % infill
(E ~2 h 15)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "parts"))
import numpy as np, trimesh                                                                     # noqa: E402
from parts.scout_v4_head_lib import (S, box, U, ycyl, to_tm, man, place, gap, sweep_hull, camera_local, horn_local, servo_frame,   # noqa: E402
                                     hole_x, YS, Y_BRK, INSERT, SG, CLR, PCTG)


def xcyl(y, z, r, x0, x1):
    import manifold3d as mf
    return mf.Manifold.cylinder(x1 - x0, r, r, 32).rotate([0, 90, 0]).translate([x0, y, z])


def frame():
    hz = S.THROAT["z1"]; ax, ay, az = S.AXIS; mx = S.MAST_X; hw = S.BLADE_W / 2
    top = hz + S.HEAD["clip"]; pz = top + S.HEAD["plate"]                                       # 87, 90
    hx = hole_x(); bx, hzs = SG["body_x"], SG["half_z"]
    drawn, cuts = [], []
    # the clip (12 Nov, as parts/scout_v4_head_frame's setup drew it)
    cw = S.LENS_HALF + 0.4 + 2.2
    drawn.append(box(mx - cw, mx + cw, -hw - 2.5, hw + 2.5, hz, top + 2.0))
    cuts += [box(mx - S.LENS_HALF - 0.4, mx + S.LENS_HALF + 0.4, -hw - 0.3, hw + 0.3, hz - 1, top - 0.5),   # the lens
             xcyl(5.0, top - 3.0, 0.8, mx, mx + 20), xcyl(5.0, top - 3.0, 2.0, mx + cw, mx + 30),          # blade A's M2, its head and driver
             box(mx - 30, mx - 1.0, -5.0 - 1.15, -5.0 + 1.15, hz + 2.0, top),                             # blade B's M2 slot
             box(mx - S.CABLE_T / 2 - 0.6, mx + S.CABLE_T / 2 + 0.6, -S.CABLE_W / 2 - 0.3, S.CABLE_W / 2 + 0.3, top - 1.0, pz + 1.0),   # the ribbon up through the plate
             box(mx - 4.6, mx + 4.6, -hw - 0.3, hw + 0.3, -50, hz)]                                        # the mast below
    # the plate
    y0, y1 = ay - Y_BRK - INSERT[1] - 10.0, ay + YS + 20.0
    xl, xr = hx[0] - 4.0, max(hx[1] + 4.0, mx + cw)
    drawn.append(box(xl, xr, y0, y1, pz - 4.0, pz))
    # the SG90's ring
    ry0, ry1, rz1 = ay + YS + SG["tab_y"][1], ay + YS + 20.0, az + hzs + CLR + 3.5
    drawn.append(box(xl, xr, ry0, ry0 + 4.0, pz - 1.0, rz1))                                    # the ring's face (the tabs on it)
    drawn += [box(x - 4.0, x + 4.0, ry0, ry1, pz - 1.0, rz1) for x in hx]                       # the tab screws' columns, deep along the body
    drawn.append(box(hx[0] - 4.0, hx[1] + 4.0, ry0, ry1, az + hzs + CLR, rz1))                  # its top bar
    body = box(ax + bx[0] - CLR, ax + bx[1] + CLR, ay + YS - 1.0, ay + YS + SG["body_y"] + 30.0, az - hzs - CLR, az + hzs + CLR)
    tabs = box(ax + SG["tab_x"][0] - CLR, ax + SG["tab_x"][1] + CLR, ay + YS - 20.0, ry0, az - hzs - CLR, az + hzs + CLR)   # the tabs and their way in from -y
    cuts += [body, tabs]
    for x in hx:
        cuts.append(ycyl(x, az, 0.85, ry0 - 3.0, ry1 + 1.0))                                    # O1.7 pilot through the column
        cuts.append(ycyl(x, az, 2.2, ay - 80.0, ay + YS + SG["tab_y"][0]))                      # the screw's head and the driver from -y
    # the idler's bracket and gusset
    b0, b1 = ay - Y_BRK - INSERT[1], ay - Y_BRK
    drawn.append(box(ax - 6.0, ax + 6.0, b0, b1, pz - 1.0, az + 7.0))
    g = trimesh.convex.convex_hull(np.array([[ax + s, y, z] for s in (-2.5, 2.5) for (y, z) in ((b0 + 0.5, pz - 1.0), (b0 + 0.5, az - 8.0), (y0 + 0.5, pz - 1.0))]))
    drawn.append(man(g))
    cuts.append(ycyl(ax, az, INSERT[0] / 2, b0 - 1.0, b1 + 1.0))                                 # the M3 insert's hole (the screw from -y)
    return U(drawn) - U(cuts)


if __name__ == "__main__":
    fr = frame()
    from parts.scout_v4_cradle.make import cradle_local
    cr = cradle_local(); cam = camera_local(); horn = horn_local(); sv = servo_frame()
    cam0 = camera_local(leds=False); leds = cam - cam0

    def cam_leds_out(t):
        """the two LED bars lowered onto the case's top and bottom from t out, through their windows"""
        from parts.scout_v4_head_lib import F0, H
        return box(F0 - 8.0, F0, -12.5, 12.5, H + t, H + 1.2 + t) + box(F0 - 8.0, F0, -12.5, 12.5, -H - 1.2 - t, -H - t)
    # the assembly checks (overlap volumes, mm3; 0 = clear)
    tilt = np.degrees(S.TILT)
    moving = to_tm(cr + cam + horn)
    swept = sweep_hull(moving, S.HEAD["range"][0], S.HEAD["range"][1])
    chk = {"the cradle, camera, LEDs and horn over the nod -20..+90 vs the frame": gap(swept, fr),
           "the SG90 seated vs the frame": gap(sv, fr),
           "the SG90 and horn slid in from 20 along -y vs the frame": max(gap(sv.translate([0, -t, 0]) + place(horn, 0.0, (0, -t, 0)), fr) for t in np.arange(0.0, 20.1, 1.0)),
           "the cradle (camera in) lowered from 45 above onto the horn at nod 0 vs the frame": max(gap(place(cr + cam, 0.0, (0, 0, t)), fr) for t in np.arange(0.0, 45.1, 1.0)),
           "the cradle lowered onto the horn vs the SG90 and horn": max(gap(place(cr, 0.0, (0, 0, t)), place(horn, 0.0) + sv) for t in np.arange(0.0, 45.1, 1.0)),
           "the camera (LED bars off: they go on through their windows once it is in) slid in from 50 behind vs the cradle":
               max(gap(cam0.translate([-t, 0, 0]), cr) for t in np.arange(0.0, 50.1, 1.0)),
           "the LED bars set in through their windows vs the cradle": max(gap(leds.translate([0, 0, s_ * t]), cr) for s_ in (1, -1) for t in [0.0]) + max(gap(cam_leds_out(t), cr) for t in np.arange(0.0, 6.1, 1.0)),
           "the SG90 seated vs the cradle and camera over the nod (the horn sits on its spline)": gap(sweep_hull(to_tm(cr + cam), S.HEAD["range"][0], S.HEAD["range"][1]), sv)}
    ok = all(v < 1e-3 for v in chk.values())
    for k, v in chk.items():
        print(f"  assembly: {k}: {'clear' if v < 1e-3 else f'OVERLAP {v:.2f} mm3'}")
    # the lowest point of the swept head over the plate
    pz = S.THROAT["z1"] + S.HEAD["clip"] + S.HEAD["plate"]
    lo = min(float(trimesh.transform_points(to_tm(cr + cam).vertices, place.__globals__["pose"](d))[:, 2].min()) for d in np.arange(-20.0, 90.1, 2.0))
    print(f"  assembly: the swept head's lowest point {lo - pz:.2f} over the plate's top")
    tm = to_tm(fr)
    pr = tm.copy(); pr.apply_translation(-pr.bounds[0])                                         # clip down, as built
    from scout_v4_probe_mech_lib import save                                                    # noqa: E402
    res = save("scout_v4_head_frame", pr, tm, f"clip down, tree supports under the plate, ~2 h; assembly {'PASS' if ok else 'FAIL'}")
    to_tm(sv).export(os.path.join(ROOT, "parts", "scout_v4_probe_servo", "part.stl"))                 # the page's SG90 at its new seat
    os.makedirs(os.path.join(ROOT, "parts", "scout_v4_probe_horn"), exist_ok=True)
    to_tm(place(horn, tilt)).export(os.path.join(ROOT, "parts", "scout_v4_probe_horn", "part.stl"))   # its horn, a part of its own: it turns with the cradle (parked)
    from topo.build import render
    render(pr, os.path.join(HERE, "preview.png"), title="scout_v4_head_frame (drawn prototype)",
           info=f"{pr.volume / 1000 * PCTG:.1f} g | solid check {res} | assembly {'PASS' if ok else 'FAIL'} | as printed: clip down")
    sys.exit(0 if ok and res == "PASS" else 1)
