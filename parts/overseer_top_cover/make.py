"""Overseer top cover (Spark 3 Nov: a cover over the centre above the FC, for looks, not airtight, good room for the cables; held by
the 8 top plate bolts; FC stack tallest point 26 above the plate; clears the props; the Noxen N mark on top, no text
(noxensys.com/assets/icon.svg); clamps seamlessly to the top plate and the arms, sandwiching the plate, no seams; aesthetic and
thin: printed in translucent PETG). Drawn only, no topology run (a cosmetic shell, no load path):
python parts/overseer_top_cover/make.py writes part.stl.

* Skirt = the top plate's own outline (sectioned from parts/top_plate/part.stl, the bought plate): a 1.2 lip down past the plate
  edge to the plate underside z 45 (0.15 clear), landing on the arm tops where the roots leave the plate, then a 45 deg flare in
  from the plate edge into the dome: plate edge and cover read as one piece, the plate sandwiched between cover and arms. The 8
  top plate bolts A (arm lib BOLT_TOP) seat on a 1.6 ring on the plate top inside a O7.2 driver well through the flare: M3 x 50
  as now (drilled bottom plate), the end 1.4 proud instead of 3.
* Dome: rises from 9 in from the plate edge (clear of the bolt heads and a driver) to a flat top at z 75.3 (stack 26 + 1.5 + 1.2
  top), 6.7 under the prop blade band (z 82; the rear 10 in disks reach r 30). Its lobes draw in along a parabola (offset
  13 t^2, never steeper than 45 deg, so it prints top down without supports) while the side faces stay out at the stack.
  Walls 1.2 normal (3 lines at 0.4: solid, no infill, the clearest translucent print).
* 8 cable notches, 12 wide x 7 tall, through flange and dome foot: one at each arm (the motor leads over the arm tops) and one on
  each side face (RX antenna, USB, telemetry).
* Logo: the N mark cut through the top, 60 wide, upright seen from above with the nose forward.
* 27.2 g (C). Gate: one body; FAILs only on thin walls (1.2 everywhere, Spark asked thin for the translucent print; no load).
* Print (Bambu A1 mini, translucent PETG, dried 65 C 8 h): top face down on the smooth PEI plate with glue stick; 0.2 layers;
  line width 0.42, 3 wall loops (= the 1.2 wall: no infill anywhere), top / bottom 6 layers, pattern concentric, 100 %;
  nozzle 255, bed 70; outer wall 40, inner wall 60, first layer 20 mm/s; max volumetric speed 8; fan 20 %; flow ratio 1.03;
  seam aligned, back; no supports, no brim, no ironing."""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
PB, PT, STACK, CLR = 45.0, 46.6, 26.0, 1.5                  # plate underside / top (mesh), stack height (Spark), clearance
T, FL, LIP, GAP = 1.2, 1.6, 1.2, 0.15                       # wall, flange, lip, lip clearance to the plate edge
ZT = PT + STACK + CLR + T                                   # top 76.1
Z0 = PT + FL                                                # dome foot 48.2
INSET, DMAX, R_SIDE, DZ = 9.0, 13.0, 40.0, 0.2              # dome foot from the plate edge, lobe draw-in, side faces, slice = layer
NOTCH_W, NOTCH_H = 12.0, 7.0
BOLT_U, BOLT_V, HOLE_R = 63.17, 15.0, 1.7                   # parts/overseer_arm_lib.py BOLT_TOP (M); M3 clearance
SEAT_R, DRIVE_R = 5.0, 3.6                                  # seat under the M3 head (O5.5); driver well O7.2
PROP_Z = 82.0                                               # prop blade band bottom (arm lib Z_THRUST)
LOGO = [[(0, 0), (78, 0), (286, 257), (210, 257), (56, 66.7), (56, 257), (0, 257)],
        [(135, 0), (215, 0), (472, 257), (392, 257)],
        [(392, 0), (472, 0), (365, 107), (325, 67)]]          # noxensys.com/assets/icon.svg (viewBox units, y down)
LOGO_W = 60.0
J = mf.JoinType.Round


def ccw(p):
    p = np.asarray(p, float)
    a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if a > 0 else p[::-1]


def plate_outline():
    m = trimesh.load(os.path.join(ROOT, "parts", "top_plate", "part.stl"))
    sec = m.section(plane_origin=[0, 0, 0.5 * (PB + PT)], plane_normal=[0, 0, 1])
    loops = [d[:, :2] for d in sec.discrete]
    area = lambda q: abs(0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    return mf.CrossSection([ccw(max(loops, key=area))]).simplify(0.05)


def prism(cs, z0, z1):
    return mf.Manifold.extrude(cs, z1 - z0).translate([0, 0, z0])


def radial(poly_tz, ang, r0, r1):
    a = np.radians(ang); ur = np.array([np.cos(a), np.sin(a), 0.0]); ut = np.array([-ur[1], ur[0], 0.0])
    return mf.Manifold.extrude(mf.CrossSection([ccw(poly_tz)]), r1 - r0).transform(np.c_[ut, [0, 0, 1.0], ur, r0 * ur])


def bolts():
    out = []
    for th in (45.0, 135.0, 225.0, 315.0):
        a = np.radians(th); ud = np.array([np.cos(a), np.sin(a)]); vd = np.array([-ud[1], ud[0]])
        out += [ud * BOLT_U + vd * s * BOLT_V for s in (1, -1)]
    return out


O = plate_outline()
B0 = O.offset(-INSET, J); SIDE = B0 ^ mf.CrossSection.circle(R_SIDE, 160)


def outer_at(z):
    t = (z - Z0) / (ZT - Z0)
    return B0.offset(-DMAX * t * t, J) + SIDE


def slope_at(z):                                            # horizontal draw-in per unit height
    return 2 * DMAX * (z - Z0) / (ZT - Z0) ** 2


def flare_at(z):                                            # the skirt: vertical lip past the plate edge, then 45 deg in to the dome
    return O.offset(GAP + LIP - max(0.0, z - PT), J)


# slices of one layer, each the outline at its mid height; the wall is measured normal to the surface (45 deg flare: T x 1.41)
outs, ins = [], []
n = int(np.ceil((ZT - PB) / DZ - 1e-9))
for i in range(n):
    z = PB + (ZT - PB) * i / n; z1 = PB + (ZT - PB) * (i + 1) / n; zm = 0.5 * (z + z1)
    f = flare_at(zm); o = f + outer_at(zm) if zm >= Z0 else f
    outs.append(prism(o, z - (0.02 if i else 0.0), z1))       # overlap: one body
    if zm < PT:
        inn = O.offset(GAP, J)                               # the lip round the plate edge
    else:
        inn = f.offset(-T * np.sqrt(2.0), J)
        if zm >= Z0:
            inn = inn + outer_at(zm).offset(-T * np.sqrt(1 + slope_at(zm) ** 2), J)
    if z1 <= ZT - T + 1e-6:
        ins.append(prism(inn, z - 0.01, z1 + 0.01))
solid = mf.Manifold.batch_boolean(outs, mf.OpType.Add)
body = solid - mf.Manifold.batch_boolean(ins, mf.OpType.Add)
# bolt seats: a ring under each head on the plate top and a tube up to the skirt, clipped to the outside; the driver well through
seats = [mf.Manifold.cylinder(FL, SEAT_R, SEAT_R, 48).translate([x, y, PT]) + mf.Manifold.cylinder(ZT - PT, DRIVE_R + T, DRIVE_R + T, 48).translate([x, y, PT])
         for x, y in bolts()]
body += mf.Manifold.batch_boolean(seats, mf.OpType.Add) ^ solid

cuts = [mf.Manifold.cylinder(FL + 2.0, HOLE_R, HOLE_R, 24).translate([x, y, PT - 1.0]) for x, y in bolts()]
cuts += [mf.Manifold.cylinder(ZT - PT, DRIVE_R, DRIVE_R, 48).translate([x, y, PT + FL]) for x, y in bolts()]
nt = [(-NOTCH_W / 2, PB - 1.0), (NOTCH_W / 2, PB - 1.0), (NOTCH_W / 2, PT + NOTCH_H - NOTCH_W / 2), (0.0, PT + NOTCH_H),
      (-NOTCH_W / 2, PT + NOTCH_H - NOTCH_W / 2)]            # pointed top: prints top down without a bridge
cuts += [radial(nt, th, 30.0, 95.0) for th in range(0, 360, 45)]
s = LOGO_W / 472.0; uc, vc = 236.0, 128.5
cuts += [prism(mf.CrossSection([ccw([(-(v - vc) * s, -(u - uc) * s) for u, v in p])]), ZT - T - 1.0, ZT + 1.0) for p in LOGO]
body -= mf.Manifold.batch_boolean(cuts, mf.OpType.Add)

g = body.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts); t.merge_vertices()
t.export(os.path.join(HERE, "part.stl"))
b = t.bounds
top = np.round(outer_at(ZT - 0.1).bounds(), 1)
print(f"{abs(t.volume) / 1000:.1f} cm3 = {abs(t.volume) * 1.27e-3:.1f} g PETG, bodies {len(t.split(only_watertight=False))}, watertight {t.is_watertight}")
print(f"box x {b[0][0]:.1f}..{b[1][0]:.1f} y {b[0][1]:.1f}..{b[1][1]:.1f} z {b[0][2]:.1f}..{b[1][2]:.1f}; top face {top}; "
      f"props {PROP_Z - b[1][2]:.1f} above; stack clear {ZT - T - PT - STACK:.1f}; side faces r {R_SIDE - T:.1f} inside (stack r ~31)")
for x, y in bolts()[:2]:
    print(f"bolt ({x:.1f}, {y:.1f}): dome foot {'HIT' if (B0.offset(3.6, J) ^ mf.CrossSection.circle(0.05).translate([x, y])).area() > 0 else 'clear (head + driver)'}")
