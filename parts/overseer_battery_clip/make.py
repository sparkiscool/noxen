"""Overseer battery clip (3 Nov): the snap that holds the Zeee pack in parts/overseer_battery_holder, as its own drawn part. A drawn
flexure is free on every side, so inside the holder the audit's room rule refused it at h 1.5 however it was cut; here it is
drawn only, no topology run: python parts/overseer_battery_clip/make.py writes part.stl. In the drone frame, as mounted.
* A PETG strip 12 x 2.5 (the 2 mm rule; 1.27 g/cm3), from the thumb end x -99.5 to the root x -34.5, its top 7 under the pack
  (z -70.6). The root (x -54.5..-34.5) bolts up to the holder's clip pad with 2 M3 x 8 into the pad's M3 x 5.7 inserts
  (O3.4 holes at x -49.5 / -39.5).
* 38 free from the pad to the hook. Release: press the thumb end down 3.5 (0.9 % strain, PETG yields at ~4 %).
* The hook: a post at x -96..-90 rises to 1 under the pack, then a wedge to 2.5 above the pack's bottom. Its vertical catch
  face is at x -90, 1 behind the pack's rear face (-89). A 6 mm ramp aft lets the pack push it down on the way in.
* Print flat (strip down, the hook up), 100 % infill, 4 walls."""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
X0, ZB = -89.5, -63.6                                       # holder cavity rear end and floor (parts/overseer_battery_holder)
W, T, GAP = 12.0, 2.5, 7.0
Z1 = ZB - GAP; Z0 = Z1 - T                                  # strip top -70.6, bottom -73.1
X_THUMB, X_ROOT = X0 - 10.0, X0 + 55.0                      # -99.5 .. -34.5
SCREWS = (X0 + 40.0, X0 + 50.0)
CATCH, HOOK, RAMP, POST = X0 - 0.5, 2.5, 6.0, 6.0


def side(poly_xz):
    """an (x, z) polygon extruded across y +-W/2"""
    cs = mf.CrossSection([np.asarray(poly_xz, float)])
    if cs.area() == 0:
        cs = mf.CrossSection([np.asarray(poly_xz, float)[::-1]])
    m = mf.Manifold.extrude(cs, W)                          # local (x, z) in the plane, extrusion along local z
    return m.transform(np.array([[1.0, 0, 0, 0], [0, 0, 1.0, -W / 2], [0, 1.0, 0, 0]]))


strip = mf.Manifold.cube([X_ROOT - X_THUMB, W, T]).translate([X_THUMB, -W / 2, Z0])
hook = side([(CATCH - POST, Z1), (CATCH, Z1), (CATCH, ZB + HOOK), (CATCH - RAMP, ZB - 1.0), (CATCH - POST, ZB - 1.0)])
holes = [mf.Manifold.cylinder(T + 2.0, 1.7, 1.7, 32).translate([x, 0.0, Z0 - 1.0]) for x in SCREWS]
m = strip + hook - mf.Manifold.batch_boolean(holes, mf.OpType.Add)
g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts)
t.export(os.path.join(HERE, "part.stl"))
b = t.bounds
print(f"{abs(t.volume):.0f} mm3 = {abs(t.volume) * 1.27e-3:.1f} g, watertight {t.is_watertight}, box x {b[0][0]:.1f}..{b[1][0]:.1f} z {b[0][2]:.1f}..{b[1][2]:.1f}")
