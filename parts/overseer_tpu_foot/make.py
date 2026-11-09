"""TPU foot for the six base nodes (8 Nov, Spark: yes, if it clears the funnel; so 6 tall, within the cage's "contact <= 6 under the
lenses" rule, not the report's 12). A TPU 95A cap glued (CA) over the node's O16 x 3 landing face: O22, 6 tall, a O16.6 x 2.6
recess on top, 2 mm round on the ground edge. Drawn only: python parts/overseer_tpu_foot/make.py writes part.stl. Print 6 at
once, flat, 100 % infill, slow (TPU). Energy: 95A TPU at ~50 % strain over 3.4 mm adds ~3 mm of stopping distance and spreads
the hit over the whole O16 face instead of the node's rim."""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
R, H, RR, RD, ROUND = 15.0, 6.0, 12.3, 2.6, 2.0          # 8 Nov: O30 x 6 on the O24 node foot (was O22 on O16), recess O24.6 x 2.6
body = mf.Manifold.cylinder(H - ROUND, R, R, 96).translate([0, 0, ROUND]) + \
    mf.Manifold.cylinder(ROUND, R - ROUND, R, 96) + \
    mf.Manifold.revolve(mf.CrossSection.circle(ROUND, 48).translate([R - ROUND, ROUND]), 96)
m = body - mf.Manifold.cylinder(RD + 0.01, RR, RR, 96).translate([0, 0, H - RD])
g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts); t.export(os.path.join(HERE, "part.stl"))
print(f"{abs(t.volume):.0f} mm3 = {abs(t.volume) * 1.21e-3:.1f} g TPU each, watertight {t.is_watertight}")
