"""Scout v4 probe BENCH turret (10 Nov, Spark: "start printing the parts needed for the bench test"): the flight turret's drawn
mechanism exactly as its setup draws it (parts/scout_v4_probe_turret/part.py bodies(): the spool's axle bosses, the N20's face plate
and its shaft's tip bearing, the throat, the pressure roller's guides with their slots and spring pockets, the drum, the four floor
bosses), joined by flat plates instead of the generated structure, less every one of the setup's cuts. So the bench
runs the flight interfaces while the flight turret's strength run goes on; not a flight part (crash and fold cases not designed for).
Drawn: python parts/scout_v4_probe_bench_turret/make.py writes print/scout_v4_probe_bench_turret.stl (part.stl in the drone frame).
  plates 3 thick: the two side walls outside the axle bosses (+y beyond the gear, -y beyond the drum), a floor on the deck's top,
  and a back wall behind the throat; bolts to parts/scout_v4_probe_bench_base with 4 x M3 x 10 into its inserts
Print: on its -y wall (the drum's end) down: the axle bores and the roller's slots print round and true; tree supports under the throat's seat and the N20 block; 0.2 layers, 4 walls, 40 % infill, ~3 h E."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "parts"))
import numpy as np, trimesh, manifold3d as mf
from parts.scout_v4_probe_turret import part as T
from parts.scout_v4_units_lib import S, box, U, DZ1, socket_bodies
from scout_v4_probe_mech_lib import save                                         # noqa: E402

T.disc_cuts = lambda: []                                                          # the props' keep-out is a flight constraint: not on the bench
T.socks_of = lambda *a, **k: []; T.check_collars = lambda *a, **k: None              # no roof rods on the bench: no RM sockets, bores or ports
B = T.bodies()
cx, cz = S.SPOOL["c"]; t = S.THROAT; mx = S.MAST_X
yp, ym = S.GEAR["y"][1] + 0.8, S.DRUM["y0"] - 0.8
lh = T.lens_half(); pz = cz - S.GEAR['c']
env = B["env"]
plates = [box(cx - 36, cx + 36, yp, yp + 3.0, DZ1, 67.0),                           # +y wall (the axle's and the shaft tip's bosses)
          box(cx - 36, cx + 36, -26.0, ym, DZ1, 67.0),                              # -y wall (the axle's boss, the drum's end), out to the floor bosses' faces: the flat side it prints on
          box(cx - 36, cx + 36, ym - 3.0, yp + 3.0, DZ1, DZ1 + 3.0),                # floor on the deck's top
          box(mx + 3.0, mx + lh + 5.0, -S.BLADE_W / 2 - 2.5, S.BLADE_W / 2 + 2.5, DZ1, t["z1"]),   # the throat's +x side filled to the back wall (the slot cut from it)
          box(mx + lh + 2.0, mx + lh + 5.0, ym - 3.0, yp + 3.0, DZ1, t["z1"]),      # back wall behind the throat, to its seat
          box(cx - 8.0, cx + 8.0, S.N20_FACE + 0.15, yp + 3.0, pz - 5.5, DZ1 + 0.1)]   # from the N20's face plate to the +y wall, below the deck's top, round the pinion (cut) and the shaft's tip bearing
solid = U(plates) + B["drawn"]
g0 = S.N20_FACE
part = solid - B["cuts"] - box(cx - 6.5, cx + 6.5, g0 - 27.5, S.SPOOL["y0"] - 0.6, DZ1 - 1.0, DZ1 + 3.1)   # 12 Nov: the floor open over the N20's can (its top z 26.5 left a 1.0 skin in the 3.0 floor: the solid check)
m = part.to_mesh(); tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
parts_ = tm.split(only_watertight=False)
tm = max(parts_, key=lambda p: abs(p.volume)) if len(parts_) > 1 else tm         # loose RM collar stubs off the plates are dropped
pr = tm.copy(); pr.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); pr.apply_translation(-pr.bounds[0] * [0, 0, 1])   # -y wall on the bed (drone y -> print z)
save("scout_v4_probe_bench_turret", pr, tm, f"-y wall down, ~3 h; {len(parts_)} piece(s)")
