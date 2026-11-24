"""Scout v4 probe BENCH turret (10 Nov, Spark: "start printing the parts needed for the bench test"): the flight turret's drawn
mechanism exactly as its setup draws it (parts/scout_v4_probe_turret/part.py bodies(): the spool's axle bosses, the N20's face plate
and its shaft's tip bearing, the throat, the pressure roller's rod guides and its leaf pocket, the drum, the four floor
bosses), joined by flat plates instead of the generated structure, less every one of the setup's cuts. So the bench
runs the flight interfaces while the flight turret's strength run goes on; not a flight part (crash and fold cases not designed for).
Drawn: python parts/scout_v4_probe_bench_turret/make.py writes print/scout_v4_probe_bench_turret.stl (part.stl in the drone frame).
  plates 3 thick: the two side walls outside the axle bosses (+y beyond the gear, -y beyond the drum), a floor on the deck's top,
  and a back wall behind the throat; bolts to parts/scout_v4_probe_bench_base with 4 x M3 x 8 into its inserts
24 Nov (Spark's build: the spool could not enter the closed drum ring, the N20's 2 x M1.6 had no driver line and their heads stood in
the pinion, the pinion had no way into its bay; systems/scout_v4_bench.assembly.json now has each part's own path, in the build's order):
  drum: a U open up and back (-x, rising 0.6 per mm: the spool's way in with its O18 drum hub and the cable's coil, its 40 T gear over
  the floor and the -x floor bosses, level it met the floor); its two walls run on straight to x -11.8 and keep the loose coil
  pinion: its bay open downward 11.92 wide (O11.42 + the fit table's mate), the turret off the base: it waits there for the shaft
  N20: no screws (the plate's two O1.8 holes stay, unused). Two rails 4.0 x 8.0 from the face plate back along the gearbox's sides,
  12.3 apart (12.0 + the fit table's press: a hand push) take its torque; the face plate stops it forward, the pinion pressed on its
  shaft (0.15 off the plate) and the base's opening (3.0 behind the can) stop it backward. Hand calculation: stall 216 N mm
  (S.N20_STALL) on the rails' 8.0 height = 27 N a rail; all of it at a rail's tip, 10 from its root: M 270 N mm on 8.0 x 4.0^2 / 6 =
  21.3 mm3 = 12.7 MPa across the layers, PCTG Zt 23 (topo/material.py): margin 1.8 (3.6 with the load spread along the rail)
  the -y wall's foot notched 13 x 0.9 over the can's top: the N20 goes in straight along +y (its shaft needs 9.85 of travel)
Print: on its -y wall (the drum's end) down: the axle bores and the roller's slots print round and true; tree supports under the throat's seat, the N20 block and its rails; 0.2 layers, 4 walls, 40 % infill, ~3 h E."""
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
          box(cx - 36, cx + 36, -26.0, ym, DZ1, 67.0),                                     # -y wall (the axle's boss, the drum's end), out to the floor bosses' faces (-26): the flat side it prints on (14 Nov: the roller's end pockets are gone, its guides end at |y| 12)
          box(cx - 36, cx + 36, ym - 3.0, yp + 3.0, DZ1, DZ1 + 3.0),                # floor on the deck's top
          box(mx + 3.0, mx + lh + 5.0, -S.BLADE_W / 2 - 2.5, S.BLADE_W / 2 + 2.5, DZ1, t["z1"]),   # the throat's +x side filled to the back wall (the slot cut from it)
          box(mx + lh + 2.0, mx + lh + 5.0, ym - 3.0, yp + 3.0, DZ1, t["z1"]),      # back wall behind the throat, to its seat
          box(cx - 8.0, cx + 8.0, S.N20_FACE + 0.15, yp + 3.0, pz - 5.5, DZ1 + 0.1)]   # from the N20's face plate to the +y wall, below the deck's top, round the pinion (cut) and the shaft's tip bearing
solid = U(plates) + B["drawn"]
g0 = S.N20_FACE
from parts.scout_v4_units_lib import TURRET_BOLTS                                               # noqa: E402
posts = U([mf.Manifold.cylinder(DZ1 - 0.0 + 30.0, 5.0 + 0.5, 5.0 + 0.5, 64).translate([x, y, -30.0]) for x, y in TURRET_BOLTS])   # 15 Nov (assembly check): the bench base's O10 insert posts under the deck's top, 0.5 clear (the N20 cheeks stood 0.5 into the +y pair)
below = box(-200, 200, -200, 200, -50, 16.0 + 0.3) - box(cx - 8.3, cx + 8.3, g0 - 30.0 + 0.2, g0 + 11.0 - 0.2, -60, 20)   # 15 Nov: what hangs below the base's top (z 16) stays 0.2 inside its opening (the N20 cheeks met its edges at 0)
part = solid - B["cuts"] - posts - below - box(cx - 6.5, cx + 6.5, g0 - 27.5, S.SPOOL["y0"] - 0.6, DZ1 - 1.0, DZ1 + 3.1)   # 12 Nov: the floor open over the N20's can (its top z 26.5 left a 1.0 skin in the 3.0 floor: the solid check)
from topo import fit as FIT                                                                    # noqa: E402
gy0, gy1 = S.GEAR["y"]; dr = S.DRUM["r"]; SLOPE = 0.6; TIP_D, HUB_D = 11.42, 7.0                   # the pinion's tip circle and hub boss (parts/scout_v4_probe_pinion/make.py)
def ysweep(r, y0, y1):                                                                         # a disc r on the spool's axis swept the spool's way out: up and back, SLOPE up per mm of -x
    c = lambda x: mf.Manifold.cylinder(y1 - y0, r, r, 96).rotate([-90, 0, 0]).translate([cx - x, y0, cz + SLOPE * x])
    return mf.Manifold.batch_hull([c(0.0), c(60.0)])
cavity = ysweep(dr + 0.3, ym, S.DRUM["y1"] + 0.5)                                              # the drum hub's and the coil's way (the ring's own r 9.3 cavity, swept)
drum_u = (ysweep(dr + 2.8, ym - 0.5, S.SPOOL["y0"] - 0.6) ^ box(cx - dr - 2.8, cx + 40, -40, 40, 0, 100)) - cavity   # the ring's 2.5 wall round it, ended at x -11.8
pin_w = FIT.across(TIP_D, "mate"); hub_w = FIT.across(HUB_D, "mate")
pinion_way = box(cx - pin_w / 2, cx + pin_w / 2, gy0 - 0.5, gy1 + 0.5, pz - 20, pz) + box(cx - hub_w / 2, cx + hub_w / 2, gy1, gy1 + 0.9, pz - 20, pz)   # the pinion (and its hub boss) up into its bay from below
n20_w = FIT.across(12.0, "press"); n20_h = FIT.across(10.0, "slide")
rails = U([box(*sorted((cx + sx * n20_w / 2, cx + sx * (n20_w / 2 + 4.0))), g0 - 10.0, g0 + 0.2, 16.0 + 0.5, DZ1) for sx in (-1, 1)])   # beside the gearbox, above the base's top (z 16)
n20_way = box(cx - 6.5, cx + 6.5, -26.1, g0 - 27.5 + 0.2, pz - n20_h / 2, pz + n20_h / 2)       # the can's way in under the -y wall's foot (the floor's opening ended at the seated can)
part = (part - cavity) + drum_u + rails - pinion_way - n20_way
m = part.to_mesh(); tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)
parts_ = tm.split(only_watertight=False)
tm = max(parts_, key=lambda p: abs(p.volume)) if len(parts_) > 1 else tm         # loose RM collar stubs off the plates are dropped
print('dropped pieces:', [(round(abs(p.volume), 1), np.round(p.bounds.mean(0), 1).tolist()) for p in parts_ if p is not tm])
pr = tm.copy(); pr.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])); pr.apply_translation(-pr.bounds[0] * [0, 0, 1])   # -y wall on the bed (drone y -> print z)
save("scout_v4_probe_bench_turret", pr, tm, f"-y wall down, ~3 h; {len(parts_)} piece(s)")
