"""Swap arm v8 PROTOTYPE lift (15 Nov 2026), two per arm: drawn, the first print (the engine setup parts/box_arm8_lift/part.py is kept for OPTIMISE).
Geometry: parts/box_arm8_lib.py lift_proto(): the setup's room solid (the 30 x 41 yoke, its 16.4 plate on the MGN12H block, the arm to the nut's lug) with
its interfaces: three o 6.2 x 28 boom sockets (resin port and three pockets each), four M3 into the block (o 3.4, each head on an o 9 washer sunk to 6 from
the block), the T8 nut's o 10.8 bore and flange slot in the lug (3.3 behind the slot: it was 1.3).
Hand calc (index as the hub's):
  yoke into the plate, its twist in the layers (A 929 mm2): 0.45 (insert at the float limit, 118 N + 3 N m)
  the four M3 on o 9 washers: 0.47 (insert off square; the bare o 5.5 heads read 1.7)
  boom sockets, a rod's moment on 28 of bore: 0.25        nut lug at the arm: 0.11 (dropped pack)
Print: the yoke's top down (bending and twist in the layers), tree supports under the lug and the arm's end (3 above the bed); 0.2 layers, 4 walls,
40 % gyroid; 152 g, ~6 h (E). 93.2 x 58.6 on the bed, 86 tall.
Assembly path: booms bonded in, then onto the MGN block's face along the axle, four M3 from the front ("lift A / B onto its block")."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import box_arm8p_lib as P
P.save("box_arm8p_lift", P.L.lift_proto(), P.rot([1, 0, 0], 180), "yoke's top down, tree supports under the lug and the arm's end", P.lift_check)
