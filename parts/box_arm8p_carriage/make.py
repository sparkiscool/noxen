"""Swap arm v8 PROTOTYPE carriage (15 Nov 2026), two per arm: drawn, the first print (the engine setup parts/box_arm8_carriage/part.py is kept for OPTIMISE).
Geometry: parts/box_arm8_lib.py carriage_proto(): the setup's room solid (the 9.85 wheel plate, the 13.7 x 23.6 spine with the MGN12 rail's seat, the rib
between the wheel pair, the 44 wide strap to the third wheel, the pick-up stop) with its interfaces: four M3 heat-set inserts (o 4.0) under the lift rail,
three V-wheel axles. The axles are new (the bores had no thread to end in, no spacer, and the third wheel's eccentric was never drawn): each an M5 x 20
button head from behind the wheel, an o 10 x 1 washer on the inner ring, an o 15 x 1.2 washer sunk in the plate's back, the plate's 3.15 web, the same
washer and an M5 nut sunk in its front; the third wheel's bore and recesses are slots, +-1.0 across the rail: pushed home on the rail, then tightened.
Hand calc (index as the hub's; wheel forces from box_arm8_lib.wheel_loads at each wheel's contact, 11.4 off its axle):
  spine and plate under the top wheel (A 569 mm2): 0.53; over the low wheel (A 552): 0.54 (dropped pack: 245 N along the axle at each wheel of the pair)
  strap to the third wheel: 0.10        axle pried on the plate through the o 15 washers: 0.65, its washers through the web: 0.21 (dropped pack)
  MGN rail's four M3 inserts at 900 N (E): 0.84 (insert off square; 0.36 in a swap)
Print: the rail seat's plane down (everything ends on it: 102.5 x 86.3 on the bed, 23.6 tall), no supports; 0.2 layers, 4 walls, 40 % gyroid; 97 g, ~4 h (E).
Assembly path: wheels, axles and the MGN rail on at the bench, then rolled onto the V rail from its tip ("carriage A / B onto its rail")."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import box_arm8p_lib as P
P.save("box_arm8p_carriage", P.L.carriage_proto(), P.rot([1, 0, 0], -90), "rail seat's plane down, no supports", P.carriage_check)
