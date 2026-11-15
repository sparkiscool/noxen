"""Box stand v2 PROTOTYPE, mast (15 Nov 2026): the arm v8 stub's face down behind the hub to the foot on both base halves; geometry,
holes, assembly path and loads in parts/box_stand2p_lib.py.
Hand check (python parts/box_stand2p_mast/make.py prints it; sections sliced from this mesh, every case of check.json): worst index 0.261 (the gate's 0.5; index = the von Mises stress over the in-layer allowable, PCTG E, SF 4 flight / 2.5 misuse / 10 sustained):
  column above the foot (W Z -124): 0.224 (box handling 3 g down + 2 g side (+X)), A 1468 mm2, W 23076 / 27358 mm3
  column through the nut windows (W Z -112): 0.204 (standing: arm v8, 11 bays with packs, the ring at its worst unbalance), A 1468 mm2, W 23076 / 27358 mm3
  column at the face's foot (W Z -40): 0.173 (box handling 3 g down + 2 g side (+X)), A 1935 mm2, W 27801 / 34757 mm3
  mast foot: front row inserts (1500 N, E), rear row through (361 N): 0.261 (insert at the drone, 106.6 N every swap)
  stub on the face: Ø15 washer through 10 of face (532 N): 0.144 (box handling 3 g down + 2 g side (+X))
  for the arm stream, not this part:   the same pull on the stub flange's inserts (arm stream) (532 N): 0.887 (box handling 3 g down + 2 g side (+X))
Print: on its +X side (W X 40 down: the tube's +X wall, 82 x 179 of it on the bed); the column's bending and the stub bolts' pull both run
in the layers. No supports (the tube's -X wall bridges 25 mm cells either side of the web; the motor notch is open); 0.2 layers, 4 walls,
25 % gyroid, PCTG at PETG + 10 C. 457.5 g solid, ~14 h (E).
Installable: python parts/box_stand2p_lib.py install (the declared path, every piece and tool line: PASS).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np
import parts.box_stand2p_lib as P

tm, line = P.save("box_stand2p_mast", P.mast(), P.rot([0, 1, 0], 90), "on its +X side")
rows = P.hand_check("mast", tm, [("column above the foot (W Z -124)", np.array([0, -32.0, -124.0]), np.array([0, 0, 1.0]), 77 * 59, 1),
                                 ("column through the nut windows (W Z -112)", np.array([0, -32.0, -112.0]), np.array([0, 0, 1.0]), None, 1),
                                 ("column at the face's foot (W Z -40)", np.array([0, -32.0, -40.0]), np.array([0, 0, 1.0]), 77 * 59, 1)],
                    P.arm_loads)
j, arm = P.mast_joints()
rows += j
line['for_the_arm_stream'] = arm
P.extra(line, rows)
