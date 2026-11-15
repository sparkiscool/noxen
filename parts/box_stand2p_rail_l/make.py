"""Box stand v2 PROTOTYPE, front rail left (-X), the right's mirror (15 Nov 2026): from the splice plate to the front foot; geometry, holes, assembly path and loads
in parts/box_stand2p_lib.py.
Hand check (python parts/box_stand2p_rail_l/make.py prints it): worst index 0.206 (the gate's 0.5; index = the von Mises stress over the in-layer allowable, PCTG E, SF 4 flight / 2.5 misuse / 10 sustained):
  rail behind its splice plate (W Y 119): 0.206 (box handling 3 g down + 2 g side (-X), uneven floor, right front foot off), A 1038 mm2, W 12220 / 10779 mm3
Print: on its floor face (W Z -211 down); the channel's top bridges 25 mm cells; no supports; 0.2 layers, 4 walls, 25 % gyroid, PCTG at
PETG + 10 C. 230.6 g solid, ~7 h (E).
Installable: python parts/box_stand2p_lib.py install (the declared path, every piece and tool line: PASS).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np
import parts.box_stand2p_lib as P
S = -1
tm, line = P.save("box_stand2p_rail_l", P.rail(S), np.eye(4), "floor face down")
rows = P.hand_check("rail_l", tm, [("rail behind its splice plate (W Y 119)", np.array([S * 149.0, 119.0, -190.0]), np.array([0, 1.0, 0]), None, 1)],
                    lambda: P.floor_loads(S, ("front",)))
P.extra(line, rows)
