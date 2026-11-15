"""Box stand v2 PROTOTYPE, foot (15 Nov 2026): TPU 95A, Ø36 x 5 with a Ø7.8 x 2.8 stud pressed into a foot pad's Ø8 x 3 hole; four.
The heaviest foot load is 188 N (box handling, an uneven floor): 0.18 MPa on the pad, nothing for TPU 95A.
Print: pad down, TPU 95A, 0.2 layers, 3 walls, 30 % infill, four on one plate. 6.6 g solid, ~1.5 h for four (E)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np
import parts.box_stand2p_lib as P
tm, line = P.save("box_stand2p_foot", P.foot(), np.eye(4), "pad down, TPU 95A, x4")
P.extra(line, [])
