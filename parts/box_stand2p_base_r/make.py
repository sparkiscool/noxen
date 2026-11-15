"""Box stand v2 PROTOTYPE, base half right (+X) (15 Nov 2026): the yoke under the mast's deck in one piece with the rear rail, the rear
foot and the splice plate; geometry, holes, assembly path and loads in parts/box_stand2p_lib.py.
Hand check (python parts/box_stand2p_base_r/make.py prints it; sections sliced from this mesh, the floor at this side's feet, every case of
check.json, flat floor and each foot lifted): worst index 0.352 (the gate's 0.5; index = the von Mises stress over the in-layer allowable, PCTG E, SF 4 flight / 2.5 misuse / 10 sustained):
  yoke box at W X 45: 0.213 (box handling 3 g down + 2 g side (+X), uneven floor, left front foot off), A 1438 mm2, W 21638 / 27554 mm3
  rail ahead of the yoke (W Y 35): 0.352 (box handling 3 g down + 2 g side (+X), uneven floor, left front foot off), A 1038 mm2, W 12220 / 10779 mm3
  rail behind the yoke (W Y -36): 0.016 (standing: arm v8, 11 bays with packs, the ring at its worst unbalance, uneven floor, left rear foot off), A 2162 mm2, W 111004 / 27952 mm3
  splice bolt pull through its washer (526 N): 0.178 (box handling 3 g down + 2 g side (+X), uneven floor, left front foot off)
  splice bolt bearing: 0.074 (standing: arm v8, 11 bays with packs, the ring at its worst unbalance, uneven floor, right rear foot off)
Print: on its floor face (W Z -211 down: every wall's edge, the yoke's floor plate and the foot pad); the yoke box's deck and the channel's
top bridge 25 mm cells; no supports; 0.2 layers, 4 walls, 25 % gyroid, PCTG at PETG + 10 C. 604.8 g solid, ~18 h (E)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np
import parts.box_stand2p_lib as P
S = 1
tm, line = P.save("box_stand2p_base_r", P.base(S), np.eye(4), "floor face down")
rows = P.hand_check("base_r", tm, [("yoke box at W X 45", np.array([S * 45.0, -1.0, -173.0]), np.array([S * 1.0, 0, 0]), 53 * 71, 1),
                                   ("rail ahead of the yoke (W Y 35)", np.array([S * 149.0, 35.0, -190.0]), np.array([0, 1.0, 0]), None, 1),
                                   ("rail behind the yoke (W Y -36)", np.array([S * 149.0, -36.0, -190.0]), np.array([0, 1.0, 0]), None, -1)],
                    lambda: P.floor_loads(S, ("rear", "front")))
rows += P.splice_check(S)
P.extra(line, rows)
