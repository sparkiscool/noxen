"""Swap arm v8 bearing fit coupon (15 Nov 2026): before the hub and the stub are printed. The 6010-2RS (50 x 80 x 16) presses into the hub's sleeve and
onto the stub's spigot at the fit table's "press" allowance (+-0.3: bore o 80.3, spigot o 49.7), which comes from a 3 mm D shaft, not an 80 mm bearing.
One piece: a ring whose bore steps down o 80.5 / 80.3 / 80.1 (5 each, the widest on top) round a plug that steps up o 49.5 / 49.7 / 49.9 (5 each, the
thinnest on top), on a 2 mm floor. Spark sets a bearing on it and reports one line: how many steps it goes down by hand in the ring (0..3) and on the
plug (0..3); 2 and 2 = the drawn sizes are right (a firm push onto the second step). No load: no hand calc (index 0).
Print: floor down, no supports, 0.2 layers, 3 walls; 37 g, ~1 h 30 (E)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import box_arm8p_lib as P
from box_arm8_lib import cyl, U
ring = cyl("z", 0, 0, 44.0, 0, 17.0, 128) - U(*[cyl("z", 0, 0, d / 2, 2.0 + 5.0 * k, 18.0, 128) for k, d in enumerate((80.1, 80.3, 80.5))])
plug = U(*[cyl("z", 0, 0, d / 2, 0, 2.0 + 5.0 * (3 - k), 128) for k, d in enumerate((49.5, 49.7, 49.9))]) - cyl("z", 0, 0, 20.0, -1, 20, 64)
floor = cyl("z", 0, 0, 44.0, 0, 2.0, 128) - cyl("z", 0, 0, 20.0, -1, 3, 64)
P.save("box_arm8p_coupon", U(ring, plug, floor), P.rot([1, 0, 0], 0), "floor down, no supports; report: steps down in the ring, steps down on the plug", lambda tm: [dict(check="fit coupon: no load", index=0.0, case=None)])
