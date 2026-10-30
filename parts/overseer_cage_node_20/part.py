"""Overseer cage node 20 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(20)
SPEC = dict(h=1.0, gen_cm3=2.5, max_worst=0.5)   # 2.5 (30 Oct, auto mode): 1.7 passed strength (0.36) but the fixed gate finds the collar cluster at (-157, 0, 110) hanging on a 1.8 mm neck (483 mm3 on 6 mm2): the same budget cause as node 2, the web to the collar too thin for the volume
