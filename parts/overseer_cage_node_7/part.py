"""Overseer cage node 7 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(7)
SPEC = dict(h=1.0, gen_cm3=5.0, max_worst=0.5)   # 28 Oct: was 3.3, worst index over 0.5 at the tie socket
