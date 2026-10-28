"""Overseer cage node 3 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(3)
SPEC = dict(h=1.0, gen_cm3=4.0, max_worst=0.5)
