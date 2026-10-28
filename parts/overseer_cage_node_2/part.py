"""Overseer cage node 2 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(2)
SPEC = dict(h=1.0, gen_cm3=2.3, max_worst=0.5)
