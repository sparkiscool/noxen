"""Overseer cage node 5 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(5)
SPEC = dict(h=1.2, gen_cm3=3.5, max_worst=0.5)   # 3.5 (29 Oct): 2.7 gave 0.66 in the landing case at the foot (engine thick6)
