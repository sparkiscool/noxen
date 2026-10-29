"""Overseer cage node 5 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(5)
SPEC = dict(h=1.2, gen_cm3=4.5, max_worst=0.5)   # 3.5 (29 Oct): 2.7 gave 0.66 in the landing case at the foot (engine thick6)   # 4.5 (29 Oct): 3.5 gave 16.0 g at 0.56, landing 6 g on the foot again; and the camera rod now goes to node 20, one socket fewer here
