"""Overseer cage node 15 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(15)
SPEC = dict(h=1.25, gen_cm3=4.9, max_worst=0.5)   # h 1.0 -> 1.25 (4 Nov): the 6.2 head-rod / core-tie socket rerun hit the 5.5 h cap 4 times at h 1.0
