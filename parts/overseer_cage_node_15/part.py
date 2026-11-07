"""Overseer cage node 15 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(15)
SPEC = dict(h=1.25, gen_cm3=6.9, max_worst=0.3)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 4.9 -> 6.9, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)   # h 1.0 -> 1.25 (4 Nov): the 6.2 head-rod / core-tie socket rerun hit the 5.5 h cap 4 times at h 1.0
