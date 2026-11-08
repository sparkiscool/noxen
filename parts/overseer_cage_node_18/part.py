"""Overseer cage node 18 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(18)
SPEC = dict(h=1.2, gen_cm3=8.4, max_worst=0.3)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 6.0 -> 8.4, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)
