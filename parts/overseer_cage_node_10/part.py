"""Overseer cage node 10 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(10)
SPEC = dict(h=1.0, gen_cm3=6.0, max_worst=0.3)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 4.3 -> 6.0, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)
