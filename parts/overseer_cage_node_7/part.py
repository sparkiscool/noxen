"""Overseer cage node 7 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(7)
SPEC = dict(h=1.0, gen_cm3=9.8, max_worst=0.3)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 7.0 -> 9.8, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)   # 28 Oct: was 3.3 then 5.0, worst index 0.60 across the layers at the core tie socket (1461 N)
