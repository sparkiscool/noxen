"""Overseer cage node 2 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(2)
SPEC = dict(h=1.0, gen_cm3=6.3, max_worst=0.3)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 4.5 -> 6.3, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)   # 4.5 (30 Oct, auto mode): 2.3 gave 1.62 on the tube 1-2 pull-out (285 N) and the two spoke collars hanging on 1.8 mm necks (run 62); room behind every socket was 0.93 (audit), so the budget was the cause: four sockets at 230-285 N plus the foot on the volume of node 20 (125 N); nodes 4 / 10 carry 196 / 326 N at 4.5 / 4.3
