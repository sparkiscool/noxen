"""Overseer cage node 6 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(6)
SPEC = dict(h=1.0, gen_cm3=5.7, max_worst=0.3)   # 10 Nov: 4.9 -> 5.7, PCTG run 0.319 (tube 6-18 pull-out 248 N; PETG 0.282)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 3.5 -> 4.9, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)   # 3.5 (30 Oct, auto mode): 2.7 gave 0.54 on the tube 6-18 pull-out (248 N, run 62), gate OK, room 0.90-0.94 behind the sockets (audit): budget, not room
