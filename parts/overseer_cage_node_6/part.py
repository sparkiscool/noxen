"""Overseer cage node 6 (v6 setup: socket walls and floors drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(6)
SPEC = dict(h=1.0, gen_cm3=3.5, max_worst=0.5)   # 3.5 (30 Oct, auto mode): 2.7 gave 0.54 on the tube 6-18 pull-out (248 N, run 62), gate OK, room 0.90-0.94 behind the sockets (audit): budget, not room
