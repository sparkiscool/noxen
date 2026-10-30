"""Overseer cage node 3 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(3)
SPEC = dict(h=1.0, gen_cm3=6.0, max_worst=0.5)   # 28 Oct: was 4.0, worst index over 0.5 at the tie socket
# rerun 30 Oct (Spark): the bay tie hub ends sit at z -9.0 in the cage file (were -5.1), so this node's bay tie socket aims 1.2 deg lower
