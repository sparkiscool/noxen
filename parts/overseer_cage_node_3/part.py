"""Overseer cage node 3 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(3)
SPEC = dict(h=1.0, gen_cm3=12.0, max_worst=0.2, settings=dict(min_member_mm=5.0))   # base ring (8 Nov, Spark: the rear base node broke in three): ground-hit case, members >= 5, limit 0.2, 2x the 30 Oct budget (8.4 -> 12.0)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 6.0 -> 8.4, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)   # 28 Oct: was 4.0, worst index over 0.5 at the tie socket
# rerun 30 Oct (Spark): the bay tie hub ends sit at z -9.0 in the cage file (were -5.1), so this node's bay tie socket aims 1.2 deg lower
