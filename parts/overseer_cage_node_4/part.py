"""Overseer cage node 4 (v6 setup: socket walls and floors and the landing face drawn, the rest generated; parts/cage_node_lib.py)."""
from parts.cage_node_lib import make
setup = make(4)
SPEC = dict(h=1.0, gen_cm3=13.5, max_worst=0.2)   # 8 Nov (Spark: go): 9.0 -> 13.5 and a Ø24 foot, the ground hit gave 0.25 at 9.0 on the Ø16 foot   # base ring (8 Nov, Spark: the rear base node broke in three): ground-hit case, limit 0.2 (the layout already lifts the member size to 6.25 so members survive the gate), 2x the 30 Oct budget (6.3 -> 9.0)   # 7 Nov (Spark: the bottom nodes cracked in the body between sockets in testing): 4.5 -> 6.3, limit 0.5 -> 0.3 (printed PETG broke at 0.30 of the engine's strength on the arms)   # 28 Oct: was 2.3, worst index over 0.5 at the tie socket
# rerun 30 Oct (Spark): the bay tie hub ends sit at z -9.0 in the cage file (were -5.1), so this node's bay tie socket aims 1.2 deg lower
