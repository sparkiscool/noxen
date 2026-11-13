# overseer_cage_node_4 feedback

## 11 Nov (Spark): the foot nodes broke again, redesign
- "The foot nodes broke again, we need to redesign them" / "so far it pretty much just snaps everywhere". Every printed bottom node (0-5) has broken repeatedly: 7 Nov cracked in the body between sockets, 8 Nov the rear base node broke in three, 10 Nov all cracked (resined back), 11 Nov broke again.
- "I want a full in depth design on this". Materials on hand: PCTG, TPU 95A (external spool, AMS cannot feed it), TPU for AMS (68D). Spark asked whether to go full TPU for the bottom nodes; the design chat's view: a rigid PCTG socket core with TPU taking the hit, but the study decides.
- Rods: Spark cuts the rods flush at the old nodes (they are glued in), so "design for one node length back for the rods": each socket meets its rod where the old printed node's collar mouth was.
- Spark 11 Nov: "these are based off the last printed nodes which was a while ago": the rods and cut points come from the printed commits (parts/overseer_tpu_foot/make.py PRINTED), not the current designs.

## 11 Nov (Spark) on foot nodes v7 (PCTG node + stud + 68D boot block)
- "I dont want this stupid design with the tpu block under, i want a whole redesign it needs to look good and not be tall needs to still fit the funnel, it needs to not crack like the other ones before as well."
- So: no TPU block or boot hanging under the node and no stud for one; low (no taller under the cage than the printed nodes), good-looking (one smooth organic body that reads like the rest of the cage), still seats in the funnel, and survives the ground hit without cracking. Rods stay cut at the old printed nodes (sockets one node length back, as v7).

## 13 Nov (Spark) on v8 as printed
- "the bottom nodes have an empty hole in the middle that shouldnt be the case because support there just rattle around" (sealed cavities: node 3 2.7, node 4 2.4, node 5 8.9 cm3; the engine kept them).
- "heavier is fine but lets rerun the nodes so they are not just filling the holes and calling that it": the engine worker makes no-sealed-void part of the optimisation and reruns nodes 2-5.
