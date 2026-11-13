# Landing funnel (parts/box_funnel/make.py: frame, end walls, strips, drive): notes

## Spark
- 13 Nov: "do a full optimization run over the whole design and build the arm as well make sure things all are compatable and get it ready to print"
- 10 Nov: both packs' latch sets must fit the funnel meant for them ("yes we need to make the battery fit the funnel do whatever you need"); the Overseer seat moved so both grip points line up.

## Design chat (session_01EtHMZPw3VtLv7n2ybfHWpH), its reading, not Spark's words
- Drawn, never through the engine; its solid check ran out of memory here and is unverified. The end-wall drive is a placeholder.
- The seat and hole were resized to the real latch v7 sets (hole 177 x 57, seat 68.8 wide, CX -0.48): parts/latch7/make.py `--fit` is the check, both packs, keep it clear.
- systems/box.assembly.json (the whole box) still shows the old funnel, latch v5/v6 context meshes and the arm v7 meshes.

## Design chat, 13 Nov: state handed to a fresh funnel worker (the first one stopped after three poses of the design in one context)
- Done and passing its checks: funnel v5 (two engine halves on four cross arms, lead-screw wall drive, four steel hold-down keys on the end walls into the receivers' end notches), the turned pose (tools/seat_check.py SEAT_TO_W, one definition: both drones nose toward the hub side), tools/box_check.py PASS with 0 overlaps against arm v8 (it reads parts/box_arm8/make.py funnel_keepout(): W X -36..69, Y 54..79, nothing below Z 161.5) and latch v7. Audit smokes at SPEC h: box_funnel_arm 0.241, box_funnel_half_tail 0.381.
- NOT passing: box_funnel_half_nose, audit 0.60 (limit 0.5), case "Overseer off centre onto the +y face, at its ear", peak at seat (57.9, 58.3, -32.6): the rail's section just inboard of the nut block's lane (the lane, |y| 46..58, h 0..12, leaves the 3 mm wall as the only web between the rib above and the room under the ledge). The tail half has the same section and passes; what differs is not found (nose rib runs to x 88, tail's to 76; the nose beam is 46 further from its ear).
- Tried, did not move it: a wider link from the ear to the nose beam (0.61 -> 0.59), a 15 mm deep run outside the arm's head from the ear to the beam (0.59), closing a 0.6 fin at the motor plate (0.60).
- Not tried: the audit's stress field for that case in both halves side by side (look before changing anything more); a web outboard of the block lane on the -y side only (no switch bar there), dropping the mirror symmetry; the nose beam as two stubs over the rail disc (seat x 114..138) with its own arm variant.
- Open besides: the stand's keel and toe are cut back from the -90 deg bay as geometry only (strength unchecked; Spark's printed demo keel would need the same cut); the enclosure sheets need 8 holes o 4.5 at W Z 121.3; the M4 screw shafts are cut to 62 (Spark does not cut metal if a part to length exists: look for one); the topo suite's slender-arms test fails the same before and after the audit.py change (an engine matter, reported to Spark).
