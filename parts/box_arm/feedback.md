# Swap arm (the box's arm, its heads and grabbers, the bay receivers): notes

## Spark
- 13 Nov: "do a full optimization run over the whole design and build the arm as well make sure things all are compatable and get it ready to print"
- 9 Nov: error tolerance at every swap step ("it should be possible to complete the swap with little issues in every situation"); forces even, nothing tilting and rubbing; "use the actual parts we have" (servo and motor specs are in the repo); every important mechanism as big as makes sense.
- 10 Nov: "is this design the best for error tolerance? i want something that will still work in most senarios"
- 8 Nov: PCTG for every printed part. 29 Oct: every bought part on Amazon.

## Design chat (session_01EtHMZPw3VtLv7n2ybfHWpH), its reading, not Spark's words
- What exists: arm v7 as meshes only (parts/box_arm_*, PETG, imported by tools/import_box.py; never built, state DESIGN) and its numbers (`python tools/dim.py "System box swap: arm"`, section I). Its scripts are in the older repo (add_repo jy202108g-bot/repo1, read only: nox/box/box-arm.md, swap_system.py, arm_v7_layout.py, arm_v7_cad.py, head_v7_concept.py, detent_v6.py, hub_v6.py, runs/arm/v7/). Its heads were for latch v4: no grabber exists for latch v7.
- The grabber is built to parts/latch7/make.py and never the other way: `blade()` (4.7 x 34 flat-topped, two hooks 25 apart), `heel()` and `pads_of()` (pads, Hall posts under the bolts' magnets), 25 N per blade (`F_BLADE`), one head for both packs (both tab lines sit on the funnel centre). parts/latch7/result.json and parts/latch7_overseer/result.json carry what the latch asks: `audit.error_budget`, `audit.parts`, `slant`, `grabber_interface`, `hold_strong`.
- Asked of the head (offered to Spark 10 Nov, he said go on): locating features that take about 3 mm across and along before the blades rise; give in tilt and roll (the latch takes 0.6 deg roll and 1.6 deg pitch rigid; the arm's worst is 0.74 / 1.28, and the Overseer receiver cannot hold more than 0.65 roll); adjustable heel pads; each bolt read by its Hall sensor before the drone is released; more push: the stepper stalls at 74 N and the insert is 57 N of plug (E) plus the pack's weight (7.5 N Overseer): at least x1.5 over that.
- Servos: DS3218 20 kg cm on hand, MG995 9.2 ordered (dimensions M).
- Clear of the funnel on the whole swap path, both packs (parts/latch7/make.py `funnel_fit`), and of every bay on the ring (systems/box.assembly.json).

## Arm worker (13 Nov), its reading, not Spark's words
- Arm v8 is drawn in parts/box_arm8/make.py (shared geometry parts/box_arm8_lib.py, setups parts/box_arm8_setups.py), system file systems/swap_arm.assembly.json, numbers parts/box_arm8/result.json. The v7 scripts repo (jy202108g-bot/repo1) could not be added to the session (refused): v8 is built from the v7 meshes and dimensions.md, not from the v7 scripts.
- State: swap cycle 502 states, both packs, 0 overlaps. Not ready to print: the hub, carriage and lift setups fail the setup audit (result.json audit_smoke) and two drawn pieces fail the solid check (solid_check).
- Open, not the arm's to decide: the plug is at the pack's far end from the rail (latch v7 to_seat: lead end = nose); the 60 / 120 deg bays; the stand's keel under the -90 deg bay; the receivers have no mounting features for a bay; a hold-down in the funnel for the 118 N push limit (v7: 74).

## Design chat, 13 Nov, on the arm worker's first report (v8 drawn, 1e4e329)
- The v7 system pass's loads and scripts are now in parts/box_arm/ref/ (the repo could not be added from the worker): the rail, V-wheel and hub frame loads come from arm_v7_system.json, not hand numbers; the missing cases (axle stall, dropped pack, indexing) are in swap_system.py.
- Bay receivers carry no landing feet (the feet are for the drone's receiver in the funnel). With that, the two slots beside the drone (60 and 120 deg) take Scout bays: 11 bays, every slot but the drone's (Spark 8 Nov: fill the wheel without getting in the way of the landing), at least 4 of them Overseer.
- A screw that takes 118 N gets a thrust bearing, not the stepper's own bearings.
- The report listed fails (three setups refused by the audit, unit body and detent arm on a neck): those are the loop, not a report. Carry on until every setup passes the audit at its SPEC h and every drawn print passes the solid check; rerun the 502 states against the latch as it is now (1c5f9a1 moved the Scout's feet and the Overseer's side walls).
