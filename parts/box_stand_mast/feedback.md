# Box stand (the built demo's mast, kb, bars, cradle keel, toe): notes

## Spark
- 13 Nov, asked in the design chat "The larger arm ring no longer clears your printed demo stand. Which change to the stand do you want?": "Hub 15 mm higher (Recommended)" = reprint the mast 15 mm longer and saw the cradle upstand off the printed keel.

## Design chat (session_01EtHMZPw3VtLv7n2ybfHWpH), 14 Nov: done here because three workers' permission checks refused the stand
- Geometry, in W (hub at the origin, so arm, funnel and seat do not move): the mast is 15 longer, a prism of its own 384 mm2 section let in at W Z -49 (between the kb's top at -52 and the stub's foot at -45): 187 long, 83.8 cm3 (was 172, 78.0). The keel is the printed one (git 33cd639) less everything above its bar's top (W Z -181 as printed): 36.7 cm3 (printed 55.2). The toe is the printed one, uncut (41.5 cm3). kb, both bars, keel and toe sit 15 lower in W (pose z -15); the floor is W Z -216.
- These three part.stl files are that geometry (mast in W; keel and toe in their printed coordinates, to be posed z -15). NOT yet in systems/box.assembly.json or tools/import_box.py on main (their STAND_NOTE, stand_versions and recut_stand still describe the cut keel): a push of systems/ or tools/ restarts every part with no result, so it waits for the runs to land.
- Checked: tools/box_check.py full run with this stand in place (local, 14 Nov): PASS, 0 overlaps. The -90 deg bay's lowest point is W Z -192.9, the keel bar's top -196: 3.1 clear; floor 23.1 under the bays.
- Print: the mast no longer stands on the 180 bed (187). File in print pose on its 60 x 187 side along the bed diagonal (160.5 x 160.5 x 60), 7,385 mm2 on the bed, solid check PASS; the tube's inner ceiling (7,163 mm2) bridges or takes support.
- NOT checked: the stand's strength with the arm v8 on it (the arm, its 11 bays with packs, the 118 N push at W Y 192). The keel bar without its cradle has the 580 mm3 section it already had over W Y 107..137; whether that carries the new overhung weight is open. Do not call the stand done.
