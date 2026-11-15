# Archive

Out of the build: nothing here is read by CI, the page or any tool. Kept so the meshes stay findable without digging
through history. Move a folder back to `parts/` (and give it an assembly line) to use it again.

- `parts/`: mesh-only folders that no assembly, tool, lib or data file referenced on 9 Nov (old box imports, Scout v3
  and scan-stick meshes, superseded latch pieces, node 17). Folders with a `make.py` and coupons the libs cite stayed in `parts/`.
- `tools/weld_stl.py`: called by nothing.
- Scout v3 (9 Nov, Spark: only the newest Scout; `systems/scout.assembly.json` is `archived`, off the page): its generated parts
  (`parts/scout_cage_node_*` x 15, `scout_arm_clip_1..4`, `scout_nose_unit`, `scout_tail_unit`: all stale, so a blanket dispatch
  would have rerun them), its mesh-only folders (`scout_cage_v21`, `scout_prop_1..4`, `camera`, `frame_plate_bee35_carbon`,
  `pack_4s_1500`, `pack_xt60_plug`), `tools/bee35_plate.py` and `dimensions_scout_v3.md` (its section of systems/dimensions.md).
  Their print meshes stay downloadable from the `meshes` release (`<part>.stl`). Kept live, because Scout v4 reads
  them: `systems/scout.assembly.json` (tools/scout_cage_loads.py's masses), `systems/scout.cage_loads.json` (imported by
  `parts/scout_node_lib.py` / `scout_units_lib.py`, which the v4 libs import), `tools/scout_cage.py` and `scout_cage_loads.py` (the
  v4 libs and tools/scout_v4_shape.py import them), `tools/scout_meshes.py` and the bought meshes tools/scout_v4_shape.py places
  (`scout_motor_1..4`, `scout_flow_mtf_02p`, `fc`, `radxa_zero_3w`, `ubec`, `xr2`). The box's seated copies (`parts/box_scout_*`)
  followed on 15 Nov (below).

A sweep moves a folder here when it is mesh-only and nothing draws, names or regenerates it (tools/sweep.py says how it decides).

15 Nov sweep (tools/sweep.py now keeps a folder only by what draws, names or regenerates it; each group checked by hand):
- Box, superseded by what the box page now draws (tools/import_box.py): `parts/box_arm_*` x 55, the arm v7 import of 3 Nov
  and its 8 Nov bays (systems/swap_arm.assembly.json, arm v8, replaced it; import_box drops `box_arm_` lines, only a re-import
  from the nox repo read `box_arm_rec0`); `parts/box_scout_*` x 17, the seated Scout v3 copies; `parts/box_overseer_*` x 18,
  seated Overseer copies (the page seats both drones from their own folders, its `box_overseer_<id>` lines name the drone's
  mesh); `parts/box_ctx_*` x 4, the 3 Nov context latches (latch v7's `box_latch7_*` replaced them); `parts/box_stand_bara`,
  `barb`, `cradle_keel`, `kb`, `toe`, the v1 demo stand the stand v2 replaced (`parts/box_stand_mast` stays: its check.py
  holds the arm's load cases stand v2 uses).
- Latch, superseded generations: `parts/latch5_*` x 12 and `parts/latch6_*` x 13 (what latch v5 and v6's make.py wrote; v7
  replaced both, their systems are archived), `parts/latch7_rib` (v7 before the 12 Nov stop bar; latch7/make.py no longer
  writes it), and the 30 Oct latch copies `parts/cartridge`, `receiver`, `pack`, `cartridge_v6_latch`, `receiver_v3_latch`.
  `systems/latch5.assembly.json` (archived, nothing writes or reads it) and its timeline `tools/latch5_motion.py` are in
  `systems/` and `tools/` here. `parts/latch5/` and `parts/latch6/` stay: latch v7 imports their make.py and reads
  latch6's result.json, and `latch7/make.py --appendix` folds `systems/latch6.appendix.html` into v7's page.
- Overseer: `parts/overseer_cage_node_0`, `1`, `8`, `9`, `12`, `14`, `16`, `19`, the 30 Oct meshes of nodes that are now mirrors
  (systems/overseer.assembly.json draws each from its `part`: 0 from 4, 1 from 3, 8 from 6, 9 from 10, 12 from 11, 14 from
  13, 16 from 15, 19 from 18).
