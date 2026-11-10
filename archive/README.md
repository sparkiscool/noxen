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
  Their print meshes stay downloadable from the `meshes` release (`<part>.stl`). Kept live, because Scout v4 or the box reads
  them: `systems/scout.assembly.json` (tools/import_box.py's seated Scout), `systems/scout.cage_loads.json` (imported by
  `parts/scout_node_lib.py` / `scout_units_lib.py`, which the v4 libs import), `tools/scout_cage.py` and `scout_cage_loads.py` (the
  v4 libs and tools/scout_v4_shape.py import them), `tools/scout_meshes.py` and the bought meshes tools/scout_v4_shape.py places
  (`scout_motor_1..4`, `scout_flow_mtf_02p`, `fc`, `radxa_zero_3w`, `ubec`, `xr2`). The box keeps its own seated copies
  (`parts/box_scout_*`).

A sweep moves a folder here when it is mesh-only, in no `systems/*.assembly.json` (by `id`, `part` or `mesh`) and named by no code.
