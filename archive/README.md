# Archive

Out of the build: nothing here is read by CI, the page or any tool. Kept so the meshes stay findable without digging
through history. Move a folder back to `parts/` (and give it an assembly line) to use it again.

- `parts/`: mesh-only folders that no assembly, tool, lib or data file referenced on 9 Nov (old box imports, Scout v3
  and scan-stick meshes, superseded latch pieces, node 17). Folders with a `make.py` and coupons the libs cite stayed in `parts/`.
- `tools/weld_stl.py`: called by nothing.

A sweep moves a folder here when it is mesh-only, in no `systems/*.assembly.json` (by `id`, `part` or `mesh`) and named by no code.
