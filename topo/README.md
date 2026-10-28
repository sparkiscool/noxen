# nox/topo: the Nox topology engine

Written 22 Oct 2026 to replace `noxopt.py`, `noxopt_local.py` and `printopt.py` for generated parts. The old files stay in the repo until the demos below have been compared; nothing in `nox/topo` imports them.

## What it does differently

| Problem with the old engine | What nox/topo does |
|---|---|
| Pads hung on 2 to 3 mm necks (bracket_e v8). The 0.9 mm erosion check passed them, and the neck test found them only after the build. | **Drawn-masked eroded filter.** The eroded design filters the design density over the non-drawn cells only, so a member is not propped up by the drawn body it meets. A wide root keeps its own density up to the drawn face; a thin neck is averaged with the empty space round it, erodes, and every load through it blows up, so the optimiser widens the root during the run. Drawn bodies stay solid and loads stay where they are. Unit test `t_masked_neck`: a 2 mm neck along a cut reads 3.2e3 x the eroded compliance of a full-width root. |
| Thin drawn plates the solid gate erodes away (bracket_jA: a 4 mm wall flange at the gate's 2.0 mm erosion) linked parts of the design that the gate then splits. | **Gate ties.** A held drawn body thinner than the gate's erosion counts in the eroded design only where generated material covers it, and a second tie set holds the largest unheld body instead of the support, so the other bodies must join it through generated material. Unit test `t_gate_ties`. |
| The masked filter treated keep-outs as "outside", so material flush against a cut was never eroded. Thin sheets and necks along cuts survived. | Constant-normalised cone filter. Every void and the grid edge count as empty, so the length scale holds against keep-outs too. |
| The length scale was set by hand per part (R in cells, floor_mm, prune_thin passes). | **Minimum member in mm** (default 3.2 mm rods, about 3.0 mm plates). The filter radius comes from a 3D calibration of the robust thresholds (`layout.calibration`), and `gate_member_mm` raises it to what the solid gate's erosion needs (2 x 2.0 mm erosion + 2.25 mm margin = 6.25 mm on the demos). There is no prune pass. |
| Unreachable setups (an access cut wider than its pad, an uncovered boss) ran for hours before failing. | **Reachability check** before the first solve. Every drawn body must be joinable by a member of the minimum size. If not, the run raises with the body's position. |
| Ties were translations only, at weight 0.25. | Ties use 3 forces plus 3 moments per unheld body, weight 0.5, on the eroded design. |
| Pardiso OOM above about 60k design cells. The separate gmg file had its own Layout. | One `Solver`. Pardiso runs on the active dofs only, and a Galerkin geometric multigrid CG takes over beyond the RAM budget. Coarse levels are built element by element and exactly. It warm-starts and solves all load cases of a support group together. |
| Polish flow, prune loops and reinforcement balls at build. | The surface is the eta_n iso-surface of the filtered field. It is smooth by construction and blends into drawn bodies with a fillet of the filter radius. It is held 0.5 mm off keep-outs and cut exactly with manifold3d. |
| Limit-cycling volume control at gain 1.0. | Damped volume-limit update (gain 0.3). |

The gate is unchanged: `nox/solid_check.py`, run on the exact built mesh.

## Formulation

- **Fields.** The design variable x lives on the design cells. x_hat is x there, 1 on drawn cells and 0 on voids. The constant-normalised cone filter gives x_tilde (voids and the grid outside count as empty). Heaviside projections at eta 0.5 (nominal or blueprint) and 0.25 (dilated) act on x_tilde (Wang, Lazarov & Sigmund 2011).
- **Eroded design (`erosion="masked"`, default).** x_tilde_e = conv(x on design, 0 elsewhere) / max(1 - conv(drawn), 0.05), projected at eta 0.75. Voids and the grid outside count as empty in numerator and denominator, drawn cells are left out of both. Drawn cells stay 1, loads stay on their nodes. The sensitivity is the filter transpose with the 1/den weight. Options: `"anchored"` (round 1: only drawn cores solid, skins eroded, loads moved to the core; it isolated thin drawn plates next to cuts) and `"frozen"` (drawn cells 1, eroded field = x_tilde).
- **Gate ties (`gate_ties=True`).** A drawn body that touches the main support and has a half thickness under `gate_thin_mm` (2.0, the solid gate's erosion cap) counts, in a separate gate realization used by the gate ties only, by its cover, x_tilde_c = conv(x on design) / max(conv(design), 0.05), projected at eta 0.75. A second tie set (3 forces + 3 moments per body, weight 0.5) holds the largest unheld body (support `_gate`) and loads every other unheld body. The real cases and the ordinary ties see the body whole, so it stays attached (crediting the cover in every case let the holder smoke test drop a held boss). Active only when such a thin held body and at least two unheld bodies exist (bracket_jA: 18 gate ties; bar_half, motor_mount: none).
- **Held ties (`held_ties=True`).** Two or more bodies held by the main support (bolt bosses on their seats) need no material to stay put, so a boss no load case leans on was left loose (holder smoke run). A third tie set holds one hub body (the largest unheld body, else the largest held one; support `_hub`) and loads every other held body with 3 forces + 3 moments, weight 0.5 (holder: 24 held ties; motor_mount: 24).
- **Objective.** The p-mean (P = 6) of c_i(eroded) / c_i,ref over the real load cases, the tie cases and the gate ties. c_i,ref is the full solid's compliance (`case_norm="solid"`, default). Against it, concentrated loads and ties swamp the distributed cases (holder organic-v7: each misuse case 1e-4 of the objective while the worst as-built index sat in misuse 10 / 12). `case_norm="frame"` renormalises every case on the first frame the run reaches; it did not lower the worst index (organic-v8 0.596 vs v7 0.569), so it is an option, off. SIMP p = 3, Emin 1e-6. Body forces (`accel`) are density-weighted and carry their sensitivities.
- **Constraints.** Dilated volume <= V*, with V* adapted (gain 0.3, reset to its fixed point at every beta step) so the blueprint meets the mass target.
- **Geometric constraints (Zhou 2015): off by default** (`geometric=True` turns them on from beta 16). Measured on bar_half topo-v1 at beta 32: the solid one read +294 because all 977 cells it counted sat at x = 1, one to two cells off a cut, where the constant-normalised filter cannot reach eta_e at all; bracket_jA topo-v1 read +6, 97 % of it from cells next to drawn bodies. Both stayed constant for 25+ iterations while MMA spent its moves on them. They now count only cells where they can be met (`_geo_masks`: x_tilde with the design all 1 reaches eta_e + 0.05 for solid, x_tilde with the design all 0 stays under eta_d - 0.05 for void). With that mask the same bar_half density reads 0.002 eps (solid) and motor_mount topo-v2 0.009 eps: the robust formulation already holds the solid length scale, so the constraints add cost and no length scale the gate needs.
- **Optimiser.** MMA (Svanberg 2007 asymptotes, dual solved by L-BFGS-B). Beta runs 1, 2, 4 (coarse grid), then 4, 8, 16, 32 (fine grid). A stage advances when it has settled. The run stops at <= 2 % grey or after max_stage iterations at beta 32.
- **Build.** The surface is the eta_n iso-surface of x_tilde (cubic upsampling, marching cubes on a field whose exact-zero samples are nudged outside, Taubin), held 0.5 mm off keep-outs, then (generated + drawn) ^ env - cuts exactly in manifold3d. Specks under 20 mm3 are dropped; sealed cavities (a hollow section in the nominal field) stay.
- **Material.** PETG, PLA or TPU, transversely isotropic about the build axis. The as-built strength report uses in-layer von Mises / A_xy and across-layer sqrt((sigma_n+/A_z)^2 + (tau_n/A_s)^2), with the same allowables as the old engine so indices compare.
- **Voxels.** Inside tests cast one +z ray per column from below the mesh (not below the grid: a keep-out reaching under the grid, as the stand setups' ring and half spaces do, inverted whole columns in round 1). A support or load region that holds no cell centre falls back to the cells it intersects (3 x 3 x 3 samples per cell), then to the part cells touching those.

## Compute plan (4 vCPU / 16 GB cloud box, 6 h cap)

- **Stage 1 (2h grid).** Beta 1 to 4 run on a grid twice as coarse, with 8x fewer cells. Most iterations happen here.
- **Stage 2 (h grid).** Warm-started from stage 1. It runs from beta 4 to the end.
- **Solver choice.** Pardiso is used while the estimated factor fits 60 % of 75 % of RAM, about 0.5 M active dofs. Above that the run uses multigrid CG: Jacobi(0.6) x2, 2:1 levels down to <= 40k dofs, Pardiso on the coarsest level, tolerance 1e-5, warm starts.
- **Time limit.** Past 75 % of `time_limit_s` the beta steps are taken as soon as allowed. Checkpoints are written every 10 iterations, and `--git` commits and pushes each one (CLAUDE.md, Heavy runs).
- **Grid size.** h defaults to about the min member / 3, the nTop rule for element size. The filter radius is then about 2.4 to 3 cells.

## Use

```bash
cd nox
python3 -m topo.test_topo            # self-test, ~3 min
python3 -m topo.run holder --git     # a demo part end to end -> runs/holder/topo-v1/
python3 -m topo.run bar_half --out ../runs/bar_half/topo-v3 --rebuild-from ../runs/bar_half/topo-v2   # build stage only
python3 -m topo.run bracket_jA --set pmean=8 --set gate_ties=false                                   # any Settings field
python3 -m topo.compare bracket_jA --new ../runs/bracket_jA/topo-v3/bracket_jA.stl   # old v8 part vs new, same grid
```

`topo.compare` maps the old print file (`print/box_<part>_v8.stl`) back to W with the inverse of the repo's print-pose rotation (`stand_v7_parts.print_pose`; bar_half 45.12 deg from `stand_v8_bar_export.py`) and a translation fitted by overlaying the drawn bodies (voxel cross-correlation, then Nelder-Mead on exact volumes; both v8 parts fit with 0.0 mm3 drawn missing, in cuts or outside the envelope). Then it runs `build.strength` for both parts on the same Part grid and load cases and writes `compare.json` beside the new part.

From a part script:

```python
from topo import Part, Settings, optimise
def setup(h):
    p = Part("my_part", h, drawn, cuts, env, build="+z")
    p.support("default", region); p.load("case", region, (fx, fy, fz), cls="misuse")
    return p
optimise(setup, h=1.0, gen_cm3=12, out_dir="runs/my_part/v1", name="my_part", settings=Settings(min_member_mm=3.2), git=True)
```

The old `arm_topo.PartProblem` setups run unchanged. See `demos/stand.py`.

## Demos

Results per part and round: `runs/topo-demo-summary.md` (round 1), `runs/topo-dev-round2.md` (round 2). The round 2 column was made with `finish="drawn"` (the engine before the organic default); organic runs are listed under Finish below.

| Name | What | Round 2 |
|---|---|---|
| `holder` | The old engine's benchmark holder (28 cases, 3 supports); the organic look test part since 23 Oct. Setup fixed for the gate (11.5 mm offset bosses, 7 mm layer under the pad; `demos/holder.py`). | organic-v11: gate PASS, 11.73 g, worst 0.493 (topo-v1: 10.26 g, 0.498, gate FAIL). `runs/holder/organic-summary.md`. |
| `bracket_e` | The bracket whose v8 bolt pads hung on necks. | Not run. Its coarse stage now builds (support band falls back to touching cells). |
| `bracket_jA` | Box stand joint bracket the old v8 shipped. | topo-v4: gate PASS, 36.02 g, worst 0.124 (v8: 38.37 g, 0.133, same grid). Organic, final engine: organic-v5, PASS, 36.85 g, 0.127. |
| `bar_half` | Box stand cross bar half the old v8 shipped. | topo-v3: gate PASS, 71.23 g, worst 0.804 (v8: 77.58 g, 0.929, same grid). |
| `motor_mount` | A generic drone motor mount (assumed dimensions, E), pads 6 mm tall. | topo-v3: gate PASS, 17.21 g, worst 0.117 (v2 with 5 mm pads: FAIL by 4 voxels). Organic, final engine: organic-v2, PASS, 18.03 g, 0.122. |

## Sources

- Wang, Lazarov, Sigmund 2011, SMO 43:767: the robust formulation.
- Trillet, Duysinx, Fernández 2021, arXiv 2101.08605: analytical length scales.
- Zhou, Lazarov, Wang, Sigmund 2015, CMAME 293:266: geometric constraints.
- Clausen & Andreassen 2017, SMO: filter boundary conditions and eroded designs detaching at boundaries.
- Cool, Aage, Sigmund 2025, arXiv 2501.09402: connectivity review.
- Development rounds: `runs/topo-demo-summary.md`, `runs/topo-dev-round2.md`.
- Amir, Aage, Lazarov 2014 and Ferrari & Sigmund 2020: multigrid CG for TO.
- Svanberg 1987 / 2007: MMA.
- Research notes behind the choices: `claude/research/topo-engine-2026-10-22.md`.

## Finish: organic by default (23 Oct)

Spark's rule: the whole part is smooth; only flat contact faces and mounting holes are geometric. The rule lives in the
engine, not in the part scripts, so every part gets it the same way (`Settings.finish`, default `"organic"`):

- **Layout.** A drawn body stays passive only as a functional skin: its cells within `skin_mm` (2.5) of a cut (holes,
  mating and contact faces, keep-outs) or of the outside of the envelope as given. A drawn body touching neither stays
  whole. A body the main support holds (bolt pads, bosses, a wall flange: the mounting features) stays whole: opening
  part of a bolt pad left its envelope-side skin with a core of its own at the gate's 2.0 mm erosion (motor_mount
  organic-v1, pad cores 0.75 in the main body, FAIL). The rest of every drawn body becomes design space, so blocky drawn plates and bosses turn into optimised,
  smooth material.
- **Build.** One smooth field, all signed distances redistanced by fast marching from continuous fields (scikit-fmm),
  so no surface carries voxel steps: the generated region (eta_n level set of the cubic-upsampled filtered field) is
  held 0.5 mm off keep-outs with a rounded `fillet_mm` blend (not a crease), the skins' free rims are rounded by
  `skin_round_mm` (1.0, an opening), and the two are smooth-unioned (blend `fillet_mm`, 2.5). The skins run 1 mm into
  the cuts and past the envelope, then `field ^ env - cuts` cuts the holes and contact faces exactly. Nothing else is
  exact or flat.
- **Function check** measures the skins (present within 1 % at the build pitch, less the free rim within
  `skin_round_mm` + 1 voxel of the design side that the build rounds on purpose) instead of whole drawn bodies; the
  solid gate is unchanged.

`--set finish=drawn` restores the v1 behaviour (drawn bodies whole and exact). The mass target `--gen` means the same in
both finishes (volume added to the drawn bodies as given): the layout adds the drawn volume it opened to the design, so
every part keeps its v1 number (bracket_jA 12 cm3; 20 cm3 filled 85 % of the design space, organic-v1).
