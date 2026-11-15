# Noxen

Drone parts in four stages (Spark 15 Nov: "we don't need generated parts for bench testing or even the first test"): CONCEPT (layout,
simulations, bought parts) -> PROTOTYPE (drawn parts Spark prints the same day) -> TEST (Spark's bench and flights) -> OPTIMISE (the
topology engine in `topo/` lightens a design the tests froze). The page at the repo's GitHub Pages URL shows every part with its stage.

## Commands
```
python parts/<name>/make.py                                       # a drawn part: writes its viewing mesh and print/<name>.stl
python tools/assembly_check.py <drone>                            # interference, fit-table clearances, assembly paths (CI runs it)
python tools/solid_check.py $(python tools/print_mesh.py fetch <name>)   # the print gate (CI runs it), on the print mesh
python -m topo.run parts/<name> --h <SPEC h> --maxit 1 --no-coarse  # OPTIMISE only: the setup audit in a minute (never a full run here)
python tools/dim.py <section>                                     # one section of systems/dimensions.md, never the whole file
python tools/sweep.py [--apply]                                   # the Monday sweep (scheduled task, fresh session): the only housekeeping
python tools/print_queue.py list                                  # print jobs: ready (waiting for Spark), approved, printing, printed
```

## The stages
- CONCEPT: `tools/<drone>_shape.py` holds the layout and every interface (bolts, sockets, rods, keep-outs, each part's assembly path and
  order), the frame model's loads, and the simulations a mechanism needs. Bought parts are chosen here, each with an Amazon listing.
- PROTOTYPE: every new part, or a changed interface, starts as a drawn `make.py` (pattern `parts/scout_v4_probe_bench_turret/make.py`):
  simple generous sections sized by a hand calculation with margin, written in its docstring; every hole, bore, slot, sleeve and press
  fit from the printer fit table (`systems/fit_table.json`, from Spark's coupons; `topo/fit.py`, the audit refuses a setup's hand-drawn one); its assembly path declared. It is print-ready when
  the solid check and the assembly check pass (CI makes, checks and publishes every changed `make.py` on a push: `tools/make_parts.py`): on the release that hour and queued (`tools/print_queue.py add <part> --orient --supports
  --settings --time`; Spark's "print <part>" = the design chat's `approve`, the desktop app prints it). A new interface Spark can't judge by eye gets a small fit coupon first (labelled choices, Spark reports one line, measures nothing).
- TEST: Spark's bench and flight results go to `parts/<part>/feedback.md` and `systems/dimensions.md` (tagged M). A prototype that fails
  goes back to PROTOTYPE at once; a mechanism that works freezes its interfaces.
- OPTIMISE (only parts Spark's tests froze, and only where mass matters): `parts/<name>/part.py`, `setup(h) -> Part` and `SPEC`; its
  drawn attachments are the prototype's tested interfaces (pads, bosses, socket walls, seats, contact faces), every bolt, rod, tool and
  assembly path cut through the whole envelope, everything between generated. Supports are the faces it bolts to; loads are the masses
  at their CG, the system's joint reactions and the handling and assembly cases (screws driven, inserts set, clips bent, a hand's push: `topo/handling.py` adds them for what the shared helpers draw, SPEC `hand_held`),
  each its own case; no guessed loads. Push to main: CI runs the engine (1-5 h), the gates, the assembly check; it replaces the prototype
  on the page (its assembly line's `"prototype": "<make.py folder>"`) only when all pass, and the prototype stays printable until then. A grid over 300k cells at the SPEC h is refused.
- A fail at any stage is a class, not one part (Spark 28 Oct): name the cause, add the check that would have caught it (`topo/audit.py`
  for setups, `tools/assembly_check.py` for fit and assembly, each with a `test_topo` case), apply the fix to every part that shares it.
  The setup audit (`topo/audit.py`, its list in the module's docstring) is fixed in the setup (room, cuts, loads), never by `--set audit=warn`.

## Sessions (Spark 15 Nov: subagents, never new sessions; the design chat stays free and usage stays low)
- The design chat is the session Spark talks to. It answers, decides, writes Spark's notes to `parts/<part>/feedback.md`, does small work itself (a question, a one-file edit) and hands everything that waits (a prototype, a smoke, the test suite, a check-in) to a subagent: the Agent tool in the background with `model: "fable"` (Spark 15 Nov: "run on Fable, and actually solve the problems"), never `create_session` (Spark 15 Nov: new sessions scramble the project). It wakes only for Spark, a subagent's return and its own one check-in per OPTIMISE push.
- One subagent per stream (a part family, a study, an engine change), never one per part, at most 3 running per design chat, one engine change at a time. The prompt names the parts, the stage, the goal and the `feedback.md` files, says to follow this file, and points at files, never pastes them. It commits on main (`git pull --rebase`, its own parts only), never on a branch of its own.
- A subagent takes its parts through the stages alone (Spark 9 Nov: "make these choices yourself") until each is print-ready or blocked on what only Spark has (a test result, a purchase). A prototype is print-ready only when it fits the assembly: every part installable along its declared path in its order, no collisions with anything (`tools/assembly_check.py`; Spark 15 Nov). It solves what it is given (builds, fixes, checks, pushes), never stops at a suggestion or a description; it never loosens a gate (`max_worst`, the solid check, the audit, the assembly check), never changes what a part is for, never calls a part accepted, reports changes as done, never as a question, and never waits: a run in CI ends its turn.
- Its return is at most 10 lines: per part its stage, PASS/FAIL, mass g, worst index or hand-calc margin, what changed and why, the print lines, the artifact link. The design chat then `git pull -q` and sends Spark each new `preview.png` (SendUserFile) with those lines. Nothing reaches Spark in between.

## Runs and check-ins (OPTIMISE)
- The subagent pushes (the push starts the runs; never dispatch after a push, 9 Nov; dispatch only a rerun with no setup change: `gh api -X POST repos/sparkiscool/noxen/actions/workflows/parts.yml/dispatches -f ref=main -f 'inputs[parts]=<names>'`) and returns `python3 tools/predict.py <parts>`'s finish. A running part whose setup changed stops itself and the new one runs.
- The design chat schedules ONE send_later into itself for then ("check runs: <parts>"), never under an hour apart, no polling; that wake only starts a check-in subagent: `git pull -q`, look at each new `preview.png` and the page, a fail goes back a stage, `python tools/build_site.py --inline` and publish `site/<drone>.html` at the `artifact` URL in `systems/<drone>.assembly.json`. Every time Spark reads is Pacific (Seattle).

## Never
- Run a full optimisation in a session; read meshes as text, logs or `inputs/`. Always look at what Spark looks at (Spark 7 Nov): every new `preview.png` and the page rendered with the headless browser, before reporting or judging a shape.
- Edit `topo/` in a part subagent, except the audit check a failure calls for; other engine changes go to an engine subagent (`python -m topo.test_topo`, and `python3 tools/predict.py --engine <tag>` in its commit). Spark never pastes a prompt.
- Send Spark to print a part that hasn't passed the assembly check, or a generated part where a drawn one would do for the test.

## Design rules
- PCTG for every printed part (Spark 8 Nov: 10-20x PETG's impact toughness; PETG settings +10 C; `topo/material.py`, E numbers until a coupon), TPU 95A feet, nothing under 2 mm (one exception: the camera M2 insert posts, Ø5.6 round Ø3.2, 1.2 wall, Spark 29 Oct by coupon), Bambu A1 mini 180 cube, the largest flat face down, layers along the part's bending wherever a load path allows.
- 3 mm and 6 mm carbon only; sockets 3.2 / 6.2 from the fit table, each with a Ø2.5 resin port (one 14 G blunt tip) 1.5 above its floor out between the members and three 0.3 resin pockets along the bore (`cage_node_lib.port_cut` / `socket_cuts(pockets=3)`, Spark 5-8 Nov); socket depth from its load (the G/flex rule in the v4 node lib).
- Numbers come from `systems/dimensions.md`, tagged M measured, C CAD, E estimate. Every bought part is proposed with an Amazon listing (Spark 29 Oct); a distributor-only part is the fallback, never the proposal.

## Changing the workflow (Spark, 28 Oct)
A new rule is merged into the process, never stacked on it: code first (an audit check, an assembly check, a step in `parts.yml`), else a change to an existing line here, a new line only when neither fits. Every addition removes what it supersedes. This file stays under 60 lines. No rulebooks, no docs folder, no agents or skills that restate rules.
