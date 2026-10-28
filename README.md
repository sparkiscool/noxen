# Noxen

One script per part, GitHub runs the topology engine, one page shows the assembly. Claude writes scripts, nothing else.

## First time (10 minutes)
1. Create an empty **public** repo on GitHub (public = free 4-core / 16 GB runners, unlimited minutes, 20 at once).
2. Push this folder to it as `main`.
3. Repo Settings -> Pages -> Source: **GitHub Actions**.
4. Repo Settings -> Actions -> General -> Workflow permissions: **Read and write**.
5. The push already started `parts` for the three test parts (motor_mount, holder, hub_deck). Watch the Actions tab.
   When they finish, the `site` workflow publishes the page; its URL is printed in that run and in Settings -> Pages.

## Every part after that
1. `claude` on your laptop in this folder: "part <name> v1: <what it mounts, what it carries>".
2. Claude writes `parts/<name>/part.py` and a line in `systems/<drone>.assembly.json`, pushes, stops.
3. Runner optimises, solid-checks, commits `result.json` + `part.stl`. Page rebuilds. Grey box becomes the real part.
4. Red on the page = failed. `claude "<name> failed"`; it reads `result.json` and fixes the setup.

Manual run of any part(s): Actions -> parts -> Run workflow -> names. Empty = everything missing or stale.

## What the runner does when it is cut off
Runner cap is 6 h. The engine checkpoints every 10 iterations; a job that hits the cap saves the checkpoint and
starts itself again for that part, up to 4 attempts (about 22 h). A crash is not retried: it shows red and needs a fix.

## Layout
```
topo/                  engine (unchanged from nox/topo except: parts/ path loader, solid_check path)
tools/solid_check.py   the gate      tools/build_site.py  page builder      tools/stale_parts.py
parts/<name>/part.py   the setup; CI adds result.json, part.stl, preview.png
systems/*.assembly.json  every drone: part ids, placeholder shapes, poses       systems/dimensions.md  the numbers
site/index.html        the viewer (three.js, reads site/assembly.json that CI writes)
```
Status on the page is computed, never stored: `result.json` present and its `part_sha` matches `part.py` -> done or failed; mismatch -> stale; missing -> pending.
