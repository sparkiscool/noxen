# Landing funnel (parts/box_funnel/make.py: frame, end walls, strips, drive): notes

## Spark
- 13 Nov: "do a full optimization run over the whole design and build the arm as well make sure things all are compatable and get it ready to print"
- 10 Nov: both packs' latch sets must fit the funnel meant for them ("yes we need to make the battery fit the funnel do whatever you need"); the Overseer seat moved so both grip points line up.

## Design chat (session_01EtHMZPw3VtLv7n2ybfHWpH), its reading, not Spark's words
- Drawn, never through the engine; its solid check ran out of memory here and is unverified. The end-wall drive is a placeholder.
- The seat and hole were resized to the real latch v7 sets (hole 177 x 57, seat 68.8 wide, CX -0.48): parts/latch7/make.py `--fit` is the check, both packs, keep it clear.
- systems/box.assembly.json (the whole box) still shows the old funnel, latch v5/v6 context meshes and the arm v7 meshes.
