# Desktop dual-board case: cooling and airflow study (job 1116)

STATUS: FINAL (25 Nov 03:30). Build section 0: it is for the layout Spark asked for on 25 Nov (the whole pad under the case, the two sides opening on a lead screw, handle on a centre spine). Section 3 is the earlier layout (fans taken out of the pad and built into the base), kept for reference only.

## 0. The layout to build: whole pad under the case, sides opening

THE OPEN SPACING: 140 mm between the boards' cooler faces (closed 54, so 43 mm of travel per side). Anything from 120 to 160 is within 5 %. Give each side 63 mm of screw travel (a 180 mm gap) so the number can be tuned in firmware.

Air from the pad that goes between the boards (middle fan guess; pad raised 30 mm): closed 54 mm: 30 CFM (14.1 L/s). Open 100: 54. 120: 62. 140: 68 (31.9 L/s). 160: 71. 180: 72. The air speed there is 0.77 m/s closed, 0.72 at 120, 0.67 at 140, 0.56 at 180. With the pad flat on the desk (5 mm under it, E) and the case closed: 13 CFM.

1. PAD: level, 30 mm clear under it all round (a printed dock: four corner risers). 20 mm is the least (94 % of the air); at 5 mm it gives 50 %. A filter sleeve round that gap costs about 11 %.
2. CASE OVER THE PAD: the case bridges the pad on two saddles at the ends of the dock; it does not stand on the mesh. Bottom an open frame: no floor plate, no base, no D0 / D1 fan decks, no feet rails. Board bottom edges 25 mm above the mesh. Centre the case on the big fan (it looks about 25 mm behind the pad's centre in the listing photo; check on arrival). The case is 412 long, the pad 380.
3. FAN POSITIONS (E, read off the listing photo, good to 15 mm): 120 mm fan at the centre of the fan field; the four 70 mm fans at (x, y) = (+-108, +-55) from it. Rule for the spacing: each cooler face stands over the outer third of its two small fans. If the measured y offset is 45 set 120; 55 set 140; 70 set 170.
4. BOARDS: cooler sides facing in, each stood so its fin-outlet edge is the TOP edge. Outer walls solid, 13 mm outside each board; at 140 they stand outside all five fans, so nothing is wasted. Open outer depth 140 + 2 x 36 = 212 (252 at the 180 limit); the pad is 300 deep.
5. LID: each half an open grille (slots 10 wide, 3 mm ribs, lid 4 thick: rib hand calc in section 3 item 7), no lid fan. The 86 mm slot that opens down the middle is the main way out. 150 mm clear above.
6. SPINE: two end posts and the handle beam, 30 wide at most. Nothing standing in the middle of the gap; if a centre post is needed, one up to 40 mm wide straight over the big fan's hub. No full-length plate thicker than 6 mm. Fixed end plates 150 wide on the posts keep the two ends shut when the sides are out.
7. MOTORS AND ELECTRONICS: lead screws, steppers, ESP32, driver and MOSFET boards in the two end spaces beyond the boards (|x| > 170), low down, never over the mesh. Lock both positions so the steppers hold with no current.
8. CLOSED POSE: light work only, about 170 W for both boards at a 10 C air rise (the four small fans are then mostly outside the box). Open carries about 380 W. Pad at full speed for heavy work.
9. PORT WINDOWS cut to the real connectors once the boards are measured. Projector 60 mm or more from the nearest fin outlet. Hub pad: both hubs are 104.9 x 30.0 x 8.9.

Second, independent run (a Fable 5.2 agent given only the sizes): for the earlier closed-box layout it found 4 to 14 CFM as drawn and about 54 CFM reshaped, the intake as the choke and no lid fan, the same picture as sections 2 and 3. A second such run on the open spacing had not come back when this was written.

The illustrated version is `cooling_study.html` beside this file. The model is `tools/desktop_case_airflow.py` (`python tools/desktop_case_airflow.py` prints every number below); the page is built by `tools/desktop_case_cooling_page.py`.

## 1. Exact part sizes

Found in Spark's Amazon order history on 25 Nov. Amazon blocked a direct read of its own pages (an "I am human" check), so each figure was read from copies of the same listing, matched by its Amazon id: Google's index of the Amazon page, the Amazon EU pages, Desertcart and manuals.plus.

| Part | Amazon id | Count | Size from the listing | In mm | Other |
|---|---|---|---|---|---|
| Kootek laptop cooling pad, model LCP04 | B01469DJLM | 1 | 14.96 x 11.81 x 1.38 in | 380 x 300 x 35 | 1.16 kg, metal mesh, USB powered; one 4.72 in fan and four 2.76 in fans (120 mm and 70 mm) |
| HOMPOW mini projector, orange | B0F43Q5B9K | 1 | 4.9 x 4.9 x 2.1 in (EU page: 12.4 x 12.4 x 5.3 cm) | 124.5 x 124.5 x 53.3 | 0.64 kg, built-in cooling fan |
| Acer USB hub, model ACER-UH501 (the "two USB strips") | B0CN3F9Y1Z | 2, both the same | 4.13 x 1.18 x 0.35 in (EU page: 10.5 x 3 x 0.9 cm) | 104.9 x 30.0 x 8.9 | 45 g, ABS body, 2 ft lead, 4 USB 3.0 ports and one USB-C 5 V power input |

Not on any listing, so not confirmed (E): the Kootek fans' thickness, speed, airflow and pressure; the projector's wattage and where its vents are; where the hub's USB-C power port sits. Both boards are still the estimate in make.py (340 x 230 x 20 with the cooler); neither has been measured, and where each board's blower inlets and fin outlets are is not known.

The case file had the pad as 381 x 295 x 33: the listing says 380 x 300 x 35. Nothing in the case changes, because only the fans are used. Both hubs are the same model, so the hub pad's second hub is 104.9 x 30.0 x 8.9 too.

## 2. What is wrong with the case as drawn (make.py at ddc31d48)

Air past the boards, middle fan guess (range over a weak to strong fan guess):

| Layout | CFM | L/s | Air warms by at 270 W |
|---|---|---|---|
| As drawn, fans standing on the floor deck D0 (as the picture shows them) | 5 (4 to 7) | 2.6 | not meaningful |
| As drawn, fans screwed under the upper deck D1 (where their holes are) | 22 (17 to 27) | 10.3 | 22 C |
| Recommended (section 3) | 55 (42 to 68) | 25.9 | 9 C |

1. The two foot rails (y 5..45 and 81..121, full length) lie under the D0 bores. The only way in from below is the 36 x 11 mm channel between them, open at the two x ends: 790 mm2 for fans that sweep 20,300 mm2.
2. The fans are drawn standing on D0 but their screw holes are in D1. On D0 they draw through that channel only, and the three 70 mm fans push air backwards through the weaker 120 mm fan.
3. The 29 base slots are 4,180 mm2. With the fans on D0 they are on the pressure side and leak; with the fans under D1 they are the whole intake, a fifth of the fan area, and the 120 mm fan's inlet is 9 mm off the floor.
4. The D1 bores are round holes under flat board edges 8 mm above: the two 13 mm outer channels get almost nothing.
5. The lid is 10,150 mm2 open (a fifth of it), and the fan under its middle pulls outside air in through the slots beside it.
6. The four port windows (8,640 mm2) let about a third of the air out low on the -x end before it has passed the boards.
7. The 120 mm fan (x 3..123) sits 33 mm under the -x end space; the boards start at x 36.

## 3. The earlier layout, for reference only (fans taken out of the pad; do not build this, build section 0)

Path: in under the floor from all four sides, up through one filter, straight up through the fans, up the gap between the boards into their cooler inlets, out of their fin outlets through collars in the lid, the spare air out of the open lid. More air in than the boards' own coolers take, so the inside is slightly above room pressure and dust only enters through the filter.

1. FEET: drop the six rail segments. Two cross feet along y, each 180 long (y -27..153) and 40 wide, at x 10..50 and x 362..402, 30 mm clear in all (27 PCTG + 3 TPU). The rods at x 12 and x 400 pass through them. Open under the floor on all four sides. Under 20 mm clear costs 7 % or more of the air; over 40 gains nothing.
2. FILTER BOX: a new 15 mm tall layer under D0: a 3 mm filter frame (open rectangle, 10 mm rim, a cross bar under each D0 tile joint) holding one sheet of filter mesh over the whole floor (406 x 120), then four solid box walls 15 tall. The filter must not sit straight on the fan bores: that costs 15 % of the air.
3. D0 IS THE FAN DECK: bores dia 113 for the 120 mm fan and dia 66 for the three 70 mm fans, with the fan screw holes (pitch 105 and 61.5) in D0, fans on top of D0 blowing up. Fan row centred under the boards: 120 at (89, 63), 70s at (192, 63), (270, 63), (348, 63). Under each bore a printed inlet ring with a 6 mm radius lip, held by the fan screws.
4. BASE WALLS SOLID: delete the slots in base_rear and base_end. HB 34 stays (25 fan + 9); it can drop if the fans measure thinner.
5. D1: one window x 36..376, y 6..120 in place of the four bores (a 3 mm lip inside each long wall, so the 13 mm outer channels stay open from below), a 6 mm rib at each tile joint (x 137.3, 274.7) and wherever a board foot lands, solid over both end spaces (x 0..36, 376..412). Rod holes stay. If the rim needs stiffness, deepen it downward (a rib under it), never widen it into the air path. STAND 8 stays (the board edges are then 20 to 30 above the fan outlets). The outer channels get most of their air at the 120 mm fan's end; that is accepted for the boards' cold sides.
6. BOARDS: positions stay (BY 16, gap 54, outer channels 13). Cooler sides toward the gap, and each board stood so its fin-outlet edge is the TOP edge. Only if a measured board has a blower inlet or hot parts on its outer face: BY 21 (gap 44, outer channels 18).
7. LID: no fan (drop FAN70_TOP and FAN_ZONE). Lid 4 thick. Slots 10 wide on a 13 pitch from x 36 to x 376, in three rows each 34 long in y (y 6..40, 46..80, 86..120): 26 x 3 x 340 = 26,500 mm2 free (20,000 is the floor: below that the flow drops). Solid over x < 36 and x > 376. Rib hand calc: 3 wide x 4 deep spanning 34, a 20 N finger push: M 170 N mm, Z 8 mm3, 21 MPa, margin 2.1 on one rib alone.
8. COLLARS (after the boards are measured): over each fin outlet a clear window in the lid with a collar 12 mm down to the fin stack. Keep them 60 mm or more from the +x end (the projector hangs there). If a board also blows out of a side edge, the end space on that side is its chimney: open the lid over it and keep D1 solid under it. Leave 150 mm clear above the lid.
9. PORT WINDOWS: cut to the real connectors once measured, under 4,000 mm2 open in all (8,640 now). Drop the three top slots in bay_end.
10. FIFTH FAN: not fitted. It adds 3 % as a lid extractor and 12 % as a fifth base fan (which needs X 424). Keep it as a spare.
11. HEIGHT: everything from D0 up moves +37 mm (feet 30 + frame 3 + box 15, in place of the 11 mm feet). Closed height about 352.

## 4. Not confirmed, and what it moves

- Fan strength (E: 120 mm 30 to 50 CFM free and 5 to 12 Pa; each 70 mm 10 to 18 CFM and 8 to 18 Pa): moves the result between 42 and 68 CFM. The layout is the same across that range.
- Where each board's blower inlets and fin outlets are: decides the collars, the port windows, which edge stands up, and item 6's alternative.
- Board size 340 x 230 x 20 (E): moves the D1 window and the lid grille with it.
- Heat per board (E: 90 to 180 W): the warming figure scales with it (5 C to 15 C over the whole range in the recommended layout).
- Fan thickness (E: 25 and 15) and the filter's resistance (E: 4 velocity heads; twice that costs 9 %).
