# Nox: parts list (22 Oct 2026)

Every part of the box, the Scout and the Overseer: what it is, how many, where it stands. One list, so a new mount or an order starts here. **Numbers are not repeated here:** body sizes and mounting holes are in `nox/dimensions.md` (section M for bought parts, the other sections for our own geometry), and the live state of every printed part is only in `nox/current-parts.md`. Sources for section M: `claude/research/2026-10-22-mounting-dimensions.md`.

**Status words:** **on hand** (in the docs as owned or built in); **ordered** (in the 15 Oct cart, arrival not recorded: CHECK); **to buy** (needed by a current design, not ordered); **priced** (on the later list, not needed yet); **CHECK** (the docs do not say). Scout v2 printed parts are retired (`archive/scout-v2/`); Scout v3 has no printed parts yet.

## Scout (aircraft; carried into v3)
| Part | Qty | Role | Status | Mount data |
|---|---|---|---|---|
| SpeedyBee Bee35 frame, aluminium bottom plate (Pro), ducts off | 1 | airframe, 153 wheelbase | on hand | M: Bee35 |
| EMAX ECO II 2004 motor (KV not recorded) | 4 | propulsion | on hand | M: ECO II 2004 |
| Gemfan Hurricane 3520 prop, 3 blade, T-mount | 4 + spares | propulsion, props on top | on hand | M: 3520 |
| MicoAir743v2-AIO-35A (FC + 35 A ESC) | 1 | flight controller | on hand | M, H |
| Grommets for the FC | 4 | soft mount (Spark 14 Oct) | CHECK (size not published) | M |
| RadioMaster XR2 (Nano) ELRS 2.4 | 1 | RC receiver, on the GPS port | on hand | M |
| MicoAir MTF-02P | 1 | optical flow + ToF | on hand | M, H |
| Radxa Zero 3W, RS107-D2E0H0W15 (2 GB, no eMMC, no header) | 1 | companion computer | on hand | M, H |
| microSD ≥ 32 GB A1 | 1 | Radxa boot | CHECK | |
| DECXIN 2633V2 USB global-shutter camera, 102° | 1 | camera | on hand | A (P); measure lens |
| M2.5 heat-set inserts for the camera (OD 3.5 x 5.7 E, Ø3.2 hole) | 4 | camera mount (Spark 23 Oct: the board takes M2.5) | CHECK | A, M |
| Camera mount v1, printed PETG (`print/scout_camera_mount_v1.stl`) | 1 | cradle, lens hood, 4 snap clips on cage tubes 7-8 / 14-15 | to print | System Scout v3 Camera |
| Camera mount stop collar, printed PETG (`print/scout_camera_mount_collar_v1.stl`) | 2 | y stop on tube 14-15, CA'd | to print | System Scout v3 Camera |
| M2.5 x 8 screws + bought M2.5 nylon or silicone washers, 1.0 thick (E), about Ø6 / Ø2.7 (E) | 4 + 4 | camera board onto the mount pads | CHECK | System Scout v3 Camera |
| 4S 1500 LiPo, XT60 | 1 + CHECK spares | flight pack | on hand | A (P) |
| 220 µF 35 V capacitor | 1 | on the battery pads | on hand | |
| 5 V 5 A UBEC | 1 | Radxa + camera power | on hand | M (typical; measure) |
| SH1.0 4-pin pigtail | 1 | Radxa UART to FC UART4 | to buy (CHECK) | |
| 90° USB-C plug or short cable | 1 | camera to Radxa | to buy (CHECK) | |
| Carbon tube 6 x 4 (OD x ID; Spark 26 Oct: 6 mm tubes for the geodesic cages) | ~15.9 m shell (63 pieces, 116 .. 446) + ~6.3 m links (30 pieces) for the Overseer cage v5 (`runs/overseer_layout/v5/checks.txt`) | geodesic cage shells, 6/4 links | to buy | O Cage v5 |
| Carbon rod Ø3.0 | per cage design | cage struts, wheel pins (decision 22) | on hand (length CHECK) | G |
| Catch hardware for the box (hold plate, charging contacts) | 1 set | box catch | to design with v3 | G |
| M2 / M3 screws, nylocs, standoffs | as built | stack, plate bolts | on hand | H |

## Overseer (development starting)
| Part | Qty | Role | Status | Mount data |
|---|---|---|---|---|
| S500 frame, PCB version with carbon fibre landing gear, 2015 edition (Spark 23 Oct, AliExpress item 2251832276919319; 500 wheelbase, listing; arms replaced by printed ones) | 1 | airframe | on hand | M: S500, O |
| SunnySky X2216 II 880 KV | 4 | propulsion | on hand | M: X2216 |
| 1045 prop | 4 + spares | propulsion | on hand | M: 1045 |
| ESCs | 4 | motor drive | CHECK (not recorded) | |
| MicoAir743-Lite FC | 1 | flight controller | on hand | M, O |
| Matek FCHUB-12S V2 PDB | 1 | power distribution | on hand | M, O |
| 4S 6000 LiPo, XT60 | 3 | flight pack | on hand | O, M |
| MicoAir MTF-02P | 1 | optical flow + ToF | on hand | M, O |
| Unitree L2 4D lidar | 1 | mapping, at the nose | on hand | M: L2, O |
| DECXIN 2633V2 camera | 2 | both beside the L2, field maximized (lidar colouring by yaw; no down camera, Spark 27 Oct) | 1 on hand, 1 to buy | A (P), O |
| RC receiver | 1 | RC (CRSF) | CHECK (model not recorded) | |
| Belly frame, 8 PETG wheels, pogo pads (no steel plate since 23 Oct) | 1 set | box catch (section G) | to design | G |

## Scan stick (hand-held L2 scanner, 23 Oct; `nox/overseer/scan_stick/scan_stick.md`)
Shares the Overseer's L2, camera and 4S 6000 pack (above); the Pi 5 is the Overseer's companion computer (resume, P).
| Part | Qty | Role | Status | Mount data |
|---|---|---|---|---|
| EXTEND-A-REACH 1.5-3 ft extension pole, 3/4-5 ACME tip (B08LL28XZW) | 1 | the stick; the unit screws onto its tip | on hand (Spark 23 Oct) | S |
| Raspberry Pi 5 + microSD ≥ 64 GB (E: ~1 GB/min) | 1 | recorder | on hand (resume) | M: Pi 5 (58 x 49, Ø2.7) |
| 5 V 5 A UBEC | 1 | Pi + camera, taped to the shelf side (Spark 23 Oct) | CHECK (box has one) | M: UBEC |
| 12 V 2 A buck (Pololu D24V22F12 class) | 1 | L2 supply | to buy | S |
| KCD1 rocker switch | 1 | main power | on hand (Parts & Resource sheet) | S |
| 12 mm momentary button + 5 mm LED + 330 Ω (typical) | 1 + 1 | record start / stop | to buy | S |
| 20 mm hook-and-loop strap, 450 mm (E) | 2 | pack | to buy | S |
| M3 heat-set inserts (Ruthex 5.7 long, Ø4.0 hole, 7 deep) | 2 | camera bracket to the head | on hand | M: inserts |
| M4 heat-set inserts (Ruthex RX-M4x8.1, Ø5.6 hole, 9.5 deep) + M4 x 12 | 4 + 4 | head to base flange | CHECK | S, M: inserts |
| M3 x 8, M3 x 12, M2.5 x 6, M2.5 x 6 | 4, 2, 4, 4 | L2 (3 under the head + 5 in the L2), camera bracket (6 pad + 6 in the insert, v7), camera (1.6 board + 4.4 in the post insert), Pi | on hand / CHECK | S |
| Cat5e 0.5 m, XT60 pigtail, DC 3.5 x 1.35 plug lead | 1 each | wiring | CHECK | |

Printed (state only in `nox/current-parts.md`): base, head, cam (nox/topo parts) and the thread test ring; files in the current-parts rows.

## Box: bought parts
| Part | Qty | Role | Status | Mount data |
|---|---|---|---|---|
| NEMA 17 stepper | 3 | axle + 2 arm heads | on hand | M: NEMA 17 |
| BTT SKR Mini E3 V3 | 1 | stepper + servo control | on hand | M: SKR |
| Mean Well LRS-350-24 | 1 | 24 V supply | on hand | M: LRS-350 |
| 5 V buck (servo supply) | 1 | spare (electronics v1: servos on a 6.0 V buck, logic on a 5.1 V 5 A buck); the logic buck if it gives ≥ 4 A | on hand | CHECK size |
| DS3218 20 kg servo | 1 | gripper 1 | on hand | M: DS3218 |
| MG995 servo (2-pack) | 1 | gripper 2 | ordered | M: MG995 (measure ears) |
| 6010-2RS bearing | 2 | hub on the stub | on hand | M: 6010 |
| 2040 extrusion | per stand | arm bars, stand | on hand | M: extrusion |
| 2020 brackets, M5 T-nuts | as needed (8 T-nuts for the arm) | extrusion joints | on hand | M: extrusion |
| GT2 80T pulley, 8 bore | 1 | axle drive, 4:1 | ordered | M: GT2 80T |
| GT2 20T pulley, 5 bore (kit) | CHECK | axle + head drives | on hand | M: GT2 20T |
| GT2 6 mm open belt, 2 m | 1 | 2 head belts + spliced axle loop | ordered | box-arm.md |
| GT2 belt connectors | per head | belt clamp (decision 27) | on hand | box-arm.md |
| GT2 toothless idler (bore 3 or 5) | 2 | head belt turnarounds (v4 demo; out in arm v5) | on hand (kit; bore CHECK) | M: idler |
| 625ZZ bearing 5 x 16 x 5 | 12 | arm v5 carriage V-wheels (2 per wheel) | to buy | box-arm.md v5 |
| 623ZZ bearing 3 x 10 x 4 | 1 | detent v6 roller on the 3.0 carbon axle (arm v7) | to buy | box-arm.md v7, runs/detent/v6/summary.md |
| GT2 16T pulley, 6 mm belt, 5 mm bore | 1 | arm v7 axle motor (5:1 on the fixed 80T; replaces the 20T) | to buy | box-arm.md v7 |
| M5 precision shim 5 x 10 x 1 | 6 | arm v5: one under each V-wheel (Y 93..94), the eccentric bush sits on one | to buy | box-arm.md v5 mechanism drawn |
| M5 x 10 socket head (ISO 4762) + M5 heat-set insert (OD 7.0 x 5.8, E) | 4 + 4 | arm rail v5 to hub v6, one per pad (the inserts go in hub v6's pads) | CHECK on hand | rails v5 (was M5 x 16, then x 12) |
| M3 x 4 grub screw | 2 | arm v5 16T pinions on the D-shafts (tapped in the printed boss) | CHECK on hand | box-arm.md v5 mechanism drawn |
| DIN 988 shim 3 x 5 x 0.5 | 2 | detent arm v2: under the arm on the roller's M3 (replaces v1's printed spacer) | to buy | box-arm.md v5 detent |
| M5 x 25 socket head + nyloc | 6 | arm v5 wheel axles | CHECK on hand | box-arm.md v5 |
| Bay contacts: copper-clad sheet, AA spring contacts | 0 | retired 23 Oct (XT60 back) | on hand, not used | |
| XT60E1-M panel mount (the half that mates the packs' plug; gender CHECK, D17) | 3 now (drone receiver + 2 live bays; +1 per later bay) | swap and bay power, on the float carrier | to buy | M: XT60; dimensions I "XT60 mate" |
| MG90S micro servo (hook latch) | 4 | hook latch plungers | to buy (CHECK: 2 may be on hand, D9) | D18; `box/box-hook-latch.md` |
| M2 x 12 screws + nuts | 12 (8 ear screws + nuts, 4 yoke pins) | hook latch | CHECK on hand | `box/box-hook-latch.md` |
| M3 x 10 + M3 x 5.7 heat-set inserts | 16 + 16 | hook latch housings on the panel v7 bosses | on hand (CHECK count) | M: inserts |
| A3144 hall switch + Ø3 x 2 magnet | 4 + 4 | hook latch: plunger fully out | to buy (A3144 pack priced below; Ø3 x 2 magnets CHECK) | `box/box-hook-latch.md` |
| BAT54 Schottky + 5 V TVS ≤ 1 µA at 3.3 V (ESD9L5.0 class, CHECK) + SMBJ15A + 1 kΩ / 100 kΩ resistors | 1 + 1 + 1 + 1 / 1 | seat sense on GPIO 24 (replaces the KW11) | to buy | `box/box-electronics.md` I/O |
| Keep-alive probe pins (DIY, panel v9 plain, `box/box-funnel.md` v9 plain): brass rod Ø3.0 (C360) cut to 4 x 36.5, tip domed + brass tube 4.0 OD / 3.0 ID cut to 4 x 3.0, soldered 14.0 from the tip | 1 rod ≥ 300 + 1 tube ≥ 300 | keep-alive 13.0 V, 2 per plain ±x face (replaces the pogo post) | to buy | `box/box-funnel.md` v9 plain; dimensions G Keep-alive probes |
| Compression spring, music wire, OD 4.0 x wire 0.25 x free length 25, ~16 active coils, rate ~0.045 N/mm (E): reads 0.54 N (55 g) at 13.0 long, 0.88 N (90 g) at 5.5 | 4 (+2 spare; an OD 4 / 0.25 wire assortment, rate checked on a kitchen scale) | keep-alive probe springs | to buy | dimensions G Keep-alive probes |
| Silicone wire 20 AWG, red (+x, B+) / black (−x, B−) | 4 x 0.15 m (one per pin, joined at the panel's 3.5 mm bullet) | keep-alive probe leads | to buy (or from the 20 AWG row below) | `box/box-electronics.md` |
| M2 x 6 pan head self-tapping (PT / plastite) + 2.5 mm cable ties | 2 + 4 | probe caps; lead strain relief in the boss groove | to buy (CHECK on hand) | `box/box-funnel.md` v9 plain |
| Endstops | CHECK | homing (or a hard stop, DEFAULT) | on hand | |
| Clear sheet, 3.0 (enclosure) | 4 | stand v7 walls | CHECK | box-stand.md |
| Brass heat-set inserts M3 / M5 | as drawn | printed-part threads | on hand | M: inserts |
| M3 / M5 screws and nuts (M4 for the PSU: CHECK) | per joint | assembly | on hand | box-stand.md Parts, box-arm.md |
| Wire: 26 / 28 / 22 AWG | ~7.8 m / ~2.8 m / ~0.9 m | arm harness | on hand | box-wiring.md cut list |
| Taidacent 12-wire slip ring | 1 | (retired: bounded-travel harness instead) | ordered, not used | |
| **Box electronics v1 (24 Oct, `box/box-electronics.md`)** | | | | |
| Raspberry Pi 4 B 2 GB + microSD ≥ 32 GB | 1 | box computer: Klipper host + `boxd` (DEFAULT; alt Pi 5) | to buy | M: Pi 4 85 x 56 (spec) |
| GPIO screw-terminal HAT for the Pi 4 | 1 | halls, seat sense, relay inputs, I2C, pull-ups | to buy | |
| PCA9685 16-ch PWM board (Adafruit 815 class) | 1 | 4 latch MG90S + 2 gripper servos | to buy | 62.5 x 25.4 (spec) |
| INA226 module + R010 2512 shunt (swap the R100) | 4 + 4 | keep-alive, 2 live bays, gripper servo stall cut (heads v7) | to buy (CHECK: INA226 on hand?) | E 32 x 22 |
| ADS1115 4-ch 16-bit ADC module + DRV5055A4 analog hall + AP2112K-3.3 LDO + 10 k / 30 k resistors | 1 + 4 + 1 + 2 / 2 | grip halls, 2 per head summed 3:1 (heads v7; the magnets are the heads') | to buy | E 28 x 18 |
| XL4016-class CC/CV buck, 8 A | 2 | keep-alive 13.0 V CC 3.0 A; servo rail 6.0 V | to buy | E 66 x 47 |
| 24 → 5.1 V 5 A buck (Pololu D36V50F5 class) | 1 | Pi + logic (or the on-hand 5 V buck if ≥ 4 A) | to buy (CHECK) | E 41 x 26 |
| DPDT relay 24 VDC coil, 8 A contacts (Omron G2RL-2 DC24 class, spec) + PCB socket | 3 | brake KB1 / KB2 (heads) / KB3 (axle): NC shorts each motor coil on itself when its brake pin is low (`box-electronics.md` Head brake) | to buy | E 29 x 13 each |
| ULN2003A (DIP-16, bare chip) + small perfboard | 1 | brake relay coils at 24 V from 3 SKR pins (not a 28BYJ-48 LED module: 5..12 V) | to buy | E 30 x 20 |
| 4-ch 5 V opto relay board (SRD-05VDC-SL-C, 10 A 30 VDC) | 1 | K1 keep-alive, K2 / K3 live bays, K4 spare (replaces the 7 x 8-ch mux, priced) | to buy | E 75 x 55 |
| 6-way blade fuse box with negative bus + blade fuses 10 / 5 / 3 / 2 A | 1 + 1 set | 24 V distribution | to buy | E 105 x 60 |
| IEC C14 fused switched inlet + mains cord | 1 + 1 | mains in, on the rear sheet | to buy (cord CHECK on hand) | cut-out 48 x 28 (E, CHECK) |
| 22 mm mushroom E-stop, NC | 1 | cuts the SKR and the servo buck | to buy | hole 22.5 (E) |
| iCharger 456DUO class dual-channel DC charger | 1 | 2 live bays, MODBUS to the Pi | to buy (CHECK the MODBUS map first; alt ToolkitRC M8D or 2 x ISDT PD60) | listing 171 x 118 x 59 |
| Nylon M3 standoffs 10 F-F + M3 x 6 + nuts; M2.5 set for the Pi / PCA | ~50 | boards on the rear sheet | CHECK on hand | |
| M4 x 6 pan head + fender washers | 4 + 4 | PSU on the rear sheet | CHECK on hand | M: LRS-350 |
| Wago 221-415 lever connectors | 3 | 6.0 V, 5.1 V, GND buses | CHECK on hand | |
| XT60 pairs; 3.5 mm bullet pairs; JST-XH 3 / 4-pin housings + crimps; bootlace ferrules; fork crimps M3.5 | 2; 2; 1 / 3; ~60; 5 | sheet disconnects, board ends | CHECK on hand | |
| 22 AWG 3-core servo extension, 1 m | 4 | hook latch servos | to buy | |
| Silicone wire 14 / 16 / 18 / 20 AWG red + black | 0.6 / 4.5 / 5.3 / 1.1 m (totals) | power nets (cut list in the doc) | CHECK on hand | |
| 10 kΩ resistors | 8 | GPIO pull-ups | CHECK on hand | |
| USB-A to USB-C 0.3 m | 1 | Pi → SKR | CHECK on hand | |

## Box: priced for later (not ordered)
| Part | Qty | Role | Mount data |
|---|---|---|---|
| A3144 hall module | 1 pack | axle index | M: A3144 |
| 5 x 2 magnets | 1 pack | axle index | |
| M6 ball plungers | 1 pack | axle detent | |
| 24 V 4010 fan | 1 | cooling; electronics v1: only if the SKR drivers run hot in the soak test (Problem 1) | M: 4010 |
| ISDT PD60 charger (listed as PD60S) + 65 W USB-C brick | 1 + 1 (2 + 2 as the alternative) | balance charging; the alternative to the dual-channel DC charger (electronics v1) | M: PD60 |
| 8-channel 5 V relay board | 7 | 7-bay mux; electronics v1 uses one 4-ch board for 2 live bays (row above) | CHECK size |
| JST-XH 4S balance pigtails, 5-pack | 2 | bay balance leads | M: JST-XH |
| Hub USB camera | 1 | vision (the box Pi is now to buy, row above) | CHECK model |

## Box: printed parts (state only in `nox/current-parts.md`)
| Part | Qty | Print file |
|---|---|---|
| Funnel panels A + B v7 (seated ±y faces) | 2 + 2 | `print/box_funnel_panel_A_v7.stl`, `print/box_funnel_panel_B_v7.stl` |
| Funnel panels v9 plain (±x faces): A px, B px, A nx, B nx | 1 each | `print/box_funnel_panel_{A,B}_v9_plain_{px,nx}.stl` |
| Funnel probe cap v9 plain | 2 | `print/box_funnel_probe_cap_v9_plain.stl` |
| Funnel hub + corner blocks | 1 + CHECK | design |
| Bay module v4i | 2 | `print/box_baymod_v4i.stl` |
| Hub v5 | 1 | `print/box_hub_v5.stl` |
| Sun post v1 | 1 | `print/box_sun_post_v1.stl` |
| Stub v3 | 1 | `print/box_stub_v3.stl` |
| Detent ring v1 + roller arm v1 | 1 + 1 | `print/box_detent_ring_v1.stl`, `print/box_detent_arm_v1.stl` |
| Idler ends A / B v3 | 1 + 1 | `print/box_idler_end_A_v3.stl`, `print/box_idler_end_B_v3.stl` |
| Pinion v4 | 2 | `print/box_pinion_v4.stl` |
| Fingers v4 (A and B, for heads A and B) | 4 | `print/box_finger_*_v4.stl` |
| Arm heads A / B v5 (v5b with the connector pad) | 1 + 1 | not in `print/` yet |
| Hook latch housing A / B v1, plunger v1 | 2 + 2, 4 | `print/box_hook_latch_housing_A_v1.stl`, `..._B_v1.stl`, `print/box_hook_latch_plunger_v1.stl` |
| Arm v5: rail A / B v1, V-wheel v1 (x6), pinion 16T v1 (x2), hub v6, heads A / B v6 with the carriages | 2, 6, 2, 1, 2 | design (`box-arm.md` v5); replaces the 2040 bars, idler ends, glides, belt clamp |
| Belt clamp coupon set v2 | 1 (on hold, decision 27) | `print/belt_clamp_coupon_set_v2.stl` |
| Mast v6 | 1 | `print/box_mast_v6.stl` |
| KB v7b | 1 | `print/box_kb_v7b.stl` |
| Cradle + front keel v6 | 1 | `print/box_cradle_keel_v6.stl` |
| Cross bar half v8 | 2 | `print/box_bar_half_v8.stl` |
| Toe v7b | 1 | `print/box_toe_v7b.stl` |
| Ring frame mid v6 | 5 | `print/box_frame_mid_v6.stl` |
| Ring frame ends L / R v7b | 1 + 1 | `print/box_frame_endL_v7b.stl`, `print/box_frame_endR_v7b.stl` |
| Corner post segment v7 | 12 | `print/box_post_seg_v7.stl` |
| Ring-to-wall brackets jA v8, jB v7, e v7c | 6 in all (split in `box-stand.md`) | `print/box_bracket_jA_v8.stl`, `print/box_bracket_jB_v7.stl`, `print/box_bracket_e_v7c.stl` |
| Arm harness: wire grommet, end clips, slot clips | 1, 2, 6 | in the wiring chat (`nox/box/wire_parts.py`) |

## Gaps to close (measure or ask Spark)
Dimension gaps (sizes, holes, threads) are the open questions Q1..Q20 at the end of `dimensions.md` M: settled when a mount for that part is designed. The gaps below are about what is on hand.
- Overseer: S500 variant, ESC model, RC receiver model.
- Scout: motor KV, number of spare packs, carbon rod stock, grommet size.
- Box: 5 V buck model (spare or logic); INA226 on hand; electronics v1 hardware marked CHECK on hand.
