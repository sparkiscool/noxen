# Overseer lights (30 Oct 2026, research only; Spark: "what light is best for our setup")

Job: light what the three DECXIN 2633V2 cameras (102° diagonal, 89 x 73, global shutter) see where there is no ambient
light, evenly, flicker-free in a short global-shutter exposure, with nothing in the L2's hemisphere or any camera's view,
off the drone's own 4S pack (12.8..16.8 V; 12 V 2 A buck is the L2's, 5 V 5 A UBEC is the Pi's), on a PETG head
(Tg 79..80 °C, HDT 62..76 °C [12]). Hover 6..10 A (E), so every watt is 1..2 % of hover. Tags as dimensions.md:
M measured, C CAD, E estimate, S spec sheet / listing; [n] is a source below.

## 1. Options

| # | Option (class) | Light | Beam vs the 89 x 73 field | CRI / CCT | Input, full / useful dim | Mass, size | Heat, on PETG? | Driver, flicker at 1 ms | FC / Pi control | IP | Price, link |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | **Cree XLamp CXB1304 Pro9 COB**, 4000 K 90 CRI, 9 V class (COB) | 458 lm at 400 mA, 85 °C (S [5]); ~400 lm at 350 mA, ~250 lm at 200 mA (E, linear) | Lambertian 120° FWHM, no optic: covers 89 x 73 with the corners at 0.63 x axis on a sphere, 0.16 x on a flat wall (E, cos⁴) | 90 min / 4000 K (S) | 8.6..9 V (S [5]); 0.40 A = 3.6 W full, 0.35 A = 3.1 W (recommended), 0.20 A = 1.8 W dim (E) | ~1 g bare, 13.35 mm square (S [5]); + 8.8 g 25 x 25 x 10 heatsink [11] | at 350 mA ~2.0 W heat (E, 35 % radiant): +36..42 K on an 18..21 K/W sink [10] = 60..67 °C sink, PETG pad behind a 1 mm air gap ~55 °C (E) = under HDT. At 700 mA +75 K: NOT on PETG | constant current, DC: no flicker at any exposure. Dim only by choosing the current (no PWM, see driver row) | on/off from Pi GPIO 3.3 V into the driver DIM pin, no MOSFET; FC: see row K | bare; the driver is IP67 | ~$2.35 at 100 (aggregator, DigiKey tray [4]), ~CAD 3 single for a 3000 K sibling [4]; DigiKey/Mouser |
| B | **Nichia 519A** x3 in series on a 20 mm MCPCB, 4000 K R9080 (bare LED array) | 428 lm per die at full drive (S, reseller [6]); 3 dies at 350 mA ~450 lm total (E) | 3 point sources you can aim (centre, ±30°) for a flatter field than one COB (E); each ~120° | Ra ≥ 90, R9 > 80 (S [6]) / 4000 K | 3 x ~3.1 V = 9.3 V at 350 mA = 3.3 W; 700 mA = 6.6 W (E) | 3 x 0.1 g + 20 mm Cu star ~2 g (E) + the same 8.8 g sink | as A per watt | as A | as A | bare | $1.68..2.01 per die, min 5 (listing [6]); star ~$2..5 (E) |
| C | **Yujileds BC 135L COB CRI 95+** (COB, colour-fidelity option) | 10 W 5600 K 1100 lm ($48) or 9 W HD5 4000 K 740 lm ($72.25) (S [7]) | Lambertian 120°, 13.5 mm | 95+ (S) / 4000 or 5600 K | 600 mA at 14..17 V (9 W HD5) or 300 mA at 32..39 V (10 W G03) (S [7], listings disagree): above the 4S floor minus 3 V, so a BOOST driver, not the LDD-L | ~1 g + sink | 9..10 W = 6 W heat: not on PETG at full; usable only dimmed to ~3 W (E) | constant current DC, same as A | same | bare | $48..72 each [7] |
| D | **Lumileds LUXEON CoB 1202 G3** L2C5-40901202E0900 (COB) | 858 lm typ (S [8]) | Lambertian 120° | 90 / 4000 K (S) | Vf not in the listing (S [8]); check the datasheet before choosing a driver | ~1 g + sink | as A per watt | as A | same | bare | $2.26 at 100 (S [8]) |
| E | **Lume Cube 2.0** (Lume Cube class) | 750 lux at 1 m, 80° beam = ~1100 lm if even (E) | 80° beam: short of 89° horizontal, corners dark | 95 / 5600 K (S [1]) | own Li-ion, USB-C 5 V charge; runs 90 min at full; running on external USB while lit not confirmed (S [1]) | 99 g, 41 x 41 x 43 (S [1]) | sealed body, OK on PETG | internal PWM dimming, frequency unpublished: flicker risk at 1 ms (E) | Bluetooth app only; no FC / Pi input | waterproof 9 m (S) | $89.95 [1] |
| F | **Ulanzi VL49 (1672)** (camera-class panel) | 800 lux at 0.5 m = 200 lux at 1 m, 6 W max (S [2]) | soft panel, very wide, weak | 95+ / 5500 K (S [2]) | own 2000 mAh battery, USB-C; "power pass-thru: no" (B&H [2]) | 90..101 g, 49-LED panel (S [2]) | fine | internal, unpublished PWM | none | none | $22.95 [2] |
| G | **FlyingTech 1.5 W 12 V headlight** (FPV class) | 1.5 W, no lumen figure (S [3]); ~120 lm (E) | 180° viewing angle (S): most light outside the field | not published, FPV white (E: 70, 6500 K) | 12 V 0.13 A from the buck | 5 g (S [3]) | fine | resistor-limited DC: no flicker; no dimming | on/off by switching 12 V (MOSFET) | none | ~£5 [3] |
| H | **Keyence CA-DBW13** 132 mm white bar (machine vision) | not published in lm; 7.3 W (S [9]) | bar light for 300 mm working distances, narrow in one axis (E) | not published / white | 12 VDC 0.6 A (S [9]) | ~90 g, 132 mm (S [9]) | metal body, fine | DC from the CA controller; dims via the controller only | none without the Keyence controller | — | ~$86 used/new eBay [9] |
| I | **Smart Vision Lights LM45 / LM75 bar**; Effilux EFFI-Flex (machine vision) | not published in lm | bar, adjustable optics (Effilux) | white, CRI not published | 24 VDC (S [9b][9c]): a boost from 4S | 75 mm bar, mass not listed (E 150 g) | fine | strobe/continuous drivers, 24 V PNP/NPN inputs | 24 V logic | IP65+ | $510 (LM75 white) [9b]; EFFI-Flex 5-LED £484 ex VAT [9c] |
| J | Drivers for A..D: **Mean Well LDD-350L / 700L** | — | — | — | in 9..36 V, out 2..32 V at Vout ≤ Vin − 3 V (S [13]): from the 4S rail a 9 V COB works down to 12 V pack; from the 12 V buck it does NOT (9 V limit) | 22.6 x 9.9 x 8.9, 6-pin DIP, ~4 g (E; weight not published) [13] | IP67, 95 % | PWM dimming 100 Hz..1 kHz only (S [14]): 1 period per 1 ms exposure = frame-to-frame brightness jumps, so NO PWM; DIM pin as on/off: > 2.5 V or open = on, < 0.8 V = off, referenced to −Vin [14] | Pi GPIO direct (3.3 V > 2.5 V) | IP67 | ~$6..9 each (distributors) |
| K | Alternative driver: **Recom RCD-24-0.70** | — | — | — | in 4.5..36 V, out 2..35 V at 700 mA; analog dimming 0..4.5 V pin (S [15]) = flicker-free dimming from a filtered Pi PWM | ~7 g, 22.4 x 12.6 x 8.5 (E) | — | PWM linear only to 200 Hz, limit 1 kHz (S [15]): PWM unusable here; analog pin fine | Pi GPIO + RC filter | — | NRND (not for new designs) [15] |

FC control (row K and A): PX4 1.18 can switch an AUX pin as a camera-trigger GPIO or drive it as a servo PWM (RC AUX
passthrough, Offboard Actuator Set via MAV_CMD_DO_SET_ACTUATOR) [16]; a servo PWM is not a logic level, so it needs a
PWM-to-switch board. The Pi colours the scan and knows when the light is needed, so the Pi GPIO is the simpler path.

## 2. How much light that is (E, the number that decides everything)

- One module at 400 lm, Lambertian: on-axis intensity 400/π = 127 cd; 127 lux at 1 m, 32 lux at 2 m, 14 lux at 3 m.
- A mid-grey exposure needs E = 250 N²/(t S) lux (incident-meter formula, constant 250 [17]). At f/2 (lens f-number not
  published: E) and ISO 800 (8x gain): 1 ms needs 1250 lux, 10 ms needs 125 lux, 20 ms needs 63 lux.
- So one 3 W module per camera lights a 1 m scene at 10 ms, a 2 m scene at 20 ms. **At 1 ms it is 40x short**: 1 ms
  would need ~50 W of LED per camera. Flyability's Elios 3 carries 16,000 lm for the same job [18]; the PETG head and
  the 4S pack cannot. The exposure must be 10..20 ms with gain; at the 30°/s colouring yaw limit that is 0.3..0.6° of
  blur, 6..13 px of 1920 (E).
- A strobe synchronised to the exposure would cut the power 10x, but the 2633V2 is a UVC board with no documented
  trigger or strobe pin (dimensions.md: no maker data), so the light is continuous.

## 3. Placement (C from the assembly, E where tagged)

- **One light per camera, outboard of its lens, emitting face flush with the lens front, aimed along the camera axis.**
  Co-located light hides the shadows of the L2 body, bosses and tubes behind the objects that cast them (as the camera
  sees them); an off-axis light (on nodes 15 / 16, say) throws head and tube shadows into the scene.
- Front lights: lenses are at (390, ±75, 122), yaw 20° out, pitch 10° down, 13 behind the L2 base plane. Light centre
  45 mm outboard along the camera's horizontal axis: **(375, ±117, 122)** (C). A point 45 mm outboard enters the 89°
  field only 46 mm ahead of the lens (45 / tan 44.5°), so a face flush with the lens is 46 mm clear; the heatsink is
  behind the face. The other camera's inner field edge crosses the centreline 165 mm ahead (dimensions.md), so the
  outboard lights are nowhere near it. The light face is 13 mm behind the L2 base plane (as the lens): outside the L2's
  360 x 90 hemisphere [19]. White LED light (phosphor ends ~750 nm) does not reach the L2's infrared band, and the L2 is
  rated for 100 klux sunlight (reseller [19]), so glare into the L2 is not a case (E).
- Rear light: lens (−288.7, 0, 58.6), facing −x, 40° down, inside the facet 13-14-5. Light centre 35 mm beside it
  (y −35, same lens plane, same axis), on node 21 or the tube 13-14 clamp; tube shadows hide behind the tubes.
- Masses (E): COB 1 g + 25 x 25 x 10 sink 8.8 g + thermal pad, 2 x M3 and 0.5 m wire 3.5 g = **13 g per light**, driver
  on the core (4 g each, 3 on the hub deck); 17 g if the driver rides on the head. Head: 336 g in the frame model →
  **362 g** with two lights (+8 %; head rods were at ≤ 126 N F, so ~136 N: under every link figure, but the head and
  node 15 / 16 reruns must carry it). Node 21: camera 25 g + light 13 g = **38 g** on three 3.0 rods (node budget was
  12 g for the print itself).
- Power: three at 350 mA, 9 V = 9.5 W ≈ 0.7 A at 14.8 V, 7..12 % of hover; 3.2 Wh over a 20 min flight, 3.6 % of the
  89 Wh pack (E). Fed from the pack rail after the rocker through a 2 A fuse (the 12 V buck cannot: Vout ≤ Vin − 3 V).

## 4. Recommendation

1. **Three Cree CXB1304 Pro9 4000 K 90 CRI COBs (A), one per camera, each on a 25 x 25 x 10 black anodised heatsink,
   each on its own Mean Well LDD-350L (J) from the pack rail, DIM pins to three Pi GPIOs: on/off, no PWM.** 1200 lm,
   9.5 W, 0.7 A, 39 g on the drone, under $40 in parts.
2. Why A: a bare Lambertian COB covers 89 x 73 with no optic; constant-current DC never flickers in any exposure; 90 CRI
   4000 K is neutral for colouring; 3 W per module is the most the PETG head can shed (sink ~65 °C, pad ~55 °C, E);
   cheapest and lightest lumens here, in stock at DigiKey / Mouser.
3. Why not the rest: E / F carry their own battery (99 g each, no external-power running confirmed, unpublished PWM);
   G is 1.5 W of cool 70-CRI light; H / I are 24 V (or controller-bound) bars built for 300 mm working distances at
   90..150 g and $86..510; C (CRI 95) needs a boost driver and is $48..72 a piece: the upgrade if colours prove off.
4. Honest limit: 400 lm per camera is 32 lux at 2 m. Set the cameras to 10..20 ms exposure with gain for colouring; the
   1 ms exposure needs 40x this light and is not reachable on this head. Test one module on a printed pad coupon with
   a thermocouple before three are printed in (pad > 65 °C = back to 250 mA).
5. Upgrade path, in order: 700 mA (6.3 W, ~750 lm) only if the coupon pad stays under 65 °C with a 30 x 30 x 15 sink;
   then B (three aimed 519A dies) for a flatter field; then C for CRI 95.

### What the design session adds to the head mount (numbers)

- Two **light pads 27 x 27**, face normal along the camera axis (yaw ±20°, pitch −10°), centres **(375, ±117, 122)**,
  face flush with the camera lens front (±2 mm), so ≥ 13 mm behind the L2 base plane; the camera board's own 8 mm plug
  keep-out stays.
- Per pad **two M3 heat-set insert bosses at 20 mm centres** (Ø7 x 6, the sink screwed through 3 mm standoffs for a
  1 mm minimum air gap to the PETG); loads: 13 g at the pad centre (17 g with the driver) in the head's 14 handling
  directions and flight 4 g, as the head's own masses.
- **Keep-out** per light: a 27 x 27 x 14 box behind the pad face (sink 10 + COB + standoffs), nothing of it ahead of the
  lens face; nothing else of the head within 46 mm ahead of the pad face at that lateral offset (the camera frustum edge).
- **Cable route**: one 2-wire 24 AWG pair per light (3 g per 0.5 m, E) from the pad, along head rod ±1 with the camera USB
  cable to node 15 / 16, down the cage to the hub deck; the rear pair along a node 21 rod to node 13 / 14 and in.
- **Core**: three LDD-350L (22.6 x 9.9 x 8.9 each) on the hub deck beside the UBEC, a pack-rail tap after the rocker
  with a 2 A fuse, a 4-wire GPIO harness (3 DIM + GND) from the Pi 40-pin.
- **Node 21 / rear clamp**: one 27 x 27 pad at y −35 beside the rear camera, face coplanar with its lens, normal −x
  pitched 40° down, same bosses; node 21 rods now see 38 g instead of 25.
- **Frame model**: head 362 g (E) at the two pad positions added to the 336 g; node 21 +13 g.

## Sources

1. Lume Cube 2.0: 750 lux at 1 m, 80°, CRI 95, 5600 K, 99 g, USB-C, $89.95 — https://www.bhphotovideo.com/c/product/1507369-REG/lume_cube_lc_v2_1_lume_cube_2_0_black.html , https://camerajabber.com/?p=87201 , https://hotrodcameras.com/products/lume-cube-2-0-daylight-balanced-portable-led-light-black-single
2. Ulanzi VL49 1672: 800 lux at 0.5 m, 6 W, CRI 95+, 5500 K, 90 g, $22.95; B&H "power pass-thru: no" — https://www.ulanzi.com/products/mini-led-video-light-ulanzi-vl49 , https://www.bhphotovideo.com/c/product/1606863-REG/ulanzi_2287_vl_49_rechargable_mini_rgb.html
3. FlyingTech 1.5 W 12 V headlight, 5 g, 180° — https://www.flyingtech.co.uk/product/racing-drone-super-bright-led-headlight-1-5w-12v-3s-jst/
4. Cree CXB1304 pricing / listings — https://zenode.ai/parts/kxtgsmqu7 (4000 K 90 CRI, 458 lm, 8.6 V), https://www.digikey.ca/en/products/detail/cree-led/CXA1304-0000-000F00B430F/4500927 , https://azcus.digikey.com/en/products/detail/cree-led/CXB1304-0000-000C0HC230G/5124946
5. Cree XLamp CXB1304 Pro9 datasheet (9 V flux at 400 mA, 85 °C; 13.35 mm; thermal section) — https://downloads.cree-led.com/files/ds/x/XLamp-CXB1304-Pro9.pdf
6. Nichia NVSW519A-V1 4000 K R9080 428 lm — https://www.ledrise.eu/leduri-individuale/nichia-leds/nichia-x19-series-high-power-leds.html ; bare 519A $1.68..2.01 min 5 — https://www.accio.ai/find-product/nichia-519a-5000k
7. Yujileds 135L CRI 95: 10 W 5600 K 1100 lm $48 — https://store.yujiintl.com/products/yujileds-cri-95-10w-cob-led-3200k-5600k-135l ; 9 W HD5 4000 K 740 lm $72.25, 600 mA 14..17 V — https://store.yujiintl.com/products/yujileds-cri-95-5mm-9w-cob-led-2700k-3200k-4000k-hd5
8. Lumileds LUXEON CoB 1202 G3 L2C5-40901202E0900, 858 lm, $2.26 at 100 — https://www.futureelectronics.com/p/lighting-solutions--light-engine-led-assemblies--chip-on-board/l2c5-40901202e0900-lumileds-6066503
9. Keyence CA-DBW13: 12 VDC, 7.3 W, ~90 g — https://www.eurosensor.com/img/pdf/keyence-ca-dbw13-beyaz-Isik-armaturu-pdf-90739.pdf , https://www.keyence.com/products/vision/vision-sys/ca-d/models/ca-dbw13/ ; eBay ~$86 — https://www.ebay.de/itm/335446700769
   9b. Smart Vision Lights LM75 white, 24 VDC, $510 — https://www.edmundoptics.com/p/smart-vision-lights-75mm-white-bar-light/57257/
   9c. Effilux EFFI-Flex 24 VDC, 5-LED £484.50 ex VAT — https://edmundoptics.co.uk/p/5-led-red-adjustable-led-bar-light/4026
10. 25 x 25 x 10 heatsink natural convection 18.1 / 21.3 K/W — https://cuidevices.com/product/thermal-management/heat-sinks/bga-heat-sinks/hsb24-252510
11. HSB24-252510 weight 8.8 g, AL6063-T5 black anodised — https://www.farnell.com/datasheets/4783065.pdf
12. PETG Tg 79..80 °C, HDT 62..76 °C — https://dbe.unibas.ch/en/research/core-facility-3d-print-lab/home/materials/petg , https://cdn.shopify.com/s/files/1/0658/6459/7724/files/TDS_PETG.pdf
13. Mean Well LDD-700L: 9..36 V in, 2..32 V out, 700 mA, 22.6 x 9.9 x 8.9, IP67 — https://azcus.digikey.com/en/products/detail/mean-well-usa-inc/LDD-700L/7704757 , https://www.trcelectronics.com/products/mean-well-ldd-700l ; Vout = Vin − 3 V — https://masterelectronics.com/en/mean-well/ldd700l-43544915.html
14. Mean Well LDD PWM 100 Hz..1 kHz, DIM > 2.5 V / open = on, < 0.8 V = off — https://forum.allaboutcircuits.com/threads/meanwell-ldd-pwm-questions.119522/ , https://www.ledsupply.com/led-drivers/mean-well-ldd-h-series-cc-step-down-mode ; LDD-500H 18 g — https://ie.farnell.com/mean-well/ldd-500h/dc-dc-converter-pwm-26w-th/dp/3596430
15. Recom RCD-24-0.70: 4.5..36 V in, 700 mA, analog + PWM dimming, PWM linear to 200 Hz / limit 1 kHz, NRND — https://www.farnell.com/datasheets/1854226.pdf , https://www.digikey.cn/en/products/detail/recom-power/RCD-24-0-70/2256303
16. PX4 camera trigger GPIO on any FMU output; RC AUX / Offboard Actuator Set outputs — https://docs.px4.io/main/en/camera/fc_connected_camera.html , https://docs.px4.io/v1.13/en/payloads/
17. Exposure formula E = C N² / (t S), C ≈ 250 — https://colour-hdri.readthedocs.io/en/v0.1.5/_modules/colour_hdri/utilities/exposure.html ; PWM shorter than the exposure = frame-to-frame brightness change — https://www.accessnewswire.com/newsroom/en/electronics-and-engineering/vadzo-imaging-explains-led-flicker-mitigation-techniques-for-industrial-1172807
18. Flyability Elios 3: 16,000 lm lighting — https://mfe-is.com/wp-content/uploads/2024/12/Elios-3_tech-specs.pdf
19. Unitree L2: 360° x 90° hemisphere, infrared pulsed laser (manual); > 100 klux anti-glare (reseller) — https://oss-global-cdn.unitree.com/static/Unitree%204D%20LiDAR%20L2%20User%20Manual.pdf , https://www.robotshop.com/products/unitree-4d-lidar-l2
