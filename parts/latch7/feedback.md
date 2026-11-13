# Latch v7, both packs (parts/latch7/make.py builds every piece): notes

## Spark
- 13 Nov: "do a full optimization run over the whole design and build the arm as well make sure things all are compatable and get it ready to print"
- 13 Nov: "isnt the latch a little to small? if the battery lands slightly slanted it might now latch properly" (answered the same day: window 1.5, seat strips at the latch line; `slant()` in make.py is the check, keep it passing)
- 12 Nov: no metal rods, printed parts instead. 10 Nov: latches and fingers the same on both packs, strong enough for the larger pack; as few bought parts as possible; XT60W (Amazon B0DZ288QD2) stays.
- 9 Nov: error tolerance at every swap step; spring and plug forces even, nothing tilting and rubbing; the real servos and motors; every important mechanism as big as makes sense.

## Design chat (session_01EtHMZPw3VtLv7n2ybfHWpH), its reading, not Spark's words
- Every piece here is drawn (receiver 95 g Scout / 210 g Overseer, cartridge 40 g, carrier, cap): none has been through the engine. The run is for the pieces with room to generate; a piece that is all attachment stays drawn and says so.
- Frozen, the arm stream builds to it: the grabber interface (`blade()`, `heel()`, `pads_of()`: wedge feet, hook windows, push bosses, heel pad zones, magnets, the tab line) and the cartridge's outline through the funnel hole. Frozen, other streams bolt to it: the receiver's mount bosses and the Scout's four landing feet (`scout_feet()`), the outline the funnel seat takes (`--fit`).
- Known and open: the Overseer receiver lets the cartridge roll 0.65 deg, the arm's worst is 0.74 (0.1 mm more side clearance, or the head gives); the plug carrier is drawn for XT60E1-M, Spark buys XT60W (B0DZ288QD2), needs its dimensions (listing, tag E until measured); strength numbers are hand models (E); the receiver's pockets have not been through FE.
- Only Spark has: the springs' force and closed length on a scale, a real XT60W pull force.
