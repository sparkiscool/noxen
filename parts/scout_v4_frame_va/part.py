"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 3, the design chat: the lightest that passes every case < 0.3 both ways is
folded into the frame, these folders deleted after). From the frame as it is (flat 16 arms, gen 100: 158.8 g, worst 0.525 BT-TB by the
tail bolt): the budget 115.0 cm3 (the setup otherwise the frame's: its audit / smoke at c4dfb15 holds; only the budget's share of the room moves).
The Radxa out of the tail tray (round 3's first ask) has nowhere to go: the deck top is the bed face and the discs' keep-out (r 80.7 from
z 27.7, 3.2 over the deck) leaves only |x| + |y| <= ~42 clear in plan, the turret's +-22 in its middle; under the deck every 65 mm run is
taken (turret parts x -8.5..18, the FC 18.5..54.5, the tail bolts' bosses across y)."""
import parts.scout_v4_frame.part as F

SPEC = dict(F.SPEC, gen_cm3=115.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_va"
    return p
