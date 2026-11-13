"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 4; the lightest passing every case < 0.3 both ways is folded into the frame, these
folders deleted after). Round 3 (the frame's setup at gen 115 / 130 / 155): 177.8 / 196.3 / 227.4 g, worst 0.459 / 0.386 / 0.338, always
BT-TB at (-45, 25, 11.7), 3.6 behind the tail battery bolt's boss, just over its support region (the boss's clamped bottom 1.5 only).
This one: the battery bolts supported on the clamped face AND the insert's 6 mm bore (the M3's shear goes in along the insert), gen 130."""
import parts.scout_v4_frame.part as F

F.REC_SUPPORT = "insert"
SPEC = dict(F.SPEC, gen_cm3=130.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_va"
    return p
