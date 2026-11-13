"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 5; the lightest passing every case < 0.3 both ways is folded into the frame, these
folders deleted after). Round 4: the battery bolts on face + insert at gen 130 / 155 -> 196.4 / 228.4 g, 0.376 / 0.321; face only at 180
-> 258.9 g, 0.306; always BT-TB by the tail bolt (index ~ V^-0.9 with the insert: 0.29 at ~174). This one: face + insert, gen 170.0."""
import parts.scout_v4_frame.part as F

F.REC_SUPPORT = "insert"
SPEC = dict(F.SPEC, gen_cm3=170.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_va"
    return p
