"""TEMPORARY variant of parts/scout_v4_frame (13 Nov, the design chat: three frame variants per round, the best folded into the frame, these
folders deleted after). One change from the frame's setup: the budget 100 -> 130 (the haunched arms as they are): where the strength / mass curve sits with the built-hold engine (hold on: 0.519 at 158.8 g, released by the solid check)."""
import parts.scout_v4_frame.part as F


SPEC = dict(F.SPEC, gen_cm3=130.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vc"
    return p
