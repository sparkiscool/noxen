"""TEMPORARY variant of parts/scout_v4_frame (13 Nov, the design chat: three frame variants per round, the best folded into the frame, these
folders deleted after). One change from the frame's setup: the arms' room a flat 12 deep band in the deck's layers, root to pod (no haunch: the run of 57802c8 on the built-hold engine read 0.834 ACROSS where the haunch's bottom chord met the body at z 9.7)."""
import parts.scout_v4_frame.part as F

F.ARM_Z0 = F.POD_Z0                                  # 12.5 at the root and the pod
SPEC = dict(F.SPEC, gen_cm3=100.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_va"
    return p
