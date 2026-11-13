"""TEMPORARY variant of parts/scout_v4_frame (13 Nov, the design chat: three frame variants per round, the best folded into the frame, these
folders deleted after). One change from the frame's setup: the arms' room a flat 16 deep band (z 8.5..24.5), root to pod: deeper chords, no haunch."""
import parts.scout_v4_frame.part as F

F.ARM_Z0 = F.POD_Z0 = F.DZ1 - 16.0
SPEC = dict(F.SPEC, gen_cm3=100.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vb"
    return p
