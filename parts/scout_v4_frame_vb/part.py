"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 2, the design chat: three frame variants per round, the best folded into the frame,
these folders deleted after). From the frame as folded (round 1's vb: 159.0 g, worst 0.59 BS0 compression across at (53, 39, -0.3), BT-TB
0.51 at the receiver's tail corner (-47, -35, 7.7)): the rooms beside the receiver's front half (x 20 to the nose block) down to z -12: BS's push runs back beside the receiver in its layers, not up the side room across them."""
import parts.scout_v4_frame.part as F

F.FRONT_Z0 = -12.0
SPEC = dict(F.SPEC)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vb"
    return p
