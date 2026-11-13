"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 2, the design chat: three frame variants per round, the best folded into the frame,
these folders deleted after). From the frame as folded (round 1's vb: 159.0 g, worst 0.59 BS0 compression across at (53, 39, -0.3), BT-TB
0.51 at the receiver's tail corner (-47, -35, 7.7)): the rooms beside the receiver out to |y| 46 (was 40; inside the funnel's 55 deg profile): room round the receiver's tail corner for BT's 551 N."""
import parts.scout_v4_frame.part as F

F.SIDE_Y1 = 46.0
SPEC = dict(F.SPEC)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_va"
    return p
