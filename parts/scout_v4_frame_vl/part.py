"""TEMPORARY variant of parts/scout_v4_frame (14 Nov: round 7's winner vh, the bolt forks at gen 130, 195.3 g, worst 0.217, given its one
keep-out run before the fold): latch v7's Scout set as the frame's keep-out (LATCH_V7: the receiver room x -48.3..63.1, the carrier and cap,
the cartridge's path down, the lowered plug's tabs, its leads out the lead end and their channel up and back to the FC), the flow sensor
forward of the receiver room (FLOW_LO x 64..80, z -6.8..-0.3, its bottom 0.6 over the funnel line + 2 at x 80), the two one-end landing
cases (LANDING_ONE_END), the boards' cuts on the run's grid (the design chat: the window's torn edges). Folded in if it passes; deleted after."""
import parts.scout_v4_cage_variant as V

F = V.apply("fork", 130.0)
F.LATCH_V7, F.FLOW_LO, F.LANDING_ONE_END = True, (64.0, -6.8), True
SPEC = dict(F.SPEC, gen_cm3=130.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vl"
    return p
