"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 7, the design chat: the architecture review's "bolt forks" as the primary line;
folded into the frame, the shape tool and the nodes if it is the lightest pass; deleted after). This one: line "fork"
(parts/scout_v4_cage_variant.py: NB / TB / S0 / S1 at z 2.7 on horizontal forks of two 3.0 rods to sockets at z 11 beside the receiver's
4 bolts; the 8 D / B rods gone; the frame model on the core's real CG, the node parts once) at gen 100.0."""
import parts.scout_v4_cage_variant as V

F = V.apply("fork", 100.0)
SPEC = dict(F.SPEC, gen_cm3=100.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vg"
    return p
