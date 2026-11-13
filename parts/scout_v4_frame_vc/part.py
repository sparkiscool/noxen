"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 6, the design chat: both cage lines at once, the lighter frame passing every case
(< 0.3 both ways, the solid check, the frame model on the frame's own mass) is folded into the frame, the shape tool and the nodes; these
folders and parts/scout_v4_cage_variant.py deleted after). This one: line "split" (parts/scout_v4_cage_variant.py) at gen 100.0."""
import parts.scout_v4_cage_variant as V

F = V.apply("split", 100.0)
SPEC = dict(F.SPEC, gen_cm3=100.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vc"
    return p
