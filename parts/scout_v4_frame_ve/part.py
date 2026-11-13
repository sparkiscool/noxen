"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 6b, the design chat: the guaranteed-pass line, the tube cage at higher budget,
the side rods BS / DS as 6/4 tube too; folded into the frame, the shape tool and the nodes if it is the lightest pass; deleted after).
This one: line "tube2" (parts/scout_v4_cage_variant.py) at gen 230.0."""
import parts.scout_v4_cage_variant as V

F = V.apply("tube2", 230.0)
SPEC = dict(F.SPEC, gen_cm3=230.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_ve"
    return p
