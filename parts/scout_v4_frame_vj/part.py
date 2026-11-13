"""TEMPORARY variant of parts/scout_v4_frame (13 Nov round 7b, Spark: "if adding a rod to the bottom of the motors can prevent snapping we
should do that instead of adding more material"): the bolt forks (parts/scout_v4_cage_variant.py line "fork") + an under-arm brace per
motor (frame.BRACES: a 3.0 carbon rod, modelled as a Part.insert with its real E A, pinned 2 short of each socket floor) at gen 70.0.
Folded in if it is the lightest pass; deleted after."""
import parts.scout_v4_cage_variant as V

F = V.apply("fork", 70.0)
F.BRACES = True
SPEC = dict(F.SPEC, gen_cm3=70.0)


def setup(h):
    p = F.setup(h)
    p.name = "scout_v4_frame_vj"
    return p
