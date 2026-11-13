"""TEMPORARY variant of parts/scout_v4_node_nb (13 Nov round 6b: the node on cage line "tube2", parts/scout_v4_cage_variant.py, for the frame
at gen 260; its tube members as 6.2 x 15+ sockets; budget toward 0.25). Folded into parts/scout_v4_node_lib.py with the winning line."""
import parts.scout_v4_cage_variant as V

setup, SPEC = V.node("tube2", "NB", 260.0, 12.0)
