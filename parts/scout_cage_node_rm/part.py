"""Scout cage node RM at (0, 0, 70.0) (4 Nov): the socket collars of every rod and tie that ends here, the rest generated.
Geometry, loads and the other nodes: parts/scout_node_lib.py (after the Overseer's parts/cage_node_lib.py)."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make("RM")
SPEC = spec_for("RM")
