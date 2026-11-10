"""Scout v4 cage node R7 (9 Nov): the socket collars of every rod that ends here, the rest generated.
Geometry and loads: parts/scout_v4_node_lib.py (the v4 cage of tools/scout_v4_shape.py, its frame model's node loads)."""
from parts.scout_v4_node_lib import make, spec_for   # noqa: F401
setup = make("R7")
SPEC = spec_for("R7")
