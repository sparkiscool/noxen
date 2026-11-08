"""Scout cage node KM0 at (0.0, 84.0, -48.0): the socket collars of every rod and tie that ends here, the rest generated
in a strut round each member and a hub (8 Nov: single again, MERGE_MM 45). Geometry, loads and the other nodes: parts/scout_node_lib.py."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make('KM0')
SPEC = spec_for('KM0')
