"""Scout cage node P2 at (-78.69999999999999, 115.0, 58.0): the socket collars of every rod and tie that ends here, the rest generated
in a strut round each member and a hub (8 Nov: single again, MERGE_MM 45). Geometry, loads and the other nodes: parts/scout_node_lib.py."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make('P2')
SPEC = spec_for('P2')
