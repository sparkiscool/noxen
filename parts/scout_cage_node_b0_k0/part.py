"""Scout cage nodes B0 + K0, one printed structure (5 Nov, Spark: nodes closer than 60 mm are one part): the socket
collars of every rod and tie that ends at either node, a printed bridge where the rod B0-K0 was, the rest generated.
Geometry, loads and the other nodes: parts/scout_node_lib.py."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make(('B0', 'K0'))
SPEC = spec_for(('B0', 'K0'))
