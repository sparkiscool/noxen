"""Scout cage nodes BM0 + KM0, one printed structure (5 Nov, Spark: nodes closer than 60 mm are one part): the socket
collars of every rod and tie that ends at either node, a printed bridge where the rod BM0-KM0 was, the rest generated.
Geometry, loads and the other nodes: parts/scout_node_lib.py."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make(('BM0', 'KM0'))
SPEC = spec_for(('BM0', 'KM0'))
