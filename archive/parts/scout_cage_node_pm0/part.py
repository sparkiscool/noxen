"""Scout cage node PM0 at (0.0, 115.0, 58.0): the ring's side mid node (cage v21, 8 Nov: the waist hoop's foot and the side leg's top), the socket collars
of every rod that ends here, the rest generated in a strut round each member and a hub. Geometry, loads and the other nodes: parts/scout_node_lib.py."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make('PM0')
SPEC = spec_for('PM0')
