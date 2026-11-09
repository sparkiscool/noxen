"""Scout cage node S0 at (0.0, 115.0, 12.0): the left side's knee (cage v21, the half cage, 8 Nov): the leg up to the ring's side mid,
the fan to the ring's two corners on that side, the two ties in from the arm-tip clips; the socket collars of every rod and tie that
ends here, the rest generated in a strut round each member and a hub. Geometry, loads and the other nodes: parts/scout_node_lib.py."""
from parts.scout_node_lib import make, spec_for   # noqa: F401
setup = make('S0')
SPEC = spec_for('S0')
