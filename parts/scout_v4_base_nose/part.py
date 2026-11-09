"""Scout v4 base, nose unit (9 Nov): see parts/scout_v4_base_lib.py (the base split at x = BX into a nose and a tail unit to fit the A1 mini)."""
from parts.scout_v4_base_lib import setup as _setup

SPEC = dict(h=1.5, gen_cm3=20.0, max_worst=0.3)   # E: the base room's 62 g estimate split in two, at the nodes' PCTG limit 0.3


def setup(h):
    return _setup("nose", h)
