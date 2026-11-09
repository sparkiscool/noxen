"""Scout v4 base, nose unit (9 Nov): see parts/scout_v4_base_lib.py (the base split at x = BX into a nose and a tail unit to fit the A1 mini)."""
from parts.scout_v4_base_lib import setup as _setup

SPEC = dict(h=1.5, gen_cm3=24.0, max_worst=0.3)   # 9 Nov: 20 cm3 stopped early at 1.67 on the DN / BN sockets (486 N); with the two bolts behind the FC, 24


def setup(h):
    return _setup("nose", h)
