"""Swap arm v8, lift (13 Nov 2026). Setup in parts/box_arm8_setups.py (lift): drawn attachments, cuts, loads from the head's cases."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from box_arm8_setups import lift as setup   # noqa: F401  (setup(h) -> Part)
SPEC = dict(h=1.8, gen_cm3=104.0, max_worst=0.5)
