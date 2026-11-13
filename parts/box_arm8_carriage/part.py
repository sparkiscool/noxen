"""Swap arm v8, carriage (13 Nov 2026). Setup in parts/box_arm8_setups.py (carriage): drawn attachments, cuts, loads from the head's cases."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from box_arm8_setups import carriage as setup   # noqa: F401  (setup(h) -> Part)
SPEC = dict(h=1.6, gen_cm3=50.0, max_worst=0.5)
