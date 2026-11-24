"""Slicing comparison bar (Spark 23 Nov: two bars of the SAME shape, one sliced planar, one non-planar, to compare the slicing).
Drawn: python parts/np_bar/make.py writes parts/np_bar/part.stl and print/np_bar.stl. A test piece, not a flight part.
Shape: a plain rectangular bar L 100 x W 25 x T 10, flat on the bed, no holes, no overhang, no support. Print it twice from one plate: once planar,
once with the non-planar slicer (wave layers). Hand calc: 10 cm x 2.5 cm x 1 cm = 25 cm3 solid, ~31 g PCTG each. Fit table: nothing to fit."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trimesh
from scout_v4_probe_mech_lib import save  # noqa: E402
m = trimesh.creation.box(extents=[100.0, 25.0, 10.0])
m.apply_translation([0, 0, 5.0])
save("np_bar", m, m.copy(), "L 100 W 25 T 10, flat face down, no support; print one planar, one non-planar")
