#!/usr/bin/env python3
"""Prove a sliced plate really prints non-planar: Z moves DURING extrusion, and how deep through each object (not a top skin).
    python tools/np_zcheck.py print/nonplanar/<plate>/plate_1.gcode
Per object half of the bed (split at y 90 by default): extruding moves, share with Z changing inside the move, and the layer-height
band (Z range among extruding moves of one layer) at the mid height of the part, i.e. whether the interior layers are curved."""
import re, sys, numpy as np

g = open(sys.argv[1], encoding="utf8", errors="replace").read().splitlines()
split = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
x = y = z = e = 0.0
rows = []
for ln in g:
    if not ln.startswith("G1") and not ln.startswith("G0"):
        continue
    w = dict((t[0], t[1:]) for t in ln.split(";")[0].split()[1:] if len(t) > 1)
    nx = float(w["X"]) if "X" in w else x; ny = float(w["Y"]) if "Y" in w else y; nz = float(w["Z"]) if "Z" in w else z
    ne = float(w["E"]) if "E" in w else 0.0
    if ln.startswith("G1") and ne > 0 and ("X" in w or "Y" in w):
        rows.append((x, y, z, nx, ny, nz))
    x, y, z = nx, ny, nz
a = np.array(rows)
print("extruding moves:", len(a))
for name, sel in (("y<%g" % split, a[:, 4] < split), ("y>=%g" % split, a[:, 4] >= split)):
    r = a[sel]
    dz = np.abs(r[:, 5] - r[:, 2])
    ex = dz > 0.005
    zmax = r[:, 5].max()
    mid = (r[:, 5] > 0.3 * zmax) & (r[:, 5] < 0.7 * zmax)
    print("%-8s moves %7d  Z changes during %5.1f%%  max |dz| per move %.3f mm  Z range %.2f..%.2f" % (name, len(r), 100 * ex.mean(), dz.max(), r[:, 5].min(), zmax))
    print("         in the mid-height band (30-70%% of the part): %d moves, %.1f%% with Z changing" % (mid.sum(), 100 * ex[mid].mean() if mid.any() else 0))
    # the interior: Z spread of moves that print on the same plane i.e. along x at fixed y -> Z range across x for a mid layer
    zz = r[mid][:, 5]
    print("         mid-band Z spread along the part (layers are curved when this exceeds a layer): %.2f mm" % (zz.max() - zz.min()))
