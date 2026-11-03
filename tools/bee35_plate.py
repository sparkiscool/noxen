"""The Scout's Bee35 main plate traced from SpeedyBee's own render (Spark 3 Nov: the carbon base is the SpeedyBee middle plate, make it
accurate; no CAD is published). Source: SpeedyBee-Bee35-Manual-EN.pdf page 3 (the main plate alone, perspective view, RGBA raster
1253 x 705 with the background transparent). The plate is a plane, so a homography from its four motor-pad centres to the known motor
positions (+-54.1, +-54.1 about the thrust centre, dimensions.md C) rectifies the silhouette exactly; the alpha channel carries the
holes. The camera mount sits on the front-left in the render, so the right half (y < 0) is traced and mirrored. Thickness 3.5,
underside z 18.5 (dimensions.md C). Residual errors: the plate's visible side faces add up to ~1 mm on the edges that faced the
camera, and the four standoff posts near the rear arms are above the plane. Tag E for the outline, C for the holes it carries.
Run: python tools/bee35_plate.py <render.png>  (writes parts/frame_plate_bee35_carbon/part.stl)
"""
import os, sys
import numpy as np, trimesh
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure
from shapely.geometry import Polygon, Point
from skimage.morphology import disk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z0, T = 18.5, 3.5
PX = 0.25                                                                            # mm per rectified pixel
PADS_IMG = {"FL": (209.6, 370.4), "FR": (658.5, 153.0), "RR": (1053.6, 390.2), "RL": (600.8, 631.4)}   # the pads' centre holes in the render
PADS_MM = {"FL": (54.1, 54.1), "FR": (54.1, -54.1), "RR": (-54.1, -54.1), "RL": (-54.1, 54.1)}


def homography(src, dst):
    """3x3 H with dst ~ H src (DLT on 4 points)"""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u]); A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.array(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def rectify(png):
    a = np.array(Image.open(png)); mask = a[..., 3] > 128
    keys = list(PADS_IMG)
    H = homography([PADS_IMG[k] for k in keys], [PADS_MM[k] for k in keys])
    Hi = np.linalg.inv(H)
    xs = np.arange(-110, 110, PX); ys = np.arange(-90, 90, PX)
    X, Y = np.meshgrid(xs, ys)                                                        # rows = y (mm), cols = x (mm)
    P = np.stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    Q = Hi @ P; u, v = Q[0] / Q[2], Q[1] / Q[2]
    inside = (u >= 0) & (u < mask.shape[1] - 1) & (v >= 0) & (v < mask.shape[0] - 1)
    M = np.zeros(X.size, bool)
    M[inside] = ndi.map_coordinates(mask.astype(float), [v[inside], u[inside]], order=1) > 0.5
    M = M.reshape(X.shape)
    right = Y < 0                                                                       # the clean half (no camera mount); mirrored over y
    Mr = np.where(right, M, False)
    Mm = Mr | Mr[::-1, :]
    Mm &= ~((X > NOSE_X) & (np.abs(Y) < 42.0))         # the camera mount's lobes at the nose (its right plate is in the traced half)
    Mm = ndi.binary_opening(Mm, structure=disk(3))      # the standoff posts' spikes on the body sides (above the plane)
    Mm = ndi.binary_closing(Mm, iterations=1)
    lab, n = ndi.label(Mm); sizes = ndi.sum(Mm, lab, range(1, n + 1)); Mm = lab == (np.argmax(sizes) + 1)
    return Mm, xs, ys, H


NOSE_X = 56.0                                            # the plate under the camera mount is hidden: solid to a straight nose edge here (E)
HOLES = ([(mx, my, 2.5) for mx, my in PADS_MM.values()]                                                          # motor bell clearance
         + [(mx + 8.49 * c, my + 8.49 * s, 1.1) for mx, my in PADS_MM.values() for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))]   # 12 x 12 M2 along the arm
         + [(1.9 + 18.03 * c, 18.03 * s, 1.5) for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))]                   # AIO 25.5 diamond at (1.9, 0), O3
         + [(3.9, 31.0, 1.6), (3.9, -31.0, 1.6), (36.0, 0.0, 1.6)])                                             # the 3 plate bolts the underside uses (C / E nose)


def polygon(Mm, xs, ys):
    pad = np.pad(Mm, 2)
    cs = measure.find_contours(pad.astype(float), 0.5)
    rings = []
    for c in cs:
        pts = np.stack([xs[0] + (c[:, 1] - 2) * PX, ys[0] + (c[:, 0] - 2) * PX], 1)
        if len(pts) >= 4:
            rings.append(Polygon(pts).buffer(0))
    rings.sort(key=lambda p: p.area, reverse=True)
    outer = rings[0]
    holes = [r for r in rings[1:] if outer.contains(r.representative_point()) and r.area > 1.0]
    poly = outer
    for h in holes:
        poly = poly.difference(h)
    for x, y, r in HOLES:                                 # the small holes the raster cannot resolve, from their known numbers
        poly = poly.difference(Point(x, y).buffer(r, 16))
    return poly.simplify(0.15), len(holes)


if __name__ == "__main__":
    png = sys.argv[1]
    Mm, xs, ys, H = rectify(png)
    out = os.path.splitext(png)[0] + "_rect.png"
    Image.fromarray((~Mm[::-1, :] * 255).astype(np.uint8)).save(out)                  # y up, x right, black = plate
    poly, nh = polygon(Mm, xs, ys)
    m = trimesh.creation.extrude_polygon(poly, T); m.apply_translation([0, 0, Z0])
    d = os.path.join(ROOT, "parts", "frame_plate_bee35_carbon"); os.makedirs(d, exist_ok=True); m.export(os.path.join(d, "part.stl"))
    b = m.bounds
    print(f"plate: area {poly.area:.0f} mm2, holes {nh}, x {b[0][0]:.1f}..{b[1][0]:.1f}, y {b[0][1]:.1f}..{b[1][1]:.1f}, "
          f"faces {len(m.faces)}, watertight {m.is_watertight}, mass at 1.55 g/cm3: {poly.area * T * 1.55e-3:.1f} g; mask {out}")
