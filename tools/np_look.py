"""(Design Agent p1158, inbox 905, Spark 23 Nov 23:04: "they need to be the same shape just the layers ... curved ... you
keep sending a corners cut off version still layer by layer")

A non-planar plate is sliced from the un-bent mesh (tools/nonplanar.py), so the pictures Bambu Studio puts in the
.gcode.3mf (the printer's screen, Bambu Handy, the plate thumbnail) showed that un-bent block, not the part that
prints. The g-code was right (tools/np_compare.py: same volume and outline), the picture was not. thumbs() redraws
those pictures from the real part meshes, so what he sees is the shape that prints. pick_1.png (the colour-coded
object picker) is left as it is.

Drawn with a small depth-buffer rasterizer (numpy + Pillow only), so hidden faces never show through.

    python tools/np_look.py <file.gcode.3mf> <part.stl> [x y]   redraw the pictures from a part placed at x, y (plate centre)
"""
import io, os, sys, zipfile
import numpy as np

PICS = ("Metadata/plate_1.png", "Metadata/plate_1_small.png", "Metadata/plate_no_light_1.png", "Metadata/top_1.png")


def _draw(meshes, color, size, elev, azim, light=True, ss=2):
    """PNG bytes: the meshes seen from elev/azim (degrees), orthographic, transparent background, part filling ~92%."""
    from PIL import Image
    tri = np.vstack([m.vertices[m.faces] for m in meshes]).astype(float)
    N = np.vstack([m.face_normals for m in meshes]).astype(float)
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])    # towards the camera
    r = np.array([-np.sin(a), np.cos(a), 0.0])                                 # screen right
    u = np.cross(d, r)                                                          # screen up
    keep = N @ d > 1e-6                                                         # faces turned to the camera
    tri, N = tri[keep], N[keep]
    c = np.array([int(color[i:i + 2], 16) / 255.0 for i in (1, 3, 5)])
    if light:
        L = np.array([-0.35, -0.55, 0.76]); L /= np.linalg.norm(L)
        k = 0.45 + 0.55 * np.clip(N @ L, 0, 1)
    else:
        k = np.ones(len(N))
    fc = np.clip(c[None, :] * k[:, None], 0, 1)
    W = size * ss
    sx, sy, z = tri @ r, tri @ u, tri @ d                                       # (n, 3) each
    lo = np.array([sx.min(), sy.min()]); hi = np.array([sx.max(), sy.max()])
    s = 0.92 * W / max((hi - lo).max(), 1e-9); mid = (lo + hi) / 2
    px = (sx - mid[0]) * s + W / 2.0; py = W / 2.0 - (sy - mid[1]) * s
    zbuf = np.full((W, W), -np.inf); img = np.zeros((W, W, 4))
    for i in range(len(tri)):
        xa, xb, xc = px[i]; ya, yb, yc = py[i]
        area = (xb - xa) * (yc - ya) - (yb - ya) * (xc - xa)
        if abs(area) < 1e-12:
            continue
        x0, x1 = max(int(np.floor(min(xa, xb, xc))), 0), min(int(np.ceil(max(xa, xb, xc))), W - 1)
        y0, y1 = max(int(np.floor(min(ya, yb, yc))), 0), min(int(np.ceil(max(ya, yb, yc))), W - 1)
        if x1 < x0 or y1 < y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((xb - X) * (yc - Y) - (yb - Y) * (xc - X)) / area
        w1 = ((xc - X) * (ya - Y) - (yc - Y) * (xa - X)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        zz = w0 * z[i, 0] + w1 * z[i, 1] + w2 * z[i, 2]
        zb = zbuf[y0:y1 + 1, x0:x1 + 1]
        up = inside & (zz > zb)
        zb[up] = zz[up]
        img[y0:y1 + 1, x0:x1 + 1][up] = (*fc[i], 1.0)
    im = Image.fromarray((img * 255).round().astype(np.uint8), "RGBA").resize((size, size), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()


def pictures(meshes, color="#F2754E"):
    """{3mf name: png bytes} of the real parts, drawn the way Bambu Studio draws its plate pictures."""
    iso = _draw(meshes, color, 512, 30, -60)
    from PIL import Image
    small = io.BytesIO(); Image.open(io.BytesIO(iso)).resize((128, 128), Image.LANCZOS).save(small, format="PNG")
    return {PICS[0]: iso, PICS[1]: small.getvalue(), PICS[2]: _draw(meshes, color, 512, 30, -60, light=False),
            PICS[3]: _draw(meshes, color, 512, 90, -90)}


def thumbs(path, meshes, color=None):
    """Replace the plate pictures in a sliced .gcode.3mf with drawings of the real parts. -> names replaced."""
    with zipfile.ZipFile(path) as zi:
        names = set(zi.namelist())
        if color is None:
            try:
                import json
                color = json.loads(zi.read("Metadata/plate_1.json")).get("filament_colors", ["#F2754E"])[0]
            except Exception:
                color = "#F2754E"
    pics = {k: v for k, v in pictures(meshes, color).items() if k in names}
    if not pics:
        return []
    tmp = path + ".look.tmp"
    with zipfile.ZipFile(path) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            zo.writestr(it, pics[it.filename]) if it.filename in pics else zo.writestr(it, zi.read(it.filename))
    os.replace(tmp, path)
    return sorted(pics)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    import trimesh
    m = trimesh.load(sys.argv[2], force="mesh")
    x, y = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (90.0, 90.0)
    m.apply_translation([x - m.bounds[:, 0].mean(), y - m.bounds[:, 1].mean(), -m.bounds[0, 2]])
    print("replaced:", thumbs(sys.argv[1], [m]))
