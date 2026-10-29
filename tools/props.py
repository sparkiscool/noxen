"""Prop meshes for the Overseer page: a two-blade 10 x 4.5 prop at each motor (29 Oct, Spark: "actual props that don't take up all
the view"; was a solid disc r 127 x 10 at the thrust plane). The swept disc stays on the page as the thin-line overlay
(systems/overseer.assembly.json overlays.prop_r) and is what the cage must clear; these blades are the look, not the keep-out.
Geometry (E): hub O14 x 9 with the M5 adapter bore, blades r 10..127, chord 16 -> 24 -> 9 mm, thickness 3.2 -> 1.2 mm, pitch
114 mm (4.5 in) twist, flat-bottomed sections. Written in the drone frame at the motors (systems/overseer.cage.json motors_xy,
prop plane z 82, dimensions.md O layout rows), blades along the arm's own diagonal so a nose-on view sees them edge-on.
Run: python tools/props.py
"""
import json, os
import numpy as np, trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_TIP, R_ROOT, PITCH_MM, Z_PROP = 127.0, 10.0, 114.3, 82.0
HUB_D, HUB_H, BORE_D = 14.0, 9.0, 5.2


def chord(r):
    t = (r - R_ROOT) / (R_TIP - R_ROOT)
    return 16.0 + 8.0 * np.sin(np.pi * min(t / 0.45, 1.0) / 2) * (1.0 if t < 0.45 else 1.0) - (15.0 * max(0.0, (t - 0.45) / 0.55) ** 1.5)


def thick(r):
    t = (r - R_ROOT) / (R_TIP - R_ROOT)
    return 3.2 - 2.0 * t


def blade(n=26, m=16):
    """one blade along +x of the hub: lofted sections, each a flat-bottomed lens of chord x thickness, twisted by the pitch angle"""
    rs = np.linspace(R_ROOT, R_TIP, n)
    rings = []
    for r in rs:
        c, th = chord(r), thick(r)
        beta = np.arctan(PITCH_MM / (2 * np.pi * r))
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        y = 0.5 * c * np.cos(a)                                     # chordwise
        z = np.where(np.sin(a) > 0, 0.5 * th * np.sin(a), 0.15 * th * np.sin(a))   # cambered top, flat bottom
        cb, sb = np.cos(beta), np.sin(beta)
        yy, zz = y * cb - z * sb, y * sb + z * cb                   # twist about the radial axis
        rings.append(np.stack([np.full(m, r), yy, zz], 1))
    V = np.vstack(rings); F = []
    for i in range(n - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c_, d = (i + 1) * m + j, (i + 1) * m + (j + 1) % m
            F += [[a, c_, b], [b, c_, d]]
    # cap the ends with a fan
    for i, flip in ((0, True), (n - 1, False)):
        cen = len(V); V = np.vstack([V, rings[i].mean(0)])
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            F.append([cen, b, a] if flip else [cen, a, b])
    return trimesh.Trimesh(V, np.array(F), process=True)


def prop(theta_deg):
    hub = trimesh.creation.cylinder(HUB_D / 2, HUB_H, sections=32)
    bore = trimesh.creation.cylinder(BORE_D / 2, HUB_H + 2, sections=24)
    b1 = blade(); b2 = blade().apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    m = trimesh.util.concatenate([hub, b1, b2])
    try:
        m = trimesh.boolean.difference([trimesh.util.concatenate([hub, b1, b2]), bore], engine="manifold")
    except Exception:
        pass
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(theta_deg), [0, 0, 1]))
    return m


def main():
    cage = json.load(open(os.path.join(ROOT, "systems", "overseer.cage.json")))
    for k, (x, y) in enumerate(cage["motors_xy"]):
        th = np.degrees(np.arctan2(y, x))                          # blades along the arm diagonal
        m = prop(th); m.apply_translation([x, y, Z_PROP])
        out = os.path.join(ROOT, "parts", f"prop_{k + 1}", "part.stl"); m.export(out)
        print(f"prop {k + 1} at ({x:.1f}, {y:.1f}, {Z_PROP}) blades at {th:.0f} deg: {len(m.faces)} faces, {m.extents.round(0).tolist()} -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
