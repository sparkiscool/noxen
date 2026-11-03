"""The shared landing funnel, drawn (3 Nov, Spark: a funnel-like base both drones sit into, centred by shape, the battery over the hole,
one landed height for both). It centres on the latch footprint, the one thing both drones carry in the same place: the receiver v3,
105.6 x 65.6 at the belly, with the cartridge hanging under it. Both drones therefore seat at the same height by construction
(proposal of 3 Nov, replacing the two-tier ring funnel; numbers E until a drop sim).
Seat frame = drone coordinates of a seated drone (x nose, y left, z up; the catch's SEAT_TO_W puts it in the box: seat x -> W Y,
seat y -> W -X, z 0 -> W Z 155.3). The receiver bottom is at z -44.3, so the seat ledge is there; the hole under it passes the
cartridge (96 x 40) and the swap head; four 55 deg faces rise 70 to the mouth, 204 x 164, for the +-40 capture; a 12 rim at the
mouth and four legs down to the ring frames' top (W Z 120 = seat z -35.3). Wall 3.0, PETG. Not drawn yet: the hold-down against the
60 N cartridge insert (four plungers into the receiver's lead-ins, concept), UHMW tape on the faces (coupon first).
Run: python parts/box_funnel/make.py  (writes part.stl in the seat frame; systems/box.assembly.json poses it with SEAT_TO_W)
"""
import os, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
SEAT_Z, FACE_DEG, H, WALL = -44.3, 55.0, 70.0, 3.0
SEAT = (-51.3, 54.3, -32.8, 32.8)                 # receiver v3 outline (parts/receiver_v3_latch/part.stl bounds, M)
HOLE = (-48.5, 51.5, -23.0, 23.0)                 # the cartridge 96 x 40 + 2.5 a side, centred with the receiver
RIM, LEG, LEG_BOTTOM = 12.0, 12.0, -35.3


def man(tri):
    return mf.Manifold(mf.Mesh(np.asarray(tri.vertices, np.float32), np.asarray(tri.faces, np.uint32)))
def hull(pts):
    return man(trimesh.convex.convex_hull(np.array(pts, float)))
def frustum(x0, x1, y0, y1, z0, z1, grow):
    """the solid between a rectangle at z0 and the same rectangle grown by `grow` on every side at z1"""
    return hull([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0 - grow, y0 - grow, z1), (x1 + grow, y0 - grow, z1), (x1 + grow, y1 + grow, z1), (x0 - grow, y1 + grow, z1)])
def box(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def funnel():
    g = H / np.tan(np.radians(FACE_DEG))                                           # 49.0: the mouth grows this much a side
    x0, x1, y0, y1 = SEAT; z0, z1 = SEAT_Z, SEAT_Z + H
    inner = frustum(x0, x1, y0, y1, z0, z1 + 1, g * (H + 1) / H)
    w = WALL / np.sin(np.radians(FACE_DEG))                                        # the horizontal offset of a 3.0 wall on a 55 deg face
    outer = frustum(x0 - w, x1 + w, y0 - w, y1 + w, z0 - WALL, z1, g)
    rim = box(x0 - g - w - RIM, x1 + g + w + RIM, y0 - g - w - RIM, y1 + g + w + RIM, z1 - 4.0, z1)
    skirt = box(HOLE[0] - WALL, HOLE[1] + WALL, HOLE[2] - WALL, HOLE[3] + WALL, z0 - 12.0, z0)
    legs = [box(cx - LEG / 2, cx + LEG / 2, cy - LEG / 2, cy + LEG / 2, LEG_BOTTOM, z1)
            for cx in (x0 - g - w - RIM / 2, x1 + g + w + RIM / 2) for cy in (y0 - g - w - RIM / 2, y1 + g + w + RIM / 2)]
    hole = box(*HOLE, z0 - 20.0, z1 + 2.0)
    body = mf.Manifold.batch_boolean([outer, rim, skirt] + legs, mf.OpType.Add) - inner - hole
    return body


if __name__ == "__main__":
    m = funnel().to_mesh()
    t = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    t.export(os.path.join(HERE, "part.stl"))
    print(f"funnel: extents {t.extents.round(1).tolist()} min {t.bounds[0].round(1).tolist()} max {t.bounds[1].round(1).tolist()}, "
          f"{t.volume * 1.27e-3:.0f} g PETG, faces {len(t.faces)}, watertight {t.is_watertight}")
