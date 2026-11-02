"""Head socket adapter (Spark 2 Nov: for tube 16-17 and its mirror 15-17, built as 3 mm rods; 4 needed, one in each end, the head
mount and nodes 15 / 16 both have 6.2 x 15 tube sockets): a 3 mm carbon rod into a 6.2 x 15 tube socket (tubes 15/16/18/19-17,
head rods +1 / -1; parts/overseer_head_mount/part.py Sock). Drawn only, no topology run: python parts/head_socket_adapter/make.py
writes part.stl (one adapter; set the copies in the slicer). Each: a sleeve O6.0 x 15 (the socket's depth; 0.1 glue gap a side in the 6.2
bore) and a collar O10 x 4 that sits on the socket mouth, bore O3.2 right through (the 3 mm rod socket bore, cage_node_lib link),
0.6 x 45 deg lead-in at the rod end. Print collar down, bore vertical. The sleeve wall is 1.4 (under the 2 mm rule, Spark to
accept): the socket wall carries it, the collar is 3.4. Glue: sleeve into the socket first, then the rod into the sleeve."""
import os, numpy as np, trimesh, manifold3d as mf

SLEEVE_R, SLEEVE_L, COLLAR_R, COLLAR_L, BORE_R, LEAD = 3.0, 15.0, 5.0, 4.0, 1.6, 0.6
N, PITCH = 1, 16.0                                                    # one per STL (the gate: one body); copies in the slicer

def one(x):
    body = mf.Manifold.cylinder(COLLAR_L, COLLAR_R, COLLAR_R, 64) + mf.Manifold.cylinder(SLEEVE_L, SLEEVE_R, SLEEVE_R, 64).translate([0, 0, COLLAR_L])
    bore = mf.Manifold.cylinder(COLLAR_L + SLEEVE_L + 2.0, BORE_R, BORE_R, 32).translate([0, 0, -1.0])
    lead = mf.Manifold.cylinder(LEAD + 0.01, BORE_R + LEAD, BORE_R, 32).translate([0, 0, -0.01])      # the rod goes in at the collar
    return (body - bore - lead).translate([x, 0, 0])

m = one(0.0)
for k in range(1, N):
    m += one(k * PITCH)
g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "part.stl"); t.export(out)
print(f"{N} adapters, {abs(t.volume) / N:.0f} mm3 = {abs(t.volume) / N * 1.27e-3:.2f} g each, watertight {t.is_watertight}")
