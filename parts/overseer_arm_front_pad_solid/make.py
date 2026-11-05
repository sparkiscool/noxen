"""Overseer front arm motor PAD, solid (5 Nov, Spark: the printed pads cracked at the motor base, printed flat with the layers across
the thrust; demo tomorrow, be safe and fast: no topology, a drawn solid part). Arm 1 (+y); arm 4 is the y-mirror (mirror in the
slicer). python parts/overseer_arm_front_pad_solid/make.py writes part.stl, in the drone frame.
* The old pad stays glued; its 4 rods (the two 6/4 tubes, spokes 1.1 and 1.3) are cut flush behind it, one socket shorter. This pad
  sits where the old one was (motor at motors_xy[0], pad top z 45) and its sockets reach one socket length further along the rods:
  tubes 17 toward the root (bore 21 long, O6.2, at z 14 / 36), spokes 12 further from the arm node (bore O3.2, 13 long).
* Drop-on: every socket is open downward (a slot narrower than the rod: 5.4 / 2.7) so the pad lowers onto the 4 cut rod ends from
  above, snaps over them and is glued (resin grooves as the nodes'); no rod comes out.
* Solid: a 20 x 37 block along the arm over both tube sockets (s len-62..len-16, z 8..45), the motor disc O36 x 6 on top, each
  spoke collar (O8.2) in a solid wedge back to the block, the arm lib's lugs, slots (r 8..9.5), head seats, driver sweeps and the
  O8.6 shaft bore, the cage members near it cut clear (r 4.5), prop disc clear.
* Print on its side (the block's flat side on the bed, the motor disc's axis horizontal): the layers run along the arm and through
  the pad, so thrust and bending are in-layer. PETG, 100 % infill, 5 walls, 0.2 layers.
Mass ~50 g (E, 1.27 g/cm3): the optimised pad was 23.5 g; the weight is the price of no lace and no wait."""
import os, sys, numpy as np, trimesh, manifold3d as mf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from parts.overseer_tube_arm_lib import TubeArm, TUBE_Z, TUBE_DEPTH
from parts.overseer_arm_lib import Z_PAD, PAD_T, cyl, along, U, NODE_R, SOCK_DEPTH, SOCK_BORE_R, SOCK_WALL

EXT_TUBE, EXT_SPOKE = 17.0, 12.0
TUBE_GAP, SPOKE_GAP, DROP = 5.4, 2.7, 60.0
BLOCK_V, BLOCK_Z0 = 10.0, 8.0


class FrontPad(TubeArm):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pad_mouth -= EXT_TUBE                                         # tube sockets 17 closer to the root
        self.node_r = NODE_R + EXT_SPOKE                                   # spoke sockets 12 further out from the node (arm lib hook)


def trough(P, u, s0, s1, w, drop):
    u = np.asarray(u, float); d = np.array([0, 0, -1.0]) - np.dot([0, 0, -1.0], u) * u; d /= np.linalg.norm(d); n = np.cross(u, d)
    return mf.Manifold.cube([s1 - s0, w, drop]).translate([s0, -w / 2, 0.0]).transform(np.c_[u, n, d, np.asarray(P, float)])


ARM = FrontPad("front", 45.0, 0, ["spoke 1.1", "spoke 1.3"], [178.94, 146.7, 8.0], "pad")
drawn, cuts, env = ARM.drawn_cuts_env()
a3 = ARM.a3; n3 = np.array([-a3[1], a3[0], 0.0]); R3 = np.array([ARM.R[0], ARM.R[1], 0.0]); L = ARM.len; M = ARM.M


def beam(s0, s1, v0, v1, z0, z1):
    return mf.Manifold.cube([s1 - s0, v1 - v0, z1 - z0]).translate([s0, v0, z0]).transform(np.c_[a3, n3, [0, 0, 1.0], R3])


body = [beam(L - 62.0, L - 16.0, -BLOCK_V, BLOCK_V, BLOCK_Z0, Z_PAD), cyl(M[0], M[1], Z_PAD - PAD_T, Z_PAD, 18.0, 64),
        beam(L - 30.0, L + 4.0, -14.0, 14.0, Z_PAD - PAD_T - 8.0, Z_PAD)]                      # block, motor disc, boss under it
for sp in ARM.spokes:                                                                            # spoke collar + solid wedge to the block
    col = along(ARM.node, sp["u"], ARM.node_r - 6.0, ARM.node_r + SOCK_DEPTH + 4.0, 5.2)       # O10.4 collar (wall 3.6: the slot lips stay over 2)
    body.append(mf.Manifold.batch_hull([col, beam(L - 34.0, L - 16.0, -BLOCK_V, BLOCK_V, BLOCK_Z0, BLOCK_Z0 + 12.0)]))
slots = []
for z in TUBE_Z:
    slots.append(trough(ARM.p(ARM.pad_mouth + TUBE_DEPTH, 0.0, z), -a3, -2.0, TUBE_DEPTH + 7.0, TUBE_GAP, DROP))
    slots.append(along(ARM.p(ARM.pad_mouth + TUBE_DEPTH, 0.0, z), -a3, -4.0, TUBE_DEPTH + 7.0, 3.1))   # the bore, 21 + long (no floor)
for sp in ARM.spokes:
    slots.append(trough(ARM.node, sp["u"], ARM.node_r - 3.0, ARM.node_r + SOCK_DEPTH + 3.0, SPOKE_GAP, DROP))
    slots.append(along(ARM.node, sp["u"], ARM.node_r - 3.0, ARM.node_r + SOCK_DEPTH + 3.0, SOCK_BORE_R))
m = (U(body) + drawn) - cuts - U(slots)
g = m.to_mesh(); t = trimesh.Trimesh(g.vert_properties[:, :3], g.tri_verts); t.merge_vertices()
parts = t.split(only_watertight=False); t = max(parts, key=lambda q: abs(q.volume)) if len(parts) > 1 else t
t.export(os.path.join(HERE, "part.stl"))
b = t.bounds
print(f"{abs(t.volume) / 1000:.1f} cm3 = {abs(t.volume) * 1.27e-3:.0f} g, bodies {len(parts)}, watertight {t.is_watertight}, box {np.round(b[1] - b[0], 1)}")
