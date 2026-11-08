"""Overseer front arm motor PAD, solid v2 (5 Nov, Spark: the printed pads cracked at the motor base, printed flat with the layers across
the thrust; demo tomorrow: no topology, a drawn solid, smooth not geometric, even more support round the motor plate). Arm 1 (+y);
arm 4 is the y-mirror (mirror in the slicer). python parts/overseer_arm_front_pad_solid/make.py writes part.stl, in the drone frame
(parts/arm_solid_lib.py: a blurred distance field meshed, then the arm lib's exact lugs added and its cuts subtracted).
* The old pad stays glued; its 4 rods (the two 6/4 tubes, spokes 1.1 and 1.3) are cut flush behind it, one socket shorter. This pad
  sits where the old one was (motor at motors_xy[0], pad top z 45) and its sockets reach one socket length further along the rods:
  tubes 17 toward the root (bore O6.2, 21 long, at z 14 / 36), spokes 12 further from the arm node (bore O3.2, 13 long).
* Slide-on from the outboard side: every socket is a channel open across the arm toward +n (outboard; a vertical drop cannot work,
  the two tubes stack on one vertical line so the upper slot would cut the lower socket open). The pad is pushed inboard onto the 4
  cut rod ends (tubes a 5.8 light press, spokes a 2.8 snap) and glued (resin grooves as the nodes'); no rod comes out. Thrust and
  bending load the closed walls; the open side is across the arm.
* Shape: a rounded 22 x 37 block along the arm over both tube sockets (s len-62..len-16, z 5..45), the motor disc O36 x 8 on a
  bell under it (O36 at the pad down to O32 at z 14, outside the driver sweeps) blended into the block, each spoke collar (O10.4)
  on a smooth capsule back into the block: one body; the lugs, slots (r 8..9.5), head seats, driver sweeps and O8.6 shaft bore cut
  as the lib draws them, the cage members near it cut clear.
* Print on its side (the block's flat side on the bed, the motor disc's axis horizontal): layers along the arm and through the pad,
  thrust and bending in-layer. PETG, 5 walls, 50 % gyroid, 0.2 layers (half the time of solid, in-layer loads)."""
import os, sys, numpy as np, manifold3d as mf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from parts.overseer_tube_arm_lib import TubeArm, TUBE_Z, TUBE_DEPTH
from parts.overseer_arm_lib import Z_PAD, along, U, NODE_R, SOCK_DEPTH, SOCK_BORE_R
from parts.arm_solid_lib import grid, rbox, cylinder, bell, capsule, softmax, mesh, export

EXT_TUBE, EXT_SPOKE = 17.0, 12.0
TUBE_GAP, SPOKE_GAP, DROP = 5.8, 2.8, 60.0                              # channel widths: a light press on the 6.0 tube, a snap on the 3.0 rod
PAD_T, BELL_Z0, BELL_R0 = 8.0, 14.0, 16.0
BLOCK = (23.0, 11.0, 20.0)                                              # half sizes along, across, z (centre s L-39, z 25: bottom z 5, 4.9 under the lower tube's tunnel)


class FrontPad(TubeArm):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.pad_mouth -= EXT_TUBE                                         # tube sockets 17 closer to the root
        self.node_r = NODE_R + EXT_SPOKE                                   # spoke sockets 12 further out from the node (arm lib hook)


SLIDE = None                                                            # set below: outboard across the arm (+n), the way the pad goes on


def trough(P, u, s0, s1, w, drop):
    """the socket's channel: the bore's width w swept from the rod axis out along the slide direction (as projected perpendicular to the
    rod), so the rod enters the socket moving along SLIDE; a rod sliding partly along its own axis just runs in its bore"""
    u = np.asarray(u, float); d = SLIDE - np.dot(SLIDE, u) * u; d /= np.linalg.norm(d); n = np.cross(u, d)
    return mf.Manifold.cube([s1 - s0, w, drop]).translate([s0, -w / 2, 0.0]).transform(np.c_[u, n, d, np.asarray(P, float)])


ARM = FrontPad("front", 45.0, 0, ["spoke 1.1", "spoke 1.3"], [178.94, 146.7, 8.0], "pad")
drawn, cuts, env = ARM.drawn_cuts_env()
a3 = ARM.a3; n3 = np.array([-a3[1], a3[0], 0.0]); R3 = np.array([ARM.R[0], ARM.R[1], 0.0]); L = ARM.len; M = ARM.M
SLIDE = n3                                                              # +n: outboard (toward +y on arm 1), free space
node = np.asarray(ARM.node, float)
ends = [node + np.asarray(sp["u"]) * (ARM.node_r + SOCK_DEPTH + 4.0) for sp in ARM.spokes]
lo = np.minimum.reduce([R3 + a3 * (L - 70.0) - 30, [M[0] - 25, M[1] - 25, 0.0]] + [e - 12 for e in ends])
hi = np.maximum.reduce([R3 + a3 * (L - 10.0) + 30, [M[0] + 25, M[1] + 25, Z_PAD + 2]] + [e + 12 for e in ends])
ax, P = grid(lo, hi)
block_c = R3 + a3 * (L - 39.0) + [0, 0, 5.0 + BLOCK[2]]
fields = [rbox(P, block_c, a3, n3, BLOCK, r=5.0), cylinder(P, M, Z_PAD - PAD_T, Z_PAD, 18.0), bell(P, M, BELL_Z0, BELL_R0, Z_PAD - PAD_T + 0.5, 18.0)]
anchor = R3 + a3 * (L - 24.0) + [0, 0, 14.0]
for sp, e in zip(ARM.spokes, ends):
    s0 = node + np.asarray(sp["u"]) * (ARM.node_r - 6.0)
    fields.append(capsule(P, s0, e, 5.2))                                  # the spoke collar
    fields.append(capsule(P, 0.5 * (s0 + e), anchor, 6.0))                 # its smooth strut into the block
body = mesh(ax, softmax(fields), z1=Z_PAD)               # exactly flat at the pad top
slots = []
for z in TUBE_Z:
    B = ARM.p(ARM.pad_mouth + TUBE_DEPTH, 0.0, z)
    slots += [trough(B, -a3, -2.0, TUBE_DEPTH + 7.0, TUBE_GAP, DROP), along(B, -a3, -4.0, TUBE_DEPTH + 7.0, 3.1)]
for sp in ARM.spokes:
    slots += [trough(node, sp["u"], ARM.node_r - 8.0, ARM.node_r + SOCK_DEPTH + 16.0, SPOKE_GAP, DROP),     # through both collar ends
              along(node, sp["u"], ARM.node_r - 8.0, ARM.node_r + SOCK_DEPTH + 16.0, SOCK_BORE_R),
              along(node, sp["u"], ARM.node_r + SOCK_DEPTH - 0.2, ARM.node_r + SOCK_DEPTH + 60.0, 7.0)]   # a flat collar end (the lib's O4.4 rod tunnel left a sliver in the rounded cap)
m = (body + drawn) - cuts - U(slots)
export(m, os.path.join(HERE, "part.stl"))
