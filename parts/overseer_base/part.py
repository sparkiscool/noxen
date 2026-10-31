"""Overseer base (v1 setup, 29 Oct; Spark: "for the base of the drone, the battery part, make it so I can screw on whatever design I
come up with in the future with 4 M3 sinks and screws; build it now with all the nodes and attach the battery mechanism after; as
compact as possible, enough room to attach the design after").

One generated PETG piece under the bottom plate (plate z -2.1..0, x -95.5..100.5, y +-59) that ends the ten cage links whose hub
ends had no part: core ties rear.1 / rear.2 (hub ends (-95.5, +-12, -1), behind the plate), left.1 / left.2 and right.1 / right.2
(hub ends (0, +-59, -1), at the plate's side edges) and the four bay ties (hub ends (30 / -90, +-56, -9.0); -5.1 until 30 Oct). Each is a coupon
socket as the cage nodes' (parts/cage_node_lib: bore 3.2 x 10 from 10.15 past the virtual centre, resin grooves, lead-in, crown,
wall 2.5, floor 3.0), so the rod ends and cut list of systems/overseer.cage.json do not move: nothing else reruns.
Drawn = attachments only: the 10 socket collars; 8 pads O9 x 6 under the plate's arm bottom screws (S500 STEP (+-52.14, +-38),
(+-38, +-52.14), O3.0 in the plate, "clear below the plate"): the arm screws grow from M3 x 8 to M3 x 14 and clamp the base under the
plate, no new holes in the carbon; 4 insert bosses O11 x 6 for the future battery mechanism, M3 x 5.7 heat-set inserts in O4.0
holes from below, faces on z -8.1 (the latch set's receiver mount plane: plate underside - 6), pattern 80 x 50 centred on the pack's
x (-20): (-60, +-25), (20, +-25). Below z -8.1 inside the slot (x -121.8..61.9, y +-33.4) nothing is drawn or generated: that is the
future design's room. The topology joins collars, pads and bosses in the 6 mm under the plate and outside it (the collars beside
and behind the plate rise to z +4 / +10, where nothing else is: the arms sit on the plate's corners).
Loads (frame model systems/overseer.cage_loads.json, as the hub deck's nose): every crash case's force on each of the 10 sockets
(N along the rod + end shear), landing with the base-loop landing case, a pull-out per tie at its largest |N|; the future battery
mechanism 750 g (E: pack ~600 + latch 150) at the pack's CG through the 4 bosses at 35 g in 14 directions, flight 4 g, landing 6 g.
Supports: the 8 pad faces against the plate. Prints upside down: the z -8.1 faces on the build plate, crowns toward -z.
Numbers: systems/dimensions.md (Overseer core: bottom plate holes, latch set pose; cage rows). v1 h 1.6 (h 1.5 = 323k cells, over
the 300k limit), gen 26 cm3 (E). v2 (29 Oct, auto mode): side room y +-84 / z 8, h 1.7, 30 cm3 (v1: index 2.09 at the side sockets, 44.4 g, run 64)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from topo.part import Part
from parts.cage_node_lib import CAGE, KIND, NODE_R, WALL, FLOOR, socket_cuts, collar_shape, along, U

SPEC = dict(h=1.7, gen_cm3=60.0, max_worst=0.5)   # v7 (30 Oct 8:30 pm, Spark): v6 gave 3.03 at the same spot (-69, +-45, -8.3), worst in the bay tie -1 pull-out alone (488 N), so the deeper side room went the wrong way: v5's room back (nothing deeper toward the battery), material 45 -> 60   # v6 (30 Oct 5:30 pm, Spark): v5 gave 2.22 in crash 7 (35 g, pack 750 g) at (-69, 43, -9), the side strip beside the slot only 16 mm deep (z -18 to -2.1) carrying the pack into the bay tie sockets; the side zones y 34..59 both sides, x -95..40, now drop to z -26 (outside the slot y +-33.4, 66 mm above the feet), volume kept   # v5 (30 Oct 8:15 am, Spark): the bay tie hub ends sit at z -9.0 in the cage file (were -5.1: the 6.2 bore had no wall under the plate at -2.1 and v4 gave 3.65 at those sockets); nothing else changed, 45 kept for a pass first   # v4 (30 Oct 1:30 am, auto mode): v3 with the rails to x -95 gave 1.55 at the same spot as v2 (1.54), now in the bay tie -1+1.2 pull-out (488 N) at (-60, 54, -7), the end of that socket where it meets the body under the plate; room did nothing between v2 and v3 (8 % of a 380 cm3 room is filled), so the lever is volume: 30 -> 45 (the node 5 history, 3.5 -> 4.5 cm3 halved its index, scales as volume^-3; 1.55 -> ~0.45 E), ~70 g      # v2: h 1.7 (the wider side room at 1.6 is 301k cells), 30 cm3 (v1 came in at 44 g with 26)   # v3 (30 Oct, auto mode): run 69 gave 1.54 at (-55, 55, -8.5), the left edge between the left.1/.2 sockets (x 0) and the bay tie -1+1.2 socket (x -90): the side rails only ran x -45..45, so the rail ended 45 mm short of that socket and the path pinched in the 16 mm under the plate edge; rails now x -95..45 (room, not volume)
G = 9.81e-3
BUILD = (0.0, 0.0, -1.0)                                          # printed upside down: the flat z -8.1 faces on the plate
Z_PLATE, Z_MOUNT = -2.1, -8.1                                     # plate underside; the future mechanism's mount plane
SLOT = (-121.8, 61.9, -33.4, 33.4)                                # the latch set's slot (dimensions.md): empty below Z_MOUNT
SCREWS = [(sx * 52.14, sy * 38.0) for sx in (1, -1) for sy in (1, -1)] + [(sx * 38.0, sy * 52.14) for sx in (1, -1) for sy in (1, -1)]
PAD_R, PAD_HOLE_R, HEAD_R = 4.5, 1.6, 3.4                         # O9 pad, O3.2 through, O6.8 head room below
BOSSES = [(-60.0, 25.0), (-60.0, -25.0), (20.0, 25.0), (20.0, -25.0)]
BOSS_R, INSERT_R, INSERT_D = 5.5, 2.0, 5.7                        # O11 boss, M3 x 5.7 insert in O4.0
PACK_G, PACK_CG = 750.0, np.array([-20.0, 0.0, -42.0])            # E: pack + latch, at the pack's centre (pack mesh -94.9..53, z -64.9..-19.9)
TIE_NAMES = ("core tie rear.1", "core tie rear.2", "core tie left.1", "core tie left.2", "core tie right.1", "core tie right.2",
             "bay tie +1+1.2", "bay tie +1-1.2", "bay tie -1+1.2", "bay tie -1-1.2")
TIE_CASES = json.load(open(os.path.join(HERE, "..", "..", "systems", "overseer.cage_loads.json")))["cases"]
DIRS14 = [np.array(v, float) / np.linalg.norm(v) for v in
          [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]]
FLIGHT = {"flight 4 g": np.array([0, 0, 4.0]), "flight 2 g x": np.array([2.0, 0, 1]), "flight 2 g y": np.array([0, 2.0, 1]), "landing 6 g": np.array([0, 0, 6.0])}


def cyl(x, y, z0, z1, r, n=40):
    import manifold3d as mf
    return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])


def box(x0, x1, y0, y1, z0, z1):
    import manifold3d as mf
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def rigid_split(pts, F, P):
    """least-norm forces at pts that carry a force F acting at point P (force and moment): the payload on a rigid link"""
    pts = np.asarray(pts, float); c = pts.mean(0); n = len(pts); Am = np.zeros((6, 3 * n))
    for i, q in enumerate(pts - c):
        Am[:3, 3 * i:3 * i + 3] = np.eye(3); Am[3:, 3 * i:3 * i + 3] = [[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]]
    Mc = np.cross(np.asarray(P, float) - c, F)
    return (np.linalg.pinv(Am) @ np.r_[F, Mc]).reshape(n, 3)


def sockets():
    """the ten sockets: name, c (virtual centre = the link's hub end), u (toward the cage node), s0, depth, bore_r, lining, axial"""
    out = []
    for nm in TIE_NAMES:
        p, q = CAGE["links"][nm]
        c, q = np.array(p, float), np.array(q, float); u = (q - c) / np.linalg.norm(q - c)
        k = KIND["link"]
        out.append(dict(name=nm, c=c, u=u, s0=NODE_R, depth=k["depth"], bore_r=k["bore_r"],
                        lining=along(c, u, NODE_R - FLOOR - 0.5, NODE_R + k["depth"], k["bore_r"] + 1.5),
                        axial=max(abs(cs["members"][nm]["N"]) for cs in TIE_CASES)))
    for i, a in enumerate(out):                                   # bores clear each other by WALL at the grooves (cage_node_lib's rule)
        for b in out[i + 1:]:
            sa = [a["c"] + a["u"] * t for t in np.linspace(a["s0"] - FLOOR, a["s0"] + a["depth"], 30)]
            sb = [b["c"] + b["u"] * t for t in np.linspace(b["s0"] - FLOOR, b["s0"] + b["depth"], 30)]
            dmin = min(np.linalg.norm(x - y) for x in sa for y in sb)
            need = a["bore_r"] + b["bore_r"] + 2 * 0.5 + WALL
            assert dmin >= need, f"{a['name']} / {b['name']}: bores {dmin:.1f} mm apart, need {need:.1f}"
    return out


def bodies():
    socks = sockets()
    collars, cuts = [], []
    for s_ in socks:
        s_end = s_["s0"] + s_["depth"]
        collars.append(collar_shape(s_["c"], s_["u"], s_["s0"] - FLOOR, s_end, s_["bore_r"], WALL, build=BUILD))
        cuts += socket_cuts(s_["c"], s_["u"], s_["s0"], s_["depth"], s_["bore_r"], build=BUILD)
        cuts.append(along(s_["c"], s_["u"], s_end, s_end + 25.0, s_["bore_r"] + 0.6))          # the rod beyond the mouth (short: only inside the envelope matters, and the cuts widen the grid)
    pads = [cyl(x, y, Z_MOUNT, Z_PLATE, PAD_R) for x, y in SCREWS]
    cuts += [cyl(x, y, Z_MOUNT - 30, Z_PLATE + 1, PAD_HOLE_R) for x, y in SCREWS]              # M3 through, up into the arm
    cuts += [cyl(x, y, Z_MOUNT - 30, Z_MOUNT, HEAD_R) for x, y in SCREWS]                     # the screw head below the pad
    bosses = [cyl(x, y, Z_MOUNT, Z_PLATE, BOSS_R) for x, y in BOSSES]
    cuts += [cyl(x, y, Z_MOUNT - 30, Z_MOUNT + INSERT_D, INSERT_R) for x, y in BOSSES]        # insert hole from below (0.3 to the plate)
    # the plate itself and everything above it over its outline (the plate's bounding box, so the sides and rear stay open)
    cuts.append(box(-95.5 - 0.3, 100.5 + 0.3, -59.0 - 0.3, 59.0 + 0.3, Z_PLATE, 20.0))
    # the future design's room: below the mount plane inside the slot
    cuts.append(box(SLOT[0], SLOT[1], SLOT[2], SLOT[3], -80.0, Z_MOUNT))
    # envelope: the 6 mm under the plate over the part's footprint; beside the plate's sides and behind its tail the room rises past
    # the plate plane for the core tie collars (nothing else is there) and drops to z -18 for the bay tie bosses
    # v2 (29 Oct, run 64: index 2.09 at the left side socket in the sideways crash, collars hanging off the main body): v1 gave the
    # sockets beside the plate an 11 mm strip (y 59..70) while their bores reach y 79 and the collars 83: the collars hung off a
    # ledge. The side room now runs to y +-84 and up to z 8 (168 wide: under the 180 plate), so the collars sit inside material.
    env = U([box(-118.0, 57.0, -70.0, 70.0, -18.0, Z_PLATE),
             box(-95.0, 45.0, 59.0, 84.0, -18.0, 8.0), box(-95.0, 45.0, -84.0, -59.0, -18.0, 8.0),
             box(-118.0, -95.5, -34.0, 34.0, -18.0, 11.0)])
    return U(collars + pads + bosses), U(cuts), env, socks, pads


def setup(h):
    drawn, cuts, env, socks, pads = bodies()
    p = Part("overseer_base", h, drawn, cuts, env, build="-z", material="PETG")
    p.support("default", U([cyl(x, y, Z_PLATE - 2.0, Z_PLATE + 0.5, PAD_R + 0.5) for x, y in SCREWS]))
    boss_pts = [np.array([x, y, Z_MOUNT]) for x, y in BOSSES]
    boss_reg = [cyl(x, y, Z_MOUNT - 0.5, Z_MOUNT + 2.5, BOSS_R + 0.5) for x, y in BOSSES]

    def tie_forces(case):
        out = []
        for s_ in socks:
            m = case["members"][s_["name"]]
            side = np.cross(s_["u"], [0, 0, 1.0]); side /= np.linalg.norm(side)
            out.append((s_["lining"], m["N"] * s_["u"] + m["shear_j"] * side))
        return out

    def frame_case(d):
        for cs in TIE_CASES:
            if cs["name"].startswith("crash") and np.allclose(cs["d"], d / np.linalg.norm(d), atol=0.02):
                return cs
        return None

    def payload(name, a, cls, extra=()):
        f = rigid_split(boss_pts, -PACK_G * G * a, PACK_CG)
        p.load_multi(name, [(boss_reg[i], f[i]) for i in range(4)] + list(extra), cls=cls, accel=-a)
    for k, d in enumerate(DIRS14):
        cs = frame_case(d)
        payload(f"crash {k} (35 g, pack 750 g)" + (" + frame ties" if cs else ""), 35 * d, "crash", tie_forces(cs) if cs else ())
    land = [cs for cs in TIE_CASES if cs["name"].startswith("landing")]
    for nm, a in FLIGHT.items():
        payload(nm, a, "flight", tie_forces(land[0]) if (nm.startswith("landing") and land) else ())
    for s_ in socks:
        p.load_multi(f"frame {s_['name']} axial {s_['axial']:.0f} N (pull-out)", [(s_["lining"], s_["axial"] * s_["u"])], cls="crash")
    return p
