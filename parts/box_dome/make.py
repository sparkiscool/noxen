"""The Scout's rain and dust dome and its folding mechanism (9 Nov, Spark: the dome of noxensys.com's dock, "only as big as the
scout, the overseer will not need to be in that dome", then "lets actually design the mechanism for the cover"). PROTOTYPE (15 Nov):
every printed piece is drawn here, in its print pose in print/box_dome_*.stl, solid-checked; the assembly check gates the set.

What it is
  A dome of coated ripstop on five 3 mm carbon hoops, all pivoting on one hinge axis across the box (W X, at W Z 161), centred on
  the seat (W Y 168). Shut, the hoops stand at hinge angles 0, 45, 90, 135, 180 deg (0 = the base at the rear, 90 the crown).
  Open, all five lie at 0, a bundle behind the box with the fabric gathered on it: the Overseer's rear cage nodes pass 50 above
  it, so it lands with the dome open; the dome is the Scout's only.
  Hoop k (k 0..4) is one 3 mm rod sprung between its two hub plates. A bent rod takes the shape of an elastica, not a circle, so
  the hoops are drawn as one (15 Nov): pinned ends, leaving its socket 61 deg from the axis, crown 263 + 8.8 k from the axis. The
  sockets lie along that tangent, so the rod puts no moment on its plates (a circle's radial sockets would hold 2.0 N m in each
  plate for good: EI / R = 130 GPa x 3.98 mm4 / 263; PCTG creeps under it). What is left is the rod's thrust along the axis,
  1.5 pi^2 EI / L^2 = 14 N a hoop (E), into the hub walls. Bending strain at the crown 0.72 % (4 k K r / L; pultruded rod 1.3 %).

How it folds: a folding fan
  Hoop 0 is fixed (its plate is the hub's inner wall). At each hub, plates 1..4 turn on one 6 mm carbon tube bonded in that
  wall; plate k carries a 3 mm carbon pin that rides in a 45 deg slot in plate k-1 (pins at r 24 and r 13 in turn, so a plate's
  own pin and its slot never meet), so neighbouring hoops stay 0..45 deg apart. A self-locking worm-gear motor at each hub turns
  plate 4 (the free edge's hoop) through a 6 mm flange coupling screwed to it: opening, hoop 4 sweeps from 180 down to 0 and
  picks up hoops 3, 2, 1 in turn; shutting, it pulls each one out after it once their slot is at its end. Shut, hoop 4 is held
  by the worm, hoop 0 is fixed, the slots fix every hoop between. A cam on each motor's outer shaft trips one of two roller limit
  switches; the box lands a drone only with the dome open and confirmed.
  The fabric is one silnylon sheet sewn in sleeves to the five hoops (drawn as four sub-strips per panel).

Printed pieces (PCTG, all in print/box_dome_*.stl in their print pose; `python parts/box_dome/make.py` prints each one's line)
  hub l / r        the bracket: a 6 thick base on the ring's top (4 x M3 x 12 up through the ring into its pilots), the inner wall
                   (plate 0 and the cheek behind it, 13.4), a stop post and two side walls that take the gearbox (32 across, slide)
                   and the switches. Prints base down, no supports.
  plate 1..4 l / r a 7.4 sector round the tube, an 8 x 14 arm along the rod's tangent with its 18 deep socket. Prints outer face
                   down, supports under the arm (it rises 29 deg).
  ring             16 segments (r 255..297, W Z 130..142) in half laps, two 3 mm carbon pins a lap: 8 spoke segments (lower
                   tongues, on the spokes first), 6 plain and 2 hub segments (upper tongues, dropped on; printed top face down).
Hand calculation (PCTG yield 46 MPa E, in-layer; SF 2.5 storm, 10 sustained)
  loads      storm 20 m/s on the shut dome: q 240 Pa x 0.141 m2 x Cd 1.2 = 41 N at 0.13 m: 5.3 N m on the axis, 2.7 N m a hub (the
             worm holds hoop 4); 20 N shear a hub. Operating (10 m/s) a quarter of it. Moving mass 0.1 kg at 0.3 m: 0.15 N m a hub.
  pins       2.7 N m at r 13: 208 N on a 3 mm pin in 7.4 of plate: 9.4 MPa bearing (18.4 allowed): 0.51
  arm        hoop 4's share at its socket, 2.7 N m about the axis on 8 x 14 (Z = 8 x 14^2 / 6 = 261 mm3): 10.3 MPa: 0.56
  flange     2.7 N m on four M3 at r 8: 84 N each in 2.6 pilots, 7.4 deep (shear area pi x 3 x 7.4 x 0.5 = 35 mm2): 2.4 MPa: 0.13
  side wall  the gearbox's torque as a couple 19 apart: 142 N at 12.4 from the base's end on 31 x 5.5 (Z 156 mm3): 11.3 MPa: 0.61
  hub screws 2.7 N m + 20 N x 19 mm on M3 pairs 36 apart: 86 N a screw, 5 mm in a 2.6 pilot (590 N E): 0.15
  socket     14 N thrust + the wind's 10 N a rod end on 18 x 3.2 of bond: under 0.5 MPa; rod sustained moment 0 (tangent socket)
Fits (systems/fit_table.json through topo/fit.py): rod and pin sockets and the tube's bore in the wall 3.2 / 6.2 (bond), each
  blind socket with its O2.5 port 1.5 above the floor and three 0.3 pockets; plates on the tube 6.3 and slots 3.3 (free);
  M3 pilots 2.6 (tap), M3 through the ring 3.2 (close), M2 switch pilots 1.6 (tap); the gearbox between the walls 32.6 (slide).
Assembly (declared for tools/assembly_check.py in assembly_block()): spoke segments on the spokes, the other segments dropped on,
  lap pins, the lid; each hub down onto the ring; each hoop (rod bonded in its two plates on the bench, hoop 4 with its
  couplings) lowered between the walls; the tubes pushed in from inside the dome; with every hoop folded to 0 the pin bores line
  up: pins 1 and 3 pushed in along r 24 (past plate 4's edge), 2 and 4 along r 13, a drop of CA on each; the motors
  slid in from outside, shafts into the couplings; the fabric last.

Bought (Amazon, Spark 29 Oct)
  hoops     KARBXON 3 x 1000 mm pultruded carbon rod (hoops cut to 743..860; pins 17 from the offcuts)   https://www.amazon.com/KARBXON-Carbon-Pultruded-Planes-Drones-Projects/dp/B09Q8CKC2L
  motors    JGY-370 12 V 6 rpm self-locking worm gear motor, double output shaft, 2 (gearbox 46 x 32 x 25, shaft 6 D x 14: E, listing)   https://www.amazon.com/JGY370-Double-Electric-Self-Locking-6RPM-150RPM/dp/B0DL2S9GPZ
  couplings 6 mm rigid flange coupling, 2 (flange O22 x 2, four holes on O16, hub O10, 12 long: E, the usual part; listing to pick: https://www.amazon.com/s?k=6mm+rigid+flange+coupling)
  switches  KW11-3Z roller lever micro switch, 4 of a 10-pack     https://www.amazon.com/clp/B0B342RXJM
  fabric    emma kites 20D silnylon, white, 60 x 36 in, 2         https://www.amazon.com/emma-kites-Waterproof-Ultralight-Resistant/dp/B09VWZQT22
  lid       24 x 24 in 3 mm UV-stabilised polycarbonate sheet    https://www.amazon.com/clp/B01KZG8NBQ
  spokes, pivots  6 mm carbon tube (the project's stock): 8 spokes, 2 pivots x 53
  screws    8 x M3 x 12 cap (hubs), 8 x M3 x 8 (flanges), 8 x M2 x 10 self-tapping (switches)
W frame (the box's): X across, Y along the axle toward the front, Z up, mm.
Run: python parts/box_dome/make.py [--quick]   (writes gore_meshes/*.stl and print/box_dome_*.stl, solid-checks each print file, then
     checks every part against both drones, shut, moving, open; --quick: no drone check)
"""
import json, math, os, sys
import numpy as np, trimesh
import manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from topo import fit as FIT                     # noqa: E402

YC, ZB = 168.0, 161.0                           # the dome's centre: W (0, YC), its hinge axis at W Z ZB
PLAN = math.radians(90.0)                       # drawn with its axis along W Y folding to -W X, then turned 90 deg about the centre: axis along W X, hubs left and right, folding to the rear (-W Y)
NH = 5                                          # hoops
SHUT = [0.0, 45.0, 90.0, 135.0, 180.0]          # hoop angles shut (deg)
GAP = 45.0                                      # the slot: neighbouring hoops 0..45 apart
ROD = 1.5                                       # 3 mm carbon rod
RHO_SOCK, SOCK_D = 50.0, 18.0                   # the socket's mouth from the axis, its depth along the rod
UM0, PITCH, CROWN0 = 258.2, 10.7, 263.0         # hoop 0's mouth from the dome's centre along the axis, the step between hoops, hoop 0's crown from the axis
PLATE_T, PLATE_R, HUB_R = 7.4, 32.0, 8.0        # hub plates: thickness, sector radius, the ring round the tube
ARM_T, ARM_W, ARM_C, ARM_S = 8.0, 14.0, 9.4, 34.0  # the socket arm: across the rod's plane, in it at its root and round the socket (a 3.1 collar), its length along the rod
RHO_PIN = (24.0, 13.0)                          # the pins: odd plates at r 24, even at r 13
PIN_LEAD = math.radians(22.0)                   # pins and slots lead their plate's arm by this, clear of its root
M = 4                                           # fabric sub-strips per panel
FAB = 0.8                                       # fabric thickness drawn
RING = (255.0, 297.0, 130.0, 142.0)             # the base ring: r in, r out, z (beside the lid, the lid's edge on its inner part)
NSEG, LAP, LAP_G = 16, 16.0, 0.3                # ring segments, the half lap's length at r 276, its end gap
LID = (142.0, 145.0, 160.0, 270.0)              # the rain lid: z, the radius it stays open within, its edge (on the ring)
BOX = dict(x=209.0, y0=-69.0, y1=340.0, top=145.0)
SPOKES = 8
BASE = (RING[3], RING[3] + 6.0)                 # the hub's base on the ring's top
PCTG = 1.27                                     # g / cm3
R_ROD, R_SLOT = FIT.bore(3.0, "bond") / 2, FIT.bore(3.0, "free") / 2
R_TUBE_B, R_TUBE_F = FIT.bore(6.0, "bond") / 2, FIT.bore(6.0, "free") / 2
R_PORT, R_TAP3, R_CLOSE3, R_TAP2 = FIT.bore(2.11, "port") / 2, FIT.bore(3.0, "tap") / 2, FIT.bore(3.0, "close") / 2, FIT.bore(2.0, "tap") / 2
GEAR_W = FIT.across(32.0, "slide")              # the JGY-370's gearbox between the side walls
POCKET = 0.3                                    # resin pockets along a bonded bore
SCREWS = ((-18.0, 265.0), (18.0, 265.0), (-18.0, 289.0), (18.0, 289.0))   # the hub's M3 into the ring: (x, along the axis from the centre)


# ---------------------------------------------------------------------------------------------------------------- the hoop's shape
def _shape(n=240):
    """the pinned elastica through the mouths with hoop 0's crown: (end angle from the axis, xi 1..-1, eta, length), per unit of
    the mouth's distance from the centre"""
    from scipy.special import ellipk
    target = (CROWN0 - RHO_SOCK) / UM0

    def run(k):
        al = 2 * math.asin(k); L = 2 * float(ellipk(k * k)); h = L / n
        psi, w = al, 0.0; x, y = [0.0], [0.0]
        f = lambda p, w_: (w_, -math.sin(p))
        for _ in range(n):
            k1 = f(psi, w); k2 = f(psi + h / 2 * k1[0], w + h / 2 * k1[1]); k3 = f(psi + h / 2 * k2[0], w + h / 2 * k2[1]); k4 = f(psi + h * k3[0], w + h * k3[1])
            pn = psi + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); w = w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            pm = (psi + pn) / 2; x.append(x[-1] + h * math.cos(pm)); y.append(y[-1] + h * math.sin(pm)); psi = pn
        s = 2.0 / x[-1]
        return al, 1.0 - np.array(x) * s, np.array(y) * s, L * s, k
    lo, hi = 0.3, 0.7
    for _ in range(40):
        mid = (lo + hi) / 2
        if run(mid)[2].max() < target: lo = mid
        else: hi = mid
    return run((lo + hi) / 2)


ALPHA, XI, ETA, LEN, K_E = _shape()
CA, SA = math.cos(ALPHA), math.sin(ALPHA)
UP_OFF = (RHO_SOCK - 31.0) / math.tan(ALPHA)    # a plate's plane beyond its mouth: where the arm's centre line is at r 31


def um(k):
    """hoop k's socket mouths, from the dome's centre along the axis"""
    return UM0 + PITCH * k


def up(k):
    """plate k's plane"""
    return um(k) + UP_OFF


def crown(k):
    return RHO_SOCK + um(k) * float(ETA.max())


def rod_len(k):
    return um(k) * LEN + 2 * SOCK_D


def rho_pin(k):
    return RHO_PIN[0] if k % 2 else RHO_PIN[1]


def Wp(u, rho, phi=0.0, n=0.0):
    """W point (drawn): u along the axis from the centre, rho from the axis at hinge angle phi (rad), n across the hoop's plane"""
    return np.array([-rho * math.cos(phi) + n * math.sin(phi), YC + u, ZB + rho * math.sin(phi) + n * math.cos(phi)])


def profile(k, off=0.0):
    """hoop k's centre line from the front mouth to the rear one as (u, rho) rows, moved off outward along its normal"""
    u, r = um(k) * XI, RHO_SOCK + um(k) * ETA
    if off:
        t = np.gradient(np.column_stack([u, r]), axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
        u, r = u + off * t[:, 1], r - off * t[:, 0]
    return np.column_stack([u, r])


def sock_axis(k, s):
    """a point on hoop k's front socket axis, s into the socket from its mouth, as (u, rho)"""
    return um(k) + s * CA, RHO_SOCK - s * SA


# ---------------------------------------------------------------------------------------------------------------- solids
def man(t):
    return mf.Manifold(mf.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))


def tri(m):
    t = m.to_mesh(); raw = trimesh.Trimesh(np.asarray(t.vert_properties)[:, :3], np.asarray(t.tri_verts), process=False)
    out = raw.copy(); out.merge_vertices(digits_vertex=4); out.update_faces(out.nondegenerate_faces()); out.remove_unreferenced_vertices()
    return out if out.is_watertight else raw      # float32 slivers: the merged mesh when it stays closed, else the manifold's own


def U(ms):
    ms = list(ms); return mf.Manifold.batch_boolean(ms, mf.OpType.Add) if len(ms) > 1 else ms[0]


def cyl(p0, p1, r, n=32):
    return man(trimesh.creation.cylinder(radius=r, segment=[list(p0), list(p1)], sections=n))


def hull(pts):
    return man(trimesh.convex.convex_hull(np.array(pts)))


def cube(x0, x1, y0, y1, z0, z1):
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def hbox(x0, x1, a0, a1, z0, z1):
    """a box at the front hub: x, along the axis from the centre, z"""
    return cube(x0, x1, YC + a0, YC + a1, z0, z1)


def prism(poly, a0, a1):
    """a shapely polygon in the axis's plane (x, z from the axis) extruded along the axis a0..a1 from the centre"""
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    out = []
    for p in polys:
        t = trimesh.creation.extrude_polygon(p, a1 - a0)
        t.apply_transform(np.array([[1.0, 0, 0, 0], [0, 0, 1.0, YC + a0], [0, 1.0, 0, ZB], [0, 0, 0, 1]]))
        if t.volume < 0: t.invert()
        out.append(man(t))
    return U(out)


def G():
    """the plan turn about the dome's centre: drawn (axis W Y, folding to -W X) -> placed (axis W X, folding to -W Y)"""
    c, s_ = math.cos(PLAN), math.sin(PLAN)
    T = np.eye(4); T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s_, s_, c
    A = np.eye(4); A[1, 3] = -YC; B = np.eye(4); B[1, 3] = YC
    return B @ T @ A


MIR = np.diag([1.0, -1.0, 1.0, 1.0]); MIR[1, 3] = 2 * YC          # the front hub -> the rear one


def turn(dphi):
    """4x4 (W, drawn): turn about the hinge axis by dphi (rad), hinge angles + dphi"""
    c, s = math.cos(dphi), math.sin(dphi)
    T = np.eye(4); T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    A = np.eye(4); A[2, 3] = -ZB; B = np.eye(4); B[2, 3] = ZB
    return B @ T @ A


def xf(m, T):
    return m.transform(np.asarray(T)[:3, :])


def side(m, sgn):
    return m if sgn > 0 else xf(m, MIR)


def placed(m):
    """a drawn solid (manifold or mesh) as a mesh in the box's W frame"""
    t = tri(m) if isinstance(m, mf.Manifold) else m.copy()
    t.apply_transform(G()); return t


SIDES = ((1, "l"), (-1, "r"))                   # the drawn front hub lands on -W X (left), the rear one on +W X


def socket_cuts(p_floor, p_out, r, e1, port_at=1.5, port_len=14.0):
    """a bonded bore from its floor out past its mouth: the bore, three POCKET deep 60 deg resin pockets from 1.0 above the floor
    out of the mouth (the first toward e1), and the O2.5 port port_at above the floor out along e1 (None: no port)"""
    p_floor, p_out = np.asarray(p_floor, float), np.asarray(p_out, float)
    t = (p_out - p_floor) / np.linalg.norm(p_out - p_floor)
    e1 = np.asarray(e1, float); e1 = e1 - (e1 @ t) * t; e1 /= np.linalg.norm(e1); e2 = np.cross(t, e1)
    cuts = [cyl(p_floor, p_out, r)]
    big = cyl(p_floor + t * 1.0, p_out, r + POCKET)
    for i in range(3):
        a = 2 * math.pi * i / 3; pts = []
        for q in (p_floor + t * 0.9, p_out + t * 0.1):
            pts.append(q)
            pts += [q + (r + 1.5) * (math.cos(b) * e1 + math.sin(b) * e2) for b in (a - math.pi / 6, a, a + math.pi / 6)]
        cuts.append(big ^ hull(pts))
    if port_at is not None:
        q = p_floor + t * port_at; cuts.append(cyl(q, q + e1 * port_len, R_PORT, 16))
    return cuts


def rod_socket(k):
    """hoop k's front socket, at hinge angle 0: its port out across the hoop's plane (+n)"""
    return socket_cuts(Wp(*sock_axis(k, SOCK_D)), Wp(*sock_axis(k, -3.0)), R_ROD, [0, 0, 1.0])


def slab(k, n0=None):
    """plate k's socket arm at hinge angle 0: ARM_T across, along the rod's tangent from the mouth ARM_S in (into the flat plate),
    ARM_C wide round the socket and ARM_W at its root; nothing beyond the plate's outer face (n0: its underside, the hub's)"""
    pts = []
    for s, hw in ((0.0, ARM_C / 2), (SOCK_D - 2.0, ARM_C / 2), (SOCK_D + 4.0, ARM_W / 2), (ARM_S, ARM_W / 2)):
        for w in (-ARM_T / 2, ARM_T / 2):
            u, rho = sock_axis(k, s)
            pts += [Wp(u + w * SA, rho + w * CA, 0.0, n) for n in (-hw if n0 is None else n0, hw)]
    return hull(pts) ^ hbox(-200.0, 200.0, 0.0, up(k) + PLATE_T / 2, 0.0, 400.0)


def _pt(r, psi):
    return (-r * math.cos(psi), r * math.sin(psi))


def _arc(r, a0, a1, n=24):
    return [_pt(r, a) for a in np.linspace(a0, a1, n)]


def plate_profile(k):
    """plate k's flat part in the axis's plane at hinge angle 0: (body, holes)"""
    from shapely.geometry import Polygon, Point, LineString
    from shapely.ops import unary_union
    d = math.radians
    L = PIN_LEAD
    span = (d(-10.0), L + d(GAP + 10.0)) if k in (0, 2) else (d(-10.0), L + d(10.0)) if k < NH - 1 else (d(-10.0), d(10.0))   # round the r 24 slot, or the r 24 pin bore; plate 4 (shut at 180 over the base) clear of the r 24 line
    body = [Polygon([(0.0, 0.0)] + _arc(PLATE_R, *span)), Point(0, 0).buffer(HUB_R if k < NH - 1 else 11.5, 32)]
    holes = [Point(0, 0).buffer(R_TUBE_F, 32)]
    if k < NH - 1:                                                           # the 45 deg slot for plate k+1's pin
        sl = LineString(_arc(rho_pin(k + 1), L, L + d(GAP)))
        holes.append(sl.buffer(R_SLOT, 12))
        if rho_pin(k + 1) < 20: body.append(sl.buffer(R_SLOT + 2.5, 12))
    if k >= 1:                                                               # its own pin's bore
        c = Point(_pt(rho_pin(k), L)); holes.append(c.buffer(R_ROD, 24))
        if rho_pin(k) < 20: body.append(c.buffer(R_ROD + 2.6, 24))
    if k == NH - 1:
        holes += [Point(_pt(8.0, d(45.0 + 90.0 * i))).buffer(R_TAP3, 16) for i in range(4)]   # the flange's four M3
    return unary_union(body), unary_union(holes)


def plate0(k):
    """plate k (1..4) at the front hub, hinge angle 0"""
    body, holes = plate_profile(k)
    return prism(body, up(k) - PLATE_T / 2, up(k) + PLATE_T / 2) + slab(k) - prism(holes, up(k) - 12.0, up(k) + 12.0) - U(rod_socket(k))


def plate(k, phi, sgn):
    return tri(side(xf(plate0(k), turn(phi)), sgn))


A_IN = up(0) - PLATE_T / 2 - 6.0                 # the inner wall's inner face
A_F0 = up(0) + PLATE_T / 2                       # its face toward plate 1
A_P4 = up(NH - 1) + PLATE_T / 2                  # plate 4's outer face: the coupling's flange on it (2), its hub 10 beyond
A_GB = A_P4 + 12.6                               # the gearbox's face (the stop post's outer face)
A_END = A_GB + FIT.across(25.0, "slide") + 14.0  # the side walls' ends, past the cam and the switches
A_TUBE = (A_IN, up(NH - 1) + 0.6)                # the pivot tube
WALL_T = 5.5


def hub0():
    """the front hub bracket (drawn frame)"""
    from shapely.geometry import Point, LineString, box
    from shapely.ops import unary_union
    body, _ = plate_profile(0)
    prof = unary_union([body, box(-PLATE_R, BASE[1] - ZB, 12.0, 0.0), box(-12.0, 0.0, 12.0, 12.0)]).difference(Point(0, 0).buffer(R_TUBE_B, 32))
    top = ZB + 12.0; gw = GEAR_W / 2
    solid = U([hbox(-56.0, 28.0, 252.0, A_GB, *BASE),                                   # the base on the ring
               prism(prof, A_IN, A_F0), slab(0, n0=BASE[1] - ZB) - cyl([0, YC + A_IN - 1.0, ZB], [0, YC + A_F0 + 1.0, ZB], R_TUBE_B),                         # the inner wall: the cheek, plate 0, hoop 0's arm down to the base
               hbox(12.0, gw, A_GB - 6.0, A_GB, BASE[1], top),                            # the gearbox's stop post, and a low one on the pins' side
               hbox(-gw, -12.0, A_GB - 6.0, A_GB, BASE[1], ZB - 8.0),
               hbox(gw, gw + WALL_T, A_GB - 6.0, A_END, BASE[0], top), hbox(-gw - WALL_T, -gw, A_GB - 6.0, A_END, BASE[0], ZB + 6.0)])   # the side walls (the pins' side lower: the way in for pins 1 and 3 passes over it)
    slot = prism(LineString(_arc(rho_pin(1), PIN_LEAD, PIN_LEAD + math.radians(GAP))).buffer(R_SLOT, 12), A_F0 - 6.5, A_F0 + 1.0)     # plate 1's pin: a blind slot
    cuts = [slot] + rod_socket(0)
    cuts += socket_cuts([0, YC + A_IN - 1.0, ZB], [0, YC + A_F0 + 1.0, ZB], R_TUBE_B, [0, 0, 1.0], port_at=1.0 + (A_F0 - A_IN) / 2, port_len=16.0)   # the tube's bore, its port down from the cheek's top
    cuts += [cyl([x, YC + a, BASE[0] - 1.0], [x, YC + a, BASE[1] + 0.5], R_TAP3, 16) for x, a in SCREWS]
    cuts += [cyl([-40.0, YC + A_END - 7.0, ZB + dz], [40.0, YC + A_END - 7.0, ZB + dz], R_TAP2, 12) for dz in (-4.75, 4.75)]     # the switches' M2 (KW11: 9.5 apart E)
    return solid - U(cuts)


def hoop_pts(k, phi):
    """hoop k's rod centre line at hinge angle phi (rad): socket floor to socket floor"""
    pr = profile(k); s0 = SOCK_D - 0.5
    a = sock_axis(k, s0); pts = [Wp(a[0], a[1], phi)] + [Wp(u, r, phi) for u, r in pr] + [Wp(-a[0], a[1], phi)]
    return pts


def tube(pts, r=ROD, sections=8):
    return U([cyl(a, b, r, sections) for a, b in zip(pts[:-1], pts[1:])])


def hoop_rod(k, phi):
    return tri(tube(hoop_pts(k, phi)[::4] + [hoop_pts(k, phi)[-1]]))


def pins(k, phi):
    """hoop k's two pins (3 mm carbon) at hinge angle phi: bonded through plate k, 6 into plate k-1's slot"""
    out = []
    a0 = up(k) + PLATE_T / 2; a1 = (up(k - 1) - PLATE_T / 2 + 1.4) if k > 1 else A_F0 - 6.0
    for sgn, _ in SIDES:
        out.append(side(cyl(Wp(a0, rho_pin(k), phi + PIN_LEAD), Wp(a1, rho_pin(k), phi + PIN_LEAD), ROD, 16), sgn))
    return tri(U(out))


def strip(k, j):
    """fabric sub-strip j of panel k (between hoops k and k+1), shut: hinge angles 45 k + 45 j / M .. + 45 / M, outside hoop k+1's
    rod, from just past the hub plates' arms (rho 58) round to the other hub"""
    pr = profile(k + 1, ROD + 1.5 + 0.3 * j); pr2 = profile(k + 1, ROD + 1.5 + 0.3 * j + FAB)
    ok = np.where((pr[:, 1] >= 58.0) & (pr2[:, 1] >= 58.0))[0]
    idx = np.unique(np.linspace(ok[0], ok[-1], 41).round().astype(int)); nth = len(idx) - 1
    p0 = math.radians(SHUT[k] + GAP * j / M); p1 = p0 + math.radians(GAP / M); nphi = 2
    V = [Wp(q[i, 0], q[i, 1], p0 + (p1 - p0) * a / nphi) for q in (pr, pr2) for a in range(nphi + 1) for i in idx]
    n = (nphi + 1) * (nth + 1)
    ix = lambda a, b, l: l * n + a * (nth + 1) + b; F = []
    for a in range(nphi):
        for b in range(nth):
            F += [[ix(a, b, 1), ix(a + 1, b, 1), ix(a + 1, b + 1, 1)], [ix(a, b, 1), ix(a + 1, b + 1, 1), ix(a, b + 1, 1)]]
            F += [[ix(a, b, 0), ix(a + 1, b + 1, 0), ix(a + 1, b, 0)], [ix(a, b, 0), ix(a, b + 1, 0), ix(a + 1, b + 1, 0)]]
    for a, w in ((0, -1), (nphi, 1)):
        for b in range(nth):
            q = [ix(a, b, 0), ix(a, b + 1, 0), ix(a, b + 1, 1), ix(a, b, 1)]
            F += ([[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if w < 0 else [[q[0], q[2], q[1]], [q[0], q[3], q[2]]])
    for b in (0, nth):
        for a in range(nphi):
            q = [ix(a, b, 0), ix(a + 1, b, 0), ix(a + 1, b, 1), ix(a, b, 1)]
            F += ([[q[0], q[1], q[2]], [q[0], q[2], q[3]]] if b == 0 else [[q[0], q[2], q[1]], [q[0], q[3], q[2]]])
    m = trimesh.Trimesh(np.array(V), np.array(F), process=True); m.merge_vertices(); trimesh.repair.fix_normals(m)
    if m.volume < 0: m.invert()
    return m


def bundle():
    """the open dome's fabric, gathered on the folded hoops (all at 0): the band between hoop 0 and hoop 4, from the ring's top to
    15 above the hoops (what a landing drone must clear; the sub-strips stand in for the fabric only while it is taut)"""
    from shapely.geometry import Polygon
    o, i = profile(NH - 1, 12.0), profile(0, -6.0)
    poly = Polygon([(-r, YC + u) for u, r in o] + [(-r, YC + u) for u, r in i[::-1]]).buffer(0)
    t = trimesh.creation.extrude_polygon(poly, ZB + 15.0 - RING[3]); t.apply_translation([0, 0, RING[3]])
    return placed(t)


def angles(a4, shutting):
    """the five hoop angles (deg) with hoop 4 at a4: opening (hoop 4 coming down) it has picked up every hoop it reached; shutting
    (going up) each hoop follows once its neighbour above is 45 ahead"""
    if shutting:
        return [0.0] + [max(0.0, a4 - GAP * (NH - 1 - k)) for k in range(1, NH - 1)] + [a4]
    return [0.0] + [min(SHUT[k], a4) for k in range(1, NH - 1)] + [a4]


def poses(a4, shutting):
    """part id -> 4x4 (W) for every moving part (drawn shut) with hoop 4 at a4 deg"""
    ph = angles(a4, shutting); out = {}
    for k in range(1, NH):
        T = G() @ turn(math.radians(ph[k] - SHUT[k])) @ np.linalg.inv(G())
        for i in [f"box_dome_hoop_{k}", f"box_dome_pins_{k}"] + [f"box_dome_plate_{k}_{s}" for _, s in SIDES]:
            out[i] = T
    for k in range(NH - 1):
        for j in range(M):
            start = ph[k] + (ph[k + 1] - ph[k]) * j / M
            out[f"box_dome_fabric_{k}{j}"] = G() @ turn(math.radians(start - (SHUT[k] + GAP * j / M))) @ np.linalg.inv(G())
    return out


# ---------------------------------------------------------------------------------------------------------------- base (placed)
def box_edge(a):
    """where the ray from the dome centre at angle a meets the box's top outline"""
    d = np.array([math.cos(a), math.sin(a)]); best = None
    for k, lim in ((0, BOX["x"]), (0, -BOX["x"]), (1, BOX["y0"] - YC), (1, BOX["y1"] - YC)):
        if abs(d[k]) > 1e-9:
            t = lim / d[k]; p = d * t
            if t > 0 and abs(p[0]) <= BOX["x"] + 1e-6 and BOX["y0"] - YC - 1e-6 <= p[1] <= BOX["y1"] - YC + 1e-6:
                best = t if best is None else min(best, t)
    return np.array([0.0, YC]) + d * best


def lid(n=128):
    """the rain lid: the box's top outline and the ring's disk, less the opening within LID[2] of the seat, the sheets and a notch
    round each hub's base (1 clear), 3 mm"""
    prism_z = lambda poly, z0, z1: mf.Manifold.extrude(mf.CrossSection([poly]), z1 - z0).translate([0, 0, z0])
    disk = [(LID[3] * math.cos(t), YC + LID[3] * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    hole = [(LID[2] * math.cos(t), YC + LID[2] * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
    outline = [(-BOX["x"], BOX["y0"]), (BOX["x"], BOX["y0"]), (BOX["x"], BOX["y1"]), (-BOX["x"], BOX["y1"])]
    sheets = [cube(x0, x1, y0, y1, LID[0] - 6.0, LID[0] + 6.0) for x0, x1, y0, y1 in
              ((206.0, 209.0, -63.0, 334.0), (-209.0, -206.0, -63.0, 334.0), (-203.0, 203.0, 337.0, 340.0), (-203.0, 203.0, -69.0, -66.0))]
    notch = [xf(side(hbox(-57.0, 29.0, 251.0, LID[3] + 5.0, LID[0] - 1.0, LID[1] + 1.0), sgn), G()) for sgn, _ in SIDES]
    return tri((prism_z(outline, LID[0], LID[1]) + prism_z(disk, LID[0], LID[1])) - prism_z(hole, LID[0] - 1, LID[1] + 1) - U(sheets + notch))


def seg_angle(j):
    return 2 * math.pi * j / NSEG


def _sector(a0, a1, z0, z1, r0=RING[0], r1=RING[1]):
    n = max(2, int(math.degrees(a1 - a0) / 1.0) + 1); aa = np.linspace(a0, a1, n)
    poly = [(r1 * math.cos(a), YC + r1 * math.sin(a)) for a in aa] + [(r0 * math.cos(a), YC + r0 * math.sin(a)) for a in aa[::-1]]
    return mf.Manifold.extrude(mf.CrossSection([poly]), z1 - z0).translate([0, 0, z0])


def ring_seg(j):
    """ring segment j (placed), centred on 22.5 j deg: odd ones carry a spoke and the lower tongues, even ones the upper tongues
    (0 and 8 the hubs' screws); two O3.2 lap pin bores at each end"""
    c = seg_angle(j); half = math.pi / NSEG; d = (LAP / 2) / 276.0; g = LAP_G / 276.0; zm = (RING[2] + RING[3]) / 2
    low = j % 2 == 1
    m = _sector(c - half + d, c + half - d, RING[2], RING[3]) + _sector(c - half - d + g, c + half + d - g, *((RING[2], zm - 0.1) if low else (zm + 0.1, RING[3])))
    cuts = []
    for e in (-1, 1):
        a = c + e * half
        cuts += [cyl([r * math.cos(a), YC + r * math.sin(a), RING[2] - 1], [r * math.cos(a), YC + r * math.sin(a), RING[3] + 1], R_ROD, 24) for r in (266.0, 286.0)]
    rad = np.array([math.cos(c), math.sin(c), 0.0]); o = np.array([0.0, YC, zm])
    if low:                                                                 # the spoke's socket: from its floor 6 inside the outer face in to the inner face, the port from the top
        cuts += socket_cuts(o + rad * (RING[1] - 6.0), o + rad * (RING[0] - 1.0), R_TUBE_B, [0, 0, 1.0], port_len=10.0)
    if j % (NSEG // 2) == 0:                                                # a hub's four M3 x 12 from below: head recess O6 x 5, shank 3.2
        T = G() @ (np.eye(4) if j == NSEG // 2 else MIR)
        for x, a in SCREWS:
            p = (T @ np.array([x, YC + a, 0.0, 1.0]))[:2]
            cuts += [cyl([*p, RING[2] - 1], [*p, RING[3] + 1], R_CLOSE3, 24), cyl([*p, RING[2] - 1], [*p, RING[2] + 5.0], 3.0, 24)]
    return m - U(cuts)


def ring_pins():
    out = []
    for j in range(NSEG):
        a = seg_angle(j) + math.pi / NSEG
        out += [cyl([r * math.cos(a), YC + r * math.sin(a), RING[2] + 0.5], [r * math.cos(a), YC + r * math.sin(a), RING[3] - 0.5], ROD, 12) for r in (266.0, 286.0)]
    return tri(U(out))


def spokes():
    out = []; zm = (RING[2] + RING[3]) / 2
    for k in range(SPOKES):
        a = 2 * math.pi * k / SPOKES + math.pi / SPOKES; d = np.array([math.cos(a), math.sin(a)])
        e = box_edge(a); p0 = [*(e - d * 30.0), zm]; p1 = [*(np.array([0, YC]) + d * (RING[1] - 6.5)), zm]
        out.append(cyl(p0, p1, 3.0, 16))
    return tri(U(out))


def drive0():
    """the front motor (JGY-370, E: gearbox 46 x 32 x 25 against the stop post, its 6 mm shaft 14 into the coupling and plate 4,
    the motor hanging below, outside the ring) and the cam on its outer shaft"""
    return U([hbox(-16.0, 16.0, A_GB, A_GB + 25.0, ZB - 36.0, ZB + 10.0),
              cyl([0, YC + A_GB + 12.5, ZB - 67.0], [0, YC + A_GB + 12.5, ZB - 36.0], 12.2, 24),
              cyl([0, YC + A_GB - 14.0, ZB], [0, YC + A_GB + 39.0, ZB], 3.0, 16),
              cyl([0, YC + A_END - 9.0, ZB], [0, YC + A_END - 5.0, ZB], 10.0, 24)])


def switches0():
    """the two KW11 limit switches on the side walls' inner faces, either side of the cam (2 x M2 self-tapping each, after the motor)"""
    return U([hbox(sx * 13.1 - 3.2, sx * 13.1 + 3.2, A_END - 12.0, A_END - 2.0, ZB - 10.0, ZB + 10.0) for sx in (-1, 1)])


def coupling0():
    """the 6 mm flange coupling on plate 4's outer face (E: flange O22 x 2, hub O10, 12 long)"""
    return U([cyl([0, YC + A_P4, ZB], [0, YC + A_P4 + 2.0, ZB], 11.0, 32), cyl([0, YC + A_P4 + 1.0, ZB], [0, YC + A_P4 + 12.0, ZB], 5.0, 24)]) \
        - cyl([0, YC + A_P4 - 1.0, ZB], [0, YC + A_P4 + 13.0, ZB], 3.0, 16)


def base():
    """every fixed part, placed: key -> mesh"""
    out = {f"ring_{j:02d}": tri(ring_seg(j)) for j in range(NSEG)}
    out.update(ring_pins=ring_pins(), spokes=spokes(), lid=lid(), hoop_0=placed(hoop_rod(0, 0.0)))
    for sgn, s in SIDES:
        out[f"hub_{s}"] = placed(side(hub0(), sgn))
        out[f"pivot_{s}"] = placed(side(cyl([0, YC + A_TUBE[0], ZB], [0, YC + A_TUBE[1], ZB], 3.0, 24), sgn))
        out[f"drive_{s}"] = placed(side(drive0(), sgn))
        out[f"coupling_{s}"] = placed(side(coupling0(), sgn))
        out[f"switches_{s}"] = placed(side(switches0(), sgn))
    return out


def moving():
    """every moving part, shut (its rest pose): id -> mesh"""
    out = {}
    for k in range(1, NH):
        phi = math.radians(SHUT[k])
        out[f"box_dome_hoop_{k}"] = placed(hoop_rod(k, phi)); out[f"box_dome_pins_{k}"] = placed(pins(k, phi))
        for sgn, s in SIDES:
            out[f"box_dome_plate_{k}_{s}"] = placed(plate(k, phi, sgn))
    out.update({f"box_dome_fabric_{k}{j}": placed(strip(k, j)) for k in range(NH - 1) for j in range(M)})
    return out


# ---------------------------------------------------------------------------------------------------------------- the box's lines
def lines():
    """the dome's lines of systems/box.assembly.json (tools/import_box.py): every piece as placed, shut"""
    I = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    L = lambda key, name, role, mat, col, note, **kw: dict(id=f"box_dome_{key}", name=name, role=role, group="dome", color=col, mesh=f"parts/box_dome/gore_meshes/{key}.stl",
                                                           pose=I, material=mat, tag="C", owner="box dome", version="v4 prototype 15 Nov", source="parts/box_dome/make.py", note=note, **kw)
    P = lambda key, name, note, f: L(key, name, "structure", "PCTG", "#3a3f46", f"drawn prototype (parts/box_dome/make.py), print/box_dome_{f}.stl: {note}", part="box_dome")
    out = []
    for j in range(NSEG):
        kind = "spoke" if j % 2 else ("hub" if j % (NSEG // 2) == 0 else "plain")
        out.append(P(f"ring_{j:02d}", f"dome base ring, segment {j} ({kind})", "r 255..297 at W Z 130..142, half laps with two 3 mm carbon pins; " + PRINT[f"ring_{kind}"][1], f"ring_{kind}"))
    for sgn, s in SIDES:
        nm = "left" if s == "l" else "right"
        out.append(P(f"hub_{s}", f"dome hub bracket, {nm}", "base on the ring (4 x M3 x 12 from below), the inner wall with plate 0, the gearbox's side walls; " + PRINT["hub_l"][1], f"hub_{s}"))
    for k in range(1, NH):
        for sgn, s in SIDES:
            out.append(P(f"plate_{k}_{s}", f"dome hoop {k} plate, {'left' if s == 'l' else 'right'}", f"turns on the hub's tube, its pin at r {rho_pin(k):.0f}; " + PRINT["plate_1_l"][1], f"plate_{k}_{s}"))
    for k in range(NH):
        out.append(L(f"hoop_{k}", f"dome hoop {k} (3 mm carbon, {rod_len(k):.0f} long, crown r {crown(k):.0f})", "bought", None, "#222222",
                     ("fixed at the base, rear: in the hubs' own sockets" if k == 0 else f"shut at {SHUT[k]:.0f} deg; sprung between its two plates, bonded in their sockets")))
    for k in range(1, NH):
        out.append(L(f"pins_{k}", f"dome hoop {k} pins (3 mm carbon x 17)", "bought", None, "#222222", f"bonded in plate {k}, riding in the 45 deg slot of {'the hub wall' if k == 1 else f'plate {k - 1}'}"))
    for k in range(NH - 1):
        for j in range(M):
            out.append(L(f"fabric_{k}{j}", f"dome fabric, panel {k} strip {j}", "bought", None, "#e8eef4",
                         f"silnylon between hoops {k} and {k + 1}, drawn as 4 strips that follow them (the real skin is one sheet sewn to the hoops)", opacity=0.35))
    for sgn, s in SIDES:
        nm = "left" if s == "l" else "right"
        out.append(L(f"pivot_{s}", f"dome pivot, {nm} (6 mm carbon tube x 53)", "bought", None, "#222222", "bonded in the hub's inner wall; plates 1..4 turn on it"))
        out.append(L(f"coupling_{s}", f"dome drive coupling, {nm} (6 mm flange coupling)", "bought", None, "#b08d57", "four M3 x 8 into plate 4, set screws on the motor's shaft (E: flange O22, holes on O16)"))
        out.append(L(f"drive_{s}", f"dome drive, {nm}: worm motor and cam", "bought", None, "#555555", "JGY-370 12 V 6 rpm double shaft, self-locking, between the hub's side walls; a cam on the outer shaft"))
        out.append(L(f"switches_{s}", f"dome limit switches, {nm} (2 x KW11)", "bought", None, "#333333", "on the side walls either side of the cam: open and shut"))
    out.append(L("ring_pins", "dome ring lap pins (32 x 3 mm carbon x 11)", "bought", None, "#222222", "two through each half lap, resin from the top"))
    out.append(L("spokes", "dome spokes (6 mm carbon, under the lid)", "bought", None, "#222222", "eight, from the sheets' tops out to the spoke segments' sockets"))
    out.append(L("lid", "rain lid (3 mm UV polycarbonate)", "bought", None, "#cfe3f1", "at the sheets' top (W Z 142..145), flush between them and out to the ring; open within r 160 of the seat; notched round the two hubs", opacity=0.3))
    return out


def assembly_block():
    """the dome's part of the box's assembly check: no contacts (every bore is cut), the order the pieces go in and their paths"""
    ax = G()[:3, :3] @ np.array([0.0, 1.0, 0.0])                       # the front hub's outward axis, placed
    even = [f"box_dome_ring_{j:02d}" for j in range(0, NSEG, 2)]
    paths = [dict(name="dome ring: the upper-tongue segments onto the spoke segments", part=even[0], **{"with": even[1:]}, order=5, out=[0, 0, 1.0], length_mm=14.0, fit="mate",
                  why="the eight spoke segments sit on the spokes first; the others drop on from above, then the lap pins"),
             dict(name="dome hubs onto the ring", part="box_dome_hub_l", **{"with": ["box_dome_hub_r"]}, order=10, out=[0, 0, 1.0], length_mm=20.0, fit="mate",
                  why="each bracket down onto the ring's top inside the lid's notch, four M3 from below")]
    for k in range(1, NH):
        ride = [f"box_dome_plate_{k}_{s}" for _, s in SIDES] + ([f"box_dome_coupling_{s}" for _, s in SIDES] if k == NH - 1 else [])
        paths.append(dict(name=f"dome hoop {k} with its plates into the hubs", part=f"box_dome_hoop_{k}", **{"with": ride}, order=20 + k, out=[0, 0, 1.0], length_mm=45.0, fit="mate",
                          why="the rod bonded in its two plates on the bench, the hoop lowered between the hubs' walls" + (" (the couplings screwed to plate 4 first)" if k == NH - 1 else "")))
    for sgn, s in SIDES:
        o = (ax * sgn).round(6).tolist()
        paths.append(dict(name=f"dome pivot tube, {s}", part=f"box_dome_pivot_{s}", order=30, out=[-v for v in o], length_mm=A_TUBE[1] - A_TUBE[0] + 2.0, fit="bond",
                          why="pushed in from inside the dome through the wall and plates 1..3 into plate 4"))
        paths.append(dict(name=f"dome motor, {s}", part=f"box_dome_drive_{s}", order=45, out=o, length_mm=16.0, fit="free", ignore=[f"box_dome_coupling_{s}"],
                          why="slid in between the side walls from outside, its shaft into the coupling (bought in bought: the coupling's own bore) and plate 4"))
    order = {"box_dome_lid": 6, "box_dome_ring_pins": 7, "box_dome_hoop_0": 12, "box_dome_pins_*": 40, "box_dome_switches_*": 46, "box_dome_fabric_*": 50}
    return dict(contacts=[], paths=paths, order=order)


# ---------------------------------------------------------------------------------------------------------------- print files
SET = "0.2 layers, 4 walls, 25 % gyroid, PCTG at PETG + 10 C"
PRINT = {  # print file -> (count, pose and supports, time E)
    "hub_l": (1, "base down, no supports", "4h30"), "hub_r": (1, "base down, no supports", "4h30"),
    "ring_spoke": (8, "bottom face down, no supports", "2h30"), "ring_plain": (6, "top face down (as laid), no supports", "2h30"), "ring_hub": (2, "top face down (as laid), no supports", "2h30"),
}
PRINT.update({f"plate_{k}_{s}": (1, "outer face down (as laid), supports under the arm only (build plate only)", "0h40") for k in range(1, NH) for _, s in SIDES})


def print_files():
    """print file name -> mesh in its print pose, on z 0"""
    rx = lambda deg: trimesh.transformations.rotation_matrix(math.radians(deg), [1, 0, 0])
    out = {}
    for sgn, s in SIDES:
        out[f"hub_{s}"] = tri(side(hub0(), sgn))
        for k in range(1, NH):
            t = plate(k, 0.0, sgn); t.apply_transform(rx(-90.0 * sgn)); out[f"plate_{k}_{s}"] = t      # the outer face (away from the dome's centre) down
    for name, j in (("ring_spoke", 1), ("ring_plain", 2), ("ring_hub", 0)):
        t = tri(ring_seg(j)); t.apply_transform(trimesh.transformations.rotation_matrix(-seg_angle(j) + math.pi / 2, [0, 0, 1], [0, YC, 0]))
        if j % 2 == 0: t.apply_transform(rx(180.0))
        out[name] = t
    for t in out.values():
        t.apply_translation(-t.bounds[0])
    return out


def write_prints(log=print):
    sys.path.insert(0, os.path.join(ROOT, "tools")); from solid_check import solid_check
    from topo.build import render
    pd = os.path.join(ROOT, "print"); os.makedirs(pd, exist_ok=True); rows = []
    for name, t in print_files().items():
        assert t.volume > 0, name
        f = os.path.join(pd, f"box_dome_{name}.stl"); t.export(f)
        t = trimesh.load(f, force="mesh"); ok, info = solid_check(t)
        n, pose, time_ = PRINT[name]; g = t.volume / 1000 * PCTG
        bed = float(t.area_faces[(t.triangles_center[:, 2] < 0.05) & (t.face_normals[:, 2] < -0.99)].sum())
        rows.append(dict(file=f"print/box_dome_{name}.stl", count=n, g=round(g, 1), bed_mm=[round(float(v), 1) for v in t.extents], bed_contact_mm2=round(bed), solid_check="PASS" if ok else "FAIL",
                         thin_walls=len((info or {}).get("thin_walls", [])), pose=pose, settings=SET, time=time_))
        render(t, os.path.join(HERE, f"preview_{name}.png"), title=f"print/box_dome_{name}.stl on its bed (PROTOTYPE)", info=f"PCTG | {rows[-1]['bed_mm']} mm | {bed:.0f} mm2 on the bed | {g:.0f} g | x {n} | solid check {rows[-1]['solid_check']}")
        log(f"box_dome_{name}: x {n}, {g:.0f} g, {rows[-1]['bed_mm']} mm, {bed:.0f} mm2 on the bed, solid check {rows[-1]['solid_check']}; {pose}; {SET}, ~{time_} (E)")
        if not ok:
            log("   " + json.dumps({k: v for k, v in (info or {}).items() if k != "thin_walls"}, default=str)[:600])
    json.dump(dict(stage="PROTOTYPE", ok=all(r["solid_check"] == "PASS" for r in rows), pieces=rows), open(os.path.join(HERE, "make.json"), "w"), indent=1)
    return rows


def picture(fixed, mov):
    """preview.png: the left hub with its plates, tube, pins, coupling and motor, shut and folded"""
    from topo.build import render
    keys = ["hub_l", "pivot_l", "coupling_l", "drive_l", "ring_08"]
    ms = [fixed[k] for k in keys] + [m for i, m in mov.items() if i.endswith("_l")]
    t = trimesh.util.concatenate(ms)
    c = (G() @ np.array([0, YC + 300.0, ZB, 1.0]))[:3]
    t = t.slice_plane(c + [80, 0, 0], [-1, 0, 0]) if hasattr(t, "slice_plane") else t
    render(t, os.path.join(HERE, "preview.png"), title="box dome: the left hub, shut (PROTOTYPE)", info="hub bracket, plates 1..4 on the tube, coupling and worm motor, a ring segment")


# ---------------------------------------------------------------------------------------------------------------- the dome's own check
def check(mov, fixed):
    """every dome part against both drones (tools/seat_check.descent_states): the fixed parts at every state of each; the folded
    hoops and plates and the gathered fabric (bundle()) at every state of each; the moving parts at every 15 deg of hoop 4 both ways
    against the seated Scout v4 (the dome moves only with it seated) and against the dome's own fixed parts; the least gap to the
    seated Scout with the dome shut. Crossings as solids (seat frame), or as sampled points for a part whose mesh is not a solid."""
    sys.path.insert(0, os.path.join(ROOT, "tools")); import seat_envelope as se, seat_check as sc
    W2S = np.linalg.inv(np.array(sc.SEAT_TO_W))
    def to_s(m, T=np.eye(4)):
        t = m.copy(); t.apply_transform(W2S @ T); return (sc.man(t), t)
    fx = {k: to_s(m) for k, m in fixed.items()}
    assert all(v[0] is not None for v in fx.values()), f"not a solid: {[k for k, v in fx.items() if v[0] is None]}"
    def at(a4, shutting):
        P = poses(a4, shutting); return {k: to_s(m, P[k]) for k, m in mov.items()}
    def drone(d):
        out = []
        for nm, m in se.parts_of(d):
            mm = sc.man(m)
            out.append((nm, mm, None if mm is not None else np.vstack([trimesh.sample.sample_surface(m, 30000)[0], m.vertices]), m.bounds))
        return out
    def near(b0, b1, off=(0, 0, 0), pad=1.0):
        return not np.any(np.minimum(b0[1], b1[1] + off) + pad <= np.maximum(b0[0], b1[0] + off))
    def cross(f, part, off=(0.0, 0.0, 0.0)):
        (fm, ft), (nm, dm, pts, bb) = f, part
        if not near(ft.bounds, bb, np.asarray(off)): return 0.0
        if dm is not None:
            return sc.volume(fm ^ dm.translate(list(off)))
        q = pts + off
        sel = np.all((q >= ft.bounds[0] - 1) & (q <= ft.bounds[1] + 1), axis=1)
        return float(ft.contains(q[sel]).sum()) if sel.any() else 0.0
    opened = {k: v for k, v in at(0.0, False).items() if "fabric" not in k}
    opened["box_dome_fabric (gathered)"] = to_s(bundle())
    assert all(v[0] is not None for v in opened.values()), f"not a solid: {[k for k, v in opened.items() if v[0] is None]}"
    res = {}; own = []
    for a4 in range(0, 181, 15):                                                   # the mechanism against itself
        for shutting in (True, False):
            for k, f in at(float(a4), shutting).items():
                for kk, g in fx.items():
                    if not near(f[1].bounds, g[1].bounds): continue
                    v = sc.volume(f[0] ^ g[0])
                    if v > 0.5: own.append(f"{k} x {kk} at hoop 4 {a4} deg {'shutting' if shutting else 'opening'}: {v:.0f} mm3")
    res["mechanism"] = (None, own)
    for d in ("overseer", "scout_v4"):
        parts = drone(d); hits = []
        for lab, ox, oy, h in sc.descent_states(d):
            for k, f in list(fx.items()) + list(opened.items()):
                for part in parts:
                    v = cross(f, part, (ox, oy, h))
                    if v > 0.5: hits.append(f"{k} (open) x {part[0][:28]} at {lab}: {v:.0f} {'mm3' if part[1] is not None else 'points'}")
        gap = None
        if d == "scout_v4":
            for a4 in range(0, 181, 15):
                for shutting in (True, False):
                    for k, f in at(float(a4), shutting).items():
                        for part in parts:
                            v = cross(f, part)
                            if v > 0.5: hits.append(f"{k} at hoop 4 {a4} deg x {part[0][:28]} seated: {v:.0f}")
            shut = at(180.0, True)
            gap = min(f[0].min_gap(dm, 80.0) for f in shut.values() for _, dm, _p, _b in parts if dm is not None)
        res[d] = (gap, hits)
    return res


if __name__ == "__main__":
    out_dir = os.path.join(HERE, "gore_meshes"); os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    for k in range(NH):
        assert rod_len(k) < 1000.0, "a hoop longer than the 1 m rod"
    for k in range(1, NH):                                                    # the stack's own clearances: flat faces, and the arms nested when folded
        assert PITCH - PLATE_T > 1.0 and PITCH * SA - ARM_T > 1.0, "the plates touch"
    mov, fixed = moving(), base()
    for k, m in list(mov.items()) + [(f"box_dome_{k}", m) for k, m in fixed.items()]:
        m.export(os.path.join(out_dir, f"{k.replace('box_dome_', '')}.stl"))
    for f in sorted(os.listdir(out_dir)):                                     # no empty or inside-out part
        t = trimesh.load(os.path.join(out_dir, f), force="mesh")
        assert len(t.faces) > 0 and t.volume > 1.0, f"{f}: empty or inside out ({len(t.faces)} faces, {t.volume:.0f} mm3)"
    print(f"dome: hoops' crowns r {crown(0):.0f}..{crown(NH - 1):.0f}, rods {rod_len(0):.0f}..{rod_len(NH - 1):.0f} long leaving their sockets {math.degrees(ALPHA):.1f} deg from the axis (W Z {ZB:.0f} at W Y {YC:.0f}), "
          f"plates at +-{up(0):.1f}..{up(NH - 1):.1f}, ring r {RING[0]:.0f}..{RING[1]:.0f}")
    rows = write_prints()
    picture(fixed, mov)
    assert all(r["solid_check"] == "PASS" for r in rows), "a print file fails the solid check"
    if "--quick" not in sys.argv:
        res = check(mov, fixed)
        for d, (gap, hits) in res.items():
            print(f"{d}: " + (f"the shut dome {gap:.1f} clear of it seated; " if gap is not None else "") + f"{len(hits)} crossings" + "".join("\n  " + h for h in hits[:14]))
        assert all(not hits and (gap is None or gap >= 20.0) for gap, hits in res.values()), "the dome's own check fails"
