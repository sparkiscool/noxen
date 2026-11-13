"""The shared landing funnel (v5, 13 Nov; v3 3 Nov after the seat-envelope study, v4 9 Nov the sliding end walls).
Both drones land at one height with the battery over the box's hole, centred by shape (Spark 3 Nov). Seat frame = drone coordinates
of the seated Scout (x nose, y left, z up); the box's SEAT_TO_W puts it in W. Faces 55 deg, wall 3.0, the two y faces to h 28 (y
capture +-20), the two x faces to h 40 (x capture Scout +-28, Overseer +-18); seat = the Overseer latch v7 receiver's outline, hole
177 x 57 for its cartridge; the nose and tail walls slide 37.2 in for the Scout (SETTINGS).

v5 (13 Nov, Spark: a full run over the whole design, ready to print). v4 was one drawn body 245 x 418: it fits no bed, its cross
beams (12 x 12 and 12 x 8 PCTG over the 418 between the side sheets) read 11 and 25 MPa under a 65 N share of the Overseer's 6 g
landing and sagged over 10 mm, and its wall drive was a placeholder. Now:
  half_nose, half_tail  the frame in two pieces bolted at seat x 0 (one M4 high behind each side wall, one M3 under each ledge):
                        side walls, ledges, the corner ears, the middle of that end's cross beam. Through the engine
                        (parts/box_funnel_half_nose|tail/part.py): drawn = the faces a drone or a wall touches, the lap plates, the
                        joint flanges, the motor seats and stops, the switch lugs; the rest is generated in the room this file gives.
                        Each prints upside down, the walls' top edges on the bed (layers level: the rails and the beam bend in
                        the layers), with supports under the ledges and the beam's wings.
  arm (4)               the cross beam's outer 162 mm each side, under the rain lid (the lid is open only within r 160 of the seat):
                        lapped on the half's beam end with four M4, out to the side sheet's inner face (two M4 through the
                        acrylic, W Z 121.3). One setup (parts/box_funnel_arm), printed twice as drawn (nose +y, tail -y) and
                        twice mirrored. The beams are 24 wide and 29 to 42 deep (v4: 12 x 12 and 12 x 8); the tail one moved from
                        seat x -121..-109 to -158..-134, behind the rails' disc (r 151 over W Y 38..73) and over the hub (r 115).
  wall_nose, wall_tail  the sliding end walls: two tabs each out through slots in the side walls to an M4 nut block per side.
  drive                 per tab one N20 gearmotor with an M4 x 55 screw shaft (4 in all, self-locking: the walls hold a landing
                        with no power), seated on the half under the side wall's overhang; a roller switch at each wall's two
                        positions (+y side) pressed by a dog on the nut block; hard stops at "out". No opposite-hand screw, no gears.
  strips                TPU, flush in the ledges; four spring probes through them (seat sense, keep-alive).
Not drawn (named, not guessed): the hold-down. The swap's insert (60 N; the arm v8 head pushes up to 118 N, design chat 13 Nov) is more than either drone weighs (Overseer
21.6 N, Scout 4.9 N) and nothing on the funnel or the drones holds them down, so the insert case has no path through the seat.
Run: python parts/box_funnel/make.py [--gate]   (writes the drawn pieces' meshes, checks the walls' travel; --gate: tools/solid_check.py
     on every drawn printed piece)
"""
import os, sys, numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))
SEAT_Z, FACE_DEG, WALL = -44.3, 55.0, 3.0            # seat frame; seat z 0 is W Z 220.3
FLOOR = 4.5                                       # the ledge under the seat plane (v5: 3.0 left 1.5 under the strips' pockets, under the 2 mm floor; now 2.5)
H_X, H_Y = 40.0, 28.0                             # the x (nose / tail) faces and the y faces end here, above the seat
CX = -0.48                                        # the Overseer receiver's centre x in the seat frame (10 Nov: its tab line at seat x 0, on the Scout's and the arm's grip line)
SEAT = (CX - 92.0, CX + 92.0, -34.4, 34.4)        # the Overseer receiver's bottom (latch v7 CAD: 183.6 x 68.0, C) + 0.2 / 0.4 a side
HOLE = (CX - 87.0, CX + 90.0, -28.5, 28.5)        # the Overseer cartridge (latch v7 CAD: x -84.0..90.0, +-27.5 at its rails) + 1.5 / 1.0
ZW = 220.3                                        # seat z 0 in W
END_CUT_X, END_CUT_Y, END_CUT_H = 100.0, 50.0, 24.0   # the y walls' ends (|x| > 100) stop at h 24 outside |y| 50 (the Overseer's loop corner at its diagonal capture limit)
CORNER_Y = 42.0                                   # the x walls' top corners (h 28..40) stop at |y| 45 (an Overseer tube at the diagonal capture limit)
TAN = np.tan(np.radians(FACE_DEG))
W_H = WALL / np.sin(np.radians(FACE_DEG))         # the wall's thickness measured level: 3.66
SETTINGS = {"overseer": 0.0, "scout": 37.2}       # how far each end wall moves in from the 184 seat (the nose wall's foot at x 54.3, 1.3 clear of the Scout cartridge's x 53)
TRAVEL = max(SETTINGS.values())
# The hold-down (13 Nov, with the latch worker): the swap arm pushes up to 118 N, more than either drone weighs. Each end wall carries
# two steel pins at its corners that enter notches in the receiver's end faces and bear down on their lower faces. SETTINGS are the
# SEATED positions (walls driven in, pins 2.0 in the notches). The pins stand out past the walls' feet, so a drone that slides down an
# end wall lands that end on two pins: the box lands a drone with the walls at "park", backs them off to "open" (the pins leave from
# under a hung end, which drops at most 9.3), then drives in to "seated"; a drone leaves with the walls at "open".
POS = {"seated": 0.0, "park": 2.5, "open": 11.0}  # how far out of its seated position each wall stands
OPEN = POS["open"]
PIN = dict(y=32.4, z=-36.5, r=1.5, length=16.0, tip=2.3)     # a 3 x 3 steel square key (DIN 6885 key steel) x 16 along x, flats up and down (the latch worker: a round pin dents the notch, 101 MPa); its tip this far inside the wall's foot (2.0 into the notch + the 0.3 seated gap)
TAB_H = (3.5, 8.5)                                # each wall reaches its nuts by two tabs, h 3.5..8.5, through a slot in each side wall (h 3..9)
TAB_BACK = 5.0                                    # the tab runs this far behind the wall's back: 8 x 5 (v5: 3 x 4 was a neck under its nut block, the gate's neck rule)
STRIP_Y, STRIP_D = (31.2, 34.0), 2.0              # TPU strips flush in the side ledges (both drones land on them); 2.0 thick (1.5 was under the 2 mm floor)
PROBE_R = 1.55                                    # bore for an R156 sleeve (O2.89)
PROBES = [(1.5 + dx, sy * 32.6) for dx in (-30.0, 30.0) for sy in (-1, 1)]   # seat-sense / keep-alive spring probes in the ledges
# ---- the drive: per tab an N20 gearmotor with an M4 x 55 screw shaft (dimensions E from the listings: body 12 x 10 x 25 with its gearbox)
SCREW_Y, SCREW_H = 52.0, 6.0                      # the screws' axes: |y|, h (under the side walls' overhang; the wall's back is at |y| 38.1 + 0.7 h)
BLOCK = (6.0, 5.5, 5.5)                           # the nut block's half sizes x, y, z round the screw (M4 nut 7.0 AF x 3.2 in a slot from above)
GEAR_FACE, MOTOR_L, SCREW_L = 39.5, 25.0, 62.0    # the gearbox face this far inboard of the seat end; body length; screw length (the M4 x 100 shaft cut to 62: 48.2 of travel)
SW = (19.8, 10.2, 6.4)                            # KW12-3 roller switch body (x, y, z as mounted: lying flat, its roller toward the block), E
DOG = 25.0                                        # the dog: a bar along the +y block's outer side, its tip this far inboard of the block's centre
SW_H = 4.0                                        # the switches' underside over the seat plane (the 60 / 120 deg bays' saddles reach z -43 under them)
SW_Y = 62.5                                       # the switches' inner face, outboard of the nut blocks
LUG_Y = (SW_Y - 0.5, SW_Y + SW[1] + 1.0)           # the lug over the switches (two M2 up through each), hanging from the rib's room
LUG_H = (SW_H + SW[2] + 0.2, SW_H + SW[2] + 3.2)
# ---- the cross beams and the arms out to the side sheets
BEAM_X = {1: (138.0, 162.0), -1: (-128.0, -104.0)}   # seat x of the nose and the tail beam (turned pose, 13 Nov: W Y = 167.1 - seat x, so the NOSE beam is on the hub side, W Y 5..29, 5 behind the arm v8 rails; the tail one at W Y 271..295)
SHEET_TOP = 145.0 - ZW                            # -75.3; the rain lid lies flush with it (W Z 142..145), open within r 160 of the seat
LID_Z = SHEET_TOP - 3.0 - 2.0                     # everything outside the lid's opening stays under this (2.0 under the lid)
BAY = (40.2, 0.577, -80.0, -45.0)                 # the two bays beside the drone (60 / 120 deg, seat x -92..92): their inner faces at |y| = 43.1 + 0.577 (z + 80), kept 2.5 off
OPEN_R = 155.0                                    # the lid's opening about seat (0, 0.9), 5 inside its edge
MID_Z = (-99.5, -58.0)                            # the beam's middle: up to 4+ under both drones' lowest pass, inside the opening only
ARM_Z = (-108.0, LID_Z)                           # the arms, under the lid
MID_Y, LAP_Y = 88.0, (44.0, 88.0)                 # the half carries the beam to |y| 88; the arm laps it over |y| 44..88
LAP_BOLTS = [(y, z) for y in (52.0, 80.0) for z in (-95.0, -84.5)]   # four M4 along x through each lap
LAP_PAD_R = 6.0                                   # the pad round each lap bolt (washer 9)
HUB_R = 125.0                                     # the arm v8 head motors sweep r 120 behind the rails (seat x < -129), + 5, about the axle
ARM_W = 24.0                                      # the arm lies beside the beam's outer face, as wide as the beam (each alone carries the lap's moment)
RING_Y, RING_Z = 166.0, -91.5                     # beyond |y| 166 the arm passes under the dome's base ring (W Z 132..142, r 255 about the seat)
SPOKE = (52.0, 72.0, -88.5, 10.0)                 # a dome spoke crosses the arm's outer 10 mm over |y| 56..68 down to z -86: |y|, top there, width
FLANGE = (200.7, 205.7, -108.0, -91.5, 20.0)      # the arm's end plate on the sheet's inner face (|y| 206): |y|, z, and its reach inboard of the beam
SHEET_BOLT_Z = -99.0
JOINT = dict(hi=(53.0, -34.0, 2.2), lo=(47.0, -52.2, 1.7))   # the low one outside the carried pack's sweep
ARM_LANE = (88.1, 109.1, -116.0, 116.0, -58.8)   # arm v8's carriage and rods at the drone: nothing below W Z 161.5 over W X -28..116, Y 58..79 (turned pose: seat x 88.1..109.1, y -28..116; kept on both sides)
PACK_R = 169.0                                    # a pack carried on the wheel sweeps r 165 about the axle over W Y 114..210 (seat x -53..43), + 4   # the halves' joint at x 0: (|y|, z, hole r) of the M4 and the M3


def man(tri):
    return mf.Manifold(mf.Mesh(np.asarray(tri.vertices, np.float32), np.asarray(tri.faces, np.uint32)))
def hull(pts):
    return man(trimesh.convex.convex_hull(np.array(pts, float)))
def frustum(x0, x1, y0, y1, z0, z1, grow):
    return hull([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0 - grow, y0 - grow, z1), (x1 + grow, y0 - grow, z1), (x1 + grow, y1 + grow, z1), (x0 - grow, y1 + grow, z1)])
def box(x0, x1, y0, y1, z0, z1):
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1)); z0, z1 = sorted((z0, z1))
    return mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
def cyl_x(x0, x1, y, z, r, n=32):
    x0, x1 = sorted((x0, x1)); return mf.Manifold.cylinder(x1 - x0, r, r, n).rotate([0, 90, 0]).translate([x0, y, z])
def cyl_y(y0, y1, x, z, r, n=32):
    y0, y1 = sorted((y0, y1)); return mf.Manifold.cylinder(y1 - y0, r, r, n).rotate([-90, 0, 0]).translate([x, y0, z])
def cyl_z(z0, z1, x, y, r, n=32):
    z0, z1 = sorted((z0, z1)); return mf.Manifold.cylinder(z1 - z0, r, r, n).translate([x, y, z0])
def U(ms):
    return mf.Manifold.batch_boolean(list(ms), mf.OpType.Add)


def square_cut(sx, sy, edge, h):
    """the half space beyond the plane square to a 55 deg face through its inner edge at height h (the face on the +-x or +-y side, its
    foot at |x| or |y| = edge)"""
    c = mf.Manifold.cube([1000.0, 1000.0, 500.0]).translate([-500.0, -500.0, 0.0]); a = 90.0 - FACE_DEG
    c = c.rotate([-sy * a, 0, 0]) if sy else c.rotate([0, sx * a, 0])
    return c.translate([sx * (edge + h / TAN), sy * (edge + h / TAN), SEAT_Z + h])


def inner():
    x0, x1, y0, y1 = SEAT; z0 = SEAT_Z
    return frustum(x0, x1, y0, y1, z0, z0 + H_X + 1, (H_X + 1) / TAN)


def shell():
    """the four 55 deg faces and the ledge round the hole, as one shell (no beams, no drive)"""
    x0, x1, y0, y1 = SEAT; z0 = SEAT_Z
    gx, gy = H_X / TAN, H_Y / TAN
    outer = frustum(x0 - W_H, x1 + W_H, y0 - W_H, y1 + W_H, z0, z0 + H_X, gx)               # the outer faces parallel to the inner ones, 3.0 apart
    outer = outer + box(x0 - W_H, x1 + W_H, y0 - W_H, y1 + W_H, z0 - FLOOR, z0 + 0.5)       # the ledge, square-edged under the walls' feet
    top_cut = box(x0 - gy + 0.5, x1 + gy - 0.5, -200, 200, z0 + H_Y, z0 + H_X + 2)        # the y walls above h 28 go
    hole = box(*HOLE, z0 - 60.0, z0 + H_X + 2.0)
    end_cut = [box(sx * END_CUT_X, sx * 300, sy * END_CUT_Y, sy * 300, z0 + END_CUT_H, z0 + H_X + 2) for sx in (-1, 1) for sy in (-1, 1)]
    corner_cut = [box(x0 - 200, x1 + 200, s_ * CORNER_Y, s_ * 300, z0 + H_Y, z0 + H_X + 2) for s_ in (-1, 1)]
    # every top edge square to its face (v5: a level cut left a 55 deg knife edge, under 2 mm for its last 2.9: the gate's thin rule)
    sq = [square_cut(0, sy, y1, H_Y) for sy in (-1, 1)] + [square_cut(sx, 0, x1 if sx > 0 else -x0, H_X) for sx in (-1, 1)]
    sq += [square_cut(0, sy, y1, END_CUT_H) ^ box(sx * END_CUT_X, sx * 300, sy * END_CUT_Y, sy * 300, -100, 100) for sx in (-1, 1) for sy in (-1, 1)]
    sq += [square_cut(sx, 0, x1 if sx > 0 else -x0, H_Y) ^ box(sx * 50.0, sx * 300, s_ * CORNER_Y, s_ * 300, -100, 100) for sx in (-1, 1) for s_ in (-1, 1)]
    return outer - top_cut - U(corner_cut + end_cut + sq) - inner() - hole


def carriage_zone(sx):
    """the space the nose (sx +1) or tail (sx -1) wall is cut from: beyond the corner hips (|x - xend| >= |y| - 33.4), between the side
    walls' inner faces, from the seat up"""
    x0, x1, y0, y1 = SEAT; z0 = SEAT_Z + 0.01; xe = x1 if sx > 0 else x0; zt = SEAT_Z + H_X + 2.0; X = 120.0
    wedge = hull([(xe - sx * y1, 0.0, z0), (xe - sx * y1, 0.0, zt), (xe + sx * X, X + y1, z0), (xe + sx * X, -X - y1, z0),
                  (xe + sx * X, X + y1, zt), (xe + sx * X, -X - y1, zt)])
    g = (zt - z0) / TAN
    trough = hull([(xe - sx * 40, y0, z0), (xe - sx * 40, y1, z0), (xe + sx * X, y0, z0), (xe + sx * X, y1, z0),
                   (xe - sx * 40, y0 - g, zt), (xe - sx * 40, y1 + g, zt), (xe + sx * X, y0 - g, zt), (xe + sx * X, y1 + g, zt)])
    return wedge ^ trough


def carriage_home(sx):
    """the nose or tail wall at the Overseer setting"""
    return shell() ^ carriage_zone(sx)


def xend(sx):
    return SEAT[1] if sx > 0 else SEAT[0]


def tab_x(sx):
    """the x span the wall's tabs cover at the Overseer setting: the wall's face at h 4 to its back at h 8 (the tab leans with the wall)"""
    xe = xend(sx)
    a, b = xe + sx * (TAB_H[0] / TAN + 0.3), xe + sx * (TAB_H[1] / TAN + W_H - 0.3 + TAB_BACK)
    return min(a, b), max(a, b)


def shift(setting="overseer", pos="seated"):
    """how far in from its drawn place (the Overseer's seated position) a wall stands"""
    return SETTINGS[setting] - POS[pos]


def block_x(sx, setting="overseer", pos="seated"):
    """the nut block's centre x at a setting and position"""
    xa, xb = tab_x(sx); return (xa + xb) / 2 - sx * shift(setting, pos)


def nuts(sx):
    """the wall's two tabs out through the side walls and its two nut blocks (an M4 nut dropped in a slot from above, the screw through
    it); on the +y block the dog that rolls over the two switches"""
    z0 = SEAT_Z; xe = xend(sx); out = []; xm = block_x(sx); zc = z0 + SCREW_H; bx, by, bz = BLOCK
    for sy in (-1, 1):
        pts = []
        for h in TAB_H:
            f, b = xe + sx * (h / TAN + 0.3), xe + sx * (h / TAN + W_H - 0.3 + TAB_BACK)
            pts += [(x, y, z0 + h) for x in (f, b) for y in (sy * 34.0, sy * (SCREW_Y - by + 1.0))]
        out.append(hull(pts))
        out.append(box(xm - bx, xm + bx, sy * (SCREW_Y - by), sy * (SCREW_Y + by), zc - bz, zc + bz))
    xd = xm - sx * DOG                                                                                   # the dog: a bar from the +y block inboard, outside the motor's seat, its tip on the switch's roller
    out.append(box(xm + sx * 2.0, xd, SCREW_Y + by - 1.0, SW_Y - 1.5, z0 + SW_H + 1.0, z0 + SW_H + SW[2] - 0.4) - box(xm - sx * (bx + 0.5), xd - sx * 1.0, SCREW_Y + by - 1.5, SCREW_Y + 6.9, -100, 100))
    for sy in (-1, 1):                                                                                   # a boss on the wall's back at each corner for its hold-down pin
        out.append(box(xe + sx * 7.5, xe + sx * (PIN["length"] - PIN["tip"] + 1.0), sy * 28.5, sy * 36.0, PIN["z"] - 4.0, PIN["z"] + 4.0))
    cut = [cyl_x(xm - 20.0, xm + 20.0, sy * SCREW_Y, zc, 2.2) for sy in (-1, 1)]
    cut += [box(xm - 1.7, xm + 1.7, sy * (SCREW_Y - 3.6), sy * (SCREW_Y + 3.6), zc - 3.6, zc + bz + 1.0) for sy in (-1, 1)]   # the nut's slot
    return U(out) - U(cut)


def pins(sx):
    """the wall's two hold-down pins, at its drawn place"""
    xe = xend(sx)
    return U([box(xe - sx * PIN["tip"], xe + sx * (PIN["length"] - PIN["tip"]), sy * (PIN["y"] - PIN["r"]), sy * (PIN["y"] + PIN["r"]), PIN["z"] - PIN["r"], PIN["z"] + PIN["r"]) for sy in (-1, 1)])


def carriage(sx, setting="overseer", pos="seated", with_pins=True):
    """the wall with its tabs, nut blocks and pin bosses (and its two steel pins), at a setting and position"""
    xe = xend(sx)
    m = carriage_home(sx) + nuts(sx) - U([box(xe - sx * 6.0, xe + sx * (PIN["length"] - PIN["tip"]), sy * (PIN["y"] - PIN["r"] + 0.03), sy * (PIN["y"] + PIN["r"] - 0.03), PIN["z"] - PIN["r"] + 0.03, PIN["z"] + PIN["r"] - 0.03) for sy in (-1, 1)])   # the keys' square sockets: a press fit, blind
    return (m + pins(sx) if with_pins else m).translate([-sx * shift(setting, pos), 0.0, 0.0])


def motor_x(sx):
    """the gearmotor along x: (rear end, gearbox face, screw tip)"""
    f = xend(sx) - sx * GEAR_FACE
    return f - sx * MOTOR_L, f, f + sx * SCREW_L


def switch_x(sx):
    """the home switch's body centre along x, one a wall (+y side): its roller at the dog's tip with the wall at its outmost
    (Overseer, open), its body inboard of the roller (the Overseer's foot nodes pass beyond seat x 94 outside the ears). Every other
    position is timed from it (0.7 mm a turn); inmost, the nut block stops on the motor's front plate."""
    return (block_x(sx, "overseer", "open") - sx * (DOG + 7.0),)


def fixed():
    """the shell without the two sliding walls: side walls, ledges, the corner ears, the sills under the walls' feet; with the tabs'
    slots, the strips' pockets and the probes' bores"""
    z0 = SEAT_Z
    f = shell() - carriage_zone(1) - carriage_zone(-1)                                           # the zones, not the walls: no coplanar faces left as fins
    cuts = []
    for sx in (1, -1):
        xa, xb = tab_x(sx); lo, hi = (xa - TRAVEL - 1.0, xb + OPEN + 1.0) if sx > 0 else (xa - OPEN - 1.0, xb + TRAVEL + 1.0)
        cuts += [box(lo, hi, sy * 33.0, sy * (SCREW_Y - BLOCK[1] + 0.5), z0 + TAB_H[0] - 0.5, z0 + TAB_H[1] + 0.5) for sy in (-1, 1)]
    cuts += [box(SEAT[0] + 1.0, SEAT[1] - 1.0, sy * STRIP_Y[0], sy * STRIP_Y[1], z0 - STRIP_D, z0 + 0.01) for sy in (-1, 1)]
    cuts += [cyl_z(z0 - 30.0, z0 + 5.0, x, y, PROBE_R, 16) for x, y in PROBES]
    f = f - U(cuts)
    return U([m for m in f.decompose() if m.volume() > 5.0])


def wall_sweep(sx):
    """everything the wall, its tabs, blocks, dog, bosses and pins pass through between its outmost and inmost positions, 0.4 clear"""
    c = carriage(sx); m = U([c.translate([-sx * d, 0, 0]) for d in np.linspace(-OPEN, TRAVEL, 26)])
    z0 = SEAT_Z; xm0, xm1 = block_x(sx, "overseer", "open"), block_x(sx, "scout"); bx, by, bz = BLOCK; zc = z0 + SCREW_H
    lo, hi = min(xm0, xm1) - bx - 0.8, max(xm0, xm1) + bx + 0.8
    b = [box(lo, hi, sy * (SCREW_Y - by - 0.5), sy * (SCREW_Y + by + 0.6), zc - bz - 0.5, zc + bz + 0.6) for sy in (-1, 1)]
    b.append(box(min(lo, xm1 - sx * (DOG + 1.0)), max(hi, xm1 - sx * (DOG + 1.0)), SCREW_Y + 6.9, SW_Y - 1.0, z0 + SW_H + 0.5, z0 + SW_H + SW[2] + 0.1))      # the dog
    xa, xb = tab_x(sx); tl, th = (xa - TRAVEL - 1.0, xb + OPEN + 1.0) if sx > 0 else (xa - OPEN - 1.0, xb + TRAVEL + 1.0)
    b += [box(tl, th, sy * 33.0, sy * (SCREW_Y - by + 0.5), z0 + TAB_H[0] - 0.5, z0 + TAB_H[1] + 0.5) for sy in (-1, 1)]
    return m + U(b)


def half_bodies(sx):
    """one half of the frame for the engine: (drawn, cuts, env, regions). regions: named bodies for the supports and loads."""
    z0 = SEAT_Z; xe = xend(sx); bxa, bxb = BEAM_X[sx]; xin, xout = (bxa, bxb) if sx > 0 else (bxb, bxa)   # the beam's inner and outer face
    xmid = (bxa + bxb) / 2; side = box(0.0, sx * 400.0, -400, 400, -400, 400)
    S = fixed() ^ side
    R = dict(shell=S); drawn = [S]; cuts = []; env = []
    mr, mf_, tip = motor_x(sx); zc = z0 + SCREW_H; bx, by, bz = BLOCK
    for sy in (-1, 1):
        Y = lambda a, b: (sy * a, sy * b)
        # the lap under the arm: the half's side of the beam over |y| 44..88
        lap = U([cyl_x(xin, xout, sy * y, z, LAP_PAD_R) for y, z in LAP_BOLTS]) ^ box(-400, 400, -400, 400, MID_Z[0], LID_Z)   # a pad round each bolt; the lap between them is generated
        R[f"lap{sy:+d}"] = lap; drawn.append(lap)
        cuts += [cyl_x(xin - sx * 14.0, xout + sx * 40.0, sy * y, z, 2.2) for y, z in LAP_BOLTS]                           # M4 x 60 through both
        cuts += [cyl_x(xin, xin - sx * 14.0, sy * y, z, 4.8) for y, z in LAP_BOLTS]                                        # the nut and washer inboard
        cuts.append(box(xout, xout + sx * 40.0, *Y(0.0, 400.0), -200.0, LID_Z + 0.3))                                      # the arm, beside the beam's outer face
        # the joint at x 0: a flange high behind the wall (M4) and one under the ledge (M3)
        (yh, zh, rh), (yl, zl, rl) = JOINT["hi"], JOINT["lo"]
        fh = box(0.0, sx * 5.0, *Y(44.0, 60.0), -41.0, -27.0); fl = box(0.0, sx * 5.0, *Y(42.0, 52.0), -56.5, z0 - FLOOR + 0.5)
        R[f"joint_hi{sy:+d}"], R[f"joint_lo{sy:+d}"] = fh, fl; drawn += [fh, fl]
        cuts += [cyl_x(-sx * 1.0, sx * 19.0, sy * yh, zh, rh), cyl_x(sx * 5.0, sx * 19.0, sy * yh, zh, 4.8),
                 cyl_x(-sx * 1.0, sx * 19.0, sy * yl, zl, rl), cyl_x(sx * 5.0, sx * 19.0, sy * yl, zl, 3.4)]
        # the motor's seat: a floor under it, a plate its gearbox face pulls on (slot for the screw), a stop behind it
        y0m, y1m = SCREW_Y - 6.5, SCREW_Y + 6.5
        floor = box(mr - sx * 4.0, mf_ + sx * 3.5, *Y(y0m, y1m), z0 - 3.5, z0 + SCREW_H - 5.1)
        front = box(mf_, mf_ + sx * 3.5, *Y(y0m, y1m), z0 - 3.5, z0 + SCREW_H + 6.0)
        rear = box(mr, mr - sx * 4.0, *Y(y0m, y1m), z0 - 3.5, z0 + SCREW_H + 2.0)
        R[f"motor_front{sy:+d}"], R[f"motor_rear{sy:+d}"] = front, rear; drawn += [floor, front, rear]
        cuts.append(cyl_x(mf_ - sx * 1.0, tip + sx * 6.0, sy * SCREW_Y, zc, 2.3))                                           # the screw, through the plate and the stop
        cuts.append(box(mf_ - sx * 1.0, mf_ + sx * 4.5, *Y(SCREW_Y - 2.3, SCREW_Y + 2.3), zc, zc + 12.0))                   # the plate's slot: the screw drops in
        cuts.append(box(mr, mf_, *Y(SCREW_Y - 6.1, SCREW_Y + 6.1), z0 + SCREW_H - 5.1, z0 + SCREW_H + 5.2))                # the motor body
        cuts.append(box(mr, mf_, *Y(SCREW_Y, 75.0), z0 + SCREW_H - 5.1, z0 + SCREW_H + 6.5))          # its way in from the side
        xt = (mr + mf_) / 2
        cuts.append(box(xt - 2.0, xt + 2.0, *Y(y0m - 3.0, y1m + 3.0), z0 - 5.6, z0 - 3.5))                                  # the cable tie's tunnel under the floor
        roof = lambda setting: box(block_x(sx, setting) - bx, block_x(sx, setting) + bx, *Y(SCREW_Y - by, SCREW_Y + 0.4), zc + bz + 0.6, zc + bz + 4.0)
        R[f"roof_overseer{sy:+d}"], R[f"roof_scout{sy:+d}"] = roof("overseer"), roof("scout")                               # over the nut block, seated: where the hold-down's pull comes up into the frame
        # the switch lug, outboard of the blocks (switches on the +y side; the lug on both keeps the half mirrored)
        so, = switch_x(sx); l0, l1 = so - SW[0] / 2 - 1.0, so + SW[0] / 2 + 1.0
        lug = box(l0, l1, *Y(*LUG_Y), z0 + LUG_H[0], z0 + LUG_H[1]); drawn.append(lug)
        for xs in (so,):
            cuts.append(box(xs - SW[0] / 2 - 0.3, xs + SW[0] / 2 + 0.3, *Y(SW_Y - 3.5, LUG_Y[1] + 0.3), z0 + SW_H - 0.4, z0 + LUG_H[0]))     # the switch and its roller
            cuts += [cyl_z(z0 + SW_H - 1.0, z0 + LUG_H[1] + 1.0, xs + d, sy * (SW_Y + 7.3), 0.85, 12) for d in (-4.75, 4.75)]                # M2 self-tapping, up into the lug
        # ---- room
        zt = z0 + 27.0
        env.append(box(0.0, sx * 92.0, *Y(38.0, 59.0), z0 - FLOOR - 0.2, zt))                                               # behind the side wall
        env.append(box(sx * 92.0, sx * 112.0, *Y(38.0, 59.0), z0 - FLOOR - 0.2, -31.0))                                      # behind the ear: full depth under h 13,
        env.append(box(sx * 90.0, sx * (112.0 if sx > 0 else 108.0), *Y(38.0, 56.0), -33.0, zt))                                                    # 5 less above it (the Overseer's foot nodes and tubes pass |y| 59 from z -28)
        env.append(box(0.0, sx * (88.0 if sx > 0 else 76.0), *Y(46.0, 74.0), z0 + 12.6, zt))                                 # a rib's room outside the wall, over the drive (the Overseer's foot nodes pass beyond its ends)
        if sx > 0:
            env.append(box(ARM_LANE[1] + 0.3, 128.0, *Y(26.0, 56.0), -63.0, -50.0))                                         # down behind the arm's lane (the rails' tips pass under at z -68),
            env.append(box(126.0, xout, *Y(20.0, 88.0), -63.0, -55.0))                                                      # and on to the beam under the Overseer's nose (it passes z -50 there)
            env.append(box(92.0, 120.0, *Y(38.0, 56.0), z0 - FLOOR - 0.2, -41.5))                                            # the ear's foot, out to the beam (the Overseer's tubes pass over it from z -38)
            env.append(box(0.0, 126.0, *Y(28.5, 56.0), -56.5, z0 - FLOOR + 0.5))                                            # under the ledge and the motors
            env.append(box(92.0, 126.0, *Y(26.0, 56.0), ARM_LANE[4] + 1.0, z0 - FLOOR + 0.5))                                  # behind the nose wall (the lane's cut takes what is under z -58.8)
        else:
            env.append(box(0.0, -112.0, *Y(28.5, 56.0), -56.5, z0 - FLOOR + 0.5))
            env.append(box(-92.0, -108.0, *Y(26.0, 56.0), -63.0, z0 - FLOOR + 0.5))                                         # behind the tail wall
            env.append(box(-92.0, xout, *Y(26.0, 44.0), -63.0, z0 - FLOOR + 0.5))                                           # on to the beam inside the Overseer's tail tubes (they pass |y| > 46 down to z -52)
    env.append(box(xin, xout, -MID_Y, MID_Y, MID_Z[0], LID_Z))                                                               # the beam's middle, under the lid's level
    env.append(box(xin, xout, -MID_Y, MID_Y, LID_Z - 1.0, MID_Z[1]) ^ cyl_z(-200.0, 100.0, 0.0, -0.9, OPEN_R, 96))            # and up through the lid's opening
    cuts.append(cyl_x(129.1, 260.0, 0.0, -ZW, HUB_R, 96))                                                                  # the wheel's hub side behind the rails
    cuts.append(cyl_x(-52.0, 62.0, 0.0, -ZW, PACK_R, 128))                                                                   # a pack carried round on the wheel
    cuts.append(arm_lane())
    y0b, kb, zb, ztop = BAY
    for sy in (-1, 1):                                                                                                       # the bays beside the drone
        poly = [(sy * (y0b - kb * 30.0), zb - 30.0), (sy * 200.0, zb - 30.0), (sy * 200.0, ztop), (sy * (y0b + kb * (ztop - zb)), ztop)]
        pr = mf.Manifold.extrude(mf.CrossSection([poly if sy > 0 else poly[::-1]]), 190.0).transform([[0, 0, 1, -95.0], [1, 0, 0, 0], [0, 1, 0, 0]])
        cuts.append(pr)
    for x, y in PROBES:
        cuts.append(cyl_z(-90.0, z0 + 5.0, x, y, PROBE_R, 16))
    cuts.append(inner()); cuts.append(box(*HOLE, -120.0, 20.0))                                                               # the drones' space and the cartridge's hole
    cuts.append(wall_sweep(sx))
    cuts.append(box(-sx * 0.0, -sx * 400.0, -400, 400, -400, 400))                                                           # the other half
    return U(drawn), U(cuts), U(env), R


def arm_bodies(sx=1, sy=1):
    """one cross arm for the engine, in the seat frame at the corner (sx, sy): (drawn, cuts, env, regions). Under the rain lid, beside
    the beam's outer face over the lap (|y| 44..88), out to the side sheet's inner face."""
    bxa, bxb = BEAM_X[sx]; xin, a0 = (bxa, bxb) if sx > 0 else (bxb, bxa); a1 = a0 + sx * ARM_W
    Y = lambda a, b: (sy * a, sy * b); f0, f1, fz0, fz1, reach = FLANGE
    lap = U([cyl_x(a0, a1, sy * y, z, LAP_PAD_R) for y, z in LAP_BOLTS]) ^ box(-400, 400, -400, 400, MID_Z[0], LID_Z)
    flange = box(a0 - sx * reach, a1, *Y(f0, f1), fz0, fz1)
    fb = [a0 - sx * (reach - 8.0), a1 - sx * 8.0]                                                                             # two M4 through the sheet
    cuts = [cyl_x(xin - sx * 14.0, a1 + sx * 14.0, sy * y, z, 2.2) for y, z in LAP_BOLTS]
    cuts += [cyl_x(a1, a1 + sx * 14.0, sy * y, z, 4.8) for y, z in LAP_BOLTS]                                                 # the heads and washers outboard
    cuts += [cyl_y(sy * (f0 - 30.0), sy * (f1 + 10.0), x, SHEET_BOLT_Z, 2.2) for x in fb]
    cuts += [cyl_y(sy * (f0 - 30.0), sy * f0, x, SHEET_BOLT_Z, 4.8) for x in fb]                                              # the nuts and washers inboard of the plate
    cuts.append(box(a0 - sx * 100.0, a0, *Y(0.0, f0 - 14.5), -200.0, 20.0))                                                   # the half's beam, inboard of the arm
    cuts.append(box(-400, 400, *Y(f1, 400.0), -200.0, 20.0))                                                                  # the sheet
    cuts.append(box(-400, 400, *Y(RING_Y, 400.0), RING_Z, 20.0))                                                              # the dome's base ring
    cuts.append(box(a1 - sx * SPOKE[3], a1 + sx * 5.0, *Y(SPOKE[0], SPOKE[1]), SPOKE[2], 20.0))                               # a dome spoke
    cuts.append(cyl_x(129.1, 260.0, 0.0, -ZW, HUB_R, 96))                                                                     # the wheel's hub side behind the rails
    env = box(a0, a1, *Y(LAP_Y[0], f1), *ARM_Z) + box(a0 - sx * reach, a1, *Y(f0 - 14.0, f1), fz0, fz1)
    return U([lap, flange]), U(cuts), env, dict(lap=lap, foot=flange, sheet_bolts=[(x, sy * f1, SHEET_BOLT_Z) for x in fb])


def arm_lane():
    """the swap arm v8's head lane under the tail wall (design chat 13 Nov, decided for the arm): nothing of the funnel in it"""
    x0, x1, y0, y1, zt = ARM_LANE
    return box(x0, x1, y0, y1, -300.0, zt)


def room(drawn, cuts, env):
    """the most a generated piece can be: its room and drawn bodies less its cuts (what every clearance check uses, so a run's result
    is clear whatever it generates)"""
    return (env + drawn) - cuts


def frame():
    """the fixed funnel as the clearance checks see it: both halves and the four arms, each at its full room"""
    out = [room(*half_bodies(sx)[:3]) for sx in (1, -1)] + [room(*arm_bodies(sx, sy)[:3]) for sx in (1, -1) for sy in (1, -1)]
    return U(out)


def drive():
    """the bought side of the drive, as solids everything must clear: four gearmotors with their screws, the four switches (+y), the
    probes' bodies under the ledges, the lap, joint and foot bolts"""
    z0 = SEAT_Z; zc = z0 + SCREW_H; out = []
    for sx in (1, -1):
        mr, mf_, tip = motor_x(sx)
        for sy in (-1, 1):
            out.append(box(mr, mf_, sy * (SCREW_Y - 6.0), sy * (SCREW_Y + 6.0), zc - 5.0, zc + 5.0))
            out.append(cyl_x(mf_, tip, sy * SCREW_Y, zc, 2.0, 16))
        for xs in switch_x(sx):
            out.append(box(xs - SW[0] / 2, xs + SW[0] / 2, SW_Y, SW_Y + SW[1], z0 + SW_H, z0 + SW_H + SW[2]))
            out.append(box(xs - 1.5, xs + 1.5, SW_Y - 1.4, SW_Y, z0 + SW_H + 1.5, z0 + SW_H + SW[2] - 1.5))                  # its roller, pressed in by the dog
        for sy in (-1, 1):
            bxa, bxb = BEAM_X[sx]
            out += [cyl_x(min(bxa, bxb + sx * ARM_W) - 4.0, max(bxb, bxa + sx * ARM_W) + 4.0, sy * y, z, 2.0, 12) for y, z in LAP_BOLTS]
            out += [cyl_y(y - sy * 9.0, y - sy * 0.05, x, z, 2.0, 12) for x, y, z in arm_bodies(sx, sy)[3]["sheet_bolts"]]   # to the sheet's face (through it and its nut outside: the sheet's)
    for sy in (-1, 1):
        (yh, zh, rh), (yl, zl, rl) = JOINT["hi"], JOINT["lo"]
        out += [cyl_x(-8.5, 8.5, sy * yh, zh, 2.0, 12), cyl_x(-8.0, 8.0, sy * yl, zl, 1.5, 12)]
    for x, y in PROBES:
        out.append(cyl_z(z0 - 14.0, z0, x, y, 1.5, 16))
    return U(out)


def strips():
    """the two TPU strips, flush in the ledges, with the probes' holes"""
    z0 = SEAT_Z
    out = [box(SEAT[0] + 1.0, SEAT[1] - 1.0, sy * STRIP_Y[0], sy * STRIP_Y[1], z0 - STRIP_D, z0) for sy in (-1, 1)]
    return U(out) - U([cyl_z(z0 - 10.0, z0 + 10.0, x, y, PROBE_R, 16) for x, y in PROBES])


def funnel(setting="overseer", pos="seated"):
    """everything the drones, the latch sets and the arm must clear, at a setting ("overseer" = the 184 seat, "scout" = the walls
    37.2 in) and wall position (POS): the frame at its full room, both walls with their nut blocks and pins, the strips, the drive"""
    return frame() + carriage(1, setting, pos) + carriage(-1, setting, pos) + strips() + drive()


def stl(m, path):
    mm = m.to_mesh(); t = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
    os.makedirs(os.path.dirname(path), exist_ok=True); t.export(path); return t


PIECES = {"box_funnel_wall_nose": lambda: carriage(1, with_pins=False), "box_funnel_wall_tail": lambda: carriage(-1, with_pins=False), "box_funnel_strips": strips,
          "box_funnel_drive": drive}
ROOMS = {"box_funnel_half_nose": lambda: room(*half_bodies(1)[:3]), "box_funnel_half_tail": lambda: room(*half_bodies(-1)[:3]),
         "box_funnel_arm": lambda: U([room(*arm_bodies(sx, sy)[:3]) for sx in (1, -1) for sy in (1, -1)])}


if __name__ == "__main__":
    PARTS = os.path.dirname(HERE)
    for pid, fn in PIECES.items():
        t = stl(fn(), os.path.join(PARTS, pid, "part.stl")); print(f"{pid}: {t.volume / 1000:.1f} cm3, extents {t.extents.round(1).tolist()}, {len(fn().decompose())} body(ies)")
    for pid, fn in ROOMS.items():
        m = fn(); bb = m.bounding_box(); print(f"{pid} room: {m.volume() / 1000:.1f} cm3, extents {[round(bb[k + 3] - bb[k], 1) for k in range(3)]}")
    fr = frame() + drive()
    for sx in (1, -1):                                            # the walls never touch the frame's room or the drive, from outmost to inmost
        for d in np.linspace(-OPEN, TRAVEL, 14):
            v = (carriage(sx).translate([-sx * d, 0, 0]) ^ fr).volume()
            assert v < 1.0, f"the {'nose' if sx > 0 else 'tail'} wall {d:.1f} in crosses the frame or the drive by {v:.0f} mm3"
    print(f"walls clear of the frame's room and the drive from {OPEN} out to {TRAVEL} in")
    if "--gate" in sys.argv:                                       # the solid gate on every drawn printed piece, body by body (the strips are six loose pieces)
        sys.path.insert(0, os.path.join(os.path.dirname(PARTS), "tools")); from solid_check import solid_check
        bad = 0
        for pid in ("box_funnel_wall_nose", "box_funnel_wall_tail", "box_funnel_strips"):
            for k, body in enumerate(PIECES[pid]().decompose()):
                mm = body.to_mesh(); t = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)
                ok, info = solid_check(t); bad += not ok
                print(f"gate {pid}[{k}] {t.extents.round(1).tolist()}: {'PASS' if ok else 'FAIL'}" + ("" if ok else f" necks {info['necks']} thin {info['thin_walls']} eroded bodies {info['bodies_eroded']}"))
        sys.exit(1 if bad else 0)
