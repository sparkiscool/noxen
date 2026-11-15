"""Landing funnel v5 PROTOTYPE pieces (15 Nov, Spark: prototypes print as drawn parts): the two frame halves and the cross arm,
drawn, so they print today while the engine pieces (parts/box_funnel_half_nose|tail, parts/box_funnel_arm) wait for OPTIMISE.

What each piece is: the engine setup's room filled solid, less every one of its cuts (parts/box_funnel/make.py half_bodies,
arm_bodies, room()). So every interface is the setup's: the landing faces and seat, the ledges' strip pockets and probe bores, the
lap bolts to the four arms, the joint flanges at seat x 0, the motor seats, plates and stops, the tab slots and the nut blocks' lanes,
the switch lugs, the hold-down keys' sweep (wall_sweep), the rims on the bed plane, every keep-out (drones, latch, arm v8 lane, wheel,
packs, bays, lid, dome). The box check (tools/box_check.py) already checks the frame at its FULL ROOM, so it covers these pieces as
they are; nothing is added outside the room except, inside bolt pockets the bolt alone passes, the nut traps below (checked clear of
the drive's bolts and the walls' sweep here, ADD_CLEAR).

Changes from the bare room, each a fix for an assembly path the room alone does not give:
  motor plate slot   the setup's slot is open UP (the screw "drops in"), but above it is the rib: filled, nothing drops in. Here it is
                     open to the +y SIDE at the screw's height (4.6 tall, out past |y| 59) and the up-slot is filled: the gearmotor goes
                     in from the side with its screw on, the body through the setup's way-in cut, the screw along the open lane.
  nut traps          a nut at the bottom of a 14 deep round pocket spins and no tool reaches it. The tail half's three joint pockets
                     (M4 high, M4 top, M3 low) and the arm's two sheet-bolt pockets get a hex trap at the pocket's far end (AF nut +
                     0.2, nut + 0.3 deep): the nut goes down the pocket, turns into the hex, the bolt from the other side catches it.
                     The nose half takes the joint bolts' heads (hex key down the pocket). The lap nuts sit in open air inboard of
                     the beam (a 7 mm spanner).
Fits: systems/fit_table.json has not landed; the funnel's current clearances: M4 clear O4.4, M3 O3.4, heads and washers O9.6, nut
traps M4 7.2 AF x 3.5 (as the walls' nut blocks), M3 5.7 AF x 2.7.

Assembly path (order): 1 the two halves bolted at x 0 (nose half: heads; tail half: nuts in the traps), 4 M4 + 2 M3 x 16.
2 the TPU strips glued in the ledges, the probe sleeves in their bores. 3 each end wall slid in along x from its end, the tabs through
the side walls' slots (open at the end), its two M4 nuts dropped in the blocks' slots from above; set it out past "open".
4 each gearmotor in from +y beside its plate, body through the way-in cut, screw through the plate's side slot along the lane; a cable
tie through the tunnel under its floor; wall pushed in to the screw's tip and the motor run to take up the nut. 5 the switches up under
their lugs, 2 M2 each from below. 6 the four arms lapped on the beam ends, 4 M4 x 60 each (heads outboard on the arm, nuts inboard of
the beam), the frame lowered into the box and each arm's end plate bolted through the side sheet, 2 M4 x 16 (nuts in the arm's traps).

Hand calculation (PCTG E: Xt 46, Zt 23 MPa, knockdown 0.85, flight SF 4 -> in-layer 9.8, across-layer 4.9 MPa allowable; misuse
SF 2.5 -> 15.6 / 7.8. Index = stress / allowable, the engine's measure, limit 0.5 for an engine part). Sections are counted as
printed: 6 walls (2.4 mm), top and bottom 2.4 mm (12 layers at 0.2), 25 % gyroid; first solid, then the 2.4 skin alone (infill ignored).
  HALF, the case the setups' FE names worst: the Overseer 6 g off centre onto the +y face at the ear, 129.5 N normal to the 55 deg
    face (106.1 N level, 74.3 N down) at x 78 (the ear). The rail is held at the joint (x 0: three bolts and the flanges) and at the
    beam: taken as a cantilever from the joint, lever 78 (conservative: the beam end is ignored). M level 8.28 N m, M down 5.80 N m.
    Weakest section: over the motor (x 27..52), where the motor body and the nut block's lane take everything under z -32.2 outside
    |y| 46: left is the rib block, |y| 38..88 (50) x z -31.7..-16.3 (15.4) (the wall and ledge below it ignored).
      solid:  S down 1976 mm3 -> 2.9 MPa, S level 6417 mm3 -> 1.3 MPa: 4.2 MPa in-layer (bending along x, layers level) -> index 0.43
      skin:   S down 1394, S level 3154 -> 4.2 + 2.6 = 6.8 MPa -> index 0.69
      the setups' FE on the room filled solid (topo/audit.py room_full, h 2.2): nose 0.384, tail 0.399 at this case: the hand calc
      agrees. Torsion (74 N at |y| 46 about the block's centre |y| 63): 1.3 N m, 0.4 MPa: negligible.
  HALF, motor front plate, the gearmotor stalled driving in (135 N on the plate). Class: FLIGHT (SF 4), as the setups have it, not
    misuse: the box drives the walls unattended at every landing and swap, the inmost position ends with the nut block on this plate
    and a jam or a lost count stalls the motor there with a drone seated; nobody's hand is in it.
    With the side slot the plate is two lips, not a span: the slot (4.6 tall, |y| 49.7 out past the room's edge) parts it at the
    screw. The gearbox face (12 x 10) bears 2.7 high on each lip, half the load a lip by statics (the screw on the centre line):
    67.5 N. The upper lip hangs from the rib (root z -31.8, the motor's way-in ceiling), load centre z -34.65: lever 2.85,
    M 192 N mm, bending ACROSS the layers (printed upside down); the lip's inboard edge joins the solid behind the side wall and is
    ignored. The lower lip stands on the floor, lever 1.45: half of that.
      3.5 thick (to 15 Nov): S 13 x 3.5^2 / 6 = 26.5 mm3 -> 7.2 MPa -> index 1.48 (the two-ended span of the first calc read 1.03).
      6.5 thick (now, make.py PLATE_T; grown inboard: the gearmotor sits 3.0 further in, the shaft is cut to 65 not 62, the plate's
      front face, the nut block's stop, the screw's tip and the walls' travel are where they were): S 91.5 mm3 -> 2.1 MPa ->
      index 0.43; shear 1.5 x 67.5 / (13 x 6.5) = 1.2 MPa -> 0.24. The up-slot's 1 mm leftovers either side of the plate (notches
      at the lip's root, to z -26.3) are filled. 3.0 x the plate's thickness in cells at the setups' h 2.2.
  HALF, end-wall landing (106.1 N through the two screws onto two plates, 53 N a plate, 26.5 N a lip): index 0.17 on the plate.
  HALF, hold-down push (118 N less the drone, up under the nut blocks' roofs, 24 to 28 N a block): the roof over each block, solid over
    |y| 46..59, 3.4 deep minimum under the rib: 28 N on a 12 x 13 roof, M 28 x 12 / 8 = 42 N mm, S 13 x 3.4^2 / 6 = 25 -> 1.7 MPa, 0.17.
  HALF, handling 50 N out on the top edge (misuse): the 3.0 wall with the rim and rib behind it: cantilever of the rib block, 50 N at 28
    over the ledge -> 1.4 N m on the 50 x 15.4 block's S level 6417 (it reaches the joint): 0.2 MPa: nothing.
  ARM, the worst FE case (Overseer off centre onto the nose wall: -26.5 N along x, 30.2 N up at the end plate; seated 6 g: 32.4 N up):
    cantilever from the lap (|y| 88) to the plate (|y| 203), 24 (x) x 27.7 (z), lever 115: M up 3.47, M along 3.05 N m;
    S 3069 / 2659 mm3 -> 1.1 + 1.1 = 2.3 MPa in-layer (printed on its outer face: layers along the arm) -> index 0.23 solid,
    skin 0.33; under the dome ring (|y| > 166) the section drops to 24 x 16.5 but the lever to 37: 1.0 + 0.6 = 1.6 MPa, 0.17
    (FE room_full: 0.241 at the ring's step). Handling 50 N down (misuse): 50 x 115 = 5.75 N m -> 1.9 MPa, 0.12. The sheet bolts'
    nut traps: 30 N on an M4 nut's face on PCTG, 0.3 MPa.
"""
import os, sys, importlib.util
import numpy as np, trimesh, manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
_sp = importlib.util.spec_from_file_location("box_funnel_make_proto", os.path.join(HERE, "make.py"))
fm = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(fm)
import funnel_kit as fk

PCTG = 1.23
M4_TRAP, M3_TRAP = (7.2, 3.5), (5.7, 2.7)          # nut AF + 0.2, nut thickness + 0.3 (M4 7.0 x 3.2, M3 5.5 x 2.4)
SLICER = "PCTG, PETG profile + 10 C (nozzle 260, bed 80), 0.2 layers, 6 walls, 12 top / 12 bottom layers, 25 % gyroid, brim 5"
NAMES = {"half_nose": "box_funnel_proto_half_nose", "half_tail": "box_funnel_proto_half_tail", "arm": "box_funnel_proto_arm"}


def hex_x(x0, x1, y, z, af):
    """a hex prism along x, flats up and down (the nut's flats level: it goes down a round pocket and turns into it)"""
    r = af / np.sqrt(3.0)
    return mf.Manifold.cylinder(abs(x1 - x0), r, r, 6).rotate([0, 0, 30]).rotate([0, 90, 0]).translate([min(x0, x1), y, z])


def hex_y(y0, y1, x, z, af):
    r = af / np.sqrt(3.0)
    return mf.Manifold.cylinder(abs(y1 - y0), r, r, 6).rotate([0, 0, 30]).rotate([-90, 0, 0]).translate([x, min(y0, y1), z])


def half(sx):
    """(the piece in the seat frame, ADD: what is drawn inside the setup's cuts)"""
    drawn, cuts, env, R = fm.half_bodies(sx)
    pre = env + drawn
    m = pre - cuts
    z0 = fm.SEAT_Z; zc = z0 + fm.SCREW_H; mr, mf_, tip = fm.motor_x(sx); add = []; rem = []
    for sy in (-1, 1):
        Y = lambda a, b: (sy * a, sy * b)
        # the motor plate: up-slot filled, side slot cut
        add.append(fm.box(mf_, mf_ + sx * fm.PLATE_T, *Y(fm.SCREW_Y - 2.31, fm.SCREW_Y + 2.31), zc + 2.3, zc + 12.0) ^ pre)
        add.append(fm.box(mf_ - sx * 1.0, mf_, *Y(fm.SCREW_Y - 2.31, fm.SCREW_Y + 2.31), zc + 6.5, zc + 12.0) ^ pre)                       # the up-slot's 1 mm either side of the plate: notches at the upper lip's root
        add.append(fm.box(mf_ + sx * fm.PLATE_T, mf_ + sx * (fm.PLATE_T + 1.0), *Y(fm.SCREW_Y - 2.31, fm.SCREW_Y + 2.31), zc + fm.BLOCK[2] + 0.6, zc + 12.0) ^ pre)
        rem.append(fm.box(mf_ - sx * 0.5, mf_ + sx * (fm.PLATE_T + 0.5), *Y(fm.SCREW_Y - 2.3, fm.ROOM_Y + 2.0), zc - 2.3, zc + 2.3))
        if sx > 0:                                                                             # the nose half: the M3's head pocket left 0.9 of floor under it (room ends at z -56.5): open to the bottom
            (yl, zl, rl) = fm.JOINT["lo"]
            rem.append(fm.box(5.0, 19.0, *Y(yl - 3.4, yl + 3.4), z0 - 20.0, zl))
        if sx < 0:                                                                             # the tail: a 1.2 shelf of room over the lane behind the ear (|y| 56..59.5, and |y| 47.6.. beyond x -108; z -32.2..-31.0: the solid check's thin rule)
            rem.append(fm.box(-88.0, -115.0, *Y(55.95, 60.0), -32.6, -30.9))
            rem.append(fm.box(-108.4, -115.0, *Y(47.6, 60.0), -32.6, -30.9))                   # and where the ear's room stops at x -108, out to the shell's end
        if sx < 0:                                                                             # the tail half: the joint's nut traps
            (yh, zh, _), (yl, zl, _), (yt, zt, _) = fm.JOINT["hi"], fm.JOINT["lo"], fm.JOINT["top"]
            for (y, z, r, (af, d)) in ((yh, zh, 2.2, M4_TRAP), (yt, zt, 2.2, M4_TRAP), (yl, zl, 1.7, M3_TRAP)):
                ring = fm.box(-5.0, -5.0 - d, sy * y - 6.0, sy * y + 6.0, z - 6.0, z + 6.0) ^ pre
                add.append(ring - hex_x(-4.9, -5.1 - d, sy * y, z, af) - fm.cyl_x(-1.0, -19.0, sy * y, z, r))
    A = fm.U(add) if add else None
    out = (m + A if A is not None else m) - fm.U(rem)
    return out, A


def arm():
    drawn, cuts, env, R = fm.arm_bodies(1, 1)
    pre = env + drawn; m = pre - cuts; f0 = fm.FLANGE[0]; af, d = M4_TRAP; add = []
    for (x, y, z) in R["sheet_bolts"]:
        ring = fm.box(x - 6.0, x + 6.0, f0 - d, f0, z - 6.0, z + 6.0) ^ pre
        add.append(ring - hex_y(f0 - d - 0.1, f0 + 0.1, x, z, af) - fm.cyl_y(f0 - 30, f0 + 10, x, z, 2.2))
    A = fm.U(add); return m + A, A


def add_clear(A, sx=None):
    """what is drawn inside the setup's cuts against what passes there: the drive's bolts and screws, the end walls' sweep, the drones'
    space and the hole (mm3, each must be 0)"""
    keep = fm.drive() + fm.inner() + fm.box(*fm.HOLE, -120.0, 20.0)
    if sx is not None:
        keep = keep + fm.wall_sweep(sx)
    return (A ^ keep).volume()


def tri(man):
    mm = man.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def printed_mass(t, pitch=0.5, skin=2.4, infill=0.25):
    """grams as sliced: everything within `skin` of the surface solid, the rest at `infill` (voxels at `pitch`)"""
    from scipy import ndimage as ndi
    lo = t.bounds[0] - pitch; n = np.ceil((t.bounds[1] - t.bounds[0]) / pitch + 2).astype(int)
    xs, ys, zs = (lo[k] + (np.arange(n[k]) + 0.5) * pitch for k in range(3))
    from solid_check import _ray_parity
    V = _ray_parity(t, xs, ys, zs).reshape(n[0], n[1], n[2])
    d = ndi.distance_transform_edt(V) * pitch
    shell = (V & (d <= skin)).sum(); core = (V & (d > skin)).sum()
    return (shell + infill * core) * pitch ** 3 / 1000 * PCTG


def build(which):
    """write parts/<name>/part.stl (seat frame; the arm at nose +y) and print/<name>.stl (print pose), gate the print file; return a row"""
    name = NAMES[which]
    if which == "arm":
        man, A = arm(); clear = add_clear(A); R = fk.UP["-x"]
        pose = "on its outer face (the side away from the beam), the lap's bolt-head face down; x 2 as is (nose +y, tail -y), x 2 mirrored"
    else:
        sx = 1 if which == "half_nose" else -1
        man, A = half(sx); clear = add_clear(A, sx); R = fk.UP["-z"]
        pose = "upside down: the rims and the rib's top on the bed (the seat frame's z -16.3 plane)"
    assert clear < 0.5, f"{name}: drawn inside the cuts crosses the drive, a wall's sweep or the drones' space by {clear:.1f} mm3"
    bodies = man.decompose(); assert len(bodies) == 1, f"{name}: {len(bodies)} bodies"
    t = tri(man); assert t.is_watertight and t.is_volume, name
    d = os.path.join(ROOT, "parts", name); os.makedirs(d, exist_ok=True); t.export(os.path.join(d, "part.stl"))
    rows = []
    for mir in ((False, True) if which == "arm" else (False,)):
        nm = name + ("_mirrored" if mir else "")
        p = fk.posed(t, R, mir); f = os.path.join(ROOT, "print", nm + ".stl"); os.makedirs(os.path.dirname(f), exist_ok=True); p.export(f)
        ms = fk.measure(p); rows.append((nm, f, p, ms))
    return name, t, A, clear, rows, pose


if __name__ == "__main__":
    for w in sys.argv[1:] or list(NAMES):
        name, t, A, clear, rows, pose = build(w)
        print(f"{name}: {t.volume / 1000:.1f} cm3 solid, {t.extents.round(1).tolist()}, inside-cut adds {A.volume():.0f} mm3 clear by {clear:.2f}")
        for nm, f, p, ms in rows:
            print(f"  {nm}: {ms}")
