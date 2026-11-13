"""Engine setups of the landing funnel's generated pieces (v5, 13 Nov): the two frame halves and the cross arm. Geometry, room and
the drawn attachments come from parts/box_funnel/make.py (half_bodies, arm_bodies); this file adds the supports and the loads.

Loads (systems/dimensions.md L and System Box catch; E unless tagged), each its own case, flight class unless said:
  drones      Overseer 2200 g (21.6 N), Scout 500 g (4.9 N); landing 6 g: 129.5 N and 29.4 N
  seated      6 g on the four feet, on the two ledges (the Overseer's feet touch along x -92..91, the Scout's at x +-40..56, walls in)
  one side    the Overseer rolled, both feet of one side first: the 6 g on one ledge
  off centre  the 6 g reaction normal to the 55 deg face it strikes (level 106.1 N, down 74.3 N for the Overseer): on a side wall at
              its middle, at the joint and at the ear; on an end wall, where the level part goes through the tabs and nut blocks into
              the screws onto the motors' front plates (106.1 N Overseer, 24.1 N Scout)
  swap        pull-out 50 N down on the seat, the Overseer seated. The arm v8 head pushes up to 118 N (design chat 13 Nov): less the
              drone's weight it goes through the four hold-down pins into the end walls, their tabs and nut blocks, and up under the
              frame over each block (24.1 N a block Overseer, 28.3 N Scout)
  walls       a gearmotor stalled on its M4 screw, 135 N (E: N20 100:1 at 6 V, 0.55 kg cm stall, M4 x 0.7, 25 % screw efficiency):
              the motor's rear on its stop (a jam going out), and the screw pulling on the front plate (a jam going in)
  handling    50 N on a wall's top edge (out, down) and on an ear (misuse)
Supports: the faces a piece bolts to. A half: its two laps under the arms and its four joint flanges at x 0. An arm: its lap; the
sheet's reactions come in at its end plate as loads (shares of the cases above by the levers between the four plates).
Print: a half upside down (build -z: the side walls' top edges on the bed, supports under the ledges and the beam's wings). Standing
on its beam face (build -x, every face along x, no support) the rails and the rib behind each wall bend across the layers and the
room filled solid read 0.53 at the ear (limit 0.5); upside down it reads 0.36. An arm on its outer face (build -x)."""
import importlib.util, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from topo.part import Part

_sp = importlib.util.spec_from_file_location("box_funnel_make", os.path.join(ROOT, "parts", "box_funnel", "make.py"))
fm = importlib.util.module_from_spec(_sp); sys.modules[_sp.name] = fm; _sp.loader.exec_module(fm)   # in sys.modules: tools/part_sha.py records it as read by the setups

G = 9.81e-3
OV, SC, LAND = 2200.0 * G, 500.0 * G, 6.0
S55, C55 = np.sin(np.radians(55.0)), np.cos(np.radians(55.0))
PULL, STALL, HAND, PUSH = 50.0, 135.0, 50.0, 118.0
SPEC_HALF = dict(h=2.2, gen_cm3=140.0, max_worst=0.5, settings=dict(near_sym=True))
SPEC_ARM = dict(h=1.2, gen_cm3=40.0, max_worst=0.5)


def half_setup(sx, h):
    drawn, cuts, env, R = fm.half_bodies(sx)
    p = Part("box_funnel_half_" + ("nose" if sx > 0 else "tail"), h, drawn, cuts, env, build="-z", material="PCTG")
    z0 = fm.SEAT_Z; S = R["shell"]
    p.support("default", fm.U([R[k] for k in R if k.startswith(("lap", "joint"))]))
    ledge = lambda sy, a, b: S ^ fm.box(sx * a, sx * b, sy * 28.0, sy * 34.4, z0 - fm.FLOOR - 0.1, z0 + 0.2)
    face = lambda sy, a, b: S ^ fm.box(sx * a, sx * b, sy * 36.0, sy * 62.0, z0 + 13.0, z0 + 28.5)
    ear = lambda sy: S ^ fm.box(sx * 96.0, sx * 130.0, sy * 36.0, sy * 62.0, z0 + 10.0, z0 + 28.5)
    n = np.array([0.0, S55, -C55])                                                         # the push on the +y face
    ovl = OV * LAND
    p.load_multi("Overseer seated, 6 g on its four feet", [(ledge(sy, 5, 88), (0, 0, -ovl / 4)) for sy in (1, -1)], cls="flight")
    p.load("Overseer one side first, 6 g on the +y ledge", ledge(1, 5, 88), (0, 0, -ovl / 2), cls="flight")
    for nm, a, b in (("middle", 32, 58), ("joint", 2, 24), ("ear", 66, 90)):
        p.load(f"Overseer off centre onto the +y face, at its {nm}", face(1, a, b), ovl * n, cls="flight")
    p.load_multi("Overseer off centre onto this end wall (walls parked): the screws pull the gearboxes onto their front plates, the wall's foot on the ledges",
                 [(R[f"motor_front{sy:+d}"], (sx * ovl * S55 / 2, 0, 0)) for sy in (1, -1)] + [(ledge(sy, 72, 91), (0, 0, -ovl * C55 / 2)) for sy in (1, -1)], cls="flight")
    for nm, w in (("Overseer", OV), ("Scout", SC)):                                        # the hold-down: pins -> wall -> tabs -> nut blocks -> up under the frame over each block
        p.load_multi(f"arm head push {PUSH:.0f} N, the {nm} held down by this wall's two pins",
                     [(R[f"roof_{nm.lower()}{sy:+d}"], (0, 0, (PUSH - w) / 4)) for sy in (1, -1)], cls="flight")
    scl = SC * LAND
    p.load_multi("Scout seated, 6 g on its two feet of this end (walls in)", [(ledge(sy, 40, 56), (0, 0, -scl / 4)) for sy in (1, -1)], cls="flight")
    p.load_multi("Scout off centre onto this end wall (walls in): the screws pull the gearboxes onto their front plates",
                 [(R[f"motor_front{sy:+d}"], (sx * scl * S55 / 2, 0, 0)) for sy in (1, -1)] + [(ledge(sy, 44, 58), (0, 0, -scl * C55 / 2)) for sy in (1, -1)], cls="flight")
    p.load_multi("swap pull-out 50 N, the Overseer seated", [(ledge(sy, 5, 88), (0, 0, -(PULL + OV) / 4)) for sy in (1, -1)], cls="flight")
    p.load("wall driven out against a jam, the +y gearmotor stalled (135 N)", R["motor_rear+1"], (-sx * STALL, 0, 0), cls="flight")
    p.load("wall driven in against a jam, the +y gearmotor stalled (135 N)", R["motor_front+1"], (sx * STALL, 0, 0), cls="flight")
    p.load("handling 50 N out on the +y wall's top edge", face(1, 32, 58), (0, HAND, 0), cls="misuse")
    p.load("handling 50 N down on the +y wall's top edge", face(1, 32, 58), (0, 0, -HAND), cls="misuse")
    p.load("handling 50 N in on the +y ear", ear(1), (-sx * HAND, 0, 0), cls="misuse")
    return p


def foot_shares():
    """the sheet's reaction on one arm's end plate (the nose +y corner) per case: (name, force on the plate, class). Levers: the
    sheets at seat y +-206, the beams at x 116 and -146."""
    xb = (sum(fm.BEAM_X[1]) / 2, sum(fm.BEAM_X[-1]) / 2); yf = 206.0; ovl = OV * LAND
    ky = lambda y: (yf + y) / (2 * yf); kx = lambda x: (x - xb[1]) / (xb[0] - xb[1])
    return [("Overseer seated, 6 g: a quarter on each foot", (0, 0, ovl / 4), "flight"),
            ("Overseer one side first, 6 g on the +y ledge", (0, 0, ovl * ky(31.4) * kx(0.0)), "flight"),
            ("Overseer off centre onto the +y face at the nose ear", (0, -ovl * S55 * kx(78.0) / 2, ovl * C55 * ky(46.0) * kx(78.0)), "flight"),
            ("Overseer off centre onto the nose wall", (-ovl * S55 / 4, 0, ovl * C55 * kx(100.0) / 2), "flight"),
            ("swap pull-out 50 N, the Overseer seated", (0, 0, (PULL + OV) / 4), "flight"),
            ("arm head push 118 N, the Overseer held down by the pins", (0, 0, -(PUSH - OV) / 4), "flight"),
            ("handling 50 N along the box on the arm", (HAND, 0, 0), "misuse"),
            ("handling 50 N down on the arm", (0, 0, -HAND), "misuse")]


def arm_setup(h):
    drawn, cuts, env, R = fm.arm_bodies(1, 1)
    p = Part("box_funnel_arm", h, drawn, cuts, env, build="-x", material="PCTG")
    p.support("default", R["lap"])
    for nm, f, cls in foot_shares():
        p.load(nm, R["foot"], f, cls=cls)
    return p
