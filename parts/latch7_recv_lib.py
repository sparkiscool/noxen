"""Latch v7 receiver as an engine part, one setup for both packs (13 Nov, Spark: a full optimisation run over the whole swap design).
parts/latch7/make.py drew the receiver whole (two 5 mm side walls, a 3 mm deck, 6 mm end walls, ribs: 95 g Scout, 210 g Overseer).
Here only what the cartridge, the bolts, the drone and the funnel touch is drawn; the walls, the deck and the ends are generated.

Frame: the print pose, mount plane down on the bed (z 0), z up = the cartridge's -y. Cartridge coords (x across, y up to the drone,
z along the pack) -> part: (x, z, Y_M - y). `POSE[pack]` is the 4 x 4 back to cartridge coords (the page's pose).

Room (env): the receiver make.py draws (so every fit, motion, slant and funnel check that passes on it passes on anything generated
inside it) + the slab between the deck and the mount plane, which the drawn ribs and bosses already cross (E: nothing of the drone
is in it but the Scout FC's 1.0 relief, which stays out of it); the Overseer's side walls out to +-34.0 (WALL_OUT). The cartridge's path,
the carrier's notch, the strip recesses and the plug path are the room's own faces. Cuts: the mount bolts, their head counterbores and the
bolt pockets, through the whole room.

Drawn (attachments only):
  bosses      the mount bolts' lining and top faces (frozen: other streams bolt to them); the Scout's from its head seat up
  feet        the pads of the Scout's four landing feet (frozen, make.scout_feet; their legs are generated in the room make.py draws for them, round
              the mount bolts' head bores), each with its hold-down block on it, the notch in its end face (make.hold_down); the Overseer's four wall-thick blocks
              where the side walls run out: landing face underneath, hold-down notch in the end face
  pockets     3 mm round each bolt pocket: the 20 deg shelf the bolt lands on, the back wall, the ceiling, the lead-in under it
  seat pads   the deck's underside where the two TPU seat strips and their keys sit
  side pads   the walls' faces the cartridge's rails bear on, at the tab line and at the four corners
  end pads    the end walls' top bands that hold the cartridge along the pack, with their lead-ins (at the corners: the middle is relieved)

Supports: the boss faces on the mount plane (the lead-end pair under the carrier's ears: the same bolts clamp them).
Loads (systems/dimensions.md L, J; make.ARM; each its own case; the swap mass at its CG, the couple about the contact as a rigid
cartridge on the shelves and the seat strips, E):
  crash 35 g in the 6 face and 8 corner directions; flight 4 g + 2 g lateral (4); catch 6 g; swap insert at the head's limit on the seats (118 N,
  arm v8's float springs, every swap near it); swap across 43.5 N on a side pad (2); swap pull 57 N with the bolts in (fault); the same 118 N on
  one pocket's lead-in (a jammed bolt, fault, 2);
  on the feet: 6 g landing in the funnel and the XT60 pull-out (57 N + the drone's weight), on four; hold-down: the head's 118 N on the seats
  against the four tongues of the funnel's end walls in the corner notches (118 N less the drone's weight, down on the notches' lower faces).
The XT60's insert and pull-out act on the plug carrier and its cap, which the lead-end bolts clamp on the lead-end bosses: they reach
this part only as compression under those bosses (supports). Not applied, no number: a foot's sideways stop force in a skewed
landing; handling (L's 50 N is for cage members; the 35 g side hit on the same wall is 76 N Scout, 263 N Overseer)."""
import importlib.util, math, os, sys
import numpy as np
import manifold3d as mf
from topo import Part

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_s = importlib.util.spec_from_file_location("latch7_make", os.path.join(ROOT, "parts", "latch7", "make.py"))
mk = importlib.util.module_from_spec(_s); sys.modules.setdefault("latch7_make", mk); _s.loader.exec_module(mk)
bl, box, cyl, U = mk.bl, mk.box, mk.cyl, mk.U

SKIN = 3.0                         # pads: 3 thick (the audit's 2.5 floor, the gate's 2.0 erosion)
WALL_OUT = {"overseer": 34.0}      # the Overseer's side walls' room: 4.9 thick as drawn (27.9..32.8) is under one member (the audit's env_neck: pockets and feet reach
#                                    the bosses only through it); out to +-34.0, where its pocket blocks already are (the funnel's seat is +-34.4)
CORNER = 12.0                      # corner side pads run this far in from each end wall
ALLUP_G = {"scout": 222.0 + 510.0, "overseer": 2200.0}      # the drone with its pack (make.main's 510 g Scout without the pack, E; the Overseer's budget, L, E)
G = 9.81e-3                        # N per gram per g
S3 = 1 / math.sqrt(3)
DIRS14 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] + [(a * S3, b * S3, c * S3) for a in (1, -1) for b in (1, -1) for c in (1, -1)]


def frame(pack):
    """cartridge coords -> the part's (print) frame, as a function on manifolds, and the 4 x 4 back"""
    ym = mk.build(pack)["r"]["Y_M"]
    f = lambda m: m.rotate([-90, 0, 0]).translate([0, 0, ym])
    return f, [[1, 0, 0, 0], [0, 0, -1, ym], [0, 1, 0, 0], [0, 0, 0, 1]]


def bodies(pack):
    """drawn pads (dict name -> manifold), cuts, room: cartridge coords"""
    B = mk.build(pack); g, r = B["g"], B["r"]; recv = B["recv"]
    rf, xw, zt, ym = r["RF"], r["XW"], r["Z_T"], r["Y_M"]
    z0, z1 = r["Z_E0"] - r["END_T"], r["Z_E1"] + r["END_T"]
    bb = recv.bounding_box(); big = box(bb[0] - 1, bb[3] + 1, bb[1] - 1, bb[4] + 1, bb[2] - 1, bb[5] + 1)
    bolts = [cyl("y", bx, bz, bl.BOLT_D / 2, -5.0, ym + 1) for bx, bz in r["MOUNT"]]
    slab = box(-xw, xw, r["Y_DK1"] - 0.5, ym, z0, z1) - bl.carrier_zone(r, bl.FIT) - U(*bolts)
    if pack in bl.FC_RELIEF:
        a_, b_ = bl.FC_RELIEF[pack]; slab = slab - box(-200, 200, ym - 1.0, ym + 1.0, zt + a_, zt + b_)
    slab = slab - U(*[p_.translate([dx, 0, dz]) for p_ in B["poles"] for dx in (-0.3, 0.3) for dz in (-0.3, 0.3)])
    if B.get("clip") is not None: slab = slab ^ B["clip"]
    room = recv + slab
    if pack in WALL_OUT:                                       # side walls out to the line the pocket blocks already reach
        wo = U(*[box(*sorted((s * (rf + 1.0), s * WALL_OUT[pack])), bl.Y_RB, ym, z0, z1) for s in (1, -1)])
        room = room + (wo - bl.carrier_zone(r, bl.FIT) - U(*bolts))
    # cuts: the holes inside drawn material (mount bolts, head counterbores, bolt pockets, through the whole room). The cartridge's path, the carrier's
    # notch and the plug path are the room's own faces: the build cuts the part to the room exactly, with no 0.5 stand-off (a 5 mm wall stays 5)
    cuts = list(bolts) + [cyl("y", bx, bz, bl.HEAD_R, -5.0, bl.HEAD_Y) for bx, bz in r["MOUNT"] if abs(bx) > rf]
    for s in (-1, 1):
        pk, _ = bl.pocket_cut(r); cuts.append(pk if s > 0 else bl.mirx(pk))
    cuts = U(*cuts) ^ big
    D = {}
    for i, (bx, bz) in enumerate(r["MOUNT"]):
        lead = abs(bz - g["Z_X"]) < 1e-6
        top = r["Y_DK0"] if lead else ym
        if abs(bx) > rf:                                       # Scout: outboard column, head seat at HEAD_Y from below
            b_ = U(cyl("y", bx, bz, bl.BOSS_R, bl.HEAD_Y, top), cyl("y", bx, bz, bl.HEAD_R + 3.0, bl.HEAD_Y, bl.HEAD_Y + 3.0))
        else:
            b_ = cyl("y", bx, bz, bl.BOSS_R, r["Y_DK1"] - bl.INS_DEPTH - 2.0, top)
        D[f"boss {i}"] = b_ ^ recv
    for s, sn in ((1, "+x"), (-1, "-x")):
        X = lambda a, b: sorted((s * a, s * b))
        D[f"pocket {sn}"] = box(*X(rf, rf + r["PK_D"] + bl.PK_WALL), bl.Y_RB, r["PK_Y1"] + bl.PK_WALL, zt - r["PK_HZ"] - bl.PK_WALL, zt + r["PK_HZ"] + bl.PK_WALL) ^ recv
        hz = mk.SEAT_HZ + 0.2 + mk.SEAT_KEY + 1.5
        D[f"seat {sn}"] = box(*X(r["X_IN"] - 1.6 - 1.5, rf + SKIN), r["Y_DK0"], r["Y_DK1"], zt - hz, zt + hz) ^ recv
        yb = (g["Y_TOP"] - 8.0, r["Y_DK0"] + 0.01)
        D[f"side {sn} tab"] = box(*X(rf, rf + SKIN), *yb, zt - mk.SEAT_HZ, zt + mk.SEAT_HZ) ^ recv
        for zf, sg, en in ((r["Z_E0"], -1, "tail"), (r["Z_E1"], 1, "lead")):
            zz = sorted((zf + sg * r["END_T"], zf - sg * CORNER))
            D[f"corner {sn} {en}"] = U(box(*X(rf, rf + SKIN), *yb, *zz),
                                       box(*X(r["X_IN"] - 1.0, rf + SKIN), r["RAIL_Y0"], r["Y_DK0"] + 0.01, *sorted((zf - sg * 0.5, zf + sg * r["END_T"])))) ^ recv
    for i, (hb, w) in enumerate(zip(B["hold_blocks"], B["hold_where"])):       # foot + hold-down, one drawn body each: the Scout's pad with the block on it, the Overseer's wall-thick block
        D[f"foot {i}"] = (hb + B["foot_pads"][i] if "foot_pads" in B else hb) ^ recv
    cuts = cuts + (B["hold_notch"] ^ big); room = room - B["hold_notch"]
    return B, D, cuts, room


def setup_for(pack, name):
    def setup(h):
        B, D, cuts, room = bodies(pack); g, r = B["g"], B["r"]; T, _ = frame(pack)
        rf, zt, ym = r["RF"], r["Z_T"], r["Y_M"]
        drawn = U(*D.values())
        p = Part(name, h, T(drawn), T(cuts), T(room), build="+z", material="PCTG")
        R = lambda m: T(m)
        sup = [cyl("y", bx, bz, bl.BOSS_R + 0.5, (r["Y_DK0"] if abs(bz - g["Z_X"]) < 1e-6 else ym) - 1.5, ym + 0.5) for bx, bz in r["MOUNT"]]
        p.support("default", R(U(*sup)))
        W = g["SWAP_M"] * G; cg = np.asarray(g["SWAP_CG"], float)
        y_rail = g["Y_TOP"] - 5.0
        shelf = {s: box(*sorted((s * rf, s * (rf + r["PK_D"]))), bl.Y_RB, r["PK_Y0"] + 0.01, zt - bl.BOLT_HZ, zt + bl.BOLT_HZ) ^ D[f"pocket {sn}"] for s, sn in ((1, "+x"), (-1, "-x"))}
        half = lambda m, e: m ^ box(-200, 200, -200, 200, *sorted((zt, zt + e * 200.0)))
        V = lambda v: (v[0], v[2], -v[1])                      # a vector, cartridge -> part
        sn = {1: "+x", -1: "-x"}

        def inertia(n):
            """the swap mass pushing with n g (a vector, cartridge coords) on the receiver: [(region, force)]"""
            F = W * np.asarray(n, float); out = []
            add = lambda m, v: out.append((R(m), V(v)))
            fy = F[1]
            if fy < 0:
                for s in (1, -1): add(shelf[s], (0, fy / 2, 0))
            elif fy > 0:
                for s in (1, -1): add(D[f"seat {sn[s]}"], (0, fy / 2, 0))
            if abs(F[0]) > 1e-9:
                s = 1 if F[0] > 0 else -1
                for nm in (f"side {sn[s]} tab", f"corner {sn[s]} tail", f"corner {sn[s]} lead"): add(D[nm], (F[0] / 3, 0, 0))
                c = abs(F[0]) * (y_rail - cg[1]) / (2 * rf)
                add(shelf[-s], (0, -c, 0)); add(D[f"seat {sn[s]}"], (0, c, 0))
            if abs(F[2]) > 1e-9:
                e = 1 if F[2] > 0 else -1; en = "lead" if e > 0 else "tail"
                for s in (1, -1): add(D[f"corner {sn[s]} {en}"], (0, 0, F[2] / 2))
                c = abs(F[2]) * (y_rail - cg[1]) / ((bl.BOLT_HZ + mk.SEAT_HZ) * 2 / 3)
                for s in (1, -1):
                    add(half(shelf[s], -e), (0, -c / 2, 0)); add(half(D[f"seat {sn[s]}"], e), (0, c / 2, 0))
            return out

        for k, d in enumerate(DIRS14):
            p.load_multi(f"crash {k} 35 g ({d[0]:+.1f} {d[1]:+.1f} {d[2]:+.1f})", inertia(35 * np.array(d)), cls="crash")
        for nm, n in (("flight 4 g + 2 g +x", (2, -4, 0)), ("flight 4 g + 2 g -x", (-2, -4, 0)), ("flight 4 g + 2 g +z", (0, -4, 2)), ("flight 4 g + 2 g -z", (0, -4, -2)), ("catch 6 g", (0, -6, 0))):
            p.load_multi(nm, inertia(n), cls="flight")
        A = mk.ARM
        p.load_multi(f"swap insert, the head at its limit {A['stall_N']:.0f} N on the seats", [(R(D[f"seat {sn[s]}"]), V((0, A["stall_N"] / 2, 0))) for s in (1, -1)], cls="flight")
        for s in (1, -1):
            p.load_multi(f"swap across {A['across_N']} N {sn[s]}", [(R(D[f"side {sn[s]} tab"]), V((s * A["across_N"], 0, 0)))], cls="flight")
            # a jammed bolt stands OVERLAP (3.0) out of the cartridge: its tip meets the lead-in's outer, lower half (x rf + 1.5 .. rf + 3), not the thin lip under the pocket's mouth
            lead_in = box(*sorted((s * (rf + bl.CAP / 2), s * (rf + bl.CAP))), bl.Y_RB, bl.Y_RB + bl.HCAP / 2 + 0.3, zt - bl.BOLT_HZ, zt + bl.BOLT_HZ) ^ D[f"pocket {sn[s]}"]
            p.load_multi(f"stall {A['stall_N']:.0f} N on the {sn[s]} lead-in (fault)", [(R(lead_in), V((0, A["stall_N"], 0)))], cls="misuse")
        p.load_multi(f"swap pull {A['xt60_pull_N']:.0f} N, bolts in (fault)", [(R(shelf[s]), V((0, -A["xt60_pull_N"] / 2, 0))) for s in (1, -1)], cls="misuse")
        feet = [D[k] for k in D if k.startswith("foot")]; Wd = ALLUP_G[pack] * G
        Fh = (A["stall_N"] - Wd) / len(feet)                   # the head's push less the drone's weight comes back down through the four end-wall tongues
        hy = mk.HOLD["y"][0]
        floors = [D[f"foot {i}"] ^ box(*sorted(w["x"]), hy - 2.0, hy, *w["z"]) for i, w in enumerate(B["hold_where"])]       # under each notch's lower face
        p.load_multi(f"hold-down: the head's {A['stall_N']:.0f} N up on the seats, held by the funnel's four end-wall tongues",
                     [(R(D[f"seat {sn[s]}"]), V((0, A["stall_N"] / 2, 0))) for s in (1, -1)] + [(R(f_), V((0, -Fh, 0))) for f_ in floors], cls="flight")
        pad = lambda f_: f_ ^ box(-200, 200, bl.Y_RB - 1, bl.Y_RB + 4.0, -500, 500)
        p.load_multi("landing 6 g on the four feet (funnel)", [(R(pad(f_)), V((0, 6 * Wd / 4, 0))) for f_ in feet], cls="flight")
        p.load_multi(f"XT60 pull-out {A['xt60_pull_N']:.0f} N + 1 g on the four feet", [(R(pad(f_)), V((0, (A["xt60_pull_N"] + Wd) / 4, 0))) for f_ in feet], cls="flight")
        return p
    return setup


def POSE_OF(pack):
    return frame(pack)[1]
