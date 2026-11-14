"""Swap arm v8 (13 Nov 2026, Spark: "build the arm as well make sure things all are compatable and get it ready to print").
The box's arm for latch v7 and both packs, drawn in this repo (arm v7 was meshes only, for latch v4, never built).

What changed from v7 and why (C unless tagged; numbers in result.json):
  1. Reach. The seat is 65 higher than v7 reached (W Z 220.3, 3 Nov) and v7's one-stage head cannot get there: a longer rail hits the
     stand bars (r 165) and the side sheets (206), a longer carriage crosses the hub centre at park. Each head is two stages on one
     T8 x 2 lead screw: the lift (a yoke on one MGN12H block, its 100 mm rail bolted along the carriage's spine) is driven; it picks
     the carriage (three V wheels on v7's rail) up at the rail's top for the last part. Grab travel 115.8 + 1.5, lift 54.1 on the carriage.
     A return spring keeps the carriage down on the lift's stop. The screw runs in front of the hub's plate beside the spine, with two
     F8-16M thrust bearings at a wall on the plate. From the yoke three o 6 carbon rods run out under the grab unit (two printed ties
     on them carry its float bolts): the lever from the tab line is in carbon and in the steel rail, not in print.
     13 Nov, turned pose and this layout: the first v8 took the load 70 across to three rods, back through a 5 mm plate to the wheels and
     out along printed booms; none of its three setups could pass. Pull-out acts at the tab line in either pose (6.7 N m about the screw).
  2. Push. v7's pinion stalled at 74 N; the insert is 64.5 N (Overseer): x1.15. The screw gives over 380 N at stall (E), limited by the
     grab unit's float springs to 118 N: x1.83. A 2 mm lead holds the pack with the power off: v7's head brake relay and its bench
     drop gate go.
  3. Heads 180 apart (v7: 90). Each head's spine, rail and screw pass beside the hub centre on their own side; at 90 they cross.
     The arm is balanced; a swap turns 180 between pull and insert.
  4. Axle motor and detent on the stub (fixed), a printed GT2 ring on the hub's sleeve: nothing of the axle drive turns with the arm.
  5. Ring R 129 (v7: 113): latch v7's receivers overlap their neighbours at 113 and 30 deg, and two Scout bays side by side still touch at 127 (ring()).
  6. Grab unit for latch v7 (parts/latch7/make.py blade, heel, pads_of, F_BLADE: built to them): two sprung pilots enter the free
     corners of the recess 5 mm before the blades and take 3.0 across and along; the unit floats on four preloaded springs
     (+-2.5 across and along, tilt and roll give once the push passes the preload share); each heel pad on an M3 jack screw;
     an SS49E in each hall post; both packs' pads on one deck.
python3 parts/box_arm8/make.py [--fast]  -> parts/box_arm8_*/part.stl, parts/box_arm8/view/*.stl, parts/box_arm8/result.json,
                                            systems/swap_arm.assembly.json
"""
import json, math, os, sys
import numpy as np, trimesh
import manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "parts")); sys.path.insert(0, os.path.join(ROOT, "tools"))
import box_arm8_lib as L
from box_arm8_lib import box, cyl, hullp, U, vol, rot180

ARTIFACT = None        # the first publish writes it into systems/swap_arm.assembly.json; kept from there on a rerun
BED = 180.0
RHO = {"PCTG": 1.23, "TPU": 1.21}
BAYS = {"overseer": (0, -60, -120, -180), "scout": (30, -30, -90, -150, -210, 60, 120)}       # 11 bays, the drone at 90. The two beside the drone are Scout bays: an Overseer one meets the funnel at this radius (ring())


def tm(m):
    me = m.to_mesh(); return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=True)
def man(t):
    return mf.Manifold(mf.Mesh(np.asarray(t.vertices, np.float32), np.asarray(t.faces, np.uint32)))
def T34(T): return np.asarray(T, float)[:3, :]


# ------------------------------------------------------------------ fixed side: stub, post, axle drive, detent (W frame)
AXLE_M = (-50.0, 40.0)        # the axle stepper beside the mast, behind the stub's ear
RING_R = 47.75                # GT2 150T pitch radius on the hub's sleeve (teeth drawn as a plain band)
def fixed_parts():
    stub = U(cyl("y", 0, 0, 45.0, 0.0, 6.0), cyl("y", 0, 0, 24.95, 6.0, 40.0),
             hullp([[x, y, z] for y in (0.0, 6.0) for x, z in ((-28, 15), (-74, 15), (-74, 65), (-28, 65), (-20, 40))]),      # the motor's ear
             hullp([[x, y, z] for y in (0.0, 6.0) for x, z in ((20, 20), (62, 28), (62, 52), (30, 52))]))                      # the detent's ear
    stub = stub - cyl("y", 0, 0, 12.0, -1, 41) - cyl("y", *AXLE_M, 11.2, -1, 7)
    stub = stub - U(*[cyl("y", 38 * math.cos(a), 38 * math.sin(a), 2.75, -1, 7) for a in np.radians([45, 135, 225, 315])])
    stub = stub - U(*[cyl("y", AXLE_M[0] + sx * 15.5, AXLE_M[1] + sz * 15.5, 1.7, -1, 7) for sx in (-1, 1) for sz in (-1, 1)])
    piv = (52.0, 40.0)                                                                                                         # detent arm pivot (an M3 shoulder in the ear)
    stub = stub + cyl("y", *piv, 5.0, 6.0, 17.2) - cyl("y", *piv, 1.5, -1, 18)
    post = U(cyl("y", 0, 0, 28.5, 40.0, 44.0), cyl("y", 0, 0, 9.0, 44.0, 62.0)) - cyl("y", 0, 0, 4.25, 39, 63)
    rc = (RING_R + 1.5 + 5.0) / math.sqrt(2)                                                                                   # roller centre on the notch ring, at 45 deg
    arm = U(hullp([[x, y, z] for y in (17.4, 26.0) for x, z in ((piv[0] - 6, piv[1] - 6), (piv[0] + 6, piv[1] + 6), (rc + 7, rc + 13), (rc - 7, rc + 13), (rc - 7, rc - 2), (rc + 7, rc - 2))]),
            cyl("y", *piv, 7.0, 17.4, 26.0)) - cyl("y", *piv, 1.6, 16, 27) - box(rc - 8, rc + 8, 19.6, 23.8, rc - 8, rc + 6.5) - cyl("y", rc, rc, 1.5, 16, 27) - cyl("y", 0, 0, RING_R + 3.0, 16, 27, 96)   # 8.6 thick: 2.2 cheeks either side of the 623ZZ
    return dict(stub=(stub, "PCTG", "print"), post=(post, "PCTG", "print"), detent_arm=(arm, "PCTG", "print"),
                detent_roller=(cyl("y", rc, rc, 5.0, 19.7, 23.7) - cyl("y", rc, rc, 1.5, 20, 26), None, "bought"),
                axle_motor=(U(box(AXLE_M[0] - 21.15, AXLE_M[0] + 21.15, -48.0, 0.0, AXLE_M[1] - 21.15, AXLE_M[1] + 21.15), cyl("y", *AXLE_M, 2.5, 0.0, 24.0)), None, "bought"),
                axle_pulley=(cyl("y", *AXLE_M, 5.1, 9.0, 19.0) + cyl("y", *AXLE_M, 6.5, 19.0, 24.0), None, "bought"),
                bearings=(cyl("y", 0, 0, 39.98, 8.0, 40.0) - cyl("y", 0, 0, 25.0, 7, 41), None, "bought"))


# ------------------------------------------------------------------ hub (turns): the setup's drawn bodies, as the placeholder until its run
def hub_drawn():
    """what parts/box_arm8_hub/part.py draws (attachments only); the generated body joins them"""
    y0, y1, R = L.PLATE
    sleeve = U(cyl("y", 0, 0, 43.0, 8.0, 45.0, 96), cyl("y", 0, 0, RING_R + 1.5, 9.5, 25.5, 96))     # a 3 mm seat round the bearings; the body round it is generated                          # the bearings go in from the stub end; a 3 mm shoulder over the outer race at the plate end
    ring = cyl("y", 0, 0, RING_R + 1.5, 9.5, 25.5, 96)   # flush with the sleeve: no step                    # one band: GT2 teeth (150T) over y 9.5..19.5 and the detent's twelve notches over 20.5..25.5 are cut in it (not modelled)
    notch = ring
    bodies = [sleeve]                                                     # no drawn plate: only the rails' seats (below); the plate between them is generated
    for h in (0, 1):
        f = (lambda m: m) if h == 0 else rot180
        for w in (L.MOTOR_WALL, L.THRUST_WALL):
            bodies += [f(box(w["x"][0], w["x"][1], w["y"][0], w["y"][1], w["u"][0], w["u"][1]))]
        bodies += [f(cyl("y", 0, u, 7.0, 44.0, y0 + 0.01)) for u in L.RAIL_BOLTS] + [f(box(-9.5, 9.5, y1 - 3.0, y1, L.RAIL[0], R - 0.5))]
    return bodies
def hub_cuts():
    y0, y1, R = L.PLATE
    c = [cyl("y", 0, 0, 40.0, 6.0, 40.0, 96), cyl("y", 0, 0, 30.0, 39.9, y0 + 0.01), cyl("y", 0, 0, 15.0, y0, y1 + 1),
         cyl("y", 0, 0, 78.0, 6.0, 26.2, 96) - cyl("y", 0, 0, RING_R + 2.5, 5, 27, 96)]            # the fixed motor's pulley, the belt and the detent roller pass under the plate
    x, y = L.SCREW
    for h in (0, 1):
        f = (lambda m: m) if h == 0 else rot180
        m = L.MOTOR
        c += [f(box(m["x"][0] - 0.3, m["x"][1] + 0.3, m["y"][0] - 0.3, m["y"][1] + 0.3, m["u"][0] - 1, m["u"][1])),        # the stepper, through the plate
              f(cyl("z", x, y, 11.3, -45.5, -41)), *[f(cyl("z", x + sx * 15.5, y + sy * 15.5, 1.7, -45.5, -41)) for sx in (-1, 1) for sy in (-1, 1)],
              f(box(x - 9.0, x + 9.0, y1 + 0.2, y + 9.0, -41.5, -11.5)),                                                   # coupler and the inner thrust bearing, in front of the plate
              f(cyl("z", x, y, 4.3, -12.5, -6.5)),
              f(box(x - 8.2, x + 8.2, y1 + 0.2, y + 9.5, -7.5, 160.0)),                                                    # the lane the screw, the outer bearing, the lock ring and the nut's lug run in
              *[f(cyl("y", 0, u, 3.2, 43, 60)) for u in L.RAIL_BOLTS],                                                      # M5 heat-set inserts
              f(box(-12.5, 12.5, y1, 82, L.RAIL[0] - 1, 160)),                                                              # rail seat
              f(box(9.0, L.SPINE["x"][1] + 0.8, y1 + 0.5, 82, -45, 160)), f(box(-L.WHEEL_R - 22.8, -9.0, y1 + 0.5, 82, 0.0, 160))]   # the lanes of the wheel pair with the carriage's spine, and of the third wheel
    return c


# ------------------------------------------------------------------ the parts of one head, at park, arm frame
def head_parts(state="grip"):
    P = {}
    f = L.faces()[state]
    P["rail"] = (L.rail(), "PCTG", "print", "hub")
    for k, m in L.screw_drive().items(): P[k] = (m, None, "bought", "hub")
    car = U(L.carriage_env(), *L.carriage_bodies()) - L.carriage_cuts()
    P["carriage"] = (car, "PCTG", "run", "carriage")
    P["wheels"] = (U(*[L.wheel(x, c) for x, c in L.WHEELS]), "PCTG", "print", "carriage")
    P["mgn_rail"] = (L.mgn_rail(), None, "bought", "carriage")
    P["lift"] = (L.lift_env() - L.lift_cuts(), "PCTG", "run", "lift")
    for k, m in L.lift_bought().items(): P[k] = (m, None, "bought", "lift")
    P["ties"] = (U(*L.ties()), "PCTG", "print", "lift")
    T = L.arm_T()
    body, pads = L.unit_body()
    P["unit_body"] = (body.transform(T), "PCTG", "print", "unit")
    P["pads"] = (U(*pads).transform(T), "TPU", "print", "unit")
    P["blade_p"] = (L.blade(1, f).transform(T), "PCTG", "print", "unit")
    P["blade_n"] = (L.blade(-1, f).transform(T), "PCTG", "print", "unit")
    P["pilots"] = (U(*[L.pilot(sx, sz) for sx, sz in L.PILOTS]).transform(T), "PCTG", "print", "unit")
    for k, m in L.unit_bought(state).items(): P[k] = (m.transform(T), None, "bought", "unit")
    return P


def groups(state="grip"):
    """the turning arm as kinematic groups (manifolds, arm frame, park): hub, then per head carriage / lift (with the unit)"""
    H = head_parts(state); G = {"hub": [U(*hub_drawn()) - U(*hub_cuts())]}
    for k, (m, mat, kind, grp) in H.items():
        for hd, f in (("A", lambda q: q), ("B", rot180)):
            key = "hub" if grp == "hub" else f"{'lift' if grp == 'unit' else grp} {hd}"
            G.setdefault(key, []).append(f(m))
    return {k: U(*v) for k, v in G.items()}


def posed(G, phi, sA, sB, packs=()):
    """the groups in W at arm angle phi (head A's bar), head travels sA, sB; packs: (head, set manifold in the arm frame at park)"""
    R = L.Ry(phi); out = {}
    for k, m in G.items():
        s = 0.0 if k == "hub" else (sA if k.endswith("A") else sB)
        d = L.carriage_s(s) if k.startswith("carriage") else (s if k.startswith("lift") else 0.0)
        u = np.array([0, 0, d if not k.endswith("B") else -d, 1.0])
        T = R.copy(); T[:3, 3] = (R @ u)[:3]
        out[k] = m.transform(T34(T))
    for hd, m in packs:
        s = sA if hd == "A" else sB
        T = R.copy(); T[:3, 3] = (R @ np.array([0, 0, s if hd == "A" else -s, 1.0]))[:3]
        out[f"pack {hd}"] = (m if hd == "A" else rot180(m)).transform(T34(T))
    return out


# ------------------------------------------------------------------ the latch sets, the bays, the box
def SEAT_TO_ARM(z=None):
    """the funnel's seat frame -> W (= the arm frame with head A at the drone slot): tools/seat_check.py's SEAT_TO_W, the one definition
    (turned pose, 13 Nov: W x = seat y, W y = 167.1 - seat x). z: the seat's height instead of the box's 220.3 (44.3 = receiver bottom at radius 0)"""
    sys.path.insert(0, os.path.join(L.ROOT, "tools")); import seat_check
    T = np.array(seat_check.SEAT_TO_W, float)
    if z is not None: T[2, 3] = z
    return T


def sets():
    """per pack, in the arm frame with the receiver's bottom at radius 0 on the grip line: receiver, cartridge (bolts out), cartridge gripped"""
    m7 = L.latch(); m6, bl = m7.m6, m7.bl; out = {}
    for pk, B in L.packs().items():
        g = B["g"]; Ts = m7.to_seat(B)
        M = (SEAT_TO_ARM(44.3) @ np.vstack([Ts, [0, 0, 0, 1]]))[:3]
        out[pk] = dict(recv=m6.receiver_all(B).transform(M), cart=(m6.cart_all(B) + m6.moving(g, 0, 1) + m6.moving(g, 0, -1) + bl.pack_body(g)).transform(M),
                       held=(m6.cart_all(B) + m6.moving(g, m6.R0, 1) + m6.moving(g, m6.R0, -1) + bl.pack_body(g)).transform(M), top=float(m6.receiver_all(B).transform(M).bounding_box()[5]))
        rb = B["recv_bay"] + (m6.receiver_all(B) - B["recv"]) if "recv_bay" in B else (m6.receiver_all(B) - B["feet"] if "feet" in B else m6.receiver_all(B))   # a bay's receiver carries no landing feet: the latch's own recv_bay where it has one (design chat 13 Nov)
        out[pk]["recv_bay"] = rb.transform(M)
        r = B["r"]; out[pk]["mount"] = [[float(v) for v in (M @ np.array([bx, r["Y_M"], bz, 1.0]))] for bx, bz in r["MOUNT"]]      # the receiver's mount bolts, arm frame, receiver bottom at radius 0
    return out


def at_radius(m, r, ang=90.0):
    return m.translate([0, 0, r]).transform(T34(L.Ry(ang)))


def saddle(S, pk):
    """how a bay receiver hangs: a 4 mm plate on the receiver's mount plane, on the receiver's own mount bolts (the pattern its drone uses), open over the pack,
    with an ear at each end; link bars join neighbouring ears into the ring"""
    bb = S[pk]["recv_bay"].bounding_box(); mp = S[pk]["mount"]; zt = max(p[2] for p in mp); t = 4.0
    x0, x1 = min(p[0] for p in mp) - 4.5, max(p[0] for p in mp) + 4.5; y0, y1 = min(p[1] for p in mp) - 5.0, max(p[1] for p in mp) + 5.0
    plate = box(x0, x1, y0, y1, zt, zt + t) - box(x0 + 10.5, x1 - 10.5, y0 + 11.0, y1 - 11.0, zt - 1, zt + t + 1)
    plate = plate - U(*[cyl("z", p[0], p[1], 1.7, zt - 1, zt + t + 1) for p in mp])
    ears = [box(-9.0, 9.0, a_, b_, zt, zt + t) - cyl("z", 0.0, (a_ + b_) / 2, 1.7, zt - 1, zt + t + 1) for a_, b_ in ((y0 - 6.5, y0 + 0.01), (y1 - 0.01, y1 + 6.5))]   # M3 links
    return U(plate, *ears)


def ring(S, R=None):
    """neighbouring receivers (with seated cartridges) at 30 deg: overlap mm3 at v7's 113 and at R_BAY, and the closest saddles"""
    out = {}
    for r in (113.0, L.R_BAY):
        a = at_radius(S["overseer"]["recv"] + S["overseer"]["cart"], r, 90); b = at_radius(S["scout"]["recv"] + S["scout"]["cart"], r, 120)
        out[f"R {r:g}: receiver sets, Overseer beside Scout"] = round(vol(a ^ b), 1)
    for a_k, b_k in (("overseer", "scout"), ("scout", "scout")):
        a = at_radius(saddle(S, a_k) + S[a_k]["recv_bay"], L.R_BAY, 90); b = at_radius(saddle(S, b_k) + S[b_k]["recv_bay"] + S[b_k]["cart"], L.R_BAY, 120)
        out[f"bay sets with saddles, {a_k} beside {b_k}, no feet"] = round(vol(a ^ b), 1)
    for pk in ("overseer", "scout"):
        d = at_radius(S[pk]["recv"] + S[pk]["cart"], L.R_DRONE, 90)
        out[f"drone ({pk}) vs the Scout bays at 60 and 120"] = round(sum(vol(d ^ at_radius(S["scout"]["recv_bay"] + saddle(S, "scout"), L.R_BAY, a_)) for a_ in (60, 120)), 1)
    fz = dict(box_statics("overseer")[0])["funnel (overseer walls)"]
    out["an Overseer bay at 60 vs the funnel"] = round(vol(at_radius(S["overseer"]["recv"], L.R_BAY, 60) ^ fz), 1)
    out["a Scout bay at 60 vs the funnel"] = round(vol(at_radius(S["scout"]["recv"], L.R_BAY, 60) ^ fz), 1)

    return out


def box_statics(setting):
    """everything in the box that does not turn, as (name, manifold) in W: the funnel at a wall setting, the stand, the sheets and PSU, both
    drones' watertight parts. Not watertight (cage rods, wires): listed, checked by nobody here."""
    fm = L.latch()._funnel(); F = fm.funnel(setting)
    Ts = SEAT_TO_ARM()[:3]
    out = [(f"funnel ({setting} walls)", F.transform(Ts))]
    a = json.load(open(os.path.join(ROOT, "systems", "box.assembly.json"))); skipped = []
    for p in a["parts"]:
        pid = p["id"]
        if pid.startswith(("box_arm", "box_swaparm", "box_latch", "box_funnel", "box_dome", "box_ctx")): continue          # box_swaparm_*: this arm, as the box page imports it
        drone = "overseer" if pid.startswith("box_overseer") else "scout" if pid.startswith("box_scout") else None
        if drone and drone != setting: continue
        if drone and any(k in pid for k in ("receiver", "cartridge", "pack", "battery", "latch", "prop")): continue
        P = np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4)
        if p.get("mesh"):
            t = trimesh.load(os.path.join(ROOT, p["mesh"]), force="mesh"); t.apply_transform(P)
            if not t.is_watertight or t.bounds[0][2] > 300: skipped.append(pid); continue
            m = man(t)
            if m.is_empty() or m.volume() <= 0: skipped.append(pid); continue
        elif p.get("box"):
            b = p["box"]; m = box(*b).transform(T34(P))
        else: continue
        out.append((pid, m))
    return out, skipped


def bb_hit(a, b, pad=0.0):
    A, B = a.bounding_box(), b.bounding_box()
    return all(A[i] - pad <= B[i + 3] and B[i] - pad <= A[i + 3] for i in range(3))


def compat(S, fast=False):
    """the swap cycle for both packs against everything: every overlap over 0.5 mm3 is listed. States: both heads parked and carrying, the arm
    through a full turn; head A's stroke at the drone (rising open under the seated cartridge, gripped, coming down with it, and going up with it);
    the same at an Overseer bay and a Scout bay; the heads against each other and their own hub in every state."""
    G = {st: groups(st) for st in ("open", "grip")}
    rows, worst, n = [], {}, 0
    def test(label, moving, statics):
        nonlocal n
        n += 1
        for mk, mm in moving.items():
            for sk, sm in statics:
                if not bb_hit(mm, sm): continue
                v = vol(mm ^ sm)
                if v > 0.5:
                    key = f"{mk} | {sk}"
                    if v > worst.get(key, (0, ""))[0]: worst[key] = (round(v, 1), label)
    def internal(label, P):
        ks = list(P)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if ks[i].startswith("pack") and ks[j].startswith("lift") and ks[i][-1] == ks[j][-1]: continue      # a pack in its own grab unit: the latch's check
                if ks[j].startswith("pack") and ks[i].startswith("lift") and ks[i][-1] == ks[j][-1]: continue
                if not bb_hit(P[ks[i]], P[ks[j]]): continue
                v = vol(P[ks[i]] ^ P[ks[j]])
                if v > 0.5:
                    key = f"{ks[i]} | {ks[j]}"
                    if v > worst.get(key, (0, ""))[0]: worst[key] = (round(v, 1), label)
    fixed = [(k, v[0]) for k, v in fixed_parts().items()]
    skipped = set()
    step = 8.0 if fast else 4.0
    for pk in ("overseer", "scout"):
        box_s, sk = box_statics(pk); skipped |= set(sk)
        held = S[pk]["held"].translate([0, 0, L.G_PARK + L.CART_DROP])                       # a carried cartridge at park, arm frame
        bays = []
        for kind, angs in BAYS.items():
            for a_ in angs:
                bays.append((f"bay {a_} ({kind}) receiver", at_radius(S[kind]["recv_bay"], L.R_BAY, a_), a_, kind))
                bays.append((f"bay {a_} ({kind}) saddle", at_radius(saddle(S, kind), L.R_BAY, a_), a_, kind))
                bays.append((f"bay {a_} ({kind}) cartridge", at_radius(S[kind]["cart"], L.R_BAY, a_), a_, kind))
        drone = [(f"drone receiver ({pk})", at_radius(S[pk]["recv"], L.R_DRONE)), (f"drone cartridge ({pk})", at_radius(S[pk]["cart"], L.R_DRONE))]
        allst = box_s + fixed + [(n_, m) for n_, m, _, _ in bays] + drone
        # 1. the turn: both heads parked, both carrying this pack
        for phi in np.arange(0, 360, 10 if fast else 5):
            P = posed(G["grip"], float(phi), 0, 0, [("A", held), ("B", held)])
            test(f"{pk}: turning, phi {phi:g}, both heads carrying", P, allst); internal(f"{pk}: turning", P)
        # 2. head A at the drone
        st_nc = box_s + fixed + [(n_, m) for n_, m, _, _ in bays]
        for s in np.arange(0, L.S_DRONE + 0.01, step):
            P = posed(G["open"], 90.0, float(s), 0, [("B", held)])
            test(f"{pk}: head A rising open at the drone, s {s:g}", P, allst); internal(f"{pk}: A rising at the drone s {s:g}", P)
            P = posed(G["grip"], 90.0, float(s), 0, [("A", held), ("B", held)])
            test(f"{pk}: head A carrying at the drone, s {s:g}", P, st_nc + drone[:1]); internal(f"{pk}: A carrying at the drone s {s:g}", P)
        P = posed(G["grip"], 90.0, L.S_DRONE, 0, [("B", held)]); test(f"{pk}: head A gripped at the drone", P, st_nc + [drone[0], (drone[1][0], at_radius(S[pk]["held"], L.R_DRONE))])
        # 3. head A at one bay of this pack's kind, and at the bays next to the drone
        for a_ in ([0, -60, -180] if pk == "overseer" else [30, 60, 120, -90]):
            others = box_s + fixed + [(n_, m) for n_, m, aa, _ in bays if aa != a_] + drone
            own = [(n_, m) for n_, m, aa, _ in bays if aa == a_]
            for s in np.arange(0, L.S_BAY + 0.01, step):
                P = posed(G["open"], float(a_), float(s), 0, [("B", held)])
                test(f"{pk}: head A rising open at bay {a_}, s {s:g}", P, others + own); internal(f"{pk}: A at bay {a_} s {s:g}", P)
                P = posed(G["grip"], float(a_), float(s), 0, [("A", held), ("B", held)])
                test(f"{pk}: head A carrying at bay {a_}, s {s:g}", P, others + own[:2])
            P = posed(G["grip"], float(a_), L.S_BAY, 0, [("B", held)]); test(f"{pk}: head A gripped at bay {a_}", P, others + own[:2] + [(own[2][0], at_radius(S[pk]["held"], L.R_BAY, a_))])
        # 4. head B at the drone (the arm turned 180)
        for s in np.arange(0, L.S_DRONE + 0.01, step * 2):
            P = posed(G["grip"], 270.0, 0, float(s), [("A", held), ("B", held)])
            test(f"{pk}: head B carrying at the drone, s {s:g}", P, st_nc + drone[:1]); internal(f"{pk}: B carrying at the drone s {s:g}", P)
    # 5. the ring itself against the box (nothing of the arm): bays, saddles, seated packs vs the stand, sheets, funnel
    ringhits = {}
    box_s, _ = box_statics("overseer")
    for nm_, m, a_, kind in bays:
        for sk, sm in box_s:
            if bb_hit(m, sm):
                v = vol(m ^ sm)
                if v > 0.5: ringhits[f"{nm_} | {sk}"] = round(v, 1)
    lane = [dict(pair=k, mm3=v, worst_at=lab) for k, (v, lab) in sorted(worst.items(), key=lambda kv: -kv[1][0]) if "funnel" in k and k.split(" | ")[0][:-2] in ("carriage", "lift")]
    hits = [dict(pair=k, mm3=v, worst_at=lab) for k, (v, lab) in sorted(worst.items(), key=lambda kv: -kv[1][0]) if "box_stand" not in k and not ("funnel" in k and k.split(" | ")[0][:-2] in ("carriage", "lift"))]
    stand = [dict(pair=k, mm3=v, worst_at=lab) for k, (v, lab) in sorted(worst.items(), key=lambda kv: -kv[1][0]) if "box_stand" in k]
    return dict(states=n, overlaps=hits, zero=not hits, ring_vs_box=ringhits, arm_vs_stand=stand, head_vs_funnel_at_the_drone=lane,
                funnel_note="the funnel as of 4bef046 (turned pose) leaves the head's lane clear below W Z 161.5 over Y 58..79: the carriage tops out at 158.3, the yoke at 160.3 with the over-travel (inside |X| 28, Y 80..112)",
                stand_note="the stand in systems/box.assembly.json is the built demo v1's (mast, keel, kb, toe, bars), drawn for v4's 8 bays at R 100; whatever it meets here is a box-structure change, not an arm one")


def funnel_keepout():
    """(x0, x1, y0, y1, z0, z1) in W: where nothing of the funnel may be, for tools/box_check.py. The head standing at the drone slot, one-sided as it is
    (measured on the posed CAD with the over-travel: everything of it over W Z 140 is inside X -33.5..66.7): V rail (top 154), wheels, spine and carriage (158.3),
    screw (145), from the hub plate's face. Beyond W Y 79 the head is narrower still and is checked as CAD (compat()): the yoke (X -26.5..27.6, Y 80..112) and its
    plate and arm (X 27.5..66, Y 71.5..112) top out at 160.3. The turning arm's sweep is funnel_sweep(), not this box."""
    return (-36.0, 69.0, 54.0, 79.0, -1000.0, 161.5)


def funnel_sweep():
    """[(y0, y1, r)]: the turning arm at park, radius about the axle by band along it (measured on the posed CAD, + 0.5): the screw's top and the V rail's
    tip in the first two, the head stepper's corner and a carried Overseer in the third, the booms and grab unit beyond"""
    return [(38.0, 54.0, 155.0), (54.0, 68.0, 158.6), (68.0, 100.0, 123.0), (100.0, 260.0, 123.0)]


# ------------------------------------------------------------------ numbers the head must meet (E unless tagged)
def numbers():
    m7 = L.latch(); f = L.faces()
    stroke = f["open"] - f["grip"]
    Fg = L.K_BLADE * (25.0 - (L.SEAT_X - (f["grip"] + 7.5))); Fo = Fg + L.K_BLADE * stroke; Fl = Fg - L.K_BLADE * (f["grip"] - f["limit"])
    cam = m7.CAM_KGCM_AT_19p5N * Fo / 19.5
    T, lead, eta = 0.59, 2.0, 0.25
    F_stall = 2 * math.pi * eta * T * 1000 / lead
    back = 12.0 * lead / (2 * math.pi * 1000)                        # N m to hold 12 N with no friction at all
    out = dict(
        travel_mm=dict(park_pad_plane=L.G_PARK, bay=L.S_BAY, drone=L.S_DRONE, over=L.S_OVER, lift_on_carriage=L.T2, carriage=round(L.S_DRONE + L.S_OVER - L.T2, 1),
                       top_wheel_rim_r=round(L.Z0 + L.S_DRONE + L.S_OVER - L.T2 + L.WHEELS[1][1] + L.WHEEL_R, 1)),
        push=dict(insert_N=L.F_INSERT, asked_N=round(L.F_NEED, 1), float_preload_N=L.P_FLOAT, float_rate_N_mm=L.K_FLOAT,
                  at_over_travel_N=round(L.P_FLOAT + L.K_FLOAT * L.S_OVER, 1), limit_N=L.F_LIMIT, margin_over_insert="x%.2f" % (L.F_LIMIT / L.F_INSERT["overseer"]),
                  screw_stall_N=round(F_stall), screw="T8 x 2 at 0.59 N m, efficiency 0.25 (E)", v7_stall_N=74.0,
                  note="the screw is driven to a position 1.5 past the seat; the float springs set the force (72 N preload, 23 N/mm), not the motor's stall; over 30 N only in the last 8 mm, where the receiver guides the cartridge"),
        power_off=dict(holds="2 mm lead, lead angle 4.5 deg: self-locking at any friction over 0.08", torque_to_hold_12N_frictionless_Nm=round(back, 4), stepper_detent_Nm=0.015,
                       note="v7's brake relay on the head motors is not needed"),
        time_s=dict(full_stroke=round((L.S_DRONE + L.S_OVER) / lead / 15.0, 1), bay_stroke=round(L.S_BAY / lead / 15.0, 1), at="900 rpm (E)", v7_full_stroke=0.6,
                    note="a swap is four full strokes and a 180 deg turn: about 18 s seated to leaving (v7: 4.3 s, E); the drone is on keep-alive power"),
        blades=dict(face_mm=f, stroke_mm=round(stroke, 2), spring_N=dict(at_grip=round(Fg, 1), open=round(Fo, 1), at_rib_limit=round(Fl, 1)), latch_asks_N=m7.F_BLADE,
                    cam_kgcm=round(cam, 2), servo_margin={k: "x%.1f" % (v / cam) for k, v in m7.SERVO.items()}),
        locating=dict(pilots="two, sprung 5.0 ahead, in the recess corners the hall posts leave free", takes_across_mm=L.PILOT["taper"], takes_along_mm=L.PILOT["taper"],
                      float_mm=2.5, arm_error_mm=dict(across_rss=round(math.radians(0.30) * L.R_DRONE, 2), across_worst=round(math.radians(0.55) * L.R_DRONE, 2), along=1.8),
                      blades_alone_take_mm=dict(across=0.9, along=1.8), ok=bool(L.PILOT["taper"] >= math.radians(0.55) * L.R_DRONE and L.PILOT["taper"] >= 1.8),
                      order="pilot tips reach the cartridge 11.4 before the pads; they are home (3.0 taken) with the blade tops still 3.4 under the cartridge"),
        give=dict(roll_Nm_to_start=dict(carrying=round(L.P_FLOAT * L.FLOAT_X / 1000, 2), inserting_overseer=round((L.P_FLOAT - L.F_INSERT["overseer"]) * L.FLOAT_X / 1000, 2)),
                  pitch_Nm_to_start=dict(carrying=round(L.P_FLOAT * 32.5 / 1000, 2), inserting_overseer=round((L.P_FLOAT - L.F_INSERT["overseer"]) * 32.5 / 1000, 2)),
                  carried_overseer_roll_Nm_1g=round(0.766 * 9.81 * 0.058, 2), reach_deg=dict(roll=round(math.degrees(math.atan(3.0 / L.FLOAT_X)), 1), pitch=round(math.degrees(math.atan(3.0 / 32.5)), 1)),
                  latch_takes_rigid_deg=dict(roll=0.6, pitch=1.6), note="stiff while carrying, gives once the insert push has taken up the preload"),
        sensing="an SS49E in each hall post, 1.2 under its bolt's magnet; the head lets go only with both bolts read out (box electronics: two analog lines per head, as v7)",
        pads="the eight pads (four per pack) glued in 0.4 deep seats in the deck, each picked from 2.0 / 2.2 / 2.4 / 2.6 / 2.8 thick at set-up (+-0.4 in 0.2 steps; the latch's height window is +-0.21). "
             "14 Nov: they were on M3 jack screws, but the holes ended under the deck and two Scout pads stood over the pilots' slots")
    return out


def loads_report():
    """the head's cases and what they put on the lift's MGN block, the nut and the wheels (the carriage and hub setups take the same numbers)"""
    out = []
    for pk in ("overseer", "scout"):
        for nm, cls, F, P, s in L.cases(pk):
            R, M, C, idx = L.mgn_wrench(F, P)
            wl, hz, ax = L.wheel_loads(F, P, s)
            out.append(dict(case=nm, cls=cls, F_N=[round(float(v), 1) for v in F], at=[round(float(v), 1) for v in P], s=s,
                            mgn_Nm=dict(pitch=round(float(M[0]) / 1000, 2), yaw=round(float(M[1]) / 1000, 2), roll=round(float(M[2]) / 1000, 2)), mgn_index=round(idx, 3), nut_N=round(float(-F[2]), 1),
                            wheel_hertz_index=hz, wheel_625zz_axial_index=ax))
    wheels = dict(swap=dict(hertz=max(r["wheel_hertz_index"] for r in out if r["cls"] == "flight"), axial_625zz=max(r["wheel_625zz_axial_index"] for r in out if r["cls"] == "flight")),
                  misuse=dict(hertz=max(r["wheel_hertz_index"] for r in out), axial_625zz=max(r["wheel_625zz_axial_index"] for r in out)),
                  model="v7's joint indices (parts/box_arm/ref/swap_system.py: flank Hertz over 1.67 x 55 MPa with 10 N eccentric preload, 625ZZ axial over C0 670 N) on v8's statics: three wheels, each a force across and one along the axle; v7's own frame (beams + contact springs) is not rerun for v8's geometry")
    return dict(cases=out, wheels=wheels,
                mgn12h=dict(index_swap=max(r["mgn_index"] for r in out if r["cls"] == "flight"), index_misuse=max(r["mgn_index"] for r in out),
                            rating=dict(C0_N=L.MGN["C0"], MP_Nm=L.MGN["MP"], MY_Nm=L.MGN["MY"], MR_Nm=L.MGN["MR"]),
                            model="moments about the block's centre over HIWIN's MGN12H static ratings, summed with the cross force over C0 (E: the listing's own make gives none)"),
                lever=dict(tab_line_from_screw_mm=round(L.Y_C - L.SCREW[1], 1), swap_pitch_Nm=round(max(abs(r["mgn_Nm"]["pitch"]) for r in out if r["cls"] == "flight"), 2),
                           note="turned pose, 13 Nov: both plugs are on the hub side of the tab line (Overseer 78, Scout 43). The head pulls at the blades (the tab line) and pushes through the float springs (12.5 hub side of it), "
                                "so in the receiver's guides the lever stays the tab line's; the guides' couple can only shorten it and is not counted. The bound-cartridge case fell from 11.0 to 2.3 N m"),
                rule="the screw pushes over 30 N only in the last 8 mm of an insert, where the receiver's guides hold the cartridge's pitch; a jam at the plug end before that stops at 30 N (0.4 N m / deg x lever, inside the float)",
                from_v7="case list and forces from parts/box_arm/ref (arm_v7_system.json, swap_system.py): insert / pull in the guides and free, stall into a jam (here the float limit), axle stall with the head held in a bay, dropped pack 50 mm, handling 3 g down + 2 g side, hand 50 N on a rail tip (hub setup)",
                missing=["power cut with a head out: not a case any more (the 2 mm screw holds; v7's worst was 2.80 on the 625ZZ)", "the ring belt's tooth shear at the axle stall (E)", "v7's frame model itself is not rerun for v8's geometry: rail bending and hub turn under load are the engine's and hand numbers",
                         "the carriage's pick-up at its stop is a case of the carriage alone (17.4 N, E: return spring 10 N + 0.25 kg at 3 g); the return spring's anchor on the carriage is not drawn"])


# ------------------------------------------------------------------ print check
def bed_fit(t):
    """the smallest of a few flat poses' footprints on the 180 bed"""
    best = None
    for ax, nm in ((0, "x up"), (1, "y up"), (2, "z up")):
        e = t.extents; fp = [e[i] for i in range(3) if i != ax]
        ok = max(fp) <= BED and e[ax] <= BED
        if not ok:                                                     # diagonal on the bed
            d = math.hypot(*fp); ok = (max(fp) <= BED * 1.38 and min(fp) <= 40 and e[ax] <= BED); nm2 = nm + ", on the diagonal"
        else: nm2 = nm
        if ok and (best is None or e[ax] < best[1]): best = (nm2, float(e[ax]), [round(float(v), 1) for v in fp])
    return best


PRINTED = {  # part folder -> (source key or callable, material, print pose note, count)
    "rail": ("rail", "PCTG", "keel side up (the ridges' flanks at 45 deg, no supports)", 2),
    "wheels": (None, "PCTG", "one V wheel, groove axis up", 6),
    "unit_body": ("unit_body", "PCTG", "base plate down", 2),
    "blade": ("blade_p", "PCTG", "outer face down", 4),
    "pilot": (None, "PCTG", "on its outer side face", 4),
    "paddle": (None, "PCTG", "flat", 2),
    "tie": (None, "PCTG", "bar down (the cheeks up)", 4),
    "pads": (None, "TPU", "top face down, 2.4 thick (set-up spares 2.0 to 2.8)", 16),
    "stub": (None, "PCTG", "flange down", 1), "post": (None, "PCTG", "flange down", 1), "detent_arm": (None, "PCTG", "flat", 1),
    "saddle_overseer": (None, "PCTG", "flat", 4), "saddle_scout": (None, "PCTG", "flat", 7), "link": (None, "PCTG", "flat", 20)}


# poses, supports and the files themselves: parts/box_arm8/kit.py (measured on each print file), parts/box_arm8/kit.json
ORDER = ["Stub side: press the two 6010-2RS into the hub's sleeve from the stub end, slide the hub on the stub, screw the post on; axle stepper and 16T pulley on the stub's ear, belt round the hub's ring, detent arm with its 623ZZ on its pivot.",
         "Each V rail on the hub's plate: two M5 x 10 into the heat-set M5 inserts.",
         "Each carriage: four M3 inserts in the spine, the MGN12 rail on it with four M3 x 12 (block on the rail, stoppers off only at the bench); three V wheels, two 625ZZ in each, on M5 x 20 axles, the third on its eccentric bush; roll it onto the V rail and set the bush until it runs without play. The tab on the spine's side is the pick-up stop: the lift's nut lug meets it as the block's top reaches the rail's end.",
         "Each screw drive: stepper on its face plate (four M3), coupler, inner F8-16M, screw through the thrust wall, outer F8-16M, lock ring; the nut on the screw with its flange's two sides filed to the body.",
         "Each lift: bond the three o 6 x 115 carbon rods in the yoke's sockets (resin through each port until it shows at the mouth), slide the two ties on and bond them at their stations (float bolt centres 45 hub side and 20 far side of the tab line); four M4 inserts in each pair of ties.",
         "Yoke on the MGN block with four M3 x 12; the nut into its lug (flange in the slot); the return spring from the carriage to the rail's channel.",
         "Each grab unit (the steps and their sweeps: parts/box_arm8/kit.json test_set): servo in from the open side, two M3 x 8 through its inner ear; paddle onto the spline, M3 x 35 from the hub end; each pilot on two cut o 3 springs, its o 3 x 9 pin in from the blade bay; blades and their four springs, the two o 3 x 66 rods through; an SS49E in each hall post; the eight TPU pads into their seats.",
         "Grab unit on the ties: four M4 shoulder bolts down through the base with a PTFE and a steel washer under each head, a float spring round each bolt between tie and base.",
         "Ring: each bay's receiver on its saddle by the receiver's own mount bolts, the 20 link bars between neighbouring ears.",
         "Set up: with a cartridge gripped, each pad must touch the pack's bottom: swap a pad for the next thickness where it does not, then glue; park by the screw's position; check both hall readings with the bolts in and out."]


def current(k):
    """the engine part's result is of the setup on main, and passed"""
    import part_sha
    d = os.path.join(ROOT, "parts", "box_arm8_" + k); r = json.load(open(os.path.join(d, "result.json")))
    return bool(r.get("strength_ok") and r.get("solid_ok") and r.get("setup_sha") == part_sha.setup_sha(os.path.join(d, "part.py")))


def print_kit(pr):
    """the print kit as parts/box_arm8/kit.py measured it (kit.json: files, poses, supports, order, test set), and the three engine parts' state"""
    gen = {k: json.load(open(os.path.join(ROOT, "parts", "box_arm8_" + k, "result.json"))) for k in ("hub", "carriage", "lift")}
    cur = {k: current(k) for k in gen}
    kj = os.path.join(HERE, "kit.json"); kit = json.load(open(kj)) if os.path.exists(kj) else {}
    return dict(hold=("All three engine parts are of the setups on main and passed." if all(cur.values()) else
                      "Drawn pieces: print. Engine parts waiting on their runs: " + ", ".join(k for k in cur if not cur[k]) + " (14 Nov: the carriage's bed plane and pick-up stop, the lift's bed, "
                      "and one load point the latch moved 0.06 mm; the 13 Nov shapes are not to be printed)"),
                generated=[dict(part="box_arm8_" + k, count=1 if k == "hub" else 2, current=cur[k], worst=g["strength"]["worst"], mass_g=g["mass_g"], solid_check="PASS" if g.get("solid_ok") else "FAIL") for k, g in gen.items()],
                kit="parts/box_arm8/kit.json", files=kit.get("files", []), print_order=kit.get("order", []), test_set=kit.get("test_set"), open=kit.get("open", []), order=ORDER)


def main():
    fast = "--fast" in sys.argv
    out_dir = os.path.join(HERE, "view"); os.makedirs(out_dir, exist_ok=True)
    S = sets(); H = head_parts("grip"); FX = fixed_parts()
    old = json.load(open(os.path.join(HERE, "result.json"))) if os.path.exists(os.path.join(HERE, "result.json")) else {}
    res = dict(version="arm v8 (13 Nov 2026)", numbers=numbers(), loads=loads_report(), ring=ring(S), bay_mount=bay_mount(S))
    for k in ("audit_smoke", "solid_check"):
        if k in old: res[k] = old[k]
    print("ring", res["ring"])
    # ---- printed pieces
    pr = {}
    def one(name, m):
        d = os.path.join(ROOT, "parts", "box_arm8_" + name); os.makedirs(d, exist_ok=True)
        t = tm(m); t.export(os.path.join(d, "part.stl"))
        bf = bed_fit(t); mat = PRINTED[name][1]
        pr[name] = dict(part="box_arm8_" + name, material=mat, count=PRINTED[name][3], g_each=round(t.volume / 1000 * RHO[mat], 1), box_mm=[round(float(v), 1) for v in t.extents],
                        bed=bf[0] if bf else None, fits_180=bool(bf), print_pose=PRINTED[name][2], watertight=bool(t.is_watertight), bodies=len(m.decompose()))
    one("rail", H["rail"][0]); one("unit_body", L.unit_body()[0]); one("blade", L.blade(1, L.faces()["grip"]))
    one("wheels", L.wheel(0, -L.Z0)); one("pilot", L.pilot(1, 1)); one("paddle", L.unit_bought("open")["paddle"]); one("tie", L.tie(L.FLOAT_Z[0])); one("pads", L.unit_body()[1][0])
    for k in ("stub", "post", "detent_arm"): one(k, FX[k][0])
    one("saddle_overseer", saddle(S, "overseer")); one("saddle_scout", saddle(S, "scout"))
    one("link", link_bar(S))
    res["printed"] = pr
    res["generated"] = {q["part"][9:]: dict(part=q["part"], setup=f"parts/{q['part']}/part.py", state=(f"run {'PASS' if q['solid_check'] == 'PASS' and q['worst'] <= 0.5 else 'FAIL'}: worst {q['worst']}, {q['mass_g']} g" if q["current"] else "setup changed 14 Nov: rerunning")) for q in print_kit(pr)["generated"]}
    res["bought"] = BOUGHT
    res["print_kit"] = print_kit(pr)
    res["compat"] = compat(S, fast)
    print("compat states", res["compat"]["states"], "overlaps", len(res["compat"]["overlaps"]))
    print("ring vs box", res["compat"]["ring_vs_box"]); print("arm vs stand", res["compat"]["arm_vs_stand"]); print("head vs funnel lane", res["compat"]["head_vs_funnel_at_the_drone"])
    for h in res["compat"]["overlaps"][:40]: print("  ", h)
    write_assembly(res, S, H, FX, out_dir)
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)


def bay_mount(S):
    """what a bay saddle needs from a receiver: for the latch worker. A bay's receiver is the drone's receiver without landing feet, hung by its own mount bolts."""
    out = {}
    for pk in ("overseer", "scout"):
        mp = S[pk]["mount"]; m = dict(overseer=0.766, scout=0.222)[pk]
        x0, x1 = min(p[0] for p in mp) - 4.5, max(p[0] for p in mp) + 4.5; y0, y1 = min(p[1] for p in mp) - 5.0 - L.Y_C, max(p[1] for p in mp) + 5.0 - L.Y_C
        w = 9.81 * m; a_ring = L.AXLE_STALL_NM / 0.0075                      # rad/s2: the axle's stall torque on the turning arm's inertia (v7: 0.0074 kg m2, E); the ring itself is fixed, this is what a head carrying a pack can put on a bay it is seated in
        out[pk] = dict(contact_outline=dict(frame="mount plane; x across, y along the pack from the tab line", outer=[round(x0, 1), round(x1, 1), round(y0, 1), round(y1, 1)],
                                            open_inside=[round(x0 + 10.5, 1), round(x1 - 10.5, 1), round(y0 + 11.0, 1), round(y1 - 11.0, 1)], thickness=4.0, note="a 10.5 / 11 mm band round the bolt pattern; open over the pack"),
                       cases=[dict(case="pack weight, bay at any ring angle (11 bays, 30 deg apart)", cls="sustained", N=round(w, 1), direction="any direction in the ring's plane, at the pack's CG"),
                              dict(case="insert by the head at the float limit", cls="misuse", N=L.F_LIMIT, direction="into the receiver (radially outward), at the tab line + 3 N m pitch"),
                              dict(case="insert, plain", cls="flight", N=round(57.0 + w, 1), direction="into the receiver"),
                              dict(case="pull-out by the head", cls="flight", N=round(57.0 + w, 1), direction="out of the receiver, the plug's 57 N at the lead end"),
                              dict(case="axle stall with a head seated in this bay (the detent's stop is the same path, smaller: 1.07 N m in v7)", cls="misuse", N=round(1000 * L.AXLE_STALL_NM / (L.R_BAY - L.CART_DROP), 1), direction="across (tangential), at the cartridge's mouth"),
                              dict(case="box handling 3 g down + 2 g side with a pack seated", cls="misuse", N=round(3.6 * w, 1), direction="worst in the ring's plane")],
                       face="the receiver's mount plane (its top, %.1f above its bottom), flat over a 12 mm band inside the bolt pattern" % max(p[2] for p in mp),
                       bolts=[[round(p[0], 2), round(p[1] - L.Y_C, 2)] for p in mp], bolts_frame="x across, along the pack from the tab line; the receiver's own r['MOUNT']",
                       bolt="the same screw and insert the drone uses (M3 assumed, o 3.4 clearance in the saddle)", no_feet=True,
                       loads_N=dict(insert_push=L.F_LIMIT, pull_out=round(57.0 + 9.81 * m, 1), across_at_the_mouth=43.5, pack_weight_any_direction=round(9.81 * m, 1)),
                       per_bolt_N=round(L.F_LIMIT / len(mp) * 1.5, 1), note="the push limit is 118 N (v7: 74): the receiver takes it on its mount bolts in a bay exactly as on the drone")
    out["asks"] = ["a receiver variant without landing feet for the 7 Scout bays (neighbouring Scout bays at 30 deg clear by under 1 mm3 at R 129 only without them)",
                   "the mount bolts reachable from above with the saddle's 4 mm plate on (head counterbores open upward)", "nothing of the receiver below its bottom plane (the head's pilots and blades pass 0.3 from it)"]
    return out


def link_bar(S):
    """a straight bar between two neighbouring saddles' ears (Overseer and Scout ears are at different radii and stations: one bar length per joint kind is the
    ring's own detail; this is the Overseer-to-Overseer chord at the top ears, the longest)"""
    r = L.R_BAY + max(p[2] for p in S["scout"]["mount"]) + 2.0; c = 2 * r * math.sin(math.radians(15.0))
    return U(box(-c / 2 - 7, c / 2 + 7, 0, 4.0, -7, 7)) - cyl("y", -c / 2, 0, 1.7, -1, 5) - cyl("y", c / 2, 0, 1.7, -1, 5)


A = "https://www.amazon.com/"
BOUGHT = [
    dict(part="head stepper", spec="NEMA 17 StepperOnline 17HS19-2004S1, 59 N cm, 2 A (dimensions M, on hand)", n=2, url=A + "STEPPERONLINE-Stepper-Bipolar-Connector-compatible/dp/B00PNEQKC0"),
    dict(part="axle stepper", spec="the same 17HS19 (on hand), now on the stub", n=1, url=A + "STEPPERONLINE-Stepper-Bipolar-Connector-compatible/dp/B00PNEQKC0"),
    dict(part="grab servo", spec="DS3218 20 kg cm (on hand); MG995 9.2 (ordered) also clears the cam", n=2, url=A + "DS3218-Digital-Servo-Torque-Waterproof/dp/B07TKTQ2NZ", check="take the 180 deg version"),
    dict(part="lead screw + nut", spec="Tr8 x 2 (2 mm lead, one start), 200 long, cut to 160, with its brass flange nut: the catch's screw", n=2, url=A + "YEJMKJ-200mm-Tr8x2-Screw-Brass/dp/B0GZZCF2XN", check="2 in the pack"),
    dict(part="coupler", spec="rigid 5 to 8 mm, L25 x D14", n=2, url=A + "uxcell-Coupling-L25xD14-Coupler-Connector/dp/B07P6X8X7Y"),
    dict(part="thrust bearing", spec="F8-16M, one each side of each screw's thrust wall", n=4, url=A + "Metal-Thrust-Ball-Roller-Bearing/dp/B0177ADHDW"),
    dict(part="screw collar", spec="8 mm lock ring, 13 x 5, outside the outer bearing (the coupler bears on the inner one)", n=2, url=A + "clp/B0D17CLMJM"),
    dict(part="lift rail", spec="MGN12 rail 100 long (uncut) with one MGN12H block: the block's four M3 hold the yoke, the rail's four M3 go into the carriage's spine", n=2,
         url=A + "Iverntech-Linear-MGN12H-Carriage-Printer/dp/B086SNDDXH", check="block 27 x 45.4, 13 high on the rail; holes 20 x 20; rail holes 25 apart from 12.5 in (M: measure, the setup's seats follow)"),
    dict(part="carbon rod 6", spec="o 6 x 500 solid pultruded carbon: three 115 long per head (the booms)", n=2, url=A + "clp/B0DK3LWS3V", check="take the round 6 x 500; 6.0 +- 0.1 in the 6.2 sockets"),
    dict(part="rail and yoke bolts", spec="M3 x 12 socket head, stainless: rail to spine (8), yoke to block (8)", n=16, url=A + "Socket-Screws-Available-Stainless-Machine/dp/B08P6DYH7V", check="100 in the pack"),
    dict(part="wheel bearing", spec="625ZZ, two per V wheel", n=12, url=A + "Double-Bearings-5x16x5mm-0-2x0-63x0-2-BlueStars/dp/B09XT9TKY8"),
    dict(part="detent roller", spec="623ZZ", n=1, url=A + "uxcell-Groove-Bearings-Metric-Shielded/dp/B085RSRF2F"),
    dict(part="hub bearing", spec="6010-2RS 50 x 80 x 16", n=2, url=A + "uxcell-6010-2RS-Pre-Lubricated-Industrial-Machinery/dp/B0G2QPXVK9"),
    dict(part="axle pulley", spec="GT2 16T, 5 bore, 6 mm belt", n=1, url=A + "Drive-Pulleys-Timing-Pulley-Alumium/dp/B014IGOQK6"),
    dict(part="axle belt", spec="GT2 6 mm open belt (on hand), a 336 mm loop lap-spliced as v7's; the splice never reaches the 16T in 360 deg of arm travel", n=1, url=A + "SIMAX3D-GT2-Timing-Belt/dp/B09C5ZKXRT"),
    dict(part="hall sensor", spec="SS49E, TO-92, one per bolt", n=4, url="https://amazon.com/Linear-Sensors-Element-Electric-Vehicle/dp/B08LD14S8P", check="fallback only a distributor sells: DRV5055A4"),
    dict(part="float bolt", spec="M4 shoulder screw, o 5 x 16 shoulder", n=8, url=A + "uxcell-Stainless-Shoulder-Length-Thread/dp/B077GYHYSB"),
    dict(part="float spring", spec="o 8 x 1.0 wire x 15, 5.8 N/mm (E from wire and coil count)", n=8, url=A + "clp/B0DYJYL4ZG", check="8 OD, 1 wire, 15 free, 10 in the pack. On a scale: 18 N at 11.9 long; the push limit follows this rate"),
    dict(part="blade spring", spec="o 6 x 0.7 wire x 25, 1.6 N/mm (E), two per blade", n=8, url=A + "uxcell-0-7mmx6mmx25mm-Stainless-Compression-Springs/dp/B076LRJH99", check="on a scale: 12.5 N at 17.1 long"),
    dict(part="pilot spring", spec="the latch's spring (o 3.0, 0.3 wire, 30 free) cut to 15, two side by side under each pilot (14 Nov: the o 6 x 20 had no way into the pocket)", n=8, url=A + "PATIKIL-Compression-Stainless-Mechanical-Assortment/dp/B0DKCRCBV2", check="E 0.25 N/mm each: the pilot must come back up by itself"),
    dict(part="paddle screw", spec="M3 x 35 socket head, through the paddle into the servo's shaft", n=2, url=A + "Socket-Screws-Available-Stainless-Machine/dp/B08P6DYH7V", check="the M3 x 12 listing's other lengths (not confirmed on it). E: 7 into the shaft; washers under the head if it bottoms"),
    dict(part="servo ear screws", spec="M3 x 8 socket head, two per servo into the unit body's tab (o 2.5, they cut their thread)", n=4, url=A + "Socket-Screws-Available-Stainless-Machine/dp/B08P6DYH7V"),
    dict(part="carriage return spring", spec="extension spring, 5 to 10 N over 60 mm, from the kit", n=2, url=A + "Extension-Different-Assortment-Stainless-Individual/dp/B0CFRF42LQ", check="rates not listed: pick on a scale"),
    dict(part="float washer", spec="PTFE washer M5 between each float bolt's head and its tie, with a steel M5 x 15 washer", n=8, url=A + "PATIKIL-Washer-Sealing-Gasket-Flanges/dp/B0H1XBGWPM"),
    dict(part="insert M5", spec="heat-set, rail bolts", n=4, url=A + "Threaded-Inserts-M5xD7xL6-Printing-Components/dp/B0H142Y3BM"),
    dict(part="insert M3 / M4", spec="heat-set: stepper faces M3 (16), MGN rail M3 (8); float bolts M4 (8, the next line)", n=24, url=A + "initeq-M3-0-5-Threaded-Inserts-Printing/dp/B077CHDSTR"),
    dict(part="insert M4", spec="heat-set M4-0.7, the float bolts screw into the ties", n=8, url=A + "clp/B07457F9ST", check="20 in the pack; the ties' holes are o 5.6: measure the insert first (M)"),
    dict(part="wheel axle", spec="M5 x 20 countersunk DIN 7991", n=6, url=A + "Socket-Screws-Countersunk-Stainless-Bright/dp/B0CJV8XJ3Y"),
    dict(part="carbon rod 3", spec="o 3 x 500: blade rods 66 long, four; pilot pins 9 long, four; on hand", n=1, url=A + "Carbon-Fiber-Length-Diameter-Suitable/dp/B09N91J1FC"),
]

COL = dict(hub="#c9d3dc", rail="#9fb4cc", carriage="#e8b04a", lift="#5f9ed1", unit="#7fc97f", bought="#8a8f96", tpu="#333333", fixed="#b7a3c9", bay="#8a94a6", pack="#2f4fbf", box="#d8d8d8")


def write_assembly(res, S, H, FX, out_dir):
    path = os.path.join(ROOT, "systems", "swap_arm.assembly.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    parts = []
    def add(pid, name, m, group, color, role, material, kin=None, opacity=None, folder=None, version="arm v8 (13 Nov)", pose=None):
        if folder: mesh = f"parts/{folder}/part.stl"
        else:
            mesh = f"parts/box_arm8/view/{pid}.stl"; tm(m).export(os.path.join(ROOT, mesh))
        parts.append(dict(id=pid, name=name, role=role, group=group, color=color, opacity=opacity, mesh=mesh, pose=pose, material=material, tag="C", owner="swap arm",
                          version=version, source="parts/box_arm8/make.py", kin=kin))
    hub = U(*hub_drawn()) - U(*hub_cuts())
    ran = {k: current(k) for k in ("hub", "carriage", "lift")}                                  # the run's shape where the result is of the setup on main, else the setup
    if ran["hub"]: add("box_arm8_hub_run", "hub (generated)", None, "hub (turns)", COL["hub"], "structure", "PCTG", dict(stage="hub"), folder="box_arm8_hub", version="run, arm v8")
    else: add("box_arm8_hub_setup", "hub (drawn attachments; the body is generated)", hub, "hub (turns)", COL["hub"], "structure", "PCTG", dict(stage="hub"), version="setup, not run")
    for k, (m, mat, kind) in FX.items():
        add("box_arm8_fx_" + k, k.replace("_", " "), m, "stub side (fixed)", COL["fixed"] if kind == "print" else COL["bought"], "structure" if kind == "print" else "bought", mat,
            folder=("box_arm8_" + k) if kind == "print" else None)
    names = dict(carriage="carriage (setup envelope; generated)", lift="lift: yoke on the MGN block, arm to the nut (setup envelope; generated)", mgn_rail="lift rail MGN12 x 100", mgn_block="MGN12H block", booms="booms: three o 6 carbon rods", ties="ties (printed, on the booms)")
    for hd, f in (("A", lambda q: q), ("B", rot180)):
        for k, (m, mat, kind, grp) in H.items():
            stage = "hub" if grp == "hub" else ("lift" if grp == "unit" else grp)
            col = COL["bought"] if kind == "bought" else COL["tpu"] if mat == "TPU" else COL[grp if grp != "hub" else "rail"]
            kin = dict(stage=stage, head=hd, blade=(1 if k == "blade_p" else -1 if k == "blade_n" else 0) * (1 if hd == "A" else -1))
            if kind == "run" and ran[k]:
                add(f"box_arm8_{hd.lower()}_{k}_run", f"head {hd}: {k} (generated)", None, f"head {hd}", col, "structure", mat, kin, folder="box_arm8_" + k, version="run, arm v8",
                    pose=None if hd == "A" else [[-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]])
                continue
            add(f"box_arm8_{hd.lower()}_{k}", f"head {hd}: {names.get(k, k.replace('_', ' '))}", f(m), f"head {hd}", col, "bought" if kind == "bought" else "structure", mat,
                kin, version="setup, not run" if kind == "run" else "arm v8 (14 Nov)")
    for kind, angs in BAYS.items():
        for a_ in angs:
            add(f"box_arm8_bay_{a_}_recv".replace("-", "m"), f"bay {a_}: {kind} receiver (latch v7)", at_radius(S[kind]["recv_bay"], L.R_BAY, a_), "bays", COL["bay"], "protected", "PCTG", opacity=0.55)
            add(f"box_arm8_bay_{a_}_saddle".replace("-", "m"), f"bay {a_}: saddle", at_radius(saddle(S, kind), L.R_BAY, a_), "bays", "#6f7c91", "structure", "PCTG")
            add(f"box_arm8_bay_{a_}_cart".replace("-", "m"), f"bay {a_}: {kind} cartridge", at_radius(S[kind]["cart"], L.R_BAY, a_), "packs", COL["pack"], "protected", None, dict(bay=a_))
    add("box_arm8_drone_recv", "drone seat: Overseer receiver", at_radius(S["overseer"]["recv"], L.R_DRONE), "drone seat", COL["bay"], "protected", "PCTG", opacity=0.45)
    add("box_arm8_drone_cart", "drone seat: Overseer cartridge (the dead pack)", at_radius(S["overseer"]["cart"], L.R_DRONE), "packs", "#bf4f2f", "protected", None, dict(pack="dead"))
    fm = L.latch()._funnel()
    add("box_arm8_funnel", "funnel (Overseer walls), from parts/box_funnel", fm.funnel("overseer").transform(SEAT_TO_ARM()[:3]), "box (context)", COL["box"], "protected", "PCTG", opacity=0.3)
    c = res["compat"]; nb = res["numbers"]
    asm = {"name": "swap arm", "title": "Swap Arm v8", "readme_view": "iso", "artifact": old.get("artifact", ARTIFACT), "machine": "box", "order": 80,
           "views": {"iso": {"eye": [330, -260, 250], "target": [0, 120, 30]}, "front": {"eye": [0, 620, 40], "target": [0, 120, 40]}, "side": {"eye": [560, 150, 60], "target": [0, 150, 60]},
                     "head A": {"eye": [150, 330, 150], "target": [10, 150, 60]}, "hub": {"eye": [160, -160, 120], "target": [0, 50, 0]}},
           "note": f"The box's swap arm, v8, for latch v7 and both packs. Two heads 180 apart; each is a lift on a bought MGN12H rail driven by a T8 x 2 screw, with three carbon rods out under the grab unit, picking its carriage up for the last "
                   f"{L.T2:g} mm: {nb['travel_mm']['drone']} mm to the drone seat. Push limited by the grab unit's float springs to {L.F_LIMIT:g} N (x{L.F_LIMIT / L.F_INSERT['overseer']:.2f} over the Overseer insert); the screw holds with the power off. "
                   f"Bays at R {L.R_BAY:g}. Swap cycle, both packs, {c['states']} states: {'0 overlaps' if c['zero'] else str(len(c['overlaps'])) + ' overlapping pairs, listed under the viewer'}. "
                   + ("Hub, carriage and lift are their runs' shapes." if all(ran.values()) else "Shown as their setups (drawn attachments and envelope), their runs pending: " + ", ".join(k for k in ran if not ran[k]) + "."),
           "cg_ref": None, "timeline": "tools/swap_arm_motion.py:timeline", "appendix": "systems/swap_arm.appendix.html", "parts": parts}
    json.dump(asm, open(path, "w"), indent=1)
    open(os.path.join(ROOT, "systems", "swap_arm.appendix.html"), "w").write(appendix(res))


def appendix(res):
    nb, c, pr = res["numbers"], res["compat"], res["printed"]
    td = lambda *v: "<tr>" + "".join(f"<td>{x}</td>" for x in v) + "</tr>"
    rows = [td("reach", f"park {nb['travel_mm']['park_pad_plane']}, bay +{nb['travel_mm']['bay']}, drone +{nb['travel_mm']['drone']} (+{nb['travel_mm']['over']} over); lift {nb['travel_mm']['lift_on_carriage']} on the carriage, carriage {nb['travel_mm']['carriage']}; top wheel rim r {nb['travel_mm']['top_wheel_rim_r']} (lane 159)"),
            td("push", f"insert {nb['push']['insert_N']['overseer']} N (Overseer); asked x1.5 = {nb['push']['asked_N']}; float springs {nb['push']['float_preload_N']} N preload, {nb['push']['at_over_travel_N']} at the commanded over-travel, limit {nb['push']['limit_N']} ({nb['push']['margin_over_insert']}); screw stall {nb['push']['screw_stall_N']} N (v7 pinion 74)"),
            td("power off", nb["power_off"]["holds"] + "; " + nb["power_off"]["note"]),
            td("time", f"full stroke {nb['time_s']['full_stroke']} s, bay stroke {nb['time_s']['bay_stroke']} s at {nb['time_s']['at']}; {nb['time_s']['note']}"),
            td("locating", f"{nb['locating']['pilots']}: take {nb['locating']['takes_across_mm']} across and along (arm worst {nb['locating']['arm_error_mm']['across_worst']} / {nb['locating']['arm_error_mm']['along']}; the blades alone 0.9 / 1.8); {nb['locating']['order']}"),
            td("give", f"roll starts at {nb['give']['roll_Nm_to_start']['carrying']} N m carrying, {nb['give']['roll_Nm_to_start']['inserting_overseer']} inserting; pitch {nb['give']['pitch_Nm_to_start']['carrying']} / {nb['give']['pitch_Nm_to_start']['inserting_overseer']}; reach {nb['give']['reach_deg']['roll']} deg roll, {nb['give']['reach_deg']['pitch']} pitch (the latch takes 0.6 / 1.6 rigid)"),
            td("blades", f"{nb['blades']['spring_N']['at_grip']} N at the grip (asked {nb['blades']['latch_asks_N']}), {nb['blades']['spring_N']['open']} open; cam {nb['blades']['cam_kgcm']} kg cm: {nb['blades']['servo_margin']}"),
            td("sensing", nb["sensing"]), td("pads", nb["pads"]),
            td("ring", "; ".join(f"{k}: {v} mm&sup3;" for k, v in res["ring"].items())),
            td("swap cycle", f"{c['states']} states, both packs, against the latch sets, the funnel at each wall setting, every bay, both drones' solid parts, the stand and sheets: " + ("0 overlaps" if c["zero"] else "<b>" + "; ".join(f"{h['pair']} {h['mm3']} mm&sup3; ({h['worst_at']})" for h in c["overlaps"][:12]) + "</b>"))]
    prn = "".join(td(v["part"], v["material"], v["count"], v["g_each"], " x ".join(str(x) for x in v["box_mm"]), v["print_pose"], "yes" if v["fits_180"] else "<b>no</b>") for v in pr.values())
    bought = "".join(td(b["part"], b["spec"], b["n"], f"<a href='{b['url']}'>listing</a>", b.get("check", "")) for b in res["bought"])
    ld = "".join(td(r["case"], r["cls"], r["F_N"], r["mgn_index"], r["nut_N"]) for r in res["loads"]["cases"] if "overseer" in r["case"])
    kit = res.get("print_kit") or {}
    ts = kit.get("test_set") or {}
    kit_html = (f"<h3>Print kit</h3><p><b>{kit.get('hold', '')}</b> Files: the release's print-&lt;name&gt;.stl, each in its print pose (parts/box_arm8/kit.json; pictures in parts/box_arm8/kit/).</p>"
                "<table><tr><th>file</th><th>set</th><th>material</th><th>n</th><th>on the bed mm</th><th>first layer mm&sup2;</th><th>supports</th><th>brim</th><th>g each</th><th>solid check</th></tr>"
                + "".join(td(f"<a href='{q['release']}'>{q['file'].split('/')[1]}</a>", q["set"], q["material"], q["count"], " x ".join(str(v) for v in q["size_on_bed_mm"]), q["first_layer_mm2"], q["supports"], q["brim"], q["grams_each"], q["solid_check"]) for q in kit.get("files", [])) + "</table>"
                + "<h3>Print order</h3><ol>" + "".join(f"<li>{o['step'][2:]}" + (f" <i>(waiting on its run: {', '.join(o['waiting'])})</i>" if o["waiting"] else "") + "</li>" for o in kit.get("print_order", [])) + "</ol>"
                + "<h3>Grab-unit test set (one head)</h3><ol>" + "".join(f"<li>{o}</li>" for o in ts.get("steps", [])) + "</ol><table><tr><th>bought for it</th><th>spec</th><th>n</th><th></th><th>check</th></tr>"
                + "".join(td(b_["part"], b_["spec"], b_["n"], f"<a href='{b_['url']}'>listing</a>", b_.get("check", "")) for b_ in ts.get("bought", [])) + "</table>"
                + "<h3>Open, measured on arrival</h3><table><tr><th>item</th><th>state</th><th>measure</th><th>if it is off</th></tr>" + "".join(td(o["item"], o["state"], o["what"], o["if_off"]) for o in kit.get("open", [])) + "</table>"
                + "<h3>Assembly order</h3><ol>" + "".join(f"<li>{o}</li>" for o in kit.get("order", [])) + "</ol>")
    return (f"<div class='arm8'><style>.arm8 table{{border-collapse:collapse;font-size:13px;margin:8px 0}}.arm8 td,.arm8 th{{border:1px solid #8884;padding:3px 7px;vertical-align:top}}</style>"
            f"<h3>What the head does</h3><table>{''.join(rows)}</table>"
            f"<h3>Printed pieces (drawn)</h3><table><tr><th>part</th><th>material</th><th>n</th><th>g each</th><th>box mm</th><th>print pose</th><th>fits 180</th></tr>{prn}</table>"
            f"<p>Generated: " + "; ".join(f"{k}: {v['state']}" for k, v in res["generated"].items()) + ".</p>"
            f"<h3>Load cases at the grab unit (each its own)</h3><table><tr><th>case</th><th>class</th><th>F N</th><th>MGN12H block index (static)</th><th>nut N</th></tr>{ld}</table>"
            f"<p>Not modelled: {'; '.join(res['loads']['missing'])}.</p>"
            f"{kit_html}<h3>Bought</h3><table><tr><th>part</th><th>spec</th><th>n</th><th></th><th>check</th></tr>{bought}</table></div>")


if __name__ == "__main__":
    main()
