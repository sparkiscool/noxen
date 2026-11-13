"""Swap arm v8 (13 Nov 2026, Spark: "build the arm as well make sure things all are compatable and get it ready to print").
The box's arm for latch v7 and both packs, drawn in this repo (arm v7 was meshes only, for latch v4, never built).

What changed from v7 and why (C unless tagged; numbers in result.json):
  1. Reach. The seat is 65 higher than v7 reached (W Z 220.3, 3 Nov) and v7's one-stage head cannot get there: a longer rail hits the
     stand bars (r 165) and the side sheets (206), a longer carriage crosses the hub centre at park. Each head is two stages on one
     T8 x 2 lead screw: the lift (block on two o 6 steel rods, four LM6UU) is driven; it picks the carriage (v7's V wheels on v7's rail)
     up for the last 58. Grab travel 117.8, carriage 59.8. A return spring keeps the carriage down on the lift's stop.
  2. Push. v7's pinion stalled at 74 N; the insert is 64.5 N (Overseer): x1.15. The screw gives over 380 N at stall (E), limited by the
     grab unit's float springs to 115 N: x1.78. A 2 mm lead holds the pack with the power off: v7's head brake relay and its bench
     drop gate go.
  3. Heads 180 apart (v7: 90). Each head's block, rods and screw pass beside the hub centre on their own side; at 90 they cross.
     The arm is balanced; a swap turns 180 between pull and insert.
  4. Axle motor and detent on the stub (fixed), a printed GT2 ring on the hub's sleeve: nothing of the axle drive turns with the arm.
  5. Ring R 127 (v7: 113): latch v7's receivers overlap their neighbours by 623 mm3 at 113 and 30 deg (ring()).
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
BAYS = {"overseer": (0, -60, -120, -180), "scout": (30, -30, -90, -150, -210)}       # 9 bays, the drone at 90. box.assembly.json's two more (Overseer at 60 / 120, 8 Nov) do not fit at R 127: ring()


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
    stub = stub + cyl("y", *piv, 5.0, 6.0, 20.0) - cyl("y", *piv, 1.5, -1, 21)
    post = U(cyl("y", 0, 0, 28.5, 40.0, 44.0), cyl("y", 0, 0, 9.0, 44.0, 62.0)) - cyl("y", 0, 0, 4.25, 39, 63)
    rc = (RING_R + 1.5 + 5.0) / math.sqrt(2)                                                                                   # roller centre on the notch ring, at 45 deg
    arm = U(hullp([[x, y, z] for y in (19.0, 27.0) for x, z in ((piv[0] - 5, piv[1] - 5), (piv[0] + 5, piv[1] + 5), (rc + 6, rc + 9), (rc - 2, rc + 9))]),
            cyl("y", *piv, 6.0, 19.0, 27.0), box(rc - 2.5, rc + 6.5, 19.0, 27.0, rc + 5.8, rc + 12.0)) - cyl("y", *piv, 1.6, 18, 28) - box(rc - 8, rc + 8, 20.9, 25.1, rc - 8, rc + 5.7) - cyl("y", rc, rc, 1.5, 18, 28)
    return dict(stub=(stub, "PCTG", "print"), post=(post, "PCTG", "print"), detent_arm=(arm, "PCTG", "print"),
                detent_roller=(cyl("y", rc, rc, 5.0, 21.0, 25.0) - cyl("y", rc, rc, 1.5, 20, 26), None, "bought"),
                axle_motor=(U(box(AXLE_M[0] - 21.15, AXLE_M[0] + 21.15, -48.0, 0.0, AXLE_M[1] - 21.15, AXLE_M[1] + 21.15), cyl("y", *AXLE_M, 2.5, 0.0, 24.0)), None, "bought"),
                axle_pulley=(cyl("y", *AXLE_M, 5.1, 9.0, 19.0) + cyl("y", *AXLE_M, 6.5, 19.0, 24.0), None, "bought"),
                bearings=(cyl("y", 0, 0, 39.98, 8.0, 40.0) - cyl("y", 0, 0, 25.0, 7, 41), None, "bought"))


# ------------------------------------------------------------------ hub (turns): the setup's drawn bodies, as the placeholder until its run
def hub_drawn():
    """what parts/box_arm8_hub/part.py draws (attachments only); the generated body joins them"""
    y0, y1, R = L.PLATE
    sleeve = cyl("y", 0, 0, RING_R + 1.5, 8.0, 45.0, 96)                          # the bearings go in from the stub end; a 3 mm shoulder over the outer race at the plate end
    ring = cyl("y", 0, 0, RING_R + 1.5, 9.5, 25.5, 96)   # flush with the sleeve: no step                    # one band: GT2 teeth (150T) over y 9.5..19.5 and the detent's twelve notches over 20.5..25.5 are cut in it (not modelled)
    notch = ring
    plate = cyl("y", 0, 0, R, y0, y1, 96) - cyl("y", 0, 0, 15.0, y0 - 1, y1 + 1)
    bodies = [sleeve, ring, notch, plate]
    for h in (0, 1):
        f = (lambda m: m) if h == 0 else rot180
        w = L.MOTOR_WALL
        bodies += [f(box(w["x"][0], w["x"][1], w["y"][0], w["y"][1], w["u"][0], w["u"][1]))]
        bodies += [f(cyl("y", 0, u, 7.0, 44.0, y0 + 0.01)) for u in L.RAIL_BOLTS]
    return bodies
def hub_cuts():
    y0, y1, R = L.PLATE
    c = [cyl("y", 0, 0, 40.0, 6.0, 40.0, 96), cyl("y", 0, 0, 30.0, 39.9, y0 + 0.01), cyl("y", 0, 0, 15.0, y0, y1 + 1),
         cyl("y", 0, 0, 78.0, 6.0, 28.0, 96) - cyl("y", 0, 0, RING_R + 2.5, 5, 29, 96)]            # the fixed motor's pulley, the belt and the detent roller pass under the plate
    x, y = L.SCREW
    for h in (0, 1):
        f = (lambda m: m) if h == 0 else rot180
        m = L.MOTOR
        c += [f(box(m["x"][0] - 0.3, m["x"][1] + 0.3, m["y"][0], m["y"][1] + 5, m["u"][0] - 1, m["u"][1])),              # the stepper, 5 into the plate
              f(cyl("z", x, y, 11.3, -48, -42)),
              *[f(cyl("z", x + sx * 15.5, y + sy * 15.5, 1.7, -48, -42)) for sx in (-1, 1) for sy in (-1, 1)],
              *[f(cyl("y", 0, u, 3.2, 43, 60)) for u in L.RAIL_BOLTS],                                                      # M5 heat-set inserts
              f(box(-12.5, 12.5, y1, 80, L.RAIL[0] - 1, 160)), f(box(10.0, 34.5, y1 + 0.5, 70, L.RAIL[0], 160)), f(box(-34.5, -10.0, y1 + 0.5, 70, L.RAIL[0], 160))]   # rail seat, wheel lanes
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
    P["rods"] = (L.rods(), None, "bought", "carriage")
    P["lift"] = (L.lift_env() - L.lift_cuts(), "PCTG", "run", "lift")
    for k, m in L.lift_bought().items(): P[k] = (m, None, "bought", "lift")
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
def sets():
    """per pack, in the arm frame with the receiver's bottom at radius 0 on the grip line: receiver, cartridge (bolts out), cartridge gripped"""
    m7 = L.latch(); m6, bl = m7.m6, m7.bl; out = {}
    for pk, B in L.packs().items():
        g = B["g"]; Ts = m7.to_seat(B)
        M = np.array([[0, -1.0, 0, 0], [1.0, 0, 0, L.Y_C], [0, 0, 1.0, 44.3]]) @ np.vstack([Ts, [0, 0, 0, 1]])
        out[pk] = dict(recv=m6.receiver_all(B).transform(M), cart=(m6.cart_all(B) + m6.moving(g, 0, 1) + m6.moving(g, 0, -1) + bl.pack_body(g)).transform(M),
                       held=(m6.cart_all(B) + m6.moving(g, m6.R0, 1) + m6.moving(g, m6.R0, -1) + bl.pack_body(g)).transform(M), top=float(m6.receiver_all(B).transform(M).bounding_box()[5]))
    return out


def at_radius(m, r, ang=90.0):
    return m.translate([0, 0, r]).transform(T34(L.Ry(ang)))


def saddle(S, pk):
    """how a bay receiver hangs: a cap over the receiver's top 14 mm with an ear at each end; link bars join neighbouring ears into the ring"""
    bb = S[pk]["recv"].bounding_box(); t = 3.0
    cap = box(bb[0] - t, bb[3] + t, bb[1] - t, bb[4] + t, bb[5] - 14.0, bb[5] + t) - box(bb[0] - 0.25, bb[3] + 0.25, bb[1] - 0.25, bb[4] + 0.25, bb[5] - 15.0, bb[5] + 0.25)
    cap = cap - box(bb[0] + 8, bb[3] - 8, bb[1] + 8, bb[4] - 8, bb[5] - 1, bb[5] + t + 1)                 # open over the pack
    ears = [box(-9.0, 9.0, y0, y1, bb[5] - 14.0, bb[5] + t) - cyl("y", 0.0, bb[5] - 6.0, 2.75, y0 - 1, y1 + 1) for y0, y1 in ((bb[1] - t - 5.0, bb[1] - t + 0.01), (bb[4] + t - 0.01, bb[4] + t + 5.0))]
    return U(cap, *ears)


def ring(S, R=None):
    """neighbouring receivers (with seated cartridges) at 30 deg: overlap mm3 at v7's 113 and at R_BAY, and the closest saddles"""
    out = {}
    for r in (113.0, L.R_BAY):
        a = at_radius(S["overseer"]["recv"] + S["overseer"]["cart"], r, 90); b = at_radius(S["scout"]["recv"] + S["scout"]["cart"], r, 120)
        out[f"R {r:g}: receiver sets, Overseer beside Scout"] = round(vol(a ^ b), 1)
    a = at_radius(saddle(S, "overseer"), L.R_BAY, 90); b = at_radius(saddle(S, "scout"), L.R_BAY, 120)
    out["saddles Overseer beside Scout"] = round(vol(a ^ (b + at_radius(S["scout"]["recv"], L.R_BAY, 120))), 1)
    for pk in ("overseer", "scout"):
        d = at_radius(S[pk]["recv"] + S[pk]["cart"], L.R_DRONE, 90)
        out[f"drone ({pk}) vs the bays at 30 and 150"] = round(sum(vol(d ^ at_radius(S["scout"]["recv"] + saddle(S, "scout"), L.R_BAY, a_)) for a_ in (30, 150)), 1)
    fz = dict(box_statics("overseer")[0])["funnel (overseer walls)"]
    out["an Overseer bay at 60 vs the funnel"] = round(vol(at_radius(S["overseer"]["recv"], L.R_BAY, 60) ^ fz), 1)
    out["a Scout bay at 60 vs the funnel"] = round(vol(at_radius(S["scout"]["recv"], L.R_BAY, 60) ^ fz), 1)
    out["a Scout bay at 60 vs the Scout bay at 30 (their landing feet)"] = round(vol(at_radius(S["scout"]["recv"], L.R_BAY, 60) ^ at_radius(S["scout"]["recv"], L.R_BAY, 30)), 1)
    return out


def box_statics(setting):
    """everything in the box that does not turn, as (name, manifold) in W: the funnel at a wall setting, the stand, the sheets and PSU, both
    drones' watertight parts. Not watertight (cage rods, wires): listed, checked by nobody here."""
    fm = L.latch()._funnel(); F = fm.funnel(setting)
    Ts = np.array([[0, -1.0, 0, 0], [1.0, 0, 0, L.Y_C], [0, 0, 1.0, 220.3]])
    out = [(f"funnel ({setting} walls)", F.transform(Ts))]
    a = json.load(open(os.path.join(ROOT, "systems", "box.assembly.json"))); skipped = []
    for p in a["parts"]:
        pid = p["id"]
        if pid.startswith(("box_arm", "box_funnel", "box_dome", "box_ctx")): continue
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
                bays.append((f"bay {a_} ({kind}) receiver", at_radius(S[kind]["recv"], L.R_BAY, a_), a_, kind))
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
        for a_ in ([0, -60, -180] if pk == "overseer" else [30, -90, -210]):
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
    hits = [dict(pair=k, mm3=v, worst_at=lab) for k, (v, lab) in sorted(worst.items(), key=lambda kv: -kv[1][0]) if "box_stand" not in k]
    stand = [dict(pair=k, mm3=v, worst_at=lab) for k, (v, lab) in sorted(worst.items(), key=lambda kv: -kv[1][0]) if "box_stand" in k]
    return dict(states=n, overlaps=hits, zero=not hits, ring_vs_box=ringhits, arm_vs_stand=stand, not_watertight_not_checked=sorted(skipped),
                stand_note="the stand in systems/box.assembly.json is the built demo v1's (mast, keel, kb, toe, bars), drawn for v4's 8 bays at R 100; whatever it meets here is a box-structure change, not an arm one")


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
                       top_wheel_rim_r=round(L.Z0 + L.S_DRONE + L.S_OVER - L.T2 + L.WHEELS[1][1] + 12.0, 1)),
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
        pads="each of the eight pads (four per pack) on its own M3 x 8 set screw from below, +-0.75; the latch's height window is +-0.21")
    return out


def loads_report():
    """the head's cases and what they put on the lift's bearings and nut (the carriage and hub setups take the same numbers)"""
    out = []
    for pk in ("overseer", "scout"):
        for nm, cls, F, P, s in L.cases(pk):
            seats, nut = L.lift_reactions(F, P, s)
            out.append(dict(case=nm, cls=cls, F_N=[round(float(v), 1) for v in F], at=[round(float(v), 1) for v in P], s=s,
                            bearing_N=[round(float(np.linalg.norm(f_)), 1) for _, f_ in seats], nut_N=round(float(nut[1][2]), 1)))
    wb = max(max(r["bearing_N"]) for r in out if r["cls"] == "flight"); wm = max(max(r["bearing_N"]) for r in out)
    return dict(cases=out, lm6uu=dict(worst_swap_N=wb, worst_misuse_N=wm, static_rating_N=265.0, index_swap=round(wb / 265.0, 2), index_misuse=round(wm / 265.0, 2)),
                rule="the screw pushes over 30 N only in the last 8 mm of an insert, where the receiver's guides hold the cartridge's pitch; a jam at the plug end before that stops at 30 N (0.4 N m / deg x lever, inside the float)",
                missing=["a stall of the axle motor against a jammed head (needs the ring belt's tooth shear, E)", "a dropped pack onto a parked head (v7: 50 N, not rerun)",
                         "the rail and V wheels at the new lever: v7's frame model is in the old repo, not rerun here (hand: wheel pair 9.5 N m / 50 mm = 190 N on two 625ZZ, static 660 N each)"])


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
    "pads": (None, "TPU", "flat, 2.0 thick", 16),
    "stub": (None, "PCTG", "flange down", 1), "post": (None, "PCTG", "flange down", 1), "detent_arm": (None, "PCTG", "flat", 1),
    "saddle_overseer": (None, "PCTG", "top down", 4), "saddle_scout": (None, "PCTG", "top down", 5), "link": (None, "PCTG", "flat", 16)}


def main():
    fast = "--fast" in sys.argv
    out_dir = os.path.join(HERE, "view"); os.makedirs(out_dir, exist_ok=True)
    S = sets(); H = head_parts("grip"); FX = fixed_parts()
    res = dict(version="arm v8 (13 Nov 2026)", numbers=numbers(), loads=loads_report(), ring=ring(S))
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
    one("wheels", L.wheel(0, -L.Z0)); one("pilot", L.pilot(1, 1)); one("paddle", L.unit_bought("open")["paddle"]); one("pads", L.unit_body()[1][0])
    for k in ("stub", "post", "detent_arm"): one(k, FX[k][0])
    one("saddle_overseer", saddle(S, "overseer")); one("saddle_scout", saddle(S, "scout"))
    one("link", link_bar(S))
    res["printed"] = pr
    res["generated"] = {k: dict(part="box_arm8_" + k, setup=f"parts/box_arm8_{k}/part.py", state="setup ready, not run") for k in ("hub", "carriage", "lift")}
    res["bought"] = BOUGHT
    res["compat"] = compat(S, fast)
    print("compat states", res["compat"]["states"], "overlaps", len(res["compat"]["overlaps"]))
    print("ring vs box", res["compat"]["ring_vs_box"]); print("arm vs stand", res["compat"]["arm_vs_stand"])
    for h in res["compat"]["overlaps"][:40]: print("  ", h)
    write_assembly(res, S, H, FX, out_dir)
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=1)


def link_bar(S):
    """a straight bar between two neighbouring saddles' ears (Overseer and Scout ears are at different radii and stations: one bar length per joint kind is the
    ring's own detail; this is the Overseer-to-Overseer chord at the top ears, the longest)"""
    r = L.R_BAY + S["scout"]["top"] - 6.0; c = 2 * r * math.sin(math.radians(15.0))
    return U(box(-c / 2 - 7, c / 2 + 7, 0, 4.0, -7, 7)) - cyl("y", -c / 2, 0, 2.75, -1, 5) - cyl("y", c / 2, 0, 2.75, -1, 5)


A = "https://www.amazon.com/"
BOUGHT = [
    dict(part="head stepper", spec="NEMA 17 StepperOnline 17HS19-2004S1, 59 N cm, 2 A (dimensions M, on hand)", n=2, url=A + "STEPPERONLINE-Stepper-Bipolar-Connector-compatible/dp/B00PNEQKC0"),
    dict(part="axle stepper", spec="the same 17HS19 (on hand), now on the stub", n=1, url=A + "STEPPERONLINE-Stepper-Bipolar-Connector-compatible/dp/B00PNEQKC0"),
    dict(part="grab servo", spec="DS3218 20 kg cm (on hand); MG995 9.2 (ordered) also clears the cam", n=2, url=A + "DS3218-Digital-Servo-Torque-Waterproof/dp/B07TKTQ2NZ", check="take the 180 deg version"),
    dict(part="lead screw + nut", spec="Tr8 x 2 (2 mm lead, one start), 200 long, cut to 160, with its brass flange nut: the catch's screw", n=2, url=A + "YEJMKJ-200mm-Tr8x2-Screw-Brass/dp/B0GZZCF2XN", check="2 in the pack"),
    dict(part="coupler", spec="rigid 5 to 8 mm, L25 x D14", n=2, url=A + "uxcell-Coupling-L25xD14-Coupler-Connector/dp/B07P6X8X7Y"),
    dict(part="thrust bearing", spec="F8-16M, one each side of the thrust wall", n=4, url=A + "Metal-Thrust-Ball-Roller-Bearing/dp/B0177ADHDW"),
    dict(part="screw collar", spec="8 mm set screw collar, 14 x 7", n=4, url=A + "uxcell-Collars-Landing-Stopper-Collar/dp/B0CQ2S44S4"),
    dict(part="lift bearing", spec="LM6UU 6 x 12 x 19, six per head", n=12, url=A + "uxcell-LM6UU-Linear-Bearings-Length/dp/B07H95CWTX"),
    dict(part="lift rod", spec="o 6 x 200 hardened chrome shaft, cut to 120, three per head", n=6, url=A + "SUOFEILAIMU-Linear-L200mm-Chrome-Plated/dp/B09KGBCRKR"),
    dict(part="wheel bearing", spec="625ZZ, two per V wheel", n=12, url=A + "Double-Bearings-5x16x5mm-0-2x0-63x0-2-BlueStars/dp/B09XT9TKY8"),
    dict(part="detent roller", spec="623ZZ", n=1, url=A + "uxcell-Groove-Bearings-Metric-Shielded/dp/B085RSRF2F"),
    dict(part="hub bearing", spec="6010-2RS 50 x 80 x 16", n=2, url=A + "uxcell-6010-2RS-Pre-Lubricated-Industrial-Machinery/dp/B0G2QPXVK9"),
    dict(part="axle pulley", spec="GT2 16T, 5 bore, 6 mm belt", n=1, url=A + "Drive-Pulleys-Timing-Pulley-Alumium/dp/B014IGOQK6"),
    dict(part="axle belt", spec="GT2 6 mm open belt (on hand), a 336 mm loop lap-spliced as v7's; the splice never reaches the 16T in 360 deg of arm travel", n=1, url=A + "SIMAX3D-GT2-Timing-Belt/dp/B09C5ZKXRT"),
    dict(part="hall sensor", spec="SS49E, TO-92, one per bolt", n=4, url="https://amazon.com/Linear-Sensors-Element-Electric-Vehicle/dp/B08LD14S8P", check="fallback only a distributor sells: DRV5055A4"),
    dict(part="float bolt", spec="M4 shoulder screw, o 5 x 16 shoulder", n=8, url=A + "uxcell-Stainless-Shoulder-Length-Thread/dp/B077GYHYSB"),
    dict(part="float spring", spec="o 8 x 1.0 wire x 15, 5 active coils, 5.8 N/mm (E)", n=8, url=A + "uxcell-0-8mmx8mmx20mm-Stainless-Compression-Springs/dp/B076LS5N73",
         check="the listing found is this series' 0.8 x 8 x 20 (1.4 N/mm): the 1.0 x 8 x 15 is what the push needs and is not found yet"),
    dict(part="blade spring", spec="o 6 x 0.7 wire x 25, 1.6 N/mm (E), two per blade", n=8, url=A + "uxcell-0-7mmx6mmx25mm-Stainless-Compression-Springs/dp/B076LRJH99", check="on a scale: 12.5 N at 17.1 long"),
    dict(part="pilot spring", spec="o 6 x 0.6 wire x 20", n=4, url=A + "uxcell-0-6mmx6mmx20mm-Stainless-Compression-Springs/dp/B076LWWTR7"),
    dict(part="carriage return spring", spec="extension spring, 5 to 10 N over 60 mm, from the kit", n=2, url=A + "Extension-Different-Assortment-Stainless-Individual/dp/B0CFRF42LQ", check="rates not listed: pick on a scale"),
    dict(part="float washer", spec="PTFE washer M5 between each float bolt's head and its tie, with a steel M5 x 15 washer", n=8, url=A + "PATIKIL-Washer-Sealing-Gasket-Flanges/dp/B0H1XBGWPM"),
    dict(part="insert M5", spec="heat-set, rail bolts", n=4, url=A + "Threaded-Inserts-M5xD7xL6-Printing-Components/dp/B0H142Y3BM"),
    dict(part="insert M3 / M4", spec="heat-set: stepper faces and servo M3 (16); float bolts M4 (8, CHECK: an M4 listing)", n=24, url=A + "initeq-M3-0-5-Threaded-Inserts-Printing/dp/B077CHDSTR"),
    dict(part="wheel axle", spec="M5 x 20 countersunk DIN 7991", n=6, url=A + "Socket-Screws-Countersunk-Stainless-Bright/dp/B0CJV8XJ3Y"),
    dict(part="carbon rod 3", spec="o 3 x 500: blade rods 66 long, four; on hand", n=1, url=A + "Carbon-Fiber-Length-Diameter-Suitable/dp/B09N91J1FC"),
]

COL = dict(hub="#c9d3dc", rail="#9fb4cc", carriage="#e8b04a", lift="#5f9ed1", unit="#7fc97f", bought="#8a8f96", tpu="#333333", fixed="#b7a3c9", bay="#8a94a6", pack="#2f4fbf", box="#d8d8d8")


def write_assembly(res, S, H, FX, out_dir):
    path = os.path.join(ROOT, "systems", "swap_arm.assembly.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    parts = []
    def add(pid, name, m, group, color, role, material, kin=None, opacity=None, folder=None, version="arm v8 (13 Nov)"):
        if folder: mesh = f"parts/{folder}/part.stl"
        else:
            mesh = f"parts/box_arm8/view/{pid}.stl"; tm(m).export(os.path.join(ROOT, mesh))
        parts.append(dict(id=pid, name=name, role=role, group=group, color=color, opacity=opacity, mesh=mesh, pose=None, material=material, tag="C", owner="swap arm",
                          version=version, source="parts/box_arm8/make.py", kin=kin))
    hub = U(*hub_drawn()) - U(*hub_cuts())
    add("box_arm8_hub_setup", "hub (drawn attachments; the body is generated)", hub, "hub (turns)", COL["hub"], "structure", "PCTG", dict(stage="hub"), version="setup, not run")
    for k, (m, mat, kind) in FX.items():
        add("box_arm8_fx_" + k, k.replace("_", " "), m, "stub side (fixed)", COL["fixed"] if kind == "print" else COL["bought"], "structure" if kind == "print" else "bought", mat,
            folder=("box_arm8_" + k) if kind == "print" else None)
    names = dict(carriage="carriage (setup envelope; generated)", lift="lift: block, cross-over, booms (setup envelope; generated)")
    for hd, f in (("A", lambda q: q), ("B", rot180)):
        for k, (m, mat, kind, grp) in H.items():
            stage = "hub" if grp == "hub" else ("lift" if grp == "unit" else grp)
            col = COL["bought"] if kind == "bought" else COL["tpu"] if mat == "TPU" else COL[grp if grp != "hub" else "rail"]
            add(f"box_arm8_{hd.lower()}_{k}", f"head {hd}: {names.get(k, k.replace('_', ' '))}", f(m), f"head {hd}", col, "bought" if kind == "bought" else "structure", mat,
                dict(stage=stage, head=hd, blade=(1 if k == "blade_p" else -1 if k == "blade_n" else 0) * (1 if hd == "A" else -1)), version="setup, not run" if kind == "run" else "arm v8 (13 Nov)")
    for kind, angs in BAYS.items():
        for a_ in angs:
            add(f"box_arm8_bay_{a_}_recv".replace("-", "m"), f"bay {a_}: {kind} receiver (latch v7)", at_radius(S[kind]["recv"], L.R_BAY, a_), "bays", COL["bay"], "protected", "PCTG", opacity=0.55)
            add(f"box_arm8_bay_{a_}_saddle".replace("-", "m"), f"bay {a_}: saddle", at_radius(saddle(S, kind), L.R_BAY, a_), "bays", "#6f7c91", "structure", "PCTG")
            add(f"box_arm8_bay_{a_}_cart".replace("-", "m"), f"bay {a_}: {kind} cartridge", at_radius(S[kind]["cart"], L.R_BAY, a_), "packs", COL["pack"], "protected", None, dict(bay=a_))
    add("box_arm8_drone_recv", "drone seat: Overseer receiver", at_radius(S["overseer"]["recv"], L.R_DRONE), "drone seat", COL["bay"], "protected", "PCTG", opacity=0.45)
    add("box_arm8_drone_cart", "drone seat: Overseer cartridge (the dead pack)", at_radius(S["overseer"]["cart"], L.R_DRONE), "packs", "#bf4f2f", "protected", None, dict(pack="dead"))
    fm = L.latch()._funnel()
    add("box_arm8_funnel", "funnel (Overseer walls), from parts/box_funnel", fm.funnel("overseer").transform(np.array([[0, -1.0, 0, 0], [1.0, 0, 0, L.Y_C], [0, 0, 1.0, 220.3]])), "box (context)", COL["box"], "protected", "PCTG", opacity=0.3)
    c = res["compat"]; nb = res["numbers"]
    asm = {"name": "swap arm", "title": "Swap Arm v8", "readme_view": "iso", "artifact": old.get("artifact", ARTIFACT), "machine": "box", "order": 80,
           "views": {"iso": {"eye": [330, -260, 250], "target": [0, 120, 30]}, "front": {"eye": [0, 620, 40], "target": [0, 120, 40]}, "side": {"eye": [560, 150, 60], "target": [0, 150, 60]},
                     "head A": {"eye": [150, 330, 150], "target": [10, 150, 60]}, "hub": {"eye": [160, -160, 120], "target": [0, 50, 0]}},
           "note": f"The box's swap arm, v8, for latch v7 and both packs. Two heads 180 apart; each is a lift on two steel rods driven by a T8 x 2 screw, picking its carriage up for the last "
                   f"{L.T2:g} mm: {nb['travel_mm']['drone']} mm to the drone seat. Push limited by the grab unit's float springs to {L.F_LIMIT:g} N (x{L.F_LIMIT / L.F_INSERT['overseer']:.2f} over the Overseer insert); the screw holds with the power off. "
                   f"Bays at R {L.R_BAY:g}. Swap cycle, both packs, {c['states']} states: {'0 overlaps' if c['zero'] else str(len(c['overlaps'])) + ' overlapping pairs, listed under the viewer'}. "
                   f"Hub, carriage and lift are shown as their setups (drawn attachments and envelope): the engine has not run them yet.",
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
    ld = "".join(td(r["case"], r["cls"], r["F_N"], max(r["bearing_N"]), r["nut_N"]) for r in res["loads"]["cases"] if "overseer" in r["case"])
    return (f"<div class='arm8'><style>.arm8 table{{border-collapse:collapse;font-size:13px;margin:8px 0}}.arm8 td,.arm8 th{{border:1px solid #8884;padding:3px 7px;vertical-align:top}}</style>"
            f"<h3>What the head does</h3><table>{''.join(rows)}</table>"
            f"<h3>Printed pieces (drawn)</h3><table><tr><th>part</th><th>material</th><th>n</th><th>g each</th><th>box mm</th><th>print pose</th><th>fits 180</th></tr>{prn}</table>"
            f"<p>Generated, setups ready and not yet run: hub, carriage, lift (parts/box_arm8_*/part.py).</p>"
            f"<h3>Load cases at the grab unit (each its own)</h3><table><tr><th>case</th><th>class</th><th>F N</th><th>worst LM6UU N (265 static)</th><th>nut N</th></tr>{ld}</table>"
            f"<p>Not modelled: {'; '.join(res['loads']['missing'])}.</p>"
            f"<h3>Bought</h3><table><tr><th>part</th><th>spec</th><th>n</th><th></th><th>check</th></tr>{bought}</table></div>")


if __name__ == "__main__":
    main()
