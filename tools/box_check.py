"""One compatibility check over the whole box (13 Nov, Spark: "make sure things all are compatable"): every pair of pieces from
different streams, in every state, zero overlap; the clearances that matter listed. Runs with the box page (tools/import_box.py and
the page build), writes systems/box.check.json and the page's appendix.

Streams (each has its own check of itself; this one is between them), all in the funnel's seat frame:
  funnel   parts/box_funnel/make.py: the frame at its full room (so a run's result is covered whatever it generates), both end
           walls at the drone's setting, the strips, the drive
  latch    latch v7, both packs (parts/latch7/make.py build): the receiver on the drone, the cartridge with its pack
  drones   systems/scout_v4 and systems/overseer as tools/seat_envelope.py seats them
  arm      systems/swap_arm.assembly.json once it exists (W frame), else the arm v7 parts of systems/box.assembly.json (superseded);
           the parts that turn with the wheel swept about the axle
  box      the rest of systems/box.assembly.json: bays, stand, enclosure, dome and rain lid
States: seated (each drone, its wall setting); landing (tools/seat_check.py descent_states, 65 a drone, its latch set on it); swap
(the cartridge down its path through the hole in 4 mm steps, the drone seated); the wheel turning (3 deg steps) under each seated drone.
Pairs left to their own stream's check: a drone and its own latch set, the arm and the cartridge it grips, the arm and the bays.
The dome's moving hoops and fabric are drawn shut: they are tested against the seated Scout only (the dome is open for every landing
and for the Overseer; its own check, parts/box_dome/make.py, covers its motion).
    python3 tools/box_check.py [--quick]      exit 1 on any overlap over 0.5 mm3 (--quick: every third landing state, 6 deg wheel steps)
"""
import importlib.util, json, os, sys, time
import numpy as np, trimesh, manifold3d as mf
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import seat_envelope as se, seat_check as sc, wheel_sweep as ws

W2S = np.linalg.inv(sc.SEAT_TO_W)
TOL = 0.5                                     # mm3
PACK = {"scout_v4": "scout", "overseer": "overseer"}


def load(rel, name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


class Body:
    """a piece: name, stream, trimesh, manifold (None when its mesh is not a solid: checked by surface samples then)"""
    def __init__(self, name, stream, tri=None, man=None):
        self.name, self.stream = name, stream
        self.man = man if man is not None else sc.man(tri)
        self.tri = tri if tri is not None else sc.to_tri(man)
        self.lo, self.hi = self.tri.bounds
        self._pts = None

    def pts(self):
        if self._pts is None:
            p, _ = trimesh.sample.sample_surface(self.tri, int(min(60000, max(1500, self.tri.area / 1.5))), seed=1)
            self._pts = np.vstack([p, self.tri.vertices])
        return self._pts

    def moved(self, off):
        b = Body.__new__(Body); b.name, b.stream = self.name, self.stream; off = np.asarray(off, float)
        b.man = None if self.man is None else self.man.translate(off.tolist())
        b.tri = self.tri.copy(); b.tri.apply_translation(off); b.lo, b.hi = self.lo + off, self.hi + off
        b._pts = None if self._pts is None else self._pts + off
        return b


def overlap(a, b):
    """mm3 both hold (exact when both are solids; else the count of a's surface samples inside b, as -n)"""
    if (a.lo > b.hi).any() or (a.hi < b.lo).any():
        return 0.0
    if a.man is not None and b.man is not None:
        return sc.volume(a.man ^ b.man)
    s, o = (a, b) if b.man is not None else (b, a)
    if o.man is None:
        return 0.0                                               # two open meshes (rod patterns): no test
    p = s.pts(); p = p[((p >= o.lo) & (p <= o.hi)).all(1)]
    return -float(o.tri.contains(p).sum()) if len(p) else 0.0


def gap(a_tree, a_lo, a_hi, b, reach=25.0):
    """the least distance from b's surface to a's (sampled, good to about 1 mm), or None beyond reach"""
    if (b.lo > a_hi + reach).any() or (b.hi < a_lo - reach).any():
        return None
    p = b.pts(); p = p[((p >= a_lo - reach) & (p <= a_hi + reach)).all(1)]
    if not len(p):
        return None
    d, _ = a_tree.query(p, distance_upper_bound=reach)
    if not np.isfinite(d).any():
        return None
    k = int(np.argmin(d)); gap.at = [round(float(v)) for v in p[k]]
    return float(d[k])


def funnel_bodies(fm):
    out = {}
    for st in fm.SETTINGS:
        out[st] = [Body("funnel frame (full room)", "funnel", man=fm.frame()), Body("funnel nose wall", "funnel", man=fm.carriage(1, st)),
                   Body("funnel tail wall", "funnel", man=fm.carriage(-1, st)), Body("funnel strips", "funnel", man=fm.strips()),
                   Body("funnel drive", "funnel", man=fm.drive())]
    return out


def latch_bodies():
    """per drone: (receiver, cartridge with its pack), seated at the height the latch's own fit found"""
    L = load("parts/latch7/make.py", "latch7_make"); out = {}
    for drone, pack in PACK.items():
        B = L.build(pack); T = L.to_seat(B); g = B["g"]
        res = os.path.join(ROOT, "parts", L.OUT[pack][0], "result.json")
        rest = float(json.load(open(res)).get("funnel", {}).get("rests_at_mm", 0.0)) if os.path.exists(res) else 0.0
        R = L.m6.receiver_all(B).transform(T).translate([0, 0, rest + 0.05])
        C = (L.m6.cart_all(B) + L.m6.moving(g, L.m6.R0, 1) + L.m6.moving(g, L.m6.R0, -1) + L.bl.pack_body(g)).transform(T).translate([0, 0, rest + 0.05])
        out[drone] = (Body(f"latch v7 receiver ({pack})", "latch", man=R), Body(f"latch v7 cartridge + pack ({pack})", "latch", man=C))
    return out


def box_bodies():
    """(static box bodies, static arm bodies, turning arm point clouds, the arm's source, pieces with no mesh in the tree)"""
    a = json.load(open(os.path.join(ROOT, "systems", "box.assembly.json")))
    new_arm = os.path.join(ROOT, "systems", "swap_arm.assembly.json")
    parts = [p for p in a["parts"] if not p.get("group", "").startswith("landing") and not p["id"].startswith(("box_funnel", "box_ctx", "box_latch"))]
    src = "arm v7 (systems/box.assembly.json, superseded)"
    if os.path.exists(new_arm):
        parts = [p for p in parts if not p["name"].startswith("arm: ")]
        b = json.load(open(new_arm)); src = "systems/swap_arm.assembly.json"
        ctx = lambda p: p.get("group") in ("drone seat", "box (context)") or (p.get("kin") or {}).get("pack") == "dead"   # its copies of the seat's latch set and of the funnel
        parts += [dict(p, name="arm: " + p["name"], _new=True) for p in b["parts"] if not ctx(p)]
    static, arm, turning, missing = [], [], [], []
    for p in parts:
        if p.get("mesh"):
            f = os.path.join(ROOT, p["mesh"])
            if not os.path.exists(f):
                missing.append(p["name"]); continue
            m = trimesh.load(f, force="mesh")
        elif "box" in p:
            b = p["box"]; m = trimesh.creation.box(extents=[b[1] - b[0], b[3] - b[2], b[5] - b[4]]); m.apply_translation([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        else:
            continue
        P = np.array(p["pose"], float) if isinstance(p.get("pose"), list) else np.eye(4)
        nm = p["name"].replace("arm: ", "")
        is_arm = p["name"].startswith("arm: ") and (p.get("_new") or not nm.startswith("rec"))
        turns = bool((p.get("kin") or {}).get("stage")) if p.get("_new") else (is_arm and nm.startswith(ws.TURN) and "placeholder" not in nm)   # v8: hub, carriage and lift stages turn with the wheel
        if turns:
            mw = m.copy(); mw.apply_transform(P); q, _ = trimesh.sample.sample_surface(mw, int(min(20000, max(1500, mw.area / 4.0))), seed=1)
            turning.append((p["name"], np.vstack([q, mw.vertices])))
        m.apply_transform(W2S @ P)
        (arm if is_arm else static).append(Body(p["name"], "arm" if is_arm else "box", tri=m))
    lane = None                                                       # the arm's head lane under the tail wall: the arm's own (parts/box_arm8/make.py funnel_keepout() -> W box), else the funnel's copy
    a8 = os.path.join(ROOT, "parts", "box_arm8", "make.py")
    if os.path.exists(a8):
        try:
            lane = getattr(load("parts/box_arm8/make.py", "box_arm8_make"), "funnel_keepout")()
        except Exception:
            lane = None
    if lane is None:
        x0, x1, y0, y1, zt = load("parts/box_funnel/make.py", "box_funnel_make").ARM_LANE
        m = trimesh.creation.box(extents=[x1 - x0, 144.0, 60.0]); m.apply_translation([(x0 + x1) / 2, -44.0, zt - 30.0])     # seat frame: y -116..28 (W X -28..116), 60 down
    else:
        x0, x1, y0, y1, z0, z1 = lane; m = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0]); m.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]); m.apply_transform(W2S)
    arm.append(Body("arm v8 head lane at the drone (keep-out)", "arm", tri=m))
    return static, arm, turning, src, missing


def main(quick=False):
    t0 = time.time(); fm = load("parts/box_funnel/make.py", "box_funnel_make")
    F = funnel_bodies(fm); LB = latch_bodies(); static, arm, turning, arm_src, missing = box_bodies()
    drones = {d: [Body(f"{d}: {nm}", "drones", tri=m) for nm, m in se.parts_of(d)] for d in PACK}
    hits, rows = [], []
    moving_dome = lambda b: b.name.startswith(("dome hoop 1", "dome hoop 2", "dome hoop 3", "dome hoop 4", "dome fabric"))
    def test(state, A, Bs):
        for a in A:
            for b in Bs:
                if a.stream == b.stream or {a.stream, b.stream} == {"latch", "arm"}:
                    continue
                if (moving_dome(a) or moving_dome(b)) and state != "scout_v4 seated":
                    continue
                v = overlap(a, b)
                if v > TOL or v < 0:
                    hits.append(dict(state=state, a=a.name, b=b.name, mm3=round(v, 1) if v > 0 else None, samples_inside=int(-v) if v < 0 else None))
    fun_all = {st: F[st] for st in F}
    # the funnel against the box it sits in (both settings), and the arm at park
    for st in F:
        test(f"static, walls at {st}", F[st], static + arm)
    test("static, the arm at park", [b for b in static if not moving_dome(b)], [b for b in arm if "keep-out" not in b.name])   # the stand, the enclosure and the dome against the arm and its bays
    for d in PACK:
        st = sc.walls_for(d); R, C = LB[d]; states = sc.descent_states(d)
        if quick:
            states = states[:1] + states[1::3]
        for lab, ox, oy, h in states:
            mv = [b.moved((ox, oy, h)) for b in drones[d] + [R, C]]
            test(f"{d} {'seated' if lab == 'seated' else 'landing ' + lab}", mv, F[st] + static + arm)
        rows.append(dict(what=f"{d}: seated and {len(states) - 1} landing states against the funnel (walls at {st}), the box and the arm at park", ok=True))
        for k in np.arange(0.0, 160.01, 4.0):                                              # the cartridge down its path, the drone seated
            test(f"{d} swap, cartridge {k:.0f} down", [C.moved((0, 0, -k))], F[st] + [b for b in static if not moving_dome(b)])
        rows.append(dict(what=f"{d}: cartridge down its swap path (0..160) against the funnel and the box", ok=True))
    # the wheel turning under each seated drone and through the funnel
    step = 6 if quick else 3; n_turn = 0
    seated = {d: drones[d] + [LB[d][0]] for d in PACK}
    for ang in range(0, 360, step):
        c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang)); Rm = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        for tn, pts in turning:
            q = (pts @ Rm.T) @ W2S[:3, :3].T + W2S[:3, 3]; lo, hi = q.min(0), q.max(0)
            for d in PACK:
                for b in F[sc.walls_for(d)] + seated[d]:
                    if (lo > b.hi).any() or (hi < b.lo).any():
                        continue
                    qq = q[((q >= b.lo) & (q <= b.hi)).all(1)]
                    if not len(qq):
                        continue
                    n = int(b.tri.contains(qq).sum()) if b.man is not None else int((trimesh.proximity.closest_point(b.tri, qq)[1] < 1.5).sum())
                    if n:
                        hits.append(dict(state=f"wheel at {ang} deg, {d} seated", a=tn, b=b.name, mm3=None, samples_inside=n)); n_turn += 1
    rows.append(dict(what=f"the wheel's {len(turning)} turning parts, every {step} deg, against the funnel and both seated drones", ok=n_turn == 0))
    # clearances that matter (sampled, about 1 mm)
    clear = []
    for st in F:
        fb = Body("funnel", "funnel", man=fm.funnel(st)); tree = cKDTree(fb.pts())
        d = [k for k in PACK if sc.walls_for(k) == st][0]
        for lab, group in (("seated " + d, [b for b in drones[d]]), (f"latch receiver ({PACK[d]})", [LB[d][0]]), (f"latch cartridge ({PACK[d]}), seated", [LB[d][1]])):
            g = []
            for b in group:
                gg = gap(tree, fb.lo, fb.hi, b)
                if gg is not None:
                    g.append((gg, f"{b.name[:60]} at seat {gap.at}"))
            if g:
                clear.append(dict(between=f"funnel (walls at {st})", and_=lab, mm=round(min(g)[0], 1), at=min(g)[1]))
        best = (None, None, None)
        for lab, ox, oy, h in sc.descent_states(d)[1:]:
            for b in drones[d]:
                gg = gap(tree, fb.lo, fb.hi, b.moved((ox, oy, h)), reach=12.0)
                if gg is not None and (best[0] is None or gg < best[0]):
                    best = (gg, f"{b.name[:60]} at seat {gap.at}", lab)
        if best[0] is not None:
            clear.append(dict(between=f"funnel (walls at {st})", and_=f"{d} landing, closest pass", mm=round(best[0], 1), at=f"{best[1]} at {best[2]}"))
        if st == "overseer":
            for b in static + arm:
                gg = gap(tree, fb.lo, fb.hi, b, reach=20.0)
                if gg is not None and gg < 12.0:
                    clear.append(dict(between="funnel", and_=b.name, mm=round(gg, 1), at=f"at seat {gap.at}"))
            near = None
            for ang in range(0, 360, step):
                c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang)); Rm = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
                for tn, pts in turning:
                    q = (pts @ Rm.T) @ W2S[:3, :3].T + W2S[:3, 3]; q = q[((q >= fb.lo - 20) & (q <= fb.hi + 20)).all(1)]
                    if len(q):
                        dd, _ = tree.query(q, distance_upper_bound=20.0)
                        if np.isfinite(dd).any() and (near is None or dd.min() < near[0]):
                            near = (float(dd.min()), f"{tn} at seat {[round(float(v)) for v in q[int(np.argmin(dd))]]}", ang)
            if near:
                clear.append(dict(between="funnel", and_="the wheel turning, closest pass", mm=round(near[0], 1), at=f"{near[1]} at {near[2]} deg"))
    seen = {}
    for h in hits:                                                    # one line a pair, its worst state
        k = (h["a"], h["b"]); w = h["mm3"] if h["mm3"] is not None else h["samples_inside"]
        if k not in seen or w > seen[k][0]:
            seen[k] = (w, h)
        seen[k] = (seen[k][0], dict(seen[k][1], states=seen[k][1].get("states", 0) + 1 if k in seen and "states" in seen[k][1] else 1))
    pairs = [dict(v[1], states=sum(1 for h in hits if (h["a"], h["b"]) == k)) for k, v in seen.items()]
    out = dict(ok=not hits, overlapping_pairs=sorted(pairs, key=lambda p: -(p["mm3"] or p["samples_inside"] or 0)), checked=rows,
               clearances_mm=sorted(clear, key=lambda c: c["mm"]), arm=arm_src, no_mesh_in_the_tree=missing, quick=quick,
               counts=dict(box=len(static), arm=len(arm), turning=len(turning), drone_parts={d: len(v) for d, v in drones.items()}), seconds=round(time.time() - t0))
    json.dump(out, open(os.path.join(ROOT, "systems", "box.check.json"), "w"), indent=1)
    print(f"box check: {'PASS, 0 overlaps' if out['ok'] else f'FAIL, {len(pairs)} overlapping pairs'} ({len(static)} box, {len(arm)} arm, {len(turning)} turning pieces; arm = {arm_src}; {out['seconds']} s)")
    for p in out["overlapping_pairs"][:40]:
        print(f"  {p['a'][:44]:44} x {p['b'][:40]:40} {p['mm3'] if p['mm3'] is not None else str(p['samples_inside']) + ' samples':>10} mm3 in {p['states']} state(s), worst {p['state']}")
    for c in out["clearances_mm"]:
        print(f"  clear {c['mm']:5.1f}  {c['between']} / {c['and_']}  {c['at']}")
    if missing:
        print("  not checked, no mesh in the tree: " + ", ".join(missing))
    return out


if __name__ == "__main__":
    sys.exit(0 if main(quick="--quick" in sys.argv)["ok"] else 1)
