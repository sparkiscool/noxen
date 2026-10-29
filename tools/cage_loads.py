"""Overseer cage frame model (29 Oct): the 35 g crash set behind node_loads in systems/overseer.cage.json.

Why a new one: the first frame model (runs/overseer_frame/v1, not in this repo) held the core rigid, so a crash force entering at
a corner node ran through a spoke, across the core and out the other arms: 1465 N in one 3 mm spoke, index 4..10 on every arm
piece, 2 kN shear on an M3 in a 1.5 mm plate. Here the core is a free rigid body with its mass, hung on its core ties, bay ties
and the four PETG arms at their own stiffness (a spoke rod is ~100x stiffer than the arm it ends on, so the arm gives and the
cage carries its load round it); the head (L2, cameras, mount, node 17) is a rigid body on nodes 15 / 16 and the tubes at 17;
the cage is 6/4 carbon tubes and 3 mm rods with rigid joints.

Cases: a 35 g stop in each of 14 directions (6 axes, 8 diagonals): the drone moving along d meets a flat surface and rocks
onto the face of its node hull that the ray from its CG along d leaves through; that face's nodes are held along the face
normal by the surface, 3-2-1 across it (no tangential clamp between them), and every mass pushes along d at 35 g (the
resultant through the CG passes inside the face, so the contact reactions are pushes; the tangential part is friction). Directions that land on one face are one case. And
landing 6 g on the six base nodes. Masses: systems/overseer.assembly.json (mass_g, placed CG): cage nodes at their centres, tubes and
rods half to each end, motors + props at the motor, half an arm at its spoke node and half at its root, the head group at its
CG, the rear camera at node 21, everything else is the core at its CG.
Out: node_loads per cage node per member: axial = largest |N| over the cases (N tension +, kept in "N"), side = largest shear
at that end, case = which; systems/overseer.cage_loads.json: the model (masses, sections), every case (supports, reactions,
each member's N and end shears), tube compression / Euler pinned-end buckling load.
E carbon 120 GPa, PETG 2.78 GPa (dimensions.md). Arm beam section A 300 mm2, I 2.5e4 mm4 (a generated 30 x 30 arm at ~37 %, E).
Run: python tools/cage_loads.py [--dry]   (--dry: print, write nothing)
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import assembly as A                                         # noqa: E402
import system_check as sc                                    # noqa: E402

G = 9.81e-3                                                  # N per g per g
E_CARBON, E_PETG, NU = 120e3, 2780.0, 0.3
TUBE = dict(E=E_CARBON, A=np.pi * (3.0 ** 2 - 2.0 ** 2), I=np.pi / 4 * (3.0 ** 4 - 2.0 ** 4))     # 6/4 carbon tube
RODS = dict(E=E_CARBON, A=np.pi * 1.5 ** 2, I=np.pi / 4 * 1.5 ** 4)                               # 3.0 solid carbon
ARM = dict(E=E_PETG, A=300.0, I=2.5e4)                                                            # generated PETG arm (E)
STIFF = dict(E=200e3, A=2000.0, I=2.0e6)                                                          # rigid-body links
CRASH_G, LAND_G = 35.0, 6.0
ROOT_R, ROOT_Z, Z_MOTOR = 66.5, 22.5, 60.0                   # arm root on the diagonal, mid-height; motor CG (arm lib)
BASE = [0, 1, 2, 3, 4, 5]
CAGE_PATH = os.path.join(ROOT, "systems", "overseer.cage.json")
OUT_PATH = os.path.join(ROOT, "systems", "overseer.cage_loads.json")
ASM_PATH = os.path.join(ROOT, "systems", "overseer.assembly.json")


def crash_dirs():
    out = []
    for ax in range(3):
        for s in (1, -1):
            d = np.zeros(3); d[ax] = s; out.append(d)
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                out.append(np.array([sx, sy, sz]) / np.sqrt(3))
    return out


def beam_k(sec, L):
    """12 x 12 local stiffness of a round-section Euler beam: dofs ux uy uz rx ry rz at i, then j"""
    E, A, I = sec["E"], sec["A"], sec["I"]; J = 2 * I; Gm = E / (2 * (1 + NU))
    k = np.zeros((12, 12)); a = E * A / L; t = Gm * J / L
    b1, b2, b3, b4 = 12 * E * I / L ** 3, 6 * E * I / L ** 2, 4 * E * I / L, 2 * E * I / L
    k[0, 0] = k[6, 6] = a; k[0, 6] = k[6, 0] = -a
    k[3, 3] = k[9, 9] = t; k[3, 9] = k[9, 3] = -t
    for u, r, s in ((1, 5, 1.0), (2, 4, -1.0)):             # uy with rz, uz with ry
        k[u, u] = k[u + 6, u + 6] = b1; k[u, u + 6] = k[u + 6, u] = -b1
        k[r, r] = k[r + 6, r + 6] = b3; k[r, r + 6] = k[r + 6, r] = b4
        k[u, r] = k[r, u] = s * b2; k[u, r + 6] = k[r + 6, u] = s * b2
        k[u + 6, r] = k[r, u + 6] = -s * b2; k[u + 6, r + 6] = k[r + 6, u + 6] = -s * b2
    return k


def rotation(p, q):
    x = q - p; L = float(np.linalg.norm(x)); x = x / L
    ref = np.array([0, 0, 1.0]) if abs(x[2]) < 0.9 else np.array([0, 1.0, 0])
    y = np.cross(ref, x); y /= np.linalg.norm(y); z = np.cross(x, y)
    R = np.vstack([x, y, z]); T = np.zeros((12, 12))
    for b in range(4):
        T[3 * b:3 * b + 3, 3 * b:3 * b + 3] = R
    return L, T


def contact(N, d, cg, tol=1.0):
    """the drone pushed along d against a flat surface settles on the face of its node hull that the ray from its CG along d
    leaves through (the drone rocks onto that face). Returns (nodes on that face, outward unit normal of the face)"""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(N); best = None
    for eq in hull.equations:                                    # n . x + off = 0, n outward
        n_, off = eq[:3], eq[3]; nd = n_ @ d
        if nd <= 1e-9:
            continue
        t = -(n_ @ cg + off) / nd
        if best is None or t < best[0] - 1e-9:
            best = (t, n_, off)
    t, n_, off = best
    touch = [int(i) for i in np.argsort(-(N @ n_)) if abs(N[i] @ n_ + off) < tol]
    assert len(touch) >= 3, touch
    return touch, n_ / np.linalg.norm(n_)


def supports(N, touch, d):
    """a hard surface normal to d at the touching nodes: each held along d; the first held in the plane too and the second
    across the line to the first (3-2-1: no rigid-body motion, no tangential clamp between the nodes)"""
    cons = [(i, np.asarray(d, float)) for i in touch]
    a, b = touch[0], touch[1]
    e = N[b] - N[a]; e -= (e @ d) * d; e /= np.linalg.norm(e); f = np.cross(d, e)
    cons += [(a, e), (a, f), (b, f)]
    return cons


class Frame:
    def __init__(self):
        self.pts, self.members, self.mass = [], [], []          # mass: (point index, g)

    def node(self, p, tol=0.5):
        p = np.asarray(p, float)
        for i, q in enumerate(self.pts):
            if np.linalg.norm(q - p) < tol:
                return i
        self.pts.append(p); return len(self.pts) - 1

    def member(self, name, i, j, sec, kind):
        self.members.append(dict(name=name, i=i, j=j, sec=sec, kind=kind))

    def solve(self, loads, constraints):
        """loads: (n, 3) force per point; constraints: [(point, unit direction)] translations held along that direction
        (Lagrange multipliers). Returns u (n, 6), per member (N at j, shear at i / j, moment at i / j), reactions per constraint."""
        n = len(self.pts); K = np.zeros((6 * n, 6 * n)); F = np.zeros(6 * n)
        cache = []
        for m in self.members:
            p, q = self.pts[m["i"]], self.pts[m["j"]]
            L, T = rotation(p, q); kl = beam_k(m["sec"], L); kg = T.T @ kl @ T
            idx = np.r_[6 * m["i"]:6 * m["i"] + 6, 6 * m["j"]:6 * m["j"] + 6]
            K[np.ix_(idx, idx)] += kg; cache.append((kl, T, idx))
        for i in range(n):
            F[6 * i:6 * i + 3] = loads[i]
        C = np.zeros((len(constraints), 6 * n))
        for r, (i, d) in enumerate(constraints):
            C[r, 6 * i:6 * i + 3] = d
        m_ = len(constraints)
        KKT = np.zeros((6 * n + m_, 6 * n + m_)); KKT[:6 * n, :6 * n] = K; KKT[:6 * n, 6 * n:] = C.T; KKT[6 * n:, :6 * n] = C
        rhs = np.r_[F, np.zeros(m_)]
        sol = np.linalg.solve(KKT, rhs); u, lam = sol[:6 * n], sol[6 * n:]
        resid = np.linalg.norm(KKT @ sol - rhs) / max(np.linalg.norm(F), 1e-9)
        if resid > 1e-6:
            raise RuntimeError(f"frame solve residual {resid:.2e}: the model has a mechanism or is ill-conditioned")
        out = []
        for m, (kl, T, idx) in zip(self.members, cache):
            f = kl @ (T @ u[idx])
            out.append(dict(N=float(f[6]), shear_i=float(np.hypot(f[1], f[2])), shear_j=float(np.hypot(f[7], f[8])),
                            M_i=float(np.hypot(f[4], f[5])), M_j=float(np.hypot(f[10], f[11]))))
        return u.reshape(n, 6), out, -lam                        # -lam: the surface's force on the node along d


def masses(asm_rows, cage):
    """group the assembly's placed masses: returns dict core=(g, cg), head=(g, cg), arms[k]=(arm g, motor g), cam21 g,
    node[g by id], tubes g, rods g"""
    core_m, core_s = 0.0, np.zeros(3); head_m, head_s = 0.0, np.zeros(3)
    arms = {k: dict(arm=0.0, motor=0.0) for k in range(4)}
    nodes, tubes, rods, cam21 = {}, 0.0, 0.0, 0.0
    for r in asm_rows:
        pid, g, c = r["id"], r["mass_g"], np.asarray(r["cg"], float)
        if pid.startswith("overseer_cage_node_"):
            nodes[int(pid.rsplit("_", 1)[1])] = g
        elif pid == "cage_shell_tubes":
            tubes += g
        elif pid == "spokes_ties_head_rods":
            rods += g
        elif pid in ("overseer_head_mount", "unitree_l2", "camera_left", "camera_right"):
            head_m += g; head_s += g * c
        elif pid == "camera_rear":
            cam21 += g
        elif pid.startswith("motor_") or pid.startswith("prop_"):
            arms[int(pid.rsplit("_", 1)[1]) - 1]["motor"] += g
        elif pid in ("overseer_arm_front_root", "overseer_arm_front_tip"):
            arms[0]["arm"] += g
        elif pid == "overseer_arm_rear":
            arms[1]["arm"] += g
        elif pid == "arm_3":
            arms[2]["arm"] += g
        elif pid in ("arm_4_root", "arm_4_tip"):
            arms[3]["arm"] += g
        else:
            core_m += g; core_s += g * c
    return dict(core=(core_m, core_s / core_m), head=(head_m, head_s / head_m), arms=arms, cam21=cam21, nodes=nodes,
                tubes=tubes, rods=rods)


def build(cage, mg):
    fr = Frame(); N = np.array(cage["nodes"], float)
    for p in N:
        fr.node(p, tol=1e-6)                                     # cage node id = point index
    nnode = len(N)
    lengths = {}
    for a, b in cage["rods"]:
        fr.member(f"tube {a}-{b}", a, b, TUBE, "tube"); lengths[f"tube {a}-{b}"] = float(np.linalg.norm(N[a] - N[b]))
    link_pts = {}
    for nm, (p, q) in cage["links"].items():
        if nm == "head rod c":
            continue                                             # both ends on the head mount (node 17 is part of it)
        i, j = fr.node(p), fr.node(q); link_pts[nm] = (i, j)
        fr.member(nm, i, j, RODS, "link"); lengths[nm] = float(np.linalg.norm(np.asarray(p) - np.asarray(q)))
    # masses: nodes, tubes and rods half to each end
    load = {}
    def add(i, g):
        load[i] = load.get(i, 0.0) + g
    for n_, g in mg["nodes"].items():
        add(n_, g)
    tube_len = sum(lengths[f"tube {a}-{b}"] for a, b in cage["rods"]); rod_len = sum(lengths[nm] for nm in link_pts)
    for a, b in cage["rods"]:
        g = mg["tubes"] * lengths[f"tube {a}-{b}"] / tube_len; add(a, g / 2); add(b, g / 2)
    for nm, (i, j) in link_pts.items():
        g = mg["rods"] * lengths[nm] / rod_len; add(i, g / 2); add(j, g / 2)
    add(21, mg["cam21"])
    # core: a rigid body at its CG, stiff links to the core tie and bay tie ends and the four arm roots
    core_g, core_c = mg["core"]; ic = fr.node(core_c, tol=1e-6); add(ic, core_g)
    core_ends = set()
    for nm, (i, j) in link_pts.items():
        if nm.startswith("core tie") or nm.startswith("bay tie"):
            core_ends.add(i if i >= nnode else j)
    for e in sorted(core_ends):
        fr.member(f"core link {e}", ic, e, STIFF, "stiff")
    # arms: root (on the core) -> spoke node -> motor, PETG beams
    arm_nodes = {}
    for k in range(4):
        M = np.r_[cage["motors_xy"][k], Z_MOTOR]; th = np.radians(45 + 90 * k)
        Rk = np.array([ROOT_R * np.cos(th), ROOT_R * np.sin(th), ROOT_Z])
        sp = cage["links"][f"spoke {k + 1}.1"]; P = np.asarray(sp[0] if fr.node(sp[0]) >= nnode else sp[1], float)
        ir, ip, im = fr.node(Rk), fr.node(P), fr.node(M)
        arm_nodes[k] = ip
        fr.member(f"arm {k + 1} root", ic, ir, STIFF, "stiff")
        fr.member(f"arm {k + 1} inboard", ir, ip, ARM, "arm")
        fr.member(f"arm {k + 1} outboard", ip, im, ARM, "arm")
        add(im, mg["arms"][k]["motor"]); add(ip, mg["arms"][k]["arm"] / 2); add(ir, mg["arms"][k]["arm"] / 2)
    # head: a rigid body at its CG on node 17 and the two head rod ends
    head_g, head_c = mg["head"]; ih = fr.node(head_c, tol=1e-6); add(ih, head_g)
    fr.member("head link 17", ih, 17, STIFF, "stiff")
    for nm in ("head rod +1", "head rod -1"):
        i, j = link_pts[nm]; e = i if i >= nnode else j
        fr.member(f"head link {nm[-2:]}", ih, e, STIFF, "stiff")
    return fr, load, lengths, nnode


def selftest():
    """the beam element against the textbook: a cantilever tip under a side force and a bar under a pull"""
    fr = Frame(); i, j = fr.node([0, 0, 0]), fr.node([200.0, 0, 0]); fr.member("c", i, j, TUBE, "tube")
    F = np.zeros((2, 3)); F[j] = [0, 0, -10.0]
    cons = [(i, np.array([1.0, 0, 0])), (i, np.array([0, 1.0, 0])), (i, np.array([0, 0, 1.0]))]
    fr2 = Frame(); fr2.pts, fr2.members = fr.pts, fr.members
    # a cantilever needs the root rotation held too: clamp it with stiff beams to two more held points
    for q in ([-1.0, 0, 0], [0, -1.0, 0]):
        k = fr2.node(q); fr2.member(f"clamp {k}", k, i, STIFF, "stiff")
        cons += [(k, np.array([1.0, 0, 0])), (k, np.array([0, 1.0, 0])), (k, np.array([0, 0, 1.0]))]
    F = np.zeros((4, 3)); F[j] = [0, 0, -10.0]
    u, mem, _ = fr2.solve(F, cons)
    want = 10.0 * 200.0 ** 3 / (3 * TUBE["E"] * TUBE["I"])
    assert abs(-u[j][2] - want) < 0.02 * want, (u[j][2], want)
    assert abs(mem[0]["shear_i"] - 10.0) < 1e-6 and abs(mem[0]["M_i"] - 2000.0) < 1e-3, mem[0]
    F = np.zeros((4, 3)); F[j] = [50.0, 0, 0]
    u, mem, _ = fr2.solve(F, cons)
    assert abs(u[j][0] - 50.0 * 200.0 / (TUBE["E"] * TUBE["A"])) < 1e-6 and abs(mem[0]["N"] - 50.0) < 1e-6, mem[0]


def run(dry=False, log=print):
    selftest()
    cage = json.load(open(CAGE_PATH))
    asm = A.load(ASM_PATH)
    rows = sc.mass_cg(asm.parts, asm.meshes())["parts"]
    by_name = {p["name"]: p["id"] for p in asm.parts}
    for r in rows:
        r["id"] = by_name[r["name"]]
    mg = masses(rows, cage)
    fr, load, lengths, nnode = build(cage, mg)
    n = len(fr.pts); total = sum(load.values())
    log(f"frame: {n} points, {len(fr.members)} members ({sum(m['kind'] == 'tube' for m in fr.members)} tubes, "
        f"{sum(m['kind'] == 'link' for m in fr.members)} rods, {sum(m['kind'] == 'arm' for m in fr.members)} arm beams); "
        f"mass {total:.0f} g: core {mg['core'][0]:.0f} at {np.round(mg['core'][1], 1).tolist()}, head {mg['head'][0]:.0f}, "
        f"cage {mg['tubes'] + mg['rods'] + sum(mg['nodes'].values()):.0f}, arms+motors "
        f"{sum(a['arm'] + a['motor'] for a in mg['arms'].values()):.0f}")
    N = np.array(cage["nodes"], float)
    cg = sum(g * fr.pts[i] for i, g in load.items()) / total
    cases, seen = [], {}
    for d in crash_dirs():
        touch, nrm = contact(N, d, cg)
        key = tuple(sorted(touch))
        if key in seen:
            continue
        seen[key] = True
        cases.append(dict(name=f"crash 35 g dir ({d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}) on face {'-'.join(map(str, touch))}",
                          d=d, n=nrm, g=CRASH_G, touch=touch))
    cases.append(dict(name="landing 6 g on the base", d=np.array([0, 0, -1.0]), n=np.array([0, 0, -1.0]), g=LAND_G, touch=BASE))
    log(f"drone CG {np.round(cg, 1).tolist()}; {len(cases) - 1} crash faces: " + ", ".join(c["name"].split("face ")[1] for c in cases[:-1]))
    results = []
    for c in cases:
        F = np.zeros((n, 3))
        for i, g in load.items():
            F[i] = g * G * c["g"] * c["d"]
        touch = list(c["touch"])
        while True:                                                  # a node the surface would have to pull lifts off
            cons = supports(N, touch, c["n"])
            u, mem, R = fr.solve(F, cons)
            reac = {}
            for (i, d), r in zip(cons, R):
                reac[int(i)] = round(float(r), 1) if abs(float(d @ c["n"])) > 0.999 else reac.get(int(i))   # the normal reaction
            lift = max(reac, key=reac.get)
            if reac[lift] <= 0.0 or len(touch) <= 3:
                break
            touch.remove(lift)
        c["touch"] = touch
        want = -total * G * c["g"] * float(c["d"] @ c["n"])                   # the surface's push, along n, over the touching nodes
        got = sum(reac.values())
        if abs(got - want) > 0.01 * abs(want) or max(reac.values()) > 0.02 * abs(want):
            raise RuntimeError(f"{c['name']}: contact reactions {reac} should sum to {want:.0f} N and all push "
                               f"(a pull means the drone would rock off this face; the old rigid-core model had both)")
        arms = {}
        for k in range(4):
            r_in = mem[[m["name"] for m in fr.members].index(f"arm {k + 1} inboard")]
            arms[k + 1] = dict(spokes={m["name"]: dict(N=round(r["N"], 1), side=round(r["shear_i"], 1)) for m, r in zip(fr.members, mem)
                                       if m["name"].startswith(f"spoke {k + 1}.")},                     # i = the arm end
                               root=dict(N=round(r_in["N"], 1), shear=round(r_in["shear_i"], 1), moment_Nmm=round(r_in["M_i"], 0)))
        results.append(dict(name=c["name"], d=np.round(c["d"], 4).tolist(), n=np.round(c["n"], 4).tolist(), g=c["g"],
                            touch=[int(i) for i in c["touch"]], reactions=reac, arms=arms,
                            members={m["name"]: dict(N=round(r["N"], 1), shear_i=round(r["shear_i"], 2), shear_j=round(r["shear_j"], 2))
                                     for m, r in zip(fr.members, mem)}))
    # node_loads: per cage node, per member ending there
    node_loads = {}
    for m in fr.members:
        if m["kind"] not in ("tube", "link"):
            continue
        for end, idx in (("i", m["i"]), ("j", m["j"])):
            if idx >= nnode:
                continue
            best = max(results, key=lambda r: abs(r["members"][m["name"]]["N"]))
            side = max(r["members"][m["name"]][f"shear_{end}"] for r in results)
            Nmax = best["members"][m["name"]]["N"]
            node_loads.setdefault(str(idx), {})[m["name"]] = dict(axial=round(abs(Nmax), 1), side=round(side, 1), N=round(Nmax, 1),
                                                                   case=best["name"], tag="F")
    for k in sorted(node_loads, key=int):
        node_loads[k] = dict(sorted(node_loads[k].items(), key=lambda kv: (not kv[0].startswith("tube"), kv[0])))
    # buckling of the tubes, pinned ends
    buck = {}
    for m in fr.members:
        if m["kind"] == "tube":
            L = lengths[m["name"]]; Pcr = np.pi ** 2 * TUBE["E"] * TUBE["I"] / L ** 2
            comp = -min(r["members"][m["name"]]["N"] for r in results)
            buck[m["name"]] = dict(L=round(L, 1), Pcr=round(Pcr, 1), compression=round(comp, 1), ratio=round(comp / Pcr, 2))
    # report
    old = cage.get("node_loads", {})
    worst = sorted(((v["axial"], k, nm) for k, d in node_loads.items() for nm, v in d.items()), reverse=True)
    log("largest member forces (new / old):")
    for ax, k, nm in worst[:12]:
        o = old.get(k, {}).get(nm, {}).get("axial")
        log(f"  node {k:>2} {nm:<18} {ax:7.0f} N / {o if o is None else round(o):>6}   {node_loads[k][nm]['case']}")
    log("spoke cases: " + "; ".join(f"{nm}: {v['case'].split('face ')[-1]}" for k, dd in node_loads.items() for nm, v in dd.items() if nm.startswith("spoke")))
    sp = [(v["axial"], nm, k) for k, d in node_loads.items() for nm, v in d.items() if nm.startswith("spoke")]
    log("spokes: " + ", ".join(f"{nm} {ax:.0f}" for ax, nm, k in sorted(sp, key=lambda t: t[1])))
    bb = sorted(buck.items(), key=lambda kv: -kv[1]["ratio"])[:5]
    log("tube buckling (compression / Euler pinned): " + ", ".join(f"{k} {v['ratio']:.2f}" for k, v in bb))
    for k in (1, 2):
        rr = max(results, key=lambda r: np.hypot(r["arms"][k]["root"]["N"], r["arms"][k]["root"]["shear"]))
        log(f"arm {k} root: largest net force {np.hypot(rr['arms'][k]['root']['N'], rr['arms'][k]['root']['shear']):.0f} N "
            f"(axial {rr['arms'][k]['root']['N']:.0f}, shear {rr['arms'][k]['root']['shear']:.0f}, moment {rr['arms'][k]['root']['moment_Nmm'] / 1000:.1f} N m) "
            f"in {rr['name'].split('dir ')[-1]}; spokes there {rr['arms'][k]['spokes']}")
    worst_case = max(results, key=lambda r: max(abs(v["N"]) for v in r["members"].values()))
    log(f"heaviest case: {worst_case['name']}, touching {worst_case['touch']}, normal reactions {list(worst_case['reactions'].values())} N")
    record = dict(note=__doc__.strip().split("\n")[0], engine="tools/cage_loads.py",
                  model=dict(points=n, members=len(fr.members), sections=dict(tube=TUBE, rod=RODS, arm=ARM),
                             masses_g=dict(core=round(mg["core"][0], 1), core_cg=np.round(mg["core"][1], 1).tolist(),
                                           head=round(mg["head"][0], 1), head_cg=np.round(mg["head"][1], 1).tolist(),
                                           cage_nodes=round(sum(mg["nodes"].values()), 1), tubes=round(mg["tubes"], 1),
                                           rods=round(mg["rods"], 1), arms={k + 1: v for k, v in mg["arms"].items()},
                                           camera_rear=mg["cam21"], total=round(total, 1)),
                             crash_g=CRASH_G, landing_g=LAND_G),
                  cases=results, buckling=buck)
    if dry:
        return node_loads, record
    json.dump(record, open(OUT_PATH, "w"), indent=1)
    s = open(CAGE_PATH).read()
    i = s.index('"node_loads"'); j = s.index("{", i); depth = 0
    for k in range(j, len(s)):
        depth += s[k] == "{"; depth -= s[k] == "}"
        if depth == 0:
            break
    block = json.dumps(node_loads, indent=1).replace("\n", "\n ")
    s = s[:i] + '"node_loads": ' + block + s[k + 1:]
    json.loads(s); open(CAGE_PATH, "w").write(s)
    log(f"wrote node_loads for {len(node_loads)} nodes to {os.path.relpath(CAGE_PATH, ROOT)} and the record to "
        f"{os.path.relpath(OUT_PATH, ROOT)}")
    return node_loads, record


if __name__ == "__main__":
    run(dry="--dry" in sys.argv)
