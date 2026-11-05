"""Scout cage frame model (4 Nov, Spark: "lets do a stiffness check on the design", PETG nodes): the cage of tools/scout_cage.py
as a 3D beam frame on the Overseer's engine (tools/cage_loads.py: Euler beams, rigid joints, 35 g crash onto the hull face the
CG ray leaves through, a pulled contact node lifts off).

Model: every cage rod and tie a 3.0 solid carbon beam (E 120 GPa); the plate with everything on and under it (motors, pack,
units, electronics) one rigid body at its CG, hung on the tie roots and the printed nodes NB / TB (in the nose and tail units,
scout_cage.PRINTED) by stiff links; masses from systems/scout.assembly.json
(cage nodes 5 g each, rods and ties 0.011 g/mm, half to each end).
Joints: a 3.0 rod in a 10 deep PETG socket turns against the socket wall: k_theta ~ E_petg L^3 / 12 ~ 230 N m/rad (Winkler, E),
5..12x the rod's own 4EI/L (19 N m/rad at 100 mm, 42 at 45), so rigid joints read the frame right to the rods' compliance; what
the PETG limits is strength: pull-out ~940 N (epoxy, pi 3 x 10 at 10 MPa, E), side bearing ~1200 N (40 MPa over 3 x 10, E),
end moment ~2.0 N m (40 MPa over d L^2 / 6, E); the rod itself bends to ~2.6 N m (1 GPa over pi r^3 / 4, E).
Cases: 35 g crash in 14 directions (6 axes, 8 diagonals; one case per hull face), landing 6 g flat on the four skid ends, rough
landing 6 g rolled 20 deg about x and pitched 20 deg about y (onto a skid edge, a nose or tail edge).
Reads: deflection (largest cage node displacement per case), rod axial force and compression / Euler pinned-end buckling load,
end shears and moments against the socket, free-free modes (lumped masses; the first elastic one is the cage's stiffness
number), and the pin-jointed count: a closed shell of n nodes needs 3n - 6 rods to stand with pins alone; every rod short of
that is one hinge the PETG sockets must hold (Spark 4 Nov: "not triangulated enough").
Run: python tools/scout_cage_loads.py [--dry] [--add A-B,C-D] [--drop A-B]   (what-ifs on the rod list, dry only)
Writes systems/scout.cage_loads.json (the record) and prints the report.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cage_loads as cl                                      # noqa: E402
import scout_cage as sc                                      # noqa: E402

OUT_PATH = os.path.join(ROOT, "systems", "scout.cage_loads.json")
ASM_PATH = os.path.join(ROOT, "systems", "scout.assembly.json")
ROD_G_MM, NODE_G = 0.011, 5.0
CAP = dict(pull_out=940.0, side=1200.0, moment_socket=2000.0, moment_rod=2650.0)    # N, N, N mm, N mm (E, docstring)
ROUGH_G, ROUGH_DEG = 6.0, 20.0


def core_mass():
    """the plate and everything on or under it: mass and CG from the assembly (every part but the cage)"""
    import assembly as A, system_check as sck
    asm = A.load(ASM_PATH)
    asm.data["parts"] = [p for p in asm.parts if not p["id"].startswith("scout_cage")]   # the cage (rods, nodes: boxes until their runs land) is not the core
    rows = sck.mass_cg(asm.parts, asm.meshes())["parts"]
    by = {p["name"]: p["id"] for p in asm.parts}
    m, s = 0.0, np.zeros(3)
    for r in rows:
        if by[r["name"]].startswith("scout_cage"):
            continue
        m += r["mass_g"]; s += r["mass_g"] * np.asarray(r["cg"], float)
    return m, s / m


def build(nodes, rods, ties, core):
    fr = cl.Frame(); keys = list(nodes)
    for k in keys:
        fr.node(nodes[k], tol=1e-6)
    P = {**nodes, **sc.TIE_ROOTS}
    for k in sc.TIE_ROOTS:
        fr.node(sc.TIE_ROOTS[k], tol=1e-6)
    pid = {k: fr.node(P[k], tol=1e-6) for k in P}
    load = {i: 0.0 for i in range(len(fr.pts))}
    for k in keys:
        load[pid[k]] += NODE_G
    for i, j in rods:
        L = float(np.linalg.norm(np.array(P[i]) - np.array(P[j])))
        fr.member(f"{i}-{j}", pid[i], pid[j], cl.RODS, "rod"); load[pid[i]] += ROD_G_MM * L / 2; load[pid[j]] += ROD_G_MM * L / 2
    for i, j in ties:
        L = float(np.linalg.norm(np.array(P[i]) - np.array(P[j])))
        fr.member(f"{i}-{j}", pid[i], pid[j], cl.RODS, "tie"); load[pid[i]] += ROD_G_MM * L / 2; load[pid[j]] += ROD_G_MM * L / 2
    cg_m, cg_c = core; ic = fr.node(cg_c, tol=1e-6); load[ic] = load.get(ic, 0.0) + cg_m
    for k in list(sc.TIE_ROOTS) + [k for k in sc.PRINTED if k in nodes]:      # the tie roots and the printed nodes ride on the plate's units
        fr.member(f"core {k}", ic, pid[k], cl.STIFF, "stiff")
    return fr, load, pid, keys


def mechanisms(nodes, rods):
    """pin-jointed: rank of the equilibrium matrix of the rod network; a rigid body of n nodes needs rank 3n - 6"""
    keys = list(nodes); idx = {k: n for n, k in enumerate(keys)}; N = np.array([nodes[k] for k in keys], float)
    B = np.zeros((3 * len(keys), len(rods)))
    for c, (i, j) in enumerate(rods):
        u = N[idx[j]] - N[idx[i]]; u /= np.linalg.norm(u)
        B[3 * idx[i]:3 * idx[i] + 3, c] = -u; B[3 * idx[j]:3 * idx[j] + 3, c] = u
    rank = int(np.linalg.matrix_rank(B, tol=1e-6))
    return 3 * len(keys) - 6 - rank, 3 * len(keys) - 6, len(rods)


def modes(fr, load, k=3):
    """free-free: the first k elastic frequencies (Hz), lumped masses (g) and a small rotary inertia (m r^2, r 5)"""
    from scipy.linalg import eigh
    K, _ = fr.assemble(); n = len(fr.pts); M = np.zeros(6 * n)
    for i in range(n):
        m = max(load.get(i, 0.0), 1e-3); M[6 * i:6 * i + 3] = m; M[6 * i + 3:6 * i + 6] = m * 25.0
    w = eigh(K, np.diag(M), eigvals_only=True)
    w = np.sqrt(np.clip(w, 0, None)) * 1000.0 / (2 * np.pi)      # K N/mm, m g -> rad/s x 1e3
    return [round(float(x), 1) for x in w[6:6 + k]]


def rough_dirs():
    t = np.radians(ROUGH_DEG)
    return [np.array([0, s * np.sin(t), -np.cos(t)]) for s in (1, -1)] + [np.array([s * np.sin(t), 0, -np.cos(t)]) for s in (1, -1)]


def run(nodes=None, rods=None, ties=None, dry=False, log=print):
    cl.selftest()
    nodes = nodes or sc.NODES; rods = rods or sc.RODS; ties = ties or sc.TIES
    core = core_mass()
    fr, load, pid, keys = build(nodes, rods, ties, core)
    n = len(fr.pts); total = sum(load.values())
    N = np.array([nodes[k] for k in keys], float)
    cg = sum(g * fr.pts[i] for i, g in load.items()) / total
    mech, need, have = mechanisms(nodes, rods)
    log(f"frame: {n} points, {len(rods)} rods + {len(ties)} ties 3.0, core {core[0]:.0f} g at {np.round(core[1], 1).tolist()}, "
        f"cage {total - core[0]:.0f} g, all {total:.0f} g at {np.round(cg, 1).tolist()}")
    log(f"pin-jointed: {have} rods of the {need} a closed shell of {len(keys)} nodes needs: {mech} hinge(s) the PETG sockets must hold")
    fq = modes(fr, load)
    log(f"free-free modes: {fq} Hz (the first is the cage's stiffness number)")
    cases, seen = [], {}
    for d in cl.crash_dirs():
        touch, nrm = cl.contact(N, d, cg); key = tuple(sorted(touch))
        if key in seen:
            continue
        seen[key] = True
        cases.append(dict(name=f"crash 35 g dir ({d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}) on {'-'.join(keys[i] for i in touch)}", d=d, n=nrm, g=cl.CRASH_G, touch=touch))
    kk = [keys.index(k) for k in ("K0", "K1", "K2", "K3")]
    cases.append(dict(name="landing 6 g flat on the skids", d=np.array([0, 0, -1.0]), n=np.array([0, 0, -1.0]), g=cl.LAND_G, touch=kk))
    for d in rough_dirs():
        touch, nrm = cl.contact(N, d, cg)
        cases.append(dict(name=f"rough landing {ROUGH_G:.0f} g dir ({d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}) on {'-'.join(keys[i] for i in touch)}", d=d, n=nrm, g=ROUGH_G, touch=touch))
    results = []
    for c in cases:
        F = np.zeros((n, 3))
        for i, g in load.items():
            F[i] = g * cl.G * c["g"] * c["d"]
        touch = list(c["touch"])
        while True:
            cons = cl.supports(N, touch, c["n"])
            u, mem, R = fr.solve(F, cons)
            reac = {}
            for (i, d), r in zip(cons, R):
                reac[int(i)] = round(float(r), 1) if abs(float(d @ c["n"])) > 0.999 else reac.get(int(i))
            lift = max(reac, key=reac.get)
            if reac[lift] <= 0.0 or len(touch) <= 3:
                break
            touch.remove(lift)
        c["touch"] = touch
        want = -total * cl.G * c["g"] * float(c["d"] @ c["n"]); got = sum(reac.values())
        if abs(got - want) > 0.01 * abs(want) or max(reac.values()) > 0.02 * abs(want):
            raise RuntimeError(f"{c['name']}: contact reactions {reac} should sum to {want:.0f} N and all push")
        disp = {k: float(np.linalg.norm(u[pid[k]][:3])) for k in keys}
        # displacement relative to the touching nodes' rigid motion is what bends the cage: take the largest over the cage less the smallest
        rel = max(disp.values()) - min(disp.values())
        results.append(dict(name=c["name"], d=np.round(c["d"], 4).tolist(), g=c["g"], touch=[keys[i] for i in touch], reactions={keys[i]: v for i, v in reac.items()},
                            deflection_mm=round(rel, 2), worst_node=max(disp, key=disp.get),
                            members={m["name"]: dict(N=round(r["N"], 1), shear_i=round(r["shear_i"], 1), shear_j=round(r["shear_j"], 1),
                                                     M_i=round(r["M_i"], 0), M_j=round(r["M_j"], 0)) for m, r in zip(fr.members, mem) if m["kind"] != "stiff"}))
    # per rod: Euler, extremes over the cases
    P = {**nodes, **sc.TIE_ROOTS}
    rodrep = {}
    for m in fr.members:
        if m["kind"] == "stiff":
            continue
        L = float(np.linalg.norm(fr.pts[m["i"]] - fr.pts[m["j"]])); Pcr = np.pi ** 2 * cl.RODS["E"] * cl.RODS["I"] / L ** 2
        comp = max(0.0, -min(r["members"][m["name"]]["N"] for r in results)); tens = max(0.0, max(r["members"][m["name"]]["N"] for r in results))
        side = max(max(r["members"][m["name"]]["shear_i"], r["members"][m["name"]]["shear_j"]) for r in results)
        mom = max(max(r["members"][m["name"]]["M_i"], r["members"][m["name"]]["M_j"]) for r in results)
        wc = max(results, key=lambda r: abs(r["members"][m["name"]]["N"]))["name"]
        rodrep[m["name"]] = dict(kind=m["kind"], L=round(L, 1), Pcr=round(Pcr, 1), compression=round(comp, 1), euler=round(comp / Pcr, 2), tension=round(tens, 1),
                                 side=round(side, 1), moment=round(mom, 0), case=wc)
    over = sorted(((v["euler"], k) for k, v in rodrep.items()), reverse=True)
    pull = sorted(((v["tension"], k) for k, v in rodrep.items()), reverse=True)
    sides = sorted(((v["side"], k) for k, v in rodrep.items()), reverse=True)
    moms = sorted(((v["moment"], k) for k, v in rodrep.items()), reverse=True)
    log("cases (deflection = largest cage node motion less the smallest, mm):")
    for r in results:
        big = max(r["members"].items(), key=lambda kv: abs(kv[1]["N"]))
        log(f"  {r['deflection_mm']:6.2f}  {big[0]:<7} {big[1]['N']:7.0f} N   {r['name']}")
    log(f"buckling (compression / Euler pinned): {sum(e > 1 for e, _ in over)} rod(s) over 1, {sum(e > 0.5 for e, _ in over)} over 0.5; worst "
        + ", ".join(f"{k} {e:.2f} ({rodrep[k]['compression']:.0f} N, L {rodrep[k]['L']:.0f})" for e, k in over[:6]))
    log(f"socket pull-out (cap {CAP['pull_out']:.0f} N E): worst " + ", ".join(f"{k} {t:.0f}" for t, k in pull[:4]))
    log(f"socket side (cap {CAP['side']:.0f} N E): worst " + ", ".join(f"{k} {t:.0f}" for t, k in sides[:4]))
    log(f"end moment (socket cap {CAP['moment_socket']:.0f}, rod {CAP['moment_rod']:.0f} N mm E): worst " + ", ".join(f"{k} {t:.0f}" for t, k in moms[:4]))
    # node_loads: per node (cage nodes and tie roots), per member ending there: largest |N| (sign kept in N) and side force, as the Overseer's
    node_loads = {}
    for m in fr.members:
        if m["kind"] not in ("rod", "tie"):
            continue
        a_, b_ = m["name"].split("-")
        for end, k in (("i", a_), ("j", b_)):
            best = max(results, key=lambda r: abs(r["members"][m["name"]]["N"]))
            side = max(r["members"][m["name"]][f"shear_{end}"] for r in results)
            Nmax = best["members"][m["name"]]["N"]
            node_loads.setdefault(k, {})[m["name"]] = dict(axial=round(abs(Nmax), 1), side=round(side, 1), N=round(Nmax, 1), case=best["name"], tag="F")
    worst_case = max(results, key=lambda r: r["deflection_mm"])
    log(f"softest case: {worst_case['name']}: {worst_case['deflection_mm']} mm at {worst_case['worst_node']}")
    record = dict(note=__doc__.strip().split("\n")[0], engine="tools/scout_cage_loads.py",
                  model=dict(points=n, rods=len(rods), ties=len(ties), section=cl.RODS, core_g=round(core[0], 1), core_cg=np.round(core[1], 1).tolist(),
                             total_g=round(total, 1), cg=np.round(cg, 1).tolist(), crash_g=cl.CRASH_G, landing_g=cl.LAND_G, rough_g=ROUGH_G, rough_deg=ROUGH_DEG,
                             capacities=CAP),
                  pin_jointed=dict(rods=have, needed=need, hinges=mech), modes_hz=fq, cases=results, rods=rodrep, node_loads=node_loads,
                  printed_nodes=sc.PRINTED, tie_roots={k: list(map(float, v)) for k, v in sc.TIE_ROOTS.items()})
    summary = dict(hinges=mech, modes=fq, euler_over_1=sum(e > 1 for e, _ in over), euler_max=over[0], deflection_max=(worst_case["deflection_mm"], worst_case["name"]),
                   pull_max=pull[0], side_max=sides[0], moment_max=moms[0])
    if not dry:
        json.dump(record, open(OUT_PATH, "w"), indent=1)
        log(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    return summary, record


if __name__ == "__main__":
    rods = list(sc.RODS)
    for a in sys.argv:
        if a.startswith("--add="):
            rods += [tuple(x.split("-")) for x in a[6:].split(",")]
        if a.startswith("--drop="):
            rods = [r for r in rods if "-".join(r) not in a[7:].split(",") and "-".join(r[::-1]) not in a[7:].split(",")]
    run(rods=rods, dry="--dry" in sys.argv or rods != list(sc.RODS))
