"""Box swap arm v7 drawn parts (24 Oct 2026, swap arm system worker): rails A / B v3 = rail v2 (arm_v6_cad.rail_v2) on the v7 stack
(arm_v7_layout: r 29..151) with the carbon tie sockets (two Ø3.3 x 10 bores rising into the bottom face at r 128, arm_v6_layout.TIE), and
the tie rods (2 x 3.0 carbon per rail, from the hub's lower rim at Y 22.5, r 78). Pinion16 v2, detent ring v3, V-wheel v1 and bush v1 carry
over unchanged. Round 3 (arm_v7_layout.LOW): rail v4 with the keel; v5 widens the ribbon channel and sinks the bolt heads. python3 arm_v7_cad.py [checks] -> runs/arm_rail/v5/ (W and print-posed
meshes, checks.json; v4 goes to print/ only once READY); NOX_ARM_STACK=v6 draws v3 (print/box_arm_rail_{A,B}_v3.stl)"""
import os, sys, math, json
import numpy as np, trimesh
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import arm_v7_layout as S7
S7.use()
import arm_v6_cad as C6, arm_v5_cad as C5, arm_v5_layout as V
from arm_cad import tm
C6.TIES = True
ROOT = os.path.normpath(os.path.join(HERE, "..", "..")); VER = "v5" if getattr(V, "RAIL_KEEL", None) else "v3"   # v5: the wide ribbon channel, bolt heads under its floor
RUN = os.path.join(ROOT, "runs", "arm_rail", VER); os.makedirs(RUN, exist_ok=True)


def rail_v3(h="A"):
    """round 2 rail (the v6 carriage stack); round 3 (arm_v7_layout.LOW) draws rail_v4"""
    return rail_v4(h) if getattr(V, "RAIL_KEEL", None) else C6.rail_v2(h)


def rail_v4(h="A"):
    """rail v4 (round 3, the low carriage): rail v2's body on the low stack (top Y 66.75, no feet: the M5 x 12 heads in the body), the keel
    X +-9 x Y 38..54 from r 89 (off the hub plate's edge) to the tip, and the tie sockets rising into the keel's bottom face at r 128."""
    k = V.RAIL_KEEL
    TI, C6.TIES = C6.TIES, False
    m = C6.rail_v2(h)
    C6.TIES = TI
    m += C6.box(-k["x"], k["x"], k["y"][0], k["y"][1] + 0.01, k["r0"], V.RAIL_R[1])
    if C6.TIES:
        for p0, p1, d in C6.tie_geom(h):
            mouth = p1 - C6.T["depth"] * d
            m -= C6.cyl_between(mouth - 2.0 * d, p1 + 0.01 * d, C6.T["bore_d"] / 2, 32)
    return m


def pinion_v3(xc=0.0, zc=0.0):
    """16T m1 v3 (round 3d, concept review r3): v2's involute on the 13.5 face (teeth Y 59.75..73.25, fully on the rack grown down to Y 59.25;
    10.25 on the rack tail from its step at Y 63), boss Y 55..59.25 (Ø14) with the M3 grub on the round shaft at Y 57.1 for axial hold only:
    the D-bore on the NEMA's flat (Y 59..74) carries the torque."""
    import manifold3d as mf
    from arm_cad import cylY, box
    import arm_cad as A
    cs, g = C5.involute_gear_cs(V.PIN_T, V.PIN_M)
    t0, t1 = V.PIN_TEETH_Y; b0, b1 = V.PIN_BOSS_Y
    ef = 0.3
    gear = mf.Manifold.extrude(cs, t1 - t0 - ef).rotate([-90, 0, 0]).translate([0, t0, 0])
    gear += mf.Manifold.extrude(cs.offset(-0.3, mf.JoinType.Round), ef + 0.01).rotate([-90, 0, 0]).translate([0, t1 - ef - 0.01, 0])
    p = gear + cylY(0, 0, b0, t0 + 0.01, V.PIN_BOSS_R, 96)                      # the boss runs up into the teeth (one body)
    r = C5.SHAFT_D / 2 + C5.BORE_CLR / 2; fd = C5.SHAFT_FLAT - C5.SHAFT_D / 2 + C5.BORE_CLR / 2
    yf = 59.0 + 0.5                                                          # the D-flat from Y 59 (dimensions M), +0.5
    p -= cylY(0, 0, b0 - 1, t1 + 1, r, 48) - (cylY(0, 0, yf, t1 + 2, r + 1, 48) - box(-fd, r + 2, yf - 1, t1 + 3, -r - 2, r + 2))
    yg = 0.5 * (b0 + b1)
    p -= A.cylX(yg, 0, -V.PIN_BOSS_R - 1, 0, C5.GRUB_TAP_D / 2, 24)             # M3 grub from -X onto the round shaft (axial hold)
    return p.translate([xc, 0, zc]), g


def pinion_v3_main(checks=True):
    from solid_check import solid_check
    run = os.path.join(ROOT, "runs", "arm_pinion16", "v3"); os.makedirs(run, exist_ok=True)
    m = pinion_v3()[0]; tm(m).export(os.path.join(run, "arm_pinion16_v3_W.stl"))
    t = C5.pose_axisY(m, top_down=True); path = os.path.join(run, "box_arm_pinion16_v3.stl"); t.export(path)
    r = dict(file=os.path.relpath(path, ROOT), mass_g=round(t.volume / 1000 * 1.27, 2), extents=np.round(t.extents, 1).tolist(),
             face=round(V.PIN_TEETH_Y[1] - V.PIN_TEETH_Y[0], 2), boss=V.PIN_BOSS_Y)
    if checks:
        ok, info = solid_check(trimesh.load(path, force="mesh"), pitch=0.1)
        r["solid_check"] = "PASS" if ok else "FAIL"; r["solid_info"] = {k: info[k] for k in ("bodies_eroded", "necks", "thin_walls") if k in info}
    json.dump(r, open(os.path.join(run, "checks.json"), "w"), indent=1, default=str); print(r)
    return r


def tie_rods(h="A"):
    """the two rods of rail h in head A's frame (B is mirrored with its rail)"""
    m = None
    for p0, p1, d in C6.tie_geom(h):
        c = C6.cyl_between(p0, p1, C6.T["rod_d"] / 2, 24); m = c if m is None else m + c
    return m


def main(checks=False):
    from solid_check import solid_check
    res = {}
    for h in ("A", "B"):
        m = rail_v3(h); tm(m).export(os.path.join(RUN, f"arm_rail_{h}_{VER}_W.stl"))
        t = C5.pose_rail(m); path = os.path.join(ROOT, "print", f"box_arm_rail_{h}_{VER}.stl") if VER == "v3" else os.path.join(RUN, f"box_arm_rail_{h}_{VER}.stl")
        t.export(path)                                                          # v4 is DESIGN until the concept check passes: runs/, not print/
        r = dict(file=os.path.relpath(path, ROOT), mass_g=round(t.volume / 1000 * 1.27, 2), extents=np.round(t.extents, 1).tolist(), bodies=len(m.decompose()))
        if checks:
            ok, info = solid_check(trimesh.load(path, force="mesh"), pitch=0.25)
            r["solid_check"] = "PASS" if ok else "FAIL"; r["solid_info"] = {k: info[k] for k in ("bodies_eroded", "necks", "thin_walls") if k in info}
        rods = tie_rods(h)
        r["rods_in_rail_mm3"] = round((m ^ rods).volume(), 2); r["rod_len"] = round(float(np.linalg.norm(np.subtract(*C6.tie_geom(h)[0][:2]))), 1)
        res[f"rail_{h}_{VER}"] = r; print(h, r)
    json.dump(res, open(os.path.join(RUN, "checks.json"), "w"), indent=1, default=str)
    return res


if __name__ == "__main__":
    if "pinion" in sys.argv: pinion_v3_main()
    else: main("checks" in sys.argv)
