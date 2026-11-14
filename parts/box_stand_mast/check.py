#!/usr/bin/env python3
"""Box stand strength with the arm v8 on it (14 Nov, design-chat worker): hand beam checks on the stand's plain sections.

Run:  python3 parts/box_stand_mast/check.py      (prints the table; writes parts/box_stand_mast/check.json)

Loads (from the system, W frame, hub at the origin, floor at W Z -216):
  * the arm v8 and its 11 bays with packs: every piece of systems/swap_arm.assembly.json but the drone seat and the
    funnel; printed pieces at their run mass (hub, carriages, lifts) or mesh volume x 1.23 (PCTG); receivers at their
    runs (latch7_ov_receiver 144.9 g, latch7_receiver 87.3 g); packs at the bay_mount weights (Overseer 7.5 N, Scout
    2.2 N, parts/box_arm8/result.json); bought parts at catalogue masses (E, the table B below).
  * the ring's unbalance as it turns (the turning mass's CG 25.9 off the axle): its X offset at the worst angle.
  * the head's push at the drone: 106.6 N every swap (flight), 118 N limit (misuse), at W Y 192.5, Z 176 (the arm's own
    cases, parts/box_arm8_lib.py): the drone's receiver pushes the arm down, the funnel (on the sheets) takes it up.
  * carrying an Overseer at full reach (flight), a dropped Overseer onto the head (150 N, misuse), the axle stall
    (5.53 N m about Y, misuse), box handling 3 g down + 2 g side (misuse).
  * what the sawn-off cradle held: the -90 demo bay module of the v1 stand (bay v4i, its XT60 pull). In v8 every bay is
    on the arm's ring: that load now reaches the stand through the stub, inside the arm's total above.
Supports: the floor only (the stand is free-standing; the v2 sheets are a layout, nothing bolts the stand to them).
Indices: topo/material.py PCTG allowables by class (KNOCKDOWN 0.85, SF flight 4, misuse 2.5, sustained 10); the gate is
the engine's max_worst 0.5 in every class.
Method: hand beam checks (the mast is a 40 x 60 tube prism; the keel and the toe are plain bars); sections from the
meshes (box_stand_* part.stl, 0.4 mm raster). The keel's moment is a LOWER bound: with the arm's weight entering at
the mast (Y -60..0) and the floor reaction's resultant forward of the bars (Y 62), the keel must carry at least
V (y_res - y) at every section y behind y_res, however the floor bears (flat floor: the best case). The uneven floor
(the bars' ends and the toe's front plate only) is the upper case.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
G = 9.81e-3                     # N per g

# material (topo/material.py PCTG)
XT, ZT, S, KD = 46.0, 23.0, 23.0, 0.85
SF = {"flight": 4.0, "misuse": 2.5, "sustained": 10.0}
def allow(cls): return XT * KD / SF[cls], ZT * KD / SF[cls], S * KD / SF[cls]

# the arm v8 with 11 bays and packs (mass.py in the worker's scratch, from systems/swap_arm.assembly.json): g, CG
ARM_G, ARM_CG = 9717.0, np.array([-2.1, 129.0, -21.4])
TURN_G, TURN_R = 8688.0, 25.9
STAND_G, STAND_CG = 462.0, np.array([0.0, 95.0, -180.0])        # the six pieces, mesh volume x 1.23 (E for PCTG)

# sections (mm2, mm3), from the meshes (W, posed): mast tube at the kb's top (W Z -67), uniform -143..-49
MAST = dict(A=384.0, Wx=6438.0, Wy=5114.0, yc=-30.0, z=-67.0, t=1.95, Am=38.0 * 58.0)
KB = dict(Wx=4396.0, Wy=3394.0, A=298.0, yc=-30.5, z=-143.0)    # kb's tenon inside the mast (W Z -150..-67): the mast's moment at its foot
KEEL = {90.0: 546.0, 110.0: 546.0, 130.0: 579.0, 150.0: 477.0, 175.0: 139.0}      # W about X (bending in the YZ plane)
TOE = {230.0: 333.0, 270.0: 333.0, 320.0: 350.0}
Y_BAR, Y_TOE_FRONT = 54.0, 335.0
X_BAR_END, X_KB = 186.0, 36.0
BAR_W = 2189.0                  # bar half v8 about the bending axis Y (fibre in Z), at X 100; Wz 1209 at the kb joint side

def cases():
    """(name, cls, [(F, at)], extra moment about the hub (N mm))"""
    Wd = (np.array([0, 0, -ARM_G * G]), ARM_CG)
    unb = np.array([0, TURN_G * G * TURN_R, 0]) * 0      # the unbalance as an X shift of the turning mass below
    sh = np.array([TURN_R * TURN_G / ARM_G, 0, 0])
    D = (Wd[0], ARM_CG + sh)
    st = (np.array([0, 0, -STAND_G * G]), STAND_CG)
    out = [("standing: arm v8, 11 bays with packs, the ring turned to its worst unbalance", "sustained", [D, st], 0.0),
           ("insert at the drone, 106.6 N every swap (+ standing)", "flight", [D, st, (np.array([0, 0, -106.6]), np.array([0, 192.5, 176.0]))], 0.0),
           ("carrying an Overseer at full reach (+ standing)", "flight", [D, st, (np.array([0, 0, -10.0]), np.array([0, 167.1, 176.0]))], 0.0),
           ("insert at the float limit, 118 N (+ standing)", "misuse", [D, st, (np.array([0, 0, -118.0]), np.array([0, 192.5, 176.0]))], 0.0),
           ("dropped Overseer onto the head at the drone, 150 N (+ standing)", "misuse", [D, st, (np.array([0, 0, -150.0]), np.array([0, 167.1, 176.0]))], 0.0),
           ("axle stall 5.53 N m about Y, head held at the drone (+ standing)", "misuse", [D, st], 5530.0),
           ("box handling 3 g down + 2 g side", "misuse", [(3 * D[0], D[1]), (np.array([2 * ARM_G * G, 0, 0]), D[1]),
                                                     (3 * st[0], st[1]), (np.array([2 * STAND_G * G, 0, 0]), st[1])], 0.0)]
    return out

def mast(loads, My_extra, cls):
    """stresses in the mast tube at the kb's top from the loads above the section (the arm's; the mast's own weight is small)"""
    o = np.array([0.0, MAST["yc"], MAST["z"]])
    F = np.zeros(3); M = np.zeros(3)
    for f, at in loads:
        if at[2] < MAST["z"]:
            continue                        # the stand's own weight acts below
        F += f; M += np.cross(at - o, f)
    M[1] += My_extra
    sig = abs(F[2]) / MAST["A"] + abs(M[0]) / MAST["Wx"] + abs(M[1]) / MAST["Wy"]     # the corner, all three added
    tau = abs(M[2]) / (2 * MAST["Am"] * MAST["t"]) + np.hypot(F[0], F[1]) / (2 * 40 * MAST["t"])
    Axy, Az, As = allow(cls)
    # printed on its side (layer normal X): the bending stress runs in the layers; the walls at Y 0 / -60 take the shear across them
    return dict(M_Nm=np.round(M / 1000, 2).tolist(), sigma=round(sig, 2), tau=round(tau, 2),
                inlayer=round(np.sqrt(sig ** 2 + 3 * tau ** 2) / Axy, 3), across=round(tau / As, 3))

def kb(loads, My_extra, cls):
    o = np.array([0.0, KB["yc"], KB["z"]]); F = np.zeros(3); M = np.zeros(3)
    for f, at in loads:
        if at[2] < MAST["z"]:
            continue
        F += f; M += np.cross(at - o, f)
    M[1] += My_extra
    sig = abs(F[2]) / KB["A"] + abs(M[0]) / KB["Wx"] + abs(M[1]) / KB["Wy"]
    return round(sig / allow(cls)[0], 3)     # printed upright (W Z the build axis: the tenon's bending stress runs ACROSS the layers: shown in-layer, across is twice this)

def floor(loads, My_extra):
    V = -sum(f[2] for f, _ in loads)
    yres = sum(-f[2] * at[1] + f[1] * at[2] for f, at in loads) / V      # moment about X of the loads, floor at Z ~ 0 offset ignored for Fy (none)
    xres = sum(-f[2] * at[0] - f[0] * (at[2] + 216.0) for f, at in loads) / V - My_extra / V
    return V, yres, xres

def keel(V, yres, cls):
    Axy = allow(cls)[0]
    flat = {y: max(0.0, V * (yres - y)) / W / Axy for y, W in {**KEEL, **TOE}.items()}
    Rf = V * (yres - Y_BAR) / (Y_TOE_FRONT - Y_BAR)
    three = {y: Rf * (Y_TOE_FRONT - y) / W / Axy for y, W in {**KEEL, **TOE}.items()}
    return Rf, flat, three

def main():
    rows = []
    for name, cls, loads, Mx in cases():
        V, yres, xres = floor(loads, Mx)
        m = mast(loads, Mx, cls)
        Rf, flat, three = keel(V, yres, cls)
        # bars: uneven floor, one bar end and the toe's front take the roll: the bar end's reaction about the kb joint
        Rb = V * abs(xres) / X_BAR_END + V * 0.5 * (1 - (yres - Y_BAR) / (Y_TOE_FRONT - Y_BAR))
        bar = Rb * (X_BAR_END - X_KB) / BAR_W / allow(cls)[0]
        rows.append(dict(case=name, cls=cls, V_N=round(V, 1), y_res=round(yres, 1), x_res=round(xres, 1), mast=m, kb_tenon_inlayer=kb(loads, Mx, cls),
                         keel_flat_floor_lower_bound={str(int(k)): round(v, 2) for k, v in flat.items()},
                         keel_uneven_floor={str(int(k)): round(v, 2) for k, v in three.items()}, toe_front_N=round(Rf, 1),
                         bar_uneven_floor=round(bar, 2)))
    for r in rows:
        kf = max(r["keel_flat_floor_lower_bound"].items(), key=lambda kv: kv[1])
        ku = max(r["keel_uneven_floor"].items(), key=lambda kv: kv[1])
        print(f"{r['case'][:62]:62s} {r['cls']:9s} V {r['V_N']:6.1f} N y_res {r['y_res']:6.1f} | mast {max(r['mast']['inlayer'], r['mast']['across']):.2f}"
              f" (M {r['mast']['M_Nm']}) | kb {r['kb_tenon_inlayer']:.2f} | keel >= {kf[1]:.2f} at Y {kf[0]}, uneven {ku[1]:.2f} at Y {ku[0]} | bar {r['bar_uneven_floor']:.2f}")
    json.dump(dict(note=__doc__.split("\n\n")[0], arm_g=ARM_G, arm_cg=ARM_CG.tolist(), rows=rows), open(os.path.join(HERE, "check.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
