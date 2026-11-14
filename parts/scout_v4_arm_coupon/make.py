"""Scout v4 arm coupon (14 Nov, the design chat for Spark: test the arm before trusting the frame's 0.20 motor index): motor 1's pod and its
arm out to the deck, cut from the frame's own generated print mesh (scout_v4_frame's print_stl), the same print orientation (deck top
down: pad and arm in the same layers as on the frame), the same pad and its 4 M2 holes, fused to a printed clamp block for a vise; a
dummy puck on the 4 M2 screws in the motor's place, with a cord eye at the motor's height.
    python parts/scout_v4_arm_coupon/make.py [frame.stl]   writes print/scout_v4_arm_coupon.stl and _puck.stl (laid flat for the A1 mini),
                                                           part.stl (off the drone, for the page) and prints the engine's break loads
The test (one paragraph, on the page and in dimensions.md T `arm_up` / `arm_side`): grip the block in a vise, arm level, pad up. Bolt the
puck on with 4 M2 x 20 from under the pad, nuts in the puck's pockets. Tie a cord loop through the puck's eye and hang a luggage scale (or
a bucket filled with water, weighed after) from it: pull straight UP slowly until it breaks and note the peak load and where it broke;
print a second coupon and pull SIDEWAYS (level, across the arm) the same way. Report: up N, where; sideways N, where.
Predicted break = the load at which the engine's as-built FE stress reaches the material strength (topo.build.strength on the coupon's
mesh, misuse class: index x 0.85 / 2.5 = 1), the block's vise faces fixed, the pull split over the 4 screws as the frame's motor cases
(the up pull on the screw heads' seats under the pad, the rest on the motor's base round each screw)."""
import json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np, trimesh, manifold3d as mf
from topo import Part
from topo.part import man, tm
from topo.layout import Layout
from topo.build import strength
from topo.material import SAFETY, KNOCKDOWN
from parts.scout_units_lib import box, cyl, rigid_split
import scout_v4_shape as S
import print_mesh

NAME = "scout_v4_arm_coupon"
MI = 0                                         # motor 1 (the frame's worst motor case, 0.20 in-layer at its arm)
DZ1, POD_T, M2 = S.DECK_TOP, 8.0, 6.0          # the pad's top (z 24.5), its depth, the 12 x 12 M2 pattern (parts/scout_v4_frame/part.py)
ARM_Z0 = DZ1 - 16.0                            # the arm's room bottom
S0 = 56.0                                      # the cut, along the arm from the drone's centre: where the arm leaves the deck (y -40)
BLK, W, BLK_Z0 = 20.0, 27.0, DZ1 - 30.0        # the clamp block: 20 long behind the cut, +-27 across (the arm's room is +-24 at its root),
                                               # 30 deep from the deck top (the arm's 16 + 14 under it: stiff, so the arm breaks first)
PUCK_R, PUCK_H, EYE_Z, EYE_R = 13.5, 12.0, 8.0, 2.2   # the dummy puck: O27 x 12, the cord eye O4.4 8 above the pad (the motor's bell)
F_TEST = 100.0                                 # N, the FE's test load (linear: the break load scales from it)


def frame_mesh():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return print_mesh.local("scout_v4_frame")


def local():
    """the drone frame -> the arm's frame: motor MI on +x, the arm along x, z unchanged (a turn about z keeps the layers)"""
    mx, my = S.MOTORS[MI]; a = -math.atan2(my, mx)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return R, float(np.hypot(mx, my))


def build(path):
    R, L = local()
    fr = trimesh.load(path, force="mesh"); fr.apply_transform(R)
    keep = box(S0 - 0.5, L + 30, -W, W, ARM_Z0 - 2, DZ1 + 0.01)                     # the arm and pod, within the arm's room across
    arm = man(fr) ^ keep
    block = box(S0 - BLK, S0 + 1.0, -W, W, BLK_Z0, DZ1)                              # fused 1.0 into the cut face
    body = arm + block
    parts = sorted(tm(body).split(only_watertight=True), key=lambda q: -q.volume)
    cp = parts[0]                                                                    # stubs of deck members the corridor cut loose
    dropped = sum(q.volume for q in parts[1:]) / 1000
    scr = [(L + dx, dy) for dx in (-M2 * math.sqrt(2), 0, M2 * math.sqrt(2)) for dy in (-M2 * math.sqrt(2), 0, M2 * math.sqrt(2))
           if abs(dx) + abs(dy) == M2 * math.sqrt(2)]                              # the 12 x 12 pattern turned 45 deg with the arm
    puck = cyl(L, 0, DZ1, DZ1 + PUCK_H, PUCK_R, 64)
    cuts = [cyl(x, y, DZ1 - 1, DZ1 + PUCK_H + 1, 1.2, 24) for x, y in scr]           # M2 clearance
    hexp = [man(trimesh.creation.cylinder(radius=4.4 / math.sqrt(3), height=2.6, sections=6)).translate([x, y, DZ1 + PUCK_H - 1.3])
            for x, y in scr]                                                         # M2 nut pockets 4.4 across flats, 2.5 deep
    eye = man(trimesh.creation.cylinder(radius=EYE_R, segment=[(L - 20, 0, DZ1 + EYE_Z), (L + 20, 0, DZ1 + EYE_Z)], sections=32))
    puck = puck - mf.Manifold.batch_boolean(cuts + hexp + [eye], mf.OpType.Add)      # eye along the arm: the cord pulls up or across
    return cp, tm(puck), block, scr, L, dropped


def fe(cp, block, scr, L, h):
    """the engine's as-built FE of the coupon: index of a F_TEST pull up and across, misuse class"""
    m = man(cp)
    room = box(S0 - BLK - 3, S0 - BLK, -2, 2, BLK_Z0, BLK_Z0 + 3)                   # a design room behind the block (Layout wants one; not built)
    p = Part(NAME, h, m, box(-500, -499, 0, 1, 0, 1), m + room, build="-z", material="PCTG", log=lambda s_: None)
    p.support("default", box(S0 - BLK - 1, S0 - BLK + 12, -W - 1, W + 1, BLK_Z0 - 1, DZ1 + 1))   # the block's back 12 held in the vise
    bot = [cyl(x, y, DZ1 - POD_T, DZ1 - POD_T + 1.5, 2.6) for x, y in scr]           # each screw head's seat under the pad
    top = [cyl(x, y, DZ1 - 1.5, DZ1, 3.5) for x, y in scr]                           # the puck's base on the pad round each screw
    at = np.array([L, 0, DZ1 + EYE_Z])
    pts = [(x, y, DZ1 - POD_T / 2) for x, y in scr]
    for nm, F in (("up", (0, 0, F_TEST)), ("side", (0, F_TEST, 0))):
        f = rigid_split(pts, np.asarray(F, float), at)
        out = []
        for i in range(4):
            up = max(0.0, f[i][2])
            out += [(bot[i], np.array([0.0, 0.0, up])), (top[i], f[i] - np.array([0.0, 0.0, up]))]
        p.load_multi(nm, out, cls="misuse")
    lay = Layout(p, volfrac=0.5, log=lambda s_: None)
    r = strength(lay, cp, log=lambda s_: None)
    k = KNOCKDOWN / SAFETY["misuse"]
    res = {}
    for row in r["rows"]:
        ix = max(row["inlayer"], row["across"])
        res[row["case"]] = dict(break_N=round(F_TEST / (ix * k), 1), inlayer=row["inlayer"], across=row["across"],
                                mode="in-layer" if row["inlayer"] >= row["across"] else "across the layers",
                                at_mm_from_motor=round(L - row["at"][0], 1), at=row["at"])
    return res


def main():
    path = frame_mesh()
    cp, puck, block, scr, L, dropped = build(path)
    pred = {f"h{h:g}": fe(cp, block, scr, L, h) for h in (1.0, 2.0)}
    # the print: deck top down (the frame's build -z), the puck flat beside it
    flip = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
    c1 = cp.copy(); c1.apply_transform(flip); c1.apply_translation(-c1.bounds[0])
    pk = puck.copy(); pk.apply_translation(-pk.bounds[0])
    pr = c1
    assert pr.bounds[1].max() <= 180, pr.bounds
    os.makedirs(os.path.join(ROOT, "print"), exist_ok=True)
    pp = os.path.join(ROOT, "print", f"{NAME}.stl"); pr.export(pp)
    pk.export(os.path.join(ROOT, "print", f"{NAME}_puck.stl"))
    d = os.path.dirname(os.path.abspath(__file__))
    dr = trimesh.util.concatenate([cp, puck]); dr.export(os.path.join(d, "part.stl"))   # arm frame; test_parts poses it
    import subprocess
    sc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solid_check.py"), pp], capture_output=True, text=True)
    sk = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solid_check.py"), pp.replace(".stl", "_puck.stl")], capture_output=True, text=True)
    out = dict(frame_mesh=os.path.relpath(path, ROOT), coupon_g=round(cp.volume / 1000 * 1.23, 1), puck_g=round(puck.volume / 1000 * 1.23, 1),
               dropped_cm3=round(dropped, 2), print_extents=np.round(pr.extents, 1).tolist(), solid_check="PASS" if sc.returncode == 0 else "FAIL",
               puck_solid_check="PASS" if sk.returncode == 0 else "FAIL",
               f_test_N=F_TEST, eye_above_pad=EYE_Z, predicted=pred)
    json.dump(out, open(os.path.join(d, "predict.json"), "w"), indent=1)
    from topo.build import render
    p1 = pred["h1"]
    render(pr, os.path.join(d, "preview.png"), title=f"{NAME}: motor 1's pod and arm, cut from the frame's mesh, as printed",
           info=f"{out['coupon_g']} g + puck {out['puck_g']} g; predicted break up {p1['up']['break_N']:.0f} N, sideways {p1['side']['break_N']:.0f} N")
    print(json.dumps(out, indent=1))
    for q in (sc, sk):
        if q.returncode:
            print(q.stdout[-1500:], q.stderr[-800:])


if __name__ == "__main__":
    main()
