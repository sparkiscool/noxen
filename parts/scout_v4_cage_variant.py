"""TEMPORARY (13 Nov, the design chat: two cage lines for the Scout v4 frame at once, the winner folded into tools/scout_v4_shape.py, the
frame and the nodes; this file and the variant folders deleted after). A frame variant's own cage, in memory: the line's nodes and rods
patched into tools/scout_v4_shape, the frame model (frame_check, write=False) solved on the frame's OWN setup mass (gen_cm3 x 1.23 + its
drawn mass, so a result cannot pass on a lighter frame's loads), its node loads put in place of systems/scout_v4.cage_loads.json for this
process, and the frame part told which end sockets hold a 6/4 tube.
  line "tube":  DN-NB and DT-TB as 6/4 tube (13 x the 3.0 rod's Euler load: I 51.1 vs 3.98 mm4), DN / DT at z 18.4 (the 6.2 collar's r 6.1
                under the bed face) and |x| 76.0 (the 15 deep socket inside the 180, clear of the Radxa).
  line "split": BT split into BT0 / BT1 at (-75.5, +-23, 11), beside the Radxa, each straight in its layers to its tail bolt; DT-TB as 6/4
                tube at (-76, 0, 18.4); DN-NB as tube too when the frame model's 3.0 Euler there is over 0.8."""
import json, math, os
import parts.scout_v4_units_lib as UL

S = UL.S
I_ROD, I_TUBE = math.pi * 3.0 ** 4 / 64, math.pi * (6.0 ** 4 - 4.0 ** 4) / 64
PULL_CAP = {3: 940.0, 6: 1880.0}                         # N: the 3.2 socket's pull-out cap (frame model), twice the bond area for the 6.2 (E)


def cage(line, gen_cm3, drawn_g=33.0):
    """the line's cage patched into tools/scout_v4_shape and solved on the frame's own mass: (node loads, tube members, report)"""
    N0 = dict(S.NODES)
    S.NODES["DN"], S.NODES["DT"] = (76.0, 0, 18.4), (-76.0, 0, 18.4)
    tubes = {"DN-NB", "DT-TB"}
    if line == "tube2":                                  # the side rods too (13 Nov: BS / DS 0.92-0.96 x Euler at gen 230 / 260 as 3.0)
        tubes |= {"DS0-S0", "DS1-S1", "BS0-S0", "BS1-S1"}
        S.NODES["DS0"], S.NODES["DS1"] = (0.0, 40.0, 18.4), (0.0, -40.0, 18.4)
    if line == "split":
        del S.NODES["BT"]
        S.NODES["BT0"], S.NODES["BT1"] = (-75.5, 23.0, 11.0), (-75.5, -23.0, 11.0)
        S.RODS[:] = [r for r in S.RODS if r != ("BT", "TB")] + [("BT0", "TB"), ("BT1", "TB")]
        S.PRINTED = tuple(k for k in S.PRINTED if k != "BT") + ("BT0", "BT1")
    A = {p["id"]: p["mass_g"] for p in json.load(open(os.path.join(S.ROOT, "systems", "scout_v4.assembly.json")))["parts"]}
    frame_g = gen_cm3 * 1.23 + drawn_g
    core = sum(v for k, v in A.items() if "cage" not in k) - A["scout_v4_frame"] + frame_g + S.BATTERY_SET[0]
    o = S.frame_check(core, write=False)
    if line == "split" and o["rods"].get("DN-NB", {}).get("compression", 0) / o["rods"]["DN-NB"]["Pcr"] <= 0.8:
        tubes.discard("DN-NB"); S.NODES["DN"] = N0["DN"]; o = S.frame_check(core, write=False)
    rep = []
    for k, v in sorted(o["rods"].items(), key=lambda kv: -kv[1]["compression"] / kv[1]["Pcr"]):
        tube = k in tubes
        eu = v["compression"] / (v["Pcr"] * (I_TUBE / I_ROD if tube else 1.0))
        rep.append((k, "6/4" if tube else "3.0", round(eu, 2), round(v["tension"]), PULL_CAP[6 if tube else 3]))
    worst = max(eu for _, _, eu, _, _ in rep); over = [k for k, kind, eu, pull, cap in rep if pull > cap]
    print(f"cage variant {line} at gen {gen_cm3}: frame {frame_g:.0f} g (own setup mass), core {core:.0f} g; tubes {sorted(tubes)}; "
          f"crash deflection {o['deflection'][0]} mm; end moment <= {o['moment'][0]:.0f} N mm ({o['moment'][1]})")
    for k, kind, eu, pull, cap in rep[:6]:
        print(f"  {k:8} {kind}: Euler {eu}, pull {pull} N (cap {cap:.0f})")
    if worst >= 1.0 or over:
        raise ValueError(f"cage variant {line} at gen {gen_cm3}: the cage fails the frame model (Euler {worst:.2f}, pull over the cap on {over})")
    return o["node_loads"], tubes, A, frame_g


def apply(line, gen_cm3, drawn_g=33.0):
    """the frame part on the line's cage: its node loads, its tube sockets, its end nodes, its ground hit on the heavier drone"""
    nl, tubes, A, frame_g = cage(line, gen_cm3, drawn_g)
    UL.NL4.clear(); UL.NL4.update(nl)
    import parts.scout_v4_frame.part as F
    F.TUBE_NODES = tuple(n for k in tubes for n in k.split("-") if n in S.PRINTED)
    if line == "split":
        F.END_NODES = ("DN", "BN", "DT", "BT0", "BT1")
    drone_g = sum(A.values()) - A["scout_v4_frame"] + frame_g + S.BATTERY_SET[0]
    F.HIT_N = round(2 * drone_g * UL.G * 600.0 / 15.0, -1); F.DRONE_G = drone_g
    return F


def node(line, key, gen_cm3, budget_cm3):
    """a cage node (NB / TB) on the line's cage, its tube members as 6.2 x 15+ sockets (units_lib.socket_depth6), for the frame at gen_cm3"""
    nl, tubes, A, frame_g = cage(line, gen_cm3)
    import parts.scout_v4_node_lib as NL
    L = NL.L
    L.LOADS = nl
    raw = L._raw_members

    def _raw(k, bridges=False):
        out = raw(k, bridges)
        for m in out:
            if m["name"] in tubes:
                f = nl.get(k, {}).get(m["name"]) or {}
                m["bore_r"], m["depth"] = UL.TUBE_BORE_R, UL.socket_depth6(f.get("axial", 0.0), f.get("side", 0.0))
        return out
    L._raw_members = _raw
    L.MOUTH_R = UL.TUBE_BORE_R + L.WALL + L.STRUT_M + 0.5
    L.MAX_START = 42.0                                   # a 6.2 bore beside its near-parallel 3.2 neighbour starts 40 out (was 36 with two 3.2s): a longer Y
    L.SOCKET_START = L._socket_starts()
    spec = NL.spec_for(key); spec["gen_cm3"] = budget_cm3
    return NL.make(key), spec
