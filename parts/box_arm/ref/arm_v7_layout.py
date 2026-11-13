"""Box swap arm v7 (24 Oct 2026, swap arm system worker; hub decisions after the v6 concept check, DEFAULT): the v6 stack (rails on the
motor plate) carrying both packs, Scout and Overseer, to 9 bays at 30 deg (Spark 24 Oct): one ring radius R 113 for every seated face (drone
seat and bays; a parked Overseer cartridge passes them), strokes +13, rails to r 151, rack tail to u -63, carbon ties under the rails back
in, the XT60 at each cartridge's closed (arm-side) end (the latch's change; the only option inside the float carrier for both packs), the
Overseer gripped on its drone's thrust axis (30 off its swap CG, body toward +Y), detent springs x1.25.
`use()` applies arm_v6_layout's v6 values and then patches arm_v5_layout (rails, rack, strokes) and arm_layout (R and the strokes derived
from it) in place, so every script that reads them builds v7. Import and call it before arm_kinematics (its stroke constants are read at
import). Numbers C unless tagged."""
import math, copy
import arm_layout as L
import arm_v5_layout as V
import arm_v6_layout as S6

R_V5 = 100.0
R7 = 113.0                              # every seated face (hub decision, DEFAULT; alt R 100 Scout / R 113 Overseer with two strokes)
DR = R7 - R_V5                          # 13: every stroke that ends on a seated face moves out by it
RAIL_END = 151.0                        # the outer wheel's contact at the commanded drone insert (u 32 at z0 111.3) + 6.4 of ridge + 1.0 (C)
WEB_U7 = (-63.0, 30.0)                  # head A's rack tail: the pinion (u = 55 - z0) stays 6 inside it at the commanded drone insert z0 111.3
WEB_U7_B = (-54.0, 30.0)                # head B never inserts at the drone (it pulls there at 102.6 and inserts at bays, 101.8 commanded): 6.4 inside;
                                        # its shorter tail clears the detent's springs and tower web (the full -63 tail met them, sweep C)
Y_C7 = float(__import__("os").environ.get("NOX_YC", "167.1"))   # round 3b (DEFAULT, hub: one funnel centred on each drone's catch centre = its
                                        # latch set's centre, the tab line there too); 24 Oct: 165 -> 167.1 for latch v4's Overseer receiver v2 (its
                                        # lead end 2.15 longer: the 3.0 capture kept, the arm's Y error ~±1..1.8 > the ±0.7 a trim needed), whose
                                        # arm-side end then clears the low stack's plate top 74.35 by 1.0 (swap_system Y_C_min 167.1, F); round 3 had 138
DETENT_SPRING_SCALE = 1.3               # concept review r3 (DEFAULT): passing a notch at a slow step rate (0.5 N.m pull-out, E) with both parked
                                        # heads carrying: 1.31 (1.55 on holding torque); a power cut mid-swap, both parked carrying: 1.14; a head
                                        # just out of a horizontal Overseer bay's side faces: 1.04 (swap_system.power_cut_head_out); x1.5: 1.21 / 1.31 / 1.20
AXLE_RATIO = float(__import__("os").environ.get("NOX_AXLE_RATIO", "5.0"))                          # round 3: a GT2 16T on the axle motor (80:16, DEFAULT): x1.25 the stall torque and x1.56 the shorted-coil
                                        # damping, so the springs can hold the Overseer and the motor still indexes past them (swap_system.power_cut_head_out)
XT60_Z = 7.0                            # the cartridge's XT60 axis 7 from its closed end (DEFAULT, asked of the latch; plug face at its y as before)
OV_GRIP_OFFSET = 30.0                   # the Overseer's tab line 30 toward the closed end from its swap CG = its drone's thrust axis (Spark B)
BAYS = dict(overseer=(0.0, -60.0, -120.0, -180.0), scout=(30.0, -30.0, -90.0, -150.0, -210.0), drone=90.0, pitch=30.0)   # box structure v2's set
                                        # (via the hub): alternating Overseer / Scout (two Overseer receivers side by side overlap 937 mm3 at R 113)
PHI_WINDOW = (-300.0, 90.0)             # head A at any bay or the drone, head B at any bay or the drone: one axle window (390 of the loop's 422)
STACK3 = __import__("os").environ.get("NOX_ARM_STACK", "low")   # round 3: "low" (DEFAULT) or "v6" (the round 2 carriage, for comparison)
TIE7 = dict(S6.TIE)                     # the tie rods as v6's alternative (rail end r 128: the stand's arm lane at low Y keeps them inside r 135)
_DONE = False


def strokes():
    z = dict(park=L.Z0_A_HOME, wait=81.0 + DR, bay=84.6 + 1.5 + DR, drone=84.6 + 5.0 + DR, insert=84.6 + 5.0 + V.DRONE_PUSH + DR)
    z["bay_cmd"] = z["bay"] + S6.OVERTRAVEL; z["insert_cmd"] = z["insert"] + S6.OVERTRAVEL
    return z


# round 3 (concept review r2 + latch d0cd1a1: the XT60 end faces the arm; heads v7 24 Oct: the plate top 74.35, 1.0 under the Overseer receiver at Y_C 165;
# the arm clears it): the LOW carriage stack. Everything under a cartridge (carriage plate X +-28, wheels, nuts, bolt heads, eccentric) ends
# at Y 74.35 (heads v7), 1.0 under the Overseer's end; the rack and pinion (X 40.5.. beside the cartridge) stay. The rail loses 6.25 of depth on top and
# gains a keel under it outboard of the hub plate (r >= 89, Y 38..54), which brings the free XT60 pitch back (swap_system.mate_pitch). C.
LOW = dict(RAIL_TOP=66.75, RIDGE_Y=62.75, WHEEL_Y=(57.75, 67.75), SPACER_Y=(67.75, 68.75), PLATE_Y=(68.75, 74.35),   # plate top 74.35 (heads v7: 1.0 under the Overseer receiver at Y_C 165; was 72.75)
           NUT_Y=(55.05, 57.75),                # DIN 439 M5 thin nut 2.7 + threadlocker (E) under each wheel; the hub plate's wheel lanes at Y 53
           AXLE_BOLT_L=dict(fixed=20.0, ecc=20.0, head="ISO 10642 countersunk, flush in the plate top"),   # tails Y 54.35 (1.35 over the hub's wheel lanes)
           ECC_FLANGE=(71.85, 74.35),           # the eccentric bush's flange flush in a plate counterbore (pin spanner from above)
           KEEL=dict(x=9.0, y=(38.0, 54.0), r0=89.0),   # rail keel X +-9 under the body from r 89 (1.0 off the hub plate's edge r 88) to the tip
           CBORE_Y=57.75, RAIL_BOLT="M5 x 10 ISO 4762 (head Y 57.75..62.75, under the channel floor)", STOP_Y=(66.75, 72.0), CHAN_Y=(62.75, 66.75),
           CHAN_X=(-3.25, 3.25), CHAN_MOUTH=5.0,  # rail v5 (box electronics PR 103: head ribbon 5 conductors ~6.1 wide, E): floor 6.5, mouth 5.0
                                        # (0.75 lip a side, as v4); the bolt heads sunk under the floor (v4 had them 3.25 up into the channel)
           STOP_R=(27.5, 31.0),                 # the rail's inner step and park stop from r 27.5: 1.0 off detent ring v4 (r 26.5); the bumper face r 31 kept
           TIE_RAIL=(41.6, 128.0))              # the tie rod end 10 up its own axis from the keel bottom at r 128 (mouth on the face)


RACK_V3_Y = (59.25, 73.25)              # concept review r3: rack teeth grow down (the top stays 1.1 under the plate top 74.35)
PIN_V3 = dict(teeth_y=(59.75, 73.25), boss_y=(55.0, 59.25))   # pinion16 v3: face 13.5 fully engaged on the rack, 10.25 on its tail (from Y 63); D-bore on the
                                        # shaft's flat (Y 59..74), boss Y 55..59.25 with an M3 grub on the round shaft for axial hold only


def _patch_low():
    lo = LOW
    V.RAIL_Y = (V.RAIL_Y[0], lo["RAIL_TOP"]); V.RIDGE_Y = lo["RIDGE_Y"]; V.WHEEL_Y = lo["WHEEL_Y"]; V.SPACER_Y = lo["SPACER_Y"]; V.PLATE_Y = lo["PLATE_Y"]
    V.NUT_Y = lo["NUT_Y"]; V.TAIL_Y = (lo["PLATE_Y"][1] - lo["AXLE_BOLT_L"]["fixed"], lo["NUT_Y"][0]); V.AXLE_BOLT_L = dict(lo["AXLE_BOLT_L"])
    e = dict(V.ECC); e["flange_y"] = lo["ECC_FLANGE"]; V.ECC = e
    V.AXLE_KEEP = dict(r=5.0, y=(lo["PLATE_Y"][1], lo["PLATE_Y"][1] + 8.0)); V.ECC_KEEP = dict(r=13.0, y=(lo["PLATE_Y"][1], lo["PLATE_Y"][1] + 1.0))
    V.RACK_TEETH_Y = RACK_V3_Y; V.PIN_TEETH_Y = PIN_V3["teeth_y"]; V.PIN_BOSS_Y = PIN_V3["boss_y"]
    V.WEB_Y = (V.RACK_TEETH_Y[0], lo["PLATE_Y"][1])
    st = dict(V.RAIL_STOP); st["y"] = lo["STOP_Y"]; st["r"] = lo["STOP_R"]; V.RAIL_STOP = st
    ch = dict(V.RAIL_CHAN); ch["y"] = lo["CHAN_Y"]; ch["x"] = lo["CHAN_X"]; ch["mouth"] = lo["CHAN_MOUTH"]; V.RAIL_CHAN = ch
    b = dict(V.RAIL_BOLTS); b["cbore_y"] = lo["CBORE_Y"]; b["bolt"] = lo["RAIL_BOLT"]; V.RAIL_BOLTS = b; V.RAIL_INSERTS = b
    V.RAIL_FOOT_X = V.RAIL_BODY_X                # no feet: the wheels (Y 57.75..) pass over the pads; the bolts sit in the body
    V.RAIL_KEEL = dict(lo["KEEL"])
    S6.TIE["rail"] = lo["TIE_RAIL"]


def _patch_v5():
    z = strokes()
    V.RAIL_R = (V.RAIL_R[0], RAIL_END)
    V.WEB_U = WEB_U7; V.WEB_U_B = WEB_U7_B
    V.AX_TEETH = int(round(80 / AXLE_RATIO))                                    # the loop strands round the axle motor's 16T
    L.P20_OD = V.AX_TEETH * 2 / math.pi                                        # the belt models (arm_kinematics belt_axle, detent) on the 16T's pitch d
    V.Z0S = (z["park"], z["wait"], z["bay"], z["drone"], z["insert"], z["bay_cmd"], z["insert_cmd"])
    S6.Z0_INSERT_MAX = z["insert_cmd"]; S6.Z0_BAY_INSERT = z["bay_cmd"]
    if STACK3 == "low": _patch_low()


def use():
    """v6 values, then the v7 patches (idempotent); registers itself so a later S6.use("v6") (any v6 script's import) keeps v7."""
    global _DONE
    if _patch_v5 not in S6.POST_V6: S6.POST_V6.append(_patch_v5)
    S6.use("v6")
    if not _DONE:
        L.R = R7
        L.R_BAY_TOP = L.R + L.CART_H; L.R_BAY_BACK = L.R_BAY_TOP + 6.0
        L.Z0_BAY = 84.6 + DR; L.Z0_DRONE = L.Z0_BAY + 5.0; L.Z0_B_WAIT = 81.0 + DR
        L.Y_C = Y_C7
        _DONE = True
    _patch_v5()
    import detent as DT                                                        # detent v6 parts with the v7 stack (the piece pass, CHOSEN_V8)
    DT.use_v6()
    return strokes()


if __name__ == "__main__":
    z = use()
    print("strokes", z)
    w = V.clearance_sweep((53.93, -43.67)); g = min(w.values())
    print("2D clearance min", round(g, 2), min(w, key=w.get))
    print("reach: wheel rim at the commanded insert r", round(math.hypot(V.WHEEL_X, z["insert_cmd"] + V.WHEELS[1][1] + V.WHEEL_OD / 2), 1),
          "plate corner", round(math.hypot(V.PLATE_X[1], z["insert_cmd"] + V.PLATE_U[1]), 1), "pinion in the rack tail by",
          round((V.PIN_Z - z["insert_cmd"]) - V.WEB_U[0], 1))
