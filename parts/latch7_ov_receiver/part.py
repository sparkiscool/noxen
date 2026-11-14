"""Overseer battery receiver (latch v7): the setup is parts/latch7_recv_lib.py, shared with the Scout's.
h 1.5: the receiver's box (68 x 184 x 69, hollow) is 295k cells there (1.4: 426k, refused). gate_member off: the side walls' room is 6.1 thick (the cartridge
inside, the funnel's seat outside), under the 6.25 member the gate's erosion asks for (the audit's env_neck), so the members are the engine's 3.2 and
the walls come out as plates the room's thickness; the solid gate judges the result as always. 76 cm3 on 22 drawn: about 120 g; 14 Nov: on the drone it now hangs on four deck bosses under the base's inserts, the six are the bay's set (room-full 0.31 in the side crash) (run 265 at 64: 105 g, gate PASS, 0.526 landing on the lead keys, see latch7_recv_lib.EAR_SEAT) (drawn whole it was 210 g)."""
from parts.latch7_recv_lib import setup_for, POSE_OF
SPEC = dict(h=1.5, gen_cm3=76.0, max_worst=0.5, settings=dict(near_sym=True, gate_member=False))
setup = setup_for("overseer", "latch7_ov_receiver")
POSE = POSE_OF("overseer")    # part (print pose, mount plane on the bed) -> cartridge coords
