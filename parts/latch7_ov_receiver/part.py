"""Overseer battery receiver (latch v7): the setup is parts/latch7_recv_lib.py, shared with the Scout's.
h 1.5: the receiver's box (68 x 184 x 69, hollow) is 295k cells there (1.4: 426k, refused). gate_member off: the side walls' room is 6.1 thick (the cartridge
inside, the funnel's seat outside), under the 6.25 member the gate's erosion asks for (the audit's env_neck), so the members are the engine's 3.2 and
the walls come out as plates the room's thickness; the solid gate judges the result as always. 64 cm3 on 17 drawn: about 100 g (drawn whole it was 210 g)."""
from parts.latch7_recv_lib import setup_for, POSE_OF
SPEC = dict(h=1.5, gen_cm3=64.0, max_worst=0.5, settings=dict(near_sym=True, gate_member=False))
setup = setup_for("overseer", "latch7_ov_receiver")
POSE = POSE_OF("overseer")    # part (print pose, mount plane on the bed) -> cartridge coords
