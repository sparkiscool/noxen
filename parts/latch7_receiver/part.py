"""Scout battery receiver (latch v7): the setup is parts/latch7_recv_lib.py, shared with the Overseer's.
h 1.2: the receiver's box (68 x 111 x 54, hollow) is 295k cells there (1.0: 484k, refused). 40 cm3 on 21 drawn: about 75 g (drawn whole it was 95 g; run 265 at 34: 65.9 g, both gates failed at the tail feet, see make.FOOT_WEB_X0); 28 left the 118 N jammed-bolt case at ~0.93 thinned (room-full 0.39).
gate_member off, as on the Overseer's: the side walls' room is 5.0 thick (cartridge inside, the drone's cage outside), under the member the gate's erosion asks for
(with the feet's wider leg room the engine routed 5.3 and the audit's env_neck refused the walls); members are the engine's 3.2, the solid gate judges as always."""
from parts.latch7_recv_lib import setup_for, POSE_OF
SPEC = dict(h=1.2, gen_cm3=40.0, max_worst=0.5, settings=dict(near_sym=True, gate_member=False))
setup = setup_for("scout", "latch7_receiver")
POSE = POSE_OF("scout")       # part (print pose, mount plane on the bed) -> cartridge coords
