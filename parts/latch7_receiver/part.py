"""Scout battery receiver (latch v7): the setup is parts/latch7_recv_lib.py, shared with the Overseer's.
h 1.2: the receiver's box (68 x 111 x 54, hollow) is 295k cells there (1.0: 484k, refused). 34 cm3 on 21 drawn: about 68 g (drawn whole it was 95 g); 28 left the 118 N jammed-bolt case at ~0.93 thinned (room-full 0.39)."""
from parts.latch7_recv_lib import setup_for, POSE_OF
SPEC = dict(h=1.2, gen_cm3=34.0, max_worst=0.5, settings=dict(near_sym=True))
setup = setup_for("scout", "latch7_receiver")
POSE = POSE_OF("scout")       # part (print pose, mount plane on the bed) -> cartridge coords
