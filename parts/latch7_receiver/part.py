"""Scout battery receiver (latch v7): the setup is parts/latch7_recv_lib.py, shared with the Overseer's.
h 1.2: the receiver's box (68 x 111 x 54, hollow) is 295k cells there (1.0: 484k, refused). 36 cm3 on 26 drawn: about 76 g: 36 cm3 on 26 drawn since 14 Nov (drawn whole it was 95 g; runs at 34 and 40, 65.9 and 74.1 g: both gates failed at the tail feet both times, see make.scout_feet: the feet's gussets are drawn now); 28 left the 118 N jammed-bolt case at ~0.93 thinned (room-full 0.39).
gate_member off, as on the Overseer's: the side walls' room is 5.0 thick (cartridge inside, the drone's cage outside), under the member the gate's erosion asks for
(with the feet's wider leg room the engine routed 5.3 and the audit's env_neck refused the walls); members are the engine's 3.2, the solid gate judges as always.
14 Nov, run on 4d3d93cf at 36: 73.5 g, solid check PASS, 0.64 across the layers at the lead notches' roots in the hold-down (0.77 before the final pass; the room filled solid
read 0.43). Two causes: the key's seat block had 3.5 of ligament behind its notch (make.HOLD sc_len: 12 now, as the Overseer's), and 36 cm3 was 52 % of the room,
too little to join that root to the leg. 46 cm3 (67 % of the room), about 86 g."""
from parts.latch7_recv_lib import setup_for, POSE_OF
SPEC = dict(h=1.2, gen_cm3=46.0, max_worst=0.5, settings=dict(near_sym=True, gate_member=False))
setup = setup_for("scout", "latch7_receiver")
POSE = POSE_OF("scout")       # part (print pose, mount plane on the bed) -> cartridge coords
