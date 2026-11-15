"""Landing funnel v5, half_tail: PROTOTYPE (drawn, 15 Nov). The engine setup's room filled solid less all its cuts, with the assembly fixes,
fits, assembly path and hand calculation in parts/box_funnel/proto.py (its docstring). The engine piece stays at OPTIMISE.
    python parts/box_funnel_proto_half_tail/make.py     writes parts/box_funnel_proto_half_tail/part.stl (seat frame) and print/box_funnel_proto_half_tail*.stl (print pose), gates them"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "parts", "box_funnel"))
import proto

name, t, A, clear, rows, pose = proto.build("half_tail")
for nm, f, p, ms in rows:
    ok = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "solid_check.py"), f], capture_output=True).returncode == 0
    print(f"{nm}: {ms['volume_cm3']} cm3 solid, {ms['bed_mm']} on the bed, {ms['bed_contact_mm2']} mm2 down, solid check {'PASS' if ok else 'FAIL'}; {pose}")
