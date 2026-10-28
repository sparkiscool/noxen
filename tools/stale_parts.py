"""Print the names of parts that need a run: part.py exists but result.json is missing or was made from an older part.py or lib (tools/part_sha.py)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_sha import stale
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = []
for py in sorted(glob.glob(os.path.join(ROOT, "parts", "*", "part.py"))):
    d = os.path.dirname(py); rj = os.path.join(d, "result.json")
    if not os.path.exists(rj) or stale(py, json.load(open(rj))):
        out.append(os.path.basename(d))
print(" ".join(out))
