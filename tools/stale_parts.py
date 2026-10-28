"""Print the names of parts that need a run: part.py exists but result.json is missing or was made from an older part.py."""
import glob, hashlib, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = []
for py in sorted(glob.glob(os.path.join(ROOT, "parts", "*", "part.py"))):
    d = os.path.dirname(py); rj = os.path.join(d, "result.json")
    sha = hashlib.sha256(open(py, "rb").read()).hexdigest()[:12]
    if not os.path.exists(rj) or json.load(open(rj)).get("part_sha") != sha:
        out.append(os.path.basename(d))
print(" ".join(out))
