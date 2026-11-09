"""The print mesh lives outside git (9 Nov, Spark: the committed meshes were 93 % of git storage, 441 MB at HEAD). Git carries a
viewing mesh per part (parts/<p>/part.stl, VIEW_FACES); the full mesh the engine exported and the gate passed is a GitHub Release
asset, one stable URL per part: https://github.com/<repo>/releases/download/meshes/<p>.stl, its url and sha256 in result.json
`print_stl` (print.json for a make.py part). Hand-made print files (print/*.stl) go to the same release as print-<name>.stl, listed in print/index.json.

    python tools/print_mesh.py publish <part> ... | --all   CI only (needs GH_TOKEN): upload parts/<p>/part.stl as <p>.stl
                                                           (replacing the old one), write result.json print_stl, then put the
                                                           viewing mesh in part.stl. --all: every part whose part.stl is not
                                                           yet a published viewing mesh. Exit 1 if any upload failed: that
                                                           part's files are left as they were (the caller commits none of it).
    python tools/print_mesh.py print-files                  CI only: every print/*.stl to the release, removed from print/,
                                                           listed in print/index.json
    python tools/print_mesh.py fetch <part> ...             the print mesh to parts/<p>/print.stl (git-ignored), sha256 checked

Readers that need the exact print geometry (tools/recut.py, the bore check in tools/stale_parts.py) call local(part); the page
and the layout tools read the viewing mesh in part.stl (volume within 0.05 %, bounds within 0.05 mm of the print mesh, 9 Nov).
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("GITHUB_REPOSITORY", "sparkiscool/noxen")
TAG = "meshes"
VIEW_FACES = 60000          # ~3 MB binary STL; the page decimates to 20k, import_box to 40k; layout tools read bounds and slices


def url(asset):
    return f"https://github.com/{REPO}/releases/download/{TAG}/{asset}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gh(*args):
    return subprocess.run(["gh", *args, "-R", REPO], capture_output=True, text=True)


def ensure_release():
    if gh("release", "view", TAG).returncode == 0:
        return
    r = gh("release", "create", TAG, "--title", "print meshes", "--latest=false",
           "--notes", "The full print mesh of every part (<part>.stl) and the hand-made print files (print-<name>.stl), replaced on each "
                      "result. Written by CI (tools/print_mesh.py); git keeps a viewing mesh. Never edit by hand.")
    if r.returncode != 0 and gh("release", "view", TAG).returncode != 0:      # another job may have created it at the same moment
        raise RuntimeError(f"release {TAG}: {r.stderr.strip()}")


def upload(path, asset):
    """path -> release asset `asset` (replacing one of that name), checked by size in the release listing. Returns its url."""
    ensure_release()
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, asset); shutil.copyfile(path, f)               # gh names the asset after the file
        size = os.path.getsize(f)
        for i in range(4):
            r = gh("release", "upload", TAG, f, "--clobber")
            if r.returncode == 0:
                v = gh("release", "view", TAG, "--json", "assets", "--jq", f'.assets[] | select(.name == "{asset}") | .size')
                if v.returncode == 0 and v.stdout.strip() == str(size):
                    return url(asset)
            print(f"print_mesh: upload of {asset} failed ({i + 1}): {(r.stderr or '').strip()[:200]}"); time.sleep(5 * (i + 1))
    raise RuntimeError(f"upload of {asset} failed")


def viewing(m, faces=VIEW_FACES):
    return m.simplify_quadric_decimation(face_count=faces) if len(m.faces) > faces else m


def record(p):
    """where the part's print_stl entry lives: result.json for a generated part, print.json for a make.py part (drawn by hand,
    no run: the top cover, the solid arms), None for a bought or placed mesh (not printed from this repo)"""
    d = os.path.join(ROOT, "parts", p)
    if os.path.exists(os.path.join(d, "result.json")):
        return os.path.join(d, "result.json")
    if os.path.exists(os.path.join(d, "make.py")):
        return os.path.join(d, "print.json")
    return None


def print_stl(p):
    rj = record(p)
    return ((json.load(open(rj)).get("print_stl") if rj and os.path.exists(rj) else None) or {})


def published(p):
    """parts/<p>/part.stl is already the viewing mesh of a published print mesh (or the part is not printed from here)"""
    stl = os.path.join(ROOT, "parts", p, "part.stl")
    if record(p) is None or not os.path.exists(stl):
        return True
    return print_stl(p).get("view_sha256") == sha256(stl)


def publish(p):
    import trimesh
    d = os.path.join(ROOT, "parts", p); rj = record(p); stl = os.path.join(d, "part.stl")
    if published(p):
        return True
    try:
        full_sha, nbytes = sha256(stl), os.path.getsize(stl)
        u = upload(stl, f"{p}.stl")
    except Exception as e:
        print(f"print_mesh: {p}: {e}; nothing changed"); return False
    m = trimesh.load(stl, force="mesh"); nf = len(m.faces)
    shutil.copyfile(stl, os.path.join(d, "print.stl"))                    # the local copy local() returns (git-ignored)
    v = viewing(m); v.export(stl)
    r = json.load(open(rj)) if os.path.exists(rj) else {}
    r["print_stl"] = dict(url=u, sha256=full_sha, bytes=nbytes, faces=nf, view_faces=len(v.faces), view_sha256=sha256(stl),
                          at=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(r, open(rj, "w"), indent=1)
    print(f"print_mesh: {p}: {nbytes / 1e6:.1f} MB print mesh -> {u}; part.stl {os.path.getsize(stl) / 1e6:.1f} MB ({len(v.faces)} faces)")
    return True


def local(p):
    """the path of the part's full print mesh: parts/<p>/print.stl (downloaded from the release when missing or not the current
    one), or part.stl itself when it was never published (an old result). Raises when the download fails."""
    d = os.path.join(ROOT, "parts", p); stl = os.path.join(d, "part.stl")
    ps = print_stl(p)
    if not ps or ps.get("view_sha256") != sha256(stl):
        return stl                                                          # part.stl is itself the print mesh
    f = os.path.join(d, "print.stl")
    if os.path.exists(f) and sha256(f) == ps["sha256"]:
        return f
    tmp = f + ".part"
    urllib.request.urlretrieve(ps["url"], tmp)                              # public repo: no token needed
    if sha256(tmp) != ps["sha256"]:
        os.remove(tmp); raise RuntimeError(f"{p}: the release's {p}.stl is not the one result.json names (sha256)")
    os.replace(tmp, f)
    return f


def print_files():
    """print/*.stl -> release print-<name>.stl, removed from print/, listed in print/index.json"""
    import glob
    idx_path = os.path.join(ROOT, "print", "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    bad = []
    for f in sorted(glob.glob(os.path.join(ROOT, "print", "*.stl"))):
        nm = os.path.basename(f)
        try:
            u = upload(f, f"print-{nm}")
        except Exception as e:
            print(f"print_mesh: {nm}: {e}; left in print/"); bad.append(nm); continue
        idx[nm] = dict(url=u, sha256=sha256(f), bytes=os.path.getsize(f), at=time.strftime("%Y-%m-%d %H:%M"))
        os.remove(f); print(f"print_mesh: print/{nm} -> {u}")
    json.dump(dict(sorted(idx.items())), open(idx_path, "w"), indent=1)
    return not bad


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "publish":
        if args == ["--all"]:
            import glob
            args = sorted(os.path.basename(os.path.dirname(r)) for r in glob.glob(os.path.join(ROOT, "parts", "*", "part.stl")))
            args = [p for p in args if not published(p)]
            print(f"print_mesh: {len(args)} to publish: {' '.join(args)}")
        bad = [p for p in args if not publish(p)]
    elif cmd == "fetch":
        bad = []
        for p in args:
            try:
                print(local(p))
            except Exception as e:
                print(f"print_mesh: {p}: {e}"); bad.append(p)
    elif cmd == "print-files":
        bad = [] if print_files() else ["print"]
    else:
        sys.exit(__doc__)
    sys.exit(1 if bad else 0)
