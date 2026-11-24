"""Scout v4 probe bench, the step-by-step assembly guide with a picture per step (23 Nov, Spark: "pull up an assembly guide for the
bench test, I pretty much printed everything").
    python tools/probe_bench_guide.py [--pics DIR]    -> artifacts/probe_bench_guide.html (one self-contained page, pictures inside it)
The pictures are drawn here from each part's parts/<name>/part.stl (drone frame, mm, z up): the parts already on in grey, the step's new
parts in colour and pulled back along the way they go in. The screws, inserts and nut are drawn from the sizes in systems/parts-list.md.
The words are in tools/probe_bench_guide_steps.json, so a change of order or wording needs no code. --pics also writes each picture as a png.
The order is the one the meshes allow (checked with manifold overlaps, 23 Nov): the pinion and the N20 go into the turret before the
turret goes on the base (the N20 needs 10 mm of travel along y, the base's opening leaves 3)."""
import base64, io, json, os, sys
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "parts")); sys.path.insert(0, os.path.join(ROOT, "tools"))
from parts.scout_v4_units_lib import TURRET_BOLTS, DZ1          # noqa: E402
import scout_v4_shape as S                                      # noqa: E402

CX, CZ = S.SPOOL["c"]
COL = {"printed": (0.17, 0.60, 0.58), "bought": (0.88, 0.54, 0.12), "hardware": (0.36, 0.45, 0.62), "done": (0.80, 0.81, 0.83)}


def load(name):
    return trimesh.load(os.path.join(ROOT, "parts", name, "part.stl"), process=True)


def ycyl(r, y0, y1, x=0.0, z=0.0, n=28):
    c = trimesh.creation.cylinder(radius=r, height=y1 - y0, sections=n)
    c.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])); c.apply_translation([x, (y0 + y1) / 2, z]); return c


def zcyl(r, z0, z1, x, y, n=28):
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=n); c.apply_translation([x, y, (z0 + z1) / 2]); return c


def parts():
    """every drawn thing by a short id: (mesh, kind)"""
    P = {}
    for k, n, kind in (("base", "scout_v4_probe_bench_base", "printed"), ("turret", "scout_v4_probe_bench_turret", "printed"),
                       ("spool", "scout_v4_probe_spool", "printed"), ("pinion", "scout_v4_probe_pinion", "printed"),
                       ("motor", "scout_v4_probe_motor", "bought"), ("head", "scout_v4_head_frame", "printed"),
                       ("cradle", "scout_v4_cradle", "printed"), ("servo", "scout_v4_probe_servo", "bought"),
                       ("horn", "scout_v4_probe_horn", "bought"), ("camera", "scout_v4_probe_tilt", "bought"),
                       ("blade_a", "scout_v4_probe_mast_a", "bought"), ("blade_b", "scout_v4_probe_mast_b", "bought")):
        P[k] = (load(n), kind)
    rol = load("scout_v4_probe_roller").split(only_watertight=False)
    tb = sorted([m for m in rol if m.extents[1] < 5], key=lambda m: m.bounds[0][1])
    P["tube1"], P["tube2"] = (tb[0], "printed"), (tb[1], "printed")
    from parts.scout_v4_head_lib import Y_BRK, INSERT            # the idler's bracket on the head frame (its insert, the cradle's pivot screw)
    ax, ay, az = [float(v) for v in S.AXIS]; b0 = ay - Y_BRK - INSERT[1]
    P["pivot"] = (trimesh.util.concatenate([ycyl(1.5, b0, b0 + 10.0, ax, az), ycyl(2.75, b0 - 2.3, b0, ax, az)]), "hardware")   # M3 x 10 from outside the bracket
    P["head_insert"] = (ycyl(2.3, b0, b0 + INSERT[1], ax, az), "hardware")
    P["rod"] = ([m for m in rol if m.extents[1] > 20][0], "bought")
    P["saddle"] = ([m for m in rol if 8 < m.extents[1] < 9][0], "printed")
    top = DZ1 + 4.07                                              # the turret's floor at its bolts (artifacts/probe_bench_fit_check.md)
    P["bolts"] = (trimesh.util.concatenate([trimesh.util.concatenate([zcyl(1.5, top - 8.0, top, x, y), zcyl(2.75, top, top + 2.3, x, y)])
                                            for x, y in TURRET_BOLTS]), "hardware")
    P["inserts"] = (trimesh.util.concatenate([zcyl(2.3, DZ1 - 5.7, DZ1, x, y) for x, y in TURRET_BOLTS]), "hardware")
    P["axle"] = (trimesh.util.concatenate([ycyl(1.5, -29.0, 18.5, CX, CZ), ycyl(2.75, 18.5, 20.8, CX, CZ)]), "hardware")   # M3 x 50, head on the gear side
    P["nyloc"] = (ycyl(3.1, -30.0, -26.0, CX, CZ, n=6), "hardware")
    for b in ("blade_a", "blade_b"):                              # the stubs above the throat, drawn 14 long (to the head's clip)
        m = P[b][0].copy(); z0 = m.bounds[0][2]; m.vertices[:, 2] = z0 + (m.vertices[:, 2] - z0) * 14.0; P[b] = (m, "bought")
    return P


# ---------------------------------------------------------------------------------------------------------------- the pictures
def basis(az, el):
    a, e = np.radians(az), np.radians(el)
    c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])          # from the scene to the eye
    r = np.array([-np.sin(a), np.cos(a), 0.0]); u = np.cross(c, r)
    return r, u, c


def raster(items, r, u, c, scale, mid, W, H):
    """z-buffer, orthographic. items: (verts, faces, rgb). Returns colour, depth, id, normal-shade buffers."""
    col = np.ones((H, W, 3), np.float32); dep = np.full((H, W), -1e9, np.float32); ids = np.zeros((H, W), np.int32)
    nrm = np.zeros((H, W, 3), np.float32)
    L = (0.35 * r + 0.55 * u + 0.76 * c); L /= np.linalg.norm(L)
    for k, (V, F, rgb) in enumerate(items, 1):
        tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1); keep = ln > 1e-9; n[keep] /= ln[keep][:, None]
        keep &= (n @ c) > 1e-4
        sx = (V @ r - mid[0]) * scale + W / 2; sy = H / 2 - (V @ u - mid[1]) * scale; sz = V @ c
        shade = 0.42 + 0.58 * np.clip(n @ L, 0, 1)
        for f in np.nonzero(keep)[0]:
            i0, i1, i2 = F[f]
            x0, y0, x1, y1, x2, y2 = sx[i0], sy[i0], sx[i1], sy[i1], sx[i2], sy[i2]
            xa, xb = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), W - 1))
            ya, yb = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), H - 1))
            if xa > xb or ya > yb:
                continue
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-9:
                continue
            X = np.arange(xa, xb + 1, dtype=np.float32)[None, :] + 0.5; Y = np.arange(ya, yb + 1, dtype=np.float32)[:, None] + 0.5
            l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
            l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
            l2 = 1.0 - l0 - l1
            z = l0 * sz[i0] + l1 * sz[i1] + l2 * sz[i2]
            sub = dep[ya:yb + 1, xa:xb + 1]
            m = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4) & (z > sub)
            if not m.any():
                continue
            sub[m] = z[m]
            col[ya:yb + 1, xa:xb + 1][m] = np.asarray(rgb, np.float32) * shade[f]
            ids[ya:yb + 1, xa:xb + 1][m] = k
            nrm[ya:yb + 1, xa:xb + 1][m] = n[f]
    return col, dep, ids, nrm


def lines(col, dep, ids, nrm, scale, strength=0.72):
    """dark outlines where the part, the depth or the face's direction changes"""
    e = np.zeros(ids.shape, bool)
    for ax in (0, 1):
        a = [slice(None)] * 2; b = [slice(None)] * 2; a[ax] = slice(1, None); b[ax] = slice(None, -1); a, b = tuple(a), tuple(b)
        d = (ids[a] != ids[b]) | (np.abs(dep[a] - dep[b]) > 1.2) | (((nrm[a] * nrm[b]).sum(-1) < 0.45) & (ids[a] > 0))
        e[a] |= d
    out = col.copy(); out[e] *= (1.0 - strength)
    return out


def picture(P, shown, new=(), ghost=(), move=None, az=-60, el=25, W=1040, H=700, ss=2, pad=1.12, fit=None):
    """shown: ids already on (grey); new: this step's ids (coloured), each moved back by move[id] (mm, drone frame); ghost: ids drawn
    see-through over the rest. Returns the PIL picture and where each new part's middle lands, there and home (pixels)."""
    move = move or {}
    r, u, c = basis(az, el)
    solid, ghosts, marks = [], [], {}
    for k in list(shown) + list(new):
        m, kind = P[k]; V = np.asarray(m.vertices, np.float64) + np.asarray(move.get(k, (0, 0, 0)))
        rgb = COL[kind] if k in new else COL["done"]
        (ghosts if k in ghost else solid).append((k, V, np.asarray(m.faces), rgb))
    allv = np.vstack([V for k, V, F, rgb in solid + ghosts if (fit is None or k in fit)])
    px, py = allv @ r, allv @ u
    mid = ((px.min() + px.max()) / 2, (py.min() + py.max()) / 2)
    Ws, Hs = W * ss, H * ss
    scale = min(Ws / ((px.max() - px.min()) * pad), Hs / ((py.max() - py.min()) * pad))
    col, dep, ids, nrm = raster([(V, F, rgb) for k, V, F, rgb in solid], r, u, c, scale, mid, Ws, Hs)
    img = lines(col, dep, ids, nrm, scale)
    if ghosts:
        gc, gd, gi, gn = raster([(V, F, rgb) for k, V, F, rgb in ghosts], r, u, c, scale, mid, Ws, Hs)
        gl = lines(gc, gd, gi, gn, scale, 0.6)
        front = (gi > 0) & (gd > dep)
        a = np.where(front, 0.34, 0.0)[..., None]
        edge = front & (np.abs(gl - gc).sum(-1) > 0.05)
        a[edge] = 0.75
        img = img * (1 - a) + gl * a
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    for k in new:
        m, kind = P[k]; home = np.asarray(m.vertices).mean(0); there = home + np.asarray(move.get(k, (0, 0, 0)))
        to_px = lambda p: (float((p @ r - mid[0]) * scale + Ws / 2) / ss, float(Hs / 2 - (p @ u - mid[1]) * scale) / ss)
        marks[k] = (to_px(there), to_px(home), kind)
    return out, marks


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(img, marks, labels, title=None):
    """an arrow from each pulled-back part to its seat and a numbered tag on it; labels: {id: number}"""
    d = ImageDraw.Draw(img, "RGBA"); f = font(19)
    for k, (there, home, kind) in marks.items():
        if k not in labels:
            continue
        rgb = tuple(int(v * 255 * 0.75) for v in COL[kind])
        dx, dy = home[0] - there[0], home[1] - there[1]; ln = (dx * dx + dy * dy) ** 0.5
        if ln > 26:
            ux, uy = dx / ln, dy / ln
            a = (there[0] + ux * 17, there[1] + uy * 17); b = (home[0] - ux * 4, home[1] - uy * 4)
            d.line([a, b], fill=rgb + (235,), width=4)
            d.polygon([b, (b[0] - ux * 15 - uy * 7, b[1] - uy * 15 + ux * 7), (b[0] - ux * 15 + uy * 7, b[1] - uy * 15 - ux * 7)], fill=rgb + (255,))
    for k, (there, home, kind) in marks.items():
        if k not in labels:
            continue
        rgb = tuple(int(v * 255 * 0.75) for v in COL[kind]); t = str(labels[k]); x, y = there
        d.ellipse([x - 15, y - 15, x + 15, y + 15], fill=(255, 255, 255, 255), outline=rgb + (255,), width=3)
        w = d.textlength(t, font=f); d.text((x - w / 2, y - 11), t, fill=(20, 24, 30, 255), font=f)
    return img


def jpeg(img):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=86, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


# ---------------------------------------------------------------------------------------------------------------- the page
CSS = """
:root{--bg:#f4f6f8;--panel:#fff;--ink:#15191f;--muted:#5b6573;--line:#d6dce4;--accent:#2b56c9;--tape:#f2c200;--tape-ink:#3a2f00;--warn:#a4570b;
--printed:#2c9a93;--bought:#e08a1e;--hardware:#5c73a0;--sans:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;--mono:ui-monospace,Consolas,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--panel:#1a1f26;--ink:#e7ebf0;--muted:#9aa5b3;--line:#2c343e;--accent:#7d9cf2;--tape:#e8b900;--tape-ink:#1c1600;--warn:#e3a35b;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:400 17px/1.5 var(--sans);padding:26px 18px 70px}
.wrap{max-width:980px;margin:0 auto;display:grid;gap:30px}h1,h2,h3{margin:0;line-height:1.15}h1{font-size:2.1rem}h2{font-size:1.45rem;border-bottom:2px solid var(--ink);padding-bottom:6px}
h3{font-size:1.22rem}p{margin:0}.lede{color:var(--muted);max-width:70ch}
.tag{display:inline-block;font:600 .72rem/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;background:var(--tape);color:var(--tape-ink);padding:5px 8px;border-radius:3px}
header,section{display:grid;gap:14px}header{justify-items:start}.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}.box h3{margin-bottom:8px;font-size:1.05rem}
ul.plain{list-style:none;margin:0;padding:0;display:grid;gap:6px}ul.plain li{display:flex;gap:10px;align-items:baseline}
.qty{font:600 .84rem var(--mono);color:var(--accent);min-width:2.4em;flex:none}
.note{border-left:5px solid var(--tape);background:var(--panel);padding:11px 15px;border-radius:0 8px 8px 0}.note.warn{border-left-color:var(--warn)}
.note b.h{display:block;margin-bottom:2px}
.steps{list-style:none;margin:0;padding:0;display:grid;gap:18px}
.step{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;display:grid;gap:12px}
.step.done{opacity:.55}.head{display:flex;align-items:center;gap:14px;cursor:pointer}
.n{font:700 1.7rem/1 var(--sans);color:var(--accent);min-width:1.5em}.head input{width:22px;height:22px;accent-color:var(--accent);margin-left:auto;flex:none}
.body{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;align-items:start}.body.nopic{grid-template-columns:1fr}
@media (max-width:760px){.body{grid-template-columns:1fr}}
figure{margin:0}figure img{display:block;width:100%;height:auto;border-radius:6px;border:1px solid var(--line);background:#fff}
figcaption{font-size:.84rem;color:var(--muted);margin-top:6px}
.key{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;font-size:.88rem}.key span b{display:inline-block;min-width:1.5em;height:1.5em;line-height:1.4em;text-align:center;border:2px solid var(--muted);border-radius:50%;margin-right:5px;font-size:.8rem}
.key .printed b{border-color:var(--printed)}.key .bought b{border-color:var(--bought)}.key .hardware b{border-color:var(--hardware)}
ol.do{margin:0;padding-left:1.25em;display:grid;gap:7px}.uses{font-size:.86rem;color:var(--muted);margin-top:10px}
.leg{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.88rem;color:var(--muted)}.leg i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px}
"""
JS = """
(function(){var K='probe-bench-guide-v1',st={};try{st=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
document.querySelectorAll('.step input').forEach(function(b){b.checked=!!st[b.id];b.closest('.step').classList.toggle('done',b.checked);
b.addEventListener('change',function(){b.closest('.step').classList.toggle('done',b.checked);st[b.id]=b.checked;try{localStorage.setItem(K,JSON.stringify(st))}catch(e){}})})})();
"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rich(s):
    """**bold** only"""
    out, on = "", False
    for i, part in enumerate(esc(s).split("**")):
        out += ("<b>" if i % 2 else "") + part + ("</b>" if i % 2 else "")
    return out


def page(G, pics):
    h = [f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
         f"<title>{esc(G['title'])}</title><style>{CSS}</style></head><body><div class=\"wrap\">",
         f"<header><span class=\"tag\">{esc(G['tag'])}</span><h1>{esc(G['title'])}</h1><p class=\"lede\">{rich(G['lede'])}</p>"
         "<div class=\"leg\"><span><i style=\"background:var(--printed)\"></i>printed part</span><span><i style=\"background:var(--bought)\"></i>bought part</span>"
         "<span><i style=\"background:var(--hardware)\"></i>screws and inserts</span><span><i style=\"background:#c9ccd1\"></i>already on</span></div></header>"]
    if pics.get("hero"):
        h.append(f"<figure><img alt=\"The finished bench build\" src=\"{pics['hero']['src']}\"><figcaption>{rich(G.get('hero_caption', ''))}</figcaption></figure>")
    if G.get("first"):
        h.append("<section><h2>" + esc(G["first_title"]) + "</h2>" + "".join(
            f"<div class=\"note warn\"><b class=\"h\">{rich(n['head'])}</b>{rich(n['text'])}</div>" for n in G["first"]) + "</section>")
    h.append("<section><h2>What you need</h2><div class=\"cols\">")
    for box in G["need"]:
        h.append(f"<div class=\"box\"><h3>{esc(box['title'])}</h3><ul class=\"plain\">" + "".join(
            f"<li><span class=\"qty\">{esc(q)}</span><span>{rich(t)}</span></li>" for q, t in box["items"]) + "</ul></div>")
    h.append("</div></section><section><h2>Build it</h2><ol class=\"steps\">")
    for i, s in enumerate(G["steps"], 1):
        pic = pics.get(s["id"])
        if s.get("group"):
            h.append(f"<li><h2>{esc(s['group'])}</h2>" + (f"<p class=\"lede\" style=\"margin-top:8px\">{rich(s['group_note'])}</p>" if s.get("group_note") else "") + "</li>")
        h.append(f"<li class=\"step\"><label class=\"head\"><span class=\"n\">{i}</span><h3>{esc(s['title'])}</h3><input type=\"checkbox\" id=\"{s['id']}\"></label>")
        h.append(f"<div class=\"body{'' if pic else ' nopic'}\">")
        if pic:
            key = "".join(f"<span class=\"{kind}\"><b>{n}</b>{esc(name)}</span>" for n, name, kind in pic["key"])
            h.append(f"<figure><img alt=\"{esc(s['title'])}\" src=\"{pic['src']}\"><div class=\"key\">{key}</div>"
                     + (f"<figcaption>{rich(s['caption'])}</figcaption>" if s.get("caption") else "") + "</figure>")
        h.append("<div><ol class=\"do\">" + "".join(f"<li>{rich(t)}</li>" for t in s["do"]) + "</ol>"
                 + "".join(f"<div class=\"note warn\" style=\"margin-top:10px\"><b class=\"h\">{rich(n['head'])}</b>{rich(n['text'])}</div>" for n in s.get("notes", []))
                 + (f"<div class=\"uses\">Uses: {rich(s['uses'])}</div>" if s.get("uses") else "") + "</div></div></li>")
    h.append("</ol></section>")
    for sec in G.get("after", []):
        h.append(f"<section><h2>{esc(sec['title'])}</h2><div class=\"box\"><ol class=\"do\">" + "".join(f"<li>{rich(t)}</li>" for t in sec["items"]) + "</ol></div></section>")
    h.append(f"<p class=\"lede\">{rich(G['foot'])}</p></div><script>{JS}</script></body></html>")
    return "".join(h)


def main(argv):
    G = json.load(open(os.path.join(ROOT, "tools", "probe_bench_guide_steps.json"), encoding="utf-8"))
    P = parts(); pics = {}
    out_dir = argv[argv.index("--pics") + 1] if "--pics" in argv else None
    only = argv[argv.index("--only") + 1].split(",") if "--only" in argv else None
    for s in [dict(id="hero", view=G.get("hero"))] + G["steps"]:
        v = s.get("view")
        if not v or (only and s["id"] not in only):
            continue
        new = [k for k, _ in v["new"]]
        img, marks = picture(P, v.get("shown", []), new, v.get("ghost", []), {k: tuple(m) for k, m in v.get("move", {}).items()},
                             v.get("az", -60), v.get("el", 25), fit=v.get("fit"), pad=v.get("pad", 1.12))
        labels = {k: i for i, (k, name) in enumerate(v["new"], 1) if name}
        img = annotate(img, marks, labels)
        pics[s["id"]] = dict(src=jpeg(img), key=[(i, name, P[k][1]) for i, (k, name) in enumerate(v["new"], 1) if name])
        if out_dir:
            os.makedirs(out_dir, exist_ok=True); img.save(os.path.join(out_dir, s["id"] + ".png"))
        print("picture", s["id"])
    if only:
        return
    dst = os.path.join(ROOT, "artifacts", "probe_bench_guide.html")
    open(dst, "w", encoding="utf-8").write(page(G, pics))
    print(dst, f"{os.path.getsize(dst) / 1e6:.2f} MB, {len(G['steps'])} steps, {len(pics)} pictures")


if __name__ == "__main__":
    main(sys.argv[1:])
