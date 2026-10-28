"""One picture per drone for the repo's front page: opens site/<drone>.html (the built viewer) in headless Chromium and screenshots
it to site/<drone>.png, turntable stopped at frame 0, in the view the assembly names as readme_view (default iso). The site workflow runs it after build_site, so the PNG on Pages is always the current assembly and the
README, which embeds it by URL, never changes. Spark, 28 Oct: visitors see the current progress with nothing done by hand.

    python tools/snap_site.py [drone ...]       default: every systems/<drone>.assembly.json that has a title
"""
import glob, json, os, sys, time
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, SETTLE_S = 1400, 900, 4.0


def drones():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "systems", "*.assembly.json"))):
        d = os.path.basename(f)[: -len(".assembly.json")]
        if json.load(open(f)).get("title") and os.path.exists(os.path.join(ROOT, "site", f"{d}.html")):
            out.append(d)
    return out


def main(names):
    with sync_playwright() as pw:
        exe = os.environ.get("PW_CHROMIUM")                         # a preinstalled Chromium (a session's); CI installs its own
        b = pw.chromium.launch(executable_path=exe, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--no-sandbox"])
        pg = b.new_page(viewport=dict(width=W, height=H))
        for d in names:
            path = os.path.join(ROOT, "site", f"{d}.html")
            view = json.load(open(os.path.join(ROOT, "systems", f"{d}.assembly.json"))).get("readme_view", "iso")
            pg.goto("file://" + path, wait_until="load", timeout=120000)
            time.sleep(SETTLE_S)                                    # meshes decode and the first frames draw
            pg.click("#play")                                       # stop the turntable at frame 0: the drone unrotated, x forward
            pg.evaluate("() => { const s = document.getElementById('scrub'); s.value = 0; s.dispatchEvent(new Event('input')); }")
            pg.get_by_role("button", name=view, exact=True).click() # the assembly's readme_view (Overseer: front, the L2 and cameras to the viewer)
            time.sleep(1.0)
            out = os.path.join(ROOT, "site", f"{d}.png")
            pg.screenshot(path=out)
            print(f"{d}: {out} {os.path.getsize(out) // 1024} kB")
        b.close()


if __name__ == "__main__":
    main(sys.argv[1:] or drones())
