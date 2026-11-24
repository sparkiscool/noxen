"""Noxen scan viewer: opens artifacts/container_scan.html as its own full-screen program window.

A page inside another page (the Noxy window) cannot capture the mouse; a window of its own can. This starts
Chrome (else Edge) in app mode, with a profile of its own, full screen on the big monitor (the one left of the
main display; the main display when there is only one). In the window: Look mode locks the mouse, Esc frees it.

    python tools/scan_viewer.py            # open it
    python tools/scan_viewer.py --shortcut # put "Noxen Scan Viewer" on the desktop
"""
import ctypes, os, subprocess, sys, tempfile
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(os.path.dirname(HERE), "artifacts", "container_scan.html")
PROFILE = r"E:\noxen_work\scan_viewer_profile" if os.path.isdir(r"E:\noxen_work") else os.path.join(tempfile.gettempdir(), "noxen_scan_viewer")
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]


def monitors():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass
    out = []

    class RECT(ctypes.Structure):
        _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]
    cb = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_double)

    def f(h, dc, r, d):
        out.append((r.contents.l, r.contents.t, r.contents.r - r.contents.l, r.contents.b - r.contents.t))
        return 1
    ctypes.windll.user32.EnumDisplayMonitors(None, None, cb(f), 0)
    return out


def big_monitor():
    m = monitors() or [(0, 0, 1920, 1080)]
    left = [x for x in m if x[0] < 0]                 # the big monitor stands left of the main display
    return min(left, key=lambda x: x[0]) if left else next((x for x in m if x[0] == 0 and x[1] == 0), m[0])


def launch(frag=""):
    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if not exe:
        raise SystemExit("no Chrome or Edge found")
    x, y, w, h = big_monitor()
    os.makedirs(PROFILE, exist_ok=True)
    url = "file:///" + PAGE.replace("\\", "/") + (("#" + frag) if frag else "")
    subprocess.Popen([exe, "--app=" + url, "--user-data-dir=" + PROFILE, "--no-first-run", "--no-default-browser-check",
                      "--window-position=%d,%d" % (x, y), "--window-size=%d,%d" % (w, h), "--start-fullscreen",
                      "--disable-features=Translate", "--disable-background-timer-throttling"],
                     creationflags=0x00000008 | 0x00000200)     # detached


def shortcut():
    vbs = os.path.join(tempfile.gettempdir(), "noxen_scan_lnk.vbs")
    pyw = sys.executable.replace("python.exe", "pythonw.exe")
    with open(vbs, "w") as f:
        f.write('Set sh = CreateObject("WScript.Shell")\r\n'
                'Set l = sh.CreateShortcut(sh.SpecialFolders("Desktop") & "\\Noxen Scan Viewer.lnk")\r\n'
                'l.TargetPath = "%s"\r\nl.Arguments = """%s"""\r\nl.WorkingDirectory = "%s"\r\n'
                'l.Description = "Noxen scan to BIM, full screen on the big monitor"\r\nl.Save\r\n' % (pyw, os.path.abspath(__file__), os.path.dirname(HERE)))
    subprocess.run(["cscript", "//nologo", vbs], check=True)
    os.remove(vbs)
    print("shortcut written on the desktop")


if __name__ == "__main__":
    shortcut() if "--shortcut" in sys.argv else launch(sys.argv[sys.argv.index("--hash") + 1] if "--hash" in sys.argv else "")
