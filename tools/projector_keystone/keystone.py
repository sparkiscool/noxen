r"""Projector keystone: captures ONE display and shows it warped (four-corner perspective) on the same display.

Standalone, free parts only (dxcam = Desktop Duplication, glfw + moderngl = OpenGL warp on the GPU).
Other displays are never touched. The overlay is always-on-top, click-through, and excluded from capture
(SetWindowDisplayAffinity), so it never films itself.

  python keystone.py                  run (adjust mode off)
  python keystone.py --adjust         start in adjust mode
  python keystone.py --autostart on|off   start with Windows (HKCU Run, pythonw)
  python keystone.py --reset          corners back to straight
  python keystone.py --display \\\\.\\DISPLAY3   pick the display (default: the one named by config, else the S2-TEK TV STK531A)

Hotkey (global): Ctrl+Alt+K  adjust mode on/off (off saves).   Ctrl+Alt+Q  quit.
Adjust mode: drag a corner with the mouse, or 1-4 / Tab pick a corner (TL TR BR BL), arrows nudge 1 px
(Shift 10 px, Ctrl 0.2 px), R reset, Esc leave. Corners are saved on leaving adjust mode and on quit
to %APPDATA%\\ProjectorKeystone\\config.json.
Other programs: create %APPDATA%\ProjectorKeystone\adjust_on.flag / adjust_off.flag to show or hide the adjust handles.
Clicks: with the warp on, a click on the projector is moved to the desktop point it shows.
"""
import ctypes, json, os, sys, threading, time
from ctypes import wintypes as w
import numpy as np

ctypes.windll.shcore.SetProcessDpiAwareness(2)
u32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32
CFG_DIR = os.path.join(os.environ.get("APPDATA", "."), "ProjectorKeystone")
CFG = os.path.join(CFG_DIR, "config.json")
AWAY = os.path.join(CFG_DIR, "away.flag")         # (p1265) Noxy's away mode: black, no capture
AW = {"on": False, "t": 0.0}
DEFAULT = {"display": None, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "fix_clicks": True}


BAK_DIR = os.path.join(os.environ.get("LOCALAPPDATA", "."), "ProjectorKeystone")
BAK = os.path.join(BAK_DIR, "corners_backup.json")  # second copy, different folder: survives a wiped or overwritten config
PROFILE = os.path.join(BAK_DIR, "profile.json")  # (p1262) HIS corners (Spark 24 Nov: "make that the default"): R / --reset come back here
FPS = 30                                          # (p1262) at most this many pictures a second: the projector needs no more


def profile():
    try:
        p = json.load(open(PROFILE, encoding="utf-8"))
        if len(p["corners"]) == 4:
            return [list(map(float, q)) for q in p["corners"]]
    except Exception:
        pass
    return [list(q) for q in DEFAULT["corners"]]


def save_profile(corners):
    os.makedirs(BAK_DIR, exist_ok=True)
    json.dump({"corners": [list(map(float, q)) for q in corners], "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(PROFILE + ".tmp", "w", encoding="utf-8"), indent=1)
    os.replace(PROFILE + ".tmp", PROFILE)


def _is_straight(c):
    return all(abs(a - b) < 1e-9 for p, q in zip(c["corners"], DEFAULT["corners"]) for a, b in zip(p, q))


def _read(path):
    try:
        c = json.load(open(path, encoding="utf-8"))
        return {**DEFAULT, **c}
    except Exception:
        return None


def load():
    c = _read(CFG)
    b = _read(BAK)
    if c is None:
        c = b or dict(DEFAULT)
        if _is_straight(c): c = {**c, "corners": profile()}      # (p1262) nothing saved: his profile
        return c
    # the config went straight without an explicit reset (overwritten by something else): bring the saved corners back
    if b and _is_straight(c) and not c.get("reset") and not _is_straight(b):
        c["corners"] = b["corners"]
    if _is_straight(c) and not c.get("reset"):
        c["corners"] = profile()                                  # (p1262)
    return c


def save(c):
    c = dict(c)
    if _is_straight(c): c["reset"] = True   # only an explicit reset (R, --reset) saves straight corners
    else: c.pop("reset", None)
    for d, f in ((CFG_DIR, CFG), (BAK_DIR, BAK)):
        os.makedirs(d, exist_ok=True)
        json.dump(c, open(f + ".tmp", "w", encoding="utf-8"), indent=1)
        os.replace(f + ".tmp", f)
    if not _is_straight(c):  # dated history of every distinct set of corners
        h = os.path.join(BAK_DIR, "history.jsonl")
        try:
            last = open(h, encoding="utf-8").read().splitlines()[-1]
        except Exception:
            last = ""
        line = json.dumps(c["corners"])
        if not last.endswith(line):
            open(h, "a", encoding="utf-8").write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + chr(10))


def autostart(on):
    import winreg
    k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE)
    if on:
        py = sys.executable.replace("python.exe", "pythonw.exe")
        winreg.SetValueEx(k, "ProjectorKeystone", 0, winreg.REG_SZ, f'"{py}" "{os.path.abspath(__file__)}"')
    else:
        try: winreg.DeleteValue(k, "ProjectorKeystone")
        except FileNotFoundError: pass
    print("autostart", "on" if on else "off")


def homography(src, dst):
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.append(Y)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1).reshape(3, 3)


class RECT(ctypes.Structure):
    _fields_ = [("l", w.LONG), ("t", w.LONG), ("r", w.LONG), ("b", w.LONG)]


def find_output(name):
    """-> (dxcam device_idx, output_idx, name, (x, y, w, h)) for the display, by Windows device name."""
    import dxcam
    f = dxcam.__factory if hasattr(dxcam, "__factory") else None
    fac = dxcam.DXFactory() if f is None else f
    # monitor ids: prefer the S2-TEK TV (STK) when no name given
    if name is None:
        name = tv_display() or r"\\.\DISPLAY3"
    for di, outs in enumerate(fac.outputs):
        for oi, o in enumerate(outs):
            if o.devicename.lower() == name.lower():
                d = o.desc.DesktopCoordinates
                return di, oi, o.devicename, (d.left, d.top, d.right - d.left, d.bottom - d.top)
    raise SystemExit("display not found: " + name)


def tv_display():
    class DD(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("n", w.WCHAR * 32), ("s", w.WCHAR * 128), ("f", w.DWORD), ("id", w.WCHAR * 128), ("k", w.WCHAR * 128)]
    i = 0
    while True:
        d = DD(); d.cb = ctypes.sizeof(d)
        if not u32.EnumDisplayDevicesW(None, i, ctypes.byref(d), 0): return None
        i += 1
        if d.f & 1:
            m = DD(); m.cb = ctypes.sizeof(m)
            if u32.EnumDisplayDevicesW(d.n, 0, ctypes.byref(m), 0) and "STK" in m.id.upper():
                return d.n


class Capture(threading.Thread):
    def __init__(self, di, oi):
        super().__init__(daemon=True)
        self.di, self.oi = di, oi
        self.frame = None; self.seq = 0; self.lock = threading.Lock(); self.run_ = True

    def run(self):
        import dxcam
        cam = None
        while self.run_:
            if AW["on"]:                              # (p1265) away: no pictures taken
                time.sleep(0.5); continue
            try:
                if cam is None:
                    cam = dxcam.create(self.di, self.oi, output_color="BGRA", processor_backend="numpy")
                t0 = time.perf_counter()
                f = cam.grab(new_frame_only=True)
                if f is not None:
                    with self.lock:
                        self.frame = f; self.seq += 1
                time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))   # (p1262) was a 2 ms busy poll
            except Exception as e:
                print("capture:", e); cam = None; time.sleep(0.5)


VS = """#version 330
in vec2 pos; out vec2 p;
void main(){ gl_Position=vec4(pos,0,1); p=vec2(pos.x*0.5+0.5, 0.5-pos.y*0.5); }"""
FS = """#version 330
uniform sampler2D tex; uniform mat3 Hi; in vec2 p; out vec4 c;
void main(){
  vec3 q = Hi*vec3(p,1.0); vec2 uv=q.xy/q.z;
  if(q.z<=0.0||uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0){ c=vec4(0,0,0,1); return; }
  c=vec4(texture(tex,uv).rgb,1.0);
}"""
VS2 = """#version 330
in vec2 pos; in vec3 col; out vec3 v; void main(){ gl_Position=vec4(pos,0,1); v=col; }"""
FS2 = """#version 330
in vec3 v; out vec4 c; void main(){ c=vec4(v,1.0); }"""


def selftest(ctx, fw, fh, corners, mw, mh):
    """Pixel check of the warp on the hidden window: the desktop's corners must land on the configured corners."""
    img = np.frombuffer(ctx.screen.read((0, 0, fw, fh), components=3), np.uint8).reshape(fh, fw, 3)[::-1]
    ok = True
    for (cx, cy), n in zip(corners, ("TL", "TR", "BR", "BL")):
        px, py = int(np.clip(cx * (fw - 1), 0, fw - 1)), int(np.clip(cy * (fh - 1), 0, fh - 1))
        ix, iy = int(np.clip(px + (3 if cx < .5 else -3), 0, fw - 1)), int(np.clip(py + (3 if cy < .5 else -3), 0, fh - 1))
        print(n, "corner pixel inside the picture:", img[iy, ix].tolist())
        ok &= img[iy, ix].sum() > 0
    print("SELFTEST", "PASS" if ok else "FAIL")


def main():
    a = sys.argv[1:]
    if "--autostart" in a:
        autostart(a[a.index("--autostart") + 1] == "on"); return
    cfg = load()
    if "--straight" in a: cfg["corners"] = [list(p) for p in DEFAULT["corners"]]; save(cfg); print("straight"); return
    if "--reset" in a: cfg["corners"] = profile(); save(cfg); print("reset to the profile"); return   # (p1262)
    if "--save-profile" in a: save_profile(cfg["corners"]); print("profile saved", cfg["corners"]); return
    if "--display" in a: cfg["display"] = a[a.index("--display") + 1]
    di, oi, dname, (mx, my, mw, mh) = find_output(cfg["display"])
    print("display", dname, mx, my, mw, mh)
    import glfw, moderngl
    glfw.init()
    glfw.window_hint(glfw.DECORATED, False); glfw.window_hint(glfw.FLOATING, True)
    glfw.window_hint(glfw.FOCUS_ON_SHOW, False); glfw.window_hint(glfw.FOCUSED, False)
    glfw.window_hint(glfw.AUTO_ICONIFY, False); glfw.window_hint(glfw.RESIZABLE, False)
    glfw.window_hint(glfw.MOUSE_PASSTHROUGH, True)
    glfw.window_hint(glfw.SCALE_TO_MONITOR, False)  # never let a DPI change resize the overlay: it must be exactly the display
    glfw.window_hint(glfw.VISIBLE, False)  # placed and sized first, shown after: no flash of a wrong-sized window on the projector
    win = glfw.create_window(mw, mh, "Projector Keystone", None, None)
    glfw.set_window_pos(win, mx, my); glfw.set_window_size(win, mw, mh)
    SELFTEST = bool(os.environ.get("KEYSTONE_SELFTEST"))  # hidden window, render once, check pixels, exit: nothing reaches any screen
    if not SELFTEST: glfw.show_window(win)
    glfw.set_window_pos(win, mx, my); glfw.set_window_size(win, mw, mh)
    glfw.make_context_current(win); glfw.swap_interval(1)
    hwnd = glfw.get_win32_window(win)
    if not os.environ.get("KEYSTONE_SHOWN_IN_CAPTURE"):  # test only: lets a screenshot see the overlay (then it films itself)
        u32.SetWindowDisplayAffinity(hwnd, 0x11)  # exclude from capture: no feedback loop
    ctx = moderngl.create_context()
    fbw, fbh = glfw.get_framebuffer_size(win)
    ctx.viewport = (0, 0, fbw, fbh)  # the picture fills the whole display whatever the window/DPI reports
    print("window", glfw.get_window_size(win), "framebuffer", (fbw, fbh), "display", (mw, mh))
    prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
    prog2 = ctx.program(vertex_shader=VS2, fragment_shader=FS2)
    vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4"))
    vao = ctx.vertex_array(prog, [(vbo, "2f", "pos")])
    vbo2 = ctx.buffer(reserve=65536)
    vao2 = ctx.vertex_array(prog2, [(vbo2, "2f 3f", "pos", "col")])
    def new_tex(tw, th):
        t = ctx.texture((tw, th), 4); t.swizzle = "BGRA"; t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        t.repeat_x = t.repeat_y = False
        return t
    tex = new_tex(mw, mh)
    cap = Capture(di, oi)
    TEST = None
    if "--test-pattern" in a:  # no capture: a grid, to see the warp in a screenshot
        yy, xx = np.mgrid[0:mh, 0:mw]
        TEST = np.zeros((mh, mw, 4), np.uint8); TEST[..., 3] = 255
        TEST[..., 0] = xx * 255 // mw; TEST[..., 1] = yy * 255 // mh; TEST[..., 2] = 90
        g = (xx % 120 < 4) | (yy % 120 < 4); TEST[g] = 255
    else:
        cap.start()

    corners = np.array(cfg["corners"], float)  # where the desktop's TL,TR,BR,BL show, 0..1 of the display
    S = {"adjust": "--adjust" in a, "sel": 0, "drag": False, "seq": 0, "quit": False, "dirty": True}

    def Hinv():
        return np.linalg.inv(homography([(0, 0), (1, 0), (1, 1), (0, 1)], corners))

    # ---- click remap (low-level mouse hook, own thread) ----
    HOOK = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, w.WPARAM, w.LPARAM)

    class MS(ctypes.Structure):
        _fields_ = [("x", w.LONG), ("y", w.LONG), ("data", w.DWORD), ("flags", w.DWORD), ("t", w.DWORD), ("extra", ctypes.c_size_t)]

    def straight():
        return np.allclose(corners, DEFAULT["corners"], atol=1e-4)

    def to_desktop(px, py):
        q = Hinv() @ np.array([(px - mx) / mw, (py - my) / mh, 1.0])
        return int(round(mx + q[0] / q[2] * mw)), int(round(my + q[1] / q[2] * mh))

    TEST_INJ = bool(os.environ.get("KEYSTONE_TEST_INJECT"))  # test only: remap injected clicks too
    EV = {0x201: 0x0002, 0x202: 0x0004, 0x204: 0x0008, 0x205: 0x0010, 0x207: 0x0020, 0x208: 0x0040}

    def hook(n, wp, lp):
        try:
            if n >= 0 and cfg["fix_clicks"] and not S["adjust"] and wp in EV and not straight():
                m = ctypes.cast(lp, ctypes.POINTER(MS)).contents
                if (m.extra != 0x4B53) and (not (m.flags & 1) or TEST_INJ) and mx <= m.x < mx + mw and my <= m.y < my + mh:
                    dx, dy = to_desktop(m.x, m.y)
                    u32.SetCursorPos(dx, dy)
                    u32.mouse_event(EV[wp], 0, 0, 0, 0x4B53)  # injected, marked: passes our own hook
                    u32.SetCursorPos(m.x, m.y)
                    return 1
        except Exception:
            pass
        return u32.CallNextHookEx(0, n, wp, lp)
    hook_c = HOOK(hook)

    # (p1270) 64-bit handles need explicit types: with ctypes' default int the module handle was cut to 32 bits, the hook never
    # installed, and clicks landed on the unwarped point (shifted against the picture)
    u32.SetWindowsHookExW.restype = ctypes.c_void_p
    u32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, w.DWORD]
    u32.CallNextHookEx.restype = ctypes.c_ssize_t
    u32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, w.WPARAM, w.LPARAM]
    k32.GetModuleHandleW.restype = ctypes.c_void_p
    k32.GetModuleHandleW.argtypes = [w.LPCWSTR]

    def hook_thread():
        h = u32.SetWindowsHookExW(14, hook_c, k32.GetModuleHandleW(None), 0)
        print("click hook", "installed" if h else "FAILED %d" % ctypes.get_last_error(), flush=True)
        msg = w.MSG()
        while u32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
            u32.TranslateMessage(ctypes.byref(msg)); u32.DispatchMessageW(ctypes.byref(msg))
    threading.Thread(target=hook_thread, daemon=True).start()

    def set_adjust(on):
        S["adjust"] = on
        glfw.set_window_attrib(win, glfw.MOUSE_PASSTHROUGH, not on)
        if on:
            u32.SetForegroundWindow(hwnd); glfw.focus_window(win)
        else:
            cfg["corners"] = corners.tolist(); save(cfg)
        S["dirty"] = True
    if S["adjust"]: set_adjust(True)

    def nudge(dx, dy, step):
        corners[S["sel"]] += (dx * step / mw, dy * step / mh)

    def cursor_norm():
        x, y = glfw.get_cursor_pos(win)
        return np.array([x / mw, y / mh])

    def on_mouse(win_, btn, act, mods):
        if not S["adjust"] or btn != 0: return
        if act == glfw.PRESS:
            c = cursor_norm()
            d = np.linalg.norm((corners - c) * [mw, mh], axis=1)
            S["sel"] = int(d.argmin()); S["drag"] = True
        else:
            S["drag"] = False
    glfw.set_mouse_button_callback(win, on_mouse)

    def on_key(win_, key, sc, act, mods):
        if not S["adjust"] or act == glfw.RELEASE: return
        step = 10 if mods & glfw.MOD_SHIFT else (0.2 if mods & glfw.MOD_CONTROL else 1)
        if key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4): S["sel"] = key - glfw.KEY_1
        elif key == glfw.KEY_TAB: S["sel"] = (S["sel"] + 1) % 4
        elif key == glfw.KEY_LEFT: nudge(-1, 0, step)
        elif key == glfw.KEY_RIGHT: nudge(1, 0, step)
        elif key == glfw.KEY_UP: nudge(0, -1, step)
        elif key == glfw.KEY_DOWN: nudge(0, 1, step)
        elif key == glfw.KEY_R: corners[:] = profile()                  # (p1262) back to his profile, not straight
        elif key == glfw.KEY_P: save_profile(corners.tolist()); print("profile saved")
        elif key == glfw.KEY_ESCAPE: set_adjust(False)
        S["dirty"] = True
    glfw.set_key_callback(win, on_key)

    def quad(x0, y0, x1, y1, col, out):
        for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)):
            out.extend([p[0] / mw * 2 - 1, 1 - p[1] / mh * 2, *col])

    def seg(p, q, th, col, out):
        p = np.array(p) * [mw, mh]; q = np.array(q) * [mw, mh]
        d = q - p; L = np.linalg.norm(d)
        if L < 1e-6: return
        n = np.array([-d[1], d[0]]) / L * th / 2
        a, b, c, e = p + n, p - n, q - n, q + n
        for P in (a, b, c, a, c, e):
            out.extend([P[0] / mw * 2 - 1, 1 - P[1] / mh * 2, *col])

    def draw_overlay():
        v = []
        P = corners
        for i in range(4):
            seg(P[i], P[(i + 1) % 4], 3, (0.1, 1, 0.3), v)
        for k in (0.25, 0.5, 0.75):  # alignment grid
            seg(P[0] + (P[1] - P[0]) * k, P[3] + (P[2] - P[3]) * k, 1.5, (0.1, 0.8, 0.9), v)
            seg(P[0] + (P[3] - P[0]) * k, P[1] + (P[2] - P[1]) * k, 1.5, (0.1, 0.8, 0.9), v)
        for i in range(4):
            x, y = P[i] * [mw, mh]
            r = 16
            quad(x - r, y - r, x + r, y + r, (1, 0.1, 0.1) if i == S["sel"] else (1, 1, 0.1), v)
        data = np.array(v, "f4")
        vbo2.write(data.tobytes())
        ctx.enable(moderngl.BLEND)
        vao2.render(moderngl.TRIANGLES, vertices=len(data) // 5)

    # ---- global hotkeys, polled ----
    def down(vk): return u32.GetAsyncKeyState(vk) & 0x8000
    prev = {"k": False}
    last_save = time.time()
    glfw.set_window_refresh_callback(win, lambda w_: S.__setitem__("dirty", True))
    while not glfw.window_should_close(win) and not S["quit"]:
        glfw.wait_events_timeout(0.5 if AW["on"] else 1.0 / FPS)        # (p1262) waits instead of spinning; (p1265) longer while away
        if os.path.exists(os.path.join(CFG_DIR, "quit.flag")):   # (p1262) clean quit for a restart
            try: os.remove(os.path.join(CFG_DIR, "quit.flag"))
            except OSError: pass
            break
        hk = down(0x11) and down(0x12) and down(0x4B)
        if hk and not prev["k"]: set_adjust(not S["adjust"])
        prev["k"] = bool(hk)
        if down(0x11) and down(0x12) and down(0x51): break
        for fl, on in (("adjust_on.flag", True), ("adjust_off.flag", False)):  # for other programs: create the file
            fp = os.path.join(CFG_DIR, fl)
            if os.path.exists(fp):
                try: os.remove(fp)
                except OSError: pass
                if S["adjust"] != on: set_adjust(on)
        if time.time() - AW["t"] > 1.0:               # (p1265) away mode, looked at once a second
            AW["t"] = time.time(); was = AW["on"]; AW["on"] = os.path.exists(AWAY)
            if was and not AW["on"]: S["dirty"] = True; S["black"] = False
        if AW["on"] and not S["adjust"]:
            if not S.get("black") or S["dirty"]:      # black once, then nothing drawn
                ctx.clear(0, 0, 0); glfw.swap_buffers(win); S["black"] = True; S["dirty"] = False
            continue
        if S["adjust"]:
            if S["drag"]:
                corners[S["sel"]] = np.clip(cursor_norm(), -0.5, 1.5)
                if time.time() - last_save > 0.5:  # saved as he drags
                    cfg["corners"] = corners.tolist(); save(cfg); last_save = time.time()
            if not glfw.get_window_attrib(win, glfw.FOCUSED) and not S["drag"]:
                pass
        with cap.lock:
            f, sq = (TEST, 1) if TEST is not None else (cap.frame, cap.seq)
        if f is not None and sq != S["seq"]:
            if (f.shape[1], f.shape[0]) != tex.size:  # capture size differs from the window (mode change): resize, never drop frames (that froze/blacked the picture)
                tex.release(); tex = new_tex(f.shape[1], f.shape[0])
            tex.write(np.ascontiguousarray(f).tobytes())
            S["seq"] = sq
            S["dirty"] = True
        if not (S["dirty"] or S["adjust"] or SELFTEST):   # (p1262) nothing new: the last picture stays up, no draw
            continue
        S["dirty"] = False
        ctx.clear(0, 0, 0)
        tex.use(0)
        prog["tex"].value = 0
        prog["Hi"].write(np.ascontiguousarray(Hinv().T, "f4").tobytes())
        vao.render(moderngl.TRIANGLES)
        if S["adjust"]: draw_overlay()
        if SELFTEST:
            selftest(ctx, fbw, fbh, corners, mw, mh); sys.stdout.flush(); os._exit(0)
        glfw.swap_buffers(win)
    cfg["corners"] = corners.tolist(); save(cfg)
    cap.run_ = False
    glfw.terminate()
    os._exit(0)


if __name__ == "__main__":
    main()
