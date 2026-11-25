"""Desktop dual-board case: the cooling study page (job 1116, 25 Nov). Draws the case as drawn and as recommended to scale, runs the
airflow model (tools/desktop_case_airflow.py) and writes one self-contained page: parts/desktop_dual_board_case/cooling_study.html.
The words live in tools/desktop_case_cooling_text.html (plain HTML with {{name}} slots this script fills).
Run: python tools/desktop_case_cooling_page.py"""
import os, sys, html
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desktop_case_airflow as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "parts", "desktop_dual_board_case", "cooling_study.html")
TEXT = os.path.join(ROOT, "tools", "desktop_case_cooling_text.html")
X, Y = 412.0, 126.0
CLOUD_LINE = ""                # one line under the title: how the second, independent run went
CLOUD_METHOD = ("A second Fable 5.2 agent, given only the sizes and none of these conclusions, redid the closed-box sums with its own fan guesses "
                "and landed in the same place: 4 to 14 CFM as drawn, about 54 CFM reshaped, the intake as the choke, no fan in the lid. "
                "It prefers the fans hung under the upper deck over a filtered suction box, where this page seats them on the floor; "
                "both reach the same airflow. A third run scored the open spacing by the air rising within 30 mm of a cooler face and put the best at 110 to 120 mm; "
                "scored by all the air between the boards it keeps rising to 160. At 120 mm both scores are within 5 % of their best, so that is the setting.")              # the same, in full, for the method section
UP = 37.0                    # the recommended floor deck sits 37 higher (30 feet + 3 filter frame + 15 box - today's 11 feet)... D0 at z 48


class Svg:
    def __init__(self, w, h, ox, oy, s, label):
        self.w, self.h, self.ox, self.oy, self.s, self.b = w, h, ox, oy, s, []
        self.label = label

    def px(self, u, z):
        return self.ox + u * self.s, self.oy - z * self.s

    def rect(self, u0, u1, z0, z1, cls, title=""):
        x, y = self.px(u0, z1)
        t = f"<title>{html.escape(title)}</title>" if title else ""
        self.b.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{(u1 - u0) * self.s:.1f}" height="{(z1 - z0) * self.s:.1f}">{t}</rect>')

    def line(self, u0, z0, u1, z1, cls):
        (x0, y0), (x1, y1) = self.px(u0, z0), self.px(u1, z1)
        self.b.append(f'<line class="{cls}" x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>')

    def arrow(self, pts, cls, w=2.5):
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in (self.px(u, z) for u, z in pts))
        self.b.append(f'<path class="{cls}" d="{d}" stroke-width="{w}" marker-end="url(#h-{cls.split()[-1]})"/>')

    def text(self, u, z, s, cls="lb", anchor="start", dx=0, dy=0):
        x, y = self.px(u, z)
        self.b.append(f'<text class="{cls}" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">{html.escape(s)}</text>')

    def vtext(self, u, z, s, cls="lb"):
        x, y = self.px(u, z)
        self.b.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="middle" transform="rotate(-90 {x:.1f} {y:.1f})" dy="4">{html.escape(s)}</text>')

    def mark(self, u, z, n, cls="bad"):
        x, y = self.px(u, z)
        self.b.append(f'<circle class="mk {cls}" cx="{x:.1f}" cy="{y:.1f}" r="10"/><text class="mkt" x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle">{n}</text>')

    def out(self):
        defs = "".join(f'<marker id="h-{c}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
                       f'<path class="hd {c}" d="M0 0 L10 5 L0 10 z"/></marker>' for c in ("cool", "hot", "weak"))
        return (f'<svg viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" role="img" aria-label="{html.escape(self.label)}">'
                f'<defs>{defs}</defs>' + "".join(self.b) + "</svg>")


def cross(rec):
    """Cut across the case (depth y left to right, z up) through a 70 mm fan."""
    s = 1.3
    g = Svg(440, 560 if rec else 500, 150, 520 if rec else 460, s, "Cut across the case, " + ("recommended" if rec else "as drawn"))
    z0 = 48.0 if rec else 11.0            # underside of the fan deck D0
    g.line(-80, 0, Y + 80, 0, "table")
    g.text(Y + 78, 0, "desk", "lb dim", "end", dy=14)
    if rec:
        g.rect(-27, 153, 0, 30, "ghost", "cross foot, 180 long, behind this cut")
        g.text(157, 15, "cross foot,", "lb dim", "start", dy=-2); g.text(157, 15, "at each end", "lb dim", "start", dy=12)
        g.rect(0, 3, 30, 48, "wall"); g.rect(Y - 3, Y, 30, 48, "wall")
        g.line(3, 31.5, Y - 3, 31.5, "filter")
        g.rect(0, 30, 48, 51, "wall"); g.rect(96, Y, 48, 51, "wall")                       # D0 with the full dia 66 bore
        g.rect(28, 98, 51, 66, "fan", "70 mm fan on the floor deck, blowing up")
        g.rect(0, 3, 51, 85, "wall"); g.rect(Y - 3, Y, 51, 85, "wall")                    # base walls, solid
        g.rect(0, 6, 85, 88, "wall"); g.rect(120, Y, 85, 88, "wall")                      # D1: one long window, a 3 mm lip inside each wall
        zb, zt, zl = 96.0, 326.0, 338.0
    else:
        g.rect(5, 45, 0, 11, "rail", "foot rail, full length"); g.rect(81, 121, 0, 11, "rail", "foot rail, full length")
        g.rect(0, 33, 11, 14, "wall"); g.rect(93, Y, 11, 14, "wall")                      # D0 with the dia 60 bore
        g.rect(28, 98, 14, 29, "fan", "70 mm fan standing on the floor deck")
        g.rect(0, 3, 14, 48, "wall"); g.rect(Y - 3, Y, 14, 22, "wall"); g.rect(Y - 3, Y, 40, 48, "wall")     # rear wall with its slot
        g.rect(0, 30, 48, 51, "wall"); g.rect(96, Y, 48, 51, "wall")                      # D1 with the dia 66 bore
        zb, zt, zl = 59.0, 289.0, 301.0
    g.rect(0, 3, zb - 8, zl, "wall"); g.rect(Y - 3, Y, zb - 8, zl, "wall")
    for y0 in (16.0, 90.0):
        g.rect(y0, y0 + 20, zb, zt, "board", "motherboard with its own cooler (size is an estimate)")
    g.vtext(26, (zb + zt) / 2 - 40, "board A", "lb onb"); g.vtext(100, (zb + zt) / 2 - 40, "board B", "lb onb")
    if rec:
        g.line(16, zl + 1.5, 110, zl + 1.5, "grille")
        g.rect(0, 16, zl, zl + 3, "wall"); g.rect(110, Y, zl, zl + 3, "wall")
        for y0 in (16.0, 36.0, 90.0, 110.0):                                              # collars from the lid down to each fin outlet
            g.line(y0, zt, y0, zl, "collar")
        for yc, sg in ((36.0, 1), (90.0, -1)):                                            # blower inlets on the faces toward the gap
            x, y = g.px(yc, zt - 55)
            g.b.append(f'<path class="blower" d="M{x:.1f} {y - 22:.1f} A22 22 0 0 {1 if sg > 0 else 0} {x:.1f} {y + 22:.1f}"/>')
        g.arrow([(-70, 14), (-8, 14), (20, 22)], "flow cool"); g.arrow([(Y + 70, 14), (Y + 8, 14), (Y - 20, 22)], "flow cool")
        g.arrow([(63, 22), (63, 46)], "flow cool", 4)
        g.arrow([(63, 70), (63, zt - 70)], "flow cool", 5)
        g.arrow([(9.5, 92), (9.5, zl + 24)], "flow cool", 2); g.arrow([(116.5, 92), (116.5, zl + 24)], "flow cool", 2)
        g.arrow([(56, zt - 86), (47, zt - 62), (40, zt - 56)], "flow cool", 2.5); g.arrow([(70, zt - 86), (79, zt - 62), (86, zt - 56)], "flow cool", 2.5)
        g.arrow([(63, zt - 40), (63, zl + 24)], "flow cool", 2.5)
        g.arrow([(26, zt + 2), (26, zl + 34)], "flow hot", 4); g.arrow([(100, zt + 2), (100, zl + 34)], "flow hot", 4)
        g.text(63, zl + 48, "hot air out through collars, spare air out the open lid", "lb", "middle", dy=-6)
        g.text(-74, 14, "in, all round", "lb", "start", dy=16)
        for n, (u, z) in {1: (-40, 26), 2: (-14, 58), 3: (140, 31), 4: (140, 68), 5: (140, 86), 6: (-14, zt - 55), 7: (140, zl + 2)}.items():
            g.mark(u, z, n, "good")
        g.line(3, 31.5, 128, 31.5, "lead"); g.line(Y, 68, 128, 68, "lead"); g.line(Y, 86.5, 128, 86.5, "lead"); g.line(Y, zl + 2, 128, zl + 2, "lead")
        g.line(36, zt - 55, -3, zt - 55, "lead"); g.line(28, 58, -3, 58, "lead")
    else:
        g.line(16, zl + 1.5, 110, zl + 1.5, "grille sparse")
        g.rect(0, 16, zl, zl + 3, "wall"); g.rect(110, Y, zl, zl + 3, "wall")
        g.rect(28, 98, zl - 15, zl, "fan ghostfan", "the fifth fan, under the middle of the lid")
        g.arrow([(63, 2), (63, 9)], "flow weak", 1.5)
        g.arrow([(63, 34), (63, 150)], "flow weak", 1.5)
        g.arrow([(Y + 60, 31), (Y + 6, 31)], "flow weak", 1.5)
        for n, (u, z) in {1: (-16, 6), 2: (-16, 40), 3: (142, 31), 4: (-16, 56), 5: (63, zl + 22)}.items():
            g.mark(u, z, n)
        g.line(5, 6, -6, 6, "lead"); g.line(28, 40, -6, 40, "lead"); g.line(0, 56, -6, 56, "lead"); g.line(63, zl + 3, 63, zl + 12, "lead")
        g.text(63, 150, "a trickle", "lb dim", "middle", dy=-8)
    g.text(63, -20, ("Recommended" if rec else "As drawn today") + ", cut across", "cap", "middle")
    return g.out()


def side(rec):
    """Long side view (x left to right, z up)."""
    s = 1.0
    g = Svg(640, 470 if rec else 430, 100, 425 if rec else 385, s, "Side view, " + ("recommended" if rec else "as drawn"))
    up = UP if rec else 0.0
    g.line(-90, 0, X + 110, 0, "table")
    zb, zt, zl = 59.0 + up, 289.0 + up, 301.0 + up
    g.rect(36, 376, zb, zt, "board side", "both boards, seen from the side (340 x 230, an estimate)")
    g.text(206, (zb + zt) / 2, "boards A and B (340 x 230, estimate)", "lb onb", "middle", dy=4)
    if rec:
        fans = [(89.0, 120, 25.0), (192.0, 70, 15.0), (270.0, 70, 15.0), (348.0, 70, 15.0)]
        g.rect(10, 50, 0, 30, "rail", "cross foot"); g.rect(362, 402, 0, 30, "rail", "cross foot")
        g.line(3, 31.5, X - 3, 31.5, "filter")
        g.rect(0, 3, 30, 85, "wall"); g.rect(X - 3, X, 30, 85, "wall")
        edges = [0.0] + [e for c, d, t in fans for e in (c - (113 if d == 120 else 66) / 2, c + (113 if d == 120 else 66) / 2)] + [X]
        for a, b in zip(edges[0::2], edges[1::2]):
            g.rect(a, b, 48, 51, "wall")
        for c, d, t in fans:
            g.rect(c - d / 2, c + d / 2, 51, 51 + t, "fan", f"{d} mm fan on the floor deck")
        g.rect(0, 36, 85, 88, "wall"); g.rect(376, X, 85, 88, "wall")
        for c in (137.3, 274.7):
            g.rect(c - 3, c + 3, 85, 88, "wall")
        g.rect(0, 36, zl, zl + 3, "wall"); g.rect(376, X, zl, zl + 3, "wall"); g.line(36, zl + 1.5, 376, zl + 1.5, "grille")
        g.rect(0, 3, 88, 150, "wall"); g.rect(0, 3, 166, zl, "wall")                         # one fitted port window left open
        g.rect(X - 3, X, 88, zl, "wall")
        for a, b in ((60.0, 150.0), (262.0, 352.0)):
            g.rect(a, b, zt, zl, "collarbox", "collar from a fin outlet to the lid (place after measuring the board)")
            g.arrow([((a + b) / 2, zt + 2), ((a + b) / 2, zl + 34)], "flow hot", 4)
        for c, d, t in fans:
            g.arrow([(c, 8), (c, 46)], "flow cool", 3 if d == 120 else 2)
            g.arrow([(c, 55 + t), (c, zb + 60)], "flow cool", 4 if d == 120 else 2.5)
        g.arrow([(206, zt - 60), (206, zl + 28)], "flow cool", 2.5)
        g.text(206, zl + 44, "lid open over the boards, shut over both end spaces", "lb", "middle")
        g.vtext(19, 200, "end space, shut top and bottom", "lb dim"); g.vtext(393, 200, "end space", "lb dim")
        g.text(-8, 158, "one fitted", "lb dim", "end", dy=-4); g.text(-8, 158, "port window", "lb dim", "end", dy=10)
        g.text(105, zt - 16, "collar to the lid", "lb onb", "middle"); g.text(307, zt - 16, "collar to the lid", "lb onb", "middle")
        g.text(-8, 31.5, "filter", "lb dim", "end", dy=4); g.text(-8, 15, "cross foot", "lb dim", "end", dy=4)
    else:
        fans = [(63.0, 120, 25.0), (166.0, 70, 15.0), (244.0, 70, 15.0), (322.0, 70, 15.0)]
        g.rect(0, X, 0, 11, "rail", "foot rails, the full length on both sides")
        for (z, bores) in ((11.0, {120: 100.0, 70: 60.0}), (48.0, {120: 113.0, 70: 66.0})):
            edges = [0.0] + [e for c, d, t in fans for e in (c - bores[d] / 2, c + bores[d] / 2)] + [X]
            for a, b in zip(edges[0::2], edges[1::2]):
                g.rect(a, b, z, z + 3, "wall")
        for c, d, t in fans:
            g.rect(c - d / 2, c + d / 2, 14, 14 + t, "fan", f"{d} mm fan")
        g.rect(0, 3, 14, 67, "wall"); g.rect(0, 3, 167, 185, "wall"); g.rect(0, 3, 285, zl, "wall")   # the -x end wall with its two port windows
        g.rect(X - 3, X, 14, zl, "wall")
        g.rect(0, 16, zl, zl + 3, "wall"); g.rect(400, X, zl, zl + 3, "wall")
        xs = [16 + 16 * k for k in range(24)]
        solid = [0.0] + [e for x in xs if not (x + 6 > 160 and x < 252) for e in (x, x + 6)] + [X]
        for a, b in zip(solid[0::2], solid[1::2]):
            g.rect(a, b, zl, zl + 3, "wall")
        g.rect(171, 241, zl - 15, zl, "fan", "the fifth fan as a lid extractor")
        g.arrow([(63, 55), (20, 75), (-16, 115)], "flow weak", 2)
        g.text(-22, 122, "out the port", "lb dim", "end"); g.text(-22, 108, "windows", "lb dim", "end")
        g.text(-8, 5.5, "foot rails", "lb dim", "end", dy=4)
        for n, (u, z) in {1: (206, 5.5), 6: (-14, 235), 5: (206, zl + 22), 7: (18, 30)}.items():
            g.mark(u, z, n)
        g.line(206, zl + 3, 206, zl + 12, "lead")
    # what hangs on the +x end, for orientation
    g.rect(X, X + 55, 118 + up, 218 + up, "ghost", "carry handle"); g.text(X + 60, 168 + up, "handle", "lb dim", dy=4)
    g.rect(X + 5, X + 60, 240 + up, 365 + up, "ghost", "projector, stowed upright (124.5 x 124.5 x 53.3)"); g.text(X + 65, 300 + up, "projector", "lb dim", dy=4)
    g.rect(X, X + 80, 14 + up, 59 + up, "ghost", "hub pad: two Acer hubs (104.9 x 30 x 8.9 each) and the webcam shelf"); g.text(X + 85, 40 + up, "hubs", "lb dim", dy=4)
    g.text(206, -20, ("Recommended" if rec else "As drawn today") + ", from the side", "cap", "middle")
    return g.out()


def plan(gap):
    """The pad from above with the boards over it: closed (54) or open."""
    closed = gap == 54
    g = Svg(470, 400, 235, 205, 1.0, "Pad and boards from above, " + ("closed" if closed else "open"))
    g.b.append('<rect class="pad" x="45" y="55" width="380" height="300" rx="8"><title>Kootek pad, 380 x 300 (exact)</title></rect>')
    for f in M.PAD_FANS:
        size, x, y = f
        cx, cy = g.px(x, y)
        g.b.append(f'<circle class="fan" cx="{cx:.1f}" cy="{cy:.1f}" r="{(113 if size == 120 else 66) / 2}"><title>{size} mm fan{"" if size == 120 else " (position read off the listing photo)"}</title></circle>')
        g.b.append(f'<circle class="hub" cx="{cx:.1f}" cy="{cy:.1f}" r="{20 if size == 120 else 12}"/>')
    h = gap / 2
    for sg in (-1, 1):
        a, b = sorted((sg * h, sg * (h + 20)))
        g.rect(-170, 170, a, b, "board", "motherboard with its cooler, cooler face toward the middle")
        a, b = sorted((sg * (h + 33), sg * (h + 36)))
        g.rect(-206, 206, a, b, "wall")
    e = 63 if closed else 75
    g.rect(-206, -203, -e, e, "wall"); g.rect(203, 206, -e, e, "wall")
    g.line(182, -h, 182, h, "dim"); g.line(176, -h, 188, -h, "dim"); g.line(176, h, 188, h, "dim")
    g.text(192, 0, f"{gap}", "cap", "start", dy=5)
    g.text(-170, h + 10, "board A", "lb onb", "start", dx=6, dy=4); g.text(-170, -h - 10, "board B", "lb onb", "start", dx=6, dy=4)
    g.text(0, 180, ("Closed for carrying: 54 mm" if closed else f"Open for cooling: {gap} mm") + ", from above", "cap", "middle")
    g.text(-186, -142, "pad 380 x 300", "lb dim", "start", dy=-2)
    if closed:
        g.text(108, 98, "small fans mostly", "lb", "middle"); g.text(108, 98, "outside the box", "lb", "middle", dy=14)
    else:
        g.text(0, 0, "big fan", "lb", "middle", dy=-28)
    return g.out()


def endview(gap):
    """The open case on its dock over the pad, seen from one end (y left to right, z up)."""
    s = 1.12
    g = Svg(540, 520, 270, 478, s, "The open case over the pad, seen from one end")
    h = gap / 2
    g.line(-235, 0, 235, 0, "table"); g.text(232, 0, "desk", "lb dim", "end", dy=14)
    g.rect(-150, -122, 0, 30, "rail", "dock riser"); g.rect(122, 150, 0, 30, "rail", "dock riser")
    g.rect(-150, 150, 30, 65, "pad", "Kootek pad, 35 thick")
    for yc in (-55, 55):
        g.rect(yc - 35, yc + 35, 42, 57, "fan ghostfan", "70 mm fans")
    g.rect(-60, 60, 38, 58, "fan", "120 mm fan")
    g.line(-150, 65, 150, 65, "grille")
    g.rect(-15, 15, 70, 344, "ghost", "end post of the fixed spine (one at each end)")
    g.rect(-15, 15, 344, 356, "wall", "handle beam on the two end posts")
    for sg in (-1, 1):
        a, b = sorted((sg * h, sg * (h + 20)))
        g.rect(a, b, 90, 320, "board", "motherboard with its cooler")
        a, b = sorted((sg * (h + 33), sg * (h + 36)))
        g.rect(a, b, 70, 336, "wall")
        g.line(sg * 28, 334, sg * (h + 33), 334, "grille")
        g.arrow([(sg * 228, 15), (sg * 160, 15), (sg * 95, 24)], "flow cool")
        g.arrow([(sg * 44, 70), (sg * (h - 9), 150), (sg * (h - 9), 262)], "flow cool", 3)
        g.arrow([(sg * (h + 10), 322), (sg * (h + 10), 384)], "flow hot", 4)
    g.vtext(-h - 10, 205, "board A", "lb onb"); g.vtext(h + 10, 205, "board B", "lb onb")
    g.arrow([(-32, 70), (-32, 388)], "flow cool", 4); g.arrow([(32, 70), (32, 388)], "flow cool", 4)
    g.line(-h, 120, h, 120, "dim"); g.line(-h, 114, -h, 126, "dim"); g.line(h, 114, h, 126, "dim")
    g.text(0, 120, f"{gap} mm", "cap", "middle", dy=-6)
    g.text(0, 400, "out through the open middle and the lid grilles", "lb", "middle")
    g.text(-228, 15, "in under the pad", "lb", "start", dy=-8)
    for n, (u, z) in {1: (-178, 44), 2: (178, 78), 3: (0, 96), 4: (-h - 58, 290), 5: (h + 62, 352), 6: (0, 372)}.items():
        g.mark(u, z, n, "good")
    g.line(-h - 46, 290, -h - 20, 290, "lead"); g.line(h + 50, 348, h + 36, 336, "lead"); g.line(166, 78, h + 36, 72, "lead")
    g.text(0, -22, f"Open for cooling: {gap} mm, from one end", "cap", "middle")
    return g.out()


def gapchart(res):
    """Air between the boards against the gap: line = nominal fans, band = low to high fan guess."""
    rows = res["pad"]["gaps"]
    w, h, left, right, top, bot = 760, 330, 64, 150, 24, 52
    sx = lambda v: left + (v - 40) / (230 - 40) * (w - left - right)
    sy = lambda v: h - bot - v / 100.0 * (h - top - bot)
    c = lambda lv, r: r[lv]["between"] * M.CFM
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Air between the boards against the gap between them">']
    for t in (0, 25, 50, 75, 100):
        o.append(f'<line class="grid" x1="{left}" y1="{sy(t):.1f}" x2="{w - right}" y2="{sy(t):.1f}"/><text class="tick" x="{left - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for r in rows:
        o.append(f'<text class="tick" x="{sx(r["gap"]):.1f}" y="{h - bot + 18}" text-anchor="middle">{r["gap"]}</text>')
    o.append(f'<text class="tick" x="{(left + w - right) / 2:.1f}" y="{h - 8}" text-anchor="middle">gap between the cooler faces, mm</text>')
    o.append(f'<text class="tick" x="{left - 8}" y="{top - 8}" text-anchor="end">CFM</text>')
    band = " ".join(f"{sx(r['gap']):.1f},{sy(c('high', r)):.1f}" for r in rows) + " " + " ".join(f"{sx(r['gap']):.1f},{sy(c('low', r)):.1f}" for r in reversed(rows))
    o.append(f'<polygon class="band" points="{band}"/>')
    o.append('<polyline class="ln" points="' + " ".join(f"{sx(r['gap']):.1f},{sy(c('nominal', r)):.1f}" for r in rows) + '"/>')
    for r in rows:
        x, y, n = sx(r["gap"]), sy(c("nominal", r)), r["nominal"]
        tip = (f"Gap {r['gap']} mm: {c('nominal', r):.0f} CFM between the boards ({c('low', r):.0f} to {c('high', r):.0f} for weak to strong fans), "
               f"moving at {n['speed']:.2f} m/s; {n['outer'] * M.CFM:.0f} CFM on the outer faces, {max(n['missed'], 0) * M.CFM:.0f} CFM past the case")
        pick = r["gap"] in (54, res["pad"]["open_gap"])
        o.append(f'<g class="row" tabindex="0" data-tip="{html.escape(tip)}"><rect class="hit" x="{x - 14:.1f}" y="{top}" width="28" height="{h - top - bot}"/>'
                 f'<circle class="dot{" pickdot" if pick else ""}" cx="{x:.1f}" cy="{y:.1f}" r="{6 if pick else 4.5}"/></g>')
    for gp, word, dy in ((54, "closed", 22), (res["pad"]["open_gap"], "open", -14)):
        r = next(x for x in rows if x["gap"] == gp)
        o.append(f'<text class="val" x="{sx(gp) + (10 if gp == 54 else 0):.1f}" y="{sy(c("nominal", r)) + dy:.1f}" text-anchor="{"start" if gp == 54 else "middle"}">{word}: {c("nominal", r):.0f} CFM</text>')
    last = rows[-1]
    o.append(f'<text class="tick" x="{w - right + 10}" y="{sy(c("nominal", last)) + 4:.1f}">middle fan guess</text>')
    o.append(f'<text class="tick" x="{w - right + 10}" y="{sy(c("high", last)) + 4:.1f}">strong fans</text><text class="tick" x="{w - right + 10}" y="{sy(c("low", last)) + 4:.1f}">weak fans</text>')
    o.append(f'<line class="axis" x1="{left}" y1="{sy(0):.1f}" x2="{w - right}" y2="{sy(0):.1f}"/></svg>')
    return "".join(o)


def gaptable(res):
    rows, pick = res["pad"]["gaps"], res["pad"]["open_gap"]
    cls = lambda r: ' class="pick"' if r["gap"] == pick else ""
    o = ['<table class="num sweep"><thead><tr><th style="text-align:left">Gap, mm</th>' + "".join(f'<th{cls(r)}>{r["gap"]}{" (closed)" if r["gap"] == 54 else ""}</th>' for r in rows) + "</tr></thead><tbody>"]
    for label, f in (("Between the boards, CFM", lambda n: f'{n["between"] * M.CFM:.0f}'), ("Air speed there, m/s", lambda n: f'{n["speed"]:.2f}'),
                     ("Onto the outer faces, CFM", lambda n: f'{n["outer"] * M.CFM:.0f}'), ("Past the case, wasted, CFM", lambda n: f'{max(n["missed"], 0) * M.CFM:.0f}')):
        o.append(f'<tr><td style="text-align:left">{label}</td>' + "".join(f'<td{cls(r)}>{f(r["nominal"])}</td>' for r in rows) + "</tr>")
    o.append("</tbody></table>")
    return "".join(o)


def cleartable(res):
    rows = res["pad"]["clearance"]
    return ('<table class="num sweep"><thead><tr>' + "".join(f'<th class="{"pick" if r["x"] == 30 else ""}">{r["x"]}</th>' for r in rows) + "</tr></thead><tbody><tr>"
            + "".join(f'<td class="{"pick" if r["x"] == 30 else ""}">{r["nominal"] * M.CFM:.0f}</td>' for r in rows) + "</tr></tbody></table>")


def shifttable(res):
    rows = res["pad"]["shift"]
    gaps = ("100", "120", "140", "160", "180")
    o = ['<table class="num sweep"><thead><tr><th style="text-align:left">Small fans, mm either side of the big fan</th>' + "".join(f"<th>gap {g}</th>" for g in gaps) + "</tr></thead><tbody>"]
    for r in rows:
        best = max(r[g] for g in gaps)
        o.append(f'<tr><td style="text-align:left">{r["y70"]:.0f}{" (read off the photo)" if r["y70"] == 55 else ""}</td>'
                 + "".join(f'<td class="{"pick" if r[g] >= 0.85 * best and all(r[k] < 0.85 * best for k in gaps[:gaps.index(g)]) else ""}">{r[g] * M.CFM:.0f}</td>' for g in gaps) + "</tr>")
    o.append("</tbody></table>")
    return "".join(o)


def bars(res):
    """Air past the boards by layout: bar = nominal fans, line = low to high fan guess."""
    rows = [(c["name"], c["low"]["flow_cfm"], c["nominal"]["flow_cfm"], c["high"]["flow_cfm"], c["nominal"]["flow_ls"]) for c in res["cases"]]
    names = {"as drawn, fans on the floor deck": "As drawn, fans on the floor deck", "as drawn, fans under the upper deck": "As drawn, fans under the upper deck",
             "open floor only": "Open floor only (today's lid)", "recommended": "Recommended", "recommended + lid fan": "Recommended + lid fan",
             "recommended, all five fans in the base": "Recommended, five fans in the base"}
    left, top, rowh, w, xmax = 250, 30, 38, 760, 80.0
    plot = w - left - 90
    h = top + rowh * len(rows) + 34
    sx = lambda v: left + v / xmax * plot
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="Airflow past the boards for each layout">']
    for t in range(0, 81, 20):
        o.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{top - 6}" x2="{sx(t):.1f}" y2="{h - 30}"/><text class="tick" x="{sx(t):.1f}" y="{h - 14}" text-anchor="middle">{t}</text>')
    o.append(f'<text class="tick" x="{sx(xmax) + 16:.1f}" y="{h - 14}">CFM</text>')
    for i, (nm, lo, nom, hi, ls) in enumerate(rows):
        y = top + i * rowh + 6
        x1 = sx(nom)
        tip = f"{names[nm]}: {nom} CFM ({ls} L/s) with the nominal fan guess; {lo} to {hi} CFM over the low to high guess"
        o.append(f'<g class="row" tabindex="0" data-tip="{html.escape(tip)}"><rect class="hit" x="0" y="{y - 8}" width="{w}" height="{rowh}"/>'
                 f'<text class="rowlb{" strong" if nm == "recommended" else ""}" x="{left - 12}" y="{y + 14}" text-anchor="end">{html.escape(names[nm])}</text>'
                 f'<path class="bar" d="M{left} {y} H{x1 - 4:.1f} a4 4 0 0 1 4 4 V{y + 16} a4 4 0 0 1 -4 4 H{left} z"/>'
                 f'<line class="range" x1="{sx(lo):.1f}" y1="{y + 10}" x2="{sx(hi):.1f}" y2="{y + 10}"/>'
                 f'<line class="range" x1="{sx(lo):.1f}" y1="{y + 5}" x2="{sx(lo):.1f}" y2="{y + 15}"/><line class="range" x1="{sx(hi):.1f}" y1="{y + 5}" x2="{sx(hi):.1f}" y2="{y + 15}"/>'
                 f'<text class="val" x="{sx(hi) + 10:.1f}" y="{y + 14}">{nom} CFM</text></g>')
    o.append(f'<line class="axis" x1="{left}" y1="{top - 6}" x2="{left}" y2="{h - 30}"/></svg>')
    return "".join(o)


def table(res):
    o = ['<table class="num"><thead><tr><th>Layout</th><th>Air past the boards</th><th>Low to high fan guess</th><th>Leaves by the lid</th><th>Bay pressure</th>'
         '<th>Air warms by, at 180 / 270 / 360 W</th></tr></thead><tbody>']
    for c in res["cases"]:
        n = c["nominal"]; f = n["flows"]
        top = max(f["lid"], 0) + f.get("lid fan", 0.0)
        share = f"{round(100 * top / (top + max(f['leaks'], 0)))} %" if n["flow_ls"] > 5 else "see note"
        rk = n["rise_K"]
        o.append(f'<tr><td>{html.escape(c["name"][0].upper() + c["name"][1:])}</td><td>{n["flow_cfm"]} CFM ({n["flow_ls"]} L/s)</td><td>{c["low"]["flow_cfm"]} to {c["high"]["flow_cfm"]} CFM</td>'
                 f'<td>{share}</td><td>{n["bay_pa"]:+.1f} Pa</td><td>{rk["low"]:.0f} / {rk["nominal"]:.0f} / {rk["high"]:.0f} °C</td></tr>')
    o.append("</tbody></table>")
    return "".join(o)


def sweep(res, key, unit, pick):
    rows = res["sweeps"][key]
    cells = "".join(f'<td class="{"pick" if r["x"] == pick else ""}">{r["nominal"] * M.CFM:.0f}</td>' for r in rows)
    heads = "".join(f'<th class="{"pick" if r["x"] == pick else ""}">{r["x"]:,}{unit}</th>' for r in rows)
    return f'<table class="num sweep"><thead><tr>{heads}</tr></thead><tbody><tr>{cells}</tr></tbody></table>'


def r10(v, to=10):
    return f"{round(v / to) * to:,}"


def main():
    res = M.run()
    c = {x["name"]: x for x in res["cases"]}
    rec, a, b = c["recommended"], c["as drawn, fans on the floor deck"], c["as drawn, fans under the upper deck"]
    pad = res["pad"]
    gp, gl, gh = ({r["gap"]: r[lv] for r in pad["gaps"]} for lv in ("nominal", "low", "high"))
    cl = {r["x"]: r["nominal"] for r in pad["clearance"]}
    slots = {
        "fig_cross_drawn": cross(False), "fig_cross_rec": cross(True), "fig_side_drawn": side(False), "fig_side_rec": side(True),
        "chart": bars(res), "table": table(res),
        "sweep_feet": sweep(res, "feet", "", 30), "sweep_lid": sweep(res, "lid", "", 25000), "sweep_filter": sweep(res, "filter", "", 4), "sweep_box": sweep(res, "box", "", 15),
        "now_lo": a["nominal"]["flow_cfm"], "now_hi": b["nominal"]["flow_cfm"], "rec": rec["nominal"]["flow_cfm"], "rec_lo": rec["low"]["flow_cfm"], "rec_hi": rec["high"]["flow_cfm"],
        "rec_ls": rec["nominal"]["flow_ls"], "rise_now": f'{b["nominal"]["rise_K"]["nominal"]:.0f}', "rise_rec": f'{rec["nominal"]["rise_K"]["nominal"]:.0f}',
        "rise_rec_lo": f'{rec["high"]["rise_K"]["low"]:.0f}', "rise_rec_hi": f'{rec["low"]["rise_K"]["high"]:.0f}',
        "back120": f'{-a["nominal"]["flows"]["fan 120"]:.1f}', "a_floor": r10(res["areas"]["floor path"]), "a_slots": r10(res["areas"]["base slots"]),
        "a_lid": r10(res["areas"]["lid slots"]), "a_port": r10(res["areas"]["port windows"]), "a_fans": r10(res["areas"]["fan 120"] + 3 * res["areas"]["fan 70"], 100),
        "lidfan_gain": f'{100 * (c["recommended + lid fan"]["nominal"]["flow_ls"] / rec["nominal"]["flow_ls"] - 1):.0f}',
        "five_gain": f'{100 * (c["recommended, all five fans in the base"]["nominal"]["flow_ls"] / rec["nominal"]["flow_ls"] - 1):.0f}',
        "leak_share": f'{100 * c["open floor only"]["nominal"]["flows"]["leaks"] / c["open floor only"]["nominal"]["flow_ls"]:.0f}',
        "off_w": res["fans_off"]["heat_w_at_15K"], "off_ls": res["fans_off"]["flow_ls"],
        "cloud_line": CLOUD_LINE, "cloud_method": CLOUD_METHOD,
        "fig_plan_closed": plan(54), "fig_plan_open": plan(pad["open_gap"]), "fig_end_open": endview(pad["open_gap"]),
        "gapchart": gapchart(res), "gaptable": gaptable(res), "cleartable": cleartable(res), "shifttable": shifttable(res),
        "gap": pad["open_gap"], "travel": (pad["open_gap"] - 54) // 2, "slot": pad["open_gap"] - 54, "open_depth": pad["open_gap"] + 2 * 36,
        "closed_cfm": f'{gp[54]["between"] * M.CFM:.0f}', "open_cfm": f'{gp[pad["open_gap"]]["between"] * M.CFM:.0f}',
        "open_lo": f'{gl[pad["open_gap"]]["between"] * M.CFM:.0f}', "open_hi": f'{gh[pad["open_gap"]]["between"] * M.CFM:.0f}',
        "closed_w": r10(pad["closed_w_10K"]), "open_w": r10(pad["open_w_10K"]), "filter_loss": pad["filter_loss_pct"],
        "flat_cfm": f'{pad["flat_between"]["nominal"] * M.CFM:.0f}',
        "clear5": f'{100 * cl[5] / cl[30]:.0f}', "clear20": f'{100 * cl[20] / cl[30]:.0f}',
        "g120": f'{100 * gp[120]["between"] / gp[180]["between"]:.0f}', "g140": f'{100 * gp[140]["between"] / gp[180]["between"]:.0f}',
    }
    page = open(TEXT, encoding="utf-8").read()
    for k, v in slots.items():
        page = page.replace("{{" + k + "}}", str(v))
    assert "{{" not in page, page[page.index("{{"):][:60]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)
    print("wrote", OUT, len(page), "bytes")


if __name__ == "__main__":
    main()
