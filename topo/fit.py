"""The printer's fit table (15 Nov, Spark's Scout v4 bench print: the camera did not slide into its sleeve, the head frame and the
cradle did not go together, though every part passed its gates). One table (systems/fit_table.json, read here) for the Bambu A1 mini printing PCTG, from the Scout v4
probe coupon (parts/scout_v4_probe_coupon, read by Spark 14 Nov; tools/scout_v4_shape.COUPON_PICK; systems/dimensions.md):
printed holes come out ~0.1-0.2 small on the diameter, outer faces slightly fat, a D bore needed +0.3 to press home, an axle bore
+0.3 to turn free. Every setup sizes its holes, bores, slots and sleeves from here instead of the bought part's nominal size, and
the assembly check (tools/assembly_check.py) holds every declared mating pair and assembly path to the same numbers. Every size
taken here is declared with its bought part (bore / across / insert); the setup audit's fit check (topo/audit.py fit_sizes) refuses a
round or a width drawn near a declared bought part at a size the table does not give for it.

    from topo import fit
    fit.bore(3.0, "free")      -> 3.3   a printed hole for a bought round of nominal 3.0 (diameter)
    fit.across(36.0, "slide")  -> 36.6  a printed pocket / sleeve / slot across a bought body of nominal 36.0
    fit.insert("M3")           -> 4.0   a heat-set insert's hole (the table's inserts row)
    fit.gap("slide")           -> 0.25  the least drawn clearance per side at a mate of the kind (the assembly check's minimum)
    fit.FIT["slide"]           -> the row: allowance (total, on the diameter or across), tag, source

allow  = what the drawn size adds to the nominal (total: diameter, or across two faces)
gap    = the least clearance per side, in the drawn geometry (the part's mesh, before the printer grows it), the assembly check
         accepts at a declared mate of that kind: half the allowance less MESH (the viewing mesh's error), and the margin a
         declared assembly path is cut wider than the piece that goes through it. Negative: an interference fit (press, tap),
         the depth the check allows the two to overlap.
"""

import json as _json, os as _os

TABLE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "systems", "fit_table.json")
_T = _json.load(open(TABLE))
GROW = float(_T["grow_mm"])       # mm a printed face stands proud of the drawn one (holes ~0.1-0.2 small on the diameter, outer faces fat): coupon M
MESH = float(_T["mesh_mm"])       # mm the viewing mesh (parts/<p>/part.stl) may sit off the print mesh (tools/print_mesh.py: bounds within 0.05)
FIT = _T["fits"]                  # kind -> allow (total, mm), tag, src (and min_gap where the kind needs no running gap)


def _row(kind):
    if kind not in FIT:
        raise KeyError(f"fit kind {kind!r}: one of {sorted(FIT)} (topo/fit.py)")
    return FIT[kind]


def bore(nominal, kind):
    """the drawn diameter of a printed hole for a bought round part of `nominal` diameter (declared: the audit's fit check)"""
    d = round(float(nominal) + _row(kind)["allow"], 3)
    _declare(nominal, kind, d, "round")
    return d


def across(nominal, kind):
    """the drawn width of a printed pocket, sleeve or slot across a bought (or printed) body `nominal` wide (declared: the audit's fit check)"""
    d = round(float(nominal) + _row(kind)["allow"], 3)
    _declare(nominal, kind, d, "box")
    return d


def gap(kind):
    """the least drawn clearance per side at a mate of this kind (negative: the overlap an interference fit may have)"""
    r = _row(kind)
    if "min_gap" in r:
        return float(r["min_gap"])
    a = r["allow"]
    return round(a / 2.0 - MESH, 3) if a > 0 else round(a / 2.0, 3)


def table():
    """rows for a report: kind, allow, gap, tag, source"""
    return [dict(kind=k, gap=gap(k), **v) for k, v in FIT.items()]


# ---- the shapes a setup draws (the audit's fit check, topo/audit.py fit_sizes) -------------------------------------------------
# While a setup is built (topo/run.py load_part_dir wraps setup(h) in shapes()), every straight cylinder (manifold3d
# Manifold.cylinder, trimesh.creation.cylinder / annulus) and every box (Manifold.cube, trimesh.creation.box) the setup and its libs
# make is recorded: the audit then holds each round and each width that falls in a bought part's fit window to the table's sizes.
_SHAPES = None


class shapes:
    """with fit.shapes() as rec: part = setup(h)   ->  rec.round: [diameter, ...], rec.box: [(x, y, z), ...] (rounded to 0.001),
    rec.features: the handling features the shared helpers declared (topo/handling.py declare)"""

    def __enter__(self):
        global _SHAPES
        import manifold3d as mf, trimesh
        self.round, self.box, self.features, self._prev = [], [], [], _SHAPES
        _SHAPES = self
        self._orig = [(mf.Manifold, "cylinder"), (mf.Manifold, "cube"), (trimesh.creation, "cylinder"), (trimesh.creation, "annulus"),
                      (trimesh.creation, "box")]
        self._orig = [(a, n, a.__dict__[n]) for a, n in self._orig]               # the raw descriptors: put back exactly as found
        o = {(id(a), n): getattr(a, n) for a, n, f in self._orig}
        mcyl, mcube = o[(id(mf.Manifold), "cylinder")], o[(id(mf.Manifold), "cube")]
        tcyl, tann, tbox = (o[(id(trimesh.creation), n)] for n in ("cylinder", "annulus", "box"))

        def m_cyl(height, radius_low, radius_high=-1.0, *a, **k):
            if _SHAPES is not None and (radius_high is None or radius_high < 0 or abs(radius_high - radius_low) < 1e-9):
                _SHAPES.round.append(round(2.0 * float(radius_low), 3))
            return mcyl(height, radius_low, radius_high, *a, **k)

        def m_cube(size=(1.0, 1.0, 1.0), *a, **k):
            if _SHAPES is not None:
                _SHAPES.box.append(tuple(round(float(v), 3) for v in size))
            return mcube(size, *a, **k)

        def t_cyl(radius=1.0, *a, **k):
            if _SHAPES is not None:
                _SHAPES.round.append(round(2.0 * float(radius), 3))
            return tcyl(radius, *a, **k)

        def t_ann(r_min, r_max, *a, **k):
            if _SHAPES is not None:
                _SHAPES.round += [round(2.0 * float(r_min), 3), round(2.0 * float(r_max), 3)]
            return tann(r_min, r_max, *a, **k)

        def t_box(extents=None, *a, **k):
            if _SHAPES is not None and extents is not None:
                _SHAPES.box.append(tuple(round(float(v), 3) for v in extents))
            return tbox(extents, *a, **k)

        mf.Manifold.cylinder, mf.Manifold.cube = staticmethod(m_cyl), staticmethod(m_cube)
        trimesh.creation.cylinder, trimesh.creation.annulus, trimesh.creation.box = t_cyl, t_ann, t_box
        return self

    def __exit__(self, *exc):
        global _SHAPES
        for a, n, f in self._orig:
            setattr(a, n, f)
        _SHAPES = self._prev
        return False


# ---- what a setup declares it holds (the audit's fit check, topo/audit.py fit_sizes) --------------------------------------------
# Every size taken from the table is recorded with the bought part it is for (its nominal), the kind and the file that asked for it
# (a part.py, a parts/ lib, a tools/ module). The audit holds every round and every width a setup draws near a declared nominal to the
# table's sizes for it: a hole drawn by hand at 3.25 next to a declared 3 mm rod is refused.
_ALL = {}                         # caller file -> [dict(nominal, kind, drawn, shape)]


def _declare(nominal, kind, drawn, shape, depth=2):
    import sys as _sys
    f = _os.path.abspath(_sys._getframe(depth).f_code.co_filename)
    _ALL.setdefault(f, []).append(dict(nominal=round(float(nominal), 3), kind=kind, drawn=round(float(drawn), 3), shape=shape))


def insert(thread):
    """the drawn hole (diameter) for a heat-set insert of this thread ("M2", "M2.5", "M3"): the table's inserts row"""
    row = _T["inserts"][thread]
    _declare(row["hole"], "insert", row["hole"], "round")
    return float(row["hole"])


def declared(py):
    """the declarations the part at py is made with: its own, its parts/ libs' and the tools/ modules' (not another part's part.py)"""
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    py = _os.path.abspath(py)
    out = []
    for f, ds in _ALL.items():
        rel = _os.path.relpath(f, root).replace(_os.sep, "/")
        if f == py or (rel.startswith("parts/") and rel.count("/") == 1) or rel.startswith("tools/"):
            out += ds
    return out


def sizes(nominal):
    """every drawn size the table gives for a bought part of this nominal: nominal + each kind's allowance, and the nominal itself
    (the bought body drawn as a keep-out or a stand-in)"""
    n = float(nominal)
    return sorted({round(n, 3)} | {round(n + float(r["allow"]), 3) for r in FIT.values()})


def window(nominal, kinds):
    """the sizes a hand-drawn variant of a declared fit would have: within 0.15 of a declared kind's drawn size ('clear' left out: a
    pass-by gap is not a fit)"""
    d = [float(nominal) + float(FIT[k]["allow"]) for k in kinds if k in FIT and k != "clear"] or [float(nominal)]
    return round(min(d) - 0.15, 3), round(max(d) + 0.15, 3)
