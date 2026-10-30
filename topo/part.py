"""A part as the designer gives it: exact bodies (manifold3d) plus supports and load cases on regions.

  drawn = mounting / must-have features (passive solid), cuts = holes and keep-outs (passive void),
  env   = the region generated material may occupy (the drawn bodies are added to it automatically).

The grid is centred on the envelope's bounding box (mirror planes of a symmetric part fall on a cell boundary, so the
voxel masks come out mirrored) and sized to a multiple of `mult` cells per axis so the multigrid solver can coarsen.
Same call pattern as the old `arm_topo.PartProblem` (support / load / accel_case), so existing part setups work.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class LoadCase:
    name: str
    F: np.ndarray                 # (ndof,) nodal forces, N
    cls: str = "misuse"           # allowable class for the strength report; None = stiffness only
    support: str = "default"
    accel: np.ndarray = None      # (3,) inertial load in g on the part's own material (density-weighted body force)
    nodes: np.ndarray = None      # loaded node ids (for the strength report's exclusion set)


def tm(m):
    import trimesh
    me = m.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=True)


def man(t):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(t.vertices, np.float32),
                               tri_verts=np.asarray(t.faces, np.uint32)))


def voxelise(mesh, xs, ys, zs):
    """inside test of a closed trimesh at the grid points xs x ys x zs (cell centres): ray parity along +z, one ray
    per (x, y) column. Returns bool (len xs, len ys, len zs)."""
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    # rays start below the mesh, not below the grid: a keep-out that reaches under the grid (the stand setups' ring
    # r 180 and BIG half spaces) put the old origin inside the mesh and inverted the parity of those columns
    # (bracket_jA topo-v1: 43k design cells inside the ring cut, the build kept 2.4 of 12 cm3 generated)
    z0 = min(float(zs[0]), float(mesh.bounds[0][2]) if len(mesh.faces) else float(zs[0])) - 5.0
    org = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, z0)], 1)
    # a tiny irrational offset keeps rays off mesh edges / vertices that sit exactly on grid lines
    org[:, 0] += 1.234e-5
    org[:, 1] += 2.345e-5
    A = np.zeros((len(xs) * len(ys), len(zs)), bool)
    if len(mesh.faces) == 0:
        return A.reshape(len(xs), len(ys), len(zs))
    # rays in chunks: trimesh's ray test holds every candidate (ray, triangle) pair at once, and a 1M-face organic mesh
    # under 160k columns ran a 16 GB box out of memory (overseer arm rear v5c, solid check at pitch 0.25)
    locs, rays = [], []
    for c0 in range(0, len(org), 20000):
        o = org[c0:c0 + 20000]
        l_, r_, _ = mesh.ray.intersects_location(o, np.tile([0, 0, 1.0], (len(o), 1)), multiple_hits=True)
        locs.append(l_.reshape(-1, 3)); rays.append(np.asarray(r_, int) + c0)
    loc, ray = np.concatenate(locs), np.concatenate(rays)
    if len(ray):
        order = np.argsort(ray, kind="stable")
        ray = ray[order]
        z = loc[order][:, 2]
        st = np.r_[0, np.where(np.diff(ray))[0] + 1, len(ray)]
        for a, b in zip(st[:-1], st[1:]):
            zz = np.sort(z[a:b])
            zz = zz[np.r_[True, np.diff(zz) > 1e-6]]
            for p, q in zip(zz[0::2], zz[1::2]):
                zq = zs + 3.4e-5                                  # a centre exactly on a face counts the same way always
                A[ray[a], (zq > p) & (zq < q)] = True
    return A.reshape(len(xs), len(ys), len(zs))


class Part:
    def __init__(self, name, h, drawn, cuts, env, build="+z", overhang=False, material="PETG", pad=1, mult=8,
                 log=print, thin_ok=()):
        self.name, self.h, self.build, self.material, self.log = name, float(h), build, material, log
        self.overhang = overhang
        # drawn bodies the setup declares deliberately thinner than the 2 mm floor (the camera M2 insert posts, O5.6 round O3.2, 1.2
        # wall: Spark 29 Oct by coupon, the one exception in CLAUDE.md). The gate reports their thin walls under thin_allowed and
        # does not fail the part for them; everything else thin still fails.
        self.thin_ok = list(thin_ok)
        self.env0_m = env                                       # the envelope as given (functional-skin rule)
        env = env + drawn
        self.drawn_m, self.cuts_m, self.env_m = drawn, cuts, env
        bb = np.array(env.bounding_box(), float)
        lo, hi = bb[:3], bb[3:]
        n = np.ceil((hi - lo) / h).astype(int) + 2 * pad
        n = (np.ceil(n / mult) * mult).astype(int)
        c = 0.5 * (lo + hi)
        self.shape = tuple(int(v) for v in n)
        self.origin = tuple(c - n * h / 2.0)
        self.axes = [self.origin[k] + (np.arange(self.shape[k]) + 0.5) * h for k in range(3)]
        self.nel = int(np.prod(self.shape))
        nx, ny, nz = self.shape
        self.nshape = (nx + 1, ny + 1, nz + 1)
        self.nn = int(np.prod(self.nshape))
        self.ndof = 3 * self.nn
        self.env_t, self.drawn_t, self.cuts_t = tm(env), tm(drawn), tm(cuts)
        self.env0_t = tm(self.env0_m)
        inside_env = self.vox(self.env_t)
        inside_cut = self.vox(self.cuts_t)
        inside_drawn = self.vox(self.drawn_t)
        self.cut = inside_cut.ravel()
        self.in_env0 = self.vox(self.env0_t).ravel()
        self.solid = (inside_drawn & ~inside_cut).ravel()
        self.void = ((~inside_env) | inside_cut).ravel() & ~self.solid
        self.design = ~self.solid & ~self.void
        self.stiff = np.ones(self.nel)                          # modulus multiplier per cell (inserts, see insert())
        self.insert_cells = np.zeros(self.nel, bool)
        self.supports = {}            # name -> bool (nn, 3) fixed dofs
        self.support_cells = {}       # name -> element ids the support region covers
        self.cases = []
        log(f"{name}: grid {self.shape} h {h} = {self.nel:,} cells; design {int(self.design.sum()):,} "
            f"({self.design.sum() * h ** 3 / 1000:.1f} cm3), drawn {int(self.solid.sum()):,}, void {int(self.void.sum()):,}")

    # ---------------------------------------------------------------------------------------------- regions
    def vox(self, t):
        return voxelise(t, *self.axes)

    def elems(self, region, within=None, what="region"):
        """cells whose centre lies in the region; if none does (a region thinner than a cell, e.g. a 1 mm support band
        on the 2h grid), the cells the region intersects: 3 x 3 x 3 sample points per cell, any inside counts."""
        t = tm(region)
        m = self.vox(t).ravel()
        if within is not None:
            m &= within
        if not m.any():
            off = (np.arange(3) - 1.0) * self.h / 3.0
            for dx in off:
                for dy in off:
                    for dz in off:
                        m |= voxelise(t, self.axes[0] + dx, self.axes[1] + dy, self.axes[2] + dz).ravel()
            how = "intersects (3 x 3 x 3 samples per cell)"
            if within is not None:
                hit = m.copy()
                m &= within
                if not m.any() and hit.any():
                    # the region lies in the boundary cells only (a band thinner than the cell beyond the part's last
                    # cell centre): take the part cells touching those cells
                    from scipy import ndimage as ndi
                    m = ndi.binary_dilation(hit.reshape(self.shape), np.ones((3, 3, 3), bool)).ravel() & within
                    how = "touches (the region lies in the void boundary cells; their part-cell neighbours)"
            self.log(f"  {what}: no cell centre in the region at h {self.h}; using the {int(m.sum())} cells it {how}")
        return np.where(m)[0]

    def nodes_of(self, el):
        nx, ny, nz = self.shape
        ex, ey, ez = np.unravel_index(el, self.shape)
        nid = lambda i, j, k: (i * (ny + 1) + j) * (nz + 1) + k
        return np.unique(np.concatenate([nid(ex + a, ey + b, ez + c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]))

    # ------------------------------------------------------------------------------------- supports and loads
    def insert(self, name, region, e_ratio, area_mm2=None, length_mm=None):
        """a bought stiff insert embedded in the part (a carbon rod in its bore): its cells are solid in the layout and
        in the strength FE with e_ratio x the part's modulus, never printed (the bore stays in the cuts, so the build
        cuts it), and left out of the strength indices. With area_mm2 and length_mm the multiplier is scaled so the
        voxels carry the real section (E A), whatever the grid makes of a thin cylinder."""
        el = self.elems(region, what=f"insert {name}")
        if len(el) == 0:
            raise ValueError(f"insert {name}: the region holds no cells")
        k = float(e_ratio)
        if area_mm2 is not None and length_mm is not None:
            k *= area_mm2 * length_mm / (len(el) * self.h ** 3)
        self.insert_cells[el] = True
        self.stiff[el] = k
        self.solid[el], self.void[el], self.design[el] = True, False, False
        self.log(f"  insert {name}: {len(el)} cells, modulus x{k:.1f}")

    def support(self, name, region):
        el = self.elems(region, self.solid | self.design, what=f"support {name}")
        if len(el) == 0:
            raise ValueError(f"support {name}: the region holds no part cells")
        nd = self.nodes_of(el)
        fx = np.zeros((self.nn, 3), bool)
        fx[nd] = True
        self.supports[name] = fx
        self.support_cells[name] = el
        self.log(f"  support {name}: {len(el)} cells, {len(nd)} nodes")

    def load(self, name, region, vec, cls="misuse", support="default", accel=None):
        el = self.elems(region, self.solid | self.design, what=f"load {name}")
        if len(el) == 0:
            raise ValueError(f"load {name}: empty region")
        nd = self.nodes_of(el)
        F = np.zeros(self.ndof)
        for a in range(3):
            F[3 * nd + a] = vec[a] / len(nd)
        self.cases.append(LoadCase(name, F, cls=cls, support=support,
                                   accel=None if accel is None else np.asarray(accel, float), nodes=nd))

    def load_multi(self, name, parts, cls="misuse", support="default", accel=None):
        """one load case made of several (region, vector) pieces, e.g. a torque as two opposite tangential pushes."""
        F = np.zeros(self.ndof)
        alln = []
        for region, vec in parts:
            el = self.elems(region, self.solid | self.design, what=f"load {name}")
            if len(el) == 0:
                raise ValueError(f"load {name}: empty region")
            nd = self.nodes_of(el)
            for a in range(3):
                F[3 * nd + a] += vec[a] / len(nd)
            alln.append(nd)
        self.cases.append(LoadCase(name, F, cls=cls, support=support,
                                   accel=None if accel is None else np.asarray(accel, float),
                                   nodes=np.unique(np.concatenate(alln))))

    def accel_case(self, name, g, cls="misuse", support="default"):
        self.cases.append(LoadCase(name, np.zeros(self.ndof), cls=cls, support=support, accel=np.asarray(g, float),
                                   nodes=np.zeros(0, int)))

    # ------------------------------------------------------------------------------------------- finish
    def main_support(self):
        cnt = {}
        for c in self.cases:
            cnt[c.support] = cnt.get(c.support, 0) + 1
        return max(cnt, key=cnt.get) if cnt else None

    def held_bodies(self):
        """drawn cells of every drawn body the main support (the one most load cases use) holds: the mounting
        features (bolt pads and bosses, a wall flange). The organic finish keeps them whole."""
        from scipy import ndimage as ndi
        sol = (self.solid & ~self.insert_cells).reshape(self.shape)
        main = self.main_support()
        if main is None:
            return np.zeros(self.nel, bool)
        lab, n = ndi.label(sol, np.ones((3, 3, 3), bool))
        sup = np.zeros(self.nel, bool)
        sup[self.support_cells[main]] = True
        ids = np.unique(lab[sup.reshape(self.shape) & sol])
        return np.isin(lab, ids[ids > 0]).ravel()

    def functional_skin(self, skin_mm):
        """drawn cells that carry a function: within skin_mm of a cut (hole, mating / contact face, keep-out) or of the
        outside of the envelope as given, plus every drawn cell outside that envelope (no design space there). A drawn
        body that touches neither keeps all its cells. Same rule as build.skin_field at the fine pitch."""
        from scipy import ndimage as ndi
        sh = self.shape
        sol = (self.solid & ~self.insert_cells).reshape(sh)
        out0 = ~self.in_env0.reshape(sh)
        ext = self.cut.reshape(sh) | out0
        d = (ndi.distance_transform_edt(~ext) - 0.5) * self.h               # cell centre to the face of the nearest cell
        skin = sol & ((d <= skin_mm) | out0)
        lab, n = ndi.label(sol, np.ones((3, 3, 3), bool))
        if n:
            has = np.asarray(ndi.maximum(skin, lab, np.arange(1, n + 1)), bool)
            keep = np.r_[False, ~has]
            skin |= keep[lab] & sol
        # mounting features stay whole: opening part of a bolt pad left its envelope-side skin with a core of its own
        # at the gate's 2.0 mm erosion, cut off from the legs (motor_mount organic-v1: pad cores 0.75 in the main body)
        return skin.ravel() | self.held_bodies() | self.insert_cells

    # ------------------------------------------------------------------------------------------------ misc
    def cell_centres(self):
        X, Y, Z = np.meshgrid(*self.axes, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)

    def summary(self):
        return dict(name=self.name, h=self.h, shape=list(self.shape), origin=[round(o, 4) for o in self.origin],
                    cells=self.nel, design=int(self.design.sum()), drawn=int(self.solid.sum()),
                    cases=[c.name for c in self.cases], supports=list(self.supports))
