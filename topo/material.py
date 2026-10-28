"""Materials for printed parts: transversely isotropic FDM elasticity and layer-aware allowables.

Units: mm, N, MPa, g/cm3. The layer normal is one grid axis (the build direction). Across-layer (Z) modulus and
strength are reduced from the in-layer values. The numbers are the ones the old engine used (Bambu TDS, Zt = 0.5 Xt
for PLA / PETG until coupons say otherwise) so strength indices stay comparable between the engines.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    E: float        # in-layer modulus, MPa
    Ez: float       # across-layer modulus, MPa
    nu: float
    Xt: float       # in-layer tensile strength, MPa
    Zt: float       # across-layer tensile strength (layer adhesion), MPa
    S: float        # interlayer shear strength, MPa
    density: float  # g/cm3


MATERIALS = {
    "PLA": Material("PLA", E=2580.0, Ez=0.8 * 2580.0, nu=0.35, Xt=36.0, Zt=18.0, S=18.0, density=1.24),
    "PETG": Material("PETG", E=2780.0, Ez=0.8 * 2780.0, nu=0.38, Xt=51.0, Zt=25.5, S=25.5, density=1.27),
    "TPU95A": Material("TPU95A", E=9.8, Ez=0.8 * 9.8, nu=0.45, Xt=27.3, Zt=21.8, S=21.8, density=1.22),
}

KNOCKDOWN = 0.85
SAFETY = {"crash": 1.1, "misuse": 2.5, "flight": 4.0, "vibration": 7.0, "sustained": 10.0}

AXES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}
VOIGT_PAIRS = {3: (0, 1), 4: (1, 2), 5: (2, 0)}          # Voigt order xx yy zz xy yz zx (engineering shear)


def allowables(mat, cls):
    """(in-layer, across-layer tension, interlayer shear) allowable stresses for a load class, MPa."""
    sf = SAFETY[cls]
    return mat.Xt * KNOCKDOWN / sf, mat.Zt * KNOCKDOWN / sf, mat.S * KNOCKDOWN / sf


def stiffness(mat, axis):
    """6x6 Voigt stiffness, transversely isotropic about grid axis `axis` (0/1/2 = layer normal)."""
    E = np.full(3, mat.E, float)
    E[axis] = mat.Ez
    S = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            S[i, j] = 1.0 / E[i] if i == j else -mat.nu / np.sqrt(E[i] * E[j])
    for k, (i, j) in VOIGT_PAIRS.items():
        G = (mat.Ez if axis in (i, j) else mat.E) / (2.0 * (1.0 + mat.nu))
        S[k, k] = 1.0 / G
    C = np.linalg.inv(S)
    return 0.5 * (C + C.T)
