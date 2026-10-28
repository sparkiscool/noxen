"""nox/topo: the Nox topology engine (22 Oct 2026). Replaces noxopt / noxopt_local / printopt for generated parts.

    from topo import Part, Layout, Settings, build_and_check, optimise
See README.md in this folder for the method, the sources and the compute plan.
"""
from .part import Part, tm, man, voxelise
from .layout import Layout, Settings, calibration, ReachabilityError
from .build import build_and_check, solid_check, strength
from .run import optimise

__all__ = ["Part", "Layout", "Settings", "calibration", "ReachabilityError", "build_and_check", "solid_check",
           "strength", "optimise", "tm", "man", "voxelise"]
