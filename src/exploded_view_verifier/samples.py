"""Small synthetic assemblies built from OCCT primitives (for demos and tests).

They are generated, not taken from any dataset, so they can be redistributed freely.
Units: millimetres.
"""
from __future__ import annotations

from . import geometry as G


def _box(x, y, z, dx, dy, dz):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def _cylinder(c, r, h, axis=(0, 0, 1)):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*c), gp_Dir(*axis)), r, h).Shape()


def _cut(a, b):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    s = BRepAlgoAPI_Cut(a, b).Shape()
    ex = TopExp_Explorer(s, TopAbs_SOLID)
    return ex.Current()


def _ring(z, r_in, r_out, h):
    return _cut(_cylinder((0, 0, z), r_out, h), _cylinder((0, 0, z - 1), r_in, h + 2))


def pin_in_plate():
    """A 40 x 40 x 10 plate with a 6.1 mm hole and a 6 mm pin through it (2 parts)."""
    plate = _cut(_box(-20, -20, 0, 40, 40, 10), _cylinder((0, 0, -1), 3.05, 12))
    pin = _cylinder((0, 0, -15), 3.0, 40)
    return {"plate": [plate], "pin": [pin]}


def shaft_stack():
    """A shaft with three stacked rings on it, sitting on a base plate (5 parts).

    The rings are coaxial with the shaft, so the centroid explosion pushes them along
    nearly the same direction: a typical case for the axial spreading repair.
    """
    base = _cut(_box(-30, -30, 0, 60, 60, 8), _cylinder((0, 0, -1), 5.05, 10))
    shaft = _cylinder((0, 0, -2), 5.0, 60)
    rings = {f"ring_{k + 1}": [_ring(8 + 6 * k, 5.05, 12 - 2 * k, 6)] for k in range(3)}
    parts = {"base": [base], "shaft": [shaft]}
    parts.update(rings)
    return parts


def caged_cube():
    """A small cube floating in the cavity of a closed hollow box (2 parts).

    The parts do not touch: the assembled state is free of interference, yet no
    straight-line disassembly exists. Useful to show that the two checks differ.
    """
    shell = _cut(_box(-25, -25, -25, 50, 50, 50), _box(-20, -20, -20, 40, 40, 40))
    inner = _box(-5, -5, -5, 10, 10, 10)
    return {"shell": [shell], "inner": [inner]}


SAMPLES = {"pin_in_plate": pin_in_plate, "shaft_stack": shaft_stack, "caged_cube": caged_cube}


def build(name: str, deflection: float = G.DEFLECTION):
    """Return the named sample as a meshed assembly ``{part: [solid]}``."""
    parts = SAMPLES[name]()
    return {p: [G.mesh(s, deflection) for s in sh] for p, sh in parts.items()}
