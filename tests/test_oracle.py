"""Verifier cases with a known mathematical answer (mm, tol 0.1, deflection 0.5)."""
import math

import pytest

pytest.importorskip("OCC")

from exploded_view_verifier import geometry as G  # noqa: E402
from exploded_view_verifier import oracle as O  # noqa: E402

TOL, DEFL = 0.1, 0.5


def box(x, y, z, dx, dy, dz):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def cube(c, a):
    return box(c[0] - a / 2, c[1] - a / 2, c[2] - a / 2, a, a, a)


def cylinder(c, r, h):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*c), gp_Dir(0, 0, 1)), r, h).Shape()


def sphere(c, r):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCC.Core.gp import gp_Pnt
    return BRepPrimAPI_MakeSphere(gp_Pnt(*c), r).Shape()


def torus(c, r1, r2):
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeTorus
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
    return BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(*c), gp_Dir(0, 0, 1)), r1, r2).Shape()


def cut(a, b):
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_SOLID
    return TopExp_Explorer(BRepAlgoAPI_Cut(a, b).Shape(), TopAbs_SOLID).Current()


def tangent_plate(r, deg, gap, z=-5.0):
    """5 x 5 x 2 plate whose large face is tangent to the circle of radius r (z axis) at
    angle ``deg``, at distance ``gap`` from the surface."""
    from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    p = box(r + gap, -2.5, z, 2, 5, 5)
    t = gp_Trsf()
    t.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(deg))
    return BRepBuilderAPI_Transform(p, t, True).Shape()


def hollow():
    return cut(box(-50, -50, -50, 100, 100, 100), box(-40, -40, -40, 80, 80, 80))  # 10 mm wall


def plate_hole(r_hole):
    return cut(box(-20, -20, 0, 40, 40, 5), cylinder((0, 0, -1), r_hole, 7))


CASES = [
    ("cubes 5 mm apart", lambda: ([cube((0, 0, 0), 10)], [cube((15, 0, 0), 10)]), "free"),
    ("cubes 0.2 mm apart (> tol)", lambda: ([cube((0, 0, 0), 10)], [cube((10.2, 0, 0), 10)]), "free"),
    ("cubes 0.05 mm apart (< tol)", lambda: ([cube((0, 0, 0), 10)], [cube((10.05, 0, 0), 10)]), "conflict"),
    ("cubes touching", lambda: ([cube((0, 0, 0), 10)], [cube((10, 0, 0), 10)]), "conflict"),
    ("cubes overlapping 2 mm", lambda: ([cube((0, 0, 0), 10)], [cube((8, 0, 0), 10)]), "conflict"),
    ("cube 10 inside cube 100", lambda: ([cube((0, 0, 0), 10)], [cube((0, 0, 0), 100)]), "conflict:inclusion"),
    ("cube 10 inside cube 100, 0.05 mm from a face",
     lambda: ([cube((44.95, 0, 0), 10)], [cube((0, 0, 0), 100)]), "conflict"),
    ("cube in the cavity of a hollow box", lambda: ([cube((0, 0, 0), 10)], [hollow()]), "free"),
    ("cube inside the wall of a hollow box", lambda: ([cube((45, 0, 0), 4)], [hollow()]), "conflict:inclusion"),
    ("cylinder r5 in hole r6 (1 mm clearance)", lambda: ([cylinder((0, 0, -10), 5, 25)], [plate_hole(6)]), "free"),
    ("cylinder r5 in hole r5.05 (0.05 clearance)",
     lambda: ([cylinder((0, 0, -10), 5, 25)], [plate_hole(5.05)]), "conflict"),
    ("cylinder r5 in hole r4 (press fit)", lambda: ([cylinder((0, 0, -10), 5, 25)], [plate_hole(4)]), "conflict"),
    ("sphere r5 inside sphere r20", lambda: ([sphere((0, 0, 0), 5)], [sphere((0, 0, 0), 20)]), "conflict:inclusion"),
    ("torus inside cube 100", lambda: ([torus((0, 0, 0), 10, 3)], [cube((0, 0, 0), 100)]), "conflict:inclusion"),
    ("torus around a cylinder (2 mm clearance)",
     lambda: ([torus((0, 0, 0), 10, 3)], [cylinder((0, 0, -20), 5, 40)]), "free"),
    ("two-solid part, one solid overlaps",
     lambda: ([cube((0, 0, 0), 10), cube((30, 0, 0), 10)], [cube((38, 0, 0), 10)]), "conflict"),
]
# Curved faces: a plate tangent to a cylinder / sphere of radius 20 between two mesh nodes;
# the mesh chords pass inside the curved solid by up to about the deflection.
for _gap, _exp in ((0.0, "conflict"), (0.05, "conflict"), (0.3, "free"), (1.5, "free")):
    CASES.append((f"cylinder r20, tangent plate, gap {_gap}",
                  (lambda g=_gap: ([cylinder((0, 0, -10), 20, 20)], [tangent_plate(20, 7.0, g)])), _exp))
for _gap, _exp in ((0.05, "conflict"), (0.3, "free")):
    CASES.append((f"sphere r20, tangent plate, gap {_gap}",
                  (lambda g=_gap: ([sphere((0, 0, 0), 20)], [tangent_plate(20, 7.0, g, z=-2.5)])), _exp))


@pytest.mark.parametrize("name,make,expected", CASES, ids=[c[0] for c in CASES])
def test_pair_verdict(name, make, expected):
    sa, sb = make()
    sa = [G.mesh(s, DEFL) for s in sa]
    sb = [G.mesh(s, DEFL) for s in sb]
    v = O.part_pair_verdict(sa, sb, TOL, {}, DEFL)
    kind = "free" if v == O.FREE else "conflict"
    assert kind == expected.split(":")[0], v
    if ":" in expected:
        assert v == expected.split(":")[1]


def _strip_face_at_x(shape, x):
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepTools import breptools
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        f = topods.Face(ex.Current())
        bb = G.bbox(f).Get()
        if abs(bb[0] - x) < 1e-6 and abs(bb[3] - x) < 1e-6:
            breptools.Clean(f)
        ex.Next()


@pytest.mark.parametrize("shift,expected", [(10.0, "conflict"), (8.0, "conflict"), (15.0, "free")])
def test_untriangulated_face(shift, expected):
    """A face without triangulation must not hide a contact."""
    a = G.mesh(cube((0, 0, 0), 10), DEFL)
    b = G.mesh(cube((shift, 0, 0), 10), DEFL)
    _strip_face_at_x(a, 5.0)
    v = O.part_pair_verdict([a], [b], TOL, {}, DEFL)
    assert ("free" if v == O.FREE else "conflict") == expected


def test_verify_plan_interface():
    asm = {"A": [G.mesh(cube((0, 0, 0), 10))], "B": [G.mesh(cube((8, 0, 0), 10))]}
    assert len(O.verify_plan(asm, {"A": [0, 0, 0], "B": [0, 0, 0]}, TOL, DEFL)) == 1
    assert O.verify_plan(asm, {"A": [-20, 0, 0], "B": [20, 0, 0]}, TOL, DEFL) == []


def test_undetermined_is_never_free(monkeypatch):
    """If every exact distance fails, the verifier abstains instead of accepting."""
    monkeypatch.setattr(O, "_exact_distance", lambda a, b: (None, False))
    asm = {"A": [G.mesh(cube((0, 0, 0), 10))], "B": [G.mesh(cube((10.05, 0, 0), 10))]}
    details = {}
    coll = O.verify_plan(asm, {}, TOL, DEFL, details=details)
    assert coll == [("A", "B")]
    assert details[("A", "B")] == O.UNDET
