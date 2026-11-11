"""Geometry helpers on top of pythonocc-core (OpenCASCADE).

An *assembly* is represented as a dict ``{part_id: [TopoDS_Shape, ...]}``: each part is a
list of solids already placed in the assembled position and meshed. An *exploded-view
plan* is a dict ``{part_id: [dx, dy, dz]}`` of rigid translations in millimetres.
"""
from __future__ import annotations

import contextlib
import io
import math
from typing import Dict, Iterable, List, Sequence, Tuple

Assembly = Dict[str, list]
Plan = Dict[str, List[float]]

TOL = 0.1          # contact tolerance (mm)
DEFLECTION = 0.5   # mesh deflection used by the verifier (mm)


def mesh(shape, deflection: float = DEFLECTION):
    """Triangulate a shape in place and return it."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    BRepMesh_IncrementalMesh(shape, deflection).Perform()
    return shape


def bbox(shape):
    """Bounding box using the triangulation when present (no gap)."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    b = Bnd_Box()
    b.SetGap(0.0)
    brepbndlib.Add(shape, b, True)
    return b


def combined_bbox(shapes: Iterable):
    from OCC.Core.Bnd import Bnd_Box
    tot = Bnd_Box()
    for s in shapes:
        tot.Add(bbox(s))
    return tot


def center_diag(shapes: Sequence) -> Tuple[Tuple[float, float, float], float]:
    """Centre and diagonal of the bounding box of a list of shapes."""
    xmin, ymin, zmin, xmax, ymax, zmax = combined_bbox(shapes).Get()
    c = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    return c, diag


def translate(shapes: Sequence, v: Sequence[float], deflection: float = DEFLECTION) -> list:
    """Return translated copies of ``shapes`` (meshed)."""
    from OCC.Core.gp import gp_Trsf, gp_Vec
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(float(v[0]), float(v[1]), float(v[2])))
    out = []
    for s in shapes:
        s2 = BRepBuilderAPI_Transform(s, t, True).Shape()
        mesh(s2, deflection)
        out.append(s2)
    return out


def in_contact(s1, s2, tol: float):
    """Mesh proximity test (BRepExtrema_ShapeProximity).

    Returns True / False, or None when OCCT reports the computation as not done. Used only
    by the disassembly test; the interference verifier (:mod:`.oracle`) does not rely on it.
    """
    from OCC.Core.BRepExtrema import BRepExtrema_ShapeProximity
    prox = BRepExtrema_ShapeProximity(s1, s2, tol)
    prox.Perform()
    if not prox.IsDone():
        return None
    ov = prox.OverlapSubShapes1()
    try:
        n = ov.Size()
    except Exception:
        n = ov.Extent()
    return n > 0


def load_solids(step_path: str) -> list:
    """Read a STEP file and return its individual solids (compounds are exploded). A root
    shape without any solid is returned as is."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    with contextlib.redirect_stdout(io.StringIO()):
        status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise ValueError(f"Cannot read STEP file: {step_path}")
    reader.TransferRoots()
    solids = []
    for i in range(1, reader.NbShapes() + 1):
        shape = reader.Shape(i)
        if shape.IsNull():
            continue
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        found = False
        while exp.More():
            solids.append(exp.Current())
            found = True
            exp.Next()
        if not found:
            # no solid (open shell, surfaces): keep the raw shape; the verifier will
            # abstain on it rather than ignore it
            solids.append(shape)
    return solids


def load_step_assembly(step_path: str, deflection: float = DEFLECTION) -> Assembly:
    """Load a STEP assembly as ``{"part_01": [solid], ...}`` (one solid = one part).

    Names and the product structure of the STEP file are not used: every solid becomes
    one part, in file order.
    """
    solids = load_solids(step_path)
    width = max(2, len(str(len(solids))))
    return {f"part_{k + 1:0{width}d}": [mesh(s, deflection)] for k, s in enumerate(solids)}


def write_step(shapes: Iterable, path: str) -> None:
    """Write shapes to a STEP file (AP214)."""
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    w = STEPControl_Writer()
    for s in shapes:
        w.Transfer(s, STEPControl_AsIs)
    with contextlib.redirect_stdout(io.StringIO()):
        status = w.Write(str(path))
    if status != IFSelect_RetDone:
        raise IOError(f"Cannot write STEP file: {path}")


def shape_points(shape, max_pts: int = 400):
    """Up to ``max_pts`` triangulation nodes of a meshed shape, as an (N, 3) array."""
    import numpy as np
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    pts = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                pts.append((p.X(), p.Y(), p.Z()))
        exp.Next()
    if not pts:
        return np.zeros((0, 3))
    arr = np.asarray(pts, dtype=float)
    if len(arr) > max_pts:                      # deterministic subsampling
        arr = arr[np.linspace(0, len(arr) - 1, max_pts).astype(int)]
    return arr


def part_points(shapes: Sequence):
    import numpy as np
    arrs = [a for a in (shape_points(s) for s in shapes) if len(a)]
    return np.vstack(arrs) if arrs else np.zeros((0, 3))
