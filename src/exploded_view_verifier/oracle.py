"""Conservative interference verifier for exploded views, with abstention.

For every pair of parts (each part is a list of placed, meshed solids) the verifier
returns one of:

* ``"free"``           - the boundaries are more than ``tol`` apart and neither solid
                         contains the other;
* ``"exact_fallback"`` - an exact B-rep distance (or a face/face intersection) shows the
                         boundaries are within ``tol``: contact or interpenetration;
* ``"inclusion"``      - one solid lies entirely inside the other;
* ``"undetermined"``   - the computation could not conclude (failed OCCT call, open
                         shell, contradictory estimates). An undetermined pair is never
                         reported as free.

A plan is accepted only if no pair is in conflict **or** undetermined.

How a pair of solids is decided
-------------------------------
1. *Candidate faces.* A pair of faces is discarded only if their exact bounding boxes,
   enlarged by ``tol``, are disjoint, or if all the boxes of their triangles are more than
   ``tol + 2 * deflection`` apart (two faces within ``tol`` have meshes within
   ``tol + 2 * deflection``, each mesh being within one deflection of its face). Faces
   without triangulation are filtered by their box only.
2. *Exact distance.* The remaining face pairs, closest first, go through
   ``BRepExtrema_DistShapeShape``; the first one at ``<= tol`` is a conflict.
3. *Consistency check (2.3+).* An exact distance larger than the mesh distance plus the
   mesh-to-surface gaps is provably wrong (observed on toroidal faces). The face/face
   section (``BRepAlgoAPI_Section``) then decides; without an intersection, a bracket on
   fine meshes (deflection 0.01 mm) decides; otherwise the verdict is undetermined.
4. *Containment.* If the boundaries are apart but the boxes overlap, a point of one solid
   is classified against the other (``BRepClass3d_SolidClassifier``). Since the
   boundaries are separated, one interior point means the whole solid is inside.

Versions
--------
``ORACLE_VERSION`` (environment variable of the same name) selects the variant:

* ``"2.3"`` (default) - steps 1-4 above, planar faces treated as having no sag;
* ``"2.3.1"`` - mesh-to-face gap equal to the deflection for planar faces too
  (chords of curved edges of a planar face can leave the face);
* ``"2.4"`` - a pair not certified free by the meshes is free only if the face/face
  section is empty (see :func:`_confirm_free_v24`);
* ``"2.2"`` - steps 1, 2 and 4 only.

Domain of validity: closed B-rep solids, rigid translations, tolerance ``tol``
(default 0.1 mm) on meshes of the given deflection (default 0.5 mm). The check is
performed at the final (exploded) position only; it says nothing about the motion
between the assembled and exploded states (see :mod:`.disassembly`).
"""
from __future__ import annotations

import os
from itertools import combinations

from . import geometry as G
from .tri_dist import tri_tri

ORACLE_VERSION = os.environ.get("ORACLE_VERSION", "2.3")
DEFLECTION = G.DEFLECTION
FINE = 0.01   # deflection of the fine meshes used to bracket a distance (mm)

FREE, EXACT, INCL, UNDET = "free", "exact_fallback", "inclusion", "undetermined"
CONFLICT_KINDS = (EXACT, INCL, UNDET)
_FREE_PAIR = "free_pair"   # internal: a face pair certified free

_FACE_CACHE: dict = {}     # id(solid) -> (solid, face data); cleared by verify_placed


def _bump(stats, key, n=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + n


# ----------------------------------------------------------------------------- OCCT calls
def _exact_distance(a, b):
    """Exact B-rep distance and inner-solution flag; (None, False) if the call fails.
    For solids, OCCT returns 0 with ``InnerSolution()`` true when one is inside the other."""
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    try:
        d = BRepExtrema_DistShapeShape(a, b)
        if d.IsDone() and d.NbSolution() > 0:
            return d.Value(), bool(d.InnerSolution())
    except Exception:
        pass
    return None, False


def _is_closed_solid(s):
    """True if every shell of the solid is closed (required by the point-in-solid test)."""
    from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_SHELL
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.BRep import BRep_Tool
    if s.ShapeType() != TopAbs_SOLID:
        return False
    ex, n = TopExp_Explorer(s, TopAbs_SHELL), 0
    while ex.More():
        n += 1
        if not BRep_Tool.IsClosed(ex.Current()):
            return False
        ex.Next()
    return n > 0


def _sample_point(s):
    """An exact point on the boundary of the solid: a vertex, else a mesh node."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_FACE
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopLoc import TopLoc_Location
    ex = TopExp_Explorer(s, TopAbs_VERTEX)
    if ex.More():
        return BRep_Tool.Pnt(topods.Vertex(ex.Current()))
    ex = TopExp_Explorer(s, TopAbs_FACE)
    while ex.More():
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(topods.Face(ex.Current()), loc)
        if tri is not None and tri.NbNodes() > 0:
            return tri.Node(1).Transformed(loc.Transformation())
        ex.Next()
    return None


def _inside(point, solid, tol):
    """'in', 'out' or None (unknown) for a point with respect to a closed solid."""
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_ON
    try:
        st = BRepClass3d_SolidClassifier(solid, point, tol).State()
    except Exception:
        return None
    if st == TopAbs_IN:
        return "in"
    if st in (TopAbs_OUT, TopAbs_ON):
        # ON cannot happen when the boundaries are more than tol apart
        return "out"
    return None


def _box_inside(inner, outer, tol):
    """Necessary condition for inclusion: the box of ``inner`` fits in that of ``outer``."""
    i, o = G.bbox(inner).Get(), G.bbox(outer).Get()
    return (all(i[k] >= o[k] - tol for k in range(3))
            and all(i[k] <= o[k] + tol for k in range(3, 6)))


def _volume(s):
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    g = GProp_GProps()
    brepgprop.VolumeProperties(s, g)
    return abs(g.Mass())


def _faces_intersect(f, g):
    """True if the boolean section of two faces contains an edge or a vertex; None if the
    operation fails."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    try:
        sec = BRepAlgoAPI_Section(f, g)
        sec.Build()
        if not sec.IsDone() or sec.HasErrors():
            return None
        sh = sec.Shape()
        return TopExp_Explorer(sh, TopAbs_EDGE).More() or TopExp_Explorer(sh, TopAbs_VERTEX).More()
    except Exception:
        return None


# ----------------------------------------------------------------------------- face data
def _face_box_safe(f, deflection):
    """Union of the optimal exact box of a face and the box of its triangulation enlarged
    by the deflection, so that a defect of either method cannot discard a face."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    bx = Bnd_Box()
    try:
        brepbndlib.AddOptimal(f, bx, False, False)
    except Exception:
        brepbndlib.Add(f, bx, False)
    if BRep_Tool.Triangulation(f, TopLoc_Location()) is not None:
        bm = Bnd_Box()
        brepbndlib.Add(f, bm, True)
        bm.Enlarge(deflection)
        bx.Add(bm)
    return bx


def _face_triangles(f):
    """Triangles (n, 3, 3) of the triangulation of a face (world frame), or None."""
    import numpy as np
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation(f, loc)
    if tri is None or tri.NbTriangles() == 0:
        return None
    tr = loc.Transformation()
    P = np.array([[q.X(), q.Y(), q.Z()] for q in
                  (tri.Node(k).Transformed(tr) for k in range(1, tri.NbNodes() + 1))])
    idx = np.array([tri.Triangle(t).Get() for t in range(1, tri.NbTriangles() + 1)]) - 1
    return P[idx]


def _face_data(s, deflection=DEFLECTION):
    """Faces of a solid: (face, exact box lo/hi, triangle boxes lo/hi or None)."""
    import numpy as np
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    hit = _FACE_CACHE.get(id(s))
    if hit is not None and hit[0] is s:
        return hit[1]
    out = []
    ex = TopExp_Explorer(s, TopAbs_FACE)
    while ex.More():
        f = topods.Face(ex.Current())
        bx = np.array(_face_box_safe(f, deflection).Get())
        V = _face_triangles(f)
        tl = th = None
        if V is not None:
            tl, th = V.min(axis=1), V.max(axis=1)
        out.append((f, bx[:3], bx[3:], tl, th))
        ex.Next()
    _FACE_CACHE[id(s)] = (s, out)
    return out


def _box_gap(lo1, hi1, lo2, hi2):
    import numpy as np
    g = np.maximum(0.0, np.maximum(lo2 - hi1, lo1 - hi2))
    return float(np.sqrt((g * g).sum()))


def _tri_box_gap(tl1, th1, tl2, th2, thr):
    """Smallest gap between a triangle box of face 1 and one of face 2, or infinity if it
    exceeds ``thr`` (a lower bound of the distance between the meshes)."""
    import numpy as np
    lo2, hi2 = tl2.min(axis=0), th2.max(axis=0)
    g1 = np.maximum(0, np.maximum(lo2 - th1, tl1 - hi2))
    k1 = np.sqrt((g1 * g1).sum(1)) <= thr
    lo1, hi1 = tl1.min(axis=0), th1.max(axis=0)
    g2 = np.maximum(0, np.maximum(lo1 - th2, tl2 - hi1))
    k2 = np.sqrt((g2 * g2).sum(1)) <= thr
    a_lo, a_hi, b_lo, b_hi = tl1[k1], th1[k1], tl2[k2], th2[k2]
    if not len(a_lo) or not len(b_lo):
        return float("inf")
    best = float("inf")
    step = max(1, 2_000_000 // max(1, len(b_lo)))
    for s in range(0, len(a_lo), step):
        g = np.maximum(0, np.maximum(b_lo[None] - a_hi[s:s + step, None],
                                     a_lo[s:s + step, None] - b_hi[None]))
        best = min(best, float(np.sqrt((g * g).sum(-1)).min()))
    return best if best <= thr else float("inf")


def _sag(f, deflection):
    """Maximum mesh-to-face gap. 2.3: 0 for a planar face; 2.3.1 and later: the deflection
    for every face (a chord of a curved edge of a planar face can leave the face)."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane
    if ORACLE_VERSION == "2.3" and BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane:
        return 0.0
    return deflection


def _mesh_distance_below(TA, TB, limit):
    """Minimum distance between the two meshes if it is < limit, else None."""
    import numpy as np
    la, ha = TA.min(axis=1), TA.max(axis=1)
    lb, hb = TB.min(axis=1), TB.max(axis=1)
    best = None
    step = max(1, 2_000_000 // max(1, len(TB)))
    for s in range(0, len(TA), step):
        g = np.maximum(0, np.maximum(lb[None] - ha[s:s + step, None], la[s:s + step, None] - hb[None]))
        ia, ib = np.nonzero(np.sqrt((g * g).sum(-1)) < limit)
        if len(ia):
            m = float(tri_tri(TA[s + ia], TB[ib]).min())
            if m < limit and (best is None or m < best):
                best = m
    return best


# ----------------------------------------------------------------------------- decisions
def _fine_mesh_decision(f, g, tol, stats=None):
    """EXACT if the distance is surely <= tol, _FREE_PAIR if surely > tol, None otherwise
    (uncertainty band, or meshing failed). Works on fine-meshed copies of the faces."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopoDS import topods
    try:
        fc = topods.Face(BRepBuilderAPI_Copy(f, True, False).Shape())
        gc = topods.Face(BRepBuilderAPI_Copy(g, True, False).Shape())
        BRepMesh_IncrementalMesh(fc, FINE, False, 0.1, True).Perform()
        BRepMesh_IncrementalMesh(gc, FINE, False, 0.1, True).Perform()
    except Exception:
        return None
    TA, TB = _face_triangles(fc), _face_triangles(gc)
    if TA is None or TB is None:
        return None
    sf, sg = _sag(f, FINE), _sag(g, FINE)
    m = _mesh_distance_below(TA, TB, tol + sf + sg + 1.0)
    _bump(stats, "fine_mesh_checks")
    if m is None or m - sf - sg > tol:
        _bump(stats, "fine_mesh_free")
        return _FREE_PAIR
    if m + sf + sg <= tol:
        _bump(stats, "fine_mesh_conflict")
        return EXACT
    return None


def _check_consistency(f, g, d, tol, deflection, stats=None):
    """2.3: EXACT if the exact distance ``d`` is provably wrong and the faces intersect;
    UNDET if it is provably wrong and nothing decides; None if ``d`` is consistent."""
    sf, sg = _sag(f, deflection), _sag(g, deflection)
    TA, TB = _face_triangles(f), _face_triangles(g)
    if TA is None or TB is None:
        return None
    if _mesh_distance_below(TA, TB, d - sf - sg - 1e-6) is None:
        return None
    _bump(stats, "exact_distance_inconsistent")
    if _faces_intersect(f, g):
        _bump(stats, "section_conflict")
        return EXACT
    v = _fine_mesh_decision(f, g, tol, stats)
    if v is not None:
        return v
    _bump(stats, "inconsistent_undetermined")
    return UNDET


def _confirm_free_v24(f, g, d, tol, deflection, stats=None):
    """2.4: the exact distance ``d`` (> tol) alone does not make the pair free.

    * If the meshes are more than ``tol + s_f + s_g`` apart, the true distance exceeds
      ``tol``: free, certified by the meshes.
    * Otherwise, two interpenetrating solids (without inclusion) have intersecting faces:
      a non-empty section is a conflict. With an empty section, ``d`` is kept only if the
      meshes are also beyond ``tol`` and ``d`` is consistent with them; in every other
      case the fine-mesh bracket decides, and the verdict is undetermined if it cannot.

    Returns EXACT, UNDET, _FREE_PAIR or None (free).
    """
    sf, sg = _sag(f, deflection), _sag(g, deflection)
    TA, TB = _face_triangles(f), _face_triangles(g)
    m = None
    if TA is not None and TB is not None:
        m = _mesh_distance_below(TA, TB, tol + sf + sg)
        if m is None:
            _bump(stats, "free_certified_by_meshes")
            return None
    _bump(stats, "section_checks")
    x = _faces_intersect(f, g)
    if x:
        _bump(stats, "section_conflict")
        return EXACT
    if x is False:
        if m is None:
            _bump(stats, "free_section_empty")
            return _FREE_PAIR
        if m > tol and d <= m + sf + sg + 1e-6:
            _bump(stats, "free_section_empty")
            return _FREE_PAIR
        _bump(stats, "exact_distance_inconsistent" if d > m + sf + sg + 1e-6 else "mesh_exact_disagree")
    else:
        _bump(stats, "section_failed")
    v = _fine_mesh_decision(f, g, tol, stats)
    if v is not None:
        return v
    _bump(stats, "inconsistent_undetermined")
    return UNDET


def _surface_verdict(a, b, tol, deflection, stats=None):
    """Boundary contact within tol (EXACT / INCL / UNDET), or None if the boundaries are
    more than tol apart. See the module docstring, steps 1-3."""
    thr = tol + 2.0 * deflection
    FA, FB = _face_data(a, deflection), _face_data(b, deflection)
    cand, untri = [], 0
    for fa, la, ha, tla, tha in FA:
        for fb, lb, hb, tlb, thb in FB:
            if _box_gap(la, ha, lb, hb) > tol:
                continue
            if tla is None or tlb is None:
                untri += 1
                cand.append((0.0, fa, fb))
                continue
            gap = _tri_box_gap(tla, tha, tlb, thb, thr)
            if gap <= thr:
                cand.append((gap, fa, fb))
    _bump(stats, "candidate_face_pairs", len(cand))
    if untri:
        _bump(stats, "untriangulated_face_pairs", untri)
    cand.sort(key=lambda c: c[0])
    for _, f, g in cand:
        _bump(stats, "face_exact_calls")
        d, _ = _exact_distance(f, g)
        if d is None:
            _bump(stats, "face_exact_failed")
            d, inner = _exact_distance(a, b)
            if d is None:
                return UNDET
            return (INCL if inner else EXACT) if d <= tol else None
        if d <= tol:
            return EXACT
        if ORACLE_VERSION == "2.4":
            v = _confirm_free_v24(f, g, d, tol, deflection, stats)
        elif ORACLE_VERSION != "2.2":
            v = _check_consistency(f, g, d, tol, deflection, stats)
        else:
            v = None
        if v is not None and v != _FREE_PAIR:
            return v
    return None


def _containment(a, b, tol, stats=None):
    """Boundaries more than tol apart: only inclusion remains to be checked."""
    if G.bbox(a).IsOut(G.bbox(b)):
        return FREE
    _bump(stats, "containment_tests")
    for inner, outer in ((a, b), (b, a)):
        # Necessary conditions; they also protect against a badly oriented solid.
        if not _box_inside(inner, outer, tol):
            continue
        if _volume(inner) > _volume(outer) * (1 + 1e-6):
            continue
        if not _is_closed_solid(outer):
            _bump(stats, "not_closed")
            return UNDET   # cannot tell whether `inner` is inside an open shell
        p = _sample_point(inner)
        if p is None:
            return UNDET
        st = _inside(p, outer, tol)
        if st is None:
            return UNDET
        if st == "in":
            return INCL
    return FREE


# ----------------------------------------------------------------------------- public API
def solid_pair_verdict(a, b, tol=G.TOL, stats=None, deflection=DEFLECTION):
    """Verdict for two placed, meshed solids."""
    if ORACLE_VERSION not in ("2.2", "2.3", "2.3.1", "2.4"):
        raise ValueError(f"Unknown ORACLE_VERSION {ORACLE_VERSION!r}")
    v = _surface_verdict(a, b, tol, deflection, stats)
    if v is not None:
        return v
    return _containment(a, b, tol, stats)


def part_pair_verdict(sa_list, sb_list, tol=G.TOL, stats=None, deflection=DEFLECTION):
    """Verdict for two parts (lists of solids). Priority: conflict > undetermined > free."""
    undetermined = False
    for a in sa_list:
        for b in sb_list:
            v = solid_pair_verdict(a, b, tol, stats, deflection)
            if v in (EXACT, INCL):
                return v
            if v == UNDET:
                undetermined = True
    return UNDET if undetermined else FREE


def verify_placed(placed, tol=G.TOL, stats=None, deflection=DEFLECTION):
    """``placed``: {part: [placed, meshed solids]}. Returns {(a, b): verdict} for every
    pair that is not free."""
    boxes = {k: G.combined_bbox(v) for k, v in placed.items()}
    for bx in boxes.values():
        bx.Enlarge(tol)
    _FACE_CACHE.clear()
    out = {}
    try:
        for a, b in combinations(list(placed), 2):
            if boxes[a].IsOut(boxes[b]):
                continue
            v = part_pair_verdict(placed[a], placed[b], tol, stats, deflection)
            if v != FREE:
                out[(a, b)] = v
    finally:
        _FACE_CACHE.clear()
    return out


def verify_plan(assembly, plan, tol=G.TOL, deflection=DEFLECTION, details=None, stats=None):
    """Apply ``plan`` to ``assembly`` and return the list of non-free pairs (conflicts and
    undetermined pairs). If ``details`` is a dict it receives {(a, b): verdict}."""
    placed = {p: G.translate(sh, plan.get(p, [0, 0, 0]), deflection) for p, sh in assembly.items()}
    res = verify_placed(placed, tol, stats, deflection)
    if details is not None:
        details.update(res)
    return list(res)
