"""Disassembly feasibility of an exploded-view plan.

An exploded view without interference in its final state does not imply a physically
possible disassembly. Two models are provided:

* :func:`simultaneous_feasible` - every part moves at once along a straight line from its
  assembled position (t = 0) to its exploded position (t = 1). A pair is *blocking* if it
  is not in contact when assembled but comes into contact during the motion.
* :func:`sequential_disassembly` - parts are removed one at a time, each along the
  direction of its plan vector. A part can be removed if, while it travels, it creates no
  new contact with the remaining parts and is clear of all of them at the end.

Both models use the mesh proximity test and a discrete set of positions; they do not
prove the absence of collision between two sampled positions, and they ignore rotations,
fasteners and mechanical constraints.
"""
from __future__ import annotations

from itertools import combinations

from . import geometry as G


def _solid_box(s, gap=0.0):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    b = Bnd_Box()
    if gap:
        b.SetGap(gap)
    brepbndlib.Add(s, b, False)   # exact geometry, not the mesh
    return b


def _union(boxes):
    from OCC.Core.Bnd import Bnd_Box
    tot = Bnd_Box()
    for b in boxes:
        tot.Add(b)
    return tot


def _unit(v):
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return [v[0] / n, v[1] / n, v[2] / n] if n > 1e-9 else None


def _pair_in_contact(shapes_a, boxes_a, shapes_b, boxes_b, tol):
    for sa, ba in zip(shapes_a, boxes_a):
        for sb, bb in zip(shapes_b, boxes_b):
            if ba.IsOut(bb):
                continue
            if G.in_contact(sa, sb, tol):
                return True
    return False


def simultaneous_feasible(assembly, plan, steps=6, tol=G.TOL, deflection=G.DEFLECTION):
    """Return (feasible, blocking_pairs) for the simultaneous straight-line model."""
    ids = list(assembly)
    contact0, blocking = {}, set()
    for k in range(steps + 1):
        t = k / steps
        placed, sboxes, obox = {}, {}, {}
        for o in ids:
            v = plan.get(o, [0, 0, 0])
            placed[o] = G.translate(assembly[o], [t * v[0], t * v[1], t * v[2]], deflection)
            sboxes[o] = [_solid_box(s, tol) for s in placed[o]]
            obox[o] = _union(sboxes[o])
        for a, b in combinations(ids, 2):
            c = (not obox[a].IsOut(obox[b])) and _pair_in_contact(
                placed[a], sboxes[a], placed[b], sboxes[b], tol)
            if k == 0:
                contact0[(a, b)] = c
            elif c and not contact0[(a, b)]:
                blocking.add((a, b))
    return not blocking, sorted(blocking)


def assembled_state(assembly, tol=G.TOL):
    """Boxes, contact graph and overall diagonal of the assembled state."""
    from OCC.Core.Bnd import Bnd_Box
    ids = list(assembly)
    sbox = {o: [_solid_box(s, tol) for s in assembly[o]] for o in ids}
    obox = {o: _union(sbox[o]) for o in ids}
    total = Bnd_Box()
    for b in obox.values():
        total.Add(b)
    xmin, ymin, zmin, xmax, ymax, zmax = total.Get()
    diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    contact0 = {}
    for a, b in combinations(ids, 2):
        c = (not obox[a].IsOut(obox[b])) and _pair_in_contact(
            assembly[a], sbox[a], assembly[b], sbox[b], tol)
        contact0[(a, b)] = contact0[(b, a)] = c
    return sbox, obox, diag, contact0


def is_removable(i, others, assembly, state, d, travel, steps=8, tol=G.TOL,
                 deflection=G.DEFLECTION):
    """Can part ``i`` leave along unit direction ``d`` (distance ``travel``) without
    hitting ``others``? No new contact during the motion, and clear at the end."""
    sbox, obox, _, contact0 = state
    last_contact = True
    for k in range(1, steps + 1):
        t = travel * k / steps
        moved = G.translate(assembly[i], [t * d[0], t * d[1], t * d[2]], deflection)
        mb = [_solid_box(s, tol) for s in moved]
        mob = _union(mb)
        any_contact = False
        for j in others:
            if mob.IsOut(obox[j]):
                continue
            if _pair_in_contact(moved, mb, assembly[j], sbox[j], tol):
                any_contact = True
                if not contact0[(i, j)]:
                    return False          # new collision: direction blocked
        last_contact = any_contact
    return not last_contact


def sequential_disassembly(assembly, plan, steps=8, tol=G.TOL, deflection=G.DEFLECTION,
                           reach=1.5):
    """Return (feasible, removal_order, stuck_parts).

    Parts are tried in sorted order so that the result is deterministic.
    """
    state = assembled_state(assembly, tol)
    travel = reach * state[2]
    contact0 = state[3]
    remaining, order = set(assembly), []
    while remaining:
        removed = None
        for i in sorted(remaining):
            others = remaining - {i}
            d = _unit(plan.get(i, [0, 0, 0]))
            if d is None:   # no vector: removable only if it touches nothing
                if not any(contact0[(i, j)] for j in others):
                    removed = i
                    break
                continue
            if is_removable(i, others, assembly, state, d, travel, steps, tol, deflection):
                removed = i
                break
        if removed is None:
            return False, order, sorted(remaining)
        order.append(removed)
        remaining.discard(removed)
    return True, order, []
