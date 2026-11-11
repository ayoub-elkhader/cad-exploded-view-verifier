"""Propose - verify - repair loop.

1. **Propose** a centroid plan.
2. **Verify** it with the interference verifier (:func:`.oracle.verify_plan`).
3. **Repair**, in this order:
   a. deterministic axial spreading of each cluster of colliding parts, with an
      increasing gap (no learning, no network);
   b. if conflicts remain, a *residual proposer* (random, scale, or an LLM) suggests new
      vectors for the parts still in conflict only; each proposal is verified, the working
      plan is replaced if it is not worse, and the best plan is kept;
   c. optionally, a feasibility repair that looks for straight-line extraction directions
      for the parts that block a sequential disassembly.

The verifier has the last word at every step: a plan is reported as accepted only if it
has no conflicting and no undetermined pair.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import disassembly as D
from . import geometry as G
from . import oracle as O
from .proposers import (ResidualContext, axial_spread, centroid_plan,
                        connected_components)

CANDIDATE_DIRECTIONS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


@dataclass
class RepairResult:
    plan: Dict[str, List[float]]
    conflicts: List[Tuple[str, str]]
    history: List[dict] = field(default_factory=list)
    oracle_calls: int = 0
    disassembly_feasible: Optional[bool] = None
    removal_order: List[str] = field(default_factory=list)
    stuck_parts: List[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def accepted(self) -> bool:
        return not self.conflicts


def deterministic_spread(assembly, factor=2.0, tol=G.TOL, deflection=G.DEFLECTION,
                         max_iter=4, verify=O.verify_plan, log=None):
    """Centroid plan, then axial spreading with gap 1.2 + 0.6 * k (k = 1..max_iter).
    Returns (plan, conflicts, oracle_calls)."""
    plan = centroid_plan(assembly, factor)
    coll = verify(assembly, plan, tol, deflection)
    calls = 1
    if log is not None:
        log.append({"step": "centroid", "conflicts": len(coll)})
    if not coll:
        return plan, coll, calls
    points = {p: G.part_points(sh) for p, sh in assembly.items()}
    best = (len(coll), dict(plan), coll)
    base = coll
    for k in range(1, max_iter + 1):
        trial = dict(centroid_plan(assembly, factor))
        for comp in connected_components(base):
            if len(comp) >= 2:
                axial_spread(trial, assembly, points, comp, 1.2 + 0.6 * k)
        c2 = verify(assembly, trial, tol, deflection)
        calls += 1
        if log is not None:
            log.append({"step": f"spread_{k}", "conflicts": len(c2)})
        if len(c2) < best[0]:
            best = (len(c2), dict(trial), c2)
        if not c2:
            break
    return best[1], best[2], calls


def residual_loop(assembly, plan, conflicts, proposer: Callable, budget=4, factor=2.0,
                  tol=G.TOL, deflection=G.DEFLECTION, case_id="case", verify=O.verify_plan,
                  log=None):
    """Up to ``budget`` proposals for the parts still in conflict.
    Returns (best_plan, best_conflicts, oracle_calls)."""
    start = dict(plan)
    best = (len(conflicts), dict(plan), list(conflicts))
    prev: List[Tuple[str, str]] = []
    calls = 0
    for rnd in range(1, budget + 1):
        if not conflicts:
            break
        ctx = ResidualContext(assembly=assembly, plan=plan, start=start, conflicts=conflicts,
                              round=rnd, factor=factor, case_id=case_id,
                              previous_conflicts=prev)
        try:
            fix = proposer(ctx) or {}
        except Exception as exc:   # a failing proposer must not crash the loop
            if log is not None:
                log.append({"step": f"proposal_{rnd}", "error": f"{type(exc).__name__}: {exc}"})
            break
        allowed = set(ctx.involved)
        fix = {p: v for p, v in fix.items() if p in allowed}
        if not fix:
            if log is not None:
                log.append({"step": f"proposal_{rnd}", "error": "no usable vector"})
            continue
        trial = dict(plan)
        trial.update(fix)
        c2 = verify(assembly, trial, tol, deflection)
        calls += 1
        if log is not None:
            log.append({"step": f"proposal_{rnd}", "proposer": getattr(proposer, "name", "custom"),
                        "conflicts": len(c2)})
        if len(c2) < best[0]:
            best = (len(c2), dict(trial), c2)
        prev = c2
        if len(c2) <= len(conflicts):
            plan, conflicts = trial, c2
    return best[1], best[2], calls


def feasibility_repair(assembly, plan, tol=G.TOL, deflection=G.DEFLECTION, steps=8,
                       reach=1.5, max_iter=3, verify=O.verify_plan):
    """For each part that blocks the sequential disassembly, try its current direction,
    the six axes and the radial direction; keep the first one along which the part can be
    extracted *and* the final state stays free of conflicts.
    Returns (plan, feasible, order, stuck, oracle_calls)."""
    ids = list(assembly)
    state = D.assembled_state(assembly, tol)
    diag = state[2]
    travel = reach * diag
    centers = {p: G.center_diag(assembly[p])[0] for p in ids}
    g = [sum(c[k] for c in centers.values()) / len(centers) for k in range(3)]
    calls = 0
    feas, order, stuck = D.sequential_disassembly(assembly, plan, steps, tol, deflection, reach)
    if feas:
        return dict(plan), True, order, [], calls
    best = dict(plan)
    for _ in range(max_iter):
        changed = False
        for i in list(stuck):
            others = [j for j in ids if j != i]
            v = best.get(i, [0, 0, 0])
            mag = max((v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5, 0.5 * diag)
            cands = []
            if D._unit(v):
                cands.append(D._unit(v))
            cands += [list(d) for d in CANDIDATE_DIRECTIONS]
            radial = D._unit([centers[i][k] - g[k] for k in range(3)])
            if radial:
                cands.append(radial)
            for d in cands:
                du = D._unit(d)
                if du is None or not D.is_removable(i, others, assembly, state, du, travel,
                                                     steps, tol, deflection):
                    continue
                trial = dict(best)
                trial[i] = [du[0] * mag, du[1] * mag, du[2] * mag]
                calls += 1
                if not verify(assembly, trial, tol, deflection):
                    best, changed = trial, True
                    break
        feas, order, stuck = D.sequential_disassembly(assembly, best, steps, tol, deflection, reach)
        if feas or not changed:
            break
    return best, feas, order, stuck, calls


def explode(assembly, factor=2.0, tol=G.TOL, deflection=G.DEFLECTION, spread_iter=4,
            proposer: Optional[Callable] = None, budget=4, check_disassembly=False,
            repair_disassembly=False, case_id="case") -> RepairResult:
    """Full pipeline: centroid -> spread -> residual proposer -> (disassembly)."""
    t0 = time.time()
    log: List[dict] = []
    plan, coll, calls = deterministic_spread(assembly, factor, tol, deflection, spread_iter, log=log)
    if coll and proposer is not None:
        plan, coll, c = residual_loop(assembly, plan, coll, proposer, budget, factor, tol,
                                      deflection, case_id, log=log)
        calls += c
    res = RepairResult(plan=plan, conflicts=coll, history=log, oracle_calls=calls)
    if check_disassembly or repair_disassembly:
        if repair_disassembly and not coll:
            plan2, feas, order, stuck, c = feasibility_repair(assembly, plan, tol, deflection)
            res.plan, res.oracle_calls = plan2, res.oracle_calls + c
            log.append({"step": "feasibility_repair", "feasible": feas})
        else:
            feas, order, stuck = D.sequential_disassembly(assembly, plan, tol=tol,
                                                          deflection=deflection)
        res.disassembly_feasible, res.removal_order, res.stuck_parts = feas, order, stuck
    res.seconds = time.time() - t0
    return res
