"""Exploded-view proposers.

* :func:`centroid_plan` - every part moves away from the global centre by
  ``factor * diagonal`` of its own bounding box.
* :func:`axial_spread` - deterministic repair of a cluster of colliding parts: the parts
  are spread along the first principal axis of their vertices (typical of coaxial stacks).
* Residual proposers, called by the repair loop with the parts still in conflict:
  :class:`RandomProposer`, :class:`ScaleProposer` and, optionally,
  :class:`exploded_view_verifier.llm.LLMProposer`.

A residual proposer is a callable ``proposer(context) -> {part_id: [dx, dy, dz]}``
where ``context`` is a :class:`ResidualContext`.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from . import geometry as G

ESCALATION = (1.0, 1.5, 2.25, 3.375)   # magnitude multiplier for rounds 1..4


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else None


def centroid_plan(assembly, factor: float = 2.0) -> Dict[str, List[float]]:
    centers, diags = {}, {}
    for p, sh in assembly.items():
        centers[p], diags[p] = G.center_diag(sh)
    n = len(centers)
    g = [sum(c[k] for c in centers.values()) / n for k in range(3)]
    plan = {}
    for p in assembly:
        d = [centers[p][k] - g[k] for k in range(3)]
        u = _unit(d) or [0.0, 0.0, 1.0]
        mag = factor * diags[p]
        plan[p] = [u[0] * mag, u[1] * mag, u[2] * mag]
    return plan


# ----------------------------------------------------------------------------- axial spread
def principal_axis(points) -> Tuple[float, float, float]:
    """First principal axis of a point cloud, by power iteration in scalar arithmetic.

    BLAS-backed numpy routines are avoided on purpose: on some Windows set-ups the BLAS
    shipped with numpy and the one loaded by pythonocc conflict and crash the process.
    """
    if len(points) < 2:
        return (1.0, 0.0, 0.0)
    c = points - points.mean(axis=0)
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    cxx, cyy, czz = float((x * x).sum()), float((y * y).sum()), float((z * z).sum())
    cxy, cxz, cyz = float((x * y).sum()), float((x * z).sum()), float((y * z).sum())
    cov = ((cxx, cxy, cxz), (cxy, cyy, cyz), (cxz, cyz, czz))
    v = [1.0, 1.0, 1.0]
    for _ in range(200):
        nv = [sum(cov[r][k] * v[k] for k in range(3)) for r in range(3)]
        n = math.sqrt(sum(a * a for a in nv))
        if n < 1e-12:
            return (1.0, 0.0, 0.0)
        v = [a / n for a in nv]
    return tuple(v)


def connected_components(pairs) -> List[List[str]]:
    adj: Dict[str, set] = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for node in sorted(adj):
        if node in seen:
            continue
        stack, comp = [node], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))
    return comps


def axial_spread(plan, assembly, points, cluster, gap):
    """Modify ``plan`` in place: spread the parts of ``cluster`` along their shared
    principal axis, spacing = (sum of half thicknesses) * gap, keeping the cluster centre."""
    import numpy as np
    arrs = [points[p] for p in cluster if len(points[p])]
    if not arrs:
        return
    allpts = np.vstack(arrs)
    if len(allpts) < 2:
        return
    ax = principal_axis(allpts)
    info = []
    for p in cluster:
        c0, _ = G.center_diag(assembly[p])
        cur = [c0[k] + plan[p][k] for k in range(3)]
        s = sum(cur[k] * ax[k] for k in range(3))
        pts = points[p]
        if len(pts):
            proj = pts[:, 0] * ax[0] + pts[:, 1] * ax[1] + pts[:, 2] * ax[2]
            ext = float(proj.max() - proj.min())
        else:
            ext = 1.0
        info.append((p, s, ext))
    info.sort(key=lambda r: r[1])
    targets, t, prev = {}, 0.0, None
    for p, s, ext in info:
        if prev is not None:
            t += (prev / 2 + ext / 2) * gap
        targets[p] = t
        prev = ext
    mean_t = sum(targets.values()) / len(targets)
    mean_s = sum(s for _, s, _ in info) / len(info)
    for p, s, _ in info:
        delta = (targets[p] - mean_t) - (s - mean_s)
        plan[p] = [plan[p][k] + delta * ax[k] for k in range(3)]


# ----------------------------------------------------------------------------- residual
@dataclass
class ResidualContext:
    """What a residual proposer receives at each round of the repair loop."""
    assembly: dict
    plan: Dict[str, List[float]]          # current working plan
    start: Dict[str, List[float]]         # plan at the start of the loop
    conflicts: List[Tuple[str, str]]      # pairs still in conflict
    round: int                            # 1-based round number
    factor: float = 2.0
    case_id: str = "case"
    previous_conflicts: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def involved(self) -> List[str]:
        return sorted({a for a, _ in self.conflicts} | {b for _, b in self.conflicts})


class RandomProposer:
    """Uniform random direction on the sphere for each part in conflict.

    Magnitude: ``max(|v|, factor * diag) * ESCALATION[round - 1]``. The seed is derived
    from ``seed``, the case id, the round and the part id, so runs are reproducible.
    """

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(self, ctx: ResidualContext):
        fix = {}
        esc = ESCALATION[min(ctx.round, len(ESCALATION)) - 1]
        for p in ctx.involved:
            v = ctx.plan.get(p, [0.0, 0.0, 0.0])
            _, diag = G.center_diag(ctx.assembly[p])
            mag = max(math.sqrt(sum(x * x for x in v)), ctx.factor * diag) * esc
            rng = random.Random(self.seed + zlib.crc32(f"{ctx.case_id}|{ctx.round}|{p}".encode()))
            u = None
            while u is None:
                u = _unit([rng.gauss(0, 1) for _ in range(3)])
            fix[p] = [x * mag for x in u]
        return fix


class ScaleProposer:
    """Keep the direction of the starting vector and multiply it by 1.5 ** round."""

    name = "scale"

    def __call__(self, ctx: ResidualContext):
        fix = {}
        for p in ctx.involved:
            v0 = ctx.start.get(p, [0.0, 0.0, 0.0])
            fix[p] = [x * (1.5 ** ctx.round) for x in v0]
        return fix
