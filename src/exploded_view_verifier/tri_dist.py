"""Vectorised exact distance between pairs of triangles (numpy).

``tri_tri(T1, T2)`` takes two arrays of shape (N, 3, 3) and returns the (N,) minimum
Euclidean distances between ``T1[k]`` and ``T2[k]`` (0 when the triangles intersect).
For two disjoint triangles the minimum is reached between a vertex and the other triangle
or between two edges; when they intersect, an edge of one crosses the other.
Algorithms: Ericson, *Real-Time Collision Detection*, 5.1.5 (point-triangle) and 5.1.9
(segment-segment).
"""
import numpy as np


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def point_tri(p, A, B, C):
    """Distance from points p (N, 3) to triangles ABC (N, 3 each)."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - B
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - C
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    q = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def put(mask, val):
        m = mask & ~done
        q[m] = val[m] if val.ndim == 2 else val
        done[m] = True

    put((d1 <= 0) & (d2 <= 0), A)
    put((d3 >= 0) & (d4 <= d3), B)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), A + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), C)
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), A + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), B + w2[:, None] * (C - B))
        den = 1.0 / (va + vb + vc)
        vv, ww = vb * den, vc * den
        rest = ~done
        q[rest] = (A + ab * vv[:, None] + ac * ww[:, None])[rest]
    return np.linalg.norm(p - q, axis=1)


def seg_seg(p1, q1, p2, q2):
    """Minimum distance between segments [p1, q1] and [p2, q2] (N, 3 each)."""
    eps = 1e-18
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    c, b = _dot(d1, r), _dot(d1, d2)
    den = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > eps, np.clip((b * f - c * e) / den, 0, 1), 0.0)
        t = np.where(e > eps, (b * s + f) / e, 0.0)
        s = np.where(t < 0, np.where(a > eps, np.clip(-c / a, 0, 1), 0.0), s)
        s = np.where(t > 1, np.where(a > eps, np.clip((b - c) / a, 0, 1), 0.0), s)
        t = np.clip(t, 0, 1)
        # degenerate cases (segment reduced to a point)
        s = np.where(a <= eps, 0.0, s)
        t = np.where((a <= eps) & (e > eps), np.clip(f / e, 0, 1), t)
        s = np.where((e <= eps) & (a > eps), np.clip(-c / a, 0, 1), s)
        t = np.where(e <= eps, 0.0, t)
    return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)


def seg_crosses_tri(p, q, A, B, C):
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.cross(B - A, C - A)
        dp, dq = _dot(p - A, n), _dot(q - A, n)
        x = p + (q - p) * (dp / (dp - dq))[:, None]
        c1 = _dot(np.cross(B - A, x - A), n)
        c2 = _dot(np.cross(C - B, x - B), n)
        c3 = _dot(np.cross(A - C, x - C), n)
        inside = ((c1 >= 0) & (c2 >= 0) & (c3 >= 0)) | ((c1 <= 0) & (c2 <= 0) & (c3 <= 0))
        return (dp * dq <= 0) & (dp != dq) & inside


def tri_tri(T1, T2):
    T1 = np.asarray(T1, float)
    T2 = np.asarray(T2, float)
    best = np.full(len(T1), np.inf)
    for k in range(3):
        best = np.minimum(best, point_tri(T1[:, k], T2[:, 0], T2[:, 1], T2[:, 2]))
        best = np.minimum(best, point_tri(T2[:, k], T1[:, 0], T1[:, 1], T1[:, 2]))
    for i in range(3):
        for j in range(3):
            best = np.minimum(best, seg_seg(T1[:, i], T1[:, (i + 1) % 3],
                                            T2[:, j], T2[:, (j + 1) % 3]))
    cross = np.zeros(len(T1), bool)
    for P, Q in ((T1, T2), (T2, T1)):
        for k in range(3):
            cross |= seg_crosses_tri(P[:, k], P[:, (k + 1) % 3], Q[:, 0], Q[:, 1], Q[:, 2])
    best[cross] = 0.0
    return best
