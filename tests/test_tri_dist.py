"""Triangle-triangle distance against a brute-force reference (pure numpy, no OCC)."""
import numpy as np

from exploded_view_verifier.tri_dist import point_tri, seg_seg, tri_tri


def _brute(t1, t2, n=60):
    """Distance between two triangles by dense sampling of both (upper bound)."""
    u = np.linspace(0, 1, n)
    uu, vv = np.meshgrid(u, u)
    m = (uu + vv) <= 1
    w = np.stack([1 - uu[m] - vv[m], uu[m], vv[m]], 1)
    # element-wise products only: BLAS-backed matmul can crash next to pythonocc on Windows
    p1 = (w[:, :, None] * t1[None]).sum(1)
    p2 = (w[:, :, None] * t2[None]).sum(1)
    d = np.sqrt(((p1[:, None] - p2[None]) ** 2).sum(-1))
    return d.min()


def test_parallel_triangles():
    t1 = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    t2 = t1 + [0, 0, 2.5]
    assert np.allclose(tri_tri(t1, t2), [2.5])


def test_intersecting_triangles_have_zero_distance():
    t1 = np.array([[[-1, -1, 0], [2, -1, 0], [-1, 2, 0]]], float)
    t2 = np.array([[[0.2, 0.2, -1], [0.2, 0.2, 1], [0.6, 0.3, 1]]], float)
    assert tri_tri(t1, t2)[0] == 0.0


def test_random_pairs_match_sampling():
    rng = np.random.default_rng(0)
    T1 = rng.normal(size=(40, 3, 3))
    T2 = rng.normal(size=(40, 3, 3)) + rng.normal(scale=3, size=(40, 1, 3))
    d = tri_tri(T1, T2)
    for k in range(len(T1)):
        ref = _brute(T1[k], T2[k])
        assert d[k] <= ref + 1e-9          # exact distance never exceeds a sampled one
        assert d[k] >= ref - 0.08          # and the sampling is close to it


def test_point_and_segment_helpers():
    A = np.array([[0, 0, 0]], float)
    B = np.array([[1, 0, 0]], float)
    C = np.array([[0, 1, 0]], float)
    assert np.allclose(point_tri(np.array([[0.2, 0.2, 3]], float), A, B, C), [3])
    assert np.allclose(point_tri(np.array([[-1, 0, 0]], float), A, B, C), [1])
    assert np.allclose(seg_seg(A, B, np.array([[0, 1, 1]], float), np.array([[1, 1, 1]], float)),
                       [np.sqrt(2)])
