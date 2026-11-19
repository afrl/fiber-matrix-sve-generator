import math
import numpy as np
import pytest
from scipy.spatial import KDTree

from fiber_matrix.models.boundary import BoundaryType, LinearBoundary
from fiber_matrix.models.fiber import Fiber, PeriodicGhostFiber, PeriodicPrimaryFiber


def _make_square_boundaries(size: float, btype: BoundaryType = BoundaryType.PERIODIC):
    pts = [
        np.array([0.0, 0.0]),
        np.array([size, 0.0]),
        np.array([size, size]),
        np.array([0.0, size]),
    ]
    boundaries = []
    for i in range(4):
        b = LinearBoundary([pts[i], pts[(i + 1) % 4]], [i, (i + 1) % 4], btype)
        b.index = i
        boundaries.append(b)
    for b1 in boundaries:
        for b2 in boundaries:
            if b1.is_pair(b2):
                b1.pair = b2
                b2.pair = b1
    return boundaries


class TestLinearBoundary:
    def test_length_and_normals(self):
        b = LinearBoundary(
            [np.array([0.0, 0.0]), np.array([10.0, 0.0])], [0, 1], BoundaryType.FINITE
        )
        assert math.isclose(b.get_length(), 10.0)
        np.testing.assert_allclose(b.get_move_vector(), np.array([0.0, 1.0]))
        assert math.isclose(b.get_signed_distance(np.array([5.0, 3.0])), 3.0)
        assert math.isclose(b.get_signed_distance(np.array([5.0, 0.0])), 0.0)
        assert math.isclose(b.get_signed_distance(np.array([5.0, -2.5])), -2.5)

    @pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
    def test_is_pair_across_scales(self, scale):
        boundaries = _make_square_boundaries(scale, BoundaryType.PERIODIC)
        assert boundaries[0].pair is boundaries[2]
        assert boundaries[1].pair is boundaries[3]
        assert not boundaries[0].is_pair(boundaries[0])
        assert not boundaries[0].is_pair(boundaries[1])

    def test_is_pair_rejects_coincident_opposite_segment_and_length_mismatch(self):
        b1 = LinearBoundary(
            [np.array([0.0, 0.0]), np.array([5.0, 0.0])], [0, 1], BoundaryType.PERIODIC
        )
        b_coincident = LinearBoundary(
            [np.array([5.0, 0.0]), np.array([0.0, 0.0])], [1, 0], BoundaryType.PERIODIC
        )
        b_shorter = LinearBoundary(
            [np.array([4.9, 5.0]), np.array([0.0, 5.0])], [2, 3], BoundaryType.PERIODIC
        )
        assert not b1.is_pair(b_coincident)
        assert not b1.is_pair(b_shorter)
        assert not b1.is_pair("not_a_boundary")

    def test_point_relative_position_numpy2_and_vertex_zero_guard(self):
        b = LinearBoundary(
            [np.array([0.0, 0.0]), np.array([4.0, 0.0])], [0, 1], BoundaryType.FINITE
        )
        assert b.get_point_relative_position(np.array([2.0, 2.0])) > 0.0
        assert b.get_point_relative_position(np.array([2.0, -2.0])) < 0.0
        assert b.get_point_relative_position(np.array([2.0, 0.0])) == 0.0
        # Exact segment start vertex previously divided by zero -> NaN
        val_at_start = b.get_point_relative_position(np.array([0.0, 0.0]))
        assert not math.isnan(val_at_start)
        assert val_at_start == 0.0

    def test_get_intersection_points_cases(self):
        b = LinearBoundary(
            [np.array([0.0, 0.0]), np.array([10.0, 0.0])], [0, 1], BoundaryType.FINITE
        )

        # Non-intersecting circle (previously sqrt(neg) -> NaN RuntimeWarning)
        pts_far = b.get_intersection_points(np.array([5.0, 4.0]), 2.5)
        assert pts_far == []

        # Tangent inside segment
        pts_tan = b.get_intersection_points(np.array([5.0, 2.5]), 2.5)
        assert len(pts_tan) == 1
        np.testing.assert_allclose(pts_tan[0], np.array([5.0, 0.0]))

        # Tangent outside segment bounds
        pts_tan_out = b.get_intersection_points(np.array([15.0, 2.5]), 2.5)
        assert pts_tan_out == []

        # Secant fully within segment
        pts_sec = b.get_intersection_points(np.array([5.0, 1.5]), 2.5)
        assert len(pts_sec) == 2
        xs = sorted(float(p[0]) for p in pts_sec)
        assert math.isclose(xs[0], 3.0)
        assert math.isclose(xs[1], 7.0)

        # Secant clipped by segment endpoint (1 intersection inside [0, 10])
        pts_clip = b.get_intersection_points(np.array([0.0, 1.5]), 2.5)
        assert len(pts_clip) == 1
        np.testing.assert_allclose(pts_clip[0], np.array([2.0, 0.0]))


class TestFiberModels:
    def test_coincident_fiber_overlap_resolves_without_nan(self):
        f0 = Fiber(np.array([5.0, 5.0]), 2.5)
        f1 = Fiber(np.array([5.0, 5.0]), 2.5)
        f0.neighbors.add(f1)
        had_overlap = f0.fix_overlap_with_neighbors([], min_space_between_fibers=0.25)
        assert had_overlap is True
        assert not np.any(np.isnan(f0.center))
        assert not np.any(np.isnan(f1.center))
        assert np.linalg.norm(f1.center - f0.center) > 5.0

    def test_finite_boundary_pushes_encroaching_and_escaped_fibers_inward(self):
        boundaries = _make_square_boundaries(20.0, BoundaryType.FINITE)
        # Encroaching bottom boundary (0 < y < 1.05 * R)
        f_encroach = Fiber(np.array([10.0, 1.0]), 2.5)
        f_encroach.adjust_for_bounds(boundaries, eps=0.05)
        assert math.isclose(f_encroach.center[1], 2.5 * 1.05, rel_tol=1e-9)

        # Escaped slightly outside bottom boundary (y < 0)
        f_escaped = Fiber(np.array([10.0, -0.8]), 2.5)
        f_escaped.adjust_for_bounds(boundaries, eps=0.05)
        assert math.isclose(f_escaped.center[1], 2.5 * 1.05, rel_tol=1e-9)

    def test_symmetric_boundary_snaps_or_pushes_without_oscillation(self):
        boundaries = _make_square_boundaries(20.0, BoundaryType.SYMMETRIC)
        # Center close to bottom symmetry plane (< R/2 = 1.25) -> snaps onto y = 0
        f_on_plane = Fiber(np.array([10.0, 0.8]), 2.5)
        f_on_plane.adjust_for_bounds(boundaries, eps=0.05)
        assert abs(f_on_plane.center[1]) < 1e-12
        # Repeated call does NOT kick fiber off the symmetry line
        f_on_plane.adjust_for_bounds(boundaries, eps=0.05)
        assert abs(f_on_plane.center[1]) < 1e-12

        # Center between R/2 and (1 + eps)*R -> pushed inward to (1 + eps)*R
        f_near = Fiber(np.array([10.0, 1.8]), 2.5)
        f_near.adjust_for_bounds(boundaries, eps=0.05)
        assert math.isclose(f_near.center[1], 2.5 * 1.05, rel_tol=1e-9)

    def test_corner_ghost_fibers_and_drifted_primary_wrapping(self):
        boundaries = _make_square_boundaries(20.0, BoundaryType.PERIODIC)
        # Fiber at (3.5, 3.5) with R=2.5 does NOT intersect x=0 or y=0,
        # yet lies within 3*R=7.5 of the corner and needs all 3 periodic ghosts (+x, +y, +x+y)
        pf = PeriodicPrimaryFiber(np.array([3.5, 3.5]), 2.5, 0, boundaries)
        ghost_coords = {tuple(np.round(g.center, 6)) for g in pf.ghost_fibers}
        assert (23.5, 3.5) in ghost_coords
        assert (3.5, 23.5) in ghost_coords
        assert (23.5, 23.5) in ghost_coords

        # Drifted primary fiber far outside bottom boundary (y = -3.0 < -R = -2.5) wraps back
        pf_drift = PeriodicPrimaryFiber(np.array([1.0, -3.0]), 2.5, 1, boundaries)
        pf_drift.adjust_for_bounds(boundaries)
        assert 0.0 <= pf_drift.center[0] <= 20.0
        assert 0.0 <= pf_drift.center[1] <= 20.0
        assert math.isclose(pf_drift.center[1], 17.0)

    def test_polydisperse_neighbor_search_finds_large_neighbor(self):
        boundaries = _make_square_boundaries(40.0, BoundaryType.FINITE)
        small = PeriodicPrimaryFiber(np.array([10.0, 10.0]), 0.5, 0, boundaries)
        # Distance = 5.0 > 8 * small.radius (4.0), yet overlapping because large.radius = 5.0!
        large = PeriodicPrimaryFiber(np.array([15.0, 10.0]), 5.0, 1, boundaries)
        fibers_with_ghosts = [small, large]
        tree = KDTree([f.center for f in fibers_with_ghosts])
        small.update_neighbors(
            tree,
            fibers_with_ghosts,
            fiber_diams_to_search=4.0,
            max_fiber_radius=5.0,
        )
        assert large in small.neighbors
