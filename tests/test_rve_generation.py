import math
from pathlib import Path
import numpy as np
import pytest

from fiber_matrix import BoundaryType, FiberRVE


def _assert_valid_packing(rve: FiberRVE, min_spacing_ratio: float, tol: float = 1e-6):
    avg_r = sum(f.radius for f in rve.fibers) / len(rve.fibers)
    min_gap = 2.0 * avg_r * min_spacing_ratio
    max_r = max(f.radius for f in rve.fibers)

    all_instances = []
    for f in rve.fibers:
        f.calc_ghost_fibers(rve.boundaries, max_fiber_radius=max_r)
        all_instances.append((f.index, f.center, f.radius))
        for g in f.ghost_fibers:
            all_instances.append((f.index, g.center, g.radius))

    for i in range(len(all_instances)):
        for j in range(i + 1, len(all_instances)):
            idx_i, ci, ri = all_instances[i]
            idx_j, cj, rj = all_instances[j]
            if idx_i == idx_j:
                continue
            dist = float(np.linalg.norm(ci - cj))
            required = ri + rj + min_gap
            assert dist >= required - tol, (
                f"Fibers {idx_i} and {idx_j} violate min spacing: "
                f"dist={dist:.6e}, required={required:.6e}"
            )


class TestRVEInitializationAndValidation:
    def test_rectangle_dimensions_aspect_ratio_and_fixed_height(self):
        rve = FiberRVE()
        w, h = rve.initialize_rectangle_rve(
            num_fibers=16, vf=0.5, avg_diam=4.0, rve_aspect_ratio=2.0
        )
        expected_area = 16 * math.pi * (2.0**2) / 0.5
        assert math.isclose(w * h, expected_area, rel_tol=1e-12)
        assert math.isclose(w / h, 2.0, rel_tol=1e-12)

        w2, h2 = rve.initialize_rectangle_rve(
            num_fibers=16, vf=0.5, avg_diam=4.0, fixed_height=20.0
        )
        assert math.isclose(h2, 20.0)
        assert math.isclose(w2 * h2, expected_area, rel_tol=1e-12)

    def test_hexagonal_dimensions_and_pairs(self):
        rve = FiberRVE()
        (s,) = rve.initialize_hexagonal_rve(num_fibers=12, vf=0.45, avg_diam=5.0)
        hex_area = 1.5 * math.sqrt(3.0) * s * s
        expected_area = 12 * math.pi * (2.5**2) / 0.45
        assert math.isclose(hex_area, expected_area, rel_tol=1e-12)
        assert len(rve.boundaries) == 6
        for i in range(3):
            assert rve.boundaries[i].pair is rve.boundaries[i + 3]

    def test_polygon_rve_ccw_and_cw_and_nonconvex(self):
        # Clockwise triangle -> auto-reoriented to CCW and scaled to target VF
        cw_tri = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        rve = FiberRVE()
        rve.initialize_polygon_rve(cw_tri, num_fibers=3, vf=0.25, avg_diam=2.0)
        vf, area = rve.place_initial_fibers()
        assert math.isclose(vf, 0.25, rel_tol=1e-9)
        assert math.isclose(area, 3 * math.pi * 1.0 / 0.25, rel_tol=1e-9)

        # Non-convex L-shaped polygon with vf=None (exact coordinates)
        l_poly = [
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 10.0],
            [10.0, 10.0],
            [10.0, 20.0],
            [0.0, 20.0],
        ]
        rve_l = FiberRVE()
        rve_l.initialize_polygon_rve(l_poly, num_fibers=4, vf=None, avg_diam=2.0)
        assert rve_l.is_inside_rve(np.array([5.0, 15.0])) is True
        assert rve_l.is_inside_rve(np.array([15.0, 15.0])) is False

    def test_mixed_boundary_types_and_mismatched_periodic_error(self):
        rve = FiberRVE()
        # Bottom/Top FINITE, Right/Left PERIODIC
        btypes = [
            BoundaryType.FINITE,
            BoundaryType.PERIODIC,
            BoundaryType.FINITE,
            BoundaryType.PERIODIC,
        ]
        rve.initialize_rectangle_rve(num_fibers=8, vf=0.4, avg_diam=5.0, boundary_type=btypes)
        assert rve.boundaries[1].type == BoundaryType.PERIODIC
        assert rve.boundaries[3].type == BoundaryType.PERIODIC
        assert rve.boundaries[0].type == BoundaryType.FINITE

        # Mismatched pair (bottom PERIODIC, top FINITE) must raise ValueError
        bad_btypes = [
            BoundaryType.PERIODIC,
            BoundaryType.FINITE,
            BoundaryType.FINITE,
            BoundaryType.FINITE,
        ]
        with pytest.raises(ValueError, match="periodic"):
            rve.initialize_rectangle_rve(
                num_fibers=4, vf=0.4, avg_diam=5.0, boundary_type=bad_btypes
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_fibers": 0, "vf": 0.5, "avg_diam": 5.0},
            {"num_fibers": 5, "vf": 0.0, "avg_diam": 5.0},
            {"num_fibers": 5, "vf": 1.0, "avg_diam": 5.0},
            {"num_fibers": 5, "vf": 0.5, "avg_diam": -1.0},
            {"num_fibers": 5, "vf": 0.5, "avg_diam": 5.0, "diam_std_dev": -0.1},
            {"num_fibers": 5, "vf": 0.5, "avg_diam": 5.0, "rve_aspect_ratio": 0.0},
            {"num_fibers": 5, "vf": 0.5, "avg_diam": 5.0, "fixed_height": -2.0},
            {"num_fibers": 2, "vf": 0.5, "radii": [1.0, -0.5]},
            {
                "num_fibers": 2,
                "vf": 0.5,
                "boundary_type": [BoundaryType.PERIODIC, BoundaryType.PERIODIC],
            },
        ],
    )
    def test_invalid_inputs_raise_value_error(self, kwargs):
        rve = FiberRVE()
        with pytest.raises(ValueError):
            rve.initialize_rectangle_rve(**kwargs)

    def test_reinitialize_clears_existing_fibers_and_positive_radii(self):
        np.random.seed(7)
        rve = FiberRVE()
        rve.initialize_rectangle_rve(
            num_fibers=50, vf=0.4, avg_diam=5.0, diam_std_dev=0.9
        )
        assert all(r >= 0.025 * 5.0 - 1e-12 for r in rve.radii)
        rve.place_initial_fibers()
        assert len(rve.fibers) == 50

        rve.initialize_rectangle_rve(num_fibers=5, vf=0.4, avg_diam=5.0)
        assert rve.fibers == []


class TestFiberPlacementAndSolving:
    def test_single_fiber_centroid_placement_without_mutating_input(self):
        rve = FiberRVE()
        w, h = rve.initialize_rectangle_rve(num_fibers=1, vf=0.3, avg_diam=5.0)
        empty_list = []
        vf, _ = rve.place_initial_fibers(specified_fiber_centers=empty_list)
        assert empty_list == []
        assert len(rve.fibers) == 1
        np.testing.assert_allclose(rve.fibers[0].center, np.array([w / 2.0, h / 2.0]))
        assert math.isclose(vf, 0.3, rel_tol=1e-9)

        with pytest.raises(ValueError, match="does not match"):
            rve.place_initial_fibers(specified_fiber_centers=[[1.0, 1.0], [2.0, 2.0]])

    def test_numpy_seed_reproducibility(self):
        centers = []
        for _ in range(2):
            np.random.seed(12345)
            rve = FiberRVE()
            rve.initialize_hexagonal_rve(
                num_fibers=18, vf=0.50, avg_diam=5.0, diam_std_dev=0.1
            )
            rve.place_initial_fibers()
            rve.solve_fiber_locations(min_spacing_ratio=0.08)
            centers.append(np.array(rve.get_fiber_centers()))
        np.testing.assert_allclose(centers[0], centers[1], rtol=0, atol=1e-14)

    def test_solve_without_visualization_and_dense_packing(self):
        np.random.seed(42)
        rve = FiberRVE()
        rve.initialize_rectangle_rve(num_fibers=30, vf=0.58, avg_diam=5.0)
        iters = rve.solve_fiber_locations(min_spacing_ratio=0.08, visualization_path=None)
        assert iters >= 1
        _assert_valid_packing(rve, min_spacing_ratio=0.08)

    def test_solve_bimodal_and_mixed_and_symmetric_packings(self):
        np.random.seed(11)
        radii = [4.0] * 6 + [1.2] * 24
        rve_bi = FiberRVE()
        rve_bi.initialize_rectangle_rve(num_fibers=len(radii), vf=0.52, radii=radii)
        rve_bi.solve_fiber_locations(min_spacing_ratio=0.05)
        _assert_valid_packing(rve_bi, min_spacing_ratio=0.05)

        # Mixed ply boundaries (FINITE top/bottom, PERIODIC left/right)
        np.random.seed(22)
        rve_ply = FiberRVE()
        rve_ply.initialize_rectangle_rve(
            num_fibers=18,
            vf=0.45,
            avg_diam=5.0,
            rve_aspect_ratio=1.8,
            boundary_type=[
                BoundaryType.FINITE,
                BoundaryType.PERIODIC,
                BoundaryType.FINITE,
                BoundaryType.PERIODIC,
            ],
        )
        rve_ply.solve_fiber_locations(min_spacing_ratio=0.08)
        _assert_valid_packing(rve_ply, min_spacing_ratio=0.08)
        for f in rve_ply.fibers:
            assert rve_ply.boundaries[0].get_signed_distance(f.center) >= f.radius
            assert rve_ply.boundaries[2].get_signed_distance(f.center) >= f.radius

        # Symmetric RVE
        np.random.seed(33)
        rve_sym = FiberRVE()
        rve_sym.initialize_rectangle_rve(
            num_fibers=12,
            vf=0.35,
            avg_diam=5.0,
            boundary_type=BoundaryType.SYMMETRIC,
        )
        rve_sym.solve_fiber_locations(min_spacing_ratio=0.08)
        _assert_valid_packing(rve_sym, min_spacing_ratio=0.08)
        for f in rve_sym.fibers:
            for b in rve_sym.boundaries:
                d = b.get_signed_distance(f.center)
                assert abs(d) < 1e-9 or d >= f.radius - 1e-6

    def test_gif_animation_export(self, tmp_path):
        np.random.seed(5)
        rve = FiberRVE()
        rve.initialize_rectangle_rve(num_fibers=8, vf=0.45, avg_diam=5.0)
        gif_path = tmp_path / "nested" / "solve.gif"
        rve.solve_fiber_locations(min_spacing_ratio=0.1, visualization_path=str(gif_path))
        assert gif_path.exists()
        assert gif_path.stat().st_size > 0

    def test_reference_verification_regression(self):
        tests_dir = Path(__file__).resolve().parent
        init_centers = np.load(tests_dir / "test_initial_fiber_centers.npy")
        final_centers = np.load(tests_dir / "test_final_fiber_centers.npy")

        rve = FiberRVE()
        rve.initialize_rectangle_rve(
            num_fibers=20,
            vf=0.6,
            avg_diam=5e-6,
            diam_std_dev=0.0,
            rve_aspect_ratio=1.0,
        )
        rve.place_initial_fibers(specified_fiber_centers=init_centers.tolist())
        rve.solve_fiber_locations(min_spacing_ratio=0.1)
        norm_diff = np.linalg.norm(np.array(rve.get_fiber_centers()) - final_centers)
        assert norm_diff < 1e-4
