import math
import gmsh
import numpy as np
import pytest

from fiber_matrix import BoundaryType, FiberRVE
import fiber_matrix.meshing.gmsh_mesher as gmsh_mesher_mod
import fiber_matrix.meshing.gmsh_mesher_3d as gmsh_mesher_3d_mod
from fiber_matrix.meshing.gmsh_mesher import GmshMesher
from fiber_matrix.meshing.gmsh_mesher_3d import GmshMesher3D
from fiber_matrix.models.boundary import LinearBoundary


def _compute_2d_mesh_group_areas(msh_path: str):
    gmsh.initialize()
    try:
        gmsh.open(msh_path)
        areas = {}
        for dim, tag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, tag)
            total = 0.0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                _, _, node_tags_list = gmsh.model.mesh.getElements(2, ent)
                for nt in node_tags_list:
                    tris = nt.reshape(-1, 3)
                    for tri in tris:
                        c0 = gmsh.model.mesh.getNode(int(tri[0]))[0]
                        c1 = gmsh.model.mesh.getNode(int(tri[1]))[0]
                        c2 = gmsh.model.mesh.getNode(int(tri[2]))[0]
                        total += 0.5 * abs(
                            (c1[0] - c0[0]) * (c2[1] - c0[1])
                            - (c1[1] - c0[1]) * (c2[0] - c0[0])
                        )
            areas[name] = total
        return areas
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _get_physical_group_names(msh_path: str):
    gmsh.initialize()
    try:
        gmsh.open(msh_path)
        return {
            (dim, gmsh.model.getPhysicalName(dim, tag))
            for dim, tag in gmsh.model.getPhysicalGroups()
        }
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


class TestMeshing2D:
    def test_periodic_square_2d_unit_and_micron_scales(self, tmp_path):
        for scale, mesh_h in [(5.0, 0.8), (5e-6, 0.8e-6)]:
            np.random.seed(10)
            rve = FiberRVE()
            rve.initialize_rectangle_rve(num_fibers=12, vf=0.50, avg_diam=scale)
            rve.solve_fiber_locations(min_spacing_ratio=0.08)
            prefix = str(tmp_path / f"sq_{scale:.0e}")
            rve.create_mesh(
                mesh_name=prefix, mesh_size_factor=mesh_h, check_periodicity=True
            )
            msh_file = prefix + ".msh"
            areas = _compute_2d_mesh_group_areas(msh_file)
            assert "Matrix" in areas and "Fibers" in areas
            mesh_vf = areas["Fibers"] / (areas["Fibers"] + areas["Matrix"])
            assert abs(mesh_vf - 0.50) < 0.02

    def test_periodic_hexagonal_2d_mesh(self, tmp_path):
        np.random.seed(14)
        rve = FiberRVE()
        rve.initialize_hexagonal_rve(num_fibers=14, vf=0.48, avg_diam=5.0)
        rve.solve_fiber_locations(min_spacing_ratio=0.08)
        prefix = str(tmp_path / "hex_periodic")
        rve.create_mesh(mesh_name=prefix, mesh_size_factor=0.8, check_periodicity=True)
        areas = _compute_2d_mesh_group_areas(prefix + ".msh")
        mesh_vf = areas["Fibers"] / (areas["Fibers"] + areas["Matrix"])
        assert abs(mesh_vf - 0.48) < 0.02

    def test_drifted_corner_fiber_periodic_2d_mesh(self, tmp_path):
        rve = FiberRVE()
        rve.initialize_rectangle_rve(num_fibers=1, vf=0.20, avg_diam=5.0)
        # Deliberately place fiber center outside bottom periodic edge near left edge
        rve.place_initial_fibers(specified_fiber_centers=[[1.0, -3.0]])
        prefix = str(tmp_path / "drifted_corner")
        rve.create_mesh(mesh_name=prefix, mesh_size_factor=0.7, check_periodicity=True)
        areas = _compute_2d_mesh_group_areas(prefix + ".msh")
        mesh_vf = areas["Fibers"] / (areas["Fibers"] + areas["Matrix"])
        assert abs(mesh_vf - 0.20) < 0.02

    def test_mixed_boundary_and_empty_fiber_2d_meshes(self, tmp_path):
        np.random.seed(9)
        rve = FiberRVE()
        rve.initialize_rectangle_rve(
            num_fibers=10,
            vf=0.42,
            avg_diam=5.0,
            boundary_type=[
                BoundaryType.FINITE,
                BoundaryType.PERIODIC,
                BoundaryType.FINITE,
                BoundaryType.PERIODIC,
            ],
        )
        rve.solve_fiber_locations(min_spacing_ratio=0.08)
        prefix = str(tmp_path / "mixed_ply")
        rve.create_mesh(mesh_name=prefix, mesh_size_factor=0.9, check_periodicity=True)
        assert (tmp_path / "mixed_ply.msh").exists()

        # Pure matrix (0 fibers) mesh
        mesher = GmshMesher(str(tmp_path / "pure_matrix"))
        mesher.create_mesh([], rve.boundaries, mesh_size_factor=2.0)
        groups = _get_physical_group_names(str(tmp_path / "pure_matrix.msh"))
        assert (2, "Matrix") in groups

    def test_gmsh_finalize_on_exception_and_headless_fallback(self, tmp_path, monkeypatch):
        # Disconnected boundaries trigger RuntimeError inside _build_and_write_mesh
        bad_boundaries = [
            LinearBoundary(
                [np.array([0.0, 0.0]), np.array([5.0, 0.0])], [0, 1], BoundaryType.FINITE
            ),
            LinearBoundary(
                [np.array([10.0, 10.0]), np.array([15.0, 10.0])],
                [2, 3],
                BoundaryType.FINITE,
            ),
        ]
        mesher = GmshMesher(str(tmp_path / "bad"))
        with pytest.raises(RuntimeError, match="connected boundary"):
            mesher.create_mesh([], bad_boundaries)
        assert not gmsh.isInitialized()

        # Simulated missing shared library (gmsh is None) raises clean ImportError
        monkeypatch.setattr(gmsh_mesher_mod, "gmsh", None)
        monkeypatch.setattr(
            gmsh_mesher_mod, "_gmsh_import_error", OSError("libGLU.so.1 missing")
        )
        with pytest.raises(ImportError, match="libGLU.so.1"):
            GmshMesher("no_gmsh")

        monkeypatch.setattr(gmsh_mesher_3d_mod, "gmsh", None)
        monkeypatch.setattr(
            gmsh_mesher_3d_mod, "_gmsh_import_error", OSError("libGLU.so.1 missing")
        )
        with pytest.raises(ImportError, match="libGLU.so.1"):
            GmshMesher3D("no_gmsh_3d")


class TestMeshing3D:
    def test_uniform_and_nonuniform_3d_square_mesh_and_groups(self, tmp_path, capsys):
        np.random.seed(6)
        rve = FiberRVE()
        rve.initialize_rectangle_rve(num_fibers=8, vf=0.45, avg_diam=5.0)
        rve.solve_fiber_locations(min_spacing_ratio=0.1)

        # Uniform mesh with check_periodicity=False should NOT print stray "None"
        capsys.readouterr()
        u_prefix = str(tmp_path / "u3d")
        rve.create_3d_mesh(
            mesh_name=u_prefix,
            thickness=4.0,
            mesh_size_factor=1.5,
            z_layers=3,
            check_periodicity=False,
            periodic_z=True,
            uniform_mesh=True,
            recombine_prisms=False,
        )
        captured = capsys.readouterr().out
        assert "None" not in captured.splitlines()

        # Non-uniform mesh with all physical groups & periodicity verification
        nu_prefix = str(tmp_path / "nu3d")
        rve.create_3d_mesh(
            mesh_name=nu_prefix,
            thickness=5.0,
            mesh_size_factor=1.5,
            z_layers=3,
            check_periodicity=True,
            periodic_z=True,
            surface_groups=True,
            composite_surface_groups=True,
            anchor_node_groups=True,
            uniform_mesh=False,
            fiber_mesh_size=0.6,
            matrix_mesh_size=1.8,
            boundary_mesh_size=0.8,
            interface_refinement_distance=0.6,
            boundary_refinement_distance=0.6,
            recombine_prisms=True,
        )
        groups = _get_physical_group_names(nu_prefix + ".msh")
        assert (3, "Matrix") in groups
        assert (3, "Fibers") in groups
        assert (0, "anchor_xyz") in groups
        assert (0, "anchor_yz") in groups
        assert (0, "anchor_z") in groups
        assert (2, "composite_left") in groups
        assert (2, "Matrix_Bottom") in groups

    def test_micron_scale_and_hexagonal_3d_periodic_meshes(self, tmp_path):
        # Micron-scale 3D RVE (verifies removal of 1e-9 tolerance floors)
        np.random.seed(15)
        rve_micro = FiberRVE()
        rve_micro.initialize_rectangle_rve(num_fibers=6, vf=0.40, avg_diam=5e-6)
        rve_micro.solve_fiber_locations(min_spacing_ratio=0.1)
        micro_prefix = str(tmp_path / "micro3d")
        rve_micro.create_3d_mesh(
            mesh_name=micro_prefix,
            thickness=5e-6,
            mesh_size_factor=1.5e-6,
            z_layers=2,
            check_periodicity=True,
            anchor_node_groups=True,
            uniform_mesh=True,
        )
        assert (tmp_path / "micro3d.msh").exists()

        # Hexagonal 3D periodic RVE with non-uniform refinement + anchor nodes
        np.random.seed(19)
        rve_hex = FiberRVE()
        rve_hex.initialize_hexagonal_rve(num_fibers=8, vf=0.42, avg_diam=5.0)
        rve_hex.solve_fiber_locations(min_spacing_ratio=0.1)
        hex_prefix = str(tmp_path / "hex3d")
        rve_hex.create_3d_mesh(
            mesh_name=hex_prefix,
            thickness=4.0,
            mesh_size_factor=1.5,
            z_layers=3,
            check_periodicity=True,
            anchor_node_groups=True,
            uniform_mesh=False,
            fiber_mesh_size=0.6,
            matrix_mesh_size=1.8,
            boundary_mesh_size=0.8,
            interface_refinement_distance=0.6,
            boundary_refinement_distance=0.6,
            recombine_prisms=True,
        )
        groups = _get_physical_group_names(hex_prefix + ".msh")
        assert (0, "anchor_xyz") in groups
        assert (0, "anchor_yz") in groups
        assert (0, "anchor_z") in groups

    def test_invalid_3d_parameters(self):
        rve = FiberRVE()
        rve.initialize_rectangle_rve(num_fibers=2, vf=0.3, avg_diam=5.0)
        rve.place_initial_fibers()
        with pytest.raises(ValueError, match="thickness"):
            rve.create_3d_mesh(thickness=0.0)
        with pytest.raises(ValueError, match="z_layers"):
            rve.create_3d_mesh(thickness=1.0, z_layers=0)
