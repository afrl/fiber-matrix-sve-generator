import math
import numpy as np
from typing import List, Tuple, Optional, Sequence, Union
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .models.boundary import LinearBoundary, BoundaryType
from .models.fiber import Fiber, PeriodicPrimaryFiber
from .generation.placement import FiberPlacementSolver
from .meshing.gmsh_mesher import GmshMesher
from .meshing.gmsh_mesher_3d import GmshMesher3D
from .visualization import plotting

try:
    from triangle import triangulate
except ImportError:
    triangulate = None


class FiberRVE:
    """A class that defines a fiber/matrix RVE.

    This facade coordinates the placement of fibers, boundary definitions,
    and meshing/visualization.
    """

    def __init__(self):
        self.fibers: List[PeriodicPrimaryFiber] = []
        self.boundaries: List[LinearBoundary] = []
        self.fiber_vf: float = 0.0
        self.radii: List[float] = []
        self.rve_dims: List[float] = []

    # ---------------------
    # Initialization
    # ---------------------

    @staticmethod
    def _sample_radii(
        num_fibers: int,
        avg_diam: float,
        diam_std_dev: float,
        radii: Optional[Sequence[float]] = None,
    ) -> List[float]:
        if radii is not None:
            if len(radii) < 1:
                raise ValueError("radii must contain at least one fiber radius.")
            parsed = [float(r) for r in radii]
            if any(r <= 0.0 for r in parsed):
                raise ValueError("All fiber radii must be strictly positive.")
            return parsed

        if num_fibers < 1:
            raise ValueError("num_fibers must be at least 1.")
        if avg_diam <= 0.0:
            raise ValueError("avg_diam must be strictly positive.")
        if diam_std_dev < 0.0:
            raise ValueError("diam_std_dev must be non-negative.")

        min_diam = 0.05 * avg_diam
        diams = diam_std_dev * avg_diam * np.random.randn(num_fibers) + avg_diam
        if diam_std_dev > 0.0:
            for idx in range(num_fibers):
                attempts = 0
                while diams[idx] < min_diam and attempts < 1000:
                    diams[idx] = (
                        diam_std_dev * avg_diam * float(np.random.randn()) + avg_diam
                    )
                    attempts += 1
                if diams[idx] < min_diam:
                    diams[idx] = min_diam
        return [float(d) / 2.0 for d in diams]

    @staticmethod
    def _resolve_boundary_types(
        boundary_type: Union[BoundaryType, Sequence[BoundaryType]],
        num_edges: int,
    ) -> List[BoundaryType]:
        if isinstance(boundary_type, BoundaryType):
            return [boundary_type for _ in range(num_edges)]
        btypes = list(boundary_type)
        if len(btypes) != num_edges:
            raise ValueError(
                f"Expected {num_edges} boundary types, got {len(btypes)}."
            )
        if not all(isinstance(bt, BoundaryType) for bt in btypes):
            raise ValueError("All boundary_type entries must be BoundaryType values.")
        return btypes

    def initialize_rectangle_rve(
        self,
        num_fibers: int,
        vf: float = 0.6,
        avg_diam: float = 5e-6,
        diam_std_dev: float = 0.0,
        rve_aspect_ratio: float = 1.0,
        fixed_height: Optional[float] = None,
        boundary_type: Union[BoundaryType, Sequence[BoundaryType]] = BoundaryType.PERIODIC,
        radii: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Calculates dimensions of the RVE and generates fiber radii distribution.

        Parameters
        ----------
        num_fibers : int
            Target number of fibers (ignored if ``radii`` is explicitly provided).
        vf : float, optional
            Target fiber volume fraction in (0, 1). Default 0.6.
        avg_diam : float, optional
            Average fiber diameter. Default 5e-6.
        diam_std_dev : float, optional
            Standard deviation of fiber diameter relative to average (scale factor). Default 0.0.
        rve_aspect_ratio : float, optional
            Ratio of width to height (w/h). Default 1.0.
        fixed_height : float, optional
            If provided, fixes RVE height and calculates width to satisfy VF.
        boundary_type : BoundaryType or Sequence[BoundaryType], optional
            Single boundary type applied to all 4 edges, or a 4-element sequence
            for [bottom, right, top, left]. Default PERIODIC.
        radii : Sequence[float], optional
            Explicit fiber radii (e.g. for bimodal or custom size distributions).

        Returns
        -------
        List[float]
            The calculated RVE dimensions [width, height].
        """
        if not (0.0 < vf < 1.0):
            raise ValueError("vf must be strictly between 0 and 1.")
        if fixed_height is None:
            if rve_aspect_ratio <= 0.0:
                raise ValueError("rve_aspect_ratio must be strictly positive.")
        elif fixed_height <= 0.0:
            raise ValueError("fixed_height must be strictly positive.")

        self.fibers = []
        self.fiber_vf = 0.0
        self.radii = self._sample_radii(num_fibers, avg_diam, diam_std_dev, radii=radii)
        total_fiber_area = sum([math.pi * a * a for a in self.radii])
        rve_area = total_fiber_area / vf

        if fixed_height is None:
            h = math.sqrt(rve_area / rve_aspect_ratio)
        else:
            h = fixed_height
        w = rve_area / h
        self.rve_dims = [w, h]

        boundary_points = [
            np.array([0.0, 0.0]),
            np.array([w, 0.0]),
            np.array([w, h]),
            np.array([0.0, h]),
        ]
        boundary_types = self._resolve_boundary_types(boundary_type, len(boundary_points))
        self._create_boundaries_from_points(boundary_points, boundary_types)

        return self.rve_dims

    def initialize_hexagonal_rve(
        self,
        num_fibers: int,
        vf: float = 0.6,
        avg_diam: float = 5e-6,
        diam_std_dev: float = 0.0,
        boundary_type: Union[BoundaryType, Sequence[BoundaryType]] = BoundaryType.PERIODIC,
        radii: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Calculates dimensions and initializes a regular hexagonal RVE.

        Parameters
        ----------
        num_fibers : int
            Target number of fibers (ignored if ``radii`` is explicitly provided).
        vf : float, optional
            Target fiber volume fraction in (0, 1). Default 0.6.
        avg_diam : float, optional
            Average fiber diameter. Default 5e-6.
        diam_std_dev : float, optional
            Standard deviation of fiber diameter. Default 0.0.
        boundary_type : BoundaryType or Sequence[BoundaryType], optional
            Single boundary type applied to all 6 edges, or a 6-element sequence.
            Default PERIODIC.
        radii : Sequence[float], optional
            Explicit fiber radii (e.g. for bimodal or custom size distributions).

        Returns
        -------
        List[float]
            The side length of the hexagon [side_length].
        """
        if not (0.0 < vf < 1.0):
            raise ValueError("vf must be strictly between 0 and 1.")

        self.fibers = []
        self.fiber_vf = 0.0
        self.radii = self._sample_radii(num_fibers, avg_diam, diam_std_dev, radii=radii)
        total_fiber_area = sum([math.pi * a * a for a in self.radii])
        rve_area = total_fiber_area / vf

        # Area of regular hexagon = (3 * sqrt(3) / 2) * s^2
        s = math.sqrt(2.0 * rve_area / (3.0 * math.sqrt(3.0)))
        self.rve_dims = [s]

        # Vertices for a "flat-topped" hexagon centered at [0, 0]
        h_val = s * math.sqrt(3.0) / 2.0
        boundary_points = [
            np.array([s, 0.0]),
            np.array([s / 2.0, h_val]),
            np.array([-s / 2.0, h_val]),
            np.array([-s, 0.0]),
            np.array([-s / 2.0, -h_val]),
            np.array([s / 2.0, -h_val]),
        ]
        boundary_types = self._resolve_boundary_types(boundary_type, len(boundary_points))
        self._create_boundaries_from_points(boundary_points, boundary_types)

        return self.rve_dims

    def initialize_polygon_rve(
        self,
        vertices: Sequence[Sequence[float]],
        num_fibers: int = 1,
        vf: Optional[float] = 0.5,
        avg_diam: float = 5e-6,
        diam_std_dev: float = 0.0,
        boundary_type: Union[BoundaryType, Sequence[BoundaryType]] = BoundaryType.FINITE,
        radii: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """Initializes an RVE from an arbitrary simple polygon.

        Vertices are automatically oriented counter-clockwise so inward normals
        point into the domain. If ``vf`` is provided, the polygon is uniformly
        scaled about its centroid so the fiber area fraction equals ``vf``.
        If ``vf`` is ``None``, the supplied vertex coordinates are used directly.
        """
        pts = [np.asarray(v, dtype=float).copy() for v in vertices]
        if len(pts) < 3:
            raise ValueError("Polygon RVE requires at least 3 vertices.")
        if any(p.shape != (2,) for p in pts):
            raise ValueError("Each polygon vertex must be a 2D coordinate [x, y].")

        n_pts = len(pts)
        signed_area = 0.5 * sum(
            pts[i][0] * pts[(i + 1) % n_pts][1] - pts[(i + 1) % n_pts][0] * pts[i][1]
            for i in range(n_pts)
        )
        if abs(signed_area) <= 1e-18:
            raise ValueError("Polygon vertices have zero area.")

        btypes = self._resolve_boundary_types(boundary_type, n_pts)
        if signed_area < 0:
            # Reverse vertex order to CCW while keeping edge-to-boundary-type correspondence
            pts = [pts[0]] + list(reversed(pts[1:]))
            btypes = list(reversed(btypes))
            signed_area = -signed_area

        self.fibers = []
        self.fiber_vf = 0.0
        self.radii = self._sample_radii(num_fibers, avg_diam, diam_std_dev, radii=radii)
        total_fiber_area = sum(math.pi * r * r for r in self.radii)

        if vf is not None:
            if not (0.0 < vf < 1.0):
                raise ValueError("vf must be strictly between 0 and 1.")
            target_area = total_fiber_area / vf
            scale = math.sqrt(target_area / signed_area)
            centroid = sum(pts) / float(n_pts)
            pts = [centroid + scale * (p - centroid) for p in pts]

        arr_pts = np.array(pts)
        extents = np.max(arr_pts, axis=0) - np.min(arr_pts, axis=0)
        self.rve_dims = [float(extents[0]), float(extents[1])]
        self._create_boundaries_from_points(pts, btypes)
        return self.rve_dims

    def _create_boundaries_from_points(
        self, boundary_points: List[np.ndarray], boundary_types: List[BoundaryType]
    ):
        self.boundaries = []
        num_points = len(boundary_points)
        for i in range(num_points):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % num_points]
            # Indices logic from original code is kept for completeness but simplified
            b = LinearBoundary(
                points=[p1, p2],
                point_indices=[i, (i + 1) % num_points],
                btype=boundary_types[i],
            )
            b.index = i
            self.boundaries.append(b)

        self._assign_boundary_pairs()

    def _assign_boundary_pairs(self):
        """Identifies and assigns periodic pairs for boundaries within the RVE.

        Raises
        ------
        ValueError
            If a periodic boundary is paired with a non-periodic boundary.
        """
        for boundary in self.boundaries:
            for boundary_to_check in self.boundaries:
                if boundary.is_pair(boundary_to_check):
                    boundary.pair = boundary_to_check
                    boundary_to_check.pair = boundary
                    if (
                        boundary.type == BoundaryType.PERIODIC
                        or boundary.pair.type == BoundaryType.PERIODIC
                    ):
                        if boundary.type != boundary.pair.type:
                            raise ValueError(
                                "One boundary of a pair was periodic but the other was not."
                            )

    # ---------------------
    # Placement
    # ---------------------

    @property
    def boundary_points(self) -> List[np.ndarray]:
        """Returns the vertices of the RVE boundary in order."""
        return [b.points[0] for b in self.boundaries]

    def place_initial_fibers(
        self,
        specified_fiber_centers: List[List[float]] = None,
        plot_triangulation: bool = False,
    ) -> Tuple[float, float]:
        """Places fibers within the RVE domain, using triangulation for random placement.

        Parameters
        ----------
        specified_fiber_centers : List[List[float]], optional
            List of [x, y] coordinates to force fiber locations.
        plot_triangulation : bool, optional
            If True and matplotlib is available, plots the RVE triangulation.

        Returns
        -------
        Tuple[float, float]
            Actual Fiber Volume Fraction (VF) and Total RVE Area.
        """
        centers = (
            list(specified_fiber_centers)
            if specified_fiber_centers is not None
            else []
        )

        points = np.array(self.boundary_points)
        if len(points) < 3:
            raise ValueError("RVE must have at least 3 boundary points to triangulate.")

        # Triangulate the polygon
        tri_in = {"vertices": points}
        tri_in["segments"] = [[i, (i + 1) % len(points)] for i in range(len(points))]

        if triangulate is None:
            raise ImportError(
                "The 'triangle' library is required for initial fiber placement. "
                "Please install it with 'pip install triangle'."
            )

        self.triangulation = triangulate(tri_in, "p")

        if plot_triangulation and plt is not None:
            fig, ax = plt.subplots()
            x = points[:, 0]
            y = points[:, 1]
            for t in self.triangulation["triangles"]:
                t_idx = [t[0], t[1], t[2], t[0]]
                ax.plot(x[t_idx], y[t_idx], "k-", alpha=0.3)
            plt.show()

        def get_tri_area(a, b, c):
            return 0.5 * abs(
                float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            )

        areas = []
        total_weighted_centroid = np.array([0.0, 0.0])
        for t in self.triangulation["triangles"]:
            a, b, c = points[t[0]], points[t[1]], points[t[2]]
            tri_area = get_tri_area(a, b, c)
            tri_centroid = (a + b + c) / 3.0
            total_weighted_centroid += tri_area * tri_centroid
            areas.append(tri_area)

        rve_area = sum(areas)
        assert rve_area > 0, "RVE area is zero."
        centroid = total_weighted_centroid / rve_area

        def weighted_choice(choices, weights):
            total = sum(weights)
            r = float(np.random.uniform(0.0, total))
            upto = 0.0
            for i, w in enumerate(weights):
                if upto + w >= r:
                    return choices[i]
                upto += w
            return choices[-1]

        self.fibers = []
        num_fibers = len(self.radii)

        if num_fibers == 0:
            self.fiber_vf = 0.0
            return (0.0, rve_area)

        if not centers:
            if num_fibers == 1:
                self.fibers.append(
                    PeriodicPrimaryFiber(
                        centroid.copy(), self.radii[0], 0, self.boundaries
                    )
                )
            else:
                # Random placement within triangles
                choice_indices = list(range(len(areas)))
                randvals = np.random.rand(num_fibers, 2)
                for i in range(num_fibers):
                    tri_idx = weighted_choice(choice_indices, areas)
                    t = self.triangulation["triangles"][tri_idx]

                    # Unified random point in triangle algorithm
                    # (1 - sqrt(r1))*A + sqrt(r1)*(1 - r2)*B + sqrt(r1)*r2*C
                    r1, r2 = randvals[i]
                    sqrt_r1 = math.sqrt(r1)
                    random_point = (
                        (1 - sqrt_r1) * points[t[0]]
                        + sqrt_r1 * (1 - r2) * points[t[1]]
                        + sqrt_r1 * r2 * points[t[2]]
                    )

                    self.fibers.append(
                        PeriodicPrimaryFiber(
                            random_point, self.radii[i], i, self.boundaries
                        )
                    )
        else:
            if len(centers) != num_fibers:
                raise ValueError(
                    f"Number of specified centers ({len(centers)}) "
                    f"does not match number of fibers ({num_fibers})."
                )
            for i in range(num_fibers):
                self.fibers.append(
                    PeriodicPrimaryFiber(
                        np.array(centers[i], dtype=float),
                        self.radii[i],
                        i,
                        self.boundaries,
                    )
                )

        max_r = max((f.radius for f in self.fibers), default=0.0)
        for fiber in self.fibers:
            fiber.adjust_for_bounds(self.boundaries)
            fiber.calc_ghost_fibers(self.boundaries, max_fiber_radius=max_r)

        self.fiber_vf = sum([math.pi * r**2 for r in self.radii]) / rve_area
        return (self.fiber_vf, rve_area)

    def is_inside_rve(self, point: np.ndarray) -> bool:
        """Checks if a point lies inside (or on) the RVE boundaries."""
        if not self.boundaries:
            return False
        pt = np.asarray(point, dtype=float)
        for b in self.boundaries:
            tol = max(b.get_length() * 1e-12, 1e-15)
            if b.get_distance_to_fiber(pt) <= tol:
                return True

        inside = False
        x, y = float(pt[0]), float(pt[1])
        for b in self.boundaries:
            x0, y0 = float(b.points[0][0]), float(b.points[0][1])
            x1, y1 = float(b.points[1][0]), float(b.points[1][1])
            if (y0 > y) != (y1 > y):
                x_intersect = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                if x < x_intersect:
                    inside = not inside
        return inside

    def solve_fiber_locations(
        self,
        min_spacing_ratio: float,
        visualization_path: Optional[str] = None,
        show_final: bool = False,
        iterations_max: int = 1000,
    ):
        """Executes the solver to resolve fiber overlaps and enforce spacing.

        Parameters
        ----------
        min_spacing_ratio : float
            Minimum separation between fibers as a ratio of the average diameter.
        visualization_path : Optional[str], optional
            If provided, saves an animation of the solution process to the given path. Default None.
        show_final : bool, optional
            If True, displays the final RVE plot to screen. Default False.
        iterations_max : int, optional
            Maximum number of solver iterations. Default 1000.
        """
        if not self.fibers:
            self.place_initial_fibers()

        solver = FiberPlacementSolver()

        # Visualization setup
        captured_frames = []
        fig = None
        ax = None
        callback = None

        if visualization_path and plt is not None:
            # Create a single figure for the animation
            fig, ax = plt.subplots(figsize=(6, 6))

            def plot_callback(iteration, fibers, boundaries):
                # Update the existing plot
                self.draw(fig=fig, ax=ax, frame=None)

                # Force draw to update canvas
                fig.canvas.draw()

                # Capture the frame from buffer
                # Note: safe approach for different backends
                try:
                    from PIL import Image

                    w, h = fig.canvas.get_width_height()
                    # buffer_rgba() returns a memoryview of the RGBA buffer
                    buf = fig.canvas.buffer_rgba()
                    image = Image.frombuffer("RGBA", (w, h), buf).convert("RGB")
                    captured_frames.append(image)
                except ImportError:
                    print("PIL (Pillow) not found. Cannot save GIF.")
                except AttributeError:
                    # Fallback for older matplotlib or different backends
                    try:
                        buf = fig.canvas.tostring_rgb()
                        image = Image.frombytes("RGB", (w, h), buf)
                        captured_frames.append(image)
                    except Exception as e:
                        print(f"Error capturing frame (fallback failed): {e}")
                except Exception as e:
                    print(f"Error capturing frame: {e}")

            callback = plot_callback
        elif visualization_path:
            print("Matplotlib not installed. Visualization disabled.")
            callback = None

        try:
            iterations = solver.solve_fiber_locations(
                self.fibers,
                self.boundaries,
                min_spacing_ratio,
                iteration_callback=callback,
                iterations_max=iterations_max,
            )
        finally:
            # Save GIF if we have frames
            if captured_frames:
                try:
                    out_path = Path(visualization_path)
                    if out_path.parent and str(out_path.parent) != ".":
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                    captured_frames[0].save(
                        out_path,
                        save_all=True,
                        append_images=captured_frames[1:],
                        optimize=False,
                        duration=200,  # ms per frame
                        loop=0,
                    )
                    print(
                        f"Animation saved to '{visualization_path}' ({len(captured_frames)} frames)."
                    )
                except Exception as e:
                    print(f"Failed to save GIF: {e}")

            # Explicitly close the figure to avoid RuntimeWarning
            if fig is not None:
                plt.close(fig)

        if show_final:
            self.draw()

        return iterations

    def get_fiber_centers(self) -> List[np.ndarray]:
        """Returns the current centers of all primary fibers.

        Returns
        -------
        List[np.ndarray]
            List of fiber center coordinates.
        """
        return [f.center for f in self.fibers]

    # ---------------------
    # Visualization
    # ---------------------

    def draw(
        self, fig=None, ax=None, frame=None, label_fibers=False, label_boundaries=False
    ):
        """Draws the current state of the RVE.

        Parameters
        ----------
        fig : Figure, optional
            Matplotlib figure.
        ax : Axes, optional
            Matplotlib axes.
        frame : int, optional
            Frame number for saving animation sequences.
        label_fibers : bool, optional
            Label fibers with indices.
        label_boundaries : bool, optional
            Label boundaries with indices.
        """
        return plotting.draw_rve(
            self.fibers, self.boundaries, fig, ax, frame, label_fibers, label_boundaries
        )

    # ---------------------
    # Meshing
    # ---------------------

    def create_mesh(
        self,
        mesh_name="FiberMatrixRVE",
        mesh_size_factor=1.0,
        visualize_gui=False,
        check_periodicity=False,
    ):
        """Generates a mesh for the current RVE configuration using GMSH.

        Parameters
        ----------
        mesh_name : str, optional
            Base name for output files (.msh, .vtk). Default "FiberMatrixRVE".
        mesh_size_factor : float, optional
            Global scaling factor for mesh element size. Default 1.0.
        visualize_gui : bool, optional
            If True, opens the GMSH GUI to visualize mesh generation steps. Default False.
        check_periodicity : bool, optional
            If True, asserts that the generated mesh nodes on periodic boundaries match. Default False.
        """
        mesher = GmshMesher(mesh_name)
        mesher.create_mesh(
            self.fibers,
            self.boundaries,
            mesh_size_factor,
            visualize_gui,
            check_periodicity,
        )

    def create_3d_mesh(
        self,
        mesh_name="FiberMatrixRVE3D",
        thickness=1.0,
        mesh_size_factor=1.0,
        z_layers=4,
        visualize_gui=False,
        check_periodicity=False,
        periodic_z=False,
        surface_groups=False,
        composite_surface_groups=False,
        anchor_node_groups=False,
        uniform_mesh=True,
        fiber_mesh_size=None,
        matrix_mesh_size=None,
        boundary_mesh_size=None,
        interface_refinement_distance=None,
        boundary_refinement_distance=None,
        recombine_prisms=False,
    ):
        """Generates a 3D volume mesh for the current RVE configuration using GMSH.

        The 3D mesh is created by extruding the fragmented 2D fiber/matrix
        geometry through ``thickness``. Periodic 2D boundary pairs become
        periodic side-surface pairs in the 3D mesh.

        Parameters
        ----------
        mesh_name : str, optional
            Base name for output files (.msh, .vtk). Default "FiberMatrixRVE3D".
        thickness : float, optional
            Extrusion thickness in the positive z direction. Default 1.0.
        mesh_size_factor : float, optional
            Global scaling factor for mesh element size. Default 1.0.
        z_layers : int, optional
            Number of mesh layers through the thickness. Default 4.
        visualize_gui : bool, optional
            If True, opens the GMSH GUI to visualize mesh generation steps. Default False.
        check_periodicity : bool, optional
            If True, asserts that generated mesh nodes on periodic surfaces match. Default False.
        periodic_z : bool, optional
            If True, also makes the top and bottom surfaces periodic. Default False.
        surface_groups : bool, optional
            If True, creates material-specific physical surface groups for
            left, right, bottom, top, front, and back. Default False.
        composite_surface_groups : bool, optional
            If True, creates whole-composite physical surface groups for
            left, right, bottom, top, front, and back. Default False.
        anchor_node_groups : bool, optional
            If True, creates 0D physical groups named anchor_xyz, anchor_yz,
            and anchor_z for mechanical constraint boundary conditions.
            Default False.
        uniform_mesh : bool, optional
            If True, uses mesh_size_factor as a global mesh size. If False,
            applies separate mesh sizes for fiber, matrix, and exterior boundary
            regions. Default True.
        fiber_mesh_size : float, optional
            Target element size on fiber surfaces for non-uniform meshes.
        matrix_mesh_size : float, optional
            Target element size on matrix surfaces for non-uniform meshes.
        boundary_mesh_size : float, optional
            Target element size on exterior domain boundaries for non-uniform meshes.
        interface_refinement_distance : float, optional
            Distance away from fiber/matrix interfaces over which the mesh
            transitions from fiber_mesh_size to matrix_mesh_size. Used only
            when uniform_mesh is False.
        boundary_refinement_distance : float, optional
            Distance away from exterior domain boundaries over which the mesh
            transitions from boundary_mesh_size to matrix_mesh_size. Used only
            when uniform_mesh is False.
        recombine_prisms : bool, optional
            If True, recombines the structured extrusion into prism/wedge
            elements instead of subdividing into tetrahedra. This can reduce
            radial-looking tetrahedral subdivision patterns. Default False.
        """
        mesher = GmshMesher3D(mesh_name)
        mesher.create_mesh(
            self.fibers,
            self.boundaries,
            thickness,
            mesh_size_factor,
            z_layers,
            visualize_gui,
            check_periodicity,
            periodic_z,
            surface_groups,
            composite_surface_groups,
            anchor_node_groups,
            uniform_mesh,
            fiber_mesh_size,
            matrix_mesh_size,
            boundary_mesh_size,
            interface_refinement_distance,
            boundary_refinement_distance,
            recombine_prisms,
        )
