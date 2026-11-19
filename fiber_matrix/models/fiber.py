import copy
import math
import numpy as np
from typing import List, Set, Optional, Any
from .boundary import LinearBoundary, BoundaryType, BoundaryNode


class Fiber:
    """The basic fiber class representing a circular inclusion.

    Parameters
    ----------
    center : np.ndarray
        The coordinates of the fiber center.
    radius : float
        The radius of the fiber.
    """

    def __init__(self, center: np.ndarray, radius: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.boundary_nodes: List[BoundaryNode] = []
        self.coord_inside_rve: Optional[List[float]] = None
        self.neighbors: Set["Fiber"] = set()
        self.vec_moved_since_neighbor_update = np.array([0.0, 0.0])

    def __str__(self):
        return f"Fiber Center: {self.center}\tr={self.radius}"

    def get_vector_to_other(self, other_fiber: "Fiber") -> np.ndarray:
        """Calculates the vector pointing from this fiber to another fiber.

        Parameters
        ----------
        other_fiber : Fiber
            The target fiber.

        Returns
        -------
        np.ndarray
            Vector from self to other.
        """
        return np.array(other_fiber.center) - np.array(self.center)

    def move(self, move_vec: np.ndarray):
        """Translates the fiber by the given vector.

        Parameters
        ----------
        move_vec : np.ndarray
            Translation vector.
        """
        self.center = self.center + np.array(move_vec)

    def fix_overlap_with_neighbors(
        self, boundaries: List[LinearBoundary], min_space_between_fibers: float
    ) -> bool:
        """Iteratively resolves overlaps with neighboring fibers.

        Parameters
        ----------
        boundaries : List[LinearBoundary]
            List of RVE boundaries for constraint checking.
        min_space_between_fibers : float
            Minimum allowed distance between fiber surfaces.

        Returns
        -------
        bool
            True if any overlap was found and corrected, False otherwise.
        """
        overlap_found = False
        # Sort neighbors for deterministic behavior
        sorted_neighbors = sorted(
            list(self.neighbors), key=lambda f: (float(f.center[0]), float(f.center[1]))
        )
        for nf in sorted_neighbors:
            vec_to_other = self.get_vector_to_other(nf)
            dist = np.linalg.norm(vec_to_other)
            sum_radii = self.radius + nf.radius
            if dist < sum_radii + min_space_between_fibers:
                overlap_found = True
                if dist < 1e-15:
                    unit_vec_to_other = np.array([1.0, 0.0])
                else:
                    unit_vec_to_other = vec_to_other / dist
                proportion_to_move_this_fiber = nf.radius / sum_radii
                proportion_to_move_other_fiber = 1.0 - proportion_to_move_this_fiber
                total_move_back = (
                    sum_radii + min_space_between_fibers - dist + sum_radii * 1e-3
                )

                this_nudge = (
                    -proportion_to_move_this_fiber * total_move_back * unit_vec_to_other
                )
                other_nudge = (
                    proportion_to_move_other_fiber * total_move_back * unit_vec_to_other
                )

                self.move(this_nudge)
                self.adjust_for_bounds(boundaries)
                nf.move(other_nudge)
                nf.adjust_for_bounds(boundaries)
        return overlap_found

    def adjust_for_bounds(self, boundaries: List[LinearBoundary], eps=5.0e-2):
        """Checks to make sure this fiber does not violate any specified boundaries
        and will move it if necessary.
        """
        target_dist = (1.0 + eps) * self.radius
        for boundary in boundaries:
            if boundary.type not in (BoundaryType.FINITE, BoundaryType.SYMMETRIC):
                continue

            seg_vec = boundary.points[1] - boundary.points[0]
            seg_len = boundary.get_length()
            if seg_len <= 0:
                continue

            signed_dist = boundary.get_signed_distance(self.center)
            proj = float(np.dot(self.center - boundary.points[0], seg_vec) / seg_len)
            within_span = -self.radius <= proj <= seg_len + self.radius

            if not (
                boundary.check_collision(self.center, self.radius, eps)
                or (signed_dist < 0.0 and within_span)
            ):
                continue

            move_vector = boundary.get_move_vector()
            if boundary.type == BoundaryType.FINITE:
                if signed_dist < target_dist:
                    self.move((target_dist - signed_dist) * move_vector)
            elif boundary.type == BoundaryType.SYMMETRIC:
                if signed_dist < self.radius / 2.0:
                    if abs(signed_dist) > max(self.radius * 1e-12, 1e-15):
                        self.move(-signed_dist * move_vector)
                elif signed_dist < target_dist:
                    self.move((target_dist - signed_dist) * move_vector)


class PeriodicPrimaryFiber(Fiber):
    """A class for storing a fiber that is in the RVE domain."""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        index: int,
        boundaries: Optional[List[LinearBoundary]] = None,
        ignore_ghost_fibers=False,
    ):
        super().__init__(center, radius)
        self.index = index
        self.ghost_fibers: List["PeriodicGhostFiber"] = []
        self.ignore_ghost_fibers = ignore_ghost_fibers
        if boundaries is not None:
            self.calc_ghost_fibers(boundaries)

    def calc_ghost_fibers(
        self,
        boundaries: List[LinearBoundary],
        max_fiber_radius: Optional[float] = None,
    ):
        self.ghost_fibers = []
        if self.ignore_ghost_fibers:
            return

        ref_radius = (
            max(self.radius, max_fiber_radius)
            if max_fiber_radius is not None
            else self.radius
        )
        search_threshold = 3.0 * ref_radius
        dedup_tol = max(self.radius * 1e-6, 1e-15)

        first_pass = []
        for boundary in boundaries:
            if boundary.type == BoundaryType.PERIODIC and boundary.pair is not None:
                pvec = boundary.get_periodic_vector()
                if pvec is None:
                    continue
                if (
                    boundary.get_distance_to_fiber(self.center) < search_threshold
                    or boundary.get_signed_distance(self.center) < 0.0
                ):
                    first_pass.append((boundary, self.center + pvec))

        candidate_centers = [center for _, center in first_pass]

        # Second pass for multi-boundary / corner periodic images
        for b1, c1 in first_pass:
            for b2 in boundaries:
                if (
                    b2.type == BoundaryType.PERIODIC
                    and b2.pair is not None
                    and b2 is not b1
                    and b2 is not b1.pair
                ):
                    pvec2 = b2.get_periodic_vector()
                    if pvec2 is None:
                        continue
                    if b2.get_distance_to_fiber(c1) < search_threshold:
                        candidate_centers.append(c1 + pvec2)

        unique_ghost_centers = []
        for gc in candidate_centers:
            if np.linalg.norm(gc - self.center) <= dedup_tol:
                continue
            if any(
                np.linalg.norm(gc - existing) <= dedup_tol
                for existing in unique_ghost_centers
            ):
                continue
            unique_ghost_centers.append(gc)

        for ghost_center in unique_ghost_centers:
            self.ghost_fibers.append(
                PeriodicGhostFiber(ghost_center, self.radius, self)
            )

    def adjust_for_bounds(self, boundaries: List[LinearBoundary], eps=5.0e-2):
        super().adjust_for_bounds(boundaries, eps)
        # Wrap primary fiber back into the primary RVE if it drifts completely
        # across a periodic boundary by more than its own radius.
        for _ in range(2):
            wrapped = False
            for boundary in boundaries:
                if boundary.type == BoundaryType.PERIODIC and boundary.pair is not None:
                    if boundary.get_signed_distance(self.center) < -self.radius:
                        pvec = boundary.get_periodic_vector()
                        if pvec is not None:
                            self.move(pvec)
                            wrapped = True
            if not wrapped:
                break

    def move(self, move_vec: np.ndarray):
        super().move(move_vec)
        self.vec_moved_since_neighbor_update += np.array(move_vec)
        for ghost in self.ghost_fibers:
            ghost.move_actual(move_vec)

    def get_distance_since_last_neighbor_update(self) -> float:
        return np.linalg.norm(self.vec_moved_since_neighbor_update)

    def get_all_copies(self) -> List[Fiber]:
        fibers = list(self.ghost_fibers)
        fibers.append(self)
        return fibers

    def update_neighbors(
        self,
        fiber_center_kd_tree,
        fibers_with_ghosts: List[Fiber],
        fiber_diams_to_search: float,
        max_fiber_radius: Optional[float] = None,
    ):
        self.neighbors = set()
        self.vec_moved_since_neighbor_update = np.array([0.0, 0.0])

        other_max_r = max_fiber_radius if max_fiber_radius is not None else self.radius
        search_dist = (self.radius + other_max_r) * fiber_diams_to_search
        valid_indices = fiber_center_kd_tree.query_ball_point(
            self.center, r=search_dist, p=2
        )

        for i in valid_indices:
            if i >= len(fibers_with_ghosts):
                continue
            other = fibers_with_ghosts[i]
            # Check logic from original: neighbors should have higher index to avoid double checking pairs
            if isinstance(other, PeriodicPrimaryFiber):
                if other.index > self.index:
                    self.neighbors.add(other)
            elif isinstance(other, PeriodicGhostFiber):
                if other.primary_fiber.index > self.index:
                    self.neighbors.add(other)

        for gf in self.ghost_fibers:
            gf.update_neighbors(
                fiber_center_kd_tree,
                fibers_with_ghosts,
                fiber_diams_to_search,
                max_fiber_radius=max_fiber_radius,
            )

    def fix_overlap_with_neighbors(
        self, boundaries: List[LinearBoundary], min_space_between_fibers: float
    ) -> bool:
        found_overlap = super().fix_overlap_with_neighbors(
            boundaries, min_space_between_fibers
        )
        for ghost_fib in self.ghost_fibers:
            found_overlap |= ghost_fib.fix_overlap_with_neighbors(
                boundaries, min_space_between_fibers
            )
        return found_overlap


class PeriodicGhostFiber(Fiber):
    def __init__(
        self, center: np.ndarray, radius: float, primary_fiber: PeriodicPrimaryFiber
    ):
        super().__init__(center, radius)
        self.primary_fiber = primary_fiber

    def move(self, move_vec: np.ndarray):
        self.primary_fiber.move(move_vec)

    def move_actual(self, move_vec: np.ndarray):
        super().move(move_vec)

    def adjust_for_bounds(self, boundaries: List[LinearBoundary], eps=5.0e-2):
        self.primary_fiber.adjust_for_bounds(boundaries, eps)

    def update_neighbors(
        self,
        fiber_center_kd_tree,
        fibers_with_ghosts: List[Fiber],
        fiber_diams_to_search: float,
        max_fiber_radius: Optional[float] = None,
    ):
        self.neighbors = set()
        self.vec_moved_since_neighbor_update = np.array([0.0, 0.0])

        other_max_r = max_fiber_radius if max_fiber_radius is not None else self.radius
        search_dist = (self.radius + other_max_r) * fiber_diams_to_search
        valid_indices = fiber_center_kd_tree.query_ball_point(
            self.center, r=search_dist, p=2
        )

        for i in valid_indices:
            if i >= len(fibers_with_ghosts):
                continue
            other = fibers_with_ghosts[i]
            if isinstance(other, PeriodicPrimaryFiber):
                if other.index > self.primary_fiber.index:
                    self.neighbors.add(other)
            elif isinstance(other, PeriodicGhostFiber):
                if other.primary_fiber.index > self.primary_fiber.index:
                    self.neighbors.add(other)
