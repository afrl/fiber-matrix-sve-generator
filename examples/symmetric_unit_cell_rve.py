import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from fiber_matrix import BoundaryType, FiberRVE


def build_rve(seed: int = 4) -> FiberRVE:
    np.random.seed(seed)
    rve = FiberRVE()
    # Note: 4 corner quarter-fibers + 4 edge half-fibers + 12 interior fibers
    # yield (4*0.25 + 4*0.5 + 12) = 15 effective full fiber cross-sections inside the cell.
    w, h = rve.initialize_rectangle_rve(
        num_fibers=20,
        vf=0.52,
        avg_diam=5.0,
        rve_aspect_ratio=1.0,
        boundary_type=BoundaryType.SYMMETRIC,
    )
    init_centers = [
        [0.0, 0.0],
        [w, 0.0],
        [w, h],
        [0.0, h],
        [0.5 * w, 0.0],
        [w, 0.5 * h],
        [0.5 * w, h],
        [0.0, 0.5 * h],
    ]
    for _ in range(20 - len(init_centers)):
        init_centers.append(
            [
                float(np.random.uniform(0.18 * w, 0.82 * w)),
                float(np.random.uniform(0.18 * h, 0.82 * h)),
            ]
        )
    rve.place_initial_fibers(specified_fiber_centers=init_centers)
    return rve


def main():
    show_guis = False
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        rve = build_rve(seed=4)
        print(f"RVE dimensions: {rve.rve_dims}")
        iterations = rve.solve_fiber_locations(
            min_spacing_ratio=0.08,
            visualization_path=Path(__file__).stem + ".gif",
        )
        print(f"Solved symmetric SVE in {iterations} iterations.")
        if show_guis:
            rve.draw()
            plt.show()

        rve.create_mesh(
            mesh_name=Path(__file__).stem,
            mesh_size_factor=0.75,
            visualize_gui=show_guis,
            check_periodicity=False,
        )
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
