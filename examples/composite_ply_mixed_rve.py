import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from fiber_matrix import BoundaryType, FiberRVE


def build_rve(seed: int = 34) -> FiberRVE:
    np.random.seed(seed)
    rve = FiberRVE()
    # Thin composite ply section: FINITE resin-rich top/bottom faces, PERIODIC left/right edges
    rve.initialize_rectangle_rve(
        num_fibers=26,
        vf=0.48,
        avg_diam=5.0,
        diam_std_dev=0.08,
        rve_aspect_ratio=1.75,
        boundary_type=[
            BoundaryType.FINITE,    # Bottom ply face
            BoundaryType.PERIODIC,  # Right periodic boundary
            BoundaryType.FINITE,    # Top ply face
            BoundaryType.PERIODIC,  # Left periodic boundary
        ],
    )
    rve.place_initial_fibers()
    return rve


def main():
    show_guis = False
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        rve = build_rve(seed=34)
        print(f"RVE dimensions: {rve.rve_dims}")
        iterations = rve.solve_fiber_locations(
            min_spacing_ratio=0.08,
            visualization_path=Path(__file__).stem + ".gif",
        )
        print(f"Solved composite ply mixed-boundary SVE in {iterations} iterations.")
        if show_guis:
            rve.draw()
            plt.show()

        rve.create_mesh(
            mesh_name=Path(__file__).stem,
            mesh_size_factor=0.80,
            visualize_gui=show_guis,
            check_periodicity=True,
        )
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
