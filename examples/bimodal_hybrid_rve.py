import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from fiber_matrix import BoundaryType, FiberRVE


def build_rve(seed: int = 17) -> FiberRVE:
    np.random.seed(seed)
    # Hybrid bimodal composite: 7 coarse structural fibers (R=4.2) + 34 fine fibers (R=1.35)
    radii = [4.2] * 7 + [1.35] * 34
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=len(radii),
        vf=0.55,
        rve_aspect_ratio=1.0,
        boundary_type=BoundaryType.PERIODIC,
        radii=radii,
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
        rve = build_rve(seed=17)
        print(f"RVE dimensions: {rve.rve_dims}")
        iterations = rve.solve_fiber_locations(
            min_spacing_ratio=0.06,
            visualization_path=Path(__file__).stem + ".gif",
        )
        print(f"Solved bimodal hybrid SVE in {iterations} iterations.")
        if show_guis:
            rve.draw()
            plt.show()

        rve.create_mesh(
            mesh_name=Path(__file__).stem,
            mesh_size_factor=0.75,
            visualize_gui=show_guis,
            check_periodicity=True,
        )
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
