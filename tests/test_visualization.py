import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fiber_matrix import FiberRVE
from fiber_matrix.visualization import plotting


def test_draw_rve_reuses_passed_axes_without_leaking_figures():
    plt.close("all")
    np.random.seed(1)
    rve = FiberRVE()
    rve.initialize_rectangle_rve(num_fibers=5, vf=0.4, avg_diam=5.0)
    rve.place_initial_fibers()

    fig, ax = plt.subplots()
    n_figs_before = len(plt.get_fignums())
    ret_fig, ret_ax = plotting.draw_rve(
        rve.fibers,
        rve.boundaries,
        ax=ax,
        label_fibers=True,
        label_boundaries=True,
    )
    assert ret_fig is fig
    assert ret_ax is ax
    assert len(plt.get_fignums()) == n_figs_before
    # Labels are drawn on target ax
    assert len(ax.texts) >= len(rve.fibers) + len(rve.boundaries)
    plt.close("all")


def test_draw_rve_frame_export_and_show_final(tmp_path, monkeypatch):
    plt.close("all")
    monkeypatch.chdir(tmp_path)

    np.random.seed(2)
    rve = FiberRVE()
    rve.initialize_rectangle_rve(num_fibers=4, vf=0.35, avg_diam=5.0)
    rve.place_initial_fibers()

    fig, _ = rve.draw(frame=0)
    assert os.path.exists(tmp_path / "frames" / "0_RVE.png")
    plt.close(fig)

    draw_called = []
    orig_draw = rve.draw

    def _spy_draw(*args, **kwargs):
        draw_called.append(True)
        return orig_draw(*args, **kwargs)

    monkeypatch.setattr(rve, "draw", _spy_draw)
    rve.solve_fiber_locations(min_spacing_ratio=0.05, show_final=True)
    assert len(draw_called) == 1
    plt.close("all")
