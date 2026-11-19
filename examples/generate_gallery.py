"""Generates the 2D SVE microstructure & finite element mesh gallery for README.md.

Run from the repository root:
    python examples/generate_gallery.py
"""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, REPO_ROOT.as_posix())

from fiber_matrix import BoundaryType, FiberRVE
from examples import (
    bimodal_hybrid_rve,
    composite_ply_mixed_rve,
    polydisperse_square_rve,
    symmetric_unit_cell_rve,
)


BOUNDARY_STYLE = {
    BoundaryType.PERIODIC: ("#16a34a", "-", "Periodic Boundary"),
    BoundaryType.FINITE: ("#dc2626", "-", "Finite Wall Boundary"),
    BoundaryType.SYMMETRIC: ("#7c3aed", "-.", "Symmetry Plane"),
}


@dataclass
class GalleryCase:
    slug: str
    title: str
    subtitle: str
    script_relpath: str
    builder: Callable[[], Tuple[FiberRVE, int]]
    mesh_size_factor: float
    check_periodicity: bool
    bimodal_split_radius: float = 0.0


def _build_periodic_square() -> Tuple[FiberRVE, int]:
    np.random.seed(8)
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=28,
        vf=0.60,
        avg_diam=5.0,
        rve_aspect_ratio=1.0,
        boundary_type=BoundaryType.PERIODIC,
    )
    rve.place_initial_fibers()
    iters = rve.solve_fiber_locations(min_spacing_ratio=0.07)
    return rve, iters


def _build_periodic_hex() -> Tuple[FiberRVE, int]:
    np.random.seed(14)
    rve = FiberRVE()
    rve.initialize_hexagonal_rve(
        num_fibers=24,
        vf=0.50,
        avg_diam=5.0,
        boundary_type=BoundaryType.PERIODIC,
    )
    rve.place_initial_fibers()
    iters = rve.solve_fiber_locations(min_spacing_ratio=0.08)
    return rve, iters


def _build_polydisperse() -> Tuple[FiberRVE, int]:
    rve = polydisperse_square_rve.build_rve(seed=21)
    iters = rve.solve_fiber_locations(min_spacing_ratio=0.08)
    return rve, iters


def _build_bimodal() -> Tuple[FiberRVE, int]:
    rve = bimodal_hybrid_rve.build_rve(seed=17)
    iters = rve.solve_fiber_locations(min_spacing_ratio=0.06)
    return rve, iters


def _build_composite_ply() -> Tuple[FiberRVE, int]:
    rve = composite_ply_mixed_rve.build_rve(seed=34)
    iters = rve.solve_fiber_locations(min_spacing_ratio=0.08)
    return rve, iters


def _build_symmetric() -> Tuple[FiberRVE, int]:
    rve = symmetric_unit_cell_rve.build_rve(seed=4)
    iters = rve.solve_fiber_locations(min_spacing_ratio=0.08)
    return rve, iters


def _extract_2d_mesh_data(msh_path: str) -> Dict[str, object]:
    """Reads triangle coordinates grouped by physical group ('Matrix', 'Fibers')."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(msh_path)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)
        node_map = {int(tag): coords[i, :2] for i, tag in enumerate(node_tags)}

        group_tris: Dict[str, List[np.ndarray]] = {"Matrix": [], "Fibers": []}
        group_areas: Dict[str, float] = {"Matrix": 0.0, "Fibers": 0.0}

        for dim, ptag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, ptag)
            tris_out = []
            area_sum = 0.0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2, ent)
                for etype, nt in zip(elem_types, elem_nodes):
                    if etype != 2:  # 3-node triangle
                        continue
                    tri_ids = nt.reshape(-1, 3)
                    for t0, t1, t2 in tri_ids:
                        p0, p1, p2 = node_map[int(t0)], node_map[int(t1)], node_map[int(t2)]
                        tris_out.append(np.array([p0, p1, p2]))
                        area_sum += 0.5 * abs(
                            (p1[0] - p0[0]) * (p2[1] - p0[1])
                            - (p1[1] - p0[1]) * (p2[0] - p0[0])
                        )
            group_tris[name] = tris_out
            group_areas[name] = area_sum

        n_tris = sum(len(v) for v in group_tris.values())
        tot_area = group_areas["Matrix"] + group_areas["Fibers"]
        mesh_vf = group_areas["Fibers"] / tot_area if tot_area > 0 else 0.0
        return {
            "num_nodes": len(node_tags),
            "num_elements": n_tris,
            "group_tris": group_tris,
            "mesh_vf": mesh_vf,
        }
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()


def _get_domain_polygon_and_bounds(rve: FiberRVE):
    verts = np.array([b.points[0] for b in rve.boundaries], dtype=float)
    cmin = verts.min(axis=0)
    cmax = verts.max(axis=0)
    span = np.max(cmax - cmin)
    return verts, cmin, cmax, span


def _plot_microstructure_ax(
    ax: plt.Axes,
    rve: FiberRVE,
    bimodal_split_radius: float = 0.0,
    show_legend: bool = True,
):
    verts, cmin, cmax, span = _get_domain_polygon_and_bounds(rve)
    pad = 0.14 * span

    # Domain background polygon
    domain_patch = mpatches.Polygon(
        verts,
        closed=True,
        facecolor="#f1f5f9",
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(domain_patch)

    # Ghost fibers (within visible frame)
    has_ghosts = False
    max_r = max((f.radius for f in rve.fibers), default=0.0)
    for f in rve.fibers:
        f.calc_ghost_fibers(rve.boundaries, max_fiber_radius=max_r)
        for g in f.ghost_fibers:
            if (
                cmin[0] - pad <= g.center[0] <= cmax[0] + pad
                and cmin[1] - pad <= g.center[1] <= cmax[1] + pad
            ):
                has_ghosts = True
                ax.add_patch(
                    mpatches.Circle(
                        g.center,
                        g.radius,
                        facecolor="#fb923c",
                        edgecolor="#c2410c",
                        linewidth=0.9,
                        linestyle="--",
                        alpha=0.35,
                        zorder=2,
                    )
                )

    # Primary fibers
    for f in rve.fibers:
        if bimodal_split_radius > 0 and f.radius < bimodal_split_radius:
            fc, ec = "#0d9488", "#115e59"
        else:
            fc, ec = "#2563eb", "#1e3a8a"
        ax.add_patch(
            mpatches.Circle(
                f.center,
                f.radius,
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.0,
                alpha=0.88,
                zorder=3,
            )
        )

    # Boundaries styled by BoundaryType
    seen_btypes = set()
    for b in rve.boundaries:
        color, ls, label = BOUNDARY_STYLE[b.type]
        lbl = label if b.type not in seen_btypes else None
        seen_btypes.add(b.type)
        ax.plot(
            [b.points[0][0], b.points[1][0]],
            [b.points[0][1], b.points[1][1]],
            color=color,
            linestyle=ls,
            linewidth=2.2,
            zorder=4,
            label=lbl,
        )

    ax.set_xlim(cmin[0] - pad, cmax[0] + pad)
    ax.set_ylim(cmin[1] - pad, cmax[1] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show_legend:
        handles = []
        if bimodal_split_radius > 0:
            handles.append(
                mpatches.Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="Coarse Fibers")
            )
            handles.append(
                mpatches.Patch(facecolor="#0d9488", edgecolor="#115e59", label="Fine Fibers")
            )
        else:
            handles.append(
                mpatches.Patch(facecolor="#2563eb", edgecolor="#1e3a8a", label="Primary Fibers")
            )
        if has_ghosts:
            handles.append(
                mpatches.Patch(
                    facecolor="#fb923c",
                    edgecolor="#c2410c",
                    alpha=0.45,
                    linestyle="--",
                    label="Periodic Ghosts",
                )
            )
        for bt in [BoundaryType.PERIODIC, BoundaryType.FINITE, BoundaryType.SYMMETRIC]:
            if bt in seen_btypes:
                c, ls, lbl = BOUNDARY_STYLE[bt]
                handles.append(plt.Line2D([0], [0], color=c, lw=2.0, ls=ls, label=lbl))
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=min(len(handles), 2),
            fontsize=8.0,
            framealpha=0.95,
            edgecolor="#cbd5e1",
        )


def _plot_mesh_ax(ax: plt.Axes, rve: FiberRVE, mesh_data: Dict[str, object]):
    verts, cmin, cmax, span = _get_domain_polygon_and_bounds(rve)
    pad = 0.06 * span

    matrix_tris = mesh_data["group_tris"].get("Matrix", [])
    fiber_tris = mesh_data["group_tris"].get("Fibers", [])

    if matrix_tris:
        ax.add_collection(
            PolyCollection(
                matrix_tris,
                facecolors="#e2e8f0",
                edgecolors="#475569",
                linewidths=0.32,
                zorder=1,
            )
        )
    if fiber_tris:
        ax.add_collection(
            PolyCollection(
                fiber_tris,
                facecolors="#3b82f6",
                edgecolors="#0f172a",
                linewidths=0.32,
                zorder=2,
            )
        )

    for b in rve.boundaries:
        color, ls, _ = BOUNDARY_STYLE[b.type]
        ax.plot(
            [b.points[0][0], b.points[1][0]],
            [b.points[0][1], b.points[1][1]],
            color=color,
            linestyle=ls,
            linewidth=1.8,
            zorder=3,
        )

    ax.set_xlim(cmin[0] - pad, cmax[0] + pad)
    ax.set_ylim(cmin[1] - pad, cmax[1] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_case_card(
    case: GalleryCase,
    rve: FiberRVE,
    iters: int,
    mesh_data: Dict[str, object],
    out_path: Path,
):
    fig, (ax_geo, ax_mesh) = plt.subplots(
        1, 2, figsize=(11.2, 5.4), facecolor="#ffffff"
    )
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.03, right=0.97, wspace=0.08)

    _plot_microstructure_ax(
        ax_geo,
        rve,
        bimodal_split_radius=case.bimodal_split_radius,
        show_legend=True,
    )
    ax_geo.set_title(
        f"SVE Microstructure ({len(rve.fibers)} fibers, solved in {iters} iters)",
        fontsize=10.5,
        fontweight="semibold",
        color="#1e293b",
        pad=8,
    )

    _plot_mesh_ax(ax_mesh, rve, mesh_data)
    periodic_tag = " | Periodic Nodes Verified" if case.check_periodicity else ""
    ax_mesh.set_title(
        f"Gmsh 2D FE Mesh ({mesh_data['num_nodes']:,} nodes, "
        f"{mesh_data['num_elements']:,} tris, VF={mesh_data['mesh_vf']*100:.1f}%{periodic_tag})",
        fontsize=10.0,
        fontweight="semibold",
        color="#1e293b",
        pad=8,
    )

    fig.suptitle(
        f"{case.title} — {case.subtitle}",
        fontsize=13.0,
        fontweight="bold",
        color="#0f172a",
        y=0.95,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def _render_overview_grid(
    rendered_items: List[Tuple[GalleryCase, FiberRVE, Dict[str, object]]],
    out_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), facecolor="#ffffff")
    fig.subplots_adjust(
        top=0.90, bottom=0.03, left=0.02, right=0.98, hspace=0.18, wspace=0.08
    )

    for ax, (case, rve, mesh_data) in zip(axes.flat, rendered_items):
        _plot_mesh_ax(ax, rve, mesh_data)
        ax.set_title(
            f"{case.title}\nVF={mesh_data['mesh_vf']*100:.1f}% | {mesh_data['num_elements']:,} elements",
            fontsize=10.5,
            fontweight="semibold",
            color="#0f172a",
            pad=6,
        )

    fig.suptitle(
        "Fiber-Matrix SVE Generator — 2D Microstructure & Finite Element Mesh Gallery",
        fontsize=15.0,
        fontweight="bold",
        color="#0f172a",
        y=0.97,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def _generate_relaxation_gif(out_gif_path: Path):
    np.random.seed(8)
    rve = FiberRVE()
    rve.initialize_rectangle_rve(
        num_fibers=24,
        vf=0.58,
        avg_diam=5.0,
        rve_aspect_ratio=1.0,
        boundary_type=BoundaryType.PERIODIC,
    )
    rve.place_initial_fibers()
    rve.solve_fiber_locations(
        min_spacing_ratio=0.08,
        visualization_path=str(out_gif_path),
    )


def main():
    gallery_dir = REPO_ROOT / "docs" / "assets" / "gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        GalleryCase(
            slug="periodic_square_sve",
            title="Periodic Square High-VF SVE",
            subtitle="Dense unidirectional packing (VF = 60%, 28 fibers)",
            script_relpath="examples/periodic_square_rve.py",
            builder=_build_periodic_square,
            mesh_size_factor=0.72,
            check_periodicity=True,
        ),
        GalleryCase(
            slug="periodic_hexagonal_sve",
            title="Periodic Hexagonal SVE",
            subtitle="3-axis hexagonal periodicity (VF = 50%, 24 fibers)",
            script_relpath="examples/periodic_hex_rve.py",
            builder=_build_periodic_hex,
            mesh_size_factor=0.75,
            check_periodicity=True,
        ),
        GalleryCase(
            slug="polydisperse_sve",
            title="Polydisperse Variable-Diameter SVE",
            subtitle="Gaussian diameter dispersion (CoV = 22%, VF = 52%)",
            script_relpath="examples/polydisperse_square_rve.py",
            builder=_build_polydisperse,
            mesh_size_factor=0.72,
            check_periodicity=True,
        ),
        GalleryCase(
            slug="bimodal_hybrid_sve",
            title="Bimodal Hybrid Composite SVE",
            subtitle="Coarse structural + fine interstitial fibers (VF = 55%, 41 fibers)",
            script_relpath="examples/bimodal_hybrid_rve.py",
            builder=_build_bimodal,
            mesh_size_factor=0.68,
            check_periodicity=True,
            bimodal_split_radius=2.5,
        ),
        GalleryCase(
            slug="composite_ply_mixed_sve",
            title="Composite Ply Mixed-Boundary SVE",
            subtitle="Periodic width + resin-rich finite top/bottom faces (VF = 48%)",
            script_relpath="examples/composite_ply_mixed_rve.py",
            builder=_build_composite_ply,
            mesh_size_factor=0.78,
            check_periodicity=True,
        ),
        GalleryCase(
            slug="symmetric_sve",
            title="Symmetric Mirror-Plane Sub-Cell SVE",
            subtitle="Reflection symmetry planes with half/quarter & interior fibers (Effective VF = 39%)",
            script_relpath="examples/symmetric_unit_cell_rve.py",
            builder=_build_symmetric,
            mesh_size_factor=0.72,
            check_periodicity=False,
        ),
    ]

    rendered_items: List[Tuple[GalleryCase, FiberRVE, Dict[str, object]]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for case in cases:
            print(f"Generating gallery case: {case.title}...")
            rve, iters = case.builder()
            mesh_prefix = os.path.join(tmpdir, case.slug)
            rve.create_mesh(
                mesh_name=mesh_prefix,
                mesh_size_factor=case.mesh_size_factor,
                visualize_gui=False,
                check_periodicity=case.check_periodicity,
            )
            mesh_data = _extract_2d_mesh_data(mesh_prefix + ".msh")
            card_path = gallery_dir / f"{case.slug}.png"
            _render_case_card(case, rve, iters, mesh_data, card_path)
            print(f"  Saved {card_path.relative_to(REPO_ROOT)}")
            rendered_items.append((case, rve, mesh_data))

    overview_path = gallery_dir / "gallery_overview.png"
    _render_overview_grid(rendered_items, overview_path)
    print(f"Saved overview grid: {overview_path.relative_to(REPO_ROOT)}")

    gif_path = gallery_dir / "packing_relaxation.gif"
    _generate_relaxation_gif(gif_path)
    print(f"Saved relaxation animation: {gif_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
