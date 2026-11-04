from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voronoi_aerogel.geometry import (
    build_periodic_voronoi,
    generate_periodic_seeds,
    periodic_face_patches,
)
from voronoi_aerogel.mesh import mesh_from_occupancy, mesh_quality
from voronoi_aerogel.selection import (
    SobolCoverageEstimator,
    build_pore_graph,
    coerce_face_graph,
)
from voronoi_aerogel.topology import analyze_winding


def _polygon_area(vertices: np.ndarray) -> float:
    return 0.5 * float(
        np.linalg.norm(
            np.sum(np.cross(vertices, np.roll(vertices, -1, axis=0)), axis=0)
        )
    )


def test_periodic_voronoi_topology_and_clipped_area() -> None:
    box = 100.0
    seeds = generate_periodic_seeds(30, box, random_seed=11)
    geometry = build_periodic_voronoi(seeds, box)
    assert len({face.key for face in geometry.faces}) == len(geometry.faces)

    pore = build_pore_graph(geometry.faces)
    face_graph = coerce_face_graph(
        geometry.face_adjacency, face_count=len(geometry.faces)
    )
    pore_report = analyze_winding(pore)
    solid_report = analyze_winding(face_graph)
    assert pore_report.connected and pore_report.has_three_axis_component
    assert solid_report.connected and solid_report.has_three_axis_component

    for face in geometry.faces[:: max(1, len(geometry.faces) // 25)]:
        clipped_area = sum(
            _polygon_area(patch) for patch in periodic_face_patches(face, box)
        )
        assert np.isclose(clipped_area, face.area_nm2, rtol=2e-8, atol=1e-8)


@dataclass
class _Face:
    vertices_nm: np.ndarray
    normal: np.ndarray
    area_nm2: float
    equivalent_diameter_nm: float
    cell_a: int = 0
    cell_b: int = 1
    shift: tuple[int, int, int] = (0, 0, 0)


def test_sobol_union_counts_sheet_overlap_once() -> None:
    box = 100.0
    vertices = np.asarray(
        [[50, 0, 0], [50, 100, 0], [50, 100, 100], [50, 0, 100]],
        dtype=float,
    )
    face = _Face(vertices, np.asarray([1.0, 0.0, 0.0]), 10_000.0, 112.84)
    estimator = SobolCoverageEstimator(
        [face, face],
        cube_size_nm=box,
        thickness_nm=10.0,
        qmc_power=16,
        seed=5,
    )
    once = estimator.union_count([0]) / estimator.sample_count
    twice = estimator.union_count([0, 1]) / estimator.sample_count
    assert abs(once - 0.1) < 0.002
    assert twice == once


def test_voxel_surface_is_closed() -> None:
    occupancy = np.zeros((24, 24, 24), dtype=bool)
    occupancy[10:14, 3:21, 2:22] = True
    mesh = mesh_from_occupancy(
        occupancy,
        cube_size_nm=120.0,
        method="lorensen",
    )
    quality = mesh_quality(mesh)
    assert quality["is_watertight"]
    assert quality["boundary_edge_count"] == 0
    assert quality["nonmanifold_edge_count"] == 0
    assert quality["volume_nm3"] > 0.0
