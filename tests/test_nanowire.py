from __future__ import annotations

import json

import numpy as np
import pytest

from voronoi_aerogel.geometry import (
    build_periodic_voronoi,
    generate_periodic_seeds,
)
from voronoi_aerogel.topology import analyze_winding
from voronoi_nanowire.exporters import (
    build_display_mesh,
    save_preview_png,
    write_metadata,
)
from voronoi_nanowire.geometry import (
    PeriodicEdge,
    build_edge_graph,
    extract_periodic_edges,
    periodic_edge_segments,
)
from voronoi_nanowire.selection import (
    CapsuleCoverageEstimator,
    select_connected_edges,
)


@pytest.fixture(scope="module")
def periodic_edges() -> tuple[float, list[PeriodicEdge]]:
    box_size_nm = 100.0
    seeds = generate_periodic_seeds(
        12,
        box_size_nm,
        min_distance_factor=0.30,
        random_seed=7,
    )
    tessellation = build_periodic_voronoi(seeds, box_size_nm)
    return box_size_nm, extract_periodic_edges(tessellation.faces, box_size_nm)


def test_periodic_edges_are_unique_three_face_edges_and_clip_conservatively(
    periodic_edges: tuple[float, list[PeriodicEdge]],
) -> None:
    box_size_nm, edges = periodic_edges
    keys = [
        (edge.start_vertex, edge.end_vertex, *edge.shift)
        for edge in edges
    ]
    assert len(edges) == len(set(keys))
    assert all(len(edge.incident_face_ids) == 3 for edge in edges)

    winding = analyze_winding(build_edge_graph(edges))
    assert winding.connected
    assert winding.has_three_axis_component

    for edge in edges:
        segments = periodic_edge_segments(edge, box_size_nm)
        assert segments
        clipped_length = sum(
            np.linalg.norm(segment[1] - segment[0]) for segment in segments
        )
        assert clipped_length == pytest.approx(edge.length_nm, rel=1.0e-10)
        assert all(
            np.all(segment >= -1.0e-9)
            and np.all(segment <= box_size_nm + 1.0e-9)
            for segment in segments
        )


def _straight_edge(edge_id: int) -> PeriodicEdge:
    start = np.asarray([25.0, 50.0, 50.0])
    end = np.asarray([75.0, 50.0, 50.0])
    return PeriodicEdge(
        id=edge_id,
        start_vertex=0,
        end_vertex=1,
        shift=(0, 0, 0),
        start_nm=start,
        end_nm=end,
        length_nm=float(np.linalg.norm(end - start)),
        incident_face_ids=(0, 1, 2),
    )


def test_capsule_union_counts_identical_rods_only_once() -> None:
    estimator = CapsuleCoverageEstimator(
        [_straight_edge(0), _straight_edge(1)],
        cube_size_nm=100.0,
        rod_diameter_nm=10.0,
        qmc_power=14,
        seed=5,
    )

    once = estimator.union_count([0])
    twice = estimator.union_count([0, 1])
    assert once > 0
    assert twice == once
    np.testing.assert_array_equal(estimator.edge_indices(0), estimator.edge_indices(1))


def test_random_selection_reaches_an_attainable_connected_target(
    periodic_edges: tuple[float, list[PeriodicEdge]],
) -> None:
    box_size_nm, edges = periodic_edges
    graph = build_edge_graph(edges)
    estimator = CapsuleCoverageEstimator(
        edges,
        cube_size_nm=box_size_nm,
        rod_diameter_nm=8.0,
        qmc_power=9,
        seed=13,
    )
    target_porosity = 0.92
    result = select_connected_edges(
        edges,
        graph,
        estimator,
        target_porosity=target_porosity,
        tolerance=0.01,
        random_seed=21,
    )
    repeated = select_connected_edges(
        edges,
        graph,
        estimator,
        target_porosity=target_porosity,
        tolerance=0.01,
        random_seed=21,
    )

    assert result.attainable_range.min_porosity <= target_porosity
    assert target_porosity <= result.attainable_range.max_porosity
    assert result.status == "REACHED"
    assert result.within_tolerance
    assert result.absolute_porosity_error <= 0.01
    assert result.winding.connected
    assert result.winding.has_three_axis_component
    assert set(result.backbone_edge_ids) <= set(result.selected_edge_ids)
    assert repeated.selected_edge_ids == result.selected_edge_ids
    assert repeated.achieved_porosity == result.achieved_porosity


def test_display_mesh_preserves_radius_and_writes_basic_outputs(tmp_path) -> None:
    segment = np.asarray([[[20.0, 50.0, 50.0], [80.0, 50.0, 50.0]]])
    radius_nm = 4.0
    tube_sides = 8
    mesh = build_display_mesh(
        segment,
        radius_nm,
        tube_sides=tube_sides,
        sphere_stacks=4,
    )

    assert mesh["segment_count"] == 1
    assert mesh["node_count"] == 2
    assert mesh["radius_nm"] == radius_nm
    assert mesh["vertices_nm"].shape == (68, 3)
    assert mesh["triangles"].shape == (112, 3)
    assert mesh["node_centres_nm"].shape == (2, 3)

    explicit_nodes_mesh = build_display_mesh(
        segment,
        radius_nm,
        tube_sides=tube_sides,
        sphere_stacks=4,
        node_centres_nm=np.asarray([[20.0, 50.0, 50.0]]),
    )
    assert explicit_nodes_mesh["node_count"] == 1
    assert explicit_nodes_mesh["triangles"].shape == (80, 3)

    cylinder_rings = mesh["vertices_nm"][: 2 * tube_sides]
    centres = np.repeat(segment[0], tube_sides, axis=0)
    np.testing.assert_allclose(
        np.linalg.norm(cylinder_rings - centres, axis=1),
        radius_nm,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    png_path = save_preview_png(
        tmp_path / "preview.png",
        mesh,
        cube_size_nm=100.0,
        dpi=40,
    )
    metadata_path = write_metadata(
        tmp_path / "metadata.json",
        {"rod_radius_nm": np.float64(radius_nm), "segment_count": 1},
    )

    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "rod_radius_nm": radius_nm,
        "segment_count": 1,
        "model_type": "periodic_voronoi_nanowire",
        "coordinate_unit": "nm",
        "mesh_purpose": "visualization_only",
    }
