from __future__ import annotations

import json

import numpy as np
import pytest

from aerogel_studio.variable_wire import (
    VariableCapsuleCoverageEstimator,
    generate_variable_wire,
    voxelize_periodic_capsules,
)
from voronoi_aerogel.mesh import mesh_quality
from voronoi_nanowire.geometry import PeriodicEdge
from voronoi_nanowire.selection import CapsuleCoverageEstimator


def _edge(edge_id: int, start=(25.0, 50.0, 50.0), end=(75.0, 50.0, 50.0)) -> PeriodicEdge:
    start, end = np.asarray(start), np.asarray(end)
    return PeriodicEdge(edge_id, 0, 1, (0, 0, 0), start, end, float(np.linalg.norm(end - start)), (0, 1, 2))


def test_actual_unequal_radii_control_union_and_constant_case_matches_core():
    edges = [_edge(0), _edge(1)]
    variable = VariableCapsuleCoverageEstimator(
        edges, diameters_nm=[8.0, 16.0], cube_size_nm=100.0, qmc_power=15, seed=13,
    )
    small, large = variable.union_count([0]), variable.union_count([1])
    assert large > 3 * small
    assert variable.union_count([0, 1]) == large
    assert variable.nominal_capsule_volume_nm3(1) > variable.nominal_capsule_volume_nm3(0)
    fixed = CapsuleCoverageEstimator(edges, cube_size_nm=100.0, rod_diameter_nm=8.0, qmc_power=15, seed=13)
    equal = VariableCapsuleCoverageEstimator(
        edges, diameters_nm=[8.0, 8.0], cube_size_nm=100.0, qmc_power=15, seed=13,
    )
    np.testing.assert_array_equal(equal.edge_indices(0), fixed.edge_indices(0))


def test_voxel_union_counts_overlap_once_and_wraps_capsules():
    edge = _edge(0, (90.0, 50.0, 50.0), (110.0, 50.0, 50.0))
    duplicate = _edge(1, (90.0, 50.0, 50.0), (110.0, 50.0, 50.0))
    once = voxelize_periodic_capsules([edge], [12.0], cube_size_nm=100.0, voxel_size_nm=2.0)
    twice = voxelize_periodic_capsules([edge, duplicate], [8.0, 12.0], cube_size_nm=100.0, voxel_size_nm=2.0)
    np.testing.assert_array_equal(once, twice)
    assert once[0].any() and once[-1].any()
    assert not once[25].any()


def test_generated_variable_wire_is_closed_clipped_and_reproducible(tmp_path):
    config = dict(
        cube_size_nm=100.0, num_seeds=12,
        rod_diameter_min_nm=8.0, rod_diameter_max_nm=10.0,
        target_porosity=0.92, porosity_tolerance=0.02,
        random_seed=7, seed_min_distance_factor=0.30,
        qmc_power=14, validation_qmc_power=15, voxel_size_nm=2.0,
    )
    vertices, triangles, metadata = generate_variable_wire(config, tmp_path, lambda _: None)
    quality = mesh_quality(vertices, triangles)
    assert quality["is_watertight"]
    assert quality["volume_nm3"] > 0
    assert np.all(vertices >= 0.0) and np.all(vertices <= 100.0)
    assert metadata["topology"]["has_three_axis_component"]
    assert abs(metadata["porosity"]["independent_qmc"] - 0.92) <= 0.02
    assert abs(metadata["porosity"]["closed_mesh_volume"] - 0.92) <= 0.02
    stats = metadata["geometry"]["diameter_statistics_nm"]
    assert 8.0 <= stats["min"] < stats["max"] <= 10.0
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["voxel_mesh"]["is_watertight"]
    with np.load(tmp_path / "wire_geometry.npz", allow_pickle=False) as saved:
        values = saved["diameters_nm"]
        expected = np.random.default_rng(np.random.SeedSequence([7, 0xD1A])).uniform(8.0, 10.0, len(values))
        np.testing.assert_array_equal(values, expected)


def test_subvoxel_diameter_is_rejected_before_generation(tmp_path):
    with pytest.raises(ValueError, match="体素间距"):
        generate_variable_wire(dict(rod_diameter_min_nm=4, rod_diameter_max_nm=6, voxel_size_nm=3), tmp_path, print)
