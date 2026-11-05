"""Variable-diameter periodic nanowires and a clipped solid voxel union.

The original command-line display generator is deliberately left untouched.
Every canonical edge receives its diameter before selection, and that same
diameter is used by both Sobol estimators and the solid voxelizer.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from math import ceil, floor, pi
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from voronoi_aerogel.geometry import build_periodic_voronoi, generate_periodic_seeds
from voronoi_aerogel.mesh import mesh_from_occupancy, mesh_quality
from voronoi_nanowire.geometry import PeriodicEdge, build_edge_graph, extract_periodic_edges
from voronoi_nanowire.selection import CapsuleCoverageEstimator, select_connected_edges


def _image_shifts(
    start: np.ndarray, end: np.ndarray, radius_nm: float, cube_size_nm: float
) -> Iterable[tuple[int, int, int]]:
    lower = np.minimum(start, end) - radius_nm
    upper = np.maximum(start, end) + radius_nm
    return product(
        *(
            range(
                int(ceil(-upper[axis] / cube_size_nm)),
                int(floor((cube_size_nm - lower[axis]) / cube_size_nm)) + 1,
            )
            for axis in range(3)
        )
    )


def _distance_squared(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    parameter = np.clip(((points - start) @ direction) / np.dot(direction, direction), 0.0, 1.0)
    delta = points - (start + parameter[:, None] * direction)
    return np.einsum("ij,ij->i", delta, delta)


class VariableCapsuleCoverageEstimator(CapsuleCoverageEstimator):
    """The core sparse union estimator with an immutable radius per edge."""

    def __init__(
        self,
        edges: Sequence[PeriodicEdge],
        *,
        diameters_nm: Sequence[float],
        cube_size_nm: float,
        qmc_power: int = 18,
        seed: int = 0,
    ) -> None:
        diameters = np.asarray(diameters_nm, dtype=float)
        if diameters.shape != (len(edges),) or not len(edges):
            raise ValueError("diameters_nm must have one value per edge")
        if not np.isfinite(diameters).all() or np.any(diameters <= 0.0):
            raise ValueError("diameters_nm must contain finite positive values")
        super().__init__(
            edges,
            cube_size_nm=cube_size_nm,
            rod_diameter_nm=float(diameters.max()),
            qmc_power=qmc_power,
            seed=seed,
        )
        self.radii_nm = {
            int(edge.id): float(diameter / 2.0)
            for edge, diameter in zip(edges, diameters)
        }

    def edge_indices(self, edge_id: int) -> np.ndarray:
        edge_id = int(edge_id)
        cached = self._cache.get(edge_id)
        if cached is not None:
            return cached
        edge = self._edges_by_id[edge_id]
        start, end = self._segment(edge)
        radius = self.radii_nm[edge_id] + self.tolerance_nm
        query_radius = 0.5 * np.linalg.norm(end - start) + radius
        chunks = []
        for shift in _image_shifts(start, end, radius, self.cube_size_nm):
            translation = np.asarray(shift, dtype=float) * self.cube_size_nm
            image_start, image_end = start + translation, end + translation
            candidates = np.asarray(
                self._point_tree.query_ball_point(0.5 * (image_start + image_end), query_radius),
                dtype=np.int64,
            )
            if len(candidates):
                inside = _distance_squared(self.points[candidates], image_start, image_end) <= radius**2
                chunks.append(candidates[inside])
        indices = (
            np.unique(np.concatenate(chunks)).astype(np.int32)
            if chunks
            else np.empty(0, dtype=np.int32)
        )
        self._cache[edge_id] = indices
        return indices

    def nominal_capsule_volume_nm3(self, edge_id: int) -> float:
        edge = self._edges_by_id[int(edge_id)]
        radius = self.radii_nm[int(edge_id)]
        start, end = self._segment(edge)
        return float(pi * radius**2 * np.linalg.norm(end - start) + 4.0 * pi * radius**3 / 3.0)


def voxelize_periodic_capsules(
    edges: Sequence[PeriodicEdge],
    diameters_nm: Sequence[float],
    *,
    cube_size_nm: float,
    voxel_size_nm: float,
) -> np.ndarray:
    """Sample the periodic capsule UNION at voxel centres in the primary cube.

    Local capsule bounding boxes avoid a full-grid distance array per rod.
    A slab limit also bounds the temporary arrays for long diagonal rods.
    """

    diameters = np.asarray(diameters_nm, dtype=float)
    if diameters.shape != (len(edges),) or not np.isfinite(diameters).all() or np.any(diameters <= 0):
        raise ValueError("diameters_nm must have one finite positive value per edge")
    if not np.isfinite([cube_size_nm, voxel_size_nm]).all() or min(cube_size_nm, voxel_size_nm) <= 0:
        raise ValueError("cube and voxel sizes must be finite and positive")
    count = int(ceil(cube_size_nm / voxel_size_nm))
    if count > 512:
        raise ValueError("体素网格超过 512^3；请增大体素间距或减小计算域。")
    spacing = cube_size_nm / count
    occupied = np.zeros((count, count, count), dtype=bool)
    tolerance = max(cube_size_nm * 1e-12, np.finfo(float).eps * 100)
    for edge, diameter in zip(edges, diameters):
        start, end = np.asarray(edge.start_nm), np.asarray(edge.end_nm)
        radius = float(diameter / 2.0)
        for shift in _image_shifts(start, end, radius + tolerance, cube_size_nm):
            translation = np.asarray(shift) * cube_size_nm
            image_start, image_end = start + translation, end + translation
            lower = np.minimum(image_start, image_end) - radius - tolerance
            upper = np.maximum(image_start, image_end) + radius + tolerance
            first = np.maximum(np.ceil(lower / spacing - 0.5).astype(int), 0)
            last = np.minimum(np.floor(upper / spacing - 0.5).astype(int), count - 1)
            if np.any(last < first):
                continue
            yz_count = int((last[1] - first[1] + 1) * (last[2] - first[2] + 1))
            slab_width = max(1, 250_000 // yz_count)
            for x_first in range(int(first[0]), int(last[0]) + 1, slab_width):
                x_last = min(x_first + slab_width - 1, int(last[0]))
                axes = (
                    (np.arange(x_first, x_last + 1) + 0.5) * spacing,
                    (np.arange(first[1], last[1] + 1) + 0.5) * spacing,
                    (np.arange(first[2], last[2] + 1) + 0.5) * spacing,
                )
                points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
                inside = _distance_squared(points, image_start, image_end) <= (radius + tolerance)**2
                slices = (
                    slice(x_first, x_last + 1),
                    slice(int(first[1]), int(last[1]) + 1),
                    slice(int(first[2]), int(last[2]) + 1),
                )
                occupied[slices] |= inside.reshape(tuple(len(axis) for axis in axes))
    return occupied


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def generate_variable_wire(
    config: dict,
    output_dir: Path,
    report: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select variable rods, verify porosity, and return a closed solid mesh.

    Physical geometry is an exact capsule union for Sobol estimation.  The
    exported surface is a finite-resolution voxel approximation of that union;
    its porosity is checked separately and never substituted for the QMC value.
    """

    size = float(config.get("cube_size_nm", 1000.0))
    minimum = float(config.get("rod_diameter_min_nm", config.get("rod_diameter_nm", 20.0)))
    maximum = float(config.get("rod_diameter_max_nm", config.get("rod_diameter_nm", 20.0)))
    spacing = float(config.get("voxel_size_nm", 5.0))
    target = float(config.get("target_porosity", 0.95))
    tolerance = float(config.get("porosity_tolerance", 0.005))
    seed_count = int(config.get("num_seeds", 400))
    random_seed = int(config.get("random_seed", 42))
    separation = float(config.get("seed_min_distance_factor", 0.30))
    qmc_power = int(config.get("qmc_power", 18))
    validation_power = int(config.get("validation_qmc_power", 19))
    if not np.isfinite([size, minimum, maximum, spacing, target, tolerance, separation]).all():
        raise ValueError("所有尺寸和孔隙率参数必须为有限数值。")
    if not 0.0 < minimum <= maximum < size:
        raise ValueError("纳米线直径必须满足 0 < 最小直径 ≤ 最大直径 < 计算域边长。")
    if not 0.0 < target < 1.0 or not 0.0 <= tolerance < 1.0:
        raise ValueError("孔隙率应在 0 和 1 之间，容差应在 [0,1) 内。")
    if seed_count < 5 or not 0 <= random_seed < 2**32 or not 0.0 <= separation < 1.0:
        raise ValueError("种子数量至少为 5，随机种子应为非负 32 位整数，种子间距因子应在 [0,1) 内。")
    if not 12 <= qmc_power <= 24 or not 12 <= validation_power <= 24:
        raise ValueError("Sobol 采样指数必须在 12 至 24 之间。")
    if spacing <= 0.0 or spacing > minimum / 2.0:
        raise ValueError("体素间距必须为正数，且不大于最小纳米线直径的一半。")
    grid_count = int(ceil(size / spacing))
    if grid_count > 512:
        raise ValueError("体素网格超过 512^3；请增大体素间距或减小计算域。")
    effective_spacing = size / grid_count

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    used_config = dict(config)
    used_config.update(
        cube_size_nm=size,
        rod_diameter_min_nm=minimum,
        rod_diameter_max_nm=maximum,
        voxel_size_nm=spacing,
        target_porosity=target,
        porosity_tolerance=tolerance,
        num_seeds=seed_count,
        random_seed=random_seed,
        seed_min_distance_factor=separation,
        qmc_power=qmc_power,
        validation_qmc_power=validation_power,
        output_dir=str(output_dir),
    )
    _write_json(output_dir / "config_used.json", used_config)
    report(f"生成 {seed_count} 个周期 Voronoi 种子")
    seeds = generate_periodic_seeds(seed_count, size, min_distance_factor=separation, random_seed=random_seed)
    tessellation = build_periodic_voronoi(seeds, size)
    edges = extract_periodic_edges(tessellation.faces, size)
    radius_rng = np.random.default_rng(np.random.SeedSequence([random_seed, 0xD1A]))
    diameters = (
        np.full(len(edges), minimum)
        if minimum == maximum
        else radius_rng.uniform(minimum, maximum, len(edges))
    )
    report(f"为 {len(edges)} 条完整逻辑边分配直径 {minimum:g}–{maximum:g} nm")
    estimator = VariableCapsuleCoverageEstimator(
        edges, diameters_nm=diameters, cube_size_nm=size, qmc_power=qmc_power, seed=random_seed,
    )
    selection = select_connected_edges(
        edges, build_edge_graph(edges), estimator,
        target_porosity=target, tolerance=tolerance, random_seed=random_seed,
    )
    if (
        selection.status != "REACHED" or not selection.within_tolerance
        or not selection.winding.connected or not selection.winding.has_three_axis_component
    ):
        details = {
            "status": selection.status,
            "target_porosity": target,
            "control_porosity": selection.achieved_porosity,
            "attainable_range": asdict(selection.attainable_range),
            "scope": "current diameter assignment and connected backbone growth strategy",
        }
        _write_json(output_dir / "feasibility.json", details)
        raise ValueError(
            f"当前直径范围、种子数和周期骨架策略不能在容差内达到目标孔隙率。"
            f"估计范围 {100 * selection.attainable_range.min_porosity:.2f}%–"
            f"{100 * selection.attainable_range.max_porosity:.2f}%；详见 feasibility.json。"
        )
    report(f"已选择 {len(selection.selected_edge_ids)} 根纳米棒，控制孔隙率 {100 * selection.achieved_porosity:.4f}%")
    selected_ids = np.asarray(selection.selected_edge_ids, dtype=np.int64)
    selected_edges = [edges[int(index)] for index in selected_ids]
    selected_diameters = diameters[selected_ids]
    independent = VariableCapsuleCoverageEstimator(
        selected_edges, diameters_nm=selected_diameters, cube_size_nm=size,
        qmc_power=validation_power, seed=(random_seed + 104729) % 2**32,
    )
    independent_porosity = 1.0 - independent.fraction(selection.selected_edge_ids)
    if abs(independent_porosity - target) > tolerance:
        _write_json(output_dir / "feasibility.json", {
            "status": "INDEPENDENT_VALIDATION_FAILURE", "target": target,
            "independent_qmc": independent_porosity, "tolerance": tolerance,
        })
        raise ValueError("独立采样孔隙率超出容差；请增加 Sobol 采样指数后重新生成。")
    report(f"独立孔隙率 {100 * independent_porosity:.4f}%；计算 {grid_count}^3 体素实体并集")
    occupied = voxelize_periodic_capsules(
        selected_edges, selected_diameters, cube_size_nm=size, voxel_size_nm=spacing,
    )
    voxel_porosity = 1.0 - float(np.count_nonzero(occupied)) / occupied.size
    report("提取封闭表面、裁剪胞元边界并检查水密性")
    mesh = mesh_from_occupancy(occupied, cube_size_nm=size)
    del occupied
    quality = mesh_quality(mesh)
    if not quality["is_watertight"]:
        raise ValueError("体素表面未通过水密性检查；请调整体素间距后重试。")
    mesh_porosity = 1.0 - float(quality["volume_nm3"]) / size**3
    if abs(mesh_porosity - target) > tolerance:
        _write_json(output_dir / "feasibility.json", {
            "status": "MESH_RESOLUTION_FAILURE", "target": target,
            "independent_qmc": independent_porosity, "voxel_count": voxel_porosity,
            "closed_mesh_volume": mesh_porosity, "tolerance": tolerance,
            "effective_spacing_nm": effective_spacing,
        })
        raise ValueError(
            f"网格孔隙率 {100 * mesh_porosity:.4f}% 超出目标容差；"
            "请减小体素间距以提高实体分辨率。"
        )
    np.savez_compressed(
        output_dir / "wire_geometry.npz",
        seeds_nm=seeds,
        edge_ids=np.asarray([edge.id for edge in edges], dtype=np.int64),
        start_nm=np.asarray([edge.start_nm for edge in edges]),
        end_nm=np.asarray([edge.end_nm for edge in edges]),
        shift=np.asarray([edge.shift for edge in edges], dtype=np.int64),
        diameters_nm=diameters,
        selected_edge_ids=selected_ids,
        config_json=json.dumps(used_config, ensure_ascii=False, sort_keys=True),
    )
    lengths = np.asarray([edge.length_nm for edge in selected_edges])
    warnings = []
    if minimum / effective_spacing < 4:
        warnings.append("最小直径少于 4 个体素；定量计算前请进一步细化并做网格收敛检查。")
    warnings.append("三向贯通针对周期逻辑骨架；单胞裁剪后可含多个封闭片段，体素边界形状为有限分辨率近似。")
    metadata = {
        "schema": "aerogel-studio-variable-wire-v1",
        "model_type": "periodic_voronoi_variable_nanowire",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_unit": "nm",
        "config": used_config,
        "assumptions": {
            "rod_volume_model": "periodic union of variable-radius capsules",
            "diameter_distribution": "uniform on canonical logical edges before selection",
            "whole_logical_edges_only": True,
            "solid_mesh": "voxel union clipped and closed at primary cube boundary",
            "nominal_diameters_are_not_mesh_exact": True,
        },
        "geometry": {
            "seed_count": len(seeds),
            "logical_face_count": len(tessellation.faces),
            "candidate_edge_count": len(edges),
            "selected_edge_count": len(selected_ids),
            "selected_edge_ids": selected_ids.tolist(),
            "backbone_edge_count": len(selection.backbone_edge_ids),
            "diameter_range_requested_nm": [minimum, maximum],
            "diameter_statistics_nm": _statistics(selected_diameters),
            "length_statistics_nm": _statistics(lengths),
            "selected_logical_length_nm": float(lengths.sum()),
        },
        "selection": {"status": selection.status, "attainable_range": asdict(selection.attainable_range)},
        "porosity": {
            "target": target,
            "tolerance": tolerance,
            "control_qmc": selection.achieved_porosity,
            "control_sample_count": estimator.sample_count,
            "independent_qmc": independent_porosity,
            "independent_sample_count": independent.sample_count,
            "voxel_count": voxel_porosity,
            "closed_mesh_volume": mesh_porosity,
        },
        "topology": {
            "connected": selection.winding.connected,
            "has_three_axis_component": selection.winding.has_three_axis_component,
            "maximum_rank": selection.winding.maximum_rank,
        },
        "voxel_mesh": {
            "requested_spacing_nm": spacing,
            "effective_spacing_nm": effective_spacing,
            "grid_shape": [grid_count] * 3,
            **quality,
        },
        "warnings": warnings,
        "files": {"geometry": "wire_geometry.npz", "config": "config_used.json", "metadata": "metadata.json"},
    }
    _write_json(output_dir / "metadata.json", metadata)
    report(f"实体网格检查通过：{len(mesh.faces)} 个三角面；网格孔隙率 {100 * mesh_porosity:.4f}%")
    return mesh.vertices_nm, mesh.faces, metadata


__all__ = ["VariableCapsuleCoverageEstimator", "voxelize_periodic_capsules", "generate_variable_wire"]
