"""端到端周期气凝胶生成流水线。

本模块把周期 Voronoi 几何、固/孔双连续拓扑、并集孔隙率控制、
体素实体化以及 COMSOL/Lumerical 文件导出串联为一个可复现流程。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import ceil, pi
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ModelConfig
from .exporters import (
    export_shell_bdf,
    save_preview,
    write_lumerical_script,
    write_metadata,
    write_readmes,
)
from .geometry import build_periodic_voronoi, generate_periodic_seeds
from .mesh import (
    effective_voxel_size_nm,
    mesh_from_occupancy,
    mesh_quality,
    voxelize_periodic_sheets,
    write_binary_stl,
)
from .selection import (
    SelectionResult,
    SobolCoverageEstimator,
    coerce_face_graph,
    construct_dual_backbone,
    select_by_connected_frontier,
)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """一次成功生成任务的核心结果与输出位置。"""

    output_dir: Path
    qmc_porosity: float
    validation_porosity: float
    voxel_porosity: float
    mesh_porosity: float
    selected_face_count: int
    metadata_path: Path


class GenerationError(RuntimeError):
    """当几何、孔隙率和双连续拓扑约束无法同时满足时抛出。"""


def _automatic_seed_count(config: ModelConfig) -> int:
    """同时根据片径尺度和目标固相体积估算 Voronoi 母点数。"""

    length = config.cube_size_nm
    diameter = config.target_diameter_nm
    # 0.57 是当前轻度抑制随机点集下“中位等效面径/平均点间距”的经验比例。
    size_count = int(round((0.57 * length / diameter) ** 3))
    one_sheet_volume = 0.25 * pi * diameter**2 * config.thickness_nm
    required_sheets = config.target_solid_fraction * length**3 / one_sheet_volume
    # 三维周期 Voronoi 中每个母点约对应 7.8 个不重复公共面；其中约四分之一
    # 落入有限片径窗口。这里只做容量预估，实际比例会在生成后重新测量。
    preferred_faces_per_seed = 7.8 * 0.25
    capacity_count = int(
        ceil(
            required_sheets
            * max(config.min_in_range_fraction, 0.25)
            / preferred_faces_per_seed
        )
    )
    return max(64, size_count, capacity_count)


def _winding_dict(report: Any) -> dict[str, Any]:
    """把周期图绕回报告转换为可写入 JSON 的普通类型。"""

    return {
        "connected": bool(report.connected),
        "component_count": len(report.components),
        "maximum_rank": int(report.maximum_rank),
        "components": [
            {
                "node_count": len(component.nodes),
                "rank": int(component.rank),
                "axis_mask": list(component.axis_mask),
                "basis": [list(vector) for vector in component.basis],
            }
            for component in report.components
        ],
    }


def _face_statistics(faces: list[Any], ids: list[int] | tuple[int, ...]) -> dict[str, Any]:
    """统计选中逻辑面的等效片径和中面面积。"""

    if not ids:
        return {"count": 0}
    diameters = np.asarray(
        [faces[index].equivalent_diameter_nm for index in ids], dtype=float
    )
    areas = np.asarray([faces[index].area_nm2 for index in ids], dtype=float)
    return {
        "count": len(ids),
        "diameter_nm": {
            "min": float(diameters.min()),
            "q10": float(np.quantile(diameters, 0.10)),
            "median": float(np.median(diameters)),
            "q90": float(np.quantile(diameters, 0.90)),
            "max": float(diameters.max()),
            "mean": float(diameters.mean()),
            "area_weighted_mean": float(np.average(diameters, weights=areas)),
        },
        "total_midsurface_area_nm2": float(areas.sum()),
    }


def _selection_acceptable(result: SelectionResult, config: ModelConfig) -> bool:
    """统一判定孔隙率、片径比例和双相三向贯通是否全部达标。"""

    return (
        result.status == "REACHED"
        and result.within_tolerance
        and result.solid_winding.connected
        and result.solid_winding.has_three_axis_component
        and result.pore_winding.connected
        and result.pore_winding.has_three_axis_component
        and result.diameter_statistics_nm.get(
            "in_preferred_range_fraction", 1.0
        )
        >= config.min_in_range_fraction - 1e-12
    )


def _next_seed_count(count: int, status: str) -> int:
    # 增加母点会提供更多候选面，也通常能减小三向绕回骨架的最低固相体积。
    factor = 1.30 if "INFEASIBLE" in status else 1.18
    return max(count + 1, int(ceil(count * factor)))


def _save_geometry_data(
    path: Path,
    seeds_nm: np.ndarray,
    faces: list[Any],
    selection: SelectionResult,
) -> None:
    """保存复现几何所需的母点、面属性和变长多边形数组。"""

    selected = list(selection.selected_face_ids)
    polygons = [faces[index].vertices_nm for index in selected]
    # 所有面顶点拼为一个数组，offsets 给出每个变长多边形的起止位置。
    offsets = np.zeros(len(polygons) + 1, dtype=np.int64)
    if polygons:
        offsets[1:] = np.cumsum([len(polygon) for polygon in polygons])
        vertices = np.concatenate(polygons, axis=0)
    else:
        vertices = np.empty((0, 3), dtype=float)
    np.savez_compressed(
        path,
        seeds_nm=seeds_nm,
        selected_face_ids=np.asarray(selected, dtype=np.int64),
        protected_pore_face_ids=np.asarray(
            selection.protected_pore_face_ids, dtype=np.int64
        ),
        selected_cell_a=np.asarray([faces[index].cell_a for index in selected]),
        selected_cell_b=np.asarray([faces[index].cell_b for index in selected]),
        selected_shift=np.asarray([faces[index].shift for index in selected]),
        selected_area_nm2=np.asarray([faces[index].area_nm2 for index in selected]),
        selected_equivalent_diameter_nm=np.asarray(
            [faces[index].equivalent_diameter_nm for index in selected]
        ),
        selected_vertex_offsets=offsets,
        selected_vertices_nm=vertices,
    )


def _solver_size_estimates(cube_size_nm: float, thickness_nm: float) -> dict[str, Any]:
    """给出不同厚度分辨率下的 FDTD 基础网格量级参考。"""

    spacings = sorted(
        {thickness_nm, thickness_nm / 2.0, thickness_nm / 3.0},
        reverse=True,
    )
    fdtd = []
    for spacing in spacings:
        cells_axis = int(ceil(cube_size_nm / spacing))
        cells = cells_axis**3
        fdtd.append(
            {
                "nominal_spacing_nm": spacing,
                "cells_per_axis_before_pml": cells_axis,
                "base_cells_before_pml": cells,
                "rough_complex_field_storage_GB": 96.0 * cells / 1e9,
            }
        )
    return {
        "fdtd_uniform_grid_reference": fdtd,
        "note": (
            "96 bytes/cell is only a field-storage scale; actual memory also "
            "depends on materials, PML, monitors and solver implementation."
        ),
    }


def generate_model(
    config: ModelConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> GenerationResult:
    """生成周期片层网络及全部仿真、可视化和验证文件。"""

    # 所有内部长度统一使用 nm；只有 BDF 导出阶段转换为 µm。
    config.validate()
    report = progress or (lambda _message: None)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.write_json(output_dir / "config_used.json")

    seed_count = config.num_seeds or _automatic_seed_count(config)
    selection: SelectionResult | None = None
    geometry = None
    estimator = None
    attempt_records: list[dict[str, Any]] = []

    for attempt in range(config.max_generation_attempts):
        attempt_seed = config.random_seed + 104_729 * attempt
        report(
            f"Attempt {attempt + 1}/{config.max_generation_attempts}: "
            f"{seed_count} periodic seeds"
        )
        # 每次失败重试使用确定性的新随机种子，因此结果仍可完全复现。
        seeds = generate_periodic_seeds(
            seed_count,
            config.cube_size_nm,
            min_distance_factor=config.seed_min_distance_factor,
            random_seed=attempt_seed,
        )
        geometry = build_periodic_voronoi(seeds, config.cube_size_nm)
        faces = geometry.faces
        report(f"  Voronoi topology: {len(faces)} logical faces")
        # 过小面积或极短边会产生病态壳单元，保留为开放孔面但不允许选作固相。
        allowed = {
            face.id
            for face in faces
            if face.area_nm2 >= config.min_face_area_nm2
            and face.min_edge_nm >= config.min_face_edge_nm
        }
        face_graph = coerce_face_graph(
            geometry.face_adjacency,
            face_count=len(faces),
        )

        try:
            # 先构造互不冲突的孔相保护网络与固相三向绕回骨架。
            backbone = construct_dual_backbone(
                faces,
                face_graph,
                cube_size_nm=config.cube_size_nm,
                thickness_nm=config.thickness_nm,
                target_porosity=config.target_porosity,
                diameter_range_nm=(
                    config.diameter_min_nm,
                    config.diameter_max_nm,
                ),
                allowed_face_ids=allowed,
                offset_bound=config.winding_offset_bound,
                random_seed=attempt_seed,
                max_pore_roots=1 if attempt == 0 else 2,
                max_solid_roots=3,
            )
            report(
                "  Backbones: "
                f"{len(backbone.solid_face_ids)} solid faces, "
                f"{len(backbone.protected_face_ids)} protected pore faces"
            )
            # 对有限厚度片层的物理并集采样；交叠区只计一次体积。
            estimator = SobolCoverageEstimator(
                faces,
                cube_size_nm=config.cube_size_nm,
                thickness_nm=config.thickness_nm,
                qmc_power=config.qmc_power,
                seed=attempt_seed,
                geometric_tolerance_nm=max(1e-8 * config.cube_size_nm, 1e-6),
            )
            # 只从已有固相的邻接前沿生长，保证新增片层始终属于同一固相网络。
            selection = select_by_connected_frontier(
                faces,
                face_graph,
                backbone,
                estimator,
                target_porosity=config.target_porosity,
                tolerance=config.porosity_tolerance,
                random_seed=attempt_seed,
                preferred_diameter_range_nm=(
                    config.diameter_min_nm,
                    config.diameter_max_nm,
                ),
                min_preferred_fraction=config.min_in_range_fraction,
                max_selected_faces=config.max_selected_faces,
                evaluate_maximum=True,
            )
            report(
                f"  Selection: {selection.status}, "
                f"porosity={100.0 * selection.achieved_porosity:.4f}%, "
                f"faces={len(selection.selected_face_ids)}"
            )
            attempt_records.append(
                {
                    "attempt": attempt + 1,
                    "random_seed": attempt_seed,
                    "num_seeds": seed_count,
                    "logical_face_count": len(faces),
                    "eligible_face_count": len(allowed),
                    "status": selection.status,
                    "achieved_porosity": selection.achieved_porosity,
                    "attainable_porosity": [
                        selection.attainable_range.min_porosity,
                        selection.attainable_range.max_porosity,
                    ],
                    "selected_face_count": len(selection.selected_face_ids),
                    "in_range_fraction": selection.diameter_statistics_nm.get(
                        "in_preferred_range_fraction"
                    ),
                }
            )
        except (RuntimeError, ValueError) as exc:
            attempt_records.append(
                {
                    "attempt": attempt + 1,
                    "random_seed": attempt_seed,
                    "num_seeds": seed_count,
                    "logical_face_count": len(faces),
                    "eligible_face_count": len(allowed),
                    "status": "BACKBONE_FAILURE",
                    "error": str(exc),
                }
            )
            selection = None

        if selection is not None and _selection_acceptable(selection, config):
            break
        if config.num_seeds is None:
            status = selection.status if selection is not None else "BACKBONE_FAILURE"
            seed_count = _next_seed_count(seed_count, status)

    if geometry is None or selection is None or estimator is None:
        feasibility_path = write_metadata(
            output_dir / "feasibility.json",
            {
                "status": "FAILED",
                "config": asdict(config),
                "attempts": attempt_records,
            },
        )
        raise GenerationError(
            "Could not construct disjoint connected periodic solid and pore "
            f"networks; details: {feasibility_path}"
        )
    if not _selection_acceptable(selection, config):
        feasibility_path = write_metadata(
            output_dir / "feasibility.json",
            {
                "status": selection.status,
                "config": asdict(config),
                "attempts": attempt_records,
                "attainable_porosity": asdict(selection.attainable_range),
            },
        )
        raise GenerationError(
            "The requested porosity/size/topology combination was not reached; "
            f"details: {feasibility_path}"
        )

    # 在昂贵的体素化之前落盘预检结果，便于判断给定参数是否可行。
    write_metadata(
        output_dir / "feasibility.json",
        {
            "status": "PASSED",
            "config": asdict(config),
            "attempts": attempt_records,
            "attainable_porosity": asdict(selection.attainable_range),
        },
    )

    faces = geometry.faces
    selected_ids = list(selection.selected_face_ids)
    # 使用不同扰动种子的独立 Sobol 点集复核，避免仅对控制点集“拟合”。
    validation_estimator = SobolCoverageEstimator(
        faces,
        cube_size_nm=config.cube_size_nm,
        thickness_nm=config.thickness_nm,
        qmc_power=config.validation_qmc_power,
        seed=config.random_seed + 1_000_003,
        geometric_tolerance_nm=max(1e-8 * config.cube_size_nm, 1e-6),
    )
    validation_count = validation_estimator.union_count(selected_ids)
    validation_porosity = 1.0 - validation_estimator.solid_fraction_from_count(
        validation_count
    )
    report(
        f"Independent QMC validation: {100.0 * validation_porosity:.4f}% porosity"
    )

    report("Voxelising periodic finite-thickness union")
    # STL 后端对全部厚片先取体素并集，再提取一个封闭外表面；不会简单拼接
    # 相互重叠的棱柱，也不会重复计算交叠体积。
    occupancy = voxelize_periodic_sheets(
        faces,
        selected_ids,
        config.cube_size_nm,
        config.thickness_nm,
        config.voxel_size_nm,
    )
    voxel_porosity = 1.0 - float(np.mean(occupancy))
    solid_mesh = mesh_from_occupancy(
        occupancy,
        cube_size_nm=config.cube_size_nm,
    )
    solid_quality = mesh_quality(solid_mesh)
    marching_method = "lewiner"
    # Lewiner 在少数数字拓扑鞍点处可能产生非流形边；经典 Lorensen 路径作为
    # 可验证的后备方案，只有水密且每条边恰有两个邻面时才接受最终 STL。
    if not bool(solid_quality["is_watertight"]):
        solid_mesh = mesh_from_occupancy(
            occupancy,
            cube_size_nm=config.cube_size_nm,
            method="lorensen",
        )
        solid_quality = mesh_quality(solid_mesh)
        marching_method = "lorensen"
    del occupancy
    if not bool(solid_quality["is_watertight"]):
        raise GenerationError(
            "The final voxel-union STL surface is not watertight; inspect mesh diagnostics"
        )
    mesh_porosity = 1.0 - float(solid_quality["volume_nm3"]) / (
        config.cube_size_nm**3
    )
    report(
        f"Closed mesh: {solid_quality['n_faces']} triangles, "
        f"porosity={100.0 * mesh_porosity:.4f}%"
    )

    stl_path = write_binary_stl(output_dir / "aerogel_solid_nm.stl", solid_mesh)
    pairs_path = output_dir / "periodic_node_pairs.json"
    # 壳模型保留精确中面和 5 nm 厚度属性，适合避免 COMSOL 体网格在厚度方向爆炸。
    shell = export_shell_bdf(
        faces,
        selected_ids,
        config.cube_size_nm,
        config.thickness_nm,
        output_dir / "aerogel_shell.bdf",
        periodic_pairs_path=pairs_path,
    )
    pair_data = shell["periodic_pairs"]
    unmatched = {
        axis: {
            "minus": len(data["unmatched_minus_node_ids"]),
            "plus": len(data["unmatched_plus_node_ids"]),
        }
        for axis, data in pair_data["axes"].items()
    }
    # 三组相对边界必须严格一一配对；否则几何虽看似周期，求解器约束仍会失败。
    if any(values["minus"] or values["plus"] for values in unmatched.values()):
        raise GenerationError(
            "Opposite shell boundaries did not produce one-to-one periodic node pairs"
        )

    preview_path = save_preview(
        output_dir / "preview.png",
        shell,
        cube_size_nm=config.cube_size_nm,
        max_faces=config.preview_face_limit,
        title=(
            f"Periodic alumina-aerogel sheets; porosity "
            f"{100.0 * validation_porosity:.2f}%"
        ),
    )
    lsf_path = write_lumerical_script(
        output_dir / "import_lumerical.lsf",
        stl_path.name,
        config.cube_size_nm,
    )
    readmes = write_readmes(output_dir, stl_filename=stl_path.name)
    data_path = output_dir / "periodic_geometry.npz"
    _save_geometry_data(data_path, geometry.seeds_nm, faces, selection)

    selected_stats = _face_statistics(faces, selected_ids)
    lower, upper = config.diameter_min_nm, config.diameter_max_nm
    in_range_count = sum(
        lower <= faces[index].equivalent_diameter_nm <= upper
        for index in selected_ids
    )
    selected_stats["preferred_range_nm"] = [lower, upper]
    selected_stats["in_preferred_range_count"] = in_range_count
    selected_stats["in_preferred_range_fraction"] = (
        in_range_count / len(selected_ids)
    )

    solid_fraction = 1.0 - selection.achieved_porosity
    reference_standard_error = (
        solid_fraction * max(1.0 - solid_fraction, 0.0) / estimator.sample_count
    ) ** 0.5
    # 汇总控制、独立验证、体素计数和封闭网格四种孔隙率，显式暴露离散误差。
    metadata: dict[str, Any] = {
        "schema": "periodic-voronoi-aerogel-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "assumptions": {
            "default_porosity_used": config.target_porosity,
            "geometry_periodicity": ["x", "y", "z"],
            "sheet_thickness_changed_by_generator": False,
            "porosity_definition": (
                "1 - physical union volume of finite-thickness sheets / cube volume"
            ),
            "fdtd_boundaries": {
                "x": "Periodic",
                "y": "Periodic",
                "z": "PML for finite-slab transmission",
            },
        },
        "generation_attempts": attempt_records,
        "geometry": {
            "num_seeds": len(geometry.seeds_nm),
            "logical_face_count": len(faces),
            "selected_face_count": len(selected_ids),
            "protected_open_pore_face_count": len(
                selection.protected_pore_face_ids
            ),
            "sheet_statistics": selected_stats,
        },
        "attainable_porosity_preflight": asdict(selection.attainable_range),
        "attainable_porosity_preflight_scope": (
            "Physical-union range after topology and mesh-quality exclusions. "
            "It is an optimistic dense-end bound; the configured preferred-"
            "diameter fraction is checked separately on the selected target state."
        ),
        "porosity": {
            "target": config.target_porosity,
            "tolerance": config.porosity_tolerance,
            "control_qmc": selection.achieved_porosity,
            "control_qmc_sample_count": estimator.sample_count,
            "independent_qmc": validation_porosity,
            "independent_qmc_sample_count": validation_estimator.sample_count,
            "voxel_count": voxel_porosity,
            "closed_mesh_volume": mesh_porosity,
            "reference_95pct_binomial_half_width": 1.96 * reference_standard_error,
        },
        "topology": {
            "solid": _winding_dict(selection.solid_winding),
            "pore": _winding_dict(selection.pore_winding),
        },
        "voxel_mesh": {
            "requested_spacing_nm": config.voxel_size_nm,
            "effective_spacing_nm": effective_voxel_size_nm(
                config.cube_size_nm, config.voxel_size_nm
            ),
            "marching_cubes_method": marching_method,
            **solid_quality,
        },
        "shell_mesh": {
            "vertices": len(shell["vertices_nm"]),
            "triangles": len(shell["triangles"]),
            "thickness_um": shell["thickness_um"],
            "periodic_pair_unmatched_counts": unmatched,
        },
        "solver_size_estimates": _solver_size_estimates(
            config.cube_size_nm, config.thickness_nm
        ),
        "files": {
            "solid_stl": stl_path.name,
            "shell_bdf": Path(shell["path"]).name,
            "periodic_pairs": pairs_path.name,
            "preview": preview_path.name,
            "lumerical_script": lsf_path.name,
            "comsol_notes": readmes["comsol"].name,
            "lumerical_notes": readmes["lumerical"].name,
            "reproducibility_data": data_path.name,
        },
    }
    metadata_path = write_metadata(output_dir / "metadata.json", metadata)
    report(f"Finished: {output_dir}")
    return GenerationResult(
        output_dir=output_dir,
        qmc_porosity=selection.achieved_porosity,
        validation_porosity=validation_porosity,
        voxel_porosity=voxel_porosity,
        mesh_porosity=mesh_porosity,
        selected_face_count=len(selected_ids),
        metadata_path=metadata_path,
    )


__all__ = ["GenerationError", "GenerationResult", "generate_model"]
