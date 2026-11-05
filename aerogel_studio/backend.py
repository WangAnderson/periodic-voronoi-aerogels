"""Qt-independent generation and export adapter for Aerogel Studio.

The VTP cache always uses nm.  Preview simplification is never used for export.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import numpy as np


PREVIEW_TRIANGLE_LIMIT = 600_000
_UNIT_SCALE = {"nm": 1.0, "um": 1.0e-3, "mm": 1.0e-6}
_FORMATS = {"stl", "obj", "ply", "step"}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _export_options(formats: list[str], unit: str) -> list[str]:
    selected = list(dict.fromkeys(str(item).lower().lstrip(".") for item in formats))
    unsupported = set(selected) - _FORMATS
    if unsupported:
        raise ValueError(f"不支持的导出格式：{', '.join(sorted(unsupported))}")
    if unit not in _UNIT_SCALE:
        raise ValueError("坐标单位必须为 nm、um 或 mm")
    return selected


def _write_exports(
    mesh: Any,
    result: dict[str, Any],
    destination: Path,
    formats: list[str],
    unit: str,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    if not formats:
        return {"exports": {}, "unit": unit, "coordinate_units_path": None}

    scaled = mesh.copy(deep=True)
    scaled.points *= _UNIT_SCALE[unit]
    stem = "sheet_solid" if result["model"] == "sheet" else "nanowire_solid"
    exported: dict[str, str] = {}
    unit_records: dict[str, Any] = {}
    for kind in formats:
        path = destination / f"{stem}_{unit}.{kind}"
        if kind == "step":
            from .cad_export import write_faceted_step

            triangles = scaled.faces.reshape((-1, 4))[:, 1:]
            write_faceted_step(path, scaled.points, triangles, unit=unit)
        else:
            scaled.save(path, binary=kind != "obj")
        exported[kind] = str(path.resolve())
        unit_records[path.name] = {
            "format": kind,
            "coordinate_unit": unit,
            "scale_from_nm": _UNIT_SCALE[unit],
            "source_result": result["result_path"],
            "mesh_purpose": "solid_mesh",
        }
        if kind == "stl":
            from .cad_export import write_lumerical_import

            script = destination / f"import_{stem}_{unit}.lsf"
            write_lumerical_import(
                script,
                path.name,
                float(result["config"]["cube_size_nm"]),
                unit,
            )
            exported["lsf"] = str(script.resolve())

    sidecar = destination / "coordinate_units.json"
    records = (
        json.loads(sidecar.read_text(encoding="utf-8"))
        if sidecar.is_file()
        else {"schema": "aerogel-studio-coordinate-units-v1", "exports": {}}
    )
    records["exports"].update(unit_records)
    records["note"] = (
        "VTP 缓存使用 nm；导出坐标按各文件记录的单位缩放。"
        "STL、OBJ、PLY 不提供通用可靠的长度单位字段，导入时请手动选择单位。"
        "STEP 为三角面片组成的实体，不是解析曲面的参数化 CAD。"
    )
    _write_json(sidecar, records)
    return {
        "exports": exported,
        "unit": unit,
        "coordinate_units_path": str(sidecar.resolve()),
    }


def generate(request: dict, report: Callable[[str], None]) -> dict:
    """Generate one complete model and write its completion manifest last.

    ``request`` contains ``model`` (sheet/nanowire), the complete ``config``,
    optional ``formats`` and optional export ``unit`` (nm/um/mm; default um).
    The caller gives each job a distinct ``config.output_dir``.
    """

    import pyvista as pv

    model = str(request["model"])
    if model not in {"sheet", "nanowire"}:
        raise ValueError("气凝胶类型必须为 sheet 或 nanowire")
    unit = str(request.get("unit", "um"))
    formats = _export_options(request.get("formats", ["stl"]), unit)
    config = dict(request["config"])
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    if model == "sheet":
        from voronoi_aerogel import ModelConfig, generate_model

        report("正在生成氧化铝片层气凝胶…")
        generated = generate_model(ModelConfig(**config), progress=report)
        metadata = json.loads(generated.metadata_path.read_text(encoding="utf-8"))
        porosity = metadata["porosity"]
        if abs(porosity["closed_mesh_volume"] - porosity["target"]) > porosity["tolerance"]:
            _write_json(
                output_dir / "feasibility.json",
                {
                    "status": "MESH_RESOLUTION_FAILURE",
                    "target": porosity["target"],
                    "independent_qmc": porosity["independent_qmc"],
                    "voxel_count": porosity["voxel_count"],
                    "closed_mesh_volume": porosity["closed_mesh_volume"],
                    "tolerance": porosity["tolerance"],
                    "requested_spacing_nm": metadata["config"]["voxel_size_nm"],
                },
            )
            raise ValueError(
                f"实体网格孔隙率 {100 * porosity['closed_mesh_volume']:.4f}% "
                f"超出目标 {100 * porosity['target']:.4f}% 的容差 "
                f"±{100 * porosity['tolerance']:g} 个百分点。"
                "请减小体素间距后重新生成；详见 feasibility.json。"
            )
        stl = output_dir / metadata["files"]["solid_stl"]
        full_mesh = pv.read(stl).triangulate()
    else:
        from .variable_wire import generate_variable_wire

        report("正在生成纳米线气凝胶实体…")
        vertices, triangles, metadata = generate_variable_wire(config, output_dir, report)
        faces = np.column_stack(
            (np.full(len(triangles), 3, dtype=np.int64), triangles)
        ).ravel()
        full_mesh = pv.PolyData(vertices, faces)

    report("正在保存完整实体网格…")
    mesh_path = output_dir / "studio_mesh.vtp"
    full_mesh.save(mesh_path, binary=True)
    preview_mesh = full_mesh
    if full_mesh.n_cells > PREVIEW_TRIANGLE_LIMIT:
        report("正在简化交互预览；导出仍使用完整网格…")
        reduction = 1.0 - PREVIEW_TRIANGLE_LIMIT / full_mesh.n_cells
        preview_mesh = full_mesh.decimate_pro(
            reduction,
            preserve_topology=True,
            splitting=False,
            boundary_vertex_deletion=False,
        )
    preview_path = output_dir / "preview_mesh.vtp"
    preview_mesh.save(preview_path, binary=True)

    result_path = output_dir / "studio_result.json"
    result = {
        "schema": "aerogel-studio-result-v1",
        "model": model,
        "config": config,
        "metadata": metadata,
        "output_dir": str(output_dir),
        "result_path": str(result_path),
        "mesh_path": str(mesh_path),
        "preview_path": str(preview_path),
        "mesh_purpose": "solid_mesh",
        "counts": {
            "full_vertices": int(full_mesh.n_points),
            "full_triangles": int(full_mesh.n_cells),
            "preview_vertices": int(preview_mesh.n_points),
            "preview_triangles": int(preview_mesh.n_cells),
        },
        "preview_simplified": bool(preview_mesh.n_cells < full_mesh.n_cells),
        "units": {"internal": "nm", "export": unit},
        "exports": {},
    }
    report("正在导出所选模型文件…")
    export = _write_exports(full_mesh, result, output_dir, formats, unit)
    result["exports"] = export["exports"]
    result["coordinate_units_path"] = export["coordinate_units_path"]
    _write_json(result_path, result)
    report("模型生成完成")
    return result


def export_mesh(
    result_path: str,
    destination: str,
    formats: list[str],
    unit: str = "um",
) -> dict:
    """Export the completed job's full mesh without re-generating its geometry."""

    import pyvista as pv

    selected = _export_options(formats, unit)
    manifest = Path(result_path).expanduser().resolve()
    result = json.loads(manifest.read_text(encoding="utf-8"))
    mesh_path = Path(result["mesh_path"])
    if not mesh_path.is_file():
        mesh_path = manifest.parent / mesh_path.name
    result["result_path"] = str(manifest)
    mesh = pv.read(mesh_path)
    exported = _write_exports(
        mesh, result, Path(destination).expanduser().resolve(), selected, unit
    )
    exported["source_result"] = str(manifest)
    return exported


def process_command(
    request: dict, report: Callable[[str], None] = print
) -> dict:
    """Small protocol entry point used by the separate worker process."""

    action = request.get("action", "generate")
    if action == "generate":
        return generate(request, report)
    if action == "export":
        report("正在导出模型文件…")
        result = export_mesh(
            request["result_path"],
            request["destination"],
            request.get("formats", ["stl"]),
            request.get("unit", "um"),
        )
        report("导出完成")
        return result
    raise ValueError(f"不支持的任务类型：{action}")


__all__ = ["generate", "export_mesh", "process_command"]
