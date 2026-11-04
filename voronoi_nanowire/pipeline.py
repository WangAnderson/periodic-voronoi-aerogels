from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import numpy as np

from voronoi_aerogel.geometry import build_periodic_voronoi, generate_periodic_seeds
from voronoi_aerogel.mesh import write_binary_stl

from .config import NanowireConfig
from .exporters import (
    build_display_mesh,
    save_preview_png,
    write_interactive_html,
    write_metadata,
)
from .geometry import build_edge_graph, extract_periodic_edges, periodic_edge_segments
from .selection import CapsuleCoverageEstimator, EdgeSelectionResult, select_connected_edges


class NanowireGenerationError(RuntimeError):
    """Raised when the requested nanowire model cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class NanowireGenerationResult:
    output_dir: Path
    control_porosity: float
    validation_porosity: float
    selected_edge_count: int
    html_path: Path
    preview_path: Path
    metadata_path: Path
    stl_path: Path | None = None
    mesh_path: Path | None = None


def _winding_dict(selection: EdgeSelectionResult) -> dict[str, object]:
    report = selection.winding
    return {
        "connected": report.connected,
        "has_three_axis_component": report.has_three_axis_component,
        "maximum_rank": report.maximum_rank,
        "components": [
            {
                "node_count": len(component.nodes),
                "rank": component.rank,
                "axis_mask": list(component.axis_mask),
                "basis": [list(vector) for vector in component.basis],
            }
            for component in report.components
        ],
    }


def _failure_details(
    config: NanowireConfig,
    selection: EdgeSelectionResult,
) -> dict[str, object]:
    return {
        "status": selection.status,
        "config": config.to_dict(),
        "target_porosity": selection.target_porosity,
        "achieved_porosity": selection.achieved_porosity,
        "absolute_porosity_error": selection.absolute_porosity_error,
        "attainable_range": asdict(selection.attainable_range),
        "topology": _winding_dict(selection),
    }


def _wrapped_true_node_centres(
    selected_edges: list[object],
    cube_size_nm: float,
) -> np.ndarray:
    """Return true periodic Voronoi endpoints in the primary unit cell.

    Display segments are clipped at the six cell faces.  Those artificial cut
    points must not receive spherical junctions; only the original Voronoi
    vertices supplied here are rendered as rounded nodes.
    """

    if not selected_edges:
        return np.empty((0, 3), dtype=float)
    endpoints = np.asarray(
        [
            point
            for edge in selected_edges
            for point in (getattr(edge, "start_nm"), getattr(edge, "end_nm"))
        ],
        dtype=float,
    )
    wrapped = np.mod(endpoints, cube_size_nm)
    tolerance = max(1.0e-9 * cube_size_nm, np.finfo(float).eps * cube_size_nm * 100)
    wrapped[np.abs(wrapped) <= tolerance] = 0.0
    wrapped[np.abs(wrapped - cube_size_nm) <= tolerance] = 0.0
    return wrapped


def generate_nanowire_model(
    config: NanowireConfig,
    *,
    export_stl: bool = False,
    save_display_mesh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> NanowireGenerationResult:
    """Generate the periodic Voronoi nanowire visualisation files."""

    config.validate()
    report = progress or (lambda _message: None)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.write_json(output_dir / "config_used.json")

    report(f"Generating {config.num_seeds} periodic Voronoi seeds")
    seeds = generate_periodic_seeds(
        config.num_seeds,
        config.cube_size_nm,
        min_distance_factor=config.seed_min_distance_factor,
        random_seed=config.random_seed,
    )
    voronoi = build_periodic_voronoi(seeds, config.cube_size_nm)
    edges = extract_periodic_edges(voronoi.faces, config.cube_size_nm)
    graph = build_edge_graph(edges)
    report(
        f"Voronoi geometry: {len(voronoi.faces)} faces, "
        f"{len(edges)} unique logical edges"
    )

    estimator = CapsuleCoverageEstimator(
        edges,
        cube_size_nm=config.cube_size_nm,
        rod_diameter_nm=config.rod_diameter_nm,
        qmc_power=config.qmc_power,
        seed=config.random_seed,
    )
    report("Selecting a connected, three-axis-periodic random edge network")
    selection = select_connected_edges(
        edges,
        graph,
        estimator,
        target_porosity=config.target_porosity,
        tolerance=config.porosity_tolerance,
        random_seed=config.random_seed,
    )
    report(
        f"Selection: {selection.status}, "
        f"porosity={100.0 * selection.achieved_porosity:.4f}%, "
        f"edges={len(selection.selected_edge_ids)}"
    )

    if (
        selection.status != "REACHED"
        or not selection.within_tolerance
        or not selection.winding.connected
        or not selection.winding.has_three_axis_component
    ):
        details_path = write_metadata(
            output_dir / "feasibility.json",
            _failure_details(config, selection),
        )
        raise NanowireGenerationError(
            "The requested diameter/porosity/topology combination is not "
            f"attainable; details: {details_path}"
        )

    report("Validating porosity with an independent Sobol point set")
    validation_estimator = CapsuleCoverageEstimator(
        edges,
        cube_size_nm=config.cube_size_nm,
        rod_diameter_nm=config.rod_diameter_nm,
        qmc_power=config.validation_qmc_power,
        seed=config.random_seed + 1_000_003,
    )
    validation_count = validation_estimator.union_count(
        selection.selected_edge_ids
    )
    validation_solid_fraction = validation_estimator.fraction_from_count(
        validation_count
    )
    validation_porosity = 1.0 - validation_solid_fraction
    validation_error = abs(validation_porosity - config.target_porosity)
    report(f"Independent porosity: {100.0 * validation_porosity:.4f}%")
    if validation_error > config.porosity_tolerance:
        details = _failure_details(config, selection)
        details.update(
            {
                "status": "INDEPENDENT_VALIDATION_FAILURE",
                "validation_porosity": validation_porosity,
                "validation_absolute_error": validation_error,
                "validation_sample_count": validation_estimator.sample_count,
            }
        )
        details_path = write_metadata(output_dir / "feasibility.json", details)
        raise NanowireGenerationError(
            "Independent porosity validation exceeded the configured "
            f"tolerance; details: {details_path}"
        )

    selected_edges = [edges[index] for index in selection.selected_edge_ids]
    segments = [
        segment
        for edge in selected_edges
        for segment in periodic_edge_segments(edge, config.cube_size_nm)
    ]
    true_node_centres = _wrapped_true_node_centres(
        selected_edges,
        config.cube_size_nm,
    )
    report(f"Building display mesh from {len(segments)} clipped rod segments")
    display_mesh = build_display_mesh(
        segments,
        config.rod_radius_nm,
        tube_sides=config.tube_sides,
        node_centres_nm=true_node_centres,
    )
    mesh_path = None
    if save_display_mesh:
        mesh_path = output_dir / "display_mesh.npz"
        np.savez_compressed(
            mesh_path,
            vertices_nm=display_mesh["vertices_nm"],
            triangles=display_mesh["triangles"],
            config_json=json.dumps(config.to_dict(), sort_keys=True),
        )
    title = f"Voronoi nanowire aerogel | Porosity = {100.0 * config.target_porosity:g}%"
    html_path = write_interactive_html(
        output_dir / "nanowire_preview.html",
        display_mesh,
        config.cube_size_nm,
        title=title,
    )
    preview_path = save_preview_png(
        output_dir / "preview.png",
        display_mesh,
        config.cube_size_nm,
        title=title,
        max_faces=max(len(display_mesh["triangles"]), 1),
        dpi=600,
        show_axes=False,
        cell_size_annotation=False,
        show_title=False,
    )
    stl_path = None
    if export_stl:
        stl_path = write_binary_stl(
            output_dir / "nanowire_display_nm.stl",
            display_mesh,
            header="Voronoi nanowire display mesh; coordinates in nm",
        )
        report(f"Display STL: {stl_path}")

    selected_length_nm = float(sum(edge.length_nm for edge in selected_edges))
    clipped_length_nm = float(
        sum(np.linalg.norm(segment[1] - segment[0]) for segment in segments)
    )
    solid_fraction = 1.0 - selection.achieved_porosity
    reference_standard_error = (
        solid_fraction * max(1.0 - solid_fraction, 0.0)
        / estimator.sample_count
    ) ** 0.5
    metadata = {
        "schema": "periodic-voronoi-nanowire-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "assumptions": {
            "periodicity": ["x", "y", "z"],
            "rod_volume_model": "periodic union of equal-radius capsules",
            "porosity_definition": "1 - capsule union volume / cube volume",
            "whole_logical_edges_only": True,
            "display_mesh_boolean_unioned": False,
            "display_mesh_purpose": "visualization only",
        },
        "geometry": {
            "seed_count": len(seeds),
            "logical_face_count": len(voronoi.faces),
            "candidate_edge_count": len(edges),
            "selected_edge_count": len(selection.selected_edge_ids),
            "selected_edge_ids": list(selection.selected_edge_ids),
            "backbone_edge_count": len(selection.backbone_edge_ids),
            "display_segment_count": len(segments),
            "selected_logical_length_nm": selected_length_nm,
            "display_clipped_length_nm": clipped_length_nm,
            "display_mesh_vertices": len(display_mesh["vertices_nm"]),
            "display_mesh_triangles": len(display_mesh["triangles"]),
            "display_true_node_count": len(display_mesh["node_centres_nm"]),
        },
        "selection": {
            "status": selection.status,
            "attainable_range": asdict(selection.attainable_range),
            "porosity_before_crossing": selection.porosity_before_crossing,
            "porosity_after_crossing": selection.porosity_after_crossing,
        },
        "porosity": {
            "target": config.target_porosity,
            "tolerance": config.porosity_tolerance,
            "control_qmc": selection.achieved_porosity,
            "control_sample_count": estimator.sample_count,
            "independent_qmc": validation_porosity,
            "independent_sample_count": validation_estimator.sample_count,
            "reference_control_95pct_half_width": (
                1.96 * reference_standard_error
            ),
        },
        "topology": _winding_dict(selection),
        "files": {
            "interactive_html": html_path.name,
            "preview_png": preview_path.name,
            "config": "config_used.json",
            "metadata": "metadata.json",
        },
    }
    if stl_path is not None:
        metadata["files"]["display_stl"] = stl_path.name
        metadata["assumptions"]["display_stl_boolean_unioned"] = False
        metadata["stl_coordinate_unit"] = "nm"
    if mesh_path is not None:
        metadata["files"]["display_mesh_npz"] = mesh_path.name
    metadata_path = write_metadata(output_dir / "metadata.json", metadata)
    report(f"Finished: {output_dir}")
    return NanowireGenerationResult(
        output_dir=output_dir,
        control_porosity=selection.achieved_porosity,
        validation_porosity=validation_porosity,
        selected_edge_count=len(selection.selected_edge_ids),
        html_path=html_path,
        preview_path=preview_path,
        metadata_path=metadata_path,
        stl_path=stl_path,
        mesh_path=mesh_path,
    )


def export_cached_display_stl(config: NanowireConfig) -> Path:
    """Export the exact display mesh saved by the image-generation step."""

    config.validate()
    output_dir = Path(config.output_dir).resolve()
    mesh_path = output_dir / "display_mesh.npz"
    metadata_path = output_dir / "metadata.json"
    used_config_path = output_dir / "config_used.json"
    if not all(path.is_file() for path in (mesh_path, metadata_path, used_config_path)):
        raise FileNotFoundError("请先运行 image 步骤，生成 display_mesh.npz")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    used_config = NanowireConfig.from_json(used_config_path).to_dict()
    expected_config = json.dumps(config.to_dict(), sort_keys=True)
    with np.load(mesh_path, allow_pickle=False) as saved:
        if (
            saved["config_json"].item() != expected_config
            or metadata["config"] != config.to_dict()
            or used_config != config.to_dict()
        ):
            raise ValueError("参数与图片步骤不一致；请先重新运行 image 步骤")
        mesh = {
            "vertices_nm": saved["vertices_nm"],
            "triangles": saved["triangles"],
        }
    if (
        len(mesh["vertices_nm"]) != metadata["geometry"]["display_mesh_vertices"]
        or len(mesh["triangles"]) != metadata["geometry"]["display_mesh_triangles"]
    ):
        raise ValueError("网格缓存与元数据不一致；请重新运行 image 步骤")

    stl_path = write_binary_stl(
        output_dir / "nanowire_display_nm.stl",
        mesh,
        header="Voronoi nanowire display mesh; coordinates in nm",
    )
    metadata["files"]["display_stl"] = stl_path.name
    metadata["assumptions"]["display_stl_boolean_unioned"] = False
    metadata["stl_coordinate_unit"] = "nm"
    write_metadata(metadata_path, metadata)
    return stl_path


__all__ = [
    "NanowireGenerationError",
    "NanowireGenerationResult",
    "export_cached_display_stl",
    "generate_nanowire_model",
]
