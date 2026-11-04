"""Display-only mesh and file exporters for a Voronoi nanowire skeleton.

All coordinates are expressed in nanometres.  The generated surface is a
visual mesh: cylinders and node spheres overlap, but no expensive Boolean
union is attempted.  This keeps the preview faithful to the requested rod
diameter without introducing a simulation-mesh dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np


MeshMapping = Mapping[str, object]


def _segment_endpoints(segment: object, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one segment from a pair, mapping, or edge-like object."""

    if isinstance(segment, Mapping):
        start = segment.get("start_nm", segment.get("start"))
        end = segment.get("end_nm", segment.get("end"))
        if start is None or end is None:
            raise ValueError(
                f"segment {index} mapping needs start_nm/end_nm (or start/end)"
            )
    elif hasattr(segment, "start_nm") and hasattr(segment, "end_nm"):
        start = getattr(segment, "start_nm")
        end = getattr(segment, "end_nm")
    else:
        values = np.asarray(segment, dtype=float)
        if values.shape != (2, 3):
            raise ValueError(
                f"segment {index} must contain two three-dimensional endpoints"
            )
        start, end = values

    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    if start_array.shape != (3,) or end_array.shape != (3,):
        raise ValueError(f"segment {index} endpoints must each have shape (3,)")
    if not np.all(np.isfinite(start_array)) or not np.all(np.isfinite(end_array)):
        raise ValueError(f"segment {index} endpoints must be finite")
    return start_array, end_array


def _normal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def _append_cylinder(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    start: np.ndarray,
    end: np.ndarray,
    radius_nm: float,
    tube_sides: int,
    *,
    cap_ends: bool = False,
) -> None:
    direction = end - start
    direction /= np.linalg.norm(direction)
    basis_u, basis_v = _normal_basis(direction)
    angles = 2.0 * np.pi * np.arange(tube_sides, dtype=float) / tube_sides
    offsets = radius_nm * (
        np.cos(angles)[:, None] * basis_u + np.sin(angles)[:, None] * basis_v
    )
    base = len(vertices)
    vertices.extend(start + offsets)
    vertices.extend(end + offsets)
    for side in range(tube_sides):
        next_side = (side + 1) % tube_sides
        start_a = base + side
        start_b = base + next_side
        end_a = base + tube_sides + side
        end_b = base + tube_sides + next_side
        # Open cylinder ends are covered by the endpoint spheres.
        triangles.append((start_a, start_b, end_b))
        triangles.append((start_a, end_b, end_a))
    if cap_ends:
        start_centre = len(vertices)
        vertices.append(start.copy())
        end_centre = len(vertices)
        vertices.append(end.copy())
        for side in range(tube_sides):
            next_side = (side + 1) % tube_sides
            start_a = base + side
            start_b = base + next_side
            end_a = base + tube_sides + side
            end_b = base + tube_sides + next_side
            triangles.append((start_centre, start_b, start_a))
            triangles.append((end_centre, end_a, end_b))


def _append_uv_sphere(
    vertices: list[np.ndarray],
    triangles: list[tuple[int, int, int]],
    centre: np.ndarray,
    radius_nm: float,
    longitude_count: int,
    stack_count: int,
) -> None:
    north = len(vertices)
    vertices.append(centre + np.asarray((0.0, 0.0, radius_nm)))

    first_ring = len(vertices)
    for stack in range(1, stack_count):
        polar_angle = np.pi * stack / stack_count
        sine = np.sin(polar_angle)
        cosine = np.cos(polar_angle)
        for longitude in range(longitude_count):
            azimuth = 2.0 * np.pi * longitude / longitude_count
            vertices.append(
                centre
                + radius_nm
                * np.asarray(
                    (
                        sine * np.cos(azimuth),
                        sine * np.sin(azimuth),
                        cosine,
                    )
                )
            )

    south = len(vertices)
    vertices.append(centre + np.asarray((0.0, 0.0, -radius_nm)))

    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        triangles.append(
            (north, first_ring + longitude, first_ring + next_longitude)
        )

    for stack in range(stack_count - 2):
        upper = first_ring + stack * longitude_count
        lower = upper + longitude_count
        for longitude in range(longitude_count):
            next_longitude = (longitude + 1) % longitude_count
            triangles.append(
                (upper + longitude, lower + longitude, lower + next_longitude)
            )
            triangles.append(
                (upper + longitude, lower + next_longitude, upper + next_longitude)
            )

    last_ring = first_ring + (stack_count - 2) * longitude_count
    for longitude in range(longitude_count):
        next_longitude = (longitude + 1) % longitude_count
        triangles.append(
            (south, last_ring + next_longitude, last_ring + longitude)
        )


def _unique_nodes(endpoints: np.ndarray, tolerance_nm: float) -> np.ndarray:
    """Merge coincident skeleton endpoints using a small spatial hash."""

    nodes: list[np.ndarray] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for point in endpoints:
        key = tuple(int(value) for value in np.rint(point / tolerance_nm))
        match: int | None = None
        for offset in product((-1, 0, 1), repeat=3):
            neighbour = tuple(key[axis] + offset[axis] for axis in range(3))
            for candidate in buckets.get(neighbour, ()):
                if np.linalg.norm(point - nodes[candidate], ord=np.inf) <= tolerance_nm:
                    match = candidate
                    break
            if match is not None:
                break
        if match is None:
            buckets.setdefault(key, []).append(len(nodes))
            nodes.append(point.copy())
    return np.asarray(nodes, dtype=float).reshape((-1, 3))


def build_display_mesh(
    segments: Sequence[object] | np.ndarray,
    radius_nm: float,
    *,
    tube_sides: int = 12,
    sphere_stacks: int | None = None,
    node_tolerance_nm: float | None = None,
    node_centres_nm: Sequence[object] | np.ndarray | None = None,
) -> dict[str, object]:
    """Build one low-polygon surface mesh from clipped centreline segments.

    Each segment receives an open circular tube.  One sphere is then added at
    every geometrically unique endpoint by default, so rods meeting at a
    Voronoi vertex share the same visible node rather than accumulating
    duplicate end balls.  When ``node_centres_nm`` is provided, spheres are
    instead added only at those explicitly supplied true node coordinates.
    In that mode every tube also receives flat end caps, so points where a
    periodic rod was merely clipped by the unit-cell boundary remain closed
    without acquiring artificial ball caps.  Explicit nodes are deduplicated
    with ``node_tolerance_nm`` in the same way as inferred endpoint nodes.

    The result is intended for rendering and is not a Boolean-unioned solid.
    """

    radius = float(radius_nm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_nm must be a positive finite number")
    if isinstance(tube_sides, bool) or int(tube_sides) != tube_sides or tube_sides < 6:
        raise ValueError("tube_sides must be an integer of at least 6")
    side_count = int(tube_sides)
    stack_count = max(4, side_count // 2) if sphere_stacks is None else sphere_stacks
    if (
        isinstance(stack_count, bool)
        or int(stack_count) != stack_count
        or stack_count < 3
    ):
        raise ValueError("sphere_stacks must be an integer of at least 3")
    stack_count = int(stack_count)

    pairs = [_segment_endpoints(segment, index) for index, segment in enumerate(segments)]
    if pairs:
        segment_array = np.asarray(pairs, dtype=float)
        lengths = np.linalg.norm(segment_array[:, 1] - segment_array[:, 0], axis=1)
        invalid = np.flatnonzero(lengths <= np.finfo(float).eps)
        if len(invalid):
            raise ValueError(f"segment {int(invalid[0])} has zero length")
        endpoints = segment_array.reshape((-1, 3))
    else:
        segment_array = np.empty((0, 2, 3), dtype=float)
        endpoints = np.empty((0, 3), dtype=float)

    if node_centres_nm is None:
        node_candidates = endpoints
    else:
        explicit_nodes = np.asarray(node_centres_nm, dtype=float)
        if explicit_nodes.size == 0:
            explicit_nodes = np.empty((0, 3), dtype=float)
        if explicit_nodes.ndim != 2 or explicit_nodes.shape[1] != 3:
            raise ValueError("node_centres_nm must have shape (N, 3)")
        if not np.all(np.isfinite(explicit_nodes)):
            raise ValueError("node_centres_nm must be finite")
        node_candidates = explicit_nodes

    coordinate_scale = max(
        float(np.max(np.abs(segment_array))) if segment_array.size else 0.0,
        float(np.max(np.abs(node_candidates))) if node_candidates.size else 0.0,
    )
    scale = max(coordinate_scale, radius, 1.0)

    tolerance = (
        max(scale * 1.0e-9, np.finfo(float).eps * scale * 100.0)
        if node_tolerance_nm is None
        else float(node_tolerance_nm)
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("node_tolerance_nm must be a positive finite number")
    nodes = _unique_nodes(node_candidates, tolerance)

    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for start, end in segment_array:
        _append_cylinder(
            vertices,
            triangles,
            start,
            end,
            radius,
            side_count,
            cap_ends=node_centres_nm is not None,
        )
    for node in nodes:
        _append_uv_sphere(
            vertices,
            triangles,
            node,
            radius,
            side_count,
            stack_count,
        )

    return {
        "vertices_nm": np.asarray(vertices, dtype=float).reshape((-1, 3)),
        "triangles": np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        "node_centres_nm": nodes,
        "segment_count": len(segment_array),
        "node_count": len(nodes),
        "radius_nm": radius,
        "tube_sides": side_count,
        "sphere_stacks": stack_count,
    }


def _mesh_arrays(mesh: MeshMapping | object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, Mapping):
        vertices_value = mesh.get("vertices_nm", mesh.get("vertices"))
        triangles_value = mesh.get("triangles", mesh.get("faces"))
    else:
        vertices_value = getattr(mesh, "vertices_nm", getattr(mesh, "vertices", None))
        triangles_value = getattr(mesh, "triangles", getattr(mesh, "faces", None))
    if vertices_value is None or triangles_value is None:
        raise ValueError("mesh needs vertices_nm and triangles")
    vertices = np.asarray(vertices_value, dtype=float)
    triangles = np.asarray(triangles_value, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh vertices_nm must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mesh triangles must have shape (M, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh vertices_nm must be finite")
    if len(triangles) and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= len(vertices)
    ):
        raise ValueError("mesh triangles contain an out-of-range vertex index")
    return vertices, triangles


def _cube_line_coordinates(cube_size_nm: float) -> tuple[list[float | None], ...]:
    coordinates: list[list[float | None]] = [[], [], []]
    for start_tuple in product((0.0, cube_size_nm), repeat=3):
        start = np.asarray(start_tuple, dtype=float)
        for axis in range(3):
            if start[axis] != 0.0:
                continue
            end = start.copy()
            end[axis] = cube_size_nm
            for coordinate in range(3):
                coordinates[coordinate].extend(
                    (float(start[coordinate]), float(end[coordinate]), None)
                )
    return coordinates[0], coordinates[1], coordinates[2]


def write_interactive_html(
    path: str | Path,
    mesh: MeshMapping | object,
    cube_size_nm: float,
    *,
    title: str = "Periodic Voronoi nanowire network",
    wire_color: str = "#d7832f",
    opacity: float = 1.0,
) -> Path:
    """Write a self-contained, offline Plotly HTML preview."""

    box_size = float(cube_size_nm)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("cube_size_nm must be a positive finite number")
    if not 0.0 < float(opacity) <= 1.0:
        raise ValueError("opacity must be in (0, 1]")
    vertices, triangles = _mesh_arrays(mesh)

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError("Plotly is required to write the interactive HTML") from exc

    mesh_trace = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color=wire_color,
        opacity=float(opacity),
        flatshading=False,
        hoverinfo="skip",
        lighting={"ambient": 0.45, "diffuse": 0.75, "specular": 0.25, "roughness": 0.65},
        lightposition={"x": 1.6 * box_size, "y": 1.3 * box_size, "z": 2.2 * box_size},
        name="Nanowires",
        showlegend=False,
    )
    cube_x, cube_y, cube_z = _cube_line_coordinates(box_size)
    border_trace = go.Scatter3d(
        x=cube_x,
        y=cube_y,
        z=cube_z,
        mode="lines",
        line={"color": "#30343b", "width": 3},
        hoverinfo="skip",
        name="Unit cell",
        showlegend=False,
    )
    figure = go.Figure(data=(mesh_trace, border_trace))
    axis_style = {
        "range": [0.0, box_size],
        "visible": False,
        "showgrid": False,
        "showticklabels": False,
        "zeroline": False,
    }
    figure.update_layout(
        title={"text": title, "x": 0.5},
        margin={"l": 0, "r": 0, "b": 24, "t": 55},
        paper_bgcolor="white",
        scene={
            "xaxis": axis_style,
            "yaxis": axis_style,
            "zaxis": axis_style,
            "bgcolor": "white",
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        },
        annotations=[
            {
                "text": f"Unit-cell size: L = {box_size:g} nm",
                "xref": "paper",
                "yref": "paper",
                "x": 0.03,
                "y": 0.025,
                "showarrow": False,
                "font": {"size": 14, "color": "#30343b"},
                "bgcolor": "rgba(255,255,255,0.92)",
                "bordercolor": "#d9dde2",
                "borderwidth": 1,
                "borderpad": 5,
            }
        ],
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        target,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={"responsive": True, "displaylogo": False},
    )
    return target


def save_preview_png(
    path: str | Path,
    mesh: MeshMapping | object,
    cube_size_nm: float,
    *,
    title: str = "Periodic Voronoi nanowire network",
    max_faces: int = 50_000,
    dpi: int = 300,
    wire_color: str = "#d98934",
    border_color: str = "#3f4650",
    show_axes: bool = False,
    cell_size_annotation: bool = True,
    show_title: bool = False,
    export_vector: bool = False,
) -> Path:
    """Render a clean, solid-looking static PNG with Matplotlib.

    ``max_faces`` is retained for API compatibility and now controls the
    number of faces placed in each Matplotlib collection.  Every triangle is
    rendered; the mesh is never thinned by dropping faces, which would turn
    narrow rods into visually broken or dashed lines.

    By default the coordinate axes, ticks, and grid are hidden.  The unit-cell
    outline and one concise size annotation provide the only spatial frame.
    """

    box_size = float(cube_size_nm)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("cube_size_nm must be a positive finite number")
    if isinstance(max_faces, bool) or int(max_faces) != max_faces or max_faces <= 0:
        raise ValueError("max_faces must be a positive integer")
    if isinstance(dpi, bool) or int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    vertices, triangles = _mesh_arrays(mesh)

    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LightSource
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10.5,
        }
    )
    figure = Figure(
        figsize=(7.2, 7.2),
        constrained_layout=True,
        facecolor="#ffffff",
    )
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111, projection="3d", facecolor="#ffffff")
    figure.patch.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    light_source = LightSource(azdeg=315, altdeg=42)
    batch_size = int(max_faces)
    for first_face in range(0, len(triangles), batch_size):
        face_batch = triangles[first_face : first_face + batch_size]
        collection = Poly3DCollection(
            vertices[face_batch],
            facecolors=wire_color,
            linewidths=0.0,
            alpha=1.0,
            shade=True,
            lightsource=light_source,
            antialiaseds=True,
        )
        collection.set_edgecolor("none")
        axis.add_collection3d(collection)

    cube_x, cube_y, cube_z = _cube_line_coordinates(box_size)
    for offset in range(0, len(cube_x), 3):
        axis.plot(
            cube_x[offset : offset + 2],
            cube_y[offset : offset + 2],
            cube_z[offset : offset + 2],
            color=border_color,
            linewidth=0.8,
            alpha=0.72,
            solid_capstyle="round",
        )
    padding = 0.035 * box_size
    axis.set_xlim(-padding, box_size + padding)
    axis.set_ylim(-padding, box_size + padding)
    axis.set_zlim(-padding, box_size + padding)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.view_init(elev=22, azim=-52)

    if show_axes:
        axis.set_xlabel("x (nm)")
        axis.set_ylabel("y (nm)")
        axis.set_zlabel("z (nm)")
        axis.grid(False)
    else:
        axis.set_axis_off()
    if show_title:
        axis.set_title(title, pad=10.0, color="#30343b")
    if cell_size_annotation:
        axis.text2D(
            0.035,
            0.035,
            f"Unit-cell size: L = {box_size:g} nm",
            transform=axis.transAxes,
            color="#30343b",
            fontsize=10.5,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "#d9dde2",
                "linewidth": 0.7,
                "alpha": 0.92,
            },
        )

    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError("static preview path must use the .png extension")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        target,
        dpi=int(dpi),
        facecolor="#ffffff",
        edgecolor="none",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    if export_vector:
        figure.savefig(target.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    return target


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_metadata(
    path: str | Path,
    metadata: Mapping[str, object] | None = None,
    **fields: Any,
) -> Path:
    """Write UTF-8 JSON metadata with NumPy values converted safely."""

    data = {} if metadata is None else dict(metadata)
    data.update(fields)
    data.setdefault("model_type", "periodic_voronoi_nanowire")
    data.setdefault("coordinate_unit", "nm")
    data.setdefault("mesh_purpose", "visualization_only")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_ready(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


write_metadata_json = write_metadata


__all__ = [
    "build_display_mesh",
    "save_preview_png",
    "write_interactive_html",
    "write_metadata",
    "write_metadata_json",
]
